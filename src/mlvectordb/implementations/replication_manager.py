"""
Реализация ReplicationManager для MLVectorDB.

Модуль предоставляет реализацию менеджера репликации с поддержкой
синхронизации данных между репликами через HTTP API.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Sequence, Any
from uuid import UUID
from dataclasses import dataclass, field

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None

from ..interfaces.replication import ReplicationManager
from ..interfaces.vector import VectorProtocol
from ..interfaces.storage_engine import StorageEngine


@dataclass
class ReplicaInfo:
    """Информация о реплике."""
    replica_id: str
    url: str
    is_primary: bool = False
    last_health_check: Optional[float] = None
    is_healthy: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)


class ReplicationManagerImpl(ReplicationManager):
    """
    Реализация менеджера репликации.
    
    Поддерживает:
    - Динамическое добавление/удаление реплик
    - Синхронизацию данных через HTTP API
    - Health check реплик
    - Асинхронную репликацию с обработкой ошибок
    """
    
    def __init__(
        self,
        primary_storage: StorageEngine,
        primary_replica_id: str = "primary",
        health_check_interval: float = 5.0,
        request_timeout: float = 5.0,
        enable_async_replication: bool = True
    ):
        """
        Инициализация менеджера репликации.
        
        Args:
            primary_storage: Первичное хранилище (локальное)
            primary_replica_id: Идентификатор первичной реплики
            health_check_interval: Интервал проверки здоровья реплик (секунды)
            request_timeout: Таймаут HTTP запросов (секунды)
            enable_async_replication: Включить асинхронную репликацию
        """
        if requests is None:
            raise ImportError(
                "Библиотека 'requests' не установлена. "
                "Установите её: pip install requests"
            )
        
        self._primary_storage = primary_storage
        self._primary_replica_id = primary_replica_id
        self._health_check_interval = health_check_interval
        self._request_timeout = request_timeout
        self._enable_async_replication = enable_async_replication
        
        self._replicas: Dict[str, ReplicaInfo] = {}
        self._lock = threading.RLock()
        
        # Настройка HTTP сессии с retry стратегией
        self._session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
        # Запуск фонового потока для health check
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()
        if health_check_interval > 0:
            self._start_health_check_thread()
    
    @property
    def replica_count(self) -> int:
        """Количество активных реплик."""
        with self._lock:
            return len([r for r in self._replicas.values() if r.is_healthy])
    
    @property
    def primary_replica(self) -> Optional[str]:
        """Идентификатор первичной реплики."""
        return self._primary_replica_id
    
    def add_replica(self, replica_id: str, replica_url: str) -> bool:
        """Добавить новую реплику."""
        if replica_id == self._primary_replica_id:
            self.logger.warning(f"Нельзя добавить реплику с ID первичной: {replica_id}")
            return False
        
        with self._lock:
            if replica_id in self._replicas:
                self.logger.warning(f"Реплика {replica_id} уже существует")
                return False
            
            replica = ReplicaInfo(
                replica_id=replica_id,
                url=replica_url.rstrip("/"),
                is_primary=False
            )
            self._replicas[replica_id] = replica
            
            # Проверяем здоровье новой реплики
            is_healthy = self._check_replica_health_internal(replica_id)
            replica.is_healthy = is_healthy
            replica.last_health_check = time.time()
            
            self.logger.info(
                f"Добавлена реплика {replica_id} по адресу {replica_url} "
                f"(здоровье: {'OK' if is_healthy else 'FAIL'})"
            )
            return True
    
    def remove_replica(self, replica_id: str) -> bool:
        """Удалить реплику."""
        if replica_id == self._primary_replica_id:
            self.logger.warning(f"Нельзя удалить первичную реплику: {replica_id}")
            return False
        
        with self._lock:
            if replica_id not in self._replicas:
                self.logger.warning(f"Реплика {replica_id} не найдена")
                return False
            
            del self._replicas[replica_id]
            self.logger.info(f"Удалена реплика {replica_id}")
            return True
    
    def replicate_write(
        self,
        vector: VectorProtocol,
        namespace: str
    ) -> Dict[str, bool]:
        """Реплицировать запись вектора на все реплики."""
        results = {}
        
        with self._lock:
            replicas = list(self._replicas.values())
        
        if not replicas:
            return results
        
        # Подготовка данных для репликации
        vector_data = {
            "values": vector.values.tolist() if hasattr(vector.values, 'tolist') else list(vector.values),
            "metadata": dict(vector.metadata)
        }
        
        if self._enable_async_replication:
            # Асинхронная репликация в фоновом потоке
            thread = threading.Thread(
                target=self._replicate_write_async,
                args=(replicas, vector_data, namespace, results),
                daemon=True
            )
            thread.start()
            # Ждем немного для получения начальных результатов
            thread.join(timeout=0.1)
        else:
            # Синхронная репликация
            self._replicate_write_sync(replicas, vector_data, namespace, results)
        
        return results
    
    def _replicate_write_sync(
        self,
        replicas: List[ReplicaInfo],
        vector_data: Dict,
        namespace: str,
        results: Dict[str, bool]
    ):
        """Синхронная репликация записи."""
        for replica in replicas:
            if not replica.is_healthy:
                results[replica.replica_id] = False
                continue
            
            try:
                url = f"{replica.url}/vectors?namespace={namespace}"
                response = self._session.post(
                    url,
                    json=vector_data,
                    timeout=self._request_timeout
                )
                results[replica.replica_id] = response.status_code == 201
                
                if response.status_code != 201:
                    self.logger.warning(
                        f"Ошибка репликации на {replica.replica_id}: "
                        f"HTTP {response.status_code}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Ошибка репликации на {replica.replica_id}: {e}",
                    exc_info=True
                )
                results[replica.replica_id] = False
                replica.is_healthy = False
    
    def _replicate_write_async(
        self,
        replicas: List[ReplicaInfo],
        vector_data: Dict,
        namespace: str,
        results: Dict[str, bool]
    ):
        """Асинхронная репликация записи."""
        self._replicate_write_sync(replicas, vector_data, namespace, results)
    
    def replicate_delete(
        self,
        vector_id: UUID,
        namespace: str
    ) -> Dict[str, bool]:
        """Реплицировать удаление вектора на все реплики."""
        results = {}
        
        with self._lock:
            replicas = list(self._replicas.values())
        
        if not replicas:
            return results
        
        delete_data = {"ids": [str(vector_id)]}
        
        for replica in replicas:
            if not replica.is_healthy:
                results[replica.replica_id] = False
                continue
            
            try:
                url = f"{replica.url}/vectors?namespace={namespace}"
                response = self._session.delete(
                    url,
                    json=delete_data,
                    timeout=self._request_timeout
                )
                results[replica.replica_id] = response.status_code == 200
                
                if response.status_code != 200:
                    self.logger.warning(
                        f"Ошибка репликации удаления на {replica.replica_id}: "
                        f"HTTP {response.status_code}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Ошибка репликации удаления на {replica.replica_id}: {e}",
                    exc_info=True
                )
                results[replica.replica_id] = False
                replica.is_healthy = False
        
        return results
    
    def replicate_batch_write(
        self,
        vectors: Sequence[VectorProtocol],
        namespace: str
    ) -> Dict[str, bool]:
        """Реплицировать пакетную запись векторов на все реплики."""
        results = {}
        
        with self._lock:
            replicas = list(self._replicas.values())
        
        if not replicas:
            return results
        
        # Подготовка данных для репликации
        vectors_data = {
            "vectors": [
                {
                    "values": v.values.tolist() if hasattr(v.values, 'tolist') else list(v.values),
                    "metadata": dict(v.metadata)
                }
                for v in vectors
            ]
        }
        
        for replica in replicas:
            if not replica.is_healthy:
                results[replica.replica_id] = False
                continue
            
            try:
                url = f"{replica.url}/vectors/batch?namespace={namespace}"
                response = self._session.put(
                    url,
                    json=vectors_data,
                    timeout=self._request_timeout * len(vectors)  # Увеличиваем таймаут для батча
                )
                results[replica.replica_id] = response.status_code == 200
                
                if response.status_code != 200:
                    self.logger.warning(
                        f"Ошибка батч-репликации на {replica.replica_id}: "
                        f"HTTP {response.status_code}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Ошибка батч-репликации на {replica.replica_id}: {e}",
                    exc_info=True
                )
                results[replica.replica_id] = False
                replica.is_healthy = False
        
        return results
    
    def check_replica_health(self, replica_id: str) -> bool:
        """Проверить работоспособность реплики."""
        with self._lock:
            if replica_id not in self._replicas:
                return False
            return self._check_replica_health_internal(replica_id)
    
    def _check_replica_health_internal(self, replica_id: str) -> bool:
        """Внутренний метод проверки здоровья реплики."""
        replica = self._replicas[replica_id]
        
        try:
            response = self._session.get(
                f"{replica.url}/health",
                timeout=2.0
            )
            is_healthy = response.status_code == 200
            replica.is_healthy = is_healthy
            replica.last_health_check = time.time()
            return is_healthy
        except Exception as e:
            self.logger.debug(f"Health check failed for {replica_id}: {e}")
            replica.is_healthy = False
            replica.last_health_check = time.time()
            return False
    
    def check_all_replicas_health(self) -> Dict[str, bool]:
        """Проверить работоспособность всех реплик."""
        results = {}
        
        with self._lock:
            replica_ids = list(self._replicas.keys())
        
        for replica_id in replica_ids:
            results[replica_id] = self.check_replica_health(replica_id)
        
        return results
    
    def sync_replica(self, replica_id: str, namespace: Optional[str] = None) -> bool:
        """
        Синхронизировать реплику с первичной.
        
        Полная синхронизация: получает все векторы из первичного хранилища
        и отправляет их на реплику через batch API.
        """
        if replica_id not in self._replicas:
            self.logger.warning(f"Реплика {replica_id} не найдена")
            return False
        
        replica = self._replicas[replica_id]
        
        # Проверяем здоровье
        is_healthy = self.check_replica_health(replica_id)
        if not is_healthy:
            self.logger.error(f"Реплика {replica_id} недоступна для синхронизации")
            return False
        
        try:
            # Получаем все векторы из первичного хранилища
            namespaces_to_sync = [namespace] if namespace else list(self._primary_storage.namespace_map.keys())
            
            total_synced = 0
            for ns in namespaces_to_sync:
                vectors = self._primary_storage.namespace_map.get(ns, [])
                
                if not vectors:
                    self.logger.debug(f"Пространство {ns} пустое, пропускаем")
                    continue
                
                # Разбиваем на батчи для эффективной передачи
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    
                    # Подготавливаем данные для batch API
                    vectors_data = {
                        "vectors": [
                            {
                                "values": v.values.tolist() if hasattr(v.values, 'tolist') else list(v.values),
                                "metadata": dict(v.metadata)
                            }
                            for v in batch
                        ]
                    }
                    
                    # Отправляем batch на реплику
                    try:
                        url = f"{replica.url}/vectors/batch?namespace={ns}"
                        response = self._session.put(
                            url,
                            json=vectors_data,
                            timeout=self._request_timeout * len(batch)
                        )
                        
                        if response.status_code == 200:
                            total_synced += len(batch)
                            self.logger.debug(
                                f"Синхронизировано {len(batch)} векторов в пространстве {ns} "
                                f"(всего: {total_synced})"
                            )
                        else:
                            self.logger.error(
                                f"Ошибка синхронизации batch в пространстве {ns}: "
                                f"HTTP {response.status_code}"
                            )
                            return False
                    except Exception as e:
                        self.logger.error(
                            f"Ошибка синхронизации batch в пространстве {ns}: {e}",
                            exc_info=True
                        )
                        return False
            
            self.logger.info(
                f"Реплика {replica_id} успешно синхронизирована: "
                f"{total_synced} векторов из {len(namespaces_to_sync)} пространств"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Критическая ошибка синхронизации реплики {replica_id}: {e}",
                exc_info=True
            )
            return False
    
    def get_replica_info(self) -> Dict[str, Any]:
        """Получить информацию о всех репликах."""
        with self._lock:
            info = {
                "primary_replica": self._primary_replica_id,
                "total_replicas": len(self._replicas),
                "healthy_replicas": len([r for r in self._replicas.values() if r.is_healthy]),
                "replicas": {}
            }
            
            for replica_id, replica in self._replicas.items():
                info["replicas"][replica_id] = {
                    "url": replica.url,
                    "is_healthy": replica.is_healthy,
                    "last_health_check": replica.last_health_check,
                    "is_primary": replica.is_primary
                }
        
        return info
    
    def list_replicas(self) -> List[str]:
        """Получить список всех реплик."""
        with self._lock:
            return list(self._replicas.keys())
    
    def _start_health_check_thread(self):
        """Запустить фоновый поток для проверки здоровья реплик."""
        if self._health_check_thread is not None:
            return
        
        def health_check_loop():
            while not self._stop_health_check.is_set():
                try:
                    self.check_all_replicas_health()
                except Exception as e:
                    self.logger.error(f"Ошибка в health check потоке: {e}", exc_info=True)
                
                self._stop_health_check.wait(self._health_check_interval)
        
        self._health_check_thread = threading.Thread(
            target=health_check_loop,
            daemon=True,
            name="ReplicationHealthCheck"
        )
        self._health_check_thread.start()
        self.logger.info("Запущен поток проверки здоровья реплик")
    
    def shutdown(self):
        """Остановить менеджер репликации."""
        self._stop_health_check.set()
        if self._health_check_thread:
            self._health_check_thread.join(timeout=2.0)
        self._session.close()
        self.logger.info("Менеджер репликации остановлен")

