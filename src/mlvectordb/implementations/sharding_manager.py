"""
Реализация ShardingManager для MLVectorDB.

Модуль предоставляет реализацию менеджера шардирования с поддержкой
распределения данных по нескольким шардам.
"""

import hashlib
import logging
import threading
import time
from typing import Dict, List, Optional, Sequence, Any
from uuid import UUID

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None

from ..interfaces.sharding import ShardingManager
from ..interfaces.vector import VectorProtocol, VectorDTO
from ..interfaces.storage_engine import StorageEngine


class ShardInfo:
    """Информация о шарде."""
    
    def __init__(self, shard_id: str, url: Optional[str] = None):
        self.shard_id = shard_id
        self.url = url
        self.last_health_check: Optional[float] = None
        self.is_healthy: bool = True
        self.vector_count: int = 0
        self.lock = threading.Lock()
    
    def __repr__(self):
        return f"ShardInfo(id={self.shard_id}, url={self.url}, healthy={self.is_healthy})"


class ShardingManagerImpl(ShardingManager):
    """
    Реализация менеджера шардирования.
    
    Поддерживает:
    - Распределение данных по шардам через хеширование
    - Динамическое добавление/удаление шардов
    - Health check шардов
    - Поддержку локальных и удаленных шардов
    """
    
    def __init__(
        self,
        shard_storages: Optional[Dict[str, StorageEngine]] = None,
        sharding_strategy: str = "hash",
        health_check_interval: float = 5.0,
        request_timeout: float = 5.0
    ):
        """
        Инициализация менеджера шардирования.
        
        Args:
            shard_storages: Словарь {shard_id: StorageEngine} для локальных шардов
            sharding_strategy: Стратегия шардирования ("hash" или "round_robin")
            health_check_interval: Интервал проверки здоровья шардов (секунды)
            request_timeout: Таймаут HTTP запросов для удаленных шардов (секунды)
        """
        if requests is None and shard_storages is None:
            # Если нет requests, но есть локальные шарды - это OK
            pass
        
        self._shard_storages: Dict[str, StorageEngine] = shard_storages or {}
        self._sharding_strategy = sharding_strategy
        self._health_check_interval = health_check_interval
        self._request_timeout = request_timeout
        
        self._shards: Dict[str, ShardInfo] = {}
        self._lock = threading.RLock()
        
        # Инициализация локальных шардов
        if self._shard_storages:
            for shard_id in self._shard_storages.keys():
                self._shards[shard_id] = ShardInfo(shard_id, url=None)
        
        # Настройка HTTP сессии для удаленных шардов
        if requests is not None:
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
        else:
            self._session = None
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
        # Запуск фонового потока для health check (только для удаленных шардов)
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()
        if health_check_interval > 0 and self._has_remote_shards():
            self._start_health_check_thread()
    
    @property
    def shard_count(self) -> int:
        """Количество активных шардов."""
        with self._lock:
            return len([s for s in self._shards.values() if s.is_healthy])
    
    def add_shard(self, shard_id: str, shard_url: Optional[str] = None) -> bool:
        """Добавить новый шард."""
        with self._lock:
            if shard_id in self._shards:
                self.logger.warning(f"Шард {shard_id} уже существует")
                return False
            
            shard = ShardInfo(shard_id, url=shard_url)
            self._shards[shard_id] = shard
            
            # Если это удаленный шард, проверяем его здоровье
            if shard_url:
                is_healthy = self._check_shard_health_internal(shard_id)
                shard.is_healthy = is_healthy
                shard.last_health_check = time.time()
                
                # Перезапускаем health check поток, если нужно
                if not self._health_check_thread and self._health_check_interval > 0:
                    self._start_health_check_thread()
            else:
                # Локальный шард всегда здоров
                shard.is_healthy = True
            
            self.logger.info(
                f"Добавлен шард {shard_id} "
                f"({'удаленный' if shard_url else 'локальный'})"
            )
            return True
    
    def remove_shard(self, shard_id: str) -> bool:
        """Удалить шард."""
        with self._lock:
            if shard_id not in self._shards:
                self.logger.warning(f"Шард {shard_id} не найден")
                return False
            
            # Не удаляем шард, если это единственный шард
            if len(self._shards) <= 1:
                self.logger.warning("Нельзя удалить последний шард")
                return False
            
            del self._shards[shard_id]
            if shard_id in self._shard_storages:
                del self._shard_storages[shard_id]
            
            self.logger.info(f"Удален шард {shard_id}")
            return True
    
    def get_shard_for_vector(
        self,
        vector: VectorProtocol,
        namespace: str = "default"
    ) -> str:
        """Определить шард для вектора."""
        return self.get_shard_for_id(vector.id, namespace)
    
    def get_shard_for_id(
        self,
        vector_id: UUID,
        namespace: str = "default"
    ) -> Optional[str]:
        """Определить шард для вектора по его ID."""
        with self._lock:
            healthy_shards = [
                shard_id for shard_id, shard in self._shards.items()
                if shard.is_healthy
            ]
            
            if not healthy_shards:
                self.logger.error("Нет доступных здоровых шардов")
                return None
            
            if len(healthy_shards) == 1:
                return healthy_shards[0]
            
            if self._sharding_strategy == "hash":
                # Хеширование по ID вектора и namespace
                key = f"{namespace}:{vector_id}"
                hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
                shard_index = hash_value % len(healthy_shards)
                return healthy_shards[shard_index]
            elif self._sharding_strategy == "round_robin":
                # Round-robin распределение
                hash_value = hash(str(vector_id))
                shard_index = abs(hash_value) % len(healthy_shards)
                return healthy_shards[shard_index]
            else:
                # По умолчанию используем хеширование
                key = f"{namespace}:{vector_id}"
                hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
                shard_index = hash_value % len(healthy_shards)
                return healthy_shards[shard_index]
    
    def get_shards_for_search(
        self,
        namespace: str = "default"
    ) -> List[str]:
        """Получить список шардов для поиска (обычно все здоровые шарды)."""
        with self._lock:
            return [
                shard_id for shard_id, shard in self._shards.items()
                if shard.is_healthy
            ]
    
    def redistribute_data(self, from_shard: str, to_shard: str) -> bool:
        """
        Перераспределить данные между шардами.
        
        Получает все векторы из исходного шарда, определяет какие должны быть
        на целевом шарде согласно стратегии шардирования, и переносит их.
        """
        if from_shard not in self._shards or to_shard not in self._shards:
            self.logger.warning(f"Один из шардов не найден: {from_shard}, {to_shard}")
            return False
        
        if from_shard == to_shard:
            self.logger.warning("Нельзя перераспределить данные в тот же шард")
            return False
        
        from_shard_info = self._shards[from_shard]
        to_shard_info = self._shards[to_shard]
        
        # Проверяем здоровье шардов
        if not self.check_shard_health(from_shard) or not self.check_shard_health(to_shard):
            self.logger.error("Один из шардов недоступен")
            return False
        
        try:
            # Получаем хранилище исходного шарда
            from_storage = self._shard_storages.get(from_shard)
            if not from_storage:
                self.logger.error(f"Исходный шард {from_shard} не является локальным")
                return False
            
            # Получаем все векторы из исходного шарда
            all_vectors = []
            namespaces = from_storage.list_namespaces
            
            for namespace in namespaces:
                vectors = from_storage.namespace_map.get(namespace, [])
                all_vectors.extend([(v, namespace) for v in vectors])
            
            if not all_vectors:
                self.logger.info(f"Исходный шард {from_shard} пуст, нечего перераспределять")
                return True
            
            self.logger.info(
                f"Найдено {len(all_vectors)} векторов в шарде {from_shard} "
                f"для перераспределения в {to_shard}"
            )
            
            # Определяем, какие векторы должны быть на целевом шарде
            vectors_to_move = []
            vectors_to_keep = []
            
            for vector, namespace in all_vectors:
                target_shard = self.get_shard_for_vector(vector, namespace)
                if target_shard == to_shard:
                    vectors_to_move.append((vector, namespace))
                else:
                    vectors_to_keep.append((vector, namespace))
            
            if not vectors_to_move:
                self.logger.info(f"Нет векторов для перемещения из {from_shard} в {to_shard}")
                return True
            
            self.logger.info(
                f"Будет перемещено {len(vectors_to_move)} векторов, "
                f"останется {len(vectors_to_keep)} векторов"
            )
            
            # Получаем хранилище целевого шарда
            to_storage = self._shard_storages.get(to_shard)
            
            if to_storage:
                # Локальный целевой шард - записываем напрямую
                # Группируем по namespace для эффективной записи
                by_namespace: Dict[str, List[VectorProtocol]] = {}
                for vector, namespace in vectors_to_move:
                    if namespace not in by_namespace:
                        by_namespace[namespace] = []
                    by_namespace[namespace].append(vector)
                
                for namespace, vectors in by_namespace.items():
                    to_storage.write_vectors(vectors, namespace)
                    self.logger.debug(
                        f"Перемещено {len(vectors)} векторов в локальный шард {to_shard} "
                        f"(пространство: {namespace})"
                    )
            elif to_shard_info.url:
                # Удаленный целевой шард - отправляем через HTTP API
                # Группируем по namespace
                by_namespace: Dict[str, List[VectorProtocol]] = {}
                for vector, namespace in vectors_to_move:
                    if namespace not in by_namespace:
                        by_namespace[namespace] = []
                    by_namespace[namespace].append(vector)
                
                for namespace, vectors in by_namespace.items():
                    # Подготавливаем данные для batch API
                    vectors_data = {
                        "vectors": [
                            {
                                "values": v.values.tolist() if hasattr(v.values, 'tolist') else list(v.values),
                                "metadata": dict(v.metadata)
                            }
                            for v in vectors
                        ]
                    }
                    
                    # Отправляем на удаленный шард
                    if self._session:
                        url = f"{to_shard_info.url}/vectors/batch?namespace={namespace}"
                        response = self._session.put(
                            url,
                            json=vectors_data,
                            timeout=self._request_timeout * len(vectors)
                        )
                        
                        if response.status_code != 200:
                            self.logger.error(
                                f"Ошибка отправки векторов на удаленный шард {to_shard}: "
                                f"HTTP {response.status_code}"
                            )
                            return False
                        
                        self.logger.debug(
                            f"Перемещено {len(vectors)} векторов на удаленный шард {to_shard} "
                            f"(пространство: {namespace})"
                        )
            else:
                self.logger.error(f"Целевой шард {to_shard} не настроен")
                return False
            
            # Удаляем перемещенные векторы из исходного шарда
            for vector, namespace in vectors_to_move:
                from_storage.delete(vector.id, namespace)
            
            self.logger.info(
                f"Перераспределение завершено: {len(vectors_to_move)} векторов "
                f"из {from_shard} в {to_shard}"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Ошибка перераспределения данных {from_shard} -> {to_shard}: {e}",
                exc_info=True
            )
            return False
    
    def check_shard_health(self, shard_id: str) -> bool:
        """Проверить работоспособность шарда."""
        with self._lock:
            if shard_id not in self._shards:
                return False
            
            shard = self._shards[shard_id]
            
            # Локальные шарды всегда здоровы (если они есть)
            if shard.url is None:
                if shard_id in self._shard_storages:
                    shard.is_healthy = True
                    return True
                else:
                    shard.is_healthy = False
                    return False
            
            # Для удаленных шардов проверяем через HTTP
            return self._check_shard_health_internal(shard_id)
    
    def _check_shard_health_internal(self, shard_id: str) -> bool:
        """Внутренний метод проверки здоровья шарда."""
        if self._session is None:
            return False
        
        shard = self._shards[shard_id]
        
        if shard.url is None:
            # Локальный шард
            return shard_id in self._shard_storages
        
        try:
            response = self._session.get(
                f"{shard.url}/health",
                timeout=2.0
            )
            is_healthy = response.status_code == 200
            shard.is_healthy = is_healthy
            shard.last_health_check = time.time()
            return is_healthy
        except Exception as e:
            self.logger.debug(f"Health check failed for shard {shard_id}: {e}")
            shard.is_healthy = False
            shard.last_health_check = time.time()
            return False
    
    def check_all_shards_health(self) -> Dict[str, bool]:
        """Проверить работоспособность всех шардов."""
        results = {}
        
        with self._lock:
            shard_ids = list(self._shards.keys())
        
        for shard_id in shard_ids:
            results[shard_id] = self.check_shard_health(shard_id)
        
        return results
    
    def get_shard_info(self) -> Dict[str, Any]:
        """Получить информацию о всех шардах."""
        with self._lock:
            info = {
                "total_shards": len(self._shards),
                "healthy_shards": len([s for s in self._shards.values() if s.is_healthy]),
                "sharding_strategy": self._sharding_strategy,
                "shards": {}
            }
            
            for shard_id, shard in self._shards.items():
                shard_info = {
                    "is_healthy": shard.is_healthy,
                    "is_local": shard.url is None,
                    "vector_count": shard.vector_count
                }
                
                if shard.url:
                    shard_info["url"] = shard.url
                    shard_info["last_health_check"] = shard.last_health_check
                
                info["shards"][shard_id] = shard_info
        
        return info
    
    def list_shards(self) -> List[str]:
        """Получить список всех шардов."""
        with self._lock:
            return list(self._shards.keys())
    
    def get_shard_storage(self, shard_id: str) -> Optional[StorageEngine]:
        """Получить StorageEngine для локального шарда."""
        return self._shard_storages.get(shard_id)
    
    def _has_remote_shards(self) -> bool:
        """Проверить, есть ли удаленные шарды."""
        return any(shard.url is not None for shard in self._shards.values())
    
    def _start_health_check_thread(self):
        """Запустить фоновый поток для проверки здоровья шардов."""
        if self._health_check_thread is not None:
            return
        
        if not self._has_remote_shards():
            return
        
        def health_check_loop():
            while not self._stop_health_check.is_set():
                try:
                    # Проверяем только удаленные шарды
                    with self._lock:
                        remote_shards = [
                            shard_id for shard_id, shard in self._shards.items()
                            if shard.url is not None
                        ]
                    
                    for shard_id in remote_shards:
                        self._check_shard_health_internal(shard_id)
                except Exception as e:
                    self.logger.error(f"Ошибка в health check потоке: {e}", exc_info=True)
                
                self._stop_health_check.wait(self._health_check_interval)
        
        self._health_check_thread = threading.Thread(
            target=health_check_loop,
            daemon=True,
            name="ShardingHealthCheck"
        )
        self._health_check_thread.start()
        self.logger.info("Запущен поток проверки здоровья шардов")
    
    def write_to_remote_shard(
        self,
        shard_id: str,
        vector: VectorProtocol,
        namespace: str
    ) -> bool:
        """
        Записать вектор на удаленный шард через HTTP API.
        
        Args:
            shard_id: Идентификатор удаленного шарда
            vector: Вектор для записи
            namespace: Пространство имен
            
        Returns:
            True если запись успешна
        """
        if self._session is None:
            self.logger.error("HTTP сессия не инициализирована")
            return False
        
        if shard_id not in self._shards:
            self.logger.error(f"Шард {shard_id} не найден")
            return False
        
        shard = self._shards[shard_id]
        if shard.url is None:
            self.logger.error(f"Шард {shard_id} не является удаленным")
            return False
        
        try:
            vector_data = {
                "values": vector.values.tolist() if hasattr(vector.values, 'tolist') else list(vector.values),
                "metadata": dict(vector.metadata)
            }
            
            url = f"{shard.url}/vectors?namespace={namespace}"
            response = self._session.post(
                url,
                json=vector_data,
                timeout=self._request_timeout
            )
            
            if response.status_code == 201:
                return True
            else:
                self.logger.error(
                    f"Ошибка записи на удаленный шард {shard_id}: "
                    f"HTTP {response.status_code}"
                )
                return False
        except Exception as e:
            self.logger.error(
                f"Ошибка записи на удаленный шард {shard_id}: {e}",
                exc_info=True
            )
            return False
    
    def read_from_remote_shard(
        self,
        shard_id: str,
        vector_id: UUID,
        namespace: str
    ) -> Optional[VectorProtocol]:
        """
        Прочитать вектор с удаленного шарда через HTTP API.
        
        Args:
            shard_id: Идентификатор удаленного шарда
            vector_id: ID вектора
            namespace: Пространство имен
            
        Returns:
            Вектор или None если не найден
        """
        if self._session is None:
            self.logger.error("HTTP сессия не инициализирована")
            return None
        
        if shard_id not in self._shards:
            self.logger.error(f"Шард {shard_id} не найден")
            return None
        
        shard = self._shards[shard_id]
        if shard.url is None:
            self.logger.error(f"Шард {shard_id} не является удаленным")
            return None
        
        try:
            # Используем endpoint для получения всех векторов namespace и фильтруем
            # В реальной реализации нужен endpoint /vectors/{vector_id}
            url = f"{shard.url}/namespaces/vectors?namespace={namespace}"
            response = self._session.get(
                url,
                timeout=self._request_timeout
            )
            
            if response.status_code == 200:
                vectors_data = response.json()
                for vec_data in vectors_data:
                    if UUID(vec_data["id"]) == vector_id:
                        # Создаем вектор из данных
                        from ..vector import Vector
                        return Vector(
                            values=vec_data["values"],
                            metadata=vec_data.get("metadata", {})
                        )
                return None
            else:
                self.logger.error(
                    f"Ошибка чтения с удаленного шарда {shard_id}: "
                    f"HTTP {response.status_code}"
                )
                return None
        except Exception as e:
            self.logger.error(
                f"Ошибка чтения с удаленного шарда {shard_id}: {e}",
                exc_info=True
            )
            return None
    
    def search_on_remote_shard(
        self,
        shard_id: str,
        query: VectorDTO,
        top_k: int,
        namespace: str,
        metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """
        Выполнить поиск на удаленном шарде через HTTP API.
        
        Args:
            shard_id: Идентификатор удаленного шарда
            query: Вектор запроса
            top_k: Количество результатов
            namespace: Пространство имен
            metric: Метрика расстояния
            
        Returns:
            Список результатов поиска
        """
        if self._session is None:
            self.logger.error("HTTP сессия не инициализирована")
            return []
        
        if shard_id not in self._shards:
            self.logger.error(f"Шард {shard_id} не найден")
            return []
        
        shard = self._shards[shard_id]
        if shard.url is None:
            self.logger.error(f"Шард {shard_id} не является удаленным")
            return []
        
        try:
            search_data = {
                "query": query.values.tolist() if hasattr(query.values, 'tolist') else list(query.values),
                "top_k": top_k,
                "metric": metric
            }
            
            url = f"{shard.url}/search?namespace={namespace}"
            response = self._session.post(
                url,
                json=search_data,
                timeout=self._request_timeout * top_k
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(
                    f"Ошибка поиска на удаленном шарде {shard_id}: "
                    f"HTTP {response.status_code}"
                )
                return []
        except Exception as e:
            self.logger.error(
                f"Ошибка поиска на удаленном шарде {shard_id}: {e}",
                exc_info=True
            )
            return []
    
    def shutdown(self):
        """Остановить менеджер шардирования."""
        self._stop_health_check.set()
        if self._health_check_thread:
            self._health_check_thread.join(timeout=2.0)
        if self._session:
            self._session.close()
        self.logger.info("Менеджер шардирования остановлен")

