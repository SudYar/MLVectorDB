"""
Реализация ShardingManager для MLVectorDB.

Модуль предоставляет реализацию менеджера шардирования с поддержкой
распределения данных по нескольким шардам.
"""

import hashlib
import logging
import threading
import time
from typing import Dict, List, Optional, Any
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
from ..implementations.storage_engine_in_memory import StorageEngineInMemory


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
        
        # Счетчик для round-robin распределения
        self._round_robin_counter: int = 0
        
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
                # Локальный шард - проверяем, есть ли для него StorageEngine
                if shard_id in self._shard_storages:
                    shard.is_healthy = True
                else:
                    # Автоматически создаем StorageEngine для локального шарда
                    # чтобы он был здоровым, как при инициализации
                    self._shard_storages[shard_id] = StorageEngineInMemory()
                    shard.is_healthy = True
                    self.logger.info(
                        f"Автоматически создан StorageEngine для локального шарда {shard_id}"
                    )
            
            self.logger.info(
                f"Добавлен шард {shard_id} "
                f"({'удаленный' if shard_url else 'локальный'})"
            )
            return True
    
    def remove_shard(self, shard_id: str) -> bool:
        """
        Удалить шард с автоматическим перераспределением данных.
        
        Перед удалением все векторы из удаляемого шарда автоматически
        перераспределяются на другие доступные шарды согласно стратегии шардирования.
        """
        with self._lock:
            if shard_id not in self._shards:
                self.logger.warning(f"Шард {shard_id} не найден")
                return False
            
            # Не удаляем шард, если это единственный шард
            if len(self._shards) <= 1:
                self.logger.warning("Нельзя удалить последний шард")
                return False
            
            # Получаем список оставшихся шардов (без удаляемого)
            remaining_shards = [
                sid for sid in self._shards.keys() 
                if sid != shard_id and self._shards[sid].is_healthy
            ]
            
            if not remaining_shards:
                self.logger.error("Нет доступных здоровых шардов для перераспределения данных")
                return False
            
            self.logger.info(
                f"Начинается удаление шарда {shard_id}. "
                f"Данные будут перераспределены на шарды: {remaining_shards}"
            )
            
            # Получаем хранилище удаляемого шарда
            shard_storage = self._shard_storages.get(shard_id)
            
            if shard_storage:
                # Локальный шард - собираем все векторы и перераспределяем
                all_vectors = []
                namespaces = shard_storage.list_namespaces
                
                for namespace in namespaces:
                    vectors = shard_storage.namespace_map.get(namespace, [])
                    all_vectors.extend([(v, namespace) for v in vectors])
                
                if all_vectors:
                    self.logger.info(
                        f"Найдено {len(all_vectors)} векторов в шарде {shard_id} "
                        f"для перераспределения"
                    )
                    
                    # Группируем векторы по целевым шардам согласно стратегии
                    vectors_by_target_shard: Dict[str, List[tuple]] = {}
                    for shard in remaining_shards:
                        vectors_by_target_shard[shard] = []
                    
                    for vector, namespace in all_vectors:
                        # Определяем целевой шард для вектора (исключая удаляемый)
                        # Временно исключаем удаляемый шард из списка для правильного хеширования
                        target_shard = self._get_shard_for_vector_excluding(
                            vector, namespace, exclude_shard=shard_id
                        )
                        if target_shard and target_shard in vectors_by_target_shard:
                            vectors_by_target_shard[target_shard].append((vector, namespace))
                        else:
                            # Если не удалось определить, распределяем равномерно
                            import random
                            target_shard = random.choice(remaining_shards)
                            vectors_by_target_shard[target_shard].append((vector, namespace))
                            self.logger.warning(
                                f"Не удалось определить целевой шард для вектора {vector.id}, "
                                f"распределен на {target_shard}"
                            )
                    
                    # Перераспределяем векторы на целевые шарды
                    total_moved = 0
                    for target_shard, vectors_to_move in vectors_by_target_shard.items():
                        if not vectors_to_move:
                            continue
                        
                        # Используем логику из redistribute_data для перемещения
                        success = self._move_vectors_to_shard(
                            vectors_to_move, target_shard, from_storage=shard_storage
                        )
                        if success:
                            total_moved += len(vectors_to_move)
                            self.logger.info(
                                f"Перераспределено {len(vectors_to_move)} векторов "
                                f"из {shard_id} в {target_shard}"
                            )
                        else:
                            self.logger.error(
                                f"Ошибка перераспределения векторов в {target_shard}"
                            )
                            return False
                    
                    # Удаляем все векторы из исходного шарда после успешного перераспределения
                    deleted_count = 0
                    for vector, namespace in all_vectors:
                        if shard_storage.delete(vector.id, namespace):
                            deleted_count += 1
                    
                    self.logger.info(
                        f"Успешно перераспределено {total_moved} векторов из {shard_id}, "
                        f"удалено {deleted_count} векторов из исходного шарда"
                    )
                else:
                    self.logger.info(f"Шард {shard_id} пуст, перераспределение не требуется")
            else:
                # Удаленный шард - предупреждаем, что данные могут быть потеряны
                shard_info = self._shards[shard_id]
                if shard_info.url:
                    self.logger.warning(
                        f"Удаляемый шард {shard_id} является удаленным ({shard_info.url}). "
                        f"Данные на удаленном шарде не могут быть автоматически перераспределены. "
                        f"Убедитесь, что данные сохранены перед удалением."
                    )
            
            # Удаляем шард из словарей
            del self._shards[shard_id]
            if shard_id in self._shard_storages:
                del self._shard_storages[shard_id]
            
            # Обновляем счетчики векторов в оставшихся шардах
            for remaining_shard_id in remaining_shards:
                self.update_shard_vector_count(remaining_shard_id)
            
            self.logger.info(f"Шард {shard_id} успешно удален")
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
        return self._get_shard_for_id_excluding(vector_id, namespace, exclude_shard=None)
    
    def _get_shard_for_id_excluding(
        self,
        vector_id: UUID,
        namespace: str = "default",
        exclude_shard: Optional[str] = None
    ) -> Optional[str]:
        """Определить шард для вектора по его ID, исключая указанный шард."""
        with self._lock:
            healthy_shards = [
                shard_id for shard_id, shard in self._shards.items()
                if shard.is_healthy and shard_id != exclude_shard
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
                # Классический round-robin: циклическое распределение по шардам
                # Выбираем следующий шард по очереди
                shard_index = self._round_robin_counter % len(healthy_shards)
                # Инкрементируем счетчик для следующего запроса
                self._round_robin_counter = (self._round_robin_counter + 1) % len(healthy_shards)
                return healthy_shards[shard_index]
            else:
                # По умолчанию используем хеширование
                key = f"{namespace}:{vector_id}"
                hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
                shard_index = hash_value % len(healthy_shards)
                return healthy_shards[shard_index]
    
    def _get_shard_for_vector_excluding(
        self,
        vector: VectorProtocol,
        namespace: str = "default",
        exclude_shard: Optional[str] = None
    ) -> Optional[str]:
        """Определить шард для вектора, исключая указанный шард."""
        return self._get_shard_for_id_excluding(vector.id, namespace, exclude_shard)
    
    def _move_vectors_to_shard(
        self,
        vectors_with_namespace: List[tuple],
        target_shard_id: str,
        from_storage: Optional[StorageEngine] = None
    ) -> bool:
        """
        Переместить векторы на целевой шард.
        
        Args:
            vectors_with_namespace: Список кортежей (вектор, namespace)
            target_shard_id: ID целевого шарда
            from_storage: Хранилище исходного шарда (опционально, для проверки)
            
        Returns:
            True если перемещение успешно
        """
        if target_shard_id not in self._shards:
            self.logger.error(f"Целевой шард {target_shard_id} не найден")
            return False
        
        target_shard_info = self._shards[target_shard_id]
        target_storage = self._shard_storages.get(target_shard_id)
        
        try:
            if target_storage:
                # Локальный целевой шард - записываем напрямую
                by_namespace: Dict[str, List[VectorProtocol]] = {}
                for vector, namespace in vectors_with_namespace:
                    if namespace not in by_namespace:
                        by_namespace[namespace] = []
                    by_namespace[namespace].append(vector)
                
                # Проверяем на дубликаты
                moved_count = 0
                for namespace, vectors in by_namespace.items():
                    existing_vectors = target_storage.namespace_map.get(namespace, [])
                    existing_ids = {v.id for v in existing_vectors}
                    
                    # Фильтруем только новые векторы
                    new_vectors = [v for v in vectors if v.id not in existing_ids]
                    
                    if new_vectors:
                        target_storage.write_vectors(new_vectors, namespace)
                        moved_count += len(new_vectors)
                
                return moved_count > 0 or len(vectors_with_namespace) == 0
                
            elif target_shard_info.url:
                # Удаленный целевой шард - отправляем через HTTP API
                by_namespace: Dict[str, List[VectorProtocol]] = {}
                for vector, namespace in vectors_with_namespace:
                    if namespace not in by_namespace:
                        by_namespace[namespace] = []
                    by_namespace[namespace].append(vector)
                
                for namespace, vectors in by_namespace.items():
                    vectors_data = {
                        "vectors": [
                            {
                                "values": v.values.tolist() if hasattr(v.values, 'tolist') else list(v.values),
                                "metadata": dict(v.metadata)
                            }
                            for v in vectors
                        ]
                    }
                    
                    if self._session:
                        url = f"{target_shard_info.url}/vectors/batch?namespace={namespace}"
                        response = self._session.put(
                            url,
                            json=vectors_data,
                            timeout=self._request_timeout * len(vectors)
                        )
                        
                        if response.status_code != 200:
                            self.logger.error(
                                f"Ошибка отправки векторов на удаленный шард {target_shard_id}: "
                                f"HTTP {response.status_code}"
                            )
                            return False
                
                return True
            else:
                self.logger.error(f"Целевой шард {target_shard_id} не настроен")
                return False
                
        except Exception as e:
            self.logger.error(
                f"Ошибка перемещения векторов в {target_shard_id}: {e}",
                exc_info=True
            )
            return False
    
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
            # согласно текущей стратегии шардирования
            vectors_to_move = []
            vectors_to_keep = []
            shard_distribution = {}  # Для статистики распределения
            
            for vector, namespace in all_vectors:
                target_shard = self.get_shard_for_vector(vector, namespace)
                if target_shard is None:
                    self.logger.warning(
                        f"Не удалось определить целевой шард для вектора {vector.id}, "
                        f"пропускаем его"
                    )
                    vectors_to_keep.append((vector, namespace))
                    continue
                
                # Собираем статистику распределения
                if target_shard not in shard_distribution:
                    shard_distribution[target_shard] = 0
                shard_distribution[target_shard] += 1
                
                if target_shard == to_shard:
                    vectors_to_move.append((vector, namespace))
                else:
                    vectors_to_keep.append((vector, namespace))
            
            # Логируем статистику распределения
            self.logger.info(
                f"Распределение векторов по шардам (согласно стратегии '{self._sharding_strategy}'): "
                + ", ".join([f"{shard}: {count}" for shard, count in shard_distribution.items()])
            )
            
            if not vectors_to_move:
                self.logger.info(
                    f"Нет векторов для перемещения из {from_shard} в {to_shard}. "
                    f"Согласно стратегии шардирования, векторы распределены по другим шардам."
                )
                return True
            
            self.logger.info(
                f"Будет перемещено {len(vectors_to_move)} векторов из {from_shard} в {to_shard}, "
                f"останется {len(vectors_to_keep)} векторов в {from_shard}"
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
                
                # Проверяем на дубликаты и записываем только новые векторы
                moved_count = 0
                skipped_count = 0
                for namespace, vectors in by_namespace.items():
                    # Проверяем, какие векторы уже существуют на целевом шарде
                    existing_vectors = to_storage.namespace_map.get(namespace, [])
                    existing_ids = {v.id for v in existing_vectors}
                    
                    # Фильтруем только те векторы, которых еще нет
                    new_vectors = [v for v in vectors if v.id not in existing_ids]
                    skipped = len(vectors) - len(new_vectors)
                    
                    if new_vectors:
                        to_storage.write_vectors(new_vectors, namespace)
                        moved_count += len(new_vectors)
                        self.logger.debug(
                            f"Перемещено {len(new_vectors)} векторов в локальный шард {to_shard} "
                            f"(пространство: {namespace})"
                        )
                    
                    if skipped > 0:
                        skipped_count += skipped
                        self.logger.warning(
                            f"Пропущено {skipped} векторов, которые уже существуют на целевом шарде {to_shard} "
                            f"(пространство: {namespace})"
                        )
                
                # Обновляем счетчик векторов в целевом шарде
                self.update_shard_vector_count(to_shard)
                
                if skipped_count > 0:
                    self.logger.info(
                        f"Перераспределение: перемещено {moved_count} векторов, "
                        f"пропущено {skipped_count} дубликатов"
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
            # Для локальных шардов удаляем только те векторы, которые были успешно перемещены
            # Для удаленных шардов удаляем все, так как успешность уже проверена через HTTP ответ
            deleted_count = 0
            vectors_to_delete = []
            
            if to_storage:
                # Для локального шарда проверяем наличие на целевом шарде перед удалением
                for vector, namespace in vectors_to_move:
                    existing_vectors = to_storage.namespace_map.get(namespace, [])
                    if any(v.id == vector.id for v in existing_vectors):
                        vectors_to_delete.append((vector.id, namespace))
            else:
                # Для удаленного шарда все векторы считаются успешно перемещенными
                # (проверка была выполнена через HTTP статус код)
                vectors_to_delete = [(v.id, ns) for v, ns in vectors_to_move]
            
            # Удаляем векторы из исходного шарда
            for vector_id, namespace in vectors_to_delete:
                if from_storage.delete(vector_id, namespace):
                    deleted_count += 1
            
            # Обновляем счетчик векторов в исходном шарде
            self.update_shard_vector_count(from_shard)
            
            self.logger.info(
                f"Перераспределение завершено: перемещено {len(vectors_to_move)} векторов, "
                f"удалено {deleted_count} векторов из {from_shard} в {to_shard}"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Ошибка перераспределения данных {from_shard} -> {to_shard}: {e}",
                exc_info=True
            )
            return False
    
    def redistribute_all_data(self) -> bool:
        """
        Перераспределить все векторы между всеми шардами согласно стратегии шардирования.
        
        Собирает все векторы из всех локальных шардов, определяет для каждого вектора
        правильный шард согласно текущей стратегии шардирования, и перемещает векторы
        на правильные шарды. Векторы, которые уже находятся на правильных шардах, не перемещаются.
        
        Returns:
            True если перераспределение успешно
        """
        with self._lock:
            # Получаем список всех здоровых локальных шардов
            local_shards = [
                shard_id for shard_id, shard in self._shards.items()
                if shard_id in self._shard_storages and shard.is_healthy
            ]
            
            if len(local_shards) <= 1:
                self.logger.info("Недостаточно шардов для перераспределения (нужно минимум 2)")
                return True  # Не ошибка, просто нечего перераспределять
            
            self.logger.info(
                f"Начинается перераспределение всех векторов между шардами: {local_shards}"
            )
            
            # Собираем все векторы из всех локальных шардов
            all_vectors_by_shard: Dict[str, List[tuple]] = {}
            total_vectors = 0
            
            for shard_id in local_shards:
                shard_storage = self._shard_storages.get(shard_id)
                if not shard_storage:
                    continue
                
                vectors_in_shard = []
                namespaces = shard_storage.list_namespaces
                
                for namespace in namespaces:
                    vectors = shard_storage.namespace_map.get(namespace, [])
                    vectors_in_shard.extend([(v, namespace) for v in vectors])
                
                if vectors_in_shard:
                    all_vectors_by_shard[shard_id] = vectors_in_shard
                    total_vectors += len(vectors_in_shard)
                    self.logger.info(
                        f"Найдено {len(vectors_in_shard)} векторов в шарде {shard_id}"
                    )
            
            if total_vectors == 0:
                self.logger.info("Нет векторов для перераспределения")
                return True
            
            self.logger.info(f"Всего найдено {total_vectors} векторов для перераспределения")
            
            # Группируем векторы по целевым шардам согласно стратегии
            vectors_by_target_shard: Dict[str, List[tuple]] = {}
            vectors_to_keep: Dict[str, List[tuple]] = {}  # Векторы, которые уже на правильных шардах
            
            for shard_id in local_shards:
                vectors_by_target_shard[shard_id] = []
                vectors_to_keep[shard_id] = []
            
            # Определяем правильный шард для каждого вектора
            for source_shard_id, vectors in all_vectors_by_shard.items():
                for vector, namespace in vectors:
                    # Определяем целевой шард для вектора согласно стратегии
                    target_shard = self.get_shard_for_vector(vector, namespace)
                    
                    if target_shard is None:
                        self.logger.warning(
                            f"Не удалось определить целевой шард для вектора {vector.id}, "
                            f"оставляем на текущем шарде {source_shard_id}"
                        )
                        vectors_to_keep[source_shard_id].append((vector, namespace))
                        continue
                    
                    if target_shard not in vectors_by_target_shard:
                        # Целевой шард не является локальным - оставляем на текущем
                        self.logger.warning(
                            f"Целевой шард {target_shard} для вектора {vector.id} не является локальным, "
                            f"оставляем на текущем шарде {source_shard_id}"
                        )
                        vectors_to_keep[source_shard_id].append((vector, namespace))
                        continue
                    
                    if target_shard == source_shard_id:
                        # Вектор уже на правильном шарде
                        vectors_to_keep[source_shard_id].append((vector, namespace))
                    else:
                        # Вектор нужно переместить
                        vectors_by_target_shard[target_shard].append((vector, namespace))
            
            # Подсчитываем статистику
            total_to_move = sum(len(vectors) for vectors in vectors_by_target_shard.values())
            total_to_keep = sum(len(vectors) for vectors in vectors_to_keep.values())
            
            self.logger.info(
                f"Статистика перераспределения: "
                f"будет перемещено {total_to_move} векторов, "
                f"останется на месте {total_to_keep} векторов"
            )
            
            if total_to_move == 0:
                self.logger.info("Все векторы уже находятся на правильных шардах")
                return True
            
            # Создаем карту векторов к исходным шардам для быстрого поиска
            vector_to_source_shard: Dict[UUID, str] = {}
            for source_shard_id, vectors in all_vectors_by_shard.items():
                for vector, namespace in vectors:
                    vector_to_source_shard[vector.id] = source_shard_id
            
            # Перераспределяем векторы на целевые шарды
            total_moved = 0
            for target_shard, vectors_to_move in vectors_by_target_shard.items():
                if not vectors_to_move:
                    continue
                
                # Используем логику из remove_shard для перемещения
                # from_storage не обязателен, так как мы удалим векторы отдельно
                success = self._move_vectors_to_shard(
                    vectors_to_move, target_shard, from_storage=None
                )
                
                if success:
                    total_moved += len(vectors_to_move)
                    self.logger.info(
                        f"Перераспределено {len(vectors_to_move)} векторов в {target_shard}"
                    )
                else:
                    self.logger.error(
                        f"Ошибка перераспределения векторов в {target_shard}"
                    )
                    return False
                
                # Удаляем перемещенные векторы из исходных шардов
                # Группируем по исходным шардам для эффективного удаления
                vectors_by_source: Dict[str, List[tuple]] = {}
                for vector, namespace in vectors_to_move:
                    source_shard_id = vector_to_source_shard.get(vector.id)
                    if source_shard_id:
                        if source_shard_id not in vectors_by_source:
                            vectors_by_source[source_shard_id] = []
                        vectors_by_source[source_shard_id].append((vector, namespace))
                
                # Удаляем векторы из исходных шардов
                for source_shard_id, vectors_to_delete in vectors_by_source.items():
                    source_storage = self._shard_storages.get(source_shard_id)
                    if source_storage:
                        deleted_count = 0
                        for vector, namespace in vectors_to_delete:
                            if source_storage.delete(vector.id, namespace):
                                deleted_count += 1
                        
                        if deleted_count > 0:
                            self.logger.debug(
                                f"Удалено {deleted_count} векторов из исходного шарда {source_shard_id}"
                            )
                            # Обновляем счетчик векторов в исходном шарде
                            self.update_shard_vector_count(source_shard_id)
                
                # Обновляем счетчик векторов в целевом шарде
                self.update_shard_vector_count(target_shard)
            
            self.logger.info(
                f"Успешно перераспределено {total_moved} векторов между всеми шардами"
            )
            return True
    
    def check_shard_health(self, shard_id: str) -> bool:
        """Проверить работоспособность шарда."""
        with self._lock:
            if shard_id not in self._shards:
                return False
            
            shard = self._shards[shard_id]
            
            # Локальные шарды - проверяем наличие StorageEngine
            if shard.url is None:
                has_storage = shard_id in self._shard_storages
                # Обновляем статус только если изменился
                if has_storage != shard.is_healthy:
                    shard.is_healthy = has_storage
                    if has_storage:
                        self.logger.debug(f"Локальный шард {shard_id} теперь здоров (StorageEngine найден)")
                    else:
                        self.logger.warning(
                            f"Локальный шард {shard_id} не имеет StorageEngine"
                        )
                return has_storage
            
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
                # Обновляем vector_count из StorageEngine для локальных шардов
                vector_count = shard.vector_count
                if shard.url is None and shard_id in self._shard_storages:
                    storage = self._shard_storages[shard_id]
                    # Подсчитываем векторы во всех namespace
                    total_vectors = 0
                    for namespace_vectors in storage.namespace_map.values():
                        total_vectors += len(namespace_vectors)
                    shard.vector_count = total_vectors
                    vector_count = total_vectors
                
                shard_info = {
                    "is_healthy": shard.is_healthy,
                    "is_local": shard.url is None,
                    "vector_count": vector_count
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
    
    def update_shard_vector_count(self, shard_id: str) -> None:
        """Обновить количество векторов в шарде из StorageEngine."""
        with self._lock:
            if shard_id not in self._shards:
                self.logger.warning(f"Шард {shard_id} не найден при обновлении vector_count")
                return
            
            shard = self._shards[shard_id]
            if shard.url is None and shard_id in self._shard_storages:
                # Локальный шард - подсчитываем векторы из StorageEngine
                storage = self._shard_storages[shard_id]
                total_vectors = 0
                for namespace_vectors in storage.namespace_map.values():
                    total_vectors += len(namespace_vectors)
                old_count = shard.vector_count
                shard.vector_count = total_vectors
                self.logger.debug(
                    f"Обновлен vector_count для шарда {shard_id}: {old_count} -> {total_vectors}"
                )
            else:
                self.logger.debug(
                    f"Не удалось обновить vector_count для шарда {shard_id}: "
                    f"url={shard.url}, has_storage={shard_id in self._shard_storages}"
                )
    
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
                "id": str(vector.id),
                "values": vector.values.tolist() if hasattr(vector.values, 'tolist') else list(vector.values),
                "metadata": dict(vector.metadata)
            }

            url = f"{shard.url}/vectors/copy?namespace={namespace}"
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
    
    def write_batch_to_remote_shard(
        self,
        shard_id: str,
        vectors_data: Dict[str, Any],
        namespace: str
    ) -> bool:
        """
        Записать batch векторов на удаленный шард через HTTP API.
        
        Args:
            shard_id: Идентификатор удаленного шарда
            vectors_data: Словарь с ключом "vectors" содержащий список векторов
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
            url = f"{shard.url}/vectors/copy/batch?namespace={namespace}"
            response = self._session.put(
                url,
                json=vectors_data,
                timeout=self._request_timeout * len(vectors_data.get("vectors", []))
            )
            
            if response.status_code == 200:
                return True
            else:
                self.logger.error(
                    f"Ошибка batch записи на удаленный шард {shard_id}: "
                    f"HTTP {response.status_code}"
                )
                return False
        except Exception as e:
            self.logger.error(
                f"Ошибка batch записи на удаленный шард {shard_id}: {e}",
                exc_info=True
            )
            return False
    
    def add_local_shard(self, shard_id: str, storage: StorageEngine) -> bool:
        """
        Добавить новый локальный шард с StorageEngine.
        
        Args:
            shard_id: Уникальный идентификатор шарда
            storage: StorageEngine для локального шарда
            
        Returns:
            True если шард успешно добавлен
        """
        with self._lock:
            if shard_id in self._shards:
                self.logger.warning(f"Шард {shard_id} уже существует")
                return False
            
            # Добавляем StorageEngine
            self._shard_storages[shard_id] = storage
            
            # Создаем ShardInfo для локального шарда
            shard = ShardInfo(shard_id, url=None)
            shard.is_healthy = True  # Локальные шарды всегда здоровы
            self._shards[shard_id] = shard
            
            self.logger.info(f"Добавлен локальный шард {shard_id}")
            return True
    
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
                results = response.json()
                # Добавляем информацию о шарде к каждому результату
                for result in results:
                    if "shard_id" not in result:
                        result["shard_id"] = shard_id
                return results
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
    
