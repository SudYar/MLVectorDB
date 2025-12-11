"""
Расширенный QueryProcessor с поддержкой репликации и шардирования.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence, List, Dict, Any, Optional
from uuid import UUID

from ..implementations.query_processor import QueryProcessor
from ..implementations.vector import Vector
from ..interfaces.index import IndexProtocol
from ..interfaces.replication import ReplicationManager
from ..interfaces.sharding import ShardingManager
from ..interfaces.storage_engine import StorageEngine
from ..interfaces.vector import VectorDTO


class QueryProcessorWithReplication(QueryProcessor):
    """
    Расширенный QueryProcessor с поддержкой репликации и шардирования.
    
    Этот класс расширяет базовый QueryProcessor, добавляя:
    - Репликацию данных на другие реплики при записи
    - Шардирование данных по нескольким шардам
    - Агрегацию результатов поиска из всех шардов
    """
    
    def __init__(
        self,
        storage_engine: StorageEngine,
        index: IndexProtocol,
        replication_manager: Optional[ReplicationManager] = None,
        sharding_manager: Optional[ShardingManager] = None
    ):
        """
        Инициализация QueryProcessor с репликацией и шардированием.
        
        Args:
            storage_engine: Первичное хранилище (используется как основной шард)
            index: Индекс для поиска
            replication_manager: Менеджер репликации (опционально)
            sharding_manager: Менеджер шардирования (опционально)
        """
        super().__init__(storage_engine, index)
        self._replication_manager = replication_manager
        self._sharding_manager = sharding_manager
        self.logger = logging.getLogger(__name__)
        
        # Если есть шардирование, нужно создать индексы для каждого шарда
        # В текущей реализации используем один индекс для всех шардов
        # Это можно улучшить в будущем

    def insert_new(self, vector: VectorDTO, namespace: str = "default") -> None:
        """Вставка вектора с поддержкой репликации и шардирования."""
        new_vec = Vector(values=vector.values, metadata=vector.metadata)

        self.insert(new_vec, namespace)

    def insert(self, vector: Vector, namespace: str = "default") -> None:
        # Определяем шард для вектора
        if self._sharding_manager:
            shard_id = self._sharding_manager.get_shard_for_vector(vector, namespace)
            if shard_id is None:
                raise RuntimeError("Не удалось определить шард для вектора")

            # Получаем хранилище для шарда
            shard_storage = self._sharding_manager.get_shard_storage(shard_id)

            if shard_storage:
                # Локальный шард - записываем ТОЛЬКО в шард
                shard_storage.write(vector, namespace)
                # Обновляем счетчик векторов в шарде
                self._sharding_manager.update_shard_vector_count(shard_id)
                self.logger.info(f"Вектор {vector.id} записан в локальный шард {shard_id}, namespace={namespace}")
            else:
                # Удаленный шард - записываем через HTTP API
                success = self._sharding_manager.write_to_remote_shard(
                    shard_id, vector, namespace
                )
                if not success:
                    # Если не удалось записать на удаленный шард, сохраняем в первичное хранилище как fallback
                    self.logger.warning(
                        f"Не удалось записать на удаленный шард {shard_id}, "
                        f"сохраняем в первичное хранилище как fallback"
                    )
                    self._storage.write(vector, namespace)
                else:
                    self.logger.debug(f"Вектор {vector.id} записан на удаленный шард {shard_id}")
        else:
            # Нет шардирования - используем обычную логику
            self._storage.write(vector, namespace)

        # Добавляем в индекс (индекс работает со всеми векторами независимо от шардирования)
        self._index.add([vector], namespace)

        # Реплицируем на другие реплики
        if self._replication_manager:
            try:
                results = self._replication_manager.replicate_write(vector, namespace)
                successful = sum(1 for v in results.values() if v)
                total = len(results)
                if successful < total:
                    self.logger.warning(
                        f"Репликация завершена частично: {successful}/{total} реплик"
                    )
            except Exception as e:
                self.logger.error(f"Ошибка репликации: {e}", exc_info=True)
                # Не прерываем выполнение, если репликация не удалась

    def upsert_many_new(self,
                        vectors: Iterable[VectorDTO],
                        namespace: str = "default") -> None:
        vecs = [Vector(values=v.values, metadata=v.metadata) for v in vectors]
        self.upsert_many(vecs, namespace)

    def upsert_many(
        self,
            vectors: Iterable[Vector],
        namespace: str = "default"
    ) -> None:
        """Массовая вставка векторов с поддержкой репликации и шардирования."""
        vecs_list = list(vectors)
        
        if not vecs_list:
            return
        
        # Распределяем векторы по шардам
        if self._sharding_manager:
            shard_vectors: Dict[str, List[Vector]] = {}
            fallback_vectors: List[Vector] = []  # Векторы для fallback в первичное хранилище
            
            for vec in vecs_list:
                shard_id = self._sharding_manager.get_shard_for_vector(vec, namespace)
                if shard_id:
                    if shard_id not in shard_vectors:
                        shard_vectors[shard_id] = []
                    shard_vectors[shard_id].append(vec)
                else:
                    # Если не удалось определить шард, используем fallback
                    fallback_vectors.append(vec)
            
            # Записываем в соответствующие шарды
            for shard_id, shard_vecs in shard_vectors.items():
                shard_storage = self._sharding_manager.get_shard_storage(shard_id)
                if shard_storage:
                    # Локальный шард - записываем ТОЛЬКО в шард
                    shard_storage.write_vectors(shard_vecs, namespace)
                    # Обновляем счетчик векторов в шарде
                    self._sharding_manager.update_shard_vector_count(shard_id)
                    self.logger.debug(
                        f"Записано {len(shard_vecs)} векторов в локальный шард {shard_id}"
                    )
                else:
                    # Удаленный шард - записываем через HTTP API batch
                    # Группируем векторы для batch запроса
                    vectors_data = {
                        "vectors": [
                            {
                                "id": str(v.id),
                                "values": v.values.tolist() if hasattr(v.values, 'tolist') else list(v.values),
                                "metadata": dict(v.metadata)
                            }
                            for v in shard_vecs
                        ]
                    }
                    
                    # Отправляем batch на удаленный шард
                    success = self._sharding_manager.write_batch_to_remote_shard(
                        shard_id, vectors_data, namespace
                    )
                    if not success:
                        # Если не удалось записать на удаленный шард, добавляем в fallback
                        self.logger.warning(
                            f"Не удалось записать batch на удаленный шард {shard_id}, "
                            f"используем fallback"
                        )
                        fallback_vectors.extend(shard_vecs)
                    else:
                        self.logger.debug(
                            f"Записано {len(shard_vecs)} векторов на удаленный шард {shard_id}"
                        )
            
            # Записываем fallback векторы в первичное хранилище
            if fallback_vectors:
                self._storage.write_vectors(fallback_vectors, namespace)
                self.logger.debug(
                    f"Записано {len(fallback_vectors)} векторов в первичное хранилище (fallback)"
                )
        else:
            # Нет шардирования
            self._storage.write_vectors(vecs_list, namespace)
        
        # Добавляем в индекс (индекс работает со всеми векторами независимо от шардирования)
        self._index.add(vecs_list, namespace)
        
        # Реплицируем на другие реплики
        if self._replication_manager:
            try:
                results = self._replication_manager.replicate_batch_write(
                    vecs_list, namespace
                )
                successful = sum(1 for v in results.values() if v)
                total = len(results)
                if successful < total:
                    self.logger.warning(
                        f"Батч-репликация завершена частично: {successful}/{total} реплик"
                    )
            except Exception as e:
                self.logger.error(f"Ошибка батч-репликации: {e}", exc_info=True)
    
    def find_similar(
        self,
        query: VectorDTO,
        top_k: int,
        namespace: str = "default",
        metric: str = "cosine",
    ) -> List[dict]:
        """Поиск похожих векторов с поддержкой шардирования."""
        # Если есть шардирование, нужно искать во всех шардах
        all_results = []
        search_results = None
        
        if self._sharding_manager:
            shard_ids = self._sharding_manager.get_shards_for_search(namespace)
            local_shard_ids = []
            
            # Ищем в каждом шарде
            for shard_id in shard_ids:
                shard_storage = self._sharding_manager.get_shard_storage(shard_id)
                
                if shard_storage:
                    # Локальный шард - используем общий индекс
                    local_shard_ids.append(shard_id)
                else:
                    # Удаленный шард - выполняем поиск через HTTP API
                    remote_results = self._sharding_manager.search_on_remote_shard(
                        shard_id, query, top_k, namespace, metric
                    )
                    # Результаты уже в формате словарей с полными данными
                    all_results.extend(remote_results)
            
            # Используем общий индекс для локальных шардов
            if local_shard_ids:
                local_search_results = self._index.search(
                    query, top_k=top_k * len(shard_ids), namespace=namespace, metric=metric
                )
                search_results = local_search_results
        else:
            # Нет шардирования - обычный поиск
            search_results = self._index.search(
                query, top_k=top_k, namespace=namespace, metric=metric
            )
        
        # Если есть результаты из удаленных шардов, они уже обогащены
        # Проверяем, нужно ли обрабатывать результаты из индекса
        if search_results:
            # Конвертируем результаты индекса в формат словарей
            for res in search_results:
                all_results.append({
                    "id": str(res.vector_id),
                    "score": res.score
                })
        
        if not all_results:
            return []
        
        # Если результаты уже в формате словарей с полными данными (из удаленных шардов),
        # используем их напрямую, но нужно добавить информацию о шарде
        if all_results and isinstance(all_results[0], dict) and "values" in all_results[0]:
            # Результаты уже обогащены (из удаленных шардов)
            # Добавляем информацию о шарде, если её нет
            if self._sharding_manager:
                for result in all_results:
                    if "shard_id" not in result:
                        vec_id = UUID(result["id"]) if isinstance(result["id"], str) else result["id"]
                        shard_id = self._sharding_manager.get_shard_for_id(vec_id, namespace)
                        result["shard_id"] = shard_id
            # Сортируем и берем top_k
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:top_k]
        
        # Обрабатываем результаты из индекса - нужно получить векторы
        ids = [UUID(res["id"]) if isinstance(res["id"], str) else res["id"] for res in all_results]
        
        # Получаем векторы из соответствующих шардов
        # Также отслеживаем, из какого шарда был получен каждый вектор
        vector_to_shard_map = {}  # {vector_id: shard_id}
        
        if self._sharding_manager:
            stored_vectors = []
            for vec_id in ids:
                shard_id = self._sharding_manager.get_shard_for_id(vec_id, namespace)
                if shard_id:
                    vector_to_shard_map[vec_id] = shard_id
                    shard_storage = self._sharding_manager.get_shard_storage(shard_id)
                    if shard_storage:
                        # Локальный шард
                        vec = shard_storage.read(vec_id, namespace)
                        if vec:
                            stored_vectors.append(vec)
                    else:
                        # Удаленный шард - читаем через HTTP API
                        vec = self._sharding_manager.read_from_remote_shard(
                            shard_id, vec_id, namespace
                        )
                        if vec:
                            stored_vectors.append(vec)
                        else:
                            # Fallback на первичное хранилище
                            vec = self._storage.read(vec_id, namespace)
                            if vec:
                                stored_vectors.append(vec)
                                vector_to_shard_map[vec_id] = None  # Из первичного хранилища
                else:
                    # Пробуем получить из первичного хранилища
                    vec = self._storage.read(vec_id, namespace)
                    if vec:
                        stored_vectors.append(vec)
                        vector_to_shard_map[vec_id] = None  # Из первичного хранилища
        else:
            stored_vectors = list(self._storage.read_vectors(ids, namespace))
            # Нет шардирования - все векторы из первичного хранилища
            for vec_id in ids:
                vector_to_shard_map[vec_id] = None
        
        # Обогащаем результаты
        vector_map = {v.id: v for v in stored_vectors if v}
        enriched = []
        for res in all_results:
            vec_id = UUID(res["id"]) if isinstance(res["id"], str) else res["id"]
            score = res["score"]
            v = vector_map.get(vec_id)
            if v:
                shard_id = vector_to_shard_map.get(vec_id)
                enriched.append({
                    "id": v.id,
                    "values": v.values.tolist() if hasattr(v.values, 'tolist') else list(v.values),
                    "metadata": dict(v.metadata),
                    "score": score,
                    "shard_id": shard_id,
                })
        
        # Сортируем по score (убывание) и берем top_k
        enriched.sort(key=lambda x: x["score"], reverse=True)
        return enriched[:top_k]
    
    def delete(
        self,
        ids: Sequence[UUID],
        namespace: str = "default"
    ) -> Sequence[UUID]:
        """Удаление векторов с поддержкой репликации и шардирования."""
        del_ids = []
        
        # Удаляем из соответствующих шардов
        if self._sharding_manager:
            updated_shards = set()  # Отслеживаем шарды, в которых были удаления
            for vid in ids:
                shard_id = self._sharding_manager.get_shard_for_id(vid, namespace)
                if shard_id:
                    shard_storage = self._sharding_manager.get_shard_storage(shard_id)
                    if shard_storage:
                        if shard_storage.delete(vid, namespace):
                            del_ids.append(vid)
                            updated_shards.add(shard_id)
                    # Также удаляем из первичного хранилища (fallback векторы)
                    if self._storage.delete(vid, namespace):
                        if vid not in del_ids:
                            del_ids.append(vid)
                else:
                    # Пробуем удалить из первичного хранилища
                    if self._storage.delete(vid, namespace):
                        del_ids.append(vid)
            
            # Обновляем счетчики векторов в шардах
            for shard_id in updated_shards:
                self._sharding_manager.update_shard_vector_count(shard_id)
        else:
            # Нет шардирования
            for vid in ids:
                if self._storage.delete(vid, namespace):
                    del_ids.append(vid)
        
        # Удаляем из индекса
        self._index.remove(del_ids, namespace)
        
        # Реплицируем удаление на другие реплики
        if self._replication_manager:
            for vid in del_ids:
                try:
                    self._replication_manager.replicate_delete(vid, namespace)
                except Exception as e:
                    self.logger.error(
                        f"Ошибка репликации удаления для {vid}: {e}",
                        exc_info=True
                    )
        
        # Проверяем необходимость пересборки индекса
        if getattr(self._index, "is_rebuild_required", None):
            if self._index.is_rebuild_required(namespace):
                source = {namespace: self._storage.namespace_map.get(namespace, [])}
                self._index.rebuild(source, metric=self._index._space)
        
        return del_ids
    
    def list_namespaces(self) -> List[str]:
        """Получить список всех namespace с учетом шардирования."""
        all_namespaces = set()
        
        # Если есть шардирование, собираем namespace из всех шардов
        if self._sharding_manager:
            shard_ids = self._sharding_manager.list_shards()
            
            for shard_id in shard_ids:
                shard_storage = self._sharding_manager.get_shard_storage(shard_id)
                if shard_storage:
                    # Локальный шард - получаем namespace
                    namespaces = shard_storage.list_namespaces
                    all_namespaces.update(namespaces)
            
            # Также проверяем первичное хранилище (для fallback векторов)
            primary_namespaces = self._storage.list_namespaces
            all_namespaces.update(primary_namespaces)
        else:
            # Нет шардирования - используем обычную логику
            all_namespaces = set(self._storage.list_namespaces)
        
        return sorted(list(all_namespaces))
    
    def get_namespace_vectors(self, namespace: str) -> List[Dict[str, Any]]:
        """Получить все векторы из namespace с учетом шардирования."""
        all_vectors_with_shard = []  # Список кортежей (вектор, shard_id)
        
        # Если есть шардирование, собираем векторы из всех шардов
        if self._sharding_manager:
            shard_info_dict = self._sharding_manager.get_shard_info()
            shard_ids = self._sharding_manager.list_shards()
            
            for shard_id in shard_ids:
                shard_details = shard_info_dict.get("shards", {}).get(shard_id, {})
                is_local = shard_details.get("is_local", True)
                
                # Проверяем локальный шард
                shard_storage = self._sharding_manager.get_shard_storage(shard_id)
                if shard_storage:
                    # Локальный шард - читаем напрямую
                    vectors = shard_storage.namespace_map.get(namespace, [])
                    for vec in vectors:
                        all_vectors_with_shard.append((vec, shard_id))
                
                # Проверяем удаленный шард
                elif not is_local and shard_details.get("url"):
                    # Удаленный шард - получаем через HTTP
                    try:
                        # Используем имплементацию напрямую для доступа к сессии
                        from ..implementations.sharding_manager import ShardingManagerImpl
                        if isinstance(self._sharding_manager, ShardingManagerImpl):
                            shard_url = shard_details["url"]
                            if hasattr(self._sharding_manager, '_session') and self._sharding_manager._session:
                                url = f"{shard_url}/namespaces/vectors?namespace={namespace}"
                                response = self._sharding_manager._session.get(
                                    url,
                                    timeout=self._sharding_manager._request_timeout
                                )
                                if response.status_code == 200:
                                    remote_vectors = response.json()
                                    for vec_data in remote_vectors:
                                        # Создаем вектор из данных
                                        vec = Vector(
                                            values=vec_data["values"],
                                            metadata=vec_data.get("metadata", {})
                                        )
                                        all_vectors_with_shard.append((vec, shard_id))
                    except Exception as e:
                        self.logger.warning(
                            f"Не удалось получить векторы с удаленного шарда {shard_id}: {e}"
                        )
            
            # Также проверяем первичное хранилище (для fallback векторов)
            fallback_vectors = self._storage.namespace_map.get(namespace, [])
            # Добавляем только те векторы, которых нет в шардах
            shard_vector_ids = {v.id for v, _ in all_vectors_with_shard}
            for vec in fallback_vectors:
                if vec.id not in shard_vector_ids:
                    # Векторы из первичного хранилища не имеют шарда
                    all_vectors_with_shard.append((vec, None))
        else:
            # Нет шардирования - используем обычную логику
            vectors = self._storage.namespace_map.get(namespace, [])
            for vec in vectors:
                all_vectors_with_shard.append((vec, None))
        
        # Конвертируем в формат словарей с информацией о шарде
        return [
            {
                "id": v.id,
                "values": v.values.tolist() if hasattr(v.values, 'tolist') else list(v.values),
                "metadata": dict(v.metadata),
                "shard_id": shard_id,
            }
            for v, shard_id in all_vectors_with_shard
        ]
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Получить информацию о хранилище с учетом репликации и шардирования."""
        info = super().get_storage_info()
        
        if self._replication_manager:
            info["replication"] = self._replication_manager.get_replica_info()
        
        if self._sharding_manager:
            info["sharding"] = self._sharding_manager.get_shard_info()
        
        return info

