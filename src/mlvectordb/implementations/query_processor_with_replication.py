"""
Расширенный QueryProcessor с поддержкой репликации и шардирования.
"""

from __future__ import annotations
from typing import Iterable, Sequence, List, Dict, Any, Optional
from uuid import UUID
import logging

from ..interfaces.vector import VectorDTO
from ..interfaces.index import IndexProtocol
from ..interfaces.query_processor import QueryProcessorProtocol
from ..interfaces.storage_engine import StorageEngine
from ..interfaces.replication import ReplicationManager
from ..interfaces.sharding import ShardingManager
from ..implementations.vector import Vector
from ..implementations.query_processor import QueryProcessor


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
    
    def insert(self, vector: VectorDTO, namespace: str = "default") -> None:
        """Вставка вектора с поддержкой репликации и шардирования."""
        new_vec = Vector(values=vector.values, metadata=vector.metadata)
        
        # Определяем шард для вектора
        if self._sharding_manager:
            shard_id = self._sharding_manager.get_shard_for_vector(new_vec, namespace)
            if shard_id is None:
                raise RuntimeError("Не удалось определить шард для вектора")
            
            # Получаем хранилище для шарда
            shard_storage = self._sharding_manager.get_shard_storage(shard_id)
            
            if shard_storage:
                # Локальный шард - записываем напрямую
                shard_storage.write(new_vec, namespace)
                self._storage.write(new_vec, namespace)  # Также в первичное хранилище
            else:
                # Удаленный шард - записываем через HTTP API
                success = self._sharding_manager.write_to_remote_shard(
                    shard_id, new_vec, namespace
                )
                if not success:
                    self.logger.warning(
                        f"Не удалось записать на удаленный шард {shard_id}, "
                        f"сохраняем в первичное хранилище"
                    )
                # Всегда сохраняем в первичное хранилище для индексации
                self._storage.write(new_vec, namespace)
        else:
            # Нет шардирования - используем обычную логику
            self._storage.write(new_vec, namespace)
        
        # Добавляем в индекс
        self._index.add([new_vec], namespace)
        
        # Реплицируем на другие реплики
        if self._replication_manager:
            try:
                results = self._replication_manager.replicate_write(new_vec, namespace)
                successful = sum(1 for v in results.values() if v)
                total = len(results)
                if successful < total:
                    self.logger.warning(
                        f"Репликация завершена частично: {successful}/{total} реплик"
                    )
            except Exception as e:
                self.logger.error(f"Ошибка репликации: {e}", exc_info=True)
                # Не прерываем выполнение, если репликация не удалась
    
    def upsert_many(
        self,
        vectors: Iterable[VectorDTO],
        namespace: str = "default"
    ) -> None:
        """Массовая вставка векторов с поддержкой репликации и шардирования."""
        vecs = [Vector(values=v.values, metadata=v.metadata) for v in vectors]
        vecs_list = list(vecs)
        
        if not vecs_list:
            return
        
        # Распределяем векторы по шардам
        if self._sharding_manager:
            shard_vectors: Dict[str, List[Vector]] = {}
            
            for vec in vecs_list:
                shard_id = self._sharding_manager.get_shard_for_vector(vec, namespace)
                if shard_id:
                    if shard_id not in shard_vectors:
                        shard_vectors[shard_id] = []
                    shard_vectors[shard_id].append(vec)
            
            # Записываем в соответствующие шарды
            for shard_id, shard_vecs in shard_vectors.items():
                shard_storage = self._sharding_manager.get_shard_storage(shard_id)
                if shard_storage:
                    # Локальный шард
                    shard_storage.write_vectors(shard_vecs, namespace)
                else:
                    # Удаленный шард - записываем через HTTP API
                    # Отправляем batch на удаленный шард
                    for vec in shard_vecs:
                        self._sharding_manager.write_to_remote_shard(
                            shard_id, vec, namespace
                        )
                # Также записываем в первичное хранилище для индексации
                self._storage.write_vectors(shard_vecs, namespace)
        else:
            # Нет шардирования
            self._storage.write_vectors(vecs_list, namespace)
        
        # Добавляем в индекс
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
        # используем их напрямую
        if all_results and isinstance(all_results[0], dict) and "values" in all_results[0]:
            # Результаты уже обогащены (из удаленных шардов)
            # Сортируем и берем top_k
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:top_k]
        
        # Обрабатываем результаты из индекса - нужно получить векторы
        ids = [UUID(res["id"]) if isinstance(res["id"], str) else res["id"] for res in all_results]
        
        # Получаем векторы из соответствующих шардов
        if self._sharding_manager:
            stored_vectors = []
            for vec_id in ids:
                shard_id = self._sharding_manager.get_shard_for_id(vec_id, namespace)
                if shard_id:
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
                else:
                    # Пробуем получить из первичного хранилища
                    vec = self._storage.read(vec_id, namespace)
                    if vec:
                        stored_vectors.append(vec)
        else:
            stored_vectors = list(self._storage.read_vectors(ids, namespace))
        
        # Обогащаем результаты
        vector_map = {v.id: v for v in stored_vectors if v}
        enriched = []
        for res in all_results:
            vec_id = UUID(res["id"]) if isinstance(res["id"], str) else res["id"]
            score = res["score"]
            v = vector_map.get(vec_id)
            if v:
                enriched.append({
                    "id": v.id,
                    "values": v.values.tolist() if hasattr(v.values, 'tolist') else list(v.values),
                    "metadata": dict(v.metadata),
                    "score": score,
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
            for vid in ids:
                shard_id = self._sharding_manager.get_shard_for_id(vid, namespace)
                if shard_id:
                    shard_storage = self._sharding_manager.get_shard_storage(shard_id)
                    if shard_storage:
                        if shard_storage.delete(vid, namespace):
                            del_ids.append(vid)
                    # Также удаляем из первичного хранилища
                    if self._storage.delete(vid, namespace):
                        if vid not in del_ids:
                            del_ids.append(vid)
                else:
                    # Пробуем удалить из первичного хранилища
                    if self._storage.delete(vid, namespace):
                        del_ids.append(vid)
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
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Получить информацию о хранилище с учетом репликации и шардирования."""
        info = super().get_storage_info()
        
        if self._replication_manager:
            info["replication"] = self._replication_manager.get_replica_info()
        
        if self._sharding_manager:
            info["sharding"] = self._sharding_manager.get_shard_info()
        
        return info

