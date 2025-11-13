from __future__ import annotations
from typing import Iterable, Sequence, List, Dict, Any
from uuid import UUID
from ..interfaces.vector import VectorDTO
from ..interfaces.index import IndexProtocol
from ..interfaces.query_processor import QueryProcessorProtocol
from ..interfaces.storage_engine import StorageEngine
from ..implementations.vector import Vector


class QueryProcessor(QueryProcessorProtocol):
    def __init__(self, storage_engine: StorageEngine, index: IndexProtocol):
        self._storage = storage_engine
        self._index = index

    def insert(self, vector: VectorDTO, namespace: str = "default") -> None:
        new_vec = Vector(values=vector.values, metadata=vector.metadata)
        self._storage.write(new_vec, namespace)
        self._index.add([new_vec], namespace)

    def upsert_many(self, vectors: Iterable[VectorDTO], namespace: str = "default") -> None:
        vecs = [Vector(values=v.values, metadata=v.metadata) for v in vectors]
        self._storage.write_vectors(vecs, namespace)
        self._index.add(vecs, namespace)

    def find_similar(
        self,
        query: VectorDTO,
        top_k: int,
        namespace: str = "default",
        metric: str = "cosine",
    ) -> List[dict]:
        search_results = self._index.search(query, top_k=top_k, namespace=namespace, metric=metric)
        if not search_results:
            return []
        ids = [res.vector_id for res in search_results]
        stored_vectors = list(self._storage.read_vectors(ids, namespace))
        vector_map = {v.id: v for v in stored_vectors if v}
        enriched = []
        for res in search_results:
            v = vector_map.get(res.vector_id)
            if v:
                enriched.append({
                    "id": v.id,
                    "values": v.values,
                    "metadata": v.metadata,
                    "score": res.score,
                })
        return enriched

    def delete(self, ids: Sequence[UUID], namespace: str = "default") -> Sequence[UUID]:
        del_id = []
        for vid in ids:
            if self._storage.delete(vid, namespace):
                del_id.append(vid)
        self._index.remove(ids, namespace)

        if getattr(self._index, "is_rebuild_required", None):
            if self._index.is_rebuild_required(namespace):
                source = {namespace: self._storage.namespace_map.get(namespace, [])}
                self._index.rebuild(source, metric=self._index._space)
        return del_id

    def list_namespaces(self) -> List[str]:
        return self._storage.list_namespaces

    def get_namespace_vectors(self, namespace: str) -> List[Dict[str, Any]]:
        vectors = self._storage.namespace_map.get(namespace, [])
        return [
            {
                "id": v.id,
                "values": v.values,
                "metadata": v.metadata,
            }
            for v in vectors
        ]

    def get_namespace_count(self, namespace: str) -> int:
        return len(self._storage.namespace_map.get(namespace, []))

    def get_storage_info(self) -> Dict[str, Any]:
        return self._storage.get_storage_info()
