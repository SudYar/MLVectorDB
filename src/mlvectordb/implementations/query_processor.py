from __future__ import annotations
from typing import Iterable, Sequence, List
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
        for v in vecs:
            self._storage.write(v, namespace)
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
        stored_vectors = list(self._storage.read_batch(ids, namespace))
        vector_map = {v.id: v for v in stored_vectors}
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

    def delete(self, ids: Sequence[UUID], namespace: str = "default") -> None:
        for vid in ids:
            self._storage.delete(vid, namespace)
        self._index.remove(ids, namespace)
