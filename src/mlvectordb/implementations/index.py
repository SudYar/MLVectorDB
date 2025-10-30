from typing import Iterable, Sequence, List, Dict, Optional
import numpy as np
import hnswlib
from ..interfaces.index import IndexProtocol, SearchResultProtocol
from ..interfaces.vector import VectorProtocol

class SearchResult(SearchResultProtocol):
    def __init__(self, id: str, score: float, vector: VectorProtocol):
        self.id = id
        self.score = score
        self.vector = vector

class HNSWIndex(IndexProtocol):
    def __init__(self, dim: int, metric: str = "cosine") -> None:
        self.dim = dim
        self.metric = metric
        self._spaces: Dict[str, hnswlib.Index] = {}
        # TODO: load vectors from storage
        self._vectors: Dict[str, Dict[int, VectorProtocol]] = {}
        self._next_id: Dict[str, int] = {}

    def _init_namespace(self, namespace: str) -> None:
        if namespace in self._spaces:
            return
        index = hnswlib.Index(space=self.metric, dim=self.dim)
        index.init_index(max_elements=200, ef_construction=10, M=16)
        index.set_ef(10)
        self._spaces[namespace] = index
        self._vectors[namespace] = {}
        self._next_id[namespace] = 0

    def add(self, vectors: Iterable[VectorProtocol]) -> None:
        grouped: Dict[str, List[VectorProtocol]] = {}
        for v in vectors:
            grouped.setdefault(v.namespace, []).append(v)

        for ns, vecs in grouped.items():
            self._init_namespace(ns)
            index = self._spaces[ns]
            ids = []
            data = []
            for v in vecs:
                internal_id = self._next_id[ns]
                self._next_id[ns] += 1
                ids.append(internal_id)
                data.append(v.to_numpy())
                self._vectors[ns][internal_id] = v
            index.add_items(np.vstack(data), ids)

    def remove(self, ids: Sequence[str], namespace: str = "default") -> None:
        if namespace not in self._vectors:
            return
        kept = [v for v in self._vectors[namespace].values() if v.id not in ids]
        self.rebuild(kept, metric=self.metric)

    def search(self, query: np.ndarray, top_k: int, namespace: str = "default", metric: str = "cosine") -> List[SearchResult]:
        if namespace not in self._spaces:
            return []
        index = self._spaces[namespace]
        labels, distances = index.knn_query(query.reshape(1, -1), k=min(top_k, len(self._vectors[namespace])))
        labels, distances = labels[0], distances[0]
        results: List[SearchResult] = []
        for lid, dist in zip(labels, distances):
            vector = self._vectors[namespace].get(int(lid))
            if vector is None:
                continue
            score = 1.0 - dist if metric == "cosine" else -dist
            results.append(SearchResult(vector.id, float(score), vector))
        return results

    def rebuild(self, source: Iterable[VectorProtocol], metric: str = "cosine") -> None:
        self.metric = metric
        self._spaces.clear()
        self._vectors.clear()
        self._next_id.clear()
        self.add(source)