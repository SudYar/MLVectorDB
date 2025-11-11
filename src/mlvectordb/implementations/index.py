import hnswlib
import numpy as np
from uuid import UUID
from typing import Iterable, Sequence, List, Mapping, Dict
from dataclasses import dataclass

from ..interfaces.index import IndexProtocol, SearchResultProtocol
from ..interfaces.vector import VectorProtocol, VectorDTO


@dataclass
class SearchResult(SearchResultProtocol):
    vector_id: UUID
    score: float


class Index(IndexProtocol):
    def __init__(self, space: str = "l2", ef_construction: int = 200, M: int = 16):
        self._indexes: Dict[str, hnswlib.Index] = {}
        self._dim: Dict[str, int] = {}
        self._uuid_to_label: Dict[str, Dict[UUID, int]] = {}
        self._label_to_uuid: Dict[str, Dict[int, UUID]] = {}
        self._space = space
        self._ef_construction = ef_construction
        self._M = M

    def _get_or_create_index(self, namespace: str, dim: int, metric: str):
        if namespace in self._indexes:
            return self._indexes[namespace]

        index = hnswlib.Index(space=metric, dim=dim)
        index.init_index(max_elements=10_000, ef_construction=self._ef_construction, M=self._M)
        index.set_ef(50)
        self._indexes[namespace] = index
        self._dim[namespace] = dim
        self._uuid_to_label[namespace] = {}
        self._label_to_uuid[namespace] = {}
        return index

    def add(self, vectors: Iterable[VectorProtocol], namespace: str) -> None:
        vectors = list(vectors)
        if not vectors:
            return
        dim = vectors[0].values.shape[0]
        index = self._get_or_create_index(namespace, dim, self._space)
        current_count = index.get_current_count()
        labels = []
        data = []
        for i, v in enumerate(vectors):
            label = current_count + i
            self._uuid_to_label[namespace][v.id] = label
            self._label_to_uuid[namespace][label] = v.id
            labels.append(label)
            data.append(v.values)
        index.add_items(np.array(data, dtype=np.float32), np.array(labels))

    def remove(self, ids: Sequence[UUID], namespace: str) -> None:
        if namespace not in self._indexes:
            return
        index = self._indexes[namespace]
        uuid_to_label = self._uuid_to_label[namespace]
        label_to_uuid = self._label_to_uuid[namespace]
        for uid in ids:
            label = uuid_to_label.pop(uid, None)
            if label is not None:
                index.mark_deleted(label)
                label_to_uuid.pop(label, None)

    def search(
    self,
    query: VectorDTO,
    top_k: int,
    namespace: str,
    metric: str
    ) -> List[SearchResultProtocol]:
        if namespace not in self._indexes:
            return []

        index = self._indexes[namespace]
        current_count = index.get_current_count()
        if current_count == 0:
            return []

        top_k = min(top_k, current_count)
        data = np.array([query.values], dtype=np.float32)

        try:
            labels, distances = index.knn_query(data, k=top_k)
        except RuntimeError:
            if top_k > 1:
                try:
                    labels, distances = index.knn_query(data, k=1)
                except RuntimeError:
                    return []
            else:
                return []

        results: List[SearchResultProtocol] = []
        for label, dist in zip(labels[0], distances[0]):
            uid = self._label_to_uuid[namespace].get(int(label))
            if uid is not None:
                score = float(dist)
                if metric == "cosine":
                    score = 1 - score
                results.append(SearchResult(vector_id=uid, score=score))
        return results

    def rebuild(
        self,
        source: Mapping[str, Iterable[VectorProtocol]],
        metric: str
    ) -> None:
        self._indexes.clear()
        self._uuid_to_label.clear()
        self._label_to_uuid.clear()
        self._dim.clear()

        for namespace, vectors in source.items():
            vectors = list(vectors)
            if not vectors:
                continue
            dim = vectors[0].values.shape[0]
            index = self._get_or_create_index(namespace, dim, metric)
            labels = []
            data = []
            for i, v in enumerate(vectors):
                self._uuid_to_label[namespace][v.id] = i
                self._label_to_uuid[namespace][i] = v.id
                labels.append(i)
                data.append(v.values)
            index.add_items(np.array(data, dtype=np.float32), np.array(labels))
