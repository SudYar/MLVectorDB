from typing import Iterable, Sequence, List, Callable, Optional
import numpy as np
from ..interfaces.query_processor import QueryProcessorProtocol
from ..interfaces.index import IndexProtocol, SearchResultProtocol
from ..interfaces.vector import VectorProtocol


class QueryProcessor(QueryProcessorProtocol):
    def __init__(self, index: IndexProtocol) -> None:
        self.index = index

    def insert(self, vector: VectorProtocol) -> None:
        self.index.add([vector])


    def upsert_many(self, vectors: Iterable[VectorProtocol]) -> None:
        self.index.add(vectors)


    def find_similar(self, query: np.ndarray, top_k: int, namespace: str = "default", filter: Optional[Callable[[VectorProtocol], bool]] = None, metric: str = "cosine") -> List[SearchResultProtocol]:
        results = self.index.search(query, top_k=top_k, namespace=namespace, metric=metric)
        if filter is None:
            return results
        return [r for r in results if filter(r.vector)]


    def delete(self, ids: Sequence[str], namespace: str = "default") -> None:
        self.index.remove(ids, namespace=namespace)