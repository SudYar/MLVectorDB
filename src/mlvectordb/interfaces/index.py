from typing import Protocol, Iterable, Sequence, List
import numpy as np
from .vector import VectorProtocol

class SearchResultProtocol(Protocol):
    id: str
    score: float
    vector: VectorProtocol

class IndexProtocol(Protocol):
    def add(self, vectors: Iterable[VectorProtocol]) -> None: ...
    def remove(self, ids: Sequence[str], namespace: str = "default") -> None: ...
    def search(self, query: np.ndarray, top_k: int, namespace: str = "default", metric: str = "cosine") -> List[SearchResultProtocol]: ...
    def rebuild(self, source: Iterable[VectorProtocol], metric: str = "cosine") -> None: ...
