from typing import Protocol, Sequence, Mapping, Any
import numpy as np

class VectorProtocol(Protocol):
    id: str
    values: Sequence[float]
    namespace: str
    metadata: Mapping[str, Any]

    def to_numpy(self) -> "np.ndarray": ...

    def dimension(self) -> int: ...
