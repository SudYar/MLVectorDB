from dataclasses import dataclass
from typing import Mapping, Any, Sequence, Optional
import numpy as np
from ..interfaces.vector import VectorProtocol

@dataclass
class Vector(VectorProtocol):
    id: str
    values: Sequence[float]
    namespace: str = "default"
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        self._array: np.ndarray = np.asarray(self.values, dtype=np.float32)

    def to_numpy(self) -> np.ndarray:
        return self._array

    def dimension(self) -> int:
        return int(self._array.shape[0])