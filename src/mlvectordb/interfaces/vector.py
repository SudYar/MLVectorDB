from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Mapping, Any, runtime_checkable
from uuid import UUID

import numpy as np


@runtime_checkable
class VectorProtocol(Protocol):
    id: UUID
    values: np.ndarray
    metadata: Mapping[str, Any]

    def __init__(self, values: Sequence[float], id: UUID | None = None,
                 metadata: Mapping[str, Any] | None = None) -> None:
        ...

    def shape(self) -> tuple:
        ...

@dataclass
class VectorDTO:
    values: Sequence[float]
    metadata: Mapping[str, Any]
