from __future__ import annotations

import uuid
from typing import Mapping, Any, Sequence
from uuid import UUID

import numpy as np

from ..interfaces.vector import VectorProtocol


class Vector(VectorProtocol):

    def __init__(self, values: Sequence[float], id: UUID | None = None,
                 metadata: Mapping[str, Any] | None = None) -> None:
        self._id: UUID = uuid.uuid4() if id is None else id
        self._values: np.ndarray = np.array(values, dtype=np.float32)
        self._metadata: Mapping[str, Any] = metadata or {}

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def shape(self) -> tuple:
        return self._values.shape

    def __repr__(self) -> str:
        return f"Vector(id={self.id}, dim={self.shape()}, metadata={self.metadata})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return False
        return (
            self.id == other.id
            and np.array_equal(self.values, other.values)
            and self.metadata == other.metadata
        )
