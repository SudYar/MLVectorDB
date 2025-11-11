import sys
from collections import defaultdict
from typing import Dict, Any, Optional, List, Mapping
from uuid import UUID

from src.mlvectordb import StorageEngine
from src.mlvectordb.interfaces.vector import VectorProtocol


class StorageEngineInMemory(StorageEngine):
    def __init__(self):
        self._storage = defaultdict(dict)

    @property
    def storage_type(self) -> str:
        return "in-memory"

    @property
    def total_vectors(self) -> int:
        return sum(len(vectors) for vectors in self._storage.values())

    @property
    def storage_size(self) -> int:
        return sum(
            sys.getsizeof(namespace) +
            sum(sys.getsizeof(v_id) + sys.getsizeof(vector.values) +
                (sys.getsizeof(vector.metadata) if vector.metadata else 0)
                for v_id, vector in vectors.items())
            for namespace, vectors in self._storage.items()
        )

    def write(self, vector: VectorProtocol, namespace: str) -> bool:
        self._storage[namespace][vector.id] = vector
        return True

    def write_vectors(self, vectors: List[VectorProtocol], namespace: str) -> List[bool]:
        return [self.write(vector, namespace) for vector in vectors]

    def read(self, vector_id: UUID, namespace: str) -> Optional[VectorProtocol]:
        return self._storage.get(namespace, {}).get(vector_id)

    def read_vectors(self, vector_ids: List[UUID], namespace: str) -> List[Optional[VectorProtocol]]:
        namespace_vectors = self._storage.get(namespace, {})
        return [namespace_vectors.get(v_id) for v_id in vector_ids]

    def delete(self, vector_id: UUID, namespace: str) -> bool:
        if namespace in self._storage and vector_id in self._storage[namespace]:
            del self._storage[namespace][vector_id]
            if not self._storage[namespace]:
                del self._storage[namespace]
            return True
        return False

    def exists(self, vector_id: UUID) -> bool:
        return any(vector_id in vectors for vectors in self._storage.values())

    def clear_all(self) -> bool:
        self._storage.clear()
        return True

    def get_storage_info(self) -> Dict[str, Any]:
        return {
            "storage_type": self.storage_type,
            "total_vectors": self.total_vectors,
            "storage_size_bytes": self.storage_size,
            "namespaces": list(self._storage.keys()),
            "vectors_per_namespace": {ns: len(vecs) for ns, vecs in self._storage.items()},
            "namespace_count": len(self._storage)
        }

    @property
    def namespace_map(self) -> Mapping[str, List[VectorProtocol]]:
        return {ns: list(vecs.values()) for ns, vecs in self._storage.items()}

    def delete_namespace(self, namespace: str) -> bool:
        try:
            if namespace in self._storage:
                del self._storage[namespace]
                return True
            return False
        except Exception:
            return False

    @property
    def list_namespaces(self) -> List[str]:
        return list(self._storage.keys())
