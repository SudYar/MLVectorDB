from collections import defaultdict
from typing import Dict, Any, Optional, List, Iterator, Mapping
import sys

from src.mlvectordb import StorageEngine, Vector
from src.mlvectordb.interfaces.vector import VectorProtocol


# --- Storage Engine Implementation ---
class StorageEngineInMemory(StorageEngine):

    def __init__(self):
        self._storage = defaultdict(dict)  # namespace -> {vector_id -> Vector}

    # ---------------- properties ----------------
    @property
    def storage_type(self) -> str:
        return "in-memory"

    @property
    def total_vectors(self) -> int:
        return sum(len(vectors) for vectors in self._storage.values())

    @property
    def storage_size(self) -> int:
        total_size = 0
        for namespace, vectors in self._storage.items():
            total_size += sys.getsizeof(namespace)
            for vector_id, vector in vectors.items():
                total_size += sys.getsizeof(vector_id)
                total_size += sys.getsizeof(vector.values)
                if vector.metadata:
                    total_size += sys.getsizeof(vector.metadata)
        return total_size

    # ---------------- CRUD operations ----------------

    def write(self, vector: Vector, namespace: str = "default") -> bool:
        try:
            self._storage[namespace][vector.id] = vector
            return True
        except Exception:
            return False

    def write_vectors(self, vectors: List[VectorProtocol], namespace: str = "default") -> List[bool]:
        results = []
        for vector in vectors:
            results.append(self.write(vector, namespace))
        return results

    def read(self, vector_id: str, namespace: str = "default") -> Optional[Vector]:
        return self._storage.get(namespace, {}).get(vector_id)

    def read_vectors(self, vector_ids: List[str], namespace: str = "default") -> List[Optional[Vector]]:
        namespace_vectors = self._storage.get(namespace, {})
        return [namespace_vectors.get(vector_id) for vector_id in vector_ids]

    def delete(self, vector_id: str, namespace: str = "default") -> bool:
        try:
            if namespace in self._storage and vector_id in self._storage[namespace]:
                del self._storage[namespace][vector_id]
                # Clean up empty namespaces
                if not self._storage[namespace]:
                    del self._storage[namespace]
                return True
            return False
        except Exception:
            return False

    def exists(self, vector_id: str, namespace: str = "default") -> bool:
        return namespace in self._storage and vector_id in self._storage[namespace]

    def iterate_vectors(
            self,
            namespace: str = "default",
            batch_size: int = 100
    ) -> Iterator[List[Vector]]:
        if namespace not in self._storage:
            return

        vectors = list(self._storage[namespace].values())
        for i in range(0, len(vectors), batch_size):
            yield vectors[i:i + batch_size]

    def clear_all(self) -> bool:
        try:
            self._storage.clear()
            return True
        except Exception:
            return False

    def get_storage_info(self) -> Dict[str, Any]:
        namespaces = list(self._storage.keys())

        return {
            "storage_type": self.storage_type,
            "total_vectors": self.total_vectors,
            "storage_size_bytes": self.storage_size,
            "namespaces": namespaces,
            "vectors_per_namespace": {
                namespace: len(vectors)
                for namespace, vectors in self._storage.items()
            },
            "namespace_count": len(namespaces)
        }

    @property
    def namespace_map(self) -> Mapping[str, List[Vector]]:
        return {
            namespace: list(vectors.values())
            for namespace, vectors in self._storage.items()
        }

    def delete_namespace(self, namespace: str) -> bool:
        try:
            if namespace in self._storage:
                del self._storage[namespace]
                return True
            return False
        except Exception:
            return False

    def list_namespaces(self) -> List[str]:
        return list(self._storage.keys())
