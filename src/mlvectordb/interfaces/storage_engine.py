"""
Интерфейс StorageEngine для MLVectorDB.

Модуль определяет Protocol-интерфейс StorageEngine для управления
постоянным хранением и извлечением векторов и метаданных.
"""

from typing import Protocol, List, Optional, Dict, Any, Iterator, Mapping
from uuid import UUID
from typing_extensions import runtime_checkable

from .vector import VectorProtocol


@runtime_checkable
class StorageEngine(Protocol):
    """Protocol-интерфейс для движков хранения в MLVectorDB."""
    
    @property
    def storage_type(self) -> str: raise NotImplementedError
    
    @property
    def total_vectors(self) -> int: raise NotImplementedError
    
    @property
    def storage_size(self) -> int: raise NotImplementedError

    def write(self, vector: VectorProtocol, namespace: str) -> bool: raise NotImplementedError

    def write_vectors(self, vectors: List[VectorProtocol], namespace: str) -> List[bool]: raise NotImplementedError
    
    def read(self, vector_id: UUID, namespace: str) -> Optional[VectorProtocol]: raise NotImplementedError

    def read_vectors(self, vector_ids: List[UUID], namespace: str) -> List[Optional[VectorProtocol]]:
        raise NotImplementedError

    def delete(self, vector_id: UUID, namespace: str) -> bool: raise NotImplementedError

    def exists(self, vector_id: UUID) -> bool: raise NotImplementedError

    def clear_all(self) -> bool: raise NotImplementedError
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Получить детальную информацию и статистику хранилища."""
        raise NotImplementedError

    @property
    def namespace_map(self) -> Mapping[str, List[VectorProtocol]]: raise NotImplementedError

    def delete_namespace(self, namespace: str) -> bool: raise NotImplementedError

    @property
    def list_namespaces(self) -> List[str]: raise NotImplementedError
