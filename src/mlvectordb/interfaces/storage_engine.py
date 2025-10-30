"""
StorageEngine interface for MLVectorDB.

This module defines the StorageEngine Protocol interface which handles the
persistent storage and retrieval of vectors and metadata.
"""

from typing import Protocol, List, Optional, Dict, Any, Iterator, Mapping
from .vector import Vector
from typing_extensions import runtime_checkable


@runtime_checkable
class StorageEngine(Protocol):
    """
    Protocol interface for StorageEngine in MLVectorDB.
    
    A StorageEngine handles the persistent storage, retrieval, and management
    of vectors and their associated metadata on disk or in memory.
    """
    
    @property
    def storage_type(self) -> str:
        """
        Type of storage engine.
        
        Returns:
            str: Storage type (e.g., "memory", "disk", "distributed")
        """
        raise NotImplementedError
    
    @property
    def total_vectors(self) -> int:
        """
        Total number of vectors stored.
        
        Returns:
            int: Number of stored vectors
        """
        raise NotImplementedError
    
    @property
    def storage_size(self) -> int:
        """
        Total storage size in bytes.
        
        Returns:
            int: Storage size in bytes
        """
        raise NotImplementedError

    def write(self, vector: Vector, namespace: str = "default") -> bool:
        """
        Store a vector persistently.

        Args:
            vector: Vector to store
            namespace: Namespace to write in

        Returns:
            bool: True if successfully stored, False otherwise
        """
        raise NotImplementedError

    def writes(self, vectors: List[Vector], namespace: str = "default") -> List[bool]:
        """
        Store multiple vectors persistently.
        
        Args:
            vectors: List of vectors to store
            namespace: Namespace to write in
            
        Returns:
            List[bool]: List of success status for each vector
        """
        raise NotImplementedError
    
    def read(self, vector_id: str, namespace: str = "default") -> Optional[Vector]:
        """
        Retrieve a vector by its ID.
        
        Args:
            vector_id: ID of vector to retrieve
            namespace: Namespace to search in
            
        Returns:
            Optional[Vector]: Vector if found, None otherwise
        """
        raise NotImplementedError

    def read_vectors(self, vector_ids: List[str], namespace: str = "default") -> List[Optional[Vector]]:
        """
        Retrieve multiple vectors by their IDs.

        Args:
            vector_ids: List of vector IDs to retrieve
            namespace: Namespace to search in

        Returns:
            List[Optional[Vector]]: List of vectors (None if not found)
        """
        raise NotImplementedError


    def delete(self, vector_id: str, namespace: str = "default") -> bool:
        """
        Delete a vector from storage.
        
        Args:
            vector_id: ID of vector to delete
            namespace: Namespace to delete from
            
        Returns:
            bool: True if successfully deleted, False otherwise
        """
        raise NotImplementedError

    def exists(self, vector_id: str) -> bool:
        """
        Check if a vector exists in storage.
        
        Args:
            vector_id: ID of vector to check
            
        Returns:
            bool: True if exists, False otherwise
        """
        raise NotImplementedError

    def iterate_vectors(
            self,
            namespace: str = "default",
            batch_size: int = 100
    ) -> Iterator[List[Vector]]:
        """
        Iterate over all vectors in a namespace in batches.

        Args:
            namespace: Namespace to iterate over
            batch_size: Size of each batch

        Yields:
            List[Vector]: Batch of vectors
        """
        raise NotImplementedError

    def clear_all(self) -> bool:
        """
        Clear all vectors from storage.
        
        Returns:
            bool: True if successfully cleared, False otherwise
        """
        raise NotImplementedError
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get detailed storage information and statistics.
        
        Returns:
            Dict[str, Any]: Storage information and statistics
        """
        raise NotImplementedError

    @property
    def namespace_map(self) -> Mapping[str, List[Vector]]:
        """
        Get mapping of namespaces to their vectors.

        Returns:
            Mapping[str, List[Vector]]: Namespace to vectors mapping
        """
        raise NotImplementedError
