"""
StorageEngine interface for MLVectorDB.

This module defines the StorageEngine Protocol interface which handles the
persistent storage and retrieval of vectors and metadata.
"""

from typing import Protocol, List, Optional, Dict, Any, Iterator
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
        ...
    
    @property
    def total_vectors(self) -> int:
        """
        Total number of vectors stored.
        
        Returns:
            int: Number of stored vectors
        """
        ...
    
    @property
    def storage_size(self) -> int:
        """
        Total storage size in bytes.
        
        Returns:
            int: Storage size in bytes
        """
        ...
    
    def store_vector(self, vector: Vector) -> bool:
        """
        Store a vector persistently.
        
        Args:
            vector: Vector to store
            
        Returns:
            bool: True if successfully stored, False otherwise
        """
        ...
    
    def store_vectors(self, vectors: List[Vector]) -> List[bool]:
        """
        Store multiple vectors persistently.
        
        Args:
            vectors: List of vectors to store
            
        Returns:
            List[bool]: List of success status for each vector
        """
        ...
    
    def retrieve_vector(self, vector_id: str) -> Optional[Vector]:
        """
        Retrieve a vector by its ID.
        
        Args:
            vector_id: ID of vector to retrieve
            
        Returns:
            Optional[Vector]: Vector if found, None otherwise
        """
        ...
    
    def retrieve_vectors(self, vector_ids: List[str]) -> List[Optional[Vector]]:
        """
        Retrieve multiple vectors by their IDs.
        
        Args:
            vector_ids: List of vector IDs to retrieve
            
        Returns:
            List[Optional[Vector]]: List of vectors (None if not found)
        """
        ...
    
    def update_vector(self, vector: Vector) -> bool:
        """
        Update an existing stored vector.
        
        Args:
            vector: Updated vector data
            
        Returns:
            bool: True if successfully updated, False otherwise
        """
        ...
    
    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector from storage.
        
        Args:
            vector_id: ID of vector to delete
            
        Returns:
            bool: True if successfully deleted, False otherwise
        """
        ...
    
    def delete_vectors(self, vector_ids: List[str]) -> List[bool]:
        """
        Delete multiple vectors from storage.
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            List[bool]: List of success status for each deletion
        """
        ...
    
    def exists(self, vector_id: str) -> bool:
        """
        Check if a vector exists in storage.
        
        Args:
            vector_id: ID of vector to check
            
        Returns:
            bool: True if exists, False otherwise
        """
        ...
    
    def list_vector_ids(
        self, 
        offset: int = 0, 
        limit: Optional[int] = None
    ) -> List[str]:
        """
        List vector IDs with pagination support.
        
        Args:
            offset: Starting offset for pagination
            limit: Maximum number of IDs to return
            
        Returns:
            List[str]: List of vector IDs
        """
        ...
    
    def iterate_vectors(
        self,
        batch_size: int = 100,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> Iterator[List[Vector]]:
        """
        Iterate over all vectors in batches.
        
        Args:
            batch_size: Size of each batch
            filter_condition: Optional metadata filter conditions
            
        Yields:
            List[Vector]: Batch of vectors
        """
        ...
    
    def query_by_metadata(
        self,
        filter_condition: Dict[str, Any],
        offset: int = 0,
        limit: Optional[int] = None
    ) -> List[Vector]:
        """
        Query vectors by metadata conditions.
        
        Args:
            filter_condition: Metadata filter conditions
            offset: Starting offset for pagination
            limit: Maximum number of vectors to return
            
        Returns:
            List[Vector]: List of matching vectors
        """
        ...
    
    def create_backup(self, backup_path: str) -> bool:
        """
        Create a backup of the storage.
        
        Args:
            backup_path: Path where to save the backup
            
        Returns:
            bool: True if successfully backed up, False otherwise
        """
        ...
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restore storage from a backup.
        
        Args:
            backup_path: Path to restore the backup from
            
        Returns:
            bool: True if successfully restored, False otherwise
        """
        ...
    
    def optimize_storage(self) -> bool:
        """
        Optimize the storage (e.g., defragmentation, compression).
        
        Returns:
            bool: True if successfully optimized, False otherwise
        """
        ...
    
    def clear_all(self) -> bool:
        """
        Clear all vectors from storage.
        
        Returns:
            bool: True if successfully cleared, False otherwise
        """
        ...
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get detailed storage information and statistics.
        
        Returns:
            Dict[str, Any]: Storage information and statistics
        """
        ...