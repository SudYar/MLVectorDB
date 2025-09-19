"""
Index interface for MLVectorDB.

This module defines the Index Protocol interface which handles the indexing
and efficient searching of vectors in the database.
"""

from typing import Protocol, List, Optional, Dict, Any, Tuple
from .vector import Vector
from typing_extensions import runtime_checkable


@runtime_checkable
class Index(Protocol):
    """
    Protocol interface for Index in MLVectorDB.
    
    An Index provides efficient storage and retrieval mechanisms for vectors,
    supporting various search algorithms like nearest neighbor search.
    """
    
    @property
    def name(self) -> str:
        """
        Name/identifier of the index.
        
        Returns:
            str: Index name
        """
        ...
    
    @property
    def dimension(self) -> int:
        """
        Dimensionality of vectors stored in this index.
        
        Returns:
            int: Vector dimension
        """
        ...
    
    @property
    def size(self) -> int:
        """
        Number of vectors currently indexed.
        
        Returns:
            int: Number of vectors in index
        """
        ...
    
    @property
    def index_type(self) -> str:
        """
        Type of index algorithm used.
        
        Returns:
            str: Index algorithm type (e.g., "flat", "ivf", "hnsw")
        """
        ...
    
    def add_vector(self, vector: Vector) -> bool:
        """
        Add a vector to the index.
        
        Args:
            vector: Vector to be indexed
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        ...
    
    def add_vectors(self, vectors: List[Vector]) -> List[bool]:
        """
        Add multiple vectors to the index.
        
        Args:
            vectors: List of vectors to be indexed
            
        Returns:
            List[bool]: List of success status for each vector
        """
        ...
    
    def remove_vector(self, vector_id: str) -> bool:
        """
        Remove a vector from the index.
        
        Args:
            vector_id: ID of vector to remove
            
        Returns:
            bool: True if successfully removed, False otherwise
        """
        ...
    
    def update_vector(self, vector: Vector) -> bool:
        """
        Update an existing vector in the index.
        
        Args:
            vector: Updated vector data
            
        Returns:
            bool: True if successfully updated, False otherwise
        """
        ...
    
    def search_knn(
        self, 
        query_vector: Vector, 
        k: int,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Vector, float]]:
        """
        Find k nearest neighbors to the query vector.
        
        Args:
            query_vector: Vector to search for
            k: Number of nearest neighbors to return
            filter_condition: Optional metadata filter conditions
            
        Returns:
            List[Tuple[Vector, float]]: List of (vector, distance) pairs
        """
        ...
    
    def range_search(
        self,
        query_vector: Vector,
        radius: float,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Vector, float]]:
        """
        Find all vectors within a given radius of the query vector.
        
        Args:
            query_vector: Vector to search around
            radius: Search radius
            filter_condition: Optional metadata filter conditions
            
        Returns:
            List[Tuple[Vector, float]]: List of (vector, distance) pairs within radius
        """
        ...
    
    def get_vector(self, vector_id: str) -> Optional[Vector]:
        """
        Retrieve a vector by its ID.
        
        Args:
            vector_id: ID of vector to retrieve
            
        Returns:
            Optional[Vector]: Vector if found, None otherwise
        """
        ...
    
    def build_index(self) -> bool:
        """
        Build/rebuild the index structure.
        
        Returns:
            bool: True if successfully built, False otherwise
        """
        ...
    
    def save_index(self, filepath: str) -> bool:
        """
        Save the index to disk.
        
        Args:
            filepath: Path where to save the index
            
        Returns:
            bool: True if successfully saved, False otherwise
        """
        ...
    
    def load_index(self, filepath: str) -> bool:
        """
        Load the index from disk.
        
        Args:
            filepath: Path to load the index from
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        ...
    
    def clear(self) -> bool:
        """
        Clear all vectors from the index.
        
        Returns:
            bool: True if successfully cleared, False otherwise
        """
        ...
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get index statistics and metadata.
        
        Returns:
            Dict[str, Any]: Dictionary containing index statistics
        """
        ...