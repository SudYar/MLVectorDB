"""
QueryProcessor interface for MLVectorDB.

This module defines the QueryProcessor Protocol interface which handles
query parsing, optimization, and execution for vector database operations.
"""

from typing import Protocol, List, Optional, Dict, Any, Union, Tuple
from .vector import Vector
from .index import Index
from .storage_engine import StorageEngine
from typing_extensions import runtime_checkable
from enum import Enum


class QueryType(Enum):
    """Types of queries supported by the query processor."""
    KNN = "knn"  # k-nearest neighbors
    RANGE = "range"  # range search
    SIMILARITY = "similarity"  # similarity search
    METADATA = "metadata"  # metadata-based query
    HYBRID = "hybrid"  # combination of vector and metadata queries


@runtime_checkable
class QueryProcessor(Protocol):
    """
    Protocol interface for QueryProcessor in MLVectorDB.
    
    A QueryProcessor handles the parsing, optimization, and execution of
    queries against vector databases, coordinating between indexes and storage.
    """
    
    @property
    def supported_query_types(self) -> List[QueryType]:
        """
        List of query types supported by this processor.
        
        Returns:
            List[QueryType]: Supported query types
        """
        ...
    
    def register_index(self, index: Index) -> bool:
        """
        Register an index for query processing.
        
        Args:
            index: Index to register
            
        Returns:
            bool: True if successfully registered, False otherwise
        """
        ...
    
    def register_storage_engine(self, storage_engine: StorageEngine) -> bool:
        """
        Register a storage engine for query processing.
        
        Args:
            storage_engine: Storage engine to register
            
        Returns:
            bool: True if successfully registered, False otherwise
        """
        ...
    
    def parse_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate a query specification.
        
        Args:
            query: Raw query specification
            
        Returns:
            Dict[str, Any]: Parsed and validated query
            
        Raises:
            ValueError: If query is invalid or malformed
        """
        ...
    
    def optimize_query(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize the query execution plan.
        
        Args:
            parsed_query: Parsed query from parse_query()
            
        Returns:
            Dict[str, Any]: Optimized query execution plan
        """
        ...
    
    def execute_knn_query(
        self,
        query_vector: Vector,
        k: int,
        filter_condition: Optional[Dict[str, Any]] = None,
        index_hint: Optional[str] = None
    ) -> List[Tuple[Vector, float]]:
        """
        Execute a k-nearest neighbors query.
        
        Args:
            query_vector: Vector to find neighbors for
            k: Number of nearest neighbors to return
            filter_condition: Optional metadata filter conditions
            index_hint: Optional hint for which index to use
            
        Returns:
            List[Tuple[Vector, float]]: List of (vector, distance) pairs
        """
        ...
    
    def execute_range_query(
        self,
        query_vector: Vector,
        radius: float,
        filter_condition: Optional[Dict[str, Any]] = None,
        index_hint: Optional[str] = None
    ) -> List[Tuple[Vector, float]]:
        """
        Execute a range search query.
        
        Args:
            query_vector: Center vector for range search
            radius: Search radius
            filter_condition: Optional metadata filter conditions
            index_hint: Optional hint for which index to use
            
        Returns:
            List[Tuple[Vector, float]]: List of (vector, distance) pairs within radius
        """
        ...
    
    def execute_similarity_query(
        self,
        query_vector: Vector,
        threshold: float,
        metric: str = "cosine",
        filter_condition: Optional[Dict[str, Any]] = None,
        index_hint: Optional[str] = None
    ) -> List[Tuple[Vector, float]]:
        """
        Execute a similarity search query.
        
        Args:
            query_vector: Vector to find similar vectors for
            threshold: Minimum similarity threshold
            metric: Similarity metric to use
            filter_condition: Optional metadata filter conditions
            index_hint: Optional hint for which index to use
            
        Returns:
            List[Tuple[Vector, float]]: List of (vector, similarity) pairs
        """
        ...
    
    def execute_metadata_query(
        self,
        filter_condition: Dict[str, Any],
        offset: int = 0,
        limit: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> List[Vector]:
        """
        Execute a metadata-based query.
        
        Args:
            filter_condition: Metadata filter conditions
            offset: Starting offset for pagination
            limit: Maximum number of results to return
            order_by: Optional field to order results by
            
        Returns:
            List[Vector]: List of matching vectors
        """
        ...
    
    def execute_hybrid_query(
        self,
        query_vector: Optional[Vector] = None,
        k: Optional[int] = None,
        radius: Optional[float] = None,
        filter_condition: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        metadata_weight: float = 0.3,
        index_hint: Optional[str] = None
    ) -> List[Tuple[Vector, float]]:
        """
        Execute a hybrid query combining vector and metadata search.
        
        Args:
            query_vector: Optional vector for vector-based search
            k: Optional k for k-nearest neighbors
            radius: Optional radius for range search
            filter_condition: Optional metadata filter conditions
            vector_weight: Weight for vector similarity in hybrid scoring
            metadata_weight: Weight for metadata matching in hybrid scoring
            index_hint: Optional hint for which index to use
            
        Returns:
            List[Tuple[Vector, float]]: List of (vector, hybrid_score) pairs
        """
        ...
    
    def execute_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a general query from specification.
        
        Args:
            query: Query specification dictionary
            
        Returns:
            Dict[str, Any]: Query results with metadata
            
        Example query format:
        {
            "type": "knn",
            "vector": [0.1, 0.2, ...],
            "k": 10,
            "filters": {"category": "documents"},
            "options": {"index_hint": "my_index"}
        }
        """
        ...
    
    def explain_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain the execution plan for a query without executing it.
        
        Args:
            query: Query specification dictionary
            
        Returns:
            Dict[str, Any]: Query execution plan and estimated costs
        """
        ...
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """
        Get query processing statistics.
        
        Returns:
            Dict[str, Any]: Statistics about processed queries
        """
        ...
    
    def create_query_cache(self, cache_size: int = 1000) -> bool:
        """
        Create or resize the query result cache.
        
        Args:
            cache_size: Maximum number of cached query results
            
        Returns:
            bool: True if successfully created/resized, False otherwise
        """
        ...
    
    def clear_query_cache(self) -> bool:
        """
        Clear the query result cache.
        
        Returns:
            bool: True if successfully cleared, False otherwise
        """
        ...