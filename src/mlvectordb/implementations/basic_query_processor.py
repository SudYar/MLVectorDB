"""
Basic implementation of QueryProcessor interface for MLVectorDB.

This module provides a concrete implementation of the QueryProcessor Protocol
that can be used with the REST API.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from ..interfaces.query_processor import QueryProcessor as QueryProcessorProtocol, QueryType
from ..interfaces.vector import Vector
from ..interfaces.index import Index
from ..interfaces.storage_engine import StorageEngine
from ..implementations.simple_vector import SimpleVector


class BasicQueryProcessor:
    """
    Basic concrete implementation of the QueryProcessor Protocol.
    
    This implementation provides simple query processing capabilities
    suitable for demonstration and REST API integration.
    """
    
    def __init__(self):
        """Initialize the query processor."""
        self._indexes: Dict[str, Index] = {}
        self._storage_engines: Dict[str, StorageEngine] = {}
        self._query_cache: Dict[str, Any] = {}
        self._query_stats = {
            "total_queries": 0,
            "knn_queries": 0,
            "range_queries": 0,
            "similarity_queries": 0,
            "metadata_queries": 0,
            "hybrid_queries": 0,
            "cache_hits": 0,
        }
    
    @property
    def supported_query_types(self) -> List[QueryType]:
        """List of query types supported by this processor."""
        return [
            QueryType.KNN,
            QueryType.RANGE,
            QueryType.SIMILARITY,
            QueryType.METADATA,
            QueryType.HYBRID,
        ]
    
    def register_index(self, index: Index) -> bool:
        """Register an index for query processing."""
        try:
            self._indexes[index.name] = index
            return True
        except Exception:
            return False
    
    def register_storage_engine(self, storage_engine: StorageEngine) -> bool:
        """Register a storage engine for query processing."""
        try:
            engine_name = storage_engine.storage_type
            self._storage_engines[engine_name] = storage_engine
            return True
        except Exception:
            return False
    
    def parse_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate a query specification."""
        required_fields = ["type"]
        
        # Validate required fields
        for field in required_fields:
            if field not in query:
                raise ValueError(f"Missing required field: {field}")
        
        query_type = query["type"]
        if query_type not in [qt.value for qt in self.supported_query_types]:
            raise ValueError(f"Unsupported query type: {query_type}")
        
        # Type-specific validation
        if query_type == QueryType.KNN.value:
            if "k" not in query:
                raise ValueError("KNN query requires 'k' parameter")
            if "vector" not in query:
                raise ValueError("KNN query requires 'vector' parameter")
        
        elif query_type == QueryType.RANGE.value:
            if "radius" not in query:
                raise ValueError("Range query requires 'radius' parameter")
            if "vector" not in query:
                raise ValueError("Range query requires 'vector' parameter")
        
        elif query_type == QueryType.SIMILARITY.value:
            if "threshold" not in query:
                raise ValueError("Similarity query requires 'threshold' parameter")
            if "vector" not in query:
                raise ValueError("Similarity query requires 'vector' parameter")
        
        elif query_type == QueryType.METADATA.value:
            if "filter" not in query:
                raise ValueError("Metadata query requires 'filter' parameter")
        
        return query.copy()
    
    def optimize_query(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the query execution plan."""
        # Basic optimization: add execution hints
        optimized = parsed_query.copy()
        
        # Add default parameters if not specified
        if "limit" not in optimized:
            optimized["limit"] = 100
        
        if "offset" not in optimized:
            optimized["offset"] = 0
        
        # Choose best index if not specified
        if "index_hint" not in optimized and self._indexes:
            # For demo purposes, use the first available index
            optimized["index_hint"] = list(self._indexes.keys())[0]
        
        return optimized
    
    def execute_knn_query(
        self,
        query_vector: Vector,
        k: int,
        filter_condition: Optional[Dict[str, Any]] = None,
        index_hint: Optional[str] = None
    ) -> List[Tuple[Vector, float]]:
        """Execute a k-nearest neighbors query."""
        self._query_stats["knn_queries"] += 1
        
        # For demo purposes, create some mock results
        # In a real implementation, this would use the actual index
        mock_results = []
        for i in range(min(k, 5)):  # Return up to 5 mock results
            mock_vector = SimpleVector(
                id=f"result_{i}",
                data=query_vector.data + np.random.normal(0, 0.1, query_vector.dimension),
                metadata={"rank": i + 1, "type": "knn_result"}
            )
            distance = query_vector.distance(mock_vector)
            mock_results.append((mock_vector, distance))
        
        return sorted(mock_results, key=lambda x: x[1])
    
    def execute_range_query(
        self,
        query_vector: Vector,
        radius: float,
        filter_condition: Optional[Dict[str, Any]] = None,
        index_hint: Optional[str] = None
    ) -> List[Tuple[Vector, float]]:
        """Execute a range search query."""
        self._query_stats["range_queries"] += 1
        
        # Mock implementation
        mock_results = []
        for i in range(3):  # Return 3 mock results within radius
            mock_vector = SimpleVector(
                id=f"range_result_{i}",
                data=query_vector.data + np.random.normal(0, radius/4, query_vector.dimension),
                metadata={"type": "range_result", "distance_ratio": i * 0.3}
            )
            distance = query_vector.distance(mock_vector)
            if distance <= radius:
                mock_results.append((mock_vector, distance))
        
        return mock_results
    
    def execute_similarity_query(
        self,
        query_vector: Vector,
        threshold: float,
        metric: str = "cosine",
        filter_condition: Optional[Dict[str, Any]] = None,
        index_hint: Optional[str] = None
    ) -> List[Tuple[Vector, float]]:
        """Execute a similarity search query."""
        self._query_stats["similarity_queries"] += 1
        
        # Mock implementation
        mock_results = []
        for i in range(3):
            mock_vector = SimpleVector(
                id=f"sim_result_{i}",
                data=query_vector.data + np.random.normal(0, 0.2, query_vector.dimension),
                metadata={"type": "similarity_result"}
            )
            similarity = query_vector.similarity(mock_vector, metric)
            if similarity >= threshold:
                mock_results.append((mock_vector, similarity))
        
        return sorted(mock_results, key=lambda x: x[1], reverse=True)
    
    def execute_metadata_query(
        self,
        filter_condition: Dict[str, Any],
        offset: int = 0,
        limit: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> List[Vector]:
        """Execute a metadata-based query."""
        self._query_stats["metadata_queries"] += 1
        
        # Mock implementation
        mock_results = []
        for i in range(min(limit or 10, 10)):
            mock_vector = SimpleVector(
                id=f"meta_result_{i}",
                data=np.random.random(3).astype(np.float32),
                metadata={**filter_condition, "result_index": i}
            )
            mock_results.append(mock_vector)
        
        return mock_results[offset:]
    
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
        """Execute a hybrid query combining vector and metadata search."""
        self._query_stats["hybrid_queries"] += 1
        
        # Mock implementation combining vector similarity and metadata matching
        if query_vector and k:
            vector_results = self.execute_knn_query(query_vector, k, filter_condition, index_hint)
            # Apply hybrid scoring
            hybrid_results = []
            for vector, distance in vector_results:
                # Simple hybrid score: inverse distance weighted by vector_weight
                vector_score = (1.0 / (1.0 + distance)) * vector_weight
                metadata_score = metadata_weight * 0.8  # Mock metadata score
                hybrid_score = vector_score + metadata_score
                hybrid_results.append((vector, hybrid_score))
            return sorted(hybrid_results, key=lambda x: x[1], reverse=True)
        
        return []
    
    def execute_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a general query from specification."""
        self._query_stats["total_queries"] += 1
        
        try:
            parsed_query = self.parse_query(query)
            optimized_query = self.optimize_query(parsed_query)
            
            query_type = optimized_query["type"]
            results = []
            
            if query_type == QueryType.KNN.value:
                query_vector = SimpleVector(
                    id="query", 
                    data=np.array(optimized_query["vector"], dtype=np.float32)
                )
                results = self.execute_knn_query(
                    query_vector, 
                    optimized_query["k"],
                    optimized_query.get("filter"),
                    optimized_query.get("index_hint")
                )
            
            elif query_type == QueryType.RANGE.value:
                query_vector = SimpleVector(
                    id="query", 
                    data=np.array(optimized_query["vector"], dtype=np.float32)
                )
                results = self.execute_range_query(
                    query_vector,
                    optimized_query["radius"],
                    optimized_query.get("filter"),
                    optimized_query.get("index_hint")
                )
            
            elif query_type == QueryType.SIMILARITY.value:
                query_vector = SimpleVector(
                    id="query", 
                    data=np.array(optimized_query["vector"], dtype=np.float32)
                )
                results = self.execute_similarity_query(
                    query_vector,
                    optimized_query["threshold"],
                    optimized_query.get("metric", "cosine"),
                    optimized_query.get("filter"),
                    optimized_query.get("index_hint")
                )
            
            elif query_type == QueryType.METADATA.value:
                metadata_results = self.execute_metadata_query(
                    optimized_query["filter"],
                    optimized_query.get("offset", 0),
                    optimized_query.get("limit")
                )
                results = [(vector, 1.0) for vector in metadata_results]
            
            # Format results
            formatted_results = []
            for vector, score in results:
                formatted_results.append({
                    "id": vector.id,
                    "data": vector.data.tolist(),
                    "metadata": vector.metadata,
                    "score": float(score)
                })
            
            return {
                "query_type": query_type,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "execution_time_ms": 42,  # Mock execution time
                "query_id": f"query_{self._query_stats['total_queries']}"
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "query_type": query.get("type", "unknown"),
                "results": [],
                "total_results": 0
            }
    
    def explain_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Explain the execution plan for a query without executing it."""
        try:
            parsed_query = self.parse_query(query)
            optimized_query = self.optimize_query(parsed_query)
            
            return {
                "query_type": optimized_query["type"],
                "execution_plan": {
                    "steps": [
                        "Parse and validate query",
                        "Optimize query parameters",
                        "Select appropriate index" if optimized_query.get("index_hint") else "Use sequential scan",
                        "Execute query",
                        "Format results"
                    ],
                    "estimated_cost": 100,  # Mock cost
                    "selected_index": optimized_query.get("index_hint", "none"),
                    "estimated_results": optimized_query.get("k", optimized_query.get("limit", 10))
                },
                "optimizations_applied": [
                    "Added default limit",
                    "Selected best available index"
                ]
            }
        except Exception as e:
            return {
                "error": str(e),
                "execution_plan": None
            }
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query processing statistics."""
        return self._query_stats.copy()
    
    def create_query_cache(self, cache_size: int = 1000) -> bool:
        """Create or resize the query result cache."""
        self._query_cache = {}
        return True
    
    def clear_query_cache(self) -> bool:
        """Clear the query result cache."""
        cache_entries = len(self._query_cache)
        self._query_cache.clear()
        return True