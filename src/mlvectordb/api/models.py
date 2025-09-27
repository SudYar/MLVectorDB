"""
Pydantic models for MLVectorDB REST API.

This module defines the request and response models used by the REST API.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum


class QueryType(str, Enum):
    """Query types supported by the API."""
    KNN = "knn"
    RANGE = "range"
    SIMILARITY = "similarity"
    METADATA = "metadata"
    HYBRID = "hybrid"


class VectorData(BaseModel):
    """Vector data model."""
    id: str = Field(..., description="Unique identifier for the vector")
    data: List[float] = Field(..., description="Vector data as list of floats")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    dimension: Optional[int] = Field(default=None, description="Vector dimension")


class QueryRequest(BaseModel):
    """Base query request model."""
    type: str = Field(..., description="Type of query to execute")
    vector: Optional[List[float]] = Field(default=None, description="Query vector data")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter conditions")
    limit: Optional[int] = Field(default=100, description="Maximum number of results to return")
    offset: Optional[int] = Field(default=0, description="Offset for pagination")
    index_hint: Optional[str] = Field(default=None, description="Hint for which index to use")


class KNNQueryRequest(QueryRequest):
    """K-nearest neighbors query request."""
    type: Literal["knn"] = Field(default="knn", description="Query type (knn)")
    vector: List[float] = Field(..., description="Query vector data")
    k: int = Field(..., gt=0, description="Number of nearest neighbors to return")


class RangeQueryRequest(QueryRequest):
    """Range search query request."""
    type: Literal["range"] = Field(default="range", description="Query type (range)")
    vector: List[float] = Field(..., description="Query vector data")
    radius: float = Field(..., gt=0.0, description="Search radius")


class SimilarityQueryRequest(QueryRequest):
    """Similarity search query request."""
    type: Literal["similarity"] = Field(default="similarity", description="Query type (similarity)")
    vector: List[float] = Field(..., description="Query vector data")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Minimum similarity threshold")
    metric: str = Field(default="cosine", description="Similarity metric to use")


class MetadataQueryRequest(QueryRequest):
    """Metadata-based query request."""
    type: Literal["metadata"] = Field(default="metadata", description="Query type (metadata)")
    filter: Dict[str, Any] = Field(..., description="Metadata filter conditions")
    order_by: Optional[str] = Field(default=None, description="Field to order results by")


class HybridQueryRequest(QueryRequest):
    """Hybrid query combining vector and metadata search."""
    type: Literal["hybrid"] = Field(default="hybrid", description="Query type (hybrid)")
    vector: Optional[List[float]] = Field(default=None, description="Query vector data")
    k: Optional[int] = Field(default=None, gt=0, description="Number of neighbors for vector search")
    radius: Optional[float] = Field(default=None, gt=0.0, description="Search radius for vector search")
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for vector similarity")
    metadata_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for metadata matching")


class QueryResult(BaseModel):
    """Single query result."""
    id: str = Field(..., description="Vector ID")
    data: List[float] = Field(..., description="Vector data")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Vector metadata")
    score: float = Field(..., description="Similarity score or distance")


class QueryResponse(BaseModel):
    """Query response model."""
    query_type: str = Field(..., description="Type of query executed")
    results: List[QueryResult] = Field(..., description="Query results")
    total_results: int = Field(..., description="Total number of results")
    execution_time_ms: Optional[float] = Field(default=None, description="Query execution time in milliseconds")
    query_id: Optional[str] = Field(default=None, description="Unique query identifier")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(default=None, description="Type of error")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class QueryExplanation(BaseModel):
    """Query execution plan explanation."""
    query_type: str = Field(..., description="Type of query")
    execution_plan: Optional[Dict[str, Any]] = Field(default=None, description="Detailed execution plan")
    estimated_cost: Optional[float] = Field(default=None, description="Estimated query cost")
    optimizations_applied: Optional[List[str]] = Field(default=None, description="Applied optimizations")


class QueryStatistics(BaseModel):
    """Query processor statistics."""
    total_queries: int = Field(..., description="Total number of queries processed")
    knn_queries: int = Field(..., description="Number of KNN queries")
    range_queries: int = Field(..., description="Number of range queries")
    similarity_queries: int = Field(..., description="Number of similarity queries")
    metadata_queries: int = Field(..., description="Number of metadata queries")
    hybrid_queries: int = Field(..., description="Number of hybrid queries")
    cache_hits: int = Field(..., description="Number of cache hits")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    query_processor_status: str = Field(..., description="Query processor status")


class IndexInfo(BaseModel):
    """Index information model."""
    name: str = Field(..., description="Index name")
    index_type: str = Field(..., description="Index algorithm type")
    dimension: int = Field(..., description="Vector dimension")
    size: int = Field(..., description="Number of indexed vectors")


class StorageInfo(BaseModel):
    """Storage engine information model."""
    storage_type: str = Field(..., description="Storage engine type")
    total_vectors: int = Field(..., description="Total number of stored vectors")
    storage_size: int = Field(..., description="Storage size in bytes")