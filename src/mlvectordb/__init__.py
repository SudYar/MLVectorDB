"""
MLVectorDB - A Vector Database implementation for Big Data Infrastructure course.

This package provides interfaces and implementations for a vector database system
including vector storage, indexing, and querying capabilities.
"""

__version__ = "0.1.0"
__author__ = "MLVectorDB Team"

from .interfaces.vector import Vector
from .interfaces.index import Index
from .interfaces.storage_engine import StorageEngine
from .interfaces.query_processor import QueryProcessor

# Import implementations
from .implementations.simple_vector import SimpleVector
from .implementations.basic_query_processor import BasicQueryProcessor

__all__ = [
    "Vector",
    "Index", 
    "StorageEngine",
    "QueryProcessor",
    "SimpleVector",
    "BasicQueryProcessor",
]