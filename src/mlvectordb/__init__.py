"""
MLVectorDB - A Vector Database implementation for Big Data Infrastructure course.

This package provides interfaces and implementations for a vector database system
including vector storage, indexing, and querying capabilities.
"""

__version__ = "0.1.0"
__author__ = "MLVectorDB Team"

from .interfaces.vector import VectorProtocol
from .interfaces.index import IndexProtocol
from .interfaces.storage_engine import StorageEngine
from .interfaces.query_processor import QueryProcessorProtocol
from .interfaces.replication import ReplicationManager
from .interfaces.sharding import ShardingManager

# Import implementations
from .implementations.vector import Vector
from .implementations.storage_engine_in_memory import StorageEngineInMemory
from .implementations.index import Index
from .implementations.replication_manager import ReplicationManagerImpl
from .implementations.sharding_manager import ShardingManagerImpl
from .implementations.query_processor_with_replication import QueryProcessorWithReplication

__all__ = [
    "VectorProtocol",
    "IndexProtocol",
    "StorageEngine",
    "QueryProcessorProtocol",
    "ReplicationManager",
    "ShardingManager",
    "Vector",
    "StorageEngineInMemory",
    "Index",
    "ReplicationManagerImpl",
    "ShardingManagerImpl",
    "QueryProcessorWithReplication",
]