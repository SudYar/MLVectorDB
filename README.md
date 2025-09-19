# MLVectorDB

A Vector Database implementation for Big Data Infrastructure course (Infrastructure for Big Data / Foundations of High-Load Application Design).

MLVectorDB is a Python-based vector database system designed to efficiently store, index, and query high-dimensional vectors. It provides a clean interface architecture using Python Protocols, making it extensible and easy to implement different storage engines, indexing algorithms, and query processors.

## Features

- **Protocol-based Architecture**: Clean interfaces defined using Python Protocols for maximum flexibility
- **Vector Operations**: Support for distance calculations, similarity searches, and vector normalization
- **Modular Design**: Separate components for storage, indexing, and query processing
- **Extensible**: Easy to implement custom storage engines, index types, and query processors
- **Type Safe**: Full type hints and Protocol compliance checking

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from mlvectordb.implementations.simple_vector import SimpleVector

# Create vectors
vector1 = SimpleVector("doc1", np.array([1.0, 2.0, 3.0]), {"category": "text"})
vector2 = SimpleVector("doc2", np.array([2.0, 3.0, 4.0]), {"category": "text"})

# Calculate distance and similarity
distance = vector1.distance(vector2, metric="euclidean")
similarity = vector1.similarity(vector2, metric="cosine")

print(f"Distance: {distance}")
print(f"Similarity: {similarity}")
```

## Architecture Overview

MLVectorDB follows a layered architecture with the following core components:

### Protocol Interfaces

All components are defined as Python Protocols, ensuring type safety and enabling multiple implementations.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Vector      │    │     Index       │    │ StorageEngine   │    │ QueryProcessor  │
│   (Protocol)    │    │   (Protocol)    │    │   (Protocol)    │    │   (Protocol)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  SimpleVector   │    │   FlatIndex     │    │  MemoryStorage  │    │ BasicProcessor  │
│ (Implementation)│    │(Implementation) │    │(Implementation) │    │(Implementation) │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Vector (Protocol)

Represents a high-dimensional vector with metadata.

**Key Methods:**
- `distance(other, metric)` - Calculate distance to another vector
- `similarity(other, metric)` - Calculate similarity to another vector  
- `normalize()` - Return normalized vector
- `to_dict() / from_dict()` - Serialization support

**Attributes:**
- `id: str` - Unique identifier
- `data: np.ndarray` - Vector data
- `metadata: dict` - Associated metadata
- `dimension: int` - Vector dimensionality

**UML Representation:**
```
┌─────────────────────────────────────┐
│              Vector                 │
│               <<Protocol>>          │
├─────────────────────────────────────┤
│ + id: str                          │
│ + data: np.ndarray                 │
│ + metadata: Optional[dict]         │
│ + dimension: int                   │
├─────────────────────────────────────┤
│ + distance(other, metric): float   │
│ + similarity(other, metric): float │
│ + normalize(): Vector              │
│ + to_dict(): dict                  │
│ + from_dict(data): Vector          │
└─────────────────────────────────────┘
```

### 2. Index (Protocol)

Manages efficient indexing and searching of vectors.

**Key Methods:**
- `add_vector(vector)` - Add vector to index
- `search_knn(query, k)` - K-nearest neighbors search
- `range_search(query, radius)` - Range-based search
- `build_index()` - Build/rebuild index structure

**Attributes:**
- `name: str` - Index identifier
- `dimension: int` - Vector dimension
- `size: int` - Number of indexed vectors
- `index_type: str` - Algorithm type

**UML Representation:**
```
┌─────────────────────────────────────┐
│               Index                 │
│               <<Protocol>>          │
├─────────────────────────────────────┤
│ + name: str                        │
│ + dimension: int                   │
│ + size: int                        │
│ + index_type: str                  │
├─────────────────────────────────────┤
│ + add_vector(vector): bool         │
│ + add_vectors(vectors): List[bool] │
│ + search_knn(query, k): List[...]  │
│ + range_search(query, r): List[...]│
│ + build_index(): bool              │
│ + save_index(path): bool           │
│ + load_index(path): bool           │
└─────────────────────────────────────┘
```

### 3. StorageEngine (Protocol)

Handles persistent storage and retrieval of vectors.

**Key Methods:**
- `store_vector(vector)` - Persist vector
- `retrieve_vector(id)` - Load vector by ID
- `query_by_metadata(filter)` - Filter by metadata
- `iterate_vectors()` - Batch iteration

**Attributes:**
- `storage_type: str` - Storage backend type
- `total_vectors: int` - Number of stored vectors
- `storage_size: int` - Storage size in bytes

**UML Representation:**
```
┌─────────────────────────────────────┐
│           StorageEngine             │
│               <<Protocol>>          │
├─────────────────────────────────────┤
│ + storage_type: str                │
│ + total_vectors: int               │
│ + storage_size: int                │
├─────────────────────────────────────┤
│ + store_vector(vector): bool       │
│ + retrieve_vector(id): Vector      │
│ + update_vector(vector): bool      │
│ + delete_vector(id): bool          │
│ + query_by_metadata(filter): List │
│ + iterate_vectors(): Iterator      │
│ + create_backup(path): bool        │
│ + restore_from_backup(path): bool  │
└─────────────────────────────────────┘
```

### 4. QueryProcessor (Protocol)

Handles query parsing, optimization, and execution.

**Key Methods:**
- `execute_knn_query()` - Execute k-NN search
- `execute_range_query()` - Execute range search
- `execute_hybrid_query()` - Combined vector/metadata search
- `parse_query()` - Parse query specification

**Attributes:**
- `supported_query_types: List[QueryType]` - Supported query types

**Query Types:**
- `KNN` - K-nearest neighbors
- `RANGE` - Range search
- `SIMILARITY` - Similarity search  
- `METADATA` - Metadata filtering
- `HYBRID` - Combined queries

**UML Representation:**
```
┌─────────────────────────────────────┐
│           QueryProcessor            │
│               <<Protocol>>          │
├─────────────────────────────────────┤
│ + supported_query_types: List      │
├─────────────────────────────────────┤
│ + parse_query(query): dict         │
│ + optimize_query(parsed): dict     │
│ + execute_knn_query(...): List     │
│ + execute_range_query(...): List   │
│ + execute_similarity_query(): List │
│ + execute_metadata_query(): List   │
│ + execute_hybrid_query(...): List  │
│ + execute_query(query): dict       │
│ + explain_query(query): dict       │
└─────────────────────────────────────┘
```

## System Relationships

```
     ┌─────────────────┐
     │ QueryProcessor  │ ──── Coordinates queries
     └─────────┬───────┘
               │
               │ uses
               ▼
     ┌─────────────────┐      ┌─────────────────┐
     │      Index      │◄────►│ StorageEngine   │
     └─────────┬───────┘      └─────────┬───────┘
               │                        │
               │ indexes                 │ stores
               ▼                        ▼
     ┌─────────────────┐                │
     │     Vector      │◄───────────────┘
     └─────────────────┘
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Type checking
mypy src/

# Code formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
```

### Project Structure

```
mlvectordb/
├── src/mlvectordb/
│   ├── interfaces/          # Protocol definitions
│   │   ├── vector.py
│   │   ├── index.py
│   │   ├── storage_engine.py
│   │   └── query_processor.py
│   └── implementations/     # Concrete implementations
│       └── simple_vector.py
├── tests/                   # Test suite
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Contributing

1. Implement new storage engines by following the `StorageEngine` protocol
2. Create new index types by implementing the `Index` protocol  
3. Add custom query processors using the `QueryProcessor` protocol
4. Extend vector functionality through the `Vector` protocol

## License

This project is developed as part of the Big Data Infrastructure course.
