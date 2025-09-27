"""Basic tests for MLVectorDB interfaces."""

import pytest
import numpy as np
from mlvectordb.interfaces.vector import Vector
from mlvectordb.interfaces.index import Index
from mlvectordb.interfaces.storage_engine import StorageEngine
from mlvectordb.interfaces.query_processor import QueryProcessor
from mlvectordb.implementations.simple_vector import SimpleVector


class TestVectorInterface:
    """Test the Vector interface and SimpleVector implementation."""
    
    def test_simple_vector_creation(self):
        """Test creating a SimpleVector."""
        data = np.array([1.0, 2.0, 3.0])
        metadata = {"category": "test", "timestamp": 123456}
        
        vector = SimpleVector(
            id="test_vector_1",
            data=data,
            metadata=metadata
        )
        
        assert vector.id == "test_vector_1"
        assert np.array_equal(vector.data, data)
        assert vector.metadata == metadata
        assert vector.dimension == 3
    
    def test_vector_protocol_compliance(self):
        """Test that SimpleVector implements Vector protocol."""
        vector = SimpleVector("test", np.array([1.0, 2.0, 3.0]))
        assert isinstance(vector, Vector)
    
    def test_vector_distance_calculation(self):
        """Test distance calculations between vectors."""
        vector1 = SimpleVector("v1", np.array([1.0, 0.0, 0.0]))
        vector2 = SimpleVector("v2", np.array([0.0, 1.0, 0.0]))
        
        # Euclidean distance
        euclidean_dist = vector1.distance(vector2, "euclidean")
        expected_euclidean = np.sqrt(2.0)  # sqrt(1^2 + 1^2)
        assert abs(euclidean_dist - expected_euclidean) < 1e-6
        
        # Manhattan distance
        manhattan_dist = vector1.distance(vector2, "manhattan")
        assert manhattan_dist == 2.0  # |1-0| + |0-1| + |0-0|
    
    def test_vector_similarity_calculation(self):
        """Test similarity calculations between vectors."""
        vector1 = SimpleVector("v1", np.array([1.0, 0.0, 0.0]))
        vector2 = SimpleVector("v2", np.array([1.0, 0.0, 0.0]))
        vector3 = SimpleVector("v3", np.array([0.0, 1.0, 0.0]))
        
        # Cosine similarity - identical vectors
        cosine_sim_identical = vector1.similarity(vector2, "cosine")
        assert abs(cosine_sim_identical - 1.0) < 1e-6
        
        # Cosine similarity - orthogonal vectors
        cosine_sim_orthogonal = vector1.similarity(vector3, "cosine")
        assert abs(cosine_sim_orthogonal - 0.0) < 1e-6
    
    def test_vector_normalization(self):
        """Test vector normalization."""
        vector = SimpleVector("v1", np.array([3.0, 4.0, 0.0]))
        normalized = vector.normalize()
        
        # Check that the normalized vector has unit length
        norm = np.linalg.norm(normalized.data)
        assert abs(norm - 1.0) < 1e-6
        
        # Check that the direction is preserved
        expected = np.array([0.6, 0.8, 0.0])
        assert np.allclose(normalized.data, expected)
    
    def test_vector_serialization(self):
        """Test vector to/from dictionary conversion."""
        original_data = np.array([1.5, 2.5, 3.5])
        original_metadata = {"type": "test", "value": 42}
        
        vector = SimpleVector(
            id="serialize_test",
            data=original_data,
            metadata=original_metadata
        )
        
        # Convert to dictionary
        vector_dict = vector.to_dict()
        
        # Check dictionary contents
        assert vector_dict["id"] == "serialize_test"
        assert vector_dict["data"] == original_data.tolist()
        assert vector_dict["metadata"] == original_metadata
        assert vector_dict["dimension"] == 3
        
        # Recreate from dictionary
        recreated = SimpleVector.from_dict(vector_dict)
        
        # Verify the recreated vector is equivalent
        assert recreated.id == vector.id
        assert np.array_equal(recreated.data, vector.data)
        assert recreated.metadata == vector.metadata
        assert recreated.dimension == vector.dimension


def test_protocol_interfaces_are_importable():
    """Test that all protocol interfaces can be imported without errors."""
    # These imports should not raise any exceptions
    from mlvectordb.interfaces.vector import Vector
    from mlvectordb.interfaces.index import Index
    from mlvectordb.interfaces.storage_engine import StorageEngine
    from mlvectordb.interfaces.query_processor import QueryProcessor, QueryType
    
    # Test that the classes are protocols
    assert hasattr(Vector, "__protocol_attrs__")
    assert hasattr(Index, "__protocol_attrs__")
    assert hasattr(StorageEngine, "__protocol_attrs__")
    assert hasattr(QueryProcessor, "__protocol_attrs__")


if __name__ == "__main__":
    pytest.main([__file__])