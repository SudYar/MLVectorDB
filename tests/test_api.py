"""Tests for the MLVectorDB REST API."""

import pytest
from fastapi.testclient import TestClient
from mlvectordb.api.main import app

# Create test client
client = TestClient(app)


class TestAPIEndpoints:
    """Test the REST API endpoints."""
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "MLVectorDB API"
        assert data["version"] == "0.1.0"
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data
        assert data["query_processor_status"] == "active"
    
    def test_knn_query(self):
        """Test KNN query endpoint."""
        query_data = {
            "type": "knn",
            "vector": [1.0, 2.0, 3.0],
            "k": 5
        }
        response = client.post("/query/knn", json=query_data)
        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "knn"
        assert "results" in data
        assert "total_results" in data
        assert "execution_time_ms" in data
    
    def test_range_query(self):
        """Test range query endpoint."""
        query_data = {
            "type": "range",
            "vector": [0.5, 1.5, 2.5],
            "radius": 2.0
        }
        response = client.post("/query/range", json=query_data)
        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "range"
        assert "results" in data
    
    def test_similarity_query(self):
        """Test similarity query endpoint."""
        query_data = {
            "type": "similarity",
            "vector": [1.0, 0.0, 0.0],
            "threshold": 0.7,
            "metric": "cosine"
        }
        response = client.post("/query/similarity", json=query_data)
        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "similarity"
        assert "results" in data
    
    def test_metadata_query(self):
        """Test metadata query endpoint."""
        query_data = {
            "type": "metadata",
            "filter": {"category": "test", "active": True}
        }
        response = client.post("/query/metadata", json=query_data)
        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "metadata"
        assert "results" in data
    
    def test_general_query_endpoint(self):
        """Test the general query endpoint."""
        query_data = {
            "type": "knn",
            "vector": [1.0, 2.0, 3.0],
            "k": 3
        }
        response = client.post("/query", json=query_data)
        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "knn"
        assert len(data["results"]) <= 3
    
    def test_explain_query(self):
        """Test query explanation endpoint."""
        query_data = {
            "type": "knn",
            "vector": [1.0, 2.0, 3.0],
            "k": 5
        }
        response = client.post("/query/explain", json=query_data)
        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "knn"
        assert "execution_plan" in data
        assert "steps" in data["execution_plan"]
    
    def test_statistics_endpoint(self):
        """Test statistics endpoint."""
        response = client.get("/statistics")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "knn_queries" in data
        assert "cache_hits" in data
    
    def test_query_types_endpoint(self):
        """Test query types endpoint."""
        response = client.get("/query-types")
        assert response.status_code == 200
        data = response.json()
        assert "supported_query_types" in data
        assert "descriptions" in data
        assert "knn" in data["supported_query_types"]
        assert "range" in data["supported_query_types"]
    
    def test_cache_clear(self):
        """Test cache clear endpoint."""
        response = client.post("/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data
    
    def test_example_endpoints(self):
        """Test example endpoints."""
        # Test example KNN
        response = client.post("/example/knn")
        assert response.status_code == 200
        
        # Test example range
        response = client.post("/example/range")
        assert response.status_code == 200
        
        # Test example similarity
        response = client.post("/example/similarity")
        assert response.status_code == 200
        
        # Test example metadata
        response = client.post("/example/metadata")
        assert response.status_code == 200
    
    def test_invalid_query_type(self):
        """Test handling of invalid query type."""
        query_data = {
            "type": "invalid_type",
            "vector": [1.0, 2.0, 3.0]
        }
        response = client.post("/query", json=query_data)
        assert response.status_code == 422  # Pydantic validation error
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        # KNN query without k parameter
        query_data = {
            "type": "knn",
            "vector": [1.0, 2.0, 3.0]
            # Missing k parameter
        }
        response = client.post("/query/knn", json=query_data)
        assert response.status_code == 422
        
        # Range query without radius
        query_data = {
            "type": "range",
            "vector": [1.0, 2.0, 3.0]
            # Missing radius parameter
        }
        response = client.post("/query/range", json=query_data)
        assert response.status_code == 422


class TestQueryProcessor:
    """Test the BasicQueryProcessor implementation."""
    
    def test_query_processor_initialization(self):
        """Test QueryProcessor initialization."""
        from mlvectordb.implementations.basic_query_processor import BasicQueryProcessor
        
        processor = BasicQueryProcessor()
        assert len(processor.supported_query_types) == 5
        
    def test_query_statistics_updates(self):
        """Test that query statistics are updated correctly."""
        # Execute a few queries and check statistics
        query_data = {
            "type": "knn", 
            "vector": [1.0, 2.0, 3.0],
            "k": 3
        }
        
        # Get initial stats
        stats_before = client.get("/statistics").json()
        initial_total = stats_before["total_queries"]
        initial_knn = stats_before["knn_queries"]
        
        # Execute query
        client.post("/query", json=query_data)
        
        # Check updated stats
        stats_after = client.get("/statistics").json()
        assert stats_after["total_queries"] == initial_total + 1
        assert stats_after["knn_queries"] == initial_knn + 1


if __name__ == "__main__":
    pytest.main([__file__])