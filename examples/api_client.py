"""
Example client for MLVectorDB REST API.

This script demonstrates how to interact with the MLVectorDB REST API.
"""

import requests
import json
from typing import Dict, Any, List


class MLVectorDBClient:
    """Simple client for MLVectorDB REST API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client with base URL."""
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def knn_query(self, vector: List[float], k: int, **kwargs) -> Dict[str, Any]:
        """Execute a K-nearest neighbors query."""
        data = {
            "type": "knn",
            "vector": vector,
            "k": k,
            **kwargs
        }
        response = self.session.post(f"{self.base_url}/query/knn", json=data)
        response.raise_for_status()
        return response.json()
    
    def range_query(self, vector: List[float], radius: float, **kwargs) -> Dict[str, Any]:
        """Execute a range search query."""
        data = {
            "type": "range",
            "vector": vector,
            "radius": radius,
            **kwargs
        }
        response = self.session.post(f"{self.base_url}/query/range", json=data)
        response.raise_for_status()
        return response.json()
    
    def similarity_query(
        self, vector: List[float], threshold: float, metric: str = "cosine", **kwargs
    ) -> Dict[str, Any]:
        """Execute a similarity search query."""
        data = {
            "type": "similarity",
            "vector": vector,
            "threshold": threshold,
            "metric": metric,
            **kwargs
        }
        response = self.session.post(f"{self.base_url}/query/similarity", json=data)
        response.raise_for_status()
        return response.json()
    
    def metadata_query(self, filter_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute a metadata-based query."""
        data = {
            "type": "metadata",
            "filter": filter_dict,
            **kwargs
        }
        response = self.session.post(f"{self.base_url}/query/metadata", json=data)
        response.raise_for_status()
        return response.json()
    
    def explain_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get execution plan explanation for a query."""
        response = self.session.post(f"{self.base_url}/query/explain", json=query_data)
        response.raise_for_status()
        return response.json()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get query processor statistics."""
        response = self.session.get(f"{self.base_url}/statistics")
        response.raise_for_status()
        return response.json()
    
    def get_supported_query_types(self) -> Dict[str, Any]:
        """Get supported query types."""
        response = self.session.get(f"{self.base_url}/query-types")
        response.raise_for_status()
        return response.json()


def main():
    """Demonstrate client usage."""
    print("MLVectorDB API Client Example")
    print("=" * 40)
    
    # Initialize client
    client = MLVectorDBClient()
    
    try:
        # Health check
        print("\n1. Health Check:")
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        
        # Get supported query types
        print("\n2. Supported Query Types:")
        query_types = client.get_supported_query_types()
        for qtype, description in query_types["descriptions"].items():
            print(f"  - {qtype}: {description}")
        
        # Example KNN query
        print("\n3. KNN Query Example:")
        knn_result = client.knn_query(
            vector=[1.0, 2.0, 3.0],
            k=3
        )
        print(f"Query type: {knn_result['query_type']}")
        print(f"Results found: {knn_result['total_results']}")
        print(f"Execution time: {knn_result['execution_time_ms']:.2f}ms")
        
        if knn_result['results']:
            print("First result:")
            first_result = knn_result['results'][0]
            print(f"  ID: {first_result['id']}")
            print(f"  Score: {first_result['score']:.4f}")
        
        # Example range query
        print("\n4. Range Query Example:")
        range_result = client.range_query(
            vector=[0.5, 1.5, 2.5],
            radius=2.0
        )
        print(f"Results within radius: {range_result['total_results']}")
        
        # Example similarity query
        print("\n5. Similarity Query Example:")
        sim_result = client.similarity_query(
            vector=[1.0, 0.0, 0.0],
            threshold=0.7
        )
        print(f"Similar vectors found: {sim_result['total_results']}")
        
        # Example metadata query
        print("\n6. Metadata Query Example:")
        meta_result = client.metadata_query(
            filter_dict={"category": "test", "active": True}
        )
        print(f"Matching vectors: {meta_result['total_results']}")
        
        # Query explanation
        print("\n7. Query Explanation Example:")
        explanation = client.explain_query({
            "type": "knn",
            "vector": [1.0, 2.0, 3.0],
            "k": 5
        })
        print(f"Execution steps: {len(explanation['execution_plan']['steps'])}")
        print("Steps:")
        for step in explanation['execution_plan']['steps']:
            print(f"  - {step}")
        
        # Statistics
        print("\n8. Query Statistics:")
        stats = client.get_statistics()
        print(f"Total queries: {stats['total_queries']}")
        print(f"KNN queries: {stats['knn_queries']}")
        print(f"Range queries: {stats['range_queries']}")
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server.")
        print("Make sure the server is running: python -m mlvectordb.api.server")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()