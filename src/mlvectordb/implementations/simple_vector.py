"""
Simple implementation of Vector interface for MLVectorDB.

This module provides a concrete implementation of the Vector Protocol.
"""

from typing import Optional, Dict, Any
import numpy as np
from ..interfaces.vector import Vector as VectorProtocol


class SimpleVector:
    """
    Simple concrete implementation of the Vector Protocol.
    
    This implementation stores vector data in numpy arrays and provides
    basic distance and similarity calculations.
    """
    
    def __init__(
        self, 
        id: str, 
        data: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a SimpleVector.
        
        Args:
            id: Unique identifier for the vector
            data: Vector data as numpy array
            metadata: Optional metadata dictionary
        """
        self._id = id
        self._data = np.array(data, dtype=np.float32)
        self._metadata = metadata or {}
    
    @property
    def id(self) -> str:
        """Unique identifier for the vector."""
        return self._id
    
    @property
    def data(self) -> np.ndarray:
        """The vector data as a numpy array."""
        return self._data.copy()
    
    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """Optional metadata associated with the vector."""
        return self._metadata.copy() if self._metadata else None
    
    @property
    def dimension(self) -> int:
        """Dimensionality of the vector."""
        return len(self._data)
    
    def distance(self, other: VectorProtocol, metric: str = "euclidean") -> float:
        """
        Calculate distance to another vector.
        
        Args:
            other: Another vector to compare with
            metric: Distance metric ("euclidean", "cosine", "manhattan")
            
        Returns:
            float: Distance between vectors
        """
        if self.dimension != other.dimension:
            raise ValueError("Vector dimensions must match")
        
        if metric == "euclidean":
            return float(np.linalg.norm(self._data - other.data))
        elif metric == "manhattan":
            return float(np.sum(np.abs(self._data - other.data)))
        elif metric == "cosine":
            # Convert cosine similarity to distance
            similarity = self.similarity(other, "cosine")
            return 1.0 - similarity
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
    
    def similarity(self, other: VectorProtocol, metric: str = "cosine") -> float:
        """
        Calculate similarity to another vector.
        
        Args:
            other: Another vector to compare with
            metric: Similarity metric ("cosine", "dot_product")
            
        Returns:
            float: Similarity score between vectors
        """
        if self.dimension != other.dimension:
            raise ValueError("Vector dimensions must match")
        
        if metric == "cosine":
            norm_self = np.linalg.norm(self._data)
            norm_other = np.linalg.norm(other.data)
            if norm_self == 0 or norm_other == 0:
                return 0.0
            return float(np.dot(self._data, other.data) / (norm_self * norm_other))
        elif metric == "dot_product":
            return float(np.dot(self._data, other.data))
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def normalize(self) -> "SimpleVector":
        """
        Return a normalized version of this vector.
        
        Returns:
            SimpleVector: Normalized vector
        """
        norm = np.linalg.norm(self._data)
        if norm == 0:
            normalized_data = self._data.copy()
        else:
            normalized_data = self._data / norm
        
        return SimpleVector(
            id=self._id,
            data=normalized_data,
            metadata=self._metadata.copy() if self._metadata else None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert vector to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the vector
        """
        return {
            "id": self._id,
            "data": self._data.tolist(),
            "metadata": self._metadata,
            "dimension": self.dimension
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleVector":
        """
        Create vector from dictionary representation.
        
        Args:
            data: Dictionary containing vector data
            
        Returns:
            SimpleVector: Vector instance created from dictionary
        """
        return cls(
            id=data["id"],
            data=np.array(data["data"], dtype=np.float32),
            metadata=data.get("metadata")
        )
    
    def __str__(self) -> str:
        """String representation of the vector."""
        return f"SimpleVector(id='{self._id}', dim={self.dimension})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the vector."""
        return (f"SimpleVector(id='{self._id}', data={self._data.tolist()}, "
                f"metadata={self._metadata})")