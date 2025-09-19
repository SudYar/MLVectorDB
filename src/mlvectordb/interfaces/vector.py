"""
Vector interface and entity for MLVectorDB.

This module defines the Vector Protocol interface which represents a vector 
entity in the vector database system.
"""

from typing import Protocol, List, Union, Any, Optional
import numpy as np
from typing_extensions import runtime_checkable


@runtime_checkable
class Vector(Protocol):
    """
    Protocol interface for Vector entity in MLVectorDB.
    
    A Vector represents a high-dimensional data point with an ID and metadata.
    It serves as the core entity stored and retrieved from the vector database.
    """
    
    @property
    def id(self) -> str:
        """
        Unique identifier for the vector.
        
        Returns:
            str: Unique vector identifier
        """
        ...
    
    @property
    def data(self) -> np.ndarray:
        """
        The vector data as a numpy array.
        
        Returns:
            np.ndarray: High-dimensional vector data
        """
        ...
    
    @property
    def metadata(self) -> Optional[dict[str, Any]]:
        """
        Optional metadata associated with the vector.
        
        Returns:
            Optional[dict[str, Any]]: Vector metadata
        """
        ...
    
    @property
    def dimension(self) -> int:
        """
        Dimensionality of the vector.
        
        Returns:
            int: Number of dimensions
        """
        ...
    
    def distance(self, other: "Vector", metric: str = "euclidean") -> float:
        """
        Calculate distance to another vector.
        
        Args:
            other: Another vector to compare with
            metric: Distance metric ("euclidean", "cosine", "manhattan")
            
        Returns:
            float: Distance between vectors
        """
        ...
    
    def similarity(self, other: "Vector", metric: str = "cosine") -> float:
        """
        Calculate similarity to another vector.
        
        Args:
            other: Another vector to compare with
            metric: Similarity metric ("cosine", "dot_product")
            
        Returns:
            float: Similarity score between vectors
        """
        ...
    
    def normalize(self) -> "Vector":
        """
        Return a normalized version of this vector.
        
        Returns:
            Vector: Normalized vector
        """
        ...
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert vector to dictionary representation.
        
        Returns:
            dict[str, Any]: Dictionary representation of the vector
        """
        ...
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vector":
        """
        Create vector from dictionary representation.
        
        Args:
            data: Dictionary containing vector data
            
        Returns:
            Vector: Vector instance created from dictionary
        """
        ...