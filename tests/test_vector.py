import unittest
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Optional, Mapping, Any
from src.mlvectordb.implementations.vector import Vector 

class TestVector(unittest.TestCase):

    def test_initialization_and_metadata_default(self):
        v = Vector(id="v1", values=[1.0, 2.0, 3.0])
        self.assertEqual(v.id, "v1")
        self.assertEqual(v.namespace, "default")
        self.assertEqual(v.metadata, {})
        np.testing.assert_array_equal(v.to_numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_initialization_with_metadata(self):
        meta = {"key": "value"}
        v = Vector(id="v2", values=[0.5, 0.5], metadata=meta)
        self.assertEqual(v.metadata, meta)
        np.testing.assert_array_equal(v.to_numpy(), np.array([0.5, 0.5], dtype=np.float32))

    def test_to_numpy_returns_array(self):
        values = [1.0, 2.0, 3.0, 4.0]
        v = Vector(id="v3", values=values)
        arr = v.to_numpy()
        self.assertIsInstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array(values, dtype=np.float32))

    def test_dimension_method(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        v = Vector(id="v4", values=values)
        self.assertEqual(v.dimension(), 5)

    def test_namespace_can_be_set(self):
        v = Vector(id="v5", values=[1.0], namespace="custom_ns")
        self.assertEqual(v.namespace, "custom_ns")

    def test_values_are_copied_to_numpy_array(self):
        values = [1.0, 2.0]
        v = Vector(id="v6", values=values)
        values[0] = 99.0
        np.testing.assert_array_equal(v.to_numpy(), np.array([1.0, 2.0], dtype=np.float32))

if __name__ == "__main__":
    unittest.main()
