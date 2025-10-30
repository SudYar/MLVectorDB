import unittest


from src.mlvectordb.implementations.storage_engine_in_memory import StorageEngineInMemory
from src.mlvectordb.implementations.vector import Vector


class TestStorageEngineInMemory(unittest.TestCase):
    """Comprehensive test suite for StorageEngineInMemory implementation."""

    def setUp(self):
        """Set up a fresh storage instance for each test."""
        self.storage = StorageEngineInMemory()
        self.sample_vectors = [
            Vector("v1", [1.0, 2.0, 3.0], {"type": "test", "category": "A"}),
            Vector("v2", [4.0, 5.0, 6.0], {"type": "test", "category": "B"}),
            Vector("v3", [7.0, 8.0, 9.0], {"type": "demo", "category": "A"}),
            Vector("v4", [10.0, 11.0, 12.0], {}),
        ]

    def tearDown(self):
        """Clean up after each test."""
        self.storage.clear_all()

    # ==================== PROPERTY TESTS ====================

    def test_initial_properties(self):
        """Test initial state properties."""
        self.assertEqual(self.storage.storage_type, "in-memory")
        self.assertEqual(self.storage.total_vectors, 0)
        self.assertIsInstance(self.storage.storage_size, int)
        self.assertEqual(self.storage.list_namespaces(), [])
        self.assertEqual(self.storage.namespace_map, {})

    def test_storage_size_increases_with_data(self):
        """Test that storage size increases when adding vectors."""
        initial_size = self.storage.storage_size

        vector = Vector("test_id", [1.0, 2.0, 3.0], {"meta": "data"})
        self.storage.write(vector, "test_ns")

        self.assertGreater(self.storage.storage_size, initial_size)

    # ==================== WRITE OPERATIONS TESTS ====================

    def test_write_single_vector(self):
        """Test writing a single vector."""
        vector = Vector("test_id", [1.0, 2.0, 3.0], {"meta": "data"})

        result = self.storage.write(vector, "test_namespace")

        self.assertTrue(result)
        self.assertTrue(self.storage.exists("test_id", "test_namespace"))
        self.assertEqual(self.storage.total_vectors, 1)

    def test_write_duplicate_vector_overwrites(self):
        """Test that writing duplicate vector overwrites the existing one."""
        vector1 = Vector("same_id", [1.0, 2.0], {"version": 1})
        vector2 = Vector("same_id", [3.0, 4.0], {"version": 2})

        self.storage.write(vector1, "ns1")
        self.storage.write(vector2, "ns1")

        stored_vector = self.storage.read("same_id", "ns1")
        self.assertIsNotNone(stored_vector)
        self.assertEqual(stored_vector.values, [3.0, 4.0])
        self.assertEqual(stored_vector.metadata["version"], 2)
        self.assertEqual(self.storage.total_vectors, 1)

    def test_write_vectors_batch(self):
        """Test batch writing of vectors."""
        results = self.storage.write_vectors(self.sample_vectors, "batch_ns")

        self.assertTrue(all(results))
        self.assertEqual(self.storage.total_vectors, len(self.sample_vectors))
        self.assertIn("batch_ns", self.storage.list_namespaces())

    def test_write_to_different_namespaces(self):
        """Test writing vectors to different namespaces."""
        vector1 = Vector("v1", [1.0, 2.0], {})
        vector2 = Vector("v2", [3.0, 4.0], {})

        self.storage.write(vector1, "namespace_a")
        self.storage.write(vector2, "namespace_b")

        self.assertEqual(self.storage.total_vectors, 2)
        self.assertEqual(len(self.storage.list_namespaces()), 2)
        self.assertTrue(self.storage.exists("v1", "namespace_a"))
        self.assertTrue(self.storage.exists("v2", "namespace_b"))

    # ==================== READ OPERATIONS TESTS ====================

    def test_read_existing_vector(self):
        """Test reading an existing vector."""
        vector = Vector("test_id", [1.0, 2.0, 3.0], {"key": "value"})
        self.storage.write(vector, "test_ns")

        retrieved = self.storage.read("test_id", "test_ns")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, "test_id")
        self.assertEqual(retrieved.values, [1.0, 2.0, 3.0])
        self.assertEqual(retrieved.metadata, {"key": "value"})

    def test_read_nonexistent_vector(self):
        """Test reading a non-existent vector."""
        result = self.storage.read("nonexistent", "default")
        self.assertIsNone(result)

    def test_read_nonexistent_namespace(self):
        """Test reading from non-existent namespace."""
        result = self.storage.read("any_id", "nonexistent_ns")
        self.assertIsNone(result)

    def test_read_vectors_batch(self):
        """Test batch reading of vectors."""
        self.storage.write_vectors(self.sample_vectors, "test_ns")

        ids_to_read = ["v1", "v2", "v5"]  # v5 doesn't exist
        results = self.storage.read_vectors(ids_to_read, "test_ns")

        self.assertEqual(len(results), 3)
        self.assertIsNotNone(results[0])  # v1 exists
        self.assertIsNotNone(results[1])  # v2 exists
        self.assertIsNone(results[2])  # v5 doesn't exist

    def test_read_vectors_empty_list(self):
        """Test reading with empty ID list."""
        results = self.storage.read_vectors([], "default")
        self.assertEqual(results, [])

    # ==================== DELETE OPERATIONS TESTS ====================

    def test_delete_existing_vector(self):
        """Test deleting an existing vector."""
        vector = Vector("to_delete", [1.0, 2.0], {})
        self.storage.write(vector, "test_ns")

        result = self.storage.delete("to_delete", "test_ns")

        self.assertTrue(result)
        self.assertFalse(self.storage.exists("to_delete", "test_ns"))
        self.assertEqual(self.storage.total_vectors, 0)

    def test_delete_nonexistent_vector(self):
        """Test deleting a non-existent vector."""
        result = self.storage.delete("nonexistent", "default")
        self.assertFalse(result)

    def test_delete_from_nonexistent_namespace(self):
        """Test deleting from non-existent namespace."""
        result = self.storage.delete("any_id", "nonexistent_ns")
        self.assertFalse(result)

    def test_delete_cleans_up_empty_namespace(self):
        """Test that deleting last vector cleans up the namespace."""
        vector = Vector("only_vector", [1.0, 2.0], {})
        self.storage.write(vector, "temp_ns")

        self.assertIn("temp_ns", self.storage.list_namespaces())

        self.storage.delete("only_vector", "temp_ns")

        self.assertNotIn("temp_ns", self.storage.list_namespaces())

    # ==================== EXISTS OPERATION TESTS ====================

    def test_exists_positive(self):
        """Test exists returns True for existing vector."""
        vector = Vector("existing", [1.0, 2.0], {})
        self.storage.write(vector, "test_ns")

        self.assertTrue(self.storage.exists("existing", "test_ns"))

    def test_exists_negative(self):
        """Test exists returns False for non-existent vector."""
        self.assertFalse(self.storage.exists("nonexistent", "default"))

    def test_exists_wrong_namespace(self):
        """Test exists returns False for vector in wrong namespace."""
        vector = Vector("v1", [1.0, 2.0], {})
        self.storage.write(vector, "ns1")

        self.assertFalse(self.storage.exists("v1", "ns2"))

    # ==================== ITERATION TESTS ====================

    def test_iterate_vectors_empty(self):
        """Test iteration over empty namespace."""
        batches = list(self.storage.iterate_vectors("empty_ns"))
        self.assertEqual(batches, [])

    def test_iterate_vectors_single_batch(self):
        """Test iteration with single batch."""
        self.storage.write_vectors(self.sample_vectors[:2], "test_ns")

        batches = list(self.storage.iterate_vectors("test_ns", batch_size=10))

        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 2)

    def test_iterate_vectors_multiple_batches(self):
        """Test iteration with multiple batches."""
        self.storage.write_vectors(self.sample_vectors, "test_ns")

        batches = list(self.storage.iterate_vectors("test_ns", batch_size=2))

        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 2)

    def test_iterate_vectors_correct_content(self):
        """Test that iteration returns correct vectors."""
        self.storage.write_vectors(self.sample_vectors[:2], "test_ns")

        batches = list(self.storage.iterate_vectors("test_ns"))
        all_vectors = [vec for batch in batches for vec in batch]

        self.assertEqual(len(all_vectors), 2)
        vector_ids = {vec.id for vec in all_vectors}
        self.assertEqual(vector_ids, {"v1", "v2"})

    # ==================== NAMESPACE OPERATIONS TESTS ====================

    def test_namespace_map(self):
        """Test namespace map property."""
        self.storage.write(Vector("v1", [1.0], {}), "ns1")
        self.storage.write(Vector("v2", [2.0], {}), "ns1")
        self.storage.write(Vector("v3", [3.0], {}), "ns2")

        namespace_map = self.storage.namespace_map

        self.assertEqual(len(namespace_map), 2)
        self.assertEqual(len(namespace_map["ns1"]), 2)
        self.assertEqual(len(namespace_map["ns2"]), 1)
        self.assertEqual(namespace_map["ns1"][0].id, "v1")
        self.assertEqual(namespace_map["ns1"][1].id, "v2")

    def test_delete_namespace(self):
        """Test deleting entire namespace."""
        self.storage.write(Vector("v1", [1.0], {}), "to_delete")
        self.storage.write(Vector("v2", [2.0], {}), "to_keep")

        result = self.storage.delete_namespace("to_delete")

        self.assertTrue(result)
        self.assertNotIn("to_delete", self.storage.list_namespaces())
        self.assertIn("to_keep", self.storage.list_namespaces())

    def test_delete_nonexistent_namespace(self):
        """Test deleting non-existent namespace."""
        result = self.storage.delete_namespace("nonexistent")
        self.assertFalse(result)

    def test_list_namespaces(self):
        """Test listing all namespaces."""
        self.storage.write(Vector("v1", [1.0], {}), "ns1")
        self.storage.write(Vector("v2", [2.0], {}), "ns2")
        self.storage.write(Vector("v3", [3.0], {}), "ns3")

        namespaces = self.storage.list_namespaces()

        self.assertEqual(len(namespaces), 3)
        self.assertIn("ns1", namespaces)
        self.assertIn("ns2", namespaces)
        self.assertIn("ns3", namespaces)

    # ==================== STORAGE INFO TESTS ====================

    def test_get_storage_info_empty(self):
        """Test storage info for empty storage."""
        info = self.storage.get_storage_info()

        self.assertEqual(info["storage_type"], "in-memory")
        self.assertEqual(info["total_vectors"], 0)
        self.assertEqual(info["namespace_count"], 0)
        self.assertEqual(info["namespaces"], [])
        self.assertEqual(info["vectors_per_namespace"], {})

    def test_get_storage_info_with_data(self):
        """Test storage info with data."""
        self.storage.write_vectors(self.sample_vectors[:2], "ns1")
        self.storage.write_vectors(self.sample_vectors[2:], "ns2")

        info = self.storage.get_storage_info()

        self.assertEqual(info["total_vectors"], 4)
        self.assertEqual(info["namespace_count"], 2)
        self.assertIn("ns1", info["namespaces"])
        self.assertIn("ns2", info["namespaces"])
        self.assertEqual(info["vectors_per_namespace"]["ns1"], 2)
        self.assertEqual(info["vectors_per_namespace"]["ns2"], 2)

    # ==================== CLEAR OPERATIONS TESTS ====================

    def test_clear_all(self):
        """Test clearing all data from storage."""
        self.storage.write_vectors(self.sample_vectors, "ns1")
        self.storage.write(Vector("v5", [13.0, 14.0], {}), "ns2")

        self.assertEqual(self.storage.total_vectors, 5)

        result = self.storage.clear_all()

        self.assertTrue(result)
        self.assertEqual(self.storage.total_vectors, 0)
        self.assertEqual(self.storage.list_namespaces(), [])

    def test_clear_all_empty(self):
        """Test clearing already empty storage."""
        result = self.storage.clear_all()
        self.assertTrue(result)

    # ==================== EDGE CASES TESTS ====================

    def test_large_batch_size(self):
        """Test iteration with batch size larger than total vectors."""
        self.storage.write_vectors(self.sample_vectors, "test_ns")

        batches = list(self.storage.iterate_vectors("test_ns", batch_size=1000))

        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 4)

    def test_batch_size_one(self):
        """Test iteration with batch size of 1."""
        self.storage.write_vectors(self.sample_vectors, "test_ns")

        batches = list(self.storage.iterate_vectors("test_ns", batch_size=1))

        self.assertEqual(len(batches), 4)
        for batch in batches:
            self.assertEqual(len(batch), 1)

    def test_vector_metadata_persistence(self):
        """Test that vector metadata is properly stored and retrieved."""
        complex_metadata = {
            "string": "value",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "data"},
            "none": None
        }

        vector = Vector("complex", [1.0, 2.0], complex_metadata)
        self.storage.write(vector, "test_ns")

        retrieved = self.storage.read("complex", "test_ns")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.metadata, complex_metadata)


if __name__ == "__main__":
    unittest.main()
