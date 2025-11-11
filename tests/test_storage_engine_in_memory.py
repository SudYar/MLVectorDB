import unittest
from uuid import UUID, uuid4

import numpy as np

from src.mlvectordb.implementations.storage_engine_in_memory import StorageEngineInMemory
from src.mlvectordb.implementations.vector import Vector


class TestStorageEngineInMemory(unittest.TestCase):
    """Комплексные тесты для in-memory реализации хранилища."""

    def setUp(self):
        """Инициализация чистого хранилища перед каждым тестом."""
        self.storage = StorageEngineInMemory()
        # Создаем тестовые векторы с UUID
        self.vector1 = Vector([1.0, 2.0, 3.0], {"type": "test", "category": "A"})
        self.vector2 = Vector([4.0, 5.0, 6.0], {"type": "test", "category": "B"})
        self.vector3 = Vector([7.0, 8.0, 9.0], {"type": "demo", "category": "A"})
        self.sample_vectors = [self.vector1, self.vector2, self.vector3]

    def tearDown(self):
        """Очистка после каждого теста."""
        self.storage.clear_all()

    def test_initial_state(self):
        """Тест начального состояния хранилища."""
        self.assertEqual(self.storage.storage_type, "in-memory")
        self.assertEqual(self.storage.total_vectors, 0)
        self.assertEqual(self.storage.storage_size, 0)
        self.assertEqual(self.storage.list_namespaces, [])

    def test_write_and_read_single_vector(self):
        """Тест записи и чтения одного вектора."""
        result = self.storage.write(self.vector1, "test_ns")

        self.assertTrue(result)
        self.assertEqual(self.storage.total_vectors, 1)

        # Проверка чтения
        retrieved = self.storage.read(self.vector1.id, "test_ns")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, self.vector1.id)
        self.assertTrue((retrieved.values == np.array([1.0, 2.0, 3.0])).all())

    # def test_write_duplicate_overwrites(self):
    #     """Тест перезаписи вектора с существующим ID."""
    #     vector_v1 = Vector(self.vector1.id, [1.0, 2.0], {"version": 1})
    #     vector_v2 = Vector(self.vector1.id, [3.0, 4.0], {"version": 2})
    #
    #     self.storage.write(vector_v1, "ns1")
    #     self.storage.write(vector_v2, "ns1")
    #
    #     stored = self.storage.read(self.vector1.id, "ns1")
    #     self.assertEqual(stored.values, [3.0, 4.0])
    #     self.assertEqual(stored.metadata["version"], 2)
    #     self.assertEqual(self.storage.total_vectors, 1)

    def test_batch_write_vectors(self):
        """Тест пакетной записи векторов."""
        results = self.storage.write_vectors(self.sample_vectors, "batch_ns")

        self.assertTrue(all(results))
        self.assertEqual(self.storage.total_vectors, 3)
        self.assertIn("batch_ns", self.storage.list_namespaces)

    def test_read_nonexistent_vector(self):
        """Тест чтения несуществующего вектора."""
        result = self.storage.read(uuid4(), "any_ns")
        self.assertIsNone(result)

    def test_batch_read_vectors(self):
        """Тест пакетного чтения векторов."""
        self.storage.write_vectors(self.sample_vectors, "test_ns")

        ids_to_read = [self.vector1.id, self.vector2.id, uuid4()]  # последний не существует
        results = self.storage.read_vectors(ids_to_read, "test_ns")

        self.assertEqual(len(results), 3)
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])
        self.assertIsNone(results[2])

    def test_delete_existing_vector(self):
        """Тест удаления существующего вектора."""
        self.storage.write(self.vector1, "test_ns")

        result = self.storage.delete(self.vector1.id, "test_ns")

        self.assertTrue(result)
        self.assertFalse(self.storage.exists(self.vector1.id))
        self.assertEqual(self.storage.total_vectors, 0)

    def test_delete_nonexistent_vector(self):
        """Тест удаления несуществующего вектора."""
        result = self.storage.delete(uuid4(), "any_ns")
        self.assertFalse(result)

    def test_delete_cleans_empty_namespace(self):
        """Тест очистки пустого пространства имен после удаления."""
        self.storage.write(self.vector1, "temp_ns")
        self.assertIn("temp_ns", self.storage.list_namespaces)

        self.storage.delete(self.vector1.id, "temp_ns")

        self.assertNotIn("temp_ns", self.storage.list_namespaces)

    def test_exists_positive(self):
        """Тест проверки существования вектора."""
        self.storage.write(self.vector1, "test_ns")
        self.assertTrue(self.storage.exists(self.vector1.id))

    def test_exists_negative(self):
        """Тест проверки отсутствия вектора."""
        self.assertFalse(self.storage.exists(uuid4()))

    def test_clear_all(self):
        """Тест полной очистки хранилища."""
        self.storage.write_vectors(self.sample_vectors, "ns1")
        self.storage.write(self.vector1, "ns2")

        result = self.storage.clear_all()

        self.assertTrue(result)
        self.assertEqual(self.storage.total_vectors, 0)
        self.assertEqual(self.storage.list_namespaces, [])

    def test_namespace_operations(self):
        """Тест операций с пространствами имен."""
        # Записываем векторы в разные пространства
        self.storage.write(self.vector1, "ns1")
        self.storage.write(self.vector2, "ns2")

        # Проверяем список пространств
        namespaces = self.storage.list_namespaces
        self.assertEqual(len(namespaces), 2)
        self.assertIn("ns1", namespaces)
        self.assertIn("ns2", namespaces)

        # Проверяем карту пространств
        namespace_map = self.storage.namespace_map
        self.assertEqual(len(namespace_map["ns1"]), 1)
        self.assertEqual(len(namespace_map["ns2"]), 1)

        # Удаляем пространство
        result = self.storage.delete_namespace("ns1")
        self.assertTrue(result)
        self.assertNotIn("ns1", self.storage.list_namespaces)

    def test_storage_info(self):
        """Тест получения информации о хранилище."""
        self.storage.write_vectors(self.sample_vectors[:2], "ns1")
        self.storage.write_vectors(self.sample_vectors[2:], "ns2")

        info = self.storage.get_storage_info()

        self.assertEqual(info["storage_type"], "in-memory")
        self.assertEqual(info["total_vectors"], 3)
        self.assertEqual(info["namespace_count"], 2)
        self.assertEqual(info["vectors_per_namespace"]["ns1"], 2)
        self.assertEqual(info["vectors_per_namespace"]["ns2"], 1)

    def test_storage_size_calculation(self):
        """Тест расчета размера хранилища."""
        initial_size = self.storage.storage_size

        self.storage.write(self.vector1, "test_ns")

        self.assertGreater(self.storage.storage_size, initial_size)

    def test_different_namespaces_isolation(self):
        """Тест изоляции разных пространств имен."""
        vector_ns1 = Vector([1.0, 2.0], {})
        vector_ns2 = Vector([3.0, 4.0], {})

        self.storage.write(vector_ns1, "namespace_a")
        self.storage.write(vector_ns2, "namespace_b")

        # Проверяем, что векторы изолированы по пространствам
        vec_a = self.storage.read(vector_ns1.id, "namespace_a")
        vec_b = self.storage.read(vector_ns2.id, "namespace_b")

        self.assertIsNotNone(vec_a)
        self.assertIsNotNone(vec_b)
        np.testing.assert_array_equal(vec_a.values, np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(vec_b.values, np.array([3.0, 4.0], dtype=np.float32))

        # Проверяем, что нельзя прочитать вектор из чужого пространства
        self.assertIsNone(self.storage.read(vector_ns1.id, "namespace_b"))
        self.assertIsNone(self.storage.read(vector_ns2.id, "namespace_a"))

    def test_vector_equality(self):
        """Тест сравнения векторов (проверка __eq__ метода)."""
        # Два разных вектора с одинаковыми данными
        vector1 = Vector([1.0, 2.0], {"key": "value"})
        vector2 = Vector([1.0, 2.0], {"key": "value"})

        # Они должны быть разными объектами с разными ID
        self.assertNotEqual(vector1, vector2)
        self.assertNotEqual(vector1.id, vector2.id)

    def test_vector_metadata_persistence(self):
        """Тест сохранения и восстановления метаданных вектора."""
        complex_metadata = {
            "string": "value",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "data"},
            "none": None
        }

        vector = Vector([1.0, 2.0], complex_metadata)
        self.storage.write(vector, "test_ns")

        retrieved = self.storage.read(vector.id, "test_ns")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.metadata, complex_metadata)

if __name__ == "__main__":
    # Запуск тестов с детализированным выводом
    unittest.main(verbosity=2)