import pytest
import numpy as np
from uuid import UUID
from src.mlvectordb.implementations.query_processor import QueryProcessor
from src.mlvectordb.implementations.index import Index
from src.mlvectordb.interfaces.vector import VectorDTO
from src.mlvectordb.implementations.vector import Vector


class InMemoryStorageEngine:
    def __init__(self):
        self.namespaceMap = {}

    def _ns(self, namespace: str):
        return self.namespaceMap.setdefault(namespace, {})

    def write(self, vector: Vector, namespace: str) -> None:
        self._ns(namespace)[vector.id] = vector

    def read(self, vector_id: UUID, namespace: str):
        return self._ns(namespace).get(vector_id)

    def read_batch(self, ids, namespace: str):
        ns = self._ns(namespace)
        return [ns[i] for i in ids if i in ns]

    def delete(self, vector_id: UUID, namespace: str = "default") -> None:
        ns = self._ns(namespace)
        ns.pop(vector_id, None)


@pytest.fixture
def storage():
    return InMemoryStorageEngine()


@pytest.fixture
def index():
    return Index(space="cosine")


@pytest.fixture
def processor(storage, index):
    return QueryProcessor(storage, index)


def make_vector(x, y, z, label):
    return VectorDTO(values=[x, y, z], metadata={"label": label})


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_insert_and_storage_integrity(processor):
    v1 = make_vector(1, 0, 0, "A")
    v2 = make_vector(0, 1, 0, "B")

    processor.insert(v1)
    processor.insert(v2)

    ns = "default"
    stored_ids = list(processor._storage._ns(ns).keys())
    assert len(stored_ids) == 2

    vectors = list(processor._storage._ns(ns).values())
    assert any(v.metadata["label"] == "A" for v in vectors)
    assert any(v.metadata["label"] == "B" for v in vectors)


def test_find_similar_correctness(processor):
    v1 = make_vector(1, 0, 0, "A")
    v2 = make_vector(0, 1, 0, "B")
    v3 = make_vector(0.8, 0.2, 0, "C")
    processor.upsert_many([v1, v2, v3])

    query = VectorDTO(values=[0.9, 0.1, 0], metadata={})
    results = processor.find_similar(query, top_k=3)

    assert len(results) == 3
    labels = [r["metadata"]["label"] for r in results]
    assert "A" in labels and "B" in labels and "C" in labels

    sims = [cosine_similarity(query.values, r["values"]) for r in results]
    sorted_sims = sorted(sims, reverse=True)
    assert sims == pytest.approx(sorted_sims, rel=1e-4)


def test_namespace_isolation(processor):
    ns1, ns2 = "alpha", "beta"
    v1 = make_vector(1, 0, 0, "X")
    v2 = make_vector(0, 1, 0, "Y")

    processor.insert(v1, namespace=ns1)
    processor.insert(v2, namespace=ns2)

    query = VectorDTO(values=[1, 0, 0], metadata={})

    res1 = processor.find_similar(query, top_k=1, namespace=ns1)
    res2 = processor.find_similar(query, top_k=1, namespace=ns2)

    assert res1[0]["metadata"]["label"] == "X"
    assert res2[0]["metadata"]["label"] == "Y"
    assert res1[0]["id"] != res2[0]["id"]


def test_delete_removes_from_storage_and_index(processor):
    v1 = make_vector(1, 0, 0, "A")
    v2 = make_vector(0, 1, 0, "B")
    processor.upsert_many([v1, v2])
    query = VectorDTO(values=[1, 0, 0], metadata={})
    before = processor.find_similar(query, top_k=2)
    assert len(before) == 2

    to_delete = [before[0]["id"]]
    processor.delete(to_delete)

    ns = "default"
    storage_data = processor._storage._ns(ns)
    assert to_delete[0] not in storage_data

    after = processor.find_similar(query, top_k=2)
    remaining_ids = [r["id"] for r in after]
    assert to_delete[0] not in remaining_ids


def test_search_with_many_vectors(processor):
    np.random.seed(42)
    vectors = [
        VectorDTO(values=np.random.rand(10).tolist(), metadata={"label": f"V{i}"})
        for i in range(100)
    ]
    processor.upsert_many(vectors)

    query = VectorDTO(values=np.random.rand(10).tolist(), metadata={})
    results = processor.find_similar(query, top_k=5)
    assert len(results) == 5
    assert all(isinstance(r["id"], UUID) for r in results)


def test_search_with_few_vectors(processor):
    vectors = [
        VectorDTO(values=[1, 0, 0], metadata={"label": "A"}),
        VectorDTO(values=[0, 1, 0], metadata={"label": "B"}),
    ]
    processor.upsert_many(vectors)
    query = VectorDTO(values=[1, 0, 0], metadata={})
    results = processor.find_similar(query, top_k=5)
    assert len(results) <= 2
    assert results[0]["metadata"]["label"] == "A"
