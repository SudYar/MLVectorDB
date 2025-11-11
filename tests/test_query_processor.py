import pytest
import numpy as np
from uuid import UUID

from src.mlvectordb import StorageEngineInMemory
from src.mlvectordb.implementations.query_processor import QueryProcessor
from src.mlvectordb.implementations.index import Index
from src.mlvectordb.interfaces.vector import VectorDTO


@pytest.fixture
def storage():
    return StorageEngineInMemory()


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
    count_vectors = processor._storage.total_vectors
    assert count_vectors == 2

    vectors = processor._storage.namespace_map[ns]
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
    storage_data = processor._storage.namespace_map[ns]
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
