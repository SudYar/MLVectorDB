import pytest
import numpy as np
from src.mlvectordb.implementations.query_processor import QueryProcessor
from src.mlvectordb.implementations.index import HNSWIndex
from src.mlvectordb.implementations.vector import Vector

@pytest.fixture
def sample_vectors() -> list[Vector]:
    return [
        Vector(id=f"v{i}", values=np.random.rand(3).tolist(), namespace="default", metadata={"category": "A" if i % 2 == 0 else "B"})
        for i in range(110)
    ]

@pytest.fixture
def index(sample_vectors: list[Vector]) -> HNSWIndex:
    dim = len(sample_vectors[0].values)
    idx = HNSWIndex(dim=dim, metric="cosine")
    idx.add(sample_vectors)
    return idx

@pytest.fixture
def processor(index: HNSWIndex) -> QueryProcessor:
    return QueryProcessor(index=index)

def test_insert_and_search(processor: QueryProcessor) -> None:
    vector = Vector(id="new_vec", values=[0.1, 0.2, 0.3], namespace="default", metadata={"type": "test"})
    processor.insert(vector)
    query = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    results = processor.find_similar(query, top_k=3)
    assert len(results) > 0
    ids = [r.id for r in results]
    assert "new_vec" in ids

def test_upsert_many_and_search(processor: QueryProcessor) -> None:
    vectors = [
        Vector(id="a", values=[0.2, 0.4, 0.6]),
        Vector(id="b", values=[0.9, 0.1, 0.5])
    ]
    processor.upsert_many(vectors)
    query = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    results = processor.find_similar(query, top_k=2)
    assert any(r.id == "a" for r in results)

def test_find_similar_with_filter(processor: QueryProcessor) -> None:
    query = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    filter_fn = lambda v: v.metadata.get("category") == "A"
    results = processor.find_similar(query, top_k=5, filter=filter_fn)
    assert all(r.vector.metadata["category"] == "A" for r in results)

def test_delete_removes_vectors(processor: QueryProcessor) -> None:
    vector = Vector(id="to_delete", values=[0.3, 0.3, 0.3])
    processor.insert(vector)
    query = np.array([0.3, 0.3, 0.3], dtype=np.float32)
    assert any(r.id == "to_delete" for r in processor.find_similar(query, top_k=5))
    processor.delete(["to_delete"])
    results_after_delete = processor.find_similar(query, top_k=5)
    assert all(r.id != "to_delete" for r in results_after_delete)

def test_namespace_isolation(processor: QueryProcessor) -> None:
    vec_default = Vector(id="v_default", values=[0.1, 0.1, 0.1], namespace="default")
    vec_other = Vector(id="v_other", values=[0.1, 0.1, 0.1], namespace="other")
    processor.insert(vec_default)
    processor.insert(vec_other)
    query = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    results_default = processor.find_similar(query, top_k=3, namespace="default")
    results_other = processor.find_similar(query, top_k=3, namespace="other")
    ids_default = [r.id for r in results_default]
    ids_other = [r.id for r in results_other]
    assert "v_default" in ids_default
    assert "v_other" in ids_other
    assert not any(i in ids_other for i in ids_default)