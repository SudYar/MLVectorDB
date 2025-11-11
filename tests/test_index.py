import pytest
import numpy as np
from src.mlvectordb.implementations.index import Index
from src.mlvectordb.implementations.vector import Vector
from src.mlvectordb.interfaces.vector import VectorDTO


@pytest.fixture(params=[2, 5, 100])
def vector_count(request):
    return request.param


@pytest.fixture
def sample_vectors(vector_count):
    np.random.seed(42)
    data = np.random.rand(vector_count, 16).astype(np.float32)
    return [Vector(values=v.tolist(), metadata={"i": i}) for i, v in enumerate(data)]


@pytest.fixture
def index():
    return Index(space="l2")


def test_add_and_search_various_sizes(index, sample_vectors):
    namespace = "varied_ns"
    index.add(sample_vectors, namespace)

    query_vec = sample_vectors[0].values + np.random.normal(0, 0.01, size=sample_vectors[0].values.shape)
    query = VectorDTO(values=query_vec, metadata={})

    results = index.search(query, top_k=5, namespace=namespace, metric="l2")

    assert len(results) > 0
    ids = {v.id for v in sample_vectors}
    for r in results:
        assert r.vector_id in ids
        assert isinstance(r.score, float)
        assert r.score >= 0.0


def test_remove_and_search_various_sizes(index, sample_vectors):
    namespace = "remove_ns"
    index.add(sample_vectors, namespace)

    to_remove = [v.id for v in sample_vectors[:2]]
    index.remove(to_remove, namespace)

    query = VectorDTO(values=sample_vectors[0].values, metadata={})
    results = index.search(query, top_k=5, namespace=namespace, metric="l2")

    removed_ids = set(to_remove)
    result_ids = {r.vector_id for r in results}
    assert not (removed_ids & result_ids)


def test_rebuild_many(index, sample_vectors):
    half = len(sample_vectors) // 2 or 1
    source = {
        "ns1": sample_vectors[:half],
        "ns2": sample_vectors[half:],
    }

    index.rebuild(source, metric="l2")

    for ns in source.keys():
        q_vec = source[ns][0].values
        q = VectorDTO(values=q_vec, metadata={})
        results = index.search(q, top_k=3, namespace=ns, metric="l2")
        assert len(results) > 0
        assert results[0].vector_id in [v.id for v in source[ns]]
