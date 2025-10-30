import pytest
import numpy as np
from src.mlvectordb.implementations.index import HNSWIndex
from src.mlvectordb.implementations.vector import Vector

@pytest.fixture
def vectors():
    vectors = []
    for i in range(100):
        v = Vector(id=f"v{i}", values=np.random.rand(4).tolist())
        vectors.append(v)
    for i in range(10):
        v = Vector(id=f"o{i}", values=np.random.rand(4).tolist(), namespace="other")
        vectors.append(v)
    return vectors

@pytest.fixture
def hnsw_index():
    return HNSWIndex(dim=4, metric="cosine")

def test_add_and_count_vectors(hnsw_index, vectors):
    hnsw_index.add(vectors)
    assert "default" in hnsw_index._spaces
    assert "other" in hnsw_index._spaces
    assert len(hnsw_index._vectors["default"]) == 100
    assert len(hnsw_index._vectors["other"]) == 10

def test_search_top_k(hnsw_index, vectors):
    hnsw_index.add(vectors)
    query = np.random.rand(4)
    top_k = 5
    results = hnsw_index.search(query, top_k=top_k)
    assert len(results) == top_k
    for r in results:
        assert isinstance(r.id, str)
        assert isinstance(r.score, float)

def test_remove_vector(hnsw_index, vectors):
    hnsw_index.add(vectors)
    remove_ids = [f"v{i}" for i in range(5)]
    hnsw_index.remove(remove_ids)
    remaining_ids = hnsw_index._vectors["default"].keys()
    for rid in remove_ids:
        assert rid not in [v.id for v in hnsw_index._vectors["default"].values()]

def test_rebuild_index(hnsw_index, vectors):
    default_vectors = [v for v in vectors[50:] if v.namespace == "default"]
    other_vectors = [v for v in vectors[50:] if v.namespace == "other"]

    hnsw_index.add(vectors[:50]) 
    hnsw_index.rebuild(vectors[50:]) 

    default_ids = [v.id for v in hnsw_index._vectors.get("default", {}).values()]
    for v in default_vectors:
        assert v.id in default_ids

    other_ids = [v.id for v in hnsw_index._vectors.get("other", {}).values()]
    for v in other_vectors:
        assert v.id in other_ids


def test_multiple_namespaces_search(hnsw_index, vectors):
    hnsw_index.add(vectors)
    query_default = np.random.rand(4)
    query_other = np.random.rand(4)
    results_default = hnsw_index.search(query_default, top_k=3, namespace="default")
    results_other = hnsw_index.search(query_other, top_k=3, namespace="other")
    assert all(r.id.startswith("v") for r in results_default)
    assert all(r.id.startswith("o") for r in results_other)

def test_metric_change(hnsw_index, vectors):
    hnsw_index.add(vectors[:50])
    query = np.random.rand(4)

    cosine_results = hnsw_index.search(query, top_k=3)
    hnsw_index.rebuild(vectors[:50], metric="l2")
    l2_results = hnsw_index.search(query, top_k=3, metric="l2")

    assert len(cosine_results) == 3
    assert len(l2_results) == 3

    valid_ids = {v.id for v in vectors[:50]}
    for res in cosine_results + l2_results:
        assert res.id in valid_ids
        assert isinstance(res.score, float)

    cosine_scores = [r.score for r in cosine_results]
    l2_scores = [r.score for r in l2_results]
    assert cosine_scores != l2_scores

