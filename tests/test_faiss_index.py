"""Unit tests for FAISSIndex."""

import numpy as np
import pytest
from src.retrieval.faiss_index import FAISSIndex


def test_faiss_index_add_and_search(tmp_path):
    dim = 4
    index = FAISSIndex(dimension=dim)

    # 3 dummy vectors
    vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    metadata = [
        {"chunk_id": "c1", "document_title": "Doc1"},
        {"chunk_id": "c2", "document_title": "Doc2"},
        {"chunk_id": "c3", "document_title": "Doc3"},
    ]

    index.add(vecs, metadata)
    assert index.ntotal == 3

    # Query closest to vec1
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    scores, retrieved_meta = index.search(query, top_k=2)

    assert len(scores) == 2
    assert len(retrieved_meta) == 2
    assert retrieved_meta[0]["chunk_id"] == "c1"

    # Test save and load
    idx_file = tmp_path / "test.bin"
    meta_file = tmp_path / "test_meta.json"
    index.save(idx_file, meta_file)

    loaded_index = FAISSIndex.load(idx_file, meta_file)
    assert loaded_index.ntotal == 3

    scores2, retrieved_meta2 = loaded_index.search(query, top_k=2)
    assert retrieved_meta2[0]["chunk_id"] == "c1"
    assert np.allclose(scores, scores2)
