"""Unit tests for BGEEmbedder."""

import numpy as np
import pytest
from src.retrieval.embedder import BGEEmbedder


@pytest.fixture(scope="module")
def embedder():
    return BGEEmbedder()


def test_embed_texts_shape_and_norm(embedder):
    texts = [
        "Doctor Strange is a 2016 American superhero film.",
        "Sinister is a 2012 supernatural horror film.",
    ]
    embeddings = embedder.embed_texts(texts)

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32

    # Verify L2 normalization: norm of each vector should be ~1.0
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_query_shape(embedder):
    query = "Who directed Doctor Strange?"
    query_vec = embedder.embed_query(query)

    assert query_vec.shape == (384,)
    assert query_vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(query_vec), 1.0, atol=1e-5)


def test_embed_empty_list(embedder):
    embeddings = embedder.embed_texts([])
    assert embeddings.shape == (0, 384)
