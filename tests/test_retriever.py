"""Unit tests for Retriever class."""

import numpy as np
import pytest
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.retriever import Retriever


@pytest.fixture
def mock_retriever():
    embedder = BGEEmbedder()
    dim = embedder.dimension
    index = FAISSIndex(dimension=dim)

    texts = [
        "Doctor Strange is a 2016 superhero film directed by Scott Derrickson.",
        "Sinister is a 2012 horror film starring Ethan Hawke.",
        "Bactris is a genus of spiny palms native to the Americas.",
    ]

    embeddings = embedder.embed_texts(texts)
    meta = [
        {
            "chunk_id": "c_001",
            "document_title": "Doctor Strange (film)",
            "sentence_indices": [0],
            "text": texts[0],
            "is_supporting_fact": True,
            "source_dataset": "HotpotQA",
            "source_question_id": "q1",
            "metadata": {},
        },
        {
            "chunk_id": "c_002",
            "document_title": "Sinister (film)",
            "sentence_indices": [0],
            "text": texts[1],
            "is_supporting_fact": True,
            "source_dataset": "HotpotQA",
            "source_question_id": "q1",
            "metadata": {},
        },
        {
            "chunk_id": "c_003",
            "document_title": "Bactris",
            "sentence_indices": [0],
            "text": texts[2],
            "is_supporting_fact": False,
            "source_dataset": "HotpotQA",
            "source_question_id": "q2",
            "metadata": {},
        },
    ]
    index.add(embeddings, meta)
    return Retriever(embedder=embedder, index=index, default_top_k=2)


def test_retriever_ranking_and_schema(mock_retriever):
    query = "Who directed the movie Doctor Strange?"
    results = mock_retriever.retrieve(query, top_k=2)

    assert len(results) == 2
    top1 = results[0]

    # Check top result is Doctor Strange
    assert top1["document_title"] == "Doctor Strange (film)"
    assert top1["rank"] == 1
    assert top1["score"] > results[1]["score"]

    # Check schema fields
    required_keys = [
        "rank",
        "chunk_id",
        "score",
        "document_title",
        "text",
        "sentence_indices",
        "is_supporting_fact",
        "provenance",
    ]
    for key in required_keys:
        assert key in top1
