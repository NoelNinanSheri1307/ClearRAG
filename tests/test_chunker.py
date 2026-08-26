"""Unit tests for chunk data structures and sentence chunking."""

import pytest
from src.ingestion.chunker import Chunk, chunk_document_by_sentence


def test_chunk_serialization():
    chunk = Chunk(
        chunk_id="test_001",
        source_dataset="HotpotQA",
        source_question_id="q_123",
        document_title="Scott Derrickson",
        sentence_indices=[0],
        text="Scott Derrickson is an American director.",
        is_supporting_fact=True,
        metadata={"category": "test"},
    )
    chunk_dict = chunk.to_dict()
    assert chunk_dict["chunk_id"] == "test_001"
    assert chunk_dict["document_title"] == "Scott Derrickson"
    assert chunk_dict["is_supporting_fact"] is True

    restored = Chunk.from_dict(chunk_dict)
    assert restored.chunk_id == chunk.chunk_id
    assert restored.text == chunk.text
    assert restored.sentence_indices == [0]


def test_chunk_document_by_sentence():
    sentences = [
        "Scott Derrickson (born July 16, 1966) is an American director.",
        "He lives in Los Angeles, California.",
        "He directed Doctor Strange.",
    ]
    supporting_facts = {("Scott Derrickson", 0), ("Scott Derrickson", 2)}

    chunks = chunk_document_by_sentence(
        source_dataset="HotpotQA",
        source_question_id="q_999",
        doc_index=0,
        document_title="Scott Derrickson",
        sentences=sentences,
        supporting_facts_set=supporting_facts,
    )

    assert len(chunks) == 3
    assert chunks[0].sentence_indices == [0]
    assert chunks[0].is_supporting_fact is True
    assert chunks[1].sentence_indices == [1]
    assert chunks[1].is_supporting_fact is False
    assert chunks[2].sentence_indices == [2]
    assert chunks[2].is_supporting_fact is True
    assert chunks[0].document_title == "Scott Derrickson"
