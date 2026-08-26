"""Unit tests for RAGPipeline and RAGResult schema."""

from unittest.mock import MagicMock
import pytest

from src.generation.prompt_builder import PromptBuilder
from src.generation.rag_pipeline import RAGPipeline, RAGResult


def test_rag_pipeline_answer_mocked():
    # 1. Mock Retriever
    mock_retriever = MagicMock()
    mock_retriever.embedder = MagicMock(model_name="BAAI/bge-small-en-v1.5")
    mock_retriever.retrieve.return_value = [
        {
            "rank": 1,
            "chunk_id": "chunk_001",
            "score": 0.85,
            "document_title": "Bactris",
            "sentence_indices": [0],
            "text": "Bactris has 64 species.",
            "is_supporting_fact": True,
        },
        {
            "rank": 2,
            "chunk_id": "chunk_002",
            "score": 0.72,
            "document_title": "Epigaea",
            "sentence_indices": [0],
            "text": "Epigaea has 3 species.",
            "is_supporting_fact": True,
        },
    ]

    # 2. Mock Generator
    mock_generator = MagicMock()
    mock_generator.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    mock_generator.generate_from_messages.return_value = ("Bactris has more species.", 45.2)

    # 3. Assemble Pipeline
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        prompt_builder=PromptBuilder(),
        default_top_k=2,
    )

    result = pipeline.answer("Which genus has more species?", top_k=2)

    # 4. Verify RAGResult schema and values
    assert isinstance(result, RAGResult)
    assert result.question == "Which genus has more species?"
    assert result.answer == "Bactris has more species."
    assert result.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert result.embedding_model == "BAAI/bge-small-en-v1.5"
    assert result.top_k == 2
    assert len(result.retrieved_context) == 2
    assert result.retrieved_context[0]["title"] == "Bactris"
    assert result.retrieved_context[0]["score"] == 0.85
    assert result.latency_generation_ms == 45.2

    # Check serialization
    res_dict = result.to_dict()
    assert "question" in res_dict
    assert "retrieved_context" in res_dict
    assert "latency_total_ms" in res_dict
