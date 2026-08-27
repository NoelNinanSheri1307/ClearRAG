"""Unit tests for Standard RAG and Verification Baselines.

Verifies:
1. Standard RAG executes unconditionally and returns structured StandardRAGResult.
2. Verification baseline classifies evidence sufficiency without generating answers.
3. Strict inference isolation: zero metadata leakage (pipelines accept only question).
"""

from unittest.mock import MagicMock
import pytest

from src.baselines.standard_rag import StandardRAGPipeline, StandardRAGResult
from src.baselines.verification_baseline import (
    VerificationBaselinePipeline,
    VerificationBaselineResult,
)
from src.verification.claims import Claim, ClaimType
from src.verification.models import (
    ClaimVerificationResult,
    SufficiencyStatus,
    VerificationResult,
    VerificationStatus,
)


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {
            "rank": 1,
            "chunk_id": "doc1_s0",
            "score": 0.88,
            "document_title": "Bactris",
            "sentence_indices": [0],
            "text": "Bactris is a genus of spiny palms.",
            "is_supporting_fact": True,
        }
    ]
    retriever.embedder.model_name = "BAAI/bge-small-en-v1.5"
    return retriever


@pytest.fixture
def mock_generator():
    generator = MagicMock()
    generator.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    generator.generate_from_messages.return_value = ("Bactris is a genus of spiny palms.", 45.0)
    return generator


@pytest.fixture
def mock_claim_extractor():
    extractor = MagicMock()
    extractor.extract_claims.return_value = [
        Claim(
            claim_id="c1",
            text="Bactris is a genus of spiny palms.",
            claim_type=ClaimType.ATOMIC_FACT,
            target_entities=["Bactris"],
            predicate="is_genus",
            source_question="Is Bactris a genus of spiny palms?",
        )
    ]
    return extractor


@pytest.fixture
def mock_evidence_verifier():
    verifier = MagicMock()
    verifier.verify_claims.return_value = [
        ClaimVerificationResult(
            claim=Claim(
                claim_id="c1",
                text="Bactris is a genus of spiny palms.",
                claim_type=ClaimType.ATOMIC_FACT,
                target_entities=["Bactris"],
                predicate="is_genus",
                source_question="Is Bactris a genus of spiny palms?",
            ),
            status=VerificationStatus.SUPPORTED,
            supporting_evidence_ids=["doc1_s0"],
            conflicting_evidence_ids=[],
            confidence_score=0.9,
            reason="Exact predicate match found.",
        )
    ]
    return verifier


@pytest.fixture
def mock_sufficiency_engine():
    engine = MagicMock()
    engine.evaluate_sufficiency.return_value = VerificationResult(
        question="Is Bactris a genus of spiny palms?",
        claims=[],
        retrieved_evidence=[],
        claim_results=[],
        overall_status=SufficiencyStatus.FULLY_SUPPORTED,
        explanation="All claims supported.",
    )
    return engine


class TestStandardRAGBaseline:
    """Tests for Standard RAG baseline pipeline."""

    def test_standard_rag_execution(self, mock_retriever, mock_generator):
        pipeline = StandardRAGPipeline(
            retriever=mock_retriever,
            generator=mock_generator,
            default_top_k=3,
        )

        result = pipeline.run("What is Bactris?")

        assert isinstance(result, StandardRAGResult)
        assert result.question == "What is Bactris?"
        assert result.answer == "Bactris is a genus of spiny palms."
        assert result.llm_called is True
        assert len(result.retrieved_evidence) == 1
        assert result.retrieved_chunk_ids == ["doc1_s0"]
        assert result.latency_retrieval_ms >= 0.0
        assert result.latency_generation_ms == 45.0
        assert result.latency_total_ms >= 0.0

        mock_generator.generate_from_messages.assert_called_once()

    def test_standard_rag_serialization(self, mock_retriever, mock_generator):
        pipeline = StandardRAGPipeline(
            retriever=mock_retriever,
            generator=mock_generator,
        )
        result = pipeline.run("What is Bactris?")
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["question"] == "What is Bactris?"
        assert "retrieved_chunk_ids" in d
        assert "latency_total_ms" in d

    def test_standard_rag_metadata_isolation(self, mock_retriever, mock_generator):
        pipeline = StandardRAGPipeline(
            retriever=mock_retriever,
            generator=mock_generator,
        )
        result = pipeline.run(question="What is Bactris?", top_k=2)
        assert result is not None
        mock_retriever.retrieve.assert_called_with("What is Bactris?", top_k=2)


class TestVerificationBaseline:
    """Tests for Evidence Verification baseline pipeline."""

    def test_verification_baseline_execution(
        self,
        mock_retriever,
        mock_claim_extractor,
        mock_evidence_verifier,
        mock_sufficiency_engine,
    ):
        pipeline = VerificationBaselinePipeline(
            retriever=mock_retriever,
            claim_extractor=mock_claim_extractor,
            evidence_verifier=mock_evidence_verifier,
            sufficiency_engine=mock_sufficiency_engine,
            default_top_k=3,
        )

        result = pipeline.run("Is Bactris a genus of spiny palms?")

        assert isinstance(result, VerificationBaselineResult)
        assert result.question == "Is Bactris a genus of spiny palms?"
        assert result.sufficiency_status == "FULLY_SUPPORTED"
        assert len(result.claims) == 1
        assert len(result.claim_verification_results) == 1
        assert result.provenance["num_supporting_evidence"] == 1
        assert result.provenance["num_conflicting_evidence"] == 0

    def test_verification_baseline_serialization(
        self,
        mock_retriever,
        mock_claim_extractor,
        mock_evidence_verifier,
        mock_sufficiency_engine,
    ):
        pipeline = VerificationBaselinePipeline(
            retriever=mock_retriever,
            claim_extractor=mock_claim_extractor,
            evidence_verifier=mock_evidence_verifier,
            sufficiency_engine=mock_sufficiency_engine,
        )

        result = pipeline.run("Is Bactris a genus?")
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["sufficiency_status"] == "FULLY_SUPPORTED"
        assert "provenance" in d
        assert "latency_verification_ms" in d
