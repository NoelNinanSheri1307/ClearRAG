"""Comprehensive unit tests for ClearRAG Decision + Abstention Layer.

Tests cover:
1.  Fully supported -> ANSWER
2.  Partial evidence -> ANSWER_WITH_CAVEAT
3.  Unsupported -> ABSTAIN
4.  Conflicting -> CONFLICT_ABSTENTION
5.  Unsupported does not invoke generator
6.  Conflicting does not invoke generator
7.  Partial evidence preserves caveat
8.  Supported answer preserves provenance
9.  Comparison requires all required claims
10. Multi-hop missing evidence is not treated as fully supported
11. Deterministic decision behavior
12. Latency fields populated
13. Structured result serialization
14. Generator failures do not corrupt verification result
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.clearrag.decision import (
    ClearRAGDecision,
    ClearRAGDecisionEngine,
    DEFAULT_DECISION_POLICY,
    GENERATION_PERMITTED_DECISIONS,
)
from src.clearrag.result import ClearRAGResult
from src.clearrag.pipeline import ClearRAGPipeline
from src.verification.claims import Claim, ClaimType
from src.verification.models import (
    ClaimVerificationResult,
    SufficiencyStatus,
    VerificationStatus,
)
from src.verification.sufficiency import SufficiencyEngine


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def decision_engine():
    return ClearRAGDecisionEngine()


@pytest.fixture
def mock_retriever():
    """Mock retriever returning configurable evidence."""
    retriever = MagicMock()
    retriever.embedder = MagicMock()
    retriever.embedder.model_name = "BAAI/bge-small-en-v1.5"
    return retriever


@pytest.fixture
def mock_generator():
    """Mock LLM generator returning a fixed answer."""
    gen = MagicMock()
    gen.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    gen.generate_from_messages = MagicMock(return_value=("Bactris has more species.", 50.0))
    return gen


def _make_pipeline(mock_retriever, mock_generator, evidence=None):
    """Helper to create a ClearRAGPipeline with mocked components."""
    if evidence is not None:
        mock_retriever.retrieve = MagicMock(return_value=evidence)
    return ClearRAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        default_top_k=5,
    )


# Evidence fixtures
FULL_EVIDENCE = [
    {"rank": 1, "chunk_id": "c1", "score": 0.85, "document_title": "Bactris",
     "text": "Bactris is a genus of about 75 species of palms."},
    {"rank": 2, "chunk_id": "c2", "score": 0.78, "document_title": "Epigaea",
     "text": "Epigaea is a genus of 3 species of flowering plants."},
]

PARTIAL_EVIDENCE = [
    {"rank": 1, "chunk_id": "c1", "score": 0.85, "document_title": "Bactris",
     "text": "Bactris is a genus of about 75 species of palms."},
]

NO_EVIDENCE = [
    {"rank": 1, "chunk_id": "d1", "score": 0.30, "document_title": "Pizza",
     "text": "Pizza is a popular Italian dish enjoyed worldwide."},
]

CONFLICTING_EVIDENCE = [
    {"rank": 1, "chunk_id": "c1", "score": 0.90, "document_title": "Thomas Carr",
     "text": "Thomas Carr was born in 1907."},
    {"rank": 2, "chunk_id": "c2", "score": 0.85, "document_title": "Thomas Carr",
     "text": "Thomas Carr was born in 1908."},
]


# ═══════════════════════════════════════════════════════════
# Test 1: Fully supported -> ANSWER
# ═══════════════════════════════════════════════════════════

def test_01_fully_supported_produces_answer(mock_retriever, mock_generator):
    """Fully supported evidence should result in ANSWER decision."""
    pipeline = _make_pipeline(mock_retriever, mock_generator, FULL_EVIDENCE)
    result = pipeline.answer("Which genus has more species, Bactris or Epigaea?")

    assert result.decision == ClearRAGDecision.ANSWER
    assert result.sufficiency_status == SufficiencyStatus.FULLY_SUPPORTED
    assert result.answer != ""
    assert not result.is_abstention


# ═══════════════════════════════════════════════════════════
# Test 2: Partial evidence -> ANSWER_WITH_CAVEAT
# ═══════════════════════════════════════════════════════════

def test_02_partial_evidence_produces_caveat(mock_retriever, mock_generator):
    """Partial evidence should result in ANSWER_WITH_CAVEAT decision."""
    pipeline = _make_pipeline(mock_retriever, mock_generator, PARTIAL_EVIDENCE)
    result = pipeline.answer("Which genus has more species, Bactris or Epigaea?")

    assert result.decision == ClearRAGDecision.ANSWER_WITH_CAVEAT
    assert result.sufficiency_status == SufficiencyStatus.PARTIALLY_SUPPORTED
    assert result.is_caveated
    assert not result.is_abstention


# ═══════════════════════════════════════════════════════════
# Test 3: Unsupported -> ABSTAIN
# ═══════════════════════════════════════════════════════════

def test_03_unsupported_produces_abstention(mock_retriever, mock_generator):
    """No supporting evidence should result in ABSTAIN decision."""
    pipeline = _make_pipeline(mock_retriever, mock_generator, NO_EVIDENCE)
    result = pipeline.answer("Which genus has more species, Bactris or Epigaea?")

    assert result.decision == ClearRAGDecision.ABSTAIN
    assert result.sufficiency_status == SufficiencyStatus.UNSUPPORTED
    assert result.is_abstention
    assert result.abstention_reason != ""


# ═══════════════════════════════════════════════════════════
# Test 4: Conflicting -> CONFLICT_ABSTENTION
# ═══════════════════════════════════════════════════════════

def test_04_conflicting_produces_conflict_abstention(mock_retriever, mock_generator):
    """Conflicting evidence should result in CONFLICT_ABSTENTION decision."""
    pipeline = _make_pipeline(mock_retriever, mock_generator, CONFLICTING_EVIDENCE)
    result = pipeline.answer("When was Thomas Carr born?")

    assert result.decision == ClearRAGDecision.CONFLICT_ABSTENTION
    assert result.sufficiency_status == SufficiencyStatus.CONFLICTING
    assert result.is_abstention
    assert result.abstention_reason != ""


# ═══════════════════════════════════════════════════════════
# Test 5: Unsupported does NOT invoke generator
# ═══════════════════════════════════════════════════════════

def test_05_unsupported_skips_generator(mock_retriever, mock_generator):
    """ABSTAIN should NOT call the LLM generator."""
    pipeline = _make_pipeline(mock_retriever, mock_generator, NO_EVIDENCE)
    result = pipeline.answer("Which genus has more species, Bactris or Epigaea?")

    assert result.decision == ClearRAGDecision.ABSTAIN
    mock_generator.generate_from_messages.assert_not_called()
    assert result.generation_latency_ms == 0.0


# ═══════════════════════════════════════════════════════════
# Test 6: Conflicting does NOT invoke generator
# ═══════════════════════════════════════════════════════════

def test_06_conflicting_skips_generator(mock_retriever, mock_generator):
    """CONFLICT_ABSTENTION should NOT call the LLM generator."""
    pipeline = _make_pipeline(mock_retriever, mock_generator, CONFLICTING_EVIDENCE)
    result = pipeline.answer("When was Thomas Carr born?")

    assert result.decision == ClearRAGDecision.CONFLICT_ABSTENTION
    mock_generator.generate_from_messages.assert_not_called()
    assert result.generation_latency_ms == 0.0


# ═══════════════════════════════════════════════════════════
# Test 7: Partial evidence preserves caveat
# ═══════════════════════════════════════════════════════════

def test_07_partial_evidence_includes_caveat_text(mock_retriever, mock_generator):
    """ANSWER_WITH_CAVEAT should include caveat prefix in the answer."""
    pipeline = _make_pipeline(mock_retriever, mock_generator, PARTIAL_EVIDENCE)
    result = pipeline.answer("Which genus has more species, Bactris or Epigaea?")

    assert result.decision == ClearRAGDecision.ANSWER_WITH_CAVEAT
    assert result.caveat_text != ""
    # The answer should contain the caveat prefix
    assert "incomplete evidence" in result.answer.lower() or "not fully verified" in result.answer.lower()


# ═══════════════════════════════════════════════════════════
# Test 8: Supported answer preserves provenance
# ═══════════════════════════════════════════════════════════

def test_08_supported_answer_preserves_provenance(mock_retriever, mock_generator):
    """ANSWER decision should preserve full provenance chain."""
    pipeline = _make_pipeline(mock_retriever, mock_generator, FULL_EVIDENCE)
    result = pipeline.answer("Which genus has more species, Bactris or Epigaea?")

    assert result.decision == ClearRAGDecision.ANSWER
    assert len(result.retrieved_evidence) == 2
    assert len(result.claims) >= 2
    assert len(result.claim_results) >= 2
    assert result.metadata.get("model_name") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert result.metadata.get("embedding_model") == "BAAI/bge-small-en-v1.5"


# ═══════════════════════════════════════════════════════════
# Test 9: Comparison requires all required claims
# ═══════════════════════════════════════════════════════════

def test_09_comparison_requires_both_entities(mock_retriever, mock_generator):
    """Comparison question with only one entity supported should NOT be FULLY_SUPPORTED."""
    pipeline = _make_pipeline(mock_retriever, mock_generator, PARTIAL_EVIDENCE)
    result = pipeline.answer("Which genus has more species, Bactris or Epigaea?")

    # Only Bactris evidence -> should not be FULLY_SUPPORTED
    assert result.sufficiency_status != SufficiencyStatus.FULLY_SUPPORTED
    assert result.decision != ClearRAGDecision.ANSWER


# ═══════════════════════════════════════════════════════════
# Test 10: Multi-hop missing evidence != fully supported
# ═══════════════════════════════════════════════════════════

def test_10_multihop_missing_link_not_fully_supported(mock_retriever, mock_generator):
    """Multi-hop / complex question with incomplete evidence should not be FULLY_SUPPORTED.

    Uses a comparison question (which the extractor decomposes into 2 claims)
    where only one entity has relevant evidence. This tests that the pipeline
    correctly identifies incomplete evidence chains.
    """
    evidence = [
        {"rank": 1, "chunk_id": "c1", "score": 0.80, "document_title": "Thomas Carr",
         "text": "Thomas Carr was born in 1907 in London."},
    ]
    pipeline = _make_pipeline(mock_retriever, mock_generator, evidence)
    # Comparison question: extractor creates 2 claims. Only Thomas Carr evidence exists.
    result = pipeline.answer("Who was born earlier, Thomas Carr or John Smith?")

    assert result.sufficiency_status != SufficiencyStatus.FULLY_SUPPORTED
    assert result.decision != ClearRAGDecision.ANSWER


# ═══════════════════════════════════════════════════════════
# Test 11: Deterministic decision behavior
# ═══════════════════════════════════════════════════════════

def test_11_deterministic_decision_engine(decision_engine):
    """Decision engine must produce deterministic, repeatable results."""
    for _ in range(10):
        assert decision_engine.decide(SufficiencyStatus.FULLY_SUPPORTED) == ClearRAGDecision.ANSWER
        assert decision_engine.decide(SufficiencyStatus.PARTIALLY_SUPPORTED) == ClearRAGDecision.ANSWER_WITH_CAVEAT
        assert decision_engine.decide(SufficiencyStatus.UNSUPPORTED) == ClearRAGDecision.ABSTAIN
        assert decision_engine.decide(SufficiencyStatus.CONFLICTING) == ClearRAGDecision.CONFLICT_ABSTENTION


# ═══════════════════════════════════════════════════════════
# Test 12: Latency fields populated
# ═══════════════════════════════════════════════════════════

def test_12_latency_fields_populated(mock_retriever, mock_generator):
    """ClearRAGResult must have populated latency fields."""
    pipeline = _make_pipeline(mock_retriever, mock_generator, FULL_EVIDENCE)
    result = pipeline.answer("Which genus has more species, Bactris or Epigaea?")

    assert result.retrieval_latency_ms >= 0.0
    assert result.verification_latency_ms >= 0.0
    assert result.total_latency_ms > 0.0
    # For ANSWER decision, generation latency should be > 0
    if result.decision == ClearRAGDecision.ANSWER:
        assert result.generation_latency_ms > 0.0


# ═══════════════════════════════════════════════════════════
# Test 13: Structured result serialization
# ═══════════════════════════════════════════════════════════

def test_13_structured_result_serialization(mock_retriever, mock_generator):
    """ClearRAGResult.to_dict() must produce valid JSON-serializable output."""
    pipeline = _make_pipeline(mock_retriever, mock_generator, FULL_EVIDENCE)
    result = pipeline.answer("Which genus has more species, Bactris or Epigaea?")

    result_dict = result.to_dict()
    # Must be JSON serializable
    serialized = json.dumps(result_dict)
    assert isinstance(serialized, str)

    # Check required keys exist
    required_keys = [
        "question", "answer", "decision", "sufficiency_status",
        "claims", "claim_results", "explanation",
        "retrieval_latency_ms", "verification_latency_ms",
        "generation_latency_ms", "total_latency_ms", "metadata",
    ]
    for key in required_keys:
        assert key in result_dict, f"Missing key: {key}"

    # Check full dict includes evidence
    full_dict = result.to_full_dict()
    assert "retrieved_evidence" in full_dict
    assert "supporting_evidence" in full_dict
    assert "conflicting_evidence" in full_dict


# ═══════════════════════════════════════════════════════════
# Test 14: Generator failures do not corrupt verification
# ═══════════════════════════════════════════════════════════

def test_14_generator_failure_preserves_verification(mock_retriever, mock_generator):
    """If the LLM generator throws an exception, verification result should be preserved."""
    mock_generator.generate_from_messages.side_effect = RuntimeError("GPU OOM")
    pipeline = _make_pipeline(mock_retriever, mock_generator, FULL_EVIDENCE)
    result = pipeline.answer("Which genus has more species, Bactris or Epigaea?")

    # Verification should still complete correctly
    assert result.sufficiency_status == SufficiencyStatus.FULLY_SUPPORTED
    assert result.decision == ClearRAGDecision.ANSWER
    # Answer should indicate error, not empty
    assert "error" in result.answer.lower() or "generation" in result.answer.lower()
    # Claims and verification results must be preserved
    assert len(result.claims) >= 2
    assert len(result.claim_results) >= 2


# ═══════════════════════════════════════════════════════════
# Additional Decision Engine unit tests
# ═══════════════════════════════════════════════════════════

def test_custom_policy():
    """Custom decision policy should override defaults."""
    custom_policy = {
        SufficiencyStatus.FULLY_SUPPORTED.value: ClearRAGDecision.ANSWER.value,
        SufficiencyStatus.PARTIALLY_SUPPORTED.value: ClearRAGDecision.ABSTAIN.value,  # Override!
        SufficiencyStatus.UNSUPPORTED.value: ClearRAGDecision.ABSTAIN.value,
        SufficiencyStatus.CONFLICTING.value: ClearRAGDecision.CONFLICT_ABSTENTION.value,
    }
    engine = ClearRAGDecisionEngine(policy=custom_policy)

    assert engine.decide(SufficiencyStatus.PARTIALLY_SUPPORTED) == ClearRAGDecision.ABSTAIN


def test_permits_generation():
    """Only ANSWER and ANSWER_WITH_CAVEAT permit generation."""
    engine = ClearRAGDecisionEngine()
    assert engine.permits_generation(ClearRAGDecision.ANSWER) is True
    assert engine.permits_generation(ClearRAGDecision.ANSWER_WITH_CAVEAT) is True
    assert engine.permits_generation(ClearRAGDecision.ABSTAIN) is False
    assert engine.permits_generation(ClearRAGDecision.CONFLICT_ABSTENTION) is False


def test_result_properties():
    """Test ClearRAGResult property helpers."""
    abstain_result = ClearRAGResult(
        question="Q", answer="A", decision=ClearRAGDecision.ABSTAIN,
        sufficiency_status=SufficiencyStatus.UNSUPPORTED,
    )
    assert abstain_result.is_abstention is True
    assert abstain_result.is_caveated is False

    caveat_result = ClearRAGResult(
        question="Q", answer="A", decision=ClearRAGDecision.ANSWER_WITH_CAVEAT,
        sufficiency_status=SufficiencyStatus.PARTIALLY_SUPPORTED,
    )
    assert caveat_result.is_abstention is False
    assert caveat_result.is_caveated is True

    answer_result = ClearRAGResult(
        question="Q", answer="A", decision=ClearRAGDecision.ANSWER,
        sufficiency_status=SufficiencyStatus.FULLY_SUPPORTED,
    )
    assert answer_result.is_abstention is False
    assert answer_result.is_caveated is False
