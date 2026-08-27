"""Unit tests for ClearRAG Grounded, Caveat, Conflict generation builders and metrics."""

import pytest

from src.generation.caveat_generator import CaveatPromptBuilder
from src.generation.conflict_generator import ConflictPromptBuilder
from src.generation.generation_metrics import (
    compute_contains_ground_truth,
    compute_exact_match,
    compute_grounding_metrics,
    compute_token_f1,
)
from src.generation.grounded_generator import GroundedPromptBuilder
from src.generation.attribution import AnswerClaimAttribution


class TestGroundedPromptBuilder:
    """Tests for GroundedPromptBuilder."""

    def test_build_grounded_messages(self):
        builder = GroundedPromptBuilder(require_citations=True)
        chunks = [
            {"document_title": "Doc A", "text": "Doc A factual passage."},
            {"document_title": "Doc B", "text": "Doc B factual passage."},
        ]
        messages = builder.build_messages("What is Doc A?", chunks)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "[1] Document: Doc A" in messages[1]["content"]
        assert "[2] Document: Doc B" in messages[1]["content"]


class TestCaveatPromptBuilder:
    """Tests for CaveatPromptBuilder."""

    def test_build_caveat_messages(self):
        builder = CaveatPromptBuilder()
        chunks = [
            {"document_title": "Doc A", "text": "Supported details for Entity A."},
        ]
        messages = builder.build_messages(
            question="Compare A and B",
            evidence_chunks=chunks,
            supported_claims=["Entity A species count"],
            unsupported_claims=["Entity B species count"],
        )
        assert len(messages) == 2
        assert "SUPPORTED INQUIRY ASPECTS" in messages[1]["content"]
        assert "MISSING / UNVERIFIED ASPECTS" in messages[1]["content"]


class TestConflictPromptBuilder:
    """Tests for ConflictPromptBuilder."""

    def test_build_conflict_messages(self):
        builder = ConflictPromptBuilder()
        chunks = [
            {"document_title": "Source 1", "text": "Entity born in 1904."},
            {"document_title": "Source 2", "text": "Entity born in 1907."},
        ]
        messages = builder.build_messages(
            question="When was entity born?",
            conflicting_chunks=chunks,
            conflict_description="Differing birth years",
        )
        assert len(messages) == 2
        assert "IDENTIFIED CONTRADICTION" in messages[1]["content"]
        assert "[Source 1]" in messages[1]["content"]


class TestGenerationMetrics:
    """Tests for Generation and Grounding Metrics."""

    def test_exact_match_and_f1(self):
        assert compute_exact_match("Thomas Carr", "thomas carr.") == 1.0
        assert compute_token_f1("Walter Hill director", "Walter Hill") > 0.60
        assert compute_contains_ground_truth("Walter Hill directed this", "Walter Hill") == 1.0

    def test_grounding_metrics_calculation(self):
        attributions = [
            AnswerClaimAttribution(
                claim_index=1,
                claim_text="Thomas Carr was born in 1904 [1].",
                cited_chunk_indices=[1],
                supporting_chunk_ids=["c1"],
                is_supported=True,
            ),
            AnswerClaimAttribution(
                claim_index=2,
                claim_text="He won three Academy Awards.",
                cited_chunk_indices=[],
                supporting_chunk_ids=[],
                is_supported=False,
            ),
        ]
        metrics = compute_grounding_metrics(
            attributions=attributions,
            evidence_chunks=[{"chunk_id": "c1", "text": "born in 1904"}],
            prediction_text="Thomas Carr was born in 1904 [1]. He won three Academy Awards.",
        )
        assert metrics["supported_claim_rate"] == 0.50
        assert metrics["unsupported_claim_rate"] == 0.50
        assert metrics["attribution_coverage"] == 0.50
