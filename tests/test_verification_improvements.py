"""Unit tests for ClearRAG Verification Improvements (Matching, Contradiction, Calibration, Isolation)."""

import pytest

from src.verification.calibration import ThresholdCalibrator
from src.verification.claims import Claim, ClaimType
from src.verification.contradiction import ContradictionDetector
from src.verification.evidence_matching import SemanticEvidenceMatcher
from src.verification.improved_verifier import ImprovedEvidenceVerifier
from src.verification.models import VerificationStatus
from src.verification.sufficiency import SufficiencyEngine


@pytest.fixture
def sample_claim():
    return Claim(
        claim_id="claim_1",
        text="What year was Thomas Carr born?",
        claim_type=ClaimType.ATOMIC_FACT,
        target_entities=["Thomas Carr"],
        predicate="birth_date",
        source_question="What year was Thomas Carr born?",
    )


class TestSemanticEvidenceMatcher:
    """Tests for SemanticEvidenceMatcher."""

    def test_semantic_evidence_matcher_supported(self, sample_claim):
        matcher = SemanticEvidenceMatcher(min_content_overlap_ratio=0.20)
        passage = "Thomas Carr was born in 1904 in Philadelphia, Pennsylvania."
        score, is_supported, expl = matcher.evaluate_passage_support(
            claim=sample_claim,
            passage_text=passage,
            passage_title="Thomas Carr (director)",
        )
        assert is_supported is True
        assert score > 0.0

    def test_semantic_evidence_matcher_unsupported_stopword_leakage_prevented(self):
        # Claim asking for location of a specific entity
        claim = Claim(
            claim_id="claim_loc",
            text="Where is Zhangye located?",
            claim_type=ClaimType.ATOMIC_FACT,
            target_entities=["Zhangye"],
            predicate="location",
            source_question="Where is Zhangye located?",
        )
        matcher = SemanticEvidenceMatcher(min_content_overlap_ratio=0.30)
        # Passage mentions Zhangye but only in an unrelated context with the word 'in'
        passage = "Zhangye was mentioned in passing during a general conversation."
        score, is_supported, expl = matcher.evaluate_passage_support(
            claim=claim,
            passage_text=passage,
            passage_title="Random Discussion",
        )
        # Should NOT be supported merely because 'in' is present
        assert is_supported is False


class TestContradictionDetector:
    """Tests for ContradictionDetector."""

    def test_contradiction_detector_date_conflict(self):
        detector = ContradictionDetector()
        chunks = [
            {"document_title": "Thomas Carr", "text": "Thomas Carr was born on July 4, 1904 in Philadelphia."},
            {"document_title": "Carr Biography", "text": "Thomas Carr was born in 1907 according to early studio records."},
        ]
        has_conflict, c_ids, c_texts, reason = detector.detect_conflict(
            target_entities=["Thomas Carr"],
            predicate="birth_date",
            evidence_chunks=chunks,
        )
        assert has_conflict is True
        assert len(c_ids) >= 2
        assert "1904" in reason or "1907" in reason

    def test_contradiction_detector_numeric_conflict(self):
        detector = ContradictionDetector()
        chunks = [
            {"document_title": "Epigaea", "text": "Epigaea is a genus comprising 3 species of shrubs."},
            {"document_title": "Epigaea Flora", "text": "Epigaea has 5 species recognized by botanists."},
        ]
        has_conflict, c_ids, c_texts, reason = detector.detect_conflict(
            target_entities=["Epigaea"],
            predicate="species_count",
            evidence_chunks=chunks,
        )
        assert has_conflict is True
        assert "3" in reason or "5" in reason

    def test_contradiction_detector_antonym_conflict(self):
        detector = ContradictionDetector()
        chunks = [
            {"document_title": "The Datsuns", "text": "The Datsuns are an active rock band formed in New Zealand."},
            {"document_title": "Rock History", "text": "The Datsuns are a defunct band that disbanded in 2012."},
        ]
        has_conflict, c_ids, c_texts, reason = detector.detect_conflict(
            target_entities=["The Datsuns"],
            predicate="general_fact",
            evidence_chunks=chunks,
        )
        assert has_conflict is True


class TestImprovedEvidenceVerifier:
    """Tests for ImprovedEvidenceVerifier."""

    def test_improved_verifier_multi_entity_claim(self):
        verifier = ImprovedEvidenceVerifier()
        claim_a = Claim(
            claim_id="c1",
            text="Bactris species count",
            claim_type=ClaimType.COMPARISON_ENTITY_A,
            target_entities=["Bactris"],
            predicate="species_count",
        )
        claim_b = Claim(
            claim_id="c2",
            text="Epigaea species count",
            claim_type=ClaimType.COMPARISON_ENTITY_B,
            target_entities=["Epigaea"],
            predicate="species_count",
        )
        chunks = [
            {"chunk_id": "b1", "document_title": "Bactris", "text": "Bactris is a genus of spiny palms with many species."},
            {"chunk_id": "e1", "document_title": "Epigaea", "text": "Epigaea comprises three species of flowering plants."},
        ]

        results = verifier.verify_claims([claim_a, claim_b], chunks)
        assert len(results) == 2
        assert results[0].status == VerificationStatus.SUPPORTED
        assert results[1].status == VerificationStatus.SUPPORTED

        engine = SufficiencyEngine()
        suff_res = engine.evaluate_sufficiency(
            question="Which genus has more species, Bactris and Epigaea?",
            claims=[claim_a, claim_b],
            claim_results=results,
            retrieved_evidence=chunks,
        )
        assert suff_res.overall_status.value == "FULLY_SUPPORTED"


class TestThresholdCalibrator:
    """Tests for ThresholdCalibrator."""

    def test_calibrator_metrics(self):
        predictions = [
            {"score": 0.85, "ground_truth": True},
            {"score": 0.70, "ground_truth": True},
            {"score": 0.40, "ground_truth": False},
            {"score": 0.20, "ground_truth": False},
        ]
        metrics = ThresholdCalibrator.evaluate_threshold(
            predictions=predictions,
            score_extractor=lambda x: x["score"],
            ground_truth_extractor=lambda x: x["ground_truth"],
            threshold=0.60,
            name="test_thresh",
        )
        assert metrics.accuracy == 100.0
        assert metrics.precision == 100.0
        assert metrics.recall == 100.0
        assert metrics.f1_score == 100.0


class TestVerificationMetadataIsolation:
    """Strict verification of zero inference metadata leakage."""

    def test_zero_leakage_interface(self):
        verifier = ImprovedEvidenceVerifier()
        import inspect
        sig = inspect.signature(verifier.verify_claim)
        params = list(sig.parameters.keys())
        assert "claim" in params
        assert "evidence_chunks" in params
        assert "benchmark_condition" not in params
        assert "gold_evidence" not in params
        assert "ground_truth" not in params
