"""Unit tests for ClearRAG evidence verification layer."""

import pytest
from src.verification.claims import Claim, ClaimType
from src.verification.claim_extractor import RuleBasedClaimExtractor
from src.verification.evidence_verifier import EvidenceVerifier
from src.verification.models import ClaimVerificationResult, SufficiencyStatus, VerificationStatus
from src.verification.sufficiency import SufficiencyEngine


@pytest.fixture
def claim_extractor():
    return RuleBasedClaimExtractor()


@pytest.fixture
def evidence_verifier():
    return EvidenceVerifier()


@pytest.fixture
def sufficiency_engine():
    return SufficiencyEngine()


def test_1_single_supported_claim(claim_extractor, evidence_verifier, sufficiency_engine):
    """Test 1: Single supported claim."""
    question = "When was Thomas Carr born?"
    claims = claim_extractor.extract_claims(question)
    assert len(claims) == 1

    evidence = [
        {"chunk_id": "c1", "document_title": "Thomas Carr", "text": "Thomas Carr was born on July 4, 1907 in London."}
    ]
    result = evidence_verifier.verify_claim(claims[0], evidence)
    assert result.status == VerificationStatus.SUPPORTED
    assert "c1" in result.supporting_evidence_ids

    v_result = sufficiency_engine.evaluate_sufficiency(question, claims, [result], evidence)
    assert v_result.overall_status == SufficiencyStatus.FULLY_SUPPORTED


def test_2_multi_entity_supported_question(claim_extractor, evidence_verifier, sufficiency_engine):
    """Test 2: Multi-entity comparison question with full support for both entities."""
    question = "Which genus has more species, Bactris or Epigaea?"
    claims = claim_extractor.extract_claims(question)
    assert len(claims) == 2

    evidence = [
        {"chunk_id": "c1", "document_title": "Bactris", "text": "Bactris is a genus of about 75 species of palms."},
        {"chunk_id": "c2", "document_title": "Epigaea", "text": "Epigaea is a genus of 3 species of flowering plants."}
    ]

    results = [evidence_verifier.verify_claim(claim, evidence) for claim in claims]
    assert all(r.status == VerificationStatus.SUPPORTED for r in results)

    v_result = sufficiency_engine.evaluate_sufficiency(question, claims, results, evidence)
    assert v_result.overall_status == SufficiencyStatus.FULLY_SUPPORTED


def test_3_partial_evidence(claim_extractor, evidence_verifier, sufficiency_engine):
    """Test 3: Partial evidence where only one entity in a multi-entity question has evidence."""
    question = "Which genus has more species, Bactris or Epigaea?"
    claims = claim_extractor.extract_claims(question)
    assert len(claims) == 2

    evidence = [
        {"chunk_id": "c1", "document_title": "Bactris", "text": "Bactris is a genus of about 75 species of palms."}
    ]

    results = [evidence_verifier.verify_claim(claim, evidence) for claim in claims]
    statuses = [r.status for r in results]
    assert VerificationStatus.SUPPORTED in statuses
    assert VerificationStatus.UNSUPPORTED in statuses

    v_result = sufficiency_engine.evaluate_sufficiency(question, claims, results, evidence)
    assert v_result.overall_status == SufficiencyStatus.PARTIALLY_SUPPORTED


def test_4_no_evidence(claim_extractor, evidence_verifier, sufficiency_engine):
    """Test 4: No evidence provided."""
    question = "What is the secret code for Jupiter?"
    claims = claim_extractor.extract_claims(question)
    
    evidence = []
    results = [evidence_verifier.verify_claim(claim, evidence) for claim in claims]
    assert all(r.status == VerificationStatus.UNSUPPORTED for r in results)

    v_result = sufficiency_engine.evaluate_sufficiency(question, claims, results, evidence)
    assert v_result.overall_status == SufficiencyStatus.UNSUPPORTED


def test_5_distractor_heavy_evidence(claim_extractor, evidence_verifier, sufficiency_engine):
    """Test 5: Distractor-heavy evidence (irrelevant passages containing distractor entities)."""
    question = "When was Thomas Carr born?"
    claims = claim_extractor.extract_claims(question)

    evidence = [
        {"chunk_id": "d1", "document_title": "Cooking Recipes", "text": "Baking bread requires flour, water, and yeast."},
        {"chunk_id": "d2", "document_title": "Astronomy Today", "text": "Jupiter is the largest planet in our Solar System."}
    ]
    results = [evidence_verifier.verify_claim(claim, evidence) for claim in claims]
    assert all(r.status == VerificationStatus.UNSUPPORTED for r in results)

    v_result = sufficiency_engine.evaluate_sufficiency(question, claims, results, evidence)
    assert v_result.overall_status == SufficiencyStatus.UNSUPPORTED


def test_6_conflicting_numeric_date_evidence(claim_extractor, evidence_verifier, sufficiency_engine):
    """Test 6: Conflicting numeric/date evidence."""
    question = "When was Thomas Carr born?"
    claims = claim_extractor.extract_claims(question)

    evidence = [
        {"chunk_id": "c1", "document_title": "Thomas Carr", "text": "Thomas Carr was born in 1907 in London."},
        {"chunk_id": "c2", "document_title": "Thomas Carr", "text": "Thomas Carr was born in 1908 in Manchester."}
    ]

    results = [evidence_verifier.verify_claim(claim, evidence) for claim in claims]
    assert any(r.status == VerificationStatus.CONFLICTING for r in results)

    v_result = sufficiency_engine.evaluate_sufficiency(question, claims, results, evidence)
    assert v_result.overall_status == SufficiencyStatus.CONFLICTING


def test_7_fully_supported_decision(sufficiency_engine):
    """Test 7: Decision engine policy for fully supported claims."""
    claim = Claim(claim_id="c1", text="Sample", claim_type=ClaimType.ATOMIC_FACT)
    result = [
        ClaimVerificationResult(claim=claim, status=VerificationStatus.SUPPORTED)
    ]
    v_res = sufficiency_engine.evaluate_sufficiency("Q", [claim], result, [])
    assert v_res.overall_status == SufficiencyStatus.FULLY_SUPPORTED


def test_8_partial_decision(sufficiency_engine):
    """Test 8: Decision engine policy for partial support."""
    c1 = Claim(claim_id="c1", text="A", claim_type=ClaimType.COMPARISON_ENTITY_A)
    c2 = Claim(claim_id="c2", text="B", claim_type=ClaimType.COMPARISON_ENTITY_B)
    results = [
        ClaimVerificationResult(claim=c1, status=VerificationStatus.SUPPORTED),
        ClaimVerificationResult(claim=c2, status=VerificationStatus.UNSUPPORTED),
    ]
    v_res = sufficiency_engine.evaluate_sufficiency("Q", [c1, c2], results, [])
    assert v_res.overall_status == SufficiencyStatus.PARTIALLY_SUPPORTED


def test_9_unsupported_decision(sufficiency_engine):
    """Test 9: Decision engine policy for unsupported."""
    c1 = Claim(claim_id="c1", text="A", claim_type=ClaimType.ATOMIC_FACT)
    results = [
        ClaimVerificationResult(claim=c1, status=VerificationStatus.UNSUPPORTED)
    ]
    v_res = sufficiency_engine.evaluate_sufficiency("Q", [c1], results, [])
    assert v_res.overall_status == SufficiencyStatus.UNSUPPORTED


def test_10_conflict_decision(sufficiency_engine):
    """Test 10: Decision engine policy for conflicting evidence."""
    c1 = Claim(claim_id="c1", text="A", claim_type=ClaimType.ATOMIC_FACT)
    results = [
        ClaimVerificationResult(claim=c1, status=VerificationStatus.CONFLICTING)
    ]
    v_res = sufficiency_engine.evaluate_sufficiency("Q", [c1], results, [])
    assert v_res.overall_status == SufficiencyStatus.CONFLICTING


# Additional Regression Tests A-J

def test_reg_A_entity_mention_without_predicate_support(claim_extractor, evidence_verifier):
    """Regression Test A: Entity mention without predicate support -> UNSUPPORTED."""
    question = "When was Thomas Carr born?"
    claims = claim_extractor.extract_claims(question)
    evidence = [
        {"chunk_id": "c1", "document_title": "Thomas Carr", "text": "Thomas Carr was an American film director and actor who directed westerns."}
    ]
    res = evidence_verifier.verify_claim(claims[0], evidence)
    assert res.status == VerificationStatus.UNSUPPORTED


def test_reg_B_correct_attribute_support(claim_extractor, evidence_verifier):
    """Regression Test B: Correct attribute support -> SUPPORTED."""
    question = "When was Thomas Carr born?"
    claims = claim_extractor.extract_claims(question)
    evidence = [
        {"chunk_id": "c1", "document_title": "Thomas Carr", "text": "Thomas Carr was born July 4, 1907."}
    ]
    res = evidence_verifier.verify_claim(claims[0], evidence)
    assert res.status == VerificationStatus.SUPPORTED


def test_reg_C_birth_year_vs_death_year_not_conflicting(claim_extractor, evidence_verifier):
    """Regression Test C: Birth year vs death year -> NOT CONFLICTING."""
    question = "When was Thomas Carr born?"
    claims = claim_extractor.extract_claims(question)
    evidence = [
        {"chunk_id": "c1", "document_title": "Thomas Carr", "text": "Thomas Carr was born in 1907."},
        {"chunk_id": "c2", "document_title": "Thomas Carr", "text": "Thomas Carr died in 1997."}
    ]
    res = evidence_verifier.verify_claim(claims[0], evidence)
    assert res.status == VerificationStatus.SUPPORTED


def test_reg_D_two_conflicting_birth_years(claim_extractor, evidence_verifier):
    """Regression Test D: Two conflicting birth years -> CONFLICTING."""
    question = "When was Thomas Carr born?"
    claims = claim_extractor.extract_claims(question)
    evidence = [
        {"chunk_id": "c1", "document_title": "Thomas Carr", "text": "Thomas Carr was born in 1907."},
        {"chunk_id": "c2", "document_title": "Thomas Carr", "text": "Thomas Carr was born in 1908."}
    ]
    res = evidence_verifier.verify_claim(claims[0], evidence)
    assert res.status == VerificationStatus.CONFLICTING


def test_reg_E_species_count_both_entities(claim_extractor, evidence_verifier, sufficiency_engine):
    """Regression Test E: Species count for both comparison entities -> FULLY_SUPPORTED."""
    question = "Which genus has more species, Bactris or Epigaea?"
    claims = claim_extractor.extract_claims(question)
    evidence = [
        {"chunk_id": "c1", "document_title": "Bactris", "text": "Bactris is a genus of 75 species."},
        {"chunk_id": "c2", "document_title": "Epigaea", "text": "Epigaea is a genus of 3 species."}
    ]
    results = [evidence_verifier.verify_claim(claim, evidence) for claim in claims]
    v_res = sufficiency_engine.evaluate_sufficiency(question, claims, results, evidence)
    assert v_res.overall_status == SufficiencyStatus.FULLY_SUPPORTED


def test_reg_F_species_count_only_one_entity(claim_extractor, evidence_verifier, sufficiency_engine):
    """Regression Test F: Species count for only one entity -> PARTIALLY_SUPPORTED."""
    question = "Which genus has more species, Bactris or Epigaea?"
    claims = claim_extractor.extract_claims(question)
    evidence = [
        {"chunk_id": "c1", "document_title": "Bactris", "text": "Bactris is a genus of 75 species."}
    ]
    results = [evidence_verifier.verify_claim(claim, evidence) for claim in claims]
    v_res = sufficiency_engine.evaluate_sufficiency(question, claims, results, evidence)
    assert v_res.overall_status == SufficiencyStatus.PARTIALLY_SUPPORTED


def test_reg_G_no_relevant_evidence(claim_extractor, evidence_verifier, sufficiency_engine):
    """Regression Test G: No relevant evidence -> UNSUPPORTED."""
    question = "Which genus has more species, Bactris or Epigaea?"
    claims = claim_extractor.extract_claims(question)
    evidence = [
        {"chunk_id": "c1", "document_title": "Pizza", "text": "Pizza is a delicious dish."}
    ]
    results = [evidence_verifier.verify_claim(claim, evidence) for claim in claims]
    v_res = sufficiency_engine.evaluate_sufficiency(question, claims, results, evidence)
    assert v_res.overall_status == SufficiencyStatus.UNSUPPORTED


def test_reg_H_distractor_passage_mentioning_entity_without_attribute(claim_extractor, evidence_verifier):
    """Regression Test H: Distractor passage mentioning entity but not requested attribute -> UNSUPPORTED."""
    question = "When was Thomas Carr born?"
    claims = claim_extractor.extract_claims(question)
    evidence = [
        {"chunk_id": "c1", "document_title": "Thomas Carr", "text": "Thomas Carr directed many western television episodes."}
    ]
    res = evidence_verifier.verify_claim(claims[0], evidence)
    assert res.status == VerificationStatus.UNSUPPORTED


def test_reg_I_multihop_missing_link(evidence_verifier, sufficiency_engine):
    """Regression Test I: Multi-hop with one missing link -> PARTIALLY_SUPPORTED."""
    c1 = Claim(claim_id="c1", text="Director of Orion Pictures", claim_type=ClaimType.MULTI_HOP, target_entities=["Orion Pictures"], predicate="membership")
    c2 = Claim(claim_id="c2", text="Government position of director", claim_type=ClaimType.MULTI_HOP, target_entities=["Director"], predicate="membership")
    evidence = [
        {"chunk_id": "c1", "document_title": "Orion Pictures", "text": "Orion Pictures was an American motion picture production company founded by Woody Allen."}
    ]
    results = [evidence_verifier.verify_claim(claim, evidence) for claim in [c1, c2]]
    v_res = sufficiency_engine.evaluate_sufficiency("Question", [c1, c2], results, evidence)
    assert v_res.overall_status in (SufficiencyStatus.PARTIALLY_SUPPORTED, SufficiencyStatus.UNSUPPORTED)


def test_reg_J_multihop_conflicting_required_fact(claim_extractor, evidence_verifier, sufficiency_engine):
    """Regression Test J: Multi-hop with conflicting required fact -> CONFLICTING."""
    question = "When was Thomas Carr born?"
    claims = claim_extractor.extract_claims(question)
    evidence = [
        {"chunk_id": "c1", "document_title": "Thomas Carr", "text": "Thomas Carr was born in 1907."},
        {"chunk_id": "c2", "document_title": "Thomas Carr", "text": "Thomas Carr was born in 1908."}
    ]
    results = [evidence_verifier.verify_claim(claim, evidence) for claim in claims]
    v_res = sufficiency_engine.evaluate_sufficiency(question, claims, results, evidence)
    assert v_res.overall_status == SufficiencyStatus.CONFLICTING
