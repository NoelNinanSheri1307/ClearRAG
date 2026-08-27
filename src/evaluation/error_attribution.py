"""Error Taxonomy and Attribution Engine for ClearRAG Controlled Evaluation.

Categorizes system failures into a formal error taxonomy:
1. RETRIEVAL_FAILURE: Gold facts not retrieved in top-k.
2. CLAIM_EXTRACTION_FAILURE: Claims missing core entity or relation.
3. VERIFICATION_FALSE_POSITIVE: Unsupported claim incorrectly classified as supported.
4. VERIFICATION_FALSE_NEGATIVE: Supported claim incorrectly classified as unsupported.
5. SUFFICIENCY_AGGREGATION_ERROR: Sufficiency engine incorrectly aggregates claim results.
6. DECISION_POLICY_ERROR: Decision engine chooses incorrect action for sufficiency status.
7. GENERATION_ERROR: LLM generates incorrect answer despite sufficient evidence.
8. EVALUATION_AMBIGUITY: Ambiguity in benchmark annotations or evaluation metrics.
9. CORRECT_EXECUTION: System executed accurately according to benchmark expectations.
"""

from enum import Enum
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Formal taxonomy categories for ClearRAG error attribution."""

    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    CLAIM_EXTRACTION_FAILURE = "CLAIM_EXTRACTION_FAILURE"
    VERIFICATION_FALSE_POSITIVE = "VERIFICATION_FALSE_POSITIVE"
    VERIFICATION_FALSE_NEGATIVE = "VERIFICATION_FALSE_NEGATIVE"
    SUFFICIENCY_AGGREGATION_ERROR = "SUFFICIENCY_AGGREGATION_ERROR"
    DECISION_POLICY_ERROR = "DECISION_POLICY_ERROR"
    GENERATION_ERROR = "GENERATION_ERROR"
    EVALUATION_AMBIGUITY = "EVALUATION_AMBIGUITY"
    CORRECT_EXECUTION = "CORRECT_EXECUTION"


def check_gold_evidence_retrieved(
    benchmark_item: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
) -> bool:
    """Determine whether gold supporting facts are present in retrieved chunks.

    Args:
        benchmark_item: Benchmark record containing retained_supporting_facts or supporting_facts.
        retrieved_chunks: List of retrieved chunk dictionaries.

    Returns:
        True if gold supporting facts are retrieved in the top-k, False otherwise.
    """
    facts = (
        benchmark_item.get("retained_supporting_facts", [])
        or benchmark_item.get("supporting_facts", [])
        or benchmark_item.get("gold_chunks", [])
    )

    if not facts:
        condition = benchmark_item.get("condition", "")
        return condition == "unsupported"

    gold_titles = set()
    for f in facts:
        if isinstance(f, dict):
            t = f.get("title", "")
        elif isinstance(f, (list, tuple)) and len(f) > 0:
            t = f[0]
        else:
            t = str(f)
        if t:
            gold_titles.add(t.strip().lower())

    if not gold_titles:
        return True

    retrieved_titles = set()
    for c in retrieved_chunks:
        t = c.get("title", c.get("document_title", ""))
        if t:
            retrieved_titles.add(t.strip().lower())

    overlap = gold_titles.intersection(retrieved_titles)
    return len(overlap) >= len(gold_titles)


def attribute_error(
    benchmark_item: Dict[str, Any],
    clearrag_result: Dict[str, Any],
    exact_match: float = 0.0,
    token_f1: float = 0.0,
    retrieved_evidence: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Attribute failure to a specific layer in the ClearRAG pipeline.

    Args:
        benchmark_item: Benchmark instance with ground truth.
        clearrag_result: Output dictionary from ClearRAGPipeline.
        exact_match: Exact match score (0.0 to 1.0).
        token_f1: Token F1 score (0.0 to 1.0).
        retrieved_evidence: Optional retrieved evidence chunks list.

    Returns:
        Dictionary containing error_category, explanation, and diagnostic flags.
    """
    condition = benchmark_item.get("condition", "unknown")
    decision = clearrag_result.get("decision", "")
    sufficiency_status = clearrag_result.get("sufficiency_status", "")
    claims_count = clearrag_result.get("claims_count", len(clearrag_result.get("claims", [])))

    chunks = retrieved_evidence or clearrag_result.get("retrieved_evidence", [])
    gold_retrieved = check_gold_evidence_retrieved(benchmark_item, chunks)

    # 1. Condition: UNSUPPORTED or CONFLICT
    if condition == "unsupported":
        if decision in ("ABSTAIN", "CONFLICT_ABSTENTION"):
            return {
                "category": ErrorCategory.CORRECT_EXECUTION.value,
                "explanation": "Correctly abstained on unsupported query.",
                "gold_evidence_retrieved": gold_retrieved,
            }
        elif sufficiency_status == "FULLY_SUPPORTED":
            return {
                "category": ErrorCategory.VERIFICATION_FALSE_POSITIVE.value,
                "explanation": "Verification falsely declared unsupported claims as fully supported.",
                "gold_evidence_retrieved": gold_retrieved,
            }
        else:
            return {
                "category": ErrorCategory.DECISION_POLICY_ERROR.value,
                "explanation": f"Decision engine generated answer ({decision}) for unsupported condition.",
                "gold_evidence_retrieved": gold_retrieved,
            }

    if condition == "conflict":
        if decision == "CONFLICT_ABSTENTION" or decision == "ABSTAIN":
            return {
                "category": ErrorCategory.CORRECT_EXECUTION.value,
                "explanation": "Correctly abstained/warned on conflicting evidence.",
                "gold_evidence_retrieved": gold_retrieved,
            }
        elif sufficiency_status != "CONFLICTING":
            return {
                "category": ErrorCategory.VERIFICATION_FALSE_NEGATIVE.value,
                "explanation": "Verifier failed to detect contradictory attribute values across passages.",
                "gold_evidence_retrieved": gold_retrieved,
            }
        else:
            return {
                "category": ErrorCategory.DECISION_POLICY_ERROR.value,
                "explanation": "Sufficiency was CONFLICTING but decision engine generated answer.",
                "gold_evidence_retrieved": gold_retrieved,
            }

    # 2. Condition: FULL_EVIDENCE / DISTRACTOR_HEAVY
    if condition in ("full_evidence", "distractor_heavy"):
        if not gold_retrieved:
            return {
                "category": ErrorCategory.RETRIEVAL_FAILURE.value,
                "explanation": "Gold supporting documents were not retrieved in top-k.",
                "gold_evidence_retrieved": False,
            }

        if claims_count == 0:
            return {
                "category": ErrorCategory.CLAIM_EXTRACTION_FAILURE.value,
                "explanation": "Rule-based claim extractor failed to extract any sub-claims from query.",
                "gold_evidence_retrieved": True,
            }

        if decision in ("ABSTAIN", "CONFLICT_ABSTENTION"):
            if sufficiency_status in ("UNSUPPORTED", "CONFLICTING"):
                return {
                    "category": ErrorCategory.VERIFICATION_FALSE_NEGATIVE.value,
                    "explanation": "Gold evidence was present but verifier marked claims unsupported/conflicting.",
                    "gold_evidence_retrieved": True,
                }
            else:
                return {
                    "category": ErrorCategory.DECISION_POLICY_ERROR.value,
                    "explanation": f"Sufficiency was {sufficiency_status} but decision was {decision}.",
                    "gold_evidence_retrieved": True,
                }

        # Generated an answer: check correctness
        if exact_match > 0.0 or token_f1 >= 0.5:
            return {
                "category": ErrorCategory.CORRECT_EXECUTION.value,
                "explanation": "Correctly answered query with high alignment to ground truth.",
                "gold_evidence_retrieved": True,
            }
        else:
            return {
                "category": ErrorCategory.GENERATION_ERROR.value,
                "explanation": "Evidence verified, but LLM generation did not match ground truth answer.",
                "gold_evidence_retrieved": True,
            }

    # 3. Condition: PARTIAL_EVIDENCE
    if condition == "partial_evidence":
        if decision == "ANSWER_WITH_CAVEAT":
            return {
                "category": ErrorCategory.CORRECT_EXECUTION.value,
                "explanation": "Correctly identified partial evidence and generated qualified answer with caveat.",
                "gold_evidence_retrieved": gold_retrieved,
            }
        elif decision == "ABSTAIN":
            return {
                "category": ErrorCategory.VERIFICATION_FALSE_NEGATIVE.value,
                "explanation": "Partial evidence present but marked completely unsupported.",
                "gold_evidence_retrieved": gold_retrieved,
            }
        else:
            return {
                "category": ErrorCategory.DECISION_POLICY_ERROR.value,
                "explanation": f"Partial evidence present but decision was {decision}.",
                "gold_evidence_retrieved": gold_retrieved,
            }

    return {
        "category": ErrorCategory.EVALUATION_AMBIGUITY.value,
        "explanation": f"Unrecognized condition {condition} or unhandled evaluation pattern.",
        "gold_evidence_retrieved": gold_retrieved,
    }
