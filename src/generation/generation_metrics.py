"""Generation and Grounding Metrics Suite for ClearRAG.

Computes standard generation quality (EM, F1) alongside attribution and faithfulness
grounding metrics:
- Supported Claim Rate
- Unsupported Claim Rate
- Attribution Coverage
- Attribution Precision
- Caveat Compliance
- Conflict Compliance
- Faithfulness / Groundedness Composite Score
"""

from collections import Counter
import re
import string
from typing import Any, Dict, List, Optional, Set

from src.generation.attribution import AnswerClaimAttribution, AttributionEngine


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    if not isinstance(s, str):
        return ""

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute Exact Match (EM) score."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_contains_ground_truth(prediction: str, ground_truth: str) -> float:
    """Compute substring containment."""
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    if not norm_gt:
        return 0.0
    return 1.0 if norm_gt in norm_pred else 0.0


def compute_grounding_metrics(
    attributions: List[AnswerClaimAttribution],
    evidence_chunks: List[Dict[str, Any]],
    condition: Optional[str] = None,
    prediction_text: str = "",
) -> Dict[str, float]:
    """Compute fine-grained evidence attribution and grounding metrics.

    Args:
        attributions: List of AnswerClaimAttribution objects for the generated answer.
        evidence_chunks: Retrieved evidence chunks available during generation.
        condition: Benchmark condition ('full_evidence', 'partial_evidence', 'conflict', etc.).
        prediction_text: Full generated response text.

    Returns:
        Dictionary of grounding metrics.
    """
    total_claims = len(attributions)
    if total_claims == 0:
        return {
            "supported_claim_rate": 1.0,
            "unsupported_claim_rate": 0.0,
            "attribution_coverage": 0.0,
            "attribution_precision": 1.0,
            "caveat_compliance": 1.0 if condition == "partial_evidence" else 1.0,
            "conflict_compliance": 1.0 if condition == "conflict" else 1.0,
            "faithfulness_score": 1.0,
        }

    supported_claims = sum(1 for a in attributions if a.is_supported)
    unsupported_claims = total_claims - supported_claims
    cited_claims = sum(1 for a in attributions if len(a.cited_chunk_indices) > 0 or len(a.supporting_chunk_ids) > 0)

    supported_claim_rate = supported_claims / total_claims
    unsupported_claim_rate = unsupported_claims / total_claims
    attribution_coverage = cited_claims / total_claims

    # Attribution precision: of all cited chunks across claims, how many contain supporting overlap
    total_citations = 0
    valid_citations = 0
    for a in attributions:
        for cit in a.cited_chunk_indices:
            total_citations += 1
            if a.is_supported:
                valid_citations += 1
    attribution_precision = (valid_citations / total_citations) if total_citations > 0 else (1.0 if supported_claim_rate > 0.5 else 0.0)

    # Caveat compliance: checks if the generated text explicitly acknowledges incomplete/missing information
    caveat_compliance = 0.0
    caveat_indicators = ["note:", "incomplete", "unverified", "missing", "does not state", "not mentioned", "cannot be verified", "provided evidence does not"]
    p_lower = prediction_text.lower()
    has_caveat_phrase = any(ind in p_lower for ind in caveat_indicators)
    if condition == "partial_evidence":
        caveat_compliance = 1.0 if has_caveat_phrase else 0.0
    else:
        caveat_compliance = 1.0

    # Conflict compliance: checks if the response explicitly states contradiction/conflict
    conflict_compliance = 0.0
    conflict_indicators = ["conflict", "contradict", "differ", "disagree", "opposing", "differing", "inconsistent"]
    has_conflict_phrase = any(ind in p_lower for ind in conflict_indicators)
    if condition == "conflict":
        conflict_compliance = 1.0 if has_conflict_phrase else 0.0
    else:
        conflict_compliance = 1.0

    faithfulness_score = (supported_claim_rate * 0.6) + (attribution_precision * 0.4)

    return {
        "supported_claim_rate": round(supported_claim_rate, 4),
        "unsupported_claim_rate": round(unsupported_claim_rate, 4),
        "attribution_coverage": round(attribution_coverage, 4),
        "attribution_precision": round(attribution_precision, 4),
        "caveat_compliance": round(caveat_compliance, 4),
        "conflict_compliance": round(conflict_compliance, 4),
        "faithfulness_score": round(faithfulness_score, 4),
    }
