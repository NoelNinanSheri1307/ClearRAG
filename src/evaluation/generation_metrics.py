"""Evaluation metrics for RAG generation quality and correctness.

Implements standard Exact Match (EM), token-level F1, substring ground-truth containment,
and per-benchmark-condition breakdown.
"""

from collections import Counter
import re
import string
from typing import Any, Dict, List, Optional


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    Standard SQuAD / HotpotQA normalization.
    """

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
    """Compute normalized Exact Match (EM) between prediction and ground truth.

    Returns:
        1.0 if normalized strings match, 0.0 otherwise.
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth.

    Returns:
        F1 score between 0.0 and 1.0.
    """
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
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_contains_ground_truth(prediction: str, ground_truth: str) -> float:
    """Check if normalized ground truth appears as a substring in the prediction.

    Returns:
        1.0 if ground truth is contained in prediction, 0.0 otherwise.
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    if not norm_gt:
        return 0.0
    return 1.0 if norm_gt in norm_pred else 0.0


def compute_generation_metrics(
    prediction: str,
    ground_truth: str,
) -> Dict[str, float]:
    """Compute all generation metrics for a single prediction."""
    return {
        "exact_match": compute_exact_match(prediction, ground_truth),
        "token_f1": round(compute_token_f1(prediction, ground_truth), 4),
        "contains_gt": compute_contains_ground_truth(prediction, ground_truth),
    }


def aggregate_generation_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics across a list of prediction records.

    Args:
        records: List of dictionaries containing 'prediction', 'ground_truth',
                 and optionally 'condition'.

    Returns:
        Dictionary with overall metrics and breakdown by condition.
    """
    if not records:
        return {
            "total_instances": 0,
            "overall": {
                "exact_match": 0.0,
                "token_f1": 0.0,
                "contains_gt": 0.0,
            },
            "by_condition": {},
        }

    total = len(records)
    em_sum = 0.0
    f1_sum = 0.0
    contains_gt_sum = 0.0

    condition_groups: Dict[str, List[Dict[str, Any]]] = {}

    for rec in records:
        pred = rec.get("prediction", "")
        gt = rec.get("ground_truth", "")
        cond = rec.get("condition", "unknown")

        metrics = compute_generation_metrics(pred, gt)
        em_sum += metrics["exact_match"]
        f1_sum += metrics["token_f1"]
        contains_gt_sum += metrics["contains_gt"]

        if cond not in condition_groups:
            condition_groups[cond] = []
        condition_groups[cond].append(metrics)

    overall_metrics = {
        "exact_match": round(em_sum / total, 4),
        "token_f1": round(f1_sum / total, 4),
        "contains_gt": round(contains_gt_sum / total, 4),
    }

    by_condition: Dict[str, Any] = {}
    for cond, metrics_list in condition_groups.items():
        n = len(metrics_list)
        by_condition[cond] = {
            "count": n,
            "exact_match": round(sum(m["exact_match"] for m in metrics_list) / n, 4),
            "token_f1": round(sum(m["token_f1"] for m in metrics_list) / n, 4),
            "contains_gt": round(sum(m["contains_gt"] for m in metrics_list) / n, 4),
        }

    return {
        "total_instances": total,
        "overall": overall_metrics,
        "by_condition": by_condition,
    }
