"""Evaluation metrics and benchmark utilities for ClearRAG."""

from src.evaluation.generation_metrics import (
    aggregate_generation_metrics,
    compute_contains_ground_truth,
    compute_exact_match,
    compute_generation_metrics,
    compute_token_f1,
    normalize_answer,
)
from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_query_retrieval_metrics,
)

__all__ = [
    "compute_query_retrieval_metrics",
    "aggregate_retrieval_metrics",
    "normalize_answer",
    "compute_exact_match",
    "compute_token_f1",
    "compute_contains_ground_truth",
    "compute_generation_metrics",
    "aggregate_generation_metrics",
]
