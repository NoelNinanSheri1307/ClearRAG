"""Evaluation package for ClearRAG."""

from src.evaluation.comparative_evaluator import ComparativeEvaluator
from src.evaluation.error_attribution import (
    ErrorCategory,
    attribute_error,
    check_gold_evidence_retrieved,
)
from src.evaluation.generation_metrics import (
    aggregate_generation_metrics,
    compute_contains_ground_truth,
    compute_exact_match,
    compute_generation_metrics,
    compute_token_f1,
    contains_ground_truth,
    exact_match_score,
    normalize_answer,
    token_f1_score,
)
from src.evaluation.oracle import OracleAnalysisResult, OracleEvaluator
from src.evaluation.plots import generate_all_evaluation_plots
from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_query_retrieval_metrics,
)

__all__ = [
    "exact_match_score",
    "token_f1_score",
    "contains_ground_truth",
    "compute_exact_match",
    "compute_token_f1",
    "compute_contains_ground_truth",
    "compute_generation_metrics",
    "aggregate_generation_metrics",
    "normalize_answer",
    "compute_query_retrieval_metrics",
    "aggregate_retrieval_metrics",
    "ErrorCategory",
    "attribute_error",
    "check_gold_evidence_retrieved",
    "OracleEvaluator",
    "OracleAnalysisResult",
    "ComparativeEvaluator",
    "generate_all_evaluation_plots",
]
