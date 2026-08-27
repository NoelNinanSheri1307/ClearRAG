"""Evaluation package for ClearRAG."""

from src.evaluation.error_attribution import (
    ErrorCategory,
    attribute_error,
    check_gold_evidence_retrieved,
)
from src.evaluation.generation_metrics import (
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
from src.evaluation.safety_utility import (
    SafetyUtilityEvaluator,
    SafetyUtilityMetrics,
)
from src.evaluation.statistical_testing import (
    StatisticalTestResult,
    bootstrap_confidence_interval,
    mcnemar_test,
    wilcoxon_paired_test,
)

__all__ = [
    "compute_exact_match",
    "compute_token_f1",
    "compute_contains_ground_truth",
    "compute_generation_metrics",
    "normalize_answer",
    "compute_query_retrieval_metrics",
    "aggregate_retrieval_metrics",
    "attribute_error",
    "check_gold_evidence_retrieved",
    "ErrorCategory",
    "mcnemar_test",
    "wilcoxon_paired_test",
    "bootstrap_confidence_interval",
    "StatisticalTestResult",
    "SafetyUtilityEvaluator",
    "SafetyUtilityMetrics",
]
