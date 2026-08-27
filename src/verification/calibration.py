"""Calibration and Threshold Optimization for ClearRAG Evidence Verification.

Performs parameter sweeps over verification thresholds to compute precision-recall curves
and identify optimal operational operating points.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Evaluation metrics for a specific verification threshold configuration."""

    threshold_name: str
    threshold_value: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


class ThresholdCalibrator:
    """Evaluates threshold sweeps across verification prediction outputs."""

    @staticmethod
    def evaluate_threshold(
        predictions: List[Dict[str, Any]],
        score_extractor: Callable[[Dict[str, Any]], float],
        ground_truth_extractor: Callable[[Dict[str, Any]], bool],
        threshold: float,
        name: str = "threshold",
    ) -> CalibrationMetrics:
        """Compute binary classification metrics at a given threshold cutoff.

        Args:
            predictions: List of prediction records.
            score_extractor: Function mapping a record to its continuous verification score.
            ground_truth_extractor: Function mapping a record to its binary ground truth status.
            threshold: Cutoff value above which a claim is predicted positive (supported).
            name: Metric name.

        Returns:
            CalibrationMetrics object.
        """
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for rec in predictions:
            score = score_extractor(rec)
            actual_positive = ground_truth_extractor(rec)
            predicted_positive = score >= threshold

            if predicted_positive and actual_positive:
                tp += 1
            elif predicted_positive and not actual_positive:
                fp += 1
            elif not predicted_positive and not actual_positive:
                tn += 1
            else:
                fn += 1

        total = len(predictions)
        accuracy = ((tp + tn) / total * 100.0) if total > 0 else 0.0
        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return CalibrationMetrics(
            threshold_name=name,
            threshold_value=round(threshold, 3),
            accuracy=round(accuracy, 2),
            precision=round(precision, 2),
            recall=round(recall, 2),
            f1_score=round(f1, 2),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )
