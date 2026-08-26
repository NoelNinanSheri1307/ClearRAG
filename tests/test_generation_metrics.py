"""Unit tests for generation metrics."""

import pytest
from src.evaluation.generation_metrics import (
    aggregate_generation_metrics,
    compute_contains_ground_truth,
    compute_exact_match,
    compute_generation_metrics,
    compute_token_f1,
    normalize_answer,
)


def test_normalize_answer():
    assert normalize_answer("The quick brown fox.") == "quick brown fox"
    assert normalize_answer("  An   Apple  ") == "apple"
    assert normalize_answer("Hello, World!") == "hello world"


def test_compute_exact_match():
    assert compute_exact_match("Paris", "The Paris") == 1.0
    assert compute_exact_match("Bactris", "bactris.") == 1.0
    assert compute_exact_match("London", "Paris") == 0.0


def test_compute_token_f1():
    # Identical
    assert compute_token_f1("Albert Einstein", "Albert Einstein") == 1.0
    # Partial overlap
    f1 = compute_token_f1("The American actor Thomas Carr", "Thomas Carr")
    assert 0.0 < f1 < 1.0
    # No overlap
    assert compute_token_f1("Paris", "London") == 0.0


def test_compute_contains_ground_truth():
    assert compute_contains_ground_truth("Thomas Carr was an American director.", "Thomas Carr") == 1.0
    assert compute_contains_ground_truth("No relevant info", "Bactris") == 0.0


def test_aggregate_generation_metrics():
    records = [
        {"prediction": "Paris", "ground_truth": "Paris", "condition": "full_evidence"},
        {"prediction": "London", "ground_truth": "Paris", "condition": "full_evidence"},
        {"prediction": "Bactris", "ground_truth": "Bactris", "condition": "unsupported"},
    ]
    agg = aggregate_generation_metrics(records)
    assert agg["total_instances"] == 3
    assert agg["overall"]["exact_match"] == pytest.approx(2.0 / 3.0, 0.01)
    assert "full_evidence" in agg["by_condition"]
    assert "unsupported" in agg["by_condition"]
    assert agg["by_condition"]["full_evidence"]["count"] == 2
    assert agg["by_condition"]["full_evidence"]["exact_match"] == 0.5
