"""Unit tests for ClearRAG Final Research Milestone (Statistical Testing, Safety-Utility, Transitions, Case Studies)."""

import numpy as np
import pytest

from src.evaluation.safety_utility import SafetyUtilityEvaluator, SafetyUtilityMetrics
from src.evaluation.statistical_testing import (
    bootstrap_confidence_interval,
    mcnemar_test,
    wilcoxon_paired_test,
)


class TestStatisticalTesting:
    """Tests for paired statistical hypothesis testing."""

    def test_mcnemar_test_significant_difference(self):
        # 100 queries: System A safe in 30 cases, System B safe in 80 cases
        a_outcomes = [True] * 30 + [False] * 70
        b_outcomes = [True] * 80 + [False] * 20
        res = mcnemar_test(a_outcomes, b_outcomes, label_a="SysA", label_b="SysB")
        assert res.is_significant_05 is True
        assert res.p_value < 0.001
        assert res.effect_size > 1.0

    def test_mcnemar_test_identical_outcomes(self):
        a_outcomes = [True] * 50 + [False] * 50
        b_outcomes = [True] * 50 + [False] * 50
        res = mcnemar_test(a_outcomes, b_outcomes)
        assert res.is_significant_05 is False
        assert res.p_value == 1.0

    def test_wilcoxon_paired_test(self):
        scores_a = [0.10, 0.15, 0.20, 0.05, 0.12] * 20
        scores_b = [0.25, 0.30, 0.35, 0.20, 0.28] * 20
        res = wilcoxon_paired_test(scores_a, scores_b, metric_name="Token F1")
        assert res.is_significant_05 is True
        assert res.effect_size > 0.5

    def test_bootstrap_confidence_interval(self):
        data = [0.10, 0.20, 0.30, 0.40, 0.50] * 20
        low, high = bootstrap_confidence_interval(data, statistic_fn=np.mean)
        assert low <= 0.30 <= high
        assert low > 0.20
        assert high < 0.40


class TestSafetyUtilityEvaluation:
    """Tests for SafetyUtilityEvaluator and transition matrices."""

    @pytest.fixture
    def sample_paired_records(self):
        return [
            {
                "id": "q1",
                "condition": "full_evidence",
                "gold_answer": "Walter Hill",
                "std_answer": "Walter Hill",
                "std_f1": 1.0,
                "std_em": 1.0,
                "std_latency_ms": 2500.0,
                "std_grounding": {"supported_claim_rate": 0.70, "unsupported_claim_rate": 0.30},
                "clearrag_answer": "Walter Hill [1]",
                "clearrag_f1": 1.0,
                "clearrag_em": 1.0,
                "clearrag_latency_ms": 2450.0,
                "clearrag_did_generate": True,
                "clearrag_decision": "ANSWER",
                "clearrag_confidence": 0.95,
                "clearrag_grounding": {"supported_claim_rate": 1.0, "unsupported_claim_rate": 0.0, "attribution_coverage": 1.0},
            },
            {
                "id": "q2",
                "condition": "unsupported",
                "gold_answer": "Unknown",
                "std_answer": "Hallucinated Entity X",
                "std_f1": 0.0,
                "std_em": 0.0,
                "std_latency_ms": 2400.0,
                "std_grounding": {"supported_claim_rate": 0.0, "unsupported_claim_rate": 1.0},
                "clearrag_answer": "I cannot answer based on available evidence.",
                "clearrag_f1": 0.0,
                "clearrag_em": 0.0,
                "clearrag_latency_ms": 95.0,
                "clearrag_did_generate": False,
                "clearrag_decision": "ABSTAIN",
                "clearrag_confidence": 0.10,
                "clearrag_grounding": {},
            },
        ]

    def test_compute_metrics(self, sample_paired_records):
        metrics_std = SafetyUtilityEvaluator.compute_metrics("Standard RAG", sample_paired_records, is_clearrag=False)
        metrics_clr = SafetyUtilityEvaluator.compute_metrics("ClearRAG", sample_paired_records, is_clearrag=True)

        assert metrics_std.answer_rate == 100.0
        assert metrics_clr.answer_rate == 50.0
        assert metrics_clr.correct_abstention_rate == 100.0
        assert metrics_clr.llm_calls_avoided == 1
        assert metrics_clr.compute_saved_percentage == 50.0

    def test_error_transition_matrix(self, sample_paired_records):
        transitions = SafetyUtilityEvaluator.compute_error_transition_matrix(sample_paired_records)
        assert transitions["total_queries"] == 2
        assert transitions["transition_counts"]["STD_HALLUCINATION -> CLEAR_CORRECT_ABSTAIN"] == 1
        assert transitions["transition_counts"]["STD_CORRECT -> CLEAR_CORRECT_ANSWER"] == 1

    def test_coverage_risk_curve(self, sample_paired_records):
        curve = SafetyUtilityEvaluator.compute_coverage_risk_curve(sample_paired_records, [0.0, 0.50, 0.90])
        assert len(curve) == 3
        assert curve[0]["coverage_percentage"] == 50.0

    def test_case_studies_selection(self, sample_paired_records):
        cases = SafetyUtilityEvaluator.select_case_studies(sample_paired_records)
        assert len(cases) >= 1
        assert "q2" in [c["id"] for c in cases]

    def test_audit_accounting_invariants(self):
        import json
        from pathlib import Path
        canon_path = Path("results/final_canonical_evaluation.json")
        if canon_path.exists():
            with open(canon_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["metadata"]["total_instances"] == 1250
            sys0 = data["systems"]["system_0_standard_rag"]
            sys4 = data["systems"]["system_4_final_clearrag"]
            assert sys0["total_answers_generated"] == 1250
            assert sys4["total_answers_generated"] + sys4["llm_calls_avoided"] == 1250

    def test_coverage_risk_quality_invariants(self):
        import json
        from pathlib import Path
        crq_path = Path("results/coverage_risk_quality.json")
        if crq_path.exists():
            with open(crq_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ops = data["operating_points"]
            assert len(ops) >= 5
            # Invariant: lowest coverage point has highest compute saved
            assert ops[0]["compute_saved_percentage"] > ops[-1]["compute_saved_percentage"]
            # Invariant: highest coverage point has higher unsupported risk than lowest
            assert ops[-1]["unsupported_claim_rate"] > ops[0]["unsupported_claim_rate"]
