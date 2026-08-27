"""Unit tests for Comparative Evaluation Framework, Error Taxonomy, and Oracle Analysis."""

import pytest

from src.evaluation.comparative_evaluator import ComparativeEvaluator
from src.evaluation.error_attribution import (
    ErrorCategory,
    attribute_error,
    check_gold_evidence_retrieved,
)
from src.evaluation.oracle import OracleEvaluator


@pytest.fixture
def sample_benchmark():
    return [
        {
            "id": "q1_full",
            "condition": "full_evidence",
            "question": "Which genus has more species, Bactris and Epigaea?",
            "ground_truth": "Bactris",
            "retained_supporting_facts": [{"title": "Bactris", "sentence_index": 0}],
            "expected_behavior": "answer",
        },
        {
            "id": "q2_unsupported",
            "condition": "unsupported",
            "question": "Did Neil Armstrong land on Mars in 1969?",
            "ground_truth": "No evidence",
            "retained_supporting_facts": [],
            "expected_behavior": "abstain",
        },
        {
            "id": "q3_conflict",
            "condition": "conflict",
            "question": "When was Thomas Carr born?",
            "ground_truth": "Conflicting",
            "retained_supporting_facts": [{"title": "Thomas Carr", "sentence_index": 0}],
            "expected_behavior": "abstain",
        },
        {
            "id": "q4_partial",
            "condition": "partial_evidence",
            "question": "What award did Iqbal Qadir win and in what year?",
            "ground_truth": "Sitara-e-Imtiaz in 1971",
            "retained_supporting_facts": [{"title": "Iqbal Qadir", "sentence_index": 0}],
            "expected_behavior": "answer_with_caveat",
        },
    ]


class TestErrorAttributionTaxonomy:
    """Tests for error taxonomy attribution logic."""

    def test_unsupported_correct_abstention(self):
        item = {"condition": "unsupported", "retained_supporting_facts": []}
        res = {"decision": "ABSTAIN", "sufficiency_status": "UNSUPPORTED", "claims": []}
        attr = attribute_error(item, res)
        assert attr["category"] == ErrorCategory.CORRECT_EXECUTION.value

    def test_unsupported_false_positive(self):
        item = {"condition": "unsupported", "retained_supporting_facts": []}
        res = {"decision": "ANSWER", "sufficiency_status": "FULLY_SUPPORTED", "claims": [{"entity": "Mars"}]}
        attr = attribute_error(item, res)
        assert attr["category"] == ErrorCategory.VERIFICATION_FALSE_POSITIVE.value

    def test_conflict_correct_abstention(self):
        item = {"condition": "conflict", "retained_supporting_facts": [{"title": "Thomas Carr"}]}
        res = {"decision": "CONFLICT_ABSTENTION", "sufficiency_status": "CONFLICTING", "claims": []}
        attr = attribute_error(item, res)
        assert attr["category"] == ErrorCategory.CORRECT_EXECUTION.value

    def test_conflict_false_negative_detection(self):
        item = {"condition": "conflict", "retained_supporting_facts": [{"title": "Thomas Carr"}]}
        res = {"decision": "ANSWER", "sufficiency_status": "FULLY_SUPPORTED", "claims": []}
        attr = attribute_error(item, res)
        assert attr["category"] == ErrorCategory.VERIFICATION_FALSE_NEGATIVE.value

    def test_full_evidence_retrieval_failure(self):
        item = {"condition": "full_evidence", "retained_supporting_facts": [{"title": "Bactris"}]}
        res = {
            "decision": "ABSTAIN",
            "sufficiency_status": "UNSUPPORTED",
            "claims": [{"entity": "Bactris"}],
            "retrieved_evidence": [{"title": "Random Document"}],
        }
        attr = attribute_error(item, res, retrieved_evidence=[{"title": "Random Document"}])
        assert attr["category"] == ErrorCategory.RETRIEVAL_FAILURE.value

    def test_full_evidence_generation_error(self):
        item = {"condition": "full_evidence", "retained_supporting_facts": [{"title": "Bactris"}]}
        res = {
            "decision": "ANSWER",
            "sufficiency_status": "FULLY_SUPPORTED",
            "claims": [{"entity": "Bactris"}],
            "retrieved_evidence": [{"title": "Bactris"}],
        }
        attr = attribute_error(
            item, res, exact_match=0.0, token_f1=0.1, retrieved_evidence=[{"title": "Bactris"}]
        )
        assert attr["category"] == ErrorCategory.GENERATION_ERROR.value

    def test_gold_evidence_retrieval_check(self):
        item = {
            "retained_supporting_facts": [
                {"title": "Bactris", "sentence_index": 0},
                {"title": "Epigaea", "sentence_index": 0},
            ]
        }
        retrieved_both = [{"title": "Bactris"}, {"title": "Epigaea"}]
        retrieved_one = [{"title": "Bactris"}]

        assert check_gold_evidence_retrieved(item, retrieved_both) is True
        assert check_gold_evidence_retrieved(item, retrieved_one) is False


class TestOracleEvaluator:
    """Tests for Oracle Theoretical Upper-Bound Analyzer."""

    def test_oracle_analysis(self, sample_benchmark):
        oracle = OracleEvaluator(sample_benchmark)
        clearrag_results = [
            {"decision": "ANSWER"},
            {"decision": "ABSTAIN"},
            {"decision": "CONFLICT_ABSTENTION"},
            {"decision": "ANSWER_WITH_CAVEAT"},
        ]
        attributions = [
            {"category": "CORRECT_EXECUTION"},
            {"category": "CORRECT_EXECUTION"},
            {"category": "CORRECT_EXECUTION"},
            {"category": "CORRECT_EXECUTION"},
        ]

        analysis = oracle.analyze_system_gap(clearrag_results, attributions)

        assert analysis.total_instances == 4
        assert analysis.oracle_theoretical_abstentions == 2  # unsupported + conflict
        assert analysis.oracle_theoretical_answers == 1      # full_evidence
        assert analysis.oracle_theoretical_caveats == 1      # partial_evidence
        assert analysis.upper_bound_safe_answer_rate == 50.0 # (1 + 1) / 4 = 50%
        assert analysis.upper_bound_safe_abstention_rate == 50.0 # 2 / 4 = 50%


class TestComparativeEvaluator:
    """Tests for ComparativeEvaluator cross-system integration."""

    def test_comparative_evaluator_full_flow(self, sample_benchmark):
        evaluator = ComparativeEvaluator(sample_benchmark)

        std_data = {
            "predictions": [
                {"id": "q1_full", "condition": "full_evidence", "prediction": "Bactris", "metrics": {"exact_match": 1.0, "token_f1": 1.0}, "latency_total_ms": 120.0},
                {"id": "q2_unsupported", "condition": "unsupported", "prediction": "Yes", "metrics": {"exact_match": 0.0, "token_f1": 0.0}, "latency_total_ms": 110.0},
                {"id": "q3_conflict", "condition": "conflict", "prediction": "1904", "metrics": {"exact_match": 0.0, "token_f1": 0.0}, "latency_total_ms": 115.0},
                {"id": "q4_partial", "condition": "partial_evidence", "prediction": "Sitara", "metrics": {"exact_match": 0.0, "token_f1": 0.5}, "latency_total_ms": 125.0},
            ]
        }

        ver_data = {
            "evaluable_accuracy": 50.0,
            "predictions": [
                {"instance_id": "q1_full", "actual_condition": "full_evidence", "predicted_status": "FULLY_SUPPORTED"},
                {"instance_id": "q2_unsupported", "actual_condition": "unsupported", "predicted_status": "UNSUPPORTED"},
                {"instance_id": "q3_conflict", "actual_condition": "conflict", "predicted_status": "FULLY_SUPPORTED"},
                {"instance_id": "q4_partial", "actual_condition": "partial_evidence", "predicted_status": "PARTIALLY_SUPPORTED"},
            ]
        }

        cr_data = {
            "predictions": [
                {"instance_id": "q1_full", "condition": "full_evidence", "decision": "ANSWER", "metrics": {"exact_match": 1.0, "token_f1": 1.0}, "total_latency_ms": 150.0, "retrieved_evidence": [{"title": "Bactris"}]},
                {"instance_id": "q2_unsupported", "condition": "unsupported", "decision": "ABSTAIN", "metrics": {"exact_match": 0.0, "token_f1": 0.0}, "total_latency_ms": 20.0, "retrieved_evidence": []},
                {"instance_id": "q3_conflict", "condition": "conflict", "decision": "CONFLICT_ABSTENTION", "metrics": {"exact_match": 0.0, "token_f1": 0.0}, "total_latency_ms": 25.0, "retrieved_evidence": [{"title": "Thomas Carr"}]},
                {"instance_id": "q4_partial", "condition": "partial_evidence", "decision": "ANSWER_WITH_CAVEAT", "metrics": {"exact_match": 0.0, "token_f1": 0.5}, "total_latency_ms": 160.0, "retrieved_evidence": [{"title": "Iqbal Qadir"}]},
            ]
        }

        results = evaluator.evaluate_all(std_data, ver_data, cr_data)

        assert results["summary"]["total_queries"] == 4
        assert results["systems"]["clearrag"]["overall_abstention_rate"] == 50.0 # 2 / 4
        assert results["systems"]["clearrag"]["llm_calls_avoided"] == 2
        assert "per_condition" in results
        assert "error_attribution" in results
        assert "oracle_upper_bound" in results

    def test_representative_traces_generation(self, sample_benchmark):
        evaluator = ComparativeEvaluator(sample_benchmark)

        std_data = {
            "predictions": [
                {"id": "q1_full", "condition": "full_evidence", "prediction": "Bactris", "metrics": {"exact_match": 1.0, "token_f1": 1.0}},
            ]
        }
        cr_data = {
            "predictions": [
                {"instance_id": "q1_full", "question": "Which genus has more species?", "condition": "full_evidence", "decision": "ANSWER", "prediction": "Bactris", "metrics": {"exact_match": 1.0, "token_f1": 1.0}, "claims_count": 1, "retrieved_evidence": [{"title": "Bactris"}]},
            ]
        }

        traces = evaluator.generate_representative_traces(std_data, cr_data, num_traces=5)
        assert len(traces) == 1
        assert traces[0]["query_id"] == "q1_full"
        assert traces[0]["clearrag_decision"] == "ANSWER"
        assert traces[0]["error_category"] == "CORRECT_EXECUTION"
