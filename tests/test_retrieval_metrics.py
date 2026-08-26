"""Unit tests for retrieval evaluation metrics."""

import pytest
from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_query_retrieval_metrics,
)


def test_compute_query_retrieval_metrics():
    retrieved = [
        {"rank": 1, "document_title": "DocA", "sentence_indices": [0]},
        {"rank": 2, "document_title": "DocB", "sentence_indices": [1]},
        {"rank": 3, "document_title": "DocC", "sentence_indices": [0]},
    ]
    gold_facts = [
        {"title": "DocA", "sentence_index": 0},
        {"title": "DocB", "sentence_index": 0},  # note: sentence 0, but retrieved sentence 1
    ]

    metrics = compute_query_retrieval_metrics(
        retrieved_results=retrieved,
        gold_supporting_facts=gold_facts,
        k_values=[1, 2, 3],
    )

    # At K=1: Retrieved DocA (sentence 0). Gold has DocA & DocB (2 docs). Doc Recall = 1/2 = 0.5. Fact Recall = 1/2 = 0.5.
    assert metrics["doc_recall@1"] == 0.5
    assert metrics["fact_recall@1"] == 0.5
    assert metrics["doc_hit@1"] == 1.0
    assert metrics["doc_full_coverage@1"] == 0.0

    # At K=2: Retrieved DocA, DocB. Doc Recall = 2/2 = 1.0. Fact Recall = 1/2 = 0.5 (DocB s1 is not s0).
    assert metrics["doc_recall@2"] == 1.0
    assert metrics["fact_recall@2"] == 0.5
    assert metrics["doc_hit@2"] == 1.0
    assert metrics["doc_full_coverage@2"] == 1.0


def test_aggregate_retrieval_metrics():
    qm1 = {"doc_recall@1": 0.5, "doc_recall@5": 1.0}
    qm2 = {"doc_recall@1": 1.0, "doc_recall@5": 1.0}

    agg = aggregate_retrieval_metrics([qm1, qm2])
    assert agg["doc_recall@1"] == 0.75
    assert agg["doc_recall@5"] == 1.0
