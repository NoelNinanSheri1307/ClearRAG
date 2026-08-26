"""Retrieval evaluation metrics module for ClearRAG.

Computes Recall@K, HitRate@K, and MRR at both Document-level and
Sentence/Chunk-level against HotpotQA gold supporting evidence annotations.
"""

from typing import Any, Dict, List, Set, Tuple


def compute_query_retrieval_metrics(
    retrieved_results: List[Dict[str, Any]],
    gold_supporting_facts: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute retrieval recall and hit-rate metrics for a single query.

    Metric Definitions:
    1. Document Recall@K (doc_recall@K):
       Fraction of unique gold supporting document titles retrieved in top-K results.
       doc_recall@K = len(retrieved_docs[:K] ∩ gold_docs) / len(gold_docs)

    2. Sentence/Fact Recall@K (fact_recall@K):
       Fraction of exact gold supporting facts (document_title, sentence_index) retrieved in top-K.
       fact_recall@K = len(retrieved_facts[:K] ∩ gold_facts) / len(gold_facts)

    3. Document Hit@K (doc_hit@K):
       1.0 if at least one gold supporting document is in top-K, else 0.0.

    4. Complete Document Coverage@K (doc_full_coverage@K):
       1.0 if ALL gold supporting documents are in top-K, else 0.0.

    Args:
        retrieved_results: Ranked list of retrieval output dicts from Retriever.
        gold_supporting_facts: List of dicts with 'title' and 'sentence_index'.
        k_values: Cutoff ranks to evaluate (default: [1, 3, 5, 10]).

    Returns:
        Dictionary mapping metric names to score values.
    """
    gold_docs: Set[str] = {str(f["title"]) for f in gold_supporting_facts}
    gold_facts: Set[Tuple[str, int]] = {
        (str(f["title"]), int(f["sentence_index"])) for f in gold_supporting_facts
    }

    metrics: Dict[str, float] = {}

    if not gold_docs or not gold_facts:
        # If no gold evidence is required (e.g. unanswerable queries in evaluation)
        for k in k_values:
            metrics[f"doc_recall@{k}"] = 0.0
            metrics[f"fact_recall@{k}"] = 0.0
            metrics[f"doc_hit@{k}"] = 0.0
            metrics[f"doc_full_coverage@{k}"] = 0.0
        return metrics

    for k in k_values:
        top_k_results = retrieved_results[:k]

        # Retrieved docs at K
        retrieved_docs_at_k: Set[str] = {
            res["document_title"] for res in top_k_results if res.get("document_title")
        }

        # Retrieved facts at K
        retrieved_facts_at_k: Set[Tuple[str, int]] = set()
        for res in top_k_results:
            title = res.get("document_title", "")
            for s_idx in res.get("sentence_indices", []):
                retrieved_facts_at_k.add((title, int(s_idx)))

        # Doc Recall & Hit
        doc_overlap = len(retrieved_docs_at_k.intersection(gold_docs))
        metrics[f"doc_recall@{k}"] = doc_overlap / len(gold_docs)
        metrics[f"doc_hit@{k}"] = 1.0 if doc_overlap > 0 else 0.0
        metrics[f"doc_full_coverage@{k}"] = 1.0 if doc_overlap == len(gold_docs) else 0.0

        # Fact Recall
        fact_overlap = len(retrieved_facts_at_k.intersection(gold_facts))
        metrics[f"fact_recall@{k}"] = fact_overlap / len(gold_facts)

    return metrics


def aggregate_retrieval_metrics(
    query_metrics_list: List[Dict[str, float]]
) -> Dict[str, float]:
    """Average metric scores across all evaluated queries.

    Args:
        query_metrics_list: List of per-query metric dictionaries.

    Returns:
        Dictionary mapping metric names to their mean scores across queries.
    """
    if not query_metrics_list:
        return {}

    aggregated: Dict[str, float] = {}
    metric_keys = query_metrics_list[0].keys()

    for key in metric_keys:
        values = [qm[key] for qm in query_metrics_list if key in qm]
        aggregated[key] = sum(values) / len(values) if values else 0.0

    return aggregated
