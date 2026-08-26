"""Evaluate ClearRAG Retrieval Pipeline on HotpotQA / ClearRAG benchmark.

Computes Document Recall@K, Sentence/Fact Recall@K, Hit Rate@K, and Full Coverage@K.

Usage:
    python scripts/evaluate_retrieval.py [--subset 250] [--output_path results/retrieval_evaluation.json]
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from tqdm import tqdm

from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_query_retrieval_metrics,
)
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("evaluate_retrieval")


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate ClearRAG Retrieval Pipeline.")
    parser.add_argument("--config", type=str, default="configs/retrieval_config.yaml")
    parser.add_argument("--eval_path", type=str, default=None, help="Path to evaluation JSON")
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="results/retrieval_evaluation.json")
    parser.add_argument("--subset", type=int, default=None, help="Number of evaluation queries (default from config)")
    parser.add_argument("--all", action="store_true", help="Evaluate on all benchmark queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset selection")
    args = parser.parse_args()

    full_cfg = load_config(Path(args.config))
    eval_cfg = full_cfg.get("evaluation", {})
    idx_cfg = full_cfg.get("indexing", {})
    emb_cfg = full_cfg.get("embedding", {})

    eval_path = Path(args.eval_path or eval_cfg.get("benchmark_path", "data/evaluation/clearrag_eval.json"))
    index_path = Path(args.index_path or idx_cfg.get("index_output_path", "data/processed/faiss_index.bin"))
    metadata_path = Path(args.metadata_path or idx_cfg.get("metadata_output_path", "data/processed/index_metadata.json"))
    output_path = Path(args.output_path)
    k_values = eval_cfg.get("k_values", [1, 3, 5, 10])
    max_k = max(k_values)
    seed = args.seed or eval_cfg.get("random_seed", 42)

    print("=" * 65)
    print("ClearRAG — Retrieval Evaluation Benchmark")
    print("=" * 65)
    print(f"Benchmark Path   : {eval_path}")
    print(f"FAISS Index Path : {index_path}")
    print(f"K Values         : {k_values}")
    print(f"Random Seed      : {seed}")

    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation benchmark dataset missing at {eval_path}")
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("FAISS index or metadata missing. Run scripts/build_index.py first.")

    # 1. Load benchmark dataset
    with open(eval_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # Filter to answerable/evidence-containing conditions for meaningful recall evaluation
    # Or evaluate across all instances
    eval_instances = benchmark_data
    if not args.all and args.subset:
        subset_size = args.subset
    elif not args.all and eval_cfg.get("default_eval_subset"):
        subset_size = eval_cfg.get("default_eval_subset")
    else:
        subset_size = len(eval_instances)

    if subset_size < len(eval_instances):
        rng = random.Random(seed)
        shuffled = list(eval_instances)
        rng.shuffle(shuffled)
        eval_instances = shuffled[:subset_size]

    print(f"Evaluating on {len(eval_instances):,} benchmark instances (Total available: {len(benchmark_data):,})...\n")

    # 2. Load Retriever
    embedder = BGEEmbedder(
        model_name=emb_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
        device=None if emb_cfg.get("device", "auto") == "auto" else emb_cfg.get("device"),
    )
    retriever = Retriever.from_saved_index(
        index_path=index_path,
        metadata_path=metadata_path,
        embedder=embedder,
        default_top_k=max_k,
    )

    # 3. Evaluate queries
    start_eval = time.perf_counter()
    all_query_metrics: List[Dict[str, float]] = []
    condition_metrics: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for item in tqdm(eval_instances, desc="Evaluating retrieval"):
        query = item.get("question", "")
        condition = item.get("condition", "general")
        gold_facts = item.get("original_supporting_facts") or item.get("supporting_facts", [])

        # Format gold facts if raw list of lists
        if gold_facts and isinstance(gold_facts[0], (list, tuple)):
            gold_facts_fmt = [{"title": str(f[0]), "sentence_index": int(f[1])} for f in gold_facts]
        else:
            gold_facts_fmt = gold_facts

        # Retrieve top-K
        retrieved = retriever.retrieve(query, top_k=max_k)

        # Compute metrics
        qm = compute_query_retrieval_metrics(
            retrieved_results=retrieved,
            gold_supporting_facts=gold_facts_fmt,
            k_values=k_values,
        )
        all_query_metrics.append(qm)
        condition_metrics[condition].append(qm)

    eval_duration = time.perf_counter() - start_eval
    overall_results = aggregate_retrieval_metrics(all_query_metrics)

    # 4. Print results table
    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total Evaluated Queries: {len(eval_instances):,}")
    print(f"Total Time             : {eval_duration:.2f} s ({len(eval_instances)/eval_duration:.1f} queries/s)")
    print("-" * 65)
    print(f"{'Metric':<25} | {'Score':>10}")
    print("-" * 65)

    for k in k_values:
        print(f"Document Recall@{k:<2}       | {overall_results.get(f'doc_recall@{k}', 0.0):>9.4f} ({overall_results.get(f'doc_recall@{k}', 0.0)*100:.1f}%)")
    print("-" * 65)
    for k in k_values:
        print(f"Sentence/Fact Recall@{k:<2}  | {overall_results.get(f'fact_recall@{k}', 0.0):>9.4f} ({overall_results.get(f'fact_recall@{k}', 0.0)*100:.1f}%)")
    print("-" * 65)
    for k in k_values:
        print(f"Document HitRate@{k:<2}      | {overall_results.get(f'doc_hit@{k}', 0.0):>9.4f} ({overall_results.get(f'doc_hit@{k}', 0.0)*100:.1f}%)")
    print("-" * 65)
    for k in k_values:
        print(f"Full Doc Coverage@{k:<2}     | {overall_results.get(f'doc_full_coverage@{k}', 0.0):>9.4f} ({overall_results.get(f'doc_full_coverage@{k}', 0.0)*100:.1f}%)")
    print("=" * 65)

    # Condition breakdown if available
    if len(condition_metrics) > 1:
        print("\nBreakdown by Benchmark Condition (Doc Recall@5 / Fact Recall@5):")
        print("-" * 65)
        for cond, qm_list in condition_metrics.items():
            agg = aggregate_retrieval_metrics(qm_list)
            doc_r5 = agg.get("doc_recall@5", 0.0) * 100
            fact_r5 = agg.get("fact_recall@5", 0.0) * 100
            doc_r10 = agg.get("doc_recall@10", 0.0) * 100
            fact_r10 = agg.get("fact_recall@10", 0.0) * 100
            print(f"  {cond:<20} ({len(qm_list):>3} queries): Doc@5={doc_r5:.1f}%, Fact@5={fact_r5:.1f}% | Doc@10={doc_r10:.1f}%, Fact@10={fact_r10:.1f}%")
        print("=" * 65)

    # 5. Save output report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "evaluation_summary": {
            "benchmark_path": str(eval_path),
            "faiss_index_path": str(index_path),
            "embedding_model": emb_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
            "num_evaluated_queries": len(eval_instances),
            "random_seed": seed,
            "evaluation_duration_seconds": eval_duration,
        },
        "overall_metrics": overall_results,
        "condition_breakdown": {
            cond: aggregate_retrieval_metrics(qm_list)
            for cond, qm_list in condition_metrics.items()
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nSaved evaluation metrics to: {output_path}")


if __name__ == "__main__":
    main()
