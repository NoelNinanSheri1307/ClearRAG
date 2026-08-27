"""ClearRAG Retrieval Experimentation & Diagnostic Evaluation Runner.

Runs Experiments A through E across all 1,250 benchmark queries:
- Experiment A: Current Baseline (Dense BGE, k=5)
- Experiment B: Parameter Tuning (Dense BGE Top-K sweep: 3, 5, 8, 10, 15, 20)
- Experiment C: Hybrid Retrieval (Dense BGE + Lexical BM25 RRF)
- Experiment D: Reranking (Hybrid Pool N=25 -> Cross-Scorer Rerank)
- Experiment E: Best Combined Configuration

Saves:
- results/retrieval_experiments.json
- results/retrieval_examples.json
- results/plots/ (5 retrieval diagnostic charts)
"""

import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Headless matplotlib
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.retrieval.bm25 import BM25Index
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossScorerReranker
from src.retrieval.retriever import Retriever
from src.evaluation.error_attribution import check_gold_evidence_retrieved
from src.evaluation.retrieval_metrics import compute_query_retrieval_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_retrieval_plots(
    experiments_data: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    """Generate 5 clear, informative retrieval diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    conditions = ["full_evidence", "partial_evidence", "unsupported", "distractor_heavy", "conflict"]
    cond_labels = ["Full Evidence", "Partial Evidence", "Unsupported", "Distractor Heavy", "Conflict"]

    # 1. Retrieval Success by Condition (Exp A vs Exp C vs Exp D vs Exp E)
    plt.figure(figsize=(10, 5), dpi=300)
    exp_keys = ["exp_a_dense_k5", "exp_c_hybrid_k5", "exp_d_rerank_k5", "exp_e_best_k10"]
    exp_labels = ["Exp A (Dense k=5)", "Exp C (Hybrid k=5)", "Exp D (Rerank k=5)", "Exp E (Best Hybrid+Rerank k=10)"]
    colors = ["#3498db", "#9b59b6", "#e67e22", "#2ecc71"]

    x = np.arange(len(conditions))
    width = 0.2

    for idx, (ek, el, color) in enumerate(zip(exp_keys, exp_labels, colors)):
        if ek in experiments_data:
            rates = [
                experiments_data[ek]["per_condition"][c]["gold_success_rate"] for c in conditions
            ]
            plt.bar(x + (idx - 1.5) * width, rates, width, label=el, color=color)

    plt.title("Gold Retrieval Success Rate by Benchmark Condition (%)", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Benchmark Condition", fontweight="bold")
    plt.ylabel("Success Rate (%)", fontweight="bold")
    plt.xticks(x, cond_labels, rotation=10)
    plt.ylim(0, 105)
    plt.legend(frameon=True)
    plt.tight_layout()
    p1 = output_dir / "retrieval_gold_success_by_condition.png"
    plt.savefig(p1)
    plt.close()
    generated["gold_success_by_condition"] = str(p1)

    # 2. Retrieval Failure Reduction across Experiments
    plt.figure(figsize=(9, 5), dpi=300)
    all_exps = [
        ("Exp A (Dense k=5)", "exp_a_dense_k5"),
        ("Exp B (Dense k=10)", "exp_b_dense_k10"),
        ("Exp B (Dense k=20)", "exp_b_dense_k20"),
        ("Exp C (Hybrid k=5)", "exp_c_hybrid_k5"),
        ("Exp C (Hybrid k=10)", "exp_c_hybrid_k10"),
        ("Exp D (Rerank k=5)", "exp_d_rerank_k5"),
        ("Exp E (Best Combined)", "exp_e_best_k10"),
    ]
    exp_names = [e[0] for e in all_exps if e[1] in experiments_data]
    fail_counts = [experiments_data[e[1]]["retrieval_failures"] for e in all_exps if e[1] in experiments_data]

    bars = plt.bar(exp_names, fail_counts, color="#e74c3c")
    plt.title("Retrieval Failure Count Across Experiments (Lower is Better)", fontsize=12, fontweight="bold", pad=15)
    plt.ylabel("Total Retrieval Failures (out of 1,250)", fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    for bar in bars:
        h = bar.get_height()
        plt.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    p2 = output_dir / "retrieval_failure_reduction.png"
    plt.savefig(p2)
    plt.close()
    generated["failure_reduction"] = str(p2)

    # 3. Recall@K Curve for Parameter Tuning (Exp B)
    plt.figure(figsize=(8, 5), dpi=300)
    k_vals = [3, 5, 8, 10, 15, 20]
    k_rates = []
    for k in k_vals:
        key = f"exp_b_dense_k{k}"
        if key in experiments_data:
            k_rates.append(experiments_data[key]["overall_gold_success_rate"])
        elif k == 5 and "exp_a_dense_k5" in experiments_data:
            k_rates.append(experiments_data["exp_a_dense_k5"]["overall_gold_success_rate"])

    if len(k_rates) == len(k_vals):
        plt.plot(k_vals, k_rates, marker="o", linewidth=2.5, color="#2980b9", label="Dense BGE Recall@K")
        plt.title("Top-K Parameter Sweep (Dense Recall Curve)", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Top-K Chunks Retrieved", fontweight="bold")
        plt.ylabel("Overall Gold Retrieval Success (%)", fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim(50, 95)
        for x_val, y_val in zip(k_vals, k_rates):
            plt.annotate(f"{y_val:.1f}%", xy=(x_val, y_val), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontweight="bold")
        plt.tight_layout()
        p3 = output_dir / "retrieval_recall_curve.png"
        plt.savefig(p3)
        plt.close()
        generated["recall_curve"] = str(p3)

    # 4. Distractor-Heavy Gold Retrieval Comparison
    plt.figure(figsize=(8, 5), dpi=300)
    dist_rates = [
        experiments_data.get(e[1], {}).get("per_condition", {}).get("distractor_heavy", {}).get("gold_success_rate", 0.0)
        for e in all_exps if e[1] in experiments_data
    ]
    bars4 = plt.bar(exp_names, dist_rates, color="#16a085")
    plt.title("Distractor-Heavy Gold Evidence Retrieval Success (%)", fontsize=12, fontweight="bold", pad=15)
    plt.ylabel("Success Rate (%)", fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 100)
    for bar in bars4:
        h = bar.get_height()
        plt.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    p4 = output_dir / "distractor_heavy_comparison.png"
    plt.savefig(p4)
    plt.close()
    generated["distractor_heavy"] = str(p4)

    # 5. Retrieval Latency Comparison (Mean vs Median)
    plt.figure(figsize=(9, 5), dpi=300)
    means = [experiments_data.get(e[1], {}).get("latency_mean_ms", 0.0) for e in all_exps if e[1] in experiments_data]
    meds = [experiments_data.get(e[1], {}).get("latency_median_ms", 0.0) for e in all_exps if e[1] in experiments_data]
    x_l = np.arange(len(exp_names))
    plt.bar(x_l - width / 2, means, width, label="Mean Latency (ms)", color="#f39c12")
    plt.bar(x_l + width / 2, meds, width, label="Median Latency (ms)", color="#d35400")
    plt.title("Retrieval Latency Across Configurations (ms)", fontsize=12, fontweight="bold", pad=15)
    plt.ylabel("Latency (ms)", fontweight="bold")
    plt.xticks(x_l, exp_names, rotation=25, ha="right")
    plt.legend(frameon=True)
    plt.tight_layout()
    p5 = output_dir / "retrieval_latency_comparison.png"
    plt.savefig(p5)
    plt.close()
    generated["latency_comparison"] = str(p5)

    return generated


def run_experiment_evaluation(
    name: str,
    retriever_fn,
    benchmark: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Evaluate a specific retriever configuration on the benchmark dataset."""
    logger.info("Evaluating %s (top_k=%d)...", name, top_k)
    latencies: List[float] = []
    gold_success_count = 0
    cond_success = defaultdict(int)
    cond_totals = defaultdict(int)
    traces = []

    for item in benchmark:
        q = item["question"]
        c = item.get("condition", "unknown")
        cond_totals[c] += 1

        t0 = time.perf_counter()
        retrieved_chunks = retriever_fn(q, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        is_success = check_gold_evidence_retrieved(item, retrieved_chunks)
        if is_success:
            gold_success_count += 1
            cond_success[c] += 1

        # Collect detailed trace for analysis
        if len(traces) < 30:
            traces.append(
                {
                    "id": item.get("id", ""),
                    "condition": c,
                    "question": q,
                    "gold_success": is_success,
                    "retained_supporting_facts": item.get("retained_supporting_facts", []),
                    "retrieved_titles": [c.get("document_title", "") for c in retrieved_chunks],
                    "latency_ms": round(lat, 2),
                }
            )

    total = len(benchmark)
    fail_count = total - gold_success_count
    success_rate = (gold_success_count / total * 100.0) if total > 0 else 0.0

    per_condition: Dict[str, Any] = {}
    for c in ["full_evidence", "partial_evidence", "unsupported", "distractor_heavy", "conflict"]:
        cnt = cond_totals[c]
        succ = cond_success[c]
        rate = (succ / cnt * 100.0) if cnt > 0 else 0.0
        per_condition[c] = {
            "total": cnt,
            "gold_success_count": succ,
            "gold_success_rate": round(rate, 2),
            "failures": cnt - succ,
        }

    return {
        "experiment_name": name,
        "top_k": top_k,
        "total_queries": total,
        "overall_gold_success_rate": round(success_rate, 2),
        "gold_success_count": gold_success_count,
        "retrieval_failures": fail_count,
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_median_ms": round(statistics.median(latencies), 2),
        "per_condition": per_condition,
        "sample_traces": traces,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ClearRAG Retrieval Experiments and Diagnostics.")
    parser.add_argument("--benchmark", type=Path, default=Path("data/evaluation/clearrag_eval.json"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/faiss_index.bin"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/index_metadata.json"))
    parser.add_argument("--bm25-cache", type=Path, default=Path("data/processed/bm25_index.pkl"))
    parser.add_argument("--output", type=Path, default=Path("results/retrieval_experiments.json"))
    parser.add_argument("--examples-output", type=Path, default=Path("results/retrieval_examples.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--generate-plots", action="store_true", default=True)

    args = parser.parse_args()

    # 1. Load benchmark dataset
    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    logger.info("Loaded %d benchmark instances.", len(benchmark))

    # 2. Load Embedder and FAISS Index
    embedder = BGEEmbedder(device="cuda")
    faiss_index = FAISSIndex.load(args.index, args.metadata)
    logger.info("Loaded FAISS index with %d vectors.", faiss_index.ntotal)

    # 3. Load or Build BM25 Index
    if args.bm25_cache.exists():
        bm25_index = BM25Index.load(args.bm25_cache)
    else:
        logger.info("Building BM25 index over %d corpus chunks...", len(faiss_index.metadata_store))
        bm25_index = BM25Index(title_weight=2.5)
        bm25_index.build_from_metadata(faiss_index.metadata_store)
        bm25_index.save(args.bm25_cache)

    # 4. Instantiate Retriever and Reranker
    reranker = CrossScorerReranker()
    hybrid_engine = HybridRetriever(
        embedder=embedder,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
    )

    experiments: Dict[str, Any] = {}
    distractor_traces = []

    # -------------------------------------------------------------
    # EXPERIMENT A: Baseline Dense (k=5) - Control
    # -------------------------------------------------------------
    def dense_retriever_fn(q, top_k=5):
        qv = embedder.embed_query(q)
        scores, metas = faiss_index.search(qv, top_k=top_k)
        return metas

    exp_a = run_experiment_evaluation("Exp A (Dense k=5 Control)", dense_retriever_fn, benchmark, top_k=5)
    experiments["exp_a_dense_k5"] = exp_a

    # -------------------------------------------------------------
    # EXPERIMENT B: Parameter Tuning (Top-K Sweep: 3, 8, 10, 15, 20)
    # -------------------------------------------------------------
    for k in [3, 8, 10, 15, 20]:
        exp_b_k = run_experiment_evaluation(f"Exp B (Dense k={k})", dense_retriever_fn, benchmark, top_k=k)
        experiments[f"exp_b_dense_k{k}"] = exp_b_k

    # -------------------------------------------------------------
    # EXPERIMENT C: Hybrid Retrieval (Dense + BM25 RRF, k=5 and k=10)
    # -------------------------------------------------------------
    def hybrid_retriever_fn(q, top_k=5):
        return hybrid_engine.retrieve(q, top_k=top_k, candidate_pool_k=30, fusion_method="rrf")

    exp_c_5 = run_experiment_evaluation("Exp C (Hybrid RRF k=5)", hybrid_retriever_fn, benchmark, top_k=5)
    experiments["exp_c_hybrid_k5"] = exp_c_5

    exp_c_10 = run_experiment_evaluation("Exp C (Hybrid RRF k=10)", hybrid_retriever_fn, benchmark, top_k=10)
    experiments["exp_c_hybrid_k10"] = exp_c_10

    # -------------------------------------------------------------
    # EXPERIMENT D: Reranking Stage (Hybrid Pool N=30 -> Cross-Scorer Rerank)
    # -------------------------------------------------------------
    def rerank_retriever_fn(q, top_k=5):
        initial_pool = hybrid_engine.retrieve(q, top_k=30, candidate_pool_k=30, fusion_method="rrf")
        return reranker.rerank(q, initial_pool, top_k=top_k)

    exp_d_5 = run_experiment_evaluation("Exp D (Hybrid + Rerank k=5)", rerank_retriever_fn, benchmark, top_k=5)
    experiments["exp_d_rerank_k5"] = exp_d_5

    exp_d_10 = run_experiment_evaluation("Exp D (Hybrid + Rerank k=10)", rerank_retriever_fn, benchmark, top_k=10)
    experiments["exp_d_rerank_k10"] = exp_d_10

    # -------------------------------------------------------------
    # EXPERIMENT E: Best Combined Configuration (Hybrid + Rerank k=10)
    # -------------------------------------------------------------
    exp_e = exp_d_10
    experiments["exp_e_best_k10"] = exp_e

    # 5. Extract Representative Distractor-Heavy Traces
    for item in benchmark:
        if item.get("condition") == "distractor_heavy":
            q = item["question"]
            res_dense = dense_retriever_fn(q, top_k=5)
            res_hybrid = hybrid_retriever_fn(q, top_k=5)
            res_rerank = rerank_retriever_fn(q, top_k=5)

            succ_dense = check_gold_evidence_retrieved(item, res_dense)
            succ_hybrid = check_gold_evidence_retrieved(item, res_hybrid)
            succ_rerank = check_gold_evidence_retrieved(item, res_rerank)

            distractor_traces.append(
                {
                    "id": item["id"],
                    "question": q,
                    "condition": "distractor_heavy",
                    "retained_supporting_facts": item.get("retained_supporting_facts", []),
                    "dense_k5_success": succ_dense,
                    "dense_k5_titles": [c.get("document_title", "") for c in res_dense],
                    "hybrid_k5_success": succ_hybrid,
                    "hybrid_k5_titles": [c.get("document_title", "") for c in res_hybrid],
                    "rerank_k5_success": succ_rerank,
                    "rerank_k5_titles": [c.get("document_title", "") for c in res_rerank],
                }
            )
            if len(distractor_traces) >= 25:
                break

    # 6. Save Experiment Outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)
    logger.info("Saved retrieval experiments to %s", args.output)

    with open(args.examples_output, "w", encoding="utf-8") as f:
        json.dump(distractor_traces, f, indent=2)
    logger.info("Saved %d distractor-heavy diagnostic traces to %s", len(distractor_traces), args.examples_output)

    # 7. Generate Plots
    if args.generate_plots:
        plots = generate_retrieval_plots(experiments, args.plots_dir)
        logger.info("Generated %d retrieval plots in %s", len(plots), args.plots_dir)

    # 8. Print Console Summary Tables
    print("\n" + "=" * 90)
    print("  CLEARRAG RETRIEVAL EXPERIMENTS (1,250 Benchmark Queries)")
    print("=" * 90)
    print(f"{'Experiment':<26} | {'Top-K':<5} | {'Gold Succ%':<10} | {'Failures':<8} | {'Distractor%':<11} | {'Full Ev%':<9} | {'Mean Lat':<9}")
    print("-" * 90)

    for ek, data in experiments.items():
        pc = data["per_condition"]
        print(
            f"{data['experiment_name']:<26} | "
            f"{data['top_k']:<5} | "
            f"{data['overall_gold_success_rate']:<10.1f} | "
            f"{data['retrieval_failures']:<8} | "
            f"{pc['distractor_heavy']['gold_success_rate']:<11.1f} | "
            f"{pc['full_evidence']['gold_success_rate']:<9.1f} | "
            f"{data['latency_mean_ms']:<9.1f}ms"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
