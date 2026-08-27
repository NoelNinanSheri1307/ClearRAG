"""Full benchmark evaluation runner for ClearRAG Decision + Abstention Layer.

Evaluates the complete ClearRAG pipeline over all 1,250 benchmark instances.

CRITICAL EVALUATION ISOLATION:
    During inference, the ClearRAG pipeline uses ONLY:
        - question
        - retrieved evidence (via FAISS)

    It MUST NOT use:
        - condition
        - expected_behavior
        - ground-truth answer
        - benchmark labels

    Those fields are accessed ONLY after inference for evaluation scoring.

Metrics:
    A. Generation quality: Exact Match, Token F1, Contains GT
    B. Evidence behavior: sufficiency status distribution
    C. Abstention behavior: abstention rates, correct abstention
    D. Safety/reliability: hallucination on unsupported, conflict-answer rate
    E. Latency: retrieval, verification, generation, total

Usage:
    python scripts/evaluate_clearrag.py
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clearrag.decision import ClearRAGDecision, ClearRAGDecisionEngine
from src.clearrag.pipeline import ClearRAGPipeline
from src.evaluation.generation_metrics import compute_generation_metrics
from src.generation.llm_generator import LLMGenerator
from src.retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ClearRAG over full benchmark.")
    parser.add_argument("--eval_path", type=str, default="data/evaluation/clearrag_eval.json")
    parser.add_argument("--output_path", type=str, default="results/clearrag_evaluation.json")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--index_path", type=str, default="data/processed/faiss_index.bin")
    parser.add_argument("--metadata_path", type=str, default="data/processed/index_metadata.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        logger.error(f"Benchmark file not found: {eval_path}")
        sys.exit(1)

    # Load benchmark
    with open(eval_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    logger.info(f"Loaded {len(benchmark)} benchmark instances.")

    # Initialize components
    logger.info("Loading retriever...")
    retriever = Retriever.from_saved_index(
        Path(args.index_path), Path(args.metadata_path), default_top_k=args.top_k
    )

    logger.info(f"Loading LLM generator ({args.model})...")
    generator = LLMGenerator(model_name=args.model)

    decision_engine = ClearRAGDecisionEngine()
    pipeline = ClearRAGPipeline(
        retriever=retriever,
        generator=generator,
        decision_engine=decision_engine,
        default_top_k=args.top_k,
    )

    # =================================================
    # Run evaluation
    # =================================================
    conditions = ["full_evidence", "partial_evidence", "unsupported", "distractor_heavy", "conflict"]
    decisions_list = [d.value for d in ClearRAGDecision]

    # Tracking structures
    predictions: List[Dict[str, Any]] = []
    decision_matrix = {c: {d: 0 for d in decisions_list} for c in conditions}
    condition_counts = {c: 0 for c in conditions}

    # Latency accumulators
    latency_retrieval = []
    latency_verification = []
    latency_generation = []
    latency_total = []

    # Generation metrics accumulators (only for non-abstention)
    gen_metrics_by_condition: Dict[str, List[Dict[str, float]]] = {c: [] for c in conditions}
    gen_metrics_all: List[Dict[str, float]] = []

    # Abstention tracking
    abstention_counts = {c: 0 for c in conditions}
    total_abstentions = 0
    total_generated = 0

    start_time = time.time()

    print("=" * 70)
    print(f"EVALUATING CLEARRAG PIPELINE ({len(benchmark)} QUERIES)")
    print("=" * 70)

    for idx, item in enumerate(benchmark, start=1):
        q_id = item.get("id", f"query_{idx}")
        question = item["question"]
        ground_truth = item.get("ground_truth", item.get("answer", ""))
        condition = item["condition"]

        # -- STRICT UNLEAKED INFERENCE --
        # Pipeline sees ONLY the question. No condition, no GT, no labels.
        result = pipeline.answer(question, top_k=args.top_k)

        # -- POST-INFERENCE EVALUATION --
        condition_counts[condition] += 1
        decision_matrix[condition][result.decision.value] += 1

        # Latencies
        latency_retrieval.append(result.retrieval_latency_ms)
        latency_verification.append(result.verification_latency_ms)
        latency_generation.append(result.generation_latency_ms)
        latency_total.append(result.total_latency_ms)

        # Track abstentions
        is_abstention = result.is_abstention
        if is_abstention:
            abstention_counts[condition] += 1
            total_abstentions += 1
        else:
            total_generated += 1

        # Compute generation metrics (against ground truth, post-inference)
        metrics = compute_generation_metrics(result.answer, ground_truth)
        gen_metrics_all.append(metrics)
        gen_metrics_by_condition[condition].append(metrics)

        # Build prediction record
        record = {
            "instance_id": q_id,
            "question": question,
            "condition": condition,
            "ground_truth": ground_truth,
            "prediction": result.answer,
            "decision": result.decision.value,
            "sufficiency_status": result.sufficiency_status.value,
            "is_abstention": is_abstention,
            "metrics": metrics,
            "retrieval_latency_ms": result.retrieval_latency_ms,
            "verification_latency_ms": result.verification_latency_ms,
            "generation_latency_ms": result.generation_latency_ms,
            "total_latency_ms": result.total_latency_ms,
            "claims_count": len(result.claims),
            "supporting_evidence_count": len(result.supporting_evidence),
            "conflicting_evidence_count": len(result.conflicting_evidence),
        }
        predictions.append(record)

        if idx % 50 == 0 or idx == len(benchmark):
            elapsed = time.time() - start_time
            abstain_rate = (total_abstentions / idx * 100)
            print(f"[{idx}/{len(benchmark)}] Elapsed: {elapsed:.1f}s | "
                  f"Abstention Rate: {abstain_rate:.1f}% | "
                  f"Generated: {total_generated}")

    total_duration = time.time() - start_time
    n = len(benchmark)

    # =================================================
    # Compute aggregate metrics
    # =================================================

    # A. Generation Quality
    def avg_metrics(metrics_list):
        if not metrics_list:
            return {"exact_match": 0.0, "token_f1": 0.0, "contains_gt": 0.0}
        return {
            "exact_match": round(sum(m["exact_match"] for m in metrics_list) / len(metrics_list), 4),
            "token_f1": round(sum(m["token_f1"] for m in metrics_list) / len(metrics_list), 4),
            "contains_gt": round(sum(m["contains_gt"] for m in metrics_list) / len(metrics_list), 4),
        }

    overall_gen_metrics = avg_metrics(gen_metrics_all)
    condition_gen_metrics = {c: avg_metrics(gen_metrics_by_condition[c]) for c in conditions}

    # Generated-only metrics (excluding abstentions)
    generated_only = [p["metrics"] for p in predictions if not p["is_abstention"]]
    generated_gen_metrics = avg_metrics(generated_only)

    # B. Evidence Behavior (sufficiency distribution)
    sufficiency_dist = {}
    for p in predictions:
        s = p["sufficiency_status"]
        sufficiency_dist[s] = sufficiency_dist.get(s, 0) + 1

    # C. Abstention Behavior
    overall_abstention_rate = round(total_abstentions / n * 100, 2) if n > 0 else 0.0

    # Correct abstentions: abstaining on unsupported or conflict
    correct_abstentions_unsupported = abstention_counts.get("unsupported", 0)
    correct_abstentions_conflict = abstention_counts.get("conflict", 0)
    correct_abstentions = correct_abstentions_unsupported + correct_abstentions_conflict

    # False answers on unsupported: queries where condition=unsupported but system answered
    false_answer_on_unsupported = condition_counts.get("unsupported", 0) - abstention_counts.get("unsupported", 0)

    # D. Safety/Reliability
    # Unsupported-answer rate: fraction of unsupported queries that got an answer
    unsupported_total = condition_counts.get("unsupported", 0)
    unsupported_answer_rate = round(false_answer_on_unsupported / unsupported_total * 100, 2) if unsupported_total > 0 else 0.0

    # Conflict-answer rate: fraction of conflict queries that got an answer
    conflict_total = condition_counts.get("conflict", 0)
    conflict_answered = conflict_total - abstention_counts.get("conflict", 0)
    conflict_answer_rate = round(conflict_answered / conflict_total * 100, 2) if conflict_total > 0 else 0.0

    # Supported-answer rate: fraction of full_evidence queries that got an answer
    full_total = condition_counts.get("full_evidence", 0)
    full_answered = full_total - abstention_counts.get("full_evidence", 0)
    supported_answer_rate = round(full_answered / full_total * 100, 2) if full_total > 0 else 0.0

    # E. Latency
    def safe_avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    def safe_p50(lst):
        if not lst:
            return 0.0
        s = sorted(lst)
        return round(s[len(s) // 2], 2)

    latency_stats = {
        "retrieval_ms": {"mean": safe_avg(latency_retrieval), "p50": safe_p50(latency_retrieval)},
        "verification_ms": {"mean": safe_avg(latency_verification), "p50": safe_p50(latency_verification)},
        "generation_ms": {"mean": safe_avg(latency_generation), "p50": safe_p50(latency_generation)},
        "total_ms": {"mean": safe_avg(latency_total), "p50": safe_p50(latency_total)},
    }

    # =================================================
    # Build output
    # =================================================
    output_data = {
        "metadata": {
            "system": "ClearRAG",
            "total_instances": n,
            "total_duration_seconds": round(total_duration, 2),
            "queries_per_second": round(n / total_duration, 2) if total_duration > 0 else 0,
            "model_name": args.model,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "top_k": args.top_k,
        },
        "generation_quality": {
            "overall": overall_gen_metrics,
            "generated_only": generated_gen_metrics,
            "by_condition": condition_gen_metrics,
        },
        "evidence_behavior": {
            "sufficiency_distribution": sufficiency_dist,
            "decision_matrix": decision_matrix,
        },
        "abstention_behavior": {
            "overall_abstention_rate": overall_abstention_rate,
            "total_abstentions": total_abstentions,
            "total_generated": total_generated,
            "abstentions_by_condition": abstention_counts,
            "correct_abstentions": correct_abstentions,
            "correct_abstention_unsupported": correct_abstentions_unsupported,
            "correct_abstention_conflict": correct_abstentions_conflict,
            "false_answer_on_unsupported": false_answer_on_unsupported,
        },
        "safety_reliability": {
            "unsupported_answer_rate": unsupported_answer_rate,
            "conflict_answer_rate": conflict_answer_rate,
            "supported_answer_rate": supported_answer_rate,
        },
        "latency": latency_stats,
        "predictions": predictions,
    }

    # Atomic write
    temp_path = output_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    if output_path.exists():
        output_path.unlink()
    temp_path.rename(output_path)

    # =================================================
    # Print summary
    # =================================================
    print("\n" + "=" * 70)
    print("CLEARRAG EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal Instances           : {n}")
    print(f"Total Duration            : {total_duration:.2f} s")
    print(f"Queries/Second            : {n / total_duration:.2f}" if total_duration > 0 else "")

    print(f"\n-- Generation Quality (all {n} instances) --")
    print(f"  Exact Match             : {overall_gen_metrics['exact_match']:.4f}")
    print(f"  Token F1                : {overall_gen_metrics['token_f1']:.4f}")
    print(f"  Contains GT             : {overall_gen_metrics['contains_gt']:.4f}")

    if generated_only:
        print(f"\n-- Generation Quality (generated-only: {len(generated_only)} instances) --")
        print(f"  Exact Match             : {generated_gen_metrics['exact_match']:.4f}")
        print(f"  Token F1                : {generated_gen_metrics['token_f1']:.4f}")
        print(f"  Contains GT             : {generated_gen_metrics['contains_gt']:.4f}")

    print(f"\n-- Abstention Behavior --")
    print(f"  Overall Abstention Rate : {overall_abstention_rate:.2f}%")
    print(f"  Total Abstentions       : {total_abstentions} / {n}")
    print(f"  Total Generated         : {total_generated} / {n}")
    print(f"  Correct Abstentions     : {correct_abstentions}")
    print(f"    Unsupported           : {correct_abstentions_unsupported} / {unsupported_total}")
    print(f"    Conflict              : {correct_abstentions_conflict} / {conflict_total}")
    print(f"  False Answer (unsup.)   : {false_answer_on_unsupported} / {unsupported_total}")

    print(f"\n-- Safety / Reliability --")
    print(f"  Unsupported Answer Rate : {unsupported_answer_rate:.2f}%")
    print(f"  Conflict Answer Rate    : {conflict_answer_rate:.2f}%")
    print(f"  Supported Answer Rate   : {supported_answer_rate:.2f}%")

    print(f"\n-- Decision Matrix --")
    header = f"{'Condition':<18} | {'ANSWER':<8} | {'CAVEAT':<8} | {'ABSTAIN':<8} | {'CONFLICT':<10}"
    print(header)
    print("-" * len(header))
    for c in conditions:
        counts = decision_matrix[c]
        print(f"{c:<18} | {counts.get('ANSWER', 0):<8} | {counts.get('ANSWER_WITH_CAVEAT', 0):<8} | "
              f"{counts.get('ABSTAIN', 0):<8} | {counts.get('CONFLICT_ABSTENTION', 0):<10}")

    print(f"\n-- Latency --")
    print(f"  Retrieval (mean)        : {latency_stats['retrieval_ms']['mean']:.2f} ms")
    print(f"  Verification (mean)     : {latency_stats['verification_ms']['mean']:.2f} ms")
    print(f"  Generation (mean)       : {latency_stats['generation_ms']['mean']:.2f} ms")
    print(f"  Total (mean)            : {latency_stats['total_ms']['mean']:.2f} ms")

    print("=" * 70)
    print(f"Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

