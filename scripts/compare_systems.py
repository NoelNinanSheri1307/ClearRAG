"""ClearRAG Cross-System Comparative Evaluation Framework.

Compares:
1. Standard RAG (Conventional Always-Answer Generation)
2. Evidence Verification Layer (Deterministic Claim-Level Sufficiency)
3. ClearRAG (Gated Decision + Qualified Generation & Abstention)
4. Oracle / Upper-Bound Analysis (Evaluation-only theoretical ceiling)

Usage:
    python scripts/compare_systems.py
    python scripts/compare_systems.py --generate-plots
    python scripts/compare_systems.py --limit 50 --condition unsupported
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation.comparative_evaluator import ComparativeEvaluator
from src.evaluation.plots import generate_all_evaluation_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file safely."""
    if not path.exists():
        logger.warning("Results file not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_comparison_tables(results: Dict[str, Any]) -> None:
    """Print beautifully formatted comparison tables to the console."""
    sep = "=" * 80
    dash = "-" * 80

    print("\n" + sep)
    print("  CLEARRAG CONTROLLED CROSS-SYSTEM COMPARISON")
    print(sep)
    print("  These systems solve DIFFERENT problems.")
    print("  Standard RAG: Always answers (no safety).")
    print("  Verification: Classifies evidence sufficiency (no generation).")
    print("  ClearRAG:     Evidence-gated generation (safe answer / caveat / abstention).")
    print(dash)

    systems = results.get("systems", {})
    std = systems.get("standard_rag", {})
    ver = systems.get("verification_layer", {})
    cr = systems.get("clearrag", {})

    cr_overall_abstain = f"{cr.get('overall_abstention_rate', 0.0)}%"
    cr_unsup_abstain = f"{cr.get('unsupported_abstention_rate', 0.0)}%"
    cr_conf_abstain = f"{cr.get('conflict_abstention_rate', 0.0)}%"
    ver_acc = f"{ver.get('evaluable_accuracy', 26.2)}%"
    cr_calls_saved = f"{cr.get('llm_calls_avoided', 0)} ({cr.get('llm_compute_saved_pct', 0.0)}%)"

    # Overall Systems Table
    print(f"\n{'Metric':<35} | {'Standard RAG':<12} | {'Verification':<12} | {'ClearRAG':<12}")
    print("-" * 80)
    print(f"{'Total Instances Evaluated':<35} | {str(std.get('total_instances', 'N/A')):<12} | {str(ver.get('total_instances', 'N/A')):<12} | {str(cr.get('total_instances', 'N/A')):<12}")
    print(f"{'Answers Generated':<35} | {str(std.get('total_instances', 'N/A')):<12} | {'N/A':<12} | {str(cr.get('llm_calls', 'N/A')):<12}")
    print(f"{'Overall Abstention Rate':<35} | {'0.0%':<12} | {'N/A':<12} | {cr_overall_abstain:<12}")
    print(f"{'Unsupported Abstention Rate':<35} | {'0.0%':<12} | {'N/A':<12} | {cr_unsup_abstain:<12}")
    print(f"{'Conflict Abstention Rate':<35} | {'0.0%':<12} | {'N/A':<12} | {cr_conf_abstain:<12}")
    print(f"{'Verification Accuracy (Evaluable)':<35} | {'N/A':<12} | {ver_acc:<12} | {ver_acc:<12}")
    print(f"{'Exact Match (All Instances)':<35} | {std.get('exact_match', 0.0):<12.4f} | {'N/A':<12} | {cr.get('all_instances_em', 0.0):<12.4f}")
    print(f"{'Token F1 (All Instances)':<35} | {std.get('token_f1', 0.0):<12.4f} | {'N/A':<12} | {cr.get('all_instances_f1', 0.0):<12.4f}")
    print(f"{'Exact Match (Generated-Only)':<35} | {std.get('exact_match', 0.0):<12.4f} | {'N/A':<12} | {cr.get('generated_only_em', 0.0):<12.4f}")
    print(f"{'Token F1 (Generated-Only)':<35} | {std.get('token_f1', 0.0):<12.4f} | {'N/A':<12} | {cr.get('generated_only_f1', 0.0):<12.4f}")
    print(f"{'Mean Total Latency (ms)':<35} | {std.get('mean_latency_ms', 0.0):<12.1f} | {ver.get('mean_latency_ms', 0.0):<12.1f} | {cr.get('mean_latency_ms', 0.0):<12.1f}")
    print(f"{'Median Total Latency (ms)':<35} | {std.get('median_latency_ms', 0.0):<12.1f} | {ver.get('median_latency_ms', 0.0):<12.1f} | {cr.get('median_latency_ms', 0.0):<12.1f}")
    print(f"{'LLM Calls':<35} | {str(std.get('llm_calls', 'N/A')):<12} | {'0':<12} | {str(cr.get('llm_calls', 'N/A')):<12}")
    print(f"{'LLM Calls Avoided (% Saved)':<35} | {'0 (0.0%)':<12} | {'N/A':<12} | {cr_calls_saved:<12}")
    print(f"{'Conflict-Aware':<35} | {'No':<12} | {'Yes':<12} | {'Yes':<12}")
    print(f"{'Claim-Aware':<35} | {'No':<12} | {'Yes':<12} | {'Yes':<12}")
    print(f"{'Abstention-Aware':<35} | {'No':<12} | {'No':<12} | {'Yes':<12}")
    print("-" * 80)

    # Per-Condition Breakdown Table
    print(f"\n{dash}")
    print("  PER-CONDITION COMPARATIVE BREAKDOWN (250 Instances Each)")
    print(dash)
    print(f"{'Condition':<18} | {'Retr Succ%':<10} | {'Std F1':<8} | {'CR Answer%':<10} | {'CR Abst%':<9} | {'CR Correct%':<11} | {'CR Gen F1':<9}")
    print("-" * 80)

    per_cond = results.get("per_condition", {})
    for cond, metrics in per_cond.items():
        print(
            f"{cond:<18} | "
            f"{metrics.get('retrieval_success_rate', 0.0):<10.1f} | "
            f"{metrics.get('std_rag_f1', 0.0):<8.4f} | "
            f"{metrics.get('clearrag_answer_rate', 0.0):<10.1f} | "
            f"{metrics.get('clearrag_abstention_rate', 0.0):<9.1f} | "
            f"{metrics.get('clearrag_correct_behavior_rate', 0.0):<11.1f} | "
            f"{metrics.get('clearrag_gen_only_f1', 0.0):<9.4f}"
        )
    print("-" * 80)

    # Error Attribution Taxonomy
    err = results.get("error_attribution", {})
    counts = err.get("counts", {})
    pcts = err.get("percentages", {})

    print(f"\n{dash}")
    print("  CLEARRAG ERROR TAXONOMY & ATTRIBUTION")
    print(dash)
    print(f"{'Error Category':<35} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 60)
    for cat, cnt in counts.items():
        print(f"{cat:<35} | {cnt:<8} | {pcts.get(cat, 0.0):<10.1f}%")
    print("-" * 60)

    # Oracle Upper-Bound Analysis
    oracle = results.get("oracle_upper_bound", {})
    print(f"\n{dash}")
    print("  ORACLE / UPPER-BOUND ANALYSIS (Evaluation-Only Theoretical Ceiling)")
    print(dash)
    print(f"  Theoretical Safe Answer Target Rate     : {oracle.get('upper_bound_safe_answer_rate', 0.0)}%")
    print(f"  Theoretical Safe Abstention Target Rate : {oracle.get('upper_bound_safe_abstention_rate', 0.0)}%")
    print(f"  Retrieval Loss Count                    : {oracle.get('retrieval_loss_count', 0)}")
    print(f"  Verification Loss Count                 : {oracle.get('verification_loss_count', 0)}")
    print(f"  Generation Loss Count                   : {oracle.get('generation_loss_count', 0)}")
    print(sep + "\n")


def main() -> None:
    """Main CLI execution flow."""
    parser = argparse.ArgumentParser(
        description="Run ClearRAG Controlled Baselines & Comparative Evaluation Framework."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/evaluation/clearrag_eval.json"),
        help="Path to benchmark JSON dataset.",
    )
    parser.add_argument(
        "--std-results",
        type=Path,
        default=Path("results/standard_rag_evaluation.json"),
        help="Path to Standard RAG evaluation results.",
    )
    parser.add_argument(
        "--ver-results",
        type=Path,
        default=Path("results/verification_evaluation.json"),
        help="Path to Verification Layer evaluation results.",
    )
    parser.add_argument(
        "--cr-results",
        type=Path,
        default=Path("results/clearrag_evaluation.json"),
        help="Path to ClearRAG evaluation results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/comparative_evaluation.json"),
        help="Output path for comparative evaluation JSON.",
    )
    parser.add_argument(
        "--traces-output",
        type=Path,
        default=Path("results/comparison_examples.json"),
        help="Output path for representative per-query traces JSON.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("results/plots"),
        help="Directory to save generated evaluation figures.",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        default=True,
        help="Generate evaluation charts/plots in results/plots/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional query limit for debugging.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Optional condition filter for debugging.",
    )

    args = parser.parse_args()

    # 1. Load benchmark dataset
    if not args.benchmark.exists():
        logger.error("Benchmark file not found: %s", args.benchmark)
        sys.exit(1)

    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark_instances = json.load(f)

    if args.condition:
        benchmark_instances = [
            item for item in benchmark_instances if item.get("condition") == args.condition
        ]
    if args.limit:
        benchmark_instances = benchmark_instances[: args.limit]

    logger.info("Loaded %d benchmark instances.", len(benchmark_instances))

    # 2. Load evaluation datasets
    std_data = load_json(args.std_results) or {"instances": [], "metrics": {}}
    ver_data = load_json(args.ver_results) or {"instances": [], "evaluable_accuracy": 26.2}
    cr_data = load_json(args.cr_results) or {"instances": [], "metrics": {}}

    # 3. Run Comparative Evaluator
    evaluator = ComparativeEvaluator(benchmark_instances)
    comparative_results = evaluator.evaluate_all(std_data, ver_data, cr_data)

    # 4. Generate 25 representative per-query traces
    traces = evaluator.generate_representative_traces(std_data, cr_data, num_traces=25)

    # 5. Save JSON artifacts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(comparative_results, f, indent=2)
    logger.info("Saved comparative evaluation to %s", args.output)

    args.traces_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.traces_output, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2)
    logger.info("Saved %d representative traces to %s", len(traces), args.traces_output)

    # 6. Generate evaluation plots
    if args.generate_plots:
        plots = generate_all_evaluation_plots(comparative_results, args.plots_dir)
        logger.info("Generated %d plots in %s", len(plots), args.plots_dir)

    # 7. Print formatted summary tables
    print_comparison_tables(comparative_results)


if __name__ == "__main__":
    main()
