"""Cross-system comparison framework for ClearRAG research.

Compares results from three systems using the same benchmark and retrieval foundation:

    System 1: Standard RAG
        - Answer generation quality (always answers)

    System 2: Verification Layer
        - Evidence sufficiency classification

    System 3: ClearRAG
        - Answer quality + evidence-grounded behavior + abstention/reliability

These systems solve DIFFERENT problems. This script does NOT produce a
misleading single-number comparison. Instead, it reports each system's
strength in its own evaluation domain.

Usage:
    python scripts/compare_systems.py
"""

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict:
    """Load JSON file, return empty dict if not found."""
    if not path.exists():
        logger.warning(f"Results file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_section(title: str):
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Compare Standard RAG, Verification, and ClearRAG systems.")
    parser.add_argument("--rag_path", type=str, default="results/standard_rag_evaluation.json")
    parser.add_argument("--verify_path", type=str, default="results/verification_evaluation.json")
    parser.add_argument("--clearrag_path", type=str, default="results/clearrag_evaluation.json")
    args = parser.parse_args()

    rag_results = load_json(Path(args.rag_path))
    verify_results = load_json(Path(args.verify_path))
    clearrag_results = load_json(Path(args.clearrag_path))

    print("=" * 70)
    print("  CLEARRAG CROSS-SYSTEM COMPARISON")
    print("=" * 70)
    print()
    print("  These systems solve DIFFERENT problems.")
    print("  A single-number comparison would be misleading.")
    print("  Each system is evaluated in its own domain.")

    # ----------------------------------------------
    # System 1: Standard RAG — Generation Quality
    # ----------------------------------------------
    print_section("SYSTEM 1: Standard RAG — Always-Answer Generation")

    if rag_results:
        meta = rag_results.get("evaluation_metadata", {})
        overall = rag_results.get("overall_metrics", {})
        print(f"  Benchmark Instances     : {meta.get('total_benchmark_instances', '?')}")
        print(f"  Model                   : {meta.get('model_name', '?')}")
        print(f"  Exact Match             : {overall.get('exact_match', '?')}")
        print(f"  Token F1                : {overall.get('token_f1', '?')}")
        print(f"  Contains GT             : {overall.get('contains_gt', '?')}")
        print()
        print("  Behavior: Standard RAG always generates an answer regardless of")
        print("  evidence quality. It has NO abstention capability.")

        breakdown = rag_results.get("condition_breakdown", {})
        if breakdown:
            print(f"\n  {'Condition':<18} | {'EM':<8} | {'F1':<8} | {'Contains':<8}")
            print(f"  {'-' * 48}")
            for cond in ["full_evidence", "partial_evidence", "unsupported", "distractor_heavy", "conflict"]:
                if cond in breakdown:
                    m = breakdown[cond]
                    print(f"  {cond:<18} | {m.get('exact_match', 0):<8.4f} | {m.get('token_f1', 0):<8.4f} | {m.get('contains_gt', 0):<8.4f}")
    else:
        print("  [NOT AVAILABLE] — Run scripts/evaluate_rag.py first.")

    # ----------------------------------------------
    # System 2: Verification Layer — Sufficiency Classification
    # ----------------------------------------------
    print_section("SYSTEM 2: Verification Layer — Evidence Sufficiency")

    if verify_results:
        meta = verify_results.get("metadata", {})
        print(f"  Benchmark Instances     : {meta.get('total_instances', '?')}")
        print(f"  Evaluable Accuracy      : {meta.get('evaluable_classification_accuracy', '?')}%")
        print(f"  Evaluable Correct       : {meta.get('evaluable_correct', '?')} / {meta.get('evaluable_instances', '?')}")
        print()
        print("  Behavior: Classifies evidence sufficiency (FULLY_SUPPORTED,")
        print("  PARTIALLY_SUPPORTED, UNSUPPORTED, CONFLICTING). Does NOT generate answers.")

        confusion = verify_results.get("confusion_matrix", {})
        if confusion:
            statuses = ["FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONFLICTING"]
            print(f"\n  {'Condition':<18} | {'FULL':<6} | {'PART':<6} | {'UNSUP':<6} | {'CONF':<6}")
            print(f"  {'-' * 52}")
            for cond in ["full_evidence", "partial_evidence", "unsupported", "distractor_heavy", "conflict"]:
                if cond in confusion:
                    m = confusion[cond]
                    print(f"  {cond:<18} | {m.get('FULLY_SUPPORTED', 0):<6} | {m.get('PARTIALLY_SUPPORTED', 0):<6} | "
                          f"{m.get('UNSUPPORTED', 0):<6} | {m.get('CONFLICTING', 0):<6}")
    else:
        print("  [NOT AVAILABLE] — Run scripts/evaluate_verification.py first.")

    # ----------------------------------------------
    # System 3: ClearRAG — Decision + Abstention + Generation
    # ----------------------------------------------
    print_section("SYSTEM 3: ClearRAG — Evidence-Grounded Decision + Abstention")

    if clearrag_results:
        meta = clearrag_results.get("metadata", {})
        gen_qual = clearrag_results.get("generation_quality", {})
        abstention = clearrag_results.get("abstention_behavior", {})
        safety = clearrag_results.get("safety_reliability", {})
        latency = clearrag_results.get("latency", {})

        overall_gen = gen_qual.get("overall", {})
        generated_gen = gen_qual.get("generated_only", {})

        print(f"  Benchmark Instances     : {meta.get('total_instances', '?')}")
        print(f"  Model                   : {meta.get('model_name', '?')}")

        print(f"\n  -- Generation Quality (all instances) --")
        print(f"  Exact Match             : {overall_gen.get('exact_match', '?')}")
        print(f"  Token F1                : {overall_gen.get('token_f1', '?')}")
        print(f"  Contains GT             : {overall_gen.get('contains_gt', '?')}")

        if generated_gen:
            print(f"\n  -- Generation Quality (generated-only) --")
            print(f"  Exact Match             : {generated_gen.get('exact_match', '?')}")
            print(f"  Token F1                : {generated_gen.get('token_f1', '?')}")
            print(f"  Contains GT             : {generated_gen.get('contains_gt', '?')}")

        print(f"\n  -- Abstention Behavior --")
        print(f"  Overall Abstention Rate : {abstention.get('overall_abstention_rate', '?')}%")
        print(f"  Total Abstentions       : {abstention.get('total_abstentions', '?')} / {meta.get('total_instances', '?')}")
        print(f"  Correct Abstentions     : {abstention.get('correct_abstentions', '?')}")
        print(f"    Unsupported           : {abstention.get('correct_abstention_unsupported', '?')}")
        print(f"    Conflict              : {abstention.get('correct_abstention_conflict', '?')}")
        print(f"  False Ans (unsupported) : {abstention.get('false_answer_on_unsupported', '?')}")

        print(f"\n  -- Safety / Reliability --")
        print(f"  Unsupported Answer Rate : {safety.get('unsupported_answer_rate', '?')}%")
        print(f"  Conflict Answer Rate    : {safety.get('conflict_answer_rate', '?')}%")
        print(f"  Supported Answer Rate   : {safety.get('supported_answer_rate', '?')}%")

        print(f"\n  Behavior: ClearRAG answers ONLY when evidence is sufficient.")
        print(f"  It abstains on unsupported queries and conflicting evidence,")
        print(f"  adding qualification caveats for partial evidence.")

        # Decision Matrix
        decision_matrix = clearrag_results.get("evidence_behavior", {}).get("decision_matrix", {})
        if decision_matrix:
            print(f"\n  {'Condition':<18} | {'ANSWER':<8} | {'CAVEAT':<8} | {'ABSTAIN':<8} | {'CONFLICT':<10}")
            print(f"  {'-' * 58}")
            for cond in ["full_evidence", "partial_evidence", "unsupported", "distractor_heavy", "conflict"]:
                if cond in decision_matrix:
                    m = decision_matrix[cond]
                    print(f"  {cond:<18} | {m.get('ANSWER', 0):<8} | {m.get('ANSWER_WITH_CAVEAT', 0):<8} | "
                          f"{m.get('ABSTAIN', 0):<8} | {m.get('CONFLICT_ABSTENTION', 0):<10}")

        # Latency
        if latency:
            print(f"\n  -- Latency (mean) --")
            print(f"  Retrieval               : {latency.get('retrieval_ms', {}).get('mean', '?')} ms")
            print(f"  Verification            : {latency.get('verification_ms', {}).get('mean', '?')} ms")
            print(f"  Generation              : {latency.get('generation_ms', {}).get('mean', '?')} ms")
            print(f"  Total                   : {latency.get('total_ms', {}).get('mean', '?')} ms")
    else:
        print("  [NOT AVAILABLE] — Run scripts/evaluate_clearrag.py first.")

    # ----------------------------------------------
    # Key Differences Summary
    # ----------------------------------------------
    print_section("KEY DIFFERENCES")
    print()
    print("  +----------------+--------------+--------------+--------------+")
    print("  | Capability     | Standard RAG | Verification | ClearRAG     |")
    print("  +----------------+--------------+--------------+--------------+")
    print("  | Retrieval      | Y            | Y            | Y            |")
    print("  | Generation     | Y (always)   | N            | Y (cond.)    |")
    print("  | Verification   | N            | Y            | Y            |")
    print("  | Abstention     | N            | N/A          | Y            |")
    print("  | Caveat         | N            | N/A          | Y            |")
    print("  | Conflict Detect| N            | Y            | Y            |")
    print("  | Evidence Audit | N            | Y            | Y            |")
    print("  +----------------+--------------+--------------+--------------+")
    print()
    print("  Standard RAG: Always answers. No safety mechanism.")
    print("  Verification: Classifies evidence. No generation.")
    print("  ClearRAG:     Answers when safe. Abstains when evidence is")
    print("                insufficient or conflicting. Adds caveats for")
    print("                partial evidence. Full provenance audit trail.")

    print("\n" + "=" * 70)
    print("  COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

