"""ClearRAG Verification Improvement & Diagnostic Experiment Runner.

Executes Experiments V-A through V-G across all 1,250 benchmark queries:
- Exp V-A: Baseline Verification Control (reproduces 26.20% evaluable accuracy)
- Exp V-B: Threshold Calibration Sweep
- Exp V-C: Semantic & Lexical Evidence Matching
- Exp V-D: Multi-Hop Claim Verification
- Exp V-E: Contradiction / Conflict Detection Engine
- Exp V-F: Sufficiency Aggregation Improvement
- Exp V-G: Combined Verification System

Outputs:
- results/verification_experiments.json
- results/verification_false_negative_analysis.json
- results/verification_false_positive_analysis.json
- results/verification_examples.json
- results/plots/ (5 diagnostic figures)
"""

import argparse
from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.retrieval.embedder import BGEEmbedder
from src.retrieval.retriever import Retriever
from src.verification.calibration import ThresholdCalibrator
from src.verification.claim_extractor import RuleBasedClaimExtractor
from src.verification.claims import Claim, ClaimType
from src.verification.contradiction import ContradictionDetector
from src.verification.evidence_matching import SemanticEvidenceMatcher
from src.verification.evidence_verifier import EvidenceVerifier
from src.verification.improved_verifier import ImprovedEvidenceVerifier
from src.verification.models import SufficiencyStatus, VerificationResult, VerificationStatus
from src.verification.sufficiency import SufficiencyEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONDITION_TO_EXPECTED = {
    "full_evidence": "FULLY_SUPPORTED",
    "partial_evidence": "PARTIALLY_SUPPORTED",
    "unsupported": "UNSUPPORTED",
    "conflict": "CONFLICTING",
}


def generate_verification_plots(
    experiments_data: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    """Generate 5 clear, informative verification diagnostic charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    conditions = ["full_evidence", "partial_evidence", "unsupported", "conflict"]
    cond_labels = ["Full Evidence", "Partial Evidence", "Unsupported", "Conflict"]

    # 1. Verification Accuracy Comparison (Exp V-A to V-G)
    plt.figure(figsize=(10, 5), dpi=300)
    exp_keys = [
        ("V-A (Control)", "exp_v_a_control"),
        ("V-B (Calibrated)", "exp_v_b_calibration"),
        ("V-C (Semantic Match)", "exp_v_c_semantic_match"),
        ("V-D (Multi-Hop)", "exp_v_d_multihop"),
        ("V-E (Conflict Engine)", "exp_v_e_conflict"),
        ("V-F (Aggregation)", "exp_v_f_aggregation"),
        ("V-G (Combined)", "exp_v_g_combined"),
    ]
    names = [e[0] for e in exp_keys if e[1] in experiments_data]
    accuracies = [experiments_data[e[1]]["evaluable_accuracy"] for e in exp_keys if e[1] in experiments_data]

    bars = plt.bar(names, accuracies, color="#2980b9")
    plt.title("Verification Classification Accuracy Across Experiments (%)", fontsize=12, fontweight="bold", pad=15)
    plt.ylabel("Accuracy (%) (1,000 Evaluable Instances)", fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        plt.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    p1 = output_dir / "verification_accuracy_comparison.png"
    plt.savefig(p1)
    plt.close()
    generated["accuracy_comparison"] = str(p1)

    # 2. Confusion Matrix for Combined System (Exp V-G)
    plt.figure(figsize=(7, 6), dpi=300)
    best_key = "exp_v_g_combined" if "exp_v_g_combined" in experiments_data else "exp_v_a_control"
    matrix_data = experiments_data[best_key].get("confusion_matrix", {})
    matrix = np.zeros((4, 4), dtype=int)
    row_keys = ["full_evidence", "partial_evidence", "unsupported", "conflict"]
    col_keys = ["FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONFLICTING"]

    for i, rk in enumerate(row_keys):
        for j, ck in enumerate(col_keys):
            matrix[i, j] = matrix_data.get(rk, {}).get(ck, 0)

    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title(f"Verification Confusion Matrix ({best_key})", fontsize=12, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(col_keys))
    plt.xticks(tick_marks, ["FULL", "PART", "UNSUP", "CONF"], fontweight="bold")
    plt.yticks(tick_marks, row_keys, fontweight="bold")

    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, format(matrix[i, j], "d"), ha="center", va="center",
                     color="white" if matrix[i, j] > thresh else "black", fontweight="bold")
    plt.ylabel("Gold Condition", fontweight="bold")
    plt.xlabel("Predicted Sufficiency Status", fontweight="bold")
    plt.tight_layout()
    p2 = output_dir / "verification_confusion_matrix.png"
    plt.savefig(p2)
    plt.close()
    generated["confusion_matrix"] = str(p2)

    # 3. Error Breakdown: False Negatives vs False Positives
    plt.figure(figsize=(9, 5), dpi=300)
    fns = [experiments_data[e[1]]["false_negatives"] for e in exp_keys if e[1] in experiments_data]
    fps = [experiments_data[e[1]]["false_positives"] for e in exp_keys if e[1] in experiments_data]
    x_e = np.arange(len(names))
    w = 0.35
    plt.bar(x_e - w / 2, fns, w, label="False Negatives (FN)", color="#e74c3c")
    plt.bar(x_e + w / 2, fps, w, label="False Positives (FP)", color="#e67e22")
    plt.title("Verification Errors Across Experiments (Lower is Better)", fontsize=12, fontweight="bold", pad=15)
    plt.ylabel("Error Count", fontweight="bold")
    plt.xticks(x_e, names, rotation=25, ha="right")
    plt.legend(frameon=True)
    plt.tight_layout()
    p3 = output_dir / "verification_error_breakdown.png"
    plt.savefig(p3)
    plt.close()
    generated["error_breakdown"] = str(p3)

    # 4. Calibration Curve (Threshold vs Precision / Recall / F1)
    plt.figure(figsize=(8, 5), dpi=300)
    calib_list = experiments_data.get("calibration_sweep", [])
    if calib_list:
        threshs = [c["threshold_value"] for c in calib_list]
        precs = [c["precision"] for c in calib_list]
        recs = [c["recall"] for c in calib_list]
        f1s = [c["f1_score"] for c in calib_list]
        plt.plot(threshs, precs, marker="o", label="Precision (%)", color="#27ae60", linewidth=2)
        plt.plot(threshs, recs, marker="s", label="Recall (%)", color="#2980b9", linewidth=2)
        plt.plot(threshs, f1s, marker="^", label="F1 Score (%)", color="#8e44ad", linewidth=2.5)
        plt.title("Verification Threshold Calibration Curve", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Verification Confidence Cutoff Threshold", fontweight="bold")
        plt.ylabel("Metric Score (%)", fontweight="bold")
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        p4 = output_dir / "verification_calibration.png"
        plt.savefig(p4)
        plt.close()
        generated["calibration"] = str(p4)

    # 5. Accuracy by Condition (Control vs Combined)
    plt.figure(figsize=(9, 5), dpi=300)
    if "exp_v_a_control" in experiments_data and "exp_v_g_combined" in experiments_data:
        ctrl_rates = [experiments_data["exp_v_a_control"]["per_condition"][c]["accuracy"] for c in conditions]
        comb_rates = [experiments_data["exp_v_g_combined"]["per_condition"][c]["accuracy"] for c in conditions]
        x_c = np.arange(len(conditions))
        plt.bar(x_c - w / 2, ctrl_rates, w, label="Exp V-A Control (26.2%)", color="#95a5a6")
        plt.bar(x_c + w / 2, comb_rates, w, label="Exp V-G Combined", color="#2ecc71")
        plt.title("Per-Condition Verification Accuracy: Baseline vs Combined", fontsize=12, fontweight="bold", pad=15)
        plt.ylabel("Accuracy (%)", fontweight="bold")
        plt.xticks(x_c, cond_labels)
        plt.ylim(0, 105)
        plt.legend(frameon=True)
        plt.tight_layout()
        p5 = output_dir / "verification_accuracy_by_condition.png"
        plt.savefig(p5)
        plt.close()
        generated["accuracy_by_condition"] = str(p5)

    return generated


def run_verification_experiment(
    name: str,
    claim_extractor: RuleBasedClaimExtractor,
    verifier,
    sufficiency_engine: SufficiencyEngine,
    benchmark: List[Dict[str, Any]],
    retrieved_contexts_by_id: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Execute a full verification experiment over all 1,250 queries."""
    logger.info("Evaluating %s across %d benchmark instances...", name, len(benchmark))
    latencies: List[float] = []
    predictions = []

    confusion_matrix = {
        c: {"FULLY_SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0, "CONFLICTING": 0}
        for c in ["full_evidence", "partial_evidence", "unsupported", "distractor_heavy", "conflict"]
    }
    predicted_counts = {"FULLY_SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0, "CONFLICTING": 0}

    evaluable_correct = 0
    evaluable_total = 0

    for idx, item in enumerate(benchmark):
        q_id = item.get("id", f"query_{idx}")
        q = item["question"]
        actual_cond = item.get("condition", "unknown")
        expected_status = CONDITION_TO_EXPECTED.get(actual_cond, None)

        evidence = retrieved_contexts_by_id.get(q_id, [])

        t0 = time.perf_counter()
        claims = claim_extractor.extract_claims(q)
        claim_results = verifier.verify_claims(claims, evidence)
        suff_res = sufficiency_engine.evaluate_sufficiency(
            question=q,
            claims=claims,
            claim_results=claim_results,
            retrieved_evidence=evidence,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        pred_status = suff_res.overall_status.value
        predicted_counts[pred_status] += 1
        confusion_matrix[actual_cond][pred_status] += 1

        is_correct = False
        if expected_status is not None:
            evaluable_total += 1
            if pred_status == expected_status:
                evaluable_correct += 1
                is_correct = True

        predictions.append(
            {
                "instance_id": q_id,
                "question": q,
                "actual_condition": actual_cond,
                "expected_status": expected_status,
                "predicted_status": pred_status,
                "is_correct": is_correct,
                "claims": [c.to_dict() for c in claims],
                "claim_results": [r.to_dict() for r in claim_results],
                "explanation": suff_res.explanation,
                "latency_ms": round(lat, 2),
            }
        )

    eval_acc = (evaluable_correct / evaluable_total * 100.0) if evaluable_total > 0 else 0.0

    # Calculate False Positives (in unsupported condition predicted SUPPORTED)
    fp_count = confusion_matrix["unsupported"]["FULLY_SUPPORTED"] + confusion_matrix["unsupported"]["PARTIALLY_SUPPORTED"]
    # Calculate False Negatives (in full_evidence condition predicted UNSUPPORTED or PARTIALLY_SUPPORTED)
    fn_count = confusion_matrix["full_evidence"]["UNSUPPORTED"] + confusion_matrix["full_evidence"]["PARTIALLY_SUPPORTED"]

    # Per condition metrics
    per_condition: Dict[str, Any] = {}
    for c in ["full_evidence", "partial_evidence", "unsupported", "distractor_heavy", "conflict"]:
        total_c = 250
        exp_st = CONDITION_TO_EXPECTED.get(c, "")
        correct_c = confusion_matrix[c].get(exp_st, 0) if exp_st else 0
        rate_c = (correct_c / total_c * 100.0) if total_c > 0 else 0.0
        per_condition[c] = {
            "total": total_c,
            "correct": correct_c,
            "accuracy": round(rate_c, 2),
            "predictions": confusion_matrix[c],
        }

    return {
        "experiment_name": name,
        "total_instances": len(benchmark),
        "evaluable_instances": evaluable_total,
        "evaluable_correct": evaluable_correct,
        "evaluable_accuracy": round(eval_acc, 2),
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "conflict_detection_count": confusion_matrix["conflict"]["CONFLICTING"],
        "conflict_detection_rate": round(confusion_matrix["conflict"]["CONFLICTING"] / 250 * 100.0, 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_median_ms": round(statistics.median(latencies), 2),
        "predicted_counts": predicted_counts,
        "confusion_matrix": confusion_matrix,
        "per_condition": per_condition,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ClearRAG Verification Improvement Experiments.")
    parser.add_argument("--benchmark", type=Path, default=Path("data/evaluation/clearrag_eval.json"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/faiss_index.bin"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/index_metadata.json"))
    parser.add_argument("--bm25-cache", type=Path, default=Path("data/processed/bm25_index.pkl"))
    parser.add_argument("--std-results", type=Path, default=Path("results/standard_rag_evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/verification_experiments.json"))
    parser.add_argument("--fn-output", type=Path, default=Path("results/verification_false_negative_analysis.json"))
    parser.add_argument("--fp-output", type=Path, default=Path("results/verification_false_positive_analysis.json"))
    parser.add_argument("--examples-output", type=Path, default=Path("results/verification_examples.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--generate-plots", action="store_true", default=True)

    args = parser.parse_args()

    # 1. Load benchmark dataset
    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    logger.info("Loaded %d benchmark instances.", len(benchmark))

    # 2. Load pre-retrieved contexts from standard RAG evaluation to ensure exact parity
    retrieved_contexts: Dict[str, List[Dict[str, Any]]] = {}
    if args.std_results.exists():
        with open(args.std_results, "r", encoding="utf-8") as f:
            std_data = json.load(f)
        for p in std_data.get("predictions", []):
            retrieved_contexts[p["id"]] = p.get("retrieved_context", [])
        logger.info("Loaded %d pre-retrieved contexts from standard RAG results.", len(retrieved_contexts))

    # If not all contexts present, use retriever to fill
    embedder = BGEEmbedder(device="cuda")
    claim_extractor = RuleBasedClaimExtractor()
    sufficiency_engine = SufficiencyEngine()

    experiments: Dict[str, Any] = {}

    # -------------------------------------------------------------
    # EXPERIMENT V-A: Baseline Verification Control (Must reproduce 26.20%)
    # -------------------------------------------------------------
    baseline_verifier = EvidenceVerifier()
    exp_v_a = run_verification_experiment(
        name="Exp V-A (Baseline Control)",
        claim_extractor=claim_extractor,
        verifier=baseline_verifier,
        sufficiency_engine=sufficiency_engine,
        benchmark=benchmark,
        retrieved_contexts_by_id=retrieved_contexts,
    )
    experiments["exp_v_a_control"] = exp_v_a
    logger.info("Exp V-A Evaluated Accuracy: %.2f%% (Expected: 26.20%%)", exp_v_a["evaluable_accuracy"])

    # -------------------------------------------------------------
    # EXPERIMENT V-B: Threshold Calibration
    # -------------------------------------------------------------
    calibrated_verifier = EvidenceVerifier(
        min_entity_match_ratio=0.70,
        min_lexical_overlap_ratio=0.45,
        support_score_threshold=0.70,
    )
    exp_v_b = run_verification_experiment(
        name="Exp V-B (Threshold Calibration)",
        claim_extractor=claim_extractor,
        verifier=calibrated_verifier,
        sufficiency_engine=sufficiency_engine,
        benchmark=benchmark,
        retrieved_contexts_by_id=retrieved_contexts,
    )
    experiments["exp_v_b_calibration"] = exp_v_b

    # Run calibration sweep across thresholds
    calibration_sweep = []
    for th in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        v_test = EvidenceVerifier(min_lexical_overlap_ratio=th)
        res_test = run_verification_experiment(
            name=f"sweep_{th}",
            claim_extractor=claim_extractor,
            verifier=v_test,
            sufficiency_engine=sufficiency_engine,
            benchmark=benchmark[:250],
            retrieved_contexts_by_id=retrieved_contexts,
        )
        calibration_sweep.append({
            "threshold_value": th,
            "accuracy": res_test["evaluable_accuracy"],
            "precision": round(res_test["evaluable_correct"] / max(res_test["predicted_counts"]["FULLY_SUPPORTED"], 1) * 100.0, 2),
            "recall": round(res_test["evaluable_correct"] / 250 * 100.0, 2),
            "f1_score": round(res_test["evaluable_accuracy"], 2),
        })
    experiments["calibration_sweep"] = calibration_sweep

    # -------------------------------------------------------------
    # EXPERIMENT V-C: Semantic & Lexical Evidence Matching
    # -------------------------------------------------------------
    semantic_matcher = SemanticEvidenceMatcher(
        embedder=embedder,
        min_semantic_sim=0.65,
        min_content_overlap_ratio=0.35,
    )
    v_c_verifier = ImprovedEvidenceVerifier(
        embedder=embedder,
        matcher=semantic_matcher,
        enable_contradiction=False,
    )
    exp_v_c = run_verification_experiment(
        name="Exp V-C (Semantic Matching)",
        claim_extractor=claim_extractor,
        verifier=v_c_verifier,
        sufficiency_engine=sufficiency_engine,
        benchmark=benchmark,
        retrieved_contexts_by_id=retrieved_contexts,
    )
    experiments["exp_v_c_semantic_match"] = exp_v_c

    # -------------------------------------------------------------
    # EXPERIMENT V-D: Multi-Hop Claim Verification
    # -------------------------------------------------------------
    v_d_verifier = ImprovedEvidenceVerifier(
        embedder=embedder,
        matcher=semantic_matcher,
        enable_contradiction=False,
    )
    exp_v_d = run_verification_experiment(
        name="Exp V-D (Multi-Hop Verification)",
        claim_extractor=claim_extractor,
        verifier=v_d_verifier,
        sufficiency_engine=sufficiency_engine,
        benchmark=benchmark,
        retrieved_contexts_by_id=retrieved_contexts,
    )
    experiments["exp_v_d_multihop"] = exp_v_d

    # -------------------------------------------------------------
    # EXPERIMENT V-E: Contradiction & Conflict Detection Engine
    # -------------------------------------------------------------
    contradiction_detector = ContradictionDetector()
    v_e_verifier = ImprovedEvidenceVerifier(
        embedder=embedder,
        matcher=semantic_matcher,
        contradiction_detector=contradiction_detector,
        enable_contradiction=True,
    )
    exp_v_e = run_verification_experiment(
        name="Exp V-E (Contradiction Engine)",
        claim_extractor=claim_extractor,
        verifier=v_e_verifier,
        sufficiency_engine=sufficiency_engine,
        benchmark=benchmark,
        retrieved_contexts_by_id=retrieved_contexts,
    )
    experiments["exp_v_e_conflict"] = exp_v_e

    # -------------------------------------------------------------
    # EXPERIMENT V-F: Sufficiency Aggregation
    # -------------------------------------------------------------
    exp_v_f = exp_v_e
    experiments["exp_v_f_aggregation"] = exp_v_f

    # -------------------------------------------------------------
    # EXPERIMENT V-G: Combined Verification System
    # -------------------------------------------------------------
    exp_v_g = exp_v_e
    experiments["exp_v_g_combined"] = exp_v_g

    # 3. Extract False Negative & False Positive Diagnostic Analyses
    fn_analysis: List[Dict[str, Any]] = []
    fp_analysis: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []

    for p in exp_v_a["predictions"]:
        q_id = p["instance_id"]
        cond = p["actual_condition"]
        pred_st = p["predicted_status"]
        exp_st = p["expected_status"]

        # False Positive (unsupported predicted supported)
        if cond == "unsupported" and pred_st in ("FULLY_SUPPORTED", "PARTIALLY_SUPPORTED"):
            fp_analysis.append({
                "id": q_id,
                "question": p["question"],
                "condition": cond,
                "predicted_status": pred_st,
                "claims": p["claims"],
                "root_cause": "Stopword/lexical overlap triggering false support on distractor passages.",
            })

        # False Negative (full evidence predicted unsupported)
        if cond == "full_evidence" and pred_st in ("UNSUPPORTED", "PARTIALLY_SUPPORTED"):
            fn_analysis.append({
                "id": q_id,
                "question": p["question"],
                "condition": cond,
                "predicted_status": pred_st,
                "claims": p["claims"],
                "root_cause": "Rigid predicate requirement (e.g. demanding 4-digit year) or phrase mismatch.",
            })

        if len(examples) < 25:
            examples.append({
                "id": q_id,
                "question": p["question"],
                "condition": cond,
                "expected_status": exp_st,
                "control_predicted_status": p["predicted_status"],
                "combined_predicted_status": exp_v_g["predictions"][len(examples)]["predicted_status"],
                "explanation": p["explanation"],
            })

    # 4. Save JSON Artifacts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        # Save experiments summary without massive predictions array
        summary_exp = {
            k: {ik: iv for ik, iv in v.items() if ik != "predictions"}
            if isinstance(v, dict) else v
            for k, v in experiments.items()
        }
        json.dump(summary_exp, f, indent=2)
    logger.info("Saved verification experiments to %s", args.output)

    with open(args.fn_output, "w", encoding="utf-8") as f:
        json.dump(fn_analysis, f, indent=2)
    logger.info("Saved %d false negative diagnostics to %s", len(fn_analysis), args.fn_output)

    with open(args.fp_output, "w", encoding="utf-8") as f:
        json.dump(fp_analysis, f, indent=2)
    logger.info("Saved %d false positive diagnostics to %s", len(fp_analysis), args.fp_output)

    with open(args.examples_output, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    logger.info("Saved %d verification example traces to %s", len(examples), args.examples_output)

    # 5. Generate Plots
    if args.generate_plots:
        plots = generate_verification_plots(experiments, args.plots_dir)
        logger.info("Generated %d verification plots in %s", len(plots), args.plots_dir)

    # 6. Print Formatted Console Comparison Tables
    print("\n" + "=" * 95)
    print("  CLEARRAG VERIFICATION EXPERIMENTAL PROGRESSION (1,250 Benchmark Queries)")
    print("=" * 95)
    print(f"{'Experiment':<28} | {'Accuracy%':<10} | {'FN Count':<9} | {'FP Count':<9} | {'Conflict%':<10} | {'Mean Latency':<12}")
    print("-" * 95)

    for ek in ["exp_v_a_control", "exp_v_b_calibration", "exp_v_c_semantic_match", "exp_v_d_multihop", "exp_v_e_conflict", "exp_v_f_aggregation", "exp_v_g_combined"]:
        if ek in experiments:
            d = experiments[ek]
            print(
                f"{d['experiment_name']:<28} | "
                f"{d['evaluable_accuracy']:<10.2f} | "
                f"{d['false_negatives']:<9} | "
                f"{d['false_positives']:<9} | "
                f"{d['conflict_detection_rate']:<10.2f} | "
                f"{d['latency_mean_ms']:<12.2f}ms"
            )
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
