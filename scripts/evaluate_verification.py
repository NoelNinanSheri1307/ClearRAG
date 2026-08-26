"""Benchmark evaluation runner for ClearRAG Evidence Verification layer."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import Retriever
from src.verification.claim_extractor import RuleBasedClaimExtractor
from src.verification.evidence_verifier import EvidenceVerifier
from src.verification.models import SufficiencyStatus
from src.verification.sufficiency import SufficiencyEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def map_condition_to_expected_status(condition: str) -> Optional[str]:
    """Map benchmark condition label to expected verification sufficiency status string.
    
    Semantic Mapping Documentation:
    --------------------------------
    1. 'full_evidence' -> FULLY_SUPPORTED
       The benchmark query contains complete gold evidence.
    2. 'partial_evidence' -> PARTIALLY_SUPPORTED
       The benchmark query has had one supporting evidence document removed.
    3. 'unsupported' -> UNSUPPORTED
       The benchmark query has had all supporting evidence documents removed.
    4. 'conflict' -> CONFLICTING
       The benchmark query contains synthetic numeric/date perturbations.
    5. 'distractor_heavy' -> None (Ambiguous / Dependent on Retrieval)
       Distractor-heavy queries retain ground-truth evidence alongside distractors.
       If Top-K retrieval succeeds in retrieving gold evidence, prediction is FULLY_SUPPORTED.
       If Top-K retrieval fails, prediction is UNSUPPORTED.
       Therefore, forcing distractor_heavy into UNSUPPORTED is methodologically ambiguous.
    """
    c = condition.lower()
    if c == "full_evidence":
        return SufficiencyStatus.FULLY_SUPPORTED.value
    elif c == "partial_evidence":
        return SufficiencyStatus.PARTIALLY_SUPPORTED.value
    elif c == "unsupported":
        return SufficiencyStatus.UNSUPPORTED.value
    elif c == "conflict":
        return SufficiencyStatus.CONFLICTING.value
    elif c == "distractor_heavy":
        # Handled separately in raw distribution reporting
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate ClearRAG evidence verification layer on benchmark.")
    parser.add_argument("--eval_path", type=str, default="data/evaluation/clearrag_eval.json", help="Path to evaluation benchmark")
    parser.add_argument("--output_path", type=str, default="results/verification_evaluation.json", help="Path to save output results")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k evidence chunks to retrieve")
    parser.add_argument("--index_path", type=str, default="data/processed/faiss_index.bin", help="FAISS index path")
    parser.add_argument("--metadata_path", type=str, default="data/processed/index_metadata.json", help="Index metadata path")
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        logger.error(f"Evaluation benchmark file not found at '{eval_path}'")
        sys.exit(1)

    logger.info(f"Loading benchmark dataset from '{eval_path}'...")
    with open(eval_path, "r", encoding="utf-8") as f:
        benchmark_instances = json.load(f)

    logger.info(f"Loaded {len(benchmark_instances)} evaluation queries.")

    retriever = Retriever.from_saved_index(
        Path(args.index_path),
        Path(args.metadata_path),
        default_top_k=args.top_k,
    )

    claim_extractor = RuleBasedClaimExtractor()
    evidence_verifier = EvidenceVerifier()
    sufficiency_engine = SufficiencyEngine()

    start_time = time.time()
    results = []
    condition_counts = {}
    predicted_counts = {
        SufficiencyStatus.FULLY_SUPPORTED.value: 0,
        SufficiencyStatus.PARTIALLY_SUPPORTED.value: 0,
        SufficiencyStatus.UNSUPPORTED.value: 0,
        SufficiencyStatus.CONFLICTING.value: 0,
    }

    conditions = ["full_evidence", "partial_evidence", "unsupported", "distractor_heavy", "conflict"]
    statuses = [
        SufficiencyStatus.FULLY_SUPPORTED.value,
        SufficiencyStatus.PARTIALLY_SUPPORTED.value,
        SufficiencyStatus.UNSUPPORTED.value,
        SufficiencyStatus.CONFLICTING.value,
    ]

    confusion_matrix = {c: {s: 0 for s in statuses} for c in conditions}
    evaluable_correct = 0
    evaluable_total = 0

    print("=================================================================")
    print(f"EVALUATING EVIDENCE VERIFICATION LAYER ({len(benchmark_instances)} QUERIES)")
    print("=================================================================")

    for idx, item in enumerate(benchmark_instances, start=1):
        q_id = item.get("id", f"query_{idx}")
        question = item["question"]
        actual_condition = item["condition"]

        # STRICT UNLEAKED INFERENCE
        retrieved_evidence = retriever.retrieve(question, top_k=args.top_k)
        claims = claim_extractor.extract_claims(question)
        claim_results = [evidence_verifier.verify_claim(claim, retrieved_evidence) for claim in claims]

        v_result = sufficiency_engine.evaluate_sufficiency(
            question=question,
            claims=claims,
            claim_results=claim_results,
            retrieved_evidence=retrieved_evidence,
        )

        predicted_status = v_result.overall_status.value
        expected_status = map_condition_to_expected_status(actual_condition)

        is_correct = None
        if expected_status is not None:
            evaluable_total += 1
            is_correct = (predicted_status == expected_status)
            if is_correct:
                evaluable_correct += 1

        confusion_matrix[actual_condition][predicted_status] += 1
        predicted_counts[predicted_status] = predicted_counts.get(predicted_status, 0) + 1
        condition_counts[actual_condition] = condition_counts.get(actual_condition, 0) + 1

        record = {
            "instance_id": q_id,
            "question": question,
            "actual_condition": actual_condition,
            "expected_status": expected_status,
            "predicted_status": predicted_status,
            "is_correct": is_correct,
            "claims": [c.to_dict() for c in claims],
            "claim_results": [r.to_dict() for r in claim_results],
            "explanation": v_result.explanation,
        }
        results.append(record)

        if idx % 100 == 0 or idx == len(benchmark_instances):
            elapsed = time.time() - start_time
            acc_str = f"{(evaluable_correct / evaluable_total * 100):.2f}%" if evaluable_total > 0 else "N/A"
            print(f"[{idx}/{len(benchmark_instances)}] Elapsed: {elapsed:.1f}s | Evaluable Accuracy: {acc_str}")

    total_duration = time.time() - start_time
    evaluable_accuracy = (evaluable_correct / evaluable_total * 100) if evaluable_total > 0 else 0.0

    output_data = {
        "metadata": {
            "total_instances": len(benchmark_instances),
            "total_duration_seconds": round(total_duration, 2),
            "queries_per_second": round(len(benchmark_instances) / total_duration, 2),
            "evaluable_instances": evaluable_total,
            "evaluable_correct": evaluable_correct,
            "evaluable_classification_accuracy": round(evaluable_accuracy, 2),
        },
        "condition_counts": condition_counts,
        "predicted_counts": predicted_counts,
        "confusion_matrix": confusion_matrix,
        "predictions": results,
    }

    temp_path = output_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    if output_path.exists():
        output_path.unlink()
    temp_path.rename(output_path)

    print("\n=================================================================")
    print("VERIFICATION EVALUATION SUMMARY")
    print("=================================================================")
    print(f"Total Evaluated Instances     : {len(benchmark_instances)}")
    print(f"Evaluation Duration           : {total_duration:.2f} s")
    print(f"Evaluable Accuracy (4 conds)  : {evaluable_accuracy:.2f}% ({evaluable_correct}/{evaluable_total})")

    print("\nRaw Prediction Distribution Matrix:")
    header = f"{'Actual Condition':<18} | {'FULLY_SUP':<10} | {'PARTIAL_SUP':<11} | {'UNSUPPORTED':<11} | {'CONFLICTING':<11}"
    print(header)
    print("-" * len(header))
    for c in conditions:
        counts = confusion_matrix[c]
        print(f"{c:<18} | {counts['FULLY_SUPPORTED']:<10} | {counts['PARTIALLY_SUPPORTED']:<11} | {counts['UNSUPPORTED']:<11} | {counts['CONFLICTING']:<11}")
    print("=================================================================")
    print(f"Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
