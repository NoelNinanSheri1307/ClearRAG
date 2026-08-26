"""ClearRAG Evaluation Dataset Validator.

Validates data/evaluation/clearrag_eval.json against structural schemas,
HotpotQA source traceability, and condition-specific integrity rules.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

BENCHMARK_PATH = Path("data/evaluation/clearrag_eval.json")
SOURCE_DATA_PATH = Path("data/raw/hotpotqa/hotpot_dev_distractor_v1.json")

EXPECTED_CONDITIONS = {
    "full_evidence": 250,
    "partial_evidence": 250,
    "unsupported": 250,
    "distractor_heavy": 250,
    "conflict": 250,
}

EXPECTED_BEHAVIORS = {
    "full_evidence": "answer",
    "partial_evidence": "abstain_or_qualify",
    "unsupported": "abstain",
    "distractor_heavy": "answer",
    "conflict": "conflict_detected_or_abstain",
}


def validate_schema(instance: Dict[str, Any], idx: int) -> Tuple[bool, str]:
    """Validate top-level schema fields and types."""
    required_fields = [
        "id",
        "source_dataset",
        "source_id",
        "condition",
        "question",
        "context",
        "ground_truth",
        "original_supporting_facts",
        "retained_supporting_facts",
        "removed_supporting_facts",
        "expected_behavior",
        "metadata",
    ]

    for field in required_fields:
        if field not in instance:
            return False, f"Instance {idx} ({instance.get('id', 'unknown')}): Missing field '{field}'"

    if not isinstance(instance["context"], list) or not instance["context"]:
        return False, f"Instance {idx}: 'context' must be a non-empty list"

    for doc_idx, doc in enumerate(instance["context"]):
        if not isinstance(doc, dict):
            return False, f"Instance {idx}, doc {doc_idx}: Document must be a dictionary"
        for k in ["title", "sentences", "source_type"]:
            if k not in doc:
                return False, f"Instance {idx}, doc {doc_idx}: Missing '{k}' in context doc"
        if not isinstance(doc["sentences"], list):
            return False, f"Instance {idx}, doc {doc_idx}: 'sentences' must be a list"
        if doc["source_type"] not in ["original", "synthetic_conflict"]:
            return False, f"Instance {idx}, doc {doc_idx}: Invalid source_type '{doc['source_type']}'"

    return True, "OK"


def validate_condition_integrity(
    instance: Dict[str, Any], raw_item: Dict[str, Any], idx: int
) -> Tuple[bool, str]:
    """Validate condition-specific logical integrity."""
    cond = instance["condition"]
    expected_bh = EXPECTED_BEHAVIORS.get(cond)

    if instance["expected_behavior"] != expected_bh:
        return False, (
            f"Instance {idx}: Expected behavior mismatch. "
            f"Got '{instance['expected_behavior']}', expected '{expected_bh}'"
        )

    # 1. Full Evidence
    if cond == "full_evidence":
        if len(instance["retained_supporting_facts"]) != len(instance["original_supporting_facts"]):
            return False, f"Instance {idx}: Full evidence retained facts count mismatch"
        if len(instance["removed_supporting_facts"]) != 0:
            return False, f"Instance {idx}: Full evidence must have 0 removed facts"

    # 2. Partial Evidence
    elif cond == "partial_evidence":
        if len(instance["removed_supporting_facts"]) < 1:
            return False, f"Instance {idx}: Partial evidence must have at least 1 removed fact"
        if len(instance["retained_supporting_facts"]) < 1:
            return False, f"Instance {idx}: Partial evidence must have at least 1 retained fact"
        if len(instance["removed_supporting_facts"]) + len(instance["retained_supporting_facts"]) != len(instance["original_supporting_facts"]):
            return False, f"Instance {idx}: Partial evidence fact count sum mismatch"

        # Check removed sentence text is absent from context
        removed_info = instance.get("metadata", {}).get("removed_fact_info", {})
        removed_text = removed_info.get("sentence_text")
        removed_title = removed_info.get("title")

        target_doc = next((d for d in instance["context"] if d["title"] == removed_title), None)
        if not target_doc:
            return False, f"Instance {idx}: Target document '{removed_title}' missing in partial context"
        if removed_text in target_doc["sentences"]:
            return False, f"Instance {idx}: Removed sentence is still present in partial context"

    # 3. Unsupported
    elif cond == "unsupported":
        if len(instance["retained_supporting_facts"]) != 0:
            return False, f"Instance {idx}: Unsupported must have 0 retained supporting facts"
        if len(instance["removed_supporting_facts"]) != len(instance["original_supporting_facts"]):
            return False, f"Instance {idx}: Unsupported removed facts count mismatch"

        # Check no original supporting doc titles are present in context
        orig_titles = {f["title"] for f in instance["original_supporting_facts"]}
        ctx_titles = {d["title"] for d in instance["context"]}
        overlap = orig_titles.intersection(ctx_titles)
        if overlap:
            return False, f"Instance {idx}: Unsupported context contains supporting docs: {overlap}"

    # 4. Distractor Heavy
    elif cond == "distractor_heavy":
        if len(instance["context"]) != 10:
            return False, f"Instance {idx}: Distractor heavy must have exactly 10 context docs"
        supporting_titles = {f["title"] for f in instance["original_supporting_facts"]}
        if len(supporting_titles) != 2:
            return False, f"Instance {idx}: Distractor heavy must have exactly 2 supporting docs"
        distractor_count = sum(1 for d in instance["context"] if d["title"] not in supporting_titles)
        if distractor_count != 8:
            return False, f"Instance {idx}: Distractor heavy must have exactly 8 distractor docs"

    # 5. Conflict
    elif cond == "conflict":
        meta = instance.get("metadata", {})
        if not meta.get("is_synthetic_conflict"):
            return False, f"Instance {idx}: Conflict metadata missing 'is_synthetic_conflict' flag"
        orig_s = meta.get("original_conflict_sentence")
        synth_s = meta.get("synthetic_conflict_sentence")
        if not orig_s or not synth_s:
            return False, f"Instance {idx}: Conflict metadata missing original or synthetic sentence"
        if orig_s == synth_s:
            return False, f"Instance {idx}: Original and synthetic conflict sentences are identical"
        synth_docs = [d for d in instance["context"] if d["source_type"] == "synthetic_conflict"]
        if len(synth_docs) != 1:
            return False, f"Instance {idx}: Conflict context must have exactly 1 synthetic conflict passage"

    return True, "OK"


def validate_evaluation_dataset():
    """Run full validation suite on the generated evaluation dataset."""
    print("=" * 55)
    print("ClearRAG Evaluation Dataset Validation")
    print("=" * 55)

    if not BENCHMARK_PATH.exists():
        print(f"\n[FAIL] Benchmark file not found at {BENCHMARK_PATH}")
        return False

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    print(f"\nTotal instances: {len(benchmark):,}\n")

    # Count by condition
    condition_counts = {c: 0 for c in EXPECTED_CONDITIONS}
    for item in benchmark:
        cond = item.get("condition")
        if cond in condition_counts:
            condition_counts[cond] += 1

    for cond, target in EXPECTED_CONDITIONS.items():
        formatted_name = cond.replace("_", " ").title()
        print(f"{formatted_name}:")
        print(f"    {condition_counts[cond]} (Target: {target})")

    # Load source HotpotQA for traceability check
    source_map: Dict[str, Dict[str, Any]] = {}
    if SOURCE_DATA_PATH.exists():
        with open(SOURCE_DATA_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        source_map = {item["_id"]: item for item in raw_data}

    # Run check suites
    schema_pass = True
    traceability_pass = True
    condition_passes = {c: True for c in EXPECTED_CONDITIONS}
    unique_ids: Set[str] = set()

    for idx, inst in enumerate(benchmark):
        # Unique ID check
        inst_id = inst.get("id")
        if inst_id in unique_ids:
            print(f"[FAIL] Duplicate ID detected: {inst_id}")
            schema_pass = False
        unique_ids.add(inst_id)

        # Schema check
        valid_schema, schema_err = validate_schema(inst, idx)
        if not valid_schema:
            print(f"[FAIL] Schema error: {schema_err}")
            schema_pass = False

        # Traceability check
        source_id = inst.get("source_id")
        if source_id not in source_map:
            print(f"[FAIL] Source ID '{source_id}' not found in HotpotQA source")
            traceability_pass = False
        else:
            raw_item = source_map[source_id]
            if inst.get("question") != raw_item.get("question"):
                print(f"[FAIL] Question text does not match source for ID {source_id}")
                traceability_pass = False
            if inst.get("ground_truth") != raw_item.get("answer"):
                print(f"[FAIL] Ground truth does not match source answer for ID {source_id}")
                traceability_pass = False

            # Condition integrity check
            cond = inst.get("condition")
            if cond in condition_passes:
                valid_cond, cond_err = validate_condition_integrity(inst, raw_item, idx)
                if not valid_cond:
                    print(f"[FAIL] Condition integrity error: {cond_err}")
                    condition_passes[cond] = False

    # Check quotas
    quota_pass = all(condition_counts[c] == EXPECTED_CONDITIONS[c] for c in EXPECTED_CONDITIONS)

    print("\nValidation:")
    print(f"    Schema:                  {'PASS' if schema_pass else 'FAIL'}")
    print(f"    Source traceability:     {'PASS' if traceability_pass else 'FAIL'}")
    print(f"    Condition quotas:        {'PASS' if quota_pass else 'FAIL'}")
    print(f"    Full evidence integrity: {'PASS' if condition_passes['full_evidence'] else 'FAIL'}")
    print(f"    Partial evidence integ:  {'PASS' if condition_passes['partial_evidence'] else 'FAIL'}")
    print(f"    Unsupported integrity:   {'PASS' if condition_passes['unsupported'] else 'FAIL'}")
    print(f"    Distractor integrity:    {'PASS' if condition_passes['distractor_heavy'] else 'FAIL'}")
    print(f"    Conflict integrity:      {'PASS' if condition_passes['conflict'] else 'FAIL'}")

    overall = (
        schema_pass
        and traceability_pass
        and quota_pass
        and all(condition_passes.values())
    )

    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    print("=" * 55)
    return overall


if __name__ == "__main__":
    validate_evaluation_dataset()
