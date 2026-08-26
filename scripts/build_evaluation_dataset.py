"""ClearRAG Evaluation Dataset Generator.

Generates a controlled, reproducible benchmark derived from HotpotQA
across five evaluation conditions:
1. Full Evidence (answerable)
2. Partial Evidence (abstain_or_qualify)
3. Unsupported / No Evidence (abstain)
4. Distractor Heavy (answerable with high noise)
5. Conflict (conflict_detected_or_abstain)

The original source data is treated as read-only.
Output dataset path: data/evaluation/clearrag_eval.json
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Configuration
SOURCE_DATA_PATH = Path("data/raw/hotpotqa/hotpot_dev_distractor_v1.json")
OUTPUT_DIR = Path("data/evaluation")
OUTPUT_FILE_PATH = OUTPUT_DIR / "clearrag_eval.json"
RANDOM_SEED = 42
TARGET_PER_CONDITION = 250

CONDITIONS = [
    "full_evidence",
    "partial_evidence",
    "unsupported",
    "distractor_heavy",
    "conflict",
]

NUM_PATTERN = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+)\b")


def perturb_numeric_sentence(sentence: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """Perturb a numeric value in a sentence deterministically.

    Preserves sentence structure, subject, and property while modifying
    the numerical value by +1.

    Args:
        sentence: Factual sentence containing numeric facts.

    Returns:
        Tuple of (synthetic_sentence, perturbation_metadata) or (None, None).
    """
    matches = list(NUM_PATTERN.finditer(sentence))
    if not matches:
        return None, None

    # Priority heuristic: Years (1000-2050) > multi-digit numbers > single digits
    def priority(m: re.Match) -> Tuple[int, int]:
        raw = m.group(0).replace(",", "")
        val = int(raw)
        if 1000 <= val <= 2050:
            return (3, len(raw))
        elif val >= 10:
            return (2, len(raw))
        else:
            return (1, len(raw))

    best_match = max(matches, key=priority)
    raw_str = best_match.group(0)
    clean_val = int(raw_str.replace(",", ""))

    # Shift value
    new_val = clean_val + 1
    new_str = f"{new_val:,}" if "," in raw_str else str(new_val)

    start, end = best_match.span()
    synth_sentence = sentence[:start] + new_str + sentence[end:]

    if synth_sentence == sentence:
        return None, None

    meta = {
        "original_value": raw_str,
        "perturbed_value": new_str,
    }
    return synth_sentence, meta


def format_supporting_facts(raw_facts: List[Any]) -> List[Dict[str, Any]]:
    """Convert raw HotpotQA [title, sentence_index] list into standardized dicts."""
    return [
        {"title": str(fact[0]), "sentence_index": int(fact[1])}
        for fact in raw_facts
    ]


def build_full_evidence_instance(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Construct a Full Evidence evaluation instance."""
    source_id = item["_id"]
    question = item["question"]
    answer = item["answer"]
    raw_context = item.get("context", [])
    raw_facts = item.get("supporting_facts", [])

    if not question or not answer or not raw_context or not raw_facts:
        return None

    context_docs = [
        {
            "title": str(doc[0]),
            "sentences": [str(s) for s in doc[1]],
            "source_type": "original",
        }
        for doc in raw_context
    ]

    supporting_facts = format_supporting_facts(raw_facts)

    return {
        "id": f"hotpot_{source_id}_full_evidence",
        "source_dataset": "HotpotQA",
        "source_id": source_id,
        "condition": "full_evidence",
        "question": question,
        "context": context_docs,
        "ground_truth": answer,
        "original_supporting_facts": supporting_facts,
        "retained_supporting_facts": supporting_facts,
        "removed_supporting_facts": [],
        "expected_behavior": "answer",
        "metadata": {
            "question_type": item.get("type"),
            "difficulty_level": item.get("level"),
            "total_context_documents": len(context_docs),
            "total_supporting_facts": len(supporting_facts),
        },
    }


def build_partial_evidence_instance(
    item: Dict[str, Any], rng: random.Random
) -> Optional[Dict[str, Any]]:
    """Construct a Partial Evidence instance by removing at least one supporting fact."""
    source_id = item["_id"]
    question = item["question"]
    answer = item["answer"]
    raw_context = item.get("context", [])
    raw_facts = item.get("supporting_facts", [])

    if not question or not answer or not raw_context or len(raw_facts) < 2:
        return None

    # Deterministically select one supporting fact to remove
    # (preserving >=1 remaining supporting fact)
    original_facts = format_supporting_facts(raw_facts)
    fact_to_remove = original_facts[-1]  # deterministic choice

    removed_title = fact_to_remove["title"]
    removed_s_idx = fact_to_remove["sentence_index"]

    removed_sentence_text = None
    context_docs = []

    for doc in raw_context:
        title = str(doc[0])
        sentences = [str(s) for s in doc[1]]

        if title == removed_title and removed_s_idx < len(sentences):
            removed_sentence_text = sentences[removed_s_idx]
            new_sentences = [
                s for i, s in enumerate(sentences) if i != removed_s_idx
            ]
        else:
            new_sentences = list(sentences)

        context_docs.append(
            {
                "title": title,
                "sentences": new_sentences,
                "source_type": "original",
            }
        )

    if not removed_sentence_text:
        return None

    retained_facts = [
        f for f in original_facts
        if not (f["title"] == removed_title and f["sentence_index"] == removed_s_idx)
    ]
    removed_facts = [fact_to_remove]

    if not retained_facts or not removed_facts:
        return None

    return {
        "id": f"hotpot_{source_id}_partial_evidence",
        "source_dataset": "HotpotQA",
        "source_id": source_id,
        "condition": "partial_evidence",
        "question": question,
        "context": context_docs,
        "ground_truth": answer,
        "original_supporting_facts": original_facts,
        "retained_supporting_facts": retained_facts,
        "removed_supporting_facts": removed_facts,
        "expected_behavior": "abstain_or_qualify",
        "metadata": {
            "question_type": item.get("type"),
            "difficulty_level": item.get("level"),
            "removal_strategy": "remove_single_supporting_fact",
            "removed_fact_info": {
                "title": removed_title,
                "original_sentence_index": removed_s_idx,
                "sentence_text": removed_sentence_text,
            },
            "total_context_documents": len(context_docs),
        },
    }


def build_unsupported_instance(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Construct an Unsupported instance by removing all gold supporting documents."""
    source_id = item["_id"]
    question = item["question"]
    answer = item["answer"]
    raw_context = item.get("context", [])
    raw_facts = item.get("supporting_facts", [])

    if not question or not answer or not raw_context or not raw_facts:
        return None

    supporting_doc_titles = {str(f[0]) for f in raw_facts}
    distractor_docs = [
        {
            "title": str(doc[0]),
            "sentences": [str(s) for s in doc[1]],
            "source_type": "original",
        }
        for doc in raw_context
        if str(doc[0]) not in supporting_doc_titles
    ]

    # Require non-empty distractor context (standard HotpotQA has 8 distractors)
    if not distractor_docs:
        return None

    original_facts = format_supporting_facts(raw_facts)

    return {
        "id": f"hotpot_{source_id}_unsupported",
        "source_dataset": "HotpotQA",
        "source_id": source_id,
        "condition": "unsupported",
        "question": question,
        "context": distractor_docs,
        "ground_truth": answer,
        "original_supporting_facts": original_facts,
        "retained_supporting_facts": [],
        "removed_supporting_facts": original_facts,
        "expected_behavior": "abstain",
        "metadata": {
            "question_type": item.get("type"),
            "difficulty_level": item.get("level"),
            "removal_strategy": "remove_all_supporting_documents",
            "retained_distractor_count": len(distractor_docs),
            "removed_supporting_doc_count": len(supporting_doc_titles),
        },
    }


def build_distractor_heavy_instance(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Construct a Distractor Heavy instance preserving standard 2-support / 8-distractor setup."""
    source_id = item["_id"]
    question = item["question"]
    answer = item["answer"]
    raw_context = item.get("context", [])
    raw_facts = item.get("supporting_facts", [])

    if not question or not answer or len(raw_context) != 10:
        return None

    supporting_doc_titles = {str(f[0]) for f in raw_facts}
    if len(supporting_doc_titles) != 2:
        return None

    distractor_count = sum(1 for doc in raw_context if str(doc[0]) not in supporting_doc_titles)
    if distractor_count != 8:
        return None

    context_docs = [
        {
            "title": str(doc[0]),
            "sentences": [str(s) for s in doc[1]],
            "source_type": "original",
        }
        for doc in raw_context
    ]

    supporting_facts = format_supporting_facts(raw_facts)

    return {
        "id": f"hotpot_{source_id}_distractor_heavy",
        "source_dataset": "HotpotQA",
        "source_id": source_id,
        "condition": "distractor_heavy",
        "question": question,
        "context": context_docs,
        "ground_truth": answer,
        "original_supporting_facts": supporting_facts,
        "retained_supporting_facts": supporting_facts,
        "removed_supporting_facts": [],
        "expected_behavior": "answer",
        "metadata": {
            "question_type": item.get("type"),
            "difficulty_level": item.get("level"),
            "supporting_document_count": len(supporting_doc_titles),
            "distractor_document_count": distractor_count,
            "total_document_count": len(context_docs),
        },
    }


def build_conflict_instance(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Construct a Conflict instance with a synthetic perturbed contradiction passage."""
    source_id = item["_id"]
    question = item["question"]
    answer = item["answer"]
    raw_context = item.get("context", [])
    raw_facts = item.get("supporting_facts", [])

    if not question or not answer or not raw_context or not raw_facts:
        return None

    ctx_map = {str(doc[0]): [str(s) for s in doc[1]] for doc in raw_context}

    # Find a supporting fact that can be safely perturbed numerically
    conflict_found = False
    original_conflict_sentence = None
    synthetic_conflict_sentence = None
    conflict_title = None
    conflict_s_idx = None
    perturb_meta = None

    for title, s_idx in raw_facts:
        title_str = str(title)
        s_idx_int = int(s_idx)
        if title_str in ctx_map and s_idx_int < len(ctx_map[title_str]):
            orig_sent = ctx_map[title_str][s_idx_int]
            synth_sent, p_meta = perturb_numeric_sentence(orig_sent)
            if synth_sent and synth_sent != orig_sent:
                conflict_found = True
                original_conflict_sentence = orig_sent
                synthetic_conflict_sentence = synth_sent
                conflict_title = title_str
                conflict_s_idx = s_idx_int
                perturb_meta = p_meta
                break

    if not conflict_found:
        return None

    # Build context: original documents + synthetic conflict passage
    context_docs = [
        {
            "title": str(doc[0]),
            "sentences": [str(s) for s in doc[1]],
            "source_type": "original",
        }
        for doc in raw_context
    ]

    # Append explicitly marked synthetic conflicting evidence
    context_docs.append(
        {
            "title": f"{conflict_title} (Conflicting Report)",
            "sentences": [synthetic_conflict_sentence],
            "source_type": "synthetic_conflict",
        }
    )

    supporting_facts = format_supporting_facts(raw_facts)

    return {
        "id": f"hotpot_{source_id}_conflict",
        "source_dataset": "HotpotQA",
        "source_id": source_id,
        "condition": "conflict",
        "question": question,
        "context": context_docs,
        "ground_truth": answer,
        "original_supporting_facts": supporting_facts,
        "retained_supporting_facts": supporting_facts,
        "removed_supporting_facts": [],
        "expected_behavior": "conflict_detected_or_abstain",
        "metadata": {
            "question_type": item.get("type"),
            "difficulty_level": item.get("level"),
            "is_synthetic_conflict": True,
            "conflict_generation_method": "deterministic_numeric_perturbation",
            "original_conflict_sentence": original_conflict_sentence,
            "synthetic_conflict_sentence": synthetic_conflict_sentence,
            "conflict_source": {
                "title": conflict_title,
                "sentence_index": conflict_s_idx,
            },
            "perturbation_details": perturb_meta,
            "total_context_documents": len(context_docs),
        },
    }


def validate_instance(inst: Dict[str, Any]) -> Tuple[bool, str]:
    """Perform validation checks on an individual evaluation instance."""
    # General checks
    if not inst.get("id"):
        return False, "Missing instance ID"
    if not inst.get("source_id"):
        return False, "Missing source ID"
    if not inst.get("question", "").strip():
        return False, "Empty question"
    if not inst.get("ground_truth", "").strip():
        return False, "Empty ground truth"
    if not inst.get("context"):
        return False, "Empty context"

    condition = inst.get("condition")
    if condition not in CONDITIONS:
        return False, f"Invalid condition: {condition}"

    # Condition specific checks
    if condition == "full_evidence":
        if len(inst["retained_supporting_facts"]) != len(inst["original_supporting_facts"]):
            return False, "Full evidence retained count mismatch"
        if len(inst["removed_supporting_facts"]) != 0:
            return False, "Full evidence has removed facts"

    elif condition == "partial_evidence":
        if len(inst["removed_supporting_facts"]) == 0:
            return False, "Partial evidence has no removed facts"
        if len(inst["retained_supporting_facts"]) == 0:
            return False, "Partial evidence has no retained facts"
        # Ensure removed sentence text is absent from context
        removed_fact = inst["removed_supporting_facts"][0]
        doc = next((d for d in inst["context"] if d["title"] == removed_fact["title"]), None)
        if not doc:
            return False, "Partial evidence target document missing"
        removed_text = inst["metadata"]["removed_fact_info"]["sentence_text"]
        if removed_text in doc["sentences"]:
            return False, "Partial evidence removed sentence still present in context"

    elif condition == "unsupported":
        if len(inst["retained_supporting_facts"]) != 0:
            return False, "Unsupported has retained supporting facts"
        if len(inst["removed_supporting_facts"]) != len(inst["original_supporting_facts"]):
            return False, "Unsupported removed facts count mismatch"
        # Ensure no original supporting docs remain in context
        orig_titles = {f["title"] for f in inst["original_supporting_facts"]}
        ctx_titles = {d["title"] for d in inst["context"]}
        if orig_titles.intersection(ctx_titles):
            return False, "Unsupported context contains supporting document titles"

    elif condition == "distractor_heavy":
        if len(inst["context"]) != 10:
            return False, "Distractor heavy context does not have 10 documents"
        if inst["metadata"]["distractor_document_count"] != 8:
            return False, "Distractor heavy does not have 8 distractors"

    elif condition == "conflict":
        meta = inst.get("metadata", {})
        if not meta.get("is_synthetic_conflict"):
            return False, "Conflict metadata missing synthetic flag"
        if not meta.get("synthetic_conflict_sentence"):
            return False, "Conflict metadata missing synthetic sentence"
        if meta.get("original_conflict_sentence") == meta.get("synthetic_conflict_sentence"):
            return False, "Conflict sentences are identical"
        synthetic_docs = [d for d in inst["context"] if d["source_type"] == "synthetic_conflict"]
        if len(synthetic_docs) != 1:
            return False, "Conflict context does not contain exactly 1 synthetic conflict passage"

    return True, "Valid"


def generate_clearrag_benchmark():
    """Main benchmark generation pipeline."""
    print("=" * 65)
    print("ClearRAG — Controlled Evaluation Benchmark Generator")
    print("=" * 65)

    if not SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Source dataset not found at {SOURCE_DATA_PATH}")

    print(f"Reading read-only source dataset: {SOURCE_DATA_PATH}")
    with open(SOURCE_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} source examples from HotpotQA.")
    print(f"Target per condition: {TARGET_PER_CONDITION} (Total: {TARGET_PER_CONDITION * len(CONDITIONS)})")
    print(f"Deterministic seed: {RANDOM_SEED}")

    # Deterministic shuffle
    rng = random.Random(RANDOM_SEED)
    shuffled_indices = list(range(len(data)))
    rng.shuffle(shuffled_indices)

    used_source_ids: Set[str] = set()
    condition_instances: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CONDITIONS}
    skipped_counts: Dict[str, int] = {c: 0 for c in CONDITIONS}

    # 1. Partial Evidence (prioritize candidates with >=2 supporting facts)
    for idx in shuffled_indices:
        if len(condition_instances["partial_evidence"]) >= TARGET_PER_CONDITION:
            break
        item = data[idx]
        if item["_id"] in used_source_ids:
            continue
        inst = build_partial_evidence_instance(item, rng)
        if inst:
            valid, msg = validate_instance(inst)
            if valid:
                condition_instances["partial_evidence"].append(inst)
                used_source_ids.add(item["_id"])
            else:
                skipped_counts["partial_evidence"] += 1
        else:
            skipped_counts["partial_evidence"] += 1

    # 2. Conflict (candidates with safely perturbable numeric facts)
    for idx in shuffled_indices:
        if len(condition_instances["conflict"]) >= TARGET_PER_CONDITION:
            break
        item = data[idx]
        if item["_id"] in used_source_ids:
            continue
        inst = build_conflict_instance(item)
        if inst:
            valid, msg = validate_instance(inst)
            if valid:
                condition_instances["conflict"].append(inst)
                used_source_ids.add(item["_id"])
            else:
                skipped_counts["conflict"] += 1
        else:
            skipped_counts["conflict"] += 1

    # 3. Unsupported (10 docs, 2 supporting, 8 distractors)
    for idx in shuffled_indices:
        if len(condition_instances["unsupported"]) >= TARGET_PER_CONDITION:
            break
        item = data[idx]
        if item["_id"] in used_source_ids:
            continue
        inst = build_unsupported_instance(item)
        if inst:
            valid, msg = validate_instance(inst)
            if valid:
                condition_instances["unsupported"].append(inst)
                used_source_ids.add(item["_id"])
            else:
                skipped_counts["unsupported"] += 1
        else:
            skipped_counts["unsupported"] += 1

    # 4. Distractor Heavy (10 docs, 2 supporting, 8 distractors)
    for idx in shuffled_indices:
        if len(condition_instances["distractor_heavy"]) >= TARGET_PER_CONDITION:
            break
        item = data[idx]
        if item["_id"] in used_source_ids:
            continue
        inst = build_distractor_heavy_instance(item)
        if inst:
            valid, msg = validate_instance(inst)
            if valid:
                condition_instances["distractor_heavy"].append(inst)
                used_source_ids.add(item["_id"])
            else:
                skipped_counts["distractor_heavy"] += 1
        else:
            skipped_counts["distractor_heavy"] += 1

    # 5. Full Evidence (standard valid examples)
    for idx in shuffled_indices:
        if len(condition_instances["full_evidence"]) >= TARGET_PER_CONDITION:
            break
        item = data[idx]
        if item["_id"] in used_source_ids:
            continue
        inst = build_full_evidence_instance(item)
        if inst:
            valid, msg = validate_instance(inst)
            if valid:
                condition_instances["full_evidence"].append(inst)
                used_source_ids.add(item["_id"])
            else:
                skipped_counts["full_evidence"] += 1
        else:
            skipped_counts["full_evidence"] += 1

    # Check completeness
    print("\nGeneration Summary:")
    print("-" * 65)
    all_instances: List[Dict[str, Any]] = []
    shortage = False

    for condition in CONDITIONS:
        count = len(condition_instances[condition])
        skipped = skipped_counts[condition]
        print(f"  {condition:<20}: {count:>4} generated ({skipped:>4} skipped/ineligible)")
        all_instances.extend(condition_instances[condition])
        if count < TARGET_PER_CONDITION:
            shortage = True
            print(f"    [WARNING] Shortage in condition '{condition}': {count}/{TARGET_PER_CONDITION}")

    print(f"\nTotal instances generated: {len(all_instances):,}")
    print(f"Total unique source HotpotQA IDs used: {len(used_source_ids):,}")

    if shortage:
        print("\n[WARNING] Could not achieve target size for all conditions.")
    else:
        print("\n[SUCCESS] All 5 conditions successfully reached exact target quotas.")

    # Write output file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_instances, f, indent=2, ensure_ascii=False)

    print(f"\nSaved benchmark dataset to: {OUTPUT_FILE_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    generate_clearrag_benchmark()
