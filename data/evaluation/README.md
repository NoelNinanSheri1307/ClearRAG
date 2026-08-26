# Evaluation Dataset Schema

## Overview
This directory contains schema definitions and evaluation datasets for the ClearRAG evaluation suite. 

- `questions.json`: Schema-validation placeholder queries.
- `clearrag_eval.json`: Controlled 1,250-instance evaluation benchmark derived from HotpotQA (`data/raw/hotpotqa/hotpot_dev_distractor_v1.json`).

---

## ClearRAG Benchmark (`clearrag_eval.json`)

The development benchmark contains **1,250 instances** partitioned across 5 controlled conditions (250 instances per condition) generated deterministically (`RANDOM_SEED = 42`):

1. **`full_evidence`**: Original HotpotQA context with supporting facts and distractors intact. Expected behavior: `answer`.
2. **`partial_evidence`**: Multi-hop instances where at least one supporting fact is removed while retaining partial evidence. Expected behavior: `abstain_or_qualify`.
3. **`unsupported`**: Supporting evidence removed while retaining distractor documents. Expected behavior: `abstain`.
4. **`distractor_heavy`**: Standard 10-document context (2 supporting, 8 distractors) testing retrieval and evidence selection under noise. Expected behavior: `answer`.
5. **`conflict`**: Factual supporting sentences perturbed via controlled deterministic numeric shifts and injected as explicitly marked synthetic conflict documents (`source_type: "synthetic_conflict"`). Expected behavior: `conflict_detected_or_abstain`.

### Generation and Validation Commands
- **Generate benchmark**:
  ```bash
  python scripts/build_evaluation_dataset.py
  ```
- **Validate benchmark**:
  ```bash
  python scripts/validate_evaluation_dataset.py
  ```

---

## Schema Fields

Each entry in `questions.json` and `clearrag_eval.json` adheres to the standardized evaluation data contract:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Unique identifier for the evaluation instance (e.g., `"hotpot_<id>_full_evidence"`). |
| `source_dataset` | `string` | Provenance dataset name (`"HotpotQA"`). |
| `source_id` | `string` | Original record ID in the source dataset. |
| `condition` | `string` | Evaluation condition (`full_evidence`, `partial_evidence`, `unsupported`, `distractor_heavy`, `conflict`). |
| `question` | `string` | The input query evaluated by the QA systems. |
| `context` | `array[object]` | List of document objects, each containing `title`, `sentences`, and `source_type` (`"original"` or `"synthetic_conflict"`). |
| `ground_truth` | `string` \| `null` | The reference answer against which model responses are compared. |
| `original_supporting_facts` | `array[object]` | Gold supporting facts (`title`, `sentence_index`). |
| `retained_supporting_facts` | `array[object]` | Supporting facts present in the modified evaluation context. |
| `removed_supporting_facts` | `array[object]` | Supporting facts intentionally omitted from the evaluation context. |
| `expected_behavior` | `string` | Target behavioral expectation (`answer`, `abstain_or_qualify`, `abstain`, `conflict_detected_or_abstain`). |
| `metadata` | `object` | Condition-specific provenance and transformation details. |

---

## Evaluation Categories & Expected Behaviors

The benchmark defines five distinct query conditions designed to test hallucination resistance across diverse retrieval scenarios:

### 1. `full_evidence`
- **Definition**: The knowledge corpus contains complete, unambiguous evidence directly answering the question.
- **Objective**: Test standard QA competence, retrieval accuracy, and fidelity to evidence without generating hallucinations.

### 2. `unsupported`
- **Definition**: The corpus contains no relevant evidence to answer the query (or gold evidence has been removed).
- **Objective**: Test selective abstention; the system should identify the absence of evidence and explicitly decline to answer rather than fabricate facts.

### 3. `partial_evidence`
- **Definition**: The corpus provides evidence for only a subset of sub-claims in a composite question.
- **Objective**: Test boundary awareness; the system should answer verified parts while explicitly declaring the missing components as unknown.

### 4. `conflict`
- **Definition**: Retrieved documents provide contradictory information (e.g., competing factual statements or conflicting values).
- **Objective**: Test conflict detection; the system should highlight the contradiction or abstain from asserting a contested fact as ground truth.

### 5. `distractor_heavy`
- **Definition**: Retrieval results contain topically related but non-pertinent or misleading distractor passages alongside target evidence.
- **Objective**: Test noise filtering and evidence verification under high semantic interference.
