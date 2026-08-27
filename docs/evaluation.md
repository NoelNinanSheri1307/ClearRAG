# ClearRAG: Controlled Baselines & Comparative Evaluation Framework

## 1. Executive Summary

This document defines the controlled experimental framework for evaluating **ClearRAG** against conventional **Standard RAG** and a standalone **Evidence Verification Layer**.

> [!IMPORTANT]
> **Empirical Baseline Integrity**:
> The current **26.20% verification accuracy** is an empirical baseline and is not presented as the final performance of ClearRAG.
> ClearRAG performance must be evaluated against controlled baselines using the same retrieval corpus and benchmark.

---

## 2. Experimental Setup & Controlled Variables

To guarantee scientific rigor and reproducibility, all three systems share identical underlying infrastructure:

| Variable | Controlled Setting | Details |
| :--- | :--- | :--- |
| **Dataset Source** | HotpotQA Dev Distractor | `data/raw/hotpotqa/hotpot_dev_distractor_v1.json` (7,405 questions) |
| **Corpus** | Structured Sentence Chunks | `data/corpus/corpus_chunks.json` (269,556 sentence chunks) |
| **Dense Embedder** | `BAAI/bge-small-en-v1.5` | 384-dimensional normalized embeddings |
| **Vector Index** | FAISS `IndexFlatIP` | `data/processed/faiss_index.bin` |
| **Retriever Top-K** | `k = 5` | Exact inner product search |
| **LLM Generator** | `Qwen/Qwen2.5-1.5B-Instruct` | Local GPU inference (`torch.float16`, `max_new_tokens=128`, `temperature=0.0`) |
| **Evaluation Benchmark** | ClearRAG Benchmark | `data/evaluation/clearrag_eval.json` (1,250 total instances) |

---

## 3. Systems Under Evaluation

```
                    ┌────────────────────────────────────────────────────────┐
                    │                      User Query                        │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │    FAISS Retriever    │
                                    │   (Top-k evidence)    │
                                    └───────────┬───────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         │                                      │                                      │
         ▼                                      ▼                                      ▼
┌──────────────────┐               ┌──────────────────────────┐           ┌──────────────────────────┐
│   Standard RAG   │               │   Verification Baseline  │           │         ClearRAG         │
│   (Always Answer)│               │  (Sufficiency Classify)  │           │   (Decision + Gating)    │
└────────┬─────────┘               └────────────┬─────────────┘           └────────────┬─────────────┘
         │                                      │                                      │
         ▼                                      ▼                                      ▼
PromptBuilder + LLM                   Claim Extraction +                      Claim Extraction +
         │                            Evidence Verification +                 Evidence Verification +
         ▼                            Sufficiency Engine                      Sufficiency Engine
   Answer Output                                │                                      │
(No Safety / Gating)                            ▼                                      ▼
                                       SufficiencyStatus                     ClearRAG DecisionEngine
                                    (No Generation Output)                             │
                                                                         ┌─────────────┴─────────────┐
                                                                         ▼                           ▼
                                                                  ANSWER / CAVEAT                 ABSTAIN
                                                                         │                           │
                                                                         ▼                           ▼
                                                                    LLM Generator          Deterministic Refusal
                                                                  (Qualified Answer)        (Skip LLM Compute)
```

### System 1: Standard RAG Baseline (`src/baselines/standard_rag.py`)
- **Pipeline**: Question $\rightarrow$ Retriever $\rightarrow$ PromptBuilder $\rightarrow$ LLM $\rightarrow$ Answer.
- **Behavior**: Always generates an answer regardless of evidence quality. Has no verification, conflict detection, or abstention capabilities.

### System 2: Evidence Verification Baseline (`src/baselines/verification_baseline.py`)
- **Pipeline**: Question $\rightarrow$ Retriever $\rightarrow$ ClaimExtractor $\rightarrow$ EvidenceVerifier $\rightarrow$ SufficiencyEngine.
- **Behavior**: Classifies evidence sufficiency into `FULLY_SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, or `CONFLICTING`. Does not generate natural language answers.

### System 3: ClearRAG (`src/clearrag/pipeline.py`)
- **Pipeline**: Question $\rightarrow$ Retriever $\rightarrow$ Claim Extraction $\rightarrow$ Verification $\rightarrow$ Sufficiency $\rightarrow$ Decision $\rightarrow$ Conditional Generation.
- **Behavior**: Answers when evidence is sufficient, qualifies answers with caveats on partial evidence, and deterministically abstains on unsupported or conflicting queries, skipping LLM generation.

### System 4: Oracle / Upper-Bound Analysis (`src/evaluation/oracle.py`)
- **NOTE**: `ORACLE / ANALYSIS ONLY` (evaluation-only diagnostic ceiling). Computes theoretical maximums if retrieval, verification, and generation had 100% precision.

---

## 4. Metadata Leakage Prevention

> [!CAUTION]
> **Strict Inference Isolation**:
> During inference, the input to every system is strictly the `question` string.
> Benchmark condition annotations (`full_evidence`, `unsupported`, `conflict`), expected behaviors, and ground truth answers are NEVER accessible to the pipeline and are only loaded post-inference by `ComparativeEvaluator`.

---

## 5. Benchmark Condition Taxonomy (1,250 Instances)

The benchmark consists of 250 instances across 5 balanced conditions:

1. **`full_evidence`** (250 instances): Complete supporting passages present in corpus. Expected: `ANSWER`.
2. **`partial_evidence`** (250 instances): Only a subset of supporting passages present in corpus. Expected: `ANSWER_WITH_CAVEAT`.
3. **`unsupported`** (250 instances): Key entity/relation evidence removed or absent from corpus. Expected: `ABSTAIN`.
4. **`distractor_heavy`** (250 instances): Gold evidence surrounded by high-similarity distractor passages. Expected: `ANSWER`.
5. **`conflict`** (250 instances): Multiple passages containing contradictory facts/dates for the target entity. Expected: `CONFLICT_ABSTENTION`.

---

## 6. Empirical Results & Comparative Tables

### Overall Cross-System Comparison

| Metric | Standard RAG | Verification Layer | ClearRAG |
| :--- | :--- | :--- | :--- |
| **Total Instances Evaluated** | 1,250 | 1,250 | 1,250 |
| **Answers Generated** | 1,250 | N/A | **887** |
| **Overall Abstention Rate** | 0.0% | N/A | **29.04%** |
| **Unsupported Abstention Rate** | 0.0% | N/A | **28.4%** |
| **Conflict Abstention Rate** | 0.0% | N/A | **29.6%** |
| **Verification Accuracy (Evaluable)** | N/A | **26.20%** | **26.20%** |
| **Exact Match (All Instances)** | 0.1168 | N/A | 0.0424 |
| **Token F1 (All Instances)** | 0.2578 | N/A | 0.1188 |
| **Exact Match (Generated-Only)** | 0.1168 | N/A | 0.0598 |
| **Token F1 (Generated-Only)** | 0.2578 | N/A | 0.1670 |
| **Mean Total Latency (ms)** | 2,079.1 ms | 55.9 ms | 2,520.5 ms |
| **Median Total Latency (ms)** | 1,575.4 ms | 55.9 ms | **1,152.3 ms** |
| **LLM Calls Executed** | 1,250 | 0 | **887** |
| **LLM Calls Avoided (% Saved)** | 0 (0.0%) | N/A | **363 (29.04%)** |
| **Conflict-Aware** | No | Yes | **Yes** |
| **Claim-Aware** | No | Yes | **Yes** |
| **Abstention-Aware** | No | No | **Yes** |

### Per-Condition Performance Breakdown

| Condition (250 each) | Retrieval Success % | Standard RAG F1 | ClearRAG Answer % | ClearRAG Abstain % | ClearRAG Correct Behavior % | ClearRAG Gen F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`full_evidence`** | 54.4% | 0.2811 | 70.4% | 29.6% | 9.6% | 0.1863 |
| **`partial_evidence`** | 83.2% | 0.2643 | 72.4% | 27.6% | 26.8% | 0.1670 |
| **`unsupported`** | 100.0% | 0.2502 | 71.6% | 28.4% | 28.4% | 0.1943 |
| **`distractor_heavy`** | 49.6% | 0.2506 | 70.0% | 30.0% | 7.6% | 0.1557 |
| **`conflict`** | 58.4% | 0.2427 | 70.4% | 29.6% | 29.6% | 0.1314 |

---

## 7. Error Taxonomy & Attribution

ClearRAG categorizes all failures into an 8-level error taxonomy:

| Error Taxonomy Category | Definition | Count | Percentage |
| :--- | :--- | :--- | :--- |
| **`RETRIEVAL_FAILURE`** | Gold supporting documents were not retrieved in top-k | 240 | 19.2% |
| **`VERIFICATION_FALSE_NEGATIVE`** | Evidence was present but verifier marked claims unsupported | 310 | 24.8% |
| **`VERIFICATION_FALSE_POSITIVE`** | Verifier falsely verified unsupported claims as supported | 123 | 9.8% |
| **`DECISION_POLICY_ERROR`** | Decision engine selected wrong policy for sufficiency status | 178 | 14.2% |
| **`GENERATION_ERROR`** | Evidence verified, but LLM generation did not match ground truth | 159 | 12.7% |
| **`CLAIM_EXTRACTION_FAILURE`** | Zero claims extracted from question | 0 | 0.0% |
| **`SUFFICIENCY_AGGREGATION_ERROR`** | Incorrect aggregation at sufficiency engine | 0 | 0.0% |
| **`CORRECT_EXECUTION`** | Pipeline executed accurately matching benchmark intent | 240 | 19.2% |

### Oracle Gap Analysis
- **Theoretical Safe Answer Target Rate**: 60.0% (Full Evidence + Distractor Heavy + Partial Evidence)
- **Theoretical Safe Abstention Target Rate**: 40.0% (Unsupported + Conflict)
- **Retrieval Loss Count**: 240 queries
- **Verification Loss Count**: 433 queries
- **Generation Loss Count**: 159 queries

---

## 8. Reproducibility Instructions

Run the complete comparison framework and generate all evaluation artifacts:

```powershell
# Run comparative evaluation and generate plots
.venv\Scripts\python.exe scripts/compare_systems.py --generate-plots

# Run unit tests across all layers
.venv\Scripts\pytest tests/ -v
```

Generated Artifacts:
- `results/comparative_evaluation.json`: Full machine-readable comparative metrics.
- `results/comparison_examples.json`: 25 representative per-query audit traces.
- `results/plots/`: 7 comparative evaluation figures.
