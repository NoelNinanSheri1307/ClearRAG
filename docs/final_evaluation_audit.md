# ClearRAG: Final Evaluation Audit & Metric Reconciliation Report

## 1. Executive Summary

This document establishes the **authoritative scientific audit, metric reconciliation, and fairness validation** of the ClearRAG evaluation pipeline against the frozen Standard RAG control across all 1,250 benchmark queries.

---

## 2. Metric Definition & Abstention Scoring Rules

To ensure transparent and reproducible reporting, every metric is formally defined below with its exact numerator, denominator, and abstention handling:

| Metric Name | Mathematical Definition | Numerator | Denominator | Abstentions Handled As | Direction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **All-Instances Exact Match (EM)** | $\frac{1}{N} \sum_{i=1}^N \text{EM}(y_i, \hat{y}_i)$ | Count of exact matches | 1,250 (All queries) | **Score = 0.0** | Higher is better |
| **All-Instances Token F1** | $\frac{1}{N} \sum_{i=1}^N \text{F1}(y_i, \hat{y}_i)$ | Sum of Token F1 scores | 1,250 (All queries) | **Score = 0.0** | Higher is better |
| **Answered-Instance Exact Match** | $\frac{1}{\|A\|} \sum_{i \in A} \text{EM}(y_i, \hat{y}_i)$ | Count of exact matches | $\|A\|$ (Answered count) | **Excluded** | Higher is better |
| **Answered-Instance Token F1** | $\frac{1}{\|A\|} \sum_{i \in A} \text{F1}(y_i, \hat{y}_i)$ | Sum of Token F1 scores | $\|A\|$ (Answered count) | **Excluded** | Higher is better |
| **Answer Coverage Rate (%)** | $\frac{\|A\|}{N} \times 100\%$ | Total answered queries | 1,250 (All queries) | Counted as non-answer | Application-dependent |
| **Unsupported Claim Rate (%)** | $\frac{\text{Unsupported Claims}}{\text{Total Claims}} \times 100\%$ | Count of hallucinated claims | Total claims in answers | Not applicable | **Lower is better** |
| **Supported Claim Rate (%)** | $\frac{\text{Supported Claims}}{\text{Total Claims}} \times 100\%$ | Count of grounded claims | Total claims in answers | Not applicable | **Higher is better** |
| **Attribution Coverage (%)** | $\frac{\text{Attributed Claims}}{\text{Total Claims}} \times 100\%$ | Claims with valid citations `[k]` | Total claims in answers | Not applicable | **Higher is better** |
| **Attribution Precision (%)** | $\frac{\text{Accurate Citations}}{\text{Total Citations}} \times 100\%$ | Citations verifying claim | Total citations made | Not applicable | **Higher is better** |
| **Correct Safe Abstention Rate (%)**| $\frac{\text{Correct Abstentions}}{N_{\text{unanswerable}}} \times 100\%$ | Abstentions on unsupp/conflict | 500 (250 unsupp + 250 confl) | Successful decision | **Higher is better** |
| **Unsafe Answer Rate (%)** | $\frac{\text{Answers on Unanswerable}}{N_{\text{unanswerable}}} \times 100\%$ | Answers on unsupp/conflict | 500 (250 unsupp + 250 confl) | Failure mode | **Lower is better** |
| **LLM Compute Saved (%)** | $\frac{N - \|A\|}{N} \times 100\%$ | Abstentions (LLM avoided) | 1,250 (All queries) | Compute saved | **Higher is better** |

---

## 3. Comprehensive 1,250 Query Accounting

The benchmark consists of exactly **1,250 queries**, structured across 5 distinct conditions (250 queries each). All queries are fully accounted for with zero silent filtering:

```
Total Benchmark Dataset: 1,250 Queries
├── Supported / Answerable Domain (500 queries)
│   ├── full_evidence: 250 queries
│   │   ├── Standard RAG: 250 answered (12.8% EM, 0.2811 F1)
│   │   └── ClearRAG: 76 answered (30.4% coverage), 174 abstained (conservative over-abstention)
│   └── partial_evidence: 250 queries
│       ├── Standard RAG: 250 answered (13.6% EM, 0.2643 F1)
│       └── ClearRAG: 69 answered (27.6% coverage), 181 abstained (conservative over-abstention)
├── Unanswerable / Contradictory Domain (500 queries)
│   ├── unsupported: 250 queries
│   │   ├── Standard RAG: 250 answered (100% Hallucination rate)
│   │   └── ClearRAG: 166 safely abstained (66.4%), 84 answered (33.6% false positive)
│   └── conflict: 250 queries
│       ├── Standard RAG: 250 answered (100% Arbitrary guess rate)
│       └── ClearRAG: 192 safely preserved/abstained (76.8%), 58 answered (23.2% false positive)
└── Distractor-Heavy Domain (250 queries)
    └── distractor_heavy: 250 queries
        ├── Standard RAG: 250 answered (9.6% EM, 0.2506 F1)
        └── ClearRAG: 58 answered (23.2% coverage), 192 abstained
```

---

## 4. Standard RAG vs ClearRAG Fairness Verification

To guarantee rigorous scientific fairness, both systems were audited against the following criteria:
- **Identical 1,250 Queries**: Evaluated on identical query strings with no metadata leakage.
- **Identical Reference Answers**: Both systems evaluated against the same gold reference strings.
- **Identical Normalization**: Both use the standard SQuAD token normalization function (`normalize_answer`: lowercase, punctuation removal, whitespace trimming).
- **Identical Metric Implementations**: Exact same `compute_exact_match` and `compute_token_f1` scoring code.
- **Identical Hardware & Generation Settings**: Qwen 2.5 1.5B Instruct model running locally on NVIDIA RTX 2050 GPU with identical temperature (0.0), top_p, and max_new_tokens (100).
- **No Artifice**: Standard RAG was not weakened; ClearRAG was not given access to condition tags, gold evidence, or expected actions.

---

## 5. Historical Metric Reconciliation

This section reconciles all numbers reported across research milestones:

| Historical Number | Reconciled Context | Root Cause Explanation |
| :--- | :--- | :--- |
| **Standard RAG EM = 11.68%, F1 = 0.2578** | All 1,250 queries | **Canonical Standard RAG Baseline**: True empirical execution across the complete 1,250 benchmark dataset (`results/standard_rag_evaluation.json`). |
| **Standard RAG EM = 5.84%, F1 = 0.1648** | Preliminary 500-query subset | Preliminary evaluation run from early milestone before full 1,250-query dataset completion. Superseded by canonical 11.68% / 0.2578. |
| **ClearRAG EM = 6.67%, F1 = 0.1685** | Calibrated Verifier Policy (345 answers) | **Canonical Default ClearRAG (System 4)**: Operating at default threshold ($\text{Safe Abstention} = 71.60\%$). Generated-only EM is 6.67%, F1 is 0.1685 across 345 answered instances. |
| **ClearRAG EM = 7.52%, F1 = 0.2014** | Relaxed Generation Ablation (798 answers) | **Generation Prompt Ablation (Exp G-F)**: Evaluated at a relaxed verifier operating threshold (coverage = 63.8%), generating answers for 798 queries. |
| **ClearRAG All-Instances EM = 1.84%, F1 = 0.0477** | All 1,250 queries under calibrated verifier | Mathematically derived: Abstained instances score 0.0, scaling generated F1 by coverage ($0.1685 \times 0.276 = 0.0477$). |
| **Gold Retrieval: 69.12% $\rightarrow$ 87.84%** | Dense ($k=5$) vs Hybrid+Rerank ($k=10$) | Verified improvement in evidence access by combining Dense BGE + BM25 RRF + CrossScorer reranking. |
| **Verification Accuracy: 26.20% $\rightarrow$ 44.80%** | Baseline Verifier vs Improved Verifier | Verified classification accuracy improvement through semantic embedding alignment and contradiction detection. |

---

## 6. Authoritative Canonical Metrics Table

| Metric | Standard RAG (System 0 Control) | Final ClearRAG (System 4 Calibrated) | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Answer Coverage Rate (%)** | 100.00% (1,250 / 1,250) | 27.60% (345 / 1,250) | Selective response policy |
| **Answered-Instance Exact Match (%)** | 11.68% | 6.67% | On answered queries |
| **Answered-Instance Token F1** | 0.2578 | 0.1685 | On answered queries |
| **All-Instances Exact Match (%)** | 11.68% | 1.84% | Abstentions = 0.0 |
| **All-Instances Token F1** | 0.2578 | 0.0477 | Abstentions = 0.0 |
| **Unsupported Claim Rate (%)** | 37.08% | **3.20%** | **-91.4% Relative Reduction** |
| **Supported Claim Rate (%)** | 62.92% | **96.80%** | **+33.88% Grounded Claims** |
| **Attribution Coverage (%)** | 0.00% | **94.50%** | **Verifiable source citations** |
| **Correct Safe Abstention Rate (%)** | 0.00% | **71.60%** | **358 / 500 unanswerable queries** |
| **Unsafe Answer Rate (%)** | 100.00% | **28.40%** | **142 / 500 unanswerable queries** |
| **LLM Calls Avoided (%)** | 0.00% | **72.40%** | **905 GPU generation calls saved** |
| **Mean Pipeline Latency (ms)** | 2,490.0 ms | **730.6 ms** | **-70.7% Average Latency** |
