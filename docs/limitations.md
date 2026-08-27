# ClearRAG: Research Limitations, Vulnerabilities & Design Trade-offs

## 1. Executive Statement of Limitations

ClearRAG is designed as a **selective, evidence-grounded, citation-attributed retrieval-augmented generation system**. It prioritizes factual safety, source provenance, and abstention on insufficient/contradictory evidence over maximizing raw answer coverage.

To maintain scientific integrity and publication readiness, this document outlines the verified limitations, empirical trade-offs, and failure modes of ClearRAG.

---

## 2. Core Methodological and Empirical Limitations

### A. Raw Answer Coverage Reduction (Selective Prediction Tradeoff)
- **Empirical Observation**: Under the calibrated sufficiency policy, ClearRAG answers **27.60% (345 / 1,250)** of benchmark queries, abstaining on **72.40%**.
- **Research Implication**: In applications requiring high coverage where any plausible answer is preferred over silence, ClearRAG's default gating is overly strict. While it eliminates 91.4% of unsupported claims, it declines to answer 71.0% of questions in the supported/partial evidence conditions where phrasing is ambiguous.

### B. Lower Raw Answer Generation Metrics (Exact Match & Token F1)
- **Empirical Observation**:
  - Standard RAG achieves **11.68% EM** and **0.2578 Token F1** across all 1,250 queries.
  - ClearRAG achieves **6.67% EM** and **0.1685 Token F1** on answered queries (**1.84% EM** and **0.0477 Token F1** across all 1,250 queries).
- **Underlying Cause**:
  1. Standard RAG's unconstrained generation produces verbose topical sentences that serendipitously contain gold tokens, inflating token F1 even when assertions are ungrounded.
  2. ClearRAG's generation is strictly constrained to concise, evidence-bounded text with bracketed citations (`[1]`, `[2]`), which penalizes token overlap against un-attributed reference strings.
  3. Under all-instances evaluation, abstained instances are scored as 0.0.

### C. Over-Abstention (Conservative Verifier False Negatives)
- **Empirical Observation**: In 145 cases (11.60% of all benchmark queries), Standard RAG generated a correct answer, but ClearRAG chose to abstain.
- **Root Cause**: The deterministic and semantic verification layer requires exact subject-predicate-object alignment. When the retrieved passage expresses a fact via complex syntactic transformations, metaphors, or multi-sentence coreference, the verifier assigns an `UNSUPPORTED` status, triggering a false abstention.

### D. Dependence on Retrieval Quality
- **Empirical Observation**: Gold evidence retrieval success is **87.84%** under Hybrid Dense+BM25 with CrossScorer reranking.
- **Limitation**: In the remaining **12.16% (152 queries)** where gold bridge passages are missed, the verification layer correctly detects that evidence is missing and abstains. ClearRAG cannot recover from retrieval failures through parametric memory.

### E. Computational Overhead of Semantic Verification
- **Empirical Observation**: The verification layer adds **~89.5 ms** of latency per query.
- **Trade-off**: While this overhead is negligible compared to the ~2.4s GPU generation time and results in a **70.7% net pipeline speedup** when queries abstain early, on 100% supported workloads with guaranteed evidence, the verification layer adds latency without altering the decision.

### F. Local Small-Model Generator Constraints (Qwen 2.5 1.5B)
- **Empirical Observation**: The local 1.5B parameter instruction model occasionally generates misaligned formatting or verbose explanations despite strict system prompting.
- **Hardware Bound**: The system was evaluated under strict consumer GPU constraints (NVIDIA GeForce RTX 2050 with 4GB VRAM in fp16). Larger frontier models (e.g. 70B+) may yield higher exact match accuracy on grounded context.

---

## 3. Threat to Validity & Generalization Boundaries

1. **Benchmark Domain**: The evaluation was performed on a multi-hop Wikipedia corpus (HotpotQA-derived). Performance on open-domain conversational queries or structured enterprise databases may exhibit different claim extraction and predicate matching characteristics.
2. **Attribution Metric Bounds**: Attribution coverage and precision are evaluated using sentence-level citation parsing and token overlap alignment. While robust, this automated metric does not replace human expert factual audits.

---

## 4. Summary Matrix of Trade-offs

| Dimension | Standard RAG | ClearRAG | Favorable System |
| :--- | :--- | :--- | :--- |
| **Factual Safety & Grounding** | Poor (37.08% unsupported claims) | **Superior (3.20% unsupported claims)** | **ClearRAG** |
| **Source Provenance** | None (0.0% attribution) | **High (94.50% attribution coverage)** | **ClearRAG** |
| **Contradiction Handling** | Arbitrary guess (0.0% safe) | **Conflict Preserved (76.8% safe)** | **ClearRAG** |
| **GPU Generation Compute** | 100% incurred (1,250 calls) | **72.40% compute saved (345 calls)** | **ClearRAG** |
| **Raw Answer Volume** | **100% coverage** | 27.60% selective coverage | **Standard RAG** |
| **Raw Token F1 (All Queries)**| **0.2578 (Inflated by guessing)** | 0.0477 (Penalized by abstention) | **Standard RAG** |
