# ClearRAG — Standard RAG Baseline Documentation

> [!IMPORTANT]
> **Baseline Disclaimer**:
> This implementation is the conventional RAG baseline and does not contain ClearRAG's verification, conflict detection, selective abstention, or evidence sufficiency mechanisms. It serves as an uncontaminated control condition to evaluate the empirical improvements of future ClearRAG verification layers.

---

## 1. Overview

The Standard RAG baseline represents the conventional retrieve-then-generate pipeline. Given an input inquiry, the system retrieves the top-$K$ relevant sentence passages from the pre-indexed knowledge base, formats them into a structured context prompt, and generates a factual answer using `Qwen/Qwen2.5-1.5B-Instruct`.

---

## 2. Architecture Pipeline

```
                     ┌───────────────────────────────┐
                     │         User Question         │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │       BGE Query Embedder      │
                     │    (BAAI/bge-small-en-v1.5)   │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │          FAISS Index          │
                     │    (Exact Cosine / 269.5k)    │
                     └───────────────┬───────────────┘
                                     │
                                     ▼ (Top-K Chunks)
                     ┌───────────────────────────────┐
                     │         PromptBuilder         │
                     │  (Context formatting + prompt)│
                     └───────────────┬───────────────┘
                                     │
                                     ▼ (Chat Messages)
                     ┌───────────────────────────────┐
                     │         LLM Generator         │
                     │  (Qwen/Qwen2.5-1.5B-Instruct) │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │       Generated Answer        │
                     │    + Provenance + Latency     │
                     └───────────────────────────────┘
```

---

## 3. Models & Hardware Setup

| Component | Model / Specification | Notes |
| :--- | :--- | :--- |
| **Retriever Model** | `BAAI/bge-small-en-v1.5` | 384 dimensions, L2-normalized |
| **Vector Index** | FAISS `IndexFlatIP` | 269,556 indexed chunks, exact cosine similarity |
| **Generator Model**| `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B parameters, causal decoder |
| **Inference Precision**| `torch.float16` on CUDA | Memory-conscious for RTX 2050 (4 GB VRAM) |
| **Generation Mode**| Deterministic (`do_sample=False`, `temperature=0.0`) | Ensures strict evaluation reproducibility |

---

## 4. Prompt Structure

The prompt formats retrieved chunks into numbered evidence blocks with provenance preservation:

```text
[System]
You are a helpful assistant. Answer the question based on the provided context. Be concise, direct, and factual.

[User]
Context:
[1] Document: Bactris
Bactris is a genus of spiny palms which is native to the Americas.

[2] Document: Epigaea
It contains three species of flowering plants in the family Ericaceae.

Question: Which genus has more species, Bactris or Epigaea?

Answer:
```

---

## 5. Result Schema & Provenance

Every inference query produces a structured `RAGResult` record:

```json
{
  "question": "Which genus has more species, Bactris or Epigaea?",
  "answer": "Bactris has more species.",
  "retrieved_context": [
    {
      "rank": 1,
      "chunk_id": "chunk_0037294",
      "score": 0.7474,
      "title": "Bactris",
      "sentence_indices": [0],
      "text": "Bactris is a genus of spiny palms which is native to the Mexico...",
      "is_supporting_fact": false
    }
  ],
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "top_k": 5,
  "latency_retrieval_ms": 7.42,
  "latency_generation_ms": 112.50,
  "latency_total_ms": 119.92
}
```

---

## 6. CLI Usage

### Interactive & Ad-Hoc Testing
```powershell
# Run sample benchmark test queries
.venv\Scripts\python.exe scripts/test_rag.py

# Test a specific question
.venv\Scripts\python.exe scripts/test_rag.py --query "Are The Datsuns and The Black Crowes both rock bands?"

# Launch interactive REPL mode
.venv\Scripts\python.exe scripts/test_rag.py --interactive
```

### Full Benchmark Evaluation (1,250 queries)
```powershell
# Run quick smoke test on 5 instances
.venv\Scripts\python.exe scripts/evaluate_rag.py --max_instances 5

# Run full evaluation across all 1,250 benchmark queries (with automatic checkpointing and resume)
.venv\Scripts\python.exe scripts/evaluate_rag.py
```

---

## 7. Output Artifacts

- **Standard RAG Predictions & Metrics**: [`results/standard_rag_evaluation.json`](file:///c:/Users/VICTUS/ClearRAG/results/standard_rag_evaluation.json)
  Contains overall Exact Match (EM), Token F1, Contains-GT, and per-condition breakdown across:
  - `full_evidence` (250 instances)
  - `partial_evidence` (250 instances)
  - `unsupported` (250 instances)
  - `distractor_heavy` (250 instances)
  - `conflict` (250 instances)

---

## 8. Limitations of the Standard RAG Baseline

1. **No Verification Mechanism**: The baseline assumes all retrieved passages are valid, relevant, and non-conflicting.
2. **Susceptibility to Distractors & Conflicts**: In corrupted or conflicting evidence conditions, the conventional baseline generates fluent but ungrounded or hallucinated answers.
3. **No Selective Abstention**: When evidence is missing (`unsupported`), conventional RAG attempts to answer regardless of evidence absence rather than signaling insufficiency.
