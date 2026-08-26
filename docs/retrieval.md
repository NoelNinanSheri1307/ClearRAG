# ClearRAG Retrieval Architecture Documentation

## 1. Overview
The ClearRAG Retrieval Pipeline is a modular, reproducible semantic search system designed for question answering on HotpotQA and benchmark evaluation. It serves as the common retrieval foundation for both the **Standard RAG baseline** and the **ClearRAG hallucination-resistant RAG pipeline**.

```
HotpotQA Contexts / Corpus
           │
           ▼
[CorpusBuilder & Sentence Chunking]  ──►  data/corpus/corpus_chunks.json
           │
           ▼
[BGE-small-en-v1.5 Embeddings]       ──►  L2 Normalized Embeddings (dim=384)
           │
           ▼
[FAISS IndexFlatIP & Metadata Store] ──►  data/processed/faiss_index.bin & index_metadata.json
           │
           ▼
[Retriever: Top-K Vector Search]     ──►  Ranked Provenance-Rich Evidence Passages
           │
           ▼
[Retrieval Evaluator (Recall@K)]     ──►  results/retrieval_evaluation.json
```

---

## 2. Component Specifications

### A. Corpus Builder & Sentence Chunking (`src/ingestion/`)
- **Strategy**: Sentence-level chunking preserving exact document boundaries and sentence indices.
- **Provenance Contract**: Every chunk carries:
  - `chunk_id`: Unique identifier (e.g. `chunk_0000123`).
  - `source_dataset`: Provenance dataset (`"HotpotQA"`).
  - `source_question_id`: Origin question ID.
  - `document_title`: Wikipedia article / document title.
  - `sentence_indices`: Exact sentence index within the document (`[0]`).
  - `text`: Raw text of the sentence.
  - `is_supporting_fact`: Boolean indicator (used only during evaluation).
  - `metadata`: Document index, sentence count, question references.

### B. Embedding Generation (`src/retrieval/embedder.py`)
- **Model**: `BAAI/bge-small-en-v1.5`
- **Embedding Dimension**: 384
- **Normalization**: L2 normalized embeddings so that inner product equals cosine similarity ($A \cdot B = \cos(\theta)$).
- **Query Prefix**: Uses BGE retrieval query instruction:
  `"Represent this sentence for searching relevant passages: {query}"`
- **Device Support**: Automatic CUDA detection (RTX 2050 4GB) with CPU fallback.

### C. FAISS Vector Indexing (`src/retrieval/faiss_index.py`)
- **Index Type**: `faiss.IndexFlatIP` (Exact Maximum Inner Product Search / Cosine Similarity).
- **Serialization**:
  - Binary FAISS index: `data/processed/faiss_index.bin`
  - Metadata mapping store: `data/processed/index_metadata.json`

### D. Evidence Retriever (`src/retrieval/retriever.py`)
- Executes query embedding and vector search.
- Returns ranked results sorted by descending similarity score:
  ```json
  {
    "rank": 1,
    "chunk_id": "chunk_0000042",
    "score": 0.8452,
    "document_title": "Scott Derrickson",
    "text": "Scott Derrickson (born July 16, 1966) is an American director...",
    "sentence_indices": [0],
    "is_supporting_fact": true,
    "provenance": {...}
  }
  ```

### E. Retrieval Evaluation Metrics (`src/evaluation/retrieval_metrics.py`)
Retrieval quality is measured against HotpotQA gold supporting facts across cutoffs $K \in \{1, 3, 5, 10\}$:
1. **Document Recall@K (`doc_recall@K`)**:
   $$\text{Doc Recall@K} = \frac{|\text{Retrieved Documents}_{1..K} \cap \text{Gold Documents}|}{|\text{Gold Documents}|}$$
2. **Sentence/Fact Recall@K (`fact_recall@K`)**:
   $$\text{Fact Recall@K} = \frac{|\text{Retrieved Facts (Title, Sent Index)}_{1..K} \cap \text{Gold Facts}|}{|\text{Gold Facts}|}$$
3. **Document HitRate@K (`doc_hit@K`)**:
   $$1.0 \text{ if at least one gold document is retrieved in Top-K, else } 0.0$$
4. **Full Document Coverage@K (`doc_full_coverage@K`)**:
   $$1.0 \text{ if ALL gold supporting documents are retrieved in Top-K, else } 0.0$$

---

## 3. Command-Line Usage

### 1. Build Corpus
```bash
python scripts/build_corpus.py
```
*Options: `--max_questions <N>`, `--output_path <path>`*

### 2. Build FAISS Index
```bash
python scripts/build_index.py
```
*Options: `--batch_size 256`, `--device cuda`*

### 3. Test Retrieval
```bash
python scripts/test_retrieval.py --query "Which genus has more species, Bactris or Epigaea?" --top_k 5
```

### 4. Evaluate Retrieval Benchmark
```bash
python scripts/evaluate_retrieval.py --subset 250
```
*Or evaluate all 1,250 benchmark queries:*
```bash
python scripts/evaluate_retrieval.py --all
```
