# ClearRAG

## Evidence-Grounded Retrieval-Augmented Generation with Verification and Safe Abstention

ClearRAG is a verification-aware Retrieval-Augmented Generation (RAG) framework designed to reduce hallucinations by requiring generated claims to be supported by retrieved evidence before an answer is produced.

Unlike conventional RAG systems that retrieve documents and directly pass them to an LLM, ClearRAG introduces an explicit verification layer between retrieval and generation. Claims are extracted, evidence is evaluated for relevance and entailment, and the system decides whether to answer, answer with a caveat, or abstain.

The core objective is not simply to generate an answer, but to determine whether the available evidence is sufficient to justify generating one.

---

## Key Idea

Standard RAG:

Query → Retrieval → LLM → Answer

ClearRAG:

Query → Retrieval → Claim Extraction → Evidence Verification → Decision → Grounded Generation / Abstention

The verification stage allows ClearRAG to distinguish between:

- Fully supported claims
- Partially supported claims
- Unsupported claims
- Contradictory evidence
- Temporally mismatched evidence
- Insufficient evidence

When evidence is insufficient, ClearRAG can skip LLM generation entirely and return a safe abstention instead of allowing the model to answer from unsupported parametric knowledge.

---

## Decision Policy

ClearRAG uses three primary outcomes:

### ANSWER

The retrieved evidence sufficiently supports the requested information.

The final response is generated using verified evidence and includes claim-level attribution where applicable.

### ANSWER_WITH_CAVEAT

The available evidence supports only part of the requested information.

ClearRAG answers the supported portion while explicitly identifying unsupported or insufficiently supported claims.

### ABSTAIN

The retrieved evidence does not provide sufficient support for a reliable answer.

In this case, generation can be skipped, preventing the LLM from filling the evidence gap from its own parametric knowledge.

---

## Core Features

- Evidence-aware RAG pipeline
- Dense and lexical retrieval
- Claim-level evidence verification
- Predicate-specific verification for relational claims
- Temporal consistency checking
- Sentence-level attribution
- Grounded answer generation
- Safe abstention
- Early generation skipping
- Standard RAG control baseline
- Paired benchmark evaluation
- Latency and GPU-compute measurement
- Statistical significance testing
- Local GPU inference
- Interactive claim and citation inspection
- Research-oriented reproducibility tooling

---

## Technology Stack

### Backend

- Python
- FastAPI
- PyTorch
- CUDA
- Hugging Face Transformers
- Dense retrieval
- BM25 / lexical retrieval
- Local LLM inference

### Models

- Embedding model: `BAAI/bge-small-en-v1.5`
- Generation model: `Qwen/Qwen2.5-1.5B-Instruct`

The exact models and configurations used for experiments are defined by the project configuration and benchmark environment.

### Frontend

- React
- TypeScript
- Vite
- Tailwind CSS

The frontend provides an interactive interface for comparing Standard RAG and ClearRAG outputs, inspecting retrieved evidence, viewing verification decisions, and examining claim-level attribution.

---

## Project Structure

```text
ClearRAG/
├── configs/             Configuration files
├── data/                Datasets and evaluation data
├── docs/                Documentation and reproducibility material
├── experiments/         Research experiments and evaluations
├── notebooks/           Experimental notebooks
├── results/             Benchmark and experiment results
├── scripts/             Utility and evaluation scripts
├── src/                 Core ClearRAG implementation
├── tests/               Automated tests
├── web/                 Web application
├── Web RAG Run/         Local GPU RAG execution interface
├── README.md
├── requirements.txt
└── pyproject.toml
