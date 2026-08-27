# ClearRAG: Decision + Abstention Layer

## Motivation

Standard Retrieval-Augmented Generation (RAG) pipelines always generate an answer, regardless of whether the retrieved evidence actually supports the question. This leads to:

1. **Hallucination**: The LLM invents plausible-sounding answers when evidence is insufficient.
2. **Confident incorrectness**: The system presents unreliable answers with the same confidence as well-supported ones.
3. **Conflict propagation**: When retrieved evidence contains contradictions, the system may arbitrarily choose one version.

**ClearRAG** addresses these problems by inserting a **decision and abstention layer** between evidence verification and answer generation. The system only generates an answer when evidence is sufficient, adds qualification caveats for partial evidence, and **explicitly refuses to answer** when evidence is insufficient or conflicting.

## Difference Between Standard RAG and ClearRAG

| Aspect | Standard RAG | ClearRAG |
| --- | --- | --- |
| Evidence assessment | None | Claim-level verification |
| Decision control | None (always generates) | Policy-based decision engine |
| Insufficient evidence | Hallucinates an answer | Abstains with explanation |
| Conflicting evidence | Picks one arbitrarily | Abstains with conflict report |
| Partial evidence | No indication | Answers with explicit caveat |
| Provenance | Limited | Full audit trail |
| Answer reliability | Unknown | Graded (supported/partial/abstained) |

## Architecture

```
User Question
    │
    ▼
┌─────────────────────────────┐
│  Retriever (FAISS + BGE)    │  ◄── Frozen
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  RuleBasedClaimExtractor    │  ◄── Frozen
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  EvidenceVerifier            │  ◄── Frozen
│  (predicate-aware support,   │
│   attribute-aware conflict)  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  SufficiencyEngine          │  ◄── Frozen
│  FULLY_SUPPORTED            │
│  PARTIALLY_SUPPORTED        │
│  UNSUPPORTED                │
│  CONFLICTING                │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  ClearRAGDecisionEngine     │  ◄── NEW
│  ANSWER                     │
│  ANSWER_WITH_CAVEAT         │
│  ABSTAIN                    │
│  CONFLICT_ABSTENTION        │
└──────────┬──────────────────┘
           │
     ┌─────┴─────────────────┐
     │                       │
  Permits               Does NOT
 Generation             Permit
     │                       │
     ▼                       ▼
┌──────────┐       ┌──────────────────┐
│  LLM     │       │ Deterministic    │
│ Generator│       │ Abstention       │
│ (Qwen)   │       │ Response         │
└────┬─────┘       └────────┬─────────┘
     │                      │
     └──────┬───────────────┘
            │
            ▼
┌─────────────────────────────┐
│  ClearRAGResult             │
│  (full provenance)          │
└─────────────────────────────┘
```

## Decision Policy

The default decision policy is a deterministic mapping from `SufficiencyStatus` to `ClearRAGDecision`:

| SufficiencyStatus | ClearRAGDecision | LLM Called? | Behavior |
| --- | --- | --- | --- |
| `FULLY_SUPPORTED` | `ANSWER` | ✓ Yes | Normal generation with verified evidence |
| `PARTIALLY_SUPPORTED` | `ANSWER_WITH_CAVEAT` | ✓ Yes | Generation with caveat prefix |
| `UNSUPPORTED` | `ABSTAIN` | ✗ No | Deterministic refusal response |
| `CONFLICTING` | `CONFLICT_ABSTENTION` | ✗ No | Deterministic conflict report |

The policy is configurable via `configs/clearrag_config.yaml`.

## Abstention Behavior

When ClearRAG abstains:

1. **No LLM call is made** — the system does not waste compute on unreliable generation.
2. A **deterministic response** is returned explaining why the system cannot answer.
3. The **abstention reason** includes the specific sufficiency explanation.
4. Full provenance (claims, evidence, verification results) is preserved.

### ABSTAIN Response
> "I cannot provide a reliable answer to this question. The retrieved evidence does not contain sufficient information to support a factual response."

### CONFLICT_ABSTENTION Response
> "I cannot provide a reliable answer to this question. The retrieved evidence contains conflicting information, making it impossible to determine a trustworthy response."

## Evidence Flow

1. **Question** → FAISS top-k retrieval → raw evidence chunks
2. **Evidence** → Claim extraction → structured claims with predicates
3. **Claims + Evidence** → Per-claim verification → SUPPORTED / UNSUPPORTED / CONFLICTING
4. **Claim results** → Sufficiency aggregation → overall status
5. **Status** → Decision engine → ANSWER / CAVEAT / ABSTAIN / CONFLICT
6. **Decision** → Conditional generation → ClearRAGResult

## Provenance

Every ClearRAGResult includes:
- The original question
- All retrieved evidence chunks with scores
- Extracted claims with predicates and entities
- Per-claim verification results with supporting/conflicting evidence IDs
- Overall sufficiency status and explanation
- The ClearRAG decision
- The answer (generated or abstention response)
- Latency breakdown (retrieval, verification, generation, total)

## Evaluation Methodology

ClearRAG is evaluated over 1,250 benchmark instances (5 conditions × 250 each):

- **Strict inference isolation**: During inference, only the question and retrieved evidence are used. Benchmark labels are accessed only post-inference.
- **Generation quality**: Exact Match, Token F1, Contains Ground Truth.
- **Abstention metrics**: Abstention rate by condition, correct abstention rate, false-answer-on-unsupported rate.
- **Safety metrics**: Unsupported-answer rate, conflict-answer rate.
- **Latency**: Per-stage and total latency.

## Limitations

1. **Rule-based verification**: The current verification layer uses heuristic rules (predicate keywords, lexical overlap) rather than neural entailment. This limits accuracy on complex or implicit claims.
2. **Abstention recall on conflict**: Attribute-aware conflict detection only triggers for birth/death year conflicts. Other conflict types (locations, names, roles) are not yet detected.
3. **Caveat quality**: The caveat prefix is static text, not dynamically generated based on which specific claims lack evidence.
4. **No reasoning chain**: ClearRAG does not explain *why* evidence is insufficient beyond listing unsupported predicates.

## Why ClearRAG Is Not Simply "RAG With a Prompt"

Adding "only answer if confident" to a RAG system prompt does NOT achieve what ClearRAG does:

1. **Prompt instructions are unreliable**: LLMs frequently ignore system prompt instructions about when to abstain.
2. **No structured verification**: A prompt-based approach has no claim extraction, no predicate verification, and no conflict detection.
3. **No deterministic control**: ClearRAG's decision is made BEFORE the LLM is called. The LLM is never invoked for abstention cases.
4. **No provenance**: A prompt-based approach cannot report which claims were verified, which evidence was supporting, or why the system chose to abstain.
5. **Testable and auditable**: ClearRAG's decision policy is a deterministic function that can be unit-tested and formally verified.

## Future: Replacing the Deterministic Verifier

The ClearRAG architecture is designed for modularity. The `EvidenceVerifier` and `RuleBasedClaimExtractor` can be replaced with:

1. **Neural entailment models** (e.g., NLI classifiers) for more accurate support verification.
2. **LLM-based claim decomposition** for better claim extraction on complex questions.
3. **Cross-encoder reranking** for more precise evidence relevance scoring.
4. **Chain-of-thought reasoning** for multi-hop evidence linking.

These replacements would slot into the existing pipeline without changing the decision engine, result structure, or evaluation framework. The `ClearRAGDecisionEngine` policy would remain identical — only the quality of the sufficiency signals would improve.
