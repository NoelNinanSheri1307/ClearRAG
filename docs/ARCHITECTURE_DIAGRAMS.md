# ClearRAG — Formal System Architecture & Specifications (Monochrome Academic Edition)

This document provides the formal **black-and-white, top-to-bottom system architecture schematics** in **Times New Roman serif typography**, strictly adhering to IEEE Transactions, ACM SIGIR, Springer, and patent office line-drawing standards.

---

## 📄 Rendered Publication Figures

* **Monochrome Line-Art PNG (300 DPI)**: [`results/plots/clearrag_architecture_schematic.png`](file:///c:/Users/VICTUS/ClearRAG/results/plots/clearrag_architecture_schematic.png)
* **Vector Graphics PDF (Publication Ready)**: [`results/plots/clearrag_architecture_schematic.pdf`](file:///c:/Users/VICTUS/ClearRAG/results/plots/clearrag_architecture_schematic.pdf)
* **Generation Script**: `python scripts/generate_architecture_diagram.py`

---

## 1. Top-to-Bottom Flowchart (Pure Monochrome)

```mermaid
flowchart TD
    %% 0. Input Query
    Q["1. USER QUERY INPUT\n• Natural Language Question (q)"]

    %% A. Hybrid Retrieval
    subgraph A["A. HYBRID RETRIEVAL & RERANKING"]
        direction TB
        subgraph A_dual["Dual-Channel Candidate Retrieval"]
            FAISS["A1. Dense FAISS Search\n• BGE-small-en-v1.5 (384-dim)\n• Depth: k_dense = 10"]
            BM25["A2. Sparse BM25 Search\n• Okapi BM25 (k1=1.5, b=0.75)\n• Depth: k_lexical = 10"]
        end
        RRF["A3. Reciprocal Rank Fusion & Cross-Scorer\n• S_RRF(d) = Σ 1 / (60 + r_m(d))\n• S_final = 0.70·S_dense + 0.30·S_BM25\n• Output: Top-10 Evidence Chunks D = {p_1, ..., p_10}"]
    end

    %% B. Evidence Verification
    subgraph B["B. EVIDENCE VERIFICATION LAYER"]
        B1["B1. Claim Extraction: Decomposes q into Propositional Claims {c_1, ..., c_m}"]
        B2["B2. Relational Predicate Checking: Validates semantic roles (winner, founder, author...)"]
        B3["B3. Temporal Constraint Verification: Enforces year matching Year(q) ⊆ Year(p)"]
        B4["B4. Semantic Alignment: Cosine Sim (τ_sim ≥ 0.60) & Lexical Overlap (≥ 0.30)"]
    end

    %% C & D. Conflict Detection & Sufficiency Engine
    subgraph CD["C & D. CONFLICT DETECTION & SUFFICIENCY ENGINE"]
        D_CONF["D. Conflict Detection: e(p_i) == e(p_j) ∧ attr(p_i) == attr(p_j) ∧ val(p_i) ≠ val(p_j)"]
        C_SUFF["C. Sufficiency Evaluation: Aggregates claim outcomes\n(Status: SUFFICIENT | PARTIAL_SUFFICIENT | UNSUPPORTED | CONFLICTING)"]
    end

    %% Decision Gate
    GATE{"4-Way Policy Gate\n(Decision Engine)"}

    %% E. Generation & Refusal Payloads
    REFUSAL["3 & 4. SAFE REFUSAL PAYLOAD\n• GENERATION SKIPPED\n• 0 GPU Tokens Generated\n• Preserves 72.40% Compute"]
    CAVEAT["2. CAVEAT GENERATION\n• Qwen 2.5 1.5B (FP16, Temp=0.0)\n• Caveat Prompt Builder\n• Explicit Missing-Fact Warning"]
    GROUNDED["1. GROUNDED GENERATION\n• Qwen 2.5 1.5B (FP16, Temp=0.0)\n• Evidence Context [1], [2]\n• Max New Tokens: 384"]

    %% F. Claim Attribution & Output
    ATTR["F. CLAIM-LEVEL ATTRIBUTION & FINAL DELIVERABLE\n• Propositional Sentence Alignment\n• Grounding Precision: 95.20% | Coverage: 94.50%\n• Verified Inline Sentence Citations [1], [2]"]

    %% Flow Connections (Top to Bottom)
    Q --> FAISS & BM25
    FAISS & BM25 --> RRF
    RRF -->|Top-10 Chunks D| B
    B -->|Verified Claim Records| CD
    CD --> GATE

    %% Gated Branching
    GATE -->|Unsupported / Contradiction| REFUSAL
    GATE -->|Partial Evidence (1/2 Claims)| CAVEAT
    GATE -->|Fully Supported Evidence| GROUNDED

    CAVEAT --> ATTR
    GROUNDED --> ATTR

    %% Monochrome IEEE Styling
    style Q fill:#FFFFFF,stroke:#000000,stroke-width:1.5px,color:#000000
    style A fill:#FFFFFF,stroke:#000000,stroke-width:1.5px,color:#000000
    style A_dual fill:#FFFFFF,stroke:#666666,stroke-width:1px,stroke-dasharray: 3 3,color:#000000
    style FAISS fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style BM25 fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style RRF fill:#FFFFFF,stroke:#000000,stroke-width:1px,color:#000000
    style B fill:#FFFFFF,stroke:#000000,stroke-width:1.5px,color:#000000
    style B1 fill:#FFFFFF,stroke:#888888,stroke-width:1px,color:#000000
    style B2 fill:#FFFFFF,stroke:#888888,stroke-width:1px,color:#000000
    style B3 fill:#FFFFFF,stroke:#888888,stroke-width:1px,color:#000000
    style B4 fill:#FFFFFF,stroke:#888888,stroke-width:1px,color:#000000
    style CD fill:#FFFFFF,stroke:#000000,stroke-width:1.5px,color:#000000
    style D_CONF fill:#FFFFFF,stroke:#888888,stroke-width:1px,color:#000000
    style C_SUFF fill:#FFFFFF,stroke:#888888,stroke-width:1px,color:#000000
    style GATE fill:#FFFFFF,stroke:#000000,stroke-width:1.5px,color:#000000
    style REFUSAL fill:#FFFFFF,stroke:#000000,stroke-width:1.5px,stroke-dasharray: 4 4,color:#000000
    style CAVEAT fill:#FFFFFF,stroke:#000000,stroke-width:1.5px,color:#000000
    style GROUNDED fill:#FFFFFF,stroke:#000000,stroke-width:1.5px,color:#000000
    style ATTR fill:#FFFFFF,stroke:#000000,stroke-width:1.5px,color:#000000
```

---

## 2. LaTeX TikZ Monochrome Line-Art Code (For Papers & Patents)

Copy and paste directly into Overleaf or standard LaTeX document classes (`article`, `IEEEtran`, `acmart`):

```latex
\documentclass[tikz,border=12pt]{standalone}
\usepackage{tikz}
\usepackage{times}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, calc, backgrounds}

\begin{document}
\begin{tikzpicture}[
    font=\fontfamily{ptm}\selectfont\small,
    >=Stealth,
    node distance=0.8cm and 1.2cm,
    box/.style={rectangle, draw=black, fill=white, line width=0.8pt, align=center, inner sep=5pt},
    widebox/.style={rectangle, draw=black, fill=white, line width=0.8pt, align=left, inner sep=6pt, text width=11.5cm},
    decision/.style={diamond, draw=black, fill=white, line width=0.8pt, align=center, inner sep=2pt, aspect=2.0},
    arrow/.style={->, line width=0.8pt, draw=black},
    skiparrow/.style={->, line width=0.8pt, draw=black, dashed},
    groupbox/.style={rectangle, draw=black, dashed, line width=0.6pt, fill=none, inner sep=5pt}
]

    % 1. Input
    \node[box, text width=7.0cm] (input) {\textbf{1. USER QUERY INPUT}\\Question $q$ (Natural Language Multi-Hop Inquiry)};

    % 2. Retrieval
    \node[box, below left=0.7cm and -0.8cm of input, text width=5.2cm] (dense) {\textbf{A1. Dense Vector Search}\\$\bullet$ BAAI/bge-small-en-v1.5 (384-dim)\\$\bullet$ FAISS IndexFlatIP ($k_{\text{dense}}=10$)};
    \node[box, below right=0.7cm and -0.8cm of input, text width=5.2cm] (bm25) {\textbf{A2. Sparse Lexical Search}\\$\bullet$ Okapi BM25 ($k_1=1.5, b=0.75$)\\$\bullet$ Inverted Index ($k_{\text{lexical}}=10$)};
    
    \node[widebox, below=0.6cm of $(dense)!0.5!(bm25)$] (rrf) {\textbf{A3. Reciprocal Rank Fusion \& Cross-Scorer}\\$\bullet$ $S_{\text{RRF}}(d) = \sum_{m} 1 / (60 + r_m(d))$\\$\bullet$ $S_{\text{final}} = 0.70 \cdot S_{\text{dense}} + 0.30 \cdot S_{\text{BM25}}$\\$\bullet$ Output: Top-10 Ranked Evidence Chunks $D = \{p_1, \dots, p_{10}\}$};

    % 3. Verification
    \node[widebox, below=0.7cm of rrf] (verifier) {\textbf{B. EVIDENCE VERIFICATION LAYER}\\$\bullet$ \textbf{B1. Claim Extractor:} Decomposes $q \to$ Atomic Claims $\{c_1, \dots, c_m\}$\\$\bullet$ \textbf{B2. Relational Verifier:} Validates semantic roles (winner, founder...)\\$\bullet$ \textbf{B3. Temporal Verifier:} Enforces year matching $\text{Year}(q) \subseteq \text{Year}(p)$\\$\bullet$ \textbf{B4. Semantic Alignment:} Cosine Sim ($\tau_{\text{sim}} \ge 0.60$) \& Overlap ($\ge 0.30$)};

    % 4. Conflict & Sufficiency
    \node[widebox, below=0.7cm of verifier] (suff) {\textbf{C \& D. CONFLICT DETECTION \& SUFFICIENCY ENGINE}\\$\bullet$ \textbf{D. Conflict Detection:} $e(p_i) == e(p_j) \land \text{attr}(p_i) == \text{attr}(p_j) \land v_i \ne v_j$\\$\bullet$ \textbf{C. Sufficiency Evaluation:} Aggregates claim outcomes\\$\quad$ (Status: SUFFICIENT $|$ PARTIAL\_SUFFICIENT $|$ UNSUPPORTED $|$ CONFLICTING)};

    % 5. Decision Gate
    \node[decision, below=0.8cm of suff] (gate) {\textbf{4-Way Policy Gate}\\\textbf{(Decision Engine)}};

    % 6. Output Branches
    \node[box, below left=1.2cm and 1.5cm of gate, dashed, text width=3.3cm] (refusal) {\textbf{3 \& 4. SAFE REFUSAL}\\$\bullet$ Gen Skipped (0 Tok)\\$\bullet$ Refusal Payload\\$\bullet$ Saves 72.4\% Compute};
    \node[box, below=1.2cm of gate, text width=3.3cm] (caveat) {\textbf{2. CAVEAT GEN}\\$\bullet$ Qwen 2.5 1.5B (FP16)\\$\bullet$ Qualified Answer\\$\bullet$ Missing-Fact Warning};
    \node[box, below right=1.2cm and 1.5cm of gate, text width=3.3cm] (grounded) {\textbf{1. GROUNDED GEN}\\$\bullet$ Qwen 2.5 1.5B (FP16)\\$\bullet$ Context $[1], [2]$\\$\bullet$ Max Tokens: 384};

    % 7. Final Attribution
    \node[widebox, below=1.0cm of caveat] (attr) {\textbf{F. CLAIM-LEVEL ATTRIBUTION \& FINAL OUTPUT}\\$\bullet$ Sentence-level claim decomposition \& bidirectional semantic alignment\\$\bullet$ Attribution Precision: 95.20\% $|$ Coverage: 94.50\% $|$ Citations $[1], [2]$};

    % Connecting Arrows
    \draw[arrow] (input) -| (dense);
    \draw[arrow] (input) -| (bm25);
    \draw[arrow] (dense) |- (rrf);
    \draw[arrow] (bm25) |- (rrf);
    \draw[arrow] (rrf) -- node[right, font=\footnotesize] {Top-10 Chunks $D$} (verifier);
    \draw[arrow] (verifier) -- node[right, font=\footnotesize] {Verified Records} (suff);
    \draw[arrow] (suff) -- (gate);

    \draw[skiparrow] (gate) -| node[above, pos=0.25, font=\footnotesize] {Refusal} (refusal);
    \draw[arrow] (gate) -- node[right, font=\footnotesize] {Partial} (caveat);
    \draw[arrow] (gate) -| node[above, pos=0.25, font=\footnotesize] {Sufficient} (grounded);

    \draw[arrow] (caveat) -- (attr);
    \draw[arrow] (grounded) |- (attr);

\end{tikzpicture}
\end{document}
```
