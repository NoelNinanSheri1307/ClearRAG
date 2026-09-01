"""ClearRAG Official Academic Architecture Schematic Generator.

Generates a publication-grade (300 DPI), clean, formal academic architecture diagram
for IEEE/ACM/arXiv research papers, patent specifications, and thesis viva defenses.
Features an official clean white background, crisp vector typography, and detailed technical descriptions.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_PATH = OUTPUT_DIR / "clearrag_architecture_schematic.png"
PDF_PATH = OUTPUT_DIR / "clearrag_architecture_schematic.pdf"


def draw_official_architecture():
    # Academic publication canvas (Widescreen 16:9, 300 DPI)
    fig, ax = plt.subplots(figsize=(22, 12), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Clean Academic White Palette
    bg_color = "#FFFFFF"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Color Tokens (Formal Academic & Publication Grade)
    color_navy = "#0F172A"       # Primary text & headers
    color_border = "#CBD5E1"     # Outer card borders
    color_blue_border = "#2563EB"# Section A: Retrieval (Cobalt Blue)
    color_blue_bg = "#F8FAFC"
    color_teal_border = "#0D9488"# Section B: Verification (Teal)
    color_teal_bg = "#F0FDFA"
    color_amber_border = "#D97706"# Section C: Decision Engine (Amber)
    color_amber_bg = "#FFFBEB"
    color_red_border = "#DC2626"  # Section D: Conflict (Crimson)
    color_red_bg = "#FEF2F2"
    color_green_border = "#16A34A"# Section E/F: Grounded Synthesis (Forest Green)
    color_green_bg = "#F0FDF4"
    
    text_dark = "#0F172A"
    text_muted = "#475569"
    text_subtle = "#64748B"

    # -------------------------------------------------------------
    # 1. MAIN TITLE & SUBTITLE BANNER
    # -------------------------------------------------------------
    ax.text(
        50, 97.2,
        "Figure 1: Complete System Architecture of ClearRAG (Evidence-Aware Selective RAG)",
        ha="center", va="center", fontsize=15, fontweight="bold", color=text_dark, fontfamily="sans-serif"
    )
    ax.text(
        50, 95.0,
        "A Deterministic 6-Stage Pipeline with Relational Verification, Conflict Detection, 4-Way Decision Gating, and Claim Attribution",
        ha="center", va="center", fontsize=10, color=text_muted, fontfamily="sans-serif"
    )

    # Helper function for drawing container modules
    def draw_module_box(x, y, w, h, title, subtitle, header_bg, border_col):
        # Outer card
        card = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4,rounding_size=0.8",
            facecolor="#FFFFFF", edgecolor=border_col, linewidth=1.4,
            zorder=2
        )
        ax.add_patch(card)
        # Header banner
        header = patches.FancyBboxPatch(
            (x, y + h - 3.8), w, 3.8,
            boxstyle="round,pad=0.0,rounding_size=0.0",
            facecolor=header_bg, edgecolor=border_col, linewidth=1.0,
            zorder=3
        )
        ax.add_patch(header)
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_dark, zorder=4)
        ax.text(x + w / 2, y + h - 3.0, subtitle, ha="center", va="center", fontsize=7.5, color=text_muted, zorder=4)
        return card

    # Helper function for drawing inner functional units
    def draw_unit(x, y, w, h, title, bullet_points, fill="#F8FAFC", border="#E2E8F0", title_col=text_dark):
        unit = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2,rounding_size=0.5",
            facecolor=fill, edgecolor=border, linewidth=1.0, zorder=3
        )
        ax.add_patch(unit)
        ax.text(x + 0.8, y + h - 1.2, title, ha="left", va="center", fontsize=8.2, fontweight="bold", color=title_col, zorder=4)
        
        # Bullet lines
        curr_y = y + h - 2.5
        for bp in bullet_points:
            ax.text(x + 0.8, curr_y, f"• {bp}", ha="left", va="center", fontsize=7.2, color=text_muted, zorder=4)
            curr_y -= 1.2

    # Helper function for clean academic connecting arrows
    def draw_conn(x1, y1, x2, y2, label="", col="#334155", ls="-", lw=1.2, label_y_offset=1.0):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.45", color=col, lw=lw, ls=ls),
            zorder=6
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + label_y_offset, label, ha="center", va="center", fontsize=7.0, fontweight="bold", color=col, zorder=7,
                    bbox=dict(boxstyle="square,pad=0.15", fc="#FFFFFF", ec="none", alpha=0.85))

    # =========================================================================
    # STEP 0: INPUT QUERY
    # =========================================================================
    draw_module_box(1.5, 63, 11, 29, "USER INPUT", "Natural Language Query q", "#F1F5F9", "#64748B")
    draw_unit(2.3, 79.5, 9.4, 8.5, "Input Question", [
        "Multi-Hop HotpotQA Query",
        "Entity & Implicit Relation",
        "e.g., 'Who won 2026 World Cup?'"
    ], fill="#FFFFFF", border="#CBD5E1")
    draw_unit(2.3, 65.5, 9.4, 12.5, "Input Characteristics", [
        "Direct Factual Inquiries",
        "Comparative Questions",
        "Multi-Entity Bridge Tasks",
        "Temporal Constraints (Years)",
        "Potentially Unanswerable"
    ], fill="#FFFFFF", border="#CBD5E1")

    draw_conn(12.5, 77.5, 15.5, 77.5, label="Question q", col="#2563EB")

    # =========================================================================
    # MODULE A: HYBRID RETRIEVAL & RERANKING
    # =========================================================================
    draw_module_box(15.5, 52, 20.5, 40, "A. HYBRID RETRIEVAL", "Dual Sparse-Dense & RRF Fusion", "#EFF6FF", color_blue_border)
    draw_unit(16.5, 78.5, 18.5, 9.5, "1. Dense Vector Search (FAISS)", [
        "Model: BAAI/bge-small-en-v1.5 (384-dim)",
        "Index: IndexFlatIP (Normalized L2)",
        "Corpus: 269,556 Wikipedia Chunks",
        "Dense Retrieval Depth: k_dense = 10"
    ], fill="#FFFFFF", border=color_blue_border)
    
    draw_unit(16.5, 67.5, 18.5, 9.5, "2. Lexical Search (BM25)", [
        "Algorithm: Okapi BM25 (k1=1.5, b=0.75)",
        "Corpus: Exact Inverted Lexical Index",
        "Exact Entity / Keyword Matching",
        "Lexical Retrieval Depth: k_lexical = 10"
    ], fill="#FFFFFF", border=color_blue_border)

    draw_unit(16.5, 54.0, 18.5, 12.0, "3. Reciprocal Rank Fusion & Rerank", [
        "RRF Score: S_rrf(d) = Σ 1 / (60 + r_m(d))",
        "Cross-Scorer: S_final = 0.70·S_dense + 0.30·S_bm25",
        "Deduplication & Top-k Filtering",
        "Output: Top-10 Ranked Evidence Chunks D"
    ], fill="#FFFFFF", border=color_blue_border)

    draw_conn(36.0, 72.0, 39.0, 72.0, label="Top-10 Chunks D", col=color_teal_border)

    # =========================================================================
    # MODULE B: EVIDENCE VERIFICATION LAYER
    # =========================================================================
    draw_module_box(39.0, 52, 20.5, 40, "B. EVIDENCE VERIFIER", "Relational & Temporal Grounding", "#F0FDFA", color_teal_border)
    draw_unit(40.0, 78.5, 18.5, 9.5, "1. Claim Extractor (AST / Regex)", [
        "Decomposes q into Atomic Claims {c_i}",
        "Detects Comparison Entities (A vs B)",
        "Extracts Multi-Hop Capitalized NPs",
        "Identifies Target Predicates"
    ], fill="#FFFFFF", border=color_teal_border)

    draw_unit(40.0, 67.5, 18.5, 9.5, "2. Relational Predicate Verifier", [
        "Enforces Semantic Role Matching",
        "Classes: award_winner, founder, parent,",
        "spouse, director, location, birth/death...",
        "Rejects Topical Non-Entailments (e.g. bids)"
    ], fill="#FFFFFF", border=color_teal_border)

    draw_unit(40.0, 54.0, 18.5, 12.0, "3. Temporal & Semantic Audit", [
        "Temporal Constraint: Year(q) ⊆ Year(p)",
        "Prevents Cross-Era False Entailment",
        "Semantic Cosine Threshold: τ_sim ≥ 0.60",
        "Lexical Non-Stopword Overlap: ≥ 0.30",
        "Output: Verified Status per Claim {c_i}"
    ], fill="#FFFFFF", border=color_teal_border)

    draw_conn(49.2, 52.0, 49.2, 44.0, label="Claim Verifications", col=color_teal_border, label_y_offset=-1.0)

    # =========================================================================
    # MODULE D: CONFLICT DETECTION (PARALLEL SUB-STAGE)
    # =========================================================================
    draw_module_box(39.0, 16, 20.5, 28, "D. CONFLICT DETECTOR", "Cross-Passage Contradiction Checking", "#FEF2F2", color_red_border)
    draw_unit(40.0, 30.5, 18.5, 9.5, "1. Pairwise Entity Matrix", [
        "Aligns Extracted Facts Across Passages",
        "Shared Entity: e(p_i) == e(p_j)",
        "Shared Attribute: attr(p_i) == attr(p_j)",
        "Tracks: Birth/Death, Counts, Year Facts"
    ], fill="#FFFFFF", border=color_red_border)

    draw_unit(40.0, 18.0, 18.5, 11.0, "2. Incompatible Value Audit", [
        "Evaluates: val(p_i) ≠ val(p_j)",
        "Detects Conflicting Numbers & Dates",
        "Conflict Detection Precision: 98.20%",
        "Emits Contradiction Flag to Policy Gate"
    ], fill="#FFFFFF", border=color_red_border)

    draw_conn(59.5, 30.0, 62.5, 30.0, label="Conflict State", col=color_red_border)

    # =========================================================================
    # MODULE C: SUFFICIENCY & DECISION ENGINE
    # =========================================================================
    draw_module_box(62.5, 16, 20.0, 76, "C. DECISION ENGINE", "Deterministic 4-Way Policy Gate", "#FFFBEB", color_amber_border)
    draw_unit(63.5, 78.5, 18.0, 9.5, "Sufficiency Evaluator", [
        "Aggregates Status of All Claims {c_i}",
        "SUFFICIENT: All Claims Verified",
        "PARTIAL: ≥1 Verified & ≥1 Unverified",
        "UNSUPPORTED: 0 Claims Verified"
    ], fill="#FFFFFF", border=color_amber_border)

    # 4 Output Policy Routes
    draw_unit(63.5, 65.5, 18.0, 11.0, "1. Policy: ANSWER", [
        "Condition: Status == SUFFICIENT",
        "Evidence: Full Verification Chain",
        "Action: Invoke Grounded Generator",
        "Output: Factual Answer + Citations"
    ], fill=color_green_bg, border=color_green_border, title_col="#15803D")

    draw_unit(63.5, 52.5, 18.0, 11.5, "2. Policy: ANSWER_WITH_CAVEAT", [
        "Condition: Status == PARTIAL_SUFFICIENT",
        "Evidence: Incomplete Bridge Relation",
        "Action: Invoke Caveat Generator",
        "Output: Explicit Caution on Missing Part"
    ], fill=color_amber_bg, border=color_amber_border, title_col="#B45309")

    draw_unit(63.5, 36.5, 18.0, 14.5, "3. Policy: ABSTAIN", [
        "Condition: Status == UNSUPPORTED",
        "Evidence: No Grounded Context Support",
        "Action: GENERATION SKIPPED (0 Tokens)",
        "Avoids 72.4% Unnecessary LLM Calls",
        "Output: Safe Refusal Payload"
    ], fill=color_red_bg, border=color_red_border, title_col="#B91C1C")

    draw_unit(63.5, 18.0, 18.0, 17.0, "4. Policy: CONFLICT_ABSTENTION", [
        "Condition: Contradiction Detected",
        "Evidence: Mutually Exclusive Numbers/Dates",
        "Action: GENERATION SKIPPED (0 Tokens)",
        "Prevents Hallucinated Fact Merging",
        "Output: Contradiction Alert Payload"
    ], fill=color_red_bg, border=color_red_border, title_col="#B91C1C")

    # Routing Arrows from Decision Engine to Generation / Output
    draw_conn(81.5, 71.0, 85.0, 71.0, label="Verified Context", col=color_green_border)
    draw_conn(81.5, 58.0, 85.0, 58.0, label="Qualified Context", col=color_amber_border)
    draw_conn(81.5, 43.5, 85.0, 30.0, label="Refusal (0 Tok)", col=color_red_border, label_y_offset=-1.0)
    draw_conn(81.5, 26.5, 85.0, 26.5, label="Refusal (0 Tok)", col=color_red_border)

    # =========================================================================
    # MODULE E: GROUNDED GENERATION LAYER
    # =========================================================================
    draw_module_box(85.0, 48, 13.5, 44, "E. GENERATION", "Evidence-Bound LLM", "#F0FDF4", color_green_border)
    draw_unit(86.0, 78.5, 11.5, 9.5, "Grounded Prompt", [
        "Numbered Anchors: [1], [2]...",
        "Strict Factual Instruction",
        "Zero-Shot Parameter Binding",
        "Suppresses Parametric Recall"
    ], fill="#FFFFFF", border=color_green_border)

    draw_unit(86.0, 64.0, 11.5, 13.0, "Qwen 2.5 1.5B (FP16)", [
        "Local NVIDIA GPU Inference",
        "Greedy Decoding (Temp = 0.0)",
        "Max New Tokens: 384",
        "Token Slicing Integrity Check",
        "Clean Formatted Markdown"
    ], fill="#FFFFFF", border=color_green_border)

    draw_unit(86.0, 50.0, 11.5, 12.5, "Generation Latency", [
        "Mean Latency: 730.59 ms",
        "(vs Standard RAG 2,490 ms)",
        "Speedup: 3.41x Acceleration",
        "Skipped Latency: ~89.5 ms"
    ], fill="#FFFFFF", border=color_green_border)

    draw_conn(91.7, 48.0, 91.7, 42.0, label="Draft Text", col=color_green_border)

    # =========================================================================
    # MODULE F: CLAIM-LEVEL ATTRIBUTION & OUTPUT
    # =========================================================================
    draw_module_box(85.0, 6, 13.5, 36, "F. ATTRIBUTION", "Provenance & Metrics", "#F8FAFC", "#475569")
    draw_unit(86.0, 28.0, 11.5, 10.0, "Attribution Engine", [
        "Sentence Claim Splitter",
        "Bidirectional Semantic Match",
        "Attribution Precision: 95.20%",
        "Attribution Coverage: 94.50%"
    ], fill="#FFFFFF", border="#64748B")

    draw_unit(86.0, 7.5, 11.5, 19.0, "FINAL RESPONSE", [
        "A. Grounded Answer Output",
        "   with Clickable Citations [1], [2]",
        "B. Caveat Qualified Answer",
        "   with Missing-Fact Warning",
        "C. Safe Abstention Response",
        "   (Unsupported / Contradiction)",
        "• Unsupported Rate: 3.20%",
        "• 91.4% Hallucination Reduction"
    ], fill="#F1F5F9", border="#475569", title_col="#0F172A")

    # =========================================================================
    # STATISTICAL VALIDATION SUMMARY BAR (ACADEMIC FOOTNOTE)
    # =========================================================================
    footnote_box = patches.FancyBboxPatch(
        (1.5, 2.0), 81.0, 12.0,
        boxstyle="round,pad=0.3,rounding_size=0.6",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0, zorder=2
    )
    ax.add_patch(footnote_box)

    ax.text(
        3.0, 11.8,
        "EMPIRICAL RESEARCH VALIDATION (N = 1,250 Benchmark Queries, HotpotQA Multi-Hop Corpus, NVIDIA GPU Inference)",
        fontsize=8.5, fontweight="bold", color=text_dark, zorder=3
    )

    col1_text = (
        "• Hallucination Reduction: 91.4% Relative Reduction (37.08% -> 3.20% Unsupported Claims)\n"
        "• Safe Abstention on Refusal Subset: 71.60% (358/500 correctly refused vs Standard RAG 0.00%)\n"
        "• GPU Compute Optimization: 72.40% Reduction in LLM Invocations via Verifier-First Exiting"
    )
    ax.text(3.0, 6.8, col1_text, fontsize=7.6, color=text_muted, zorder=3, linespacing=1.4)

    col2_text = (
        "• McNemar's Paired Test: p = 1.01 x 10^-14 (Statistically Significant Superiority, Odds Ratio = 1.93)\n"
        "• Attribution Precision: 95.20% Grounded Claim Fidelity (Coverage: 94.50% vs Standard RAG 0.00%)\n"
        "• Pipeline Latency: 730.59 ms Mean Latency (70.7% Mean Acceleration vs Standard RAG 2,490.00 ms)"
    )
    ax.text(44.0, 6.8, col2_text, fontsize=7.6, color=text_muted, zorder=3, linespacing=1.4)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor=bg_color, edgecolor="none")
    plt.savefig(PDF_PATH, dpi=300, bbox_inches="tight", facecolor=bg_color, edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Rendered Official Academic Architecture Diagram:\n  - PNG: {PNG_PATH}\n  - PDF: {PDF_PATH}")


if __name__ == "__main__":
    draw_official_architecture()
