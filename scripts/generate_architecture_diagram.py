"""ClearRAG Formal Academic Architecture Schematic Generator (Black & White).

Generates a formal black-and-white, top-to-bottom schematic diagram in Times New Roman
with large, highly legible typography and zero clipping, designed specifically for IEEE/ACM publication standards,
patent line-art specifications, and academic thesis submissions.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set global font to Times New Roman / Serif
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_PATH = OUTPUT_DIR / "clearrag_architecture_schematic.png"
PDF_PATH = OUTPUT_DIR / "clearrag_architecture_schematic.pdf"


def draw_bw_top_to_bottom_architecture():
    # Large-format vertical portrait canvas (15 x 22.5 in) with generous margin bounds
    fig, ax = plt.subplots(figsize=(15, 22.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 180)
    ax.axis("off")

    # Strict Monochrome Palette
    bg_color = "#FFFFFF"
    box_bg = "#FFFFFF"
    box_edge = "#000000"
    text_color = "#000000"

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # -------------------------------------------------------------
    # HEADER TITLE
    # -------------------------------------------------------------
    ax.text(
        50, 175.0,
        "Figure 1: Architecture of the ClearRAG System",
        ha="center", va="center", fontsize=18, fontweight="bold", color=text_color
    )
    ax.text(
        50, 171.8,
        "Top-to-bottom procedural workflow for evidence-aware selective generation.",
        ha="center", va="center", fontsize=13, style="italic", color=text_color
    )

    # Helper function for primary rectangular blocks
    def draw_box(x, y, w, h, title="", lines=None, lw=1.4, ls="-", fill=box_bg, title_size=12.5, line_size=10.8, line_spacing=2.6):
        box = patches.Rectangle(
            (x, y), w, h,
            facecolor=fill, edgecolor=box_edge, linewidth=lw, linestyle=ls, zorder=2
        )
        ax.add_patch(box)
        if title:
            ax.text(
                x + w / 2, y + h - 2.8, title,
                ha="center", va="center", fontsize=title_size, fontweight="bold", color=text_color, zorder=3
            )
        if lines:
            start_y = y + h - 5.8
            for i, line in enumerate(lines):
                ax.text(
                    x + 2.2, start_y - (i * line_spacing), f"• {line}",
                    ha="left", va="center", fontsize=line_size, color=text_color, zorder=3
                )
        return box

    # Helper function for decision diamond
    def draw_diamond(cx, cy, w, h, text=""):
        diamond = patches.Polygon(
            [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
            closed=True, facecolor=box_bg, edgecolor=box_edge, linewidth=1.4, zorder=2
        )
        ax.add_patch(diamond)
        if text:
            ax.text(cx, cy, text, ha="center", va="center", fontsize=11.5, fontweight="bold", color=text_color, zorder=3)

    # Helper for directional connecting arrows
    def draw_arrow(x1, y1, x2, y2, label="", ls="-", label_offset=(0, 0), ha="center", label_size=10.2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.5", color=box_edge, lw=1.3, linestyle=ls),
            zorder=4
        )
        if label:
            mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
            ax.text(
                mx, my, label,
                ha=ha, va="center", fontsize=label_size, color=text_color, zorder=5,
                bbox=dict(boxstyle="square,pad=0.2", fc=bg_color, ec="none")
            )

    # =========================================================================
    # 0. INPUT
    # =========================================================================
    draw_box(18, 156.0, 64, 11.5, title="1. USER QUERY INPUT", lines=[
        "Input Question q (Natural Language Multi-Hop Factual Inquiry)",
        "Contains Implicit Relations, Multi-Entity Comparisons, and Temporal Constraints"
    ], title_size=13.0, line_size=11.0, line_spacing=2.6)

    # =========================================================================
    # A. HYBRID RETRIEVAL & RERANKING
    # =========================================================================
    # Sub-box 1: Dense Retrieval
    draw_box(6, 134.0, 42, 16.0, title="A1. Dense Vector Search", lines=[
        "Model: BAAI/bge-small-en-v1.5 (384-dim)",
        "Index: FAISS IndexFlatIP (Normalized L2)",
        "Retrieval Depth: k_dense = 10 passages"
    ], title_size=12.0, line_size=10.5, line_spacing=2.6)

    # Sub-box 2: Sparse Retrieval
    draw_box(52, 134.0, 42, 16.0, title="A2. Sparse Lexical Search", lines=[
        "Algorithm: Okapi BM25 (k1 = 1.5, b = 0.75)",
        "Index: Inverted Term Index (269,556 chunks)",
        "Retrieval Depth: k_lexical = 10 passages"
    ], title_size=12.0, line_size=10.5, line_spacing=2.6)

    # Arrows from Query to Dual Retrievers
    draw_arrow(50, 156.0, 27, 150.0, label="")
    draw_arrow(50, 156.0, 73, 150.0, label="")

    # Sub-box 3: Fusion & Reranking
    draw_box(8, 112.0, 84, 16.0, title="A3. Reciprocal Rank Fusion & Cross-Scorer Reranking", lines=[
        "Reciprocal Rank Fusion (RRF): S_RRF(d) = Σ 1 / (60 + rank_m(d))",
        "Convex Combination Reranking: S_final(d) = 0.70 · S_dense(d) + 0.30 · S_BM25(d)",
        "Output: Top-10 Ranked Candidate Evidence Chunks D = {p_1, p_2, ..., p_10}"
    ], title_size=12.5, line_size=10.8, line_spacing=2.6)

    draw_arrow(27, 134.0, 36, 128.0, label="")
    draw_arrow(73, 134.0, 64, 128.0, label="")
    draw_arrow(50, 112.0, 50, 105.0, label="Top-10 Evidence Chunks D")

    # =========================================================================
    # B. EVIDENCE VERIFICATION LAYER
    # =========================================================================
    draw_box(6, 78.0, 88, 25.0, title="B. EVIDENCE VERIFICATION LAYER", lines=[
        "B1. Claim Extraction: Decomposes question q into atomic claims {c_1, c_2, ..., c_m}",
        "B2. Relational Predicate Verification: Validates action roles (winner, founder, author, parent, location...)",
        "    Rejects topical false positives where context mentions entities without fulfilling requested relation",
        "B3. Temporal Year Constraint: Enforces $\\mathrm{Year}(q) \\subseteq \\mathrm{Year}(p)$ to prevent cross-era historical mismatches",
        "B4. Semantic Alignment: Requires Cosine Similarity ($\\tau_{sim} \\geq 0.60$) and Lexical Overlap ($\\geq 0.30$)",
        "Output: Verified Status and Evidential Support Chain for each extracted claim"
    ], title_size=13.0, line_size=10.6, line_spacing=2.7)

    draw_arrow(50, 78.0, 50, 71.0, label="Verified Claim Records")

    # =========================================================================
    # D. CONFLICT DETECTION & SUFFICIENCY ENGINE
    # =========================================================================
    draw_box(6, 49.0, 88, 20.0, title="C & D. CONFLICT DETECTION & SUFFICIENCY ENGINE", lines=[
        "D. Cross-Passage Conflict Detection: Audits pairwise entity-attribute assertions across passages",
        "   Criterion: Entity(p_i) == Entity(p_j) and Attribute(p_i) == Attribute(p_j) and Value(p_i) != Value(p_j)",
        "   Detects incompatible numerical statistics, population counts, construction dates, and birth/death years",
        "C. Sufficiency Evaluation: Aggregates verified claim statuses into overall Sufficiency Status",
        "   (Overall Status: SUFFICIENT | PARTIAL_SUFFICIENT | UNSUPPORTED | CONFLICTING)"
    ], title_size=13.0, line_size=10.6, line_spacing=2.7)

    draw_arrow(50, 49.0, 50, 42.5, label="")

    # =========================================================================
    # DECISION GATE (Diamond)
    # =========================================================================
    draw_diamond(50, 36.0, 38, 12.0, text="4-Way Policy Gate\n(Decision Engine)")

    # Output Branches from Decision Gate
    draw_arrow(31, 36.0, 16, 36.0, label="")
    draw_arrow(16, 36.0, 16, 25.5, label="Unsupported /\nContradiction", label_offset=(0, 3.2), ha="right")

    draw_arrow(50, 30.0, 50, 25.5, label="Partial Evidence\n(1/2 Claims)", label_offset=(0.8, 0), ha="left")

    draw_arrow(69, 36.0, 84, 36.0, label="")
    draw_arrow(84, 36.0, 84, 25.5, label="Fully Supported\nEvidence", label_offset=(0, 3.2), ha="left")

    # =========================================================================
    # E. GENERATION & REFUSAL PAYLOADS
    # =========================================================================
    # Branch 1: Safe Abstention (Left)
    draw_box(3, 11.5, 27, 14.0, title="3 & 4. SAFE REFUSAL", lines=[
        "GENERATION SKIPPED",
        "0 GPU Tokens Generated",
        "Safe Refusal Payload",
        "Preserves 72.4% Compute"
    ], ls="--", title_size=11.2, line_size=9.8, line_spacing=2.3)

    # Branch 2: Answer with Caveat (Middle)
    draw_box(33, 11.5, 34, 14.0, title="2. CAVEAT GENERATION", lines=[
        "Qwen 2.5 1.5B (FP16, Temp=0.0)",
        "Caveat Prompt Builder",
        "Emits Qualified Factual Answer",
        "Explicit Warning on Missing Part"
    ], title_size=11.2, line_size=9.8, line_spacing=2.3)

    # Branch 3: Grounded Answer Generation (Right)
    draw_box(70, 11.5, 27, 14.0, title="1. GROUNDED GEN", lines=[
        "Qwen 2.5 1.5B (FP16)",
        "Grounded Context [1], [2]",
        "Greedy (Temp = 0.0)",
        "Max New Tokens: 384"
    ], title_size=11.2, line_size=9.8, line_spacing=2.3)

    # Arrows to Final Output & Attribution
    draw_arrow(50, 11.5, 50, 8.5, label="")
    draw_arrow(83.5, 11.5, 68, 8.5, label="")

    # =========================================================================
    # F. CLAIM-LEVEL ATTRIBUTION & FINAL DELIVERABLE
    # =========================================================================
    draw_box(12, 1.5, 76, 7.0, title="F. CLAIM-LEVEL ATTRIBUTION & FINAL OUTPUT", lines=[
        "Propositional Sentence Decomposition & Bidirectional Semantic Alignment against Source Chunks",
        "Grounding Precision: 95.20% | Attribution Coverage: 94.50% | Verified Citations [1], [2]"
    ], title_size=11.8, line_size=10.0, line_spacing=2.1)

    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor=bg_color, edgecolor="none")
    plt.savefig(PDF_PATH, dpi=300, bbox_inches="tight", facecolor=bg_color, edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Rendered Clean B&W Academic Architecture Diagram (Large Font, Full Bounds):\n  - PNG: {PNG_PATH}\n  - PDF: {PDF_PATH}")


if __name__ == "__main__":
    draw_bw_top_to_bottom_architecture()
