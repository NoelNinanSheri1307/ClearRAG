"""ClearRAG manual inspection CLI.

Runs the full ClearRAG pipeline on a single question and prints
a detailed breakdown of every pipeline stage for human inspection.

Usage:
    python scripts/test_clearrag.py --question "Which genus has more species, Bactris or Epigaea?"
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.clearrag.decision import ClearRAGDecisionEngine
from src.clearrag.pipeline import ClearRAGPipeline
from src.generation.llm_generator import LLMGenerator
from src.retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-' * 65}")
    print(f"  {title}")
    print(f"{'-' * 65}")


def main():
    parser = argparse.ArgumentParser(description="ClearRAG manual inspection CLI.")
    parser.add_argument("--question", type=str,
                        default="Which genus has more species, Bactris or Epigaea?",
                        help="Question to process through ClearRAG")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K evidence chunks")
    parser.add_argument("--index_path", type=str, default="data/processed/faiss_index.bin")
    parser.add_argument("--metadata_path", type=str, default="data/processed/index_metadata.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    print("=" * 65)
    print("  CLEAR RAG — DECISION + ABSTENTION LAYER INSPECTION")
    print("=" * 65)

    # ----------------------------------------------
    # Initialize components
    # ----------------------------------------------
    index_path = Path(args.index_path)
    metadata_path = Path(args.metadata_path)

    if not index_path.exists() or not metadata_path.exists():
        logger.error("FAISS index not found. Cannot run ClearRAG pipeline.")
        sys.exit(1)

    logger.info("Loading retriever...")
    retriever = Retriever.from_saved_index(index_path, metadata_path, default_top_k=args.top_k)

    logger.info(f"Loading LLM generator ({args.model})...")
    generator = LLMGenerator(model_name=args.model)

    decision_engine = ClearRAGDecisionEngine()
    pipeline = ClearRAGPipeline(
        retriever=retriever,
        generator=generator,
        decision_engine=decision_engine,
        default_top_k=args.top_k,
    )

    # ----------------------------------------------
    # Execute pipeline
    # ----------------------------------------------
    print_section("QUESTION")
    print(f"  {args.question}")

    result = pipeline.answer(args.question, top_k=args.top_k)

    # ----------------------------------------------
    # Print Results
    # ----------------------------------------------

    print_section("RETRIEVED EVIDENCE")
    for ev in result.retrieved_evidence:
        print(f"  Rank {ev.get('rank', '?')} | {ev.get('document_title', 'Untitled')} | Score: {ev.get('score', 0):.4f}")
        print(f"  Text: {ev.get('text', '')[:200]}")
        print()

    print_section("EXTRACTED CLAIMS")
    for claim in result.claims:
        print(f"  ID: {claim.get('claim_id', '?')}")
        print(f"  Text: {claim.get('text', '')}")
        print(f"  Type: {claim.get('claim_type', '?')}")
        print(f"  Predicate: {claim.get('predicate', '?')}")
        print(f"  Entities: {claim.get('target_entities', [])}")
        print()

    print_section("CLAIM VERIFICATION")
    for cr in result.claim_results:
        print(f"  Claim: {cr.get('claim', {}).get('claim_id', '?')}")
        print(f"  Status: {cr.get('status', '?')}")
        print(f"  Support IDs: {cr.get('supporting_evidence_ids', [])}")
        print(f"  Conflict IDs: {cr.get('conflicting_evidence_ids', [])}")
        print(f"  Reason: {cr.get('reason', '')}")
        print()

    print_section("SUFFICIENCY")
    print(f"  Status: {result.sufficiency_status.value}")
    print(f"  Explanation: {result.explanation}")

    print_section("CLEARRAG DECISION")
    print(f"  Decision: {result.decision.value}")
    print(f"  Is Abstention: {result.is_abstention}")
    print(f"  Is Caveated: {result.is_caveated}")

    print_section("ANSWER")
    print(f"  {result.answer}")

    if result.abstention_reason:
        print_section("ABSTENTION/CAVEAT REASON")
        print(f"  {result.abstention_reason}")

    if result.caveat_text:
        print_section("CAVEAT TEXT")
        print(f"  {result.caveat_text}")

    print_section("PROVENANCE")
    print(f"  Retrieved chunks: {len(result.retrieved_evidence)}")
    print(f"  Supporting chunks: {len(result.supporting_evidence)}")
    print(f"  Conflicting chunks: {len(result.conflicting_evidence)}")
    print(f"  Claims extracted: {len(result.claims)}")
    print(f"  Model: {result.metadata.get('model_name', '?')}")
    print(f"  Embedder: {result.metadata.get('embedding_model', '?')}")

    print_section("LATENCIES")
    print(f"  Retrieval:      {result.retrieval_latency_ms:>10.2f} ms")
    print(f"  Verification:   {result.verification_latency_ms:>10.2f} ms")
    print(f"  Generation:     {result.generation_latency_ms:>10.2f} ms")
    print(f"  Total:          {result.total_latency_ms:>10.2f} ms")

    print("\n" + "=" * 65)
    print("  INSPECTION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()

