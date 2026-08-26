"""CLI testing and inspection script for ClearRAG Evidence Verification layer."""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import Retriever
from src.verification.claim_extractor import RuleBasedClaimExtractor
from src.verification.evidence_verifier import EvidenceVerifier
from src.verification.sufficiency import SufficiencyEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test and inspect ClearRAG evidence verification.")
    parser.add_argument("--question", type=str, default="Which genus has more species, Bactris or Epigaea?", help="Question string")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K evidence chunks to retrieve")
    parser.add_argument("--index_path", type=str, default="data/processed/faiss_index.bin", help="FAISS index path")
    parser.add_argument("--metadata_path", type=str, default="data/processed/index_metadata.json", help="Index metadata path")
    args = parser.parse_args()

    print("=================================================================")
    print("CLEAR RAG — EVIDENCE VERIFICATION CLI INSPECTION")
    print("=================================================================")
    print(f"Question: {args.question}\n")

    # 1. Load retriever
    index_path = Path(args.index_path)
    metadata_path = Path(args.metadata_path)
    if index_path.exists() and metadata_path.exists():
        logger.info(f"Loading retriever from index '{index_path}'...")
        retriever = Retriever.from_saved_index(index_path, metadata_path, default_top_k=args.top_k)
        retrieved_evidence = retriever.retrieve(args.question, top_k=args.top_k)
    else:
        logger.warning("FAISS index not found. Running with mock evidence for CLI inspection.")
        retrieved_evidence = [
            {"rank": 1, "chunk_id": "chunk_1", "score": 0.82, "document_title": "Bactris", "text": "Bactris is a genus of about 75 species of palms."},
            {"rank": 2, "chunk_id": "chunk_2", "score": 0.76, "document_title": "Epigaea", "text": "Epigaea is a genus of 3 species of flowering plants."}
        ]

    # Print Retrieved Evidence
    print("Retrieved Evidence")
    print("------------------")
    for doc in retrieved_evidence:
        print(f"Rank {doc.get('rank', 'N/A')} | Doc: {doc.get('document_title', 'Untitled')} | Score: {doc.get('score', 0.0):.4f}")
        print(f"Text: {doc.get('text', '')}\n")

    # 2. Extract Claims
    extractor = RuleBasedClaimExtractor()
    claims = extractor.extract_claims(args.question)

    # 3. Verify Claims
    verifier = EvidenceVerifier()
    claim_results = [verifier.verify_claim(claim, retrieved_evidence) for claim in claims]

    print("Claims")
    print("------------------")
    for res in claim_results:
        print(f"Claim ID   : {res.claim.claim_id}")
        print(f"Text       : {res.claim.text}")
        print(f"Status     : {res.status.value}")
        print(f"Support IDs: {res.supporting_evidence_ids}")
        print(f"ConflictIDs: {res.conflicting_evidence_ids}")
        print(f"Reason     : {res.reason}\n")

    # 4. Evaluate Sufficiency
    sufficiency_engine = SufficiencyEngine()
    final_result = sufficiency_engine.evaluate_sufficiency(
        question=args.question,
        claims=claims,
        claim_results=claim_results,
        retrieved_evidence=retrieved_evidence,
    )

    print("Final Verification")
    print("-------------------")
    print(f"Status: {final_result.overall_status.value}")
    print(f"Reason: {final_result.explanation}")
    print("=================================================================")


if __name__ == "__main__":
    main()
