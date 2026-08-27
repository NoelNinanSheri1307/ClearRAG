"""Evidence Verification Baseline for ClearRAG Controlled Evaluation.

This baseline performs deterministic claim extraction, evidence verification, and sufficiency classification.
It does NOT generate natural language answers or call the LLM generator.
"""

from dataclasses import asdict, dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional

from src.retrieval.retriever import Retriever
from src.verification.claim_extractor import RuleBasedClaimExtractor
from src.verification.claims import Claim
from src.verification.evidence_verifier import EvidenceVerifier
from src.verification.models import (
    ClaimVerificationResult,
    SufficiencyStatus,
    VerificationResult,
    VerificationStatus,
)
from src.verification.sufficiency import SufficiencyEngine

logger = logging.getLogger(__name__)


@dataclass
class VerificationBaselineResult:
    """Structured output container for Evidence Verification baseline inference."""

    question: str
    claims: List[Dict[str, Any]]
    claim_verification_results: List[Dict[str, Any]]
    sufficiency_status: str
    explanation: str
    retrieved_chunk_ids: List[str]
    provenance: Dict[str, Any]
    latency_retrieval_ms: float
    latency_verification_ms: float
    latency_total_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class VerificationBaselinePipeline:
    """Standalone Evidence Verification baseline pipeline.

    Computes sufficiency classification and evidence support without answer generation.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        claim_extractor: Optional[RuleBasedClaimExtractor] = None,
        evidence_verifier: Optional[EvidenceVerifier] = None,
        sufficiency_engine: Optional[SufficiencyEngine] = None,
        default_top_k: int = 5,
    ):
        """Initialize Verification baseline pipeline."""
        self.retriever = retriever or Retriever()
        self.claim_extractor = claim_extractor or RuleBasedClaimExtractor()
        self.evidence_verifier = evidence_verifier or EvidenceVerifier()
        self.sufficiency_engine = sufficiency_engine or SufficiencyEngine()
        self.default_top_k = default_top_k

    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> VerificationBaselineResult:
        """Run verification baseline on input question.

        CRITICAL: Takes ONLY question and top_k. No ground truth or metadata.

        Args:
            question: Natural language question.
            top_k: Number of chunks to retrieve.

        Returns:
            VerificationBaselineResult with sufficiency classification.
        """
        k = top_k if top_k is not None else self.default_top_k
        t_start = time.perf_counter()

        # 1. Retrieval
        t_ret_start = time.perf_counter()
        retrieved_evidence = self.retriever.retrieve(question, top_k=k)
        t_ret_end = time.perf_counter()
        latency_retrieval_ms = (t_ret_end - t_ret_start) * 1000.0

        # 2. Claim Extraction
        t_ver_start = time.perf_counter()
        claims = self.claim_extractor.extract_claims(question)

        # 3. Evidence Verification
        claim_results = self.evidence_verifier.verify_claims(claims, retrieved_evidence)

        # 4. Sufficiency Evaluation
        sufficiency_res: VerificationResult = self.sufficiency_engine.evaluate_sufficiency(
            question=question,
            claims=claims,
            claim_results=claim_results,
            retrieved_evidence=retrieved_evidence,
        )
        t_ver_end = time.perf_counter()
        latency_verification_ms = (t_ver_end - t_ver_start) * 1000.0
        latency_total_ms = (t_ver_end - t_start) * 1000.0

        retrieved_chunk_ids = [
            chunk.get("chunk_id", f"chunk_{i}")
            for i, chunk in enumerate(retrieved_evidence)
        ]

        # Count evidence provenance
        num_supporting = sum(
            len(r.supporting_evidence_ids) for r in claim_results
        )
        num_conflicting = sum(
            len(r.conflicting_evidence_ids) for r in claim_results
        )

        provenance = {
            "num_claims": len(claims),
            "num_supporting_evidence": num_supporting,
            "num_conflicting_evidence": num_conflicting,
            "retriever_model": getattr(getattr(self.retriever, "embedder", None), "model_name", "BAAI/bge-small-en-v1.5"),
        }

        return VerificationBaselineResult(
            question=question,
            claims=[c.to_dict() for c in claims],
            claim_verification_results=[r.to_dict() for r in claim_results],
            sufficiency_status=sufficiency_res.overall_status.value,
            explanation=sufficiency_res.explanation,
            retrieved_chunk_ids=retrieved_chunk_ids,
            provenance=provenance,
            latency_retrieval_ms=round(latency_retrieval_ms, 2),
            latency_verification_ms=round(latency_verification_ms, 2),
            latency_total_ms=round(latency_total_ms, 2),
        )
