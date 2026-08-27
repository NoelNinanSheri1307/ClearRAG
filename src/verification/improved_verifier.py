"""Improved Evidence Verification Pipeline for ClearRAG.

Integrates semantic embedding similarity, non-stopword content alignment,
multi-hop entity relation checking, and attribute-aware contradiction detection.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from src.retrieval.embedder import BGEEmbedder
from src.verification.claims import Claim, ClaimType
from src.verification.contradiction import ContradictionDetector
from src.verification.evidence_matching import SemanticEvidenceMatcher
from src.verification.models import ClaimVerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


class ImprovedEvidenceVerifier:
    """Advanced evidence verification engine combining semantic, lexical, and contradiction reasoning."""

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        matcher: Optional[SemanticEvidenceMatcher] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        min_semantic_sim: float = 0.65,
        min_content_overlap_ratio: float = 0.35,
        enable_contradiction: bool = True,
    ):
        """Initialize ImprovedEvidenceVerifier.

        Args:
            embedder: BGEEmbedder instance.
            matcher: SemanticEvidenceMatcher instance.
            contradiction_detector: ContradictionDetector instance.
            min_semantic_sim: Minimum semantic similarity threshold.
            min_content_overlap_ratio: Minimum non-stopword content overlap ratio.
            enable_contradiction: Whether to execute contradiction detection.
        """
        self.embedder = embedder
        self.matcher = matcher or SemanticEvidenceMatcher(
            embedder=self.embedder,
            min_semantic_sim=min_semantic_sim,
            min_content_overlap_ratio=min_content_overlap_ratio,
        )
        self.contradiction_detector = contradiction_detector or ContradictionDetector()
        self.enable_contradiction = enable_contradiction

    def verify_claim(
        self,
        claim: Claim,
        evidence_chunks: List[Dict[str, Any]],
    ) -> ClaimVerificationResult:
        """Verify a single claim against retrieved evidence chunks.

        Args:
            claim: Extracted Claim object.
            evidence_chunks: List of retrieved evidence chunk dictionaries.

        Returns:
            Structured ClaimVerificationResult.
        """
        if not evidence_chunks:
            return ClaimVerificationResult(
                claim=claim,
                status=VerificationStatus.UNSUPPORTED,
                confidence_score=0.0,
                reason="No retrieved evidence chunks available.",
            )

        # 1. Contradiction Detection across candidate passages
        if self.enable_contradiction and claim.target_entities:
            has_conflict, c_ids, c_texts, c_reason = self.contradiction_detector.detect_conflict(
                target_entities=claim.target_entities,
                predicate=claim.predicate,
                evidence_chunks=evidence_chunks,
            )
            if has_conflict:
                return ClaimVerificationResult(
                    claim=claim,
                    status=VerificationStatus.CONFLICTING,
                    conflicting_evidence_ids=c_ids,
                    conflicting_evidence_texts=c_texts,
                    confidence_score=1.0,
                    reason=c_reason,
                )

        # 2. Passage-by-Passage Support Evaluation
        supporting_ids = []
        supporting_texts = []
        max_support_score = 0.0
        explanations = []

        for chunk in evidence_chunks:
            cid = str(chunk.get("chunk_id", chunk.get("rank", "unknown")))
            text = chunk.get("text", "")
            title = chunk.get("document_title", "")

            score, is_supported, expl = self.matcher.evaluate_passage_support(
                claim=claim,
                passage_text=text,
                passage_title=title,
            )

            if is_supported:
                supporting_ids.append(cid)
                supporting_texts.append(text)
                if score > max_support_score:
                    max_support_score = score
                    explanations.append(expl)

        # 3. Formulate Support Status
        if supporting_ids:
            return ClaimVerificationResult(
                claim=claim,
                status=VerificationStatus.SUPPORTED,
                supporting_evidence_ids=supporting_ids,
                supporting_evidence_texts=supporting_texts,
                confidence_score=round(max_support_score, 4),
                reason=f"Claim supported by {len(supporting_ids)} chunk(s). Best match: {explanations[-1] if explanations else ''}",
            )
        else:
            return ClaimVerificationResult(
                claim=claim,
                status=VerificationStatus.UNSUPPORTED,
                confidence_score=0.0,
                reason=f"Retrieved evidence lacks required factual/predicate grounding for '{claim.predicate}'.",
            )

    def verify_claims(
        self,
        claims: List[Claim],
        evidence_chunks: List[Dict[str, Any]],
    ) -> List[ClaimVerificationResult]:
        """Verify multiple extracted claims against retrieved evidence chunks."""
        return [self.verify_claim(c, evidence_chunks) for c in claims]
