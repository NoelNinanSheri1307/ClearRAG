"""Structured result models for ClearRAG evidence verification layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.verification.claims import Claim


class VerificationStatus(str, Enum):
    """Status of evidence verification for an individual claim."""
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONFLICTING = "CONFLICTING"


class SufficiencyStatus(str, Enum):
    """Overall evidence sufficiency status for a question."""
    FULLY_SUPPORTED = "FULLY_SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONFLICTING = "CONFLICTING"


@dataclass
class ClaimVerificationResult:
    """Provenance-rich verification result for a single claim."""
    claim: Claim
    status: VerificationStatus
    supporting_evidence_ids: List[str] = field(default_factory=list)
    supporting_evidence_texts: List[str] = field(default_factory=list)
    conflicting_evidence_ids: List[str] = field(default_factory=list)
    conflicting_evidence_texts: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert claim verification result to dictionary representation."""
        return {
            "claim": self.claim.to_dict(),
            "status": self.status.value,
            "supporting_evidence_ids": self.supporting_evidence_ids,
            "supporting_evidence_texts": self.supporting_evidence_texts,
            "conflicting_evidence_ids": self.conflicting_evidence_ids,
            "conflicting_evidence_texts": self.conflicting_evidence_texts,
            "confidence_score": self.confidence_score,
            "reason": self.reason,
        }


@dataclass
class VerificationResult:
    """Structured, inspectable result for end-to-end evidence verification."""
    question: str
    claims: List[Claim]
    retrieved_evidence: List[Dict[str, Any]]
    claim_results: List[ClaimVerificationResult]
    overall_status: SufficiencyStatus
    explanation: str
    verifier_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert full verification result to dictionary representation."""
        return {
            "question": self.question,
            "claims": [c.to_dict() for c in self.claims],
            "retrieved_evidence_count": len(self.retrieved_evidence),
            "claim_results": [r.to_dict() for r in self.claim_results],
            "overall_status": self.overall_status.value,
            "explanation": self.explanation,
            "verifier_metadata": self.verifier_metadata,
        }
