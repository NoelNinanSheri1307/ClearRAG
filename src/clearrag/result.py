"""Structured result model for ClearRAG pipeline output.

Contains full provenance, latency breakdown, decision audit trail,
and evidence verification details for every ClearRAG query.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.clearrag.decision import ClearRAGDecision
from src.verification.models import SufficiencyStatus


@dataclass
class ClearRAGResult:
    """Structured, inspectable result for a complete ClearRAG pipeline execution.

    Preserves full provenance chain:
        Question -> Retrieved Evidence -> Claims -> Claim Verification
        -> Sufficiency -> Decision -> (Conditional Generation) -> Answer
    """

    # Core fields
    question: str
    answer: str
    decision: ClearRAGDecision
    sufficiency_status: SufficiencyStatus

    # Claim-level detail
    claims: List[Dict[str, Any]] = field(default_factory=list)
    claim_results: List[Dict[str, Any]] = field(default_factory=list)

    # Evidence provenance
    retrieved_evidence: List[Dict[str, Any]] = field(default_factory=list)
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    conflicting_evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Explanations
    explanation: str = ""
    abstention_reason: str = ""
    caveat_text: str = ""

    # Latency breakdown (milliseconds)
    retrieval_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Fine-grained attribution and grounding
    attributions: List[Dict[str, Any]] = field(default_factory=list)
    grounding_metrics: Dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ClearRAGResult to a JSON-compatible dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "decision": self.decision.value,
            "sufficiency_status": self.sufficiency_status.value,
            "claims": self.claims,
            "claim_results": self.claim_results,
            "retrieved_evidence_count": len(self.retrieved_evidence),
            "supporting_evidence_count": len(self.supporting_evidence),
            "conflicting_evidence_count": len(self.conflicting_evidence),
            "explanation": self.explanation,
            "abstention_reason": self.abstention_reason,
            "caveat_text": self.caveat_text,
            "attributions": self.attributions,
            "grounding_metrics": self.grounding_metrics,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "verification_latency_ms": round(self.verification_latency_ms, 2),
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "metadata": self.metadata,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Serialize with full evidence text included (for detailed inspection)."""
        d = self.to_dict()
        d["retrieved_evidence"] = self.retrieved_evidence
        d["supporting_evidence"] = self.supporting_evidence
        d["conflicting_evidence"] = self.conflicting_evidence
        return d

    @property
    def is_abstention(self) -> bool:
        """Check if this result is an abstention."""
        return self.decision in (
            ClearRAGDecision.ABSTAIN,
            ClearRAGDecision.CONFLICT_ABSTENTION,
        )

    @property
    def is_caveated(self) -> bool:
        """Check if this result includes a caveat qualification."""
        return self.decision == ClearRAGDecision.ANSWER_WITH_CAVEAT

    @property
    def generated_answer(self) -> bool:
        """Check if the LLM generator was invoked for this result."""
        return self.generation_latency_ms > 0.0
