"""Sufficiency decision engine for ClearRAG verification layer."""

import logging
from typing import Any, Dict, List, Optional

from src.verification.claims import Claim
from src.verification.models import (
    ClaimVerificationResult,
    SufficiencyStatus,
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class SufficiencyEngine:
    """Combines individual claim verification results into an overall sufficiency decision policy."""

    def evaluate_sufficiency(
        self,
        question: str,
        claims: List[Claim],
        claim_results: List[ClaimVerificationResult],
        retrieved_evidence: List[Dict[str, Any]],
        verifier_metadata: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Apply deterministic sufficiency decision policy.

        Policy Logic:
        1. CONFLICTING: If any claim verification result has status CONFLICTING.
        2. FULLY_SUPPORTED: If all claims have status SUPPORTED.
        3. PARTIALLY_SUPPORTED: If at least one claim is SUPPORTED and at least one is UNSUPPORTED.
        4. UNSUPPORTED: If all claims have status UNSUPPORTED (or no claims were extracted).

        Args:
            question: Input question string.
            claims: List of extracted Claim objects.
            claim_results: List of ClaimVerificationResult objects.
            retrieved_evidence: List of retrieved evidence dictionary objects.
            verifier_metadata: Optional metadata dictionary.

        Returns:
            Structured VerificationResult object.
        """
        if verifier_metadata is None:
            verifier_metadata = {}

        if not claim_results:
            return VerificationResult(
                question=question,
                claims=claims,
                retrieved_evidence=retrieved_evidence,
                claim_results=claim_results,
                overall_status=SufficiencyStatus.UNSUPPORTED,
                explanation="No claims extracted or verified for question.",
                verifier_metadata=verifier_metadata,
            )

        # Count statuses
        num_claims = len(claim_results)
        num_supported = sum(1 for r in claim_results if r.status == VerificationStatus.SUPPORTED)
        num_unsupported = sum(1 for r in claim_results if r.status == VerificationStatus.UNSUPPORTED)
        num_conflicting = sum(1 for r in claim_results if r.status == VerificationStatus.CONFLICTING)

        # Decision Policy Logic
        if num_conflicting > 0:
            overall_status = SufficiencyStatus.CONFLICTING
            explanation = f"Conflicting evidence detected across {num_conflicting} claim(s)."
        elif num_supported == num_claims and num_claims > 0:
            overall_status = SufficiencyStatus.FULLY_SUPPORTED
            explanation = f"All {num_claims} required claim(s) are fully supported by evidence."
        elif num_supported > 0 and num_unsupported > 0:
            overall_status = SufficiencyStatus.PARTIALLY_SUPPORTED
            explanation = f"Partial support: {num_supported}/{num_claims} claim(s) supported; {num_unsupported} unsupported."
        else:
            overall_status = SufficiencyStatus.UNSUPPORTED
            explanation = f"None of the {num_claims} claim(s) have sufficient supporting evidence."

        return VerificationResult(
            question=question,
            claims=claims,
            retrieved_evidence=retrieved_evidence,
            claim_results=claim_results,
            overall_status=overall_status,
            explanation=explanation,
            verifier_metadata=verifier_metadata,
        )
