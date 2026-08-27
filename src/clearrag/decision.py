"""ClearRAG Decision + Abstention Layer.

Implements the decision engine that determines whether ClearRAG should:
- ANSWER (fully supported)
- ANSWER_WITH_CAVEAT (partially supported)
- ABSTAIN (unsupported)
- CONFLICT_ABSTENTION (conflicting evidence)
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional

from src.verification.models import SufficiencyStatus

logger = logging.getLogger(__name__)


class ClearRAGDecision(str, Enum):
    """ClearRAG system decision after evidence verification."""
    ANSWER = "ANSWER"
    ANSWER_WITH_CAVEAT = "ANSWER_WITH_CAVEAT"
    ABSTAIN = "ABSTAIN"
    CONFLICT_ABSTENTION = "CONFLICT_ABSTENTION"


# Default decision policy mapping
DEFAULT_DECISION_POLICY: Dict[str, str] = {
    SufficiencyStatus.FULLY_SUPPORTED.value: ClearRAGDecision.ANSWER.value,
    SufficiencyStatus.PARTIALLY_SUPPORTED.value: ClearRAGDecision.ANSWER_WITH_CAVEAT.value,
    SufficiencyStatus.UNSUPPORTED.value: ClearRAGDecision.ABSTAIN.value,
    SufficiencyStatus.CONFLICTING.value: ClearRAGDecision.CONFLICT_ABSTENTION.value,
}

# Decisions that permit LLM generation
GENERATION_PERMITTED_DECISIONS = {
    ClearRAGDecision.ANSWER,
    ClearRAGDecision.ANSWER_WITH_CAVEAT,
}

# Default abstention responses
DEFAULT_ABSTENTION_RESPONSES: Dict[str, str] = {
    ClearRAGDecision.ABSTAIN.value: (
        "I cannot provide a reliable answer to this question. "
        "The retrieved evidence does not contain sufficient information "
        "to support a factual response."
    ),
    ClearRAGDecision.CONFLICT_ABSTENTION.value: (
        "I cannot provide a reliable answer to this question. "
        "The retrieved evidence contains conflicting information, "
        "making it impossible to determine a trustworthy response."
    ),
}

# Default caveat prefix
DEFAULT_CAVEAT_PREFIX = (
    "Note: The following answer is based on incomplete evidence. "
    "Some aspects of the question could not be fully verified. "
)


class ClearRAGDecisionEngine:
    """Deterministic decision engine that maps evidence sufficiency to ClearRAG actions.

    Configurable via a policy dictionary mapping SufficiencyStatus values
    to ClearRAGDecision values.
    """

    def __init__(
        self,
        policy: Optional[Dict[str, str]] = None,
        abstention_responses: Optional[Dict[str, str]] = None,
        caveat_prefix: Optional[str] = None,
    ):
        """Initialize ClearRAGDecisionEngine.

        Args:
            policy: Optional custom mapping from SufficiencyStatus.value
                    to ClearRAGDecision.value. Falls back to DEFAULT_DECISION_POLICY.
            abstention_responses: Optional custom abstention response text
                                  keyed by ClearRAGDecision.value.
            caveat_prefix: Optional prefix text for caveat-qualified answers.
        """
        self.policy = policy or dict(DEFAULT_DECISION_POLICY)
        self.abstention_responses = abstention_responses or dict(DEFAULT_ABSTENTION_RESPONSES)
        self.caveat_prefix = caveat_prefix or DEFAULT_CAVEAT_PREFIX

        # Validate policy keys
        for status_val in self.policy:
            try:
                SufficiencyStatus(status_val)
            except ValueError:
                logger.warning(f"Unknown SufficiencyStatus in policy: '{status_val}'")

        # Validate policy values
        for decision_val in self.policy.values():
            try:
                ClearRAGDecision(decision_val)
            except ValueError:
                logger.warning(f"Unknown ClearRAGDecision in policy: '{decision_val}'")

    def decide(self, sufficiency_status: SufficiencyStatus) -> ClearRAGDecision:
        """Apply deterministic decision policy.

        Args:
            sufficiency_status: The evidence sufficiency status from
                                the SufficiencyEngine.

        Returns:
            ClearRAGDecision enum value.
        """
        decision_str = self.policy.get(
            sufficiency_status.value,
            ClearRAGDecision.ABSTAIN.value,
        )
        decision = ClearRAGDecision(decision_str)

        logger.info(
            f"ClearRAG Decision: {sufficiency_status.value} -> {decision.value}"
        )
        return decision

    def permits_generation(self, decision: ClearRAGDecision) -> bool:
        """Check if the decision permits LLM generation.

        Args:
            decision: The ClearRAG decision.

        Returns:
            True if generation is permitted.
        """
        return decision in GENERATION_PERMITTED_DECISIONS

    def get_abstention_response(self, decision: ClearRAGDecision) -> str:
        """Get the deterministic abstention response text.

        Args:
            decision: The ClearRAG decision (should be ABSTAIN or CONFLICT_ABSTENTION).

        Returns:
            Abstention response string.
        """
        return self.abstention_responses.get(
            decision.value,
            f"Unable to provide a reliable answer. Decision: {decision.value}",
        )

    def get_abstention_reason(
        self,
        decision: ClearRAGDecision,
        sufficiency_explanation: str,
    ) -> str:
        """Generate a human-readable abstention reason.

        Args:
            decision: The ClearRAG decision.
            sufficiency_explanation: The explanation from the SufficiencyEngine.

        Returns:
            Explanation string for why the system abstained.
        """
        if decision == ClearRAGDecision.ABSTAIN:
            return (
                f"ABSTENTION: The evidence verification layer determined that "
                f"retrieved evidence is insufficient to answer this question. "
                f"Detail: {sufficiency_explanation}"
            )
        elif decision == ClearRAGDecision.CONFLICT_ABSTENTION:
            return (
                f"CONFLICT ABSTENTION: The evidence verification layer detected "
                f"conflicting information in the retrieved evidence. "
                f"Providing an answer risks propagating incorrect information. "
                f"Detail: {sufficiency_explanation}"
            )
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine configuration."""
        return {
            "policy": self.policy,
            "abstention_responses": self.abstention_responses,
            "caveat_prefix": self.caveat_prefix,
            "generation_permitted_decisions": [
                d.value for d in GENERATION_PERMITTED_DECISIONS
            ],
        }
