"""ClearRAG Decision + Abstention Layer.

This module implements the core ClearRAG innovation: evidence-grounded
decision control that determines whether to answer, qualify, or abstain
based on verified evidence sufficiency.
"""

from src.clearrag.decision import (
    ClearRAGDecision,
    ClearRAGDecisionEngine,
    DEFAULT_DECISION_POLICY,
    GENERATION_PERMITTED_DECISIONS,
)
from src.clearrag.result import ClearRAGResult
from src.clearrag.pipeline import ClearRAGPipeline

__all__ = [
    "ClearRAGDecision",
    "ClearRAGDecisionEngine",
    "ClearRAGResult",
    "ClearRAGPipeline",
    "DEFAULT_DECISION_POLICY",
    "GENERATION_PERMITTED_DECISIONS",
]
