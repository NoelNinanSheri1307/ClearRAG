"""Baselines package for ClearRAG controlled comparative evaluation."""

from src.baselines.standard_rag import StandardRAGPipeline, StandardRAGResult
from src.baselines.verification_baseline import (
    VerificationBaselinePipeline,
    VerificationBaselineResult,
)

__all__ = [
    "StandardRAGPipeline",
    "StandardRAGResult",
    "VerificationBaselinePipeline",
    "VerificationBaselineResult",
]
