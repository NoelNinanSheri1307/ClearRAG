"""Verification package for ClearRAG."""

from src.verification.calibration import CalibrationMetrics, ThresholdCalibrator
from src.verification.claim_extractor import BaseClaimExtractor, RuleBasedClaimExtractor
from src.verification.claims import Claim, ClaimType
from src.verification.contradiction import ContradictionDetector
from src.verification.evidence_matching import SemanticEvidenceMatcher
from src.verification.evidence_verifier import EvidenceVerifier
from src.verification.improved_verifier import ImprovedEvidenceVerifier
from src.verification.models import (
    ClaimVerificationResult,
    SufficiencyStatus,
    VerificationResult,
    VerificationStatus,
)
from src.verification.sufficiency import SufficiencyEngine

__all__ = [
    "Claim",
    "ClaimType",
    "VerificationStatus",
    "SufficiencyStatus",
    "ClaimVerificationResult",
    "VerificationResult",
    "BaseClaimExtractor",
    "RuleBasedClaimExtractor",
    "EvidenceVerifier",
    "ImprovedEvidenceVerifier",
    "SemanticEvidenceMatcher",
    "ContradictionDetector",
    "ThresholdCalibrator",
    "CalibrationMetrics",
    "SufficiencyEngine",
]
