"""Generation and Synthesis Package for ClearRAG."""

from src.generation.attribution import AnswerClaimAttribution, AttributionEngine
from src.generation.caveat_generator import CaveatPromptBuilder
from src.generation.conflict_generator import ConflictPromptBuilder
from src.generation.generation_metrics import (
    compute_contains_ground_truth,
    compute_exact_match,
    compute_grounding_metrics,
    compute_token_f1,
    normalize_answer,
)
from src.generation.grounded_generator import GroundedPromptBuilder
from src.generation.llm_generator import LLMGenerator
from src.generation.prompt_builder import PromptBuilder
from src.generation.rag_pipeline import RAGPipeline

__all__ = [
    "LLMGenerator",
    "PromptBuilder",
    "RAGPipeline",
    "GroundedPromptBuilder",
    "CaveatPromptBuilder",
    "ConflictPromptBuilder",
    "AttributionEngine",
    "AnswerClaimAttribution",
    "normalize_answer",
    "compute_exact_match",
    "compute_token_f1",
    "compute_contains_ground_truth",
    "compute_grounding_metrics",
]
