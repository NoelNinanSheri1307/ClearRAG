"""Grounded Prompt Builder and Evidence-Bound Synthesis for ClearRAG.

Constructs strict, attribution-aware prompts that require language models to ground
every assertion in verified context chunks and cite evidence anchors [1], [2].
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GROUNDED_SYSTEM_PROMPT = (
    "You are a strictly grounded factual assistant. "
    "Your objective is to answer the user's question using ONLY the facts directly stated in the provided context.\n"
    "RULES:\n"
    "1. Answer concisely, directly, and truthfully.\n"
    "2. Do NOT extrapolate, speculate, or introduce external knowledge not present in the context.\n"
    "3. Support your answer statements by citing the context chunk number, e.g., [1] or [2].\n"
    "4. If the provided context does not contain sufficient facts to answer the question, state that clearly."
)


class GroundedPromptBuilder:
    """Constructs evidence-bounded prompt templates with numbered citation anchors."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        require_citations: bool = True,
    ):
        """Initialize GroundedPromptBuilder.

        Args:
            system_prompt: Custom system instruction override.
            require_citations: Whether to instruct the model to cite chunk numbers [1], [2].
        """
        self.system_prompt = system_prompt or GROUNDED_SYSTEM_PROMPT
        self.require_citations = require_citations

    def format_grounded_context(
        self,
        evidence_chunks: List[Dict[str, Any]],
        verified_claims: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Format evidence chunks into structured, numbered blocks with verified claim anchors.

        Args:
            evidence_chunks: List of retrieved evidence dictionaries.
            verified_claims: Optional verified claim status summaries.

        Returns:
            Formatted context string.
        """
        if not evidence_chunks:
            return "No relevant verified evidence available."

        context_blocks = []
        for i, chunk in enumerate(evidence_chunks, start=1):
            title = chunk.get("document_title", "").strip()
            text = chunk.get("text", "").strip()
            header = f"[{i}] Document: {title}" if title else f"[{i}]"
            context_blocks.append(f"{header}\n{text}")

        context_str = "\n\n".join(context_blocks)

        if verified_claims:
            claim_summaries = []
            for c in verified_claims:
                cid = c.get("claim_id", "")
                ctext = c.get("text", "")
                claim_summaries.append(f"- Verified Requirement {cid}: {ctext}")
            claims_header = "VERIFIED INQUIRY REQUIREMENTS:\n" + "\n".join(claim_summaries)
            return f"{claims_header}\n\nEVIDENCE PASSAGES:\n{context_str}"

        return context_str

    def build_user_prompt(
        self,
        question: str,
        evidence_chunks: List[Dict[str, Any]],
        verified_claims: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Construct the grounded user prompt."""
        context_text = self.format_grounded_context(evidence_chunks, verified_claims)
        citation_instruction = (
            "Cite the supporting evidence chunk numbers (e.g. [1], [2]) for each factual statement.\n"
            if self.require_citations
            else ""
        )
        return (
            f"Context Information:\n"
            f"---------------------\n"
            f"{context_text}\n"
            f"---------------------\n\n"
            f"Question: {question.strip()}\n\n"
            f"{citation_instruction}"
            f"Answer:"
        )

    def build_messages(
        self,
        question: str,
        evidence_chunks: List[Dict[str, Any]],
        verified_claims: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        """Build messages for LLM chat completion."""
        user_content = self.build_user_prompt(question, evidence_chunks, verified_claims)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
