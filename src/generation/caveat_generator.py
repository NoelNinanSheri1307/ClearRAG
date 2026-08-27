"""Caveat-Aware Prompt Builder and Synthesis for ClearRAG.

Constructs dual-aspect prompt templates for partial-evidence decisions, explicitly
separating supported facts from missing/unverified inquiries to prevent hallucinated completion.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CAVEAT_SYSTEM_PROMPT = (
    "You are a careful, truthful assistant operating under incomplete evidence.\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Answer ONLY what is explicitly supported by the provided context.\n"
    "2. Explicitly acknowledge what information is missing or unverified.\n"
    "3. Do NOT guess, assume, or invent facts for the missing parts.\n"
    "4. Formulate your response in two parts: what is supported, followed by a clear caveat stating what cannot be verified from the evidence."
)


class CaveatPromptBuilder:
    """Constructs caveat-aware prompts for partial evidence situations."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        caveat_prefix: str = "Note: Incomplete evidence. ",
    ):
        """Initialize CaveatPromptBuilder.

        Args:
            system_prompt: Custom system prompt override.
            caveat_prefix: Prefix string for caveat qualifications.
        """
        self.system_prompt = system_prompt or CAVEAT_SYSTEM_PROMPT
        self.caveat_prefix = caveat_prefix

    def format_partial_context(
        self,
        evidence_chunks: List[Dict[str, Any]],
        supported_claims: Optional[List[str]] = None,
        unsupported_claims: Optional[List[str]] = None,
    ) -> str:
        """Format retrieved context with explicit support annotations."""
        context_blocks = []
        for i, chunk in enumerate(evidence_chunks, start=1):
            title = chunk.get("document_title", "").strip()
            text = chunk.get("text", "").strip()
            header = f"[{i}] Document: {title}" if title else f"[{i}]"
            context_blocks.append(f"{header}\n{text}")

        context_str = "\n\n".join(context_blocks)
        meta_blocks = []
        if supported_claims:
            meta_blocks.append("SUPPORTED INQUIRY ASPECTS:\n" + "\n".join(f"- {c}" for c in supported_claims))
        if unsupported_claims:
            meta_blocks.append("MISSING / UNVERIFIED ASPECTS:\n" + "\n".join(f"- {c}" for c in unsupported_claims))

        meta_header = "\n\n".join(meta_blocks)
        if meta_header:
            return f"{meta_header}\n\nEVIDENCE PASSAGES:\n{context_str}"
        return context_str

    def build_user_prompt(
        self,
        question: str,
        evidence_chunks: List[Dict[str, Any]],
        supported_claims: Optional[List[str]] = None,
        unsupported_claims: Optional[List[str]] = None,
    ) -> str:
        """Build user prompt for caveat synthesis."""
        context_text = self.format_partial_context(
            evidence_chunks, supported_claims, unsupported_claims
        )
        return (
            f"Context Information:\n"
            f"---------------------\n"
            f"{context_text}\n"
            f"---------------------\n\n"
            f"Question: {question.strip()}\n\n"
            f"Instructions: Provide an answer covering only what is supported by the context, "
            f"and clearly state what remains unverified.\n\n"
            f"Answer:"
        )

    def build_messages(
        self,
        question: str,
        evidence_chunks: List[Dict[str, Any]],
        supported_claims: Optional[List[str]] = None,
        unsupported_claims: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages list for caveat-aware generation."""
        user_content = self.build_user_prompt(
            question, evidence_chunks, supported_claims, unsupported_claims
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
