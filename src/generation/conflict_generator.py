"""Conflict-Aware Prompt Builder and Explanation Synthesis for ClearRAG.

Constructs multi-perspective contradiction summaries that explicitly outline opposing claims
across retrieved passages rather than silently or arbitrarily picking one side.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONFLICT_SYSTEM_PROMPT = (
    "You are an impartial, truth-preserving research assistant.\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. The provided evidence contains contradictory or conflicting factual assertions.\n"
    "2. Do NOT choose or favor one side over the other.\n"
    "3. State clearly that the available evidence is in conflict.\n"
    "4. Summarize the differing claims presented by each source passage."
)


class ConflictPromptBuilder:
    """Constructs conflict explanation prompts for contradictory evidence cases."""

    def __init__(self, system_prompt: Optional[str] = None):
        """Initialize ConflictPromptBuilder."""
        self.system_prompt = system_prompt or CONFLICT_SYSTEM_PROMPT

    def format_conflict_context(
        self,
        conflicting_chunks: List[Dict[str, Any]],
        conflict_description: str = "",
    ) -> str:
        """Format conflicting evidence blocks."""
        blocks = []
        for i, chunk in enumerate(conflicting_chunks, start=1):
            title = chunk.get("document_title", "").strip()
            text = chunk.get("text", "").strip()
            header = f"[Source {i}] Document: {title}" if title else f"[Source {i}]"
            blocks.append(f"{header}\n{text}")

        context_str = "\n\n".join(blocks)
        if conflict_description:
            return f"IDENTIFIED CONTRADICTION:\n{conflict_description}\n\nCONFLICTING PASSAGES:\n{context_str}"
        return context_str

    def build_user_prompt(
        self,
        question: str,
        conflicting_chunks: List[Dict[str, Any]],
        conflict_description: str = "",
    ) -> str:
        """Build user prompt for conflict explanation."""
        context_text = self.format_conflict_context(conflicting_chunks, conflict_description)
        return (
            f"Context Information (Conflicting Evidence):\n"
            f"-----------------------------------------\n"
            f"{context_text}\n"
            f"-----------------------------------------\n\n"
            f"Question: {question.strip()}\n\n"
            f"Instructions: Explain that the evidence contains a direct factual conflict, "
            f"and summarize the conflicting facts from each source.\n\n"
            f"Answer:"
        )

    def build_messages(
        self,
        question: str,
        conflicting_chunks: List[Dict[str, Any]],
        conflict_description: str = "",
    ) -> List[Dict[str, str]]:
        """Build chat messages list for conflict synthesis."""
        user_content = self.build_user_prompt(question, conflicting_chunks, conflict_description)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
