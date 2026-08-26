"""Standard RAG Prompt Builder for ClearRAG.

Formats retrieved evidence chunks and user questions into clean, deterministic
prompts for instruction-tuned language models.
"""

from typing import Any, Dict, List, Optional


class PromptBuilder:
    """Constructs conventional RAG prompts from questions and retrieved evidence."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer the question based on the provided context. "
        "Be concise, direct, and factual."
    )

    def __init__(self, system_prompt: Optional[str] = None):
        """Initialize the PromptBuilder.

        Args:
            system_prompt: Optional custom system instructions.
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format a list of retrieved chunk dictionaries into numbered context text.

        Args:
            retrieved_chunks: List of dictionaries containing at minimum 'text'
                              and optionally 'document_title', 'rank'.

        Returns:
            Formatted context string.
        """
        if not retrieved_chunks:
            return "No relevant context found."

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            title = chunk.get("document_title", "").strip()
            text = chunk.get("text", "").strip()
            if title:
                context_blocks.append(f"[{i}] Document: {title}\n{text}")
            else:
                context_blocks.append(f"[{i}] {text}")

        return "\n\n".join(context_blocks)

    def build_user_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """Build the combined context and question user message.

        Args:
            question: User inquiry string.
            retrieved_chunks: List of retrieved evidence chunk dictionaries.

        Returns:
            Formatted user message string.
        """
        context_text = self.format_context(retrieved_chunks)
        return (
            f"Context:\n{context_text}\n\n"
            f"Question: {question.strip()}\n\n"
            f"Answer:"
        )

    def build_messages(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Build a standard chat message list suitable for HuggingFace chat templates.

        Args:
            question: User inquiry string.
            retrieved_chunks: List of retrieved evidence chunk dictionaries.

        Returns:
            List of message dictionaries with 'role' and 'content'.
        """
        user_content = self.build_user_prompt(question, retrieved_chunks)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
