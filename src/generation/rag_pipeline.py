"""Standard RAG Pipeline for ClearRAG.

Connects the verified Retriever, PromptBuilder, and LLMGenerator into a complete,
deterministic Question-Answering baseline pipeline.
"""

from dataclasses import asdict, dataclass
import logging
import time
from typing import Any, Dict, List, Optional

from src.generation.llm_generator import LLMGenerator
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Encapsulates the structured output and provenance of a RAG query."""

    question: str
    answer: str
    retrieved_context: List[Dict[str, Any]]
    model: str
    embedding_model: str
    top_k: int
    latency_retrieval_ms: float
    latency_generation_ms: float
    latency_total_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert RAG result to dictionary."""
        return asdict(self)


class RAGPipeline:
    """Conventional Standard RAG baseline pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        generator: LLMGenerator,
        prompt_builder: Optional[PromptBuilder] = None,
        default_top_k: int = 5,
    ):
        """Initialize the RAGPipeline.

        Args:
            retriever: Initialized Retriever instance.
            generator: Initialized LLMGenerator instance.
            prompt_builder: Optional custom PromptBuilder instance.
            default_top_k: Default number of evidence chunks to retrieve (default: 5).
        """
        self.retriever = retriever
        self.generator = generator
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.default_top_k = default_top_k

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> RAGResult:
        """Execute end-to-end Standard RAG on a query.

        Args:
            question: User inquiry string.
            top_k: Top-K passages to retrieve (defaults to self.default_top_k).
            max_new_tokens: Max new generation tokens.
            temperature: LLM temperature.
            do_sample: LLM sampling flag.

        Returns:
            RAGResult containing answer, full retrieved context, and latencies.
        """
        k = top_k if top_k is not None else self.default_top_k
        total_start = time.perf_counter()

        # 1. Retrieve evidence chunks
        retrieval_start = time.perf_counter()
        raw_chunks = self.retriever.retrieve(question, top_k=k)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # 2. Format structured context records for provenance preservation
        retrieved_context: List[Dict[str, Any]] = []
        for chunk in raw_chunks:
            retrieved_context.append(
                {
                    "rank": chunk.get("rank", 0),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "score": round(chunk.get("score", 0.0), 4),
                    "title": chunk.get("document_title", ""),
                    "sentence_indices": chunk.get("sentence_indices", []),
                    "text": chunk.get("text", ""),
                    "is_supporting_fact": chunk.get("is_supporting_fact", False),
                }
            )

        # 3. Build formatted chat prompt
        messages = self.prompt_builder.build_messages(question, retrieved_context)

        # 4. Generate answer
        answer_text, gen_latency_ms = self.generator.generate_from_messages(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        total_latency_ms = (time.perf_counter() - total_start) * 1000.0

        embedding_model_name = getattr(
            self.retriever.embedder, "model_name", "BAAI/bge-small-en-v1.5"
        )

        return RAGResult(
            question=question,
            answer=answer_text,
            retrieved_context=retrieved_context,
            model=self.generator.model_name,
            embedding_model=embedding_model_name,
            top_k=k,
            latency_retrieval_ms=round(retrieval_latency_ms, 2),
            latency_generation_ms=round(gen_latency_ms, 2),
            latency_total_ms=round(total_latency_ms, 2),
        )
