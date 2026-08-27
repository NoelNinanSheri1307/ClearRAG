"""Standard RAG Baseline for ClearRAG Controlled Evaluation.

Implements conventional Always-Answer Retrieval-Augmented Generation:
Question -> Retriever -> PromptBuilder -> LLM -> Answer.

Does NOT use claim extraction, verification, sufficiency engine,
ClearRAG decision engine, abstention logic, conflict detection, or caveat generation.
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
class StandardRAGResult:
    """Structured output from Standard RAG Baseline with provenance and latencies."""

    question: str
    answer: str
    retrieved_evidence: List[Dict[str, Any]]
    retrieved_chunk_ids: List[str]
    model: str
    embedding_model: str
    top_k: int
    llm_called: bool
    latency_retrieval_ms: float
    latency_generation_ms: float
    latency_total_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class StandardRAGPipeline:
    """Standard RAG Baseline Pipeline that always attempts to answer."""

    def __init__(
        self,
        retriever: Retriever,
        generator: LLMGenerator,
        prompt_builder: Optional[PromptBuilder] = None,
        default_top_k: int = 5,
    ):
        """Initialize the Standard RAG Baseline.

        Args:
            retriever: Retriever instance.
            generator: LLMGenerator instance.
            prompt_builder: Optional custom PromptBuilder.
            default_top_k: Number of chunks to retrieve (default: 5).
        """
        self.retriever = retriever
        self.generator = generator
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.default_top_k = default_top_k

    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> StandardRAGResult:
        """Run the Standard RAG pipeline on a question.

        Args:
            question: Query string (received WITHOUT any ground truth metadata).
            top_k: Number of passages to retrieve.
            max_new_tokens: Max new generation tokens.
            temperature: LLM temperature.
            do_sample: LLM sampling flag.

        Returns:
            StandardRAGResult containing answer, evidence, chunk IDs, and latency breakdown.
        """
        k = top_k if top_k is not None else self.default_top_k
        total_start = time.perf_counter()

        # 1. Retrieve evidence chunks
        retrieval_start = time.perf_counter()
        raw_chunks = self.retriever.retrieve(question, top_k=k)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        retrieved_chunk_ids: List[str] = []
        retrieved_evidence: List[Dict[str, Any]] = []

        for chunk in raw_chunks:
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id:
                retrieved_chunk_ids.append(chunk_id)

            retrieved_evidence.append(
                {
                    "rank": chunk.get("rank", 0),
                    "chunk_id": chunk_id,
                    "score": round(chunk.get("score", 0.0), 4),
                    "title": chunk.get("document_title", ""),
                    "sentence_indices": chunk.get("sentence_indices", []),
                    "text": chunk.get("text", ""),
                    "is_supporting_fact": chunk.get("is_supporting_fact", False),
                }
            )

        # 2. Build formatted prompt
        messages = self.prompt_builder.build_messages(question, retrieved_evidence)

        # 3. Generate answer unconditionally
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

        return StandardRAGResult(
            question=question,
            answer=answer_text,
            retrieved_evidence=retrieved_evidence,
            retrieved_chunk_ids=retrieved_chunk_ids,
            model=self.generator.model_name,
            embedding_model=embedding_model_name,
            top_k=k,
            llm_called=True,
            latency_retrieval_ms=round(retrieval_latency_ms, 2),
            latency_generation_ms=round(gen_latency_ms, 2),
            latency_total_ms=round(total_latency_ms, 2),
        )

    def answer(self, question: str, **kwargs: Any) -> StandardRAGResult:
        """Alias for run() to maintain interface consistency."""
        return self.run(question, **kwargs)
