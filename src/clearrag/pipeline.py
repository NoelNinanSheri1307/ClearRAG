"""ClearRAG end-to-end pipeline.

Orchestrates the full ClearRAG flow:
    Retrieve -> Extract Claims -> Verify Evidence -> Sufficiency
    -> Decision -> (Conditional Generation) -> ClearRAGResult

For ABSTAIN and CONFLICT_ABSTENTION, the LLM generator is NOT invoked.
For ANSWER_WITH_CAVEAT, the generator receives a caveat-aware prompt.
For ANSWER, the generator receives only verified supporting evidence.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from src.clearrag.decision import ClearRAGDecision, ClearRAGDecisionEngine
from src.clearrag.result import ClearRAGResult
from src.generation.llm_generator import LLMGenerator
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.retriever import Retriever
from src.verification.claim_extractor import RuleBasedClaimExtractor
from src.verification.evidence_verifier import EvidenceVerifier
from src.verification.models import SufficiencyStatus, VerificationStatus
from src.verification.sufficiency import SufficiencyEngine

logger = logging.getLogger(__name__)


# Caveat-aware system prompt for ANSWER_WITH_CAVEAT decisions
CAVEAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based on the provided context. "
    "Be concise, direct, and factual. "
    "IMPORTANT: Some evidence may be incomplete or missing. "
    "Clearly indicate what is supported by evidence and what could not be verified. "
    "Do not invent information not present in the provided context."
)

# Standard system prompt for ANSWER decisions
ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based on the provided context. "
    "Be concise, direct, and factual. Only use information from the provided context."
)


class ClearRAGPipeline:
    """ClearRAG pipeline with evidence-grounded decision control and conditional generation.

    Architecture:
        Question
            -> Retriever (frozen)
            -> RuleBasedClaimExtractor (frozen)
            -> EvidenceVerifier (frozen)
            -> SufficiencyEngine (frozen)
            -> ClearRAGDecisionEngine (new)
            -> Conditional LLMGenerator
            -> ClearRAGResult

    This class integrates the existing retrieval + verification foundation
    with the new decision + abstention layer without modifying existing components.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: LLMGenerator,
        decision_engine: Optional[ClearRAGDecisionEngine] = None,
        claim_extractor: Optional[RuleBasedClaimExtractor] = None,
        evidence_verifier: Optional[EvidenceVerifier] = None,
        sufficiency_engine: Optional[SufficiencyEngine] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        default_top_k: int = 5,
    ):
        """Initialize ClearRAGPipeline.

        Args:
            retriever: Initialized Retriever instance (frozen).
            generator: Initialized LLMGenerator instance (used only when permitted).
            decision_engine: Optional ClearRAGDecisionEngine (default policy if None).
            claim_extractor: Optional claim extractor (default RuleBasedClaimExtractor).
            evidence_verifier: Optional evidence verifier (default EvidenceVerifier).
            sufficiency_engine: Optional sufficiency engine (default SufficiencyEngine).
            prompt_builder: Optional prompt builder (default PromptBuilder).
            default_top_k: Default number of evidence chunks to retrieve.
        """
        self.retriever = retriever
        self.generator = generator
        self.decision_engine = decision_engine or ClearRAGDecisionEngine()
        self.claim_extractor = claim_extractor or RuleBasedClaimExtractor()
        self.evidence_verifier = evidence_verifier or EvidenceVerifier()
        self.sufficiency_engine = sufficiency_engine or SufficiencyEngine()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.default_top_k = default_top_k

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> ClearRAGResult:
        """Execute the full ClearRAG pipeline on a question.

        Steps:
            1. Retrieve evidence
            2. Extract claims
            3. Verify claims against evidence
            4. Calculate sufficiency
            5. Make decision (answer/abstain)
            6. Conditionally generate answer
            7. Construct ClearRAGResult with full provenance

        Args:
            question: User question string.
            top_k: Number of evidence chunks to retrieve.
            max_new_tokens: Maximum generation tokens (if generation occurs).

        Returns:
            ClearRAGResult with full provenance and audit trail.
        """
        k = top_k if top_k is not None else self.default_top_k
        total_start = time.perf_counter()

        # ──────────────────────────────────────────────
        # Step 1: Retrieve evidence (frozen retrieval layer)
        # ──────────────────────────────────────────────
        retrieval_start = time.perf_counter()
        raw_chunks = self.retriever.retrieve(question, top_k=k)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # Format retrieved context for provenance
        retrieved_evidence: List[Dict[str, Any]] = []
        for chunk in raw_chunks:
            retrieved_evidence.append({
                "rank": chunk.get("rank", 0),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": round(chunk.get("score", 0.0), 4),
                "document_title": chunk.get("document_title", ""),
                "text": chunk.get("text", ""),
                "sentence_indices": chunk.get("sentence_indices", []),
                "is_supporting_fact": chunk.get("is_supporting_fact", False),
            })

        # ──────────────────────────────────────────────
        # Step 2–4: Verification pipeline (frozen verification layer)
        # ──────────────────────────────────────────────
        verification_start = time.perf_counter()

        # 2. Extract claims
        claims = self.claim_extractor.extract_claims(question)

        # 3. Verify each claim against evidence
        claim_results = [
            self.evidence_verifier.verify_claim(claim, raw_chunks)
            for claim in claims
        ]

        # 4. Calculate overall sufficiency
        verification_result = self.sufficiency_engine.evaluate_sufficiency(
            question=question,
            claims=claims,
            claim_results=claim_results,
            retrieved_evidence=raw_chunks,
        )

        verification_latency_ms = (time.perf_counter() - verification_start) * 1000.0

        # ──────────────────────────────────────────────
        # Step 5: ClearRAG Decision
        # ──────────────────────────────────────────────
        decision = self.decision_engine.decide(verification_result.overall_status)

        # Collect supporting and conflicting evidence for provenance
        supporting_evidence = self._collect_supporting_evidence(claim_results, retrieved_evidence)
        conflicting_evidence = self._collect_conflicting_evidence(claim_results, retrieved_evidence)

        # ──────────────────────────────────────────────
        # Step 6: Conditional generation
        # ──────────────────────────────────────────────
        answer_text = ""
        generation_latency_ms = 0.0
        abstention_reason = ""
        caveat_text = ""

        if self.decision_engine.permits_generation(decision):
            try:
                answer_text, generation_latency_ms, caveat_text = self._generate_answer(
                    question=question,
                    decision=decision,
                    retrieved_evidence=retrieved_evidence,
                    supporting_evidence=supporting_evidence,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                # Generation failure does not corrupt verification result
                answer_text = (
                    "An error occurred during answer generation. "
                    "The evidence verification completed successfully."
                )
                generation_latency_ms = 0.0
        else:
            # ABSTAIN or CONFLICT_ABSTENTION: deterministic response, no LLM call
            answer_text = self.decision_engine.get_abstention_response(decision)
            abstention_reason = self.decision_engine.get_abstention_reason(
                decision, verification_result.explanation
            )

        total_latency_ms = (time.perf_counter() - total_start) * 1000.0

        # ──────────────────────────────────────────────
        # Step 7: Construct ClearRAGResult
        # ──────────────────────────────────────────────
        return ClearRAGResult(
            question=question,
            answer=answer_text,
            decision=decision,
            sufficiency_status=verification_result.overall_status,
            claims=[c.to_dict() for c in claims],
            claim_results=[r.to_dict() for r in claim_results],
            retrieved_evidence=retrieved_evidence,
            supporting_evidence=supporting_evidence,
            conflicting_evidence=conflicting_evidence,
            explanation=verification_result.explanation,
            abstention_reason=abstention_reason,
            caveat_text=caveat_text,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            verification_latency_ms=round(verification_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            metadata={
                "top_k": k,
                "model_name": self.generator.model_name,
                "embedding_model": getattr(
                    self.retriever.embedder, "model_name", "BAAI/bge-small-en-v1.5"
                ),
                "decision_policy": self.decision_engine.policy,
            },
        )

    def _generate_answer(
        self,
        question: str,
        decision: ClearRAGDecision,
        retrieved_evidence: List[Dict[str, Any]],
        supporting_evidence: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> tuple:
        """Generate answer using the LLM, with decision-appropriate prompting.

        Args:
            question: The user question.
            decision: The ClearRAG decision.
            retrieved_evidence: All retrieved evidence chunks.
            supporting_evidence: Only the verified supporting evidence.
            max_new_tokens: Max generation tokens.

        Returns:
            Tuple of (answer_text, generation_latency_ms, caveat_text).
        """
        caveat_text = ""

        if decision == ClearRAGDecision.ANSWER:
            # ANSWER: Use supporting evidence preferentially; fall back to all evidence
            context_chunks = supporting_evidence if supporting_evidence else retrieved_evidence
            system_prompt = ANSWER_SYSTEM_PROMPT
        elif decision == ClearRAGDecision.ANSWER_WITH_CAVEAT:
            # ANSWER_WITH_CAVEAT: Use all evidence with caveat-aware prompt
            context_chunks = retrieved_evidence
            system_prompt = CAVEAT_SYSTEM_PROMPT
            caveat_text = self.decision_engine.caveat_prefix
        else:
            # Should not reach here if permits_generation is checked
            return "", 0.0, ""

        # Build prompt using the existing PromptBuilder
        builder = PromptBuilder(system_prompt=system_prompt)
        messages = builder.build_messages(question, context_chunks)

        # Generate
        answer_text, gen_latency_ms = self.generator.generate_from_messages(
            messages, max_new_tokens=max_new_tokens
        )

        # Prepend caveat for ANSWER_WITH_CAVEAT
        if caveat_text and answer_text:
            answer_text = f"{caveat_text}{answer_text}"

        return answer_text, gen_latency_ms, caveat_text

    def _collect_supporting_evidence(
        self,
        claim_results: list,
        retrieved_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Collect evidence chunks that were identified as supporting claims."""
        supporting_ids = set()
        for cr in claim_results:
            if cr.status == VerificationStatus.SUPPORTED:
                supporting_ids.update(cr.supporting_evidence_ids)

        return [
            chunk for chunk in retrieved_evidence
            if str(chunk.get("chunk_id", "")) in supporting_ids
        ]

    def _collect_conflicting_evidence(
        self,
        claim_results: list,
        retrieved_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Collect evidence chunks identified as conflicting."""
        conflicting_ids = set()
        for cr in claim_results:
            if cr.status == VerificationStatus.CONFLICTING:
                conflicting_ids.update(cr.conflicting_evidence_ids)

        return [
            chunk for chunk in retrieved_evidence
            if str(chunk.get("chunk_id", "")) in conflicting_ids
        ]
