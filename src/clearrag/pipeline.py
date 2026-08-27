"""ClearRAG end-to-end pipeline.

Orchestrates the full ClearRAG flow:
    Retrieve -> Extract Claims -> Verify Evidence -> Sufficiency
    -> Decision -> (Conditional Grounded Generation) -> Attribution -> ClearRAGResult

For ABSTAIN and CONFLICT_ABSTENTION, the LLM generator is NOT invoked.
For ANSWER_WITH_CAVEAT, the generator receives a caveat-aware prompt.
For ANSWER, the generator receives verified supporting evidence and grounding instructions.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from src.clearrag.decision import ClearRAGDecision, ClearRAGDecisionEngine
from src.clearrag.result import ClearRAGResult
from src.generation.attribution import AttributionEngine
from src.generation.caveat_generator import CaveatPromptBuilder
from src.generation.generation_metrics import compute_grounding_metrics
from src.generation.grounded_generator import GroundedPromptBuilder
from src.generation.llm_generator import LLMGenerator
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.retriever import Retriever
from src.verification.claim_extractor import RuleBasedClaimExtractor
from src.verification.evidence_verifier import EvidenceVerifier
from src.verification.models import SufficiencyStatus, VerificationStatus
from src.verification.sufficiency import SufficiencyEngine

logger = logging.getLogger(__name__)

# Standard prompt presets
CAVEAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based on the provided context. "
    "Be concise, direct, and factual. "
    "IMPORTANT: Some evidence may be incomplete or missing. "
    "Clearly indicate what is supported by evidence and what could not be verified. "
    "Do not invent information not present in the provided context."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based on the provided context. "
    "Be concise, direct, and factual. Only use information from the provided context."
)


class ClearRAGPipeline:
    """ClearRAG pipeline with evidence-grounded decision control, conditional generation, and attribution."""

    def __init__(
        self,
        retriever: Retriever,
        generator: LLMGenerator,
        decision_engine: Optional[ClearRAGDecisionEngine] = None,
        claim_extractor: Optional[RuleBasedClaimExtractor] = None,
        evidence_verifier: Optional[EvidenceVerifier] = None,
        sufficiency_engine: Optional[SufficiencyEngine] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        grounded_builder: Optional[GroundedPromptBuilder] = None,
        caveat_builder: Optional[CaveatPromptBuilder] = None,
        attribution_engine: Optional[AttributionEngine] = None,
        default_top_k: int = 5,
        generation_mode: str = "standard",
    ):
        """Initialize ClearRAGPipeline.

        Args:
            retriever: Initialized Retriever instance.
            generator: Initialized LLMGenerator instance.
            decision_engine: Decision policy engine.
            claim_extractor: Claim extractor instance.
            evidence_verifier: Evidence verifier instance.
            sufficiency_engine: Sufficiency engine instance.
            prompt_builder: Standard prompt builder.
            grounded_builder: Grounded prompt builder.
            caveat_builder: Caveat prompt builder.
            attribution_engine: Attribution engine for fine-grained claim tracing.
            default_top_k: Number of evidence chunks to retrieve.
            generation_mode: 'standard' or 'grounded'.
        """
        self.retriever = retriever
        self.generator = generator
        self.decision_engine = decision_engine or ClearRAGDecisionEngine()
        self.claim_extractor = claim_extractor or RuleBasedClaimExtractor()
        self.evidence_verifier = evidence_verifier or EvidenceVerifier()
        self.sufficiency_engine = sufficiency_engine or SufficiencyEngine()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.grounded_builder = grounded_builder or GroundedPromptBuilder()
        self.caveat_builder = caveat_builder or CaveatPromptBuilder()
        self.attribution_engine = attribution_engine or AttributionEngine()
        self.default_top_k = default_top_k
        self.generation_mode = generation_mode

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        generation_mode: Optional[str] = None,
    ) -> ClearRAGResult:
        """Execute the full ClearRAG pipeline on a question.

        Steps:
            1. Retrieve evidence
            2. Extract claims
            3. Verify claims against evidence
            4. Calculate sufficiency
            5. Make decision (answer/abstain)
            6. Conditionally generate answer
            7. Attribute generated answer claims to evidence
            8. Construct ClearRAGResult with full provenance
        """
        k = top_k if top_k is not None else self.default_top_k
        mode = generation_mode or self.generation_mode
        total_start = time.perf_counter()

        # Step 1: Retrieve evidence
        retrieval_start = time.perf_counter()
        raw_chunks = self.retriever.retrieve(question, top_k=k)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

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

        # Step 2-4: Verification & Sufficiency
        verification_start = time.perf_counter()
        claims = self.claim_extractor.extract_claims(question)
        claim_results = [
            self.evidence_verifier.verify_claim(claim, raw_chunks)
            for claim in claims
        ]
        verification_result = self.sufficiency_engine.evaluate_sufficiency(
            question=question,
            claims=claims,
            claim_results=claim_results,
            retrieved_evidence=raw_chunks,
        )
        verification_latency_ms = (time.perf_counter() - verification_start) * 1000.0

        # Step 5: Decision
        decision = self.decision_engine.decide(verification_result.overall_status)
        supporting_evidence = self._collect_supporting_evidence(claim_results, retrieved_evidence)
        conflicting_evidence = self._collect_conflicting_evidence(claim_results, retrieved_evidence)

        # Step 6: Conditional Generation
        answer_text = ""
        generation_latency_ms = 0.0
        abstention_reason = ""
        caveat_text = ""
        attributions: List[Dict[str, Any]] = []
        grounding_metrics: Dict[str, float] = {}

        if self.decision_engine.permits_generation(decision):
            try:
                answer_text, generation_latency_ms, caveat_text = self._generate_answer(
                    question=question,
                    decision=decision,
                    retrieved_evidence=retrieved_evidence,
                    supporting_evidence=supporting_evidence,
                    claims=claims,
                    claim_results=claim_results,
                    max_new_tokens=max_new_tokens,
                    mode=mode,
                )
                # Step 7: Attribution Analysis
                if answer_text:
                    attr_objs = self.attribution_engine.attribute_answer(
                        answer_text=answer_text,
                        evidence_chunks=supporting_evidence or retrieved_evidence,
                        verified_claims=[c.to_dict() for c in claims],
                    )
                    attributions = [a.to_dict() for a in attr_objs]
                    grounding_metrics = compute_grounding_metrics(
                        attributions=attr_objs,
                        evidence_chunks=supporting_evidence or retrieved_evidence,
                        prediction_text=answer_text,
                    )
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                answer_text = (
                    "An error occurred during answer generation. "
                    "The evidence verification completed successfully."
                )
                generation_latency_ms = 0.0
        else:
            answer_text = self.decision_engine.get_abstention_response(decision)
            abstention_reason = self.decision_engine.get_abstention_reason(
                decision, verification_result.explanation
            )

        total_latency_ms = (time.perf_counter() - total_start) * 1000.0

        # Step 8: Return structured result
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
            attributions=attributions,
            grounding_metrics=grounding_metrics,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            verification_latency_ms=round(verification_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            metadata={
                "top_k": k,
                "generation_mode": mode,
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
        claims: Optional[list] = None,
        claim_results: Optional[list] = None,
        max_new_tokens: Optional[int] = None,
        mode: str = "standard",
    ) -> tuple:
        """Generate answer with decision-appropriate and mode-appropriate prompting."""
        caveat_text = ""

        if decision == ClearRAGDecision.ANSWER:
            context_chunks = supporting_evidence if supporting_evidence else retrieved_evidence
            if mode == "grounded":
                messages = self.grounded_builder.build_messages(
                    question=question,
                    evidence_chunks=context_chunks,
                    verified_claims=[c.to_dict() for c in claims] if claims else None,
                )
            else:
                builder = PromptBuilder(system_prompt=ANSWER_SYSTEM_PROMPT)
                messages = builder.build_messages(question, context_chunks)

        elif decision == ClearRAGDecision.ANSWER_WITH_CAVEAT:
            context_chunks = retrieved_evidence
            caveat_text = self.decision_engine.caveat_prefix

            if mode == "grounded":
                supported_texts = [cr.claim.text for cr in (claim_results or []) if cr.status == VerificationStatus.SUPPORTED]
                unsupported_texts = [cr.claim.text for cr in (claim_results or []) if cr.status != VerificationStatus.SUPPORTED]
                messages = self.caveat_builder.build_messages(
                    question=question,
                    evidence_chunks=context_chunks,
                    supported_claims=supported_texts,
                    unsupported_claims=unsupported_texts,
                )
            else:
                builder = PromptBuilder(system_prompt=CAVEAT_SYSTEM_PROMPT)
                messages = builder.build_messages(question, context_chunks)
        else:
            return "", 0.0, ""

        answer_text, gen_latency_ms = self.generator.generate_from_messages(
            messages, max_new_tokens=max_new_tokens
        )

        if caveat_text and answer_text and mode != "grounded":
            answer_text = f"{caveat_text}{answer_text}"

        return answer_text, gen_latency_ms, caveat_text

    def _collect_supporting_evidence(
        self,
        claim_results: list,
        retrieved_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Collect evidence chunks identified as supporting claims."""
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
