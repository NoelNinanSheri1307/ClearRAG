"""Claim-level and Passage-level Attribution Engine for ClearRAG.

Decomposes generated answers into atomic sentences/claims and maps each claim
back to supporting evidence chunks and verified claim IDs.
"""

from dataclasses import asdict, dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AnswerClaimAttribution:
    """Attribution metadata for a single sentence/claim in the generated answer."""

    claim_index: int
    claim_text: str
    cited_chunk_indices: List[int] = field(default_factory=list)
    supporting_chunk_ids: List[str] = field(default_factory=list)
    verified_claim_ids: List[str] = field(default_factory=list)
    is_supported: bool = False
    grounding_confidence: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert attribution to dictionary."""
        return asdict(self)


class AttributionEngine:
    """Aligns generated answer sentences with retrieved evidence chunks and verified claims."""

    def __init__(
        self,
        min_token_overlap_ratio: float = 0.30,
        stop_words: Optional[Set[str]] = None,
    ):
        """Initialize AttributionEngine.

        Args:
            min_token_overlap_ratio: Minimum content token overlap ratio for grounding.
            stop_words: Set of words to exclude from lexical overlap calculation.
        """
        self.min_token_overlap_ratio = min_token_overlap_ratio
        self.stop_words = stop_words or {
            "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "and", "or", "but", "if", "then", "which", "what",
            "who", "whom", "whose", "where", "when", "why", "how", "this", "that",
            "these", "those", "it", "its", "they", "them", "their", "also", "however",
        }

    def segment_sentences(self, text: str) -> List[str]:
        """Split answer text into constituent sentences/claims."""
        clean = text.strip()
        if not clean:
            return []
        # Split on period, question mark, or newline while preserving abbreviations
        raw_sentences = re.split(r"(?<=[.?!])\s+|\n+", clean)
        sentences = [s.strip() for s in raw_sentences if s.strip() and len(s.strip()) > 3]
        return sentences if sentences else [clean]

    def extract_explicit_citations(self, text: str) -> List[int]:
        """Extract explicit bracketed citation numbers like [1], [2], [1, 2] from text."""
        citations = []
        matches = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text)
        for m in matches:
            for num_str in m.split(","):
                try:
                    citations.append(int(num_str.strip()))
                except ValueError:
                    pass
        return sorted(list(set(citations)))

    def attribute_answer(
        self,
        answer_text: str,
        evidence_chunks: List[Dict[str, Any]],
        verified_claims: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AnswerClaimAttribution]:
        """Attribute each sentence in the answer to retrieved evidence and verified claims.

        Args:
            answer_text: The complete generated response string.
            evidence_chunks: List of retrieved evidence chunk dictionaries.
            verified_claims: Optional list of verified claim dictionaries from the pipeline.

        Returns:
            List of AnswerClaimAttribution objects.
        """
        sentences = self.segment_sentences(answer_text)
        attributions: List[AnswerClaimAttribution] = []

        chunk_tokens_map = {}
        for idx, chunk in enumerate(evidence_chunks, start=1):
            full_text = f"{chunk.get('document_title', '')} {chunk.get('text', '')}".lower()
            tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", full_text)) - self.stop_words
            chunk_tokens_map[idx] = (tokens, chunk)

        for c_idx, sentence in enumerate(sentences, start=1):
            explicit_cits = self.extract_explicit_citations(sentence)
            sent_lower = sentence.lower()
            sent_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", sent_lower)) - self.stop_words

            supporting_ids = []
            matched_cits = list(explicit_cits)
            best_overlap = 0.0

            # 1. Check explicit citations first
            for cit in explicit_cits:
                if cit in chunk_tokens_map:
                    chunk_tokens, chunk_obj = chunk_tokens_map[cit]
                    cid = str(chunk_obj.get("chunk_id", f"rank_{cit}"))
                    supporting_ids.append(cid)

            # 2. Check lexical overlap across all chunks if no explicit citations or to supplement
            for cit_idx, (chunk_tokens, chunk_obj) in chunk_tokens_map.items():
                if sent_tokens and chunk_tokens:
                    overlap = len(sent_tokens.intersection(chunk_tokens)) / len(sent_tokens)
                    if overlap > best_overlap:
                        best_overlap = overlap
                    if overlap >= self.min_token_overlap_ratio and cit_idx not in matched_cits:
                        matched_cits.append(cit_idx)
                        cid = str(chunk_obj.get("chunk_id", f"rank_{cit_idx}"))
                        if cid not in supporting_ids:
                            supporting_ids.append(cid)

            is_supported = len(supporting_ids) > 0 or best_overlap >= self.min_token_overlap_ratio
            confidence = round(max(best_overlap, 1.0 if explicit_cits else 0.0), 3)

            # Link to verified claims if provided
            matched_claim_ids = []
            if verified_claims:
                for vc in verified_claims:
                    vc_id = vc.get("claim_id", "")
                    vc_text = vc.get("text", "").lower()
                    vc_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", vc_text)) - self.stop_words
                    if sent_tokens and vc_tokens:
                        if len(sent_tokens.intersection(vc_tokens)) / len(sent_tokens) >= 0.25:
                            matched_claim_ids.append(vc_id)

            explanation = (
                f"Supported by {len(supporting_ids)} chunk(s) (overlap={best_overlap:.2f}, "
                f"explicit_citations={explicit_cits})"
                if is_supported
                else f"No sufficient grounding found (max_overlap={best_overlap:.2f})"
            )

            attributions.append(
                AnswerClaimAttribution(
                    claim_index=c_idx,
                    claim_text=sentence,
                    cited_chunk_indices=matched_cits,
                    supporting_chunk_ids=supporting_ids,
                    verified_claim_ids=matched_claim_ids,
                    is_supported=is_supported,
                    grounding_confidence=confidence,
                    explanation=explanation,
                )
            )

        return attributions
