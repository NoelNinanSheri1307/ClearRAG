"""Reranker module for ClearRAG evidence candidate pools.

Re-ranks top-N candidate passages using fine-grained token-level cross-scoring,
exact named-entity alignment, and semantic similarity scoring.
"""

from collections import Counter, defaultdict
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_key_terms(text: str) -> List[str]:
    """Extract lowercase significant words (excluding common stop words)."""
    stop_words = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "and", "or", "but", "if", "then", "which", "what",
        "who", "whom", "whose", "where", "when", "why", "how", "this", "that",
        "these", "those", "it", "its", "they", "them", "their", "more", "most",
        "less", "least", "than", "as", "both", "either", "neither",
    }
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [t for t in tokens if t not in stop_words and len(t) > 1]


class CrossScorerReranker:
    """Fast, deterministic entity-aware cross-scorer reranker for retrieved candidate pools."""

    def __init__(
        self,
        title_exact_match_boost: float = 3.0,
        entity_token_boost: float = 2.0,
        content_overlap_weight: float = 1.0,
        initial_rank_decay: float = 0.05,
        max_chunks_per_doc: int = 2,
    ):
        """Initialize CrossScorerReranker.

        Args:
            title_exact_match_boost: Weight boost for title tokens present in question.
            entity_token_boost: Boost for capitalized/named entities matching chunk text.
            content_overlap_weight: Weight for question term overlap with chunk body.
            initial_rank_decay: Weight discount based on incoming retrieval rank.
            max_chunks_per_doc: Maximum number of chunks allowed from the same document title.
        """
        self.title_exact_match_boost = title_exact_match_boost
        self.entity_token_boost = entity_token_boost
        self.content_overlap_weight = content_overlap_weight
        self.initial_rank_decay = initial_rank_decay
        self.max_chunks_per_doc = max_chunks_per_doc

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank a pool of candidate chunk dictionaries for a given query.

        Args:
            query: Input natural language query.
            candidates: List of candidate chunk dictionaries from initial retrieval.
            top_k: Number of highest-ranked chunks to return.

        Returns:
            Reranked list of chunk dictionaries with updated scores and ranks.
        """
        if not candidates:
            return []

        query_terms = extract_key_terms(query)
        query_terms_set = set(query_terms)

        # Extract capitalized entities from query
        query_entities = re.findall(r"\b[A-Z][a-zA-Z0-9_-]*(?:\s+[A-Z][a-zA-Z0-9_-]*)*\b", query)
        query_entity_tokens = set()
        for ent in query_entities:
            query_entity_tokens.update(extract_key_terms(ent))

        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

        for idx, cand in enumerate(candidates):
            title = cand.get("document_title", "")
            text = cand.get("text", "")
            initial_score = float(cand.get("score", 1.0))
            incoming_rank = cand.get("rank", idx + 1)

            title_terms = extract_key_terms(title)
            text_terms = extract_key_terms(text)

            title_terms_set = set(title_terms)
            text_terms_set = set(text_terms)

            # 1. Title match score (exact token matches in title)
            title_overlap = len(query_terms_set.intersection(title_terms_set))
            title_score = title_overlap * self.title_exact_match_boost

            # 2. Named entity match score
            entity_overlap = len(query_entity_tokens.intersection(title_terms_set.union(text_terms_set)))
            entity_score = entity_overlap * self.entity_token_boost

            # 3. Content term coverage (recall of query terms in chunk text)
            content_overlap = len(query_terms_set.intersection(text_terms_set))
            term_coverage = content_overlap / len(query_terms_set) if query_terms_set else 0.0
            content_score = term_coverage * self.content_overlap_weight

            # 4. Rank preservation penalty (small decaying bias towards top initial candidates)
            rank_bonus = 1.0 / (1.0 + self.initial_rank_decay * incoming_rank)

            # Combined cross-score
            total_rerank_score = (
                title_score + entity_score + content_score + (initial_score * 0.5) + rank_bonus
            )

            scored_candidates.append((total_rerank_score, cand))

        # Sort descending by rerank score
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        doc_counts: Dict[str, int] = defaultdict(int)

        for score, cand in sorted_candidates:
            title = cand.get("document_title", "")
            if self.max_chunks_per_doc > 0 and title and doc_counts[title] >= self.max_chunks_per_doc:
                continue

            doc_counts[title] += 1
            rank = len(results) + 1

            chunk_copy = dict(cand)
            chunk_copy["rank"] = rank
            chunk_copy["score"] = round(float(score), 4)
            provenance = dict(chunk_copy.get("provenance", {}))
            provenance["reranked"] = True
            provenance["rerank_score"] = round(float(score), 4)
            chunk_copy["provenance"] = provenance
            results.append(chunk_copy)

            if len(results) >= top_k:
                break

        return results
