"""Semantic and Lexical Evidence Matching Engine for ClearRAG.

Evaluates claim support using dense semantic embedding similarity (BGE Embedder)
combined with non-stopword lexical alignment and exact entity presence.
"""

from collections import Counter
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.retrieval.embedder import BGEEmbedder
from src.verification.claims import Claim

logger = logging.getLogger(__name__)

STOP_WORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "and", "or", "but", "if", "then", "which", "what",
    "who", "whom", "whose", "where", "when", "why", "how", "this", "that",
    "these", "those", "it", "its", "they", "them", "their", "more", "most",
    "less", "least", "than", "as", "both", "either", "neither", "also", "into",
}


def extract_content_tokens(text: str) -> List[str]:
    """Extract lowercase non-stopword alphanumeric tokens."""
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


class SemanticEvidenceMatcher:
    """Matches claims against evidence passages using semantic similarity and lexical alignment."""

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        min_semantic_sim: float = 0.65,
        min_content_overlap_ratio: float = 0.35,
        min_entity_coverage_ratio: float = 0.50,
    ):
        """Initialize SemanticEvidenceMatcher.

        Args:
            embedder: BGEEmbedder instance.
            min_semantic_sim: Minimum cosine similarity threshold for semantic support.
            min_content_overlap_ratio: Minimum content token overlap ratio.
            min_entity_coverage_ratio: Minimum target entity token overlap ratio.
        """
        self.embedder = embedder
        self.min_semantic_sim = min_semantic_sim
        self.min_content_overlap_ratio = min_content_overlap_ratio
        self.min_entity_coverage_ratio = min_entity_coverage_ratio

    def evaluate_passage_support(
        self,
        claim: Claim,
        passage_text: str,
        passage_title: str = "",
    ) -> Tuple[float, bool, str]:
        """Evaluate if a single passage supports a given claim.

        Args:
            claim: Structured Claim object.
            passage_text: Evidence passage text.
            passage_title: Document title of the passage.

        Returns:
            Tuple of (support_score, is_supported, explanation).
        """
        full_passage = f"{passage_title} {passage_text}".strip()
        p_lower = full_passage.lower()

        if not p_lower:
            return 0.0, False, "Empty passage"

        # 1. Entity Coverage Check
        target_entities = [e.lower() for e in claim.target_entities if e]
        entity_found = True
        if target_entities:
            matched_entities = sum(1 for e in target_entities if e in p_lower)
            entity_ratio = matched_entities / len(target_entities)
            if entity_ratio < self.min_entity_coverage_ratio:
                entity_found = False

        if not entity_found:
            return 0.0, False, f"Target entities {target_entities} not found in passage"

        # 2. Content Token Overlap Check (Excluding generic stopwords)
        claim_content_tokens = set(extract_content_tokens(claim.source_question or claim.text))
        # Remove entity tokens from content tokens to isolate predicate requirements
        for ent in target_entities:
            claim_content_tokens.difference_update(extract_content_tokens(ent))

        passage_content_tokens = set(extract_content_tokens(full_passage))

        lexical_overlap_ratio = 1.0
        if claim_content_tokens:
            overlap_count = len(claim_content_tokens.intersection(passage_content_tokens))
            lexical_overlap_ratio = overlap_count / len(claim_content_tokens)

        # 3. Predicate & Attribute-Specific Verification
        predicate_satisfied, pred_reason = self._check_predicate_satisfaction(
            claim.predicate, full_passage, p_lower
        )

        # 4. Semantic Similarity (Dense Vector Cosine Similarity)
        semantic_sim = 0.0
        if self.embedder is not None:
            claim_text = claim.source_question or claim.text
            claim_vec = self.embedder.embed_query(claim_text)
            passage_vec = self.embedder.embed_texts([full_passage])[0]
            semantic_sim = float(np.dot(claim_vec, passage_vec))

        # Composite Support Decision
        # Either semantic similarity is very high with entity presence,
        # OR explicit predicate rules and content overlap are satisfied.
        is_supported = False
        if entity_found and predicate_satisfied and (lexical_overlap_ratio >= self.min_content_overlap_ratio or semantic_sim >= self.min_semantic_sim):
            is_supported = True
        elif entity_found and semantic_sim >= 0.78 and lexical_overlap_ratio >= 0.20:
            is_supported = True

        combined_score = max(semantic_sim, lexical_overlap_ratio if predicate_satisfied else 0.0)

        explanation = (
            f"EntityMatch={entity_found}, PredicateSatisfied={predicate_satisfied} ({pred_reason}), "
            f"LexOverlap={lexical_overlap_ratio:.2f}, SemanticSim={semantic_sim:.2f}"
        )
        return float(combined_score), is_supported, explanation

    def _check_predicate_satisfaction(
        self,
        predicate: str,
        passage: str,
        p_lower: str,
    ) -> Tuple[bool, str]:
        """Check if attribute/predicate requirements are met in the passage."""
        if predicate == "birth_date":
            has_birth_token = any(k in p_lower for k in ["born", "birth", "b.", "birthdate", "baptized"])
            has_year = bool(re.search(r"\b(16|17|18|19|20)\d{2}\b", passage))
            return (has_birth_token or has_year), f"birth_token={has_birth_token}, has_year={has_year}"

        elif predicate == "death_date":
            has_death_token = any(k in p_lower for k in ["died", "death", "d.", "killed", "passed away", "buried", "assassinated"])
            has_year = bool(re.search(r"\b(16|17|18|19|20)\d{2}\b", passage))
            return (has_death_token or has_year), f"death_token={has_death_token}, has_year={has_year}"

        elif predicate == "species_count":
            has_species = any(k in p_lower for k in ["species", "genus", "taxa", "family", "subspecies", "shrubs", "plants", "trees"])
            return has_species, f"has_species_indicator={has_species}"

        elif predicate == "population":
            has_pop = any(k in p_lower for k in ["population", "inhabitants", "people", "census", "residents", "pop."])
            has_number = bool(re.search(r"\b\d{1,3}(?:,\d{3})+|\b\d{4,}\b", passage))
            return (has_pop or has_number), f"has_pop={has_pop}, has_num={has_number}"

        elif predicate == "location":
            # Avoid single token 'in', require substantive location indicators or capitalized proper nouns
            has_loc = any(k in p_lower for k in ["located", "situated", "city", "town", "state", "country", "province", "county", "capital", "district", "village", "municipality"])
            return has_loc, f"has_loc_indicator={has_loc}"

        elif predicate == "release_date":
            has_rel = any(k in p_lower for k in ["released", "release", "published", "album", "film", "movie", "single", "series", "written", "directed", "created", "recorded", "premiered", "starring"])
            return has_rel, f"has_rel_indicator={has_rel}"

        elif predicate == "membership":
            has_mem = any(k in p_lower for k in ["starred", "played", "actor", "actress", "member", "band", "group", "guitarist", "vocalist", "drummer", "founder"])
            return has_mem, f"has_mem_indicator={has_mem}"

        # Default general_fact
        return True, "general_fact"
