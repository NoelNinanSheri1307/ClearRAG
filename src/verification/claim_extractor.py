"""Claim extraction interface and implementations for ClearRAG."""

from abc import ABC, abstractmethod
import logging
import re
from typing import List, Tuple

from src.verification.claims import Claim, ClaimType

logger = logging.getLogger(__name__)


class BaseClaimExtractor(ABC):
    """Abstract interface for claim extraction strategies."""

    @abstractmethod
    def extract_claims(self, question: str) -> List[Claim]:
        """Extract structured claims required to answer a question.

        Args:
            question: Input question string.

        Returns:
            List of extracted Claim objects.
        """
        pass


class RuleBasedClaimExtractor(BaseClaimExtractor):
    """Deterministic, rule-based claim extractor for QA questions."""

    def __init__(self, max_claims: int = 4):
        """Initialize RuleBasedClaimExtractor.

        Args:
            max_claims: Maximum claims allowed per question.
        """
        self.max_claims = max_claims

    def extract_claims(self, question: str) -> List[Claim]:
        """Extract claims from question based on linguistic and entity patterns.

        Args:
            question: Input question string.

        Returns:
            List of Claim objects.
        """
        clean_q = question.strip()
        if not clean_q:
            return []

        predicate = self._detect_predicate(clean_q)

        # 1. Check for comparison / dual-entity question patterns
        comparison_entities = self._extract_comparison_entities(clean_q)
        if comparison_entities and len(comparison_entities) == 2:
            entity_a, entity_b = comparison_entities
            claim_a = Claim(
                claim_id="claim_1",
                text=f"Factual details/attributes for entity '{entity_a}' regarding '{predicate}' in question: {clean_q}",
                claim_type=ClaimType.COMPARISON_ENTITY_A,
                target_entities=[entity_a],
                predicate=predicate,
                source_question=clean_q,
                metadata={"entity_role": "entity_a"},
            )
            claim_b = Claim(
                claim_id="claim_2",
                text=f"Factual details/attributes for entity '{entity_b}' regarding '{predicate}' in question: {clean_q}",
                claim_type=ClaimType.COMPARISON_ENTITY_B,
                target_entities=[entity_b],
                predicate=predicate,
                source_question=clean_q,
                metadata={"entity_role": "entity_b"},
            )
            return [claim_a, claim_b]

        # 2. Extract capitalized entity names for multi-entity / bridge questions
        entities = self._extract_named_entities(clean_q)
        if len(entities) >= 2:
            claims = []
            for idx, entity in enumerate(entities[: self.max_claims], start=1):
                claims.append(
                    Claim(
                        claim_id=f"claim_{idx}",
                        text=f"Factual details for entity '{entity}' regarding '{predicate}' relevant to: {clean_q}",
                        claim_type=ClaimType.MULTI_HOP,
                        target_entities=[entity],
                        predicate=predicate,
                        source_question=clean_q,
                    )
                )
            return claims

        # 3. Default single atomic factual claim
        primary_entity = entities[0] if entities else ""
        return [
            Claim(
                claim_id="claim_1",
                text=clean_q,
                claim_type=ClaimType.ATOMIC_FACT,
                target_entities=[primary_entity] if primary_entity else [],
                predicate=predicate,
                source_question=clean_q,
            )
        ]

    def _detect_predicate(self, question: str) -> str:
        """Identify the requested attribute/predicate from question text."""
        q_lower = question.lower()
        if re.search(r"\b(won|winner|champion|championship|victory|victor|title|defeated|beat|gold medal|trophy)\b", q_lower):
            return "award_winner"
        elif re.search(r"\b(founded|founder|started|established|originated|invented|inventor)\b", q_lower):
            return "founder_creator"
        elif re.search(r"\b(directed|director|filmmaker|wrote|author|novelist|composed|composer)\b", q_lower):
            return "director_author"
        elif re.search(r"\b(mother|father|parent|son|daughter|child|mom|dad)\b", q_lower):
            return "parent_child"
        elif re.search(r"\b(married|spouse|wife|husband|wedding)\b", q_lower):
            return "spouse_marriage"
        elif re.search(r"\b(born|birth|b\.)\b", q_lower):
            return "birth_date"
        elif re.search(r"\b(died|death|d\.)\b", q_lower):
            return "death_date"
        elif re.search(r"\b(species|genus)\b", q_lower):
            return "species_count"
        elif re.search(r"\b(population|inhabitants|people)\b", q_lower):
            return "population"
        elif re.search(r"\b(located|location|where|city|town|state|country|province|county|capital)\b", q_lower):
            return "location"
        elif re.search(r"\b(released|release|year|when|published|created)\b", q_lower):
            return "release_date"
        elif re.search(r"\b(starred|played|actor|actress|member|band|group|cast)\b", q_lower):
            return "membership"
        return "general_fact"

    def _extract_comparison_entities(self, question: str) -> List[str]:
        """Detect comparison patterns such as 'Which X, A or B?' or 'Who was Y, A or B?'."""
        or_pattern = r"(?:which|who|where|what|is|are|was|were).*?([^,?:;]+?)\s+(?:or|and)\s+([^,?:;]+?)\?$"
        match = re.search(or_pattern, question, re.IGNORECASE)
        if match:
            cand_a = match.group(1).strip()
            cand_b = match.group(2).strip()
            cand_a = re.sub(r"^(?:between|both|either|the|a|an)\s+", "", cand_a, flags=re.IGNORECASE).strip()
            cand_b = re.sub(r"^(?:between|both|either|the|a|an)\s+", "", cand_b, flags=re.IGNORECASE).strip()
            if cand_a and cand_b and cand_a.lower() != cand_b.lower():
                return [cand_a, cand_b]

        or_mid_pattern = r"\b([A-Z][a-zA-Z0-9_\-\.\s]{1,30})\s+or\s+([A-Z][a-zA-Z0-9_\-\.\s]{1,30})\b"
        match_mid = re.search(or_mid_pattern, question)
        if match_mid:
            cand_a = match_mid.group(1).strip()
            cand_b = match_mid.group(2).strip()
            if cand_a and cand_b and cand_a.lower() != cand_b.lower():
                return [cand_a, cand_b]

        return []

    def _extract_named_entities(self, question: str) -> List[str]:
        """Extract capitalized noun phrases representing candidate named entities."""
        ignore_words = {"Which", "What", "Who", "Where", "When", "Why", "How", "Is", "Are", "Was", "Were", "Did", "Does", "Do", "The", "A", "An"}
        words = question.split()
        if words and words[0] in ignore_words:
            words = words[1:]

        text = " ".join(words)
        matches = re.findall(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b", text)
        result = []
        for m in matches:
            cleaned = m.strip("?.,!")
            if cleaned and cleaned not in ignore_words and cleaned not in result:
                result.append(cleaned)
        return result
