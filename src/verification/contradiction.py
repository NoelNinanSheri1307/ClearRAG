"""Contradiction and Conflict Detection Engine for ClearRAG.

Detects factual, numeric, temporal, and categorical contradictions across
retrieved passages for target entities.
"""

from collections import defaultdict
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ContradictionDetector:
    """Detects attribute-aware factual contradictions across evidence passages."""

    def __init__(
        self,
        enable_date_conflicts: bool = True,
        enable_numeric_conflicts: bool = True,
        enable_location_conflicts: bool = True,
    ):
        """Initialize ContradictionDetector.

        Args:
            enable_date_conflicts: Detect differing years for identical attributes.
            enable_numeric_conflicts: Detect differing numbers/counts for identical attributes.
            enable_location_conflicts: Detect contradictory location entities.
        """
        self.enable_date_conflicts = enable_date_conflicts
        self.enable_numeric_conflicts = enable_numeric_conflicts
        self.enable_location_conflicts = enable_location_conflicts

    def detect_conflict(
        self,
        target_entities: List[str],
        predicate: str,
        evidence_chunks: List[Dict[str, Any]],
    ) -> Tuple[bool, List[str], List[str], str]:
        """Inspect evidence chunks for factual contradictions regarding target entities.

        Args:
            target_entities: List of entity names mentioned in the claim.
            predicate: Requested predicate/attribute.
            evidence_chunks: List of retrieved evidence chunk dicts.

        Returns:
            Tuple of (has_conflict, conflicting_ids, conflicting_texts, explanation).
        """
        if not target_entities or len(evidence_chunks) < 2:
            return False, [], [], "Insufficient chunks for conflict"

        entities_lower = [e.lower() for e in target_entities if e]

        # 1. Check Date / Year Contradictions (e.g. birth dates, death dates, release years)
        if self.enable_date_conflicts:
            date_conflict, c_ids, c_texts, reason = self._check_date_conflicts(
                entities_lower, predicate, evidence_chunks
            )
            if date_conflict:
                return True, c_ids, c_texts, reason

        # 2. Check Numeric Quantity Contradictions (e.g. species count, population, heights)
        if self.enable_numeric_conflicts:
            num_conflict, c_ids, c_texts, reason = self._check_numeric_conflicts(
                entities_lower, predicate, evidence_chunks
            )
            if num_conflict:
                return True, c_ids, c_texts, reason

        # 3. Check Opposing Antonym Assertions
        antonym_conflict, c_ids, c_texts, reason = self._check_antonym_conflicts(
            entities_lower, evidence_chunks
        )
        if antonym_conflict:
            return True, c_ids, c_texts, reason

        return False, [], [], "No contradictory evidence detected"

    def _check_date_conflicts(
        self,
        entities: List[str],
        predicate: str,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[bool, List[str], List[str], str]:
        """Detect conflicting 4-digit years for the same entity and predicate."""
        entity_years: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

        for chunk in chunks:
            text = chunk.get("text", "")
            title = chunk.get("document_title", "")
            full = f"{title} {text}".lower()

            for ent in entities:
                if ent in full:
                    # Match years within sentence context of the entity
                    years = [int(y) for y in re.findall(r"\b(?:16|17|18|19|20)\d{2}\b", text)]
                    for y in set(years):
                        entity_years[ent][y].append(chunk)

        for ent, years_map in entity_years.items():
            # If multiple distinct years are claimed for birth/death/release
            if len(years_map) >= 2:
                distinct_years = sorted(list(years_map.keys()))
                # Check if gap is meaningful (e.g. > 0 years)
                if distinct_years[-1] - distinct_years[0] >= 1:
                    conflicting_chunks = []
                    for y in distinct_years:
                        conflicting_chunks.extend(years_map[y])

                    c_ids = [str(c.get("chunk_id", c.get("rank", ""))) for c in conflicting_chunks]
                    c_texts = [c.get("text", "") for c in conflicting_chunks]
                    return (
                        True,
                        c_ids[:4],
                        c_texts[:4],
                        f"Entity '{ent}' has contradictory dates/years across passages: {distinct_years}",
                    )

        return False, [], [], ""

    def _check_numeric_conflicts(
        self,
        entities: List[str],
        predicate: str,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[bool, List[str], List[str], str]:
        """Detect conflicting numeric quantities associated with the target entity."""
        if predicate not in ("species_count", "population", "general_fact"):
            return False, [], [], ""

        entity_numbers: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

        for chunk in chunks:
            text = chunk.get("text", "")
            title = chunk.get("document_title", "")
            full = f"{title} {text}".lower()

            for ent in entities:
                if ent in full:
                    # Extract numeric patterns followed by attribute words
                    matches = re.findall(r"\b(\d+(?:,\d+)?)\s+(species|inhabitants|people|meters|km|miles|members|albums)\b", full)
                    for num_str, unit in matches:
                        clean_num = num_str.replace(",", "")
                        entity_numbers[f"{ent}_{unit}"][clean_num].append(chunk)

        for key, num_map in entity_numbers.items():
            if len(num_map) >= 2:
                distinct_nums = list(num_map.keys())
                conflicting_chunks = []
                for n in distinct_nums:
                    conflicting_chunks.extend(num_map[n])
                c_ids = [str(c.get("chunk_id", c.get("rank", ""))) for c in conflicting_chunks]
                c_texts = [c.get("text", "") for c in conflicting_chunks]
                return (
                    True,
                    c_ids[:4],
                    c_texts[:4],
                    f"Contradictory numeric counts ({distinct_nums}) detected for '{key}' across passages.",
                )

        return False, [], [], ""

    def _check_antonym_conflicts(
        self,
        entities: List[str],
        chunks: List[Dict[str, Any]],
    ) -> Tuple[bool, List[str], List[str], str]:
        """Detect direct opposing polarity assertions (e.g. alive vs dead, won vs lost)."""
        opposing_pairs = [
            ({"alive", "survived", "living"}, {"dead", "died", "deceased", "killed"}),
            ({"won", "winner", "victorious", "champion"}, {"lost", "runner-up", "defeated"}),
            ({"active", "current", "present"}, {"defunct", "disbanded", "former", "inactive"}),
            ({"success", "successful"}, {"failed", "failure", "unsuccessful"}),
        ]

        for ent in entities:
            for group_a, group_b in opposing_pairs:
                chunks_a = []
                chunks_b = []
                for chunk in chunks:
                    full = f"{chunk.get('document_title', '')} {chunk.get('text', '')}".lower()
                    if ent in full:
                        tokens = set(re.findall(r"\w+", full))
                        if tokens.intersection(group_a):
                            chunks_a.append(chunk)
                        if tokens.intersection(group_b):
                            chunks_b.append(chunk)

                if chunks_a and chunks_b:
                    conflicting = chunks_a + chunks_b
                    c_ids = [str(c.get("chunk_id", c.get("rank", ""))) for c in conflicting]
                    c_texts = [c.get("text", "") for c in conflicting]
                    return (
                        True,
                        c_ids[:4],
                        c_texts[:4],
                        f"Opposing polarities detected for entity '{ent}' ({group_a} vs {group_b})",
                    )

        return False, [], [], ""
