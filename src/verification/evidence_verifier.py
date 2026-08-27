"""Evidence verification and conflict detection module for ClearRAG."""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.verification.claims import Claim
from src.verification.models import ClaimVerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


class EvidenceVerifier:
    """Evaluates evidence support and detects attribute-aware factual/numeric conflicts for extracted claims."""

    def __init__(
        self,
        min_entity_match_ratio: float = 0.5,
        min_lexical_overlap_ratio: float = 0.30,
        support_score_threshold: float = 0.60,
        enable_conflict_detection: bool = True,
    ):
        """Initialize EvidenceVerifier."""
        self.min_entity_match_ratio = min_entity_match_ratio
        self.min_lexical_overlap_ratio = min_lexical_overlap_ratio
        self.support_score_threshold = support_score_threshold
        self.enable_conflict_detection = enable_conflict_detection

    def verify_claim(
        self,
        claim: Claim,
        evidence_chunks: List[Dict[str, Any]],
    ) -> ClaimVerificationResult:
        """Verify a single claim against retrieved evidence chunks."""
        if not evidence_chunks:
            return ClaimVerificationResult(
                claim=claim,
                status=VerificationStatus.UNSUPPORTED,
                confidence_score=0.0,
                reason="No evidence chunks provided.",
            )

        supporting_ids = []
        supporting_texts = []
        max_support_score = 0.0

        entities = [e.lower() for e in claim.target_entities if e]

        # 1. Attribute-aware conflict detection across evidence chunks for claim entities
        if self.enable_conflict_detection and entities:
            has_conflict, c_ids, c_texts, c_reason = self._detect_attribute_aware_conflict(entities, claim.predicate, evidence_chunks)
            if has_conflict:
                return ClaimVerificationResult(
                    claim=claim,
                    status=VerificationStatus.CONFLICTING,
                    conflicting_evidence_ids=c_ids,
                    conflicting_evidence_texts=c_texts,
                    confidence_score=1.0,
                    reason=f"Conflicting evidence detected for attribute '{claim.predicate}': {c_reason}",
                )

        # 2. Evaluate support for claim predicate chunk by chunk
        for chunk in evidence_chunks:
            chunk_id = str(chunk.get("chunk_id", chunk.get("rank", "unknown")))
            text = chunk.get("text", "")
            title = chunk.get("document_title", "")
            full_passage = f"{title} {text}".strip()

            score, is_supported, reason = self._evaluate_predicate_support(claim, entities, full_passage)

            if is_supported:
                supporting_ids.append(chunk_id)
                supporting_texts.append(text)
                if score > max_support_score:
                    max_support_score = score

        if supporting_ids:
            return ClaimVerificationResult(
                claim=claim,
                status=VerificationStatus.SUPPORTED,
                supporting_evidence_ids=supporting_ids,
                supporting_evidence_texts=supporting_texts,
                confidence_score=max_support_score,
                reason=f"Claim predicate '{claim.predicate}' supported by {len(supporting_ids)} chunk(s). Max score={max_support_score:.2f}",
            )
        else:
            return ClaimVerificationResult(
                claim=claim,
                status=VerificationStatus.UNSUPPORTED,
                confidence_score=0.0,
                reason=f"Retrieved evidence lacks required predicate support for '{claim.predicate}'.",
            )

    def verify_claims(
        self,
        claims: List[Claim],
        evidence_chunks: List[Dict[str, Any]],
    ) -> List[ClaimVerificationResult]:
        """Verify multiple extracted claims against retrieved evidence chunks."""
        return [self.verify_claim(c, evidence_chunks) for c in claims]

    def _evaluate_predicate_support(
        self,
        claim: Claim,
        target_entities: List[str],
        passage: str,
    ) -> Tuple[float, bool, str]:
        """Evaluate if a passage supports the claim's specific predicate/attribute."""
        p_lower = passage.lower()
        if not p_lower:
            return 0.0, False, "Empty passage"

        # Entity match check
        entity_found = True
        if target_entities:
            entity_matches = sum(1 for e in target_entities if e in p_lower)
            entity_ratio = entity_matches / len(target_entities)
            if entity_ratio < self.min_entity_match_ratio:
                entity_found = False

        if not entity_found:
            return 0.0, False, "Target entities missing in passage"

        # Predicate-specific verification check
        predicate_satisfied = False
        pred = claim.predicate

        if pred == "birth_date":
            # Must mention birth tokens and a year/date
            has_birth_token = any(k in p_lower for k in ["born", "birth", "b.", "birthdate"])
            has_year = bool(re.search(r"\b(17|18|19|20)\d{2}\b", passage))
            predicate_satisfied = has_birth_token and has_year
        elif pred == "death_date":
            # Must mention death tokens and a year/date
            has_death_token = any(k in p_lower for k in ["died", "death", "d.", "passed away"])
            has_year = bool(re.search(r"\b(17|18|19|20)\d{2}\b", passage))
            predicate_satisfied = has_death_token and has_year
        elif pred == "species_count":
            predicate_satisfied = any(k in p_lower for k in ["species", "genus", "taxa", "family"])
        elif pred == "population":
            has_pop_token = any(k in p_lower for k in ["population", "inhabitants", "people", "census"])
            has_number = bool(re.search(r"\b\d{1,3}(?:,\d{3})+|\b\d{4,}\b", passage))
            predicate_satisfied = has_pop_token or has_number
        elif pred == "location":
            predicate_satisfied = any(k in p_lower for k in ["located", "in", "city", "town", "state", "country", "province", "county", "capital", "district"])
        elif pred == "release_date":
            has_rel_token = any(k in p_lower for k in ["released", "release", "published", "album", "film", "movie", "single", "series", "written", "directed", "created"])
            has_year = bool(re.search(r"\b(18|19|20)\d{2}\b", passage))
            predicate_satisfied = has_rel_token and has_year
        else:
            # Fallback general_fact: requiring lexical overlap beyond entity names
            claim_tokens = self._tokenize(claim.source_question or claim.text)
            passage_tokens = self._tokenize(passage)
            if claim_tokens:
                overlap = len(claim_tokens.intersection(passage_tokens))
                overlap_ratio = overlap / len(claim_tokens)
                predicate_satisfied = overlap_ratio >= self.min_lexical_overlap_ratio
            else:
                predicate_satisfied = True

        score = 1.0 if (entity_found and predicate_satisfied) else 0.0
        return score, predicate_satisfied, f"EntityMatch={entity_found}, PredicateSatisfied={predicate_satisfied}"

    def _detect_attribute_aware_conflict(
        self,
        entities: List[str],
        target_predicate: str,
        evidence_chunks: List[Dict[str, Any]],
    ) -> Tuple[bool, List[str], List[str], str]:
        """Detect conflicting numeric or date values ONLY when referring to the SAME entity AND SAME attribute."""
        entity_attribute_facts: List[Dict[str, Any]] = []

        for chunk in evidence_chunks:
            chunk_id = str(chunk.get("chunk_id", chunk.get("rank", "unknown")))
            text = chunk.get("text", "")
            title = chunk.get("document_title", "")
            full_passage = f"{title} {text}"
            p_lower = full_passage.lower()

            matching_entities = [e for e in entities if e in p_lower]
            if not matching_entities:
                continue

            # Extract birth years specifically
            birth_years = set()
            birth_match = re.findall(r"\b(?:born|birth)\b.*?\b(17\d{2}|18\d{2}|19\d{2}|20\d{2})\b", full_passage, re.IGNORECASE)
            if birth_match:
                birth_years.update(birth_match)

            # Extract death years specifically
            death_years = set()
            death_match = re.findall(r"\b(?:died|death)\b.*?\b(17\d{2}|18\d{2}|19\d{2}|20\d{2})\b", full_passage, re.IGNORECASE)
            if death_match:
                death_years.update(death_match)

            entity_attribute_facts.append({
                "chunk_id": chunk_id,
                "text": text,
                "title": title,
                "matching_entities": matching_entities,
                "birth_years": birth_years,
                "death_years": death_years,
            })

        # Compare pairs of passages for exact attribute conflicts
        for i in range(len(entity_attribute_facts)):
            for j in range(i + 1, len(entity_attribute_facts)):
                fact_i = entity_attribute_facts[i]
                fact_j = entity_attribute_facts[j]

                # Check shared entity
                shared = set(fact_i["matching_entities"]).intersection(set(fact_j["matching_entities"]))
                if not shared:
                    continue

                # Check birth year conflict for the same entity
                if fact_i["birth_years"] and fact_j["birth_years"] and fact_i["birth_years"] != fact_j["birth_years"]:
                    return (
                        True,
                        [fact_i["chunk_id"], fact_j["chunk_id"]],
                        [fact_i["text"], fact_j["text"]],
                        f"Incompatible birth years for entity '{list(shared)[0]}' ({fact_i['birth_years']} vs {fact_j['birth_years']})",
                    )

                # Check death year conflict for the same entity
                if fact_i["death_years"] and fact_j["death_years"] and fact_i["death_years"] != fact_j["death_years"]:
                    return (
                        True,
                        [fact_i["chunk_id"], fact_j["chunk_id"]],
                        [fact_i["text"], fact_j["text"]],
                        f"Incompatible death years for entity '{list(shared)[0]}' ({fact_i['death_years']} vs {fact_j['death_years']})",
                    )

        return False, [], [], ""

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize string into non-stopword set."""
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "what", "which", "who", "where", "when", "how"}
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {w for w in words if w not in stopwords and len(w) > 1}
