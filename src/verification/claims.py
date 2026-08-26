"""Structured claim representations for ClearRAG verification layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ClaimType(str, Enum):
    """Enumeration of claim types for questions."""
    ATOMIC_FACT = "atomic_fact"
    COMPARISON_ENTITY_A = "comparison_entity_a"
    COMPARISON_ENTITY_B = "comparison_entity_b"
    MULTI_HOP = "multi_hop"


@dataclass
class Claim:
    """Structured representation of an extracted claim."""
    claim_id: str
    text: str
    claim_type: ClaimType
    target_entities: List[str] = field(default_factory=list)
    predicate: str = "general_fact"  # E.g. birth_date, death_date, species_count, population, location, release_date
    source_question: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert claim to dictionary representation."""
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type.value,
            "target_entities": self.target_entities,
            "predicate": self.predicate,
            "source_question": self.source_question,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        """Create Claim from dictionary."""
        return cls(
            claim_id=data["claim_id"],
            text=data["text"],
            claim_type=ClaimType(data.get("claim_type", ClaimType.ATOMIC_FACT.value)),
            target_entities=data.get("target_entities", []),
            predicate=data.get("predicate", "general_fact"),
            source_question=data.get("source_question", ""),
            metadata=data.get("metadata", {}),
        )
