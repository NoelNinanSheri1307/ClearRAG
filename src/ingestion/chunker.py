"""Chunk representation and chunking strategies for ClearRAG.

Preserves exact provenance (source dataset, question ID, document title,
and sentence indices) to support evaluation and evidence verification.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """Represents a text chunk with complete provenance metadata."""

    chunk_id: str
    source_dataset: str
    source_question_id: str
    document_title: str
    sentence_indices: List[int]
    text: str
    is_supporting_fact: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create a Chunk instance from a dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            source_dataset=data["source_dataset"],
            source_question_id=data["source_question_id"],
            document_title=data["document_title"],
            sentence_indices=data["sentence_indices"],
            text=data["text"],
            is_supporting_fact=data.get("is_supporting_fact", False),
            metadata=data.get("metadata", {}),
        )


def chunk_document_by_sentence(
    source_dataset: str,
    source_question_id: str,
    doc_index: int,
    document_title: str,
    sentences: List[str],
    supporting_facts_set: Optional[set] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """Chunk a document into sentence-level chunks with exact provenance.

    Args:
        source_dataset: Origin dataset name (e.g. 'HotpotQA').
        source_question_id: ID of the question/context item.
        doc_index: Numerical index of the document in the context list.
        document_title: Title of the document / Wikipedia article.
        sentences: List of sentences belonging to the document.
        supporting_facts_set: Set of (title, sentence_index) tuples representing gold facts.
        metadata: Optional additional metadata dict.

    Returns:
        List of Chunk objects, one per non-empty sentence.
    """
    if supporting_facts_set is None:
        supporting_facts_set = set()
    if metadata is None:
        metadata = {}

    chunks: List[Chunk] = []

    for sentence_idx, raw_sentence in enumerate(sentences):
        sentence_text = raw_sentence.strip()
        if not sentence_text:
            continue

        chunk_id = f"{source_question_id}_d{doc_index}_s{sentence_idx}"
        is_support = (document_title, sentence_idx) in supporting_facts_set

        chunk = Chunk(
            chunk_id=chunk_id,
            source_dataset=source_dataset,
            source_question_id=source_question_id,
            document_title=document_title,
            sentence_indices=[sentence_idx],
            text=sentence_text,
            is_supporting_fact=is_support,
            metadata={
                **metadata,
                "doc_index": doc_index,
                "total_sentences_in_doc": len(sentences),
            },
        )
        chunks.append(chunk)

    return chunks
