"""Corpus builder module for ClearRAG.

Processes raw HotpotQA questions and contexts into structured, provenance-preserving
corpus chunks ready for embedding and retrieval indexing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class CorpusBuilder:
    """Builds a structured chunk corpus from raw HotpotQA data."""

    def __init__(self, deduplicate_by_title_and_idx: bool = True):
        """Initialize CorpusBuilder.

        Args:
            deduplicate_by_title_and_idx: If True, merges identical sentences from
                the same document title across questions while recording all
                associated question provenance IDs.
        """
        self.deduplicate = deduplicate_by_title_and_idx

    def build_from_hotpotqa(
        self,
        raw_data: List[Dict[str, Any]],
        max_questions: Optional[int] = None,
    ) -> List[Chunk]:
        """Build corpus chunks from a list of HotpotQA question objects.

        Args:
            raw_data: List of raw HotpotQA dictionaries.
            max_questions: Optional cap on the number of questions to process.

        Returns:
            List of Chunk objects with complete provenance.
        """
        if max_questions is not None and max_questions > 0:
            items = raw_data[:max_questions]
        else:
            items = raw_data

        seen_keys: Dict[Tuple[str, int], Chunk] = {}
        chunks_list: List[Chunk] = []

        for item_idx, item in enumerate(items):
            source_id = str(item.get("_id", f"item_{item_idx}"))
            raw_context = item.get("context", [])
            raw_facts = item.get("supporting_facts", [])

            supporting_facts_set: Set[Tuple[str, int]] = {
                (str(f[0]), int(f[1])) for f in raw_facts
            }

            for doc_idx, doc in enumerate(raw_context):
                doc_title = str(doc[0])
                sentences = doc[1] if len(doc) > 1 else []

                for sentence_idx, raw_sent in enumerate(sentences):
                    sentence_text = str(raw_sent).strip()
                    if not sentence_text:
                        continue

                    key = (doc_title, sentence_idx)
                    is_support = key in supporting_facts_set

                    if self.deduplicate and key in seen_keys:
                        # Append question provenance to existing chunk
                        existing_chunk = seen_keys[key]
                        if source_id not in existing_chunk.metadata["source_question_ids"]:
                            existing_chunk.metadata["source_question_ids"].append(source_id)
                        if is_support and source_id not in existing_chunk.metadata["supporting_question_ids"]:
                            existing_chunk.metadata["supporting_question_ids"].append(source_id)
                            existing_chunk.is_supporting_fact = True
                        continue

                    # Create new chunk
                    chunk_id = f"chunk_{len(chunks_list):07d}"
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        source_dataset="HotpotQA",
                        source_question_id=source_id,
                        document_title=doc_title,
                        sentence_indices=[sentence_idx],
                        text=sentence_text,
                        is_supporting_fact=is_support,
                        metadata={
                            "doc_index": doc_idx,
                            "source_question_ids": [source_id],
                            "supporting_question_ids": [source_id] if is_support else [],
                            "total_doc_sentences": len(sentences),
                        },
                    )

                    if self.deduplicate:
                        seen_keys[key] = chunk

                    chunks_list.append(chunk)

        logger.info(
            f"Built corpus with {len(chunks_list):,} chunks from {len(items):,} questions."
        )
        return chunks_list

    def save_corpus(self, chunks: List[Chunk], output_path: Path) -> None:
        """Save list of chunks to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [chunk.to_dict() for chunk in chunks]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks):,} chunks to {output_path}")

    def load_corpus(self, input_path: Path) -> List[Chunk]:
        """Load list of chunks from a JSON file."""
        if not input_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Chunk.from_dict(item) for item in data]
