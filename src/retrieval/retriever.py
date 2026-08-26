"""Retriever module for ClearRAG.

Combines query embedding and vector indexing to execute top-k semantic search
returning provenance-rich evidence passages.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.retrieval.embedder import BGEEmbedder
from src.retrieval.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


class Retriever:
    """End-to-end evidence retriever combining BGE embeddings and FAISS index."""

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        index: Optional[FAISSIndex] = None,
        default_top_k: int = 10,
    ):
        """Initialize the Retriever.

        Args:
            embedder: BGEEmbedder instance.
            index: FAISSIndex instance.
            default_top_k: Default number of documents/chunks to retrieve.
        """
        self.embedder = embedder or BGEEmbedder()
        self.index = index
        self.default_top_k = default_top_k

    @classmethod
    def from_saved_index(
        cls,
        index_path: Path,
        metadata_path: Path,
        embedder: Optional[BGEEmbedder] = None,
        default_top_k: int = 10,
    ) -> "Retriever":
        """Load a retriever instance directly from saved index and metadata files."""
        loaded_index = FAISSIndex.load(index_path, metadata_path)
        active_embedder = embedder or BGEEmbedder()
        return cls(embedder=active_embedder, index=loaded_index, default_top_k=default_top_k)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K relevant chunks for a question.

        Args:
            query: Question or retrieval query string.
            top_k: Number of chunks to retrieve (defaults to self.default_top_k).

        Returns:
            List of ranked result dictionaries sorted by descending score.
        """
        if self.index is None:
            raise ValueError("Retriever has no initialized or loaded FAISS index.")

        k = top_k if top_k is not None else self.default_top_k
        query_vector = self.embedder.embed_query(query)

        scores, meta_list = self.index.search(query_vector, top_k=k)

        results: List[Dict[str, Any]] = []
        for rank, (score, meta) in enumerate(zip(scores, meta_list), start=1):
            results.append(
                {
                    "rank": rank,
                    "chunk_id": meta.get("chunk_id", f"rank_{rank}"),
                    "score": float(score),
                    "document_title": meta.get("document_title", ""),
                    "text": meta.get("text", ""),
                    "sentence_indices": meta.get("sentence_indices", []),
                    "is_supporting_fact": meta.get("is_supporting_fact", False),
                    "provenance": {
                        "source_dataset": meta.get("source_dataset", ""),
                        "source_question_id": meta.get("source_question_id", ""),
                        "metadata": meta.get("metadata", {}),
                    },
                }
            )

        return results
