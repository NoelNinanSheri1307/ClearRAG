"""Retriever module for ClearRAG.

Combines Dense Vector Search (FAISS + BGE-small), Lexical BM25 Search,
Hybrid RRF Fusion, and Cross-Scorer Reranking to deliver high-precision evidence.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.retrieval.bm25 import BM25Index
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossScorerReranker

logger = logging.getLogger(__name__)


class Retriever:
    """End-to-end multi-modal evidence retriever for ClearRAG."""

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        index: Optional[FAISSIndex] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[CrossScorerReranker] = None,
        default_top_k: int = 10,
        mode: str = "dense",
    ):
        """Initialize the Retriever.

        Args:
            embedder: BGEEmbedder instance.
            index: FAISSIndex instance.
            bm25_index: BM25Index instance.
            reranker: CrossScorerReranker instance.
            default_top_k: Default number of documents/chunks to retrieve.
            mode: Retrieval strategy mode: 'dense', 'bm25', 'hybrid', 'hybrid_rerank'.
        """
        self.embedder = embedder or BGEEmbedder()
        self.index = index
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.default_top_k = default_top_k
        self.mode = mode

        # Setup internal hybrid retriever if components exist
        self._hybrid_engine: Optional[HybridRetriever] = None
        if self.index is not None and self.bm25_index is not None:
            self._hybrid_engine = HybridRetriever(
                embedder=self.embedder,
                faiss_index=self.index,
                bm25_index=self.bm25_index,
                default_top_k=self.default_top_k,
            )

    @classmethod
    def from_saved_index(
        cls,
        index_path: Path,
        metadata_path: Path,
        bm25_path: Optional[Path] = None,
        embedder: Optional[BGEEmbedder] = None,
        default_top_k: int = 10,
        mode: str = "dense",
        enable_reranker: bool = False,
    ) -> "Retriever":
        """Load a retriever instance directly from saved index and metadata files."""
        loaded_index = FAISSIndex.load(index_path, metadata_path)
        active_embedder = embedder or BGEEmbedder()

        bm25_idx = None
        if bm25_path is not None and bm25_path.exists():
            bm25_idx = BM25Index.load(bm25_path)
        elif mode in ("hybrid", "hybrid_rerank", "bm25"):
            # Build BM25 index in memory from the loaded FAISS metadata store
            bm25_idx = BM25Index()
            bm25_idx.build_from_metadata(loaded_index.metadata_store)

        reranker_instance = CrossScorerReranker() if (enable_reranker or mode == "hybrid_rerank") else None

        return cls(
            embedder=active_embedder,
            index=loaded_index,
            bm25_index=bm25_idx,
            reranker=reranker_instance,
            default_top_k=default_top_k,
            mode=mode,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        candidate_pool_k: int = 25,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K relevant chunks for a question.

        Args:
            query: Question or retrieval query string.
            top_k: Number of chunks to retrieve (defaults to self.default_top_k).
            mode: Override active retrieval strategy ('dense', 'bm25', 'hybrid', 'hybrid_rerank').
            candidate_pool_k: Number of candidates to fetch before reranking.

        Returns:
            List of ranked result dictionaries sorted by descending score.
        """
        active_mode = mode or self.mode
        k = top_k if top_k is not None else self.default_top_k

        # 1. Lexical BM25 Only
        if active_mode == "bm25":
            if self.bm25_index is None:
                raise ValueError("Retriever has no initialized BM25 index.")
            raw_bm25 = self.bm25_index.search(query, top_k=k)
            results = []
            for rank, (doc_idx, score) in enumerate(raw_bm25, start=1):
                meta = self.bm25_index.doc_metadata[doc_idx]
                results.append(
                    {
                        "rank": rank,
                        "chunk_id": meta.get("chunk_id", f"doc_{doc_idx}"),
                        "score": round(float(score), 4),
                        "document_title": meta.get("document_title", ""),
                        "text": meta.get("text", ""),
                        "sentence_indices": meta.get("sentence_indices", []),
                        "is_supporting_fact": meta.get("is_supporting_fact", False),
                        "provenance": {
                            "source_dataset": meta.get("source_dataset", ""),
                            "source_question_id": meta.get("source_question_id", ""),
                            "strategy": "bm25",
                        },
                    }
                )
            return results

        # 2. Hybrid (Dense + BM25 RRF)
        if active_mode == "hybrid":
            if self._hybrid_engine is None and (self.index is not None and self.bm25_index is not None):
                self._hybrid_engine = HybridRetriever(
                    embedder=self.embedder,
                    faiss_index=self.index,
                    bm25_index=self.bm25_index,
                )
            if self._hybrid_engine is not None:
                return self._hybrid_engine.retrieve(query, top_k=k, candidate_pool_k=candidate_pool_k)

        # 3. Hybrid + Reranking
        if active_mode == "hybrid_rerank":
            if self._hybrid_engine is None and (self.index is not None and self.bm25_index is not None):
                self._hybrid_engine = HybridRetriever(
                    embedder=self.embedder,
                    faiss_index=self.index,
                    bm25_index=self.bm25_index,
                )
            reranker = self.reranker or CrossScorerReranker()
            initial_pool = (
                self._hybrid_engine.retrieve(query, top_k=candidate_pool_k, candidate_pool_k=candidate_pool_k)
                if self._hybrid_engine is not None
                else self._retrieve_dense(query, top_k=candidate_pool_k)
            )
            return reranker.rerank(query, initial_pool, top_k=k)

        # 4. Default: Dense Vector Search
        return self._retrieve_dense(query, top_k=k)

    def _retrieve_dense(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Execute standard dense vector search against FAISS index."""
        if self.index is None:
            raise ValueError("Retriever has no initialized or loaded FAISS index.")

        query_vector = self.embedder.embed_query(query)
        scores, meta_list = self.index.search(query_vector, top_k=top_k)

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
                        "strategy": "dense",
                        "metadata": meta.get("metadata", {}),
                    },
                }
            )

        return results
