"""Hybrid Retrieval Engine for ClearRAG.

Fuses Dense Vector Search (BGE-small / FAISS) and Lexical Search (BM25)
using Reciprocal Rank Fusion (RRF) or Convex Combination Score Fusion.
"""

from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.retrieval.bm25 import BM25Index
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid Retriever combining Dense Vector and BM25 Lexical search."""

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        faiss_index: Optional[FAISSIndex] = None,
        bm25_index: Optional[BM25Index] = None,
        dense_weight: float = 0.5,
        lexical_weight: float = 0.5,
        rrf_c: int = 60,
        default_top_k: int = 10,
    ):
        """Initialize Hybrid Retriever.

        Args:
            embedder: BGEEmbedder instance.
            faiss_index: FAISSIndex instance.
            bm25_index: BM25Index instance.
            dense_weight: Weight for dense rankings/scores.
            lexical_weight: Weight for BM25 rankings/scores.
            rrf_c: Smoothing constant for Reciprocal Rank Fusion (default: 60).
            default_top_k: Default final candidate count to return.
        """
        self.embedder = embedder or BGEEmbedder()
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.rrf_c = rrf_c
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_pool_k: int = 40,
        fusion_method: str = "rrf",
    ) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval using RRF or convex combination fusion.

        Args:
            query: Input question string.
            top_k: Final number of ranked chunks to return.
            candidate_pool_k: Number of candidates retrieved from each individual retriever.
            fusion_method: 'rrf' (Reciprocal Rank Fusion) or 'weighted_sum'.

        Returns:
            List of ranked chunk dictionaries sorted by descending hybrid score.
        """
        k = top_k if top_k is not None else self.default_top_k

        # 1. Dense FAISS Retrieval
        dense_candidates: List[Dict[str, Any]] = []
        if self.faiss_index is not None and self.embedder is not None:
            query_vector = self.embedder.embed_query(query)
            scores, meta_list = self.faiss_index.search(query_vector, top_k=candidate_pool_k)
            for rank, (score, meta) in enumerate(zip(scores, meta_list), start=1):
                dense_candidates.append(
                    {
                        "rank": rank,
                        "score": float(score),
                        "meta": meta,
                        "chunk_id": meta.get("chunk_id", ""),
                    }
                )

        # 2. Lexical BM25 Retrieval
        lexical_candidates: List[Dict[str, Any]] = []
        if self.bm25_index is not None:
            bm25_results = self.bm25_index.search(query, top_k=candidate_pool_k)
            for rank, (doc_idx, score) in enumerate(bm25_results, start=1):
                meta = (
                    self.bm25_index.doc_metadata[doc_idx]
                    if doc_idx < len(self.bm25_index.doc_metadata)
                    else {}
                )
                lexical_candidates.append(
                    {
                        "rank": rank,
                        "score": float(score),
                        "meta": meta,
                        "chunk_id": meta.get("chunk_id", ""),
                    }
                )

        # 3. Fuse Rankings / Scores
        if fusion_method == "rrf":
            fused_results = self._reciprocal_rank_fusion(
                dense_candidates, lexical_candidates, top_k=k
            )
        else:
            fused_results = self._weighted_score_fusion(
                dense_candidates, lexical_candidates, top_k=k
            )

        return fused_results

    def _reciprocal_rank_fusion(
        self,
        dense_candidates: List[Dict[str, Any]],
        lexical_candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Merge candidate lists using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_meta_map: Dict[str, Dict[str, Any]] = {}

        for cand in dense_candidates:
            cid = cand["chunk_id"]
            if not cid:
                continue
            rank = cand["rank"]
            rrf_scores[cid] += self.dense_weight / (self.rrf_c + rank)
            if cid not in chunk_meta_map:
                chunk_meta_map[cid] = cand["meta"]

        for cand in lexical_candidates:
            cid = cand["chunk_id"]
            if not cid:
                continue
            rank = cand["rank"]
            rrf_scores[cid] += self.lexical_weight / (self.rrf_c + rank)
            if cid not in chunk_meta_map:
                chunk_meta_map[cid] = cand["meta"]

        ranked_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, (cid, score) in enumerate(ranked_items, start=1):
            meta = chunk_meta_map.get(cid, {})
            results.append(
                {
                    "rank": rank,
                    "chunk_id": cid,
                    "score": round(float(score), 6),
                    "document_title": meta.get("document_title", ""),
                    "text": meta.get("text", ""),
                    "sentence_indices": meta.get("sentence_indices", []),
                    "is_supporting_fact": meta.get("is_supporting_fact", False),
                    "provenance": {
                        "source_dataset": meta.get("source_dataset", ""),
                        "source_question_id": meta.get("source_question_id", ""),
                        "fusion_method": "rrf",
                        "metadata": meta.get("metadata", {}),
                    },
                }
            )

        return results

    def _weighted_score_fusion(
        self,
        dense_candidates: List[Dict[str, Any]],
        lexical_candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Merge candidate lists using min-max normalized weighted scores."""
        dense_scores = [c["score"] for c in dense_candidates]
        lex_scores = [c["score"] for c in lexical_candidates]

        min_dense, max_dense = (min(dense_scores), max(dense_scores)) if dense_scores else (0.0, 1.0)
        min_lex, max_lex = (min(lex_scores), max(lex_scores)) if lex_scores else (0.0, 1.0)

        range_dense = max_dense - min_dense if max_dense > min_dense else 1.0
        range_lex = max_lex - min_lex if max_lex > min_lex else 1.0

        fused_scores: Dict[str, float] = defaultdict(float)
        chunk_meta_map: Dict[str, Dict[str, Any]] = {}

        for cand in dense_candidates:
            cid = cand["chunk_id"]
            if not cid:
                continue
            norm_score = (cand["score"] - min_dense) / range_dense
            fused_scores[cid] += self.dense_weight * norm_score
            if cid not in chunk_meta_map:
                chunk_meta_map[cid] = cand["meta"]

        for cand in lexical_candidates:
            cid = cand["chunk_id"]
            if not cid:
                continue
            norm_score = (cand["score"] - min_lex) / range_lex
            fused_scores[cid] += self.lexical_weight * norm_score
            if cid not in chunk_meta_map:
                chunk_meta_map[cid] = cand["meta"]

        ranked_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, (cid, score) in enumerate(ranked_items, start=1):
            meta = chunk_meta_map.get(cid, {})
            results.append(
                {
                    "rank": rank,
                    "chunk_id": cid,
                    "score": round(float(score), 6),
                    "document_title": meta.get("document_title", ""),
                    "text": meta.get("text", ""),
                    "sentence_indices": meta.get("sentence_indices", []),
                    "is_supporting_fact": meta.get("is_supporting_fact", False),
                    "provenance": {
                        "source_dataset": meta.get("source_dataset", ""),
                        "source_question_id": meta.get("source_question_id", ""),
                        "fusion_method": "weighted_sum",
                        "metadata": meta.get("metadata", {}),
                    },
                }
            )

        return results
