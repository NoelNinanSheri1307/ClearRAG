"""BM25 Inverted Index Lexical Retriever for ClearRAG.

Provides sub-millisecond keyword and entity-aware retrieval over the frozen corpus
with title weighting and persistent disk serialization.
"""

from collections import defaultdict
import json
import logging
import math
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def default_tokenizer(text: str) -> List[str]:
    """Extract lowercase alphanumeric word tokens."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


class BM25Index:
    """Inverted Index BM25 lexical retriever with document title weighting."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        title_weight: float = 2.5,
    ):
        """Initialize BM25 Index.

        Args:
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            title_weight: Multiplier for tokens appearing in document titles.
        """
        self.k1 = k1
        self.b = b
        self.title_weight = title_weight

        self.num_docs: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: List[float] = []
        self.doc_metadata: List[Dict[str, Any]] = []

        # Inverted index: term -> list of (doc_index, weighted_tf)
        self.inverted_index: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        # Term IDF cache: term -> idf
        self.idf_cache: Dict[str, float] = {}

    def build_from_metadata(
        self,
        metadata_records: List[Dict[str, Any]],
        show_progress: bool = False,
    ) -> None:
        """Build BM25 index from a list of corpus chunk metadata dictionaries.

        Args:
            metadata_records: List of chunk metadata dicts (must have 'text' and 'document_title').
            show_progress: Whether to log build progress.
        """
        self.num_docs = len(metadata_records)
        self.doc_metadata = metadata_records
        self.doc_lengths = []
        self.inverted_index.clear()
        self.idf_cache.clear()

        total_length: float = 0.0
        doc_term_freqs: List[Dict[str, float]] = []

        for doc_idx, doc in enumerate(metadata_records):
            title = doc.get("document_title", "")
            text = doc.get("text", "")

            title_tokens = default_tokenizer(title)
            text_tokens = default_tokenizer(text)

            tf_map: Dict[str, float] = defaultdict(float)
            for t in text_tokens:
                tf_map[t] += 1.0
            for t in title_tokens:
                tf_map[t] += self.title_weight

            effective_len = len(text_tokens) + (len(title_tokens) * self.title_weight)
            self.doc_lengths.append(effective_len)
            total_length += effective_len
            doc_term_freqs.append(tf_map)

        self.avg_doc_len = total_length / self.num_docs if self.num_docs > 0 else 1.0

        # Build inverted index and compute document frequencies
        doc_freqs: Dict[str, int] = defaultdict(int)
        for doc_idx, tf_map in enumerate(doc_term_freqs):
            for term, weighted_tf in tf_map.items():
                self.inverted_index[term].append((doc_idx, weighted_tf))
                doc_freqs[term] += 1

        # Compute IDFs with standard BM25 smoothed formulation
        for term, df in doc_freqs.items():
            self.idf_cache[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

        logger.info(
            "Built BM25 index over %d documents (vocab size: %d, avg len: %.1f)",
            self.num_docs,
            len(self.inverted_index),
            self.avg_doc_len,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Search BM25 index for top-K matching documents.

        Args:
            query: Query text string.
            top_k: Number of highest scoring documents to return.

        Returns:
            List of (doc_index, score) tuples sorted by descending score.
        """
        if self.num_docs == 0:
            return []

        query_tokens = default_tokenizer(query)
        if not query_tokens:
            return []

        doc_scores: Dict[int, float] = defaultdict(float)

        for term in set(query_tokens):
            idf = self.idf_cache.get(term)
            if idf is None or term not in self.inverted_index:
                continue

            postings = self.inverted_index[term]
            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf * (tf * (self.k1 + 1.0)) / denom
                doc_scores[doc_idx] += term_score

        if not doc_scores:
            return []

        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked_docs

    def save(self, file_path: Path) -> None:
        """Save BM25 index to a binary pickle file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "title_weight": self.title_weight,
            "num_docs": self.num_docs,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "doc_metadata": self.doc_metadata,
            "inverted_index": dict(self.inverted_index),
            "idf_cache": self.idf_cache,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved BM25 index to %s", file_path)

    @classmethod
    def load(cls, file_path: Path) -> "BM25Index":
        """Load BM25 index from a binary pickle file."""
        if not file_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
            title_weight=data.get("title_weight", 2.5),
        )
        instance.num_docs = data["num_docs"]
        instance.avg_doc_len = data["avg_doc_len"]
        instance.doc_lengths = data["doc_lengths"]
        instance.doc_metadata = data["doc_metadata"]
        instance.inverted_index = defaultdict(list, data["inverted_index"])
        instance.idf_cache = data["idf_cache"]
        logger.info("Loaded BM25 index from %s (%d documents)", file_path, instance.num_docs)
        return instance
