"""FAISS Index management for ClearRAG vector retrieval.

Wraps faiss.IndexFlatIP for exact cosine similarity search over normalized
embeddings, with persistent disk serialization of indices and chunk metadata mappings.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS exact inner product (cosine similarity) index."""

    def __init__(self, dimension: int = 384):
        """Initialize FAISSIndex.

        Args:
            dimension: Dimensionality of the embedding vectors (default: 384 for bge-small).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store: List[Dict[str, Any]] = []

    @property
    def ntotal(self) -> int:
        """Return total number of vectors in the index."""
        return self.index.ntotal

    def add(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """Add embedding vectors and their corresponding chunk metadata to the index.

        Args:
            embeddings: 2D numpy array of shape (N, dimension) and float32 dtype.
            metadata_list: List of N metadata dictionaries corresponding to the vectors.
        """
        if embeddings.shape[0] != len(metadata_list):
            raise ValueError(
                f"Count mismatch: received {embeddings.shape[0]} embeddings "
                f"and {len(metadata_list)} metadata records."
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        # Ensure float32 and contiguous
        embeddings_c = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings_c)
        self.metadata_store.extend(metadata_list)
        logger.info(
            f"Added {embeddings.shape[0]:,} vectors to FAISS index. Total: {self.ntotal:,}"
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Search the index for top_k most similar vectors.

        Args:
            query_vector: 1D or 2D numpy array (1, dimension) or (dimension,).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple of (scores_array, list_of_metadata_dicts).
        """
        if self.ntotal == 0:
            return np.empty((0,), dtype=np.float32), []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_c = np.ascontiguousarray(query_vector, dtype=np.float32)
        k = min(top_k, self.ntotal)

        scores, indices = self.index.search(query_c, k)

        retrieved_scores = scores[0]
        retrieved_indices = indices[0]

        retrieved_meta = [
            self.metadata_store[idx]
            for idx in retrieved_indices
            if idx != -1 and idx < len(self.metadata_store)
        ]

        return retrieved_scores, retrieved_meta

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Save FAISS binary index and JSON metadata store to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved FAISS index ({self.ntotal:,} vectors) to {index_path}")
        logger.info(f"Saved metadata ({len(self.metadata_store):,} entries) to {metadata_path}")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "FAISSIndex":
        """Load FAISS binary index and metadata from disk."""
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        raw_index = faiss.read_index(str(index_path))
        dimension = raw_index.d

        instance = cls(dimension=dimension)
        instance.index = raw_index

        with open(metadata_path, "r", encoding="utf-8") as f:
            instance.metadata_store = json.load(f)

        if instance.index.ntotal != len(instance.metadata_store):
            logger.warning(
                f"Index count ({instance.index.ntotal}) != metadata count ({len(instance.metadata_store)})"
            )

        logger.info(f"Loaded FAISS index with {instance.ntotal:,} vectors from {index_path}")
        return instance
