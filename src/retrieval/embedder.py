"""Embedding module for ClearRAG retrieval.

Wraps SentenceTransformer with BAAI/bge-small-en-v1.5, supporting CUDA detection,
batch encoding, L2 normalization, and query instruction formatting.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """Embedding generator using BAAI/bge-small-en-v1.5."""

    DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DEFAULT_DIMENSION = 384
    DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 256,
        normalize_embeddings: bool = True,
        query_instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
    ):
        """Initialize the BGE Embedder.

        Args:
            model_name: HuggingFace model ID.
            device: 'cuda', 'cpu', or None for automatic detection.
            batch_size: Batch size for batch encoding.
            normalize_embeddings: If True, output vectors are L2-normalized.
            query_instruction: Prefix for query encoding (recommended for BGE retrieval).
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.query_instruction = query_instruction or ""

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension() or self.DEFAULT_DIMENSION
        else:
            self.dimension = self.model.get_sentence_embedding_dimension() or self.DEFAULT_DIMENSION
        logger.info(f"Loaded embedder: dim={self.dimension}, normalize={self.normalize_embeddings}")

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode a list of text strings into normalized float32 embeddings.

        Args:
            texts: List of text passages/sentences.
            batch_size: Override default batch size if provided.
            show_progress_bar: Whether to display tqdm progress bar.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim) with dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        bs = batch_size or self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string with the BGE retrieval instruction.

        Args:
            query: Input query text.

        Returns:
            1D numpy array of shape (embedding_dim,) with dtype float32.
        """
        formatted_query = f"{self.query_instruction}{query}" if self.query_instruction else query
        embedding = self.model.encode(
            formatted_query,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)
