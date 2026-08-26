"""Build FAISS vector index using BGE-small-en-v1.5 embeddings.

Usage:
    python scripts/build_index.py [--corpus_path data/corpus/corpus_chunks.json] [--batch_size 256]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from tqdm import tqdm

from src.ingestion.corpus_builder import CorpusBuilder
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.faiss_index import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("build_index")


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Build FAISS Index from ClearRAG Corpus.")
    parser.add_argument("--config", type=str, default="configs/retrieval_config.yaml", help="Path to config")
    parser.add_argument("--corpus_path", type=str, default=None, help="Path to corpus JSON")
    parser.add_argument("--index_path", type=str, default=None, help="Output path for FAISS binary index")
    parser.add_argument("--metadata_path", type=str, default=None, help="Output path for index metadata")
    parser.add_argument("--batch_size", type=int, default=None, help="Embedding batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--max_chunks", type=int, default=None, help="Cap number of chunks to index for debugging")
    args = parser.parse_args()

    full_cfg = load_config(Path(args.config))
    corp_cfg = full_cfg.get("corpus", {})
    emb_cfg = full_cfg.get("embedding", {})
    idx_cfg = full_cfg.get("indexing", {})

    corpus_path = Path(args.corpus_path or corp_cfg.get("corpus_output_path", "data/corpus/corpus_chunks.json"))
    index_path = Path(args.index_path or idx_cfg.get("index_output_path", "data/processed/faiss_index.bin"))
    metadata_path = Path(args.metadata_path or idx_cfg.get("metadata_output_path", "data/processed/index_metadata.json"))
    batch_size = args.batch_size or emb_cfg.get("batch_size", 256)
    model_name = emb_cfg.get("model_name", "BAAI/bge-small-en-v1.5")
    device_arg = args.device or emb_cfg.get("device", "auto")
    device = None if device_arg == "auto" else device_arg

    print("=" * 65)
    print("ClearRAG — FAISS Index Builder")
    print("=" * 65)
    print(f"Corpus Path      : {corpus_path}")
    print(f"Index Output Path: {index_path}")
    print(f"Metadata Path    : {metadata_path}")
    print(f"Model Name       : {model_name}")
    print(f"Batch Size       : {batch_size}")

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_path}. Please run scripts/build_corpus.py first."
        )

    # 1. Load corpus chunks
    start_load = time.perf_counter()
    builder = CorpusBuilder()
    chunks = builder.load_corpus(corpus_path)
    if args.max_chunks and args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    load_time = time.perf_counter() - start_load
    print(f"Loaded {len(chunks):,} chunks in {load_time:.2f}s")

    # 2. Format passage texts with title context for retrieval
    # Format: "{Title}: {Sentence}" ensures subject context is preserved
    passage_texts = [
        f"{chunk.document_title}: {chunk.text}" if chunk.document_title else chunk.text
        for chunk in chunks
    ]
    chunk_metadata = [chunk.to_dict() for chunk in chunks]

    # 3. Initialize embedder
    embedder = BGEEmbedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    print(f"Device Used      : {embedder.device}")
    if embedder.device == "cuda":
        print(f"GPU Name         : {torch.cuda.get_device_name(0)}")

    # 4. Generate embeddings
    print(f"\nGenerating embeddings for {len(passage_texts):,} passages (batch_size={batch_size})...")
    start_embed = time.perf_counter()
    embeddings = embedder.embed_texts(passage_texts, batch_size=batch_size, show_progress_bar=True)
    embed_time = time.perf_counter() - start_embed
    throughput = len(passage_texts) / embed_time if embed_time > 0 else 0
    print(f"Embedded {len(passage_texts):,} vectors in {embed_time:.2f}s ({throughput:.1f} vectors/s)")

    # 5. Build and save FAISS index
    start_index = time.perf_counter()
    faiss_index = FAISSIndex(dimension=embedder.dimension)
    faiss_index.add(embeddings, chunk_metadata)
    faiss_index.save(index_path, metadata_path)
    index_time = time.perf_counter() - start_index

    index_size_mb = index_path.stat().st_size / (1024 * 1024)
    meta_size_mb = metadata_path.stat().st_size / (1024 * 1024)

    print("\nFAISS Index Summary:")
    print("-" * 65)
    print(f"Total Vectors Indexed: {faiss_index.ntotal:,}")
    print(f"Vector Dimension     : {embedder.dimension}")
    print(f"FAISS Index Type     : IndexFlatIP (Exact Cosine Similarity)")
    print(f"Index File Size      : {index_size_mb:.2f} MB")
    print(f"Metadata File Size   : {meta_size_mb:.2f} MB")
    print(f"Total Pipeline Time  : {(load_time + embed_time + index_time):.2f}s")
    if embedder.device == "cuda":
        alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"GPU Memory Allocated : {alloc_mb:.2f} MB")
    print("=" * 65)


if __name__ == "__main__":
    main()
