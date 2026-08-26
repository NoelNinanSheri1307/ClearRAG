"""Test retrieval with queries on the ClearRAG FAISS index.

Usage:
    python scripts/test_retrieval.py [--query "Which genus has more species, Bactris or Epigaea?"] [--top_k 5]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.retrieval.embedder import BGEEmbedder
from src.retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


SAMPLE_QUERIES = [
    "Which genus has more species, Bactris or Epigaea?",
    "Who was born first out of Thomas Carr and Joyce Wieland?",
    "Are The Datsuns and The Black Crowes both rock bands?",
]


def test_single_query(retriever: Retriever, query: str, top_k: int):
    print(f"\nQuery: '{query}'")
    print("-" * 65)

    start = time.perf_counter()
    results = retriever.retrieve(query, top_k=top_k)
    elapsed = time.perf_counter() - start

    print(f"Retrieved {len(results)} chunks in {elapsed * 1000:.2f} ms:\n")
    for res in results:
        rank = res["rank"]
        score = res["score"]
        title = res["document_title"]
        sent_indices = res["sentence_indices"]
        text = res["text"]
        chunk_id = res["chunk_id"]

        print(f"[{rank}] Score: {score:.4f} | Title: '{title}' (Sentence: {sent_indices})")
        print(f"    Chunk ID : {chunk_id}")
        print(f"    Text     : {text[:160]}{'...' if len(text) > 160 else ''}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Test ClearRAG Retriever.")
    parser.add_argument("--config", type=str, default="configs/retrieval_config.yaml")
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--query", type=str, default=None, help="Specific question to search")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to retrieve")
    args = parser.parse_args()

    full_cfg = load_config(Path(args.config))
    idx_cfg = full_cfg.get("indexing", {})
    emb_cfg = full_cfg.get("embedding", {})

    index_path = Path(args.index_path or idx_cfg.get("index_output_path", "data/processed/faiss_index.bin"))
    metadata_path = Path(args.metadata_path or idx_cfg.get("metadata_output_path", "data/processed/index_metadata.json"))
    top_k = args.top_k or full_cfg.get("retrieval", {}).get("default_top_k", 10)

    print("=" * 65)
    print("ClearRAG — Test Retrieval")
    print("=" * 65)
    print(f"Index Path   : {index_path}")
    print(f"Metadata Path: {metadata_path}")
    print(f"Top-K        : {top_k}")

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"FAISS index or metadata missing. Please run scripts/build_index.py first."
        )

    embedder = BGEEmbedder(
        model_name=emb_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
        device=None if emb_cfg.get("device", "auto") == "auto" else emb_cfg.get("device"),
    )

    retriever = Retriever.from_saved_index(
        index_path=index_path,
        metadata_path=metadata_path,
        embedder=embedder,
        default_top_k=top_k,
    )

    if args.query:
        test_single_query(retriever, args.query, top_k)
    else:
        print("\nExecuting sample benchmark test queries:")
        for q in SAMPLE_QUERIES:
            test_single_query(retriever, q, top_k)

    print("=" * 65)


if __name__ == "__main__":
    main()
