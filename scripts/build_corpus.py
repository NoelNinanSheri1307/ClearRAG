"""Build structured chunk corpus from raw HotpotQA dataset.

Usage:
    python scripts/build_corpus.py [--max_questions 500] [--output_path data/corpus/corpus_chunks.json]
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

import yaml

from src.ingestion.corpus_builder import CorpusBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("build_corpus")


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Build ClearRAG Corpus from HotpotQA.")
    parser.add_argument("--config", type=str, default="configs/retrieval_config.yaml", help="Path to config file")
    parser.add_argument("--raw_path", type=str, default=None, help="Path to raw HotpotQA JSON")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for corpus JSON")
    parser.add_argument("--max_questions", type=int, default=None, help="Limit number of questions to process")
    parser.add_argument("--no_dedup", action="store_true", help="Disable cross-question document deduplication")
    args = parser.parse_args()

    cfg = load_config(Path(args.config)).get("corpus", {})

    raw_path = Path(args.raw_path or cfg.get("raw_data_path", "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"))
    output_path = Path(args.output_path or cfg.get("corpus_output_path", "data/corpus/corpus_chunks.json"))
    max_questions = args.max_questions if args.max_questions is not None else cfg.get("max_questions", None)
    deduplicate = not args.no_dedup if args.no_dedup else cfg.get("deduplicate", True)

    print("=" * 65)
    print("ClearRAG — Corpus Builder")
    print("=" * 65)
    print(f"Source data      : {raw_path}")
    print(f"Output path      : {output_path}")
    print(f"Max questions    : {max_questions if max_questions else 'ALL (7,405)'}")
    print(f"Deduplicate      : {deduplicate}")

    if not raw_path.exists():
        raise FileNotFoundError(f"Source HotpotQA data file not found at {raw_path}")

    start_time = time.perf_counter()
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    builder = CorpusBuilder(deduplicate_by_title_and_idx=deduplicate)
    chunks = builder.build_from_hotpotqa(raw_data, max_questions=max_questions)
    builder.save_corpus(chunks, output_path)

    elapsed = time.perf_counter() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("\nCorpus Build Summary:")
    print("-" * 65)
    print(f"Questions processed: {len(raw_data[:max_questions]) if max_questions else len(raw_data):,}")
    print(f"Total chunks created: {len(chunks):,}")
    print(f"Output file size   : {file_size_mb:.2f} MB")
    print(f"Processing time    : {elapsed:.2f} seconds")
    print("=" * 65)


if __name__ == "__main__":
    main()
