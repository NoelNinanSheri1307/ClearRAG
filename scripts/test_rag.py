"""Test Standard RAG Baseline interactively or with sample queries.

Usage:
    python scripts/test_rag.py
    python scripts/test_rag.py --query "Which genus has more species, Bactris or Epigaea?"
"""

import argparse
import logging
from pathlib import Path
import sys
import yaml

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generation.llm_generator import LLMGenerator
from src.generation.prompt_builder import PromptBuilder
from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_rag")


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Test ClearRAG Standard RAG Baseline.")
    parser.add_argument("--config", type=str, default="configs/rag_config.yaml", help="Path to config")
    parser.add_argument("--query", type=str, default=None, help="Question to answer (runs interactive if None)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-K evidence chunks to retrieve")
    parser.add_argument("--interactive", action="store_true", help="Force interactive mode")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    paths_cfg = cfg.get("paths", {})
    llm_cfg = cfg.get("llm", {})
    ret_cfg = cfg.get("retrieval", {})
    prompt_cfg = cfg.get("prompt", {})

    index_path = REPO_ROOT / paths_cfg.get("index_path", "data/processed/faiss_index.bin")
    metadata_path = REPO_ROOT / paths_cfg.get("metadata_path", "data/processed/index_metadata.json")
    top_k = args.top_k or ret_cfg.get("top_k", 5)

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"FAISS index or metadata not found at {index_path}. "
            "Please run scripts/build_index.py first."
        )

    print("=" * 65)
    print("ClearRAG — Standard RAG Baseline Test")
    print("=" * 65)
    print(f"Index Path   : {index_path}")
    print(f"Metadata Path: {metadata_path}")
    print(f"LLM Model    : {llm_cfg.get('model_name', 'Qwen/Qwen2.5-1.5B-Instruct')}")
    print(f"Top-K        : {top_k}")
    print("=" * 65)

    # 1. Initialize retriever
    embedder = BGEEmbedder(model_name=ret_cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"))
    retriever = Retriever.from_saved_index(
        index_path=index_path,
        metadata_path=metadata_path,
        embedder=embedder,
        default_top_k=top_k,
    )

    # 2. Initialize LLM generator
    generator = LLMGenerator(
        model_name=llm_cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
        device=llm_cfg.get("device", "auto"),
        torch_dtype=llm_cfg.get("torch_dtype", "float16"),
        default_max_new_tokens=llm_cfg.get("max_new_tokens", 128),
        default_temperature=llm_cfg.get("temperature", 0.0),
        default_do_sample=llm_cfg.get("do_sample", False),
    )

    # 3. Assemble RAG pipeline
    prompt_builder = PromptBuilder(system_prompt=prompt_cfg.get("system_prompt"))
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        prompt_builder=prompt_builder,
        default_top_k=top_k,
    )

    def run_query(q: str):
        print("\n" + "=" * 65)
        print(f"Question: {q}")
        print("-" * 65)

        result = rag_pipeline.answer(q, top_k=top_k)

        print("Retrieved Evidence:")
        for chunk in result.retrieved_context:
            title = chunk.get("title", "Unknown")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")
            rank = chunk.get("rank", 0)
            print(f"  [{rank}] {title} (Score: {score:.4f}):\n      {text}")

        print("\nGenerated Answer:")
        print(f"  {result.answer}")
        print("-" * 65)
        print(
            f"Latency: Retrieval={result.latency_retrieval_ms:.1f}ms | "
            f"Generation={result.latency_generation_ms:.1f}ms | "
            f"Total={result.latency_total_ms:.1f}ms"
        )
        print("=" * 65)

    if args.query:
        run_query(args.query)
    elif args.interactive:
        print("\nEntering interactive mode. Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_q = input("\nQuestion: ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit", "q"):
                    break
                run_query(user_q)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nSession ended.")
    else:
        sample_queries = [
            "Which genus has more species, Bactris or Epigaea?",
            "Who was born first out of Thomas Carr and Joyce Wieland?",
            "Are The Datsuns and The Black Crowes both rock bands?",
        ]
        print("\nRunning sample benchmark test queries:")
        for q in sample_queries:
            run_query(q)


if __name__ == "__main__":
    main()
