"""Evaluate Standard RAG Baseline on the ClearRAG 1,250-instance Benchmark.

Evaluates generation and retrieval performance across 5 controlled conditions:
- full_evidence
- partial_evidence
- unsupported
- distractor_heavy
- conflict

Features:
- Checkpointing and automatic resumption
- GPU memory conscious batching
- Exact Match, Token F1, and Substring Contains-GT computation
- Detailed condition breakdown

Usage:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --max_instances 5   # Smoke test
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
import time
from tqdm import tqdm
import yaml

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.generation_metrics import (
    aggregate_generation_metrics,
    compute_generation_metrics,
)
from src.generation.llm_generator import LLMGenerator
from src.generation.prompt_builder import PromptBuilder
from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_rag")


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate ClearRAG Standard RAG Baseline.")
    parser.add_argument("--config", type=str, default="configs/rag_config.yaml", help="Path to config")
    parser.add_argument("--benchmark_path", type=str, default=None, help="Path to benchmark JSON")
    parser.add_argument("--results_path", type=str, default=None, help="Path to output results JSON")
    parser.add_argument("--top_k", type=int, default=None, help="Retrieval Top-K")
    parser.add_argument("--max_instances", type=int, default=None, help="Cap instances for quick smoke testing")
    parser.add_argument("--no_resume", action="store_true", help="Overwrite existing results without resuming")
    parser.add_argument("--checkpoint_every", type=int, default=25, help="Save frequency during evaluation")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    paths_cfg = cfg.get("paths", {})
    llm_cfg = cfg.get("llm", {})
    ret_cfg = cfg.get("retrieval", {})
    prompt_cfg = cfg.get("prompt", {})
    eval_cfg = cfg.get("evaluation", {})

    index_path = REPO_ROOT / paths_cfg.get("index_path", "data/processed/faiss_index.bin")
    metadata_path = REPO_ROOT / paths_cfg.get("metadata_path", "data/processed/index_metadata.json")
    benchmark_path = REPO_ROOT / (args.benchmark_path or paths_cfg.get("benchmark_path", "data/evaluation/clearrag_eval.json"))
    results_path = REPO_ROOT / (args.results_path or paths_cfg.get("results_path", "results/standard_rag_evaluation.json"))
    top_k = args.top_k or ret_cfg.get("top_k", 5)
    checkpoint_every = args.checkpoint_every or eval_cfg.get("checkpoint_every", 25)

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found at {benchmark_path}")
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"FAISS index or metadata not found at {index_path}")

    # 1. Load benchmark instances
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    if args.max_instances and args.max_instances > 0:
        benchmark_data = benchmark_data[: args.max_instances]

    print("=" * 65)
    print("ClearRAG — Standard RAG Baseline Evaluation")
    print("=" * 65)
    print(f"Benchmark Path   : {benchmark_path}")
    print(f"Total Instances  : {len(benchmark_data):,}")
    print(f"Index Path       : {index_path}")
    print(f"Results Path     : {results_path}")
    print(f"Top-K Evidence   : {top_k}")
    print(f"Model            : {llm_cfg.get('model_name', 'Qwen/Qwen2.5-1.5B-Instruct')}")
    print("=" * 65)

    # 2. Check for existing progress to resume
    completed_records = []
    completed_ids = set()

    if not args.no_resume and results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                existing_payload = json.load(f)
                completed_records = existing_payload.get("predictions", [])
                completed_ids = {r["id"] for r in completed_records if "id" in r}
                print(f"Resuming from existing results: {len(completed_records)} already evaluated.")
        except Exception as e:
            logger.warning(f"Could not load existing results file to resume ({e}). Starting fresh.")
            completed_records = []
            completed_ids = set()

    # 3. Initialize RAG pipeline
    embedder = BGEEmbedder(model_name=ret_cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"))
    retriever = Retriever.from_saved_index(
        index_path=index_path,
        metadata_path=metadata_path,
        embedder=embedder,
        default_top_k=top_k,
    )
    generator = LLMGenerator(
        model_name=llm_cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
        device=llm_cfg.get("device", "auto"),
        torch_dtype=llm_cfg.get("torch_dtype", "float16"),
        default_max_new_tokens=llm_cfg.get("max_new_tokens", 128),
        default_temperature=llm_cfg.get("temperature", 0.0),
        default_do_sample=llm_cfg.get("do_sample", False),
    )
    prompt_builder = PromptBuilder(system_prompt=prompt_cfg.get("system_prompt"))
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        prompt_builder=prompt_builder,
        default_top_k=top_k,
    )

    def save_checkpoint(records_list: list, is_final: bool = False):
        results_path.parent.mkdir(parents=True, exist_ok=True)
        agg_metrics = aggregate_generation_metrics(records_list)
        payload = {
            "evaluation_metadata": {
                "benchmark_path": str(benchmark_path),
                "total_benchmark_instances": len(benchmark_data),
                "completed_instances": len(records_list),
                "model_name": generator.model_name,
                "embedding_model": embedder.model_name,
                "top_k": top_k,
                "device": generator.device,
                "timestamp": datetime.datetime.now().isoformat(),
                "is_final": is_final,
            },
            "overall_metrics": agg_metrics["overall"],
            "condition_breakdown": agg_metrics["by_condition"],
            "predictions": records_list,
        }
        # Write atomically with retry for Windows file locks
        tmp_path = results_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        for attempt in range(5):
            try:
                tmp_path.replace(results_path)
                break
            except (PermissionError, OSError):
                time.sleep(0.2)
        else:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    # 4. Evaluation Loop
    start_eval_time = time.perf_counter()
    newly_evaluated = 0

    with tqdm(total=len(benchmark_data), initial=len(completed_records), desc="Evaluating Standard RAG") as pbar:
        for idx, instance in enumerate(benchmark_data):
            inst_id = instance.get("id", f"inst_{idx}")
            if inst_id in completed_ids:
                continue

            question = instance.get("question", "")
            ground_truth = instance.get("ground_truth", "")
            condition = instance.get("condition", "unknown")

            # Run RAG
            rag_res = rag_pipeline.answer(question, top_k=top_k)
            prediction = rag_res.answer

            # Compute instance metrics
            metrics = compute_generation_metrics(prediction, ground_truth)

            record = {
                "id": inst_id,
                "condition": condition,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "metrics": metrics,
                "retrieved_context": rag_res.retrieved_context,
                "latency_retrieval_ms": rag_res.latency_retrieval_ms,
                "latency_generation_ms": rag_res.latency_generation_ms,
                "latency_total_ms": rag_res.latency_total_ms,
            }

            completed_records.append(record)
            completed_ids.add(inst_id)
            newly_evaluated += 1
            pbar.update(1)

            # Periodic checkpoint
            if newly_evaluated % checkpoint_every == 0:
                save_checkpoint(completed_records, is_final=False)

    # Final save
    total_duration_sec = time.perf_counter() - start_eval_time
    save_checkpoint(completed_records, is_final=True)

    # 5. Print Summary Results
    final_agg = aggregate_generation_metrics(completed_records)
    print("\n" + "=" * 65)
    print("STANDARD RAG EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total Evaluated Instances: {len(completed_records):,}")
    print(f"Evaluation Duration      : {total_duration_sec:.2f} s")
    if newly_evaluated > 0:
        print(f"Average Speed            : {newly_evaluated / total_duration_sec:.2f} queries/s")
    print("-" * 65)
    print("Overall Generation Metrics:")
    print(f"  Exact Match (EM)       : {final_agg['overall']['exact_match'] * 100:.2f}%")
    print(f"  Token F1 Score         : {final_agg['overall']['token_f1'] * 100:.2f}%")
    print(f"  Contains Ground Truth  : {final_agg['overall']['contains_gt'] * 100:.2f}%")
    print("=" * 65)
    print("\nBreakdown by Benchmark Condition:")
    print("-" * 65)
    print(f"{'Condition':<20} | {'Count':<6} | {'EM (%)':<8} | {'F1 (%)':<8} | {'Contains GT (%)':<15}")
    print("-" * 65)
    for cond, cond_metrics in final_agg["by_condition"].items():
        count = cond_metrics["count"]
        em_pct = cond_metrics["exact_match"] * 100
        f1_pct = cond_metrics["token_f1"] * 100
        cgt_pct = cond_metrics["contains_gt"] * 100
        print(f"{cond:<20} | {count:<6} | {em_pct:<8.2f} | {f1_pct:<8.2f} | {cgt_pct:<15.2f}")
    print("=" * 65)
    print(f"\nResults successfully saved to: {results_path}")


if __name__ == "__main__":
    main()
