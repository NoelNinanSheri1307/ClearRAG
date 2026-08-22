import time
import torch
from sentence_transformers import SentenceTransformer


def test_embedding():
    model_name = "BAAI/bge-small-en-v1.5"
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{model_name}' on device: {device}...")
    
    # Load model
    model = SentenceTransformer(model_name, device=device)
    
    test_sentences = [
        "Retrieval augmented generation combines language models with external knowledge.",
        "A retrieval system searches a knowledge base for relevant documents.",
        "The capital of France is Paris.",
        "Machine learning models learn patterns from data.",
    ]
    
    # Encode and measure time
    start_time = time.perf_counter()
    embeddings = model.encode(test_sentences, convert_to_numpy=True)
    elapsed_time = time.perf_counter() - start_time
    
    # Print results
    print("\n" + "=" * 50)
    print("Embedding Benchmark Results")
    print("=" * 50)
    print(f"Model Name         : {model_name}")
    print(f"Device             : {device}")
    if device == "cuda":
        print(f"GPU Name           : {torch.cuda.get_device_name(0)}")
    print(f"Embedding Dimension: {embeddings.shape[1]}")
    print(f"Number of Sentences: {len(test_sentences)}")
    print(f"Embedding Shape    : {embeddings.shape}")
    print(f"Encoding Time      : {elapsed_time:.4f} seconds ({elapsed_time * 1000:.2f} ms)")
    
    # CUDA memory metrics if available
    if device == "cuda":
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"GPU Memory Alloc   : {allocated_mb:.2f} MB")
        print(f"GPU Memory Reserved: {reserved_mb:.2f} MB")
    print("=" * 50)


if __name__ == "__main__":
    test_embedding()
