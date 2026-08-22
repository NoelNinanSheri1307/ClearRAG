import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_llm():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    prompt_text = "Explain in two sentences what Retrieval-Augmented Generation is."

    # Device & dtype detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model '{model_name}' on {device} (dtype: {torch_dtype})...")

    # Measure model and tokenizer load time
    start_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to("cpu")
    load_time = time.perf_counter() - start_load

    # Prepare chat formatted input
    messages = [
        {"role": "user", "content": prompt_text}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)

    # Generate response
    start_gen = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_time = time.perf_counter() - start_gen

    # Decode only the generated response tokens
    input_len = model_inputs.input_ids.shape[1]
    response = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()

    # Print benchmark report
    print("\n" + "=" * 55)
    print("LLM Benchmark Results")
    print("=" * 55)
    print(f"Model Name          : {model_name}")
    print(f"Device              : {device}")
    if device == "cuda":
        print(f"GPU Name            : {torch.cuda.get_device_name(0)}")
    print(f"Model Loading Time  : {load_time:.2f} s")
    print(f"Generation Time     : {gen_time:.2f} s")
    print("-" * 55)
    print("Generated Response:")
    print(response)
    print("-" * 55)

    if device == "cuda":
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"GPU Memory Allocated: {allocated_mb:.2f} MB")
        print(f"GPU Memory Reserved : {reserved_mb:.2f} MB")
    print("=" * 55)


if __name__ == "__main__":
    test_llm()
