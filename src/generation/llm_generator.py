"""LLM Generator module for ClearRAG.

Loads instruction-tuned models (default: Qwen/Qwen2.5-1.5B-Instruct) using Hugging Face
Transformers and executes deterministic text generation on GPU in FP16.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LLMGenerator:
    """Encapsulates local LLM loading and deterministic inference."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        default_max_new_tokens: int = 384,
        default_temperature: float = 0.0,
        default_do_sample: bool = False,
    ):
        """Initialize the LLMGenerator.

        Args:
            model_name: HuggingFace model identifier.
            device: 'cuda', 'cpu', or None for auto-detection.
            torch_dtype: 'float16', 'float32', torch.float16, or None for auto.
            default_max_new_tokens: Maximum tokens to generate per response.
            default_temperature: Sampling temperature (0.0 for greedy).
            default_do_sample: Whether to sample tokens (default: False for deterministic).
        """
        self.model_name = model_name
        self.default_max_new_tokens = default_max_new_tokens
        self.default_temperature = default_temperature
        self.default_do_sample = default_do_sample

        # 1. Device resolution
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 2. Dtype resolution
        if torch_dtype is None or torch_dtype == "auto":
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        elif isinstance(torch_dtype, str):
            self.dtype = torch.float16 if torch_dtype == "float16" else torch.float32
        else:
            self.dtype = torch_dtype

        logger.info(
            f"Initializing LLMGenerator with '{self.model_name}' on device '{self.device}' (dtype={self.dtype})"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Load Tokenizer & Model
        start_load = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )
        if self.device == "cuda":
            self.model.to("cuda")
        elif self.device == "cpu":
            self.model.to("cpu")

        self.model.eval()
        self.load_time_seconds = time.perf_counter() - start_load
        logger.info(f"Loaded LLM '{self.model_name}' in {self.load_time_seconds:.2f}s")

    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> Tuple[str, float]:
        """Generate response from structured chat messages.

        Args:
            messages: List of message dicts (role, content).
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample.

        Returns:
            Tuple of (generated_answer_text, latency_ms).
        """
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.generate(
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> Tuple[str, float]:
        """Generate text from a raw formatted prompt string.

        Args:
            prompt: Formatted prompt string.
            max_new_tokens: Max new tokens.
            temperature: Generation temperature.
            do_sample: Sampling boolean flag.

        Returns:
            Tuple of (generated_text, latency_ms).
        """
        max_tokens = max_new_tokens or self.default_max_new_tokens
        temp = temperature if temperature is not None else self.default_temperature
        sample = do_sample if do_sample is not None else self.default_do_sample

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        input_token_len = model_inputs.input_ids.shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": sample,
        }
        if sample and temp > 0.0:
            gen_kwargs["temperature"] = temp

        start_gen = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs,
            )
        latency_ms = (time.perf_counter() - start_gen) * 1000.0

        # Slice off the prompt tokens to get only the model's new answer
        response_tokens = generated_ids[0][input_token_len:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        return response_text, latency_ms
