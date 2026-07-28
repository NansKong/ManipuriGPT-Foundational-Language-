"""
SpeedBenchmark module (`app/evaluation/speed_benchmark.py`).
Measures inference speed, throughput (tokens/sec), prompt latency, peak VRAM memory usage, and load time.
"""

import time
import torch
from typing import Dict, Any, List, Callable, Optional
from app.utils.logger import logger


class SpeedBenchmark:
    """Hardware and inference throughput profiler."""

    def __init__(self, model_loader_fn: Optional[Callable] = None):
        self.model_loader_fn = model_loader_fn

    def measure_model_load_time(self, load_fn: Optional[Callable] = None) -> float:
        """Measures exact wall-clock time required to load the model into memory."""
        fn = load_fn or self.model_loader_fn
        if not fn:
            return 0.0

        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        return round(elapsed, 4)

    def benchmark_inference(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        max_new_tokens: int = 64,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Profiles generation throughput (tokens/sec), average latency, and GPU VRAM footprint."""
        model.eval()
        
        latencies = []
        total_tokens_generated = 0
        peak_vram_gb = 0.0

        if torch.cuda.is_available() and "cuda" in str(device).lower():
            torch.cuda.reset_peak_memory_stats()

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            start_time = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            gen_tokens = output_ids.size(1) - inputs["input_ids"].size(1)
            total_tokens_generated += gen_tokens
            latencies.append(latency_ms)

        avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
        total_time_sec = sum(latencies) / 1000.0
        tokens_per_sec = (total_tokens_generated / total_time_sec) if total_time_sec > 0 else 0.0

        if torch.cuda.is_available() and "cuda" in str(device).lower():
            peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)

        return {
            "tokens_per_sec": round(tokens_per_sec, 2),
            "average_latency_ms": round(avg_latency_ms, 2),
            "total_tokens_generated": total_tokens_generated,
            "peak_vram_gb": peak_vram_gb,
            "device": str(device)
        }
