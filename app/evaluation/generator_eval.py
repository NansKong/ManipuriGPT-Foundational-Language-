"""
GeneratorEvaluator module (`app/evaluation/generator_eval.py`).
Evaluates text generation quality across multi-sampling strategies (Greedy, Top-k, Top-p, Temperature)
and computes quantitative diversity metrics (Distinct-1/2/3, Self-BLEU, Repeated n-grams).
"""

import math
import torch
from typing import Dict, Any, List, Optional
from app.utils.logger import logger


class GeneratorEvaluator:
    """Multi-decoding strategy and text diversity evaluation engine."""

    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.1
    ) -> str:
        """Generates text from prompt using specified decoding parameters."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                if top_k is not None:
                    gen_kwargs["top_k"] = top_k
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

            generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return generated
        except Exception as e:
            logger.error(f"GeneratorEvaluator: Generation failed for prompt '{prompt}': {e}")
            return prompt

    def compute_distinct_n(self, texts: List[str], n: int) -> float:
        """Computes Distinct-N metric (ratio of unique n-grams to total n-grams)."""
        ngrams = []
        for text in texts:
            words = text.split() if text else []
            for i in range(len(words) - n + 1):
                ngrams.append(tuple(words[i:i+n]))

        if not ngrams:
            return 0.0
        return round(len(set(ngrams)) / len(ngrams), 4)

    def compute_self_bleu(self, texts: List[str]) -> float:
        """Computes Self-BLEU pairwise similarity score (lower indicates higher diversity)."""
        if len(texts) < 2:
            return 0.0

        try:
            from app.evaluation.metrics import MetricRegistry
            metrics = MetricRegistry()
            bleu_scores = []
            for i, hyp in enumerate(texts):
                refs = [ref for j, ref in enumerate(texts) if j != i]
                score = metrics.calculate("bleu", [hyp] * len(refs), refs)
                bleu_scores.append(score)

            return round(sum(bleu_scores) / len(bleu_scores), 2)
        except Exception:
            return 0.0

    def evaluate_decoding_strategies(self, prompts: List[str]) -> Dict[str, Any]:
        """Compares generation outputs and diversity across Greedy, Top-k, Top-p, and Temperatures."""
        strategies = {
            "greedy": {"do_sample": False},
            "top_k_50": {"do_sample": True, "temperature": 0.7, "top_k": 50},
            "top_p_0.9": {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
            "temp_0.2": {"do_sample": True, "temperature": 0.2},
            "temp_0.7": {"do_sample": True, "temperature": 0.7},
            "temp_1.0": {"do_sample": True, "temperature": 1.0}
        }

        results = {}

        for strat_name, kwargs in strategies.items():
            generated_texts = []
            sample_outputs = []
            for prompt in prompts:
                text = self.generate_text(prompt, **kwargs)
                generated_texts.append(text)
                sample_outputs.append({"prompt": prompt, "generated": text})

            d1 = self.compute_distinct_n(generated_texts, 1)
            d2 = self.compute_distinct_n(generated_texts, 2)
            d3 = self.compute_distinct_n(generated_texts, 3)
            self_bleu = self.compute_self_bleu(generated_texts)

            results[strat_name] = {
                "distinct_1": d1,
                "distinct_2": d2,
                "distinct_3": d3,
                "self_bleu": self_bleu,
                "samples": sample_outputs
            }

        return results
