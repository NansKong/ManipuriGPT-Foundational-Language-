"""
PerplexityEvaluator module (`app/evaluation/perplexity_eval.py`).
Computes overall and script-wise perplexity on held-out text splits across Meitei Mayek, Bengali, and Mixed scripts.
"""

import math
import re
import torch
from typing import Dict, Any, List, Optional
from app.utils.logger import logger


class PerplexityEvaluator:
    """Evaluates cross-entropy loss and perplexity (PPL = exp(loss)) broken down by script."""

    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        self.meitei_pat = re.compile(r'[\uABC0-\uABFF\u1C80-\u1C8F]')
        self.bengali_pat = re.compile(r'[\u0980-\u09FF]')

    def classify_script(self, text: str) -> str:
        """Classifies text into Meitei Mayek, Bengali script, or Mixed script."""
        has_meitei = bool(self.meitei_pat.search(text))
        has_bengali = bool(self.bengali_pat.search(text))
        if has_meitei and has_bengali:
            return "mixed"
        elif has_meitei:
            return "meitei"
        elif has_bengali:
            return "bengali"
        return "other"

    def compute_sequence_loss(self, text: str) -> Optional[float]:
        """Computes average per-token cross-entropy loss for a single text sequence."""
        if not text or not text.strip():
            return None

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if inputs["input_ids"].size(1) <= 1:
                return None

            with torch.no_grad():
                labels = inputs["input_ids"].clone()
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss.item()
                return loss
        except Exception as e:
            logger.debug(f"PerplexityEvaluator: Loss computation failed for sequence: {e}")
            return None

    def evaluate_texts(self, texts: List[str]) -> Dict[str, Any]:
        """Computes overall and script-wise perplexity across a corpus of texts."""
        overall_losses = []
        script_losses = {"meitei": [], "bengali": [], "mixed": [], "other": []}

        for text in texts:
            loss = self.compute_sequence_loss(text)
            if loss is not None and not math.isnan(loss) and not math.isinf(loss):
                overall_losses.append(loss)
                script = self.classify_script(text)
                script_losses[script].append(loss)

        def calc_ppl(losses: List[float]) -> float:
            if not losses:
                return float("nan")
            avg_loss = sum(losses) / len(losses)
            return round(math.exp(min(avg_loss, 20.0)), 4)

        overall_ppl = calc_ppl(overall_losses)
        meitei_ppl = calc_ppl(script_losses["meitei"])
        bengali_ppl = calc_ppl(script_losses["bengali"])
        mixed_ppl = calc_ppl(script_losses["mixed"])

        # Qualitatively grade overall PPL
        qualitative = "Unknown"
        if not math.isnan(overall_ppl):
            if overall_ppl < 10:
                qualitative = "Excellent (<10)"
            elif overall_ppl <= 20:
                qualitative = "Very Good (10–20)"
            elif overall_ppl <= 50:
                qualitative = "Good (20–50)"
            elif overall_ppl <= 100:
                qualitative = "Understandable (50–100)"
            else:
                qualitative = "Poor (>100)"

        return {
            "overall_ppl": overall_ppl,
            "overall_loss": round(sum(overall_losses)/len(overall_losses), 4) if overall_losses else float("nan"),
            "qualitative_meaning": qualitative,
            "samples_evaluated": len(overall_losses),
            "script_wise": {
                "meitei_mayek_ppl": meitei_ppl,
                "meitei_samples": len(script_losses["meitei"]),
                "bengali_script_ppl": bengali_ppl,
                "bengali_samples": len(script_losses["bengali"]),
                "mixed_script_ppl": mixed_ppl,
                "mixed_samples": len(script_losses["mixed"])
            }
        }
