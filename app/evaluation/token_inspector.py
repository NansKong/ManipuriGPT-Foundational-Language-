"""
TokenInspector module (`app/evaluation/token_inspector.py`).
Inspects next-token predictions, top-k probabilities, token confidence, entropy, and grammar/formatting.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from app.utils.logger import logger


class TokenInspector:
    """Analyzes next-token logits and probability distributions for given prompts."""

    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def inspect_prompt(self, prompt: str, top_k: int = 5) -> Dict[str, Any]:
        """Inspects top-k next token predictions and confidence metrics for a seed prompt."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[0, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)

            top_probs, top_indices = torch.topk(probs, top_k)
            top_tokens = []
            for prob, idx in zip(top_probs, top_indices):
                token_str = self.tokenizer.decode([idx.item()])
                top_tokens.append({
                    "token_id": idx.item(),
                    "token_text": token_str,
                    "probability": round(prob.item(), 4)
                })

            top_1_confidence = top_tokens[0]["probability"] if top_tokens else 0.0
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

            return {
                "prompt": prompt,
                "top_1_token": top_tokens[0]["token_text"] if top_tokens else "",
                "top_1_confidence": top_1_confidence,
                "entropy": round(entropy, 4),
                "top_k_predictions": top_tokens
            }
        except Exception as e:
            logger.error(f"TokenInspector: Error inspecting prompt '{prompt}': {e}")
            return {
                "prompt": prompt,
                "error": str(e)
            }

    def inspect_canonical_prompts(self, prompts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Runs next-token inspection over standard Manipuri seed prompts."""
        if not prompts:
            prompts = ["ꯑꯩ", "অদুগা", "ꯃꯅꯤꯄꯨꯔ", "ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯑꯁꯤ", "মণিপুরী ভাষা"]

        results = []
        for prompt in prompts:
            res = self.inspect_prompt(prompt)
            results.append(res)
        return results
