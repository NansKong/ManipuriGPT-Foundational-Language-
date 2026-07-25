"""
EM-ALBERT Evaluation & Benchmarking Module (`app/evaluation/em_albert_eval.py`).

Provides evaluation, embedding representation comparison, and downstream probing
using the pretrained EM-ALBERT model stored at `W0316/Manipuri resources/EM-ALBERT/em-albert`.

Placed strictly inside the evaluation/benchmarking layer.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.utils.logger import logger

DEFAULT_EM_ALBERT_PATH = Path("cache/datasets/W0316/Manipuri resources/EM-ALBERT/em-albert")


class EMAlbertEvaluator:
    """
    Evaluator leveraging EM-ALBERT for representation benchmarking, sentence similarity,
    and downstream Manipuri NLP evaluation.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_EM_ALBERT_PATH
        self._tokenizer = None
        self._model = None
        self._loaded = False

    def is_available(self) -> bool:
        """Checks if EM-ALBERT model directory or weight file exists on disk."""
        return self.model_path.exists() or (self.model_path.parent / "em-albert").exists()

    def load(self) -> bool:
        """Loads EM-ALBERT model and tokenizer using Hugging Face Transformers."""
        if self._loaded:
            return True

        if not self.is_available():
            logger.warning(f"EMAlbertEvaluator: EM-ALBERT path not found at {self.model_path}")
            return False

        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._model = AutoModel.from_pretrained(str(self.model_path))
            self._model.eval()
            self._loaded = True
            logger.info(f"EMAlbertEvaluator: Successfully loaded EM-ALBERT from {self.model_path}")
            return True
        except Exception as e:
            logger.warning(f"EMAlbertEvaluator: Failed to load EM-ALBERT model via transformers ({e})")
            return False

    def extract_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """
        Extracts pooled CLS embeddings for a list of input texts.
        """
        if not self._loaded:
            if not self.load():
                return {"status": "unavailable", "embeddings": []}

        try:
            import torch
            inputs = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()

            return {
                "status": "success",
                "model_name": "EM-ALBERT",
                "embeddings": cls_embeddings,
                "count": len(texts),
            }
        except Exception as e:
            logger.error(f"EMAlbertEvaluator: Error extracting embeddings: {e}")
            return {"status": "error", "message": str(e), "embeddings": []}

    def compute_representation_similarity(self, text_a: str, text_b: str) -> float:
        """
        Computes cosine similarity between two text representations extracted from EM-ALBERT.
        """
        res = self.extract_embeddings([text_a, text_b])
        if res.get("status") != "success" or len(res.get("embeddings", [])) < 2:
            return 0.0

        emb_a = res["embeddings"][0]
        emb_b = res["embeddings"][1]

        dot = sum(x * y for x, y in zip(emb_a, emb_b))
        norm_a = sum(x * x for x in emb_a) ** 0.5
        norm_b = sum(y * y for y in emb_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))
