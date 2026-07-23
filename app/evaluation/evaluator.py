"""
ManipuriEvaluator module orchestrating automated benchmark evaluation (Phase 5).
Calculates scorecard metrics across tasks: `chat`, `translation`, `qa`, `reasoning`.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from app.evaluation.metrics import MetricRegistry
from app.utils.logger import logger


class ManipuriEvaluator:
    """
    Automated evaluation engine. Takes generated model outputs against benchmark datasets
    and produces a structured, multi-metric evaluation scorecard.
    """
    def __init__(self):
        self.metrics = MetricRegistry()

    def evaluate_task(
        self,
        task_type: str,
        predictions: List[str],
        references: List[str],
        log_losses: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Runs relevant metric evaluations tailored to the target task type.
        """
        t_clean = task_type.lower().strip()
        logger.info(f"ManipuriEvaluator: Evaluating {len(predictions)} samples for task '{task_type}'...")

        scorecard: Dict[str, Any] = {
            "task": task_type,
            "samples_evaluated": len(predictions),
            "perplexity": self.metrics.calculate("perplexity", predictions, references, log_losses=log_losses)
        }

        if t_clean in ["translation", "paraphrase"]:
            scorecard["bleu"] = self.metrics.calculate("bleu", predictions, references)
            scorecard["chrf"] = self.metrics.calculate("chrf", predictions, references)
            scorecard["bertscore"] = self.metrics.calculate("bertscore", predictions, references)
        elif t_clean in ["qa", "classification"]:
            scorecard["accuracy"] = self.metrics.calculate("accuracy", predictions, references)
            scorecard["f1"] = self.metrics.calculate("f1", predictions, references)
            scorecard["hallucination_rate"] = self.metrics.calculate("hallucination_rate", predictions, references)
        elif t_clean in ["chat", "summarization", "instruction"]:
            scorecard["rouge"] = self.metrics.calculate("rouge", predictions, references)
            scorecard["bertscore"] = self.metrics.calculate("bertscore", predictions, references)
            scorecard["hallucination_rate"] = self.metrics.calculate("hallucination_rate", predictions, references)
        else:
            # General evaluation computes all standard scores
            scorecard["bleu"] = self.metrics.calculate("bleu", predictions, references)
            scorecard["rouge"] = self.metrics.calculate("rouge", predictions, references)
            scorecard["accuracy"] = self.metrics.calculate("accuracy", predictions, references)

        logger.info(f"ManipuriEvaluator: Scorecard generated for '{task_type}' -> {scorecard}")
        return scorecard
