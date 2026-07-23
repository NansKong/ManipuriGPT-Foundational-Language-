"""
ManipuriGPT Evaluation Suite Module (Phase 5).
Exports metrics registry, automated scorecard evaluator, and human evaluation pipeline.
"""

from app.evaluation.metrics import MetricRegistry
from app.evaluation.evaluator import ManipuriEvaluator
from app.evaluation.human import HumanEvaluationPipeline

__all__ = [
    "MetricRegistry",
    "ManipuriEvaluator",
    "HumanEvaluationPipeline",
]
