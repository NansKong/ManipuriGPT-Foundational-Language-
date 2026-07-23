"""
ManipuriGPT Tokenization Training, Benchmarking, and Evaluation Module (Phase 5.3).
Bridges core tokenization architecture from `app.tokenizer` with Phase 5.3 training,
evaluation, and export suites.
"""

# Re-export core tokenizer modules for seamless backward & forward compatibility
from app.tokenizer.context import PipelineContext
from app.tokenizer.normalizer import ConversationNormalizer
from app.tokenizer.tokenizer_manager import TokenizerManager
from app.tokenizer.formatter import PromptFormatter
from app.tokenizer.templates import apply_chat_template, CHAT_TEMPLATES

from app.tokenization.trainer import TokenizerTrainer
from app.tokenization.benchmark import TokenizerBenchmarker
from app.tokenization.evaluator import TokenizerEvaluator

__all__ = [
    "PipelineContext",
    "ConversationNormalizer",
    "TokenizerManager",
    "PromptFormatter",
    "apply_chat_template",
    "CHAT_TEMPLATES",
    "TokenizerTrainer",
    "TokenizerBenchmarker",
    "TokenizerEvaluator",
]
