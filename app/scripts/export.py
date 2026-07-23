"""
Production CLI script for exporting checkpoints to distribution targets (`Phase 5`).
Usage: `python -m app.scripts.export --checkpoint artifacts/models/checkpoints --model smollm_135m --targets hf gguf onnx`
"""

import argparse
import sys
from typing import Optional, List, Dict, Any, Union, Tuple
from app.exports.exporter import UnifiedExporter
from app.utils.logger import logger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Model Export CLI")
    parser.add_argument("--checkpoint", type=str, default="artifacts/models/checkpoints", help="Path to saved model checkpoint directory")
    parser.add_argument("--model", type=str, default="smollm_135m", help="Model identifier/name")
    parser.add_argument("--targets", nargs="+", default=["hf", "gguf", "onnx"], choices=["hf", "gguf", "onnx"], help="Export distribution targets")
    parser.add_argument("--gguf-quant", type=str, default="Q4_K_M", help="GGUF quantization level")
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)
    logger.info(f"CLI: Launching model export for '{parsed.model}' across targets: {parsed.targets}")

    exporter = UnifiedExporter()
    results = exporter.export_all(
        checkpoint_dir=parsed.checkpoint,
        model_name=parsed.model,
        targets=parsed.targets,
        gguf_quantization=parsed.gguf_quant,
        simulate=True
    )
    logger.info(f"CLI: Export operations completed -> {list(results.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
