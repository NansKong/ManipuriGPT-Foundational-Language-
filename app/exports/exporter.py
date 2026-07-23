"""
UnifiedExporter module coordinating model export targets (`hf`, `gguf`, `onnx`).
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from app.exports.hf_export import HFHubExporter
from app.exports.gguf_export import GGUFExporter
from app.exports.onnx_export import ONNXExporter
from app.utils.logger import logger


class UnifiedExporter:
    """
    Unified high-level interface for exporting trained Manipuri checkpoints across targets:
    `hf` (Hugging Face Hub), `gguf` (`llama.cpp` quantization), `onnx` (ONNX Runtime).
    """
    def __init__(self):
        self.hf_exporter = HFHubExporter()
        self.gguf_exporter = GGUFExporter()
        self.onnx_exporter = ONNXExporter()

    def export_all(
        self,
        checkpoint_dir: str,
        model_name: str,
        targets: Optional[List[str]] = None,
        hf_repo_id: Optional[str] = None,
        gguf_quantization: str = "Q4_K_M",
        simulate: bool = True
    ) -> Dict[str, Any]:
        """
        Runs exports for all requested targets (`hf`, `gguf`, `onnx`).
        """
        if not targets:
            targets = ["hf", "gguf", "onnx"]
        
        results: Dict[str, Any] = {}
        logger.info(f"UnifiedExporter: Starting export of '{checkpoint_dir}' for targets: {targets}")

        for target in targets:
            t_clean = target.lower().strip()
            if t_clean == "hf":
                repo = hf_repo_id or f"manipurigpt/{model_name}"
                results["hf"] = self.hf_exporter.export(checkpoint_dir, repo_id=repo, simulate=simulate)
            elif t_clean == "gguf":
                results["gguf"] = self.gguf_exporter.export(checkpoint_dir, model_name=model_name, quantization=gguf_quantization, simulate=simulate)
            elif t_clean == "onnx":
                results["onnx"] = self.onnx_exporter.export(checkpoint_dir, model_name=model_name, simulate=simulate)
            else:
                logger.warning(f"UnifiedExporter: Unknown target '{target}' skipped.")

        logger.info(f"UnifiedExporter: Completed multi-target exports -> {list(results.keys())}")
        return results
