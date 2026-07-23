import os
import json
import subprocess
from datetime import datetime
from typing import Union, Dict, Any, Optional
from datasets import Dataset, DatasetDict
from app.utils.logger import logger

class TokenizedDatasetExporter:
    """
    Exports tokenized Hugging Face datasets to Arrow, Parquet, JSONL, or CSV
    under centralized artifacts/ directory, automatically initializes standard directories,
    and writes manifest.json enriched with Git commit and version metadata.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.output_dir = config.get("output_dir", "artifacts/datasets/tokenized")
        self.format = config.get("format", "arrow").lower()
        self._ensure_artifacts_hierarchy()

    def _ensure_artifacts_hierarchy(self) -> None:
        """Initializes standard top-level artifact directories."""
        base_dirs = [
            "artifacts/datasets",
            "artifacts/models",
            "artifacts/reports",
            "artifacts/logs",
            "artifacts/benchmarks",
            "artifacts/exports"
        ]
        for d in base_dirs:
            os.makedirs(d, exist_ok=True)

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    def export(
        self,
        dataset: Union[Dataset, DatasetDict],
        format_override: Optional[str] = None,
        output_dir_override: Optional[str] = None,
        file_prefix: str = "tokenized_dataset",
        metadata: Optional[Dict[str, Any]] = None,
        ctx: Optional[Any] = None
    ) -> None:
        """Exports dataset along with enriched manifest.json."""
        fmt = (format_override or self.format).lower()
        out_dir = output_dir_override or self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"TokenizedDatasetExporter: Exporting dataset to '{out_dir}' in format '{fmt}'")

        if isinstance(dataset, DatasetDict):
            total_samples = 0
            for split_name, ds in dataset.items():
                dest = os.path.join(out_dir, f"{file_prefix}_{split_name}")
                self._export_single(ds, fmt, dest)
                if hasattr(ds, "__len__"):
                    try:
                        total_samples += len(ds)
                    except TypeError:
                        pass
        else:
            dest = os.path.join(out_dir, file_prefix)
            self._export_single(dataset, fmt, dest)
            total_samples = len(dataset) if hasattr(dataset, "__len__") else 0

        self._save_manifest(out_dir, total_samples, metadata or {}, ctx=ctx)

    def _export_single(self, dataset: Dataset, fmt: str, dest_prefix: str) -> None:
        if fmt == "arrow":
            dataset.save_to_disk(dest_prefix)
            logger.info(f"TokenizedDatasetExporter: Saved Arrow dataset to '{dest_prefix}'")
        elif fmt == "jsonl":
            dest_file = f"{dest_prefix}.jsonl"
            dataset.to_json(dest_file)
            logger.info(f"TokenizedDatasetExporter: Saved JSONL dataset to '{dest_file}'")
        elif fmt == "parquet":
            dest_file = f"{dest_prefix}.parquet"
            dataset.to_parquet(dest_file)
            logger.info(f"TokenizedDatasetExporter: Saved Parquet dataset to '{dest_file}'")
        elif fmt == "csv":
            dest_file = f"{dest_prefix}.csv"
            dataset.to_csv(dest_file)
            logger.info(f"TokenizedDatasetExporter: Saved CSV dataset to '{dest_file}'")
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

    def _save_manifest(self, out_dir: str, total_samples: int, metadata: Dict[str, Any], ctx: Optional[Any] = None) -> None:
        manifest_path = os.path.join(out_dir, "manifest.json")
        model_name = metadata.get("model", getattr(ctx, "model_name", "Qwen2.5-3B"))
        task_name = metadata.get("task", getattr(ctx, "task_name", "translation"))
        
        manifest_data = {
            "model": model_name,
            "task": task_name,
            "git_commit": self._get_git_commit(),
            "pipeline_version": metadata.get("pipeline_version", "0.4.0"),
            "dataset_version": metadata.get("dataset_version", "v2"),
            "tokenizer_version": metadata.get("tokenizer_version", "huggingface"),
            "samples": total_samples or metadata.get("samples", 0),
            "tokens": metadata.get("tokens", 0),
            "packed": metadata.get("packed", getattr(ctx, "packing", True)),
            "max_length": metadata.get("max_length", getattr(ctx, "max_length", 2048)),
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        logger.info(f"TokenizedDatasetExporter: Saved dataset manifest to '{manifest_path}'")
