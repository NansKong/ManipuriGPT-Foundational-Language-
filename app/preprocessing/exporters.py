import os
from typing import Union, Dict, Any
from datasets import Dataset, DatasetDict
from app.utils.logger import logger

class DatasetExporter:
    """
    Component for exporting cleaned/processed datasets in various formats
    (Arrow, JSONL, CSV, Parquet) sharing a common interface.
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.format = config.get("format", "arrow").lower()
        self.output_dir = config.get("output_dir", "data/processed")
        self.version = config.get("version", "")

    def get_resolved_output_dir(self, output_dir_override: str = None) -> str:
        """Resolves the output directory, appending version if specified."""
        base_dir = output_dir_override or self.output_dir
        if self.version:
            return os.path.join(base_dir, self.version)
        return base_dir

    def export(
        self, 
        dataset: Union[Dataset, DatasetDict], 
        format_override: str = None, 
        output_dir_override: str = None,
        file_prefix: str = "dataset"
    ) -> list[str]:
        """
        Exports the dataset or dataset dictionary to the target directory.
        Returns a list of exported file paths.
        """
        format_type = (format_override or self.format).lower()
        output_dir = self.get_resolved_output_dir(output_dir_override)
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Exporter: Exporting dataset to '{output_dir}' using format '{format_type}'")

        exported_paths = []
        if isinstance(dataset, DatasetDict):
            for split_name, ds in dataset.items():
                split_path = os.path.join(output_dir, f"{file_prefix}_{split_name}")
                written_path = self._export_single(ds, format_type, split_path)
                exported_paths.append(written_path)
        else:
            split_path = os.path.join(output_dir, file_prefix)
            written_path = self._export_single(dataset, format_type, split_path)
            exported_paths.append(written_path)
            
        return exported_paths

    def _export_single(self, dataset: Dataset, format_type: str, dest_path_prefix: str) -> str:
        """Helper to export a single Dataset object. Returns the actual exported file path."""
        if format_type == "arrow":
            # Save using HuggingFace disk format (which includes metadata)
            dataset.save_to_disk(dest_path_prefix)
            logger.info(f"Exporter: Exported Arrow dataset to '{dest_path_prefix}'")
            return dest_path_prefix
            
        elif format_type == "jsonl":
            dest_file = f"{dest_path_prefix}.jsonl"
            dataset.to_json(dest_file)
            logger.info(f"Exporter: Exported JSONL to '{dest_file}'")
            return dest_file
            
        elif format_type == "csv":
            dest_file = f"{dest_path_prefix}.csv"
            dataset.to_csv(dest_file)
            logger.info(f"Exporter: Exported CSV to '{dest_file}'")
            return dest_file
            
        elif format_type == "parquet":
            dest_file = f"{dest_path_prefix}.parquet"
            dataset.to_parquet(dest_file)
            logger.info(f"Exporter: Exported Parquet to '{dest_file}'")
            return dest_file
            
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

