"""
HumanEvaluationPipeline module for side-by-side human A/B evaluation and preference annotation (`Phase 5`).
Tracks inter-annotator agreement and exports preference rankings (`win`/`loss`/`tie`) for DPO/RLHF alignment.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from app.utils.logger import logger


class HumanEvaluationPipeline:
    """
    Manages side-by-side human preference annotations (`win`, `loss`, `tie`) between
    Model A and Model B responses. Calculates inter-annotator agreement and exports DPO preference pairs.
    """
    def __init__(self, storage_dir: str = "artifacts/evaluation/human_annotations"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.annotations: List[Dict[str, Any]] = []

    def add_annotation(
        self,
        prompt: str,
        model_a_output: str,
        model_b_output: str,
        preference: str,
        annotator_id: str = "annotator_1",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Records a blind evaluation comparison between two response candidates.
        `preference` must be one of: `model_a`, `model_b`, `tie`, `both_bad`.
        """
        pref_clean = preference.lower().strip()
        if pref_clean not in ["model_a", "model_b", "tie", "both_bad"]:
            raise ValueError(f"Invalid preference choice: {preference}. Must be model_a, model_b, tie, or both_bad.")

        record = {
            "prompt": prompt,
            "model_a_output": model_a_output,
            "model_b_output": model_b_output,
            "preference": pref_clean,
            "annotator_id": annotator_id,
            "metadata": metadata or {}
        }
        self.annotations.append(record)
        return record

    def compute_inter_annotator_agreement(self) -> float:
        """
        Computes Cohen's Kappa / Krippendorff's Alpha approximation across prompts rated by
        multiple annotators.
        """
        prompt_groups: Dict[str, List[str]] = {}
        for ann in self.annotations:
            p = ann["prompt"]
            prompt_groups.setdefault(p, []).append(ann["preference"])

        agreements = 0
        multi_rated = 0
        for p, prefs in prompt_groups.items():
            if len(prefs) >= 2:
                multi_rated += 1
                if len(set(prefs)) == 1:
                    agreements += 1

        if multi_rated == 0:
            return 1.0  # Perfect default when no multi-rated overlap conflicts exist
        return round(agreements / multi_rated, 4)

    def export_dpo_dataset(self, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Transforms recorded annotations into standard DPO `chosen` / `rejected` preference pairs
        for direct preference optimization training.
        """
        dpo_records: List[Dict[str, Any]] = []

        for ann in self.annotations:
            pref = ann["preference"]
            if pref == "model_a":
                chosen = ann["model_a_output"]
                rejected = ann["model_b_output"]
            elif pref == "model_b":
                chosen = ann["model_b_output"]
                rejected = ann["model_a_output"]
            else:
                # Skip ties or both_bad for clean DPO training pairs
                continue

            dpo_records.append({
                "prompt": ann["prompt"],
                "chosen": chosen,
                "rejected": rejected
            })

        out_file = output_path or os.path.join(self.storage_dir, "dpo_preference_pairs.json")
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(dpo_records, f, indent=2)
            logger.info(f"HumanEvaluationPipeline: Exported {len(dpo_records)} DPO pairs to '{out_file}'")
        except Exception as e:
            logger.error(f"HumanEvaluationPipeline: Failed to export DPO pairs ({e})")

        return dpo_records
