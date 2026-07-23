from typing import Dict, Any, List, Optional
from app.utils.logger import logger

class TokenizationValidator:
    """
    Validates tokenized examples before passing to the training backend.
    Checks for empty input_ids, matching attention_mask and labels lengths, and max_length bounds.
    """
    def __init__(self, max_length: int = 2048, check_labels: bool = True):
        self.max_length = max_length
        self.check_labels = check_labels

    def validate_example(self, example: Dict[str, Any]) -> bool:
        """
        Validates a single tokenized dictionary. Returns True if valid, raises or returns False otherwise.
        """
        input_ids = example.get("input_ids")
        if input_ids is None or len(input_ids) == 0:
            logger.warning("TokenizationValidator: Rejected sample with empty input_ids.")
            return False

        if len(input_ids) > self.max_length:
            logger.warning(f"TokenizationValidator: Sample exceeds max_length ({len(input_ids)} > {self.max_length}).")
            return False

        attention_mask = example.get("attention_mask")
        if attention_mask is not None and len(attention_mask) != len(input_ids):
            logger.warning(f"TokenizationValidator: Mismatch between input_ids ({len(input_ids)}) and attention_mask ({len(attention_mask)}).")
            return False

        if self.check_labels:
            labels = example.get("labels")
            if labels is not None and len(labels) != len(input_ids):
                logger.warning(f"TokenizationValidator: Mismatch between input_ids ({len(input_ids)}) and labels ({len(labels)}).")
                return False

        return True

    def validate_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Filters a batch dictionary to only retain valid samples.
        """
        valid_indices = []
        input_ids_list = batch.get("input_ids", [])
        attention_mask_list = batch.get("attention_mask", [None] * len(input_ids_list))
        labels_list = batch.get("labels", [None] * len(input_ids_list))

        for idx in range(len(input_ids_list)):
            sample = {
                "input_ids": input_ids_list[idx],
                "attention_mask": attention_mask_list[idx] if attention_mask_list[idx] is not None else None,
                "labels": labels_list[idx] if labels_list[idx] is not None else None
            }
            if self.validate_example(sample):
                valid_indices.append(idx)

        # Reconstruct batch with valid indices
        filtered_batch = {}
        for k, v_list in batch.items():
            filtered_batch[k] = [v_list[i] for i in valid_indices]

        return filtered_batch
