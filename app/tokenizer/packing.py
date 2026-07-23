from typing import Dict, Any, List
from app.utils.logger import logger

class SequencePacker:
    """
    Concatenates multiple tokenized samples separated by eos_token_id into uniform chunks of max_length.
    Improves training throughput and GPU utilization by eliminating padding overhead.
    """
    def __init__(self, max_length: int = 2048, eos_token_id: int = 2, pad_token_id: int = 0):
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def pack(self, examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        """
        Packs a batch dictionary containing lists of lists ('input_ids', 'attention_mask', 'labels')
        into dense blocks of size `self.max_length`.
        """
        # Concatenate all sequences across the batch
        concatenated: Dict[str, List[int]] = {k: [] for k in examples.keys()}
        
        # Ensure we process input_ids, attention_mask, and labels
        keys = list(examples.keys())
        num_examples = len(examples.get("input_ids", []))
        
        if num_examples == 0:
            return {k: [] for k in keys}

        for i in range(num_examples):
            for k in keys:
                seq = list(examples[k][i])
                # Ensure each sequence ends with eos token if it's input_ids or labels
                if k in ["input_ids", "labels"] and (not seq or seq[-1] != self.eos_token_id):
                    seq.append(self.eos_token_id)
                elif k == "attention_mask" and (len(seq) < len(examples["input_ids"][i]) + 1):
                    seq.append(1)
                concatenated[k].extend(seq)

        # Calculate total chunks of max_length we can form
        total_length = len(concatenated.get("input_ids", []))
        total_chunks = total_length // self.max_length

        if total_chunks == 0:
            # If total concatenated tokens is less than one chunk, pad or drop depending on need
            # Here we pack into one padded chunk if total_length > 0
            if total_length > 0:
                packed: Dict[str, List[List[int]]] = {}
                pad_len = self.max_length - total_length
                for k in keys:
                    pad_val = self.pad_token_id if k != "labels" else -100
                    if k == "attention_mask":
                        pad_val = 0
                    packed[k] = [concatenated[k] + [pad_val] * pad_len]
                return packed
            return {k: [] for k in keys}

        # Slice into chunks of self.max_length
        packed = {k: [] for k in keys}
        for chunk_idx in range(total_chunks):
            start = chunk_idx * self.max_length
            end = start + self.max_length
            for k in keys:
                packed[k].append(concatenated[k][start:end])

        logger.debug(f"SequencePacker: Packed {num_examples} samples into {total_chunks} blocks of length {self.max_length}.")
        return packed
