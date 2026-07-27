"""
Sequence Packing Engine for Pretraining (`app/training/packer.py`).

Packs tokenized text sequences into fixed context length blocks (512 / 1024 / 2048 / 4096 tokens)
with EOS boundary separation to eliminate padding waste and maximize GPU training efficiency.
"""

import os
import json
from typing import List, Dict, Any, Generator, Optional

from app.utils.logger import logger


class SequencePacker:
    """Concatenates variable-length tokenized sequences into fixed context blocks."""

    def __init__(self, tokenizer_path: str, max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len
        self.tokenizer_path = tokenizer_path
        self.sp = None
        if os.path.exists(tokenizer_path):
            try:
                import sentencepiece as spm
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load(tokenizer_path)
            except Exception as e:
                logger.warning(f"SequencePacker: Failed to load SentencePiece tokenizer: {e}")

    def pack_texts(self, texts: List[str]) -> Generator[Dict[str, Any], None, None]:
        """Pack raw text sequences into fixed max_seq_len token blocks."""
        if not self.sp:
            raise RuntimeError("SentencePiece tokenizer is required to pack sequences.")

        eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else 3
        current_block: List[int] = []
        packed_count = 0

        for text in texts:
            if not text:
                continue
            token_ids = self.sp.Encode(str(text).strip())
            if not token_ids:
                continue

            token_ids.append(eos_id)

            for tid in token_ids:
                current_block.append(tid)
                if len(current_block) == self.max_seq_len:
                    yield {
                        "input_ids": list(current_block),
                        "attention_mask": [1] * self.max_seq_len,
                        "seq_len": self.max_seq_len
                    }
                    packed_count += 1
                    current_block = []

        # Remaining trailing tokens if any (padded with pad_id = 0)
        if current_block:
            pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
            pad_len = self.max_seq_len - len(current_block)
            mask = [1] * len(current_block) + [0] * pad_len
            padded_block = current_block + [pad_id] * pad_len
            yield {
                "input_ids": padded_block,
                "attention_mask": mask,
                "seq_len": self.max_seq_len
            }

    def pack_and_save_shards(
        self,
        texts: List[str],
        output_dir: str,
        seq_lens: List[int] = [512, 1024, 2048, 4096],
        shard_size: int = 5000
    ) -> Dict[int, int]:
        """Generate packed Parquet dataset shards for multiple context window sizes."""
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets library is required to save packed shards.")

        results = {}
        for s_len in seq_lens:
            self.max_seq_len = s_len
            target_sub_dir = os.path.join(output_dir, f"packed_seq_{s_len}")
            os.makedirs(target_sub_dir, exist_ok=True)

            logger.info(f"Packing dataset for context window max_seq_len={s_len}...")
            packed_rows = list(self.pack_texts(texts))
            logger.info(f"  Generated {len(packed_rows):,} packed blocks of length {s_len}")

            rows_in_packed = len(packed_rows)
            shard_idx = 0
            for start_i in range(0, rows_in_packed, shard_size):
                end_i = min(start_i + shard_size, rows_in_packed)
                chunk = packed_rows[start_i:end_i]
                ds = Dataset.from_list(chunk)
                shard_filename = f"packed-shard-{shard_idx:05d}.parquet"
                ds.to_parquet(os.path.join(target_sub_dir, shard_filename))
                shard_idx += 1

            results[s_len] = len(packed_rows)

        return results
