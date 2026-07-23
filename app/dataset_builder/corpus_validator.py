"""
CorpusValidator module (`app/dataset_builder/corpus_validator.py`).
Performs rigorous quantitative and qualitative validation on the assembled corpus:
- Duplicate percentage & retention
- Vocabulary coverage & unknown token rate (<unk> density)
- Sequence length statistics (characters & subword tokens)
- Token distribution percentiles (p10, p50, p90, p99, max)
- Script balance across Meitei Mayek, Bengali script, Latin/Romanized, and mixed
- Source distribution across parallel & monolingual datasets
Generates both `corpus_report.json` and Hugging Face `README.md` (Dataset Card).
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Union, List, Optional
from datasets import Dataset, DatasetDict
from app.utils.logger import logger


class CorpusValidator:
    """
    Evaluates dataset splits against Phase 5.4 validation criteria before pretraining.
    """
    def __init__(self, tokenizer_path: Optional[str] = None):
        self.tokenizer_path = tokenizer_path
        self.sp = None
        self.unk_id = 0
        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                import sentencepiece as spm
                self.sp = spm.SentencePieceProcessor()
                self.sp.load(tokenizer_path)
                self.unk_id = self.sp.unk_id()
                logger.info(f"CorpusValidator: Loaded SentencePiece tokenizer from '{tokenizer_path}'")
            except Exception as e:
                logger.warning(f"CorpusValidator: Could not load SentencePiece model ({e})")

    def _tokenize(self, text: str) -> List[int]:
        if self.sp:
            return self.sp.encode_as_ids(text)
        # Approximate subword tokenization if tokenizer not loaded (words * 1.4)
        return [0] * int(len(text.split()) * 1.4)

    def evaluate(
        self,
        dataset_or_dict: Union[Dataset, DatasetDict, List[Dict[str, Any]]],
        raw_count_before_dedup: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Computes comprehensive evaluation metrics across all splits and sequences.
        """
        logger.info("CorpusValidator: Evaluating corpus quality and tokenization statistics...")

        splits_data: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(dataset_or_dict, DatasetDict):
            for name, ds in dataset_or_dict.items():
                splits_data[name] = [row for row in ds]
        elif isinstance(dataset_or_dict, Dataset):
            splits_data["train"] = [row for row in dataset_or_dict]
        elif isinstance(dataset_or_dict, list):
            splits_data["train"] = dataset_or_dict
        else:
            raise TypeError("Unsupported dataset format passed to CorpusValidator.evaluate()")

        total_sequences = 0
        total_characters = 0
        total_tokens = 0
        total_unk_tokens = 0
        all_token_lengths: List[int] = []

        script_counts: Dict[str, int] = {"meitei": 0, "bengali": 0, "latin": 0, "mixed": 0, "unknown": 0}
        source_counts: Dict[str, int] = {}
        source_tokens: Dict[str, int] = {}
        split_summaries: Dict[str, Any] = {}

        for split_name, rows in splits_data.items():
            split_seqs = len(rows)
            split_chars = 0
            split_tokens = 0
            split_unks = 0
            split_token_lens: List[int] = []

            for row in rows:
                text = row.get("text", "")
                if not text:
                    continue

                chars_len = len(text)
                tokens = self._tokenize(text)
                tok_len = len(tokens)
                unks = tokens.count(self.unk_id) if self.sp else 0

                split_chars += chars_len
                split_tokens += tok_len
                split_unks += unks
                split_token_lens.append(tok_len)

                # Script tracking
                script = str(row.get("script", "unknown")).lower().strip()
                if script in script_counts:
                    script_counts[script] += 1
                else:
                    script_counts["unknown"] += 1

                # Source tracking
                src = str(row.get("source", row.get("source_dataset", "unknown"))).strip()
                source_counts[src] = source_counts.get(src, 0) + 1
                source_tokens[src] = source_tokens.get(src, 0) + tok_len

            total_sequences += split_seqs
            total_characters += split_chars
            total_tokens += split_tokens
            total_unk_tokens += split_unks
            all_token_lengths.extend(split_token_lens)

            avg_char = round(split_chars / max(split_seqs, 1), 2)
            avg_tok = round(split_tokens / max(split_seqs, 1), 2)
            split_summaries[split_name] = {
                "sequences": split_seqs,
                "characters": split_chars,
                "tokens": split_tokens,
                "unk_tokens": split_unks,
                "avg_sequence_length_chars": avg_char,
                "avg_sequence_length_tokens": avg_tok,
                "unk_rate": round((split_unks / max(split_tokens, 1)) * 100, 4)
            }

        # Calculate duplicate statistics
        duplicate_pct = 0.0
        retention_pct = 100.0
        if raw_count_before_dedup and raw_count_before_dedup > 0:
            dups = max(0, raw_count_before_dedup - total_sequences)
            duplicate_pct = round((dups / raw_count_before_dedup) * 100, 2)
            retention_pct = round((total_sequences / raw_count_before_dedup) * 100, 2)

        # Calculate percentiles (p10, p50, p90, p99, max)
        sorted_lens = sorted(all_token_lengths)
        n = len(sorted_lens)
        percentiles = {
            "p10": sorted_lens[int(n * 0.10)] if n else 0,
            "p50": sorted_lens[int(n * 0.50)] if n else 0,
            "p90": sorted_lens[int(n * 0.90)] if n else 0,
            "p99": sorted_lens[int(n * 0.99)] if n else 0,
            "max": sorted_lens[-1] if n else 0,
        }

        # Calculate script balance percentage
        script_balance_pct = {
            k: round((v / max(total_sequences, 1)) * 100, 2) for k, v in script_counts.items()
        }

        # Calculate source distribution percentage
        source_distribution_pct = {
            k: {
                "sequences": v,
                "pct_sequences": round((v / max(total_sequences, 1)) * 100, 2),
                "tokens": source_tokens.get(k, 0),
                "pct_tokens": round((source_tokens.get(k, 0) / max(total_tokens, 1)) * 100, 2)
            }
            for k, v in source_counts.items()
        }

        overall_unk_rate = round((total_unk_tokens / max(total_tokens, 1)) * 100, 4)
        avg_overall_char = round(total_characters / max(total_sequences, 1), 2)
        avg_overall_tok = round(total_tokens / max(total_sequences, 1), 2)

        report = {
            "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": "5.4",
            "tokenizer_used": self.tokenizer_path or "estimated_subwords",
            "overall_statistics": {
                "total_sequences": total_sequences,
                "total_characters": total_characters,
                "total_tokens": total_tokens,
                "total_unk_tokens": total_unk_tokens,
                "unknown_token_rate_pct": overall_unk_rate,
                "avg_sequence_length_chars": avg_overall_char,
                "avg_sequence_length_tokens": avg_overall_tok,
            },
            "deduplication_metrics": {
                "raw_examples_before_deduplication": raw_count_before_dedup or total_sequences,
                "clean_sequences_retained": total_sequences,
                "duplicate_percentage": duplicate_pct,
                "retention_percentage": retention_pct
            },
            "token_length_distribution": percentiles,
            "script_balance_pct": script_balance_pct,
            "source_distribution": source_distribution_pct,
            "split_summaries": split_summaries
        }

        logger.info(f"CorpusValidator: Evaluation complete. Total sequences={total_sequences:,}, tokens={total_tokens:,}, UNK rate={overall_unk_rate}%")
        return report

    def save_report(
        self,
        report_data: Dict[str, Any],
        output_dir: str = "artifacts/datasets/ManipuriGPT-Corpus-v1/metadata",
        dataset_card_path: str = "artifacts/datasets/ManipuriGPT-Corpus-v1/README.md"
    ) -> Dict[str, str]:
        """
        Writes `corpus_report.json` and generates the Hugging Face `README.md` dataset card.
        """
        os.makedirs(output_dir, exist_ok=True)
        report_json_path = os.path.join(output_dir, "corpus_report.json")

        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"CorpusValidator: Saved corpus report to '{report_json_path}'")

        # Generate README.md Dataset Card
        os.makedirs(os.path.dirname(dataset_card_path), exist_ok=True)
        card_content = self._generate_dataset_card_markdown(report_data)
        with open(dataset_card_path, "w", encoding="utf-8") as f:
            f.write(card_content)
        logger.info(f"CorpusValidator: Saved Dataset Card to '{dataset_card_path}'")

        return {
            "corpus_report.json": report_json_path,
            "README.md": dataset_card_path
        }

    def _generate_dataset_card_markdown(self, report: Dict[str, Any]) -> str:
        overall = report.get("overall_statistics", {})
        dedup = report.get("deduplication_metrics", {})
        dist = report.get("token_length_distribution", {})
        splits = report.get("split_summaries", {})
        scripts = report.get("script_balance_pct", {})
        sources = report.get("source_distribution", {})

        splits_table = "\n".join([
            f"| `{split}` | {data['sequences']:,} | {data['characters']:,} | {data['tokens']:,} | {data['avg_sequence_length_tokens']} | {data['unk_rate']}% |"
            for split, data in splits.items()
        ])

        sources_table = "\n".join([
            f"| `{src}` | {data['sequences']:,} ({data['pct_sequences']}%) | {data['tokens']:,} ({data['pct_tokens']}%) |"
            for src, data in sources.items()
        ])

        return f"""---
language:
- mni
license: apache-2.0
task_categories:
- text-generation
- translation
tags:
- manipuri
- meiteilon
- meitei-mayek
- foundation-model
size_categories:
- 100K<n<1M
---

# ManipuriGPT-Corpus-v1: Training-Ready Sharded Foundation Corpus

**ManipuriGPT-Corpus-v1** is a research-grade, deduplicated, and quality-validated dataset built specifically for pretraining and fine-tuning open foundation language models for the Manipuri (Meiteilon) language across all native scripts.

## 📊 Dataset Summary

* **Total Sequences**: {overall.get('total_sequences', 0):,}
* **Total Subword Tokens**: ~{overall.get('total_tokens', 0):,}
* **Total Characters**: {overall.get('total_characters', 0):,}
* **Unknown Token Rate (`<unk>`)**: `{overall.get('unknown_token_rate_pct', 0)}%`
* **Duplicate Removal Rate**: `{dedup.get('duplicate_percentage', 0)}%` (Retention: `{dedup.get('retention_percentage', 100)}%`)
* **Tokenizer Artifact**: Tokenizer v1 (`{report.get('tokenizer_used', 'SentencePiece')}`)

---

## 🗂️ Dataset Splits (`98 / 1 / 1`)

| Split Name | Sequences | Characters | Subword Tokens | Avg Tokens / Sequence | `<unk>` Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
{splits_table}

---

## 📈 Token Length Percentiles

| p10 | p50 (Median) | p90 | p99 | Max Sequence |
| :---: | :---: | :---: | :---: | :---: |
| {dist.get('p10', 0)} | {dist.get('p50', 0)} | {dist.get('p90', 0)} | {dist.get('p99', 0)} | {dist.get('max', 0)} |

---

## 🌐 Script Balance & Writing Systems

* **Meitei Mayek Script**: `{scripts.get('meitei', 0)}%`
* **Bengali Script**: `{scripts.get('bengali', 0)}%`
* **Latin / Romanized Script**: `{scripts.get('latin', 0)}%`
* **Mixed / Multi-script**: `{scripts.get('mixed', 0)}%`

---

## 📚 Source Distribution

| Source / Parallel Dataset | Sequences (% of Total) | Tokens (% of Total) |
| :--- | :--- | :--- |
{sources_table}

---

## 🛠️ Preprocessing & Validation Pipeline (Phase 5.4)

Every sequence in this dataset has undergone:
1. **Unicode Normalization & Cleaning**: NFC normalization, quote standardization, URL/PII/Markdown stripping (`cleaner.py`, `normalizer.py`).
2. **Multi-Stage Deduplication**: Exact SHA256 hashing and MinHash / LSH fuzzy deduplication (`minhash_deduplicator.py`).
3. **Quality & Toxicity Screening**: Heuristic symbol-to-word ratio checks and hate-speech keyword filtration (`quality_scorer.py`).
4. **Subword Tokenization & Validation**: Verified against Tokenizer v1 with `<unk>` rate monitoring (`corpus_validator.py`).
5. **Deterministic Splitting & Sharding**: Split into 98/1/1 and saved as chunked Parquet files with SHA256 checksums (`dataset_assembler.py`).

---

## 🚀 Usage in Hugging Face Datasets

```python
from datasets import load_from_disk, load_dataset

# Load locally from directory
dataset = load_from_disk("artifacts/datasets/ManipuriGPT-Corpus-v1")
print(dataset["train"][0])

# Or load from parquet shards
dataset = load_dataset("parquet", data_files={{
    "train": "train/*.parquet",
    "validation": "validation/*.parquet",
    "test": "test/*.parquet"
}})
```
"""
