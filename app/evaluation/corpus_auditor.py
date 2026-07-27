"""
ManipuriGPT Corpus Quality Audit Engine (`app/evaluation/corpus_auditor.py`).

Provides comprehensive quantitative analysis of ManipuriGPT corpus snapshots:
  1. Source Novelty & Retention Rates
  2. Domain/Category Token Balance
  3. Outlier Analysis (Longest/Shortest docs, OCR noise, repeated tokens)
  4. Script Distribution (Meitei Mayek, Bengali script, Romanized, Mixed)
  5. Token & Character Frequency Distributions (Top 1,000)
  6. Unicode Normalization & Cleaning Statistics
"""

import os
import math
import json
import hashlib
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from app.utils.logger import logger


class CorpusAuditor:
    """Engine for performing deep qualitative & quantitative corpus audits."""

    def __init__(self, tokenizer_path: Optional[str] = None):
        self.tokenizer_path = tokenizer_path
        self.sp = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                import sentencepiece as spm
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load(tokenizer_path)
                logger.info(f"CorpusAuditor loaded SentencePiece tokenizer from {tokenizer_path}")
            except Exception as e:
                logger.warning(f"CorpusAuditor could not load tokenizer: {e}")

    def audit_dataset_directory(
        self,
        dataset_dir: str,
        prior_source_stats: Optional[Dict[str, Dict[str, int]]] = None
    ) -> Dict[str, Any]:
        """Audit all Parquet shards inside a dataset directory (train/validation/test)."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for CorpusAuditor.")

        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        all_rows: List[Dict[str, Any]] = []

        for split in ["train", "validation", "test"]:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for f in sorted(os.listdir(split_dir)):
                if not f.endswith(".parquet"):
                    continue
                fp = os.path.join(split_dir, f)
                try:
                    table = pq.read_table(fp)
                    df = table.to_pandas()
                    for record in df.to_dict(orient="records"):
                        record["_split"] = split
                        all_rows.append(record)
                except Exception as e:
                    logger.warning(f"Error reading shard {fp}: {e}")

        logger.info(f"CorpusAuditor: Read {len(all_rows):,} total records from '{dataset_dir}'")
        return self.audit_records(all_rows, prior_source_stats=prior_source_stats)

    def audit_records(
        self,
        records: List[Dict[str, Any]],
        prior_source_stats: Optional[Dict[str, Dict[str, int]]] = None
    ) -> Dict[str, Any]:
        """Perform audit metrics calculation on a list of corpus sequence dictionaries."""
        if not records:
            return {"error": "No records to audit."}

        total_seqs = len(records)
        total_chars = 0
        total_words = 0
        total_tokens = 0
        total_unk_tokens = 0

        source_seq_counts: Counter = Counter()
        category_seq_counts: Counter = Counter()
        category_token_counts: Counter = Counter()
        script_seq_counts: Counter = Counter()
        language_seq_counts: Counter = Counter()

        token_freq: Counter = Counter()
        char_freq: Counter = Counter()

        doc_lengths: List[int] = []
        longest_doc: Optional[Dict[str, Any]] = None
        shortest_doc: Optional[Dict[str, Any]] = None

        ocr_noise_outliers: List[Dict[str, Any]] = []
        repeated_token_outliers: List[Dict[str, Any]] = []

        vocab_seen: set = set()
        vocab_growth_by_source: Dict[str, int] = {}

        for idx, rec in enumerate(records):
            text = str(rec.get("text", "")).strip()
            if not text:
                continue

            c_len = len(text)
            words = text.split()
            w_len = len(words)
            doc_lengths.append(c_len)

            total_chars += c_len
            total_words += w_len

            # Character frequency
            for char in text:
                char_freq[char] += 1

            # Tokenization (SentencePiece if available, else whitespace word tokens)
            if self.sp is not None:
                token_ids = self.sp.Encode(text)
                t_len = len(token_ids)
                total_tokens += t_len
                unk_id = self.sp.unk_id()
                for tid in token_ids:
                    token_str = self.sp.IdToPiece(tid)
                    token_freq[token_str] += 1
                    if tid == unk_id:
                        total_unk_tokens += 1
                    vocab_seen.add(tid)
            else:
                t_len = w_len
                total_tokens += w_len
                for w in words:
                    token_freq[w] += 1
                    vocab_seen.add(w)

            src = str(rec.get("source", "unknown"))
            cat = str(rec.get("category", "unknown"))
            scr = str(rec.get("script", "unknown"))
            lang = str(rec.get("language", "unknown"))

            source_seq_counts[src] += 1
            category_seq_counts[cat] += 1
            category_token_counts[cat] += t_len
            script_seq_counts[scr] += 1
            language_seq_counts[lang] += 1

            vocab_growth_by_source[src] = len(vocab_seen)

            doc_info = {
                "sequence_index": idx,
                "text_snippet": text[:120] + ("..." if len(text) > 120 else ""),
                "char_length": c_len,
                "word_length": w_len,
                "token_length": t_len,
                "source": src,
                "category": cat,
                "script": scr,
                "document_id": str(rec.get("document_id", ""))
            }

            if longest_doc is None or c_len > longest_doc["char_length"]:
                longest_doc = doc_info
            if shortest_doc is None or c_len < shortest_doc["char_length"]:
                shortest_doc = doc_info

            # Outlier Detection: High Latin density in OCR/Non-Latin document
            if scr in ["meitei", "bengali"]:
                latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
                latin_density = latin_chars / max(c_len, 1)
                if latin_density > 0.35:
                    ocr_noise_outliers.append({
                        **doc_info,
                        "latin_density_pct": round(latin_density * 100, 2)
                    })

            # Outlier Detection: Repeated tokens (degenerate strings)
            if words:
                top_word_count = Counter(words).most_common(1)[0][1]
                if top_word_count / max(w_len, 1) > 0.60 and w_len > 15:
                    repeated_token_outliers.append({
                        **doc_info,
                        "repeated_word_ratio_pct": round(top_word_count / w_len * 100, 2)
                    })

        avg_c_len = total_chars / max(total_seqs, 1)
        avg_w_len = total_words / max(total_seqs, 1)
        avg_t_len = total_tokens / max(total_seqs, 1)
        unk_rate_pct = (total_unk_tokens / max(total_tokens, 1)) * 100.0

        # Source Novelty Table Computation
        source_novelty_report = []
        if prior_source_stats:
            for src_name, stats in prior_source_stats.items():
                raw = stats.get("raw", 0)
                kept = stats.get("kept", 0)
                dups = stats.get("duplicates", 0)
                novelty_pct = (kept / max(raw, 1)) * 100.0 if raw > 0 else 0.0
                source_novelty_report.append({
                    "source": src_name,
                    "raw_scanned": raw,
                    "unique_kept": kept,
                    "duplicates": dups,
                    "novelty_pct": round(novelty_pct, 2)
                })
        else:
            for src_name, kept in source_seq_counts.items():
                source_novelty_report.append({
                    "source": src_name,
                    "unique_kept": kept
                })

        # Top 1,000 Tokens & Characters
        top_1000_tokens = [
            {"token": tok, "frequency": count, "pct": round(count / max(total_tokens, 1) * 100, 4)}
            for tok, count in token_freq.most_common(1000)
        ]
        top_1000_chars = [
            {"char": repr(ch), "frequency": count, "pct": round(count / max(total_chars, 1) * 100, 4)}
            for ch, count in char_freq.most_common(1000)
        ]

        report = {
            "summary": {
                "total_sequences": total_seqs,
                "total_characters": total_chars,
                "total_words": total_words,
                "total_tokens": total_tokens,
                "total_unk_tokens": total_unk_tokens,
                "unknown_token_rate_pct": round(unk_rate_pct, 4),
                "avg_doc_length_chars": round(avg_c_len, 2),
                "avg_doc_length_words": round(avg_w_len, 2),
                "avg_doc_length_tokens": round(avg_t_len, 2),
                "unique_vocab_size_observed": len(vocab_seen),
            },
            "source_novelty": source_novelty_report,
            "domain_balance": {
                cat: {
                    "sequences": category_seq_counts[cat],
                    "tokens": category_token_counts[cat],
                    "seq_pct": round(category_seq_counts[cat] / max(total_seqs, 1) * 100, 2),
                    "token_pct": round(category_token_counts[cat] / max(total_tokens, 1) * 100, 2),
                }
                for cat in category_seq_counts
            },
            "script_distribution": {
                scr: {
                    "count": count,
                    "pct": round(count / max(total_seqs, 1) * 100, 2)
                }
                for scr, count in script_seq_counts.most_common()
            },
            "language_distribution": dict(language_seq_counts.most_common()),
            "outliers_and_long_docs": {
                "longest_document": longest_doc,
                "shortest_document": shortest_doc,
                "ocr_noise_outliers_count": len(ocr_noise_outliers),
                "ocr_noise_outliers_sample": ocr_noise_outliers[:10],
                "repeated_token_outliers_count": len(repeated_token_outliers),
                "repeated_token_outliers_sample": repeated_token_outliers[:10],
            },
            "vocabulary_growth_by_source": vocab_growth_by_source,
            "top_1000_tokens": top_1000_tokens,
            "top_1000_chars": top_1000_chars,
        }

        return report
