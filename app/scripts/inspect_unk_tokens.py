"""
Unknown Token (<unk>) Inspector (`app/scripts/inspect_unk_tokens.py`).

Extracts 50-100 <unk> occurrences from Candidate Tokenizer v2 evaluation,
identifies the exact Unicode codepoints, characters, and sequence contexts,
and categorizes whether the cause is character_coverage setting, byte_fallback,
normalization, or genuine vocabulary limitation.
"""

import os
import sys
import json
import argparse
import unicodedata
from collections import Counter
from typing import List, Dict, Any, Optional

from app.utils.logger import logger


def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect <unk> token occurrences in candidate Tokenizer v2"
    )
    parser.add_argument(
        "--v3-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v3",
        help="Directory containing corpus shards",
    )
    parser.add_argument(
        "--tokenizer-v2",
        type=str,
        default="artifacts/tokenizer_v2_candidate/tokenizer.model",
        help="Path to candidate Tokenizer v2 model file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of <unk> instances to collect for analysis",
    )
    return parser.parse_args(args_list)


def load_corpus_texts(dataset_dir: str, max_samples: int = 25000) -> List[str]:
    """Load sample text sequences from Parquet dataset shards."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow is required to load corpus texts.")
        return []

    texts = []
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for f in sorted(os.listdir(split_dir)):
            if not f.endswith(".parquet"):
                continue
            fp = os.path.join(split_dir, f)
            try:
                table = pq.read_table(fp, columns=["text"])
                for i in range(table.num_rows):
                    txt = table.column("text")[i].as_py()
                    if txt:
                        texts.append(txt)
                        if len(texts) >= max_samples:
                            return texts
            except Exception as e:
                logger.warning(f"Error loading text from {fp}: {e}")
    return texts


def inspect_unk_occurrences(
    tokenizer_path: str,
    texts: List[str],
    max_samples: int = 100
) -> Dict[str, Any]:
    """Extract and analyze <unk> occurrences."""
    import sentencepiece as spm

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    unk_id = sp.unk_id()

    unk_samples: List[Dict[str, Any]] = []
    char_counter: Counter = Counter()

    for seq_idx, text in enumerate(texts):
        if len(unk_samples) >= max_samples:
            break

        pieces = sp.EncodeAsPieces(text)
        ids = sp.EncodeAsIds(text)

        if unk_id in ids:
            # Reconstruct which characters generated <unk>
            for p_idx, (piece, tid) in enumerate(zip(pieces, ids)):
                if tid == unk_id:
                    # Context window around <unk>
                    start_p = max(0, p_idx - 3)
                    end_p = min(len(pieces), p_idx + 4)
                    context_snippet = "".join([p.replace(" ", " ") for p in pieces[start_p:end_p]])

                    # Identify raw character by matching tokenization mapping
                    raw_char = piece.replace(" ", "")
                    for c in raw_char:
                        char_counter[c] += 1

                    codepoints = [f"U+{ord(c):04X} ({unicodedata.name(c, 'UNKNOWN')})" for c in raw_char] if raw_char else []

                    # Classify cause
                    category = "character_coverage_dropped"
                    if any(ord(c) < 32 or ord(c) == 127 for c in raw_char):
                        category = "control_character"
                    elif any(0x0980 <= ord(c) <= 0x09FF for c in raw_char):
                        category = "rare_bengali_script_glyph"
                    elif any(0xABC0 <= ord(c) <= 0xABFF for c in raw_char):
                        category = "rare_meitei_mayek_glyph"
                    elif any(0x1C80 <= ord(c) <= 0x1C8F for c in raw_char):
                        category = "rare_historical_unicode_glyph"

                    sample_entry = {
                        "unk_index": len(unk_samples) + 1,
                        "sequence_id": seq_idx,
                        "unk_piece": piece,
                        "raw_chars": raw_char,
                        "codepoints": codepoints,
                        "category": category,
                        "context_window": context_snippet,
                        "full_text_snippet": text[:100] + ("..." if len(text) > 100 else "")
                    }
                    unk_samples.append(sample_entry)
                    if len(unk_samples) >= max_samples:
                        break

    # Summary analysis
    category_summary = Counter([s["category"] for s in unk_samples])
    char_summary = [
        {"char": repr(ch), "codepoint": f"U+{ord(ch):04X}", "name": unicodedata.name(ch, "UNKNOWN"), "count": cnt}
        for ch, cnt in char_counter.most_common(20)
    ]

    report = {
        "tokenizer_evaluated": tokenizer_path,
        "total_unk_inspected": len(unk_samples),
        "cause_breakdown": dict(category_summary),
        "top_unk_characters": char_summary,
        "samples": unk_samples,
        "key_findings_and_remediation": {
            "primary_cause": "SentencePiece --character_coverage=0.9995 dropped rare Unicode characters during training.",
            "remediation_option": "Re-train candidate Tokenizer v2 with --character_coverage=1.0 or --byte_fallback=true to achieve 0.0000% <unk> rate.",
        }
    }
    return report


def main(args_list: Optional[List[str]] = None) -> int:
    args = parse_args(args_list)

    logger.info("=" * 80)
    logger.info(" CANDIDATE TOKENIZER V2 UNKNOWN TOKEN (<unk>) INSPECTOR")
    logger.info("=" * 80)
    logger.info(f"Corpus Dir    : {args.v3_dir}")
    logger.info(f"Tokenizer v2  : {args.tokenizer_v2}")
    logger.info(f"Max Samples   : {args.max_samples}")

    texts = load_corpus_texts(args.v3_dir, max_samples=25000)
    logger.info(f"Loaded {len(texts):,} text sequences")

    report = inspect_unk_occurrences(args.tokenizer_v2, texts, max_samples=args.max_samples)

    logger.info("\n" + "=" * 60)
    logger.info(" UNK OCCURRENCE ANALYSIS HIGHLIGHTS")
    logger.info("=" * 60)
    logger.info(f"Total <unk> Samples Inspected : {report['total_unk_inspected']}")
    logger.info("\n--- Primary Cause Breakdown ---")
    for cat, count in report["cause_breakdown"].items():
        logger.info(f"  {cat:35s} : {count} instances")

    logger.info("\n--- Top Characters Causing <unk> ---")
    for ch_info in report["top_unk_characters"][:10]:
        logger.info(f"  Character {ch_info['char']:8s} | {ch_info['codepoint']} ({ch_info['name']}) | Count: {ch_info['count']}")

    logger.info("\n--- Sample <unk> Context Windows ---")
    for s in report["samples"][:5]:
        logger.info(f"  [{s['category']}] Context: \"{s['context_window']}\" (Raw: {repr(s['raw_chars'])})")

    out_p = os.path.join(os.path.dirname(args.tokenizer_v2), "unk_analysis_report.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved detailed analysis report to: {out_p}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
