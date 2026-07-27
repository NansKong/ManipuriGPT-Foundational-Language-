"""
ManipuriGPT Candidate Tokenizer v2 Trainer & Evaluator CLI (`app/scripts/train_eval_tokenizer_v2.py`).

Trains candidate Tokenizer v2 on the expanded corpus and objectively evaluates it
against Tokenizer v1 across tokens/seq, compression ratio, unknown rate, vocabulary
coverage, vocabulary utilization %, and token entropy.

Usage:
  python -m app.scripts.train_eval_tokenizer_v2 --v3-dir artifacts/datasets/ManipuriGPT-Corpus-v3
"""

import os
import sys
import json
import argparse
from typing import List, Optional

from app.utils.logger import logger
from app.tokenization.compare_tokenizers import TokenizerComparator


def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ManipuriGPT Candidate Tokenizer v2 Trainer & Evaluator"
    )
    parser.add_argument(
        "--v3-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v3",
        help="Directory containing Phase 5.5 / v3 corpus shards",
    )
    parser.add_argument(
        "--v1-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v1",
        help="Directory containing v1 corpus shards",
    )
    parser.add_argument(
        "--tokenizer-v1",
        type=str,
        default="artifacts/tokenizer_v1/tokenizer.model",
        help="Path to frozen Tokenizer v1 model file",
    )
    parser.add_argument(
        "--tokenizer-v2-dir",
        type=str,
        default="artifacts/tokenizer_v2_candidate",
        help="Output directory for candidate Tokenizer v2",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target vocabulary size for candidate Tokenizer v2",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip candidate training if Tokenizer v2 model already exists",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="Character coverage ratio for SentencePiece (use 1.0 to eliminate <unk> tokens)",
    )
    parser.add_argument(
        "--byte-fallback",
        action="store_true",
        default=True,
        help="Enable byte fallback for out-of-vocabulary characters",
    )
    return parser.parse_args(args_list)


from app.preprocessing.cleaner import TextCleaner


def load_corpus_texts(dataset_dir: str, max_samples: int = 20000) -> List[str]:
    """Load sample text sequences from Parquet dataset shards and sanitize using TextCleaner."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow is required to load corpus texts.")
        return []

    cleaner = TextCleaner({"remove_foreign_scripts": True, "normalize_whitespace": True, "remove_control_chars": True})
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
                        cleaned_txt = cleaner.clean(str(txt))
                        if cleaned_txt:
                            texts.append(cleaned_txt)
                            if len(texts) >= max_samples:
                                logger.info(f"TextCleaner Removal Summary during corpus text loading: {cleaner.get_removal_summary()}")
                                return texts
            except Exception as e:
                logger.warning(f"Error loading text from {fp}: {e}")

    logger.info(f"TextCleaner Removal Summary during corpus text loading: {cleaner.get_removal_summary()}")
    return texts


def train_candidate_tokenizer_v2(
    texts: List[str],
    output_dir: str,
    vocab_size: int = 32000,
    character_coverage: float = 1.0,
    byte_fallback: bool = True
) -> str:
    """Train SentencePiece BPE tokenizer on text sequences."""
    import sentencepiece as spm
    os.makedirs(output_dir, exist_ok=True)
    temp_txt_path = os.path.join(output_dir, "training_input.txt")

    logger.info(f"Writing {len(texts):,} text samples to temporary file for Tokenizer v2 training...")
    import re
    with open(temp_txt_path, "w", encoding="utf-8") as f:
        for t in texts:
            cleaned = re.sub(r"\s+", " ", str(t)).strip()
            if cleaned:
                f.write(cleaned + "\n")

    model_prefix = os.path.join(output_dir, "tokenizer")
    spm_args = (
        f"--input={temp_txt_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage={character_coverage} "
        f"--byte_fallback={'true' if byte_fallback else 'false'} "
        f"--model_type=bpe "
        f"--pad_id=0 "
        f"--unk_id=1 "
        f"--bos_id=2 "
        f"--eos_id=3 "
        f"--user_defined_symbols=<meitei>,<bengali>,<romanized>,<mask "
        f"--normalization_rule_name=identity"
    )

    logger.info(f"Training SentencePiece candidate Tokenizer v2 (vocab_size={vocab_size}, coverage={character_coverage}, byte_fallback={byte_fallback})...")
    spm.SentencePieceTrainer.Train(spm_args)

    model_path = f"{model_prefix}.model"
    if os.path.exists(temp_txt_path):
        os.remove(temp_txt_path)

    logger.info(f"Candidate Tokenizer v2 trained successfully: {model_path}")
    return model_path


def main(args_list: Optional[List[str]] = None) -> int:
    args = parse_args(args_list)

    logger.info("=" * 80)
    logger.info(" MANIPURIGPT TOKENIZER V1 VS CANDIDATE TOKENIZER V2 EVALUATOR")
    logger.info("=" * 80)

    target_dir = args.v3_dir
    if not os.path.exists(target_dir):
        target_dir = args.v1_dir

    if not os.path.exists(target_dir):
        logger.error(f"No corpus directory found at '{args.v3_dir}' or '{args.v1_dir}'. Aborting.")
        return 1

    texts = load_corpus_texts(target_dir, max_samples=25000)
    logger.info(f"Loaded {len(texts):,} evaluation text sequences from '{target_dir}'")

    if not texts:
        logger.error("0 evaluation text sequences found! Aborting.")
        return 1

    v1_model = args.tokenizer_v1
    if not os.path.exists(v1_model):
        logger.error(f"Tokenizer v1 model missing at '{v1_model}'. Aborting.")
        return 1

    v2_model = os.path.join(args.tokenizer_v2_dir, "tokenizer.model")
    if not args.skip_train or not os.path.exists(v2_model):
        try:
            v2_model = train_candidate_tokenizer_v2(
                texts,
                args.tokenizer_v2_dir,
                vocab_size=args.vocab_size,
                character_coverage=args.character_coverage,
                byte_fallback=args.byte_fallback
            )
        except Exception as e:
            logger.warning(f"Could not train Candidate Tokenizer v2: {e}. Falling back to evaluating Tokenizer v1.")
            v2_model = v1_model

    comparator = TokenizerComparator()
    results = comparator.compare(v1_model, v2_model, texts)

    v1 = results["tokenizer_v1"]
    v2 = results["tokenizer_v2"]

    logger.info("\n" + "=" * 80)
    logger.info(" TOKENIZER EVALUATION & COMPARISON MATRIX")
    logger.info("=" * 80)
    logger.info(f"{'Metric':35s} | {'Tokenizer v1':20s} | {'Tokenizer v2 (Cand)':20s}")
    logger.info("-" * 80)
    logger.info(f"{'Vocabulary Size':35s} | {v1['vocab_size']:<20,} | {v2['vocab_size']:<20,}")
    logger.info(f"{'Used Vocabulary Size':35s} | {v1['used_vocab_size']:<20,} | {v2['used_vocab_size']:<20,}")
    logger.info(f"{'Vocabulary Utilization (%)':35s} | {v1['vocab_utilization_pct']:<20.2f} | {v2['vocab_utilization_pct']:<20.2f}")
    logger.info(f"{'Avg Tokens / Sequence':35s} | {v1['avg_tokens_per_sequence']:<20.2f} | {v2['avg_tokens_per_sequence']:<20.2f}")
    logger.info(f"{'Compression (Bytes / Token)':35s} | {v1['compression_ratio_bytes_per_token']:<20.3f} | {v2['compression_ratio_bytes_per_token']:<20.3f}")
    logger.info(f"{'Compression (Chars / Token)':35s} | {v1['compression_ratio_chars_per_token']:<20.3f} | {v2['compression_ratio_chars_per_token']:<20.3f}")
    logger.info(f"{'Unknown Token Rate (`<unk>%`)':35s} | {v1['unknown_rate_pct']:<20.4f} | {v2['unknown_rate_pct']:<20.4f}")
    logger.info(f"{'Token Entropy H (bits)':35s} | {v1['token_entropy_bits']:<20.4f} | {v2['token_entropy_bits']:<20.4f}")
    logger.info("-" * 80)
    logger.info(f" RECOMMENDATION : {results['recommendation']}")
    logger.info(f" REASON         : {results['decision_reason']}")
    logger.info("=" * 80)

    # Save comparison report
    output_path = os.path.join(args.tokenizer_v2_dir, "tokenizer_comparison_report.json")
    os.makedirs(args.tokenizer_v2_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved tokenizer comparison report to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
