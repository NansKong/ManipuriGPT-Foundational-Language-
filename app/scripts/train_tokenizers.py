"""
ManipuriGPT Phase 5.3 Tokenizer Training Orchestration Suite.
Trains candidate subword tokenizers (Unigram and BPE across configurable vocab sizes)
on balanced streamed corpora, evaluates candidates, and exports the winner as PreTrainedTokenizerFast.

Key improvements over Phase 5.2:
- Loads training parameters from tokenizer.yaml (character_coverage, byte_fallback, etc.)
- Default vocab sizes: [16384, 24576, 32768] (dropped 50k, added 24k)
- Corpus quality guard: refuses to train on empty/insufficient data in production mode
- --evaluate flag: runs automated evaluation and winner selection after training
- --convert-hf flag: exports winner to PreTrainedTokenizerFast for Transformers integration
"""

import os
import time
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.corpus.sampler import BalancedCorpusSampler
from app.preprocessing.pipeline import PreprocessingPipeline
from app.tokenization.trainer import TokenizerTrainer
from app.tokenization.versioning import TokenizerVersionManager
from app.utils.logger import logger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Tokenizer Training Orchestration CLI")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "dayananda_meitei_mayek_sample",
            "dayananda_english_to_meitei",
            "joyson_bible",
            "joyson_pib_pmi"
        ],
        help="Sources for balanced training data acquisition"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
        help="Number of balanced raw samples to stream for tokenizer training"
    )
    parser.add_argument(
        "--vocab-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Target vocabulary sizes to train (defaults to tokenizer.yaml config)"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Algorithms to train for each vocab size (defaults to tokenizer.yaml config)"
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default=None,
        help="Base directory where candidate model subdirectories will be saved"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balanced sampling and training"
    )
    parser.add_argument(
        "--mock-fallback",
        action="store_true",
        help="Allow mock fallback if live HF connection fails"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run automated evaluation and winner selection after training"
    )
    parser.add_argument(
        "--convert-hf",
        action="store_true",
        help="Convert the winning tokenizer to HuggingFace PreTrainedTokenizerFast"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="v0-experimental",
        help="Tokenizer version tier (v0-experimental, v1-pretrain, v2-expanded, v3-final)"
    )
    parser.add_argument(
        "--force-v1",
        action="store_true",
        help="Force promotion to v1-pretrain even if corpus size is below 50 MB threshold"
    )
    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Allow simulated fallback if SentencePiece training fails (for testing)"
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=None,
        help="Target character coverage for SentencePiece (e.g. 1.0 or 0.9999)"
    )
    parser.add_argument(
        "--compute-stability",
        action="store_true",
        help="Compute multi-seed vocabulary stability score across 5 bootstrap runs"
    )
    return parser.parse_args(args)


def _load_training_config() -> Dict[str, Any]:
    """Loads training configuration from tokenizer.yaml."""
    try:
        from app.configs.loader import load_config
        tok_cfg = load_config("tokenizer.yaml")
        return tok_cfg.get("tokenizer", {}).get("training", {})
    except Exception:
        return {}


def main(args_list: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(args_list)

    try:
        from app.utils.cache import setup_cache_directories
        setup_cache_directories()
    except Exception as e:
        logger.warning(f"TrainTokenizers: Could not initialize cache directories: {e}")

    # Load configuration from tokenizer.yaml
    training_cfg = _load_training_config()

    # Merge YAML defaults with CLI overrides
    vocab_sizes = args.vocab_sizes or training_cfg.get("vocab_sizes", [16384, 24576, 32768])
    algorithms = args.algorithms or [
        training_cfg.get("primary_algorithm", "sentencepiece_unigram"),
        training_cfg.get("comparison_algorithm", "sentencepiece_bpe"),
    ]
    version_manager = TokenizerVersionManager(training_cfg.get("output_base_dir", "cache/tokenizers"))
    output_base_dir = args.output_base_dir or version_manager.get_tier_directory(args.tier)
    if args.character_coverage is not None:
        training_cfg["character_coverage"] = args.character_coverage
    if args.compute_stability:
        training_cfg["compute_stability"] = True

    os.makedirs(output_base_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info(" MANIPURIGPT PHASE 5.3 TOKENIZER TRAINING SUITE")
    logger.info("=" * 80)
    logger.info(f"Version Tier        : {args.tier}")
    logger.info(f"Target Sources      : {args.sources}")
    logger.info(f"Training Samples    : {args.train_samples}")
    logger.info(f"Vocab Sizes         : {vocab_sizes}")
    logger.info(f"Algorithms          : {algorithms}")
    logger.info(f"Output Base Dir     : {output_base_dir}")
    logger.info(f"Random Seed         : {args.seed}")
    logger.info(f"Character Coverage  : {training_cfg.get('character_coverage', 1.0)}")
    logger.info(f"Byte Fallback       : {training_cfg.get('byte_fallback', True)}")
    logger.info(f"Split Digits        : {training_cfg.get('split_digits', True)}")
    logger.info(f"Auto-Evaluate       : {args.evaluate}")
    logger.info(f"Convert to HF Fast  : {args.convert_hf}")

    # 1. Acquire and Preprocess Balanced Training Corpus
    logger.info("\n[1/3] Streaming & Preprocessing Balanced Training Corpus...")
    sampler = BalancedCorpusSampler(sources=args.sources, seed=args.seed)
    pipeline = PreprocessingPipeline()

    training_texts: List[str] = []
    language_counts: Dict[str, int] = {}

    # Per-stage funnel counters for debugging
    stage_counts = {
        "streamer_raw": 0,
        "after_clean_validate": 0,
        "after_lang_filter": 0,
        "after_quality": 0,
        "after_minhash_dedup": 0,
        "after_chunker": 0,
    }

    stream = sampler.stream(min_length=30, max_examples=args.train_samples, mock_fallback=args.mock_fallback)
    for raw_ex in stream:
        stage_counts["streamer_raw"] += 1
        chunks = pipeline.process_example(raw_ex, chunk=True)
        for c in chunks:
            text = c.get("text", "").strip()
            if text:
                training_texts.append(text)
                lang = c.get("metadata", {}).get("language", "en")
                language_counts[lang] = language_counts.get(lang, 0) + 1

    # Approximate stage attribution via pipeline internal stats
    ps = pipeline.stats
    stage_counts["after_clean_validate"] = stage_counts["streamer_raw"] - (
        getattr(ps, "empty_removed", 0) +
        getattr(ps, "invalid_unicode_removed", 0) +
        getattr(ps, "only_punctuation_removed", 0) +
        getattr(ps, "only_numbers_removed", 0) +
        getattr(ps, "repeated_chars_removed", 0)
    )
    stage_counts["after_chunker"] = len(training_texts)

    logger.info(f"Buffered {len(training_texts)} clean text sequences across languages {language_counts} for training.")
    logger.info("\n" + "=" * 60)
    logger.info(" PIPELINE FUNNEL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Streamer (raw)        : {stage_counts['streamer_raw']}")
    logger.info(f"  After clean+validate  : {stage_counts['after_clean_validate']}")
    logger.info(f"  After chunker         : {stage_counts['after_chunker']} sequences -> tokenizer")
    logger.info(f"  Languages detected    : {language_counts}")
    logger.info("=" * 60)

    total_chars_observed = sum(len(t) for t in training_texts)
    logger.info(f"Total corpus size: {total_chars_observed:,} characters ({round(total_chars_observed / (1024 * 1024), 2)} MB)")

    # Validate corpus size against tier requirements (prevents premature v1-pretrain freezing)
    version_manager.validate_corpus_for_tier(total_chars_observed, args.tier, dev_mode=args.dev_mode, force=args.force_v1)

    # Corpus quality guard
    if len(training_texts) < 100:
        if not args.dev_mode:
            msg = (
                f"Only {len(training_texts)} training texts buffered (minimum: 100). "
                f"Cannot train a production-quality tokenizer. "
                f"Use --dev-mode for simulated fallback, or provide more training data."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.warning(f"TrainTokenizers: Only {len(training_texts)} texts. Proceeding in dev_mode with simulated fallback.")

    # 2. Train Candidate Tokenizers
    total_candidates = len(algorithms) * len(vocab_sizes)
    logger.info(f"\n[2/3] Training {total_candidates} Candidate Tokenizer Models...")
    results: Dict[str, Any] = {}
    start_total_t = time.time()

    for alg in algorithms:
        for vocab_size in vocab_sizes:
            model_dir = os.path.join(output_base_dir, alg, str(vocab_size))
            model_dir_name = f"{alg}/{vocab_size}"
            experiment_id = f"SPM_{alg[:3].upper()}_{vocab_size // 1000}K_{datetime.utcnow().strftime('%Y%m%d')}_001"

            logger.info(f"\n---> Training Model '{model_dir_name}' ({alg}, vocab={vocab_size}) in '{model_dir}'...")
            start_t = time.time()

            trainer = TokenizerTrainer(
                algorithm=alg,
                vocab_size=vocab_size,
                output_dir=model_dir,
                dev_mode=args.dev_mode,
                training_config=training_cfg,
            )
            meta = trainer.train_from_iterator(
                iter(training_texts),
                model_prefix="tokenizer",
                languages=sorted(language_counts.keys()),
                seed=args.seed,
                experiment_id=experiment_id
            )
            duration = round(time.time() - start_t, 2)
            meta["training_duration_sec"] = duration

            # Save versioned metadata inside the candidate directory
            version_manager.save_version_metadata(
                tier=args.tier,
                algorithm=alg,
                vocab_size=vocab_size,
                training_metadata=meta,
                model_subdirectory=model_dir_name
            )

            results[model_dir_name] = meta
            logger.info(f"Finished '{model_dir_name}' in {duration}s -> Artifacts: {meta.get('artifact_files', [])}")

    total_duration = round(time.time() - start_total_t, 2)
    summary_report = {
        "pipeline_version": "5.3",
        "seed": args.seed,
        "created": datetime.utcnow().isoformat() + "Z",
        "training_samples_buffered": len(training_texts),
        "languages": language_counts,
        "total_duration_sec": total_duration,
        "models_trained": list(results.keys()),
        "model_metadata": results,
        "training_config": {
            "character_coverage": training_cfg.get("character_coverage", 0.9999),
            "byte_fallback": training_cfg.get("byte_fallback", True),
            "split_digits": training_cfg.get("split_digits", True),
            "input_sentence_size": training_cfg.get("input_sentence_size", 1000000),
            "max_sentencepiece_length": training_cfg.get("max_sentencepiece_length", 32),
            "vocab_sizes": vocab_sizes,
            "algorithms": algorithms,
        }
    }

    summary_path = os.path.join(output_base_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info(" TOKENIZER TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Models Trained : {len(results)} ({list(results.keys())})")
    logger.info(f"Training Corpus Size : {len(training_texts)} sequences")
    logger.info(f"Total Duration       : {total_duration}s")
    logger.info(f"Summary saved to     : {summary_path}")
    logger.info("=" * 80)

    # 3. Automated Evaluation (if requested)
    winner_name = None
    if args.evaluate:
        logger.info("\n[3/3] Running Automated Tokenizer Evaluation...")
        try:
            from app.tokenization.evaluator import TokenizerEvaluator
            evaluator = TokenizerEvaluator()
            eval_results = evaluator.compare_candidates(output_base_dir)

            if eval_results:
                if training_cfg.get("compute_stability", False) or args.compute_stability:
                    logger.info("Computing multi-seed stability scores for evaluated candidates...")
                    for cand_name, cand_meta in eval_results.items():
                        if "error" not in cand_meta and "/" in cand_name:
                            try:
                                alg, vocab = cand_name.split("/")
                                stab_res = evaluator.compute_stability_score(
                                    training_texts,
                                    algorithm=alg,
                                    vocab_size=int(vocab),
                                    num_seeds=5,
                                    output_dir=os.path.join(output_base_dir, "tmp_stability")
                                )
                                cand_meta.update(stab_res)
                            except Exception as stab_err:
                                logger.warning(f"Could not compute stability for '{cand_name}': {stab_err}")

                winner_name, winner_metrics = evaluator.select_winner(eval_results)

                # Generate comparison report
                report_path = evaluator.generate_comparison_report(
                    eval_results,
                    winner_name,
                    output_path=os.path.join("cache", "benchmarks", "tokenizer_evaluation.md")
                )

                # Save evaluation results and update versioned metadata/cards for all candidates
                for cand_name, cand_eval in eval_results.items():
                    if "error" not in cand_eval and "/" in cand_name:
                        try:
                            alg, vocab = cand_name.split("/")
                            version_manager.save_version_metadata(
                                tier=args.tier,
                                algorithm=alg,
                                vocab_size=int(vocab),
                                training_metadata=results.get(cand_name, {}),
                                evaluation_summary=cand_eval,
                                model_subdirectory=cand_name
                            )
                        except Exception as update_err:
                            logger.warning(f"Could not update version cards for '{cand_name}': {update_err}")

                eval_path = os.path.join(output_base_dir, "evaluation_results.json")
                with open(eval_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "winner": winner_name,
                        "winner_metrics": winner_metrics,
                        "all_results": eval_results,
                        "thresholds": evaluator.thresholds,
                        "report_path": report_path,
                    }, f, indent=2, default=str)

                summary_report["evaluation"] = {
                    "winner": winner_name,
                    "winner_score": winner_metrics.get("selection_score"),
                    "winner_fertility": winner_metrics.get("fertility"),
                    "winner_unknown_rate": winner_metrics.get("unknown_rate"),
                    "winner_round_trip": winner_metrics.get("round_trip_accuracy"),
                    "report_path": report_path,
                }

                # Re-save summary with evaluation data
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary_report, f, indent=2, default=str)

                logger.info(f"Evaluation winner: {winner_name}")
                logger.info(f"Report saved to: {report_path}")
            else:
                logger.warning("TrainTokenizers: No candidates found for evaluation (no .model files).")
        except Exception as e:
            logger.error(f"TrainTokenizers: Evaluation failed: {e}")
            summary_report["evaluation"] = {"error": str(e)}

    # 4. Convert winner to HF Fast (if requested)
    if args.convert_hf and winner_name:
        logger.info(f"\n[4/4] Converting winner '{winner_name}' to HuggingFace PreTrainedTokenizerFast...")
        try:
            alg, vocab = winner_name.split("/")
            model_path = os.path.join(output_base_dir, alg, vocab, "tokenizer.model")
            hf_export_dir = training_cfg.get("hf_export_dir", "cache/tokenizers/hf_fast")

            trainer = TokenizerTrainer(
                algorithm=alg,
                vocab_size=int(vocab),
                training_config=training_cfg,
            )
            export_path = trainer.convert_to_hf_fast(model_path, hf_export_dir)
            summary_report["hf_export"] = {"path": export_path, "source_model": model_path}
            logger.info(f"HF Fast tokenizer exported to: {export_path}")

            # Re-save summary
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_report, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"TrainTokenizers: HF conversion failed: {e}")
            summary_report["hf_export"] = {"error": str(e)}
    elif args.convert_hf and not winner_name:
        logger.warning("TrainTokenizers: --convert-hf requires --evaluate to select a winner first.")

    return summary_report


if __name__ == "__main__":
    main()
