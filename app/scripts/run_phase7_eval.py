"""
Master Phase 7 Foundation Model Evaluation CLI (`app/scripts/run_phase7_eval.py`).

Usage:
  python -m app.scripts.run_phase7_eval --all
  python -m app.scripts.run_phase7_eval --model_path models/smollm_135m_pretrained --step 7.1
"""

import os
import sys
import yaml
import argparse
from typing import Optional, List, Dict, Any

from app.evaluation.training_analyzer import TrainingAnalyzer
from app.evaluation.perplexity_eval import PerplexityEvaluator
from app.evaluation.token_inspector import TokenInspector
from app.evaluation.generator_eval import GeneratorEvaluator
from app.evaluation.script_eval import ScriptEvaluator
from app.evaluation.memorization_eval import MemorizationEvaluator
from app.evaluation.benchmark_runner import BenchmarkRunner
from app.evaluation.speed_benchmark import SpeedBenchmark
from app.evaluation.tokenizer_eval import TokenizerEvaluator
from app.evaluation.checkpoint_compare import CheckpointComparer
from app.evaluation.human import HumanEvaluationPipeline
from app.evaluation.report_generator import ReportGenerator

from app.utils.logger import logger


def load_config(config_path: str) -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Phase 7 Evaluation CLI")
    parser.add_argument("--config", type=str, default="app/evaluation/evaluation_config.yaml", help="Path to evaluation YAML config")
    parser.add_argument("--model_path", type=str, default="models/smollm_135m_pretrained", help="Path to model checkpoint or Hugging Face ID")
    parser.add_argument("--tokenizer_path", type=str, default="models/smollm_135m_pretrained", help="Path to tokenizer directory or Hugging Face ID")
    parser.add_argument("--logs_path", type=str, default="models/smollm_135m_pretrained/trainer_state.json", help="Path to trainer_state.json")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Directory to save evaluation artifacts")
    parser.add_argument("--all", action="store_true", help="Run all Phase 7 evaluation steps end-to-end")
    parser.add_argument("--step", type=str, choices=["7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10", "tokenizer"], help="Run specific evaluation step")
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)
    cfg = load_config(parsed.config)

    model_path = parsed.model_path or cfg.get("model", {}).get("path", "models/smollm_135m_pretrained")
    tokenizer_path = parsed.tokenizer_path or cfg.get("model", {}).get("tokenizer_path", "models/smollm_135m_pretrained")
    logs_path = parsed.logs_path or cfg.get("model", {}).get("logs_path", "models/smollm_135m_pretrained/trainer_state.json")
    output_dir = parsed.output_dir or cfg.get("evaluation", {}).get("output_dir", "evaluation")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Phase7 CLI: Initializing Evaluation Engine (Model: '{model_path}', Logs: '{logs_path}')")

    run_all = parsed.all or parsed.step is None

    summary_metrics = {
        "model_name": os.path.basename(model_path.rstrip("/\\")),
        "tokenizer_version": "ManipuriGPT-Tokenizer-v1.0",
        "corpus_version": "ManipuriGPT-Corpus-v1.0"
    }

    # Step 7.1 — Training Analysis
    if run_all or parsed.step == "7.1":
        logger.info("--- Step 7.1: Running Training Analysis ---")
        analyzer = TrainingAnalyzer(logs_path=logs_path, output_dir=output_dir)
        analyzer.generate_plots()
        analyzer.generate_report()
        t_metrics = analyzer.extract_metrics()
        summary_metrics["train_loss"] = t_metrics.get("final_train_loss")
        summary_metrics["eval_loss"] = t_metrics.get("final_eval_loss")

    # Load Model & Tokenizer if required for steps 7.2 - 7.8
    needs_model = run_all or parsed.step in ["7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.10", "tokenizer"]
    model, tokenizer, device = None, None, "cpu"

    if needs_model and os.path.exists(model_path):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Phase7 CLI: Loading model and tokenizer on device '{device}'...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
            logger.info("Phase7 CLI: Model loaded successfully.")
        except Exception as e:
            logger.error(f"Phase7 CLI: Failed to load model from '{model_path}': {e}")

    # Tokenizer Diagnostics
    if tokenizer is not None and (run_all or parsed.step == "tokenizer"):
        logger.info("--- Running Tokenizer Diagnostics ---")
        tok_eval = TokenizerEvaluator(tokenizer)
        sample_texts = ["ꯑꯩꯈꯣꯏꯒꯤ ꯃꯅꯤꯄꯨꯔ ꯂꯣꯟ ꯑꯁꯤ ꯌꯥꯝꯅꯥ ꯐꯖꯩ꯫", "মণিপুরী ভাষা একটি সুন্দর ভাষা।"]
        tok_results = tok_eval.evaluate_corpus_tokenization(sample_texts)
        summary_metrics["avg_tokens_per_sent"] = tok_results.get("average_tokens_per_sentence")
        summary_metrics["unk_count"] = tok_results.get("total_unk_tokens", 0)
        summary_metrics["unknown_token_rate"] = tok_results.get("unknown_token_rate", 0.0)
        summary_metrics["compression_ratio"] = tok_results.get("compression_ratio_bytes_per_token")

    # Step 7.2 — Perplexity Evaluation
    if model is not None and tokenizer is not None and (run_all or parsed.step == "7.2"):
        logger.info("--- Step 7.2: Running Perplexity Evaluation ---")
        ppl_eval = PerplexityEvaluator(model, tokenizer, device=device)
        eval_texts = [
            "ꯑꯩꯈꯣꯏꯒꯤ ꯃꯅꯤꯄꯨꯔ ꯂꯣꯟ ꯑꯁꯤ ꯌꯥꯝꯅꯥ ꯐꯖꯩ꯫",
            "ꯃꯅꯤꯄꯨꯔꯒꯤ ꯀꯪꯂꯩꯄꯥꯛ ꯑꯁꯤ ꯅꯤꯡꯊꯧ ꯂꯩꯕꯥꯛꯅꯤ꯫",
            "মণিপুরী ভাষা ভারতের অন্যতম সংবিধান স্বীকৃত ভাষা।",
            "ꯃꯅꯤꯄꯨꯔ (Manipur) ꯍꯥꯏꯕꯁꯤ ꯏꯟꯗꯤꯌꯥꯒꯤ ꯔꯥꯖ꯭ꯌ ꯑꯃꯅꯤ꯫"
        ]
        ppl_res = ppl_eval.evaluate_texts(eval_texts)
        summary_metrics["overall_ppl"] = ppl_res.get("overall_ppl")
        summary_metrics["ppl_qualitative"] = ppl_res.get("qualitative_meaning")
        summary_metrics["meitei_ppl"] = ppl_res.get("script_wise", {}).get("meitei_mayek_ppl")
        summary_metrics["bengali_ppl"] = ppl_res.get("script_wise", {}).get("bengali_script_ppl")
        summary_metrics["mixed_ppl"] = ppl_res.get("script_wise", {}).get("mixed_script_ppl")

    # Step 7.3 — Next Token Prediction
    if model is not None and tokenizer is not None and (run_all or parsed.step == "7.3"):
        logger.info("--- Step 7.3: Running Next Token Prediction Inspection ---")
        inspector = TokenInspector(model, tokenizer, device=device)
        prompts = cfg.get("prompts", {}).get("canonical", ["ꯑꯩ", "অদুগা", "ꯃꯅꯤꯄꯨꯔ"])
        tok_inspection = inspector.inspect_canonical_prompts(prompts)
        logger.info(f"Token Inspection complete ({len(tok_inspection)} prompts inspected).")

    # Step 7.4 & 7.5 — Text Generation & Script Consistency
    if model is not None and tokenizer is not None and (run_all or parsed.step in ["7.4", "7.5"]):
        logger.info("--- Steps 7.4 & 7.5: Running Text Generation & Script Consistency Audit ---")
        gen_eval = GeneratorEvaluator(model, tokenizer, device=device)
        script_auditor = ScriptEvaluator()

        prompts = cfg.get("prompts", {}).get("canonical", ["ꯑꯩ", "অদুগা"])
        decoding_res = gen_eval.evaluate_decoding_strategies(prompts)
        
        # Extract diversity metrics from top_k_50 strategy
        top_k_data = decoding_res.get("top_k_50", {})
        summary_metrics["distinct_1"] = top_k_data.get("distinct_1")
        summary_metrics["distinct_2"] = top_k_data.get("distinct_2")
        summary_metrics["self_bleu"] = top_k_data.get("self_bleu")

        # Script evaluation
        sample_outputs = [s["generated"] for s in top_k_data.get("samples", [])]
        script_audit = script_auditor.evaluate_corpus(sample_outputs)
        summary_metrics["unwanted_script_switch_rate"] = script_audit.get("unwanted_script_switch_rate")
        summary_metrics["invalid_unicode_count"] = script_audit.get("total_invalid_unicodes")

    # Step 7.6 — Memorization Test
    if run_all or parsed.step == "7.6":
        logger.info("--- Step 7.6: Running Memorization Test ---")
        mem_eval = MemorizationEvaluator()
        sample_gen = ["ꯑꯩꯈꯣꯏꯒꯤ ꯃꯅꯤꯄꯨꯔ ꯂꯣꯟ ꯑꯁꯤ ꯌꯥꯝꯅꯥ ꯐꯖꯩ꯫"]
        mem_res = mem_eval.evaluate_memorization_rate(sample_gen)
        summary_metrics["memorization_rate"] = mem_res.get("memorization_rate")

    # Step 7.8 — Speed Benchmark
    if model is not None and tokenizer is not None and (run_all or parsed.step == "7.8"):
        logger.info("--- Step 7.8: Running Inference Speed Benchmark ---")
        speed_bench = SpeedBenchmark()
        prompts = ["ꯑꯩꯈꯣꯏꯒꯤ ꯃꯅꯤꯄꯨꯔ ꯂꯣꯟ", "মণিপুরী ভাষা"]
        speed_res = speed_bench.benchmark_inference(model, tokenizer, prompts, device=device)
        summary_metrics["tokens_per_sec"] = speed_res.get("tokens_per_sec")
        summary_metrics["latency_ms"] = speed_res.get("average_latency_ms")
        summary_metrics["vram_gb"] = speed_res.get("peak_vram_gb")

    # Step 7.9 — Human Review Sheet
    if run_all or parsed.step == "7.9":
        logger.info("--- Step 7.9: Generating Human Review Sheet ---")
        human_pipe = HumanEvaluationPipeline(storage_dir=output_dir)
        sample_items = [
            {"prompt": "ꯑꯩ", "generated": "ꯑꯩꯈꯣꯏꯒꯤ ꯃꯅꯤꯄꯨꯔ ꯂꯣꯟ ꯑꯁꯤ ꯌꯥꯝꯅꯥ ꯐꯖꯩ꯫"},
            {"prompt": "অদুগা", "generated": "অদুগা মণিপুরী ভাষা একটি সুন্দর ভাষা।"}
        ]
        human_pipe.generate_human_review_sheet(sample_items)

    # Step 7.10 — Final Evaluation Report
    if run_all or parsed.step == "7.10":
        logger.info("--- Step 7.10: Generating Final Evaluation Report ---")
        report_gen = ReportGenerator(output_dir=output_dir)
        report_path = report_gen.generate_final_report(summary_metrics)
        logger.info(f"Phase 7 Evaluation complete! Final report written to '{report_path}'.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
