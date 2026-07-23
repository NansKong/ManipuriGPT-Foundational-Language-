"""
Card Generator (`app/tokenization/card_generator.py`).
Automatically generates publication-ready Hugging Face Tokenizer Cards (`README.md`)
and detailed architectural reports (`model_card.md`) inside each tokenizer candidate directory.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional
from app.utils.logger import logger


def generate_tokenizer_cards(complete_meta: Dict[str, Any], target_dir: str) -> Dict[str, str]:
    """
    Generates `README.md` and `model_card.md` inside `target_dir` using metadata recorded during Phase 5.3+ training.
    """
    os.makedirs(target_dir, exist_ok=True)

    tier = complete_meta.get("version_tier", "v0-experimental")
    algorithm = complete_meta.get("algorithm", "sentencepiece_unigram")
    vocab_size = complete_meta.get("vocab_size", 16384)
    git_commit = complete_meta.get("git_commit", "unknown")
    created_at = complete_meta.get("created_at", datetime.utcnow().isoformat() + "Z")
    
    # Extract training config & summary
    t_meta = complete_meta.get("training_metadata", {})
    t_cfg = t_meta.get("training_config", {})
    eval_summary = complete_meta.get("evaluation_summary", {})
    
    char_coverage = t_cfg.get("character_coverage", 1.0)
    byte_fallback = t_cfg.get("byte_fallback", True)
    split_digits = t_cfg.get("split_digits", True)
    
    # Extract corpus stats
    total_chars = complete_meta.get("token_count", t_meta.get("total_characters_observed", 0))
    total_mb = round(total_chars / (1024 * 1024), 2)
    sample_count = complete_meta.get("sentence_count", t_meta.get("training_samples", 0))
    lang_counts = t_meta.get("language_counts", {})
    
    # Extract stability & metrics
    fertility = eval_summary.get("fertility", "N/A")
    compression = eval_summary.get("compression_ratio", "N/A")
    unk_rate = eval_summary.get("unknown_rate", "N/A")
    round_trip = eval_summary.get("round_trip_accuracy", "N/A")
    entropy = eval_summary.get("entropy", "N/A")
    utilization = round(eval_summary.get("vocab_utilization", 0.0) * 100, 1) if isinstance(eval_summary.get("vocab_utilization"), (int, float)) else "N/A"
    aggl = eval_summary.get("agglutination_fragmentation", "N/A")
    stability_score = eval_summary.get("stability_score", "N/A")
    stability_rating = eval_summary.get("stability_rating", "Not Evaluated")

    # 1. Generate Hugging Face README.md (with YAML frontmatter)
    readme_lines = [
        "---",
        "language:",
        "- mni",
        "- bn",
        "- en",
        "tags:",
        "- tokenizer",
        "- manipurigpt",
        "- sentencepiece",
        "- indic",
        "- meitei-mayek",
        f"- {tier}",
        "license: apache-2.0",
        "---",
        "",
        f"# ManipuriGPT Tokenizer (`{algorithm}` - `{vocab_size}` vocab)",
        "",
        f"This tokenizer is part of the **ManipuriGPT Phase 5.3+** open foundation model project, operating in the **`{tier}`** version tier.",
        "",
        "## Model Specifications",
        "",
        f"- **Algorithm**: `{algorithm}`",
        f"- **Vocabulary Size**: `{vocab_size:,}`",
        f"- **Character Coverage**: `{char_coverage}`",
        f"- **Byte Fallback**: `{byte_fallback}`",
        f"- **Split Digits**: `{split_digits}`",
        f"- **Version Tier**: `{tier}`",
        f"- **Created At**: `{created_at}`",
        f"- **Git Commit**: `{git_commit}`",
        "",
        "## Training Corpus Statistics",
        "",
        f"- **Total Characters**: `{total_chars:,}` (`{total_mb}` MB)",
        f"- **Total Sentences / Samples**: `{sample_count:,}`",
        f"- **Language Distribution**: `{lang_counts}`",
        "",
        "## Evaluation Metrics & Stability",
        "",
        "| Metric | Value | Description |",
        "| :--- | :---: | :--- |",
        f"| **Fertility** | `{fertility}` | Average number of subword tokens produced per word |",
        f"| **Compression Ratio** | `{compression}` | Ratio of raw character length to token length |",
        f"| **Unknown Rate** | `{unk_rate}` | Proportion of out-of-vocabulary fallback tokens (`<unk>`) |",
        f"| **Round-Trip Accuracy** | `{round_trip}` | Exact string reconstruction accuracy after decode(encode(text)) |",
        f"| **Vocabulary Utilization** | `{utilization}%` | Percentage of learned subwords actively emitted during evaluation |",
        f"| **Token Frequency Entropy** | `{entropy}` | Shannon entropy over emitted token probabilities |",
        f"| **Agglutination Fragmentation** | `{aggl}` | Average pieces per word on top 100 longest compound Manipuri words |",
        f"| **Multi-Seed Stability** | `{stability_score}%` (`{stability_rating}`) | Pairwise vocabulary overlap across multi-seed bootstrap training |",
        "",
        "## Intended Use & Multi-Script Architecture",
        "",
        "ManipuriGPT utilizes a non-destructive multi-script data representation separating the user experience from the internal model representation:",
        "",
        "```",
        "User Input (Bengali / Meitei Mayek / English)",
        "                    │",
        "                    ▼",
        "             Input Adapter",
        "                    │",
        "                    ▼",
        "      Canonical / Multi-Script Tokenizer",
        "                    │",
        "                    ▼",
        "       ManipuriGPT Foundation LLM",
        "```",
        "",
        "## Usage (Hugging Face Transformers)",
        "",
        "```python",
        "from transformers import AutoTokenizer",
        "",
        "# Load directly using offset_mapping-compliant fast tokenizer",
        f"tokenizer = AutoTokenizer.from_pretrained('./{tier}/{algorithm}_{vocab_size}')",
        "",
        "text = 'ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯇꯝꯂꯤ — মণিপুরী ভাষা'",
        "tokens = tokenizer.tokenize(text)",
        "print('Tokens:', tokens)",
        "```",
        ""
    ]
    readme_path = os.path.join(target_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))

    # 2. Generate detailed model_card.md
    model_card_lines = [
        f"# ManipuriGPT Tokenizer Architectural Model Card — `{algorithm}_{vocab_size}`",
        "",
        "## Phase 5.3+ Production Refinement Notes",
        "",
        "As established during the research lead review (`9.95/10` specification), Phase 5.3+ enforces:",
        "1. **Non-Destructive Multi-Script Tracking**: Preprocessing preserves original Bengali and Meitei Mayek scripts cleanly while storing `original_text` and `canonical_text` in metadata.",
        "2. **Corpus Size Guard (`50 MB` Pretrain Requirement)**: Tokenizers in tier `v0-experimental` are intended for engineering and verification. Promotion to `v1-pretrain` requires at least 50 MB of clean training data (`v2-expanded` >= 250 MB, `v3-final` >= 500 MB).",
        "3. **Multi-Seed Stability Assessment**: Evaluates vocabulary overlap stability across distinct initialization seeds to guarantee production consistency.",
        "",
        "## Per-Script Performance Breakdown",
        ""
    ]
    
    per_script = eval_summary.get("per_script", {})
    if per_script:
        model_card_lines.extend([
            "| Script | Fertility | Compression Ratio | Unknown Rate | Round-Trip Accuracy | Avg Chars/Token | Agglutination |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"
        ])
        for sname in ["meitei_mayek", "bengali", "latin", "mixed"]:
            sm = per_script.get(sname, {})
            if sm:
                model_card_lines.append(
                    f"| `{sname}` "
                    f"| {sm.get('fertility', '?')} "
                    f"| {sm.get('compression_ratio', '?')} "
                    f"| {sm.get('unknown_rate', '?')} "
                    f"| {sm.get('round_trip_accuracy', '?')} "
                    f"| {sm.get('avg_chars_per_token', '?')} "
                    f"| {sm.get('agglutination_fragmentation', '?')} |"
                )
        model_card_lines.append("")
    else:
        model_card_lines.extend(["*(No per-script evaluation summary recorded)*", ""])

    model_card_path = os.path.join(target_dir, "model_card.md")
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write("\n".join(model_card_lines))

    logger.info(f"CardGenerator: Generated README.md and model_card.md inside '{target_dir}'")
    return {"readme": readme_path, "model_card": model_card_path}
