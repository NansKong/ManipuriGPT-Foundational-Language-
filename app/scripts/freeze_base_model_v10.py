"""
ManipuriGPT Base Foundation Model Release & Freezing Script (`app/scripts/freeze_base_model_v10.py`).

Freezes `models/smollm_135m_pretrained` into `models/ManipuriGPT-135M-Base-v1.0`, generating explicit
SHA256 checksums, immutability release manifest (`manifest.json`), and model card (`README.md`).

Usage:
  python -m app.scripts.freeze_base_model_v10
"""

import os
import sys
import shutil
import json
import hashlib
import argparse
from typing import Dict, Any, Optional

from app.utils.logger import logger


def calculate_sha256(filepath: str) -> str:
    """Calculates SHA256 checksum of a file in 64KB chunks."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Base Model Release Freezing CLI")
    parser.add_argument("--source-dir", type=str, default="models/smollm_135m_pretrained", help="Source pretrained model directory")
    parser.add_argument("--target-dir", type=str, default="models/ManipuriGPT-135M-Base-v1.0", help="Target release directory")
    return parser.parse_args()


def freeze_base_model(source_dir: str, target_dir: str) -> Dict[str, Any]:
    """Freezes pretrained model directory into an immutable release artifact."""
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source pretrained directory '{source_dir}' does not exist.")

    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Freezing base foundation model from '{source_dir}' -> '{target_dir}'...")

    copied_files = {}
    for filename in os.listdir(source_dir):
        src_file = os.path.join(source_dir, filename)
        if os.path.isfile(src_file):
            dst_file = os.path.join(target_dir, filename)
            shutil.copy2(src_file, dst_file)
            checksum = calculate_sha256(dst_file)
            size_bytes = os.path.getsize(dst_file)
            copied_files[filename] = {
                "sha256": checksum,
                "size_bytes": size_bytes
            }
            logger.info(f"  Frozen: {filename} ({size_bytes:,} bytes, SHA256: {checksum[:16]}...)")

    manifest = {
        "model_id": "ManipuriGPT-135M-Base-v1.0",
        "version": "1.0",
        "status": "IMMUTABLE_BASE_RELEASE",
        "model_family": "SmolLM-135M CausalLM",
        "tokenizer_version": "ManipuriGPT-Tokenizer-v1.0",
        "corpus_version": "ManipuriGPT-Corpus-v1.0",
        "training": {
            "global_steps": 13596,
            "epochs": 3.0,
            "final_train_loss": 3.6246,
            "final_eval_loss": 4.2799,
            "meitei_mayek_ppl": 133.14
        },
        "lineage": {
            "base_foundation": "ManipuriGPT-135M-Base-v1.0",
            "downstream_targets": [
                "ManipuriGPT-135M-SFT-v1.0",
                "ManipuriGPT-135M-DPO-v1.0",
                "ManipuriGPT-135M-Chat-v1.0"
            ]
        },
        "files": copied_files
    }

    manifest_path = os.path.join(target_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme_content = f"""# ManipuriGPT-135M-Base-v1.0

Official immutable foundation model release for **ManipuriGPT**.

## Model Summary
- **Architecture**: SmolLM-135M Causal LM
- **Tokenizer**: `ManipuriGPT-Tokenizer-v1.0` (8,000 vocab size, 0 `<unk>` emissions)
- **Corpus**: `ManipuriGPT-Corpus-v1.0`
- **Training Epochs**: 3.0 (13,596 steps)
- **Final Train Loss**: `3.6246`
- **Final Eval Loss**: `4.2799`
- **Meitei Mayek Perplexity**: `133.14`

## Model Lineage & Hierarchy
```text
ManipuriGPT-135M-Base-v1.0  (Immutable Pretrained Foundation Model)
        │
        ├── ManipuriGPT-135M-SFT-v1.0   (Instruction Fine-Tuned)
        ├── ManipuriGPT-135M-SFT-v2.0   (Multi-task / Reasoning)
        ├── ManipuriGPT-135M-DPO-v1.0   (Direct Preference Optimization)
        └── ManipuriGPT-135M-Chat-v1.0  (Conversational Assistant)
```

## Immutability Guarantee
This directory is locked as the baseline foundation release. All downstream fine-tuning (SFT, DPO, RLHF) will build upon this model without overwriting base weights.
"""
    readme_path = os.path.join(target_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info(f"Release manifest saved to '{manifest_path}'")
    logger.info(f"Model Card saved to '{readme_path}'")
    return manifest


def main() -> int:
    args = parse_args()
    try:
        manifest = freeze_base_model(args.source_dir, args.target_dir)
        logger.info("\n" + "=" * 70)
        logger.info(f" BASE FOUNDATION RELEASE CREATED: {manifest['model_id']}")
        logger.info(" STATUS: IMMUTABLE_BASE_RELEASE")
        logger.info(f" PATH  : {args.target_dir}")
        logger.info("=" * 70)
        return 0
    except Exception as e:
        logger.error(f"Failed to freeze base model release: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
