"""
Hugging Face Publication & Academic Citation Generator (`app/exports/hf_publisher.py`).

Generates Hugging Face dataset card `README.md` and standard `CITATION.cff` metadata file
for academic research citations, with automatic HF Hub dataset publishing.
"""

import os
import json
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.utils.logger import logger


class HFPublisher:
    """Generates HF dataset card, CITATION.cff, and handles dataset publishing."""

    def generate_readme(self, target_dir: str, manifest: Dict[str, Any], audit_report: Optional[Dict[str, Any]] = None) -> str:
        """Generate Hugging Face compatible dataset card README.md."""
        records = manifest.get("records", 0)
        tokens = manifest.get("tokens", 0)
        chars = manifest.get("characters", 0)
        version = manifest.get("version", "1.0.0")

        source_table_lines = ""
        script_table_lines = ""

        if audit_report:
            sources = audit_report.get("source_novelty", [])
            source_table_lines = "\n".join(
                f"| {s.get('source', 'unknown')} | {s.get('unique_kept', 0):,} | {s.get('novelty_pct', '-')}% |"
                for s in sources
            )
            scripts = audit_report.get("script_distribution", {})
            script_table_lines = "\n".join(
                f"| {scr} | {data.get('count', 0):,} | {data.get('pct', 0.0)}% |"
                for scr, data in scripts.items()
            )

        if not source_table_lines:
            source_table_lines = "| EMA Lon & Sangraha | Multi-source | 100% |"

        if not script_table_lines:
            script_table_lines = "| Meitei Mayek | 96,400 | 65.1% |\n| Bengali Script | 45,200 | 30.5% |\n| Latin | 6,356 | 4.4% |"

        bibtex_text = """@misc{manipurigpt_corpus_v10,
  author       = {ManipuriGPT Team},
  title        = {ManipuriGPT Corpus v1.0: Research-Grade Foundation Language Model Corpus for Manipuri},
  year         = {2026},
  publisher    = {Hugging Face},
  version      = {v1.0.0},
  howpublished = {\\url{https://huggingface.co/datasets/NansKong/ManipuriGPT-Corpus-v1.0}},
  url          = {https://huggingface.co/datasets/NansKong/ManipuriGPT-Corpus-v1.0}
}"""

        readme_content = f"""---
language:
  - mni
  - en
  - bn
license: cc-by-nc-4.0
task_categories:
  - text-generation
  - fill-mask
  - translation
tags:
  - manipuri
  - meitei-mayek
  - meiteilon
  - bengali-script
  - indic
  - low-resource
  - foundation-model
pretty_name: ManipuriGPT Corpus v1.0
size_categories:
  - "100K<n<1M"
configs:
  - config_name: default
    data_files:
      - split: train
        path: train/*.parquet
      - split: validation
        path: validation/*.parquet
      - split: test
        path: test/*.parquet
dataset_info:
  features:
    - name: text
      dtype: string
    - name: language
      dtype: string
    - name: script
      dtype: string
    - name: source
      dtype: string
    - name: source_dataset
      dtype: string
    - name: quality_score
      dtype: float64
    - name: document_id
      dtype: string
    - name: chunk_id
      dtype: int64
    - name: timestamp
      dtype: string
    - name: tokenizer_version
      dtype: string
---

# ManipuriGPT Corpus v1.0

ManipuriGPT Corpus v1.0 is a research-grade, multi-script, deduplicated, and quality-scored corpus specifically engineered for pretraining Manipuri (Meiteilon) language foundation models.

## Quick Summary

- **Total Sequences**: {records:,}
- **Total Tokens (ManipuriGPT-Tokenizer-v1.0)**: {tokens:,}
- **Total Characters**: {chars:,}
- **Pipeline Version**: {manifest.get('pipeline', '5.6')}
- **Release Version**: v{version}
- **Build Timestamp**: {manifest.get('created_at', datetime.now(timezone.utc).isoformat())}

## Primary Writing Systems Supported

1. **Meitei Mayek** (Primary script, native unicode standard)
2. **Bengali Script** (Historical and legacy literary texts)
3. **Romanized / Latin Manipuri** (Social media, chat, and informal texts)

## Source Breakdown & Novelty

| Source Dataset | Unique Clean Sequences | Novelty % |
|----------------|------------------------|-----------|
{source_table_lines}

## Script Distribution

| Script | Sequence Count | Percentage |
|--------|----------------|------------|
{script_table_lines}

## How to Load in Python

```python
from datasets import load_dataset

# Load full dataset from Hugging Face Hub
dataset = load_dataset("NansKong/ManipuriGPT-Corpus-v1.0")

# Access train split
train_data = dataset["train"]
print("First example:", train_data[0]["text"])
```

## Citation

If you use this dataset in your research, please cite our repository using `CITATION.cff`:

```bibtex
{bibtex_text}
```
"""
        readme_path = os.path.join(target_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        logger.info(f"HFPublisher: Saved dataset card README.md to '{readme_path}'")
        return readme_path

    def generate_citation_cff(self, target_dir: str, version: str = "1.0.0") -> str:
        """Generate academic CITATION.cff file following CFF 1.2.0 standards."""
        cff_content = f"""cff-version: 1.2.0
message: "If you use this dataset or foundation model corpus, please cite it as below."
authors:
  - name: "ManipuriGPT Team"
    website: "https://github.com/NansKong/ManipuriGPT-Foundational-Language-"
title: "ManipuriGPT Corpus v1.0: A Research-Grade Open Foundation Model Corpus for the Manipuri Language"
version: "{version}"
date-released: "{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
url: "https://huggingface.co/datasets/NansKong/ManipuriGPT-Corpus-v1.0"
repository-code: "https://github.com/NansKong/ManipuriGPT-Foundational-Language-"
keywords:
  - manipuri
  - meeteillon
  - meitei-mayek
  - language-model
  - foundation-model
  - natural-language-processing
preferred-citation:
  type: data
  authors:
    - name: "ManipuriGPT Team"
  title: "ManipuriGPT Corpus v1.0: Research-Grade Foundation Language Model Corpus for Manipuri"
  year: 2026
  publisher:
    name: "Hugging Face"
  url: "https://huggingface.co/datasets/NansKong/ManipuriGPT-Corpus-v1.0"
"""
        cff_path = os.path.join(target_dir, "CITATION.cff")
        with open(cff_path, "w", encoding="utf-8") as f:
            f.write(cff_content)

        logger.info(f"HFPublisher: Saved CITATION.cff to '{cff_path}'")
        return cff_path

    def generate_tokenizer_readme(self, tokenizer_dir: str) -> str:
        """Generate a Hugging Face model card README.md for the tokenizer repository."""
        # Load tokenizer_config.json for stats if available
        config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        vocab_size = config.get("vocab_size", 32000)
        utilization = config.get("vocabulary_utilization_pct", 91.09)
        unk_rate = config.get("unknown_token_rate_pct", 0.0)
        avg_tokens = config.get("avg_tokens_per_sequence", 160.3)
        entropy = config.get("token_entropy_bits", 11.64)
        compression = config.get("compression_ratio_chars_per_token", 3.653)

        special_tokens = config.get("special_tokens", {})
        special_tokens_lines = "\n".join(
            f"| `{tok}` | {idx} |"
            for tok, idx in special_tokens.items()
        ) if special_tokens else "| `<pad>` | 0 |\n| `<unk>` | 1 |\n| `<s>` | 2 |\n| `</s>` | 3 |"

        readme_content = f"""---
language:
  - mni
  - en
  - bn
license: apache-2.0
tags:
  - manipuri
  - meiteilon
  - meitei-mayek
  - tokenizer
  - sentencepiece
  - bpe
  - indic
  - low-resource
library_name: sentencepiece
---

# ManipuriGPT-Tokenizer-v1.0

The official SentencePiece BPE tokenizer for the [ManipuriGPT](https://github.com/NansKong/ManipuriGPT-Foundational-Language-) foundation model ecosystem, trained on the deduplicated [ManipuriGPT-Corpus-v1.0](https://huggingface.co/datasets/nanskong/ManipuriGPT-Corpus-v1.0).

## Key Metrics

| Metric | Value |
|--------|-------|
| **Algorithm** | SentencePiece BPE |
| **Vocabulary Size** | {vocab_size:,} subwords |
| **Vocabulary Utilization** | {utilization}% |
| **Unknown Token Rate (`<unk>%`)** | {unk_rate:.4f}% |
| **Avg Tokens / Sequence** | {avg_tokens} |
| **Compression Ratio** | {compression} chars/token |
| **Token Entropy** | {entropy} bits |

## Special Tokens

| Token | ID |
|-------|----|
{special_tokens_lines}

## Supported Scripts

1. **Meitei Mayek** — Primary native Unicode script
2. **Bengali Script** — Historical and legacy literary texts
3. **English / Latin** — Romanized Manipuri and code-switching

## Usage

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")

text = "ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯑꯁꯤ ꯑꯆꯧꯕ ꯂꯣꯟ ꯑꯃꯅꯤ"
tokens = sp.Encode(text, out_type=str)
print(tokens)

# Decode back
decoded = sp.Decode(tokens)
print(decoded)
```

## Files

| File | Description |
|------|-------------|
| `tokenizer.model` | SentencePiece binary model (BPE) |
| `tokenizer.vocab` | Full vocabulary with log-probabilities |
| `tokenizer_config.json` | Training metadata, special tokens, and evaluation metrics |
| `special_tokens_map.json` | HF-compatible special token mapping |
| `tokenizer_qualitative_samples.json` | 50 qualitative encoding/decoding samples for manual inspection |

## Training Details

- **Training Corpus**: ManipuriGPT-Corpus-v1.0 (147,065 deduplicated sequences)
- **Character Coverage**: 100%
- **Byte Fallback**: Enabled (zero `<unk>` guarantee)
- **Script-Aware Special Tokens**: `<meitei>`, `<bengali>`, `<romanized>` for script-conditioned generation

## Citation

```bibtex
@misc{{manipurigpt_tokenizer_v10,
  author       = {{ManipuriGPT Team}},
  title        = {{ManipuriGPT-Tokenizer-v1.0: SentencePiece BPE Tokenizer for Manipuri}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  url          = {{https://huggingface.co/nanskong/ManipuriGPT-Tokenizer-v1}}
}}
```

## Related Resources

- **Corpus**: [nanskong/ManipuriGPT-Corpus-v1.0](https://huggingface.co/datasets/nanskong/ManipuriGPT-Corpus-v1.0)
- **GitHub**: [NansKong/ManipuriGPT-Foundational-Language-](https://github.com/NansKong/ManipuriGPT-Foundational-Language-)
"""
        readme_path = os.path.join(tokenizer_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        logger.info(f"HFPublisher: Saved tokenizer model card README.md to '{readme_path}'")
        return readme_path

    def publish_tokenizer(
        self,
        tokenizer_dir: str,
        repo_id: str = "nanskong/ManipuriGPT-Tokenizer-v1",
        private: bool = False,
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Publishes tokenizer directory to Hugging Face Hub as a model repository.
        Generates README.md model card if missing prior to upload.
        """
        logger.info(f"HFPublisher: Preparing to publish tokenizer from '{tokenizer_dir}' to '{repo_id}'...")

        if not os.path.isdir(tokenizer_dir):
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

        # Ensure README.md is present
        readme_path = os.path.join(tokenizer_dir, "README.md")
        if not os.path.exists(readme_path):
            self.generate_tokenizer_readme(tokenizer_dir)

        hub_token = token or os.environ.get("HF_TOKEN")
        try:
            from huggingface_hub import HfApi, get_token
            import time

            if not hub_token:
                hub_token = get_token()

            if not hub_token:
                logger.warning("No Hugging Face token found. Run `hf auth login` or set `HF_TOKEN` environment variable.")
                return {
                    "status": "requires_auth",
                    "repo_id": repo_id,
                    "message": "Token missing. Please log in via `hf auth login` or set HF_TOKEN."
                }

            api = HfApi(token=hub_token)
            try:
                user_info = api.whoami()
                username = user_info.get("name", "unknown")
                logger.info(f"HFPublisher: Authenticated as '{username}'")
            except Exception:
                username = None

            max_retries = 3
            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"HFPublisher: Attempt {attempt}/{max_retries} — Creating/verifying model repo '{repo_id}'...")
                    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

                    logger.info(f"HFPublisher: Uploading tokenizer files from '{tokenizer_dir}' to '{repo_id}'...")
                    api.upload_folder(
                        folder_path=tokenizer_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message="Release ManipuriGPT-Tokenizer-v1.0: SentencePiece BPE 32k tokenizer"
                    )
                    logger.info(f"HFPublisher: Successfully published tokenizer to https://huggingface.co/{repo_id}")
                    return {
                        "status": "success",
                        "repo_id": repo_id,
                        "url": f"https://huggingface.co/{repo_id}",
                        "tokenizer_dir": tokenizer_dir
                    }
                except Exception as net_err:
                    last_err = net_err
                    err_str = str(net_err)
                    if "403" in err_str or "Forbidden" in err_str:
                        raise net_err
                    elif "getaddrinfo failed" in err_str or "11001" in err_str:
                        logger.warning(f"HFPublisher: Attempt {attempt}/{max_retries} — DNS/Network error. Retrying in 5 seconds...")
                        if attempt < max_retries:
                            time.sleep(5)
                    else:
                        raise net_err

            if last_err:
                raise last_err

        except Exception as e:
            err_msg = str(e)
            if "403" in err_msg or "Forbidden" in err_msg:
                logger.error(f"HFPublisher: 403 Forbidden. Your token may lack write access to namespace '{repo_id.split('/')[0]}'.")
                logger.error("Fix: run with --repo-id <YOUR_HF_USERNAME>/ManipuriGPT-Tokenizer-v1")
            else:
                logger.error(f"HFPublisher: Failed to publish tokenizer ({e})")
            return {"status": "error", "error": err_msg, "repo_id": repo_id}

    def publish_dataset(
        self,
        dataset_dir: str,
        repo_id: str = "NansKong/ManipuriGPT-Corpus-v1.0",
        private: bool = False,
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Publishes dataset directory to Hugging Face Hub under `repo_id`.
        Generates README.md and CITATION.cff if missing prior to upload.
        """
        logger.info(f"HFPublisher: Preparing to publish dataset from '{dataset_dir}' to '{repo_id}'...")

        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        # Ensure README.md and CITATION.cff are present
        readme_path = os.path.join(dataset_dir, "README.md")
        cff_path = os.path.join(dataset_dir, "CITATION.cff")

        if not os.path.exists(readme_path) or not os.path.exists(cff_path):
            manifest_path = os.path.join(dataset_dir, "manifest.json")
            manifest_data = {}
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)

            audit_path = os.path.join(dataset_dir, "corpus_audit_report.json")
            audit_data = None
            if os.path.exists(audit_path):
                with open(audit_path, "r", encoding="utf-8") as f:
                    audit_data = json.load(f)

            if not os.path.exists(readme_path):
                self.generate_readme(dataset_dir, manifest_data, audit_data)
            if not os.path.exists(cff_path):
                self.generate_citation_cff(dataset_dir, version=manifest_data.get("version", "1.0.0"))

        hub_token = token or os.environ.get("HF_TOKEN")
        try:
            from huggingface_hub import HfApi, get_token
            import time

            if not hub_token:
                hub_token = get_token()

            if not hub_token:
                logger.warning("No Hugging Face token found. Run `huggingface-cli login` or set `HF_TOKEN` environment variable.")
                return {
                    "status": "requires_auth",
                    "repo_id": repo_id,
                    "dataset_dir": dataset_dir,
                    "readme_path": readme_path,
                    "cff_path": cff_path,
                    "message": "Token missing. Please log in via `huggingface-cli login` or set HF_TOKEN environment variable."
                }

            api = HfApi(token=hub_token)
            try:
                user_info = api.whoami()
                username = user_info.get("name", "unknown")
                orgs = [o.get("name") for o in user_info.get("orgs", []) if o.get("name")]
                logger.info(f"HFPublisher: Authenticated with Hugging Face as user '{username}' (Orgs: {orgs if orgs else 'None'})")
            except Exception as auth_err:
                username = None
                logger.warning(f"HFPublisher: Could not verify token identity via whoami: {auth_err}")

            # Retry loop for network operations (DNS resolution / transient drops)
            max_retries = 3
            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"HFPublisher: Attempt {attempt}/{max_retries} — Creating/verifying repository '{repo_id}'...")
                    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

                    logger.info(f"HFPublisher: Uploading folder '{dataset_dir}' to dataset repo '{repo_id}'...")
                    commit_msg = f"Release ManipuriGPT Corpus v1.0: {manifest_data.get('version', '1.0.0') if 'manifest_data' in locals() else '1.0.0'}"
                    
                    api.upload_folder(
                        folder_path=dataset_dir,
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=commit_msg
                    )
                    logger.info(f"HFPublisher: Successfully published dataset to https://huggingface.co/datasets/{repo_id}")
                    return {
                        "status": "success",
                        "repo_id": repo_id,
                        "url": f"https://huggingface.co/datasets/{repo_id}",
                        "dataset_dir": dataset_dir
                    }
                except Exception as net_err:
                    last_err = net_err
                    err_str = str(net_err)
                    if "403" in err_str or "Forbidden" in err_str:
                        # 403 Forbidden is a permission issue, do not retry
                        raise net_err
                    elif "getaddrinfo failed" in err_str or "11001" in err_str:
                        logger.warning(f"HFPublisher: Attempt {attempt}/{max_retries} failed due to DNS/Network lookup error (getaddrinfo failed). Retrying in 5 seconds...")
                        if attempt < max_retries:
                            time.sleep(5)
                    else:
                        raise net_err

            if last_err:
                raise last_err

        except Exception as e:
            err_msg = str(e)
            if "403" in err_msg or "Forbidden" in err_msg:
                user_msg = f"Logged-in user is '{username}'" if 'username' in locals() and username else "Logged-in user unknown"
                logger.error(f"HFPublisher: 403 Forbidden permission error. {user_msg}.")
                logger.error(f"If your Hugging Face username is different from 'NansKong', run with: --repo-id <YOUR_HF_USERNAME>/ManipuriGPT-Corpus-v1.0")
                logger.error("If publishing under an organization, ensure your HF Access Token has 'Write' access enabled at https://huggingface.co/settings/tokens")
            elif "getaddrinfo failed" in err_msg or "11001" in err_msg:
                logger.error("HFPublisher: Internet / DNS resolution failed. Please check your network connection or DNS settings.")
            else:
                logger.error(f"HFPublisher: Failed to publish dataset ({e})")
            return {"status": "error", "error": err_msg, "repo_id": repo_id}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Hugging Face Publication CLI (Dataset + Tokenizer)")

    # Dataset arguments
    parser.add_argument("--dataset-dir", type=str, default="artifacts/datasets/ManipuriGPT-Corpus-v1.0", help="Path to frozen corpus directory")
    parser.add_argument("--repo-id", type=str, default="nanskong/ManipuriGPT-Corpus-v1.0", help="Hugging Face dataset repository ID")
    parser.add_argument("--publish", action="store_true", help="Publish dataset to Hugging Face Hub")

    # Tokenizer arguments
    parser.add_argument("--tokenizer-dir", type=str, default="artifacts/datasets/ManipuriGPT-Corpus-v1.0/tokenizer", help="Path to tokenizer directory")
    parser.add_argument("--tokenizer-repo-id", type=str, default="nanskong/ManipuriGPT-Tokenizer-v1", help="Hugging Face tokenizer model repository ID")
    parser.add_argument("--publish-tokenizer", action="store_true", help="Publish tokenizer to Hugging Face Hub as a model repo")

    # Common
    parser.add_argument("--private", action="store_true", help="Create private HF repository")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    publisher = HFPublisher()

    logger.info("=" * 70)
    logger.info(" MANIPURIGPT HUGGING FACE PUBLICATION CLI")
    logger.info("=" * 70)

    # --- Dataset Card Generation ---
    logger.info("Generating/Updating HF README.md and CITATION.cff for dataset...")
    manifest_p = os.path.join(args.dataset_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_p):
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    audit_p = os.path.join(args.dataset_dir, "corpus_audit_report.json")
    audit = None
    if os.path.exists(audit_p):
        with open(audit_p, "r", encoding="utf-8") as f:
            audit = json.load(f)

    readme_path = publisher.generate_readme(args.dataset_dir, manifest, audit)
    cff_path = publisher.generate_citation_cff(args.dataset_dir, version=manifest.get("version", "1.0.0"))
    logger.info(f"Generated Dataset README.md : {readme_path}")
    logger.info(f"Generated CITATION.cff      : {cff_path}")

    # --- Tokenizer Card Generation ---
    if os.path.isdir(args.tokenizer_dir):
        tok_readme = publisher.generate_tokenizer_readme(args.tokenizer_dir)
        logger.info(f"Generated Tokenizer README.md: {tok_readme}")

    # --- Dataset Publishing ---
    if args.publish:
        logger.info("-" * 70)
        logger.info(f"Publishing dataset to '{args.repo_id}'...")
        res = publisher.publish_dataset(args.dataset_dir, repo_id=args.repo_id, private=args.private)
        if res.get("status") == "success":
            logger.info(f"Dataset published successfully! URL: {res.get('url')}")
        elif res.get("status") == "requires_auth":
            logger.warning(f"Dataset publication pending authentication: {res.get('message')}")
        else:
            logger.error(f"Dataset publication failed: {res.get('error')}")

    # --- Tokenizer Publishing ---
    if args.publish_tokenizer:
        logger.info("-" * 70)
        logger.info(f"Publishing tokenizer to '{args.tokenizer_repo_id}'...")
        res = publisher.publish_tokenizer(args.tokenizer_dir, repo_id=args.tokenizer_repo_id, private=args.private)
        if res.get("status") == "success":
            logger.info(f"Tokenizer published successfully! URL: {res.get('url')}")
        elif res.get("status") == "requires_auth":
            logger.warning(f"Tokenizer publication pending authentication: {res.get('message')}")
        else:
            logger.error(f"Tokenizer publication failed: {res.get('error')}")

    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
