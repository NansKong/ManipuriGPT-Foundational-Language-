# ManipuriGPT

**ManipuriGPT** is an open, reproducible research project for building language technology for Manipuri (Meiteilon). It is designed to develop a Manipuri-aware foundation-model ecosystem—from documented corpus engineering, multi-script processing, tokenizer research, continued pretraining, instruction tuning, and evaluation to model export and public release.

The project follows the vision described in [Proposal.pdf](Proposal.pdf): support Manipuri across **Meitei Mayek** (native Unicode standard), **Bengali Script** (historical literary texts), and **Romanized / Latin Manipuri** (informal communication), while keeping data sources, processing, and experiments reproducible.

---

## Official Hugging Face Releases

| Resource | Type | Hugging Face Repository | Description |
| :--- | :--- | :--- | :--- |
| **`ManipuriGPT-Corpus-v1.0`** | Dataset | [nanskong/ManipuriGPT-Corpus-v1.0](https://huggingface.co/datasets/nanskong/ManipuriGPT-Corpus-v1.0) | Research-grade, deduplicated, quality-scored multi-script Manipuri corpus |
| **`ManipuriGPT-Tokenizer-v1.0`** | Tokenizer | [nanskong/ManipuriGPT-Tokenizer-v1](https://huggingface.co/nanskong/ManipuriGPT-Tokenizer-v1) | 8,000 vocab SentencePiece BPE tokenizer with 0% `<unk>` error rate |
| **`ManipuriGPT-135M-Base-v1.0`** | **Base Model** | [nanskong/ManipuriGPT-135M-Base-v1.0](https://huggingface.co/nanskong/ManipuriGPT-135M-Base-v1.0) | **Official Pretrained Foundation Model** (SmolLM-135M base, 13.5k steps) |

---

## Project Goals

- Build a licensed, reproducible Manipuri corpus from streaming and local sources.
- Normalize, clean, deduplicate, validate, and shard corpus data across Meitei Mayek, Bengali script, and Latin.
- Train and evaluate Manipuri-aware tokenizers with zero unknown (`<unk>`) token emissions.
- Support language-adaptive continued pretraining and instruction tuning with LoRA/QLoRA on SmolLM architectures.
- Conduct rigorous evaluation across perplexity, script consistency, multi-sampling generation, translation, QA, and reasoning.
- Export trained checkpoints for Hugging Face, GGUF, and ONNX use.

---

## Repository Capabilities

- **Evaluated Tokenizers**: Evaluated candidate tokenizers and frozen standard `tokenizer.model` (0.00% `<unk>` rate).
- **Master Corpus Scaling Engine (`run_phase55.py`)**: Cross-version deduplication and non-overwriting release snapshots (`ManipuriGPT-Corpus-v1`, `v2`, `v3`).
- **OCR Artifact Cleaning**: Specialized OCR cleaning, companion JSON sidecar metadata mapping, and Latin noise density filtering for scanned PDF archives (`d:/manipuri corpus`).
- **Phase 7 Evaluation Suite (`run_phase7_eval.py`)**: 10-step automated evaluation engine profiling training loss curves, script-wise perplexity, multi-sampling decoding diversity, script consistency, memorization, and inference throughput.

---

## Phase 7 — Pretraining & Evaluation Results

ManipuriGPT has completed Phase 7 foundation model pretraining and evaluation across 13.5k steps (3.0 Epochs).

| Metric | Value | Notes / Status |
| :--- | :--- | :--- |
| **Pretraining Epochs** | **3.0 Epochs** (13,596 steps) | Full corpus coverage |
| **Final Train Loss** | `3.6246` | Smooth cross-entropy convergence |
| **Final Eval Loss** | `4.2799` | Best eval loss: `4.2693` |
| **Meitei Mayek PPL** | **`133.14`** | Strong language representation on native script |
| **Bengali Script PPL** | `1508.89` | Script-switching representation |
| **Tokenizer `<unk>` Rate** | **0.0000%** | Zero unknown tokens emitted |
| **Compression Ratio** | `7.125 bytes/token` | Highly efficient subword compression |
| **Generation Diversity** | Distinct-1: `0.8683` \| Distinct-2: `0.9849` | Self-BLEU: `4.22` (High output variety) |

---

## Model Lineage & Hierarchy

```text
ManipuriGPT-135M-Base-v1.0  (Immutable Base Foundation Model Release)
        │
        ├── ManipuriGPT-135M-SFT-v1.0   (Phase 8: Instruction Fine-Tuning)
        ├── ManipuriGPT-135M-SFT-v2.0   (Multi-Task & Reasoning Tuning)
        ├── ManipuriGPT-135M-DPO-v1.0   (Direct Preference Optimization)
        └── ManipuriGPT-135M-Chat-v1.0  (Conversational Assistant)
```

---

## Quickstart Python Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "nanskong/ManipuriGPT-135M-Base-v1.0"

# Load model and tokenizer from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# Generate Manipuri text continuation
prompt = "ꯑꯩ"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_k=50, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Architecture

```text
Configured data sources
        |
        v
Corpus ingestion / streaming
        |
        v
Cleaning -> normalization -> script detection -> deduplication -> validation
        |
        v
Sharded unified corpus
        |
        +--> Tokenizer training and evaluation
        |
        +--> Continued pretraining / instruction tuning
                       |
                       v
                 Evaluation and export
```

Dataset sources and processing metadata live in `app/configs/datasets.yaml`; runtime behavior is intended to be changed through configuration rather than hard-coded URLs.

---

## Repository Layout

```text
app/
  configs/          YAML configuration and loading helpers
  corpus/           Source registry, acquisition, streaming, and sampling
  datasets/         Dataset registry, loading, validation, and builders
  preprocessing/    Corpus cleaning and transformation pipeline
  tokenization/     Tokenizer training, evaluation, and versioning
  training/         Training backends, Tesla T4 FP16 auto-selection, and Trainer
  evaluation/       Phase 7 multi-perspective evaluation suite:
                      ├── training_analyzer.py      (Loss curves & training_report.md)
                      ├── perplexity_eval.py        (Script-wise PPL: Meitei/Bengali)
                      ├── token_inspector.py        (Next-token probability distributions)
                      ├── generator_eval.py         (Multi-sampling decoding & diversity)
                      ├── script_eval.py            (Script consistency & Unicode health)
                      ├── memorization_eval.py      (Verbatim memorization check)
                      ├── benchmark_runner.py       (Task benchmarks: Translation, OCR)
                      ├── speed_benchmark.py        (Tokens/sec, latency & VRAM)
                      ├── tokenizer_eval.py         (Tokenizer health diagnostics)
                      ├── checkpoint_compare.py     (Multi-checkpoint scorecards)
                      ├── human.py                  (Offline human_review.md generator)
                      └── report_generator.py       (Final final_eval_report.md)
  exports/          HF, GGUF, ONNX exporters and HFPublisher engine
  scripts/          CLI entry points:
                      ├── run_phase55.py            (Master corpus scaling CLI)
                      ├── freeze_corpus_v10.py       (Corpus snapshot freezer)
                      ├── freeze_base_model_v10.py   (Base model release freezer)
                      ├── run_phase7_eval.py        (Master Phase 7 evaluation CLI)
                      └── export.py                 (Export & Hugging Face publisher)
docs/               Project documentation and phase design specs
evaluation/         Phase 7 evaluation artifacts, reports, and plots
Proposal.pdf        Project proposal and architectural roadmap
```

---

## Setup

Requires Python 3.10 or newer. Create an isolated environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Core development and tests
pip install -r requirements/dev.txt

# Corpus, tokenizer, training, and evaluation workflows
pip install -r requirements/training.txt
```

---

## Common Workflows & CLI Entry Points

Run commands from the repository root.

### 1. Ingest a corpus sample

```powershell
python -m app.scripts.ingest --source wikipedia --limit 1000
```

### 2. Execute Master Corpus Scaling & Snapshot Build

```powershell
python -m app.scripts.run_phase55 `
  --output-dir artifacts/datasets/ManipuriGPT-Corpus-v3 `
  --v1-dir artifacts/datasets/ManipuriGPT-Corpus-v1 `
  --v2-dir artifacts/datasets/ManipuriGPT-Corpus-v2 `
  --skip-freeze
```

### 3. Run Complete Phase 7 Evaluation Suite

```powershell
python -m app.scripts.run_phase7_eval --all
```

Or run individual diagnostic steps:
```powershell
python -m app.scripts.run_phase7_eval --step 7.1   # Training Loss Curves & Report
python -m app.scripts.run_phase7_eval --step 7.2   # Script-wise Perplexity
python -m app.scripts.run_phase7_eval --step 7.8   # Inference Speed Profiling
```

### 4. Freeze Base Foundation Release

```powershell
python -m app.scripts.freeze_base_model_v10 --source-dir models/smollm_135m_pretrained --target-dir models/ManipuriGPT-135M-Base-v1.0
```

### 5. Publish Model / Tokenizer to Hugging Face Hub

```powershell
python -m app.exports.hf_publisher --model-dir models/ManipuriGPT-135M-Base-v1.0 --model-repo-id nanskong/ManipuriGPT-135M-Base-v1.0 --publish-model
```

---

## Detailed Documentation & Dataset Citations

For comprehensive technical history, pretraining step diagnostics, dataset catalog breakdown, and complete BibTeX citations, see:
* **[docs/PROJECT_HISTORY_AND_CITATIONS.md](docs/PROJECT_HISTORY_AND_CITATIONS.md)**

---

## Citation & Academic Use

If you use ManipuriGPT models, tokenizers, or datasets in your research, please cite:

```bibtex
@misc{manipurigpt_base_v10,
  author       = {ManipuriGPT Team},
  title        = {ManipuriGPT-135M-Base-v1.0: Research-Grade Open Foundation Model for Manipuri},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/nanskong/ManipuriGPT-135M-Base-v1.0}
}
```

---

## License

This repository and model weights are released under the [Apache 2.0 License](LICENSE).
