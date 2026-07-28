# ManipuriGPT

**ManipuriGPT** is an open, research-grade, reproducible foundation model ecosystem for the **Manipuri (Meiteilon)** language. It is designed to develop a Manipuri-aware language technology pipeline—from corpus engineering, multi-script tokenization, and continued pretraining to comprehensive evaluation, instruction tuning, and deployment.

The project supports Manipuri across its primary writing systems: **Meitei Mayek** (native Unicode standard), **Bengali Script** (historical literary texts), and **Romanized / Latin Manipuri** (informal communication).

---

## Official Hugging Face Releases

| Resource | Type | Hugging Face Repository | Description |
| :--- | :--- | :--- | :--- |
| **`ManipuriGPT-Corpus-v1.0`** | Dataset | [nanskong/ManipuriGPT-Corpus-v1.0](https://huggingface.co/datasets/nanskong/ManipuriGPT-Corpus-v1.0) | Research-grade, deduplicated, quality-scored multi-script Manipuri corpus |
| **`ManipuriGPT-Tokenizer-v1.0`** | Tokenizer | [nanskong/ManipuriGPT-Tokenizer-v1](https://huggingface.co/nanskong/ManipuriGPT-Tokenizer-v1) | 8,000 vocab SentencePiece BPE tokenizer with 0% `<unk>` error rate |
| **`ManipuriGPT-135M-Base-v1.0`** | **Base Model** | [nanskong/ManipuriGPT-135M-Base-v1.0](https://huggingface.co/nanskong/ManipuriGPT-135M-Base-v1.0) | **Official Pretrained Foundation Model** (SmolLM-135M base, 13.5k steps) |

---

## Phase 7 — Pretraining & Evaluation Metrics

ManipuriGPT has completed Phase 7 (Foundation Model Evaluation) across 13.5k pretraining steps.

### Pretraining Diagnostics Summary

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

### Load Pretrained Base Model with `transformers`

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

## Repository Layout & Modular Components

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

## Common Workflows & CLI Entry Points

### 1. Run Complete Phase 7 Evaluation Suite

```powershell
python -m app.scripts.run_phase7_eval --all
```

Or execute individual diagnostic steps:
```powershell
# Training Loss Curves & Report
python -m app.scripts.run_phase7_eval --step 7.1

# Script-wise Perplexity
python -m app.scripts.run_phase7_eval --step 7.2

# Inference Speed Profiling
python -m app.scripts.run_phase7_eval --step 7.8
```

### 2. Freeze Base Foundation Release

```powershell
python -m app.scripts.freeze_base_model_v10 --source-dir models/smollm_135m_pretrained --target-dir models/ManipuriGPT-135M-Base-v1.0
```

### 3. Publish Base Model to Hugging Face Hub

```powershell
python -m app.exports.hf_publisher --model-dir models/ManipuriGPT-135M-Base-v1.0 --model-repo-id nanskong/ManipuriGPT-135M-Base-v1.0 --publish-model
```

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
