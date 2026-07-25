# ManipuriGPT

**ManipuriGPT** is an open, reproducible research project for building language technology for Manipuri (Meiteilon). It is designed to develop a Manipuri-aware foundation-model ecosystem—not just a chatbot—through a documented corpus, multi-script processing, tokenizer research, continued pretraining, instruction tuning, and evaluation.

The project follows the vision described in [Proposal.pdf](Proposal.pdf): support Manipuri across **Meitei Mayek**, **Romanized Manipuri**, and **Bengali-script Manipuri**, while keeping data sources, processing, and experiments reproducible.

## Project goals

- Build a licensed, reproducible Manipuri corpus from streaming and local sources.
- Normalize, clean, deduplicate, validate, and shard corpus data.
- Train and evaluate Manipuri-aware tokenizers.
- Support language-adaptive continued pretraining and instruction tuning with LoRA/QLoRA.
- Evaluate translation, chat, question answering, and reasoning workloads.
- Export trained checkpoints for Hugging Face, GGUF, and ONNX use.

## Current repository capabilities

The repository currently contains modular Python components for:

- Evaluated candidate tokenizers and frozen standard `tokenizer.model` (`v1`, 0.00% `<unk>` rate).
- Production master corpus scaling engine (`run_phase55.py`) with cross-version deduplication and non-overwriting release snapshots (`ManipuriGPT-Corpus-v1`, `ManipuriGPT-Corpus-v2`, `ManipuriGPT-Corpus-v3`).
- Specialized OCR artifact cleaning, companion JSON sidecar metadata mapping, and Latin noise density filtering for scanned PDF data (`d_drive_manipuri_corpus_processed`).
- Decoupled EM-Suite utilities: `EMFastTextEngine` (post-processing utility for semantic lookup and OCR candidate ranking) and `EMAlbertEvaluator` (evaluation/benchmarking layer).

## Release Snapshots & Corpus Build Results

| Release Snapshot | Status | Total Sequences | Total Tokens | `<unk>` Token Rate | Key Sources Included |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ManipuriGPT-Corpus-v1** | Baseline | 143,891 | ~2,844,564 | **0.0000%** | Local processed PDFs, Dayananda Meitei Mayek |
| **ManipuriGPT-Corpus-v2** | Scaled | Expanded | Scaled | **0.0000%** | `v1` + Sangraha, EMA Lon, Joyson Parallel |
| **ManipuriGPT-Corpus-v3** | **Latest Release** | **4,065 (new)** | **738,068** | **0.0000%** | `v1` + `v2` + `joyson_monolingual` + `D:/manipuri corpus/processed` |

### Detailed `v3` Corpus Expansion Metrics

- **New Unique Sequences Added**: 4,065
- **Total Tokens**: 738,068 tokens (0.00% unknown token rate)
- **Cross-Version Duplicates Removed**: 2,754,516 (99.62% duplicate rate eliminated against `v1` & `v2`)
- **Scanned OCR Lines Filtered**: 1,145 garbled lines pruned via regex page marker cleaner & Latin density threshold (>35% Latin char density)
- **Script Distribution**: 64.4% Meitei Mayek, 26.3% Bengali Script, 7.4% Latin, 1.9% Mixed

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

## Repository layout

```text
app/
  configs/          YAML configuration and loading helpers
  corpus/           source registry, acquisition, streaming, and sampling
  datasets/         dataset registry, loading, validation, and builders
  preprocessing/    corpus cleaning and transformation pipeline
  tokenization/     tokenizer training, evaluation, and versioning
  tokenizer/        runtime tokenizer utilities and dataset preparation
  training/         training configuration, backends, callbacks, and trainer
  evaluation/       metrics and evaluation suite
  inference/        inference engine and validation helpers
  exports/          Hugging Face, GGUF, and ONNX exporters
  scripts/          command-line workflow entry points
docs/               project documentation and phase notes
requirements/       dependency groups
Proposal.pdf        project proposal and roadmap
```

## Setup

Requires Python 3.10 or newer. Create an isolated environment, then install the dependencies appropriate for the work you plan to do.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Core development and tests
pip install -r requirements/dev.txt

# Corpus, tokenizer, training, and evaluation workflows
pip install -r requirements/training.txt

# Optional FastAPI serving dependencies
pip install -r requirements/api.txt
```

Copy the example environment file if a workflow requires local configuration or credentials:

```powershell
Copy-Item .env.example .env
```

Never commit `.env`, downloaded corpora, caches, checkpoints, or model exports. They are intentionally excluded by `.gitignore`.

## Configuration and data access

Review `app/configs/datasets.yaml` before acquiring data. Each source describes its provider, split, script, domain, license, priority, download method, and processing pipeline.

Some configured Hugging Face sources are gated or require accepted terms. Authenticate only with a personal access token that has the minimum necessary **read** permission; place it in your local environment configuration, never in source code or Git.

The pipeline defaults to streaming where possible. Respect each dataset's licence, terms, and redistribution restrictions before training on or publishing derived data.

## Common workflows

Run commands from the repository root.

### Ingest a small corpus sample

```powershell
python -m app.scripts.ingest --source wikipedia --limit 1000
```

Use `--no-mock` to require a live source instead of allowing the development fallback.

### Preprocess and shard a corpus

```powershell
python -m app.scripts.preprocess_shards `
  --sources dayananda_meitei_mayek_sample dayananda_english_to_meitei `
  --limit 5000 `
  --shard-size 1000 `
  --format parquet `
  --output-dir artifacts/datasets/phase52
```

Add `--resume` to continue from the output manifest after an interruption.

### Execute Master Corpus Scaling & Snapshot Build (Phase 5.5)

Run the master scaling pipeline with cross-version deduplication (`v1` and `v2`) to produce standard Parquet shards and `manifest.json`:

```powershell
python -m app.scripts.run_phase55 `
  --output-dir artifacts/datasets/ManipuriGPT-Corpus-v3 `
  --v1-dir artifacts/datasets/ManipuriGPT-Corpus-v1 `
  --v2-dir artifacts/datasets/ManipuriGPT-Corpus-v2 `
  --skip-freeze
```

### Train tokenizer candidates

```powershell
python -m app.scripts.train_tokenizers `
  --train-samples 5000 `
  --evaluate `
  --convert-hf `
  --tier v0-experimental
```

Candidate tokenizer settings—including algorithms and vocabulary sizes—are in `app/configs/tokenizer.yaml`.

### Validate a training configuration

Start with a dry run before initiating a costly training job:

```powershell
python -m app.scripts.train `
  --model smollm_135m `
  --mode qlora `
  --backend peft `
  --epochs 3 `
  --dry-run
```

Training defaults are defined in `app/configs/training.yaml`. Choose hardware-appropriate precision and batch size; the proposal targets both a constrained local GPU and cloud notebook environments.

### Evaluate and export

```powershell
python -m app.scripts.evaluate --model smollm_135m --task translation

python -m app.scripts.export `
  --checkpoint artifacts/models/checkpoints `
  --model smollm_135m `
  --targets hf gguf onnx
```

Exports are simulated by the current CLI implementation; validate an exported artifact before publishing it.

## Quality checks

Run the test suite locally:

```powershell
pytest
```

Useful focused runs:

```powershell
pytest tests/preprocessing
pytest tests/tokenizer
pytest -m "not slow"
```

Test files are intentionally ignored by this repository's current Git policy, but they remain available locally for validation.

## Reproducibility principles

- Use YAML configuration for sources and experiment settings.
- Record dataset licences, source metadata, and preprocessing decisions.
- Keep generated data, cache files, checkpoints, and binary exports outside version control.
- Preserve manifests, metrics, and configuration snapshots with each experiment.
- Do not publish models or datasets without verifying licensing, data provenance, and evaluation results.

## Roadmap

The proposal organizes the work around corpus research and ingestion, preprocessing, tokenizer research, language-adaptive pretraining, instruction fine-tuning, evaluation, RAG, and public release. See [docs/phases.md](docs/phases.md) and [Proposal.pdf](Proposal.pdf) for the detailed plan.

## Contributing

Contributions should be modular, configuration-driven, reproducible, and accompanied by local validation. Do not add secrets, raw restricted datasets, caches, checkpoints, or large model binaries to commits.
