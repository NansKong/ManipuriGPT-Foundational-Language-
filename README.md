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

- Dataset registration, loading, validation, caching, and translation/chat dataset construction.
- Streaming corpus acquisition and balanced sampling.
- Preprocessing: language and script detection, cleaning, Unicode normalization, PII removal, deduplication, quality scoring, chunking, splitting, and export.
- Tokenizer training, candidate evaluation, versioning, formatting, packing, and Hugging Face conversion.
- Configuration-driven training backends and modes including full fine-tuning, LoRA, QLoRA, SFT, DPO, ORPO, and continued pretraining.
- Evaluation, inference validation, experiment tracking, and model export.

The proposal also includes future work such as a production FastAPI service, retrieval-augmented generation (RAG), OCR resources, a web demo, CI/CD, public benchmarks, and public model/data releases. Treat these as roadmap items unless their implementation is added to the repository.

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
