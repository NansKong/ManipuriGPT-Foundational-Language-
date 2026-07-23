# ManipuriGPT Project Phases

This document outlines the detailed development phases for the ManipuriGPT project.

---

## Phase 1 — Repository Foundation (Week 1)

Establishing the core structure of the repository.

```text
ManipuriGPT/
│
├── app/
│   ├── datasets/
│   ├── preprocessing/
│   ├── tokenizer/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── api/
│   ├── utils/
│   └── configs/
│
├── scripts/
│
├── notebooks/
│
├── docs/
│
├── tests/
│
├── models/
│
├── docker/
│
├── requirements.txt
├── README.md
├── .env.example
└── pyproject.toml
```

---

## Phase 2 — Data Layer

Implementing dataset ingestion strategies, prioritizing streaming.

### Key Datasets (Hugging Face)
* **Translation**: English ↔ Meitei Mayek Translation
* **Monolingual/Corpora**: Manipuri Corpus, OSCAR, CC100, FLoRes, OPUS, Wikipedia
* **AI4Bharat**: IndicCorp

*Note: Only datasets unavailable on Hugging Face will be downloaded locally.*

### Registry Component
We will create a centralized file `app/datasets/dataset_registry.py` which maintains:
* Dataset source
* Dataset splits
* Target languages
* Streaming capability configurations
* Custom preprocessing rules

---

## Phase 3 — Corpus Pipeline

An end-to-end data processing workflow for raw textual data:

```text
Raw Dataset
      │
      ▼
Language Detection
      │
Cleaning
      │
Deduplication
      │
Normalization (Unicode & Script normalization)
      │
Sentence Validation (Filtering short/corrupt texts)
      │
Train / Validation Split
      │
Tokenization
      │
Arrow Dataset
```

* **Caching**: Everything is fully cached. No repeated preprocessing.

---

## Phase 4 — Training

Robust training configurations supporting resource-constrained environments.

### Supported Environments
* Kaggle Notebooks
* Google Colab
* Lightning AI
* Hugging Face ZeroGPU (where applicable)

### Training Methods
* LoRA (Low-Rank Adaptation)
* QLoRA (Quantized LoRA)
* Full Fine-tuning
* Continued Pretraining (DAPT)

### Optimizations
* Automatic GPU / accelerator detection
* Mixed Precision training (FP16/BF16)
* Gradient Checkpointing
* Resume from Checkpoints

---

## Phase 5 — Supported Model Families

Modular support for popular open-source model families.

* **Initial Models**: Llama 3.2, Qwen 2.5, Mistral, Gemma
* **Future Additions**: DeepSeek, Phi, SmolLM

*Note: Model selection and hyperparameter loading are fully configuration-driven.*

---

## Phase 6 — Evaluation

Automatic validation suite running at the end of each training phase.

### Metrics Supported
* **Translation & Generation Quality**: BLEU, ROUGE, ChrF, COMET
* **Language Modeling**: Perplexity
* **Human-in-the-loop**: Human Evaluation Samples generator

### Output Artifacts
Every training run automatically outputs:
* `evaluation.json`
* `metrics.csv`
* `graphs/` (loss curves, evaluation metrics over training steps)
* `predictions.csv` (sample generation targets vs. model outputs)

---

## Phase 7 — API

A fast, production-ready inference API built with FastAPI.

### API Endpoints
* `GET /health` - Health status and hardware usage stats
* `GET /model-info` - Loaded model architecture and parameters
* `POST /translate` - Translation between scripts and languages (English ↔ Manipuri)
* `POST /chat` - Interactive text generation/assistant conversations
* `POST /evaluate` - Run evaluation metrics on custom reference/prediction lists

---

## Phase 8 — Web Demo

A simple, user-friendly interactive web interface.

### Features
* **Translation Workspace**:
  ```text
  English
  ──────────────
  How are you?

  ↓

  Meitei Mayek
  ──────────────
  ꯅꯪ ꯀꯔꯤ ꯇꯧꯔꯤꯕꯅꯣ?
  ```
* **Utility Features**:
  * Copy to clipboard
  * Download translations/outputs
  * Session history tracking

---

## Phase 9 — CI/CD

Automated validation pipelines via GitHub Actions.

### Automation Tasks
* Code formatting validation
* Linter checks
* Unit tests execution
* Model evaluation validation tests
* Docker image build verification

---

## Phase 10 — Documentation

Deliverable guides and project references.

### Required Documentation
1. **Installation Guide**: Local, Colab, and Docker setup instructions.
2. **Dataset Documentation**: Description of data registry, schemas, and processing rules.
3. **Model Card**: Detail model base, training hyperparameters, and script capability.
4. **Training Guide**: How to run tokenizer and model training pipelines.
5. **API Documentation**: OpenAPI/Swagger specifications and request examples.
6. **Contributing Guide**: Style guidelines and branch workflow conventions.
