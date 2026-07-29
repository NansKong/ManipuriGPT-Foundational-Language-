# ManipuriGPT: Technical History, Execution Roadmap & Dataset Citations

This document provides a comprehensive record of all engineering phases, corpus sources, dataset preprocessing, tokenizer development, foundation model pretraining, evaluation diagnostics, and academic citations for the **ManipuriGPT** project.

---

## 1. Project Overview & Vision

**ManipuriGPT** is an open, reproducible research project dedicated to building foundation model language technology for **Manipuri (Meiteilon)**. Manipuri is a low-resource Tibeto-Burman language primarily spoken in Manipur, India, written across three distinct writing systems:
1. **Meitei Mayek** (`\uABC0-\uABFF`, `\u1C80-\u1C8F`): The native Unicode standard script.
2. **Bengali Script** (`\u0980-\u09FF`): Historically used for literary, religious, and academic texts since the 18th century.
3. **Romanized / Latin Manipuri**: Widely used in modern digital communication and social media.

---

## 2. Phase-by-Phase Execution History

```text
Phase 0-4: Data Engineering & Preprocessing Engine
    ├── Multi-script Unicode normalization & cleaning
    ├── Script detection & Latin noise density filtering
    └── SentencePiece BPE Tokenizer training (0% <unk>)
            │
            v
Phase 5: Master Corpus Scaling & Snapshot Freezing
    ├── 2.75M cross-version duplicates eliminated
    ├── OCR noise cleaning & companion JSON sidecar metadata
    └── Immutable release: ManipuriGPT-Corpus-v1.0 & ManipuriGPT-Tokenizer-v1.0
            │
            v
Phase 6: Pretraining Setup & Held-out Benchmark Generation
    ├── Sequence packing (512 token context window)
    └── 4 held-out task benchmarks (PPL, Translation, Script, OCR)
            │
            v
Phase 7: Foundation Pretraining & 10-Step Evaluation Suite
    ├── 13,596 steps (3.0 Epochs) on Tesla T4 FP16
    ├── Final train loss: 3.6246 | Eval loss: 4.2799 | Meitei Mayek PPL: 133.14
    └── Official base release: ManipuriGPT-135M-Base-v1.0
```

### Key Milestones Achieved

- **Phase 0–4 (Corpus & Tokenization Core)**: Developed modular Python preprocessing modules supporting streaming and local data ingestion. Trained subword tokenizers evaluated across vocabulary utilization, subword entropy, and zero `<unk>` emissions.
- **Phase 5 (Master Corpus Scaling & Freezing)**: Constructed the master scaling CLI (`run_phase55.py`), executing cross-version deduplication that removed over 2,754,516 duplicate sequences across v1 and v2 snapshots. Cleaned OCR PDF data from the **Manipuri Corpus & OCR Pipeline** using regex page marker cleaners and Latin character density thresholds.
- **Phase 6 (Pretraining Preparation & Benchmark Suite)**: Packed tokenized sequences into uniform 512-token context windows for SmolLM-135M Causal LM training. Built held-out benchmark datasets for Perplexity, Translation, Script Conversion, and OCR Restoration.
- **Phase 7 (Foundation Pretraining & Modular Evaluation)**: Executed 3 full epochs (13,596 global steps). Implemented a 10-step modular evaluation suite (`run_phase7_eval.py`) covering training analysis, script-wise perplexity, next-token prediction, multi-decoding strategies, script consistency, memorization testing, speed profiling, and report generation.
- **Base Model Release Freezing**: Promoted the step-13,596 checkpoint to `ManipuriGPT-135M-Base-v1.0`, computed SHA256 checksums, generated `manifest.json`, and published the base model weights, dataset, and tokenizer to the Hugging Face Hub.

---

## 3. Data Sources & Corpus Catalog

| Source Name | Provider / Repository | Type | Primary Script | License | Description / Attribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ManipuriGPT-Corpus-v1.0** | Hugging Face: `nanskong/ManipuriGPT-Corpus-v1.0` | Monolingual | Multi-Script | CC-BY-NC-4.0 | Frozen, deduplicated, quality-scored master pretraining corpus. |
| **IndicCorp V2 (mni)** | Hugging Face: `ai4bharat/IndicCorpV2` | Monolingual | Bengali | CC-BY-4.0 | AI4Bharat large-scale Indic text corpus for Manipuri. |
| **Sangraha / Joyson** | Hugging Face: `joyson117/manipuri-monolingual-corpus` | Monolingual | Bengali | Research Only | News and web corpus for Manipuri. |
| **Dayananda Meitei Mayek** | Hugging Face: `DayanandaThokchom/meitei_mayek_sample` | Monolingual | Meitei Mayek | CC-BY-4.0 | Native Meitei Mayek text corpus by Dayananda Thokchom. |
| **Dayananda Parallel** | Hugging Face: `DayanandaThokchom/meitei-mayek-to-english` | Parallel | Meitei Mayek | CC-BY-4.0 | Meitei Mayek to English parallel translation corpus. |
| **FLORES+ (mni_Beng)** | Hugging Face: `openlanguagedata/flores_plus` | Translation | Bengali | CC-BY-SA-4.0 | FLORES+ benchmark split for Manipuri (Bengali script). |
| **Manipuri Corpus & OCR Pipeline** | GitHub: [`NansKong/Manipuri_Corpus`](https://github.com/NansKong/Manipuri_Corpus) | Monolingual | Mixed | Proprietary/Academic | 15 processed PDFs (4,158 pages, 857,464 words) of Manipuri books, dictionaries, and scanned archives. |

---

## 4. Summary of Evaluation Metrics (Phase 7)

```text
======================================================================
MANIPURIGPT FOUNDATION MODEL EVALUATION SUMMARY
======================================================================
Model Identifier          : ManipuriGPT-135M-Base-v1.0
Base Architecture         : SmolLM-135M Causal LM
Pretraining Steps         : 13,596 Global Steps (3.0 Epochs)
Final Training Loss       : 3.6246
Final Validation Loss     : 4.2799
Best Validation Loss      : 4.2693

PERPLEXITY DIAGNOSTICS:
  - Overall Perplexity    : 244.28
  - Meitei Mayek PPL      : 133.14 (Primary Target Script)
  - Bengali Script PPL    : 1508.89

TOKENIZER HEALTH:
  - Vocabulary Size       : 8,000 subwords (SentencePiece BPE)
  - Unknown (<unk>) Rate  : 0.0000% (Zero <unk> emissions)
  - Compression Ratio     : 7.125 bytes / token

GENERATION DIVERSITY & SPEED:
  - Distinct-1 Score      : 0.8683
  - Distinct-2 Score      : 0.9849
  - Self-BLEU Score       : 4.22
  - Throughput (CPU)      : 9.91 tokens / second
======================================================================
```

---

## 5. Official Hugging Face Repositories

- **Pretrained Base Model**: [nanskong/ManipuriGPT-135M-Base-v1.0](https://huggingface.co/nanskong/ManipuriGPT-135M-Base-v1.0)
- **Subword Tokenizer**: [nanskong/ManipuriGPT-Tokenizer-v1](https://huggingface.co/nanskong/ManipuriGPT-Tokenizer-v1)
- **Pretraining Dataset**: [nanskong/ManipuriGPT-Corpus-v1.0](https://huggingface.co/datasets/nanskong/ManipuriGPT-Corpus-v1.0)

---

## 6. Academic Citations & BibTeX Registry

If you use ManipuriGPT models, tokenizers, datasets, or evaluation tools in your research, please cite the corresponding entries below:

```bibtex
@misc{manipurigpt_base_v10_2026,
  author       = {ManipuriGPT Team},
  title        = {ManipuriGPT-135M-Base-v1.0: Research-Grade Open Foundation Model for Manipuri},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/nanskong/ManipuriGPT-135M-Base-v1.0}}
}

@misc{manipurigpt_tokenizer_v10_2026,
  author       = {ManipuriGPT Team},
  title        = {ManipuriGPT-Tokenizer-v1.0: SentencePiece BPE Tokenizer for Manipuri},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/nanskong/ManipuriGPT-Tokenizer-v1}}
}

@misc{manipurigpt_corpus_v10_2026,
  author       = {ManipuriGPT Team},
  title        = {ManipuriGPT Corpus v1.0: A Research-Grade Open Foundation Model Corpus for the Manipuri Language},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/nanskong/ManipuriGPT-Corpus-v1.0}}
}

@inproceedings{kudo-richardson-2018-sentencepiece,
  title     = "{S}entence{P}iece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing",
  author    = "Kudo, Taku  and  Richardson, John",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
  year      = "2018",
  pages     = "66--71"
}

@misc{ai4bharat_indiccorp_v2,
  title        = {IndicCorp V2: Large-Scale Multilingual Corpora for Indic Languages},
  author       = {AI4Bharat Team},
  year         = {2023},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/ai4bharat/IndicCorpV2}}
}

@misc{flores_plus_2024,
  title        = {FLORES+: Open Multilingual Translation Evaluation Benchmark},
  author       = {Open Language Data Initiative},
  year         = {2024},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/openlanguagedata/flores_plus}}
}
```
