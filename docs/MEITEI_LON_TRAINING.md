# Meitei-lon training update

18 November 2026

Meitei-lon is the primary language. The assistant prompt defaults to Meitei Mayek, matches Bengali-script and Romanized Meitei-lon input, and follows explicit requests for another language or script.

## Tokenizer and model

Use the existing 135M-named base as the starting checkpoint. Its learned vocabulary contains 32,000 IDs. The archived first 2,048-token pretraining block matches the frozen SentencePiece encoder, including document-ending EOS tokens.

The new export in `artifacts/tokenizers/Meitei-lon-32k-sp-compatible` reproduces that encoder's normalization and initial word marker. All 153,882 v1.1 corpus rows produce identical token sequences. Three-script fixtures, whitespace, control-token text and language-marker cases are tested. Packing uses the same frozen SentencePiece model. New SFT exports carry their tokenizer for inference.

The existing v1.1 adapter was evaluated with its original encoding, corrected padding metadata and the new Meitei-lon prompt. It returned zero correct labels on 150 script-identification examples, with 50 per script. Its 26 Romanized translation outputs had zero normalized exact matches. Recorded outputs include repetition and Meitei Mayek answers to explicit Romanized requests. These results establish a baseline for the new tasks. Exact matching alone cannot assess translation meaning or acceptable spelling variants.

Continue pretraining and then train a fresh LoRA adapter using the repaired data. The one-step checks establish that the checkpoint and optimizer work. Fluent-reader evaluation is still needed to establish usable language quality.

## Four implemented changes

1. **Training inputs and execution.** The general launcher defaults to the local Manipuri base, repaired tokenizer and prepared three-script corpus. The SFT launcher selects regrouped OCR-reviewed records. Production loading, initialization and saving failures propagate. Explicit dry runs use a separate output subdirectory. Input hashes and release split sizes are recorded and checked on resume.
2. **Consistent encoding.** Fast encoding matches frozen SentencePiece. Vocabulary size mismatches fail for the Manipuri preset. Corpus examples end with EOS, packing retains tails, and SFT supervision preserves answer tokens when a prompt exceeds the context limit. Assistant responses train the existing EOS ID.
3. **Held-out evaluation.** Translation directions, shared sentence variants and similar aligned pairs are grouped before splitting. Book-derived SFT tasks use whole-book holdouts with approximate per-task balance. Exact held-out text is excluded from the new continued-pretraining input. The benchmark builder uses held-out records with explicit translation and conversion references. Missing reference sets are marked unavailable. Synthetic OCR corruption remains labeled as synthetic.
4. **Romanized text and recognition.** Training now contains 52 Romanized sentence rows, up from 24. The 28 additions were transcribed against M024 PDF pages 46–47. Fourteen printed bilingual examples supply both translation directions. Page 48 supplies 13 reserved pairs in both directions. There are 768 script-identification training examples and 150 reserved test examples.

The textbook's spelling and tone-mark backslashes are retained. This small addition needs expansion with fluent-speaker-checked everyday writing and spelling variants. The identification task explicitly assumes Meitei-lon input. Distinguishing Romanized Manipuri from English, and Bengali-script Manipuri from Bengali-language text, requires a separate labeled challenge set.

## Data quality and provenance

The local release is built from corpus v1.1 revision `f8581dfedd5d32347d86eef6f3020ab0f33b9783`, the OCR-reviewed SFT copies and `app/corpus/romanized_course.json`. The manifest records exact file hashes and counts.

The preparation script sets aside 776 existing translation rows whose alignment has at least ten words on one side and at most two on the other. They remain available in `artifacts/training/meitei-lon/translation_alignment_review.jsonl`. This heuristic identifies candidates for review. It does not prove that an alignment is wrong. Checked textbook alignments are retained.

Source-book grouping protects the new SFT split. The already-trained v1.0 base has historical exposure to the original corpus. New splits cannot undo that exposure. Nearby-edit grouping is heuristic and broader semantic duplication remains possible.

## Commands

The preparation script expects the corpus v1.1 Parquet splits under `artifacts/datasets/ManipuriGPT-Corpus-v1.1/data` and the OCR-reviewed instruction files `merged_sft_v1_train_ocr_reviewed.jsonl` and `merged_sft_v1_val_ocr_reviewed.jsonl` under `artifacts/book_ocr_review`. It also reads the local inputs `app/corpus/romanized_course.json` and `app/corpus/native_speaker_feedback.json`. The separate corpus release builder reads `app/corpus/reviewed_book_excerpts.jsonl`. Obtain these review inputs from the project maintainer if preparing the data on another machine.

Place the local 135M base checkpoint and its tokenizer export in `models/ManipuriGPT-135M-Base-v1.0`. The comparison also uses the existing adapter at `models/ManipuriGPT-SFT-v1.1`.

Run from the project root with the existing project interpreter. The check verifies prepared file hashes and required model files. Each training stage starts only when invoked explicitly. Both stages default to one epoch and accept `--max-steps` for a shorter pilot.

```powershell
& .\manipuri\Scripts\python.exe -m app.scripts.run_meitei_lon check
& .\manipuri\Scripts\python.exe -m app.scripts.run_meitei_lon pretrain
& .\manipuri\Scripts\python.exe -m app.scripts.run_meitei_lon fine-tune
& .\manipuri\Scripts\python.exe -m app.scripts.run_meitei_lon evaluate
```

The run directory defaults to `artifacts/models/meitei-lon-v1.2`. Pass the same `--run-dir` to all stages if you use another directory. Use a new directory for each experiment. Fine-tuning initializes a fresh LoRA adapter on the continued-pretraining model and uses the reviewed instruction data.

Evaluation generates candidate and v1.1 baseline answers on the same held-out data. The run's `evaluation/candidate` and `evaluation/baseline` directories contain predictions, references and exact-match scores. Review the answers for meaning, naturalness and script. The native-speaker examples remain held out for this comparison.

If a prepared split changes, rebuild it with `python -m app.scripts.prepare_meitei_lon` using the project interpreter.

The continued-pretraining preset uses a conservative learning rate of `1e-5`, batch size 1, gradient accumulation 8 and context length 512. These are initial implementation choices requiring validation-loss monitoring. Full training is a separate run from the one-step verification.

The dedicated SFT trainer uses LoRA. The main trainer's separate QLoRA mode requires its quantization dependencies.

The dependency constraints record the existing project environment. Its interpreter is `D:\ManipuriGPT\manipuri\Scripts\python.exe`. It contains PyTorch 2.5.1+cu121, Transformers 5.14.0, PEFT 0.19.1 and Accelerate 1.14.0. `pip check` reports no broken requirements.

Fresh continued-pretraining and SFT each passed a real optimizer step, changed saved weights and reloaded with finite outputs. Sixteen focused tests passed. Additional checks cover empty documents and one-token packing tails. The existing environment can execute these paths.

The normal training CLI was also verified against all 145,121 prepared training rows with a one-step limit. It evaluated 4,340 validation rows, saved changed attention weights and reloaded with finite outputs. The selected weight's maximum change was approximately `1.05e-5`. Evidence is saved in `artifacts/training/meitei-lon/training_verification.json` and `verification.log`.

**Optimizer resume remains blocked by the installed Transformers safety check.** It requires PyTorch 2.6 or newer when loading optimizer state. Input-hash matching was verified separately, including rejection of changed settings. A PyTorch download was stopped after the user asked to inspect the existing environments. Package installation did not run.

The corpus environment at `E:\manipuri corpus\corpa` has PyTorch 2.5.1+cu121, Transformers 5.14.1, EasyOCR and pytesseract. PEFT, Datasets and SentencePiece are absent there. Use the training project's `manipuri` environment for the training commands above.

The installed packages retain their loading checks. A future optimizer-resume environment update can use the matching versions from the official installation matrix. [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/)

## Artifacts

- Prepared corpus and SFT data live in `artifacts/training/meitei-lon`.
- `manifest.json` records dataset hashes, counts and translation-pair overlap.
- `evaluation/ManipuriGPT-SFT-v1.1` contains baseline predictions and exact-match counts.
- `app/corpus/romanized_course.json` preserves transcriptions and printed glosses with PDF page identifiers.
