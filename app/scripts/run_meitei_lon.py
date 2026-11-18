"""Run the reviewed Meitei-lon training stages in an isolated local directory."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from app.training.meitei_lon import BASE, ROOT, SFT_DATA, TOKENIZER, TRAINING_DATA, file_digest


def check_inputs():
    manifest = json.loads((TRAINING_DATA / "manifest.json").read_text(encoding="utf-8"))
    corpus = TRAINING_DATA / "corpus" / "data"
    required = {
        "corpus": {"train", "validation", "test"},
        "sft": {"train", "validation", "script_identification_test", "romanized_translation_test",
                "native_speaker_translation_test", "native_speaker_conversation_test"},
    }
    for section, directory, suffix in (("corpus", corpus, ".parquet"), ("sft", SFT_DATA, ".jsonl")):
        missing = required[section] - manifest[section].keys()
        if missing:
            raise ValueError(f"Manifest is missing {section} splits: {', '.join(sorted(missing))}")
        for split, entry in manifest[section].items():
            path = directory / f"{split}{suffix}"
            if entry["rows"] <= 0 or not path.is_file() or file_digest(path) != entry["sha256"]:
                raise ValueError(f"Prepared split missing, empty or changed: {path}. Run app.scripts.prepare_meitei_lon.")
    if manifest["shared_translation_pairs_train_vs_heldout"]:
        raise ValueError("The prepared release contains shared training and held-out translation pairs.")
    if file_digest(TOKENIZER / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("The prepared tokenizer differs from the release manifest.")
    require_files(TOKENIZER, "tokenizer_config.json")
    require_files(BASE, "config.json", "model.safetensors")
    return manifest


def require_files(directory, *names):
    for name in names:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty required file: {path}")


def require_fresh_output(directory):
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f"Choose an unused run directory. Output already exists: {directory}")


def run_module(module, *args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT, check=True)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be at least 1.")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="check", choices=("check", "pretrain", "fine-tune", "evaluate"))
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/models/meitei-lon-v1.2")
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--max-steps", type=positive_int, help="Optional optimizer-step limit for a pilot")
    args = parser.parse_args(argv)
    manifest = check_inputs()
    if args.stage == "check":
        print(f"Verified {manifest['corpus']['train']['rows']:,} corpus rows and "
              f"{manifest['sft']['train']['rows']:,} instruction rows against the prepared manifest.")
        print(f"Base: {BASE}\nTokenizer: {TOKENIZER}\nRun directory: {args.run_dir.resolve()}")
        return

    run_dir = args.run_dir.resolve()
    cpt = run_dir / "continued-pretraining"
    base = cpt / "final_model"
    adapter = run_dir / "adapter"
    limits = ["--max-steps", args.max_steps] if args.max_steps is not None else []
    if args.stage == "pretrain":
        require_fresh_output(cpt)
        run_module("app.scripts.train", "--mode", "continued_pretraining",
                   "--model-name-or-path", BASE, "--tokenizer-path", TOKENIZER,
                   "--dataset-source", TRAINING_DATA / "corpus", "--output-dir", cpt,
                   "--epochs", args.epochs, *limits)
        return

    require_files(base, "config.json", "model.safetensors")
    require_files(cpt, "meitei_lon_run_inputs.json")
    run_inputs = json.loads((cpt / "meitei_lon_run_inputs.json").read_text(encoding="utf-8"))
    if run_inputs["release"]["corpus"] != manifest["corpus"] or run_inputs["tokenizer"]["tokenizer.json"] != manifest["tokenizer_sha256"]:
        raise ValueError("The continued-pretraining checkpoint used a different corpus or tokenizer.")
    if args.stage == "fine-tune":
        require_fresh_output(adapter)
        run_module("app.scripts.run_sft_train", "--base-model", base, "--output-dir", adapter,
                   "--epochs", args.epochs, *limits)
        return

    require_files(adapter, "adapter_config.json", "adapter_model.safetensors", "tokenizer.json")
    models = (
        ("candidate", base, adapter),
        ("baseline", BASE, ROOT / "models/ManipuriGPT-SFT-v1.1"),
    )
    for _, _, model_adapter in models:
        require_files(model_adapter, "adapter_config.json", "adapter_model.safetensors")
    for name, model_base, model_adapter in models:
        run_module("app.scripts.evaluate_meitei_lon", "--base", model_base, "--adapter", model_adapter,
                   "--output-dir", run_dir / "evaluation" / name)


if __name__ == "__main__":
    main()
