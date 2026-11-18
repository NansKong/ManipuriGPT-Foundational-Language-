from pathlib import Path
import hashlib
import json
import logging

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/ManipuriGPT-135M-Base-v1.0"
CORPUS = ROOT / "artifacts/datasets/ManipuriGPT-Corpus-v1.1"
TOKENIZER = ROOT / "artifacts/tokenizers/Meitei-lon-32k-sp-compatible"
TRAINING_DATA = ROOT / "artifacts/training/meitei-lon"
SFT_DATA = TRAINING_DATA / "sft"
REVIEW = ROOT / "artifacts/book_ocr_review"


def adapter_tokenizer(adapter, base=BASE, **kwargs):
    """Old adapters declared an out-of-vocabulary ChatML pad token in their HF config."""
    from transformers import AutoTokenizer
    path = Path(adapter)
    tokenizer = AutoTokenizer.from_pretrained(path if (path / "tokenizer.json").exists() else base,
        pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>", **kwargs)
    if len(tokenizer) != 32000 or tokenizer.pad_token_id != 0 or tokenizer.eos_token_id != 3:
        raise ValueError("Adapter tokenizer is incompatible with the 32,000-row ManipuriGPT base")
    return tokenizer


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_run_inputs(output, resume, model_dir, tokenizer_dir, splits, settings):
    """Compare actual local inputs with the original run before resuming its optimizer."""
    def signatures(folder, names):
        return {p.name: file_digest(p) for name in names for p in sorted(Path(folder).glob(name))}

    signature = {
        "base": signatures(model_dir, ("config.json", "*.safetensors")),
        "tokenizer": signatures(tokenizer_dir, ("tokenizer.json", "tokenizer_config.json")),
        "data": {p.name: file_digest(p) for p in splits},
        "settings": settings,
    }
    if not signature["base"] or not signature["tokenizer"]:
        raise ValueError("The Meitei-lon preset requires a local model and tokenizer")
    manifest = TRAINING_DATA / "manifest.json"
    signature["release"] = json.loads(manifest.read_text(encoding="utf-8"))
    for path in splits:
        section = "corpus" if path.suffix == ".parquet" else "sft"
        expected = signature["release"][section][path.stem]["sha256"]
        if signature["data"][path.name] != expected:
            raise ValueError(f"Prepared split changed since manifest creation: {path}")
    if signature["tokenizer"].get("tokenizer.json") != signature["release"]["tokenizer_sha256"]:
        raise ValueError("Tokenizer export changed since manifest creation")
    name = "meitei_lon_run_inputs.json"
    if resume:
        original = Path(resume).resolve().parent / name
        if not original.is_file() or json.loads(original.read_text(encoding="utf-8")) != signature:
            raise ValueError("Checkpoint data, model, tokenizer or training settings differ from this run")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != signature:
        raise ValueError("Output directory belongs to a different training run. Select a fresh output directory.")
    path.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.getLogger("manipurigpt").info("Recorded input hashes and split sizes in %s", path)


def export_tokenizer():
    """Preserve the learned BPE and reproduce SentencePiece's ASCII-space handling."""
    from tokenizers import Tokenizer, Regex, normalizers, decoders, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    data = json.loads((BASE / "tokenizer.json").read_text(encoding="utf-8"))
    data["added_tokens"] = [token for token in data["added_tokens"] if token["id"] < 4]
    data["model"]["ignore_merges"] = True
    engine = Tokenizer.from_str(json.dumps(data))
    engine.pre_tokenizer = pre_tokenizers.Split(Regex("<meitei>|<bengali>|<romanized>|<mask"), behavior="isolated")
    engine.no_truncation()
    engine.no_padding()
    engine.normalizer = normalizers.Sequence([
        normalizers.Replace(Regex("^[ ▁]+|[ ▁]+$"), ""),
        normalizers.Replace(Regex("[ ▁]+"), "▁"),
        normalizers.Prepend("▁"),
    ])
    engine.decoder = decoders.Sequence([
        decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(),
        decoders.Strip(" ", 1, 0),
    ])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=engine, bos_token="<s>",
        eos_token="</s>", pad_token="<pad>", unk_token="<unk>", model_max_length=8192,
        split_special_tokens=True)
    fixtures = {"ꯃꯩꯇꯩ": [5704, 6200], "মণিপুরী": [1239],
                "hidaak layge.": [632, 6240, 532, 26030, 909, 31723]}
    for text, expected in fixtures.items():
        if tokenizer.encode(text, add_special_tokens=False) != expected:
            raise ValueError(f"Tokenizer export failed the SentencePiece fixture for {text!r}")
    tokenizer.save_pretrained(TOKENIZER)
    return TOKENIZER


if __name__ == "__main__":
    print(export_tokenizer())
