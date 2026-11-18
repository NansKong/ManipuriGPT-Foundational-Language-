"""Build the v1.1 three-script corpus from the pinned v1.0 shards and source data."""
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

BASE = Path("artifacts/datasets/ManipuriGPT-Corpus-v1.0")
OUTPUT = BASE.with_name("ManipuriGPT-Corpus-v1.1")
AK = Path("cache/ocr_release_inputs/aksharantar/mni.zip")
SPLITS = ("test", "validation", "train")
SCHEMA = pa.schema([(k, pa.string()) for k in (
    "text", "language", "script", "language_script", "source", "source_id",
    "label_evidence", "unit", "native_form"
)] + [("pdf_page", pa.int32())])
OCR = {"d_drive_manipuri_corpus_processed", "local_processed_pdfs"}


def norm(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def matches_script(text, script):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters or "\ufffd" in text or any(unicodedata.category(c) in {"Co", "Cn"} for c in text):
        return False
    if script == "Mtei":
        return all("\uabc0" <= c <= "\uabff" or "\uaae0" <= c <= "\uaaff" for c in letters)
    if script == "Beng":
        return all("\u0980" <= c <= "\u09ff" for c in letters)
    return all("LATIN" in unicodedata.name(c, "") for c in letters)


def bengali_reference(candidates):
    matched = set()
    for family, column in (("english-manipuri-parallel-corpus", "Manipuri"), ("manipuri-monolingual-corpus", "text")):
        files = list((Path("cache/datasets") / ("joyson117___" + family)).rglob("*.arrow"))
        if not files:
            raise FileNotFoundError(f"Missing reference corpus {family}")
        for path in files:
            with pa.memory_map(str(path)) as stream:
                for batch in ipc.open_stream(stream):
                    for value in batch.column(column).to_pylist():
                        if value:
                            key = norm(value)
                            if key in candidates:
                                matched.add(key)
    return matched


def build():
    rows = {s: [] for s in SPLITS}
    seen = set()
    exclusions = Counter()

    def add(split, text, script, source, key, evidence, unit, native=None, page=None):
        text = norm(text)
        if not matches_script(text, script):
            exclusions["script_or_encoding"] += 1
            return
        normalized = text.casefold()
        if normalized in seen:
            exclusions["duplicate"] += 1
            return
        seen.add(normalized)
        rows[split].append(dict(text=text, language="mni", script=script, language_script="mni-" + script,
                                source=source, source_id=key, label_evidence=evidence, unit=unit,
                                native_form=native, pdf_page=page))

    # Verify the frozen input files against the release manifest.
    for line in (BASE / "checksums/parquet_shards.sha256").read_text().splitlines():
        digest, filename = line.split()
        if hashlib.sha256((BASE / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source shard changed: {filename}")
    old = {s: [] for s in SPLITS}
    for split in SPLITS:
        for file in sorted((BASE / split).glob("*.parquet")):
            old[split].extend((r, f"{split}/{file.name}#{i}") for i, r in enumerate(pq.read_table(file).to_pylist()))
    candidates = {norm(r["text"]) for values in old.values() for r, _ in values if r["script"] == "bengali"}
    bengali = bengali_reference(candidates)
    verified = [json.loads(l) for l in Path("app/corpus/reviewed_book_excerpts.jsonl").read_text("utf8").splitlines()]
    native_splits = {}
    with zipfile.ZipFile(AK) as archive:
        for split in SPLITS:
            suffix = "valid" if split == "validation" else split
            for line in archive.open(f"mni_{suffix}.json"):
                r = json.loads(line)
                native, roman = norm(r["native word"]), norm(r["english word"])
                if r["source"] not in {"AK-Freq", "IndicCorp"} or not matches_script(native, "Mtei") or not re.fullmatch("[a-zA-Z]+", roman):
                    continue
                if native_splits.setdefault(native, split) != split:
                    continue
                add(split, roman, "Latn", "ai4bharat/Aksharantar", f"{suffix}/{r['unique_identifier']}",
                    r["source"], "word", native=native)
            for i, v in enumerate(verified):
                page_split = {39: "validation", 41: "test"}.get(v["page"], "train")
                if split == page_split:
                    add(split, v["text"], v["tag"].split("-")[1], v["document"], f"{v['document']}/p{v['page']}/{i}",
                        "checked_against_scan", v["kind"], page=v["page"])
            for r, key in old[split]:
                if r["source"] in OCR:
                    exclusions["legacy_book_ocr_without_page_provenance"] += 1
                    continue
                if r["script"] == "meitei" and r["language"] == "mni":
                    script, evidence = "Mtei", "v1.0_language_and_script"
                elif r["script"] == "bengali" and norm(r["text"]) in bengali:
                    script, evidence = "Beng", "exact_manipuri_source_match"
                elif r["script"] == "bengali" and r["language"] == "mni" and r["source"] in {"local_ema_lon_mono","local_ema_lon_parallel","joyson_monolingual","joyson_pib_pmi"}:
                    script, evidence = "Beng", "named_manipuri_source"
                else:
                    exclusions["unresolved_language_or_mixed_script"] += 1
                    continue
                add(split, r["text"], script, "nanskong/ManipuriGPT-Corpus-v1.0", key, evidence, "passage")
    counts, hashes = {}, {}
    for split, records in rows.items():
        counts[split] = dict(Counter(r["language_script"] for r in records))
        assert set(counts[split]) == {"mni-Latn", "mni-Mtei", "mni-Beng"}
        path = OUTPUT / "data" / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), path, compression="zstd")
        assert pq.read_table(path).to_pylist() == records
        hashes[path.relative_to(OUTPUT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(counts=counts, sha256=hashes, exclusions=dict(exclusions),
                  sources={"v1.0":"729b6202f26b80348c0e1218e61231a4d7aad3fc","Aksharantar":"e418c1fc928d9f5393af33268472cf20c1891be8"})
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    build()
