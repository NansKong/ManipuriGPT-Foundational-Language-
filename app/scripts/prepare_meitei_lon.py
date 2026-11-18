"""Prepare the local three-script training release from the frozen OCR-reviewed inputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq

from app.sft.splits import canonical, grouped_split
from app.training.meitei_lon import CORPUS, REVIEW, ROOT, SFT_DATA, TOKENIZER, TRAINING_DATA, export_tokenizer


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def write_parquet(path, table):
    temp = path.with_suffix(".tmp")
    pq.write_table(table, temp)
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def course_records():
    pages = json.loads((ROOT / "app/corpus/romanized_course.json").read_text(encoding="utf-8"))
    for page, lines in pages.items():
        for index, (text, *translation) in enumerate(lines):
            row = dict(text=text, source="M024", source_id=f"M024/p{page}/{index}",
                       pdf_page=int(page), script="Latn", language="mni", language_script="mni-Latn",
                       label_evidence="checked_against_scan", unit="sentence", native_form=None)
            yield row, translation


def translation_rows(row, english):
    for src, tgt, direction, target in [
        (row["text"], english, "latn_to_en", "English"),
        (english, row["text"], "en_to_latn", "Romanized Meitei-lon"),
    ]:
        yield dict(id=f"{row['source_id']}/{direction}", instruction=f"Translate into {target}.",
                   input=src, output=tgt, task_type=f"translation_{direction}", document="M024",
                   source=row["source_id"], language="mni", pdf_page=row["pdf_page"])


def surfaces(rows):
    return {canonical(r[field]) for r in rows for field in ("input", "output") if r[field]}


def identification_rows(table, split, blocked):
    rng = random.Random(554)
    by_script = {}
    seen = set(blocked)
    for row in table.select(["text", "language_script", "source_id", "unit"]).to_pylist():
        label = row["language_script"]
        if label not in ("mni-Mtei", "mni-Beng", "mni-Latn") or len(row["text"]) > 450:
            continue
        key = canonical(row["text"])
        if not key or key in seen:
            continue
        seen.add(key)
        by_script.setdefault(label, []).append(row)
    result = []
    for label in ("mni-Mtei", "mni-Beng", "mni-Latn"):
        pool = by_script[label]
        rng.shuffle(pool)
        pool.sort(key=lambda r: r["unit"] in ("word", "headword"))
        for row in pool[:256 if split == "train" else 50]:
            result.append(dict(id=f"identify/{split}/{row['source_id']}", input=row["text"], output=label,
                               instruction="Identify the writing system of this Meitei-lon text. Answer with mni-Mtei, mni-Beng or mni-Latn.",
                               task_type="script_identification", source=row["source_id"], language="mni"))
    return result


def main():
    export_tokenizer()
    out = TRAINING_DATA / "corpus/data"
    out.mkdir(parents=True, exist_ok=True)
    tables = {s: pq.read_table(CORPUS / "data" / f"{s}.parquet") for s in ("train", "validation", "test")}
    feedback_path = ROOT / "app/corpus/native_speaker_feedback.json"
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    # Keep source transcriptions intact. Apply speaker edits to the derived release.
    rows = tables["test"].to_pylist()
    for row in rows:
        if row["source"] == "M024" and row["pdf_page"] == 41 and row["text"] == "achu\\mbara ?":
            row["text"] = "achumbara ?"
            row["label_evidence"] = "native_speaker_corrected"
    tables["test"] = pa.Table.from_pylist(rows, schema=tables["test"].schema)
    existing = {canonical(t) for table in tables.values() for t in table["text"].to_pylist()}
    additions, bilingual, reference = [], [], []
    for row, gloss in course_records():
        if row["pdf_page"] == 48:
            if row["source_id"] == "M024/p48/1":
                row["text"] = row["text"].replace("machaaka", "machaga")
            reference.extend(translation_rows(row, gloss[0]))
        elif canonical(row["text"]) not in existing:
            additions.append(row)
            existing.add(canonical(row["text"]))
            if gloss:
                bilingual.extend(translation_rows(row, gloss[0]))
    speaker_reference = []
    for item in feedback["examples"]:
        if "preferred_meitei_mayek" in item:
            speaker_reference.append(dict(
                id=f"native_review/{item['example']}/en_to_mtei",
                instruction="Translate into Meitei-lon using Meitei Mayek.",
                input=item["english"], output=item["preferred_meitei_mayek"],
                task_type="translation_en_to_mm", source=item["id"],
                review_status="native_speaker_corrected", language="mni"))
    speaker_reference.append(dict(id="native_review/7/latn_to_en",
        instruction="Translate into English.", input="achumbara", output="Is it true?",
        task_type="translation_latn_to_en", source="M024/p41/36",
        review_status="native_speaker_corrected", language="mni"))
    conversation_reference = []
    for item in feedback["examples"]:
        if "preferred_conversational_reply" in item:
            source_id = item["id"].removeprefix("identify/test/")
            original = next(row for row in rows if row["source_id"] == source_id)
            conversation_reference.append(dict(
                id=f"native_review/{item['example']}/conversation",
                instruction="Reply naturally in Meitei-lon using Bengali script.",
                input=original["text"], output=item["preferred_conversational_reply"],
                task_type="conversation", source=source_id, language="mni",
                input_meaning=item["input_meaning"], reply_meaning=item["reply_meaning"],
                reply_romanized=item["reply_romanized"], review_status="native_speaker_corrected"))
            speaker_reference.append(dict(
                id=f"native_review/{item['example']}/beng_to_en",
                instruction="Translate into English.", input=original["text"],
                output=item["input_meaning"], task_type="translation_bn_to_en",
                source=source_id, language="mni", review_status="native_speaker_corrected"))
    heldout = surfaces(reference + speaker_reference + conversation_reference)
    for s in ("validation", "test"):
        heldout.update(map(canonical, tables[s]["text"].to_pylist()))
    candidates = []
    for split in ("train", "val"):
        path = REVIEW / f"merged_sft_v1_{split}_ocr_reviewed.jsonl"
        with path.open(encoding="utf-8") as stream:
            candidates.extend(json.loads(line) for line in stream if line.strip())
    if not candidates or not reference:
        raise ValueError("Missing reviewed SFT or page-aligned reference data")
    candidates = [r for r in candidates if not surfaces([r]) & heldout]
    # A single-word answer aligned to a long sentence often means only a fragment
    # was copied. Preserve both directions for review without guessing a correction.
    review_queue = []
    approved = []
    for row in candidates:
        lengths = sorted(len(s.split()) for s in (row["input"], row["output"]))
        if row["task_type"].startswith("translation") and lengths[0] <= 2 and lengths[1] >= 10:
            review_queue.append(dict(**row, review_reason="possible_truncated_alignment"))
        else:
            approved.append(row)
    write_jsonl(TRAINING_DATA / "translation_alignment_review.jsonl", review_queue)
    train, val = grouped_split(approved)
    blocked = heldout | surfaces(val)
    train.extend(r for r in bilingual if not surfaces([r]) & blocked)
    def translation_pairs(rows):
        return {tuple(sorted((canonical(r["input"]), canonical(r["output"])))) for r in rows
                if r["task_type"].startswith("translation")}
    overlap = translation_pairs(train) & (translation_pairs(val) | translation_pairs(reference + speaker_reference))
    if overlap:
        raise ValueError(f"{len(overlap)} translation pairs shared with held-out splits")
    training_table = pa.concat_tables([tables["train"], pa.Table.from_pylist(additions, schema=tables["train"].schema)])
    tables["train"] = training_table.filter(pa.array([canonical(t) not in blocked for t in training_table["text"].to_pylist()]))
    for name, table in tables.items():
        write_parquet(out / f"{name}.parquet", table)
    identification = identification_rows(tables["train"], "train", blocked)
    train.extend(identification)
    train_texts = set(map(canonical, tables["train"]["text"].to_pylist())) | surfaces(train)
    identification_test = identification_rows(tables["test"], "test", train_texts | surfaces(val))
    files = {"train": train, "validation": val, "romanized_translation_test": reference,
             "native_speaker_translation_test": speaker_reference,
             "native_speaker_conversation_test": conversation_reference,
             "script_identification_test": identification_test}
    for name, rows in files.items():
        write_jsonl(SFT_DATA / f"{name}.jsonl", rows)
    manifest = {
        "primary_language": "Meitei-lon (mni)",
        "translation_alignment_review_rows": len(review_queue),
        "shared_translation_pairs_train_vs_heldout": len(overlap),
        "corpus_base_revision": "nanskong/ManipuriGPT-Corpus-v1.1@f8581dfedd5d32347d86eef6f3020ab0f33b9783",
        "source_files": {str(p.relative_to(ROOT)): sha(p) for p in
                         [ROOT / "app/corpus/romanized_course.json", feedback_path] +
                         [REVIEW / f"merged_sft_v1_{s}_ocr_reviewed.jsonl" for s in ("train", "val")]},
        "tokenizer_sha256": sha(TOKENIZER / "tokenizer.json"),
        "corpus": {s: {"rows": len(t), "sha256": sha(out / f"{s}.parquet")} for s,t in tables.items()},
        "sft": {s: {"rows": len(rows), "sha256": sha(SFT_DATA / f"{s}.jsonl")} for s,rows in files.items()},
        "romanized_training_sentences": sum(1 for tag, unit in zip(tables["train"]["language_script"].to_pylist(),
            tables["train"]["unit"].to_pylist()) if tag == "mni-Latn" and unit == "sentence"),
        "identification_test_by_class": dict(Counter(r["output"] for r in identification_test)),
    }
    dest = TRAINING_DATA / "manifest.json"
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp.replace(dest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
