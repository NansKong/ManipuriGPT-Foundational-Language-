"""Generate predictions on the reserved script identification and textbook translation pages."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.sft.prompt_formatter import format_chatml_record
from app.sft.splits import canonical
from app.training.meitei_lon import BASE, SFT_DATA, TRAINING_DATA, adapter_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, help="Directory for predictions and scores")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = adapter_tokenizer(args.adapter, args.base, padding_side="left", local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16 if device == "cuda" else torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).eval().to(device)
    dest = args.output_dir or TRAINING_DATA / "evaluation" / args.adapter.name
    dest.mkdir(parents=True, exist_ok=True)
    n, correct = Counter(), Counter()
    for task in ("script_identification_test", "romanized_translation_test",
                 "native_speaker_translation_test", "native_speaker_conversation_test"):
        rows = [json.loads(line) for line in (SFT_DATA / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()]
        if not rows:
            raise ValueError(f"No references in {task}")
        with (dest / f"{task}.jsonl").open("w", encoding="utf-8") as stream, torch.inference_mode():
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8]
                prompts = [format_chatml_record(r["instruction"], r["input"])["prompt_text"] for r in batch]
                inputs = tok(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=16 if task.startswith("script") else 48,
                                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                predictions = tok.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                for row, prediction in zip(batch, predictions):
                    prediction = prediction.split("<|im_end|>")[0].strip()
                    label = row["output"] if task.startswith("script") else row["task_type"]
                    match = canonical(prediction) == canonical(row["output"])
                    n[label] += 1
                    correct[label] += int(match)
                    stream.write(json.dumps(dict(**row, prediction=prediction, normalized_exact_match=match), ensure_ascii=False) + "\n")
    scores = {key: {"normalized_exact_matches": correct[key], "total": count} for key, count in n.items()}
    (dest / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
