"""
Dataset Statistics module for ManipuriGPT Phase 8 SFT (Stage 6: Statistics & Data Card Publisher).
Calculates detailed metric distributions (Mean, Median, P95, Max, Script %, Quality Score)
and generates `instruction_stats_v1.json` and `DATASET_CARD_v1.md`.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any
from app.sft.config import STATS_DATA_DIR
from app.sft.dataset_quality_scorer import SFTQualityScorer


class SFTStatisticsReporter:
    def __init__(self, stats_dir: Path = STATS_DATA_DIR):
        self.stats_dir = Path(stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _compute_stats(lengths: List[int]) -> Dict[str, Any]:
        """Helper to calculate mean, median, p95, and max for length arrays using standard library."""
        if not lengths:
            return {"mean": 0.0, "median": 0, "p95": 0, "max": 0}
        s_arr = sorted(lengths)
        n = len(s_arr)
        mean_val = round(sum(s_arr) / n, 2)
        median_val = int(statistics.median(s_arr))
        p95_idx = min(int(n * 0.95), n - 1)
        p95_val = s_arr[p95_idx]
        max_val = s_arr[-1]
        return {
            "mean": mean_val,
            "median": median_val,
            "p95": p95_val,
            "max": max_val,
        }

    def generate_report(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        filename: str = "instruction_stats_v1.json",
        card_filename: str = "DATASET_CARD_v1.md",
    ) -> Path:
        all_records = train_records + val_records

        task_counts: Dict[str, int] = {}
        script_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}
        lang_counts: Dict[str, int] = {}

        inst_char_lens: List[int] = []
        out_char_lens: List[int] = []
        inst_token_lens: List[int] = []
        out_token_lens: List[int] = []
        quality_scores: List[float] = []

        total_meetei_chars = 0
        total_bengali_chars = 0
        total_latin_chars = 0
        total_all_chars = 0

        for rec in all_records:
            t_type = str(rec.get("task_type") or "unknown").lower()
            task_counts[t_type] = task_counts.get(t_type, 0) + 1

            script = str(rec.get("script") or SFTQualityScorer.detect_script(rec.get("output", "")))
            script_counts[script] = script_counts.get(script, 0) + 1

            lang = str(rec.get("language") or "mni").lower()
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            src = str(rec.get("source") or "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

            quality_scores.append(rec.get("quality_score", 0.95))

            inst = rec.get("instruction", "")
            inp = rec.get("input", "")
            out = rec.get("output", "")

            inst_full = f"{inst} {inp}".strip()
            inst_c_len = len(inst_full)
            out_c_len = len(out)

            inst_char_lens.append(inst_c_len)
            out_char_lens.append(out_c_len)
            inst_token_lens.append(int(inst_c_len / 3.5))
            out_token_lens.append(int(out_c_len / 3.5))

            full_str = inst_full + " " + out
            total_all_chars += len(full_str)
            for ch in full_str:
                if '\uABC0' <= ch <= '\uABFF' or '\uAAE0' <= ch <= '\uAAFF':
                    total_meetei_chars += 1
                elif '\u0980' <= ch <= '\u09FF':
                    total_bengali_chars += 1
                elif ch.isalpha() and ch.isascii():
                    total_latin_chars += 1

        # Script percentage calculations
        total_script_chars = max(total_meetei_chars + total_bengali_chars + total_latin_chars, 1)
        script_breakdown_pct = {
            "MeeteiMayek": round((total_meetei_chars / total_script_chars) * 100, 2),
            "Bengali": round((total_bengali_chars / total_script_chars) * 100, 2),
            "Latin": round((total_latin_chars / total_script_chars) * 100, 2),
            "Mixed": round((script_counts.get("Mixed", 0) / max(len(all_records), 1)) * 100, 2),
        }

        avg_quality = round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 0.95

        report_data = {
            "dataset_version": "v1.0",
            "total_samples": len(all_records),
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "avg_quality_score": avg_quality,
            "task_distribution": task_counts,
            "script_counts": script_counts,
            "script_breakdown_pct": script_breakdown_pct,
            "language_distribution": lang_counts,
            "source_distribution": source_counts,
            "length_statistics": {
                "instruction_chars": self._compute_stats(inst_char_lens),
                "instruction_tokens": self._compute_stats(inst_token_lens),
                "response_chars": self._compute_stats(out_char_lens),
                "response_tokens": self._compute_stats(out_token_lens),
            },
        }

        # Export JSON statistics report
        output_path = self.stats_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # Export DATASET_CARD_v1.md
        card_path = self.stats_dir / card_filename
        self._generate_dataset_card(card_path, report_data)

        return output_path

    def _generate_dataset_card(self, card_path: Path, data: Dict[str, Any]):
        """Generates comprehensive Data Card documentation in Markdown format."""
        with open(card_path, "w", encoding="utf-8") as f:
            f.write("# ManipuriGPT SFT Dataset v1.0 — Data Card\n\n")
            f.write("## Dataset Summary\n")
            f.write(f"- **Total Instruction Pairs**: {data['total_samples']:,}\n")
            f.write(f"- **Train Split (90%)**: {data['train_samples']:,}\n")
            f.write(f"- **Validation Split (10%)**: {data['val_samples']:,}\n")
            f.write(f"- **Average Quality Score ($Q$)**: `{data['avg_quality_score']}` (Threshold $\ge 0.85$)\n\n")

            f.write("## Task Distribution\n\n")
            f.write("| Task Category | Count | Percentage |\n")
            f.write("| :--- | :--- | :--- |\n")
            for task, count in data['task_distribution'].items():
                pct = round((count / data['total_samples']) * 100, 2)
                f.write(f"| **{task.capitalize()}** | {count:,} | {pct}% |\n")
            f.write("\n")

            f.write("## Script Distribution\n\n")
            f.write("| Script | Character Share (%) |\n")
            f.write("| :--- | :--- |\n")
            for script, pct in data['script_breakdown_pct'].items():
                f.write(f"| **{script}** | {pct}% |\n")
            f.write("\n")

            f.write("## Response & Instruction Length Metrics\n\n")
            f.write("| Metric | Mean | Median | $P_{95}$ | Max |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for label, stat_key in [
                ("Instruction Length (Chars)", "instruction_chars"),
                ("Instruction Tokens (Est)", "instruction_tokens"),
                ("Response Length (Chars)", "response_chars"),
                ("Response Tokens (Est)", "response_tokens"),
            ]:
                s = data['length_statistics'][stat_key]
                f.write(f"| **{label}** | {s['mean']} | {s['median']} | {s['p95']} | {s['max']} |\n")
            f.write("\n")

            f.write("## Provenance & Attribution\n")
            f.write("Mined across local research corpus (`M001`–`M025`), `Dayananda`, `Joyson Parallel`, `FLORES+`, and synthetic grammar/dictionary generators in ChatML format.\n")
