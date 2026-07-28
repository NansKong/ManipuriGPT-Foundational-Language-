"""
HumanEvaluationPipeline module (`app/evaluation/human.py`).
Generates offline `evaluation/human_review.md` sheets and tracks human A/B / rating annotations (Phase 7).
"""

import os
import json
from typing import Dict, Any, List, Optional
from app.utils.logger import logger


class HumanEvaluationPipeline:
    """Manages human review sheets (human_review.md) and preference annotations."""

    def __init__(self, storage_dir: str = "evaluation"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.annotations: List[Dict[str, Any]] = []

    def generate_human_review_sheet(self, sample_generations: List[Dict[str, str]], output_file: Optional[str] = None) -> str:
        """Generates a structured, reusable offline review sheet (`evaluation/human_review.md`)."""
        out_path = output_file or os.path.join(self.storage_dir, "human_review.md")

        sections = [
            "# ManipuriGPT Human Evaluation Review Sheet",
            "",
            "Please rate each model output on a **1–5 scale** across the 6 criteria below:",
            "1. **Fluency** (1 = ungrammatical/gibberish, 5 = perfectly natural)",
            "2. **Grammar** (1 = invalid syntax, 5 = flawless grammar)",
            "3. **Cultural Correctness** (1 = culturally inaccurate, 5 = authentic)",
            "4. **Meaning Preservation** (1 = lost context, 5 = accurate context)",
            "5. **Naturalness** (1 = robotic, 5 = native speaker level)",
            "6. **Readability** (1 = unreadable, 5 = effortless reading)",
            "",
            "---",
            ""
        ]

        for i, item in enumerate(sample_generations, 1):
            prompt = item.get("prompt", "")
            generated = item.get("generated", "")
            sections.append(f"### Sample #{i}")
            sections.append(f"**Prompt**: `{prompt}`")
            sections.append(f"**Generated Text**: `{generated}`")
            sections.append("")
            sections.append("| Criterion | Score (1-5) | Notes / Comments |")
            sections.append("| --- | --- | --- |")
            sections.append("| Fluency | [ ] | |")
            sections.append("| Grammar | [ ] | |")
            sections.append("| Cultural Correctness | [ ] | |")
            sections.append("| Meaning Preservation | [ ] | |")
            sections.append("| Naturalness | [ ] | |")
            sections.append("| Readability | [ ] | |")
            sections.append("")
            sections.append("---")
            sections.append("")

        md_content = "\n".join(sections)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"HumanEvaluationPipeline: Written review sheet to '{out_path}'")
        return out_path
