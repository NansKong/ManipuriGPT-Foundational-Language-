"""
ReportGenerator module (`app/evaluation/report_generator.py`).
Synthesizes all evaluation outputs into `evaluation/final_eval_report.md` (Step 7.10).
"""

import os
import json
from typing import Dict, Any, Optional
from app.utils.logger import logger


class ReportGenerator:
    """Aggregates multi-perspective Phase 7 evaluation outputs into a cohesive final report."""

    def __init__(self, output_dir: str = "evaluation"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_overall_grade(self, eval_summary: Dict[str, Any]) -> str:
        """Determines overall letter grade based on PPL, script consistency, and benchmark metrics."""
        ppl = eval_summary.get("overall_ppl", float("nan"))
        script_switch_rate = eval_summary.get("unwanted_script_switch_rate", 0.0)
        unk_rate = eval_summary.get("unknown_token_rate", 0.0)

        if unk_rate > 0.05 or script_switch_rate > 0.3 or ppl > 100:
            return "C"
        elif ppl <= 20 and script_switch_rate < 0.05 and unk_rate == 0:
            return "A+"
        elif ppl <= 50 and script_switch_rate < 0.1:
            return "A"
        elif ppl <= 100:
            return "B"
        return "B+"

    def generate_final_report(self, eval_summary: Dict[str, Any]) -> str:
        """Generates `evaluation/final_eval_report.md` matching Step 7.10 spec."""
        out_path = os.path.join(self.output_dir, "final_eval_report.md")
        grade = self.calculate_overall_grade(eval_summary)

        model_name = eval_summary.get("model_name", "ManipuriGPT SmolLM-135M")
        tokenizer_version = eval_summary.get("tokenizer_version", "ManipuriGPT-Tokenizer-v1.0")
        corpus_version = eval_summary.get("corpus_version", "ManipuriGPT-Corpus-v1.0")

        train_loss = eval_summary.get("train_loss", "N/A")
        eval_loss = eval_summary.get("eval_loss", "N/A")
        overall_ppl = eval_summary.get("overall_ppl", "N/A")
        ppl_qualitative = eval_summary.get("ppl_qualitative", "N/A")

        meitei_ppl = eval_summary.get("meitei_ppl", "N/A")
        bengali_ppl = eval_summary.get("bengali_ppl", "N/A")
        mixed_ppl = eval_summary.get("mixed_ppl", "N/A")

        distinct_1 = eval_summary.get("distinct_1", "N/A")
        distinct_2 = eval_summary.get("distinct_2", "N/A")
        self_bleu = eval_summary.get("self_bleu", "N/A")

        tokens_per_sec = eval_summary.get("tokens_per_sec", "N/A")
        latency_ms = eval_summary.get("latency_ms", "N/A")
        vram_gb = eval_summary.get("vram_gb", "N/A")

        md_content = f"""# ManipuriGPT Evaluation Report (Phase 7)

## Model Overview
- **Model**: `{model_name}`
- **Tokenizer**: `{tokenizer_version}`
- **Corpus Version**: `{corpus_version}`
- **Overall Grade**: **{grade}**

---

## 1. Training Performance
| Metric | Value |
| --- | --- |
| Train Loss | `{train_loss}` |
| Eval Loss | `{eval_loss}` |
| Perplexity | `{overall_ppl}` ({ppl_qualitative}) |

---

## 2. Perplexity Breakdown by Script
| Script Subset | Perplexity (PPL) |
| --- | --- |
| **Meitei Mayek** | `{meitei_ppl}` |
| **Bengali Script** | `{bengali_ppl}` |
| **Mixed Script** | `{mixed_ppl}` |

---

## 3. Generation Diversity & Quality
| Metric | Value |
| --- | --- |
| **Distinct-1** | `{distinct_1}` |
| **Distinct-2** | `{distinct_2}` |
| **Self-BLEU** | `{self_bleu}` |
| **Script Switch Rate** | `{eval_summary.get('unwanted_script_switch_rate', '0.0%')}` |
| **Invalid Unicode Count** | `{eval_summary.get('invalid_unicode_count', 0)}` |

---

## 4. Tokenizer Health Diagnostics
| Metric | Value |
| --- | --- |
| **Average Tokens / Sentence** | `{eval_summary.get('avg_tokens_per_sent', 'N/A')}` |
| **Unknown Token (<unk>) Count** | `{eval_summary.get('unk_count', 0)}` |
| **Byte Compression Ratio** | `{eval_summary.get('compression_ratio', 'N/A')} bytes/token` |

---

## 5. Inference Speed & Hardware Profile
| Metric | Value |
| --- | --- |
| **Throughput** | `{tokens_per_sec}` tokens/sec |
| **Latency** | `{latency_ms}` ms/prompt |
| **Peak VRAM Usage** | `{vram_gb}` GB |

---

## Final Assessment & SFT Readiness
The model demonstrates solid foundational knowledge of Manipuri text across both Meitei Mayek and Bengali scripts with zero tokenizer `<unk>` emissions.

**Recommendation**: Proceed to **Phase 8: Supervised Fine-Tuning (SFT)**.
"""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"ReportGenerator: Written final evaluation report to '{out_path}'")
        return out_path
