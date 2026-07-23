import json
import os
from typing import Dict, Any, List
from app.utils.logger import logger

class TokenStatisticsTracker:
    """
    Tracks tokenization metrics: average tokens, maximum length, minimum length,
    number of truncated samples, and total packed blocks.
    Generates standardized reporting files: summary.md, summary.json, and histograms.png.
    """
    def __init__(self):
        self.token_lengths: List[int] = []
        self.truncated_count = 0
        self.packed_blocks = 0
        self.total_processed = 0

    def record_sample(self, token_len: int, max_length: int):
        """Records metrics for an un-packed token sequence."""
        self.total_processed += 1
        self.token_lengths.append(token_len)
        if token_len >= max_length:
            self.truncated_count += 1

    def record_packed_blocks(self, num_blocks: int):
        """Records the total count of sequence packed blocks."""
        self.packed_blocks += num_blocks

    def generate_summary(self) -> Dict[str, Any]:
        """Generates dictionary of token statistics."""
        if not self.token_lengths:
            return {
                "avg_tokens": 0,
                "max_tokens": 0,
                "min_tokens": 0,
                "truncated": self.truncated_count,
                "packed": self.packed_blocks,
                "total_processed": self.total_processed
            }

        avg_len = sum(self.token_lengths) / len(self.token_lengths)
        return {
            "avg_tokens": round(avg_len, 2),
            "max_tokens": max(self.token_lengths),
            "min_tokens": min(self.token_lengths),
            "truncated": self.truncated_count,
            "packed": self.packed_blocks,
            "total_processed": self.total_processed
        }

    def generate_histogram_text(self) -> str:
        """Generates an ASCII bar chart histogram of token length distributions."""
        if not self.token_lengths:
            return "No token lengths recorded for histogram."
        
        bins = {"<128": 0, "128-512": 0, "512-1024": 0, "1024-2048": 0, ">2048": 0}
        for length in self.token_lengths:
            if length < 128:
                bins["<128"] += 1
            elif length <= 512:
                bins["128-512"] += 1
            elif length <= 1024:
                bins["512-1024"] += 1
            elif length <= 2048:
                bins["1024-2048"] += 1
            else:
                bins[">2048"] += 1

        total = len(self.token_lengths)
        lines = ["### Token Length Distribution Histogram\n"]
        for label, count in bins.items():
            pct = (count / total) * 100 if total > 0 else 0
            bar = "█" * int(pct // 4)
            lines.append(f"`{label:<9}` | {bar:<25} | {count} ({pct:.1f}%)")
        return "\n".join(lines)

    def generate_markdown(self) -> str:
        """Generates markdown report content including ASCII histogram."""
        stats = self.generate_summary()
        md = [
            "# Tokenization & Dataset Statistics Summary Report\n",
            "## Summary Metrics",
            f"- **Total Raw Samples Processed**: {stats['total_processed']}",
            f"- **Average Token Length**: {stats['avg_tokens']} tokens",
            f"- **Minimum Token Length**: {stats['min_tokens']} tokens",
            f"- **Maximum Token Length**: {stats['max_tokens']} tokens",
            f"- **Truncated Samples**: {stats['truncated']}",
            f"- **Packed Blocks Generated**: {stats['packed']}\n",
            self.generate_histogram_text(),
            "\n"
        ]
        return "\n".join(md)

    def save_markdown_report(self, path: str = "artifacts/reports/summary.md") -> None:
        """Saves markdown report to file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())
        logger.info(f"TokenStatisticsTracker: Saved markdown report to '{path}'")

    def save_json_report(self, path: str = "artifacts/reports/summary.json") -> None:
        """Saves JSON report to file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.generate_summary(), f, indent=2)
        logger.info(f"TokenStatisticsTracker: Saved JSON report to '{path}'")

    def generate_plots(self, output_dir: str = "artifacts/reports", filename: str = "histograms.png") -> None:
        """
        Attempts to generate and save matplotlib histogram plot of token lengths.
        Gracefully falls back if matplotlib is not installed.
        """
        if not self.token_lengths:
            return
        try:
            import matplotlib.pyplot as plt
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, filename)

            plt.figure(figsize=(8, 5))
            plt.hist(self.token_lengths, bins=30, color="skyblue", edgecolor="black")
            plt.title("Token Length Distribution")
            plt.xlabel("Token Length")
            plt.ylabel("Sample Count")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"TokenStatisticsTracker: Saved histogram plot to '{plot_path}'")
        except ImportError:
            logger.debug("TokenStatisticsTracker: matplotlib not installed, skipping histogram plot PNG generation.")
        except Exception as e:
            logger.warning(f"TokenStatisticsTracker: Could not generate histogram plot: {e}")

    def save_reports(self, output_dir: str = "artifacts/reports", prefix: str = "summary") -> None:
        """Saves both markdown and JSON reports, and generates plots if available."""
        os.makedirs(output_dir, exist_ok=True)
        self.save_markdown_report(os.path.join(output_dir, f"{prefix}.md"))
        self.save_json_report(os.path.join(output_dir, f"{prefix}.json"))
        self.generate_plots(output_dir=output_dir)
