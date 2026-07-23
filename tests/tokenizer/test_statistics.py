import os
import json
from app.tokenizer.statistics import TokenStatisticsTracker

def test_statistics_tracker_recording():
    tracker = TokenStatisticsTracker()
    tracker.record_sample(100, max_length=512)
    tracker.record_sample(600, max_length=512)
    tracker.record_packed_blocks(5)

    stats = tracker.generate_summary()
    assert stats["total_processed"] == 2
    assert stats["avg_tokens"] == 350.0
    assert stats["max_tokens"] == 600
    assert stats["min_tokens"] == 100
    assert stats["truncated"] == 1
    assert stats["packed"] == 5

def test_statistics_tracker_reports(tmp_path):
    tracker = TokenStatisticsTracker()
    tracker.record_sample(200, max_length=512)

    md_path = os.path.join(tmp_path, "summary.md")
    json_path = os.path.join(tmp_path, "summary.json")

    tracker.save_markdown_report(md_path)
    tracker.save_json_report(json_path)
    tracker.generate_plots(output_dir=str(tmp_path), filename="histograms.png")

    assert os.path.exists(md_path)
    assert os.path.exists(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_processed"] == 1
        assert data["avg_tokens"] == 200.0
