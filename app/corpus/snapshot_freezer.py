"""
Snapshot Freezer module for ManipuriGPT Phase 8 Corpus Evolution (`app/corpus/snapshot_freezer.py`).
Freezes `ManipuriGPT-Corpus-v4`, calculates SHA256 checksums, and generates Hugging Face dataset manifest.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


class CorpusSnapshotFreezer:
    """Freezes Corpus-v4 snapshot into immutable frozen release directory."""

    def __init__(self, frozen_dir: Optional[Path] = None):
        self.frozen_dir = Path(frozen_dir) if frozen_dir else Path("artifacts/datasets/ManipuriGPT-Corpus-v4")
        self.frozen_dir.mkdir(parents=True, exist_ok=True)

    def freeze_snapshot(
        self, candidate_sequences: List[Dict[str, Any]], acceptance_report: Dict[str, Any]
    ) -> Tuple[Path, Dict[str, Any]]:
        frozen_data_file = self.frozen_dir / "corpus_v4_frozen.jsonl"

        sha256_hash = hashlib.sha256()

        with open(frozen_data_file, "w", encoding="utf-8") as f:
            for seq in candidate_sequences:
                line = json.dumps(seq, ensure_ascii=False) + "\n"
                f.write(line)
                sha256_hash.update(line.encode("utf-8"))

        digest = sha256_hash.hexdigest()

        # Build Manifest & HuggingFace Dataset Card metadata
        manifest = {
            "dataset_name": "ManipuriGPT-Corpus-v4",
            "hf_repository": "nanskong/ManipuriGPT-Corpus-v4",
            "version": "v4.0",
            "status": "frozen_immutable",
            "sha256_checksum": digest,
            "total_sequences": len(candidate_sequences),
            "acceptance_report": acceptance_report,
            "files": [
                {
                    "filename": "corpus_v4_frozen.jsonl",
                    "sha256": digest,
                    "records": len(candidate_sequences),
                }
            ],
        }

        manifest_file = self.frozen_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Generate README.md dataset card
        readme_file = self.frozen_dir / "README.md"
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write("# ManipuriGPT Corpus v4.0 (Frozen Snapshot)\n\n")
            f.write(f"**Repository**: `nanskong/ManipuriGPT-Corpus-v4`  \n")
            f.write(f"**Status**: `Frozen & Immutable`  \n")
            f.write(f"**SHA256**: `{digest}`  \n\n")
            f.write("## Dataset Summary\n")
            f.write(f"- **Total Sequences**: {len(candidate_sequences):,}\n")
            f.write(f"- **Acceptance Gate Passed**: {acceptance_report.get('passed', False)}\n\n")

        return frozen_data_file, manifest
