"""
Master Execution CLI Script for ManipuriGPT Phase 8 Track 1:
Corpus Evolution, Versioning & Tokenizer Audit (`app/scripts/run_track1_pipeline.py`).

Executes:
  1. Ingestion of 18 prose documents -> Candidate Corpus-v4
  2. Corpus Acceptance Gate -> `acceptance_report_v4.json`
  3. Snapshot Freezing -> `ManipuriGPT-Corpus-v4` (SHA256 & Hugging Face Manifest)
  4. Tokenizer Compatibility Audit -> `Tokenizer-v1.0` Decision Gate
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.corpus.track1_processor import Track1CorpusProcessor
from app.corpus.acceptance_gate import CorpusAcceptanceGate
from app.corpus.snapshot_freezer import CorpusSnapshotFreezer
from app.tokenization.tokenizer_audit import Track1TokenizerAuditor


def main():
    print("=" * 80)
    print(" MANIPURIGPT PHASE 8 — TRACK 1: CORPUS EVOLUTION & TOKENIZER AUDIT")
    print("=" * 80)

    # Step 1: Candidate Corpus-v4 Ingestion
    print("\n[Step 1/4] Ingesting Track 1 Prose Documents (18 Documents)...")
    processor = Track1CorpusProcessor()
    sequences, proc_summary = processor.process_track1_documents()

    print(f"  Ingested Sequences : {proc_summary['total_sequences']:,}")
    print(f"  Deduplicated Out   : {proc_summary['deduplicated_removed']:,}")

    # Step 2: Corpus Acceptance Gate
    print("\n[Step 2/4] Running Corpus Acceptance Gate...")
    gate = CorpusAcceptanceGate()
    passed, acceptance_report = gate.run_audit(sequences)

    print(f"  Acceptance Gate Passed : {passed}")
    print(f"  Quality Ratio          : {acceptance_report['quality_ratio_pct']}%")
    print(f"  Report Saved To        : {acceptance_report['report_file']}")

    if not passed:
        print("\n[Track 1 Pipeline Warning] Corpus Acceptance Gate did not pass threshold.")

    # Step 3: Snapshot Freezing
    print("\n[Step 3/4] Freezing Snapshot: ManipuriGPT-Corpus-v4...")
    freezer = CorpusSnapshotFreezer()
    frozen_file, manifest = freezer.freeze_snapshot(sequences, acceptance_report)

    print(f"  Frozen Dataset Data : {frozen_file}")
    print(f"  SHA256 Checksum     : {manifest['sha256_checksum']}")
    print(f"  HF Repository Tag   : {manifest['hf_repository']}")

    # Step 4: Tokenizer Compatibility Audit
    print("\n[Step 4/4] Conducting Tokenizer Compatibility Audit on Corpus-v4...")
    auditor = Track1TokenizerAuditor(tokenizer_path="nanskong/ManipuriGPT-Tokenizer-v1")
    audit_report = auditor.audit_corpus(sequences)

    audit_out_path = Path("artifacts/datasets/ManipuriGPT-Corpus-v4/tokenizer_audit_report.json")
    audit_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_out_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    print(f"  Audited Tokenizer     : {audit_report['audited_tokenizer']}")
    print(f"  Unknown Token Rate    : {audit_report['unknown_token_rate_pct']}%")
    print(f"  Compression Ratio     : {audit_report['compression_ratio_bytes_per_token']} bytes/token")
    print(f"  Decision Gate Status  : {audit_report['decision_gate']['status']}")
    print(f"  Recommended Tokenizer : {audit_report['decision_gate']['recommended_tokenizer']}")
    print(f"  Audit Report Saved To : {audit_out_path}")

    print("\n" + "=" * 80)
    print(" TRACK 1 EXECUTION COMPLETE — CORPUS-V4 FROZEN & TOKENIZER AUDITED")
    print("=" * 80)


if __name__ == "__main__":
    main()
