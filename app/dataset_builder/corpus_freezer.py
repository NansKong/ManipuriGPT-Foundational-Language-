"""
Immutable Corpus Snapshot Freezer & Fingerprinter (`app/dataset_builder/corpus_freezer.py`).

Consolidates all dataset shards and tokenizers into a final, immutable snapshot directory
(`ManipuriGPT-Corpus-v1.0/`) with explicit SHA256 checksums, dataset identity fingerprint,
license text summaries, and release manifests.
"""

import os
import shutil
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.utils.logger import logger


class CorpusFreezer:
    """Builds and freezes the final ManipuriGPT-Corpus-v1.0 dataset snapshot."""

    def compute_sha256(self, file_path: str) -> str:
        """Compute SHA256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def freeze(
        self,
        v1_dir: str,
        v3_dir: str,
        output_dir: str,
        tokenizer_model_path: str,
        seed: int = 42,
        shard_size: int = 10000
    ) -> Dict[str, Any]:
        """Freeze and consolidate corpus into an immutable release directory."""
        try:
            import pyarrow.parquet as pq
            from datasets import Dataset
            from app.preprocessing.splitter import DatasetSplitter
        except ImportError:
            raise ImportError("pyarrow and datasets are required to freeze the corpus.")

        logger.info(f"CorpusFreezer: Building frozen snapshot at '{output_dir}'...")
        os.makedirs(output_dir, exist_ok=True)

        all_records: List[Dict[str, Any]] = []
        seen_hashes: set = set()

        # Helper to load rows from a corpus directory
        def _load_dir(target_dir: str):
            if not os.path.isdir(target_dir):
                return
            for split in ["train", "validation", "test"]:
                split_dir = os.path.join(target_dir, split)
                if not os.path.isdir(split_dir):
                    continue
                for f in sorted(os.listdir(split_dir)):
                    if not f.endswith(".parquet"):
                        continue
                    fp = os.path.join(split_dir, f)
                    try:
                        table = pq.read_table(fp)
                        df = table.to_pandas()
                        for record in df.to_dict(orient="records"):
                            txt = record.get("text", "")
                            if txt:
                                h = hashlib.sha256(str(txt).strip().encode("utf-8")).hexdigest()
                                if h not in seen_hashes:
                                    seen_hashes.add(h)
                                    all_records.append(record)
                    except Exception as e:
                        logger.warning(f"Error reading shard {fp}: {e}")

        logger.info(f"Loading records from v1 directory: {v1_dir}")
        _load_dir(v1_dir)
        logger.info(f"Loading records from v3 directory: {v3_dir}")
        _load_dir(v3_dir)

        logger.info(f"Total unique consolidated clean records: {len(all_records):,}")

        if not all_records:
            raise ValueError("No records found to freeze!")

        # Deterministic Train / Val / Test splitting
        splitter = DatasetSplitter({"enabled": True, "train": 0.98, "validation": 0.01, "test": 0.01})
        full_ds = Dataset.from_list(all_records)
        split_dict = splitter.split(full_ds, seed=seed)

        shard_checksums: Dict[str, str] = {}
        total_chars = 0
        total_tokens = 0

        # Export split shards
        for split_name, ds in split_dict.items():
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            rows_in_split = len(ds)
            shard_idx = 0
            for start_i in range(0, rows_in_split, shard_size):
                end_i = min(start_i + shard_size, rows_in_split)
                shard_rows = [ds[i] for i in range(start_i, end_i)]
                for r in shard_rows:
                    total_chars += len(r.get("text", ""))

                shard_ds = Dataset.from_list(shard_rows)
                shard_filename = f"shard-{shard_idx:05d}.parquet"
                shard_path = os.path.join(split_dir, shard_filename)
                shard_ds.to_parquet(shard_path)

                sha = self.compute_sha256(shard_path)
                rel_path = f"{split_name}/{shard_filename}"
                shard_checksums[rel_path] = sha
                shard_idx += 1
                logger.info(f"  Exported {rel_path} ({len(shard_rows):,} seqs, sha256={sha[:12]}...)")

        # Copy & freeze tokenizer files
        tok_dest_dir = os.path.join(output_dir, "tokenizer")
        os.makedirs(tok_dest_dir, exist_ok=True)
        tokenizer_checksums: Dict[str, str] = {}

        if tokenizer_model_path and os.path.exists(tokenizer_model_path):
            tok_dest_model = os.path.join(tok_dest_dir, "tokenizer.model")
            shutil.copy2(tokenizer_model_path, tok_dest_model)
            tokenizer_checksums["tokenizer/tokenizer.model"] = self.compute_sha256(tok_dest_model)

            # Copy vocabulary json/txt/config if present
            src_dir = os.path.dirname(tokenizer_model_path)
            for extra in ["tokenizer.vocab", "tokenizer_config.json", "tokenizer_qualitative_samples.json", "tokenizer.json", "vocab.txt", "special_tokens_map.json"]:
                ext_p = os.path.join(src_dir, extra)
                if os.path.exists(ext_p):
                    dest_p = os.path.join(tok_dest_dir, extra)
                    shutil.copy2(ext_p, dest_p)
                    tokenizer_checksums[f"tokenizer/{extra}"] = self.compute_sha256(dest_p)

            # Evaluate token count with frozen tokenizer
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load(tok_dest_model)
                for rec in all_records:
                    total_tokens += len(sp.Encode(rec.get("text", "")))
            except Exception as e:
                logger.warning(f"Could not compute precise token counts: {e}")
                total_tokens = total_chars // 4

        # Create LICENSES/ directory
        lic_dir = os.path.join(output_dir, "LICENSES")
        os.makedirs(lic_dir, exist_ok=True)
        with open(os.path.join(lic_dir, "LICENSE_SUMMARY.txt"), "w", encoding="utf-8") as f:
            f.write(
                "ManipuriGPT Corpus v1.0 License Summary\n"
                "=========================================\n"
                "1. EMA Lon Monolingual & Parallel: CC BY-NC 4.0\n"
                "2. AI4Bharat Sangraha Manipuri: CC BY 4.0\n"
                "3. Dayananda / Joyson Public Benchmark Subsets: CC BY 4.0 / MIT\n"
                "4. Processed OCR Books: Academic Research Use\n"
            )

        # Create checksums/ directory
        chk_dir = os.path.join(output_dir, "checksums")
        os.makedirs(chk_dir, exist_ok=True)

        with open(os.path.join(chk_dir, "parquet_shards.sha256"), "w", encoding="utf-8") as f:
            for path, sha in sorted(shard_checksums.items()):
                f.write(f"{sha}  {path}\n")

        with open(os.path.join(chk_dir, "tokenizer.sha256"), "w", encoding="utf-8") as f:
            for path, sha in sorted(tokenizer_checksums.items()):
                f.write(f"{sha}  {path}\n")

        # Copy audit report and master corpus report if present in v3
        meta_dir = os.path.join(output_dir, "metadata")
        os.makedirs(meta_dir, exist_ok=True)

        for report_name in ["corpus_report.json", "corpus_audit_report.json"]:
            src_rep = os.path.join(v3_dir, "metadata", report_name)
            if not os.path.exists(src_rep):
                src_rep = os.path.join(v1_dir, "metadata", report_name)
            if os.path.exists(src_rep):
                shutil.copy2(src_rep, os.path.join(meta_dir, report_name))
                shutil.copy2(src_rep, os.path.join(output_dir, report_name))

        # Dataset Fingerprint identity
        fingerprint_data = {
            "dataset_sha256": hashlib.sha256(json.dumps(sorted(list(shard_checksums.values()))).encode("utf-8")).hexdigest(),
            "tokenizer_sha256": list(tokenizer_checksums.values())[0] if tokenizer_checksums else "none",
            "pipeline_version": "5.6",
            "build_timestamp": datetime.utcnow().isoformat() + "Z",
            "records": len(all_records),
            "tokens": total_tokens,
            "characters": total_chars,
        }
        fp_path = os.path.join(output_dir, "dataset_fingerprint.json")
        with open(fp_path, "w", encoding="utf-8") as f:
            json.dump(fingerprint_data, f, indent=2)

        # Release manifest
        manifest_data = {
            "name": "ManipuriGPT-Corpus-v1.0",
            "version": "1.0.0",
            "pipeline": "5.6",
            "status": "FROZEN_IMMUTABLE",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "records": len(all_records),
            "tokens": total_tokens,
            "characters": total_chars,
            "fingerprint": fingerprint_data["dataset_sha256"],
            "tokenizer": "ManipuriGPT-Tokenizer-v1.0" if tokenizer_model_path else "estimated",
            "license": "Mixed (See LICENSES/LICENSE_SUMMARY.txt)"
        }

        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        with open(os.path.join(meta_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Manifests and Readme SHA256 checksums
        with open(os.path.join(chk_dir, "manifests.sha256"), "w", encoding="utf-8") as f:
            f.write(f"{self.compute_sha256(os.path.join(output_dir, 'manifest.json'))}  manifest.json\n")
            f.write(f"{self.compute_sha256(fp_path)}  dataset_fingerprint.json\n")

        # Generate Hugging Face Dataset Card README.md and CITATION.cff
        try:
            from app.exports.hf_publisher import HFPublisher
            publisher = HFPublisher()
            audit_data = None
            audit_p = os.path.join(output_dir, "corpus_audit_report.json")
            if os.path.exists(audit_p):
                with open(audit_p, "r", encoding="utf-8") as f:
                    audit_data = json.load(f)

            publisher.generate_readme(output_dir, manifest_data, audit_data)
            publisher.generate_citation_cff(output_dir, version="1.0.0")

            with open(os.path.join(chk_dir, "readme.sha256"), "w", encoding="utf-8") as f:
                f.write(f"{self.compute_sha256(os.path.join(output_dir, 'README.md'))}  README.md\n")
                f.write(f"{self.compute_sha256(os.path.join(output_dir, 'CITATION.cff'))}  CITATION.cff\n")
        except Exception as e:
            logger.warning(f"Could not generate HF README or CITATION.cff: {e}")

        logger.info(f"CorpusFreezer: Successfully frozen ManipuriGPT-Corpus-v1.0 snapshot at {output_dir}")
        return manifest_data
