"""
Track 1 Processor module for ManipuriGPT Phase 8 Corpus Evolution (`app/corpus/track1_processor.py`).
Ingests 18 prose documents (literature, grammar, phonology, native Meetei Mayek books M019 & M020),
applies OCR quality filters, deduplication, and formats Candidate Corpus-v4.
"""

from app.corpus.book_ocr import review_reason

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Track 1 Document Inventory (18 Prose Documents)
TRACK1_DOCUMENT_IDS = [
    "M001", "M002", "M005", "M006", "M007", "M009", "M010",
    "M011", "M013", "M014", "M015", "M016", "M017", "M018",
    "M019", "M020", "M022", "M024"
]

TRACK1_CATALOG: Dict[str, Dict[str, Any]] = {
    "M001": {"title": "A Collection of Manipuri Folktales", "category": "Literature", "script": "bengali"},
    "M002": {"title": "Ahangi Manipuri Mopi Lairik", "category": "Cultural", "script": "bengali"},
    "M005": {"title": "Manipuri Lerole Amasung Manipuri Khunnai", "category": "Cultural", "script": "bengali"},
    "M006": {"title": "Manipuri Nursery Rhymes", "category": "Cultural", "script": "bengali"},
    "M007": {"title": "Manipuri Phonology (Primary)", "category": "Phonology", "script": "latin"},
    "M009": {"title": "Structure of Manipuri (Metteiron Meaning)", "category": "Grammar", "script": "latin/bengali"},
    "M010": {"title": "Manipuri Kabitagi Chhanda", "category": "Textbook", "script": "bengali"},
    "M011": {"title": "Matamgi Manipuri Grammar Amasung Composition", "category": "Grammar", "script": "bengali/meetei"},
    "M013": {"title": "Manipuri Longei", "category": "Literature", "script": "bengali/meetei"},
    "M014": {"title": "Structural Analysis of Manipuri Language", "category": "Grammar", "script": "latin"},
    "M015": {"title": "Manipurgi Ningthoukon Amasung Chatna-Lonchat", "category": "Cultural", "script": "bengali"},
    "M016": {"title": "Critical Essays of Manipuri Literature", "category": "Literature", "script": "bengali"},
    "M017": {"title": "Manipuri Langei", "category": "Literature", "script": "romanized"},
    "M018": {"title": "Meeteilonmit Manipuri Grammar", "category": "Grammar", "script": "bengali"},
    "M019": {"title": "Nunggee Mashak Utliba Kanglei Haraobagi Jagoi", "category": "Cultural", "script": "meetei_mayek"},
    "M020": {"title": "Manipuri Phunggawarigi Saktam", "category": "Cultural", "script": "meetei_mayek"},
    "M022": {"title": "Manipuri Phonetic Reader", "category": "Phonology", "script": "latin"},
    "M024": {"title": "Introduction to Manipuri Language & Grammar", "category": "Grammar", "script": "latin"},
}


class Track1CorpusProcessor:
    """Manages ingestion and Candidate Corpus-v4 staging for Track 1."""

    def __init__(self, corpus_source_dir: Optional[Path] = None, output_v4_dir: Optional[Path] = None):
        self.output_dir = Path(output_v4_dir) if output_v4_dir else Path("artifacts/datasets/Candidate-Corpus-v4")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Candidate search directories (Strictly Disk E as primary source)
        self.search_dirs = [
            Path("E:/manipuri corpus"),
            Path("E:/manipuri_corpus"),
            Path("artifacts/datasets/ManipuriGPT-Corpus-v3"),
            Path("artifacts/datasets/ManipuriGPT-Corpus-v1"),
            Path("data/processed"),
        ]
        if corpus_source_dir:
            self.search_dirs.insert(0, Path(corpus_source_dir))

    def _find_document_file(self, doc_id: str) -> Optional[Path]:
        for s_dir in self.search_dirs:
            if not s_dir.exists():
                continue
            processed = s_dir / "processed" / f"{doc_id}.jsonl"
            if processed.is_file():
                return processed
            # Check direct JSONL file
            cand = s_dir / f"{doc_id}.jsonl"
            if cand.exists():
                return cand
            # Check jsonl in subdirectories
            for root, _, files in os.walk(s_dir):
                if f"{doc_id}.jsonl" in files:
                    return Path(root) / f"{doc_id}.jsonl"
        return None

    def process_track1_documents(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        processed_sequences = []
        doc_stats = {}
        seen_hashes = set()
        dedup_count = 0

        print(f"[Track 1 Processor] Ingesting {len(TRACK1_DOCUMENT_IDS)} prose documents...")

        for doc_id in TRACK1_DOCUMENT_IDS:
            metadata = TRACK1_CATALOG.get(doc_id, {"title": f"Doc {doc_id}", "script": "unknown"})
            doc_file = self._find_document_file(doc_id)

            words_count = 0
            seq_count = 0
            review_pages = []

            if doc_file and doc_file.exists():
                with open(doc_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            text = record.get("text", "").strip()
                            reason = review_reason(doc_id, record)
                            if reason:
                                review_pages.append({"page": record.get("page"), "reason": reason})
                                continue
                            if not text:
                                continue

                            # Hash deduplication
                            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                            if h in seen_hashes:
                                dedup_count += 1
                                continue
                            seen_hashes.add(h)

                            w_len = len(text.split())
                            words_count += w_len
                            seq_count += 1

                            processed_sequences.append({
                                "id": f"v4_{doc_id}_{seq_count:06d}",
                                "document_id": doc_id,
                                "text": text,
                                "page": record.get("page"),
                                "page_id": record.get("id"),
                                "word_count": w_len,
                                "script": record.get("script") or metadata["script"],
                                "category": metadata.get("category", "General"),
                                "source": metadata["title"],
                            })
                        except Exception:
                            continue
            doc_stats[doc_id] = {
                "sequences": seq_count,
                "words": words_count,
                "review_pages": review_pages,
                "title": metadata["title"],
                "script": metadata["script"],
                "source_file": str(doc_file) if doc_file else None,
            }

        # Save Candidate Corpus-v4 JSONL
        out_jsonl = self.output_dir / "candidate_corpus_v4.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for item in processed_sequences:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        summary = {
            "total_documents": len(TRACK1_DOCUMENT_IDS),
            "total_sequences": len(processed_sequences),
            "deduplicated_removed": dedup_count,
            "output_jsonl": str(out_jsonl),
            "doc_stats": doc_stats,
        }

        return processed_sequences, summary
