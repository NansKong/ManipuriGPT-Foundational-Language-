"""
TranslationDatasetBuilder module for assembling parallel translation datasets across:
`English <-> Manipuri (Meitei Mayek)`, `English <-> Manipuri (Bengali Script)`, and `Hindi <-> Manipuri`.
Includes alignment verification and automatic quality scoring.
"""

from typing import Dict, Any, List, Optional, Tuple
from app.utils.logger import logger


class TranslationDatasetBuilder:
    """
    Builds and validates parallel sentence pairs for translation modeling across English,
    Hindi, and Manipuri scripts (Meitei Mayek / Bengali).
    """
    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "mni",
        min_length_ratio: float = 0.25,
        max_length_ratio: float = 4.0
    ):
        self.source_lang = source_lang.lower()
        self.target_lang = target_lang.lower()
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio

    def validate_and_score_pair(self, source_text: str, target_text: str) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validates parallel alignment and calculates heuristic alignment quality score (0.0 to 1.0).
        Checks length ratio consistency and empty sentence checks.
        """
        if not source_text or not target_text:
            return False, 0.0, {"reason": "empty_sentence"}

        src_clean = source_text.strip()
        tgt_clean = target_text.strip()
        if not src_clean or not tgt_clean:
            return False, 0.0, {"reason": "empty_after_strip"}

        src_len = len(src_clean.split())
        tgt_len = len(tgt_clean.split())

        ratio = tgt_len / max(src_len, 1)
        if ratio < self.min_length_ratio or ratio > self.max_length_ratio:
            return False, 0.2, {"reason": "abnormal_length_ratio", "ratio": round(ratio, 2)}

        # Base quality score inversely proportional to ratio skew from ideal 1.2
        ideal_ratio = 1.2
        skew = abs(ratio - ideal_ratio)
        score = max(0.0, min(1.0, 0.95 - (skew * 0.2)))

        return True, round(score, 3), {
            "src_words": src_len,
            "tgt_words": tgt_len,
            "length_ratio": round(ratio, 2)
        }

    def build_parallel_records(self, raw_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transforms raw parallel dictionary entries into standard seq2seq/instruction records.
        Each valid record contains `translation` (`source`, `target`) and alignment `quality_score`.
        """
        valid_records: List[Dict[str, Any]] = []

        for pair in raw_pairs:
            src_text = pair.get(self.source_lang, pair.get("source", ""))
            tgt_text = pair.get(self.target_lang, pair.get("target", ""))

            is_valid, score, details = self.validate_and_score_pair(src_text, tgt_text)
            if not is_valid:
                continue

            record = {
                "translation": {
                    self.source_lang: src_text.strip(),
                    self.target_lang: tgt_text.strip()
                },
                "metadata": {
                    "source_lang": self.source_lang,
                    "target_lang": self.target_lang,
                    "alignment_score": score,
                    "alignment_details": details
                }
            }
            valid_records.append(record)

        logger.info(f"TranslationDatasetBuilder: Validated {len(valid_records)} of {len(raw_pairs)} parallel pairs.")
        return valid_records

    def build_from_source(
        self,
        source: Any,
        limit: int = 5000,
        mock_fallback: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Streams parallel translation pairs from a real corpus source (e.g., 'opus', 'opensubtitles', 'flores'),
        validates alignment, scores quality, and formats them.
        """
        from app.corpus.acquisition import CorpusAcquisitionManager
        logger.info(f"TranslationDatasetBuilder: Building parallel dataset from source '{source}' (limit={limit}, mock_fallback={mock_fallback})...")
        mgr = CorpusAcquisitionManager()
        spec = mgr.get_source(source) if isinstance(source, str) else source
        if not spec:
            raise KeyError(f"Source '{source}' not found in registry.")

        stream = mgr.stream_source(spec, max_examples=limit, mock_fallback=mock_fallback)
        raw_pairs = []
        for ex in stream:
            if isinstance(ex, dict):
                # Check for standard translation dictionary structures
                if "translation" in ex and isinstance(ex["translation"], dict):
                    raw_pairs.append(ex["translation"])
                else:
                    raw_pairs.append(ex)

        return self.build_parallel_records(raw_pairs)
