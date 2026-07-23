"""
MinHashDeduplicator module for near-duplicate detection and removal across streamed chunks
using Locality-Sensitive Hashing (LSH) and MinHash signatures.

Optimized for high-throughput streaming (250K+ sentences):
- Uses LSH band-based candidate retrieval instead of brute-force O(n²) comparison
- Stores signatures as tuples for faster hashing and comparison
- Uses xxhash (with hashlib fallback) for faster n-gram hashing
- Pre-computes hash coefficients to avoid per-ngram multiplication overhead
"""

import hashlib
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict

try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False


class MinHashDeduplicator:
    """
    Near-duplicate deduplication using MinHash signatures (num_perm=128) and LSH bands.
    Helps eliminate redundant web crawl or scraped documents with minor variations.

    Performance: O(1) average-case lookup per document via LSH band indexing,
    vs. previous O(n) brute-force scan over all stored signatures.
    """
    def __init__(
        self,
        num_perm: int = 128,
        num_bands: int = 16,
        similarity_threshold: float = 0.85,
        ngram_size: int = 5
    ):
        if num_perm % num_bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be evenly divisible by num_bands ({num_bands})")
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.similarity_threshold = similarity_threshold
        self.ngram_size = ngram_size

        # Pre-compute hash coefficients for MinHash permutations
        self._coeff_a = tuple((i + 1) for i in range(num_perm))
        self._coeff_b = tuple((i * 0x5bd1e995) for i in range(num_perm))

        # In-memory LSH buckets: band_index -> {band_hash -> set(doc_ids)}
        self.lsh_buckets: List[Dict[int, Set[int]]] = [defaultdict(set) for _ in range(num_bands)]
        # Store signatures as tuples for faster comparison
        self.doc_signatures: Dict[int, Tuple[int, ...]] = {}
        self.next_doc_id = 0

    def reset(self) -> None:
        """Resets the LSH buckets and document signatures index."""
        self.lsh_buckets = [defaultdict(set) for _ in range(self.num_bands)]
        self.doc_signatures.clear()
        self.next_doc_id = 0

    def _get_ngrams(self, text: str) -> Set[str]:
        """
        Extracts character n-grams from cleaned lowercased text for fine-grained near-duplicate comparison.
        """
        clean = " ".join(text.lower().split())
        size = min(self.ngram_size, len(clean)) if len(clean) > 0 else 1
        return {clean[i:i+size] for i in range(len(clean) - size + 1)}

    def _compute_minhash(self, ngrams: Set[str]) -> Tuple[int, ...]:
        """
        Computes num_perm-permutation MinHash signature using deterministic hashing.
        Returns a tuple (immutable, hashable) instead of list for performance.
        """
        num_perm = self.num_perm
        signature = [0xFFFFFFFF] * num_perm
        coeff_a = self._coeff_a
        coeff_b = self._coeff_b

        for ngram in ngrams:
            encoded = ngram.encode("utf-8")
            if HAS_XXHASH:
                base_hash = xxhash.xxh64(encoded).intdigest() & 0xFFFFFFFF
            else:
                base_hash = int.from_bytes(hashlib.md5(encoded).digest()[:8], byteorder="little") & 0xFFFFFFFF

            for i in range(num_perm):
                h = (coeff_a[i] * base_hash + coeff_b[i]) & 0xFFFFFFFF
                if h < signature[i]:
                    signature[i] = h

        return tuple(signature)

    def _get_band_hash(self, signature: Tuple[int, ...], band_idx: int) -> int:
        """Compute a fast hash for a band slice of the signature."""
        start = band_idx * self.rows_per_band
        end = start + self.rows_per_band
        return hash(signature[start:end])

    def _get_lsh_candidates(self, signature: Tuple[int, ...]) -> Set[int]:
        """
        Retrieve candidate doc_ids from LSH buckets that share at least one band hash.
        This is the key optimization: instead of comparing against ALL stored signatures,
        we only compare against candidates that collide in at least one LSH band.
        """
        candidates: Set[int] = set()
        for band_idx in range(self.num_bands):
            band_hash = self._get_band_hash(signature, band_idx)
            bucket = self.lsh_buckets[band_idx]
            if band_hash in bucket:
                candidates.update(bucket[band_hash])
        return candidates

    def is_duplicate_or_add(self, text: str, doc_id: Optional[int] = None) -> Tuple[bool, Optional[int]]:
        """
        Checks if the text is a near-duplicate against existing LSH buckets and document signatures.
        If it is unique, adds its signature to the index and returns (False, new_doc_id).
        If it is a duplicate, returns (True, matched_doc_id).
        """
        if not text or not isinstance(text, str):
            return True, None

        ngrams = self._get_ngrams(text)
        signature = self._compute_minhash(ngrams)

        # Use LSH bands to find candidate matches (O(1) average instead of O(n) brute-force)
        candidates = self._get_lsh_candidates(signature)
        threshold = self.similarity_threshold
        num_perm = self.num_perm
        doc_sigs = self.doc_signatures

        for cand_id in candidates:
            cand_sig = doc_sigs[cand_id]
            # Fast vectorized comparison using tuple iteration
            matches = sum(a == b for a, b in zip(signature, cand_sig))
            est_jaccard = matches / num_perm
            if est_jaccard >= threshold:
                return True, cand_id

        # Unique document: register in LSH buckets and signature store
        current_id = doc_id if doc_id is not None else self.next_doc_id
        self.next_doc_id = max(self.next_doc_id, current_id + 1)
        self.doc_signatures[current_id] = signature

        for band_idx in range(self.num_bands):
            band_hash = self._get_band_hash(signature, band_idx)
            self.lsh_buckets[band_idx][band_hash].add(current_id)

        return False, current_id

    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a dataset dictionary. Returns None if text is a near-duplicate.
        Otherwise enriches metadata and returns the example.
        """
        text = example.get("text", "")
        is_dup, matched_id = self.is_duplicate_or_add(text)
        if is_dup:
            return None

        metadata = example.get("metadata", {}).copy()
        metadata["minhash_deduplicated"] = True
        metadata["doc_id"] = matched_id
        result = example.copy()
        result["metadata"] = metadata
        return result
