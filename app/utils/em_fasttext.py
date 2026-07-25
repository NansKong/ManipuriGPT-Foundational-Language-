"""
EM-FT FastText Utility Module (`app/utils/em_fasttext.py`).

Provides optional post-processing utilities for Manipuri text using the fastText
model/vectors (`cc.mni.300.bin` / `cc.mni.300.vec` located in `W0316/Manipuri resources/EM-FT`).

Capabilities:
  1. Semantic similarity calculation between sentences/tokens
  2. Lexical search and nearest-neighbor query
  3. Candidate OCR correction and dictionary lookup scoring
  4. Spelling correction and quality filtering assistance
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from app.utils.logger import logger


DEFAULT_EM_FT_DIR = Path("cache/datasets/W0316/Manipuri resources/EM-FT")
DEFAULT_BIN_PATH = DEFAULT_EM_FT_DIR / "cc.mni.300.bin"
DEFAULT_VEC_PATH = DEFAULT_EM_FT_DIR / "cc.mni.300.vec"


class EMFastTextEngine:
    """
    Decoupled optional post-processing engine leveraging fastText pre-trained embeddings for Manipuri.
    Lazy-loads vectors/binary model to ensure light overhead when fastText is not invoked.
    """

    def __init__(self, bin_path: Optional[str] = None, vec_path: Optional[str] = None):
        self.bin_path = Path(bin_path) if bin_path else DEFAULT_BIN_PATH
        self.vec_path = Path(vec_path) if vec_path else DEFAULT_VEC_PATH
        self._model = None
        self._vectors = {}
        self._loaded = False

    def is_available(self) -> bool:
        """Checks if fastText binary or vector files exist on disk."""
        return self.bin_path.exists() or self.vec_path.exists()

    def load(self, force_vec_fallback: bool = False) -> bool:
        """
        Loads fastText model into memory.
        Attempts fasttext Python package first, falling back to gensim or raw vector parser.
        """
        if self._loaded:
            return True

        if not self.is_available():
            logger.warning(
                f"EMFastTextEngine: Neither {self.bin_path} nor {self.vec_path} exists."
            )
            return False

        # Attempt fasttext binary loading
        if not force_vec_fallback and self.bin_path.exists():
            try:
                import fasttext
                self._model = fasttext.load_model(str(self.bin_path))
                self._loaded = True
                logger.info(f"EMFastTextEngine: Loaded binary model from {self.bin_path}")
                return True
            except ImportError:
                logger.info("EMFastTextEngine: fasttext package not installed. Trying vector fallback.")
            except Exception as e:
                logger.warning(f"EMFastTextEngine: Failed to load binary model ({e}). Trying vector fallback.")

        # Vector fallback loading (.vec)
        if self.vec_path.exists():
            try:
                self._load_vec_file(str(self.vec_path))
                self._loaded = True
                logger.info(f"EMFastTextEngine: Loaded {len(self._vectors):,} word vectors from {self.vec_path}")
                return True
            except Exception as e:
                logger.error(f"EMFastTextEngine: Error reading .vec file: {e}")
                return False

        return False

    def _load_vec_file(self, path: str, max_words: int = 50000) -> None:
        """Lightweight parsing of .vec format file."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip().split()
            # First line can contain header: num_words dim
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) > 2:
                    word = parts[0]
                    try:
                        vec = [float(x) for x in parts[1:]]
                        self._vectors[word] = vec
                    except ValueError:
                        continue
                if len(self._vectors) >= max_words:
                    break

    def get_word_vector(self, word: str) -> Optional[List[float]]:
        """Returns 300-dim embedding vector for a given word."""
        if not self._loaded:
            if not self.load():
                return None

        if self._model is not None:
            return self._model.get_word_vector(word).tolist()
        elif word in self._vectors:
            return self._vectors[word]
        return None

    def get_sentence_vector(self, sentence: str) -> Optional[List[float]]:
        """Returns sentence vector via fastText average or model method."""
        if not self._loaded:
            if not self.load():
                return None

        if self._model is not None and hasattr(self._model, "get_sentence_vector"):
            return self._model.get_sentence_vector(sentence.replace("\n", " ")).tolist()

        # Manual average over words
        words = sentence.strip().split()
        vecs = [self.get_word_vector(w) for w in words if self.get_word_vector(w) is not None]
        if not vecs:
            return None

        dim = len(vecs[0])
        avg = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
        return avg

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Calculates cosine similarity between two sentences."""
        v1 = self.get_sentence_vector(text1)
        v2 = self.get_sentence_vector(text2)

        if v1 is None or v2 is None:
            return 0.0

        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def rank_ocr_candidates(self, noisy_word: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Ranks candidate dictionary/OCR correction words based on vector cosine similarity.
        """
        v_noisy = self.get_word_vector(noisy_word)
        scored: List[Tuple[str, float]] = []

        for cand in candidates:
            v_cand = self.get_word_vector(cand)
            if v_cand is not None and v_noisy is not None:
                dot = sum(a * b for a, b in zip(v_noisy, v_cand))
                norm1 = sum(a * a for a in v_noisy) ** 0.5
                norm2 = sum(b * b for b in v_cand) ** 0.5
                sim = dot / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0
            else:
                sim = 0.0
            scored.append((cand, sim))

        return sorted(scored, key=lambda x: -x[1])
