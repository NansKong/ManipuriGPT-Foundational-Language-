"""
SequenceChunker module for token-aware, word-aware, and character-aware sliding window
chunking of large documents into uniform context windows prior to tokenization.
"""

from typing import List, Dict, Any, Union


class SequenceChunker:
    """
    Splits long document strings into smaller, uniform chunks with configurable overlap.
    Supports word-level or character-level approximation windows.
    """
    def __init__(
        self,
        max_chunk_size: int = 2048,
        chunk_overlap: int = 128,
        mode: str = "word"  # "word", "char"
    ):
        if chunk_overlap >= max_chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than max_chunk_size")
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.mode = mode.lower()

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of maximum size `max_chunk_size` overlapping by `chunk_overlap`.
        """
        if not text or not isinstance(text, str):
            return []

        clean = text.strip()
        if not clean:
            return []

        if self.mode == "word":
            units = clean.split()
            delimiter = " "
        else:
            units = list(clean)
            delimiter = ""

        if len(units) <= self.max_chunk_size:
            return [clean]

        chunks: List[str] = []
        step = self.max_chunk_size - self.chunk_overlap
        for i in range(0, len(units), step):
            chunk_units = units[i:i + self.max_chunk_size]
            chunk_str = delimiter.join(chunk_units).strip()
            if chunk_str:
                chunks.append(chunk_str)
            if i + self.max_chunk_size >= len(units):
                break

        return chunks

    def process_example(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Takes a single dataset example and yields one or more chunked examples.
        Preserves metadata and adds `chunk_index` and `total_chunks`.
        """
        text = example.get("text", "")
        chunks = self.chunk_text(text)
        
        results: List[Dict[str, Any]] = []
        for idx, chunk_str in enumerate(chunks):
            metadata = example.get("metadata", {}).copy()
            metadata["chunk_index"] = idx
            metadata["total_chunks"] = len(chunks)
            
            chunk_ex = example.copy()
            chunk_ex["text"] = chunk_str
            chunk_ex["metadata"] = metadata
            results.append(chunk_ex)

        return results
