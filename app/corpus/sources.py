"""
Corpus source definitions and configuration specifications for streaming ingestion.
Supports multiple high-quality and large-scale datasets listed in Phase 5 specification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class CorpusSourceSpec:
    """
    Configuration specification for a single dataset stream source.
    """
    name: str
    source_type: str  # "hf", "web", "file", "api"
    dataset_path: str
    subset: Optional[str] = None
    split: str = "train"
    supported_languages: List[str] = field(default_factory=lambda: ["en", "mni", "bn"])
    supports_streaming: bool = True
    caching_strategy: str = "stream"  # "local_cache", "shard_prefetch", "stream"
    default_text_column: str = "text"
    metadata_columns: List[str] = field(default_factory=list)
    description: str = ""
    license: str = "Various"
    extra_configs: Dict[str, Any] = field(default_factory=dict)


# Centralized registry of all supported Phase 5 corpus sources
SOURCE_REGISTRY: Dict[str, CorpusSourceSpec] = {
    "huggingface_datasets": CorpusSourceSpec(
        name="huggingface_datasets",
        source_type="hf",
        dataset_path="allenai/c4",
        subset="en",
        split="train",
        supported_languages=["en", "mni", "bn"],
        default_text_column="text",
        description="General Hugging Face datasets streaming hub endpoint."
    ),
    "common_crawl": CorpusSourceSpec(
        name="common_crawl",
        source_type="hf",
        dataset_path="allenai/c4",
        subset="en",
        split="train",
        supported_languages=["en"],
        default_text_column="text",
        description="Common Crawl web dumps via clean streaming subsets."
    ),
    "wikipedia": CorpusSourceSpec(
        name="wikipedia",
        source_type="hf",
        dataset_path="wikimedia/wikipedia",
        subset="20231101.en",
        split="train",
        supported_languages=["en", "bn", "mni"],
        caching_strategy="local_cache",
        default_text_column="text",
        metadata_columns=["url", "title"],
        description="Wikipedia articles dumped across target languages."
    ),
    "oscar": CorpusSourceSpec(
        name="oscar",
        source_type="hf",
        dataset_path="allenai/c4",
        subset="multilingual",
        split="train",
        supported_languages=["en", "bn"],
        default_text_column="text",
        metadata_columns=["url", "timestamp"],
        description="mC4 multilingual web dataset (replacing gated OSCAR/CulturaX)."
    ),
    "cc100": CorpusSourceSpec(
        name="cc100",
        source_type="hf",
        dataset_path="DKYoon/SlimPajama-6B",
        subset=None,
        split="train",
        supported_languages=["en"],
        default_text_column="text",
        metadata_columns=["meta", "redpajama_set_name"],
        description="SlimPajama-6B open pretraining dataset (replacing deprecated CC100 script)."
    ),
    "slimpajama": CorpusSourceSpec(
        name="slimpajama",
        source_type="hf",
        dataset_path="DKYoon/SlimPajama-6B",
        subset=None,
        split="train",
        supported_languages=["en"],
        default_text_column="text",
        metadata_columns=["meta", "redpajama_set_name"],
        description="SlimPajama-6B open pretraining dataset for development."
    ),
    "fineweb": CorpusSourceSpec(
        name="fineweb",
        source_type="hf",
        dataset_path="HuggingFaceFW/fineweb",
        subset="sample-10BT",
        split="train",
        supported_languages=["en"],
        default_text_column="text",
        metadata_columns=["url", "dump", "file_path"],
        description="High-quality filtered educational and informative web dataset."
    ),
    "c4": CorpusSourceSpec(
        name="c4",
        source_type="hf",
        dataset_path="allenai/c4",
        subset="en",
        split="train",
        supported_languages=["en"],
        default_text_column="text",
        metadata_columns=["url", "timestamp"],
        description="Colossal Clean Crawled Corpus (C4)."
    ),
    "arxiv": CorpusSourceSpec(
        name="arxiv",
        source_type="hf",
        dataset_path="CurationCorp/arxiv-abstracts",
        subset=None,
        split="train",
        supported_languages=["en"],
        default_text_column="abstract",
        metadata_columns=["title", "authors", "categories"],
        description="Scientific paper abstracts and STEM documents from arXiv."
    ),
    "pubmed": CorpusSourceSpec(
        name="pubmed",
        source_type="hf",
        dataset_path="ncbi/pubmed",
        subset=None,
        split="train",
        supported_languages=["en"],
        default_text_column="AbstractText",
        metadata_columns=["Title", "PMID"],
        description="Biomedical literature and clinical research abstracts."
    ),
    "stackexchange": CorpusSourceSpec(
        name="stackexchange",
        source_type="hf",
        dataset_path="HuggingFaceH4/stack-exchange-preferences",
        subset=None,
        split="train",
        supported_languages=["en"],
        default_text_column="question",
        metadata_columns=["answers"],
        description="Q&A technical and scientific discussions from StackExchange."
    ),
    "github_code": CorpusSourceSpec(
        name="github_code",
        source_type="hf",
        dataset_path="codeparrot/github-code",
        subset="all-all",
        split="train",
        supported_languages=["en", "code"],
        default_text_column="code",
        metadata_columns=["repo_name", "path", "language"],
        description="High-quality open source code repositories from GitHub."
    ),
    "opensubtitles": CorpusSourceSpec(
        name="opensubtitles",
        source_type="hf",
        dataset_path="open_subtitles",
        subset="en-bn",
        split="train",
        supported_languages=["en", "bn", "mni"],
        default_text_column="translation",
        description="Conversational subtitle alignments for dialogue modeling."
    ),
    "wiktionary": CorpusSourceSpec(
        name="wiktionary",
        source_type="hf",
        dataset_path="kaist-ai/wiktionary",
        subset="en",
        split="train",
        supported_languages=["en", "mni", "bn"],
        default_text_column="text",
        metadata_columns=["title", "definitions"],
        description="Multilingual dictionary definitions and etymological entries."
    ),
    "opus": CorpusSourceSpec(
        name="opus",
        source_type="hf",
        dataset_path="Helsinki-NLP/opus-100",
        subset="bn-en",
        split="train",
        supported_languages=["en", "bn", "mni"],
        default_text_column="translation",
        description="OPUS-100 parallel translation corpus."
    ),
    "ai4bharat": CorpusSourceSpec(
        name="ai4bharat",
        source_type="hf",
        dataset_path="ai4bharat/sangraha",
        subset="verified",
        split="mni",
        supported_languages=["mni", "bn", "hi", "en"],
        caching_strategy="local_cache",
        default_text_column="text",
        metadata_columns=["source", "language", "script"],
        description="AI4Bharat Indic languages dataset suite including Sangraha (split='mni')."
    ),
    "manipuri_specific": CorpusSourceSpec(
        name="manipuri_specific",
        source_type="hf",
        dataset_path="local/manipuri_corpus",
        subset="meitei_mayek",
        split="train",
        supported_languages=["mni"],
        caching_strategy="local_cache",
        default_text_column="text",
        metadata_columns=["script", "source", "category"],
        description="Curated Manipuri domain corpus across Meitei Mayek, Romanized, and Bengali scripts."
    ),
    "dayananda_meitei_mayek_sample": CorpusSourceSpec(
        name="dayananda_meitei_mayek_sample",
        source_type="hf",
        dataset_path="DayanandaThokchom/meitei_mayek_sample",
        subset=None,
        split="train",
        supported_languages=["mni"],
        caching_strategy="local_cache",
        default_text_column="meitei_mayek",
        metadata_columns=["english", "meitei_only"],
        description="Clean Meitei Mayek sample corpus curated by Dayananda Thokchom."
    ),
    "dayananda_meitei_to_english": CorpusSourceSpec(
        name="dayananda_meitei_to_english",
        source_type="hf",
        dataset_path="DayanandaThokchom/meitei-mayek-to-english",
        subset=None,
        split="train",
        supported_languages=["mni", "en"],
        caching_strategy="local_cache",
        default_text_column="meitei_mayek",
        metadata_columns=["english"],
        description="Parallel corpus aligning Meitei Lon in canonical Meitei Mayek script with English."
    ),
    "dayananda_english_to_meitei": CorpusSourceSpec(
        name="dayananda_english_to_meitei",
        source_type="hf",
        dataset_path="DayanandaThokchom/english-TO-meitei-mayek",
        subset=None,
        split="train",
        supported_languages=["mni", "en"],
        caching_strategy="local_cache",
        default_text_column="meitei_mayek",
        metadata_columns=["english", "src_lang", "tgt_lang"],
        description="Parallel corpus aligning English with canonical Meitei Mayek script by Dayananda Thokchom."
    ),
    "joyson_bible": CorpusSourceSpec(
        name="joyson_bible",
        source_type="hf",
        dataset_path="joyson117/english-manipuri-parallel-corpus",
        subset="bible",
        split="bible",
        supported_languages=["mni", "en"],
        caching_strategy="local_cache",
        default_text_column="Manipuri",
        metadata_columns=["English"],
        description="Bible parallel corpus (31K sentences) — English ↔ Manipuri by Joyson."
    ),
    "joyson_pib_pmi": CorpusSourceSpec(
        name="joyson_pib_pmi",
        source_type="hf",
        dataset_path="joyson117/english-manipuri-parallel-corpus",
        subset="pib-pmi",
        split="pib_pmi",
        supported_languages=["mni", "en"],
        caching_strategy="local_cache",
        default_text_column="Manipuri",
        metadata_columns=["English"],
        description="PIB-PMI parallel corpus (497K sentences) — English ↔ Manipuri by Joyson."
    ),

    # ---------------------------------------------------------------
    # Phase 5.5 — Local file sources for master corpus scaling
    # ---------------------------------------------------------------
    "local_processed_pdfs": CorpusSourceSpec(
        name="local_processed_pdfs",
        source_type="local",
        dataset_path="cache/processed",
        split="train",
        supported_languages=["mni", "en"],
        caching_strategy="local_cache",
        default_text_column="text",
        metadata_columns=["category", "source", "script", "quality", "ocr_confidence"],
        description="OCR'd Manipuri PDFs from Bharatavani / CIIL (14 JSONL files, ~3.8K records).",
        license="To be determined",
        extra_configs={"format": "jsonl"},
    ),
    "local_ema_lon_mono": CorpusSourceSpec(
        name="local_ema_lon_mono",
        source_type="local",
        dataset_path="cache/datasets/W0316/Manipuri resources/EM Corpus/EM Corpus monolingual Manipuri/monolingual_Manipuri_v2",
        split="train",
        supported_languages=["mni"],
        caching_strategy="local_cache",
        default_text_column="text",
        description="EMA Lon monolingual Manipuri corpus (~2K lines).",
        license="CC BY-NC 4.0",
        extra_configs={"format": "txt"},
    ),
    "local_ema_lon_parallel": CorpusSourceSpec(
        name="local_ema_lon_parallel",
        source_type="local",
        dataset_path="cache/datasets/W0316/Manipuri resources/EM Corpus/EM Corpus parallel Manipuri-English/bilingual_manipuri_english_v2",
        split="train",
        supported_languages=["mni", "en"],
        caching_strategy="local_cache",
        default_text_column="text",
        description="EMA Lon bilingual Manipuri-English corpus (~8.9K pairs).",
        license="CC BY-NC 4.0",
        extra_configs={"format": "tsv"},
    ),
    "local_sangraha_cached": CorpusSourceSpec(
        name="local_sangraha_cached",
        source_type="local",
        dataset_path="cache/datasets/ai4bharat___sangraha",
        split="train",
        supported_languages=["mni"],
        caching_strategy="local_cache",
        default_text_column="text",
        description="AI4Bharat Sangraha Manipuri (local HF cache, ~112K records).",
        license="CC BY 4.0",
        extra_configs={"format": "arrow"},
    ),
    "local_dayananda_meitei": CorpusSourceSpec(
        name="local_dayananda_meitei",
        source_type="local",
        dataset_path="cache/datasets/DayanandaThokchom___meitei_mayek_sample",
        split="train",
        supported_languages=["mni"],
        caching_strategy="local_cache",
        default_text_column="meitei_mayek",
        metadata_columns=["english", "meitei_only"],
        description="Dayananda Thokchom Meitei Mayek sample (local cache, ~110K records).",
        license="Various",
        extra_configs={"format": "parquet"},
    ),
    "local_dayananda_eng_to_meitei": CorpusSourceSpec(
        name="local_dayananda_eng_to_meitei",
        source_type="local",
        dataset_path="cache/datasets/DayanandaThokchom___english-to-meitei-mayek",
        split="train",
        supported_languages=["mni", "en"],
        caching_strategy="local_cache",
        default_text_column="meitei_mayek",
        metadata_columns=["english"],
        description="Dayananda Thokchom English-to-Meitei Mayek (local cache, ~28K pairs).",
        license="Various",
        extra_configs={"format": "parquet"},
    ),
    "local_joyson_parallel": CorpusSourceSpec(
        name="local_joyson_parallel",
        source_type="local",
        dataset_path="cache/datasets/joyson117___english-manipuri-parallel-corpus",
        split="train",
        supported_languages=["mni", "en"],
        caching_strategy="local_cache",
        default_text_column="Manipuri",
        metadata_columns=["English"],
        description="Joyson English-Manipuri parallel corpus (local cache, ~18K pairs).",
        license="Various",
        extra_configs={"format": "arrow"},
    ),
}


def get_source_spec(name: str) -> CorpusSourceSpec:
    """
    Retrieves a CorpusSourceSpec from the registry by canonical name.
    """
    key = name.lower().replace(" ", "_")
    if key not in SOURCE_REGISTRY:
        available = sorted(SOURCE_REGISTRY.keys())
        raise KeyError(f"Source '{name}' not found in SOURCE_REGISTRY. Available: {available}")
    return SOURCE_REGISTRY[key]
