import pytest
from datasets import Dataset
try:
    from datasets import IterableDataset
except ImportError:
    IterableDataset = None

from app.tokenizer.pipeline import TokenizationPipeline
from app.tokenizer.tokenizer_manager import TokenizerManager
from app.tokenizer.context import PipelineContext

class MockTokenizerForPipeline:
    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "</s>"
        self.mask_token = "[MASK]"
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.mask_token_id = 4
        self.padding_side = "right"

    def __call__(self, text, **kwargs):
        tokens = [ord(c) % 50 + 10 for c in text] + [self.eos_token_id]
        if kwargs.get("truncation") and kwargs.get("max_length"):
            max_l = kwargs["max_length"]
            if len(tokens) > max_l:
                tokens = tokens[:max_l]
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens)
        }

@pytest.fixture
def dummy_tokenizer_manager():
    return TokenizerManager(
        config={"max_length": 128, "padding_side": "right"},
        tokenizer_instance=MockTokenizerForPipeline()
    )

def test_tokenization_pipeline_with_context_and_hooks(dummy_tokenizer_manager):
    pipeline = TokenizationPipeline(
        config={"max_length": 128, "packing": False},
        tokenizer_manager=dummy_tokenizer_manager
    )
    ctx = PipelineContext(
        dataset_name="test_flores",
        task="translation",
        packing=False,
        max_length=128
    )
    data = [
        {"en": "Hello world", "mni": "Chat"},
        {"en": "", "mni": "Empty source should be skipped by validate()"},
        {"en": "Good morning", "mni": "Ayuk"}
    ]
    ds = Dataset.from_list(data)
    out_ds = pipeline.run(ds, ctx=ctx)
    
    assert len(out_ds) == 2
    assert "input_ids" in out_ds.column_names
    assert "labels" in out_ds.column_names
    
    stages_completed = [s["stage"] for s in ctx.stage_history if s["status"] == "completed"]
    assert "prepare_and_tokenize" in stages_completed
    assert "validate" in stages_completed

def test_tokenization_pipeline_iterable_dataset(dummy_tokenizer_manager):
    if IterableDataset is None:
        pytest.skip("IterableDataset not supported in this environment")
    
    def gen_data():
        yield {"text": "Sentence one for pretraining."}
        yield {"text": "Sentence two for pretraining."}

    iter_ds = IterableDataset.from_generator(gen_data)
    pipeline = TokenizationPipeline(
        config={"max_length": 128, "packing": False},
        tokenizer_manager=dummy_tokenizer_manager
    )
    ctx = PipelineContext(dataset_name="stream_pretrain", task="pretraining", packing=False)
    out_iter = pipeline.run(iter_ds, ctx=ctx)
    first_item = next(iter(out_iter))
    assert "input_ids" in first_item
    assert "labels" in first_item
