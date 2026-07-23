import pytest
from app.models import model_registry, ModelSpecification
from app.tokenizer.context import PipelineContext

class DummyTokenizerForModels:
    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.pad_token_id = 0

def test_model_registry_lookup_families():
    models = model_registry.list_models()
    assert "qwen2.5" in models
    assert "qwen2.5-7b" in models
    assert "llama3.2" in models
    assert "llama3.2-1b" in models
    assert "gemma3" in models
    assert "gemma3-12b" in models
    assert "mistral" in models

    spec = model_registry.get("qwen2.5")
    assert spec.max_context_length == 32768
    assert "q_proj" in spec.lora_target_modules

def test_model_specification_tokenizer():
    spec = model_registry.get("llama3.2")
    dummy = DummyTokenizerForModels()
    manager = spec.tokenizer(tokenizer_instance=dummy)
    assert manager.max_length == 131072
    assert manager.tokenizer == dummy

def test_pipeline_context_initialization():
    spec = model_registry.get("qwen2.5")
    ctx = PipelineContext(
        dataset_name="flores",
        model=spec,
        task="translation",
        artifact_dir="artifacts/"
    )
    assert ctx.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert ctx.model_short_name == "qwen2.5"
    assert ctx.task_name == "translation"
    assert ctx.get_artifact_path("reports/summary.md") == "artifacts/reports/summary.md"
