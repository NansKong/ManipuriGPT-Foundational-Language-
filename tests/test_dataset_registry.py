import pytest
from app.datasets.registry import registry, DatasetRegistry

def test_default_registry_keys():
    expected_keys = ["flores", "manipuri_corpus", "indiccorp"]
    datasets = registry.list_datasets()
    for key in expected_keys:
        assert key in datasets

def test_registry_get_metadata():
    meta = registry.get_metadata("flores")
    assert meta["repo"] == "openlanguagedata/flores_plus"
    assert meta["type"] == "translation"
    assert meta["streaming"] is True
    assert meta["subset"] == "mni_Beng"
    assert meta["provider"] == "huggingface"

def test_registry_get_metadata_missing():
    with pytest.raises(KeyError):
        registry.get_metadata("missing_dataset_name")

def test_dynamic_registration():
    custom_registry = DatasetRegistry()
    custom_metadata = {
        "provider": "huggingface",
        "repo": "custom/repo",
        "streaming": False,
        "type": "monolingual"
    }
    custom_registry.register("new_dataset", custom_metadata)
    assert "new_dataset" in custom_registry.list_datasets()
    assert custom_registry.get_metadata("new_dataset")["repo"] == "custom/repo"
