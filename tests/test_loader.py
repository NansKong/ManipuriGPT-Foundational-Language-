import pytest
import httpx
from app.datasets.registry import DatasetRegistry


@pytest.mark.slow
def test_load_flores_streaming():
    """
    Integration test: Load the FLORES+ mni_Beng dataset in streaming mode.
    Uses openlanguagedata/flores_plus (successor to deprecated facebook/flores).
    Skips gracefully on network timeouts / HuggingFace outages.
    """
    custom_registry = DatasetRegistry()

    try:
        result = custom_registry.load("flores", split="dev", use_streaming=True)
    except (httpx.ReadTimeout, httpx.ConnectTimeout, ConnectionError, OSError) as e:
        pytest.skip(f"HuggingFace unreachable (transient network issue): {e}")

    # Verify we got a valid iterable dataset back
    assert result is not None

    # Take one sample and verify it has data
    sample = next(iter(result))
    assert sample is not None
    assert isinstance(sample, dict)
    # FLORES+ has a "sentence" column for monolingual configs
    assert "sentence" in sample or "id" in sample


def test_load_local_dataset():
    """
    Integration test: Verify local file loading raises FileNotFoundError
    when the file doesn't exist (no mocks — real code path).
    """
    custom_registry = DatasetRegistry()
    custom_metadata = {
        "provider": "local",
        "streaming": False,
        "type": "translation",
        "local_path": "data/nonexistent_file.jsonl"
    }
    custom_registry.register("local_missing", custom_metadata)

    with pytest.raises(FileNotFoundError):
        custom_registry.load("local_missing")
