import pytest
from app.tokenization.versioning import TokenizerVersionManager


def test_tokenizer_version_manager_tier_directories(tmp_path):
    mgr = TokenizerVersionManager(base_output_dir=str(tmp_path))
    
    assert mgr.get_tier_directory("v0-experimental") == str(tmp_path / "v0-experimental")
    assert mgr.get_tier_directory("v1-pretrain") == str(tmp_path / "v1-pretrain")
    
    with pytest.raises(ValueError):
        mgr.get_tier_directory("v99-invalid")


def test_validate_corpus_for_tier():
    mgr = TokenizerVersionManager()
    
    # v0-experimental allows any size
    assert mgr.validate_corpus_for_tier(1000, "v0-experimental") is True
    
    # v1-pretrain requires >= 50 MB (50 * 1024 * 1024 = 52428800 bytes)
    with pytest.raises(RuntimeError, match="below the minimum threshold"):
        mgr.validate_corpus_for_tier(10 * 1024 * 1024, "v1-pretrain")
        
    # Overriding with dev_mode or force
    assert mgr.validate_corpus_for_tier(10 * 1024 * 1024, "v1-pretrain", dev_mode=True) is False
    assert mgr.validate_corpus_for_tier(10 * 1024 * 1024, "v1-pretrain", force=True) is False
    
    # Valid size >= 50 MB passes cleanly
    assert mgr.validate_corpus_for_tier(60 * 1024 * 1024, "v1-pretrain") is True


def test_save_version_metadata_and_cards(tmp_path):
    import os
    mgr = TokenizerVersionManager(base_output_dir=str(tmp_path))
    
    meta_path = mgr.save_version_metadata(
        tier="v0-experimental",
        algorithm="sentencepiece_unigram",
        vocab_size=8192,
        training_metadata={"training_samples": 500, "total_characters_observed": 50000},
        evaluation_summary={"fertility": 1.25, "stability_score": 99.2, "stability_rating": "High (>=98%)"}
    )
    
    target_dir = os.path.dirname(meta_path)
    assert os.path.exists(meta_path)
    assert os.path.exists(os.path.join(target_dir, "training_config.json"))
    assert os.path.exists(os.path.join(target_dir, "README.md"))
    assert os.path.exists(os.path.join(target_dir, "model_card.md"))
    
    with open(os.path.join(target_dir, "README.md"), "r", encoding="utf-8") as f:
        readme_content = f.read()
        assert "sentencepiece_unigram" in readme_content
        assert "8,192" in readme_content
        assert "99.2%" in readme_content
