import pytest
from app.tokenizer.packing import SequencePacker

def test_sequence_packer_basic_chunking():
    packer = SequencePacker(max_length=10, eos_token_id=2, pad_token_id=0)
    batch = {
        "input_ids": [[1, 3], [4, 5, 6], [7, 8, 9, 10, 11]],
        "attention_mask": [[1, 1], [1, 1, 1], [1, 1, 1, 1, 1]],
        "labels": [[1, 3], [4, 5, 6], [7, 8, 9, 10, 11]]
    }
    # Each sequence gets + [2] when concatenated if not ending in 2
    # [1,3,2] (len 3) + [4,5,6,2] (len 4) + [7,8,9,10,11,2] (len 6) = 13 tokens total
    # With max_length=10, we should get 1 complete chunk of length 10
    packed = packer.pack(batch)
    assert len(packed["input_ids"]) == 1
    assert len(packed["input_ids"][0]) == 10
    assert packed["input_ids"][0] == [1, 3, 2, 4, 5, 6, 2, 7, 8, 9]

def test_sequence_packer_short_sequence_padding():
    packer = SequencePacker(max_length=10, eos_token_id=2, pad_token_id=0)
    batch = {
        "input_ids": [[1, 3]],
        "attention_mask": [[1, 1]],
        "labels": [[1, 3]]
    }
    # [1, 3, 2] -> total 3 tokens. Less than max_length 10. Should pad to 10.
    packed = packer.pack(batch)
    assert len(packed["input_ids"]) == 1
    assert len(packed["input_ids"][0]) == 10
    assert packed["input_ids"][0][:3] == [1, 3, 2]
    assert packed["input_ids"][0][3:] == [0] * 7
    # Labels should be padded with -100
    assert packed["labels"][0][3:] == [-100] * 7
    assert packed["attention_mask"][0][3:] == [0] * 7
