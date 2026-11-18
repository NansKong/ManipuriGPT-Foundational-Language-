"""
Prompt Formatter module for ManipuriGPT Phase 8 SFT.
Handles ChatML prompt construction and loss masking (-100 masking for system/user tokens).
"""

from typing import Dict, Any, Tuple, List, Optional
from app.sft.config import SYSTEM_PROMPT, CHATML_START, CHATML_END, IGNORE_INDEX


def format_chatml_record(
    instruction: str,
    input_text: str = "",
    output_text: str = "",
    system_prompt: str = SYSTEM_PROMPT
) -> Dict[str, str]:
    """
    Formats instruction, input, and output into standard ChatML components.
    """
    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content = f"{user_content}\n\n{input_text.strip()}"

    system_turn = f"{CHATML_START}system\n{system_prompt}{CHATML_END}\n"
    user_turn = f"{CHATML_START}user\n{user_content}{CHATML_END}\n"
    assistant_turn = f"{CHATML_START}assistant\n{output_text.strip()}{CHATML_END}"

    full_prompt = f"{system_turn}{user_turn}{assistant_turn}"
    prompt_without_response = f"{system_turn}{user_turn}{CHATML_START}assistant\n"

    return {
        "full_text": full_prompt,
        "prompt_text": prompt_without_response,
        "response_text": output_text.strip(),
        "system_prompt": system_prompt,
    }


class ChatMLCollator:
    """
    PyTorch / HuggingFace Dataset DataCollator that tokenizes ChatML sequences
    and constructs labels with -100 masking over non-assistant tokens.
    """

    def __init__(self, tokenizer: Any, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for item in batch:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")

            formatted = format_chatml_record(instruction, input_text, output_text)

            if self.max_seq_length < 3 or not formatted["response_text"]:
                raise ValueError("SFT requires a nonempty answer and at least three token positions")
            prompt = self.tokenizer(formatted["prompt_text"], add_special_tokens=False)["input_ids"]
            answer = self.tokenizer(formatted["response_text"], add_special_tokens=False)["input_ids"]
            # Preserve the assistant header at the right end of overlong prompts.
            answer = answer[:self.max_seq_length - 2] + [self.tokenizer.eos_token_id]
            prompt = prompt[-(self.max_seq_length - len(answer)):]
            input_ids = prompt + answer
            attention_mask = [1] * len(input_ids)
            labels = [IGNORE_INDEX] * len(prompt) + answer

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        # Pad to max length in batch
        max_len = max(len(ids) for ids in input_ids_batch)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, lbls in zip(input_ids_batch, attention_mask_batch, labels_batch):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(lbls + [IGNORE_INDEX] * pad_len)

        import torch
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
