"""
Inference CLI module for ManipuriGPT Phase 8 ChatML Playground.
Loads base model + SFT QLoRA adapter for interactive chat and evaluation.
"""

import sys
import torch
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from app.sft.config import (
    SYSTEM_PROMPT,
    DEFAULT_OUTPUT_MODEL_DIR, resolve_model_path,
    CHATML_START,
    CHATML_END,
)
from app.sft.prompt_formatter import format_chatml_record
from app.training.meitei_lon import adapter_tokenizer


class ManipuriGPTChatSession:
    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model_path: str = resolve_model_path(),
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if model_path is None:
            legacy = DEFAULT_OUTPUT_MODEL_DIR.parent / "ManipuriGPT-SFT-v1.1"
            model_path = str(DEFAULT_OUTPUT_MODEL_DIR if (DEFAULT_OUTPUT_MODEL_DIR / "adapter_config.json").exists() else legacy)
        self.system_prompt = system_prompt
        print(f"[Inference] Loading tokenizer from: {model_path}")
        self.tokenizer = adapter_tokenizer(model_path, base_model_path)

        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[Inference] Loading base model: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        if Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists():
            print(f"[Inference] Loading LoRA adapter from: {model_path}")
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            print("[Inference] No adapter found, using base model directly.")
            self.model = base_model

        self.model.eval()

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        formatted = format_chatml_record(
            instruction=instruction,
            input_text=input_text,
            system_prompt=self.system_prompt,
        )
        prompt_text = formatted["prompt_text"]

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0,
                eos_token_id=self.tokenizer.encode(CHATML_END)[0]
                if CHATML_END in self.tokenizer.get_vocab()
                else self.tokenizer.eos_token_id,
            )

        gen_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return gen_text.strip()


def run_cli():
    print("=" * 60)
    print("      ManipuriGPT Interactive ChatML CLI Playground")
    print("=" * 60)

    session = ManipuriGPTChatSession()
    print("\nType your instruction/question below (or 'exit' to quit):\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Exiting playground.")
                break
            if not user_input:
                continue

            response = session.generate(instruction=user_input)
            print(f"\nAssistant > {response}\n")
        except KeyboardInterrupt:
            print("\nExiting playground.")
            break


if __name__ == "__main__":
    run_cli()
