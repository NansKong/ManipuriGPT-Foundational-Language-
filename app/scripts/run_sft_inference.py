"""
ManipuriGPT Phase 8 SFT Model Interactive Testing & Inference CLI.
Loads the fine-tuned ManipuriGPT-SFT-v1.0 model and adapter for instruction-following tests.
Usage:
    python app/scripts/run_sft_inference.py
    python app/scripts/run_sft_inference.py --prompt "Translate to Manipuri: Hello, how are you?"
"""

import sys
import argparse
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

BASE_MODEL_PATH = str(BASE_DIR / "models" / "ManipuriGPT-135M-Base-v1.0")
DEFAULT_SFT_V11 = BASE_DIR / "models" / "ManipuriGPT-SFT-v1.1"
DEFAULT_SFT_V10 = BASE_DIR / "models" / "ManipuriGPT-SFT-v1.0"
SFT_MODEL_PATH = str(next((p for p in (BASE_DIR / "models/Meitei-lon-SFT-v1.2", DEFAULT_SFT_V11, DEFAULT_SFT_V10) if (p / "adapter_config.json").exists()), DEFAULT_SFT_V11))

from app.sft.config import SYSTEM_PROMPT
from app.training.meitei_lon import adapter_tokenizer

SAMPLE_PROMPTS = [
    (
        "English to Meetei Mayek Translation",
        "Provide the Meetei Mayek translation for the English sentence below.",
        "Manipur is a beautiful state in India."
    ),
    (
        "Meetei Mayek Conversation",
        "Continue the conversation naturally in Manipuri.",
        "ꯈꯨꯔꯨꯝꯖꯔꯤ! ꯅꯉꯣꯟꯗ ꯋꯥꯍꯪ ꯑꯃ ꯍꯪꯖꯒꯦ꯫"
    ),
    (
        "Meetei Mayek Definition",
        "Provide the definition and explanation for the Manipuri word.",
        "ꯑꯉꯥꯡꯍꯜ"
    ),
    (
        "Bengali Script to Meetei Mayek Conversion",
        "Change the script of this text from Bengali to native Meetei Mayek.",
        "মণিপুরী ভাষা অত্যন্ত প্রাচীন।"
    ),
    (
        "General QA",
        "What is Meetei Mayek and how many base letters does it have?",
        ""
    ),
]


def load_sft_model(sft_path: str = SFT_MODEL_PATH, base_path: str = BASE_MODEL_PATH):
    print("=========================================================")
    print("  Loading ManipuriGPT SFT Model & LoRA Adapter")
    print("=========================================================")
    print(f"[Loading] Base Model: {base_path}")
    print(f"[Loading] SFT Adapter: {sft_path}")

    # Load tokenizer directly from Base Model to preserve exact 32K vocab without adding pseudo special tokens
    tokenizer = adapter_tokenizer(sft_path, base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Running on {device.upper()}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, sft_path)
    model.config.use_cache = True
    model.eval()

    vocab_size = len(tokenizer)
    embed_size = base_model.get_input_embeddings().weight.shape[0]
    print(f"[Tokenizer] Vocab Size: {vocab_size} | Model Embedding Size: {embed_size}")
    print("[Loaded] Model and Adapter successfully loaded into memory!\n")
    return model, tokenizer, device


def format_chatml(instruction: str, input_text: str = "", system_prompt: str = SYSTEM_PROMPT) -> str:
    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content = f"{user_content}\n\n{input_text.strip()}"
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    device: str = "cuda",
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    formatted_prompt = format_chatml(instruction, input_text)

    # Tokenize full prompt with exact subwords (add_special_tokens=False)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=40,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice off prompt input tokens
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Clean any residual ChatML tags if present
    for tag in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>"]:
        if tag in response:
            response = response.split(tag)[0]

    return response.strip()


def run_benchmark_tests(model, tokenizer, device):
    print("=========================================================")
    print("  Running Quick SFT Sample Benchmark Tests")
    print("=========================================================\n")

    for idx, (task_label, instruction, input_text) in enumerate(SAMPLE_PROMPTS, 1):
        print(f"--- Test #{idx}: {task_label} ---")
        print(f"INSTRUCTION: {instruction}")
        if input_text:
            print(f"INPUT: {input_text}")
        response = generate_response(model, tokenizer, instruction, input_text, device)
        print(f"RESPONSE:\n{response}")
        print("-" * 55 + "\n")


def interactive_loop(model, tokenizer, device):
    print("=========================================================")
    print("  ManipuriGPT SFT Interactive Terminal (Type 'exit' to quit)")
    print("=========================================================\n")

    while True:
        try:
            user_input = input("\n[User Prompt] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive testing session.")
                break

            print("\nGenerating response...")
            response = generate_response(model, tokenizer, user_input, "", device)
            print(f"\n[ManipuriGPT Response]:\n{response}\n")
        except KeyboardInterrupt:
            print("\nExiting interactive testing session.")
            break


def main():
    parser = argparse.ArgumentParser(description="ManipuriGPT SFT Model Testing CLI")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to evaluate")
    parser.add_argument("--benchmark", action="store_true", help="Run preset sample prompts benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max output tokens (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3)")
    parser.add_argument("--adapter", default=SFT_MODEL_PATH)
    parser.add_argument("--base-model", default=BASE_MODEL_PATH)
    args = parser.parse_args()

    model, tokenizer, device = load_sft_model(args.adapter, args.base_model)

    if args.prompt:
        print(f"PROMPT: {args.prompt}")
        response = generate_response(
            model, tokenizer, args.prompt, "", device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"\n[ManipuriGPT Response]:\n{response}\n")
    elif args.benchmark:
        run_benchmark_tests(model, tokenizer, device)
    else:
        # Default run sample benchmark then enter interactive loop
        run_benchmark_tests(model, tokenizer, device)
        interactive_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()
