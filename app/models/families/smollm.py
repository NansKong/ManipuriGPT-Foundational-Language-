"""
SmolLM model family specifications (Phase 5).
Compact, highly efficient foundation models for on-device and specialized fine-tuning.
"""

from app.models.registry import ModelSpecification, model_registry

SMOLLM_135M = ModelSpecification(
    short_name="smollm_135m",
    name="HuggingFaceTB/SmolLM2-135M-Instruct",
    max_context_length=8192,
    chat_template="{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["chat", "instruction", "pretraining"],
    eos_token="<|im_end|>",
    bos_token="<|im_start|>",
    pad_token="<|im_end|>",
    memory_estimates={"4bit_vram_gb": 1.2, "8bit_vram_gb": 1.8, "16bit_vram_gb": 2.8},
    rope_scaling={"type": "default", "factor": 1.0}
)

SMOLLM_360M = ModelSpecification(
    short_name="smollm_360m",
    name="HuggingFaceTB/SmolLM2-360M-Instruct",
    max_context_length=8192,
    chat_template=SMOLLM_135M.chat_template,
    lora_target_modules=SMOLLM_135M.lora_target_modules,
    supported_tasks=["chat", "instruction", "pretraining"],
    eos_token="<|im_end|>",
    bos_token="<|im_start|>",
    pad_token="<|im_end|>",
    memory_estimates={"4bit_vram_gb": 1.6, "8bit_vram_gb": 2.6, "16bit_vram_gb": 4.5},
    rope_scaling={"type": "default", "factor": 1.0}
)

SMOLLM_1_7B = ModelSpecification(
    short_name="smollm_1_7b",
    name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    max_context_length=8192,
    chat_template=SMOLLM_135M.chat_template,
    lora_target_modules=SMOLLM_135M.lora_target_modules,
    supported_tasks=["chat", "instruction", "pretraining", "reasoning"],
    eos_token="<|im_end|>",
    bos_token="<|im_start|>",
    pad_token="<|im_end|>",
    memory_estimates={"4bit_vram_gb": 3.2, "8bit_vram_gb": 5.0, "16bit_vram_gb": 8.5},
    rope_scaling={"type": "default", "factor": 1.0}
)

model_registry.register(SMOLLM_135M)
model_registry.register(SMOLLM_360M)
model_registry.register(SMOLLM_1_7B)
