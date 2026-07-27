"""
Phi model family specifications (Phase 5).
Includes Phi-3 and Phi-3.5 variants with memory estimates and RoPE scaling.
"""

from app.models.registry import ModelSpecification, model_registry

PHI_3_MINI_4K = ModelSpecification(
    short_name="phi_3_mini_4k",
    name="microsoft/Phi-3-mini-4k-instruct",
    max_context_length=4096,
    chat_template="{% for message in messages %}{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}<|end|>\n{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}<|end|>\n{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}<|end|>\n{% endif %}{% endfor %}",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
    supported_tasks=["chat", "instruction", "reasoning"],
    eos_token="<|end|>",
    bos_token="<s>",
    pad_token="<|end|>",
    memory_estimates={"4bit_vram_gb": 3.8, "8bit_vram_gb": 6.5, "16bit_vram_gb": 12.0},
    rope_scaling={"type": "su", "factor": 1.0}
)

PHI_3_MEDIUM_128K = ModelSpecification(
    short_name="phi_3_medium_128k",
    name="microsoft/Phi-3-medium-128k-instruct",
    max_context_length=131072,
    chat_template=PHI_3_MINI_4K.chat_template,
    lora_target_modules=PHI_3_MINI_4K.lora_target_modules,
    supported_tasks=["chat", "instruction", "reasoning", "rag"],
    eos_token="<|end|>",
    bos_token="<s>",
    pad_token="<|end|>",
    memory_estimates={"4bit_vram_gb": 11.5, "8bit_vram_gb": 22.0, "16bit_vram_gb": 44.0},
    rope_scaling={"type": "su", "factor": 32.0}
)

PHI_3_5_MINI = ModelSpecification(
    short_name="phi_3_5_mini",
    name="microsoft/Phi-3.5-mini-instruct",
    max_context_length=131072,
    chat_template=PHI_3_MINI_4K.chat_template,
    lora_target_modules=PHI_3_MINI_4K.lora_target_modules,
    supported_tasks=["chat", "instruction", "reasoning", "code"],
    eos_token="<|end|>",
    bos_token="<s>",
    pad_token="<|end|>",
    memory_estimates={"4bit_vram_gb": 4.2, "8bit_vram_gb": 7.5, "16bit_vram_gb": 14.0},
    rope_scaling={"type": "su", "factor": 32.0}
)

model_registry.register(PHI_3_MINI_4K)
model_registry.register(PHI_3_MEDIUM_128K)
model_registry.register(PHI_3_5_MINI)
