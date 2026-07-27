"""
Falcon model family specifications (Phase 5).
Includes state-space hybrid architectures and scale defaults.
"""

from app.models.registry import ModelSpecification, model_registry

FALCON_MAMBA_7B = ModelSpecification(
    short_name="falcon_mamba_7b",
    name="tiiuae/falcon-mamba-7b",
    max_context_length=8192,
    chat_template="{% for message in messages %}{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}<|endoftext|>\n{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}<|endoftext|>\n{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}<|endoftext|>\n{% endif %}{% endfor %}",
    lora_target_modules=["in_proj", "x_proj", "dt_proj", "out_proj"],
    supported_tasks=["chat", "instruction", "pretraining"],
    eos_token="<|endoftext|>",
    bos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
    memory_estimates={"4bit_vram_gb": 6.8, "8bit_vram_gb": 12.0, "16bit_vram_gb": 22.0},
    rope_scaling={"type": "default", "factor": 1.0}
)

FALCON_180B_DEFAULTS = ModelSpecification(
    short_name="falcon_180b_defaults",
    name="tiiuae/falcon-180B-chat",
    max_context_length=2048,
    chat_template=FALCON_MAMBA_7B.chat_template,
    lora_target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    supported_tasks=["chat", "instruction", "reasoning"],
    eos_token="<|endoftext|>",
    bos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
    memory_estimates={"4bit_vram_gb": 110.0, "8bit_vram_gb": 210.0, "16bit_vram_gb": 400.0},
    rope_scaling={"type": "default", "factor": 1.0}
)

model_registry.register(FALCON_MAMBA_7B)
model_registry.register(FALCON_180B_DEFAULTS)
