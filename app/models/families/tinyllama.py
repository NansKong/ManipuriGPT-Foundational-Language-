"""
TinyLlama model family specifications (Phase 5).
Lightweight 1.1B foundation model trained across 3T tokens.
"""

from app.models.registry import ModelSpecification, model_registry

TINYLLAMA_1_1B = ModelSpecification(
    short_name="tinyllama_1_1b",
    name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_context_length=2048,
    chat_template="{% for message in messages %}{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}</s>\n{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}</s>\n{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}</s>\n{% endif %}{% endfor %}",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["chat", "instruction", "pretraining", "translation"],
    eos_token="</s>",
    bos_token="<s>",
    pad_token="</s>",
    memory_estimates={"4bit_vram_gb": 2.5, "8bit_vram_gb": 4.0, "16bit_vram_gb": 6.8},
    rope_scaling={"type": "default", "factor": 1.0}
)

model_registry.register(TINYLLAMA_1_1B)
