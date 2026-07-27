"""
Aya model family specifications (Phase 5).
Cohere's state-of-the-art multilingual instruction tuned foundation models covering 23+ languages.
"""

from app.models.registry import ModelSpecification, model_registry

AYA_23_8B = ModelSpecification(
    short_name="aya_23_8b",
    name="CohereForAI/aya-23-8B",
    max_context_length=8192,
    chat_template="{% for message in messages %}{% if message['role'] == 'system' %}<|START_OF_TURN_TOKEN|<|SYSTEM_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>{% elif message['role'] == 'user' %}<|START_OF_TURN_TOKEN|<|USER_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>{% elif message['role'] == 'assistant' %}<|START_OF_TURN_TOKEN|<|CHATBOT_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>{% endif %}{% endfor %}",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["chat", "instruction", "translation", "qa", "pretraining"],
    eos_token="<|END_OF_TURN_TOKEN|>",
    bos_token="<|START_OF_TURN_TOKEN|>",
    pad_token="<|END_OF_TURN_TOKEN|>",
    memory_estimates={"4bit_vram_gb": 7.8, "8bit_vram_gb": 14.0, "16bit_vram_gb": 25.5},
    rope_scaling={"type": "default", "factor": 1.0}
)

AYA_EXPANSE_8B = ModelSpecification(
    short_name="aya_expanse_8b",
    name="CohereForAI/aya-expanse-8b",
    max_context_length=8192,
    chat_template=AYA_23_8B.chat_template,
    lora_target_modules=AYA_23_8B.lora_target_modules,
    supported_tasks=["chat", "instruction", "translation", "qa", "reasoning"],
    eos_token="<|END_OF_TURN_TOKEN|>",
    bos_token="<|START_OF_TURN_TOKEN|>",
    pad_token="<|END_OF_TURN_TOKEN|>",
    memory_estimates={"4bit_vram_gb": 8.0, "8bit_vram_gb": 14.5, "16bit_vram_gb": 26.0},
    rope_scaling={"type": "default", "factor": 1.0}
)

model_registry.register(AYA_23_8B)
model_registry.register(AYA_EXPANSE_8B)
