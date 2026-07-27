"""
DeepSeek model family specifications (Phase 5).
Includes R1 reasoning distillation variants optimized for deep analysis and math/code logic.
"""

from app.models.registry import ModelSpecification, model_registry

DEEPSEEK_R1_DISTILL_QWEN_1_5B = ModelSpecification(
    short_name="deepseek_r1_distill_qwen_1_5b",
    name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    max_context_length=32768,
    chat_template="{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["reasoning", "chat", "code", "instruction"],
    eos_token="<|im_end|>",
    bos_token="<|im_start|>",
    pad_token="<|im_end|>",
    memory_estimates={"4bit_vram_gb": 3.0, "8bit_vram_gb": 4.8, "16bit_vram_gb": 8.5},
    rope_scaling={"type": "yarn", "factor": 4.0}
)

DEEPSEEK_R1_DISTILL_LLAMA_8B = ModelSpecification(
    short_name="deepseek_r1_distill_llama_8b",
    name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_context_length=32768,
    chat_template="{% for message in messages %}{% if message['role'] == 'system' %}<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["reasoning", "chat", "code", "instruction", "qa"],
    eos_token="<|eot_id|>",
    bos_token="<|begin_of_text|>",
    pad_token="<|eot_id|>",
    memory_estimates={"4bit_vram_gb": 8.0, "8bit_vram_gb": 14.5, "16bit_vram_gb": 26.0},
    rope_scaling={"type": "yarn", "factor": 4.0}
)

model_registry.register(DEEPSEEK_R1_DISTILL_QWEN_1_5B)
model_registry.register(DEEPSEEK_R1_DISTILL_LLAMA_8B)
