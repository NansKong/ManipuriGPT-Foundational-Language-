from app.models.registry import ModelSpecification, model_registry

qwen_2_5_3b = ModelSpecification(
    short_name="qwen_2_5_3b",
    aliases=["qwen2.5", "qwen2.5-3b"],
    name="Qwen/Qwen2.5-3B-Instruct",
    max_context_length=32768,
    chat_template="qwen",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["translation", "instruction", "chat", "pretraining"],
    eos_token="<|im_end|>",
    pad_token="<|im_end|>",
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True
    }
)

qwen_2_5_7b = ModelSpecification(
    short_name="qwen_2_5_7b",
    aliases=["qwen2.5-7b"],
    name="Qwen/Qwen2.5-7B-Instruct",
    max_context_length=32768,
    chat_template="qwen",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["translation", "instruction", "chat", "pretraining"],
    eos_token="<|im_end|>",
    pad_token="<|im_end|>"
)

model_registry.register(qwen_2_5_3b)
model_registry.register(qwen_2_5_7b)
