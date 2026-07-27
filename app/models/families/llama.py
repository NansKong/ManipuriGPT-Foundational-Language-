from app.models.registry import ModelSpecification, model_registry

llama_3_2_3b = ModelSpecification(
    short_name="llama_3_2_3b",
    aliases=["llama3.2", "llama3.2-3b"],
    name="meta-llama/Llama-3.2-3B-Instruct",
    max_context_length=131072,
    chat_template="llama",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["translation", "instruction", "chat", "pretraining"],
    eos_token="<|eot_id|>",
    pad_token="<|eot_id|>",
    bos_token="<|begin_of_text|>"
)

llama_3_2_1b = ModelSpecification(
    short_name="llama_3_2_1b",
    aliases=["llama3.2-1b"],
    name="meta-llama/Llama-3.2-1B-Instruct",
    max_context_length=131072,
    chat_template="llama",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["translation", "instruction", "chat", "pretraining"],
    eos_token="<|eot_id|>",
    pad_token="<|eot_id|>",
    bos_token="<|begin_of_text|>"
)

model_registry.register(llama_3_2_3b)
model_registry.register(llama_3_2_1b)
