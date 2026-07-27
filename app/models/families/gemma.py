from app.models.registry import ModelSpecification, model_registry

gemma_3_4b = ModelSpecification(
    short_name="gemma_3_4b",
    aliases=["gemma3", "gemma3-4b"],
    name="google/gemma-3-4b-it",
    max_context_length=8192,
    chat_template="gemma",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["translation", "instruction", "chat", "pretraining"],
    eos_token="<eos>",
    pad_token="<pad>",
    bos_token="<bos>"
)

gemma_3_12b = ModelSpecification(
    short_name="gemma_3_12b",
    aliases=["gemma3-12b"],
    name="google/gemma-3-12b-it",
    max_context_length=8192,
    chat_template="gemma",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["translation", "instruction", "chat", "pretraining"],
    eos_token="<eos>",
    pad_token="<pad>",
    bos_token="<bos>"
)

model_registry.register(gemma_3_4b)
model_registry.register(gemma_3_12b)
