from app.models.registry import ModelSpecification, model_registry

mistral_7b_v03 = ModelSpecification(
    short_name="mistral_7b",
    aliases=["mistral"],
    name="mistralai/Mistral-7B-Instruct-v0.3",
    max_context_length=32768,
    chat_template="mistral",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    supported_tasks=["translation", "instruction", "chat", "pretraining"],
    eos_token="</s>",
    pad_token="</s>",
    bos_token="<s>"
)

model_registry.register(mistral_7b_v03)
