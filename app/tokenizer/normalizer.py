from typing import Dict, Any, List

class ConversationNormalizer:
    """
    Standardizes heterogeneous chat/conversation data formats (ShareGPT from/value,
    Alpaca instruction/output, OpenAssistant, or raw user/assistant keys)
    into the mandatory Normalized Conversation Schema:
    [{'role': 'system|user|assistant', 'content': '...'}]
    Ensures PromptFormatter never receives raw source formats.
    """
    @staticmethod
    def normalize(example: Dict[str, Any]) -> List[Dict[str, str]]:
        normalized_messages = []
        if "messages" in example and isinstance(example["messages"], list):
            for msg in example["messages"]:
                role = str(msg.get("role", "user")).lower()
                content = str(msg.get("content", "")).strip()
                normalized_messages.append({"role": role, "content": content})
        elif "conversations" in example and isinstance(example["conversations"], list):
            # ShareGPT style format
            for msg in example["conversations"]:
                raw_from = str(msg.get("from", msg.get("role", "user"))).lower()
                role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
                role = role_map.get(raw_from, "user")
                content = str(msg.get("value", msg.get("content", ""))).strip()
                normalized_messages.append({"role": role, "content": content})
        else:
            # Fallback check system/user/assistant keys
            system_msg = example.get("system")
            if system_msg:
                normalized_messages.append({"role": "system", "content": str(system_msg).strip()})
            user_msg = example.get("user") or example.get("human") or ""
            assistant_msg = example.get("assistant") or example.get("gpt") or ""
            if user_msg:
                normalized_messages.append({"role": "user", "content": str(user_msg).strip()})
            if assistant_msg:
                normalized_messages.append({"role": "assistant", "content": str(assistant_msg).strip()})
        return normalized_messages
