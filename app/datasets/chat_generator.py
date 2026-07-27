"""
ChatDatasetGenerator module for normalizing conversational datasets across diverse schemas
into a unified target schema: `{"messages": [{"role": "system|user|assistant", "content": "..."}], "metadata": {}}`.
"""

from typing import Dict, Any, List, Optional, Union
from app.utils.logger import logger


class ChatDatasetGenerator:
    """
    Normalizes heterogeneous conversation schemas into standard `{"messages": [...], "metadata": {}}` records.
    Supported source formats: `ShareGPT`, `UltraChat`, `OpenAssistant`, `LMSYS`, `Dolly`,
    `Self-Instruct`/`Alpaca`, `Evol-Instruct`, and `Custom Manipuri conversations`.
    """
    def __init__(self, default_system_prompt: str = "You are ManipuriGPT, an AI coding and linguistic assistant for the Manipuri language."):
        self.default_system_prompt = default_system_prompt

    def normalize_record(self, raw_record: Dict[str, Any], source_format: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Normalizes a single dictionary record to standard schema.
        Automatically detects format if `source_format` is not explicitly provided.
        """
        if not raw_record or not isinstance(raw_record, dict):
            return None

        # Detect source format if not specified
        fmt = (source_format or self._detect_format(raw_record)).lower()
        messages: List[Dict[str, str]] = []
        metadata = raw_record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}
        metadata["source_format"] = fmt

        # 1. ShareGPT & LMSYS format: `conversations` with `from` and `value`
        if fmt in ["sharegpt", "lmsys"] or "conversations" in raw_record:
            convs = raw_record.get("conversations", [])
            for turn in convs:
                if not isinstance(turn, dict):
                    continue
                role_raw = turn.get("from", turn.get("role", "")).lower()
                content = turn.get("value", turn.get("text", turn.get("content", "")))
                role = self._map_role(role_raw)
                if content and role:
                    messages.append({"role": role, "content": str(content).strip()})

        # 2. Alpaca / Dolly / Self-Instruct / Evol-Instruct: instruction, input/context, output/response
        elif fmt in ["alpaca", "dolly", "self-instruct", "evol-instruct"] or ("instruction" in raw_record and ("output" in raw_record or "response" in raw_record)):
            instruction = raw_record.get("instruction", "")
            context = raw_record.get("input", raw_record.get("context", ""))
            output = raw_record.get("output", raw_record.get("response", ""))

            user_content = str(instruction).strip()
            if context and str(context).strip():
                user_content = f"{user_content}\n\nContext:\n{str(context).strip()}"

            if user_content:
                messages.append({"role": "user", "content": user_content})
            if output and str(output).strip():
                messages.append({"role": "assistant", "content": str(output).strip()})

        # 3. UltraChat format: `data` or `messages` strings or dict turns
        elif fmt == "ultrachat" or "data" in raw_record:
            data_items = raw_record.get("data", raw_record.get("messages", []))
            for i, turn in enumerate(data_items):
                role = "user" if i % 2 == 0 else "assistant"
                if isinstance(turn, dict):
                    content = turn.get("content", turn.get("text", ""))
                    role = self._map_role(turn.get("role", role))
                else:
                    content = str(turn)
                if content:
                    messages.append({"role": role, "content": str(content).strip()})

        # 4. OpenAssistant format: `text` and `role`
        elif fmt == "openassistant" or ("role" in raw_record and "text" in raw_record):
            role = self._map_role(raw_record.get("role", "user"))
            content = raw_record.get("text", "")
            if content and role:
                messages.append({"role": role, "content": str(content).strip()})

        # 5. Standard / Custom Manipuri messages schema
        elif "messages" in raw_record:
            for turn in raw_record["messages"]:
                if isinstance(turn, dict) and "role" in turn and "content" in turn:
                    role = self._map_role(turn["role"])
                    if role and turn["content"]:
                        messages.append({"role": role, "content": str(turn["content"]).strip()})

        if not messages or not any(m["role"] == "user" for m in messages):
            return None

        # Ensure system prompt exists as first message if not present
        if messages[0]["role"] != "system" and self.default_system_prompt:
            messages.insert(0, {"role": "system", "content": self.default_system_prompt})

        return {
            "messages": messages,
            "metadata": metadata
        }

    def _detect_format(self, record: Dict[str, Any]) -> str:
        if "conversations" in record:
            return "sharegpt"
        if "instruction" in record and ("output" in record or "response" in record):
            return "alpaca"
        if "data" in record:
            return "ultrachat"
        if "role" in record and "text" in record:
            return "openassistant"
        if "messages" in record:
            return "standard"
        return "unknown"

    def _map_role(self, raw_role: str) -> str:
        role_clean = str(raw_role).lower().strip()
        if role_clean in ["human", "user", "prompter", "qu"]:
            return "user"
        elif role_clean in ["gpt", "assistant", "bot", "ai", "model"]:
            return "assistant"
        elif role_clean in ["system", "sys"]:
            return "system"
        return ""
