from typing import Any, Dict, Optional, Union
from app.utils.logger import logger

try:
    from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
except ImportError:
    DataCollatorForLanguageModeling = None
    DataCollatorForSeq2Seq = None

class DataCollatorManager:
    """
    Resolves and configures the appropriate data collator based on task objective or explicit selection.
    Supports causal language modeling (mlm=False) and seq2seq collation.
    """
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def get_collator(self, collator_type: str = "causal_lm", **kwargs) -> Any:
        """
        Returns the requested collator.
        collator_type: 'causal_lm', 'seq2seq', or 'mlm'
        """
        if DataCollatorForLanguageModeling is None:
            raise RuntimeError("transformers package is not installed. Cannot create DataCollators.")

        key = collator_type.lower()
        if key in ["causal_lm", "clm"]:
            logger.debug("DataCollatorManager: Returning DataCollatorForLanguageModeling (mlm=False)")
            return DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=kwargs.get("pad_to_multiple_of", 8)
            )
        elif key in ["seq2seq", "s2s"]:
            if DataCollatorForSeq2Seq is None:
                raise RuntimeError("DataCollatorForSeq2Seq not found in transformers.")
            logger.debug("DataCollatorManager: Returning DataCollatorForSeq2Seq")
            return DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=kwargs.get("pad_to_multiple_of", 8)
            )
        elif key == "mlm":
            logger.debug("DataCollatorManager: Returning DataCollatorForLanguageModeling (mlm=True)")
            return DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=kwargs.get("mlm_probability", 0.15),
                pad_to_multiple_of=kwargs.get("pad_to_multiple_of", 8)
            )
        else:
            logger.warning(f"DataCollatorManager: Unknown collator type '{collator_type}', defaulting to causal_lm.")
            return DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=kwargs.get("pad_to_multiple_of", 8)
            )
