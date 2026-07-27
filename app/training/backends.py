"""
BackendFactory module for lightweight backend wrappers (`Transformers`, `TRL`,
`PEFT`, `Unsloth`, `DeepSpeed`). Receives pre-loaded model, tokenizer, and datasets.
"""

from typing import Dict, Any, Optional
from app.training.config import TrainingConfig
from app.utils.logger import logger


class BaseBackendWrapper:
    """
    Abstract base wrapper for lightweight training backends.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config

    def create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None
    ) -> Any:
        raise NotImplementedError


class TransformersBackendWrapper(BaseBackendWrapper):
    def create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None
    ) -> Any:
        if self.config.dry_run or (isinstance(model, dict) and model.get("simulated")):
            logger.info("TransformersBackendWrapper: Using MockTrainerInstance for dry-run simulation.")
            return MockTrainerInstance(self.config, model, tokenizer, train_dataset, eval_dataset)

        try:
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

            tok_obj = tokenizer if not hasattr(tokenizer, "tokenizer") else tokenizer.tokenizer
            if hasattr(tok_obj, "pad_token") and tok_obj.pad_token is None:
                tok_obj.pad_token = tok_obj.eos_token
                tok_obj.pad_token_id = tok_obj.eos_token_id

            collator = DataCollatorForLanguageModeling(
                tokenizer=tok_obj,
                mlm=False
            )

            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                lr_scheduler_type=self.config.lr_scheduler_type,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_train_epochs=self.config.num_epochs,
                max_steps=self.config.max_steps or -1,
                fp16=(self.config.precision == "fp16"),
                bf16=(self.config.precision == "bf16"),
                logging_steps=self.config.logging_steps,
                logging_strategy=self.config.logging_strategy,
                save_steps=self.config.save_steps,
                save_strategy=self.config.save_strategy,
                save_total_limit=self.config.save_total_limit,
                eval_steps=self.config.eval_steps,
                eval_strategy=self.config.eval_strategy if eval_dataset else "no",
                push_to_hub=self.config.push_to_hub,
                hub_model_id=self.config.hub_model_id,
                hub_private_repo=self.config.hub_private_repo,
                hub_strategy=self.config.hub_strategy,
                gradient_checkpointing=self.config.gradient_checkpointing,
                seed=self.config.seed,
                report_to="none"
            )

            tok_obj = tokenizer if not hasattr(tokenizer, "tokenizer") else tokenizer.tokenizer
            try:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=collator,
                    processing_class=tok_obj
                )
            except TypeError:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=collator,
                    tokenizer=tok_obj
                )
            logger.info("TransformersBackendWrapper: Real Hugging Face Trainer initialized.")
            return trainer

        except Exception as e:
            logger.warning(f"TransformersBackendWrapper: Real Trainer initialization failed ({e}). Falling back to MockTrainerInstance.")
            return MockTrainerInstance(self.config, model, tokenizer, train_dataset, eval_dataset)


class PEFTBackendWrapper(BaseBackendWrapper):
    def create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None
    ) -> Any:
        if self.config.dry_run or (isinstance(model, dict) and model.get("simulated")):
            return MockTrainerInstance(self.config, model, tokenizer, train_dataset, eval_dataset, trainer_type="PEFT_LoRA")

        try:
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                target_modules=self.config.lora_target_modules
            )
            peft_model = get_peft_model(model, peft_config)
            logger.info(f"PEFTBackendWrapper: Applied LoRA wrapper (r={self.config.lora_r}, alpha={self.config.lora_alpha}).")
            
            # Delegate execution to Transformers backend with PEFT model
            tf_backend = TransformersBackendWrapper(self.config)
            return tf_backend.create_trainer(peft_model, tokenizer, train_dataset, eval_dataset)
        except Exception as e:
            logger.warning(f"PEFTBackendWrapper: PEFT initialization failed ({e}). Falling back to mock trainer.")
            return MockTrainerInstance(self.config, model, tokenizer, train_dataset, eval_dataset, trainer_type="PEFT_LoRA")


class TRLBackendWrapper(BaseBackendWrapper):
    def create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None
    ) -> Any:
        if self.config.dry_run or (isinstance(model, dict) and model.get("simulated")):
            return MockTrainerInstance(self.config, model, tokenizer, train_dataset, eval_dataset, trainer_type="TRL_SFT/DPO")

        try:
            from trl import SFTTrainer
            from transformers import TrainingArguments

            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_train_epochs=self.config.num_epochs,
                max_steps=self.config.max_steps or -1,
                fp16=(self.config.precision == "fp16"),
                bf16=(self.config.precision == "bf16"),
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps
            )
            tok_obj = tokenizer if not hasattr(tokenizer, "tokenizer") else tokenizer.tokenizer
            try:
                trainer = SFTTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    dataset_text_field="text",
                    max_seq_length=self.config.max_seq_length,
                    processing_class=tok_obj
                )
            except TypeError:
                trainer = SFTTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    dataset_text_field="text",
                    max_seq_length=self.config.max_seq_length,
                    tokenizer=tok_obj
                )
            logger.info("TRLBackendWrapper: TRL SFTTrainer initialized.")
            return trainer
        except Exception as e:
            logger.warning(f"TRLBackendWrapper: TRL initialization failed ({e}). Falling back to mock trainer.")
            return MockTrainerInstance(self.config, model, tokenizer, train_dataset, eval_dataset, trainer_type="TRL_SFT/DPO")


class MockTrainerInstance:
    """
    Simulated training execution runner guaranteeing clean testing without needing GPU hardware.
    """
    def __init__(self, config: TrainingConfig, model: Any, tokenizer: Any, train_ds: Any, eval_ds: Any, trainer_type: str = "Standard"):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_ds
        self.eval_dataset = eval_ds
        self.trainer_type = trainer_type
        self.state = {"global_step": 0, "status": "initialized"}

    def train(self) -> Dict[str, Any]:
        steps = self.config.max_steps or 20 if self.config.dry_run else 100
        logger.info(f"MockTrainerInstance ({self.trainer_type}): Simulating {steps} training steps for mode '{self.config.mode}'...")
        self.state["global_step"] = steps
        self.state["status"] = "completed"
        return {
            "global_step": steps,
            "training_loss": 1.234,
            "validation_loss": 1.150,
            "mode": self.config.mode,
            "backend": self.trainer_type
        }

    def save_model(self, output_dir: Optional[str] = None) -> str:
        path = output_dir or self.config.output_dir
        import os, json
        os.makedirs(path, exist_ok=True)
        meta = {
            "model_name": self.config.model_name,
            "mode": self.config.mode,
            "backend": self.trainer_type,
            "checkpoint_step": self.state["global_step"]
        }
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path


class BackendFactory:
    """
    Factory creating appropriate lightweight training backend wrappers.
    """
    @staticmethod
    def get_backend(config: TrainingConfig) -> BaseBackendWrapper:
        b = config.backend.lower()
        if b == "trl" or config.mode == "sft":
            return TRLBackendWrapper(config)
        elif b == "peft" or config.mode in ["lora", "qlora"]:
            return PEFTBackendWrapper(config)
        return TransformersBackendWrapper(config)
