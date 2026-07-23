from typing import Dict, Any, Union, Optional
from datasets import Dataset, DatasetDict
from app.tokenizer.pipeline import TokenizationPipeline
from app.tokenizer.tokenizer_manager import TokenizerManager
from app.tokenizer.validators import TokenizationValidator
from app.tokenizer.packing import SequencePacker
from app.tokenizer.statistics import TokenStatisticsTracker
from app.tokenizer.context import PipelineContext
from app.configs.settings import settings
from app.utils.logger import logger

class DatasetBuilder:
    """
    Core engine for Phase 4: Transforms clean datasets into model-specific, training-ready datasets.
    Orchestrates execution cleanly via TokenizationPipeline and rich PipelineContext.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tokenizer_manager: Optional[TokenizerManager] = None,
        validator: Optional[TokenizationValidator] = None,
        packer: Optional[SequencePacker] = None,
        stats_tracker: Optional[TokenStatisticsTracker] = None,
        pipeline: Optional[TokenizationPipeline] = None
    ):
        config = config or {}
        settings_config = {}
        if hasattr(settings, "tokenizer") and hasattr(settings.tokenizer, "to_dict"):
            settings_config = settings.tokenizer.to_dict()
        elif isinstance(settings, dict):
            settings_config = settings
            
        self.max_length = config.get("max_length", settings_config.get("max_length", 2048))
        self.packing_enabled = config.get("packing", settings_config.get("packing", True))
        
        self.tokenizer_manager = tokenizer_manager or TokenizerManager(config)
        self.validator = validator or TokenizationValidator(max_length=self.max_length)
        self.stats = stats_tracker or TokenStatisticsTracker()
        
        if packer is not None:
            self.packer = packer
        else:
            self.packer = SequencePacker(
                max_length=self.max_length,
                eos_token_id=self.tokenizer_manager.get_eos_token_id(),
                pad_token_id=self.tokenizer_manager.get_pad_token_id()
            )

        self.pipeline = pipeline or TokenizationPipeline(
            config=config,
            tokenizer_manager=self.tokenizer_manager,
            validator=self.validator,
            packer=self.packer,
            stats_tracker=self.stats
        )

    def build(
        self,
        dataset: Union[Dataset, DatasetDict, Any],
        ctx: Optional[PipelineContext] = None,
        task_name: Optional[str] = "translation",
        num_proc: Optional[int] = None,
        **kwargs
    ) -> Union[Dataset, DatasetDict, Any]:
        """
        Processes a dataset using TokenizationPipeline with PipelineContext.
        """
        if ctx is None:
            max_len = kwargs.pop("max_length", self.max_length)
            pack_flag = kwargs.pop("packing", self.packing_enabled)
            ctx = PipelineContext(
                dataset_name="dataset",
                task=task_name or "translation",
                num_proc=num_proc,
                max_length=max_len,
                packing=pack_flag,
                **kwargs
            )
        ctx.log("DatasetBuilder: Delegating build to TokenizationPipeline.")
        return self.pipeline.run(dataset, ctx=ctx, task_name=task_name, num_proc=num_proc, **kwargs)
