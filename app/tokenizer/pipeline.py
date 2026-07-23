from typing import Dict, Any, Union, Optional
from datasets import Dataset, DatasetDict
try:
    from datasets import IterableDataset
except ImportError:
    IterableDataset = tuple

from app.tasks.manager import task_manager
from app.tokenizer.tokenizer_manager import TokenizerManager
from app.tokenizer.validators import TokenizationValidator
from app.tokenizer.packing import SequencePacker
from app.tokenizer.statistics import TokenStatisticsTracker
from app.tokenizer.context import PipelineContext
from app.configs.settings import settings
from app.utils.logger import logger

class TokenizationPipeline:
    """
    Dedicated pipeline for executing tokenization workflows across memory or streamed datasets.
    Orchestrates prepare/validate/format/tokenize -> sequence packing -> validation -> stats tracking.
    Enforces PipelineContext and lifecycle hooks across all stages.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tokenizer_manager: Optional[TokenizerManager] = None,
        validator: Optional[TokenizationValidator] = None,
        packer: Optional[SequencePacker] = None,
        stats_tracker: Optional[TokenStatisticsTracker] = None
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
        
        if packer is not None:
            self.packer = packer
        else:
            self.packer = SequencePacker(
                max_length=self.max_length,
                eos_token_id=self.tokenizer_manager.get_eos_token_id(),
                pad_token_id=self.tokenizer_manager.get_pad_token_id()
            )
            
        self.stats = stats_tracker or TokenStatisticsTracker()

    def before_run(self, ctx: PipelineContext, dataset: Any) -> None:
        """Hook called before the tokenization pipeline executes."""
        ctx.log(f"Pipeline started for dataset '{ctx.dataset_name}' with task '{ctx.task_name}' (packing={ctx.packing})")

    def after_run(self, ctx: PipelineContext, result: Any) -> None:
        """Hook called after the tokenization pipeline completes execution."""
        ctx.log("Pipeline completed successfully.")

    def before_stage(self, ctx: PipelineContext, stage_name: str) -> None:
        """Hook called before entering a specific processing stage."""
        ctx.stage_history.append({"stage": stage_name, "status": "started"})
        ctx.log(f"Stage '{stage_name}' started.", level="debug")

    def after_stage(self, ctx: PipelineContext, stage_name: str, stage_data: Any) -> None:
        """Hook called after completing a processing stage."""
        ctx.stage_history.append({"stage": stage_name, "status": "completed"})
        ctx.log(f"Stage '{stage_name}' completed.", level="debug")

    def run(
        self,
        dataset: Union[Dataset, DatasetDict, Any],
        ctx: Optional[PipelineContext] = None,
        task_name: Optional[str] = None,
        num_proc: Optional[int] = None,
        **kwargs
    ) -> Union[Dataset, DatasetDict, Any]:
        """
        Runs the tokenization pipeline with rich PipelineContext and lifecycle hooks.
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

        self.before_run(ctx, dataset)
        task = task_manager.get(ctx.task_name)
        tokenizer_instance = self.tokenizer_manager.tokenizer

        if isinstance(dataset, DatasetDict):
            result = DatasetDict()
            for split, ds in dataset.items():
                ctx.log(f"Processing split '{split}'")
                result[split] = self._run_single(ds, task, tokenizer_instance, ctx, num_proc=ctx.num_proc or num_proc, **kwargs)
            self.after_run(ctx, result)
            return result
        else:
            result = self._run_single(dataset, task, tokenizer_instance, ctx, num_proc=ctx.num_proc or num_proc, **kwargs)
            self.after_run(ctx, result)
            return result

    def run_on_source(
        self,
        source: Union[str, Any],
        task_name: str = "translation",
        limit: int = 5000,
        mock_fallback: bool = False,
        **kwargs
    ) -> Any:
        """
        Runs the tokenization pipeline directly on real streamed items from a corpus source.
        """
        from app.corpus.acquisition import CorpusAcquisitionManager
        logger.info(f"TokenizationPipeline: Running directly on source '{source}' (task={task_name}, limit={limit}, mock_fallback={mock_fallback})...")
        mgr = CorpusAcquisitionManager()
        spec = mgr.get_source(source) if isinstance(source, str) else source
        if not spec:
            raise KeyError(f"Source '{source}' not found in registry.")

        stream = mgr.stream_source(spec, max_examples=limit, mock_fallback=mock_fallback)
        records = [ex if isinstance(ex, dict) else {"text": str(ex)} for ex in stream]
        from datasets import Dataset
        ds = Dataset.from_list(records)
        return self.run(ds, task_name=task_name, **kwargs)

    def _run_single(self, dataset: Any, task: Any, tokenizer: Any, ctx: PipelineContext, num_proc: Optional[int] = None, **kwargs) -> Any:
        if not hasattr(dataset, "map"):
            from datasets import Dataset
            if isinstance(dataset, (list, tuple)):
                dataset = Dataset.from_list(list(dataset))
            else:
                records = [ex if isinstance(ex, dict) else {"text": str(ex)} for ex in dataset]
                dataset = Dataset.from_list(records)

        is_iterable = isinstance(dataset, IterableDataset) or hasattr(dataset, "n_shards") or hasattr(dataset, "_ex_iterable")
        
        remove_cols = None
        if hasattr(dataset, "column_names") and dataset.column_names:
            remove_cols = dataset.column_names
        elif hasattr(dataset, "features") and dataset.features:
            remove_cols = list(dataset.features.keys())

        # Stage 1: Prepare & Tokenize
        self.before_stage(ctx, "prepare_and_tokenize")
        def _map_fn(example):
            out = task.prepare_example(example, tokenizer, ctx.max_length, ctx=ctx, **kwargs)
            if out.get("input_ids"):
                self.stats.record_sample(len(out["input_ids"]), ctx.max_length)
            return out

        map_kwargs = {"batched": False}
        if remove_cols:
            map_kwargs["remove_columns"] = remove_cols
        if not is_iterable and num_proc is not None:
            map_kwargs["num_proc"] = num_proc

        tokenized_ds = dataset.map(_map_fn, **map_kwargs)
        self.after_stage(ctx, "prepare_and_tokenize", tokenized_ds)

        # Stage 2: Validate & Filter
        self.before_stage(ctx, "validate")
        def _filter_fn(batch):
            return self.validator.validate_batch(batch)

        validated_ds = tokenized_ds.map(
            _filter_fn,
            batched=True,
            batch_size=1000
        )
        self.after_stage(ctx, "validate", validated_ds)

        # Stage 3: Optional Sequence Packing
        if ctx.packing:
            self.before_stage(ctx, "packing")
            packed_ds = validated_ds.map(
                self.packer.pack,
                batched=True,
                batch_size=1000
            )
            if not is_iterable and hasattr(packed_ds, "__len__"):
                try:
                    self.stats.record_packed_blocks(len(packed_ds))
                except TypeError:
                    pass
            self.after_stage(ctx, "packing", packed_ds)
            return packed_ds

        return validated_ds
