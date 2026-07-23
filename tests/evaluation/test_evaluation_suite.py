"""
Unit test for evaluation suite modules (`MetricRegistry`, `ManipuriEvaluator`, `HumanEvaluationPipeline`).
Verifies linguistic metrics, task scorecards, agreement tracking, and DPO dataset export.
"""

import os
import pytest
from app.evaluation.metrics import MetricRegistry
from app.evaluation.evaluator import ManipuriEvaluator
from app.evaluation.human import HumanEvaluationPipeline


def test_metric_registry_calculations():
    registry = MetricRegistry()
    preds = ["Manipuri language is official in Manipur."]
    refs = ["Manipuri language is one of the official languages of Manipur."]

    bleu = registry.calculate("bleu", preds, refs)
    chrf = registry.calculate("chrf", preds, refs)
    rouge = registry.calculate("rouge", preds, refs)

    assert bleu > 0.0
    assert chrf > 0.0
    assert rouge > 0.0


def test_manipuri_evaluator_scorecard():
    evaluator = ManipuriEvaluator()
    preds = ["ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ"]
    refs = ["ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ"]

    scorecard = evaluator.evaluate_task("translation", preds, refs)
    assert scorecard["task"] == "translation"
    assert "bleu" in scorecard
    assert "chrf" in scorecard
    assert scorecard["samples_evaluated"] == 1


def test_human_evaluation_pipeline_and_dpo_export(tmp_path):
    storage = str(tmp_path / "human_eval")
    pipeline = HumanEvaluationPipeline(storage_dir=storage)

    pipeline.add_annotation(
        prompt="Translate Hello",
        model_a_output="ꯈꯨꯔꯨꯝꯖꯔꯤ",
        model_b_output="wrong output",
        preference="model_a",
        annotator_id="ann_1"
    )
    pipeline.add_annotation(
        prompt="Translate Hello",
        model_a_output="ꯈꯨꯔꯨꯝꯖꯔꯤ",
        model_b_output="wrong output",
        preference="model_a",
        annotator_id="ann_2"
    )

    agreement = pipeline.compute_inter_annotator_agreement()
    assert agreement == 1.0

    dpo_pairs = pipeline.export_dpo_dataset()
    assert len(dpo_pairs) == 2
    assert dpo_pairs[0]["chosen"] == "ꯈꯨꯔꯨꯝꯖꯔꯤ"
    assert os.path.exists(os.path.join(storage, "dpo_preference_pairs.json"))
