"""
Unit test for QualityScorer and ToxicityFilter modules (Phase 5).
Verifies fluency heuristic scoring, character length/symbol checks, and toxic keyword filtering.
"""

import pytest
from app.preprocessing.quality_scorer import QualityScorer, ToxicityFilter


def test_quality_scorer_good_and_bad_text():
    scorer = QualityScorer(min_score=0.45)
    
    good_text = "Manipuri language and script alignment pipeline has been refined and verified across multiple test suites."
    score, details = scorer.compute_score(good_text)
    assert score >= 0.45
    assert scorer.is_acceptable(good_text)

    # High symbol density or gibberish
    bad_text = "@@@@ #### $$$$ %%%%% &&&&& ******* ^^^^^^^"
    bad_score, details = scorer.compute_score(bad_text)
    assert not scorer.is_acceptable(bad_text)


def test_toxicity_filter():
    tox_filter = ToxicityFilter()
    clean_ex = {"text": "Manipuri is spoken primarily in northeast India."}
    toxic_ex = {"text": "This sample contains hate_speech_sample_kw inside."}

    res_clean = tox_filter.filter_example(clean_ex)
    assert res_clean is not None

    res_toxic = tox_filter.filter_example(toxic_ex)
    assert res_toxic is None
