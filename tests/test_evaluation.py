import pytest
from martingale_audit.evaluation import compare_labels, check_provenance
from martingale_audit.extract import LLMExtractor, _parse_json_list
from martingale_audit.io import parse_ts
from martingale_audit.schema import Message, Observation


def test_incomplete_reference_is_not_negative_ground_truth():
    ref = [Observation("c", 0, "a", .7, 1, "1"), Observation("c", 1, "a", .3, 0, "2")]
    pred = [Observation("c", 0, "a", .9, 0, "1"), Observation("c", 2, "a", .5, 0, "3")]
    r = compare_labels(ref, pred)
    assert r["coverage"] == .5
    assert r["belief_mae_on_matches"] == pytest.approx(.2)
    assert r["evidence_agreement_on_matches"] == 0
    assert r["unlabelled_prediction_pairs"] == 1
    assert "precision" not in r


def test_source_identity_and_millisecond_rounding():
    ref = [Observation("c", 10.123, "a", .7, 1, "1")]
    assert check_provenance(ref, [Message("1", 10.1234, "a", "text")])["passed"]
    assert not check_provenance(ref, [Message("1", 10.1234, "wrong", "text")])["passed"]


def test_claim_outcomes_never_enter_annotation_prompt():
    class Judge:
        name = "test"
        def complete(self, prompt, max_tokens=2000):
            assert "secret outcome note" not in prompt
            assert "'truth':" not in prompt
            assert "Claim text" in prompt
            return '[]'
    LLMExtractor(Judge()).extract([Message("1", 0, "a", "hello")],
        {"c": {"text": "Claim text", "truth": True, "note": "secret outcome note"}})


def test_invalid_model_json_is_not_silently_no_stance():
    with pytest.raises(ValueError):
        _parse_json_list("I cannot format this")
    assert _parse_json_list("[]") == []


def test_visibility_channels_cannot_be_silently_pooled():
    with pytest.raises(ValueError, match="visibility"):
        LLMExtractor(None).extract([Message("1", 0, "a", "x", "private"),
                                   Message("2", 1, "b", "y", "public")], {"c": "claim"})


def test_naive_dates_are_explicitly_utc():
    assert parse_ts("2026-10-04", 0) == parse_ts("2026-10-04T00:00:00Z", 0)
