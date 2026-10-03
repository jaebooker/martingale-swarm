import json
import random

import pytest

from martingale_audit import EProcess, Observation, PRESETS, StepConfig, audit, build_steps, e_bh
from martingale_audit import cascade
from martingale_audit.extract import LLMExtractor, RegexExtractor
from martingale_audit.schema import Message
from martingale_audit.synth import CONSENSUS_SWARM_ROSTER, SynthConfig, simulate


def test_wealth_never_grows_without_a_bet():
    ep = EProcess()
    assert ep.next_bet() == 0.0
    ep.update(1.0)
    assert ep.wealth == 1.0          # first stake is always zero


def test_bet_uses_only_the_past():
    ep = EProcess()
    ep.update_many([0.2, 0.3])
    lam = ep.next_bet()
    before = ep.log_wealth
    ep.update(-1.0)
    assert ep.log_wealth == pytest.approx(before + __import__("math").log1p(-lam))


def test_ville_bound_holds_under_the_null():
    """Fair +/-1 steps scaled into [-1, 1]. Share of runs ever reaching 1/alpha
    must stay under alpha (with slack for Monte Carlo error)."""
    rng = random.Random(1)
    alpha, runs, hits = 0.05, 2000, 0
    for _ in range(runs):
        ep = EProcess()
        for _ in range(300):
            ep.update(rng.choice([-0.3, 0.3]))
        hits += ep.rejected(alpha)
    assert hits / runs <= alpha + 0.015


def test_detects_positive_drift():
    rng = random.Random(2)
    ep = EProcess()
    for _ in range(300):
        ep.update(0.05 + rng.uniform(-0.1, 0.1))
    assert ep.rejected(0.05)


def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        EProcess().update(1.5)


def test_e_bh():
    assert e_bh({"a": 1000, "b": 50, "c": 1, "d": 0.5}, 0.05) == ["a", "b"]
    assert e_bh({"a": 10, "b": 5}, 0.05) == []


def _obs(rows):
    return [Observation("c", float(t), a, b, e) for t, (a, b, e) in enumerate(rows)]


def test_step_counts_only_evidence_free_intervals():
    rows = [("peer", 0.9, 0), ("me", 0.2, 0), ("me", 0.4, 0),      # evidence-free move toward peer
            ("peer", 0.9, 1), ("me", 0.8, 0)]                      # move after peer posts evidence
    steps = build_steps(_obs(rows), StepConfig(peer_time="current", gate=True, ref_lag=0))
    mine = [s for s in steps if s.agent == "me"]
    assert [s.counted for s in mine] == [True, False]
    assert mine[0].x == 1.0 and mine[0].delta == pytest.approx(0.2)


def test_own_evidence_closes_the_gate():
    rows = [("peer", 0.9, 0), ("me", 0.2, 0), ("me", 0.7, 1)]
    steps = build_steps(_obs(rows), StepConfig(gate=True, ref_lag=0))
    assert not [s for s in steps if s.agent == "me"][0].counted


def test_previous_peer_time_ignores_later_peer_posts():
    rows = [("peer", 0.1, 0), ("me", 0.5, 0), ("peer", 0.9, 0), ("me", 0.6, 0)]
    cfg = StepConfig(peer_time="previous", gate=False, ref_lag=0, statistic="magnitude")
    s = [s for s in build_steps(_obs(rows), cfg) if s.agent == "me"][0]
    assert s.gap < 0 and s.x == pytest.approx(-0.1)     # peer stood at 0.1 when I last spoke


def test_ref_lag_needs_three_statements():
    rows = [("peer", 0.9, 0), ("me", 0.2, 0), ("me", 0.4, 0)]
    assert not [s for s in build_steps(_obs(rows), StepConfig(ref_lag=1)) if s.agent == "me"]


def test_regex_extractor_round_trip():
    messages, truth = simulate(config=SynthConfig(n_claims=3, rounds=3))
    got = RegexExtractor().extract(messages)
    assert len(got) == len(truth)
    for g, t in zip(got, truth):
        assert g.agent == t.agent and g.evidence == t.evidence
        assert abs(g.belief - t.belief) <= 0.005 + 1e-9


def test_end_to_end_flags_herders_only():
    messages, _ = simulate(config=SynthConfig(seed=0))
    res = audit(RegexExtractor().extract(messages), PRESETS["conformity"],
                groups=CONSENSUS_SWARM_ROSTER)
    flagged = {a for a, r in res.agents.items() if r.flagged}
    assert flagged == {"Answer_Agent_4", "Answer_Agent_5"}
    assert res.groups["herder"]["e_value"] >= 20


class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, prompt, max_tokens=2000):
        self.prompts.append(prompt)
        return self.reply


def test_llm_extractor_parses_and_filters():
    msgs = [Message("1", 1, "a", "the form went through"),
            Message("2", 2, "b", "agreed")]
    reply = "Here you go:\n" + json.dumps([
        {"id": "1", "claim": "c01", "belief": 0.9, "evidence": 1},
        {"id": "2", "claim": "c01", "belief": 0.8, "evidence": 0},
        {"id": "2", "claim": "nope", "belief": 0.8},          # unknown claim
        {"id": "9", "claim": "c01", "belief": 0.8},           # unknown message
        {"id": "1", "claim": "c01", "belief": 7},             # out of range
    ])
    obs = LLMExtractor(FakeLLM(reply)).extract(msgs, {"c01": "The form was submitted."})
    assert [(o.agent, o.belief, o.evidence) for o in obs] == [("a", 0.9, 1.0), ("b", 0.8, 0.0)]


def test_llm_extractor_survives_garbage():
    msgs = [Message("1", 1, "a", "x")]
    assert LLMExtractor(FakeLLM("sorry, no")).extract(msgs, {"c01": "y"}) == []


def test_cascade_marks_evidence_free_switch():
    rows = [("a", 0.9, 1), ("b", 0.9, 0), ("me", 0.2, 0), ("me", 0.8, 0)]
    ad = cascade.trace(_obs(rows))
    mine = [x for x in ad if x.agent == "me"]
    assert mine[-1].switched and mine[-1].evidence_free and mine[-1].peers_on_side == 1.0
    assert cascade.summary(ad)["me"]["joined_majority_without_evidence"] == 1
    assert cascade.summary(ad)["a"]["originated"] == 1
