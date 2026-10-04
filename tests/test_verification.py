import json
from fractions import Fraction

import pytest

from martingale_audit.schema import Observation
from martingale_audit.verification import verify_betting_kernel, verify_trace


def obs(ts, agent="a", belief=0.9, evidence=0, msg_id=None, claim="c"):
    return Observation(claim, ts, agent, belief, evidence, msg_id or f"{claim}:{agent}:{ts}")


def findings(report, kind):
    return [f for f in report["findings"] if f["kind"] == kind]


def test_absence_witness_and_future_evidence_do_not_license_adoption():
    rows = [obs(1, belief=0.5), obs(2), obs(3, agent="b", evidence=1)]
    report = verify_trace(rows)
    candidate, = findings(report, "unsupported_stance_adoption")
    assert [r["input_index"] for r in candidate["witness"]] == [0, 1]
    assert candidate["context"]["checked_prefix_indices"] == [0]
    assert report["summary"]["unsupported_adoptions"] == 1
    assert report["summary"]["adoptions_with_evidence"] == 1
    json.dumps(report, allow_nan=False)


def test_evidence_can_be_prior_or_in_the_adopting_message():
    report = verify_trace([obs(1, evidence=1), obs(2, agent="b"), obs(3, agent="c", belief=0.1)])
    assert report["summary"]["adoptions_with_evidence"] == 3
    assert report["summary"]["unsupported_adoptions"] == 0


def test_evidence_from_another_claim_does_not_license_adoption():
    report = verify_trace([obs(1, evidence=1, claim="other"), obs(2)])
    assert report["summary"]["unsupported_adoptions"] == 1


def test_timestamp_tie_is_inconclusive_regardless_of_input_order():
    rows = [obs(1, agent="a"), obs(1, agent="b", evidence=1)]
    for ordered in (rows, list(reversed(rows))):
        report = verify_trace(ordered)
        assert report["summary"]["unsupported_adoptions"] == 0
        assert report["summary"]["ambiguous_adoptions"] == 1
        assert report["summary"]["ordering_ambiguities"] == 1
        assert report["summary"]["policy_status"] == "inconclusive"


def test_neutral_resets_stance_but_repeated_stance_is_not_new_adoption():
    report = verify_trace([obs(1), obs(2), obs(3, belief=0.5), obs(4)])
    assert report["summary"]["unsupported_adoptions"] == 2


def test_tied_previous_agent_states_do_not_force_a_false_switch():
    rows = [obs(1, belief=0.1, msg_id="x"), obs(1, belief=0.9, msg_id="y"), obs(2)]
    report = verify_trace(rows)
    assert report["summary"]["unsupported_adoptions"] == 0
    assert report["summary"]["ambiguous_adoptions"] == 3


def test_message_can_annotate_multiple_claims_but_duplicate_claim_is_reported():
    first = obs(1, msg_id="shared", evidence=1)
    report = verify_trace([first, obs(1, msg_id="shared", evidence=1, claim="other"), first])
    assert len(findings(report, "duplicate_observation")) == 1
    assert not findings(report, "conflicting_message_id")
    assert report["summary"]["adoptions_checked"] == 2


def test_conflicting_duplicate_and_nonfinite_values_are_inconclusive():
    for rows in ([obs(1, msg_id="same"), obs(1, belief=0.1, msg_id="same")],
                 [obs(float("nan"))]):
        report = verify_trace(rows)
        assert report["summary"]["uncheckable_claims"] == 1
        assert report["summary"]["unsupported_adoptions"] == 0
        assert report["summary"]["policy_status"] == "inconclusive"
        json.dumps(report, allow_nan=False)


def test_missing_source_id_keeps_input_index_for_review():
    report = verify_trace([Observation("c", 1, "a", 0.9)])
    missing, = findings(report, "missing_message_id")
    assert missing["witness"][0]["input_index"] == 0
    assert report["summary"]["integrity_issues"] == 1


def test_custom_threshold_and_input_sequence_are_respected():
    rows = [obs(2, agent="b"), obs(1, evidence=0.5)]
    snapshot = list(rows)
    assert verify_trace(rows)["summary"]["unsupported_adoptions"] == 0
    assert verify_trace(rows, evidence_threshold=0.6)["summary"]["unsupported_adoptions"] == 2
    assert rows == snapshot


def test_missing_solver_is_explicit_not_a_simulated_proof(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def without_z3(name, *args, **kwargs):
        if name == "z3":
            raise ImportError("deliberately absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_z3)
    report = verify_betting_kernel()
    assert report["status"] == "unavailable"
    assert report["obligations"] == []


def test_symbolic_obligations_and_mutants():
    pytest.importorskip("z3")
    report = verify_betting_kernel()
    assert report["status"] == "proved"
    assert len(report["obligations"]) == 3
    assert all(o["solver_result"] == "unsat" for o in report["obligations"])
    assert len(report["mutation_checks"]) == 2
    assert all(m["solver_result"] == "sat" and m["counterexample"] for m in report["mutation_checks"])
    excess, lookahead = [m["counterexample"] for m in report["mutation_checks"]]
    assert Fraction(excess["wealth"]) * (1 + Fraction(excess["stake"]) * Fraction(excess["outcome"])) < 0
    factor = ((1 + Fraction(lookahead["positive_stake"])) + (1 - Fraction(lookahead["negative_stake"]))) / 2
    assert factor == Fraction(5, 4)
