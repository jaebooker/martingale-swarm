"""Narrow trace-policy checks and optional symbolic betting-kernel obligations.

Neither checker verifies natural-language annotations, the statistical null,
or the safety of a swarm. Findings are about the supplied finite trace only.
"""
from collections import Counter, defaultdict
from itertools import groupby
import math
from typing import Iterable

from .schema import Observation


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _witness(index, observation):
    """Keep source indices even when message IDs are missing or duplicated."""
    return {
        "input_index": index,
        "msg_id": observation.msg_id,
        "claim": observation.claim,
        "ts": observation.ts if _finite_number(observation.ts) else repr(observation.ts),
        "agent": observation.agent,
        "belief": observation.belief if _finite_number(observation.belief) else repr(observation.belief),
        "evidence": observation.evidence if _finite_number(observation.evidence) else repr(observation.evidence),
    }


def verify_trace(observations: Iterable[Observation], evidence_threshold: float = 0.5,
                 margin: float = 0.1) -> dict:
    """Check evidence availability before first, renewed, or opposite stances.

    A stance is belief <= 0.5-margin or >= 0.5+margin. A neutral observation
    resets the agent's stance. The policy requires any earlier annotation of
    evidence for the claim, or evidence on the adopting observation itself.
    Same-time evidence from another message cannot establish a prior event.

    An unsupported adoption is a review candidate relative to the annotations,
    not proof of missing real-world evidence or an unjustified belief. Evidence
    need not support the adopted side: this schema cannot express that relation.
    Input is not modified. The returned report is JSON-serializable.
    """
    if not _finite_number(evidence_threshold) or not 0 < evidence_threshold <= 1:
        raise ValueError("evidence_threshold must be in (0, 1]")
    if not _finite_number(margin) or not 0 < margin < 0.5:
        raise ValueError("margin must be in (0, 0.5)")

    rows = list(enumerate(observations))
    findings = []
    by_claim = defaultdict(list)
    by_message = defaultdict(list)
    by_key = defaultdict(list)
    tainted_claims = set()

    def add(kind, description, witness, **context):
        findings.append({"kind": kind, "description": description,
                         "witness": [_witness(i, o) for i, o in witness],
                         "context": context})

    for index, observation in rows:
        by_claim[observation.claim].append((index, observation))
        if not isinstance(observation.msg_id, str) or not observation.msg_id.strip():
            add("missing_message_id", "Source message ID is missing; use the input index for review.",
                [(index, observation)])
        else:
            by_message[observation.msg_id].append((index, observation))
            by_key[(observation.claim, observation.msg_id)].append((index, observation))
        numeric_ok = (_finite_number(observation.ts)
                      and _finite_number(observation.belief) and 0 <= observation.belief <= 1
                      and _finite_number(observation.evidence) and 0 <= observation.evidence <= 1)
        if not numeric_ok:
            tainted_claims.add(observation.claim)
            add("invalid_observation", "Timestamp must be finite; belief and evidence must be in [0, 1].",
                [(index, observation)])

    # One message may legitimately annotate several different claims.
    duplicate_indices = set()
    for (claim, message_id), group in by_key.items():
        if len(group) > 1:
            identical = all(o == group[0][1] for _, o in group[1:])
            add("duplicate_observation", "A claim/message pair appears more than once.",
                group, identical=identical)
            if identical:
                duplicate_indices.update(i for i, _ in group[1:])
            else:
                tainted_claims.add(claim)
    for message_id, group in by_message.items():
        if any((o.agent, o.ts) != (group[0][1].agent, group[0][1].ts) for _, o in group[1:]):
            add("conflicting_message_id", "A source message ID has inconsistent author or timestamp.", group)
            tainted_claims.update(o.claim for _, o in group)

    checked = supported = unsupported = ambiguous = 0
    for claim, claim_rows in sorted(by_claim.items()):
        if claim in tainted_claims:
            add("uncheckable_claim", "Conflicting or invalid records make this claim's policy check inconclusive.",
                claim_rows[:1], claim=claim)
            continue
        ordered = sorted((r for r in claim_rows if r[0] not in duplicate_indices), key=lambda r: r[1].ts)
        prior_sides = {}  # set-valued after unresolved same-agent timestamp ties
        previous_rows = {}
        prior_evidence = []
        prefix = []
        for timestamp, batch_iter in groupby(ordered, key=lambda row: row[1].ts):
            batch = list(batch_iter)
            same_agent = Counter(o.agent for _, o in batch)
            batch_evidence = [(i, o) for i, o in batch if o.evidence >= evidence_threshold]
            if len(batch) > 1:
                add("ambiguous_timestamp_order", "Timestamp ties do not establish message order within this claim.", batch)
            next_sides = defaultdict(set)
            for index, observation in batch:
                side = (1 if observation.belief >= 0.5 + margin else
                        -1 if observation.belief <= 0.5 - margin else 0)
                next_sides[observation.agent].add(side)
                previous = prior_sides.get(observation.agent, {0})
                if side == 0 or (previous == {side} and same_agent[observation.agent] == 1):
                    continue
                witness = previous_rows.get(observation.agent, []) + [(index, observation)]
                if same_agent[observation.agent] > 1 or side in previous:
                    ambiguous += 1
                    add("ambiguous_adoption", "Timestamp order leaves the existence of this adoption unresolved.",
                        witness, claim=claim)
                    continue
                checked += 1
                if prior_evidence or observation.evidence >= evidence_threshold:
                    supported += 1
                    continue
                if batch_evidence:
                    ambiguous += 1
                    add("ambiguous_adoption", "Evidence is annotated at the same timestamp in another message; prior availability is unresolved.",
                        witness + batch_evidence, claim=claim)
                    continue
                unsupported += 1
                add("unsupported_stance_adoption",
                    "Review candidate: this stance was adopted without prior or same-message evidence in the supplied annotations.",
                    witness, claim=claim, adopted_side=side,
                    checked_prefix_indices=[i for i, _ in prefix],
                    prior_annotated_evidence_count=0,
                    absence_scope="Only the supplied observations for this claim; omitted evidence is unknown.")
            for agent, sides in next_sides.items():
                prior_sides[agent] = sides
                previous_rows[agent] = [(i, o) for i, o in batch if o.agent == agent]
            prior_evidence.extend(batch_evidence)
            prefix.extend(batch)

    kinds = Counter(f["kind"] for f in findings)
    integrity_issues = sum(kinds[k] for k in ("missing_message_id", "invalid_observation",
                                             "duplicate_observation", "conflicting_message_id"))
    policy_status = ("review_candidates" if unsupported else
                     "inconclusive" if ambiguous or tainted_claims else "satisfied_on_supplied_trace")
    return {
        "policy": {"name": "annotated_evidence_before_stance", "version": 1,
                   "evidence_threshold": evidence_threshold, "margin": margin,
                   "neutral_resets_stance": True, "timestamp_ties_are_unordered": True},
        "summary": {"observations": len(rows), "claims": len(by_claim),
                    "adoptions_checked": checked, "adoptions_with_evidence": supported,
                    "unsupported_adoptions": unsupported, "ambiguous_adoptions": ambiguous,
                    "integrity_issues": integrity_issues,
                    "ordering_ambiguities": kinds["ambiguous_timestamp_order"],
                    "uncheckable_claims": len(tainted_claims), "policy_status": policy_status},
        "findings": findings,
        "scope_note": "Trace-policy findings depend on incomplete, fallible annotations. They do not establish truth, irrationality, herding, or swarm safety.",
    }


def verify_betting_kernel() -> dict:
    """Ask Z3 to refute exact-real one-step obligations; no import-time extra.

    This verifies mathematical formulas, not the floating-point implementation.
    The conditional-expectation reduction assumes the bet is predictable.
    """
    try:
        import z3
    except ImportError:
        return {"status": "unavailable", "reason": "Install z3-solver to run symbolic obligations.",
                "obligations": [], "mutation_checks": []}

    wealth, stake, outcome, mean = z3.Reals("wealth stake outcome mean")
    obligations = []

    def check(name, assumptions, negated_property, variables):
        solver = z3.Solver()
        solver.set(timeout=10000)
        solver.add(*assumptions, negated_property)
        result = solver.check()
        report = {"name": name, "solver_result": str(result),
                  "assumptions": [str(a) for a in assumptions],
                  "negated_property": str(negated_property)}
        if result == z3.sat:
            model = solver.model()
            report["counterexample"] = {str(v): str(model.eval(v, model_completion=True)) for v in variables}
        elif result == z3.unknown:
            report["reason"] = solver.reason_unknown()
        return report

    stake_bounds = [stake >= 0, stake <= 1]
    outcome_bounds = [outcome >= -1, outcome <= 1]
    obligations.append(check("one_step_factor_nonnegative", stake_bounds + outcome_bounds,
                             1 + stake * outcome < 0, [stake, outcome]))
    obligations.append(check("one_step_wealth_nonnegative", [wealth >= 0] + stake_bounds + outcome_bounds,
                             wealth * (1 + stake * outcome) < 0, [wealth, stake, outcome]))
    obligations.append(check("conditional_expected_wealth_nonincrease",
                             [wealth >= 0, mean >= -1, mean <= 0] + stake_bounds,
                             wealth * (1 + stake * mean) > wealth, [wealth, stake, mean]))

    mutants = [check("mutant_allows_stake_up_to_two",
                     [wealth > 0, stake >= 0, stake <= 2] + outcome_bounds,
                     wealth * (1 + stake * outcome) < 0, [wealth, stake, outcome])]
    # A fair +/-1 outcome has mean zero. Choosing the stake after observing its
    # sign breaks predictability even though each chosen stake stays in [0, 1].
    positive_stake, negative_stake = z3.Reals("positive_stake negative_stake")
    expected_wealth = wealth * ((1 + positive_stake) + (1 - negative_stake)) / 2
    mutants.append(check("mutant_bets_after_observing_outcome",
                         [wealth > 0, positive_stake == z3.RealVal("1/2"), negative_stake == 0],
                         expected_wealth > wealth, [wealth, positive_stake, negative_stake]))
    return {
        "status": "proved" if all(o["solver_result"] == "unsat" for o in obligations)
                  and all(m["solver_result"] == "sat" for m in mutants) else "not_verified",
        "method": "Z3 exact-real symbolic negation of one-step algebraic obligations",
        "solver_version": z3.get_version_string(),
        "obligations": obligations, "mutation_checks": mutants,
        "assumptions": ["wealth >= 0", "0 <= stake <= 1", "-1 <= outcome <= 1",
                        "Conditional mean of outcome given the past is <= 0.",
                        "Wealth and stake are fixed given the past; stake does not see the outcome."],
        "limitations": ["Conditional expectations are represented by a symbolic mean; measure theory is not encoded.",
                        "No proof of Ville's inequality, extraction accuracy, the null assumption, trace completeness, or swarm safety.",
                        "No proof of the Python implementation or its floating-point arithmetic."],
    }
