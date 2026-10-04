"""Turn belief observations into a sequence of bets.

For agent i's k-th statement on a claim we form

    delta     = p_k - p_{k-1}                      how far the agent moved
    direction = sign(m - p_ref)                    where the peers were
    x         = direction * delta                  movement toward the peers, in [-1, 1]

and decide whether the step counts (`counted`). The e-test in `etest.py` bets on
x > 0. Three knobs fix what the null hypothesis is, so they are worth reading:

peer_time
    "current"   m is the mean of peers' latest beliefs just before the agent's
                k-th statement. Catches immediate conformity.
    "previous"  m is the peer mean just before the agent's (k-1)-th statement, so
                the direction was already fixed when the agent last spoke. With
                gate=False, magnitude scores can test a conditional-mean null
                if observed beliefs are martingales in the bettor's filtration.
                Asynchronous statements and extraction noise do not guarantee this.

gate
    If True, a step counts only when nobody (the agent included) introduced new
    evidence on the claim between the agent's two statements. The null becomes:
    "in evidence-free intervals the agent does not, in expectation, move toward
    its peers."

ref_lag
    Which of the agent's own earlier beliefs sets the direction. 0 uses p_{k-1},
    1 uses p_{k-2}. Beliefs are extracted with error, and p_{k-1} appears in
    delta with a minus sign, so with ref_lag=0 extraction noise alone makes x
    positive on average (regression to the mean looks like herding). ref_lag=1
    separates the readings used for direction and movement. This can reduce
    same-reading coupling, but overlapping increments still have conditional
    dependence even with independent measurement noise. It is a sensitivity
    choice, not a proof of noise robustness or an anytime error guarantee.

statistic
    "sign"       x = direction * sign(delta), ignoring moves smaller than
                 `min_move`. Null: a move is no more likely to be toward the
                 peers than away from them. Uses only the direction of each
                 move. This may be useful for ordinal labels, but remains
                 dependent on extraction and an unverified conditional null.
    "magnitude"  x = direction * delta. Null: the expected move toward the peers
                 is not positive. A latent-belief martingale does not imply
                 this property for noisy extracted labels. A martingale can
                 also make many small moves one way and rare large moves back.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .schema import Observation


@dataclass(frozen=True)
class StepConfig:
    peer_time: str = "current"       # "current" | "previous"
    gate: bool = True
    ref_lag: int = 1                 # 0 | 1
    evidence_threshold: float = 0.5
    min_gap: float = 0.02            # ignore steps where peers and agent already agree
    statistic: str = "sign"          # "sign" | "magnitude"
    min_move: float = 0.02           # sign statistic: smaller moves count as no move

    def __post_init__(self):
        if self.peer_time not in ("current", "previous"):
            raise ValueError("peer_time must be 'current' or 'previous'")
        if self.ref_lag not in (0, 1):
            raise ValueError("ref_lag must be 0 or 1")
        if self.statistic not in ("sign", "magnitude"):
            raise ValueError("statistic must be 'sign' or 'magnitude'")


@dataclass(frozen=True)
class Step:
    claim: str
    ts: float
    agent: str
    delta: float
    gap: float                # m - p_ref; its sign is the bet direction
    evidence_free: bool
    counted: bool
    x: float                  # the bet outcome, in [-1, 1]; positive = toward peers
    msg_id: Optional[str] = None

    @property
    def direction(self) -> int:
        return (self.gap > 0) - (self.gap < 0)


@dataclass
class _Past:
    idx: int
    belief: float
    peer_mean: Optional[float]   # peer mean just before this statement


def _peer_mean(latest: Dict[str, float], agent: str) -> Optional[float]:
    vals = [b for a, b in latest.items() if a != agent]
    return sum(vals) / len(vals) if vals else None


def build_steps(observations: Iterable[Observation],
                config: StepConfig = StepConfig()) -> List[Step]:
    by_claim: Dict[str, List[Observation]] = defaultdict(list)
    for o in observations:
        if not 0.0 <= o.belief <= 1.0:
            raise ValueError(f"belief out of range: {o}")
        by_claim[o.claim].append(o)

    steps: List[Step] = []
    for claim, obs in by_claim.items():
        obs.sort(key=lambda o: o.ts)          # stable: ties keep input order
        latest: Dict[str, float] = {}
        history: Dict[str, List[_Past]] = defaultdict(list)
        last_evidence_idx = -1
        for idx, o in enumerate(obs):
            m_now = _peer_mean(latest, o.agent)
            if o.evidence >= config.evidence_threshold:
                last_evidence_idx = idx
            hist = history[o.agent]
            if len(hist) >= 1 + config.ref_lag:
                prev = hist[-1]
                ref = hist[-1 - config.ref_lag]
                m = m_now if config.peer_time == "current" else prev.peer_mean
                if m is not None:
                    gap = m - ref.belief
                    if abs(gap) < config.min_gap:
                        gap = 0.0
                    evidence_free = last_evidence_idx <= prev.idx
                    delta = o.belief - prev.belief
                    d = (gap > 0) - (gap < 0)
                    if config.statistic == "sign":
                        move = 0 if abs(delta) < config.min_move else (1 if delta > 0 else -1)
                        x = float(d * move)
                    else:
                        x = d * delta
                    counted = d != 0 and (evidence_free or not config.gate)
                    steps.append(Step(claim, o.ts, o.agent, delta, gap, evidence_free,
                                      counted, x, o.msg_id))
            hist.append(_Past(idx, o.belief, m_now))
            latest[o.agent] = o.belief
    steps.sort(key=lambda s: s.ts)
    return steps
