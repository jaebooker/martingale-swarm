"""Trace how a claim spreads: who took which side, when, and on what basis."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .schema import Observation


@dataclass(frozen=True)
class Adoption:
    claim: str
    ts: float
    agent: str
    side: int                  # +1 believes the claim, -1 disbelieves
    switched: bool             # False = first stance, True = changed sides
    evidence_free: bool        # no new evidence since the agent last spoke
    unsupported: bool          # nobody had offered any evidence on the claim yet
    peers_on_side: float       # share of peers already on that side
    own_evidence: bool = False # this very message introduced evidence
    msg_id: str | None = None


def trace(observations: Iterable[Observation], evidence_threshold: float = 0.5,
          margin: float = 0.1) -> List[Adoption]:
    """An agent holds a side once its belief is beyond 0.5 +/- margin."""
    by_claim: Dict[str, List[Observation]] = defaultdict(list)
    for o in observations:
        by_claim[o.claim].append(o)
    out: List[Adoption] = []
    for claim, obs in by_claim.items():
        obs.sort(key=lambda o: o.ts)
        side: Dict[str, int] = {}
        last_idx: Dict[str, int] = {}
        last_evidence = -1
        for idx, o in enumerate(obs):
            if o.evidence >= evidence_threshold:
                last_evidence = idx
            s = 1 if o.belief >= 0.5 + margin else -1 if o.belief <= 0.5 - margin else 0
            if s != 0 and side.get(o.agent, 0) != s:
                peers = [v for a, v in side.items() if a != o.agent and v != 0]
                out.append(Adoption(
                    claim=claim, ts=o.ts, agent=o.agent, side=s,
                    switched=o.agent in side and side[o.agent] != 0,
                    evidence_free=last_evidence <= last_idx.get(o.agent, -1),
                    unsupported=last_evidence < 0,
                    peers_on_side=(sum(v == s for v in peers) / len(peers)) if peers else 0.0,
                    own_evidence=o.evidence >= evidence_threshold,
                    msg_id=o.msg_id))
            if s != 0:
                side[o.agent] = s
            last_idx[o.agent] = idx
    out.sort(key=lambda a: a.ts)
    return out


def summary(adoptions: List[Adoption]) -> Dict[str, Dict[str, float]]:
    """Per agent: side switches, how many were evidence-free moves onto the side
    most peers already held, and how its first stances were formed.

    `first_echo` counts first stances that joined a side a peer already held
    while the message itself offered no evidence. `first_verified` counts first
    stances that came with the agent's own evidence."""
    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "switches": 0, "evidence_free_switches": 0, "joined_majority_without_evidence": 0,
        "originated": 0, "first_stances": 0, "first_echo": 0, "first_verified": 0})
    for a in adoptions:
        d = agg[a.agent]
        if not a.switched:
            d["first_stances"] += 1
            if a.own_evidence:
                d["first_verified"] += 1
            elif a.peers_on_side > 0.0:
                d["first_echo"] += 1
        if a.switched:
            d["switches"] += 1
            if a.evidence_free:
                d["evidence_free_switches"] += 1
                if a.peers_on_side > 0.5:
                    d["joined_majority_without_evidence"] += 1
        elif a.peers_on_side == 0.0:
            d["originated"] += 1
    return dict(agg)
