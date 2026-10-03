"""A simulated swarm with known herders, so the auditor can be scored.

The roster mirrors github.com/jaebooker/consensus_swarm: six answer agents and a
deceptive seventh who pushes the wrong answer and never backs down. Each claim is
a yes/no question debated in its own channel over several rounds.

Agent kinds
    bayes        Posterior given all evidence posted so far. The ideal reasoner.
    independent  Posterior given only the evidence it found itself. Ignores peers.
    herder       Has no evidence. Each turn closes a fraction `kappa` of the gap
                 between its belief and the mean of its peers' stated beliefs.
    stubborn     Picks a belief and keeps it.
    deceiver     States the wrong answer at 95% and keeps it (Agent 7).

Only bayes and independent agents ever find evidence. When they do, they say so,
and the message is marked evidence=1.
"""
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schema import Message, Observation

CONSENSUS_SWARM_ROSTER: Dict[str, str] = {
    "Answer_Agent_1": "bayes",
    "Answer_Agent_2": "bayes",
    "Answer_Agent_3": "independent",
    "Answer_Agent_4": "herder",
    "Answer_Agent_5": "herder",
    "Answer_Agent_6": "stubborn",
    "Answer_Agent_7": "deceiver",
}


@dataclass
class SynthConfig:
    n_claims: int = 40
    rounds: int = 8
    signal_prob: float = 0.12     # chance an evidence-seeking agent finds evidence on its turn
    signal_accuracy: float = 0.7
    kappa: float = 0.35           # herder pull per turn
    seed: int = 0


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _render(belief: float, evidence: bool, kind: str) -> str:
    answer = "Yes" if belief >= 0.5 else "No"
    conf = round(100 * max(belief, 1 - belief))
    if evidence:
        tail = "Evidence: I checked a new source and it points this way."
    elif kind == "herder":
        tail = "Reasoning: Looking at what everyone else has said, this seems right."
    else:
        tail = "Reasoning: My view is unchanged."
    return f"Answer: {answer}\nConfidence: {conf}%\n{tail}"


def simulate(roster: Dict[str, str] = CONSENSUS_SWARM_ROSTER,
             config: SynthConfig = SynthConfig()) -> Tuple[List[Message], List[Observation]]:
    """Returns (messages, ground-truth observations)."""
    rng = random.Random(config.seed)
    llr = math.log(config.signal_accuracy / (1 - config.signal_accuracy))
    messages: List[Message] = []
    truth_obs: List[Observation] = []
    t = 0.0
    for c in range(config.n_claims):
        claim = f"claim-{c:03d}"
        truth = rng.random() < 0.5
        public = 0.0                                   # log-odds from all posted evidence
        own = {a: 0.0 for a in roster}                 # log-odds from own evidence
        fixed = {a: rng.uniform(0.15, 0.85) for a in roster}
        stated: Dict[str, float] = {}
        for _ in range(config.rounds):
            order = list(roster)
            rng.shuffle(order)
            for agent in order:
                kind = roster[agent]
                evidence = False
                if kind in ("bayes", "independent") and rng.random() < config.signal_prob:
                    correct = rng.random() < config.signal_accuracy
                    s = llr if correct == truth else -llr
                    public += s
                    own[agent] += s
                    evidence = True
                if kind == "bayes":
                    belief = _sigmoid(public)
                elif kind == "independent":
                    belief = _sigmoid(own[agent])
                elif kind == "herder":
                    prev = stated.get(agent, fixed[agent])
                    peers = [b for a, b in stated.items() if a != agent]
                    m = sum(peers) / len(peers) if peers else prev
                    belief = prev + config.kappa * (m - prev)
                elif kind == "stubborn":
                    belief = fixed[agent]
                elif kind == "deceiver":
                    belief = 0.05 if truth else 0.95
                else:
                    raise ValueError(f"unknown agent kind: {kind}")
                belief = min(0.99, max(0.01, belief))
                stated[agent] = belief
                t += 1.0
                mid = f"m{int(t):06d}"
                messages.append(Message(mid, t, agent, _render(belief, evidence, kind),
                                        channel=claim, model=kind))
                truth_obs.append(Observation(claim, t, agent, belief, float(evidence), mid))
    return messages, truth_obs


def add_noise(observations: List[Observation], sigma: float, seed: int = 0) -> List[Observation]:
    """Model extraction error: independent Gaussian noise on every belief reading."""
    if sigma <= 0:
        return list(observations)
    rng = random.Random(seed)
    return [Observation(o.claim, o.ts, o.agent,
                        min(1.0, max(0.0, o.belief + rng.gauss(0.0, sigma))),
                        o.evidence, o.msg_id) for o in observations]
