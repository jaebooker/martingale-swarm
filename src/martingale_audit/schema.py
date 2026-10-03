"""Two record types. Everything else in the package is a function between them.

Message      raw transcript line, as loaded from a dataset.
Observation  one agent's belief in one claim at one moment, as produced by an
             extractor. This is the only thing the statistics ever see.
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class Message:
    id: str
    ts: float                 # any monotone clock: unix seconds, turn index, ...
    agent: str
    text: str
    channel: str = "main"     # messages in one channel are mutually visible
    model: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    claim: str                # claim id
    ts: float
    agent: str
    belief: float             # P(claim is true) attributed to the agent, in [0, 1]
    evidence: float = 0.0     # in [0, 1]: does this message add new evidence on the claim?
    msg_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)
