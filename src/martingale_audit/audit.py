"""Compute transcript scores and conditional-null betting diagnostics.

The betting theorem does not verify its assumptions for extracted labels.
See docs/STATISTICAL_VALIDITY.md before interpreting these as inference.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .etest import EProcess, _check_alpha, e_bh, merge
from .schema import Observation
from .trajectories import Step, StepConfig, build_steps

PRESETS: Dict[str, StepConfig] = {
    # Exploratory sign score. A conditional no-toward-drift null is needed
    # separately; lack of an evidence label does not establish that null.
    "conformity": StepConfig(peer_time="current", gate=True, ref_lag=1, statistic="sign"),
    # Lagged magnitude score. The conditional null must hold for observed
    # scores in the bettor's filtration, not merely for a latent belief.
    "martingale": StepConfig(peer_time="previous", gate=False, ref_lag=1,
                             statistic="magnitude"),
    # Unlagged sensitivity score, vulnerable to same-reading noise coupling.
    "naive": StepConfig(peer_time="current", gate=True, ref_lag=0, statistic="sign"),
}


@dataclass
class AgentResult:
    agent: str
    n_steps: int                 # steps that counted
    e_value: float               # wealth at the end
    p_value: float               # anytime-valid only under the conditional score null
    flagged: bool                # crossed 1/alpha at some point (Ville)
    discovery: bool = False      # survives e-BH across all agents
    pull: Optional[float] = None # share of the gap to peers closed per counted step
    first_flag_ts: Optional[float] = None
    path: List[List[float]] = field(default_factory=list)   # [ts, wealth]


@dataclass
class AuditResult:
    config: StepConfig
    alpha: float
    agents: Dict[str, AgentResult]
    groups: Dict[str, Dict[str, float]]
    steps: List[Step]
    pooled: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, with_paths: bool = True) -> dict:
        agents = {}
        for a, r in self.agents.items():
            d = dict(r.__dict__)
            if not with_paths:
                d.pop("path")
            agents[a] = d
        return {"config": self.config.__dict__, "alpha": self.alpha,
                "agents": agents, "groups": self.groups, "pooled": self.pooled}

    def table(self) -> str:
        rows = sorted(self.agents.values(), key=lambda r: -r.e_value)
        out = ["Conditional-null diagnostics; assumptions for extracted labels are unverified.",
               "", "| agent | steps | e-value | p (conditional anytime) | pull | flagged | e-BH |",
               "|---|---:|---:|---:|---:|:-:|:-:|"]
        for r in rows:
            pull = "" if r.pull is None else f"{r.pull:+.2f}"
            out.append(f"| {r.agent} | {r.n_steps} | {r.e_value:.3g} | {r.p_value:.3g} "
                       f"| {pull} | {'yes' if r.flagged else ''} | {'yes' if r.discovery else ''} |")
        if self.groups:
            out += ["", "| group | agents | e-value | flagged |", "|---|---:|---:|:-:|"]
            for g, d in sorted(self.groups.items(), key=lambda kv: -kv[1]["e_value"]):
                out.append(f"| {g} | {int(d['n_agents'])} | {d['e_value']:.3g} "
                           f"| {'yes' if d['e_value'] >= 1 / self.alpha else ''} |")
        if self.pooled:
            p = self.pooled
            out += ["", f"Swarm summary: {int(p['n_steps'])} counted steps "
                        f"({int(p['toward'])} toward peers, {int(p['away'])} away, "
                        f"{int(p['no_move'])} no move). Mean terminal agent wealth "
                        f"{p['e_value']:.3g}; inverse mean {p['p_value']:.3g} "
                        "(fixed-horizon conditional p-value).",
                    f"Concatenated wealth {p['diagnostic_concatenated_wealth']:.3g} "
                    "is descriptive only; cross-agent dependence is not corrected."]
        return "\n".join(out)


def audit(observations: Iterable[Observation],
          config: StepConfig = PRESETS["conformity"],
          alpha: float = 0.05,
          groups: Optional[Dict[str, str]] = None,
          max_bet: Optional[float] = None) -> AuditResult:
    """`groups` maps agent -> group name (for example its model family).

    `max_bet` caps the stake in (0, 1). The default is 0.5 for
    the sign statistic, where one wrong call costs the full stake, and 0.9 for
    the magnitude statistic, where moves are small and a low cap wastes power.

    Per-agent guarantees require a conditional-null score sequence. The pooled
    headline averages current per-agent wealth, not interleaved increments.
    Its inverse is a fixed-horizon conditional p-value for a prespecified
    agent family. Extracted labels do not establish these assumptions.
    """
    _check_alpha(alpha)
    if max_bet is None:
        max_bet = 0.5 if config.statistic == "sign" else 0.9
    observations = list(observations)
    steps = build_steps(observations, config)
    per_agent: Dict[str, List[Step]] = defaultdict(list)
    for s in steps:
        if s.counted:
            per_agent[s.agent].append(s)

    results: Dict[str, AgentResult] = {}
    for agent in sorted({o.agent for o in observations}):
        ep = EProcess(max_bet=max_bet)
        path, first_flag = [], None
        moved = gap_total = 0.0
        for s in per_agent.get(agent, []):
            ep.update(s.x)
            path.append([s.ts, ep.wealth])
            moved += s.direction * s.delta
            gap_total += abs(s.gap)
            if first_flag is None and ep.rejected(alpha):
                first_flag = s.ts
        results[agent] = AgentResult(
            agent=agent, n_steps=ep.n, e_value=ep.wealth, p_value=ep.p_value,
            flagged=ep.rejected(alpha), pull=(moved / gap_total if gap_total else None),
            first_flag_ts=first_flag, path=path)

    for a in e_bh({a: r.e_value for a, r in results.items()}, alpha):
        results[a].discovery = True

    group_out: Dict[str, Dict[str, float]] = {}
    if groups:
        members: Dict[str, List[float]] = defaultdict(list)
        for a, r in results.items():
            members[groups.get(a, "unknown")].append(r.e_value)
        group_out = {g: {"e_value": merge(v), "n_agents": float(len(v))}
                     for g, v in members.items()}
    # Diagnostic only: interleaving increments requires a conditional null in
    # the pooled filtration. Correlated agent updates do not establish it.
    pool = EProcess(max_bet=max_bet)
    counted = [s for s in steps if s.counted]
    for s in counted:
        pool.update(s.x)
    mean_wealth = merge(r.e_value for r in results.values())
    pooled = {"n_steps": float(pool.n), "n_agents": float(len(results)),
              "e_value": mean_wealth, "p_value": min(1.0, 1.0 / mean_wealth) if mean_wealth else 1.0,
              "flagged": float(mean_wealth >= 1.0 / alpha),
              "scope": "fixed_horizon_mean_of_terminal_agent_e_values",
              "inference": "conditional_score_null_and_prespecified_family_required; unverified_for_extracted_labels",
              "diagnostic_concatenated_wealth": pool.wealth,
              "toward": float(sum(s.x > 0 for s in counted)),
              "away": float(sum(s.x < 0 for s in counted)),
              "no_move": float(sum(s.x == 0 for s in counted))}
    return AuditResult(config, alpha, results, group_out, steps, pooled)
