"""Score the auditor on simulated swarms where the herders are known.

For each preset and each level of extraction noise we report, per agent kind,
the share of agents flagged at level alpha. For herders that share is power.
For every other kind it is the false positive rate and should stay under alpha.
"""
from collections import defaultdict
from typing import Dict, List

from .audit import PRESETS, audit
from .synth import CONSENSUS_SWARM_ROSTER, SynthConfig, add_noise, simulate

KINDS = ["herder", "bayes", "independent", "stubborn", "deceiver"]


def run(trials: int = 200, noise_levels=(0.0, 0.05, 0.1), alpha: float = 0.05,
        presets=("conformity", "martingale", "naive"),
        synth: SynthConfig = SynthConfig()) -> List[dict]:
    rows = []
    for sigma in noise_levels:
        tallies = {p: defaultdict(lambda: [0, 0]) for p in presets}
        pulls = {p: defaultdict(list) for p in presets}
        for trial in range(trials):
            cfg = SynthConfig(**{**synth.__dict__, "seed": synth.seed + trial})
            _, truth = simulate(CONSENSUS_SWARM_ROSTER, cfg)
            obs = add_noise(truth, sigma, seed=10_000 + trial)
            for p in presets:
                res = audit(obs, PRESETS[p], alpha=alpha)
                for agent, r in res.agents.items():
                    kind = CONSENSUS_SWARM_ROSTER[agent]
                    tallies[p][kind][0] += int(r.flagged)
                    tallies[p][kind][1] += 1
                    if r.pull is not None:
                        pulls[p][kind].append(r.pull)
        for p in presets:
            for kind in KINDS:
                hit, n = tallies[p][kind]
                pl = pulls[p][kind]
                rows.append({"preset": p, "noise": sigma, "kind": kind,
                             "flag_rate": hit / n if n else None, "n": n,
                             "mean_pull": sum(pl) / len(pl) if pl else None})
    return rows


def table(rows: List[dict]) -> str:
    out = []
    for preset in dict.fromkeys(r["preset"] for r in rows):
        noises = list(dict.fromkeys(r["noise"] for r in rows))
        out.append(f"\n### {preset}\n")
        out.append("| agent kind | " + " | ".join(f"noise {s:g}" for s in noises) + " |")
        out.append("|---|" + "---:|" * len(noises))
        for kind in KINDS:
            cells = []
            for s in noises:
                r = next(r for r in rows if r["preset"] == preset
                         and r["noise"] == s and r["kind"] == kind)
                cells.append(f"{100 * r['flag_rate']:.1f}%")
            out.append(f"| {kind} | " + " | ".join(cells) + " |")
    return "\n".join(out)
