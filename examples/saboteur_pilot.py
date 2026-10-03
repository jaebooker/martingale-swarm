"""Run the audit on the hand-labelled AI Village saboteur-game pilot.

    python examples/saboteur_pilot.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from martingale_audit import PRESETS, StepConfig, audit, cascade  # noqa: E402
from martingale_audit.io import load_observations  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "ai_village_saboteur"
obs = load_observations(str(DATA / "observations.jsonl"))
groups = json.loads((DATA / "groups.json").read_text())
claims = json.loads((DATA / "claims.json").read_text())

print(f"{len(obs)} observations, {len({o.agent for o in obs})} agents, {len(claims)} claims\n")

configs = {
    "conformity (noise-robust, ref_lag=1)": PRESETS["conformity"],
    "conformity, ref_lag=0 (upper bound, see README)": PRESETS["naive"],
    "no evidence gate, ref_lag=0": StepConfig(peer_time="current", gate=False, ref_lag=0),
}
out = {}
for name, cfg in configs.items():
    res = audit(obs, cfg, groups=groups)
    out[name] = res.to_dict(with_paths=False)
    print(f"## {name}\n")
    print(res.table(), "\n")

adoptions = cascade.trace(obs)
summ = cascade.summary(adoptions)
print("## How first stances were formed\n")
print("| agent | first stances | with own evidence | echoed a peer, no evidence | echo share |")
print("|---|---:|---:|---:|---:|")
tot = Counter()
by_group = defaultdict(Counter)
for a, d in sorted(summ.items(), key=lambda kv: -kv[1]["first_echo"] / max(1, kv[1]["first_stances"])):
    n = int(d["first_stances"])
    print(f"| {a} | {n} | {int(d['first_verified'])} | {int(d['first_echo'])} "
          f"| {100 * d['first_echo'] / max(1, n):.0f}% |")
    for k in ("first_stances", "first_verified", "first_echo"):
        tot[k] += d[k]
        by_group[groups.get(a, "?")][k] += d[k]
print(f"| **all** | {int(tot['first_stances'])} | {int(tot['first_verified'])} | "
      f"{int(tot['first_echo'])} | {100 * tot['first_echo'] / tot['first_stances']:.0f}% |\n")
print("| developer | first stances | echo share |")
print("|---|---:|---:|")
for g, d in sorted(by_group.items()):
    print(f"| {g} | {int(d['first_stances'])} | {100 * d['first_echo'] / max(1, d['first_stances']):.0f}% |")

print("\n## Echoed first stances by whether the claim turned out true\n")
split = defaultdict(Counter)
for a in adoptions:
    if a.switched or a.side != 1:
        continue
    t = claims[a.claim]["truth"]
    key = "true" if t is True else "false" if t is False else "unresolved"
    split[key]["accusing first stances"] += 1
    if not a.own_evidence and a.peers_on_side > 0:
        split[key]["echoed"] += 1
print("| claim outcome | accusing first stances | echoed without own evidence |")
print("|---|---:|---:|")
for k in ("true", "false", "unresolved"):
    d = split[k]
    print(f"| {k} | {d['accusing first stances']} | {d['echoed']} |")

print("\n## Evidence-free side switches\n")
for a in adoptions:
    if a.switched and a.evidence_free:
        side = "accuses" if a.side == 1 else "clears"
        print(f"- {a.claim}: {a.agent} now {side}; {100 * a.peers_on_side:.0f}% of peers already there")

(DATA / "results.json").write_text(json.dumps(out, indent=1))
