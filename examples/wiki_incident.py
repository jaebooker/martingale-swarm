"""Audit of the German wiki relay incident (collusion.wiki export).

    python examples/wiki_incident.py /path/to/full-wiki-logs

The export is not redistributed here. It needs revisions.jsonl and pages.jsonl.

Two analyses.

1. Technique spread. Agents used third-party proxy and markdown hosts to reach
   pages they could not open directly. For each account's first use of such a
   host we ask whether the line was copied verbatim from another account's
   earlier text or written fresh.

2. A value cascade with a known reversal. In the "oecd-equity" task, parallel
   cohorts relayed four answers read from a one-decimal workbook (9.70, 9.90,
   16.40, 14.60). Later cohorts rendered the live chart and posted two-decimal
   values (9.69, 9.91, 16.38, 14.59). Each signed message that asserts either
   value is one observation for the herding audit.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from martingale_audit import Observation, PRESETS, StepConfig, audit, cascade  # noqa: E402
from martingale_audit.verification import verify_trace  # noqa: E402

RELAY_HOSTS = {
    "jqp.vercel.app", "md.succ.ai", "markdown.new", "allorigins.hexlet.app", "r.jina.ai",
    "pure.md", "cors.bwa.workers.dev", "www.proxymule.com", "md.dhr.wtf", "corsmirror.com",
    "api.allorigins.win", "api.cors.lol", "webcrawlerapi.com", "tinyurl.com", "is.gd",
    "vanderbi.lt", "api.counterapi.dev",
}
HOST = re.compile(r"https?://([A-Za-z0-9.\-]+)")
SIGNATURE = re.compile(r"--\s*([A-Za-z0-9_]+)\s*$")
EVIDENCE = re.compile(r"independent|verif|reproduc|replicat|rendered|proof|screenshot|\bDOM\b|"
                      r"tooltip|bypass|raw (value|response|query)|querydata|workbook|xlsx", re.I)
OWN_CHECK = re.compile(r"\b(I|we)\b[^.]{0,60}\b(independently|reproduc\w*|replicat\w*|rendered|"
                       r"verified|bypassed|confirmed via|queried)|independently (bypassed|rendered|"
                       r"reproduc\w*|confirmed|verified)", re.I)
# claim -> (workbook value pattern, chart value pattern)
VALUE_CLAIMS = {
    "SVK=14.60": (r"14\.60\b|14\.6\b(?!\d)", r"14\.59"),
    "HUN=9.90": (r"\b9\.90\b", r"\b9\.91\b"),
    "POL=16.40": (r"16\.40\b|16\.4\b(?!\d)", r"16\.38"),
    "CZE=9.70": (r"\b9\.70\b", r"\b9\.69\b"),
}


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def norm(line):
    return re.sub(r"\d{6,}(\.\d+)?", "#", line.strip())


def load(root):
    pages = {}
    for line in open(root / "pages.jsonl"):
        p = json.loads(line)
        pages[p["page_id"]] = p
    revs = [json.loads(line) for line in open(root / "revisions.jsonl")]
    for r in revs:
        r["t"] = ts(r["time"])
    revs.sort(key=lambda r: (r["t"], r["page_id"], r["seq"]))
    return pages, revs


def technique_spread(pages, revs):
    seen, prev, first_use, origin = {}, {}, {}, {}
    for r in revs:
        body = r["body"] or ""
        old = {norm(x) for x in prev.get(r["page_id"], "").splitlines()}
        new = [norm(x) for x in body.splitlines() if len(x.strip()) > 30 and norm(x) not in old]
        for line in new:
            for h in {x.lower() for x in HOST.findall(line)} & RELAY_HOSTS:
                key = (r["label"], h)
                if h not in origin:
                    origin[h] = r["time"][:10]
                    first_use[key] = "origin"
                elif key not in first_use:
                    src = seen.get(line)
                    first_use[key] = ("verbatim" if src and src != r["label"]
                                      else "self" if src else "fresh")
        for line in new:
            seen.setdefault(line, r["label"])
        prev[r["page_id"]] = body
    print("## Technique spread\n")
    print(f"{len({r['label'] for r in revs})} account names, {len({r['ip16'] for r in revs})} /16 "
          f"networks, {len(revs)} revisions, {len(pages)} pages\n")
    print("| relay host | first seen | accounts | copied verbatim | wrote fresh |")
    print("|---|---|---:|---:|---:|")
    total = Counter()
    rows = []
    for h in RELAY_HOSTS:
        c = Counter(v for (_, hh), v in first_use.items() if hh == h and v != "origin")
        total.update(c)
        rows.append((c["verbatim"] + c["fresh"] + c["self"], h, c))
    for n, h, c in sorted(rows, reverse=True):
        print(f"| {h} | {origin.get(h, '')} | {n} | {c['verbatim']} | {c['fresh']} |")
    n = total["verbatim"] + total["fresh"]
    print(f"\nAcross hosts: {total['verbatim']} of {n} first uses "
          f"({100 * total['verbatim'] / n:.0f}%) carried a line copied verbatim from another "
          f"account. The rest wrote the line fresh, which does not show they had not read one.\n")


def value_observations(pages, revs):
    """One observation per first appearance of a signed line asserting a value."""
    seen, obs, own = set(), [], set()
    for r in revs:
        if pages[r["page_id"]]["page_family"] != "oecd-equity":
            continue
        for k, line in enumerate((r["body"] or "").splitlines()):
            line = line.strip()
            if len(line) < 25 or line in seen:
                continue
            seen.add(line)
            m = SIGNATURE.search(line)
            who = m.group(1) if m else r["label"]
            for claim, (old, new) in VALUE_CLAIMS.items():
                has_old, has_new = re.search(old, line), re.search(new, line)
                if not (has_old or has_new):
                    continue
                # A line naming both values is a correction: it asserts the new one.
                belief = 0.1 if has_new else 0.9
                mid = f"{r['rev_id']}#{k}"
                obs.append(Observation(claim, r["t"], who, belief,
                                       1.0 if EVIDENCE.search(line) else 0.0, mid))
                if OWN_CHECK.search(line):
                    own.add(mid)
    return obs, own


def value_cascade(pages, revs):
    obs, own = value_observations(pages, revs)
    print("## The workbook-value cascade (oecd-equity)\n")
    print(f"{len(obs)} assertions by {len({o.agent for o in obs})} signers on "
          f"{len(VALUE_CLAIMS)} values\n")
    print("| claim | signers on workbook value | still posting after the correction | "
          "of those, moved | of those, stayed | first chart-value post (UTC) |")
    print("|---|---:|---:|---:|---:|---|")
    for claim in VALUE_CLAIMS:
        by = defaultdict(list)
        for o in obs:
            if o.claim == claim:
                by[o.agent].append(o)
        firsts = [o.ts for o in obs if o.claim == claim and o.belief < 0.5]
        t0 = min(firsts)
        held = [a for a, v in by.items() if any(o.belief > 0.5 and o.ts < t0 for o in v)]
        after = [a for a in held if any(o.ts >= t0 for o in by[a])]
        moved = [a for a in after if by[a][-1].belief < 0.5]
        first = datetime.fromtimestamp(t0, timezone.utc).strftime("%m-%d %H:%M")
        print(f"| {claim} | {len(held)} | {len(after)} | {len(moved)} | "
              f"{len(after) - len(moved)} | {first} |")

    ad = cascade.trace(obs)
    up = [a for a in ad if a.side == 1 and not a.switched]
    echo = [a for a in up if not a.own_evidence and a.peers_on_side > 0]
    sw = [a for a in ad if a.switched and a.side == -1]
    print(f"\nFirst stances on a workbook value: {len(up)}. Of these, {len(echo)} "
          f"({100 * len(echo) / max(1, len(up)):.0f}%) repeated a value a peer had already "
          f"posted and cited no source of their own.")
    print(f"Switches to the chart value: {len(sw)}. With no evidence posted since the signer "
          f"last spoke: {sum(a.evidence_free for a in sw)}. Switching message describes the "
          f"signer's own check: {sum(a.msg_id in own for a in sw)}. The rest switched on "
          f"evidence other signers had posted.\n")

    for name, cfg in (("conformity (lagged)", PRESETS["conformity"]),
                      ("conformity, ref_lag=0", PRESETS["naive"]),
                      ("no evidence gate, ref_lag=0", StepConfig(gate=False, ref_lag=0))):
        p = audit(obs, cfg).pooled
        print(f"- {name}: {int(p['n_steps'])} counted steps, {int(p['toward'])} toward peers, "
              f"{int(p['away'])} away; swarm e-value {p['e_value']:.3g} "
              f"(diagnostic concatenated {p['diagnostic_concatenated_wealth']:.3g})")
    s = verify_trace(obs)["summary"]
    print(f"\nTrace policy: {s['adoptions_checked']} adoptions checked, "
          f"{s['unsupported_adoptions']} with no annotated evidence before them, "
          f"{s['ambiguous_adoptions']} ambiguous.")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "full-wiki-logs")
    pages, revs = load(root)
    technique_spread(pages, revs)
    value_cascade(pages, revs)
