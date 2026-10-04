"""Small deterministic, outcome-blind local-model agreement check.

Uses full target messages (up to --max-chars) and six preceding messages.
The selected message manifest is written before any model inference.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from martingale_audit.evaluation import compare_labels, check_provenance
from martingale_audit.extract import STANCE_PROMPT, _fmt, _parse_json_list, claim_texts
from martingale_audit.io import load_aivillage, load_observations, save_jsonl
from martingale_audit.llm import OpenAICompatLLM, CachedLLM
from martingale_audit.schema import Observation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chat", required=True)
    p.add_argument("--agents", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--limit", type=int, default=12, help="0 selects all reference message IDs")
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--out", default="out/calibration")
    p.add_argument("--cache", default=".cache/calibration.jsonl")
    args = p.parse_args()
    if args.limit < 0 or args.max_chars < 1:
        p.error("limit must be nonnegative and max-chars positive")
    root = Path(__file__).resolve().parents[1]
    data = root / "data/ai_village_saboteur"
    reference = load_observations(data / "observations.jsonl")
    claims = json.loads((data / "claims.json").read_text())
    messages = load_aivillage(args.chat, args.agents, room="18a3", start="2026-03-05", end="2026-03-14")
    provenance = check_provenance(reference, messages)
    if not provenance["passed"]:
        raise ValueError("Reference source provenance failed")
    index = {m.id: i for i, m in enumerate(messages)}
    ids = sorted({o.msg_id for o in reference}, key=lambda mid: hashlib.sha256(("pilot-calibration-v1:" + mid).encode()).hexdigest())
    ids = ids[:args.limit] if args.limit else ids
    ids.sort(key=lambda mid: index[mid])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "selection": "First N unique reference message IDs sorted by SHA256(pilot-calibration-v1: + id), then chronological order", "model": args.model,
                "base_url": args.base_url, "message_ids": ids, "max_chars_per_message": args.max_chars,
                "reference_sha256": hashlib.sha256((data / "observations.jsonl").read_bytes()).hexdigest(),
                "claims_sha256": hashlib.sha256((data / "claims.json").read_bytes()).hexdigest(),
                "outcomes_sent_to_model": False, "scope": "Positive-reference-message subset; not a full-window extraction evaluation"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    llm = CachedLLM(OpenAICompatLLM(args.model, args.base_url, timeout=90), args.cache)
    predictions, failures, truncations = [], [], []
    for j, mid in enumerate(ids):
        i = index[mid]
        m = messages[i]
        day = datetime.fromtimestamp(m.ts, timezone.utc).date().isoformat()
        active = claim_texts({k: v for k, v in claims.items() if v["day"] == day})
        context = messages[max(0, i - 6):i]
        truncations += [x.id for x in context + [m] if len(x.text) > args.max_chars]
        prompt = STANCE_PROMPT.format(claims="\n".join(f"{k}: {v}" for k, v in active.items()), context=_fmt(context, args.max_chars), batch=_fmt([m], args.max_chars))
        try:
            rows = _parse_json_list(llm.complete(prompt, max_tokens=800))
            seen = set()
            for row in rows:
                if str(row["id"]) != mid or str(row["claim"]) not in active:
                    raise ValueError("Model returned an unknown message or claim")
                key = str(row["claim"])
                if key in seen:
                    raise ValueError("Model returned duplicate annotations")
                belief, evidence = float(row["belief"]), float(row["evidence"])
                if not (0 <= belief <= 1 and 0 <= evidence <= 1):
                    raise ValueError("Model returned out-of-range labels")
                seen.add(key)
            predictions.extend(Observation(str(r["claim"]), m.ts, m.agent, float(r["belief"]), float(r["evidence"]), mid) for r in rows)
        except Exception as exc:
            failures.append({"msg_id": mid, "error": type(exc).__name__, "detail": str(exc)[:180]})
        print(f"{j + 1}/{len(ids)} messages complete; {len(failures)} failures", flush=True)
        save_jsonl(predictions, out / "predictions.jsonl")
    selected = [o for o in reference if o.msg_id in set(ids)]
    report = {"manifest": manifest, "provenance": provenance, "agreement": compare_labels(selected, predictions),
              "failures": failures, "truncated_message_ids": sorted(set(truncations))}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["agreement"], indent=2))


if __name__ == "__main__":
    main()
