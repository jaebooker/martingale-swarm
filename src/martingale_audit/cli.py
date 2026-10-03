"""Command line entry point: python -m martingale_audit <command>."""
import argparse
import json
import sys
from pathlib import Path

from . import bench as bench_mod
from . import cascade, io
from .audit import PRESETS, audit
from .extract import LLMExtractor, RegexExtractor
from .synth import CONSENSUS_SWARM_ROSTER, SynthConfig, simulate


def _mapping(arg):
    return json.loads(arg) if arg else None


def _load(args):
    if getattr(args, "aivillage_agents", None):
        msgs = io.load_aivillage(args.input, args.aivillage_agents, room=args.room,
                                 start=args.start, end=args.end)
        return msgs[:args.limit] if args.limit else msgs
    if args.hf:
        return io.load_hf(args.hf, split=args.split, config=args.hf_config,
                          mapping=_mapping(args.mapping), limit=args.limit)
    msgs = io.load_messages(args.input, _mapping(args.mapping))
    return msgs[:args.limit] if args.limit else msgs


def cmd_demo(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    messages, _ = simulate(config=SynthConfig(seed=args.seed))
    io.save_jsonl(messages, out / "messages.jsonl")
    obs = RegexExtractor().extract(messages)          # full pipeline, from text
    io.save_jsonl(obs, out / "observations.jsonl")
    res = audit(obs, PRESETS[args.preset], groups=CONSENSUS_SWARM_ROSTER)
    (out / "audit.json").write_text(json.dumps(res.to_dict(), indent=1))
    print(res.table())
    try:
        from .report import plot_wealth
        plot_wealth(res, str(out / "wealth.png"), labels=CONSENSUS_SWARM_ROSTER)
        print(f"\nplot: {out / 'wealth.png'}")
    except ImportError:
        print("\n(matplotlib not installed, skipping plot)")


def cmd_bench(args):
    rows = bench_mod.run(trials=args.trials)
    print(bench_mod.table(rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))


def cmd_peek(args):
    if args.hf:
        from datasets import load_dataset
        rows = load_dataset(args.hf, args.hf_config, split=args.split, streaming=True)
    else:
        rows = io.read_jsonl(args.input)
    print(io.peek(rows, args.n))


def _llm(args):
    from .llm import AnthropicLLM, CachedLLM, OpenAICompatLLM
    if args.backend == "openai":
        if not args.model:
            sys.exit("--model is required with --backend openai (e.g. --model llama3.1:8b)")
        return CachedLLM(OpenAICompatLLM(args.model, args.base_url), args.cache)
    return CachedLLM(AnthropicLLM(args.model), args.cache)


def cmd_claims(args):
    claims = LLMExtractor(_llm(args)).discover_claims(_load(args), n=args.n)
    Path(args.out).write_text(json.dumps(claims, indent=1))
    print(json.dumps(claims, indent=1))
    print(f"\nEdit {args.out} by hand, then run `extract`.", file=sys.stderr)


def cmd_extract(args):
    claims = json.loads(Path(args.claims).read_text())
    obs = LLMExtractor(_llm(args), batch_size=args.batch_size).extract(_load(args), claims)
    io.save_jsonl(obs, args.out)
    print(f"{len(obs)} observations -> {args.out}")


def cmd_audit(args):
    obs = io.load_observations(args.observations)
    groups = json.loads(Path(args.groups).read_text()) if args.groups else None
    res = audit(obs, PRESETS[args.preset], alpha=args.alpha, groups=groups)
    print(res.table())
    adoptions = cascade.trace(obs)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(json.dumps(res.to_dict(), indent=1))
        (out / "cascade.json").write_text(json.dumps(
            {"adoptions": [a.__dict__ for a in adoptions],
             "summary": cascade.summary(adoptions)}, indent=1))
        try:
            from .report import plot_wealth
            plot_wealth(res, str(out / "wealth.png"), labels=groups)
        except ImportError:
            pass


def main(argv=None):
    p = argparse.ArgumentParser(prog="martingale-audit", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp):
        sp.add_argument("--input", help="JSONL (or .jsonl.gz) of messages")
        sp.add_argument("--hf", help="Hugging Face dataset name")
        sp.add_argument("--hf-config")
        sp.add_argument("--split", default="train")
        sp.add_argument("--mapping", help='JSON, e.g. \'{"agent": "sender"}\'')
        sp.add_argument("--limit", type=int)
        sp.add_argument("--aivillage-agents", help="agents.jsonl.gz; treats --input as "
                        "the AI Village chat_messages.jsonl.gz")
        sp.add_argument("--room", help="AI Village room id prefix")
        sp.add_argument("--start", help="keep messages at or after this time (ISO)")
        sp.add_argument("--end", help="keep messages before this time (ISO)")

    def llm_args(sp):
        sp.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic",
                        help="'openai' = any OpenAI-compatible server, e.g. Ollama")
        sp.add_argument("--base-url", default="http://localhost:11434/v1")
        sp.add_argument("--model")
        sp.add_argument("--cache", default=".cache/llm.jsonl")

    sp = sub.add_parser("demo", help="simulate a swarm and audit it end to end")
    sp.add_argument("--out", default="out/demo")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--preset", default="conformity", choices=PRESETS)
    sp.set_defaults(fn=cmd_demo)

    sp = sub.add_parser("bench", help="power and false positive rates on simulated swarms")
    sp.add_argument("--trials", type=int, default=200)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_bench)

    sp = sub.add_parser("peek", help="show a dataset's fields")
    data_args(sp)
    sp.add_argument("-n", type=int, default=3)
    sp.set_defaults(fn=cmd_peek)

    sp = sub.add_parser("claims", help="propose contested claims (LLM)")
    data_args(sp)
    llm_args(sp)
    sp.add_argument("-n", type=int, default=15)
    sp.add_argument("--out", default="claims.json")
    sp.set_defaults(fn=cmd_claims)

    sp = sub.add_parser("extract", help="read beliefs off a transcript (LLM)")
    data_args(sp)
    llm_args(sp)
    sp.add_argument("--claims", default="claims.json")
    sp.add_argument("--batch-size", type=int, default=20)
    sp.add_argument("--out", default="observations.jsonl")
    sp.set_defaults(fn=cmd_extract)

    sp = sub.add_parser("audit", help="run the herding test on extracted beliefs")
    sp.add_argument("observations")
    sp.add_argument("--preset", default="conformity", choices=PRESETS)
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--groups", help="JSON file mapping agent -> group (e.g. model family)")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_audit)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
