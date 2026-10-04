"""Loading and saving. Datasets differ in column names, so loaders take a mapping
from our field names to theirs, e.g. {"agent": "sender", "text": "content"}."""
import gzip
import json
import re
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .schema import Message, Observation

DEFAULT_MAPPING = {"id": "id", "ts": "ts", "agent": "agent", "text": "text",
                   "channel": "channel", "model": "model"}


def _open(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_jsonl(path: str) -> Iterator[dict]:
    with _open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_ts(value, fallback: float) -> float:
    """Accepts numbers, numeric strings and ISO-8601 timestamps."""
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        pass
    s = s.replace("Z", "+00:00")
    # Python < 3.11 only accepts 3 or 6 fractional digits.
    s = re.sub(r"\.(\d+)", lambda m: "." + (m.group(1) + "000000")[:6], s, count=1)
    try:
        dt = datetime.fromisoformat(s)
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
    except ValueError:
        return fallback


def to_messages(records: Iterable[dict], mapping: Optional[Dict[str, str]] = None) -> List[Message]:
    mp = {**DEFAULT_MAPPING, **(mapping or {})}
    out = []
    for i, r in enumerate(records):
        text = r.get(mp["text"])
        agent = r.get(mp["agent"])
        if not text or agent is None:
            continue
        out.append(Message(
            id=str(r.get(mp["id"], i)), ts=parse_ts(r.get(mp["ts"]), float(i)),
            agent=str(agent), text=str(text),
            channel=str(r.get(mp["channel"], "main")),
            model=(str(r[mp["model"]]) if r.get(mp["model"]) is not None else None)))
    out.sort(key=lambda m: m.ts)
    return out


def load_messages(path: str, mapping: Optional[Dict[str, str]] = None) -> List[Message]:
    return to_messages(read_jsonl(path), mapping)


def load_hf(name: str, split: str = "train", config: Optional[str] = None,
            mapping: Optional[Dict[str, str]] = None, limit: Optional[int] = None) -> List[Message]:
    """Needs `pip install datasets` and, for gated datasets, `huggingface-cli login`."""
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, streaming=True)
    rows = islice(ds, limit) if limit is not None else ds
    return to_messages(rows, mapping)


def load_aivillage(chat_path: str, agents_path: str, room: Optional[str] = None,
                   start: Optional[str] = None, end: Optional[str] = None,
                   agents_only: bool = True) -> List[Message]:
    """AI Village export (aidigestorg/ai-village): chat_messages.jsonl.gz joined
    to agents.jsonl.gz for names and models. `room` is a room-id prefix."""
    agents = {r["id"]: r for r in read_jsonl(agents_path)}
    lo = parse_ts(start, float("-inf")) if start else float("-inf")
    hi = parse_ts(end, float("inf")) if end else float("inf")
    out = []
    for r in read_jsonl(chat_path):
        if agents_only and r.get("speaker_type") != "agent":
            continue
        if room and not str(r.get("room_id", "")).startswith(room):
            continue
        ts = parse_ts(r["created_at"], 0.0)
        if not lo <= ts < hi or not r.get("content"):
            continue
        a = agents.get(r.get("agent_speaker_id"), {})
        out.append(Message(id=r["id"], ts=ts, agent=a.get("name", "unknown"),
                           text=r["content"], channel=str(r["room_id"]),
                           model=a.get("model_string")))
    out.sort(key=lambda m: m.ts)
    return out


def peek(records: Iterable[dict], n: int = 3) -> str:
    """Show field names and a few truncated rows, to work out a mapping."""
    lines = []
    for i, r in enumerate(records):
        if i >= n:
            break
        lines.append(json.dumps({k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                                 for k, v in r.items()}, default=str))
    return "\n".join(lines)


def save_jsonl(items: Iterable, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it.to_dict()) + "\n")


def load_observations(path: str) -> List[Observation]:
    return [Observation(**r) for r in read_jsonl(path)]
