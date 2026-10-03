"""Thin LLM wrapper. Anything with `complete(prompt) -> str` works as a backend."""
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Protocol


class LLM(Protocol):
    name: str

    def complete(self, prompt: str, max_tokens: int = 2000) -> str: ...


class AnthropicLLM:
    """Needs `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: Optional[str] = None):
        import anthropic
        self.name = model or os.environ.get("MARTINGALE_AUDIT_MODEL", "claude-haiku-4-5")
        self._client = anthropic.Anthropic()

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        r = self._client.messages.create(
            model=self.name, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class CachedLLM:
    """Disk cache keyed on (model, prompt). Extraction over a large transcript is
    the expensive step, so a crashed run should resume for free."""

    def __init__(self, inner: LLM, path: str = ".cache/llm.jsonl"):
        self.inner = inner
        self.name = inner.name
        self.path = Path(path)
        self._cache = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["k"]] = rec["v"]

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        key = hashlib.sha256(f"{self.name}\0{max_tokens}\0{prompt}".encode()).hexdigest()
        if key not in self._cache:
            value = self.inner.complete(prompt, max_tokens)
            self._cache[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"k": key, "v": value}) + "\n")
        return self._cache[key]
