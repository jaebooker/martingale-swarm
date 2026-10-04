"""Run: python examples/verify_kernel.py (optional dependency: z3-solver)."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from martingale_audit.verification import verify_betting_kernel


if __name__ == "__main__":
    report = verify_betting_kernel()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "proved" else 1)
