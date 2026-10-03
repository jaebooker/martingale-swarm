"""Plot the wealth of the bet against each agent. Needs matplotlib."""
import math
from typing import Dict, Optional

from .audit import AuditResult


def plot_wealth(result: AuditResult, path: str, labels: Optional[Dict[str, str]] = None,
                title: str = "Evidence of herding, by agent") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for agent, r in sorted(result.agents.items()):
        if not r.path:
            continue
        ys = [math.log10(max(w, 1e-12)) for _, w in r.path]
        label = f"{agent} ({labels[agent]})" if labels and agent in labels else agent
        ax.plot(range(1, len(ys) + 1), ys, label=label, linewidth=1.6)
    ax.axhline(math.log10(1 / result.alpha), color="black", linestyle="--", linewidth=1)
    ax.text(0.5, math.log10(1 / result.alpha), f" flag at 1/alpha = {1 / result.alpha:g}",
            va="bottom", fontsize=9)
    ax.set_xlabel("counted steps for that agent")
    ax.set_ylabel("log10 wealth (e-value)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
