"""Testing by betting.

Null hypothesis for a sequence x_1, x_2, ... in [-1, 1]:

    H0:  E[x_t | everything before t] <= 0   for every t.

Start with wealth 1 and at each step stake a fraction lam_t in [0, max_bet] on
x_t being positive, where lam_t depends only on the past:

    W_t = W_{t-1} * (1 + lam_t * x_t).

Under H0, W is a non-negative supermartingale, so by Ville's inequality

    P( W ever reaches 1/alpha ) <= alpha.

That bound holds at every step simultaneously. You can look after each message,
stop whenever you like, and the error rate is still alpha. No fixed sample size
and no correction for peeking.

The staking rule is aGRAPA (Waudby-Smith & Ramdas, "Estimating means of bounded
random variables by betting", 2023), truncated to one-sided bets. The rule only
affects power. Validity needs nothing more than lam_t being fixed before x_t is seen.
"""
import math
from typing import Dict, Iterable, List


class EProcess:
    def __init__(self, max_bet: float = 0.5, prior_var: float = 0.25):
        if not 0.0 < max_bet < 1.0:
            raise ValueError("max_bet must be in (0, 1) so wealth stays positive")
        self.max_bet = max_bet
        self.prior_var = prior_var
        self.n = 0
        self._sum = 0.0
        self._sumsq = 0.0
        self.log_wealth = 0.0
        self.max_log_wealth = 0.0
        self.path: List[float] = []

    def next_bet(self) -> float:
        """Stake for the next observation. Uses past observations only."""
        if self.n == 0:
            return 0.0
        mean = self._sum / (self.n + 1)                       # shrunk toward 0
        var = (self.prior_var + self._sumsq) / (self.n + 1) - mean * mean
        if mean <= 0.0:
            return 0.0
        return min(self.max_bet, mean / (max(var, 1e-12) + mean * mean))

    def update(self, x: float) -> float:
        if not -1.0 <= x <= 1.0:
            raise ValueError("x must lie in [-1, 1]")
        lam = self.next_bet()
        self.log_wealth += math.log1p(lam * x)
        self.max_log_wealth = max(self.max_log_wealth, self.log_wealth)
        self.n += 1
        self._sum += x
        self._sumsq += x * x
        self.path.append(self.wealth)
        return self.wealth

    def update_many(self, xs: Iterable[float]) -> "EProcess":
        for x in xs:
            self.update(x)
        return self

    @property
    def wealth(self) -> float:
        """E-value if you stop now. Valid at any stopping time."""
        return math.exp(self.log_wealth)

    @property
    def max_wealth(self) -> float:
        return math.exp(self.max_log_wealth)

    @property
    def p_value(self) -> float:
        """Anytime-valid p-value: 1 / (largest wealth reached so far)."""
        return min(1.0, math.exp(-self.max_log_wealth))

    def rejected(self, alpha: float = 0.05) -> bool:
        return self.max_log_wealth >= math.log(1.0 / alpha)


def e_bh(e_values: Dict[str, float], alpha: float = 0.05) -> List[str]:
    """e-BH (Wang & Ramdas 2022). Controls the false discovery rate at alpha
    under arbitrary dependence between the e-values, which matters here because
    agents in one swarm are anything but independent."""
    items = sorted(e_values.items(), key=lambda kv: -kv[1])
    n = len(items)
    k_star = 0
    for k, (_, e) in enumerate(items, start=1):
        if e >= n / (alpha * k):
            k_star = k
    return [name for name, _ in items[:k_star]]


def merge(e_values: Iterable[float]) -> float:
    """The mean of e-values is an e-value, whatever their dependence. Used to
    pool agents into a group (a model family, a team)."""
    vals = list(e_values)
    return sum(vals) / len(vals) if vals else 1.0
