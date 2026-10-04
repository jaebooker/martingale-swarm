# What the numbers do and do not license

The betting argument is a theorem with a hypothesis. This page states the hypothesis, says where the pipeline can break it, and says which reported numbers survive.

## The theorem

Let x_1, x_2, ... lie in [-1, 1]. Suppose that for every t, E[x_t | F_{t-1}] <= 0, where F_{t-1} is everything the bettor has seen before step t. Let the stake lam_t in [0, 1) be a function of F_{t-1}. Then W_t = prod (1 + lam_s x_s) is a non-negative supermartingale, and P(sup W_t >= 1/alpha) <= alpha.

Everything below is about whether E[x_t | F_{t-1}] <= 0 is a fair description of "this agent is not herding" for scores extracted from a transcript.

## Where the hypothesis can fail without any herding

**Extraction noise.** x_t is built from readings of beliefs. With `ref_lag=0` one noisy reading sets both the direction and the start of the move, and E[x_t] > 0 from noise alone. The benchmark shows false positive rates up to 87%.

With `ref_lag=1` the direction comes from an earlier reading. That removes the unconditional bias for independent noise. It does not make the conditional mean zero: consecutive moves share a reading, so x_{t+1} is predictable from x_t. In simulation this dependence is negative, the staking rule bets with the running mean, and false positives stay under alpha at every noise level tested. That is an empirical statement about one noise model and one staking rule. It is not a proof.

**Missed evidence.** The gate assumes every evidence post is labelled. An unlabelled post that several agents respond to makes them all move the same way in what looks like an evidence-free interval.

**Strategic speech.** A saboteur's public position is a move in a game. It is not a belief.

**Selection.** Claims chosen after seeing the data, or agents added to a group after seeing their results, break any guarantee.

## What each reported number is

| number | valid when | status in the pilot |
|---|---|---|
| Per-agent e-value | the conditional null holds for that agent's extracted scores | Unverified. No agent comes close to a flag, so nothing rests on it. |
| e-BH discoveries | per-agent e-values are valid, family fixed in advance, applied once | Same. |
| Group and swarm e-value (mean of per-agent terminal e-values) | per-agent e-values are valid. No assumption about dependence between agents. | 1.04. No evidence. |
| Concatenated wealth (all agents' steps in time order, one bettor) | the null holds given the whole public history: after seeing agent A move, agent B is still no more likely to move toward peers | Diagnostic only. One missed evidence post breaks it, and the pilot labels were made from excerpts. |
| Toward / away counts | nothing. They are counts. | 5 toward, 0 away (lagged). 12 toward, 1 away (unlagged). |
| Benchmark flag rates | the simulator's noise model | Reported as simulation results. |

## What to say

For the pilot: the counts, the four episodes, the first-stance table, and the sentence "no inferential claim is made". The swarm-level e-value under the conservative pooling is 1.04.

For the method: the unlagged score manufactures herding from measurement error, the lagged score does not do so in simulation, and the betting kernel's one-step algebra is solver-checked (`docs/VERIFICATION.md`).
