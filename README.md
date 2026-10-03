# martingale-audit

Find out which agents in a swarm change their minds because of evidence and which change them because of each other, with a bound on how often the tool is wrong.

Input: a multi-agent transcript. Output, per agent and per model family:

- an **e-value** for herding, with an error guarantee that holds however often you look
- a **pull** estimate: the share of the gap to its peers the agent closes per evidence-free step
- a **cascade trace** per claim: who took which side, in what order, and whether they had evidence

This is the second version of [mixture_of_search_agents_martingale](https://github.com/jaebooker/mixture_of_search_agents_martingale). Version one used a martingale score as the reward inside a debate system. This version takes the mediator out of the swarm and points it at swarms other people are running.

## Quickstart

```bash
pip install -e ".[dev,plot]"
pytest                               # 15 tests, no network
python -m martingale_audit demo      # simulate a 7-agent swarm, audit it from raw text
python -m martingale_audit bench     # power and false positive rates (about 30 s)
```

The demo reproduces the [consensus_swarm](https://github.com/jaebooker/consensus_swarm) roster: six answer agents and a deceptive seventh. Two of the six are herders. The auditor flags those two and nobody else.

![wealth by agent](docs/demo_wealth.png)

## How the test works

For each claim, every agent has a sequence of stated beliefs. Each time an agent restates its belief we ask one question: did it move toward where its peers stood?

We then bet on the answer. Start with wealth 1. Before each step, stake a fraction of wealth on "this agent will move toward its peers", using only what happened before. If the null hypothesis holds, the bet is fair or worse, so wealth is a non-negative supermartingale. Ville's inequality then gives

    P(wealth ever reaches 1/alpha) <= alpha.

Wealth of 20 is evidence at the 5% level. The bound holds at every step at once, so you can audit a live swarm message by message and stop the moment an agent is flagged. Across many agents, e-BH controls the false discovery rate under arbitrary dependence, which matters because agents in one swarm are not independent.

There are two presets. They test different nulls and you should say which one you ran.

| preset | null hypothesis | a rejection means |
|---|---|---|
| `conformity` | In intervals where nobody posts new evidence, the agent is no more likely to move toward its peers than away. | The agent follows the crowd without evidence. |
| `martingale` | The agent's belief is a martingale: its next move cannot be predicted from where the peers already stood when it last spoke. | The agent's updates are predictable from old information. No Bayesian does that. |

`conformity` is the headline test. It uses only the direction of each move, which is the part of an extracted belief you can trust. `martingale` is the stricter claim and needs no evidence labels at all, but it uses magnitudes and is more sensitive to extraction noise.

## Benchmark

Share of agents flagged at alpha = 0.05 over 300 simulated swarms (40 claims, 8 rounds each). "Noise" is the standard deviation of the error added to every belief reading, on a 0 to 1 scale. For herders the number is power. For everyone else it is the false positive rate and should be under 5%.

**conformity**

| agent kind | noise 0 | noise 0.05 | noise 0.1 |
|---|---:|---:|---:|
| herder | 100.0% | 41.7% | 5.7% |
| bayes | 0.0% | 0.3% | 0.2% |
| independent | 0.0% | 0.0% | 1.0% |
| stubborn | 0.0% | 0.0% | 0.0% |
| deceiver | 0.0% | 0.0% | 0.0% |

**martingale**

| agent kind | noise 0 | noise 0.05 | noise 0.1 |
|---|---:|---:|---:|
| herder | 88.3% | 28.7% | 6.8% |
| all others | 0.0% | 0.0% | 0.0% |

**naive** (the conformity test without the noise correction described below)

| agent kind | noise 0 | noise 0.05 | noise 0.1 |
|---|---:|---:|---:|
| herder | 100.0% | 100.0% | 100.0% |
| bayes | 0.0% | 5.8% | 43.0% |
| independent | 0.0% | 24.0% | 87.0% |
| stubborn | 0.0% | 4.3% | 37.0% |

Three things to take from this.

1. **The error bound holds.** False positives stay well under 5% in both real presets at every noise level.
2. **Power is bought with data.** The simulated herders are subtle, moving about 0.02 per step. At noise 0.05, power is 42% with 40 claims and 99.5% with 160.
3. **The obvious version of this test is broken.** See the next section.

Raw numbers are in `docs/bench.json`.

## The regression-to-the-mean trap

The natural way to score herding is: take the agent's last belief, see whether the peers are above or below it, and check whether the agent's next move goes that way. Version one of this project did that.

It fails once beliefs are measured with error. If a reading is too low by chance, the peers look like they are above the agent, and the next reading comes back up by chance. That registers as a move toward the peers. An agent that never changes its mind gets flagged 37% of the time at noise 0.1, and an independent reasoner 87% of the time.

The fix is `ref_lag=1`: decide the direction from the agent's belief two statements back, and measure the move between the last two. The two readings in the move no longer appear in the direction, so independent noise cancels. It costs one statement per agent per claim. Both real presets use it.

Any LLM-judged belief has this kind of error, so this applies to any herding measure built on LLM stance labels.

## Running on real transcripts

Real transcripts have no "Confidence: 80%" lines, so beliefs are read off the text by a model in two steps.

```bash
pip install -e ".[llm,hf]"
export ANTHROPIC_API_KEY=...

# 1. See what the fields are called and write a mapping
python -m martingale_audit peek --hf aidigestorg/ai-village

MAP='{"agent": "<sender field>", "text": "<content field>", "ts": "<time field>", "channel": "<room field>"}'

# 2. Propose contested claims from a sample. Edit claims.json by hand afterwards.
python -m martingale_audit claims --hf aidigestorg/ai-village --mapping "$MAP" --limit 2000

# 3. Read beliefs and evidence off every message. Cached, so reruns are free.
python -m martingale_audit extract --hf aidigestorg/ai-village --mapping "$MAP" --limit 2000

# 4. Audit. groups.json maps agent name -> model family.
python -m martingale_audit audit observations.jsonl --groups groups.json --out out/village
```

Any JSONL file works in place of `--hf` via `--input`. Messages in the same `channel` are assumed visible to everyone in it.

## What the guarantee covers

The error bound is a statement about the belief sequence the test is given. It says: if that sequence satisfies the null, the chance of a flag is at most alpha. Three things sit outside it.

- **Extraction.** The beliefs and evidence labels come from a model. `ref_lag=1` protects against noise that is independent from one reading to the next. It does not protect against a judge that is biased in a correlated way, for example one that reads agreement into any message that follows a confident peer.
- **The evidence gate.** The `conformity` test trusts the label "this message adds new evidence". An agent that fabricates evidence passes the gate. The `martingale` preset does not use the gate.
- **The meaning of a rejection.** Deferring to peers can be rational. If the peers are usually right, following them is good inference, and the economics literature calls the result an information cascade. `conformity` measures a behaviour. It does not show the behaviour was a mistake. `martingale` is the test that is inconsistent with Bayesian updating.

Claim selection is the main researcher degree of freedom. Fix the claim list before looking at audit results.

## Layout

```
src/martingale_audit/
  schema.py        Message, Observation
  extract.py       RegexExtractor, LLMExtractor (claim discovery, stance reading)
  trajectories.py  observations -> bets; the nulls are defined here
  etest.py         the betting test, e-BH, pooling
  audit.py         per-agent and per-group results; presets
  cascade.py       adoption order and evidence-free side switches per claim
  synth.py         simulated swarm with known herders
  bench.py         power and false positive rates
  io.py, llm.py, report.py, cli.py
```

The core has no dependencies. `anthropic`, `datasets` and `matplotlib` are optional.

## Status

Tested: the betting test, step construction, both extractors' parsing, the cascade trace, the full pipeline on simulated text, and the benchmark above.

Not yet run: the LLM extractor against a live model, and anything on real transcripts. `llm.py` and `io.load_hf` are untested against their services.

## Next

1. Run on AI Village and report herding by model family.
2. Hand-label 100 messages and measure the extractor's error, to place real data on the noise axis of the benchmark.
3. Replace the simulator with a live run of consensus_swarm, so the benchmark includes real model text.
4. Temporal-logic monitors over the cascade trace, for example "no agent adopts a claim before any agent has cited evidence for it". `cascade.trace` already records the `unsupported` flag this needs.

## References

- Shafer, "Testing by betting", JRSS-A 2021
- Waudby-Smith and Ramdas, "Estimating means of bounded random variables by betting", JRSS-B 2024
- Wang and Ramdas, "False discovery rate control with e-values", JRSS-B 2022
- Bikhchandani, Hirshleifer and Welch, "A theory of fads, fashion, custom, and cultural change as informational cascades", JPE 1992
