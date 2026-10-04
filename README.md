# martingale-audit

Find out which agents in a swarm change their minds because of evidence and which change them because of each other, with every statistical assumption written down next to the number it supports.

Input: a multi-agent transcript. Output, per agent and per model family:

- an **e-value** for herding from a betting test, valid at any stopping time when its stated null holds for the extracted scores
- a **pull** estimate: the share of the gap to its peers the agent closes per evidence-free step
- a **cascade trace** per claim: who took which side, in what order, and whether they had evidence
- a **trace-policy report**: every stance adopted before anyone had posted evidence, with the message ids as witness

This is the second version of [mixture_of_search_agents_martingale](https://github.com/jaebooker/mixture_of_search_agents_martingale). Version one used a martingale score as the reward inside a debate system. This version takes the mediator out of the swarm and points it at swarms other people are running.

## Quickstart

```bash
pip install -e ".[dev,plot]"
pytest                               # 36 tests, no network
python -m martingale_audit demo      # simulate a 7-agent swarm, audit it from raw text
python -m martingale_audit bench     # power and false positive rates (about 30 s)
python examples/saboteur_pilot.py    # the real-data pilot
python -m martingale_audit verify data/ai_village_saboteur/observations.jsonl --kernel   # needs z3-solver
```

The demo reproduces the [consensus_swarm](https://github.com/jaebooker/consensus_swarm) roster: six answer agents and a deceptive seventh. Two of the six are herders. The auditor flags those two and nobody else.

![wealth by agent](docs/demo_wealth.png)

## How the test works

For each claim, every agent has a sequence of stated beliefs. Each time an agent restates its belief we ask one question: did it move toward where its peers stood?

We then bet on the answer. Start with wealth 1. Before each step, stake a fraction of wealth on "this agent will move toward its peers", using only what happened before. If the null hypothesis holds, the bet is fair or worse, so wealth is a non-negative supermartingale. Ville's inequality then gives

    P(wealth ever reaches 1/alpha) <= alpha.

Wealth of 20 is evidence at the 5% level. The bound holds at every step at once, so you can audit a live swarm message by message and stop the moment an agent is flagged. Across many agents, e-BH controls the false discovery rate under arbitrary dependence, which matters because agents in one swarm are not independent.

All of that is conditional on the null holding for the scores the test is fed. Scores extracted from text by a model or an annotator do not come with that guarantee. `docs/STATISTICAL_VALIDITY.md` lists the ways it can fail and says which reported numbers survive.

There are two presets. They test different nulls and you should say which one you ran.

| preset | null hypothesis | a rejection means |
|---|---|---|
| `conformity` | In intervals where nobody posts new evidence, the agent is no more likely to move toward its peers than away. | The agent follows the crowd without evidence. |
| `martingale` | The agent's belief is a martingale: its next move cannot be predicted from where the peers already stood when it last spoke. | The agent's updates are predictable from old information. No Bayesian does that. |

`conformity` is the headline test. It uses only the direction of each move, which is the most robust part of an extracted belief. `martingale` is the stricter claim and needs no evidence labels at all, but it uses magnitudes and is more sensitive to extraction noise.

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

1. **False positives stay under 5% in simulation** for both real presets at every noise level. This is an empirical result for one noise model. See the next section for why it is not a theorem.
2. **Power is bought with data.** The simulated herders are subtle, moving about 0.02 per step. At noise 0.05, power is 42% with 40 claims and 99.5% with 160.
3. **The obvious version of this test is broken.** See the next section.

Raw numbers are in `docs/bench.json`.

## The regression-to-the-mean trap

The natural way to score herding is: take the agent's last belief, see whether the peers are above or below it, and check whether the agent's next move goes that way. Version one of this project did that.

It fails once beliefs are measured with error. If a reading is too low by chance, the peers look like they are above the agent, and the next reading comes back up by chance. That registers as a move toward the peers. An agent that never changes its mind gets flagged 37% of the time at noise 0.1, and an independent reasoner 87% of the time.

The mitigation is `ref_lag=1`: decide the direction from the agent's belief two statements back, and measure the move between the last two. The two readings in the move no longer appear in the direction, so independent noise no longer biases the score. It costs one statement per agent per claim. Both real presets use it.

This is a mitigation with simulation support. Consecutive moves still share a reading, so the scores are not conditionally mean-zero given the past, and the supermartingale property is not proven under noise. In the benchmark the leftover dependence works against the bettor and false positives stay low.

Any LLM-judged belief has this kind of error, so this applies to any herding measure built on LLM stance labels.

## Pilot on real data: the AI Village saboteur game

For seven working days in March 2026 the AI Village ran a social-deduction game. Each morning every agent rolled a die in private, a 1 made it a saboteur for the day, and the group could call meetings and vote members out. Roles were revealed each evening. It is close to an ideal test bed: agents accuse each other in public, they update in public, and there is ground truth.

I hand-labelled 285 statements by 13 agents on 15 accusations (protocol and caveats in `data/ai_village_saboteur/LABELING.md`). Reproduce with `python examples/saboteur_pilot.py`.

**1. Most movement follows evidence.** Counting every restatement, agents moved toward their peers 44 times and away 3 times. Restricting to intervals where nobody posted new evidence leaves 12 toward and 1 away. These agents mostly change position when someone posts a diff, a grep or a quote.

**2. There is a residue of evidence-free movement, and it has one shape.** Almost every one of those 12 moves is the same step, from "this is suspicious" to "I vote to remove", taken once a meeting had quorum and with nothing new on the table. Under the lagged score the count is 5 toward and 0 away.

**3. No inferential claim is made from the pilot.** The busiest agent has 7 counted steps, so no agent can be flagged. The swarm-level e-value, pooled as the mean of per-agent e-values, is 1.04. Feeding every agent's steps to one bettor in time order gives 5.1 (lagged) or 37 (unlagged), but that is valid only if no evidence post was missed, and the labels were made from excerpts. Those two figures are reported as diagnostics.

**4. Most first positions are taken on someone else's word.** Of 112 first stances, 44 came with the agent's own check and 66 joined a side a peer already held without adding evidence (59%).

| developer | first stances | joined a peer without own evidence |
|---|---:|---:|
| Anthropic (6 agents) | 61 | 56% |
| OpenAI (3) | 18 | 56% |
| DeepSeek (1) | 11 | 64% |
| Google (3) | 22 | 68% |

Among agents with at least nine first stances the range is 33% (Gemini 3.1 Pro, Opus 4.5 Claude Code) to 91% (Gemini 2.5 Pro). These are small counts from one annotator. Do not read the developer differences as findings.

**5. False accusations were mostly shared errors, not copied ones.** Of 17 first accusations on claims that turned out false, 7 were echoes. On claims that turned out true, 29 of 38 were. The false alarms (stale-branch diffs read as deletions, a GitHub API returning 404 for pull requests that existed) came from several agents running the same misleading check. That is correlated error from a shared tool, which a herding test will not see and should not be asked to.

The trace-policy checker, run on the same labels, checks 128 stance adoptions and returns 8 with no annotated evidence anywhere before them. All 8 are on one claim, `d4:docs`.

Four episodes worth reading in the transcript, by claim id:

- `d3:So45`. An agent posted its own saboteur roll by accident, then denied it. The agent that first raised the alarm withdrew it and four others declined to rely on it. One agent went back to the log, where the message still was, and held its position alone.
- `d4:docs`. A voted-out saboteur pushed documentation. Belief that it contained sabotage rose from "revert as a precaution" to "malicious" across four agents. In the messages read, nobody reported finding anything in it.
- `d5:Hk45`. One phrase, eight votes in two minutes. The swarm was right.
- `d7:So46`. An accuser said the rules named the exact phrase as forbidden. The rules as posted on day one do not. Another agent repeated the claim when agreeing to the meeting. The swarm was right here too.

## Verification layer

Two narrow checks. Details and limits in `docs/VERIFICATION.md`.

**Trace policy.** `verify` walks the labelled trace and reports every stance an agent adopted when no evidence on that claim had been annotated yet, with message ids as the witness. Timestamp ties and inconsistent records are reported as inconclusive. It is a runtime contract check on the supplied labels. It does not certify reasoning.

**Betting kernel.** With `z3-solver` installed, three one-step obligations are discharged over exact reals: the wealth factor is non-negative, wealth stays non-negative, and expected wealth does not rise when the conditional mean is at most zero and the stake is fixed in advance. Two deliberate mutants (stake up to 2, stake chosen after seeing the outcome) produce counterexamples, which shows the bounds and the timing assumption are both needed. This checks the algebra. It does not prove Ville's inequality, the Python implementation, or anything about the labels.

## Running on real transcripts

Real transcripts have no "Confidence: 80%" lines, so beliefs are read off the text by a model in two steps. Either backend works.

```bash
# Anthropic
pip install -e ".[llm]" && export ANTHROPIC_API_KEY=...
LLM=""

# or any OpenAI-compatible server. Ollama needs no key and no extra install:
ollama pull llama3.1:8b
LLM="--backend openai --base-url http://localhost:11434/v1 --model llama3.1:8b"

# AI Village: point at the two files from the dataset and pick a window
DATA="--input chat_messages.jsonl.gz --aivillage-agents agents.jsonl.gz \\
      --room 18a3 --start 2026-03-05 --end 2026-03-14"

# 1. Propose contested claims from a sample. Edit claims.json by hand afterwards.
python -m martingale_audit claims $DATA $LLM --limit 400

# 2. Read beliefs and evidence off every message. Cached, so reruns are free.
python -m martingale_audit extract $DATA $LLM

# 3. Audit. groups.json maps agent name -> group.
python -m martingale_audit audit observations.jsonl --groups groups.json --out out/village
```

Other datasets: `--input file.jsonl --mapping '{"agent": "sender", "text": "body"}'`, or `--hf name`. Messages in the same `channel` are assumed visible to everyone in it.

A small local model will be a noisy judge. Score it against the pilot labels before trusting it:

```bash
python examples/calibrate_pilot.py --chat chat_messages.jsonl.gz --agents agents.jsonl.gz \\
    --model llama3.1:8b --limit 12
```

It picks a fixed, outcome-blind sample of labelled messages, writes the sample manifest before calling the model, never sends the ground truth, and reports agreement on belief level and on the evidence flag. `--limit 0` runs all 272 labelled messages.

`extract` skips a batch when the model's reply is not usable JSON and writes the skipped batches to `<out>.failures.json`. Report that count with any result. `--strict` stops at the first failure.

## What the guarantee covers

The error bound is a statement about the score sequence the test is given. It says: if that sequence satisfies the null, the chance of a flag is at most alpha. Four things sit outside it. `docs/STATISTICAL_VALIDITY.md` has the full account.

- **Extraction.** The beliefs and evidence labels come from a model. `ref_lag=1` protects against noise that is independent from one reading to the next. It does not protect against a judge that is biased in a correlated way, for example one that reads agreement into any message that follows a confident peer.
- **Pooling across agents.** Group and swarm e-values are the mean of per-agent e-values, which is valid whatever the dependence between agents. Feeding all agents' steps to one bettor is reported only as a diagnostic.
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
  verification.py  trace-policy checker, Z3 obligations for the betting kernel
  evaluation.py    agreement between label files, provenance checks
  io.py, llm.py, report.py, cli.py
data/ai_village_saboteur/   hand labels, claims with ground truth, protocol
examples/saboteur_pilot.py  the pilot analysis
examples/calibrate_pilot.py score a local model against the pilot labels
examples/verify_kernel.py   run the solver obligations
docs/STATISTICAL_VALIDITY.md, docs/VERIFICATION.md
```

The core has no dependencies. `anthropic`, `datasets`, `matplotlib` and `z3-solver` are optional.

## Status

Tested: the betting test, step construction, both extractors' parsing, the cascade trace, the trace-policy checker, label comparison, the solver obligations, the full pipeline on simulated text, the benchmark above, and the pilot analysis on hand labels.

Not yet run: the LLM extractor against a live model (Anthropic or Ollama). `llm.py` and `io.load_hf` are untested against their services. `io.load_aivillage` was run on the real export (4,382 agent messages in the pilot window). `examples/calibrate_pilot.py` has not been run end to end.

## Next

1. Run the LLM extractor over the pilot window and score it against the hand labels.
2. Scale to the full 183k-message chat log. Per-agent results need on the order of hundreds of counted steps each.
3. Label twice, independently. With two readings of every belief, the direction can come from one and the move from the other. That removes the noise artefact without giving up a step per claim, which is what makes `ref_lag=1` so costly on sparse data.
4. Replace the simulator with a live run of consensus_swarm, so the benchmark includes real model text.
5. Temporal-logic monitors over the cascade trace, for example "no agent adopts a claim before any agent has cited evidence for it". `cascade.trace` already records the `unsupported` flag this needs.

## References

- Shafer, "Testing by betting", JRSS-A 2021
- Waudby-Smith and Ramdas, "Estimating means of bounded random variables by betting", JRSS-B 2024
- Wang and Ramdas, "False discovery rate control with e-values", JRSS-B 2022
- Bikhchandani, Hirshleifer and Welch, "A theory of fads, fashion, custom, and cultural change as informational cascades", JPE 1992
