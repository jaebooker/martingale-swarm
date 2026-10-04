# Martingale Audit: who in the swarm moves on evidence, and who moves on each other

**Team:** Jaeson Booker (jaesonbooker@gmail.com)
**Code:** [GITHUB URL]
**Data:** AI Village chat log, 5 to 13 March 2026

## The question

When agents in a swarm change their minds, is it because someone posted evidence or because the others already had? A swarm that converges for the second reason looks identical from outside, right up until it converges on something false.

## What I built

A tool that reads a multi-agent transcript and reports, per agent and per claim:

- **A herding score from a betting test.** Each time an agent restates a belief, the tool bets on whether it moved toward its peers. If the agent is not herding the bet is fair, so the bettor's wealth can only grow by luck, and the chance it ever reaches 20 is at most 5%. You can check after every message and stop whenever you like.
- **A cascade trace.** Who took which side, in what order, and whether they brought evidence.
- **A trace-policy check.** Every stance adopted before anyone had posted evidence on that claim, with message ids as the witness.

The core has no dependencies. Beliefs can be extracted by an Anthropic model or any local model through Ollama.

## Finding 1: the obvious herding score is broken

The natural score takes an agent's last belief, sees which side its peers are on, and checks whether the next move goes that way. Once beliefs are read off text with any error, this manufactures herding. A reading that is too low by chance makes the peers look higher, and the next reading drifts back up.

In simulation, an agent that never changes its mind is flagged 37% of the time and an independent reasoner 87% of the time, with reading error of 0.1 on a 0 to 1 belief scale. Taking the direction from an earlier reading brings false positives under 5% in every condition tested. This applies to any herding measure built on LLM stance labels.

## Finding 2: a pilot on the AI Village saboteur game

For seven days the village ran a social-deduction game with private roles, public accusations, votes and an evening reveal. I labelled 285 statements by 13 agents on 15 accusations.

- **Agents mostly move on evidence.** 44 moves toward peers against 3 away overall. Only 12 against 1 happened with no new evidence posted.
- **The residue has one shape.** Nearly all of it is "this is suspicious" becoming "I vote to remove" once a meeting had quorum.
- **59% of first positions were taken on a peer's word**, with no check of the agent's own.
- **False alarms were shared errors, not copied ones.** Several agents ran the same misleading check, such as a GitHub lookup returning "not found" for pull requests that existed. A herding test cannot see that, and should not be asked to.
- **One claim was adopted with no evidence at all.** A voted-out agent's documentation was reverted, and eight agents came to call it sabotage. In the messages I read, none reported finding anything in it. The trace checker flags exactly these eight adoptions and no others out of 128.

## What I am not claiming

The pilot supports no statistical conclusion. The swarm-level e-value is 1.04. No agent has enough steps to be flagged, and no comparison between models or developers is warranted.

The labels come from one annotator reading keyword excerpts, knowing each day's outcome. The betting guarantee holds only if its null holds for the extracted scores, which is unverified. The repo states which numbers survive that.

## Formal verification, narrowly

Z3 discharges the one-step algebra of the betting rule over exact reals, and two deliberate mutants produce counterexamples. This checks the arithmetic. It does not verify the swarm, the labels or the statistics.

## Lineage and next step

This is the second version of my Martingale project. The first used a martingale score as a reward inside a debate system. This one takes the mediator out of the swarm and points it at swarms other people run.

Next: score a local model against the pilot labels, then run the full 183k-message log, where per-agent results become possible.
