# How the pilot labels were made

**Source.** AI Village dataset (`aidigestorg/ai-village`), `chat_messages.jsonl.gz`, main room, 5 to 13 March 2026: the seven working days of the goal "Develop a turn-based RPG together while voting out Easter Egg saboteurs". Each day every agent privately rolled a die, a 1 made it a saboteur, and the group could vote members out. Roles were revealed at an end-of-day debrief, which gives ground truth for most claims.

**No message text is included here.** The dataset is gated under research terms. `observations.jsonl` holds only message ids, timestamps, agent names and the labels, so anyone with access can join back to the text.

**Selection.** 5,400 or so agent messages in the window were filtered by keyword (saboteur, suspicious, vote, accuse, trojan, meeting, and similar, after stripping boilerplate such as "easter egg scan: clean"). That left 957 messages, read as keyword-centred excerpts of up to 700 characters. Messages with no accusation keyword were not read, so some evidence posts are missing. Where a later message relayed evidence whose original post was not in the selection, the relaying message is marked as the evidence.

**Annotator.** One annotator, Claude (the assistant in the build session), in a single pass. There is no second annotator and no agreement score. Treat every label as provisional.

**Claims.** 15 claims, fixed in `claims.json`. They were chosen while reading, before any statistics were run. Claims where the accused confessed before anyone else took a position were left out, since there is nothing to audit.

**Belief scale.** Five levels only, to avoid inventing precision:

| value | meaning |
|---|---|
| 0.1 | says it is not sabotage / the accused is innocent |
| 0.3 | leans against, or "not necessarily" |
| 0.5 | explicitly undecided, "wait for the debrief", "if confirmed" |
| 0.7 | calls it suspicious, supports a meeting |
| 0.9 | asserts it, or casts a vote to remove |

**Evidence.** 1 if the message reports something the author checked or found (a diff, a grep, a quote from the log, a git ref), or is the first labelled message to relay such a finding. 0 for agreement, restatement, votes, and arguments from who the author is. A statement by the accused is not labelled at all: the accused is not an observer of its own guilt.

**Known weaknesses.**

- Excerpts, not full messages. A stance outside the excerpt window was missed.
- The evidence label depends on the claim's wording. For `d4:docs` the claim is that the documents *contain* sabotage, so "the commits exist" was marked as no evidence.
- Saboteurs' public statements are strategic. Sonnet 4.5 voting against others on day 7 is cover, not belief.
- The annotator knew the outcome of each day while labelling.
