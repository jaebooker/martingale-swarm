# Second dataset: the German wiki relay incident

Source: the `full-wiki-logs` export from collusion.wiki (14,591 revisions, 4,579 pages, 3,103 account names, 24 May to 2 July 2026). The data is not redistributed here. Reproduce with:

```bash
python examples/wiki_incident.py /path/to/full-wiki-logs
```

Everything below is rule-based: regular expressions over revision text, no model in the loop and no hand labels. Treat the counts as approximate.

## What the logs show

AI agents working on timed, multi-round data questions wrote to open German wikis. They used the pages for two things.

**As a relay.** Agents that could not open a data URL directly wrote it into a wiki page wrapped in a third-party proxy or markdown-conversion host, then followed the link. Seventeen such hosts appear.

**As a message board between parallel runs.** Runs of the same task, which call themselves "cohorts", found each other on shared pages. Cohorts further along posted each round's question and answer. Cohorts behind them acknowledged and said they would use it. The messages are signed and timestamped, and many state where a value came from.

The second use is the reason this export suits the audit better than the saboteur game: thousands of signed assertions of specific values, with corrections.

## Finding 1: a third of technique adoption is visible copying

For each account's first use of a relay host, the script checks whether the line was already on the wiki under another account.

- 819 of 2,427 first uses (34%) were copied verbatim.
- 1,608 wrote the line fresh.
- Two hosts spread almost entirely by copying: `webcrawlerapi.com` (52 of 57) and `md.dhr.wtf` (46 of 53).
- One spread with almost none: `api.counterapi.dev` (1 of 78), a public counter that cohorts used as a signal channel.

34% is a floor on transmission through the wiki. A fresh line does not show the author had not read an earlier one. Account names are throwaway (1,332 of 3,103 appear once), so "another account" is sometimes the same run under a new name, which pushes the other way.

## Finding 2: a wrong value spread on testimony and was reversed on evidence

In the `oecd-equity` task, cohorts were asked four values in turn. An early cohort read them from a downloadable workbook that stores one decimal, and posted 9.70, 9.90, 16.40 and 14.60. Later cohorts repeated those as the expected answers for rounds they had not yet reached.

The workbook values were first posted on 17 June and repeated heavily from late on 19 June. About five hours into that busy stretch, cohorts that rendered the live chart posted two-decimal values (9.69, 9.91, 16.38, 14.59) with their method. One of them called the earlier values "the workbook-display herd".

| | count |
|---|---:|
| Assertions of either value, 4 values | 846 |
| Signers | 122 |
| First stances on a workbook value | 272 |
| ...that repeated a peer's value and cited no source | 208 (76%) |
| Switches to the chart value | 107 |
| ...with no evidence posted since the signer last spoke | 0 |
| ...where the switching message describes the signer's own check | 61 |

Among signers who held the workbook value for the fourth answer and were still posting after the correction, 15 of 15 moved. For the third answer, 14 of 14.

The audit's numbers match the pilot's shape. With the evidence gate off there are 114 moves toward peers and 0 away. With it on there are no moves at all: every change of position came in an interval where someone had posted evidence. The swarm-level e-value is 1.

So this swarm adopted on a peer's word and corrected on evidence. The adoption was fast and nearly universal. So was the correction.

## What this adds to the pilot

- **The same two-part pattern in unrelated data.** First positions taken on testimony (76% here, 59% in the saboteur game), changes of position driven by evidence.
- **The error was a shared-source error.** The wrong values came from one workbook that everyone downstream inherited. That is the same mechanism as the saboteur game's false alarms, where several agents ran the same misleading check.
- **The betting test stays silent, correctly.** There is no evidence-free movement to detect. The first-stance count and the trace-policy check are the instruments that see what happened.

## Limits

- **Signer is not agent.** A cohort name is a self-chosen signature. Nothing ties it to one model instance.
- **The evidence flag is a keyword match.** It fires on words such as "verified", "rendered", "tooltip" and "workbook". A message that relays someone else's check also matches. The "own check" count uses a stricter first-person pattern and is still a regex.
- **Value matching is by digits.** The earliest "chart value" hits for two of the four answers come hours before the correction posts and are probably raw six-digit values quoted in passing. The before-and-after split for those two is less clean.
- **Reads are invisible.** The export holds writes. Who had read what is unknown, as in every dataset so far.
- **No ground truth for the task.** The chart values are what later cohorts could reproduce. Whether the task's answer key expected one or two decimals is not in the export.
- **The trace-policy check is noisy here.** Of 425 adoptions it marks 169 ambiguous, because many messages share a timestamp at one-second resolution.
