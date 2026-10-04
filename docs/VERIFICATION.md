# What the verification layer establishes

This is a finite-trace policy checker plus three symbolic obligations for the
betting kernel. It does **not** formally verify a swarm, an annotation model, or
the statistical assumptions behind a reported e-value.

## Annotation-policy review

`martingale_audit.verification.verify_trace(observations)` takes the existing
`Observation` records and returns a JSON-serializable report without writing
files. The default policy treats beliefs at or above 0.6 as supporting a claim
and beliefs at or below 0.4 as opposing it. An intermediate belief resets the
current stance. First stances, stance changes, and returning to a stance after
a neutral statement are adoption events; repetitions are not.

An adoption passes the policy if any earlier observation for that claim is
annotated as evidence, or the adopting observation itself is annotated as
evidence. The evidence threshold defaults to 0.5. This is an **availability**
policy: legacy labels do not say whether evidence supports a particular side,
whether an agent saw it, or whether it is valid. One piece of earlier evidence
can satisfy the policy for all later adoptions on that claim. This deliberately
weak policy is useful for finding auditable gaps, not for certifying reasoning.

Each unsupported adoption is a **review candidate relative to the supplied
annotations**. Its witness contains source message IDs, original input indices,
claim, agent, timestamp, belief, and evidence labels. The checked prefix indices
identify the observations over which absence of earlier evidence was checked.
Evidence omitted by selection or annotation remains unknown. In particular,
the pilot's keyword-centred excerpts cannot establish that the real transcript
contained no evidence.

Tied timestamps are unordered. Evidence from another message at the same
timestamp makes prior availability inconclusive, independent of input order.
Multiple statements by the same agent at one timestamp also leave its stance
order unresolved. The checker carries the possible previous sides forward;
it does not arbitrarily choose one. Missing IDs, duplicate claim/message pairs,
conflicting source metadata, and invalid numbers are reported separately.
Identical duplicates are collapsed only for policy checking. Conflicting or
invalid records make the affected claim inconclusive. A message ID appearing
on several different claims is permitted when its author and timestamp match.

The report's `summary` includes `unsupported_adoptions`, `ambiguous_adoptions`,
`integrity_issues`, `ordering_ambiguities`, and `uncheckable_claims`. The status
`satisfied_on_supplied_trace` means this particular evidence-availability policy
has no identified counterexample on the supplied labels; it is not a quality,
truth, safety, or statistical certificate. Integrity warnings remain separate.

## Symbolic betting-kernel checks

Install the optional solver and run:

```sh
python -m pip install z3-solver
python examples/verify_kernel.py
```

The script prints JSON and succeeds only if Z3 reports `unsat` for the negation
of all three exact-real obligations and `sat` for both deliberate mutants:

1. If `0 <= lambda <= 1` and `-1 <= x <= 1`, then `1 + lambda*x >= 0`.
2. With the same bounds and `W >= 0`, updated wealth `W*(1 + lambda*x) >= 0`.
3. If also the conditional mean `mu` is in `[-1, 0]`, then
   `W*(1 + lambda*mu) <= W`.

These are symbolic algebraic proofs over exact real variables, not enumeration
of a bounded sample grid. The third obligation represents the algebraic step
in a supermartingale argument: replacing `x` by its conditional mean requires
wealth and the stake to be fixed given the past. The script assumes this
predictability; it does not derive it from the Python program or the transcript.
It does not encode conditional-expectation measure theory or prove Ville's
inequality.

The mutation checks produce concrete rational counterexamples. Allowing a stake
up to 2 can make wealth negative. Choosing a stake of 1/2 after observing a
positive fair-coin outcome and 0 after observing a negative one raises expected
wealth by a factor of 1.25, despite zero-mean outcomes and bounded stakes. The
second example shows why the timing assumption is essential.

The solver checks mathematical formulas, **not a refinement proof of
`EProcess`**, its stake rule, floating-point arithmetic, extraction accuracy,
data selection, conditional null, aggregation, or downstream decisions.
Without `z3-solver` the report says `unavailable` and the script exits nonzero;
it never substitutes simulation for a proof. The Python trace checker has no
solver dependency.
