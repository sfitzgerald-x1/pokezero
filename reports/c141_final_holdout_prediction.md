# C141 — final-holdout prediction, registered before the sweep

C116 Phase 4 item 13: the single measurement on the reserved final holdout. This document
is committed **before** the sweep runs, and the sweep is run **once**. There is no second
attempt, so a prediction registered afterwards would be worthless.

## The window, and why it is not the reserved floor

The reservation begins at seed **19,200,000** and is enforced by the guard added in #1122,
which checks the whole requested span and every seed present in the records, and refuses
without `--final-holdout-i-mean-it`.

I contaminated the first 60 seeds of that window earlier in this program: a convenience
shell loop executed 60 games of it before the guard existed. That is disclosed in
`reports/rust-fidelity/final_holdout_contamination_disclosure.md` (external) and the JSON
was deleted unread. So the measurement window is **`19,200,060`–`19,200,259`**, disjoint
from the contaminated 60, chosen by me rather than deferred, and 200 games like the other
two windows.

## The build

`main` `16857e06`, engine fingerprint **`44ee1430708cbb55`, 71 patches** — verified by
`engine_build_fingerprint.py --check` immediately before the run. This is the same
fingerprint the dev and holdout measurements carry, and #1155 was documentation-only, so
no engine input has moved since.

## State on the two open windows, on this build

| window | diverged | rows |
|---|---|---|
| dev `19,000,000–199` | **1** | `19000191/63` |
| holdout `19,100,000–199` | **0** | — |

## Prediction

**A small non-zero count, most likely 0–3, dominated by shapes already in the ledger.**

Reasoning, stated so it can be checked against the outcome rather than reinterpreted after
it:

- Dev and holdout are 1 and 0 on this build, so the per-window rate is now well under one
  row per 200 games. A third window drawn from the same distribution should look similar.
- The final holdout has never been swept, so **every latent shape gets its first
  opportunity here.** Six of the seven rows this program closed were closed after being
  seen; a window that has never been seen has had no such attrition. That argues for
  *more* than zero, not less.
- The known-gaps ledger (`c138`) lists reachable-but-unobserved shapes — Farfetch'd's
  Stick, the double-faint tie, PP above 10, the Toxic min-1 clamp — any of which could
  surface here for the first time.

**I am not predicting zero.** A zero would be a pleasant result and I would report it, but
predicting it would be predicting the flattering outcome on a window selected precisely
because nothing has been fitted to it.

## What this measurement can and cannot say

It is one unbiased sample of the real distribution on a window nothing was tuned against.
That is its whole value, and it is why it is spent once.

It **cannot** say the engine matches Showdown everywhere, and three standing caveats from
`c138` apply to this number exactly as they do to the others:

- **~9% of measured boundaries are accepted through a widened bar** — an underivable sleep
  counter is neither skipped nor guessed but enumerated over up to 64 worlds, accepted if
  any one matches (Constraint 7). Monotone toward matching.
- **The simultaneous last-mon double faint is a terminal-value divergence** that no
  differential counter can see; it corrupts search value rather than fidelity reporting.
- **~13% of boundaries are single-seat** and skipped before comparison, so the coverage
  denominator is ~87%, not 100% (`c132`).

## Falsifier

**If the count is large — say above 5 — or if any row is a shape the ledger does not
already contain, the program's claim that the residue is understood is wrong**, and that
is the outcome worth learning. A new shape here is more informative than a clean sweep,
because it would mean the two development windows had been exhausted rather than the
mechanism understood.

The run will be reported exactly as it comes out, including the seed and step of every
divergence, whatever the number.
