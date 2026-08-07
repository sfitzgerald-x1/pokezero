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

---

# OUTCOME — appended after the single run. The falsifier fired.

Run once, on `19,200,060`–`19,200,259`, 200 games, build `44ee1430708cbb55` / 71 patches,
`main` `16857e06`. Artifact: `reports/artifacts/c141_final_holdout_sweep.json`. Exit 1,
which is the differential's by-design code when divergences exist.

| | dev | holdout | **final holdout** |
|---|---|---|---|
| boundaries_measured | 15,503 | 15,579 | **16,274** |
| boundaries_full_round | 15,968 | 16,155 | **16,837** |
| matched | 15,502 | 15,579 | **16,268** |
| **diverged** | 1 | 0 | **2** |
| all-branches-lossy | 0 | 0 | **4** |
| lossy_render | 0 | 3 | **14** |
| engine_errors | 0 | 0 | **0** |

**The count landed inside the predicted 0–3. Everything else about the prediction was
wrong, and in the direction that matters.**

## The falsifier fired on shape, not on count

The registered falsifier read: *"if the count is large — say above 5 — or if any row is a
shape the ledger does not already contain, the program's claim that the residue is
understood is wrong."*

Both rows are shapes the ledger does not contain, and both classes are strings neither
development window ever emitted.

**`19200131/129`** — `roll_scaled_component`, `p1: doubleedge / p2: batonpass`, gating
exact, 2 branches:

```
pct=6.25: p1 roll-scaled components differ: observed=[('recoil', -18)] engine=[('recoil', -32)]
```

Recoil. Not a residual, not a heal label, not a partition artifact — a **recoil magnitude**
disagreement, on a class that has never appeared in this program's residue.

**`19200244/115`** — `component_mismatch:heal|itemleftovers`, `p1: flamethrower /
p2: fireblast`, gating exact, 9 branches:

```
pct=49.03: p1 attributed components differ: observed_only=[('heal', 36)] engine_only=[('itemleftovers', 37)]
pct=10.74: p1 attributed components differ: observed_only=[('heal', 36)] engine_only=[]
```

An unattributed heal of 36 against a Leftovers tick of 37 — an **attribution** mismatch
with a magnitude disagreement inside it, and again a class string new to the residue.

## A second, quieter finding: four boundaries were adjudicated as neither

`skip:strict_all_branches_lossy = 4`. Those boundaries are counted in
`boundaries_measured` but are neither matched nor diverged, which is why
`matched + diverged == measured` — an identity this program has asserted repeatedly, and
which I asserted throughout this session — **does not hold here**. The correct identity is
`matched + diverged + all_branches_lossy == measured` (16,268 + 2 + 4 = 16,274), and it
held on dev and holdout only because that counter was zero on both.

`strict:lossy_render` is 14 here against 3 on holdout and 0 on dev. So the unseen window
exercises renderer paths the development windows do not, and four boundaries could not be
adjudicated at all. That population is invisible in a divergence count.

## What this measurement establishes

The engine is close on an unbiased sample nothing was tuned against: **2 divergences in
16,274 measured boundaries, zero engine errors.** That is a real result and it is the
number this window was reserved to produce.

It also establishes that **the residue was not understood** — it was exhausted on two
windows. Six of the seven rows this program closed were closed after being seen, and the
first window that had never been seen produced two shapes in classes that had never
appeared, plus a four-boundary adjudication hole that had never appeared. The prediction
argued for exactly this and then guessed the wrong specifics; the count was right for the
wrong reason.

## What it cannot say, unchanged from the pre-registration

- **~9% of measured boundaries are accepted through a widened bar** (Constraint 7):
  `gating_support_based` is 1,284 of 16,274 here, 7.9%.
- **~3.3% of full-round boundaries are single-seat and skipped** (1,767 of 16,837), plus
  205 struggle, 141 volatile-unsupported, 18 self-request-state, 4 materialization and 2
  encore-move-unknown exits.
- **The double-faint terminal-value tie is invisible to every counter above.**

## The window is now spent

`19,200,060`–`19,200,259` has been measured and must never be measured again. Seeds
`19,200,000`–`19,200,059` were contaminated earlier and are also spent. The remainder of
the reserved range above `19,200,260` is untouched and is the only clean reserve left; the
#1122 guard still refuses it without an explicit opt-in.

**Neither of the two rows should be fixed against this window.** Diagnose them on
generated boundaries or on new dev seeds. Fixing against the final holdout is exactly the
fitting this reservation exists to prevent, and it would spend the reserve twice.
