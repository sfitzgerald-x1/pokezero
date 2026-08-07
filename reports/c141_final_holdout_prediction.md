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

# OUTCOME — appended after the single run, then corrected twice by review

Run once, on `19,200,060`–`19,200,259`, 200 games, build `44ee1430708cbb55` / 71 patches.
Artifact: `reports/artifacts/c141_final_holdout_sweep.json`. The run executed **at the
prediction commit** `3687d205` with `source_tree: clean`, so the pre-registration is
verifiable rather than asserted.

**Headline: 1 divergence and 1 withheld boundary in 16,274 measured boundaries, zero
engine errors.**

| | dev | holdout | **final holdout** |
|---|---|---|---|
| boundaries_measured | 15,503 | 15,579 | **16,274** |
| matched | 15,502 | 15,579 | **16,268** |
| reported diverged | 1 | 0 | **2** |
| **of which a rump artifact** | 0 | 0 | **1** |
| **genuine divergences** | 1 | 0 | **1** |
| all-branches-lossy | 0 | 0 | **4** |
| engine_errors | 0 | 0 | **0** |

## The falsifier did NOT fire, and my first outcome note said it did

I wrote that both rows were shapes the ledger does not contain. **Both halves were false.**

- `roll_scaled_component` has fired **5× on dev and 2–4× on the validation holdout** across
  nine committed artifacts. `c138` H15 already lists it among the six classes ever seen.
- `component_mismatch:` fired on the holdout as `component_mismatch:itemleftovers|leechseed`.

**`19200244/115` is ledger entry G8**, still open — the collapsed lethal Leech Seed drain,
`hp_after_move + leftovers < maxhp/8`. Verified: 11 + 25 = 36 < 50.875. It is the same
mechanism as dev row `19000191/63`, diagnosed in `c140` one commit before this sweep ran,
off by one in the same way (36 vs 37; 28 vs 29). The class string differs only because the
engine's arm labelled its component `itemleftovers` instead of `heal`.

**`19200131/129` is not a divergence at all.** Replaying the retained state recovers two
arms: the non-crit arm at **93.75 %** mass, carrying `recoil −19`, was **dropped** as
`attract_empty_tail_ambiguous:paralyzed+cannot_act`; the crit arm at 6.25 %, carrying
`−32`, survived; the observation is `−18`. The observation *is* the dropped arm.
Allowlisting that one marker turns the boundary into **matched**. The verdict rested on
one-sixteenth of the enumerated mass.

So the corrected reading of this window is **1 divergence + 1 withheld boundary**.

## The identity finding was already ledgered — as H14

`skip:strict_all_branches_lossy = 4`, and `matched + diverged == measured` fails here. But
this is **not** a discovery: `c138` H14 already states the correct form, and states it more
completely than I did, including a term I dropped:

> `boundaries_measured == matched + diverged + engine_error + skip:strict_all_branches_lossy`

`docs/engine_divergence_ledger_20260728.md` carries it as a standing rule, derived from a
live 40-row instance in closed PR #1037. This window is the **first live firing** of a
gap H14 had already classified reachable-in-principle. No report ever asserted the
two-term form beyond its own data, and no code guard on this harness covers it.

## Coverage, corrected

My first note said "~3.3 % of full-round boundaries are single-seat" and called it unchanged
from the pre-registration. Wrong by 3×, and it was a silent revision. Single-seat
boundaries are counted **before** `boundaries_full_round`, so they are not in that
denominator at all. The reconciliation from `c132`:

- total 16,837 + 1,767 = **18,604**; single-seat **9.5 %**; coverage **16,274 / 18,604 =
  87.5 %** — which *vindicates* the pre-registered ~87 %.
- within-full-round exits total 563: 205 struggle, 169 substitute-health-unknown, 141
  volatile-unsupported, 24 prestate-mismatch, 18 self-request-state, 4 materialization,
  2 encore-move-unknown. `16,837 − 563 = 16,274`. My first list omitted the 169 and the 24
  and therefore did not reconcile — the exact omission `c132` warns about by name.
- `gating_support_based` 1,284 / 16,274 = **7.9 %** accepted through the enumerated
  sleep-counter widening (Constraint 7). That figure was right.

## What this measurement establishes

**One genuine divergence in 16,274 measured boundaries on an unbiased window nothing was
tuned against**, and it is an instance of a gap already diagnosed and already open in the
ledger. Zero engine errors.

My first outcome note concluded "the residue was not understood, it was exhausted on two
windows." That was over-correction in the pessimistic direction and it was wrong on the
facts. The residue looks **more** understood than that: the heal/Leech-Seed family
predicted the one real row, and the other row was never a divergence.

What the window did surface, and could not have been surfaced otherwise, is
**instrumentation**: the first live firing of H14, and a rump-branch adjudication that had
been documented as possible and never seen in a reported result. Both are now measurable.

## What it cannot say

- ~7.9 % of measured boundaries accepted through the sleep-counter widening (Constraint 7).
- Coverage is 87.5 %, not 100 % — 1,767 single-seat boundaries and 563 in-path exits are
  outside the measured set.
- The double-faint terminal-value tie is invisible to every counter here.
- The true rump-boundary count on this window is **unknowable**: `strict:lossy_render = 14`
  counts branch drops, not boundaries, so it is somewhere in 1–14, and the measurement that
  would settle it is the forbidden one.

## The window is spent

`19,200,060`–`19,200,259` must never be measured again; `19,200,000`–`19,200,059` was
contaminated earlier and is also spent. Above `19,200,260` is the only clean reserve, and
#1122's guard still refuses it without an explicit opt-in.

**Neither row may be fixed against this window.** Both were diagnosed on generated
boundaries and retained-state replays, which is not a re-measurement.
