# C119 — Phase 2 scoping: what enumerate-then-merge would and would not absorb

C116 Phase 2 is "one spike, one measurement" followed by a three-way decision. This is the
input that decision needs and does not yet have: **how many residue rows the change actually
reaches.** No code, no flag, no spike — the spike is item 6 and this is what should be true
before it is written.

Era: `main` `48468b67`. Residue **dev 6 / holdout 14**, both after B1's two halves.

## 1. The claim being scoped

C116 §3 argues the partition cascade has a closed form: evaluate each of the 16 rolls through
the move and the ordered residual phase, produce an observable vector per roll, merge rolls
with identical vectors. It lists what that deletes — A4, A7, the residual mirror and its
bisection, the f32 comparator, most of the mass-leak surface — and prices it at ~32
ordered-phase evaluations per fan boundary against ~10 bisection evaluations today.

**The argument is sound and I am not disputing it.** What §3 does not say, and what reads as
though it does, is how much of the *current* residue it retires. Answer: **about a fifth.**

## 2. The split, derived from the corrected C117 filings

Enumerate-then-merge abolishes the **collapse tax** — a fan collapsed before its consequences
are computed. It cannot change a move-legality predicate, a renderer tag, a boundary-pairing
marker, or a missing mechanic.

| holdout rows | cause | absorbed by (a)? |
|---|---|---|
| 2 | Pain Split collapse tax — the amount is a function of the arm's representative roll | **yes** |
| 2 | A8, the residual mirror reading status pre-move | **yes** — a per-roll evaluator runs the phase *after* the move, so the secondary is visible by construction |
| 1 | A9, a planned Wish heal never rendered | **yes**, if the evaluator renders what it simulates |
| 11 | B1 — renderer tag | no *(already fixed, #1081/#1086)* |
| 3 | A1 — faint/forced-switch residual **placement** | no — a harness pairing question |
| 2 | A10 — Belly Drum pays `maxhp/2` at +6 | no — a move-legality predicate |
| 2 | leechseed-for-Leftovers renderer mis-tag | no |
| 1 | `19100180/24` unowned side-condition divergence | unknown |

**5 of 25 absorbed, 20 untouched** — and of the 14 rows still divergent on the holdout,
**5 absorbed / 9 untouched.** On the dev window's 6, by the same filings: the 2 A1 rows, A4,
A5, A7 and A6 — so **1 or 2 absorbed at most** (A4 and A7 are collapse-class; A1, A5, A6 are
not).

I am stating this at its true strength rather than its most useful one, because the previous
version of this program's arithmetic was corrected three times for the opposite habit. The
reviewer of C117 independently derived the same 5/20 and asked me to re-derive it before it
entered a document; the numbers above come from the corrected §4 table, not from that message.

## 3. What that does and does not imply

**It does not weaken the case for outcome (a).** The plan's argument was never "this closes
the residue" — it was that the *cascade* has a closed form and that each hand-built partition
adds the next one's surface area. That argument is about **future** cost, and it is
independently supported: A4 and A7 are queued instances, and A8 is a *new* instance that
appeared in a window nobody had swept. Three of the five absorbed rows were discovered after
§3 was written.

**It does mean the decision cannot be justified by residue count.** If outcome (a) is taken
expecting the queue to empty, it will look like a failure at 20 untouched rows. The
justification has to be the one §3 actually makes — abolishing the tax so the next observable
costs an arm-merge rather than a partition — plus M5, the f32 comparator, which is a
sweep-invisible defect the enumeration deletes outright.

**M5 is the strongest single argument and it is independent of row count.** The comparator
mis-counts kill rolls for 195 `(max, threshold)` pairs in the audited range, 22 at interior
thresholds, one-directional (always undercounting), across five call sites. Enumeration
replaces it with exact integer counting. All three of the plan's outcomes are required to
close M5, so that part is not a decision at all.

## 4. Obligations the decision record must carry

Registered here so they are not rediscovered later:

1. **If the partition stack is RETAINED for any consumer** (outcomes (b) or (c)), the
   crit-fan residual split, `fixed_damage` and multi-hit get mass-gate fixtures before the
   decision is recorded as closed. They are currently uncovered and were deferred *only*
   because outcome (a) deletes those paths. Deferring them under (b) or (c) is not defensible.
2. **The bail set is unreachable by the current mass gate's design** — a scalar quiet-turn
   tick cannot represent Sitrus's non-monotone threshold heal. Covering it needs a different
   reconstruction, not another fixture.
3. **Throughput must be measured at the production config** (d4-s1024 decisions/sec), not
   inferred from the per-evaluation cost. The plan already says this; it is repeated because
   §3's "~32 vs ~10 evaluations" is a *cost model*, not a measurement.
4. **The spike must be behind a flag**, so the two engines can be swept on the same tree — the
   only way to get a single-variable comparison on a change this size.

## 5. Recommendation

Write the spike, and pre-register the prediction as **5 of 25 holdout rows and 1–2 of 6 dev
rows**, with M5 closed. If the spike closes materially more than that, the extra rows are
evidence my filings are wrong somewhere and should be replayed rather than celebrated. If it
closes fewer, the collapse-class attributions are wrong.

That is the point of writing the number down first.
