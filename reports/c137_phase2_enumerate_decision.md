# C137 — C116 Phase 2 decision: enumeration is the oracle, not the fix

**This document was rewritten after independent review, and the decision changed.** The
first version chose "adopt enumeration for the differential harness only" and, on that
basis, cancelled two engine fixes the program had already specced. That was wrong, and
the way it was wrong is instructive enough to keep on the record: §5.

**Decision: enumeration ships as a flag-gated reference oracle, used to validate the
collapsed path. The differential keeps measuring the shipping configuration. The engine
fixes in c133 §3 and c135 §5 are un-cancelled and should be implemented, with their
masses checked against the oracle.**

## 1. What the spike measured, and what it did not

All numbers re-derived from `spike_artifacts/`; the two demonstrations were re-run,
because they have **no committed artifact** — a gap that let a wrong number survive into
the first version of this document (§5).

**Enumeration is exact where the collapsed path is not.** Charizard Fire Blast into a
defender that survives every roll, reconstructed independently in Python:

| | branches | predicted damage × burn cells matched |
|---|---|---|
| collapsed | 5 | **0 of 64** — the fan collapses to a single arm at 145, and no legal roll deals 145 |
| enumerated | **65** | **64 of 64** to 1e-9 |

**A8's ordering defect is real and enumeration prices it correctly.** Defender 206/404,
burn tick 50, non-crit rolls 133–157; of 32 outcomes, 16 lethal on the hit, 1 lethal only
if the burn lands, 15 never lethal:

| | independent KO mass |
|---|---|
| truth | 5.810547 % |
| collapsed engine | **5.312500 %** — short by exactly the burn-dependent roll |
| enumerated | **5.810547 %, delta 0** |

**Residue, both windows, 200 games, one build, flag the only variable:** dev 2 → 0,
holdout 4 → 2, nothing opened, boundaries and gating counters identical, `engine_errors`
0 on all four.

**Throughput, and the correction that matters.** Measured at **depth 4 / 1024 sims**:
`midgame_3v3` 2.38 ms → 8,881.75 ms per decision, 3,732×, with 229× the leaf evaluations.
**Production is `search_depth: 2`, `search_sims: 256`** (`src/pokezero/engine_search.py`),
and enumeration is gated on `depth < DAMAGE_BRANCH_DEPTH = 2`
(`rust/pokezero-search/src/tree.rs`), so at production depth there is nothing *below* the
enumerated plies for the fan-out to multiply through. **The production-config regression
is unmeasured**, and is plausibly one to two orders of magnitude smaller. The first
version of this document called 8.9 s "the production config". It is not.

## 2. Why "adopt for the harness only" is the wrong decision

**It closes the rows in the instrument, not in the engine.** The four collapse-class rows
stop being *reported*. The crit-straddle gap and the A8 pre-secondary threshold read
remain in the engine that ships, at plies 1–2, which is exactly where they move KO pricing
at decision margins.

**And it stops the fidelity gate from testing the shipping path.** These are demonstrably
different engines. From the spike's own bench file, same position, same five seeds:

```
collapsed:  ember, ember, tackle, tackle, ember     (argmax unstable)
enumerated: tackle, tackle, tackle, tackle, tackle
```

A differential running enumerated certifies the configuration that plays `tackle` while
production plays `ember`. Every fidelity claim would then attest a code path production
never takes, and a regression in the collapsed damage-branch or residual-partition
surface — the surface that ships — becomes invisible to the 200-game gate.

**The artifact cannot even record which path ran.** The four sweep JSONs contain zero
occurrences of the flag, and `engine_fingerprint`, `source_commit` and `build_check` are
byte-identical between the on and off runs. A certification sweep would attest a
fingerprint that does not determine the behaviour it measured. That is a direct cost of
the single-build design the first version presented as an unalloyed virtue.

**"Zero throughput risk by construction" was backwards.** The property that makes the risk
zero is *default-off*, not *runtime env read*. A cargo feature would make search
enumeration impossible by construction; a process-global `OnceLock` read makes it possible
by accident. Worse, the patch disables the mirror **unconditionally**
(`residual_threshold_opt = if enumerate_damage_rolls() { None } else { … }`) while
enumerating only under the additional `branch_on_damage && fixed_damage.is_none()`. A
searching process with the flag set therefore gets, at every ply `depth >= 2`, a **third
configuration that has never been measured**: collapsed representative *and* no residual
partition.

## 3. The decision, and why it is better than either original option

The alternative the first version never argued against is the one the program already had
on its books:

- `reports/c133_collapsed_roll_disposition.md`: `19000074/27` is an **engine fix, ~15
  lines, mirroring existing code**; the A8 pair is an **engine fix — make the threshold
  status-aware**.
- `reports/c135_roll_divergent_lethality_adjudication.md` §5 gives the corrected recipe:
  one residual-kill arm per distinct threshold, priced at its own `tᵢ`, with
  **disjoint-band** masses — explicitly *not* the minimum over statuses, which c133 §4
  shows destroys an arm the engine emits today.

Those close the same four rows **in the shipping engine**, at the same zero throughput
cost, and they leave the differential measuring what production runs.

The obvious objection is that this family has already burned **three wrong hand-derived
mass recipes**, which is why C134 §3 froze it. That objection is now answerable, and this
is the spike's real contribution:

> **Enumeration is an exact oracle for the collapsed path's masses.** For any fixture,
> enumerate the fan and compare the collapsed arms' masses against the enumerated truth.
> A wrong recipe stops being something review has to catch by reading, and becomes a test
> that fails.

That is precisely what was missing when the three recipes were wrong. It is a better use
of the spike than replacing the measurement, because it makes the *engine* correct rather
than making the instrument stop noticing.

**So:** enumeration lands behind its default-off flag as a reference implementation,
consumed by tests. The differential is **not** switched to it. c133 §3 and c135 §5 are
un-cancelled, and each must be validated against the oracle and swept on both windows with
a registered "nothing opened" falsifier before it is believed.

## 4. What remains open

- **The engine fixes are not written.** This document only un-cancels them.
- **The enumerated path has no arm-structure or mass pin of its own.** If it is used as an
  oracle, it must be pinned, or a wrong oracle silently blesses a wrong recipe. The
  unpinned multi-hit semantic change — enumeration applies a per-hit roll shared across
  hits, replacing the collapsed path's total→per-hit conversion — is exactly what that
  hole would hide.
- **The f32 comparator (C116 M5) still executes in search**, and is untouched by any of
  this. Re-derived independently from the shipped expression over max_damage 1..400: 173
  max values where the top rung lands below `floor(max)`, **195 kill-count mismatches, 195
  undercounts, 0 overcounts**, 22 at interior thresholds; at max = 120 with threshold 108
  it counts 10 kill rolls against a true 11. Closing it needs C116(c)'s integer rewrite.
- **`counters.strict:sleeptalk_union_branch` moved 126 → 617 (dev) and 105 → 612
  (holdout)** between the two spike configurations. Probably branch multiplicity, but
  "identical trajectories" is asserted elsewhere and this is the one non-gating counter
  that moved fivefold. Unexplained.

## 5. How the first version of this document went wrong

Kept because the failure is more useful than the conclusion.

1. **It argued against a strawman.** It framed option (c) as "reject and buy nothing", so
   "nothing beats four rows at no throughput cost" followed trivially. The real
   alternative — implement the two specced engine fixes — closes the same rows in the
   shipping build, and the document then *cancelled those fixes* citing a measurement that
   never touched search.
2. **It said "at the production config" for a depth-4/1024 measurement.** Production is
   depth 2 / 256. The heading was honest; the sentence that rejected option (a) was not.
3. **It reported 34 branches for the enumerated Fire Blast fixture. The answer is 65.**
   34 is the *other* demonstration's branch count. It is self-refuting — 64 cells cannot
   be filled by 34 branches, since each branch maps to one cell — and it survived because
   the demonstrations have no committed artifact while the PR body claimed every figure
   came from one.
4. **The survivorship argument was misattributed and wrong.** C119 annotates every "no" by
   *class* — "it cannot change a move-legality predicate, a renderer tag, a
   boundary-pairing marker, or a missing mechanic" — never by tractability, and **11 of
   the 20 rows were already marked fixed in C119's own table**. The claim that C119's
   prediction was "conservative" is also false: it registered 2 holdout rows firmly and
   exactly 2 closed. That reading was reached by pooling dev and holdout into a
   holdout-scoped prediction.
5. **Smaller:** seven `residual-mass-*` probes, not six; only 3 of the 14 failures are
   branch-count assertions; and `19000191/63` was labelled with the superseded C111/C115
   "collapsed lethal arm" diagnosis that c133 explicitly corrected.

The through-line is that every one of these overstated the case for the conclusion the
document had already reached. The measurements were sound; the argument built on them was
not, and it was not adversarial against itself.
