# C138 — the two collapse-class engine fixes, registered before any post-fix measurement

Registered 2026-08-06, before either patch was built or installed. The baseline in §2 was
re-derived from the base commit on this branch's own build; no figure here is carried
forward from another document.

These are the two fixes `reports/c137_phase2_enumerate_decision.md` un-cancelled:
`reports/c133_collapsed_roll_disposition.md` §3 (the crit-straddle sub-split) and
`reports/c135_roll_divergent_lethality_adjudication.md` §5 (the status-aware residual
threshold). They ship as two commits on one branch — see §6 for why one branch and not
two PRs.

## 1. What each fix is

**(A) The crit-straddle residual sub-split.** When the CRIT fan straddles the KO threshold,
Case B partitions it into crit-kill and crit-survive and never consults the residual
threshold. The residual sub-split therefore existed at two of the three partition sites and
was missing from the third. Dev row `19000074/27`: the priced crit fan is
`[214, 216, 219, 221, 224, 226, 229, 231, 234, 236, 239, 241, 244, 246, 249, 252]` against
244 HP with a sandstorm threshold of 229. Showdown rolled **241 — roll 96, a member of the
engine's own fan** — while the engine emitted arms only at 244 (the defender's HP) and 227
(the mean of the twelve non-KO rolls, not a fan member at all).

**(B) The status-aware residual threshold, and the disjoint-band mass rule.**
`residual_lethality_threshold` reads the defender's PRE-MOVE state, so a Fire move's own
burn secondary is invisible and the split never fires. Holdout rows `19100107/135` (Sacred
Fire into Roselia 175/254) and `19100191/5` (Fire Blast into Ninjask 225/225): both
defenders unstatused, the mirror declines, the whole non-crit fan collapses onto a surviving
representative — and the observed roll is again a member of the engine's own fan.

The fix passes `choice` in so secondaries are visible, and takes the **union** of the
distinct thresholds. Explicitly **not** the minimum: c133 §4 records a measured
counterexample where `min` drops the threshold below the fan's lowest roll, every
`min_roll < threshold` guard stops firing, and an arm the engine emits today is destroyed.

Because the thresholds are **nested** (a status only adds a tick, so every status-aware
threshold is at or below the pre-move one), each arm carries the **disjoint band**
`#{rolls in [t_i, t_i+1)}` priced at its own `t_i`, the survive arm keeps
`average_surviving_damage` over the rolls below `t_1`, and — this is the part c135 §5 states
incompletely — **the KO threshold participates in the same nest as the ceiling wherever a
KO arm already exists.** At Case A and at the crit-straddle site the top residual band is
therefore `#{rolls in [t_k, hp)}`, not `#{rolls >= t_k}`; the literal reading of c135 §5
would let the top arm absorb the hit-lethal rolls and double-count them against the KO arm.
c133 §4 is primary here: it names the engine's existing
`num_residual_only = num_at_or_above - num_kill_rolls` as "exactly the band between the
residual threshold and the KO threshold", and the fix generalises that subtraction rather
than replacing it. With a one-entry ladder every site reduces to the arithmetic it ran
before.

## 2. Baseline, re-derived from the base commit

`origin/main` `2ec0cb13`, fingerprint-verified (`68 patches, 907bea70abd1bf86`), one build,
200 games per window. Artifacts:
`reports/artifacts/c138_collapsefix_main_{dev,holdout}_sweep.json`.

| window | full_round | measured | matched | diverged | engine_errors |
|---|---|---|---|---|---|
| dev `19,000,000–199` | 15,968 | 15,503 | 15,501 | **2** | 0 |
| holdout `19,100,000–199` | 16,155 | 15,579 | 15,575 | **4** | 0 |

Complete `divergence_classes` census (not read off `repros`, which is capped at
`keep_repro=25`):

| window | class | count | rows |
|---|---|---|---|
| dev | `component_magnitude:heal` | 1 | `19000191/63` |
| dev | `component_missing_in_engine:sandstorm` | 1 | **`19000074/27`** |
| holdout | `component_missing_in_engine:itemleftovers` | 2 | `19100170/71`, `19100170/72` |
| holdout | `limit:roll_divergent_lethality` | 2 | **`19100107/135`**, **`19100191/5`** |

Holdout reads 4, not the 5 c133 §7 cites, because `19100180/24` closed in between
(#1144). That is a difference in the baseline, not in this measurement.

## 3. Prediction

**After (A) alone:**

| window | predicted |
|---|---|
| dev | **2 → 1**, closing exactly `19000074/27`, the lone `component_missing_in_engine:sandstorm` |
| holdout | **4 → 4, unchanged** — no holdout row is the crit-straddle mechanism |

**After (A) + (B):**

| window | predicted |
|---|---|
| dev | **1 → 1, unchanged** |
| holdout | **4 → 2**, closing exactly the two `limit:roll_divergent_lethality` rows |

Also predicted, on all four post-fix runs: `boundaries_measured` and
`boundaries_full_round` unchanged at 15,503 / 15,968 (dev) and 15,579 / 16,155 (holdout);
`engine_errors: 0`; the identity `matched + diverged == measured` holding.

Rows that must **survive** every run, because none of them is either mechanism:

- `19000191/63` — c133 §3 disposes of it as "the arm exists; its representative mis-prices a
  roll-dependent drain", needing enumeration or a matcher change. Neither fix touches it.
- `19100170/71` and `19100170/72` — `component_missing_in_engine:itemleftovers`, not a
  partition defect.

## 4. Falsifier

**If anything opens on either window, or either boundary count changes, or a predicted row
survives, or a row closes that is not named above, the fix that produced the run is
withdrawn.** This clause has caught two consecutive patches in this program that every other
gate passed, and a third was caught only because a reviewer built a shape no sweep
contained. A withdrawn fix with a clear diagnosis is an acceptable outcome; a believed fix
without this measurement is not.

Note the asymmetry that makes "nothing opened" the load-bearing half. Both fixes **re-price
survive representatives** — c133 §7 names three, `227 → 220` on `19000074/27`,
`203 → 197` on `19100191/5` and `157 → 150` on `19100107/135` — and the comparator's
fallback acceptance window `[0.92·eng − 1, 1.09·eng + 1]` is not invariant under that
re-pricing. How much currently-matched mass rides on that fallback is unmeasured
(c135 §6). The sweep is the only instrument that sees it.

## 5. The oracle, and why it is registered here too

This family has burned **three wrong hand-derived mass recipes**, and the reason C134 §3
froze it. What is different now is that an exact oracle exists: the enumerate-then-merge
spike patch emits one arm per distinct `floor(max * r / 100)` for `r` in `85..=100` at mass
1/16, resolves lethality inside `run_move` rather than in a mirror, and was verified against
an independent reconstruction (Fire Blast, 64 of 64 damage × burn cells to 1e-9; an A8
KO-mass fixture at 5.810547 % against independent truth 5.810547 %, delta 0, where the
collapsed path gives 5.312500 %).

So a fourth wrong recipe should be a **failing test**, not something review has to catch by
reading. `tests/test_collapsed_arm_mass_oracle.py` is that test.

**The functional is outcome mass, deliberately.** A correct collapsed path *cannot* agree
with enumerated truth arm-for-arm — that is what collapsing means. The comparison is a
coarsening: for each fixture, the total probability mass landing on each
`(defender faints?, defender's end status)` cell. That is the functional the spike's own A8
demonstration uses, and the one on which the disjoint-band rule is exact. Nobody should
later "strengthen" it into an arm-for-arm comparison, which can never pass.

**The oracle cannot be toggled in-process.** `ENUMERATE_DAMAGE_ROLLS` is a `OnceLock`
initialised from `std::env::var` on first call, so one process is one engine, permanently.
The enumerated truth is therefore produced in a **separate process against a separate
build** and committed as a pinned artifact (`tests/data/collapsed_arm_mass_oracle.json`),
which also answers c137 §4's objection that an unpinned oracle can silently bless a wrong
recipe. The test then checks three things against each other: the shipping engine's
functional, the pinned enumerated functional, and a pure-Python reconstruction that shares
no partition arithmetic with either.

Registered prediction for the oracle test, so it is falsifiable too: on the base commit the
crit-straddle fixture and both status-aware fixtures are **RED**, and the
`min-would-destroy-an-arm` fixture is **GREEN** — it is a control that only a
minimum-over-statuses implementation breaks.

## 6. Two commits, one branch, one PR

The two fixes are not cleanly separable into two PRs. (B) does not add a threshold beside
(A)'s; it replaces the single `residual_threshold_opt` that all four partition sites read
with a ladder, and rewrites (A)'s own emission in the same edit. A second PR would therefore
carry a diff against code that only exists in the first, and the patch stack is
append-only and ordered, so the two patch files must land in a fixed order anyway.

They are still **two commits**, each with its own clean-room digests and its own sweep pair,
so the sweep attributes each closure to the fix that produced it. c133 §7 is the reason
attribution matters: the last engine fix in this residue had a correct mechanism, a verified
Showdown citation, a red-on-main pin and green unit gates, and still opened 38 dev / 40
holdout rows against its single closure.
