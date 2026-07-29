# §8 acceptance re-bench — staged plan

**Status: STAGED, NOT LAUNCHED.** Nothing runs until the seat-orientation fix
(PR #937) merges and the owner says so.

This is the run that settles §10 and §11 of `docs/mcts_degradation_findings.md`.
Cluster-side artifacts (image, seed reservation, job specs, launcher) live in the
private deploy repo under `mcts/`; this file is the part that belongs in the
open: what is being tested, on what seeds, and what result would kill it.

---

## 1. The prediction, stated before the run

> **p1 stays where it is and p2 rises to meet it.**
>
> If p2 does not recover, **§10 and §11 both retract**, the way §5 did.

§11 measured, on the pre-fix build, a search seat gap of **+0.34 to +0.59** in
every deep or high-simulation cell, with p2 pinned at exactly 0.000 in three of
them and p1 flat-to-rising. The fix reflects the model's self-relative value once
at the seat boundary. If that was the cause, the gap collapses toward the
control's (±0.11, null). If the gap survives, the mechanism claim was wrong
regardless of how clean the audit looked.

Bracketing numbers for the acceptance cell (`d4-s1024` was never run as a search
cell pre-fix; its neighbours bound it):

| reference cell | p1 | p2 |
|---|---|---|
| control (raw v raw) | 0.550 | 0.500 |
| `d4-s2048` | 0.480 | 0.140 |
| `d6-s1024` | 0.460 | 0.260 |

## 2. Acceptance criterion (§8, unchanged)

> search ≥ 0.500 vs raw at `d4`/`s1024` over 200 mirrored-pair games, on a build
> whose SHA is recorded in the report.

Reported as: **per seat with Wilson 95%**, plus the pooled mirrored-**pair** mean
with a deterministic percentile bootstrap. Screening language only
(`mcts_eval.scoring.parity_label`) — never "parity achieved" at this n.

The pooled number alone is exactly what hid the defect for two campaigns. It is
reported second, never first.

## 3. What is structurally different from the grids

**Pairing is WITHIN-seed.** Each battle seed is played twice — once with search
seated p1, once seated p2 — so both seats face the *same two teams*.

`power_h2h.py`, which produced §4 and §9, derived the seat from seed parity
(`seed % 2`). Every n=50 arm therefore held two *disjoint* seed sets, and a
seat comparison was also a team comparison. That is §11.7's standing weakness,
and it is the one thing this run must not inherit.

Runner: `scripts/mcts_acceptance_h2h.py`. It emits
`pokezero.mcts_eval.scoring.GameResult` rows, so the in-house fail-closed merge
applies unchanged: a pair missing a seat is an error, never a silently
half-scored row.

Report: `scripts/mcts_acceptance_report.py`.

## 4. Arms

| arm | cell | pairs | games | seeds |
|---|---|---|---|---|
| search | `d4-s1024-b64-w4` | 220 | 440 | 7800000–7800219 |
| control | raw vs raw | 220 | 440 | **the same 7800000–7800219** |

220 rather than 200 so up to 20 pairs can be excluded and still clear the §8 bar.
Excluded pairs are replaced from a reserved spare band in a pre-registered order
(`mcts_eval.scoring.promote_spare_pairs`); a pair excluded from either arm is
excluded from both, so the arms always score the same pair set.

The control shares the search arm's seeds deliberately, and under within-seed
pairing it changes character in a way worth stating before the run:

* **Pooled, it is a strict equality test, not a noisy null.** Two identical
  deterministic policies playing the same seed from both seats produce the *same
  battle with the labels swapped*, so the control's pooled pair mean must be
  **exactly 0.500**. Any deviation is harness nondeterminism — something
  seat- or battle-id-dependent — and invalidates the pairing for both arms.
* **Per seat, it is the real payload.** The control's p1/p2 split measures how
  this seed set's *teams* split. That is the baseline the search arm's per-seat
  numbers must be read against: if these 220 seeds happen to favour the p1-slot
  team, search-as-p1 will look better than search-as-p2 for reasons that have
  nothing to do with value orientation. Within-seed pairing cancels that in the
  pooled number but not in the seat split — and the seat split is the whole
  point of this run.

Seeds are disjoint from everything already burned — the pre-fix grids
(600000–600099, 500000–500279), the fix-development bench seeds, and every prior
MCTS reservation. Both the new reservation and a retroactive record of the
previously unregistered blocks are filed in the private repo; they validate
against the in-house `assert_seed_ranges_are_unreserved`.

## 5. Build provenance

The measurement is only as good as the build, and a stale build does not error —
it produces a plausible number.

1. The image is built from the **public repo at the #937 merge commit**, so it is
   the branch exactly as merged.
2. The image stamps `scripts/engine_build_fingerprint.py --write` after **both**
   the crate and the `poke_engine` wheel are installed, which is that script's
   own stated contract. (It did not do this before; staging this run is what
   surfaced the gap. Without the stamp every shard would have died on "no build
   stamp — the installed engine's provenance is unknown".)
3. Every shard re-checks the fingerprint in-pod before its first game, and
   asserts it equals the value recorded in the staged config.
4. Every shard records the fingerprint in its output. The report **refuses to
   merge shards from two different builds** — a merged number across builds is
   not a measurement.

If review forces any change under `third_party/` or `rust/pokezero-search/src/`,
the fingerprint changes: rebuild the image and update the staged config in the
same commit.

## 6. Optional strength probe — not part of the acceptance claim

`d6-s4096`, 100 pairs, seeds 7810000–7810099. Owner-gated, to run **after** the
acceptance cell.

It is the most informative cell on record: the strongest p1 result (0.590) and
the deepest p2 collapse (0.000). If the fix is real this is where search should
show its largest gain. It is also ~36 GPU-hours, which is why it is a separate
decision and why its result may not be cited for or against the §8 bar.

## 7. Reading the result

| outcome | reading |
|---|---|
| p2 rises to p1's level, pooled > 0.500 | the §10/§11 mechanism holds; §8 met; the pre-fix grid numbers stay void |
| p2 rises but pooled ≤ 0.500 | mechanism holds, search still does not pay for itself; a new question, not a retraction |
| p2 does not rise | **§10 and §11 retract.** The inversion was real and the fix is still correct, but it was not the cause of the decay |
| p1 moves materially | something else changed; suspect the build, not the seat |

The residual §11.7 questions this run also settles: whether the correctly-oriented
seat carries a small real depth effect (p1 sat at 0.46–0.48 for d2/d6/d8 against
0.55–0.56 for control and d1, all overlapping at n=50), and whether the elevated
fallback in the largest cells matters.
