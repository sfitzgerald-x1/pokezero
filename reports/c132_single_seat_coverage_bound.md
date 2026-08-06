# C132 — the differential measures ~87 % of boundaries, not the ~96.6 % it reports

An instrument bound, not a defect. Found while refuting candidate cause A12. It constrains what any
fidelity claim built on this differential can assert, so it is written down and pinned rather than
left implicit.

## 1. The two counters are disjoint

`scripts/engine_transition_differential.py`:

```python
if set(requested) == {"p1", "p2"}:
    counts["boundaries_full_round"] += 1
    prepared = _prepare_boundary(...)
else:
    counts["skip:single_seat_boundary"] += 1
```

A single-seat ply — one where only one side is asked to act, because the other is waiting — is
counted in `skip:single_seat_boundary` and **never** in `boundaries_full_round`. The two sets do not
overlap, and their sum is every boundary the sweep encountered.

`measured_fraction_of_full_rounds` therefore divides by `boundaries_full_round`, a denominator that
**excludes the single-seat population entirely**. It is not wrong for what it says; it is easy to read
as coverage, and it is not coverage.

## 2. The numbers

From the artifacts committed with C131:

| window | all boundaries | single-seat, never compared | measured | measured / all | reported `measured_fraction_of_full_rounds` |
|---|---|---|---|---|---|
| dev `19,000,000–199` | 17,710 | 1,742 (9.8 %) | 15,432 | **87.1 %** | 0.9664 |
| holdout `19,100,000–199` | 17,968 | 1,813 (10.1 %) | 15,551 | **86.5 %** | 0.9626 |

About 8.7 single-seat boundaries per game in dev, 9.1 in holdout.

> An earlier draft of this stated "10.9 % of full-round boundaries", dividing 1,742 by 15,968. That
> denominator is wrong — the sets are disjoint, so the ratio means nothing. Caught by reading the
> counting code rather than trusting the arithmetic.

## 3. What lives in the unmeasured population

Every **post-move-faint replacement ply**. When a Pokémon faints to a move, gen 3 pauses mid-turn for
the replacement; the surviving side gets `wait`, so the ply is single-seat and is skipped.

That matters because **the residual phase is deferred onto exactly those plies**. Measured by driving
Showdown directly in `gen3customgame`: the move-faint ply itself carries no residuals, and the ply
after the replacement carries the whole phase — weather upkeep, both actives' chips, items, status.
The engine defers it identically; its branch ends in `ToggleSide*ForceSwitch` and the next transition
carries the full list.

So the differential has never compared a deferred residual phase. Both simulators agree there as far
as the spot checks in `agents/reports/rust-fidelity/a12_candidate_residuals_skipped_on_move_faint.md`
go, but agreement on a handful of hand-built states is not the same as 400 games of measurement, and
the distinction should not be blurred.

This is also why a probe searching for a move-faint and a weather chip **in the same ply** found zero
hits across 180 games: by construction they are never in the same ply. The probe was fine; its premise
was wrong.

## 4. Why this is a bound and not a bug

A single-seat ply genuinely cannot be compared the way a full round is — there is no second action to
model, and the engine's transition for it is a different shape. Skipping it is the right call. What
was missing is the statement of what that costs, so a reader of a residue figure knows the residue is
over 87 % of boundaries and not all of them.

**Consequence for the program's terminal claim:** any statement of the form "N divergences over two
200-game windows" is a statement about full-round boundaries. It must either say so, or be
accompanied by this bound. It must not be phrased as though every boundary were compared.

## 5. Pin

`tests/test_single_seat_coverage_bound.py` asserts that both counters are present in any produced
report and that they are disjoint, so the denominator is always recoverable and this bound stays
computable. It does **not** pin the exact fraction — that legitimately drifts with the pool and the
seed window, and pinning it would produce a gate that fails for the wrong reason.
