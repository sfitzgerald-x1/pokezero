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
overlap, and their sum is every boundary that reached the counting fork. (A boundary aborted earlier
for `abort:no_legal_action` is in neither; it is 0 in both windows, so the totals below are exact.)

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

**The accounting reconciles exactly**, which is what makes 17,710 the real total rather than two
counters picked out of many:

| window | measured | + exits inside the full-round path | = full_round | + single-seat | = all |
|---|---|---|---|---|---|
| dev | 15,432 | 536 | 15,968 | 1,742 | **17,710** |
| holdout | 15,551 | 604 | 16,155 | 1,813 | **17,968** |

The exits **observed in these windows** are `skip:world_unsupported:*`, `skip:unmappable_choice:*`,
`world_prestate_mismatch` and `limit:world_substitute_health_unknown`. Others exist and are 0 here
(`skip:no_materialization:*`, `skip:no_action_candidates`, `skip:world_error:*`);
`skip:strict_all_branches_lossy` is **not** an exit — it fires after `boundaries_measured` has already
incremented. Two traps to avoid when re-deriving this:

- `world_prestate_mismatch`'s four `:p1_hp` / `:p1_status` / `:p2_hp` / `:p2_status` sub-counters **sum
  to the parent** (39 dev, 68 holdout), so adding parent and children double-counts.
- `limit:world_substitute_health_unknown` reads like an annotation but is a genuine exit; omit it and
  the reconciliation misses by exactly its count.

Summing `boundaries_measured` plus **every** `skip:*` counter gives 17,544 against a `full_round` of
15,968 — which does not reconcile, and is what first made me doubt the total. (The `skip:*` counters
alone sum to 2,112; an earlier draft of this sentence said 17,544 was the `skip:*` sum, omitting the
`measured +`.)

> An earlier draft of this stated "10.9 % of full-round boundaries", dividing 1,742 by 15,968. That
> denominator is wrong — the sets are disjoint, so the ratio means nothing. Caught by reading the
> counting code rather than trusting the arithmetic.

## 3. What lives in the unmeasured population

**Almost every post-move-faint replacement ply.** When a Pokémon faints to a move, gen 3 pauses
mid-turn for the replacement; the surviving side gets `wait`, so the ply is single-seat and skipped.

> **The exception, and an earlier draft asserted there was none.** That draft said "*Every*
> post-move-faint replacement ply is single-seat" and concluded "the differential has **never**
> compared a deferred residual phase". Both are false. When **both** actives faint in the same ply —
> Explosion, Selfdestruct, Destiny Bond, a recoil KO, all present in gen3 randbats — there is no
> survivor to receive `wait`: both sides get `forceSwitch`, `requested == {"p1","p2"}`, and the
> replacement ply is a **full-round boundary that IS measured**. Demonstrated in Showdown
> (`gen3customgame`, sandstorm up, Tauros Explosion): both requests come back
> `forceSwitch=true wait=false`, and the following ply carries both switches plus the full residual
> phase. Corroborating from the harness side, `_is_forced_replacement_ply` is called only from
> `evaluate_boundary_strict`, which only full-round boundaries reach — so the harness already assumes
> replacement-shaped plies can be compared.
>
> The quantitative bound is unaffected (87 % vs 96.6 % does not move), and the harness handles the
> double-faint case correctly. But a universal claim was the *justification* for why the bound matters,
> in a report whose whole purpose is to stop an over-read — so it is corrected rather than softened
> silently.

That matters because **the residual phase is deferred onto exactly those plies**. Measured by driving
Showdown directly in `gen3customgame`: the move-faint ply itself carries no residuals, and the ply
after the replacement carries the whole phase — weather upkeep, both actives' chips, items, status.
The engine defers it identically; its branch ends in `ToggleSide*ForceSwitch` and the next transition
carries the full list.

So the differential has compared a deferred residual phase only in the double-faint case above. Both
simulators agree there as far
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

`tests/test_single_seat_coverage_bound.py` asserts that both counters are present, that the
full-round path reconciles (`measured + in-path exits == full_round`), and that the single-seat share
stays material — so the denominator is always recoverable and this bound stays computable. It does
**not** pin the exact fraction, which legitimately drifts with the pool and the seed window; a gate on
it would fail for the wrong reason.

> **What these pins cannot do, stated because it is easy to over-read them.** They read four
> *committed* artifacts. They import nothing from `scripts/`, exercise no live code path, and
> therefore **cannot go red from a change to the counting logic** — only from someone editing those
> JSON files. If a future edit hoisted `counts["boundaries_full_round"] += 1` above the `if` and nobody
> regenerated the artifacts, these pins would stay green.
>
> That is exactly how the first version of them was defeated: a review folded single-seat plies into
> `boundaries_full_round` *and recomputed the derived fraction the way the live writer does*, and all
> three pins passed. My own red-run had appeared to catch it only because hand-editing the JSON left
> `measured_fraction_of_full_rounds` stale — an artifact of mutating data instead of code. The
> reconciliation assertion added since catches that mutation, but the structural limitation remains.
>
> The cheap way to get one live-coupled pin: the single-seat arm never calls `_prepare_boundary`, so it
> needs no engine. A stub `env` whose `requested_players()` returns one seat, driven through
> `run_game`, would assert on the real fork. Filed, not done here.
