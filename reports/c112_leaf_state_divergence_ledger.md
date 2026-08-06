# C112 — source-level cause and disposition for every `state`-class row in the leaf differential

> **Scope and honesty note.** This ledger attributes **88 of 106** `state`-class rows on
> golden-v2 to two verified source-level causes with a disposition each. **Three classes
> (26 rows on golden-v2, 10 on scenarios) are recorded as OPEN, deliberately unattributed**,
> with the measured direction and the exact next check for each. C111 v1 shipped three wrong
> attributions and had to be withdrawn; guessing is the known failure mode in this format, so
> the open rows say "open" rather than carrying a plausible cause.

## Era and provenance

Read on `main` at `df4a0fce`, with both poke-engine build artifacts freshly re-vendored
(#1119 patched the vendored `gen3/choice_effects.rs`; a stale tree makes three ability tests
fail and is invisible from the harness).

| corpus | observation schema | tables | boundaries | compared | divergent | artifact |
|---|---|---|---|---|---|---|
| `corpus/golden-v2` | `pokezero.observation.v3` | `/tmp/tables_v3.json` | 1008 | 737 | 567 | `reports/c112_leaf_state_golden_v2.json` |
| `corpus/golden-v2-scenarios` | `pokezero.observation.v2.2` | `corpus/encoder_tables_v2.2.json` | 369 | 273 | 174 | `reports/c112_leaf_state_scenarios.json` |

`matched + diverged == compared` on both (170+567 = 737; 99+174 = 273). Compared is
boundaries minus skips, and is the denominator every rate below uses — **not** the boundary
count, which credits capacity the run did not exercise.

Producing commands, verbatim:

```sh
# corpora (both regenerated; corpus/ is gitignored)
PYTHONPATH=src python -m pokezero.golden_corpus --showdown-root "$POKEZERO_SHOWDOWN_ROOT" \
    --games 10 --seed-start 1000 --out corpus/golden-v2 --belief-set-source on \
    --observation-schema pokezero.observation.v3
PYTHONPATH=src python -m pokezero.golden_corpus_scenarios --showdown-root "$POKEZERO_SHOWDOWN_ROOT" \
    --out corpus/golden-v2-scenarios --belief-set-source on

# tables
PYTHONPATH=src python scripts/export_encoder_tables.py --showdown-root "$POKEZERO_SHOWDOWN_ROOT" \
    --observation-schema v3 --out /tmp/tables_v3.json

# the measurement
PYTHONPATH=src:scripts python scripts/leaf_vs_reality.py --corpus corpus/golden-v2 \
    --tables /tmp/tables_v3.json --json reports/c112_leaf_state_golden_v2.json
PYTHONPATH=src:scripts python scripts/leaf_vs_reality.py --corpus corpus/golden-v2-scenarios \
    --tables corpus/encoder_tables_v2.2.json --json reports/c112_leaf_state_scenarios.json
```

**The scenarios generator takes no `--observation-schema`** and defaults to v2.2, which is why
the two corpora are read at different schemas. Running the scenarios corpus against v3 tables
skips **273 of 369** boundaries on `encode_error:ValueError` and the harness then reports
`matchup gate INERT on this corpus: only 0 of 369 same-seat boundaries were compared` — worth
recording because that INERT line is the only thing distinguishing that run from a pass.

## The table

| class | rows (gv2 / scen) | cause | disposition |
|---|---|---|---|
| `NUMERIC_SELF_WISH_TURNS` | 17 / 0 | **S1** adjacent-key write | harness fix (one line) |
| `NUMERIC_OPP_WISH_TURNS` | 17 / 0 | **S1** | harness fix (one line) |
| `NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF` | 4 / 0 | **S1**, different predicate | harness fix |
| `NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP` | 4 / 0 | **S1**, different predicate | harness fix |
| `NUMERIC_STALL_COUNTER` (self) | 23 / 0 | **S2** never plumbed; not in engine state | **demonstrated-unreachable from engine state** → encoder-side decision |
| `NUMERIC_STALL_COUNTER` (opponent) | 23 / 0 | **S2** | as above |
| `NUMERIC_TOXIC_STAGE` (self) | 12 / 0 | **OPEN — S3** | open (two candidate mechanisms below) |
| `NUMERIC_TOXIC_STAGE` (opponent) | 12 / 0 | **OPEN — S3** | open |
| `NUMERIC_ENCORE_TURNS` | 2 / 0 | **OPEN — S4** | open (one tick behind; candidate: tagged engine-model) |
| action surface (`NUMERIC_ACTIVE`, `NUMERIC_LEGAL`, `legal_action_mask`) | 5 / 10 | **OPEN — S5** | open |

Row counts are per (array, block, column) family; one boundary can carry several, which is why
the rows sum above the 106/2 divergent-boundary counts.

## S1 — the leaf writes an ADJACENT metadata key, not the one the column reads (42 rows)

**Verified at source.** `leaf.rs:1221-1222`:

```rust
md.insert("self_wish_pending".into(), json!(self_side.wish.0 != 0));
md.insert("opponent_wish_pending".into(), json!(opp_side.wish.0 != 0));
```

The leaf sets `self_wish_pending` / `opponent_wish_pending` as **booleans**. The diverging
columns are `NUMERIC_SELF_WISH_TURNS` / `NUMERIC_OPP_WISH_TURNS`, which read the *separate*
state fields `self_wish_turns` / `opponent_wish_turns` (`showdown.py:1719-1720`, value
`min(1, remaining / 2)` per `showdown.py:517`). Those are never set at the leaf, so they hold
their dataclass default of `0` — which is exactly the measured signature: **got 0.0, want 0.5**
on all 34 rows, never the reverse.

Both `NUMERIC_SELF_WISH_PENDING` and `NUMERIC_SELF_WISH_TURNS` exist as distinct columns
(`showdown.py:907,911`), so this is not a duplicate — the leaf populates one member of a pair
and silently leaves the other at zero.

**The count is available and is being discarded.** `pub wish: (i8, i16)` (`state.rs:1208`) — the
first member IS the remaining-turns counter, and `!= 0` throws it away. So this is a one-line
harness fix, not a modelling gap: write the count beside the bool.

Same shape, different predicate, for the sleep-clause pair. The leaf sets
`self_sleep_clause_used` (`leaf.rs:1233,1240`, with a genuine engage/release derivation at
`:1254-1261`), while the column reads `self_sleep_clause_blocks` (`showdown.py:1729-1730`) —
documented there as "an opposing mon is currently asleep from a sleep OUR side inflicted",
which is a *different question* from whether the clause has been used. Never set → `False` →
**got 0.0, want 1.0** on all 8 rows.

**Disposition: harness fix.** Neither is a comparison limit and neither needs an encoder change.

## S2 — `NUMERIC_STALL_COUNTER` is not derivable from engine state (46 rows)

**Verified two ways.** `stall_counter` has **zero** occurrences in
`rust/pokezero-search/src/leaf.rs`, so the leaf never writes it and the column is 0 at every
depth — matching the measured **got 0.0, want 0.125** on all 46 rows.

And it cannot simply be plumbed: `third_party/poke-engine-src/src/state.rs` carries no
stall/protect/consecutive-protect counter at all (the only `stall` substring matches are
`terastallized`). The counter is protocol-derived — the Protect/Detect success chain — which is
why `showdown.py` has 31 references to it and the engine none.

**Disposition: demonstrated-unreachable from engine state.** This is not a harness bug to fix
in `leaf.rs`; the value has to come from the root row or the fold, which makes it the same
shape as the k0 feature pack and therefore an **encoder-side design decision**, not a defect.
Two options, both owner calls, deliberately not made here:

1. root-freeze it explicitly (add to a documented frozen set, as the k0 pack is), or
2. carry it on the fold and surface it through `ProductsData` — the route taken for
   `matchup_counters` in #1118.

Until one is chosen, 46 of the 106 rows are **correctly** divergent and the `state` class is
the wrong bucket for them: `state` is documented as "MUST-MATCH engine-state-derived cells",
and this cell is by construction not engine-state-derived. **Reclassifying it is a
`classify()` change and is out of this ledger's documentation-only scope** — but it is the
single largest contributor to the 106 and the reason that number should not be read as 106
encoder defects.

## S3 — `NUMERIC_TOXIC_STAGE`: OPEN, 24 rows

Measured **got 0.0, want 0.1333** (= 2/15, i.e. reality is at Toxic stage 2 while the leaf
reads 0). Two candidate mechanisms, both live, and I have not established which fires:

1. **The line replay did not escalate.** The leaf's toxic stage is *deliberately* line-driven,
   not engine-derived — `leaf.rs:298-302` states it: "the parser's toxic stage and … LINES, not
   of engine state — the engine ticks its toxic counter on every … faint-pending ply ticks the
   engine but never the parser". The parser escalates on `|turn|` lines only (review F1), so a
   synthesized line set without the `|turn|` boundary leaves `meta.toxic` at 0.
2. **The leaf's own guard zeroed it.** `leaf.rs:1265-1267` zeroes a nonzero stage when
   `active.hp <= 0 || active.status != PokemonStatus::TOXIC`. The engine does carry
   `toxic_count` (`state.rs:707`) and a `TOXIC` status, so a status representation mismatch
   would zero a correct stage.

**Next check, precise:** instrument `LeafMeta` at one of the 12 example boundaries
(`golden-gen3randombattle-1000#[56,57]` p1) and print `meta.toxic[side]` *before* the guard.
Nonzero-before-guard proves (2); zero proves (1). Both dispositions differ — (1) is a
harness/line-replay fix, (2) is a status-mapping fix — so attributing without this measurement
is exactly the C111 v1 error.

## S4 — `NUMERIC_ENCORE_TURNS`: OPEN, 2 rows

Measured **got 0.1667 (1/6), want 0.3333 (2/6)** — one tick behind, not absent, which
distinguishes it from S1/S2. The candidate cause is the already-tagged engine-model deviation
"Encore volatile not applied" (`docs/leaf_observation_column_map.md`, `engine_model` row), in
which case these rows are misfiled into `state` and belong in `engine_model`. Not asserted:
that class is documented as *not applied*, whereas these rows show a value that IS applied and
merely lags, so the existing tag may not be the right one.

**Next check:** replay `golden-gen3randombattle-1003#[8,9]` p1 and compare the engine's
`ENCORE` volatile duration against the parser's `opponent_encore_elapsed` tick, to see whether
the lag is the documented non-application or a separate off-by-one.

## S5 — the action surface: OPEN, 5 rows on golden-v2 and 10 on scenarios

`NUMERIC_ACTIVE`, `NUMERIC_LEGAL` and three `legal_action_mask` slots diverge on a single
boundary per corpus (`golden-gen3randombattle-1009#[16,17]` p2), all in the same direction:
**got True/1.0, want False/0.0** — the leaf believes actions are legal that reality does not.
Five rows on one boundary is one disagreement, not five.

This is the only `state` family whose direction implies the leaf is *permissive* rather than
*empty*, which makes it the highest-severity open row here: a search that prices illegal
actions is worse than one reading a stale counter.

**Next check:** dump the leaf's `action_candidates` and reality's for that boundary and diff
the three slots. Note the scenarios corpus also reports 7 `skip:action_unmapped`, which may be
the same mechanism surfacing as a skip rather than a divergence.

## What this ledger does not claim

- It is **not** a completeness claim over `state`: 18 of 106 golden-v2 rows are open.
- It is documentation only — no classifier change, no encoder change, no harness change.
- The **scenarios** corpus contributes only S5 (2 rows). Its `state` class is otherwise clean,
  so every S1–S4 count above is golden-v2 only.
- `self_moveset_mismatch` (11 skips on scenarios) and the residue-row classes are owned by the
  fallback-burndown and rust-fidelity lanes; they appear here only as skip counts and are
  **not** attributed. A note belongs in those ledgers, not a fix here.
