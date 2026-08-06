# C112 v3 — source-level cause and disposition for every `state`-class row in the leaf differential

> **v3 covers the corpus the task named, which v2 did not.** v2 ledgered `golden-v2`
> (106 boundaries, v3 schema) and the scenarios corpus (2), and presented that as the
> ledger of "the 124". The 124 comes from a **v4** corpus, and v2's class set is a strict
> SUBSET of it: four families were entirely unattributed — `NUMERIC_SLEEP_TURNS`
> (self + opponent) and `CATEGORY_VOLATILE_OFFSET` (self + opponent). I substituted a
> corpus that produced a tidier number. v3 measures all three corpora and attributes
> **every one of the 18 families / 138 rows** the fixed harness surfaces, with a cause and
> a disposition each — no class is left open.
>
> v3 also closes the two classes v2 left open (S4 encore, S5 action surface) rather than
> recording a next check, because the task's disposition vocabulary is
> {encoder fix, harness fix, demonstrated-unreachable with a pool check} and "open" is not
> in it.

# C112 v2 — superseded header, retained for traceability

> **v2 corrects v1 substantially after independent review. v1 should not be cited.**
> Three of v1's central claims were wrong and one of them re-committed the exact error
> C111 v1 was withdrawn for. Every correction is marked **[CORRECTION]** below.
>
> - **v1's S2 was FALSE.** It called `NUMERIC_STALL_COUNTER` "demonstrated-unreachable
>   from engine state" on a grep for `stall` that was a substring false-negative. The
>   engine names it `protect`, and production already seeds it. Those 46 rows are a
>   harness fix, not a design decision. See S2.
> - **v1's coverage arithmetic did not close.** It gave three numbers for one quantity
>   (18/26/31) and divided a ROW count by a BOUNDARY count. See "Units".
> - **v1 claimed the regenerated golden-v2 is comparable to the published row.** It is
>   not, and that claim was the sole justification for lifting UNVERIFIED. See "What is
>   and is not comparable".
> - v1 also left S3 and S4 "open" when the committed artifact already answered S3;
>   understated the scenarios action-surface incidence; claimed a direction "on all N
>   rows" from an artifact storing one example per family; cited a tautology as
>   evidence; and stamped its era one day in the future.

## Era and provenance

Read on `main` at `df4a0fce`, **2026-08-05**, with both poke-engine build artifacts
freshly re-vendored (#1119 patched the vendored `gen3/choice_effects.rs`; a stale tree
fails three ability tests and is invisible from this harness).

| corpus | `rows.jsonl` sha256 | observation schema | tables | boundaries | compared | divergent | artifact |
|---|---|---|---|---|---|---|---|
| `corpus/golden-v2` | `e8d4db0772c65648…` | `pokezero.observation.v3` | v3, exported below | 1008 | 737 | 567 | `reports/c112_leaf_state_golden_v2.json` |
| `corpus/golden-v2-scenarios` | `1931040f087f317a…` | `pokezero.observation.v2.2` | `corpus/encoder_tables_v2.2.json` | 369 | 273 | 174 | `reports/c112_leaf_state_scenarios.json` |
| `corpus/golden-v4` **(the 124)** | `ac5a202a8145a89e…` | `pokezero.observation.v4` | `corpus/encoder_tables_v4.json` | 1271 | 956 | 917 | `reports/c112_leaf_state_golden_v4.json` |

**[CORRECTION]** v1 recorded no corpus hash, so "regenerate before citing" was
unfalsifiable, and cited an ephemeral `/tmp` tables path. `corpus/` is gitignored, so the
hashes above are the only way a future reader can tell whether their regeneration matches.

Producing commands, verbatim:

```sh
PYTHONPATH=src python -m pokezero.golden_corpus --showdown-root "$POKEZERO_SHOWDOWN_ROOT" \
    --games 10 --seed-start 1000 --out corpus/golden-v2 --belief-set-source on \
    --observation-schema pokezero.observation.v3
PYTHONPATH=src python -m pokezero.golden_corpus_scenarios --showdown-root "$POKEZERO_SHOWDOWN_ROOT" \
    --out corpus/golden-v2-scenarios --belief-set-source on
PYTHONPATH=src python scripts/export_encoder_tables.py --showdown-root "$POKEZERO_SHOWDOWN_ROOT" \
    --observation-schema v3 --out /tmp/tables_v3.json
PYTHONPATH=src:scripts python scripts/leaf_vs_reality.py --corpus corpus/golden-v2 \
    --tables /tmp/tables_v3.json --json reports/c112_leaf_state_golden_v2.json
PYTHONPATH=src:scripts python scripts/leaf_vs_reality.py --corpus corpus/golden-v2-scenarios \
    --tables corpus/encoder_tables_v2.2.json --json reports/c112_leaf_state_scenarios.json

# the 124 -- the corpus the task named, which v2 omitted
PYTHONPATH=src python -m pokezero.golden_corpus --showdown-root "$POKEZERO_SHOWDOWN_ROOT" \
    --games 12 --seed-start 1000 --out corpus/golden-v4 --belief-set-source on \
    --observation-schema pokezero.observation.v4
PYTHONPATH=src python scripts/export_encoder_tables.py --showdown-root "$POKEZERO_SHOWDOWN_ROOT" \
    --observation-schema v4 --out corpus/encoder_tables_v4.json
PYTHONPATH=src:scripts python scripts/leaf_vs_reality.py --corpus corpus/golden-v4 \
    --tables corpus/encoder_tables_v4.json --json reports/c112_leaf_state_golden_v4.json
```

`golden_corpus_scenarios` takes no `--observation-schema` and defaults to v2.2, which is
why the two corpora are read at different schemas. Against v3 tables the scenarios corpus
skips **273 of 369** boundaries on `encode_error:ValueError`, and the run then prints
`matchup gate INERT on this corpus: only 0 of 369 same-seat boundaries were compared` —
that INERT line being the only thing distinguishing it from a pass.

## Units — read this before any count below

**[CORRECTION]** v1 mixed two incompatible denominators. This ledger states coverage in
**ROWS** throughout:

- A **row** is one `(array, block, column)` family. The `state` families sum to
  **119 rows** (`turn` contributes none) on golden-v2 and **10** on scenarios.
- `class_rows.state` counts **BOUNDARIES**: **106** on golden-v2, **2** on scenarios.
- **The mapping between them is not derivable from the artifact.** Families are
  incremented once per boundary (`leaf_vs_reality.py:892-893`) and no per-boundary family
  list is stored, so the distinct boundaries any group of families covers is only bounded
  — for v1's S1+S2 group, to `[23, 88]`. v1's headline "88 of 106" divided rows by
  boundaries and is withdrawn. Any future coverage claim in boundaries needs the harness
  to record per-boundary family sets first; that is a harness change and out of scope here.

Coverage on **the 124** (`corpus/golden-v4`): **138 of 138 rows attributed**, across 18
families, no class open — S1 46 + S2 52 + S3 28 + S4 2 + S5 8 + S6 2. Of those, 110 are
source-verified and 28 rest partly on side-symmetry (S3's opponent half, 14 rows, and S6's,
1 row), which is carried as a split for the reason below.

On `golden-v2`: **119 of 119 rows**, same causes minus S6 (which does not surface there) and
with S5 contributing 5 rather than 8. Scenarios: **10 of 10**, all S5.

The verified/inferred split is carried rather than merged because that is how a ledger of
this kind gets over-read — and its predecessor was withdrawn for exactly that.

The 12/12 split is carried here and not only in S3's prose, because merging a verified count
with an inferred one is how a ledger of this kind gets over-read — and its predecessor was
withdrawn for exactly that.

## The table

| class | rows (v4 / gv2 / scen) | cause | disposition |
|---|---|---|---|
| `NUMERIC_SELF_WISH_TURNS` | 17 / 17 / 0 | **S1** adjacent-key write | harness fix |
| `NUMERIC_OPP_WISH_TURNS` | 17 / 17 / 0 | **S1** | harness fix |
| `NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF` | 6 / 4 / 0 | **S1**, different predicate | harness fix |
| `NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP` | 6 / 4 / 0 | **S1**, different predicate | harness fix |
| `NUMERIC_STALL_COUNTER` (self) | 26 / 23 / 0 | **S2** omitted from the side-condition export | harness fix |
| `NUMERIC_STALL_COUNTER` (opponent) | 26 / 23 / 0 | **S2** | harness fix |
| `NUMERIC_TOXIC_STAGE` (self) | 14 / 12 / 0 | **S3** line replay did not escalate | harness fix |
| `NUMERIC_TOXIC_STAGE` (opponent) | 14 / 12 / 0 | **S3** | harness fix (by symmetry) |
| `NUMERIC_ENCORE_TURNS` | 2 / 2 / 0 | **S4** in-branch Encore unappliable + line-derived tag | classifier fix |
| `CATEGORY_VOLATILE_OFFSET` (self) | 2 / 0 / 0 | **S5** forced-recharge split by side | production + gate fix → **task 4** |
| `CATEGORY_VOLATILE_OFFSET` (opponent) | 1 / 0 / 0 | **S5** | → **task 4** |
| `NUMERIC_ACTIVE` (action) | 1 / 1 / 2 | **S5** | → **task 4** |
| `NUMERIC_LEGAL` (action) | 1 / 1 / 2 | **S5** | → **task 4** |
| `legal_action_mask` action0 | 1 / 1 / 0 | **S5** | → **task 4** |
| `legal_action_mask` action1 | 1 / 1 / 2 | **S5** | → **task 4** |
| `legal_action_mask` action3 | 1 / 1 / 2 | **S5** | → **task 4** |
| `NUMERIC_SLEEP_TURNS` (self) | 1 / 0 / 0 | **S6** harness opts into an approximated counter | harness fix |
| `NUMERIC_SLEEP_TURNS` (opponent) | 1 / 0 / 0 | **S6** | harness fix (by symmetry) |

**[CORRECTION]** v2's table listed ten rows and omitted `CATEGORY_VOLATILE_OFFSET` and
`NUMERIC_SLEEP_TURNS` entirely, because it ledgered a corpus on which they do not surface.
`legal_action_mask action2` appears on scenarios only (2 rows) and is folded into S5;
scenarios' action set is `action1/2/3` rather than `action0/1/3`.

## S1 — the leaf writes an ADJACENT metadata key, not the one the column reads (46 rows on v4, 42 on gv2)

Unchanged from v1; independently re-verified, including that no other site writes the
target keys.

`leaf.rs:1221-1222` sets `self_wish_pending` / `opponent_wish_pending` as **booleans** from
`side.wish.0 != 0`. The diverging columns read `self_wish_turns` / `opponent_wish_turns`
(`showdown.py:1719-1720`; value `min(1, remaining / 2)` per `:517`), consumed at
`encoder.rs:1289,1297`. A repo-wide grep for `wish_turns|sleep_clause_blocks` under `rust/`
finds **only reads, no writer**, so the keys hold their dataclass default of 0 — matching
got 0.0 / want 0.5. Both `NUMERIC_SELF_WISH_PENDING` and `NUMERIC_SELF_WISH_TURNS` exist as
distinct columns (`showdown.py:907,911`), so the leaf populates one member of a pair and
leaves the other at zero.

**The count is available and discarded**: `pub wish: (i8, i16)` (`state.rs:1208`), first
member is the remaining-turns counter; `set_wish` sets `wish.0 = 2`.

Same shape, different predicate, for the sleep-clause pair: the leaf sets
`self_sleep_clause_used` (`leaf.rs:1233,1240`, with a real engage/release derivation at
`:1254-1261`) while the columns read `self_sleep_clause_blocks` (`showdown.py:1729-1730`,
`encoder.rs:1275,1282`) — documented there as a different question.

**Disposition: harness fix.** Neither is a comparison limit.

## S2 — `NUMERIC_STALL_COUNTER` is omitted from the leaf's side-condition export (52 rows on v4, 46 on gv2)

**[CORRECTION] — this replaces v1's "demonstrated-unreachable from engine state", which
was false.** v1 grepped `leaf.rs` and `state.rs` for `stall`, found only `terastallized`,
and concluded the value does not exist in the engine. The engine names it **`protect`**:

- `third_party/poke-engine-src/src/state.rs:699` — `pub protect: i8` on `SideConditions`.
- `gen3/generate_instructions.rs:4941,4952` — `+= 1` on each successful PROTECT/ENDURE,
  reset to 0 otherwise. That is the same "consecutive successful stall uses" quantity as
  `showdown.py::_update_stall_counter`.
- `gen3/generate_instructions.rs:2774,2796` — the engine reads it as the consecutive-protect
  success chance, so it is load-bearing, not vestigial.
- **`src/pokezero/engine_world.py:1265` — `side_conditions["protect"] = stall_counter`.**
  Production already seeds the engine counter from the parser's, and the comment at
  `:1252-1258` states the semantics match with no offset.

So at a leaf `side.side_conditions.protect` **is** the live stall counter, and
`leaf.rs:2220 side_condition_counts()` already exports `spikes/reflect/lightscreen/
safeguard/mist` from that exact struct while omitting `protect`. Same one-line shape as S1.

**Disposition: harness fix.** The root-freeze-vs-`ProductsData` decision v1 posed to the
owner does not exist, and v1's escalation that "46 of the 106 are arguably in the wrong
class" is withdrawn — that argument rested on the value being unreachable.

**The caveat v1 should have carried instead**, which is a fidelity note rather than a
blocker: the engine's `protect` is a **side** condition and does not reset on switch-out or
faint the way the parser's per-active counter does, so plumbing it is exact only within a
stint. Whoever writes the fix must decide what to emit across a switch.

**[CORRECTION]** v1 also claimed `showdown.py` "has 31 references"; that reproduces as
neither the line nor the occurrence count. The claim is dropped rather than restated.

## S3 — the toxic line replay did not escalate (28 rows on v4, 24 on gv2)

**[CORRECTION] — v1 left this open with two candidate mechanisms. The committed artifact
already discriminates them, so it is closed here with no new measurement.**

The leaf's toxic stage is deliberately line-driven, not engine-derived (`leaf.rs:298-302`:
"the parser's toxic stage … LINES, not of engine state — the engine ticks its toxic counter
on every … faint-pending ply ticks the engine but never the parser"). The parser escalates
on `|turn|` lines only (review F1). v1's two candidates were:

1. the line replay never escalated, leaving `meta.toxic` at 0; or
2. the guard at `leaf.rs:1265-1267` zeroed a correct stage when
   `active.hp <= 0 || active.status != PokemonStatus::TOXIC`.

**(2) is excluded by the artifact** — but only via instruments that can actually move, which
took a correction. Both guard arms are observable as other columns:

- the `active.hp <= 0` arm rides **`NUMERIC_LEGAL`** (`encoder.rs:2282-2285`:
  `if condition.fainted { 0.0 } else { 1.0 }`), where `condition.fainted` is parsed from the
  leaf's own condition string (`encoder.rs:555`). That string is written from the same `p.hp`
  the guard reads — literally the same expression, `if hp <= 0 { return "0 fnt" }`
  (`leaf.rs:195-198`) against the guard's `active.hp <= 0` (`leaf.rs:1266`), with no rounding,
  clamp or separate faint flag that could drift.

  The write is gated: the primary path (`leaf.rs:1432-1434`) re-derives the string only
  `if changed` (`:1430-1431`), otherwise the root ledger's string is retained by design
  (`:1421-1428`); `leaf.rs:1506` is the fallback for a first-time-active self mon. That gate
  cannot hide a fainted leaf mon: for the guard to fire while `NUMERIC_LEGAL` stays 1.0 you
  would need `p.hp <= 0` AND `snapshot.hp == p.hp`, i.e. the mon was already at hp <= 0 at the
  root — in which case the retained root string is itself `"0 fnt"` and `fainted` is still
  true. The only escape is a recorded root-ledger skew, and `ledger_skew` is **0** on
  golden-v2.

  `NUMERIC_LEGAL` is not in `EPISTEMIC_PREFIXES`, so a `self_team` divergence would fall to
  the `state` fallback and appear in the table above. It is also a **live** instrument in this
  corpus, which is the check the old version lacked: in `corpus/golden-v2/arrays.npz` the
  self-team `NUMERIC_LEGAL` cells take both values (2055 zeros of 6168, in 787 of 1028 rows),
  where `NUMERIC_PRESENT` is `{1.0}` and the self-side `attention_mask` is `{True}`.
- the `active.status != TOXIC` arm rides **`CATEGORY_SECONDARY`** (`encoder.rs:2258-2262`).
  It is the engine's own field, not the belief ledger: `belief` is `None` for
  `Role::SelfTeam` (`encoder.rs:1992-1994`), so the status falls through to
  `condition.status` ← md `"status"` ← `status_code(p.status)` (`leaf.rs:1505`). Without that
  step the argument would be comparing the belief ledger against itself. It is also written
  unconditionally (a NONE status renders `status:`), so absence cannot mean "never written".

Both are **absent from `self_team` under EVERY class**, not just `state` — checked, because a
`CATEGORY_SECONDARY` divergence would route to `ledger_skew` under a `curestatus` tag
(`leaf_vs_reality.py:346-352`) or `engine_model` under
`transform|recharge|encore|baton_pass` (`:355`), and a `state`-only check would have missed
those. So neither arm can have fired.

**[CORRECTION]** An earlier revision of this section cited `NUMERIC_PRESENT` and
`self_team`'s `attention_mask` as the liveness instruments. Both are **constants**:
`NUMERIC_PRESENT` is set unconditionally to 1.0 for every team token
(`encoder.rs:2286` — and `docs/leaf_observation_column_map.md:83` already classifies it `C |
constants`) and the self-side attention mask depends only on team size
(`encoder.rs:1110-1112`). Neither can ever diverge, so reading their absence as a
measurement was the same false-negative move as v1's `grep stall` — an instrument that was
never going to fire. `NUMERIC_LEGAL` is the one that carries the signal.

That leaves (1). **Disposition: harness fix** — the synthesized line set does not carry the
`|turn|` boundary the parser escalates on.

Scope limit, stated rather than glossed: this argument is clean for the **12 self-side**
rows only. The opponent's 12 are attributed **by side-symmetry** — the guard and
`meta.toxic[engine_side]` are one loop over both sides (`leaf.rs:1230-1269`) and the
`|turn|`-escalation mechanism is side-agnostic — and the symmetry is an inference, counted
separately in Units.

The opponent side has **no analogue of the discriminator, in principle**: for
`Role::Opponent`, `belief = opponent_beliefs.get(&species_key)` (`encoder.rs:1993`), so its
status token comes from `belief.status()` first and could not evidence the engine's
`active.status` even if it had agreed. Worse, `opp_membership` routes every
`opponent_team` column to `epistemic` on 78 boundaries (`leaf_vs_reality.py:334-335`), so an
opponent status divergence is masked by construction — which is also what the 3-row
`epistemic` toxic family is, a symptom of that sweep rather than an independent anomaly.

## S4 — in-branch Encore is unappliable, and the tag that should catch it is line-derived (2 rows)

**[CORRECTION] — v2 left this open. It is closed here from the harness's own comments.**

Measured got 1/6, want 2/6 — one tick **behind**.

The cause is documented in the harness: "the vendored gen3 engine does not apply the Encore
volatile from a branch (world construction fail-closes on ROOT encore states for the same
reason: `encore_move_unknown`)" (`leaf_vs_reality.py:609-611`). So an Encore that exists
across the boundary cannot tick in the leaf's world.

Why these 2 rows land in `state` rather than the `engine_model` class that already exists for
exactly this: the tag is set by a **literal line match** —

```python
if any("|Encore" in line for source in (synthesized, row_next.get("event_slice") or ())
       for line in source):
    tags.add("encore")          # leaf_vs_reality.py:612-617
```

— and `:355` routes tagged boundaries to `engine_model` before the fallback. A mon already
under Encore *before* the replayed window emits no new `|Encore` line, so it gets no tag and
falls through to `state` at `:382`. The artifact shows both halves of the partition:
`engine_model / self_team / NUMERIC_ENCORE_TURNS rows=2` **and** the `state` pair.

**Disposition: classifier fix.** Derive the tag from the Encore volatile's presence in the
root or leaf state rather than from a line match, and these rows join the `engine_model`
class where the deviation is already accepted. No encoder or engine change.

## S5 — the forced-recharge split by side (8 rows on v4, 5 on gv2, 10 on scenarios)

**[CORRECTION] — v2 called this "the action surface, OPEN" and missed that it is one cause
with the volatile columns, and that the cause is already documented.**

All of these rows are one mechanism, and the v4 corpus is what makes it visible: the
volatile columns v2 never ledgered share a boundary with the action surface.

`CATEGORY_VOLATILE_OFFSET` diverges with vocab id **877 = `volatile:mustrecharge`**
(decoded from `corpus/encoder_tables_v4.json`):

- `golden-gen3randombattle-1009#[16,17]`, self side: got **877**, want 0 — the leaf's own mon
  carries `mustrecharge` where reality does not. The **same boundary** carries every
  action-surface row: `NUMERIC_ACTIVE` and `NUMERIC_LEGAL` got 1.0 want 0.0, and
  `legal_action_mask` action0/1/3 got True want False.
- `#[18,19]`, opponent side: got 0, want **877** — the mirror, one boundary later.

That is precisely what `docs/leaf_observation_column_map.md` already records about A1: the
opponent side is refreshed live from the branch's own `volatile_statuses`, while the **self**
side is root-frozen deliberately, because `engine_search.py::_recharging_slots` returns the
opponent slot or nothing and never ours — so "the production root world currently lets our
recharging mon pick any move". The leaf therefore holds a self-side `mustrecharge` while its
own action surface presents moves as legal: the column map's phrase is that this member "did
not merely go stale at a leaf but **contradicted the same observation's action surface**".

**Disposition: production + gate fix, and it is task 4 of this goal, not a separate finding.**
The column map also records why the gates cannot currently catch a bad fix:
`leaf_root_parity.py`, `leaf_vs_reality.py`, `prior_mapping_assert.py` and
`fidelity_gate_events.py` all derive `recharging` for BOTH slots from the recorded chosen
candidate, so they build a different world than production and would ratify a symmetric
self-side write rather than catch it. Task 4's ordering — gates first, then
`_recharging_slots` — is exactly right, and these 8 rows are the measured incidence it should
close.

**Severity note retained from v2, now with a cause:** this is still the only family where the
leaf is *permissive* rather than empty, and a search pricing illegal actions is worse than one
reading a stale counter.

## S6 — the harness opts into an approximated sleep counter (2 rows)

**[CORRECTION] — v2 never ledgered this class; it does not surface on the corpus v2 measured.**

Measured got 0.4 (2/5), want 0.2 (1/5) — the leaf is one tick **ahead**, the opposite
direction from S4.

`leaf_vs_reality.py:448` constructs every world with `approximate_sleep_turns=True`, and
`engine_world.py:137-139` documents what that means: "mapping `slp` with `sleep_turns=0`
('just fell asleep') — a documented approximation for search POCs; **the real fix is public
sleep-counter tracking in the replay state**". So the leaf's counter restarts at 0 at world
construction and ticks from the root, while reality's parser counts observed `\|cant …\|slp`
lines since the `\|-status\|slp`. A mon that fell asleep before the root diverges by
construction.

**Disposition: harness fix**, and the harness already names the real one. The divergence is an
artifact of the comparison setup the differential opts into, not an encoder defect — which is
why it belongs in an excuse class rather than `state`, the same argument S4 makes.
Opponent-side row attributed by symmetry (one loop, side-agnostic mechanism).

## What is and is not comparable

**[CORRECTION] — v1 claimed the regenerated golden-v2 is comparable to the published
closure row because it reproduces 1008 boundaries. That is false, and it was the sole
justification for lifting UNVERIFIED from the `leaf_vs_reality` tables.**

Boundary count cannot detect the change: `boundaries = decisions − 2·games` (verified:
gv2 manifest `decisions 1028, games 10` → 1008; scenarios `405, 18` → 369), and the
trajectories come from `RandomLegalPolicy` / `SimpleLegalPolicy`, whose `select_action`
reads **only** `legal_action_mask` and a seeded RNG (`policy.py:151-171`). It is
structurally invariant to observation-schema and engine changes, so 1008 could not have
moved.

Meanwhile the contents did change: the regenerated corpus is v3, and `--observation-schema`
only reached `golden_corpus` on 2026-08-04, while the published row is stamped 2026-07-19.
Four of the five `state` families are columns that did not exist then —
`NUMERIC_SLEEP_CLAUSE_BLOCKS_*` (2026-07-20), `NUMERIC_STALL_COUNTER` and
`NUMERIC_SELF_WISH_TURNS` and `NUMERIC_ENCORE_TURNS` (2026-07-21).

**Consequences, applied in `docs/leaf_observation_column_map.md`:**

1. The published closure rows are **RETRACTED for both corpora**, not corrected. No
   per-class delta (`fold 440→422` etc.) is attributable to the harness fix.
2. The re-derived rows are published as a **new measurement at their own era**, not as a
   correction of the old ones.
3. **v1's causal story for the published `state = 0` is also wrong.** A run that skipped
   100% of boundaries cannot emit `fold 440, epistemic 322, engine_roll 313`. The published
   zero is explained by four of the five families not being columns yet — a corpus/schema
   cause, not the harness break.

## What this ledger does not claim

- Coverage is in **rows**, and the row→boundary mapping is not derivable from the artifact.
- **No class is open.** Every one of the 18 families the fixed harness surfaces on the 124 has
  a cause and a disposition. Two dispositions point elsewhere rather than proposing work here:
  S5 is task 4 of this goal, and S4 is a `classify()` change this documentation-only pass does
  not make.
- Three dispositions rest partly on **side-symmetry** (S3 opponent 14 rows, S6 opponent 1 row);
  counted separately in Units and marked in the table.
- Documentation only: no classifier, encoder or harness change.
- **[CORRECTION]** v1 wrote "got 0.0, want 0.5 on all 34 rows". The harness stores **one
  example per family** (`leaf_vs_reality.py:874-875`), so direction is *inferred from
  source* for S1/S2/S3, not measured per row. The uniformity claim is withdrawn.
- **[CORRECTION]** v1 offered `matched + diverged == compared` as evidence. `compared` is
  *defined* as `exact + divergent` (`leaf_vs_reality.py:972`), so the identity is a
  tautology and no harness assertion exists. The non-vacuous identity —
  `sum(all counts) == boundaries` — is not checked by the harness either; verified by hand
  here (1008 and 369) and it holds. Mechanizing it is task 3's subject.
- `self_moveset_mismatch` (11 scenarios skips) and the residue-row classes are owned by the
  fallback-burndown and rust-fidelity lanes; they appear only as skip counts and are **not**
  attributed. A note belongs in those ledgers.
