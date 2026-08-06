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

## v2's header, retained for traceability — NOT this ledger's claims

> Everything in the block below is v2's text. It is kept so the corrections are traceable and
> is **superseded** by the v3 header above; do not read it as current.

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

Golden-v2 and scenarios read on `main` at `df4a0fce`; the v4 corpus (this revision's subject)
read on `main` at `d57a26ac`, two commits later. `scripts/leaf_vs_reality.py` is unchanged
across that range, so no measurement differs, but the provenance must describe the tree that
produced each run rather than one date for all three. with both poke-engine build artifacts
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

Coverage on the v4 corpus (`corpus/golden-v4`): **138 of 138 rows attributed** across 18
families, no class open — S1 46 + S2 52 + S3 28 + S4 2 + S5 2 + S5b 1 + S6 5 + S7 2 = 138.

**123 source-verified, 15 resting partly on side-symmetry** (S3's opponent half, 14 rows, and
S7's opponent row, 1). An earlier revision of this sentence said "110 source-verified and 28
… (14 rows, and 1 row)" — 14 + 1 is 15, not 28, so it stated two values for one quantity. That
is the defect this ledger's own header says v1 was corrected for, committed in the sentence
that announces the correction.

**Note on "the 124".** 124 is `class_rows.state`, a count of BOUNDARIES; 138 is the sum of
per-family boundary incidences, which this document calls rows. They are different units and
the Units section forbids juxtaposing them, so the coverage claim is stated over rows only.

On `golden-v2`: **119 of 119 rows**, same causes minus S6 (which does not surface there) and
with S5 contributing 5 rather than 8. Scenarios: **10 of 10**, all S5.

The verified/inferred split is carried rather than merged because that is how a ledger of
this kind gets over-read — and its predecessor was withdrawn for exactly that.



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
| `NUMERIC_ENCORE_TURNS` | 2 / 2 / 0 | **S4** Encore seeded at a floor and never advanced | encoder fix |
| `CATEGORY_VOLATILE_OFFSET` (self) | 2 / 0 / 0 | **S5** self-side recharge root-freeze | production + gate fix → **task 4** |
| `CATEGORY_VOLATILE_OFFSET` (opponent) | 1 / 0 / 0 | **S5b** recharge consumed a ply early, tag escapes seat-locally | accepted `engine_model` deviation + tag fix |
| `NUMERIC_ACTIVE` (action) | 1 / 1 / 2 | **S6** stale Choice lock | encoder fix |
| `NUMERIC_LEGAL` (action) | 1 / 1 / 2 | **S6** | encoder fix |
| `legal_action_mask` action0 | 1 / 1 / 0 | **S6** | encoder fix |
| `legal_action_mask` action1 | 1 / 1 / 2 | **S6** | encoder fix |
| `legal_action_mask` action3 | 1 / 1 / 2 | **S6** | encoder fix |
| `NUMERIC_SLEEP_TURNS` (self) | 1 / 0 / 0 | **S7** Sleep-Talk turn refund not modelled | encoder fix |
| `NUMERIC_SLEEP_TURNS` (opponent) | 1 / 0 / 0 | **S7**, same mon from the other seat | encoder fix |

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

**Scope of that liveness evidence:** it is a **golden-v2** measurement and was not re-run on
the v4 corpus, even though v4 is what licenses S3's 28 rows here. The conclusion still holds on
v4 — no `self_team` `NUMERIC_LEGAL` or `CATEGORY_SECONDARY` family exists in the v4 artifact —
but this ledger does not demonstrate the instrument is live there, so v4's half of S3 inherits
golden-v2's liveness check rather than carrying its own.

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

Scope limit, stated rather than glossed: this argument is clean for the self-side
rows only. The opponent's 14 (12 on golden-v2) are attributed **by side-symmetry** — the guard and
`meta.toxic[engine_side]` are one loop over both sides (`leaf.rs:1230-1269`) and the
`|turn|`-escalation mechanism is side-agnostic — and the symmetry is an inference, counted
separately in Units.

The opponent side has **no analogue of the discriminator, in principle**: for
`Role::Opponent`, `belief = opponent_beliefs.get(&species_key)` (`encoder.rs:1993`), so its
status token comes from `belief.status()` first and could not evidence the engine's
`active.status` even if it had agreed. Worse, `opp_membership` routes every
`opponent_team` column to `epistemic` on 78 boundaries (golden-v2; **94** on v4) (`leaf_vs_reality.py:334-335`), so an
opponent status divergence is masked by construction — which is also what the `epistemic` toxic family (3 rows on golden-v2, **5** on v4) is, a symptom of that sweep rather than an independent anomaly.

## S4 — Encore is seeded at a deliberate floor and never advanced (2 rows)

**[CORRECTION] — an earlier revision said root Encore states "fail-close" and that an
in-branch Encore "cannot tick", disposition "classifier fix". Both were wrong.**

Root Encore states do **not** fail-close in general. `engine_world.py:1292-1304` raises only
when the locked move index cannot be resolved; boundary `1003#[8,9]` p1 has
`self_active_volatiles: ['encore']` at the root and **was compared, not skipped**. What that
path does is `engine_world.py:1324` — `volatile_durations["encore"] = 1` — documented at
`:1316-1322` as "a deliberate floor … the true elapsed count is not observable from the
request … Deriving the real value from observation history is follow-up work". Root elapsed is
1, reality's leaf is 2, the leaf reads 1: consistent with a seeded floor that does not advance,
not with non-application.

**And it must not be folded into `engine_model`.** The two partitions have **opposite
signatures**: the `engine_model` example is got 0.0 / want 1/6 (Encore never applied), the
`state` pair is got 1/6 / want 2/6 (applied, not advanced). Routing the second into the first
via a `classify()` change — the previous revision's proposal — would bury a distinct mechanism
under an already-accepted excuse.

The tag observation stands and is worth keeping: the `encore` tag is a literal `"|Encore"` line
match (`leaf_vs_reality.py:612-617`) routed at `:355`, so a mon already under Encore before the
replayed window emits no line and falls to the `state` fallback (`:383`). That explains which
bucket these rows land in; it is not the cause of the value being wrong.

**Disposition: encoder fix** — advance the seeded Encore counter, which `engine_world.py`
already names as follow-up work.

## S5 — the self-side recharge root-freeze (2 rows)

`CATEGORY_VOLATILE_OFFSET` (self) diverges with vocab id **877 = `volatile:mustrecharge`**
(decoded from `corpus/encoder_tables_v4.json`): got 877, want 0 — the leaf's own mon carries
`mustrecharge` where reality does not.

This is the A1 split `docs/leaf_observation_column_map.md` documents: the opponent side is
refreshed live from the branch's `volatile_statuses` while the **self** side is root-frozen
deliberately, because the live producer returns the opponent slot or nothing and never ours.

**[CORRECTION]** An earlier revision cited `engine_search.py::_recharging_slots` as the
producer for this measurement. That is production's producer, not this harness's:
`leaf_vs_reality.py:430-439` derives `recharging` for **both** slots from the recorded chosen
candidate and passes it at `:450`. The ledger was quoting the column map's own caveat that
these gates "build a different world than production" while using the production path to
attribute a gate measurement. The A1 attribution survives — the frozen self-side volatile is
what the column reads — but the citation was wrong, and task 4's gates-first ordering exists
precisely because of this discrepancy.

**Disposition: production + gate fix → task 4.** These 2 rows are the measured incidence.

**[CORRECTION]** The previous revision claimed 8 rows for this cause, folding in the five
action-surface rows and the opponent volatile row. Both are separate mechanisms — S6 and S5b.
`CATEGORY_VOLATILE_OFFSET` (self) carries `rows=2` and the artifact stores one example per
family, so the second row's boundary is unverified; the id-877 reading rests on the example.

## S5b — the recharge consumed a ply early, escaping its tag seat-locally (1 row)

`CATEGORY_VOLATILE_OFFSET` (opponent) at `1009#[18,19]`, seat **p1**, direction **inverted**:
got 0 / want 877 — the leaf *lost* a recharge reality still has. Reality at round 18 p1 is
`request_kind: force_switch` with `opponent_must_recharge=True`.

**[CORRECTION]** The previous revision called this "the opponent mirror" of S5. It is not: a
different seat, the opposite direction, and a different mechanism — the harness's own
documented deviation, "the engine consumes the recharge one ply early on **faint-replacement
plies**" (`leaf_vs_reality.py:660-663`). It escapes the `recharge` tag only because `:664-675`
inspects the **seat's own** target-row candidates and p1's row 19 carries no `recharge`
candidate.

**Disposition: accepted `engine_model` deviation plus a tag fix** — widen the tag's lookup
beyond the seat's own candidates. Not task 4.

## S6 — the world seeds benched mons with the previous stint's Choice lock (5 rows)

**[CORRECTION] — the previous revision filed these five rows under the recharge cause. The
corpus refutes that.** At `1009` round 17 seat p2, reality has `self_must_recharge = False`,
the active mon is **Slaking holding `choiceband`**, and `self_last_used_move = hyperbeam`. The
mask is `[F,F,T,F,T,T,T,T,F]` — exactly the Choice-locked move legal plus switches. Round 15
(fresh switch-in) is all-legal and round 16 is the recharge request, so the round-17 shape is a
**Choice lock**, not a recharge. A recharge mechanism also cannot explain why the three
non-Hyper-Beam moves diverge while hyperbeam does not.

`leaf.rs:1638-1646` already names this bug family: "the world constructor seeds benched mons
with their LAST STINT's cached disabled bits (the payload caches per-mon move state) and the
engine never re-enables them on a branch switch — present the request semantics instead
(leaf-vs-reality repro: **Choice-Band Nidoking fresh switch-in shows all four moves legal**)."

**Consequence for task 4, stated because the previous revision got it backwards:** making
`_recharging_slots` symmetric would clear S5's 2 rows and leave these 5 untouched — the leaf's
engine surface already has no `MUSTRECHARGE` at that leaf. The previous claim that "these 8
rows are the measured incidence task 4 should close" was false.

**Disposition: encoder fix.** This remains the highest-severity family: the only one where the
leaf is *permissive* rather than empty, and a search pricing illegal actions is worse than one
reading a stale counter.

## S7 — the Sleep-Talk turn refund is not modelled at the leaf (2 rows)

**[CORRECTION] — the previous revision blamed `approximate_sleep_turns=True`. That instrument
does not feed this column and predicts the wrong direction.**

`NUMERIC_SLEEP_TURNS` reads `exact.sleep_turns()` (`encoder.rs:2329`), which is the JSON ledger
field `sleep_turns` (`encoder.rs:1464`), written by `leaf.rs:1461-1470` as `base + count` where
`base` is the **root ledger's** value — the comment there says "Root sleepers keep their ledger
base." `approximate_sleep_turns` only seeds poke-engine's internal wake-RNG counter
(`engine_world.py:1735-1738`) and cannot move this cell. It also predicts the wrong sign: a
`sleep_turns=0` approximation makes the leaf **lower**, while the measurement is got 0.4 (2/5)
against want 0.2 (1/5) — the leaf is **higher**.

The real cause, checked on the cited boundary. At `1011#[60,61]`, p1's benched Arbok has
`sleep_turns=2, sleep_skipped_turns=1`, and the branch contains
`|switch|selfa: Arbok|Arbok, L85, M|241/241 slp`. Gen3 **refunds** the turns spent on Sleep
Talk / Snore when the sleeper pivots in (`time += skippedTime` in `slp.onSwitchIn`), which
`belief.py:1526-1533` applies as `sleep_turns = max(0, sleep_turns - sleep_skipped_turns)` —
so reality drops to 1. `LeafMeta.sleep` (`leaf.rs:322-329`) is keyed
`(started, cant_count)` with **no skipped term**, so the leaf cannot apply the refund and
carries the root's 2.

**Disposition: encoder fix**, and the `state` class it landed in is **correct** — the previous
revision's "an artifact of the comparison setup, not an encoder defect" removed a genuine
defect from the defect list on a mechanism that cannot produce it.

Both rows are the **same Arbok seen from the two seats**, one event, directly checkable — not a
side-symmetry inference, which the previous revision also claimed.

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
