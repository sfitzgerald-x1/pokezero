# C112 v3 — source-level cause and disposition for every `state`-class row in the leaf differential

> **v3 covers the corpus the task named, which v2 did not.** v2 ledgered `golden-v2`
> (106 boundaries, v3 schema) and the scenarios corpus (2), and presented that as the
> ledger of "the 124". The 124 comes from a **v4** corpus, and v2's class set is a strict
> SUBSET of it: four families were entirely unattributed — `NUMERIC_SLEEP_TURNS`
> (self + opponent) and `CATEGORY_VOLATILE_OFFSET` (self + opponent). I substituted a
> corpus that produced a tidier number. v3 measures all three corpora and groups all 18
> families / 138 rows into six mechanisms. **110 of the 138 rows carry a source-level cause
> and a disposition; 28 (P2, toxic) carry a measurement, a pinned failure mode and an open
> mechanism.** So the task's "a cause per class" is delivered for five of six classes and
> NOT for P2 — said here because the header is the part that gets quoted, and an earlier
> revision of this line claimed "no class is left open" while the body withdrew it.
>
> v3 closes the two classes v2 left open (encore, now P1; the action surface, now P5) rather than
> recording a next check, because the task's disposition vocabulary is
> {encoder fix, harness fix, demonstrated-unreachable with a pool check} and "open" is not
> in it. Dispositions name the **owning file** instead, because `harness fix` / `encoder fix` was
> applied inconsistently and read four production changes as test-only work. Five of six name a
> file; P2 names none, because its mechanism is not established.

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
produced each run rather than one date for all three.
Both poke-engine build artifacts were freshly re-vendored (#1119 patched the vendored `gen3/choice_effects.rs`; a stale tree
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

- A **row** is one `(array, block, column)` family **at one boundary** — a family x boundary
  incidence (`leaf_vs_reality.py:892-893`), which is why 18 families carry 138 rows. The `state` families sum to
  **119 rows** (`turn` contributes none) on golden-v2 and **10** on scenarios.
- `class_rows.state` counts **BOUNDARIES**: **106** on golden-v2, **2** on scenarios.
- **The mapping between them is not derivable from the artifact.** Families are
  incremented once per boundary (`leaf_vs_reality.py:892-893`) and no per-boundary family
  list is stored, so the distinct boundaries any group of families covers is only bounded
  — for v1's S1+S2 group, to `[23, 88]`. v1's headline "88 of 106" divided rows by
  boundaries and is withdrawn. Any future coverage claim in boundaries needs the harness
  to record per-boundary family sets first; that is a harness change and out of scope here.

Coverage on the v4 corpus (`corpus/golden-v4`): the harness surfaces **18 `state` families / 138
rows**, and this ledger groups all 138 into six mechanisms — P1 100 + P2 28 + P3 2 + P4 1 + P5 5 +
P6 2 = 138. **110 rows carry a source-level cause and a disposition** (P1, P3, P4, P5, P6). **28 do
not** — P2, toxic, which carries a measured rate, a pinned failure mode and an open mechanism.

Every attributed row rests on source-level verification, subject to P3's row-level caveat stated in
that section (its second row's boundary is unverified, and the id-877 reading rests on the one
stored example). The ledger's one remaining **inference** sits *inside* the open class: P2's
opponent 14 would inherit the self half's cause by side-symmetry (`leaf.rs:1230-1269`) once that
cause exists.

**[CORRECTION]** Two earlier revisions stated a verified/inferred split — "110 source-verified and
28 … (14 rows, and 1 row)", then "123 / 15", then "124 source-verified, 14 resting on
side-symmetry". All three are withdrawn. The last was the worst: it counted P2's self-side 14 as
verified two paragraphs after declaring P2 unattributed, and described the other 14 as an inference
from a cause that does not exist. **No split is stated over unattributed rows**, and no row count
of 124 is **asserted** anywhere — the numeral appears only in this withdrawal and as the corpus's
BOUNDARY count. 110 + 14 colliding with it was a coincidence, not a measurement. Four consecutive edits to this paragraph each fixed the wrong
sentence and left the composition unread.

**Note on "the 124".** 124 is `class_rows.state`, a count of BOUNDARIES; 138 is the sum of
per-family boundary incidences, which this document calls rows. They are different units and
the Units section forbids juxtaposing them, so the coverage claim is stated over rows only.

On `golden-v2`: **119 of 119 rows** — P1 (**90** = 17+17+4+4+23+23+2), P2 (24) and P5 (5). P3, P4 and P6 do not surface
there. Scenarios: **10 of 10**, all **P5**, the same recharge-request Choice lock, corroborated on
its own boundary.

The verified/inferred split is carried rather than merged because that is how a ledger of
this kind gets over-read — and its predecessor was withdrawn for exactly that.


## The table

| class | rows (v4 / gv2 / scen) | cause | disposition |
|---|---|---|---|
| `NUMERIC_SELF_WISH_TURNS` | 17 / 17 / 0 | **P1** root-frozen md passthrough | harness fix |
| `NUMERIC_OPP_WISH_TURNS` | 17 / 17 / 0 | **P1** | harness fix |
| `NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF` | 6 / 4 / 0 | **P1** | harness fix |
| `NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP` | 6 / 4 / 0 | **P1** | harness fix |
| `NUMERIC_STALL_COUNTER` (self) | 26 / 23 / 0 | **P1** | harness fix |
| `NUMERIC_STALL_COUNTER` (opponent) | 26 / 23 / 0 | **P1** | harness fix |
| `NUMERIC_ENCORE_TURNS` | 2 / 2 / 0 | **P1** | harness fix |
| `NUMERIC_TOXIC_STAGE` (self) | 14 / 12 / 0 | **P2** line replay did not escalate | harness fix |
| `NUMERIC_TOXIC_STAGE` (opponent) | 14 / 12 / 0 | **P2** | harness fix (by symmetry) |
| `CATEGORY_VOLATILE_OFFSET` (self) | 2 / 0 / 0 | **P3** self-side recharge root-freeze | production + gate fix → **task 4** |
| `CATEGORY_VOLATILE_OFFSET` (opponent) | 1 / 0 / 0 | **P4** `recharging` never seeded on a faint-replacement round | harness fix |
| `NUMERIC_ACTIVE` (action) | 1 / 1 / 2 | **P5** recharge request carries no `disabled` bits | harness fix |
| `NUMERIC_LEGAL` (action) | 1 / 1 / 2 | **P5** | harness fix |
| `legal_action_mask` action0 | 1 / 1 / 0 | **P5** | harness fix |
| `legal_action_mask` action1 | 1 / 1 / 2 | **P5** | harness fix |
| `legal_action_mask` action3 | 1 / 1 / 2 | **P5** | harness fix |
| `NUMERIC_SLEEP_TURNS` (self) | 1 / 0 / 0 | **P6** Sleep-Talk refund not modelled | encoder fix |
| `NUMERIC_SLEEP_TURNS` (opponent) | 1 / 0 / 0 | **P6**, same mon from the other seat | encoder fix |

P1 100 + P2 28 + P3 2 + P4 1 + P5 5 + P6 2 = **138**.

**Dispositions name the owning FILE, not a `harness fix` / `encoder fix` label.** Those two words
were applied inconsistently in an earlier revision — P1 and P6 are the same file with different
labels, and P5's fix is in production world construction while reading as test-only work. Someone
scheduling off this table would have read four rows as harness work that in fact change the leaf
encoding search uses:

| cause | the fix lands in | production or harness |
|---|---|---|
| P1 | `rust/pokezero-search/src/leaf.rs` (write the md keys) | **production** |
| P2 | not established — see the section | unknown |
| P3 | `engine_search.py::_recharging_slots` + the four gates | **production**, = task 4 |
| P4 | `scripts/leaf_vs_reality.py:430-439` | harness |
| P5 | `engine_world.py` world construction / the leaf's request-shape handling | **production** |
| P6 | `rust/pokezero-search/src/leaf.rs` (`LeafMeta.sleep`) | **production** |

Only P4 is harness-only. Four of the six touch the leaf encoding or the world production shares. `legal_action_mask action2` is
scenarios-only (2 rows) and belongs to P5; scenarios' locked-out set is `action1/2/3` rather
than `action0/1/3` because Hyper Beam sits at a different index there.

**[CORRECTION] — the cause structure changed twice under review and this is the third form.**
v2 had ten families and four causes. v3's first form had eight causes, three of which
(`S1` wish/sleep-clause, `S2` stall, `S4` encore) turned out to be **one** cause, and two of
which (`S5b`, `S6`) were attributed to mechanisms the corpus refutes. The 100 rows now under
P1 were previously presented as three unrelated defects with three different fixes.

## P1 — root-frozen metadata passthrough (100 rows)

**This subsumes v3's S1, S2 and S4, which were three sections proposing three fixes for one
mechanism.**

`leaf.rs:1131` is `let mut row = self.root.clone()`, and `md` is
`row["observation_metadata"]` mutated **in place** (`leaf.rs:1142-1145`). So every metadata key
the leaf does not explicitly overwrite keeps the **root's parser value**, frozen, at every
depth.

These seven columns read md keys that `rust/` never writes:

| column | md key it reads | written anywhere in `rust/`? |
|---|---|---|
| `NUMERIC_SELF_WISH_TURNS` / `_OPP_` | `self_wish_turns` / `opponent_wish_turns` | no — `encoder.rs:1289,1297` read only |
| `NUMERIC_SLEEP_CLAUSE_BLOCKS_*` | `*_sleep_clause_blocks` | no — `encoder.rs:1275,1282` read only |
| `NUMERIC_STALL_COUNTER` | `{prefix}_stall_counter` | no — `encoder.rs:2160` read only |
| `NUMERIC_ENCORE_TURNS` | `{prefix}_encore_elapsed` | no — `encoder.rs:2162` read only |

`grep -rn "encore_elapsed\|stall_counter" rust/` returns exactly those two reads and nothing
else. The stall counter and encore elapsed sit in the same little `(md key, column, divisor)`
table at `encoder.rs:2158-2168` as `confusion_elapsed` and `wrap_trap_elapsed`.

**Three specific claims from the previous revision are withdrawn:**

1. *"the keys hold their dataclass default of 0"* (wish/sleep-clause). They hold the **root's**
   value. `got 0.0` on those boundaries is because the Wish was set **in-branch**, not because
   of a default — the encore rows prove passthrough carries real non-zero root values
   (`1003` rounds 6-12 carry `self_encore_elapsed` 0,0,1,2,3,4).
2. *"`side_condition_counts()` omits `protect`, add it"* (stall). `NUMERIC_STALL_COUNTER` reads
   md `{prefix}_stall_counter`, **not** `*_side_condition_counts`, so adding `protect` to that
   export would not move the column. The C112 v2 finding that the engine *carries* the counter
   as `protect` (`state.rs:699`, seeded at `engine_world.py:1265`) still stands and is what
   makes the fix cheap — but the write has to land on the md key.
3. *"Encore is seeded at a deliberate floor and never advanced"*. `engine_world.py:1324`'s
   `volatile_durations["encore"] = 1` seeds poke-engine's own volatile duration, which this
   column never reads. That is the `approximate_sleep_turns` error (P6) under a different name.
   It looked consistent only by coincidence: at `1003#[8,9]` the root's parser
   `self_encore_elapsed` is 1 **and** the seeded floor is 1, so the two are indistinguishable
   at that boundary.

**The "opposite signatures" argument is also withdrawn.** The previous revision argued the
`engine_model` encore family (got 0.0) and the `state` pair (got 1/6) are different mechanisms.
Both are `got == the root's value`: the `engine_model` example is `1003#[6,8]` where the root
carries 0, the `state` example is `#[8,9]` where it carries 1. Same mechanism, different root
number.

The tag observation survives as an explanation of **which class** these land in, not of the
value: the `encore` tag is a literal `"|Encore"` line match (`leaf_vs_reality.py:612-617`)
routed at `:355`, so a mon already under Encore before the replayed window gets no tag and
falls to the `state` fallback at `:383`.

### The class is decidable, and it has three latent members

`get(md, …)` occurs only in `encoder.rs` (43 sites, no other file), so the enumeration is closed:
diff the md keys `encoder.rs` reads against the keys `leaf.rs` rewrites, and the remainder is
root-frozen by construction. That is a decision procedure, not four instances.

| md key | column | on golden-v4 |
|---|---|---|
| `*_wish_turns` | `NUMERIC_{SELF,OPP}_WISH_TURNS` | diverges 17/17 |
| `*_sleep_clause_blocks` | `NUMERIC_SLEEP_CLAUSE_BLOCKS_*` | diverges 6/6 |
| `{prefix}_stall_counter` | `NUMERIC_STALL_COUNTER` | diverges 26/26 |
| `{prefix}_encore_elapsed` | `NUMERIC_ENCORE_TURNS` | diverges 2 |
| `{prefix}_confusion_elapsed` | `NUMERIC_CONFUSION_TURNS` | **latent** — never nonzero in 1271 rows |
| `{prefix}_wrap_trap_elapsed` | `NUMERIC_WRAP_TRAP_TURNS` | **latent** |
| `{prefix}_meanlook_trap` (`encoder.rs:2174`) | `NUMERIC_MEANLOOK_TRAP` | **latent** |

**Membership rests on the closed write set, not on the measurement.** `leaf.rs`'s md writes are 17
literal `md.insert` keys plus `key_sc` / `key_tox`, and nothing post-processes md between
`encoder.encode_leaf` (`leaf_vs_reality.py:594`) and the comparison. The three latent keys are not
in that set, so they are members **by construction**, whatever a corpus shows — the "never nonzero
in 1271 rows" column above is evidence about **incidence**, not about membership. Stating it the
other way round would be the absence-of-signal argument this ledger has already been corrected for
twice.

Positive evidence that those zeros are "nothing happened" rather than a dead instrument: golden-v4
contains **zero** confusion, wrap-family (`wrap/whirlpool/clamp/firespin/bind/sandtomb`) and
mean-look/spider-web events across **31,018** event lines, and the parser has live increment paths
for all three (`showdown.py:2617`, `:2632`, `:5108`, with resets at `:2484`, `:2497`,
`:2504-2505`). Claimed narrowly: the leaf never writes the key. Whether the parser produces correct
values in a game that exercises them is not tested here.

**P1 is the undeclared half of a class the harness already names.** The declared half is the same
mechanism with a rationale attached and its own class, which is why it never reaches `state`:
`self_must_recharge` (that is P3) and the `root_frozen_pack` set — `{prefix}_{truant_loaf,
choice_locked, item_swapped, last_damage_dealt, last_damage_taken, last_used_move,
arrived_by_baton_pass, traced_ability}` plus the v4 credit block (`encoder.rs:1311-1317`), the
910-boundary `root_frozen_pack` class in the artifact.

`opponent_matchup_switch_evidence` (`encoder.rs:2234`) is also never written in `leaf.rs`, but it
is **not** a P1 member: the matchup pair is fold-written through `ProductsData` since #1118 and
carries its own `matchup_fold` class and allowance.

**Passthrough signature, measured.** A pure passthrough must diverge on exactly the boundaries
where reality's value changes across the pair:

| key | boundaries where reality changes | rows in the artifact |
|---|---|---|
| `stall_counter` (per side) | 27 | 26 `state` + 1 `engine_model` = **27** |
| `wish_turns` (per side) | 18 | 17 (1 skipped) |
| `sleep_clause_blocks` (per side) | 8 | 6 |
| `encore_elapsed` | 7 | 4, plus 6 `encore_move_unknown` skips |
| confusion / wrap-trap / mean-look | 0 | 0 |

Stall matches to the row. The latent three are latent because the corpus never exercises them,
not because the mechanism differs.

**Disposition: write the md keys at the leaf — all SEVEN key families, in
`rust/pokezero-search/src/leaf.rs`.** Fixing only the four that currently diverge leaves the bug to
return the first time a Confusion, Wrap or Mean Look game enters a corpus. All four values are available: the wish counter is `side.wish.0`
(`state.rs:1208`), the stall counter is `side_conditions.protect` (`state.rs:699`), and the
sleep-clause predicate is derivable from the same sleeper bookkeeping `leaf.rs:1254-1261`
already computes for `*_sleep_clause_used`.

**Caveat carried from C112 v2, still true:** the engine's `protect` is a **side** condition and
does not reset on switch-out the way the parser's per-active counter does, so writing it is
exact only within a stint.

## P2 — the toxic write is live and 91% correct; the residual mechanism is NOT established (28 rows on v4, 24 on gv2)

**[CORRECTION] — earlier revisions titled this "the line replay did not escalate" and gave a
disposition. Neither is earned, and the measurement that proves the write works also refutes the
headline.**

The parser's toxic stage **changes across 153 (self) / 155 (opponent) boundaries** on golden-v4,
and only **14 / 14** diverge. A root-frozen passthrough diverges on all of them — P1's stall
counter does, 27 of 27 — so the leaf's line-driven write is live and correct on ~91% of the
boundaries where it must move. That is a positive measurement of a moving instrument, which is
what this ledger keeps demanding of itself, and it is why P2 is **not** a P1 member.

But 14 of 153 is not "did not escalate"; it is "fails on 9% of them", and that needs its own
mechanism. **I do not have one.** Two candidate leads, one already refuted:

- *Refuted:* that the synthesized line set lacks the `\|turn\|` boundary the parser escalates on.
  `synthesized` is `branch["events"]` rendered by **production's** event renderer
  (`leaf_vs_reality.py:558` → `envstep.rs:198` → `events.rs`), and `events.rs:1088,1112` **does**
  emit `\|turn\|N+1` — though only when the ply completes the battle turn (`:1065-1072`). An
  earlier revision asserted this as the cause; it is production's stream either way, so the
  mechanism could not have been harness-confined.
- *Refuted by measurement:* that the failures sit on boundaries whose round pair is non-adjacent
  (a faint/force-switch ply that does not complete the turn). Traced every toxic-column divergence
  on golden-v4: **26 of 33 sit on ADJACENT same-seat pairs**, 7 on gapped ones. Definition, because the number is
otherwise unreproducible: consecutive same-seat rows differ by **+1** in `decision_round_index`
(both seats share the index), and a pair can still skip a round when only the other seat acted. My
first pass used +2 as "adjacent" and would have reported the opposite conclusion. (33 = the 28
  `state` rows plus the 5 `epistemic` opponent rows, which the column-name trace cannot separate.)
  So non-adjacency does not explain it.

### The failure mode IS pinned, even though the cause is not

**`got = 0.0` in every stored example** — all three toxic families, across all three classes
(`state/self_team` 14 rows, `state/opponent_team` 14, `epistemic/opponent_team` 5), report
`got 0.0 / want 0.1333` (= 2/15). A missed *escalation* of a live counter would give
`got = want − 1`; a counter that never started, or was cleared, gives `got = 0`. **So the residual
is in the start/clear paths, not the escalation path** — which is why no variant of "did not
escalate" belongs in this section's title, and why the two refuted leads above were both looking in
the wrong place.

**Named next check, grep-shaped.** The two implementations' arithmetic matches (both invert an exact
residual as `damage / (max/16)`: `leaf.rs:494-533` against `showdown.py:2944-2963`). The one
structural difference is **`toxic_stage_known`**, a tri-state the parser carries and gates *both* its
escalation (`showdown.py:2586`) and its inference (`:2944-2963`, `:1998-2000`) on, distinguishing
"stage 0" from "stage unknown". `LeafMeta` has `toxic`, `active_toxic` and `toxic_reentry_pending`
and **no counterpart**.

**And one case study that does NOT resolve, recorded so nobody adopts a story that merely fits.**
At `1000#[56,57]` p1 the root's active is Latias at stage 0 while the leaf's is Volbeat, toxiced at
round 48, switched out, re-entering inside the branch (`|switch|selfa: Volbeat|…|72/275 tox` then
`|-damage|…|4/275 tox|[from] psn`). Reality reports **2** — but 72 → 4 of 275 is damage 68, unit 17,
`68 % 17 == 0`, i.e. stage **4** under both residual-inversion rules. So reality's 2 comes from a
path neither implementation's arithmetic explains, and "the leaf cannot restore a re-entering mon's
stage" is a story that *fits* rather than a cause. Adopting it would be the encore-floor error again,
which looked consistent only because the root happened to hold the same number.

**Disposition: undetermined.** **Task acceptance note: this is the one class of the six without a
source-level cause — 28 of 138 rows.**

### The guard is excluded



The leaf's toxic stage is deliberately line-driven, not engine-derived (`leaf.rs:298-302`), and
the parser escalates on `\|turn\|` lines only. Two candidate mechanisms: the replay never
escalated, or the guard at `leaf.rs:1265-1267` zeroed a correct stage when
`active.hp <= 0 || active.status != PokemonStatus::TOXIC`.

**The guard is excluded**, via instruments that can actually move:

- the `hp <= 0` arm rides **`NUMERIC_LEGAL`** (`encoder.rs:2282-2285`), whose `condition.fainted`
  comes from the leaf's own condition string, written by the *same expression* as the guard —
  `if hp <= 0 { return "0 fnt" }` (`leaf.rs:195-198`) against `active.hp <= 0` (`leaf.rs:1266`).
- the status arm rides **`CATEGORY_SECONDARY`** (`encoder.rs:2258-2262`), which for self-team
  tokens is the engine's own field and not the belief ledger, because `belief` is `None` for
  `Role::SelfTeam` (`encoder.rs:1992-1994`).

Both are **absent from `self_team` under EVERY class**, not just `state` — checked, because a
`CATEGORY_SECONDARY` divergence would route to `ledger_skew` or `engine_model` under other tags.

**[CORRECTION]** An earlier revision cited `NUMERIC_PRESENT` and the self-side `attention_mask`
as the liveness instruments. Both are **constants** (`encoder.rs:2286`, `:1110-1112`) and can
never diverge, so reading their absence measured nothing.

**Scope, twice over.** The argument is clean for the self side; the opponent's 14 rows are
attributed **by side-symmetry** (one loop over both sides, `leaf.rs:1230-1269`, side-agnostic
mechanism), and the opponent has no analogue in principle because its status token comes from
`belief.status()` first (`encoder.rs:1993`) while `opp_membership` sweeps opponent columns into
`epistemic` on 94 v4 boundaries. And the `NUMERIC_LEGAL` liveness evidence is a **golden-v2**
measurement (2055 zeros of 6168 cells, 787 of 1028 rows) not re-run on v4; the conclusion holds
there (no such family in the v4 artifact) but v4's half inherits gv2's liveness check.

**Disposition: harness fix.**

## P3 — the self-side recharge root-freeze (2 rows)

`CATEGORY_VOLATILE_OFFSET` (self) diverges with vocab **877 = `volatile:mustrecharge`**: got 877,
want 0 — the leaf's own mon carries it where reality does not. This is the A1 split
`docs/leaf_observation_column_map.md` documents: the opponent side is refreshed live from the
branch's `volatile_statuses`, the self side is root-frozen because the live producer never
returns our slot.

**[CORRECTION]** An earlier revision cited `engine_search.py::_recharging_slots` as the producer
for this measurement. That is production's; this harness derives `recharging` for **both** slots
from the recorded chosen candidate (`leaf_vs_reality.py:430-439`, passed at `:450`). The ledger
was quoting the column map's caveat that these gates "build a different world than production"
while using the production path to attribute a gate measurement.

`CATEGORY_VOLATILE_OFFSET` (self) carries `rows=2` and the artifact stores one example per
family, so the second row's boundary is unverified; the id-877 reading rests on the example.

**Disposition: production + gate fix → task 4**, and these 2 rows are its measured incidence —
**not** the 8 an earlier revision claimed.

## P4 — `recharging` is never seeded on a faint-replacement round (1 row)

`CATEGORY_VOLATILE_OFFSET` (opponent) at `1009#[18,19]`, seat p1, direction **inverted**: got 0 /
want 877 — the leaf *lost* a recharge reality still has.

**[CORRECTION]** An earlier revision called this "the opponent mirror" of P3 and then "the engine
consumes the recharge a ply early". Neither: the recharge was **never in the world to consume**.
`leaf_vs_reality.py:430-439` looks up `decisions[(battle_id, round_n, slot)]` for both slots at
the root round and tests that row's chosen candidate — and battle 1009 has **no p2 decision row
at round 18** (round 18 is p1's force-switch after a faint). So `recharging == ()`, `MUSTRECHARGE`
never enters the constructed world, and `leaf.rs:1320-1323` correctly reports the absence of a
volatile that was never seeded. Slaking's Hyper Beam was at round 17, before the root, so the
branch cannot set it either.

**Disposition: harness fix** — seed `recharging` from the opponent's most recent decision row, or
from the payload's `opponent_must_recharge`, rather than from a same-round chosen candidate that
does not exist on faint-replacement rounds. Filing this as an accepted deviation by widening a
tag, as the previous revision proposed, would route a harness world-construction gap into an
excuse class — the move P1 repudiates.

This is the **third** finding where the harness's `recharging` derivation is the thing at fault
(P3's citation correction, this cause, and the gate caveat task 4 exists to fix).

## P5 — the recharge request carries no `disabled` bits, so the world loses a Choice lock (5 rows on v4, 5 on gv2, 10 on scenarios)

**[CORRECTION] — an earlier revision filed these under the recharge cause, then under "the world
seeds benched mons with the previous stint's Choice lock". Both are wrong.**

Reality at `1009` round 17 seat p2: `self_must_recharge = False`, active **Slaking holding
`choiceband`**, `self_last_used_move = hyperbeam`, mask `[F,F,T,F,T,T,T,T,F]` — exactly the
locked-into move legal plus switches. So the shape is a **Choice lock**, and a recharge mechanism
cannot explain why the three non-Hyper-Beam moves diverge while hyperbeam agrees.

The instrument, chased rather than assumed: `engine_world.py:1885-1890` reads the request's active
move rows into `known_pp[id] = (pp, disabled)`, carried into `MoveSpec` at `:1946,:1967`;
`leaf.rs:1663` reads that per-move `disabled` bit. Dumping `public_materialization` for 1009 p2:
at **round 16 (the root, a recharge request)** all four moves are `"disabled": false`; at **round
17 (reality's leaf)** three are `true` and hyperbeam `false`.

So the world is seeded **truthfully** from the root's own request — the recharge request simply
reports every move enabled — and reality re-asserts the lock one request later. The constructed
world cannot re-derive it (`use_last_used_move` is off, `leaf.rs:331-334`).

Slaking was the **active** mon at the root, seeded from its own current request, so nothing is
benched and nothing is stale-per-stint. `leaf.rs:1638-1646`, which the previous revision cited as
already naming this bug, is the **opposite polarity**: it describes stale bits making a fresh
switch-in too *restrictive*, and its remedy (`if fresh_switch_in { false }`) would make this row
worse.

**Corroboration, which also relocates the scenarios rows.** Scenarios is the same mechanism, not
P3: `golden-scenario-hyperbeam_recharge-91000#[2,3]` p1 has a recharge request at the root round 2
and a round-3 mask of `[1,0,0,0,1,0,0,0,0]` with `choiceband` — only the locked-into move legal.
Two corpora, two recharge-request roots, and in both the diverging slots are exactly the
locked-out moves while the locked-into move agrees. The index difference (`action0/1/3` on v4 vs
`action1/2/3` on scenarios) is just where Hyper Beam sits in each move list.

**Disposition: harness fix.** Highest-severity family: the only one where the leaf is
*permissive* rather than empty, and a search pricing illegal actions is worse than one reading a
stale counter.

## P6 — the Sleep-Talk turn refund is not modelled at the leaf (2 rows)

**[CORRECTION] — an earlier revision blamed `approximate_sleep_turns=True`. That instrument does
not feed this column and predicts the wrong direction.**

`NUMERIC_SLEEP_TURNS` reads `exact.sleep_turns()` (`encoder.rs:2329`) = the JSON ledger field
(`encoder.rs:1464`), written by `leaf.rs:1461-1470` as `base + count` with `base` the **root
ledger's** value ("Root sleepers keep their ledger base"). `approximate_sleep_turns` only seeds
poke-engine's internal wake-RNG counter (`engine_world.py:1735-1738`). It also predicts the wrong
sign: a `sleep_turns=0` approximation makes the leaf **lower**, while the measurement is got 0.4
(2/5) against want 0.2 (1/5) — **higher**.

Real cause, checked at `1011#[60,61]`: p1's benched Arbok has `sleep_turns=2,
sleep_skipped_turns=1` and the branch switches it in. Gen3 **refunds** turns spent on Sleep Talk /
Snore on pivot-in (`time += skippedTime` in `slp.onSwitchIn`), applied at `belief.py:1526-1533` as
`max(0, sleep_turns - sleep_skipped_turns)`, so reality drops to 1. `LeafMeta.sleep`
(`leaf.rs:322-329`) is `(started, cant_count)` with **no skipped term**, so the leaf cannot apply
the refund and carries the root's 2.

**Disposition: encoder fix**, and the `state` class is **correct** — the previous revision's
"artifact of the comparison setup, not an encoder defect" removed a genuine defect from the defect
list on a mechanism incapable of producing it.

Both rows are the **same Arbok from two seats**, one event, directly checkable — not a symmetry
inference.

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
Four of the `state` families as they stood in v2's framing are columns that did not exist then —
`NUMERIC_SLEEP_CLAUSE_BLOCKS_*` (2026-07-20), `NUMERIC_STALL_COUNTER` and
`NUMERIC_SELF_WISH_TURNS` and `NUMERIC_ENCORE_TURNS` (2026-07-21).

**Consequences, applied in `docs/leaf_observation_column_map.md`:**

1. The published closure rows are **RETRACTED for both corpora**, not corrected. No
   per-class delta (`fold 440→422` etc.) is attributable to the harness fix.
2. The re-derived rows are published as a **new measurement at their own era**, not as a
   correction of the old ones.
3. **v1's causal story for the published `state = 0` is also wrong.** A run that skipped
   100% of boundaries cannot emit `fold 440, epistemic 322, engine_roll 313`. The published
   zero is explained by four of the families as v2 framed them not being columns yet — a corpus/schema
   cause, not the harness break.

## What this ledger does not claim

- Coverage is in **rows**, and the row→boundary mapping is not derivable from the artifact.
- **One class has no cause: P2, 28 rows.** The other five do, with the owning file named.
  P3 points at task 4 of this goal. No disposition is a `classify()` change — an earlier revision
  proposed one for encore and it was withdrawn when that cause turned out to be P1.
  `demonstrated-unreachable` is used nowhere; the one place v2 claimed it, it was false.
- **One** disposition rests on side-symmetry: P2's opponent half, 14 rows. Counted separately in
  Units and marked in the table. Two earlier revisions said "three … " and then listed two.
- Documentation only: no classifier, encoder or harness change.
- **[CORRECTION]** v1 wrote "got 0.0, want 0.5 on all 34 rows". The harness stores **one
  example per family** (`leaf_vs_reality.py:874-875`), so direction is *inferred from
  source* for P1 and P2, not measured per row. The uniformity claim is withdrawn.
- **[CORRECTION]** v1 offered `matched + diverged == compared` as evidence. `compared` is
  *defined* as `exact + divergent` (`leaf_vs_reality.py:972`), so the identity is a
  tautology and no harness assertion exists. The non-vacuous identity —
  `sum(all counts) == boundaries` — is not checked by the harness either; verified by hand
  here (1008 and 369) and it holds. Mechanizing it is task 3's subject.
- `self_moveset_mismatch` (11 scenarios skips) and the residue-row classes are owned by the
  fallback-burndown and rust-fidelity lanes; they appear only as skip counts and are **not**
  attributed. A note belongs in those ledgers.
