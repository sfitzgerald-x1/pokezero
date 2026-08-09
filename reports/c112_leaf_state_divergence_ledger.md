# C112 v3 — source-level cause and disposition for every `state`-class row in the leaf differential

> **v3 covers the corpus the task named, which v2 did not.** v2 ledgered `golden-v2`
> (106 boundaries, v3 schema) and the scenarios corpus (2), and presented that as the
> ledger of "the 124". The 124 comes from a **v4** corpus, and v2's class set is a strict
> SUBSET of it: four families were entirely unattributed — `NUMERIC_SLEEP_TURNS`
> (self + opponent) and `CATEGORY_VOLATILE_OFFSET` (self + opponent). I substituted a
> corpus that produced a tidier number. v3 measures all three corpora and groups all 18
> families / 138 rows into six mechanisms. **All 138 rows carry a source-level cause
> and a disposition.** P2 (toxic, 28 rows) was open in an earlier revision and is closed here: the
> event renderer emits status-free condition strings, so the leaf's replay cannot see that a mon
> switching in mid-branch is badly poisoned. Evidenced by a counterfactual (re-injecting the status
> suffix makes the case study exact) and a 33/33 signature on v4, 27/27 on golden-v2. Three earlier
> attributions for this class are withdrawn inside P2, including two of my own.
>
> v3 closes the two classes v2 left open (encore, now P1; the action surface, now P5) rather than
> recording a next check, because the task's disposition vocabulary is
> {encoder fix, harness fix, demonstrated-unreachable with a pool check} and "open" is not
> in it. Dispositions name the **owning file** instead, because `harness fix` / `encoder fix` was
> applied inconsistently and read four production changes as test-only work. All six name a file.
> P2's is `events.rs`, and its section states plainly that the fix is a production behaviour change
> with named consumers and two contract assertions — not a small one.

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
P6 2 = 138. **All 138 rows carry a source-level cause and a disposition.** P2 (toxic, 28 rows) was open in an
earlier revision and is closed here: the event renderer emits status-free condition strings
(`events.rs`'s `fn hp_condition`), so the leaf's replay cannot see an in-branch toxic entry. Closed on a
counterfactual plus a 33/33 signature on v4 and 27/27 on golden-v2, with three earlier attributions
for the class withdrawn.

> **AT HEAD THESE ARE 16 FAMILIES / 135 ROWS.** P4's 1 row closed in #1156 and P3's 2 closed with
> the freeze lift; both families are gone entirely. The figures above are the ledger-time
> attribution and are kept because the attribution itself is what this document is for — but do
> not quote them as current coverage. A previous correction updated the BOUNDARY count
> (124→123→122) and left these row counts untouched, which is the unit this document actually
> states coverage in.

> **`leaf.rs` LINE CITATIONS IN THIS DOCUMENT ARE ERA-STAMPED AND HAVE DRIFTED.** The P3
> freeze lift changed that file by **+5 lines** net, so every citation below the edit
> (~1290) is low by roughly that much; citations above it are unaffected. They are left as
> written rather than renumbered, because renumbering is what keeps failing: three separate
> line references in this effort went stale **in the commit that introduced or fixed them**.
> Resolve by symbol or by grepping the quoted expression, not by line.

Every attributed row rests on source-level verification, subject to P3's row-level caveat stated in
that section (its second row's boundary is unverified, and the id-877 reading rests on the one
stored example). P2's opponent 14 are covered by the same mechanism as the self 14 — all 33
divergent toxic cells share the in-branch-switch signature — so no side-symmetry inference remains
anywhere in this ledger.

**[CORRECTION]** Two earlier revisions stated a verified/inferred split — "110 source-verified and
28 … (14 rows, and 1 row)", then "123 / 15", then "124 source-verified, 14 resting on
side-symmetry". All three are withdrawn. The last was the worst: it counted P2's self-side 14 as
verified two paragraphs after declaring P2 unattributed, and described the other 14 as an inference
from a cause that does not exist. **No split is stated over unattributed rows**, and no row count
of 124 is **asserted** anywhere — the numeral appears only in this withdrawal and as the corpus's
BOUNDARY count. 110 + 14 colliding with it was a coincidence, not a measurement. Four consecutive edits to this paragraph each fixed the wrong
sentence and left the composition unread.

**Note on "the 124", and why it is now 122.** 124 is `class_rows.state`, a count of BOUNDARIES;
138 is the sum of per-family boundary incidences, which this document calls rows. They are
different units and the Units section forbids juxtaposing them, so the coverage claim is stated
over rows only.

**The number moved twice after this ledger was written, both times because a cause here was
closed.** Do not read 124 as a live figure:

| when | `class_rows.state` | why |
|---|---|---|
| at ledger time | **124** | as attributed below |
| after #1156 | 123 | P4 closed — the candidate-derived block this ledger cites as P4's fix site was replaced wholesale |
| after the P3 freeze lift | **122** | P3 closed — see its section |

Measured at the tree that carries this correction, `python scripts/leaf_vs_reality.py --corpus
corpus/golden-v4 --tables corpus/encoder_tables_v4.json`:
`DEFECT-CLASS (state+turn) divergent boundaries: 122`.

Lifting the P3 freeze also cleared 2 of the 3 rows in the `engine_model` class's
`self_team/CATEGORY_VOLATILE_OFFSET` family; `engine_model` boundaries went 31 -> 30.

**CORRECTED — my first explanation of this was invented.** I wrote that those rows were
"attributed to `volatile:encore` (864) but were mustrecharge displacing the volatile written at
that offset". Review dumped every cell in the family and it is false twice over:

```
MAIN (5 cells)                                    BRANCH (1 cell)
  engine_model 1003 p1 12->13  got 864 want 0       engine_model 1003 p1 12->13  got 864 want 0
  engine_model 1009 p2 15->16  got 0   want 877
  engine_model 1009 p2 17->19  got 0   want 877
  state        1009 p2 16->17  got 877 want 0
  state        1009 p2 19->20  got 877 want 0
```

The two cells that left `engine_model` were **never 864** — they are `got=0 / want=877`, the leaf
writing nothing where reality had the flag. The one genuine encore cell is the one that
**survived**. Nothing was displaced, and `encoder.rs:1638-1641` sorts the bag by raw name, so
`"encore" < "mustrecharge"` and mustrecharge can never displace encore from `offset+0` anyway.

I read the family's printed exemplar — `got 864`, which is the survivor — and generalised it to
all three. That is the same error C141 had already had to correct one report earlier.

**What actually happened, which is the better result:** the lift fixes 4 cells in BOTH
directions — 2 spurious (a stale flag written where reality had none) and 2 missing (no flag
written where reality had one).

**The arithmetic, stated because it otherwise reads wrong:** each class dropped 1 boundary while
its family dropped 2 cells (`state` 123->122, `engine_model` 31->30). One boundary in each pair
still carries another divergent family of the same class, so clearing a cell there frees no
boundary. Classes count boundaries; families count cells.

On `golden-v2`: **119 of 119 rows** — P1 (**90** = 17+17+4+4+23+23+2), P2 (24) and P5 (5). P3, P4 and P6 do not surface
there. Scenarios: **10 of 10**, all **P5**, the same recharge-request Choice lock, corroborated on
its own boundary.

The verified/inferred split is carried rather than merged because that is how a ledger of
this kind gets over-read — and its predecessor was withdrawn for exactly that.


## The table

| class | rows (v4 / gv2 / scen) | cause | disposition |
|---|---|---|---|
| `NUMERIC_SELF_WISH_TURNS` | 17 / 17 / 0 | **P1** root-frozen md passthrough | **production** — `leaf.rs` |
| `NUMERIC_OPP_WISH_TURNS` | 17 / 17 / 0 | **P1** | **production** — `leaf.rs` |
| `NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF` | 6 / 4 / 0 | **P1** | **production** — `leaf.rs` |
| `NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP` | 6 / 4 / 0 | **P1** | **production** — `leaf.rs` |
| `NUMERIC_STALL_COUNTER` (self) | 26 / 23 / 0 | **P1** | **production** — `leaf.rs` |
| `NUMERIC_STALL_COUNTER` (opponent) | 26 / 23 / 0 | **P1** | **production** — `leaf.rs` |
| `NUMERIC_ENCORE_TURNS` | 2 / 2 / 0 | **P1** | **production** — `leaf.rs` |
| `NUMERIC_TOXIC_STAGE` (self) | 14 / 12 / 0 | **P2** renderer emits status-free conditions | **production** — `events.rs` |
| `NUMERIC_TOXIC_STAGE` (opponent) | 14 / 12 / 0 | **P2**, same mechanism | **production** — `events.rs` |
| `CATEGORY_VOLATILE_OFFSET` (self) | 2 / 0 / 0 | **P3** self-side recharge root-freeze | **CLOSED** — gates #1156, freeze lifted in `leaf.rs` (this branch) |
| `CATEGORY_VOLATILE_OFFSET` (opponent) | 1 / 0 / 0 | **P4** `recharging` never seeded on a faint-replacement round | **CLOSED** — #1156 replaced the candidate-derived block |
| `NUMERIC_ACTIVE` (action) | 1 / 1 / 2 | **P5** recharge request carries no `disabled` bits | **production** — `engine_world.py` |
| `NUMERIC_LEGAL` (action) | 1 / 1 / 2 | **P5** | **production** — `engine_world.py` |
| `legal_action_mask` action0 | 1 / 1 / 0 | **P5** | **production** — `engine_world.py` |
| `legal_action_mask` action1 | 1 / 1 / 2 | **P5** | **production** — `engine_world.py` |
| `legal_action_mask` action3 | 1 / 1 / 2 | **P5** | **production** — `engine_world.py` |
| `NUMERIC_SLEEP_TURNS` (self) | 1 / 0 / 0 | **P6** Sleep-Talk refund not modelled | **production** — `leaf.rs` (`LeafMeta.sleep`) |
| `NUMERIC_SLEEP_TURNS` (opponent) | 1 / 0 / 0 | **P6**, same mon from the other seat | **production** — `leaf.rs` (`LeafMeta.sleep`) |

P1 100 + P2 28 + P3 2 + P4 1 + P5 5 + P6 2 = **138**.

**Dispositions name the owning FILE, not a `harness fix` / `encoder fix` label.** Those two words
were applied inconsistently in an earlier revision — P1 and P6 are the same file with different
labels, and P5's fix is in production world construction while reading as test-only work. Someone
scheduling off this table would have read four rows as harness work that in fact change the leaf
encoding search uses:

| cause | the fix lands in | production or harness |
|---|---|---|
| P1 | `rust/pokezero-search/src/leaf.rs` (write the md keys) | **production** |
| P2 | `rust/pokezero-search/src/events.rs` (both options root there — see P2) | **production** |
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
   as `protect` (`state.rs:699`, seeded at `engine_world.py:1286`) still stands and is what
   makes the fix cheap — but the write has to land on the md key.
3. *"Encore is seeded at a deliberate floor and never advanced"*. `engine_world.py:1345`'s
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

## P2 — the event renderer emits status-free condition strings, so the replay cannot see an in-branch toxic entry (28 rows on v4, 24 on gv2)

**Cause.** `events.rs`'s `fn hp_condition` (and `hp_percent_condition`) returns `"0 fnt"`, a percent form, or
`"{hp}/{maxhp}"` and **never appends a status token**. So every condition string in a synthesized
branch is status-free, including the `|switch|` line (`events.rs:1171-1172`). Compare
`1000#[56,57]` p1:

```
synthesized: |switch|p1a: Volbeat|Volbeat, L94, M|72/275          <- no " tox"
             |-damage|p1a: Volbeat|4/275|[from] Spikes
             |-heal|p1a: Volbeat|21/275|[from] item: Leftovers
             |-damage|p1a: Volbeat|4/275|[from] psn
reality:     |switch|p1a: Volbeat|Volbeat, L94, M|72/275 tox
             |-damage|p1a: Volbeat|4/275 tox|[from] Spikes
             |-heal|p1a: Volbeat|21/275 tox|[from] item: Leftovers
             |-damage|p1a: Volbeat|4/275 tox|[from] psn
```

The leaf's `|switch|` handler calls `clear_toxic_meta` (`leaf.rs:763`), then
`update_active_condition`, which sets `active_toxic` only if the condition string carries a status
token (`leaf.rs:546`). It never does, so `active_toxic` stays `false`,
`reseed_toxic_from_residual` bails at its first guard (`leaf.rs:499`), the stage stays 0, and
`|turn|` does not escalate 0.

**Counterfactual — a passing intervention, which is what makes this a cause rather than a story
that fits.** Re-injecting only the ` tox` suffix, changing nothing else:

```
plain    [56,57] token1 got= 0.0     want= 0.1333
patched  [56,57] token1 got= 0.1333  want= 0.1333
```

Corpus-wide the same crude positional restoration takes **21 of 33** divergent toxic cells exact;
the residual 12 are **all** opponent tokens (indices 8-12) the crude patch could not align.

The 21 split **14 self (14/14) and 7 opponent (7/19)**, so the intervention passes on **both
sides**. The opponent rows are therefore *directly demonstrated*, not inferred from the self side —
which is the stronger basis for this ledger carrying no side-symmetry inference.

**Signature, on both corpora.** Every divergent toxic cell sits on a branch containing a `|switch|`
or `|drag|`, and `got = 0.0` in every one:

| corpus | divergent toxic cells | on a branch with `\|switch\|`/`\|drag\|` | `got` |
|---|---|---|---|
| `golden-v4` | 33 (14 self + 14 opp `state`, 5 opp `epistemic`) | **33 / 33** | all 0.0 |
| `golden-v2` | 27 (12 self + 12 opp `state`, 3 opp `epistemic`) | **27 / 27** | all 0.0 |

The gv2 half of this section's row count was an inherited assertion until it was measured; it is now
a measurement, on the second corpus.

**P2 is not a P1 passthrough member**, and the same measurement shows why: the parser's stage
changes across 153/155 boundaries and only 14/14 diverge, so the leaf's line-driven write is live
and ~91% correct where a passthrough would diverge on all of them (P1's stall counter does, 27 of
27). The 5 `epistemic` opponent rows travel with the 28 `state` rows because a column-name trace
cannot separate them by class.

**The leaf already implements the parser's whole algorithm identically** — `|-status|` → 1
(`leaf.rs:748`), exact-residual inversion (`:494-533`), `|turn|` escalation (`:630-631`), and the
rounded-residual gate `toxic_reentry_pending` (`:511-513`) which is exactly the parser's one live use
of `toxic_stage_known` (`showdown.py:2944-2945`). **The delta is the input, not the algorithm.**

### Disposition: encoder fix in `events.rs`, and it is NOT small

Both options are `events.rs`-rooted, and the narrowness runs opposite to what an earlier revision
claimed:

- **Narrower:** emit a switch-in `ActiveStatusTransition`. Those are built at `events.rs:230-238`
  from `ChangeStatusInstruction` only — a switch-in is not a status change, which is why production
  carries this bug too. Touches `LeafMeta` alone. Note the "fix it in `leaf.rs`" phrasing an earlier
  revision used is not possible: `evolve_leaf_meta_with_status_transitions`
  (`leaf.rs:603-608`) takes `(meta, lines, ctx, transitions)` and has **no access to engine state**.

  Two traps for whoever implements it, recorded because this ledger has already shipped a
  disposition that would have changed no row — one that changes a row to the WRONG value is worth
  two sentences. (i) **Ordering:** transitions for offset *k* are applied *before* `lines[k]`
  (`leaf.rs:611-621`), so a transition emitted at the switch line's own offset is wiped by
  `clear_toxic_meta` on that same line; it must be offset+1. (ii) **The existing arm clobbers two
  fields:** `leaf.rs:613-616` sets `toxic = 1` *and* `toxic_reentry_pending = false` — the first
  pre-empts the residual inversion (harmless at `1000#[56,57]` only because the answer happens to
  be 1), the second disables the `/100` re-entry inference at `:511-513`, which is the very
  mechanism this section credits the leaf for implementing correctly. A switch-in needs
  `active_toxic = true` **without** touching either. That is a new arm, not the existing one.
- **Wider:** append the status suffix in `hp_condition`. `render_branch_events` is production's
  leaf-pricing path — `model.rs:972`, whose `rendered.lines` feed `fold.advance_in_place` (`:993`),
  `evolve_self_order` (`:1001`, `:1007`), `evolve_leaf_meta_with_status_transitions` (`:1012`) and
  `encode_leaf` (`:1019`), plus `envstep.rs::env_step`. It also breaks **two in-crate assertions
  that encode the current shape as a contract**: `events.rs:6509-6516` ("Damage lines carry plain
  ASCII cur/max integers (fold input contract)") and `events.rs:6753-6756`
  (`assert_eq!(base, "100", …)` after `split_once('/')`). And `leaf.rs:541-544` documents "no status
  token means 'unchanged', not 'cured'" — once absence can mean "no status", that rule becomes a
  stale-status bug and must be revisited in the same change.

So this is a production behaviour change with named consumers and two contract assertions, not a
one-line encoder fix.

**Harness/production asymmetry, recorded because this section asserts "production":** the pyo3
`LeafEncoder::encode_leaf` (`leaf.rs:2308-2321`) goes through `branch_context(lines)` →
`evolve_leaf_meta` with **no** transitions, while production passes
`rendered.active_status_transitions`. For the 5 `(switch, |-status|)` cells both reach TOXIC by
different routes, so the counts are unaffected — but the harness is not a byte-identical proxy for
production on this path.

### Prior attributions for this class, all withdrawn

Also checked and refuted, recorded because a completeness claim needs them and because the second
is the first thing a reviewer re-asks:

- **The synthesized stream lacks the `\|turn\|` boundary the parser escalates on.** It does not —
  `events.rs:1088,1112` emit `\|turn\|N+1`, though only when the ply completes the battle turn. An
  earlier revision asserted this as the cause; it is production's stream either way, so the
  mechanism could never have been harness-confined. Folded into (1) below as its purported
  mechanism.
- **The failures sit on non-adjacent round pairs** (a faint/force-switch ply that does not complete
  the turn). Measured and refuted: **26 of 33 sit on ADJACENT same-seat pairs**, 7 on gapped ones.
  Definition, because the number is otherwise unreproducible — consecutive same-seat rows differ by
  **+1** in `decision_round_index` (both seats share the index) and a pair can still skip a round
  when only the other seat acted. My first pass used +2 as "adjacent" and would have reported the
  opposite conclusion.

1. *"The line replay did not escalate."* Refuted by measurement: the parser's stage changes across
   153/155 boundaries and only 14/14 diverge, so the write is live and ~91% correct (the
   classification consequence is stated in the cause block above). Its purported mechanism — a
   missing `\|turn\|` — is refuted separately above.
2. *"The residual is the absent `toxic_stage_known` tri-state in `LeafMeta`."* The cited gates
   (`showdown.py:1994-1999`, `:2047`) are inside `_ReplayParser.from_snapshot`, a legacy-snapshot
   fallback that `continue`s whenever the snapshot carries the field — which the only construction
   site always does (`:3639` → `:3660`). Dead for anything this codebase produces: the same error as
   blaming `approximate_sleep_turns` (P6) and the seeded encore floor (P1). The proposed fix would
   also have changed no row, since a boolean cannot turn 0 into 2 when the arithmetic producing the
   value is never reached. And `LeafMeta` has counterparts in both directions —
   `toxic_reentry_pending` and `active_toxic`, the latter documented as whether the active is
   "publicly known to retain badly poisoned status".
3. *"Every toxiced mon reports stage 2 on entry, and reality's 2 comes from a path neither
   implementation's arithmetic explains."* Both halves false. The three cited cases are ordinary
   arithmetic — `|-status|..tox` sets 1 (`showdown.py:4971-4975`), the next `|turn|` escalates to 2
   (`:2585-2587`) — and one of them switched in **clean** via Baton Pass and was toxed afterwards.
   The corpus also holds stage-**1** rows where the decision is sampled before the next `|turn|`, so
   "2" was a sampling artifact. And the unexplained arithmetic was **my own wrong subtraction**:
   72 → 4 is the **Spikes** line (3 layers, 275/4 = 68); the psn residual is two lines later,
   21 → 4 = 17 = one unit → stage 1 (`showdown.py:2951-2963`), then `|turn|` → 2.

### The guard is excluded

The leaf's own zeroing guard (`leaf.rs:1265-1267`,
`active.hp <= 0 || active.status != PokemonStatus::TOXIC`) is **not** what produces these rows, and
the counterfactual confirms it: with the status suffix restored the same guard passes and the leaf
emits 2. The two arms are observable as other columns and both are absent from `self_team` under
every class — the `hp <= 0` arm rides `NUMERIC_LEGAL` (`encoder.rs:2282-2285`), whose
`condition.fainted` comes from the same expression as the guard (`leaf.rs:195-198` against `:1266`);
the status arm rides `CATEGORY_SECONDARY` (`encoder.rs:2258-2262`), which for self-team tokens is
the engine's own field since `belief` is `None` for `Role::SelfTeam` (`encoder.rs:1992-1994`).

**[CORRECTION]** An earlier revision cited `NUMERIC_PRESENT` and the self-side `attention_mask` as
the liveness instruments. Both are **constants** (`encoder.rs:2286`, `:1110-1112`) and can never
diverge, so reading their absence measured nothing.

**Scope of the absence argument, restored — a rewrite deleted it while keeping the argument that
needs it.** Absence is evidence only if the instrument moves, and the proof that `NUMERIC_LEGAL`
moves is a **golden-v2** measurement (2055 zeros of 6168 cells, in 787 of 1028 rows) that was
**never re-run on v4**, so v4's half inherits gv2's liveness check. On the opponent side there is no
analogue in principle: its status token comes from `belief.status()` first (`encoder.rs:1993`), and
`opp_membership` sweeps opponent columns into `epistemic` on 94 v4 boundaries. Stating the lesson
about constants immediately above the place it had stopped being applied is exactly the failure this
`[CORRECTION]` describes.

**So the guard is excluded on two different grounds, not one.** On the **21** cells the
counterfactual takes exact, it is excluded directly — the same guard passes and the leaf emits 2.
On the remaining **12 — all opponent tokens**, where the crude patch could not align, the exclusion
rests on the shared switch signature (27/27 and 33/33) **and on the counterfactual passing for 7
sibling opponent cells (7/19)** — same class of cell, same intervention, guard passes. It does NOT
rest on the absence argument: the caveat above is a **disqualification** for the opponent side, not
a discount, since that side has no analogue in principle. The absence argument stays scoped to the
self side, where it holds.

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

**CLOSED.** The gate half landed in #1156 (`production_recharging_slots`, all four gates) and the
symmetry half in the same PR (`_recharging_slots` returns our own slot from the parser's
`self_must_recharge`). The freeze itself is now lifted in `leaf.rs`: both sides derive the flag
from the branch's own `volatile_statuses`.

Why the freeze had to go rather than merely being allowed to stay: making `_recharging_slots`
symmetric meant `engine_search` started BUILDING self-recharge worlds that previously failed
closed as `self_request_state_unsupported`, so the stale root flag became reachable in production
search at depth > 0 where it never had been. The symmetry fix turned a dormant defect into a live
one; leaving it would have been strictly worse than before task 4.

Measured after the lift, same producing command as above:

- the id-877 self-side family is **gone** (2 rows -> 0);
- `class_rows.state` 123 -> **122**;
- `leaf_root_parity` stays at **diverged 0** — depth-0 parity is intact, which was the freeze's
  entire rationale and the thing that had to be checked;
- and 2 of the 3 `engine_model` `self_team/CATEGORY_VOLATILE_OFFSET` cells cleared too. They were
  `got=0 / want=877` — the leaf writing NOTHING where reality had the flag — not, as an earlier
  revision of this ledger claimed, encore rows displaced by mustrecharge. See the corrected cell
  dump in the "Note on the 124" section; the lift fixes 4 cells in both directions.

The second row's boundary was unverified when this section was written (the artifact stores one
example per family). The lift clearing exactly 2 rows confirms both were id-877.

## P4 — `recharging` is never seeded on a faint-replacement round (1 row)

`CATEGORY_VOLATILE_OFFSET` (opponent) at `1009#[18,19]`, seat p1, direction **inverted**: got 0 /
want 877 — the leaf *lost* a recharge reality still has.

**CLOSED by #1156**, incidentally rather than deliberately: this section's named fix site,
`scripts/leaf_vs_reality.py:430-439`, is the candidate-derived block that task 4 replaced
wholesale with `production_recharging_slots`. Round 18 has no p2 decision row, so the candidate
rule could not see p2's lock; the parser tracker can. `class_rows.state` 124 -> 123.

**[CORRECTION]** An earlier revision called this "the opponent mirror" of P3 and then "the engine
consumes the recharge a ply early". Neither: the recharge was **never in the world to consume**.
`leaf_vs_reality.py:430-439` looks up `decisions[(battle_id, round_n, slot)]` for both slots at
the root round and tests that row's chosen candidate — and battle 1009 has **no p2 decision row
at round 18** (round 18 is p1's force-switch after a faint). So `recharging == ()`, `MUSTRECHARGE`
never enters the constructed world, and `leaf.rs:1320-1323` correctly reports the absence of a
volatile that was never seeded. Slaking's Hyper Beam was at round 17, before the root, so the
branch cannot set it either.

**Disposition: harness** — `scripts/leaf_vs_reality.py`, matching this file's own file-location
table (`P4 | scripts/leaf_vs_reality.py | harness`) and its "Only P4 is harness-only" line.

*Correction on a correction:* a previous pass relabelled this `production`, claiming the original
`harness fix` contradicted the table. It did not — P4 is the one genuinely harness-only cause, and
the table said so. That pass was fixing P5, whose body WAS wrong, and swept P4 along with it. The
label is restored; what was actually stale here is the prescribed remedy below, which #1156
superseded by replacing the block wholesale (see the CLOSED note above). Seed `recharging` from
the opponent's most recent decision row, or
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

The instrument, chased rather than assumed: `engine_world.py:1953-1958` reads the request's active
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

**Disposition: production** — `engine_world.py` world construction, per this file's own file-location table. (An earlier revision said `harness fix` here, which is the contradiction the table at the top now resolves.) Highest-severity family: the only one where the leaf is
*permissive* rather than empty, and a search pricing illegal actions is worse than one reading a
stale counter.

## P6 — the Sleep-Talk turn refund is not modelled at the leaf (2 rows)

**[CORRECTION] — an earlier revision blamed `approximate_sleep_turns=True`. That instrument does
not feed this column and predicts the wrong direction.**

`NUMERIC_SLEEP_TURNS` reads `exact.sleep_turns()` (`encoder.rs:2329`) = the JSON ledger field
(`encoder.rs:1464`), written by `leaf.rs:1461-1470` as `base + count` with `base` the **root
ledger's** value ("Root sleepers keep their ledger base"). `approximate_sleep_turns` only seeds
poke-engine's internal wake-RNG counter (`engine_world.py:1803-1806`). It also predicts the wrong
sign: a `sleep_turns=0` approximation makes the leaf **lower**, while the measurement is got 0.4
(2/5) against want 0.2 (1/5) — **higher**.

Real cause, checked at `1011#[60,61]`: p1's benched Arbok has `sleep_turns=2,
sleep_skipped_turns=1` and the branch switches it in. Gen3 **refunds** turns spent on Sleep Talk /
Snore on pivot-in (`time += skippedTime` in `slp.onSwitchIn`), applied at `belief.py:1526-1533` as
`max(0, sleep_turns - sleep_skipped_turns)`, so reality drops to 1. `LeafMeta.sleep`
(`leaf.rs:322-329`) is `(started, cant_count)` with **no skipped term**, so the leaf cannot apply
the refund and carries the root's 2.

**Disposition: production** — `events.rs`, per this file's own file-location table (an earlier revision said the label-only `encoder fix`, which `:20`'s "All six name a file" rule forbids) — and the `state` class is **correct** — the previous revision's
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
- **All six classes have a cause**, with the owning file named. P2 was open in two earlier
  revisions; the check those revisions named was itself refuted, and P2 is closed on a different
  mechanism with a passing counterfactual.
  P3 points at task 4 of this goal. No disposition is a `classify()` change — an earlier revision
  proposed one for encore and it was withdrawn when that cause turned out to be P1.
  `demonstrated-unreachable` is used nowhere; the one place v2 claimed it, it was false.
- **No disposition rests on side-symmetry.** P2's opponent rows were attributed that way in an
  earlier revision; they now share the self side's measured signature (33/33 on v4, 27/27 on gv2), so
  the inference is gone rather than relocated. Two earlier revisions said "three" and listed two, and
  a third claimed the table marked it after this commit's predecessor had deleted the marker.
- Documentation only: no classifier, encoder or harness change.
- **[CORRECTION]** v1 wrote "got 0.0, want 0.5 on all 34 rows". The harness stores **one
  example per family** (`leaf_vs_reality.py:874-875`), so direction is *inferred from
  source* for **P1**, not measured per row. The uniformity claim is **withdrawn in full**; for P2
  only the `got` half was later re-established, per row, by re-encoding every boundary and reading
  every cell rather than reading the artifact — 33/33 on v4 and 27/27 on gv2, all `got = 0.0`.
  **`want` is NOT uniform:** v4 `{0.1333: 32, 0.0667: 1}`, gv2 `{0.1333: 26, 0.0667: 1}`. That odd
  cell corroborates rather than troubles the account — `0.0667` is stage **1**, exactly the mid-turn
  `force_switch` sampling case withdrawal (3) describes. This section is the conservative backstop,
  so it must neither discount the document's strongest measurement nor let a half-claim stand as a
  whole one.
- **[CORRECTION]** v1 offered `matched + diverged == compared` as evidence. `compared` is
  *defined* as `exact + divergent` (`leaf_vs_reality.py:972`), so the identity is a
  tautology and no harness assertion exists. The non-vacuous identity —
  `sum(all counts) == boundaries` — is not checked by the harness either; verified by hand
  here (1008 and 369) and it holds. Mechanizing it is task 3's subject.
- `self_moveset_mismatch` (11 scenarios skips) and the residue-row classes are owned by the
  fallback-burndown and rust-fidelity lanes; they appear only as skip counts and are **not**
  attributed. **FILED** for the fallback-burndown half: see the note appended to
  `reports/c111_residue_row_causes.md`. The rust-fidelity half is still owed — this sentence
  said "those ledgers", plural, and only one has been written to.
- **`scripts/engine_transition_differential.py` is a FIFTH differential and is NOT adopted.** It
  still derives `recharging` from the RECORDED CHOSEN CANDIDATE -- the rule the four gates dropped
  in #1156 for being circular, since it seeds the world from the very thing the harness is
  checking. It also does not adopt `scripts/differential_denominator.py`. It DOES publish a
  measured count -- an earlier revision of this note said it did not, which was wrong. The problem
  is that nothing GATES on it. Both of its two SUCCESS-path exits are, token for token with the
  source's line wrap collapsed,
  `return 1 if (report["transitions_diverged"] or report["engine_errors"] or partition) else 0`.
  (There are also `return 2` refusal exits for bad invocation, which are not the path in question.)
  The third term is real but does not help: `partition` is `verdict_partition_failures`, which
  checks that the verdict counts SUM to `boundaries_measured` — plus that the fields are present
  and well-formed, but no coverage floor. A run that skipped every boundary sums correctly, so the
  partition closes, and with no divergence and no engine error the run exits 0 having measured
  nothing. (Cited by expression, not line: a first version of this note gave five
  line numbers, every one off by exactly 5, because they were measured against a 14-line draft of
  a comment that shipped at 19. A later revision then paraphrased this expression instead of
  quoting it, dropping the `partition` term and the `report[...]` lookups — same defect, one level
  down.) Publishing without gating is the exact shape the denominator rule exists to close.

  Left as-is on purpose: it sits outside the four differentials the denominator (#1154) and
  recharge (#1156) work scoped, and changing it unscoped is how a "while I'm here" edit lands
  without the mutation testing the other four received.

  **Adopting it means:** `fidelity_gate_events.production_recharging_slots` for the derivation,
  `differential_denominator.check_denominator/gate` for the denominator, and a red run for each
  per the house rule in `docs/engine_fidelity_program_20260801.md`.

  Recorded HERE rather than as a comment in that file, which is where a previous revision put it.
  That file is under a certification pin: `tests/test_c26_damage_composition_readout.py` hashes it
  and requires the hash to match `reports/certification_contract_lifecycle.json`'s registered
  identity, or for the lifecycle to declare a successor-pending divergence recording the new bytes.
  A comment-only edit changed its sha256 and broke that guard -- exactly what the pin exists to
  catch. So adopting the denominator rule there is a real change plus a lifecycle re-registration,
  on top of the three steps above, and not a drive-by.
