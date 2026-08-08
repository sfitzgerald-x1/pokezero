# C148 — the Sleep Talk double `damage_dealt` reset: the probe, the artifacts, and the record

**This is a follow-up to #1170, which is already merged as `cf3c03d3`.** It changes no engine
behaviour. It supplies the thing that PR shipped without — a committed, runnable probe and its
output — and corrects the record where the merged text overstates what was measured.

**Scope, stated first.** #1170 closes a real engine defect on the **in-memory tree fold**
(`model.rs` / `tree.rs`), at depth ≥ 2. It closes **nothing** on the transition-differential sweep
corpus, and it cannot: on all three of that corpus's engine entry points the carry-over the defect
needs is presented as zero — two of them because the field does not survive serialization, the
third because the pyo3 binding hardcodes it away (§2). The four committed sweeps are a **null**, and
the null was **predicted from a committed control arm before the sweeps were read**, not discovered
afterwards.

**The merged commit title overstates all of this and cannot be changed.** #1170 squash-merged as
`cf3c03d3` under *"the Sleep Talk double damage-dealt reset, which is all of
`none_matched:shape_length`"*. That is not what the body it merged establishes, and it is not what
this report claims: the closure is measured on the fold path, `shape_length` on the sweep corpus was
never measured at all, and the change also closes `shape_structure` on the probe population — so
"all of `shape_length`" is wrong in both directions. A reader following `git log` should take the
body, the G38 ledger cell and this report as the claim, not the subject line. The same note is
recorded in `reports/c138_known_gaps_ledger.md` so it is discoverable from the ledger.

**Why this follow-up exists.** Independent review found that #1170's diff carried no probe and no
artifacts, so its headline `8,613 -> 0` was not reproducible by any reader. Re-verified here on
`cf3c03d3`, and stated precisely because a looser version of it was wrong on my first pass:
**`none_matched` appears in zero committed JSON** -- `grep -rl none_matched` over `reports/*.json`,
`reports/artifacts/*.json` and `docs/` returns **0 files**. The string
`sleeptalk_called_unidentified` does appear in **three** (`reports/c9_decomposition.json`,
`reports/c12_decomposition.json`, `reports/c54_sleeptalk_render_contract_mismatch.json`), but only
inside free-text prose fields, with **no shape breakdown and no counts** -- so the class has still
never been quantified in a committed artifact. Three claims from the merged text are **withdrawn**
in §6, and the numbers that replace them are in §3.

---

## 1. The defect

`generate_instructions_from_move` opens by emitting the turn-start `damage_dealt` carry-over reset.
The Sleep Talk block calls `state.reverse_instructions(&incoming_instructions.instruction_list)`
before recursing into the callee, which **undoes** that reset and restores the pre-reset carry-over
into `state`. The callee's own call re-enters the same opening, reads the restored carry-over, and
pushes the **same reset** onto a list that already contains it. `reset_damage_dealt` reads `state`
and only appends — it never mutates the side — so it cannot see the queued instruction.

Two of the three sub-fields are non-idempotent under double emission:
`ChangeDamageDealtDamage` is a **delta** (`0 - damage`), so twice lands on `-damage`;
`ToggleDamageDealtHitSubstitute` is a **toggle**, so twice restores the flag the reset meant to
clear. Only `ChangeDamageDealtMoveCatagory` is an absolute set and idempotent.

The guard is one predicate: `state.use_damage_dealt && !choice.sleep_talk_move`. The callee is not
a second action; it is the same action continuing one level down.

**Renderer consequence.** `consume_move_prelude` (`rust/pokezero-search/src/events.rs`) eats *every*
leading damage-dealt instruction and walks **past** `SetSleepTurns`, so it eats both copies even
though they are not adjacent (`[reset, SetSleepTurns, reset, …]`), while
`identify_sleep_talk_called` regenerates exactly one. The divergence is at **index 0**, which is why
the class registers `shape_length` and never `shape_branch_is_prefix_of_tail` or
`shape_tail_is_prefix_of_branch`.

## 2. Reachability — the full enumeration, and a correction to my own

**A previous revision of this report said `engine_transition_differential.py:2057` calling
`pokezero_search.branch_events` was "the only way the sweep corpus enters the engine". That was a
false totality claim, in the report whose purpose is retiring one, and it appeared in four places.**
It is wrong: `branch_events` is **one of three** entry points, and the other two were never
examined. The conclusion survives — but on a mechanism I had not cited, and the correction matters
more than the citation, because §3's control arm is only as good as the enumeration behind it. I
predicted the null correctly via one route while two went unchecked. The enumeration below is
re-derived from
`grep -n 'poke_engine\.\|pokezero_search\.' scripts/engine_transition_differential.py`, not taken
from review.

**Route 1 — `:2057`, `pokezero_search.branch_events(state.to_string(), …)`.** Serializes first, so
`State::deserialize` runs. `Side::serialize` (`third_party/poke-engine-src/src/state.rs:1292`)
formats **29 fields and `damage_dealt` is not one of them**, and `Side::deserialize` (`:1394`)
hardcodes `damage_dealt: DamageDealt::default()`. **Carry presented: zero.**

**Route 2 — `:2304`, `poke_engine.generate_instructions(state, …)` on the LIVE Python `State`.**
This is the route the previous revision missed, and it is the interesting one: the argument is
`state`, **not** `state.to_string()`. Those states come from `build_poke_engine_state`
(`src/pokezero/poke_engine_adapter.py:298`), which builds through the pyo3 constructors and never
calls `from_string` — so **`State::deserialize` is never involved**. And it *does* reach the guarded
function: `poke-engine-py/src/lib.rs:1067` → `:1092` `generate_instructions_from_move_pair`, the
caller of `generate_instructions_from_move`. The carry is zeroed anyway, by a **different
mechanism**: the `PySide → Side` conversion hardcodes `damage_dealt: Default::default()` at
`poke-engine-py/src/lib.rs:263`, and `PyState → State` sets `use_damage_dealt: false` at `:84`
before calling `state.set_conditional_mechanics()` at `:86`. **Carry presented: zero.**

**Route 3 — `:842` and `:2075`, `poke_engine.calculate_damage(…)`.**
`poke-engine-py/src/lib.rs:1099` → `calculate_both_damage_rolls`
(`src/gen3/generate_instructions.rs:6437`) → `calculate_damage_rolls`. This route **never calls
`generate_instructions_from_move` at all**, so the guarded block is not on it — a stronger statement
than a zero carry. (`:841` additionally feeds it a `State.from_string` state, i.e. route 1's
zeroing.)

**Why zero is decisive.** `DamageDealt::default()` is
`{damage: 0, move_category: Physical, hit_substitute: false}` (`state.rs:586`), and all three of
`reset_damage_dealt`'s guards test against exactly those values — so on a zero carry it emits
**nothing**, and emitting nothing twice is emitting nothing. Both zeroing routes still run
`set_conditional_mechanics`, so the **flag** is set correctly from movesets while the **value** is
zero: exactly the configuration that makes this defect invisible to the corpus while leaving it live
in the fold.

**So the sweep corpus cannot observe this fix on any of its three entry points**, and §5 is a null
by construction.

`pokezero_search.env_step` (`envstep.rs`) is a fourth consumer, outside the differential. It is
zeroed by route 1's mechanism and additionally returns `post_state: state.serialize()`, so the carry
cannot survive *between* calls either.

The one reachable path is the in-memory tree fold, which applies and reverses instructions on a
live `State` without re-serializing, at **depth ≥ 2** — a ply-1 `set_damage_dealt` supplies the
carry that ply 2 then doubles.

## 3. The measurement, committed and reproducible

`rust/pokezero-search/examples/gen3_sleeptalk_none_matched_census.rs`, run on two builds whose
vendored trees differ **in exactly one line** (`diff -r` over the two
`third_party/poke-engine-src` trees returns that one hunk and nothing else).

Population: the #1048 attribution oracle's builder — 10 collision-prone movesets × 4 defenders
(Counter, Mirror Coat, Substitute, Flail) × both move orders = **80 cells, 2,025 branches** — with
two additions, `state.use_damage_dealt = true` and a non-default `side_two.damage_dealt`. Both are
set **directly** rather than via a Counter moveset, because `set_conditional_mechanics` runs inside
`State::deserialize` and not on field assignment; a probe that only added Counter to a moveset
would leave the flag false and measure nothing against either engine.

The probe runs the **same population twice**, once with the carry-over and once with it zeroed. The
zero-carry arm is the **reachability control**: it is what a deserialized corpus boundary looks
like.

| | branches | refused | `none_matched:shape_length` | `none_matched:shape_structure` | attributed |
|---|---|---|---|---|---|
| **guard reverted**, carry-over 137 | 2,025 | **2,025** | **2,025** | 954 | 0 |
| **guarded**, carry-over 137 | 2,025 | **0** | **0** | 0 | 1,840 |
| **guard reverted**, carry-over 0 | 2,025 | 0 | 0 | 0 | 1,840 |
| **guarded**, carry-over 0 | 2,025 | 0 | 0 | 0 | 1,840 |

Artifacts: `reports/artifacts/c148_sleeptalk_double_reset_census_{base,gate}.json`.

Read it in three parts.

1. **On the reachable path the class goes 2,025 → 0** and every refused branch carried
   `shape_length`. On this population the guard is the whole of it.
2. **`shape_structure` also goes 954 → 0.** So the defect is not confined to `shape_length` even
   here — a second reason the original title's "which is all of `none_matched:shape_length`" was
   wrong in both directions.
3. **The two zero-carry rows are identical to each other.** That is the prediction, registered
   before §5 was read: a corpus that only ever presents a zero carry cannot move.

**End-state neutrality, recorded as a digest so it can be checked rather than believed.** Each arm
buckets branch mass per `(cell, serialized end state)` in integer micro-percent and digests the
whole map. **All four arms return `5b30690c…`, 1,899 distinct end states, 2,025 branches.**

Do not over-read that. `Side::serialize` omits `damage_dealt`, so this projection **structurally
excludes the field being repaired** — the matching digest says the observable outcome distribution
does not move, not that the corruption is harmless. On this population the corruption is invisible
in the projection for a specific reason: Counter's `gen3_fixed_damage_amount` returns `damage * 2`,
which is `-274` under the defect and `0` under the guard, and **neither produces a damage event**,
so the two land on the same serialized state by different routes. A population that separates them
is not committed here; see §6.

**By exhaustion, Counter is the only reader that can carry the corruption.** gen3 has exactly three
readers of the struct field — Counter (`generate_instructions.rs:1498`), Mirror Coat (`:1507`) and
Focus Punch (`choice_effects.rs:266`); every other mention is inside `reset_damage_dealt` /
`set_damage_dealt`, the writers, and the `abilities.rs` hits are a local `i16` parameter, not the
field. Mirror Coat gates on `move_category == Special` and the category write is an absolute set —
the one idempotent sub-field — so it is immune. Focus Punch needs `damage > 0`, which both `-137`
and `0` fail.

## 4. Pins

Seven crate tests, the whole of
`rust/pokezero-search/tests/gen3_sleeptalk_damage_dealt_reset.rs`, all seven named in
`.github/workflows/engine-fidelity-gates.yml`. They are the only tests in the repo that reach the
guarded block: the #1048 oracle and every other sleep-talk fixture build state through
`State::default()`, which leaves `use_damage_dealt` **false**, so reverting the guard leaves all of
them green.

**Mutation battery — all eight rows re-run for this revision, none carried forward.** Independent
review had reproduced only M1; M2–M7 were self-reported. Each mutant was applied to the vendored
tree in a dedicated worktree and the file re-run.

| Mutant | reset_once | 2nd-moving | non_dmg_cleared | dmg_records | counter_cleared | ordinary_move | tracking_off | killed |
|---|---|---|---|---|---|---|---|---|
| M0 unmutated (control) | ok | ok | ok | ok | ok | ok | ok | n/a |
| M1 revert guard | FAIL | FAIL | FAIL | FAIL | FAIL | ok | ok | yes |
| M2 drop the flag conjunct | ok | ok | ok | ok | ok | ok | **FAIL** | yes |
| M3 invert → `&& choice.sleep_talk_move` | FAIL | ok | ok | ok | ok | FAIL | ok | yes |
| M4 delete the whole reset block | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | ok | yes |
| M5 wrong guard → `&& !choice.first_move` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | ok | yes |
| M6 `\|\|` instead of `&&` | FAIL | FAIL | FAIL | FAIL | FAIL | ok | FAIL | yes |
| M7 `&& (!sleep_talk_move \|\| !first_move)` | ok | **FAIL** | ok | ok | ok | ok | ok | yes |

7/7 killed. Sole killers: `with_damage_dealt_tracking_off_no_reset_is_emitted` for M2 and
`a_second_moving_sleep_talk_user_also_emits_the_reset_exactly_once` for M7. M7 is the load-bearing
one — `new_choice.first_move = choice.first_move` propagates move order into the callee, so that
predicate restores the double reset for every second-moving Sleep Talk user, and it survived the
original six because all six gave the sleeper speed 500.

`non_damaging_cleared`, `damaging_records` and `counter_cleared` kill the same set {M1, M4, M5, M6},
so none is a unique killer against this battery. Kept because they pin three distinct properties.

## 5. The sweeps — a registered null

`scripts/engine_transition_differential.py --games 200 --matcher strict`, collapsed roll path, on
both builds and both permitted windows. Four artifacts committed as
`reports/artifacts/c148_sleeptalk_double_reset_{base,gate}_{dev,holdout}_sweep.json`.
The final holdout was **not** run.

| | dev `19,000,000–19,000,199` | | validation holdout `19,100,000–19,100,199` | |
|---|---|---|---|---|
| | **base** `30f8b1f8…` | **gate** `de29e3dc…` | **base** `30f8b1f8…` | **gate** `de29e3dc…` |
| `boundaries_full_round` | 15968 | 15968 | 16155 | 16155 |
| `boundaries_measured` | 15503 | 15503 | 15579 | 15579 |
| `transitions_matched` | 15502 | 15502 | 15579 | 15579 |
| `transitions_diverged` | 1 | 1 | 0 | 0 |
| `engine_errors` | 0 | 0 | 0 | 0 |
| `divergence_classes` | `{component_magnitude:heal: 1}` | same | `{}` | `{}` |
| `gating:exact` | 14156 | 14156 | 14148 | 14148 |
| `gating:support` | 1347 | 1347 | 1431 | 1431 |

**The whole `counters` block is byte-identical between base and gate on both windows** — compared
as dicts, 23 keys on dev and 19 on holdout.

Diffing the *entire* artifact leaf-by-leaf, with **no key excluded**, base and gate differ in
**seven** leaves across both windows — four on dev, three on holdout — and none is engine
behaviour:

| leaf | windows | what it is |
|---|---|---|
| `/checkpoint_provenance/distinct[0]` | dev, holdout | the `engine_fingerprint`: two different builds, by design |
| `/elapsed_seconds` | dev, holdout | wall clock |
| `/games_per_hour` | dev, holdout | wall clock |
| `/repros[0]/protocol[1]` | dev | a `\|t:\|` Unix-timestamp protocol line in the retained repro |

A previous revision of this report said **two**, and got there by filtering `elapsed_seconds` and
`games_per_hour` out of the comparison and then describing the result as *"the entire artifact
leaf-by-leaf"*. The four timing leaves were named nowhere. The substance is unchanged — none of the
seven is engine behaviour — but this count was offered as the *stronger* claim than the `counters`
comparison above it, so it had to be right and was not. It is now derived with no filter at all.

**Zero rows opened and zero closed. That is the registered expectation from §3's control arm, not a
disappointment**, and it is committed so the null is on record rather than absent. A reader who
wants to falsify this change should attack §3, not §5: §5 cannot discriminate.

The dev and holdout scalars reproduce C147's independently
(`reports/c147_g33b_residual_bucket_gate.md` §5), which is a useful cross-check that these builds
behave like `main` outside the guarded block.

**Build identity, and the ordering discipline.** Both trees were built from clean with
`scripts/build_search_crate_engine.sh`, exit captured directly and not through a pipe. Gate:
the merged head (`origin/main` `32829210` merged in) plus this branch's edits, 73 patches,
fingerprint **`de29e3dc79c80659`**, confirmed by both `--print` (source-derived from tracked bytes)
and `--check` (against the installed stamp). Base: the same merged commit with the guard predicate
reverted and the tree-sha pin disabled -- a throwaway measurement tree that is never committed --
73 patches, fingerprint **`30f8b1f855e4fade`**. `diff -r` over the two vendored `third_party/poke-engine-src` trees returns
that one hunk and nothing else.

Every edit to a fingerprint-covered tracked file (here, `third_party/poke-engine-gen3-patches.txt`)
was made **before** the build, and the fingerprint measured **last**. That ordering is not
incidental: a first attempt at this work measured, then edited the manifest's comments, which moved
the fingerprint out from under the artifacts -- the exact fault `scripts/engine_build_fingerprint.py`
documents at its `PATCH_LIST.read_bytes()` call. Those sweeps were discarded and re-run rather than
argued around.

## 5b. What this follow-up does NOT change

No engine behaviour, no patch body, no test. The only fingerprint-covered edit is comment text in
`third_party/poke-engine-gen3-patches.txt`, replacing withdrawn figures with the measured ones;
`git diff` on that file shows **only** comment lines — no patch name, no ordering, no patch body.

Re-derived on this branch rather than carried, because #1170's own history is a record of figures
going stale within the hour:

| | value | how |
|---|---|---|
| patch stack | **73** | `engine_build_fingerprint.py --print` |
| crate suite | **443** passed / **443** distinct names / **0** failed | `RUSTFLAGS="-C debug-assertions=yes" cargo test --release`, summed with the gate step's own expression |
| named CI pins | **36**, all resolving | the workflow's own `grep -qxE` over that run |
| `_EXPECTED_SWEEP_ARTIFACTS` | 87 → **91** | `_sweep_reports()` in both trees; set difference exactly the four sweeps, nothing removed; live at 90 and 92 |
| `_EXPECTED_COUNTER_ARTIFACTS` | 360 → **366** | `counter_artifacts()` in both trees; set difference exactly the six; live at 365 and 367 |

The floor stays **443** and the pin loop stays **36**: this branch adds a cargo *example*, which
`cargo test` compiles but does not run and which contributes no test name. Both figures come from a
run on this tree, not from `main`'s comment.

**Then `main` moved to `32829210` (#1169) and this branch merged it, so the figures above are
superseded — recorded rather than overwritten, because the drift is the lesson.** #1169 adds
`rust/pokezero-search/src/abort_telemetry.rs` and touches `events.rs`, both fingerprint-covered, so
nothing was carried across the merge. Re-measured on the merged tree:

| | at `cf3c03d3` | on the merged tree | how |
|---|---|---|---|
| crate suite | 443 / 443 distinct / 0 failed | **451 / 451 distinct / 0 failed** | the gate step's own summing expression |
| named CI pins | 36, all resolving | **44, all resolving** | the workflow's own `grep -qxE` |
| `#[should_panic]` | 5 | **5** (so a strict `^test NAME \.\.\. ok` grep returns 446) | — |
| floor | 443 | **451**, unchanged from `main` | fails at a measured 450, passes at 451 |

An earlier revision of this report's PR body said *"a figure of 451 with a 44-name pin union does
not describe `main`"* and *"`main`'s own workflow reads 443"*. Both were true of `cf3c03d3` and are
**false of `32829210`**: a real run on the merged tree gives exactly **451** and **44**. Withdrawn,
and the reason it happened is worth naming — it was a correctly-scoped statement about a base that
moved, which is the same failure mode as an uncited totality claim: right when written, unqualified
when read.

## 6. Withdrawn, and not verified

**Withdrawn — two claims this branch itself made, found in review of this PR.**

0a. **"`engine_transition_differential.py:2057` … is the only way the sweep corpus enters the
   engine."** A **false totality claim in the report that exists to retire one**, and it appeared
   in four places — the same count as the "~46%" another PR withdrew. There are **three** entry
   points; §2 now enumerates all three, re-derived from the source rather than from review, and
   cites the `poke-engine-py/src/lib.rs:263` / `:84` / `:86` mechanism that actually zeroes the
   route I had missed. The conclusion is unchanged; the enumeration behind it was not there.
   This is the more serious of the two, because §3's control arm is presented as having
   **predicted** the null — and a prediction is worth exactly as much as the enumeration behind
   it. I predicted correctly through one route while two went unexamined.

0b. **"base and gate differ in exactly two leaves."** The measured answer is **seven**:
   `/checkpoint_provenance/distinct[0]` ×2, `/elapsed_seconds` ×2, `/games_per_hour` ×2 and
   `/repros[0]/protocol[1]` ×1. My comparison filtered `elapsed_seconds` and `games_per_hour` out
   and then described the result as *"the entire artifact leaf-by-leaf"*. The substance holds —
   none of the seven is engine behaviour — but the leaf count was offered as the **stronger**
   claim than the `counters` comparison, so being wrong about it is not a footnote. Corrected in
   §5, with all seven named.

**Withdrawn — three claims from #1170, the merged change this follows up.**

1. **The title's "which is all of `none_matched:shape_length`".** The class was never measured on
   the sweep corpus — `grep -rl none_matched` over `reports/*.json`, `reports/artifacts/*.json`
   and `docs/` returns **0 files**. (Three committed JSONs do mention
   `sleeptalk_called_unidentified` in prose fields; none carries a shape breakdown or a count.)
   The only
   quantification of the class was a **source comment** (`events.rs`, "era 61's largest
   world-failure class — 4,786 worlds, 33.3%") from a sweep whose artifact was never committed.
   §3 measures a *probe population*, not that class, and §2 shows the corpus that produced the
   era-61 figure cannot observe this fix. The title now names the path that was measured.
2. **The headline `8,613 → 0`.** No artifact ever supported it, and it counted **aggregate
   `none_matched`**, not `shape_length` — so it was not even the quantity the title named.
   Replaced by §3's committed `2,025 → 0`, which is a smaller, narrower, reproducible claim.
3. **The 5,184-cell and 4,368-cell decompositions** (`594 identical`, `4,590 differ`, `70 differing
   cells`, `26 triples`, `18 full-HP cells`). These came from an uncommitted diagnosis run on the
   72-patch stack and are **not re-measured here**. They are dropped rather than restated. The
   qualitative reader-exhaustion argument they supported is retained in §3 because it is provable
   from source and was independently confirmed by review; the counts are not.

**Also corrected.** The first revision said "#1166 has merged, as `99c77eb7`". `99c77eb7` is
**#1168**; #1166 merged as `f1c3b3aa` and #1171 as `b71bc2fd`. And it described the tail pin as
`[-14:]` → `[-16:]`, "two patches appended"; the diff is `[-15:]` → `[-16:]`, **one** appended.

**Not verified / inherited.**

- **No campaign-era measurement and no fallback-rate claim.** §3's figures are probe-population
  branch counts over a synthetic `State::default()` stat block.
- **No Showdown differential evidence for the repaired value.** That the guarded engine is the
  *correct* one is argued from the engine's own `gen3_fixed_damage_amount` and from gen 3's rules,
  not measured against the real simulator. §5 cannot supply it, per §2.
- **The observable consequence of the corruption is unmeasured.** §3 shows the end-state projection
  does not move on *this* population, and explains why. A population where Counter's `-274` and `0`
  land on different serialized states would show it; none is committed.
- **The in-memory fold is not swept end to end.** §3 reaches the guarded block through
  `render_branch_events` on a live `State`, which is the fold's regime, but it is a probe, not a
  fold run at depth ≥ 2 through `tree.rs`. Building that harness is the honest next step and is not
  claimed here.
