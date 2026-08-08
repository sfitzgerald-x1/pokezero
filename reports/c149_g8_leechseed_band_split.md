# C149 — ledger G8, built and swept: the residual-kill arm splits per roll where the killing residual is Leech Seed

**Outcome: the change SHIPS. The prediction registered at `7613f3e0` landed on every term, and
nothing opened on either window.** `reports/c149_g8_leechseed_band_split_prediction.md` is that
prediction and is unedited; this file is the outcome, and §7 records the one thing the prediction
got wrong.

`19000191/63` — the last divergent row on either permitted window, open since C99 and diagnosed six
different ways before c140 got it right — is closed by an engine change rather than by a comparator
widening or a hand-picked constant. **Ledger G8 stays OPEN**: its second instance is in the final
holdout, on a code path this change cannot reach.

| | base | split |
|---|---|---|
| patches | 73 | **74** |
| engine fingerprint | `de29e3dc79c80659` | `8e912b45544034e6` |
| dev `19,000,000–199` | 15,503 measured / 15,502 matched / **1 diverged** / 0 errors, exit 1 | 15,503 / **15,503** / **0** / 0, exit **0** |
| holdout `19,100,000–199` | 15,579 / 15,579 / 0 / 0, exit 0 | 15,579 / 15,579 / **0** / 0, exit 0 |
| row `19000191/63` | 14 branches, `diverged`, 12 misses | **38 branches, `matched`, 0 misses** |
| that row's p2 mirror-heal census | `{29: 93.9062 %}` | `{22, 24, 25, 26, 27, 28: 5.7129 each; 29: 59.6289}` |

Both rows of that table are single-variable across two builds of **this branch's tree**: the base
column is a throwaway 73-patch build made here (manifest entry and tree-sha pin reverted, never
committed) which reproduced the fingerprint `de29e3dc79c80659` **exactly**, so it is the same engine
the baselines were measured on and not a lookalike.

**Read every residue count below with its accept bar and its coverage bound.** `15,503` is not
15,503 exactly-matched boundaries and is not the window's boundary population:

* **Accept bar.** ~9 % are accepted by an enumerated widening over **up to 64 hidden sleep-counter
  assignments**, accepted if *any* matches, tallied as `gating:support` — dev
  **1,347/15,503 = 8.689 %**, holdout **1,431/15,579 = 9.185 %**. `gating:exact` is 14,156 and
  14,148. Ledger H21.
* **Coverage bound.** Single-seat plies never enter `boundaries_full_round`, so real coverage is
  `measured / (full_round + single_seat)` — dev `15,503 / (15,968 + 1,742)` = **87.5 %**, holdout
  `15,579 / (16,155 + 1,813)` = **86.7 %** — against the 0.9709 and 0.9643 the artifacts
  self-report. `reports/c132_single_seat_coverage_bound.md`, ledger H1.

Both figures are re-derived from `reports/artifacts/c149_*_sweep.json` in this branch, not carried.
**This PR also retrofits both qualifiers into `reports/c140_last_dev_row_diagnosis.md`**, which
quoted `1/15,503` three times and carried neither: `grep -cn 'gating\|accept bar\|87\|sleep-counter\|hidden counter\|coverage'` over that file returned **0** before this change.

## 1. What changed, and the scope argument

c140 §7a, built. In
`third_party/poke-engine-src/src/gen3/generate_instructions.rs`, where the partition emits a
residual-**kill** arm *and* the killing residual **transfers HP to the other side** — in gen 3 that
is Leech Seed alone; Nightmare and Ghost-Curse are damage-only — the single arm priced at the band's
threshold becomes **one arm per roll of the band, each at `1/16`** of the crit-split factor.

The mechanism, from c140 §5: a residual that kills is capped by the HP that happened to be left, so
its magnitude inherits the damage roll. The harness knows this on the drained side and tags it
`capped_lethal`, which is roll-scaled and forgiven. Leech Seed **transfers** the capped amount, and
it lands on the other side as a bare silent `heal` that the comparator checks **exactly**. Over the
lethal band the drain `min(maxhp/8, hp_after_move + leftovers)` is injective in the roll, so one arm
priced one roll and the other six had no arm at all.

### The four call sites, classified by `ceiling` — read, not carried

`grep -n 'residual_disjoint_bands' third_party/poke-engine-src/src/gen3/generate_instructions.rs`
returns the definition at `:1976` and exactly four call sites:

| call | `ceiling` | loop var | site emits a hit-KO arm? | in scope |
|---|---|---|---|---|
| `:4073` (loop `:4085`) | `defender_active.hp` | `num_residual_only` | yes, at `defender_active.hp` | no |
| `:4176` (loop `:4185`) | **`i16::MAX`** | `num_residual_kill_rolls` | no — this fan cannot kill on the hit | **yes** |
| `:4259` (loop `:4268`) | `defender_active.hp` | `num_residual_only` | yes, the crit-straddle KO arm | no |
| `:4309` (loop `:4317`) | **`i16::MAX`** | `num_crit_residual_kills` | no | **yes** |

The handoff supplied this classification with line numbers offset by about ten; it was re-derived
here against the tree at `1c94f071` and the *classification* is confirmed while the line numbers are
corrected above.

> **The two excluded sites are excluded for BLAST RADIUS, not because they are immune.** The same
> capped-lethal-drain arithmetic runs at both, and a Leech-Seeded defender dying to the residual
> inside a band bounded by its own HP has the same roll-dependent mirror. This report does not claim
> otherwise and does not measure it. Extending the split there is a candidate change, unbuilt and
> unswept.

### Two properties that make it a strict improvement rather than a trade

**(a) Roll values come from the exact integer fan.** `residual_band_roll_fan` computes
`floor(max * r / 100)` for `r` in `85..=100` — character-for-character the expression
`push_enumerated_rolls` and the shipped 32-arm `pending_hp_reading_move` enumeration already use.
Deliberately **not** the f32 accumulator in `compare_health_with_damage_multiples`, which drifts
below the true rung (C116 M5); a drifted value is a damage amount Showdown can never deal, so an arm
priced at it can only miss.

**(b) Mass is still priced by the comparator's own count, and a disagreement declines the split.**
Each band keeps the roll count `residual_disjoint_bands` gave it, and
`push_per_roll_residual_kill_arms` emits exactly that many arms at `1/16` — but only after checking
that the integer fan really does put that many rolls in the band's half-open window `[t_i, t_i+1)`.
If it does not, it pushes **nothing** and the caller emits the single collapsed arm it emits today.
The check runs before any push, so a partial emission can never leave the fallback arm
double-counting.

That fallback is **reachable and measured, not theoretical**
(`reports/artifacts/c149_fan_basis_census.json`, reproducible via
`scripts/c149_fan_basis_census.py`): over `max_damage` in `10..=600` and every integer threshold
strictly above the f32 fan's floor, **288 of 27,318 windows disagree — 1.054 %**, touching **256 of
the 591** `max_damage` values. That rectangle is the whole scope of the census and nothing wider.

The consequence is the property that separates this from c140 §7c, which §6a measured as an even
trade: **on any given band the split is a strict improvement or a no-op, never a trade.** A band
whose collapsed arm happens to sit on the roll Showdown threw either keeps that arm (the counts
disagree, so the split declines) or gains an arm at every roll including that one. There is no
configuration in which the split removes a matching arm.

## 2. The row, single-variable

`reports/artifacts/c149_row_replay_{base,split,oracle}.json`, all through the shipped
`cert_sweep_reread.reread_row` → `evaluate_boundary_strict` and the shipped
`pokezero_search.branch_events`. Nothing is reimplemented.

| | branches | verdict | misses | mass sum | p2 `heal` magnitudes, by mass |
|---|---|---|---|---|---|
| base, `de29e3dc` | 14 | `diverged` | 12 | 100.000000 % | `{29: 93.9062}` |
| **split, `8e912b45`** | **38** | **`matched`** | **0** | 100.000000 % | `{22: 5.7129, 24: 5.7129, 25: 5.7129, 26: 5.7129, 27: 5.7129, 28: 5.7129, 29: 59.6289}` |
| oracle, `8e912b45`, `POKEZERO_ENUMERATE_ROLLS=1` | 1015 | `matched` | 0 | 100.000000 % | **identical to the row above** |

Three things worth separating out.

**38 was predicted before it was measured**, and derived rather than carried from c140's estimate:
the band is the seven rolls `[108, 109, 110, 111, 112, 113, 115]` of a 115-max fan at threshold 108,
four upstream arms (Thunderbolt's paralysis × crit splits) each carry one residual-kill arm, so
`14 − 4 + 4 × 7 = 38`. It agrees with c140's arithmetic estimate, which was explicitly unmeasured.

**The base column reproduces c140 §1 and §2 exactly** — 14 arms, the four residual-kill arms failing
at 34.61 / 2.88 / 2.31 / 0.19 % with `observed_only=[('heal', 28)] engine_only=[('heal', 29)]`, and
the census `{29: 93.9062 %}` c140 §2 reported. c140 measured those on fingerprint `44ee1430`; this
is the same numbers on `de29e3dc`, so the row's behaviour is stable across those two eras rather
than an artefact of either.

**The split reproduces the enumeration oracle's mirror distribution byte for byte, at 38 branches
instead of 1,015, on ONE build.** The flag is a `OnceLock` read at process start, so flag-off and
flag-on are one engine and two processes — single-variable by construction, the same discipline
c140 §6 used. This is the strongest single statement in the report: on this boundary the collapsed
path now carries exactly the mirror-heal information the oracle carries, at 3.7 % of the branch
count. It is **one boundary**, and no claim is made that the equality generalises.

## 3. Both windows, both builds — the falsifier did not fire

`--matcher strict --games 200`, exit captured directly and never through a pipe.
`reports/artifacts/c149_{base,split}_{dev,holdout}_sweep.json`.

### Dev — predicted changed

| counter | base | split | predicted |
|---|---|---|---|
| `transition:diverged` | 1 | **key absent** | 0 ✔ |
| `transition:matched` | 15502 | **15503** | 15503 ✔ |
| `divergence_class:component_magnitude:heal` | 1 | **key absent** | absent ✔ |
| `strict:diverged_on_full_branch_set` | 1 | **key absent** | absent ✔ |

### Dev — predicted unchanged, and unchanged

Every remaining key of the `counters` block is identical, and **no key was added**:
`boundaries_full_round` 15968 · `boundaries_measured` 15503 · `gating:exact` 14156 ·
`gating:support` 1347 · `hidden_counter_support:confusion` 1 · `hidden_counter_support:sleep` 1352 ·
`limit:world_substitute_health_unknown` 131 · `skip:single_seat_boundary` 1742 ·
`skip:unmappable_choice:struggle_not_submittable` 118 ·
`skip:world_unsupported:encore_move_unknown` 2 ·
`skip:world_unsupported:materialization_blocker` 18 ·
`skip:world_unsupported:self_request_state_unsupported` 13 ·
`skip:world_unsupported:volatile_unsupported` 144 · `strict:sleeptalk_union_branch` 126 ·
`world_prestate_mismatch` 39 / `:p1_hp` 7 / `:p1_status` 14 / `:p2_hp` 13 / `:p2_status` 5.

`engine_errors` 0, `divergence_classes` `{}`, exit **0**.

### Holdout

**The whole `counters` block is byte-identical to the same-tree base** — zero keys added, zero
removed, zero values changed — at 15,579 measured / 15,579 matched / 0 diverged / 0 engine_errors,
exit 0.

**Zero holdout rows closed was the registered expectation, not a disappointment**: that window had
no divergence left to close, so it is a **safety measurement only and cannot corroborate the
mechanism**. A reader attacking this change should attack §1's scope argument or §2, not §3's
holdout half — it cannot discriminate.

### F1 and F2, explicitly

**F1 — nothing opened.** No boundary that matched on base diverges on split, on either window: dev
goes 1 → 0 and holdout stays 0 → 0, and neither fixed-build artifact carries a `divergence_classes`
key the base does not. **F2 — engine health.** `engine_errors` 0 on both windows; `grep -c "COUNTER
INTEGRITY"` over all four sweep logs returns 0, so the `verdict_partition_failures` self-check
closed the four-term identity on every artifact; branch mass is 100.000000 % on the target row on
both builds. **F3 — mechanism.** Row `matched`, branch count exactly 38.

### Provenance, including one thing that looks wrong in the artifacts

The two fixed-build sweeps record `"source_tree": "dirty"` and
`"source_commit": "7613f3e0…"` — the prediction commit — because the engine change was in the
working tree, uncommitted, while they ran. That ordering is deliberate: the prediction had to be
committed **before** the fixed build was measured. The authoritative identity is the engine
fingerprint, `8e912b45544034e6…` on all 200 checkpoint records of both artifacts, and §6 records
`engine_build_fingerprint.py --print` reproducing that same value from the **tracked bytes of the
committed tree** — so the artifacts do correspond to what ships despite the flag.

Every fingerprint-covered tracked file (`third_party/poke-engine-gen3-patches.txt` and the new patch
file) was edited **before** the build and the fingerprint measured **last**, which is the ordering
`scripts/engine_build_fingerprint.py` documents at its `PATCH_LIST.read_bytes()` call — the manifest
is hashed whole, comments included, so a prose edit after a measurement moves the stamp out from
under the artifacts.

## 4. The dev baseline was run twice and reproduces

The handoff's dev baseline and an independent re-derivation on the same clean tree differ on exactly
**three** leaves, compared leaf-by-leaf with **no filter at all**:

| leaf | handoff | re-derived |
|---|---|---|
| `/elapsed_seconds` | 493.99 | 488.55 |
| `/games_per_hour` | 1457.5 | 1473.7 |
| `/repros[0]/protocol[1]` | `\|t:\|1786179409` | `\|t:\|1786180878` |

Two wall-clock scalars and one Unix-timestamp protocol line. Every counter, verdict scalar and the
whole `checkpoint_provenance` block are identical. The committed artifact is the re-derivation.

## 5. The pins, and what each actually kills

`rust/pokezero-search/tests/gen3_leechseed_residual_band_split.rs`, **eight** tests, on two fixtures
that reach one call site each.

The controls are **poisoned, not clean**, and that is the load-bearing design choice. A clean
unstatused control is worthless here: with no residual there is no lethality threshold,
`residual_disjoint_bands` returns `None`, and no band arm is emitted on either build — so an
unstatused control passes with the gate deleted. Gen 3 poison ticks `maxhp / 8` and Leech Seed
drains `maxhp / 8`, so each control shares its fixture's entire arithmetic — same threshold, same
band, same roll count — and differs only in whether the killing residual transfers.

Mutation results, run rather than reasoned about:

| mutant | red | green |
|---|---|---|
| full revert of `generate_instructions.rs` to the 73-patch preimage | **5 of 8** | the two `a_poisoned_defender_*` controls (correctly — they assert unchanged behaviour) and `every_fixture_still_sums_to_one_hundred_percent` |
| `if defender_leech_seeded {` → `if true {` at **both** sites | **3**: both `a_poisoned_defender_*` controls and `the_split_conserves_…` | all five split assertions |

So the two controls are the **sole killers of the gate deletion**, and the five split assertions are
sole killers of the split's absence. `every_fixture_still_sums_to_one_hundred_percent` kills neither
and is not claimed to: it is a conservation check on a change whose whole shape — `n` arms of `1/16`
replacing one of `n/16` — is the shape that silently loses mass, and `update_percentage` has no
conservation check of its own.

**`every_split_arm_lands_on_a_roll_showdown_can_throw` is explicitly NOT claimed as a sole killer.**
Fixture A's two fan bases agree, so swapping the integer fan back for the f32 accumulator leaves it
green. What kills that mutant is the source-text assertion added to
`tests/test_poke_engine_patch_stack.py` on the fan expression itself, alongside a
`count(...) == 2` assertion on `if defender_leech_seeded {` that fails if the gate is dropped from
either site.

## 6. Build identity and the gate arithmetic

| | value | how |
|---|---|---|
| patch stack | 73 → **74** | clean-room replay into a scratch sdist extraction |
| backends | all `git-apply` except the two known adjacent `patch-fallback`s at indices 46, 47 | the replay's own report |
| `PATCHED_TARGET_TREE_SHA256` | `05bdf844…` → **`7334738d06c11894…`** | clean-room replay, **never** read off the vendored tree |
| `generate_instructions.rs` | `209b938f…` → **`9fd568c65b125fa1…`** | same replay |
| `items.rs`, `abilities.rs`, `choice_effects.rs` | **unchanged** | the drift control: vendored-source drift would have moved all four |
| tail pin | 16 → **17** entries, **grown not slid** | `tests/test_poke_engine_patch_stack.py` |
| crate suite floor | 451 → **459** | see below |
| `--test test_gen3` | **32**, unchanged | run |
| `Engine lib suite` | **5**, unchanged | run |
| `_EXPECTED_SWEEP_ARTIFACTS` | 91 → **95** | `_sweep_reports()` in both trees |
| `_EXPECTED_COUNTER_ARTIFACTS` | 366 → **374** | `counter_artifacts()` in both trees |

**The two artifact-corpus pins move independently and were re-derived from their own selectors, not
from each other and not by arithmetic.** Each was obtained by executing the selector *function
itself* against a worktree of `origin/main` at `8d63dcce` and against this tree:

* `_EXPECTED_SWEEP_ARTIFACTS` 91 → 95, set difference exactly
  `c149_{base,split}_{dev,holdout}_sweep.json`, **nothing removed**.
* `_EXPECTED_COUNTER_ARTIFACTS` 366 → 374, set difference exactly the **eight**
  `reports/artifacts/c149_*.json`, **nothing removed**.

Eight files added, four sweep-corpus members: the three `c149_row_replay_*.json` are single-row
replays and `c149_fan_basis_census.json` is a pure-arithmetic census, so none carries a top-level
`boundaries_measured`. That is why the two counts differ by more than the sweeps, and it is why the
two corpora cannot check each other.

Both measured **after** merging `origin/main`, deliberately: that merge modifies
`docs/token-format/turn16-token-dump.json`, and the counter corpus selects on counter-shaped leaves
rather than on filenames, so a content change alone can move a member in or out. It did not — but a
base taken before the merge could not have said so. Both confirmed live: **95 passes while 94 and 96
fail; 374 passes while 373 and 375 fail.** Neither module's test count moved (26 and 16), so the
workflow's `Ran 26 tests` and `Ran 16 tests` guards still match.

**Everything above was re-run after the merge**, not carried across it: fingerprint `--check` still
reports 74 patches / `8e912b45544034e6`, the crate suite still sums to 459 with all eight pins
resolving, both engine suites are still 32 and 5, the patch-stack module is still `Ran 4 tests OK`,
and the row still replays `matched` at 38 branches with mass 100.000000 %. `origin/main` touched no
fingerprint-covered path, so the two sweeps were not re-run; the fingerprint is the evidence for
that and it is unchanged.

The new digest was independently confirmed a second way: applying the committed patch file to the
73-patch preimage reproduces `9fd568c65b125fa1…` exactly, which is the same value the clean-room
replay produced from a fresh sdist.

**The crate floor is a measurement, not `451 + 8`.** Base is this same tree with
`generate_instructions.rs` reverted to its 73-patch preimage and the new test file moved aside — the
crate suite compiles the vendored engine directly, so that is a real base build. Differencing the
test-NAME sets, per the discipline the workflow's own comments arrived at the hard way:

```
base   451 sum / 451 distinct names      head   459 sum / 459 distinct names
ADDED    8   (exactly the eight in the new file, all eight named in the workflow pin loop)
REMOVED  0
```

No duplicate name in either build, so the set difference is exact. Discrimination replayed against
the real log with the step body: **460 fails, 459 passes, 459 fails on any single deletion, 458
silently absorbs one.**

Five `#[should_panic]` tests mean a strict `^test NAME \.\.\. ok` grep returns 454, not 459; none of
the eight new pins is `#[should_panic]`, so the workflow's `grep -qxE` is safe for them — checked by
resolving all eight against the run these figures came from.

## 7. Withdrawn, and not measured

**Withdrawn — one sentence in this branch's own registered prediction.** §1 of
`reports/c149_g8_leechseed_band_split_prediction.md` reads *"the enumeration's coverage of the
sleep-counter space is **87.5 %**, not the ~96.6 % the metric self-reports."* The **number** is
right for the dev window on this era's artifacts, but the **description conflates two unrelated
things**: 87.5 % is the *boundary* coverage bound (`measured / (full_round + single_seat)`, ledger
H1, `reports/c132_single_seat_coverage_bound.md`), and it has nothing to do with the sleep-counter
enumeration, which is the separate ~9 % `gating:support` accept bar. The prediction file is left
unedited by design; the corrected statement is at the top of this report, with **both** windows —
dev 87.5 %, holdout 86.7 % — rather than dev alone standing in for both. This is a small error of
exactly the kind this program keeps finding: a correct figure attached to the wrong quantity.

**Caught by CI, not by me.** The first push went red on one step: `_MENTION_ALLOWLIST` in
`tests/test_roll_enumeration_scope.py` is an **exact** ledger of every tracked file naming
`POKEZERO_ENUMERATE_ROLLS`, and this report names it in §2 where it records how the oracle column
was produced. c140 §0 documents hitting the same gate for the same reason and I had read that
paragraph. The cause was a process gap rather than a reasoning one: I ran the crate suite and the
two artifact-corpus modules locally but not the workflow's full Python battery, so a gate that
exists and does run found something eighteen locally-run modules would have found first. All
eighteen were then run locally and the battery is green, with two modules failing only on a
`numpy` gap in this venv — `tests.test_leaf_self_recharge_derivation` and
`tests.test_spread_gate_provenance`, both shape-only steps that skip in CI, both untouched by this
branch, and both verified failing identically at the base commit `1c94f071`.

**A second red run, and it was not this branch.** After the fix above, `origin/main` moved to
`c2227d3a` (#1184) while the PR was open. That commit adds a prose reference to
`POKEZERO_ENUMERATE_ROLLS` in `tests/test_engine_terminal_residual_roll_limit.py` **without** the
matching ledger entry, so `origin/main` fails this same gate on its own — verified by running the
module in a clean worktree of `c2227d3a` before touching anything, rather than inferred from the
merge. CI tests the PR *merge* commit, so the break lands on whichever branch merges main next;
the entry is added here with that attribution rather than left for a separate PR. Both of main's
new commits touch only `tests/`, no fingerprint-covered path, so `--check` still reports 74 patches
/ `8e912b45544034e6` and neither sweep was re-run.

Not measured, and stated as such:

* **The two `defender_active.hp`-ceiling call sites.** Same arithmetic, deliberately untouched. No
  claim that they are immune; no measurement either way.
* **`19200244/115`**, the second confirmed G8 instance in the **final holdout**. Unreachable by this
  change — its arm is priced at the *survive representative*, not at a residual threshold, because
  every threshold there lies below the fan minimum and `residual_disjoint_bands`'s
  `min_roll < threshold` guard cannot pass. **The final holdout was not swept and must not be.**
* **How often the split fires across the two windows.** The sweep is verdict-level and carries no
  counter for it. A boundary where the gate is true but no roll of the band was thrown is invisible
  to a verdict comparison, so "nothing opened" is consistent with the split firing many times or
  few. Unmeasured. C147 measured the analogous reach separately and this report does not.
* **How often the count-guard fallback fires across the two windows.** Its reachability is measured
  arithmetically over a `(10..=600)` rectangle, not observed in a sweep.
* **Whether the mirror-census equality with the oracle generalises.** Measured on one boundary.
* Whether the change moves search strength. Not measured; 38 branches where there were 14 is a real
  cost at every ply the gate is true, and no benchmark was run.

## 8. Disposition

**`19000191/63` — closed, by engine change, under the shipping collapsed path.** Both permitted
windows are now at **0 divergences**: dev 15,503/15,503 and holdout 15,579/15,579, at the accept bar
and coverage bound stated at the top of this report.

**Ledger G8 stays OPEN**, and the cell says why: the second instance is on a different code path in
a window this work must not sweep, and two call sites carrying the same arithmetic were left alone
for blast radius.

c140's recommendation was *do not ship 7a*, on maintenance cost — a second partition mechanism in a
family that has produced three wrong hand-derived mass recipes. That objection is real and is not
dismissed: what answers it is that the mass recipe here is **not hand-derived**. The band count
still comes from `residual_disjoint_bands`, the split only redistributes it, and the arithmetic is
checked before any arm is pushed. c140 §7 also does not accept maintenance cost as a disposition,
which is why 7a was the route taken over 7b (a comparator widening whose falsifier is vacuous) and
7c (a re-pricing c140 §6a measured as an even trade, on `n = 1`).
