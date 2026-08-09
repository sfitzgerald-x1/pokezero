# C152 — the mass gate's owed fixtures: C119 obligation 1, discharged for four families

**What this closes.** `reports/c137_phase2_enumerate_decision.md` §4 and
`tests/test_branch_mass_reconstruction.py`'s docstring record the same live commitment:
because Phase 2 **retained** the partition stack for the shipping engine, the mass-gate
fixtures C119 required *"get fixtures before that decision is recorded as closed."* They
did not exist. They do now, as a second matrix `OWED_CASES` in the same module, with
seven fixtures, seven test methods and a mutation run behind every one.

**What this does NOT close, stated first because the two sources disagree by one entry.**
The two lists are not identical and the difference is load-bearing:

| source | families named |
|---|---|
| c137 §4 | crit-fan residual split · `fixed_damage` · multi-hit · the Wish / Rain Dish / Leech Seed / partial-trap mirror steps — **four** |
| the gate's docstring | the same four, **plus the bail set** — five |

The docstring rules the bail set out of fixture scope in the same sentence that names it
(*"unreachable BY THIS DESIGN — a scalar quiet-turn tick cannot represent Sitrus's
non-monotone threshold heal — so covering it needs a different reconstruction, not
another fixture"*), and c137 §4 omits it. **This branch discharges the four and carries
the bail set forward verbatim.** It is not covered, and nothing here should be read as
covering it.

The fourth family is one sentence in both sources and is **four independent branches of
`residual_phase_final_hp`** — orders 7, 10.3, 10.5 and 10.9. It is split into four
entries in `OWED_FAMILIES`, because a single fixture reaching one of them would have
satisfied a one-entry check while leaving the other three unguarded. One of the four,
**Rain Dish**, is discharged by a reachability measurement rather than a fixture (§2).

## 1. The matrix

Seven fixtures. Every number below is re-derived from the built engine at fingerprint
`bfdbe1c04876edcd…`, 74 patches, base `d20cf840`.

| fixture | family | hp/maxhp | tick | reconstruction | engine |
|---|---|---|---|---|---|
| `crit-fan-residual-sand` | crit-fan residual split | 250/255 sand | 15 | 1.4062 % | 1.4063 % |
| `fixed-damage-seismictoss` | `fixed_damage` | 105/244 burn | 30 | 100.0000 % | 100.0000 % |
| `multihit-bonemerang-sand` | multi-hit | 120/244 sand | 15 | 79.4531 % | 79.4531 % |
| `leechseed-mirror` | mirror 10.5 | 140/244 seeded | 30 | 58.3594 % | 58.3594 % |
| `leechseed-crit-fan` | mirror 10.5 | 250/255 seeded | 31 | 3.8672 % | 3.8672 % |
| `partial-trap-mirror` | mirror 10.9 | 130/244 trapped | 15 | 37.2656 % | 37.2656 % |
| `wish-mirror` | mirror 7 | 140/244 burn + Wish | −84 | 5.6250 % | 5.6250 % |

Three design points worth recording because each was a wrong turn first:

- **The crit-FAN site did not need a second attacker profile.** The docstring says it
  does. It is cheaper than that: `CASES`'s crit fan tops out at exactly `MAXHP = 244`, so
  no `hp <= 244` can sit above it — raising the *defender's* `maxhp` to 255 puts the crit
  fan `[207..244]` strictly below 250 HP and reaches the fourth partition site with the
  same attacker.
- **`fixed_damage` has no fan, so "straddle" has to mean something else.** Seismic Toss
  deals `level` = 81 with no 85–100 roll and no crit; `calculate_damage` returns a
  **one**-element list for it, and the fixture's `rolls=False` descriptor is checked
  against that length so a move that grows a crit entry fails by name instead of being
  read as `damages[1]` by accident. The fixture is placed so that the fixed value is the
  *only* reason the answer is 100 %: threshold 76, fixed 81 clears it, and a roll-scaled
  Seismic Toss would fan `[68..81]` with seven rolls below the threshold — 50.0000 %.
- **The multi-hit fan this gate must enumerate is the TOTAL.** The comparator is handed
  `max_damage_dealt * hit_count`, so a per-hit reconstruction would compare two different
  bases and agree with the engine for the wrong reason.

## 2. Pool reachability, before any fixture claims to cover anything

Convention from `reports/c138_known_gaps_ledger.md` §1.1 — a verdict **and** the
instrument that produced it. Instruments are the ones §1.2 prescribes, against the
vendored Showdown at `f76228a1354b5d0f307ca2d16101294ad3a2308b`.

**Move reachability** — the union of every set's `movepool` in
`data/random-battles/gen3/sets.json`: **393 sets over 220 species, 125 distinct move ids**
(matching c138 §1.3 exactly). Classified through `Dex.mod('gen3')`:

| family | verdict | measurement |
|---|---|---|
| multi-hit | **REACHABLE** | of the 125 pool moves, **exactly one** has a `multihit` field: `bonemerang` (`multihit: 2`), on **1** species (Marowak) |
| `fixed_damage` | **REACHABLE** | of the 125, **exactly one** has a `damage` field: `seismictoss` (`damage: "level"`), on **11** species. **0** have `ohko` |
| mirror 7 — Wish | **REACHABLE** | `wish` on **16** species |
| mirror 10.5 — Leech Seed | **REACHABLE** | `leechseed` on **12** species |
| mirror 10.9 — partial trap | **REACHABLE** | of the 125, **exactly one** sets `volatileStatus: 'partiallytrapped'`: `wrap`, on **1** species (Shuckle) |
| mirror 10.3 — **Rain Dish** | **UNREACHABLE** | see below |

**Rain Dish is UNREACHABLE in the pool and no fixture is shipped for it.** Instrument:
the union of every set's `abilities`, **71 distinct ability names** over the same 393
sets — identical to c138's count — and Rain Dish is not one of them. Nor is **Dry Skin**,
the other order-10.3 healer the mirror's own comment flags. **Trace *is* in the 71**, and
cannot manufacture it: Trace copies the *opponent's* ability, and no pool member has Rain
Dish to copy — the cross-side check c138's R26 correction requires. The step is
gen3-legal (`Dex.mod('gen3').abilities.get('raindish')` resolves, `num: 44`) and live in
`residual_phase_final_hp`, so this is unreachable **in the pool**, not in gen3.

Two assertions carry that verdict, and they cover different things — the distinction
matters, because the first revision of this branch conflated them and the c137 bullet
inherited a claim broader than what ran:

- `test_every_owed_family_has_a_fixture_or_a_reachability_verdict` asserts the mapping
  from `OWED_FAMILIES` is total in **both** directions and that no family is both shipped
  and waived. It says nothing about the pool.
- `test_the_rain_dish_waiver_is_backed_by_the_committed_census` asserts
  `POOL_UNREACHABLE`'s **figures** — 220 species / 393 sets / 71 abilities / 0 Rain Dish /
  0 Dry Skin / Trace present / `f76228a1` — against
  `tests/data/c152_pool_reachability_census.json`, regenerated out of process by
  `scripts/c152_pool_reachability_census.py`. `POOL_UNREACHABLE` is structured data rather
  than prose so there is something to compare.

**What the second one does not do**, stated because a waiver is the only thing standing
between this branch and an undischarged obligation: it does **not** re-derive the census
against a live pool. CI builds no Showdown checkout and this module's workflow step
forbids skips outright, so a live derivation cannot run in the mass gate at all. A
Showdown bump that put Rain Dish on a gen3 set would leave artifact, gate and waiver
green and wrong. The census therefore records the commit it was taken at, which bounds
the staleness and makes regeneration a reviewable act. Both directions were mutated to
confirm the pin is not inert: setting the census's `Rain Dish` to 1 reddens it, and
drifting the waiver's `distinct_abilities` to 70 reddens it.

**Scope of these negatives.** Each "exactly one" / "not one of them" above is exactly as
wide as the glob that produced it: the union of `movepool` and `abilities` over every set
in `data/random-battles/gen3/sets.json` at `f76228a1`. It says nothing about items (c138
§1.2: `sets.json` is the wrong instrument for those), nothing about other generations, and
nothing about frequency — only presence.

## 3. Discrimination: every fixture reddens, and the old matrix sees none of it

Seven mutations, one per fixture, each built through the **real** build path
(`pip download poke-engine==0.0.47` → `verify_poke_engine_source.py` →
`apply_poke_engine_patches.py` → mutate → `uv pip install`) and each measured by running
`python -m unittest tests.test_branch_mass_reconstruction`.

> **A first attempt at this measured nothing and reported all-green.** It mutated
> `third_party/poke-engine-src/`, which `setup_poke_engine.sh` never reads — that script
> re-downloads the sdist into a temp dir and applies the patch stack there. Seven
> mutations, seven clean builds, zero effect. Recorded because "all seven mutants are
> green" is exactly what a working gate and a disconnected harness look like from the
> outside, and the only thing that distinguished them was disbelief.

| # | mutation | reddens | figures |
|---|---|---|---|
| M1 | crit-nokill site: push the band's arm two below its threshold (mass-conserving) | `crit-fan-residual-sand` | engine **0.0000 %** vs 1.4062 % |
| M2 | give a `damageCallback` move the 0.925 representative roll | `fixed-damage-seismictoss` | engine **0.0000 %** vs 100.0000 % |
| M3 | drop `hit_count` from `ko_max_damage` | `multihit-bonemerang-sand` | engine **90.0000 %** vs 79.4531 % |
| M4 | delete mirror step 10.5 (Leech Seed) | `leechseed-mirror`, `leechseed-crit-fan`, and the split pin | 90.0000 % vs 58.3594 %; 5.6250 % vs 3.8672 % |
| M5 | delete mirror step 10.9 (partial trap) | `partial-trap-mirror` | engine **5.6250 %** vs 37.2656 % |
| M6 | delete mirror step 7 (Wish heal) | `test_a_resolving_wish_leaves_the_fan_collapsed` | shape `[112]` → **`[106, 110]`** |
| M7 | disable patch 74's per-roll split | `test_the_leechseed_bands_are_split_per_roll` | `[106, 111 … 122]` → **`[106, 110]`**; crit site `[112, 211, 219 … 244]` → **`[112, 211, 219]`** |

**`test_ko_mass_matches_independent_reconstruction` — the pre-existing nine-fixture
matrix — stayed GREEN on all seven.** That is the coverage claim, and it is measured
rather than argued: before this branch the module was blind to every one of these seven
engine defects.

### 3.1 Two assertions that cannot be the KO mass, and why

- **Patch 74 is invisible to any KO-mass functional (M7).** Splitting a *lethal* band
  into one arm per roll moves no mass between faint and alive — every arm was lethal
  before and after. M7 leaves all seven KO-mass rows green. The mass gate that exists to
  catch mass errors in the Leech Seed family therefore covers the family the newest patch
  operates on via the *mirror step*, and covers *the split itself* only through the shape
  pin. Both of the patch's two call sites are reached, and reached separately:
  `leechseed-mirror` sits on the non-crit unbounded-ceiling site and `leechseed-crit-fan`
  on the crit one, so a regression at either site alone still reddens.
- **The Wish fixture's KO-mass row is a control, not the assertion (M6).** A resolving
  Wish heals `min(maxhp − hp, maxhp/2)` *before* every damage tick, and `maxhp/2` strictly
  exceeds the sum of every tick the mirror models (16ths and 8ths totalling at most
  `3·maxhp/8`), so a resolving Wish makes residual death impossible. Deleting the order-7
  heal leaves the engine's KO mass **and** the reconstruction at 5.6250 %, because the real
  end-of-turn phase still applies the Wish and refuses the KO the mispriced mirror
  expected. Only the branch shape moves. This is stated in the fixture comment rather than
  left for a reader to discover, because a row that cannot fail reads as coverage.
  `hp = 140` is chosen for exactly this: at 130 the mispriced threshold falls *below* the
  fan floor, the fan saturates, the shape stays collapsed and M6 is invisible to every
  assertion in the file.

## 4. A measured finding: Case B's non-crit residual band is still on a per-hit basis

Designing the multi-hit fixture surfaced this. It is **not new** — it is
`reports/c129_hitcount_ko_threshold.md` §6 **N1**, filed there as *"Case B's residual arms
are the last basis mix in the file"*, pre-existing on `main` and explicitly not fixed —
but c129's repro is a hand-built state and this is an independent reproduction in the
gate's own idiom, with the mass error quantified:

`bonemerang`, defender Normal/typeless **140/244**, burned, no weather:

```
per-hit max 61 -> per-hit fan [51..61];  total fan [103..122];  burn tick 30
residual threshold 111; ten of the sixteen TOTAL rolls are residual-lethal

engine KO mass                90.0000 %   (one arm: 56 | 56 | 28 = 140, the whole 15/16)
total-fan reconstruction      58.3594 %
delta                         31.6406 points
```

The Case B non-crit block calls `residual_disjoint_bands(max_damage_dealt, …)` with the
**per-hit** maximum against a **total**-basis threshold; no roll of `[51..61]` reaches
111, `total_rolls` is 0, the partition declines, and the collapsed representative
`0.925 × 61 = 56` is applied twice for 112 — which the burn then finishes. Case A, four
lines above, scales the same basis by `hit_count` and its comment says precisely why.

**No fixture is shipped for it**, deliberately: a red fixture is not coverage, and pinning
the current behaviour as expected would be worse. `multihit-bonemerang-sand` is placed at
the **case-a** site, where the hit-count scaling exists and is correct, and
`test_the_owed_matrix_is_not_vacuous` asserts it stays there (`per-hit max < hp <= total
fan top`) so it cannot drift onto the broken site and start pinning the defect. **Scope:
the multi-hit family is covered at the case-a site only.** The case-b non-crit residual
site remains uncovered and open, as c129 §6 N1.

## 5. Verification

Built per the standing recipe: `python3.14 -m venv .venv --system-site-packages`,
`uv pip install --python ./.venv/bin/python maturin`,
`bash scripts/build_search_crate_engine.sh` → **exit 0**, fingerprint
`bfdbe1c04876edcd1957e7a360c5086cfc7eae32ccf3ba0e71d137bd76df3990`, **74 patches** —
re-derived from the build's own stamp, not carried from the handoff.

| module | result |
|---|---|
| `tests.test_branch_mass_reconstruction` (this gate) | **Ran 14, OK** (was 6) |
| `tests.test_collapsed_arm_mass_oracle` | Ran 7, OK |
| `tests.test_roll_enumeration_scope` | Ran 17, OK |
| `tests.test_poke_engine_patch_stack` | Ran 4, OK |
| `tests.test_never_fired_counter_census` | Ran 16, OK |
| `tests.test_boundary_verdict_partition` | Ran 26, OK |
| `tests.test_ledger_table_uniformity` | Ran 17, OK |

**The CI count guard moved with the suite**: `Ran 6 tests` → `Ran 14 tests` in the
`Mass gate` step of `.github/workflows/engine-fidelity-gates.yml`. Those guards are
invisible to a local run and have reddened CI here before. Every other count pin listed
above is unchanged and was re-measured, not assumed.

**Base drift, recorded rather than papered over.** The handoff pinned `main` at
`66ee1869`. `origin/main` was `d20cf840` when this branch was cut — one commit ahead
(#1195, the refusal recorder) — and this branch is based on `d20cf840`. The patch stack
(74), fingerprint and every count pin were re-derived at that base.

**No sweep was run and none is claimed.** This branch adds no sweep artifact, so
`_EXPECTED_SWEEP_ARTIFACTS` (95) and `_EXPECTED_COUNTER_ARTIFACTS` (375) are untouched;
`test_never_fired_counter_census` and `test_boundary_verdict_partition`, which own them,
are green at their pinned counts. No seed window was swept, and in particular nothing at
or above `19,200,000` was touched.

## 6. What independent review added

Review of #1198 re-applied **all seven** mutations through its own replica of the build
path and reproduced every figure to four decimals, and confirmed the central claim
directly: **the pre-existing nine had FAIL count 0 on every one of the seven.** It also
established three things this branch had not:

- **The case-B guard is load-bearing, prospectively.** Moving
  `multihit-bonemerang-sand` to `hp=140` — the case-B site — makes
  `test_the_owed_matrix_is_not_vacuous` go **red** while the KO-mass row stays **green**,
  because at that site the mass row agrees *for the wrong reason* (§4). Without the
  structural assertion a drifted fixture would have read as coverage. That is the exact
  defect class this repository keeps finding, caught before it shipped rather than after.
- **The bail-set exclusion pre-dates this work.** The sentence ruling it out of fixture
  scope is byte-identical at `d20cf840`; this branch only reflowed it. So the four-vs-five
  reconciliation in this document's header dropped nothing and was not authored to shrink
  the obligation. In code the bail set is the three early `return None`s in
  `residual_phase_final_hp` (SITRUSBERRY, LUMBERRY, SHEDSKIN), and a scalar quiet-turn
  tick genuinely cannot express Sitrus's non-monotone threshold heal.
- **The fourth-bullet split is structural, not editorial.** The four mirror steps are four
  separate `if` blocks in `generate_instructions.rs`, so treating the bullet as one family
  would have let one fixture stand for four branches.

And it found three scope defects, all fixed here:

| | defect | fix |
|---|---|---|
| **F1** | The c137 bullet said the Rain Dish verdict was "machine-checked by `POOL_UNREACHABLE`". `POOL_UNREACHABLE` held **prose**; the only assertion checked the family→(fixture\|verdict) mapping was total. Nothing re-derived 71/0. §2 of this document stated it correctly; the bullet compressed it into a stronger claim. | Bullet corrected, and the underlying fact is now pinned: `POOL_UNREACHABLE` is structured figures, `scripts/c152_pool_reachability_census.py` produces `tests/data/c152_pool_reachability_census.json`, and `test_the_rain_dish_waiver_is_backed_by_the_committed_census` compares them (14th test; CI guard 13 → 14). Its own limits are in §2. |
| **F2** | `test_every_owed_family_has_a_fixture_or_a_reachability_verdict` catches a family losing its **last** fixture (relabelling both Leech Seed fixtures reddens it) but not a **mislabel that leaves both families populated** (repointing `leechseed-crit-fan`'s `covers` at `mirror-step-partial-trap` reads 14/14 OK). Its docstring claimed only the former, so this is documentation, not a defect. | Docstring now states the gap and names what holds the line instead — `test_the_owed_matrix_is_not_vacuous`, which asserts each fixture's shape structurally, so a mislabel is a documentation defect rather than a coverage one. |
| **F3** | The c137 bullet omitted that the Wish row is a control. A reader of the bullet alone could count seven mass-discriminating fixtures where there are **six**. | Added to the bullet as scope item (iv). It was already disclosed correctly in the test file and in §3.1 here. |

## 7. What remains open after this

- **The bail set** — the fifth entry in the docstring's uncovered list, out of fixture
  scope by that same sentence. Needs a different reconstruction, not another fixture.
- **Case B's non-crit residual band basis** (§4) — c129 §6 N1, unchanged.
- **c137 §4's other open items** are untouched: the enumerated path's own arm-structure
  pin, the f32 comparator (C116 M5), and the fivefold
  `counters.strict:sleeptalk_union_branch` move.
