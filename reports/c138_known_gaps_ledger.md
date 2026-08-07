# C138 — the known-gaps ledger, pool-reachability filtered

C116 Phase 4 item 14. A **known gap** is a place where the engine, or the harness, is known
to differ from gen3 Showdown or to be unable to express something — *whether or not any sweep
row currently shows it*. The ledger's job is to make the program's blind spots explicit so a
fidelity claim can be read honestly.

> **On the C116 citation.** As in C121–C126 and C131–C136: the plan is not in this repository,
> so it is provenance for why this document exists, never evidence for a claim.

> **Relation to `reports/c125_known_gaps_ledger.md`.** C125 is the same item, scoped to six
> candidates (three dropped, three carried). It is not superseded — its three carried rows
> reappear here as G1, G2, G18 — but it is not the ledger item 14 asks for either. This is the
> exhaustive pass. Where C125 and this document disagree, the disagreement is called out.

**Shape of the document.** §3 carries **78 rows** (55 engine/renderer/leaf, 23 harness/process);
§4 drops **27** candidates as verified unreachable; §3.5 names every exit the differential can
emit and neither window did. A handful of §3 rows carry an UNREACHABLE verdict in place rather
than moving to §4 — G28 and G32 outright, and the split rows G12, G16 and G27 in half — because
they are named, cited gaps whose *status* a reader needs at the place they will look for them.

> **Revision note.** This is the post-review revision. One §4 verdict was **withdrawn as wrong**
> (R26 → G49), two survived with their evidence replaced, one §3 row survived with its reasoning
> replaced, one UNKNOWN was settled statically, one row was re-characterised against its own
> artifact, four counts were corrected and nine gaps were added. Every such change is marked ⚠
> in place and summarised at the end of §8. Nothing was quietly repaired.

---

## 1. Method

### 1.1 Reachability comes first

The filter runs before classification, because several candidate gaps are unreachable in gen3
random battles and would otherwise pad the ledger into uselessness. Every row below carries a
reachability verdict **and the instrument that produced it**.

Three verdicts, and only three:

| verdict | meaning |
|---|---|
| **REACHABLE** | the shape can occur in gen3 randbats; the row then says whether a current sweep row shows it |
| **UNREACHABLE** | measured absent from the pool, with the measurement stated |
| **UNKNOWN** | not determined, with the exact measurement that would settle it named |

"UNKNOWN with a named next measurement" is a better entry than a confident wrong
classification. **Three** rows below are UNKNOWN and say so — H8, H12 and H19. (H11 was the fourth
until 2026-08-07, when `reports/c145_itemleftovers_row_adjudication.md` classified it; an earlier
draft said six: it counted G1, which is classified REACHABLE and merely carries an open
*follow-up* measurement in §7.1, and G28, which is now settled UNREACHABLE by a static argument.)

### 1.2 Which instrument answers which question

C124 established that `sets.json` is the wrong instrument for items. That holds, and this pass
adds a second correction: `sets.json` is also the wrong instrument for **pairings**, because a
gap that needs two things on the *same* Pokémon (Sleep Talk + Haze; Rest + Insomnia) is a
per-**set** question, not a per-species one.

| question | instrument | why |
|---|---|---|
| is a **move** reachable? | union of every set's `movepool` in `data/random-battles/gen3/sets.json` | 125 distinct move ids over 393 sets / 220 species |
| is an **ability** reachable? | union of every set's `abilities` | 71 distinct; **narrower than the dex** — Marowak's dex abilities are Rock Head / Lightning Rod, but its sets list only Rock Head, and Lightning Rod is absent from the pool entirely |
| is an **item** reachable? | `getItem` in `data/random-battles/gen3/teams.ts`, **never** `sets.json` | a gen3 `sets.json` entry has no item field at all |
| is a **same-side pairing** reachable? | per-**set** co-occurrence, not per-species | Haze (4 species) and Sleep Talk (40 species) are both reachable and never co-occur on one set |
| is a **cross-side pairing** reachable? | **independent per-team rates, multiplied** — never a movepool check on the affected species | Trick is `target: normal`, so the *opponent* uses it. White Herb's holders do not need `trick` in their own movepool. See the correction at R26 |
| does a mechanic exist in gen3 **at all**? | `Dex.mod('gen3').abilities.get(id)` / `.moves.get(id)` → `gen` and `isNonstandard` | Snow Warning reports `gen: 4, isNonstandard: "Future"` |

### 1.3 What was measured, and how to reproduce it

All measurements are against the vendored Showdown at
the vendored Showdown checkout (`$POKEZERO_SHOWDOWN_ROOT`, resolved by
`pokezero.local_showdown.default_showdown_root`), commit
`f76228a1354b5d0f307ca2d16101294ad3a2308b`.

**Static pool census** (`data/random-battles/gen3/sets.json`):

```
220 species keys · 393 sets · 125 distinct move ids · 71 distinct ability names
roles: Setup Sweeper 65, Bulky Attacker 59, Fast Attacker 55, Wallbreaker 53,
       Bulky Support 50, Staller 49, Berry Sweeper 23, Bulky Setup 21, Generalist 18
```

**Generative census** — one run, 4,000 `gen3randombattle` teams = 24,000 Pokémon, via
`Teams.generate("gen3randombattle", {seed})` against `dist/sim/index.js`. Every figure below is
from that single run, so they are mutually consistent. This is the instrument that settles
items, and it agrees exactly with reading `getItem`:

```
ITEMS — exactly 13 distinct, over 24,000 Pokémon
  Leftovers 17355 · Choice Band 2984 · Petaya Berry 1042 · Salac Berry 1020
  Lum Berry 466 · Liechi Berry 386 · Soul Dew 201 · White Herb 149
  Thick Club 132 · Light Ball 102 · Stick 85 · Silk Scarf 41 · Twisted Spoon 37
ABILITIES — 71 distinct, identical to the sets.json union
MOVES     — 125 distinct, identical to the sets.json union
NATURES   — unset on 24,000/24,000 Pokémon (i.e. neutral)
LEVELS    — 66..100
SPECIES   — all 220 sets.json keys covered within these 4,000 teams
```

> **Counts, and which of them are seed-stable.** The 13 items, 71 abilities, 125 moves,
> 24,000/24,000 unset natures and 220/220 key coverage are structural and reproduce under any
> seed scheme. Raw *per-species* tallies are not: the number of distinct species **names** was
> 245 here (220 keys + 25 cosmetic Unown formes) and a reviewer running a different seed scheme
> measured 246, and single-species frequencies move by tens of percent. Wherever a row below
> needs a rate rather than a presence/absence fact, it is quoted as a rate measured across
> **three independent seed schemes** and not as one raw count.

The 13-item universe is **closed**, and closed for a structural reason, not a sampling one:
gen3's `getItem` override returns on every path and ends `return 'Leftovers'`, so control never
reaches the gen4/gen5 superclass. Reading the code and generating 24,000 mons give the same 13.
That single fact retires a large block of candidate gaps at once — see §4.

**Engine source.** `third_party/poke-engine-src/` is gitignored and regenerated, so it is cited
by **symbol** throughout. For this pass it was rebuilt exactly as the repo does it:
`poke-engine==0.0.47` sdist, sha256 verified by `scripts/verify_poke_engine_source.py` against
`third_party/poke-engine-base-source.json`, then all 70 patches applied by
`scripts/apply_poke_engine_patches.py` (exit 0, all applied).

**Live histogram.** `reports/artifacts/c136_faintcancels_fix_{dev,holdout}_sweep.json`, the
newest committed post-fix pair. Both `build_check: gated`, `matcher: strict`, 200 games,
source commit `aeaee2b1`, engine fingerprint `e8047b56…`. Seed windows `19,000,000–199` (dev)
and `19,100,000–199` (holdout) — both below the `19,200,000` reserved holdout floor.

### 1.4 Limits of this method

Stated because the reachability filter is the load-bearing part and is easy to over-trust:

- **Reachable is not the same as observed, and unobserved is not the same as absent.** Two
  200-game windows are ~32k full-round boundaries. A shape with a 1-in-50,000 boundary
  incidence is reachable and would show zero rows.
- **The pool is pinned to one Showdown commit.** `sets.json` is upstream data and changes.
  Every "0 of 220" below is a statement about `f76228a1`, not a theorem.
- **UNREACHABLE-in-pool does not mean the engine is right.** It means the defect cannot be
  reached *by this program's format*. Several rows below are unreachable and still real; they
  are recorded so a later format change does not silently re-arm them.
- **A per-set co-occurrence check answers "can one Pokémon have both", not "can the generator
  actually pick both".** Sets have 4-move draws from larger movepools, so co-occurrence in a
  movepool is an upper bound on reachability, and absence from every movepool is decisive. All
  same-side pairing verdicts below are of the decisive (absence) kind, except where the row says
  otherwise. **G14 is the row where an upper bound was wrongly read as reachability**, and it is
  corrected in place rather than quietly repaired.
- **A movepool check is only valid for a move that targets its own user.** A `target: normal`
  move is used *by the opponent*, so "species X's movepool lacks move Y" says nothing about
  whether X can be hit by Y. This is the error that produced the wrong R26 verdict in the first
  version of this document; §8 records the resulting rule and R26 records the correction.
- **No engine was built and no new sweep was run for this document.** Every "currently
  observed" column reads the committed c136 artifacts. Rows marked UNKNOWN are the ones where
  that was not enough.

---

## 2. Coverage: the denominator every row below sits inside

Re-derived from the c136 artifacts rather than carried from C132, whose table was computed on
the older C131 artifacts:

| window | all boundaries | single-seat, never compared | in-path exits | measured | measured / all |
|---|---|---|---|---|---|
| dev `19,000,000–199` | 17,710 | 1,742 (9.84 %) | 465 | 15,503 | **87.54 %** |
| holdout `19,100,000–199` | 17,968 | 1,813 (10.09 %) | 576 | 15,579 | **86.70 %** |

The full-round path reconciles exactly in both windows
(`measured + in-path exits == boundaries_full_round`: 15,503 + 465 = 15,968; 15,579 + 576 =
16,155), which is what makes 17,710 and 17,968 the real totals. The reported
`measured_fraction_of_full_rounds` (0.9709 / 0.9643) divides by a denominator that excludes the
single-seat population entirely.

**This is itself gap H1 below**, and it is the gap that conditions every other row: a divergence
count is a count over ~87 % of boundaries.

---

## 3. The ledger

Classes: **E** engine differs from gen3 Showdown · **H** harness/instrument cannot measure or
express · **R** renderer (the crate's protocol reconstruction) · **X** cross-cutting/process.

Legend for *Observed*: **yes** = a row in the c136 dev or holdout artifact; **no** = zero rows
in those windows; **n/a** = the gap is not of a kind the differential can emit a row for.

### 3.0 Terminal-value gap — read this one first

One gap does not belong in a table, because it corrupts **search value** rather than the
differential's component comparison, and so is invisible to every counter in §2.

| # | Gap | Class | Reachability evidence | Observed |
|---|---|---|---|---|
| **G0** | **A simultaneous last-mon double faint is scored as a win for side two, where gen3 Showdown ties it.** `third_party/poke-engine-gen3-terminal-options.patch`, verbatim: *"KNOWN DIVERGENCE, deliberately NOT addressed here and pinned as-is: a simultaneous last-mon double faint. gen3 Showdown TIES it (`Battle.checkWin`: `this.win(faintData && this.gen > 4 ? faintData.target.side : null)` — the `gen > 4` guard makes gen3 pass `null`), whereas `battle_is_over` tests side one first and returns -1.0, awarding it to side two."* Re-pinned as still-open in `poke-engine-gen3-battle-end-residuals.patch`: *"The #888 double-wipe tie divergence is untouched and re-pinned here."* Deferred because encoding a tie needs a sentinel in a `{0.0, 1.0, -1.0}` contract with no free value, in the shared `src/state.rs`, across three consumers — one of which (`src/search.rs`) uses the value as a **sign multiplier**. | E | **REACHABLE** — via exactly the population H3 already grants: Explosion (25 species), Selfdestruct (3), Destiny Bond (4) and recoil KOs all produce a same-ply double faint, and C132 §3 demonstrates the double-faint replacement ply live in `gen3customgame`. It needs that ply to be each side's *last* mon. | **no** |

**Why this ranks above everything in §3.1.** Every other engine row changes a transition the
differential compares. This one changes a **terminal value**, so it is asymmetric (always
favours side two), it propagates through backup into every ancestor node, and `search.rs` uses
it as a sign multiplier. It is also the one gap whose zero-row count is *guaranteed* rather than
merely observed: a terminal is not a boundary, so no `divergence_class` can ever fire for it.

### 3.1 Engine gaps — REACHABLE

| # | Gap | Class | Reachability evidence | Observed |
|---|---|---|---|---|
| **G1** | **Stick's +2 crit ratio is not modelled.** The gen3 `Items` enum in the engine (`src/gen3/items.rs`, `define_enum_with_from_str! { Items { … } default = UNKNOWNITEM }`) has **no `STICK` variant** — `grep -c STICK` over the whole engine `src/` returns hits only for `STICKYHOLD`/`STICKYWEB`. The whole of gen3 crit rate is `fn critical_hit_chance(choice, defender_ability)` in `generate_instructions.rs`: `BATTLEARMOR`/`SHELLARMOR` → `0.0`, else `guaranteed_crit()` → `1.0`, else `increased_crit_ratio()` → `1/8`, else `BASE_CRIT_CHANCE = 1/16`. **Four arms, no item term.** Showdown gen3: `stick.onModifyCritRatio` returns `critRatio + 2` when `this.toID(user.baseSpecies.baseSpecies) === 'farfetchd'` (`data/items.ts`; the built `dist/` compiles this to an equivalent species-id test), and `sim/battle-actions.ts` uses `critMult = [0,16,8,4,3,2]` for `gen <= 5`, so Return (`critRatio: 1`) goes from **1/16 to 1/4**. | E | **REACHABLE, and deterministic — from code, not from a rate.** `getItem`: `if (species.id === 'farfetchd') return 'Stick';` — unconditional and unguarded, so the holding rate is 100 % by construction and no sample is needed. (An earlier draft quoted "435 of 435 over 20,000 teams"; the 100 % is code-certain but the denominator is seed-scheme dependent and did not reproduce for a reviewer, so the count is dropped.) Farfetch'd's single set is `agility / batonpass / return / swordsdance`, so its only damaging move is the physical Return the boost applies to. | **no** |
| **G2** | **Cross-side Leech Seed is not modelled.** The engine models only the defender's own seed; order 10.5 is cross-side and speed-major, so a faster seeder's drain is emitted at the *victim's* slot. Recorded in `reports/c115_program_state.md` §5 and in the engine's own doc comment at `gen3/generate_instructions.rs` (`"…LIQUIDOOZE it damages instead. Only the defender's own seed is modelled."`). | E | REACHABLE. `leechseed` on **12 of 220** species / 12 sets. | was yes (`19100014/35`, `19100193/46`); **no** in c136 — the *attribution* was fixed, the modelling gap was not |
| **G3** | **`19100014/35`'s 10 % arm is a Leech-Seed-miss branch against a Showdown hit.** `reports/c131_leechseed_heal_label.md` §2: "no rendering change can make a miss branch reproduce a hit." Distinct sub-arm of G2. | E | REACHABLE (as G2). | no (row closed via its 90 % arm only) |
| **G4** | **`compare_health_with_damage_multiples` accumulates the roll ladder in f32.** Re-derived here, not transcribed — see §5. | E | REACHABLE: it is on the KO-threshold split path for every damaging move. | no direct row; **it is the shared machinery under G6/G7 and both `limit:roll_divergent_lethality` rows** |
| **G5** | **Wish heals the resolving *active*'s `maxhp/2`, not the *caster*'s.** `src/pokezero/engine_world.py` `_build_side_spec` (wish block): "The amount is IGNORED by poke-engine, which heals the resolving active's maxhp/2 — a known low-severity deviation from gen3 (true heal = the CASTER's maxhp/2)". | E | REACHABLE. `wish` on **16 of 220** species; **24 sets** pair `wish` with `protect`, the shape that most often survives to resolution on a different active. | no |
| **G6** | **Residual-lethality threshold is read from the defender's *pre-move* state**, before the move's own secondary is known, so a Fire move's burn tick kills a defender the collapsed representative leaves alive. `reports/c135_roll_divergent_lethality_adjudication.md` §3. | E | REACHABLE: burn secondaries on `fireblast` (28 species), `sacredfire` (1), `willowisp` (7), `flamethrower` (15), `overheat` (4). | **was yes** — `19100107/135`, `19100191/5`; ⚠ **CLOSED 2026-08-06** by the collapse-class engine fixes (`reports/c138_collapse_class_engine_fixes_prediction.md` §7): the rows below closed, measured on both windows with nothing opened. The gap description above is the pre-fix state and is kept as the record of it. |
| **G7** | **The crit-straddle path has no residual sub-split.** `reports/c133_collapsed_roll_disposition.md` §3: the crit population straddles both a KO threshold and a sand-lethality threshold, and the code splits only on KO. | E | REACHABLE. Requires a residual chip; in this pool sand is the dominant source, from Sand Stream on **Tyranitar** (all 3 of its sets) — see §4 R4 for why sand is the *only* chipping weather. | **was yes** — `19000074/27`; ⚠ **CLOSED 2026-08-06** by the collapse-class engine fixes (`reports/c138_collapse_class_engine_fixes_prediction.md` §7): the row below closed, measured on both windows with nothing opened. The gap description above is the pre-fix state and is kept as the record of it. |
| **G8** | **The collapsed lethal arm mis-prices a roll-dependent Leech Seed drain** (`hp_after_move + leftovers < maxhp/8` clamp). `reports/c111_residue_row_causes.md` Addendum 2. ⚠ **DIAGNOSED 2026-08-07** — `reports/c140_last_dev_row_diagnosis.md`. Not a limit: the enumeration oracle emits the arm at the observed roll (−109, mirror heal 28) and the shipped comparator accepts it, measured. Over the 7-roll lethal band the drain is **injective** (108→29, 109→28, 110→27, 111→26, 112→25, 113→24, 115→22), so a fixed representative prices **exactly one** of the seven rolls and the shipping engine is already at that bound. ⚠ **CORRECTED 2026-08-07 (same PR):** an earlier version of this cell said "Not closable by a representative either." **That is false** — representative 109 closes it, measured on the full 7×7 representative-versus-thrown-roll matrix, which is the identity (c140 §6a). Re-pricing 108→109 is a **trade**, not a fix: it opens the boundaries where Showdown throws 108, at equal `1/16` mass, and it would be fitting an engine constant to a single dev observation. The engine's own `floor(mean(band))` convention — the rule that yields the `−101` survive representative — gives 111, which also diverges. Disposition: **closed by enumeration, retained under the collapsed path; this gap stays OPEN.** ⚠ **SECOND CONFIRMED INSTANCE 2026-08-07 — `19200244/115`, final holdout** (`reports/c143_heal_attribution_diagnosis.md`). Same arithmetic, second window, different HP configuration: maxhp 407 / hp-after-move 11 / Leftovers 25 / mirror 36, against c140's 235 / 14 / 14 / 28. The band is **14 rolls wide**, not 7, and the mirror is injective over all 14 — so the gap is not specific to one HP configuration and the "one arm prices one roll" bound is a property of the band, not of the row. **It also exhibits a sub-case c140 did not measure, and the shipping engine is BELOW c140's bound in it.** c140's arm was priced *at a residual threshold* (108, a fan member) via `residual_disjoint_bands`; here every residual threshold lies below the fan minimum, that function's `min_roll < threshold` guard cannot pass, and the single non-KO arm is priced at the **survive representative** — `sum(band)//len(band) = 145`, which is **not a member of its own 16-roll fan**. Its mirror 37 is therefore unachievable, and the row matches **0 of 14** rolls rather than 1 of 14 (measured over the **whole band plus the off-fan shipping value — 15 representatives × 14 columns, 210 cells** — through the unmodified shipped comparator). **c140 §6a's "exactly one" needs two scope conditions its own matrix satisfied without stating them**: the representative must be inside the fan, and its mirror must not saturate the seeder's max HP. Both excluded cases are exhibited here — a non-fan representative prices **zero**, and the two saturating ones price **two**. ⚠ **The rule doing that is NOT H8's proportional window, and an earlier revision of this cell said it was.** H8's `[0.92·eng − 1, 1.09·eng + 1]` around the engine's cap of 45 is `[40.40, 50.05]`, which would admit **five** achievable mirrors (47, 46, 44, 43, 41) against the **two** measured. The binding rule is the `_to_full` branch of `roll_components_agree` (`scripts/engine_transition_differential.py:984-1020`), which `continue`s before any later window: its magnitude bound `abs(abs(obs) − abs(eng)) > _damage_difference + 1` with `\|49 − 45\| = 4` gives `[40, 50]`, and its **direction** test `if _obs_damage > _eng_damage and abs(obs) < abs(eng): return False` forces the mirror `>= 45` — leaving exactly {47, 46}, confirmed by direct calls to the shipped function (44/43/41 rejected, 45/46/47 accepted). That tolerance is **adjudicated, not incidental**: `docs/engine_divergence_ledger_20260728.md` §B.4 (`:1005`) filed the `magnitude:heal` class UNRESOLVED and §C.2 (`:1278`) settled it as a matcher defect, prescribing that a capped heal is roll-scaled with *"only the magnitude relaxed, and only in the capped direction (clipping can only reduce, so the test is an **inequality, not a window**)"* — which is the direct refutation of the H8 window attribution, and whose worked case (seed 1310001/72, 251 from 2 HP against 247 from 6 HP) is the same one the code comment names. `I3_roll_inherited` (**H19**) is the still-unadjudicated remainder of the shape, tied to B.4 by `reports/c101_i3_painsplit_tolerance_derivation.json:43`. It is **not** H8's `pre_legal`-absent fallback; mis-attributing it inflated H8's reach, which matters because H8's own cell says "UNKNOWN how much" mass rides on it. **Saturation is a census, not a sample:** only representatives 135 and 136 make the repricer write a `268/268` line, so they are *the* saturating members of the band. All seven of c140's band values were fan members and none saturated, so its bound is sound within its scope; read it as scoped. **Consequence c140's row did not have:** because 145 prices zero rather than one, re-pricing this arm to a non-saturating fan member is a strict **0 → 1** gain here, not the even permutation c140 §6a(ii) analysed — so the "wash" argument against re-pricing does not apply to an off-fan representative, though c140's other two objections (fitted to the sample; the engine's own convention is what produces 145) do. Snapping a mean-priced representative to the nearest fan member is filed as a **candidate rule change, unbuilt and unswept** — and note the nearest are **144 and 146, a tie at distance 1, of which only 146 closes this row**, so the principled snap closes this boundary only if the tie-break is chosen to fit the sample. A **third** collapse sits on the same boundary, on a **different code path**: Fire Blast into Moltres is survivable across its whole fan, so it takes the `max_damage_dealt < defender_active.hp` branch, whose rule is `(max_damage_dealt as f32 * 0.925) as i16` = `49 × 0.925 → 45` (`generate_instructions.rs:4027`), against an observed 49. ⚠ An earlier revision called this a fan mean; `720//16` is also 45 here, which is a coincidence of this fan and not the rule. Either collapse alone would block an arm, so the collapsed path's failure here is over-determined. Second instance carries a companion renderer defect, filed separately as **G33b**; that one is what made the row's class string look novel. | E | REACHABLE (as G2, plus Leftovers, which is 72 % of all generated items). | **yes** — `19000191/63` (dev) and `19200244/115` (final holdout) |
| **G9** | **Multi-hit moves share one damage roll across all hits.** `reports/c129_hitcount_ko_threshold.md` §6: Showdown rolls each hit independently (128 + 121 = 249 on `19100113/62`); the engine applies one roll to the whole move, so a two-hit move can only produce even totals. | E | **REACHABLE, but far narrower than it looks.** The pool's *only* multi-hit move is `bonemerang` (fixed 2 hits), on **Marowak alone**, on 1 of its 2 sets. Measured: 340 of 652 generated Marowak carried it. Every `[2,5]`-hit move (Bullet Seed, Rock Blast, Icicle Spear, Pin Missile, Fury Attack, Doubleslap, Comet Punch, Fury Swipes, Arm Thrust, Barrage, Spike Cannon) is **0 of 220**. | no (`19100113/62` closed by the hit-count partition, not by this) |
| **G10** | **N1 — Case B's residual arms mix per-hit and total bases**, so a KO can be priced as a residual death. `reports/c129_hitcount_ko_threshold.md` §6. | E | REACHABLE only through G9's Bonemerang, since `hit_count > 1` is its precondition. | no |
| **G11** | **N2 — the survive arm truncates**, under-dealing by up to `hit_count − 1` HP. Same source. | E | REACHABLE, same precondition as G10; magnitude bounded at 1 HP for a 2-hit move. | no |
| **G12** | **`residual_phase_final_hp` models no Curse or Nightmare tick**, so the residual threshold is silently absent for those two sources. Same source. | E | **SPLIT.** Curse: REACHABLE — `curse` on **5 of 220** species. Nightmare: **UNREACHABLE** — `nightmare` is 0 of 220. | no |
| **G13** | **A non-Ghost curser carries a spurious engine CURSE volatile** the real protocol never starts. `rust/pokezero-search/src/leaf.rs` `VOLATILE_MAP` doc: "the gen3 engine applies the base Curse choice … with no Ghost/non-Ghost split". | E | **REACHABLE, and it is the only reachable half.** All 5 Curse carriers are non-Ghost (Dunsparce Normal, Miltank Normal, Muk Poison, Regirock Rock, Snorlax Normal); **0 Ghost-type species in the pool carries `curse`**, so the *companion* deviation the same comment names — Ghost-curse target placement — is UNREACHABLE. | no |
| **G14** | **Hustle + Choice Band truncates twice**, a real ±1 damage divergence. `src/pokezero/gen3_damage.py` `Gen3DamageContext.attack_direct_mods`. | E | **REACHABLE, and 100 % of Hustle holders — but not for the reason first given.** Hustle is on Delibird alone. Measured over 60,000 Pokémon: Delibird generates **only three movesets**, every one of them containing Ice Beam, so `counter.get('Physical')` is **3, never 4** — the first Choice Band branch (`Physical >= 4`) never fires. Choice Band comes from the **second** branch, `Physical >= 3 && role === 'Wallbreaker' && counter.get('Special')`, which Delibird satisfies on all three movesets (its only role is Wallbreaker and Ice Beam supplies the Special). The verdict survives; the reasoning does not. ⚠ **This row is the worked example of §1.4's upper-bound trap** — the original evidence inferred `Physical >= 4` from the six-move *movepool* instead of measuring the four-move *draw*. | no |
| **G15** | **Sleep Talk's callee has `before_move` applied twice**, so a Sleep-Talk-called Fire/Ice move into a Thick Fat defender gets its base power quartered. `reports/c102_consumed_choice_double_mutation.json`; the crate's own note calls the dangerous direction "a false MATCH, not a divergence". | E/R | REACHABLE. `sleeptalk` 40 species; Thick Fat 6 species (Dewgong, Grumpig, Hariyama, Miltank, Snorlax, Walrein); Fire/Ice moves are widespread (`icebeam` 58, `flamethrower` 15, `fireblast` 28). | no — and c102 states the differential **cannot** see it, because the failure mode is a false match |
| **G16** | **Lum Berry's cure fires at residual order 10.4 instead of immediately.** `third_party/poke-engine-gen3-residual-speed-order.patch`, on symbol `item_end_of_turn`: "LUM / CHESTO are a KNOWN, SEPARATE divergence … gen3 gives them no residual handler at all … the real fix is an immediate cure and is not attempted here." | E | **SPLIT.** Lum Berry: REACHABLE — 466 of 24,000 generated Pokémon. Chesto Berry: **UNREACHABLE** — not in the 13-item universe. | no |
| **G17** | **Effect Spore's SLEEP third stays refused for a same-turn waking attacker.** `reports/c121_a5_wake_before_contact.md` §3. | E | REACHABLE. Effect Spore on Breloom and Parasect (2 of 220); requires a contact move into them from a mon waking that turn. | no |
| **G18** | **White Herb's `onResidual` order-29 trigger is not wired.** `reports/c126_a6_white_herb.md` §5b. Needs a stat drop surviving to end-of-turn with no move, ability or switch in between. | E | REACHABLE, and the *holder* is deterministic: `getItem` returns White Herb unconditionally for `deoxys` and `deoxysattack` (149 of 24,000 generated). The order-29 *path* is the narrow part. | no |
| **G19** | **The engine implements no Endless Battle Clause.** `src/pokezero/engine_env.py` module docstring: "Showdown ends a non-progressing game; the engine does not … Measured incidence with a random-legal policy: 2 of 120 seeds exceeded 1500 plies against a median of 80." | E | REACHABLE — the pool is full of recovery (`rest` 46, `recover` 16, `protect` 43, `toxic` 114) and Leftovers is 72 % of items. | n/a for the differential; it manifests as `abort:max_steps`, which is **1** in the c136 holdout |
| **G20** | **Unrecognised volatile *and item* names deserialize silently to a default rather than failing.** `third_party/poke-engine-gen3-state-roundtrip.patch` records this for volatiles ("`from_str` has the same `default = NONE` fallback for UNRECOGNISED names"); the same macro gives `Items` `default = UNKNOWNITEM`, which is the mechanism by which G1 is *silent* rather than loud. | E | REACHABLE — G1 is a live instance of it. | no |
| **G21** | **Opponent PP is always modelled as full.** `src/pokezero/engine_world.py` `_move_specs`: "Opponent PP decrements are not tracked publicly yet: full PP is a documented exemption." Interacts with Encore's PP-zero termination. | E/H | REACHABLE — every battle. | no |
| **G21b** | **The engine does not decrement PP above 10 at all — for either side, on every move.** This is the *engine* half that G21's harness half hides, and it is the wider of the two. `gen3/generate_instructions.rs`: `if active.moves[&choice.move_index].pp < 10 { … }`, guarded by the upstream comment *"most of the time pp decrement doesn't matter and just adds another instruction so we only decrement pp if the move is under 10 pp since that is when it starts to matter"*. Corroborated by `poke-engine-gen3-transform.patch` approximation 2, which calls it a "pre-existing cost-tracking threshold … it applies to every move equally". Directly undercuts the two mechanics G21 itself names: **Struggle onset** (a side is only Struggle-locked once every slot hits 0) and **Encore's PP-zero termination** (`move_fails_encore(...) \\|\\| move_slot.pp <= 0`). | E | REACHABLE — every battle, every move above 10 PP. Most of the pool sits above the threshold. Gen3 **max** PP at the default 3 PP Ups is **24** for `shadowball` (38 species, base 15) and **16** for `earthquake` (72), `rockslide` (42), `substitute` (75) and `toxic` (114) — all base 10. ⚠ An earlier draft transposed the `shadowball` and `substitute` figures and called them "base PP"; base is 15/10/10/10/10 and none of it is above the threshold. The verdict is untouched — all five max values exceed 10 — but a transposed number in a ledger invites exactly the re-derivation the ledger exists to prevent. | no |
| **G22** | **Encore's elapsed-turn counter is seeded at a floor of 1.** `engine_world.py` `_build_side_spec` (encore block): the true elapsed count is not observable from the request. Gen3 Encore runs `this.random(3,7)`. | E/H | REACHABLE. `encore` on **16 of 220** species. | no directly; `skip:world_unsupported:encore_move_unknown` fires 2 dev / 1 holdout |
| **G23** | **Substitute health after a surviving hit is unknowable, so the world fails closed.** `engine_world.py` `approximate_substitute_health`; `reports/c16_substitute_depletion_prediction.md`. | E/H | REACHABLE, and common. `substitute` on **75 of 220** species — the single most widespread non-`toxic` move in the pool. The exactly-derivable depletions it names (Seismic Toss reachable at 11 species; Night Shade, Dragon Rage, Sonic Boom all 0) are mostly unreachable. | **yes as an exit** — `limit:world_substitute_health_unknown` 131 dev / 139 holdout |
| **G24** | **A sleeping active cannot be expressed exactly, and the differential recovers by *weakening its own accept bar*.** `slp` is deliberately absent from `_STATUS_CODES` (`engine_world.py`: "guessing them biases wake-up odds (fail closed by default)"), so the strict build raises `status_unsupported`. `approximate_sleep_turns=True` — the "just fell asleep" approximation — is a **search-POC** opt-in and was **OFF** in both c136 sweeps (`approximate_sleep_turns: false` in the artifact). What the differential does instead, traced end to end: `--approximate-sleep` is `action="store_true"`, so it is opt-in and default **OFF**; with it off an underivable counter raises `status_unsupported`; `hidden_counter_recovery` maps that to `"sleep"`; the world is rebuilt with `approximate_sleep_turns=True`; and `_sleep_counter_variants` then enumerates `sleep_turns` 0..`MAX_SLEEP_TURNS`(4) plus `rest_turns` 1..`MAX_REST_TURNS`(2) — **7 variants per asleep side**, cross-product truncated at `MAX_HIDDEN_COUNTER_WORLDS = 64`. `evaluate_boundary_strict` then returns matched on the first passing branch of **any** variant. That is not a wrong guess, it is a **widened accept set** — the more dangerous shape, because it can only ever convert a divergence into a match. ⚠ An earlier draft of this row said the approximation was "default ON" and described a biased guess. Both were wrong: the flag's own default was misread, and the mechanism is enumeration, not approximation. | E/H | REACHABLE and dominant. `hypnosis` 11, `sleeppowder` 9, `spore` 3, `lovelykiss` 1, `yawn` 1 species. | **yes, and it is the largest single caveat on the match count** — `hidden_counter_support:sleep` 1,352 dev / 1,435 holdout, and `gating:support` 1,347 dev / 1,431 holdout against `gating:exact` 14,156 / 14,148. So **8.7 %** of dev and **9.2 %** of holdout matched boundaries were accepted under the widened bar, not the exact one |
| **G25** | **`approximate_partial_trap_turns`: the engine models PARTIALLYTRAPPED with no duration counter at all**, so the trap is *unbounded* and one-sided in the trapper's favour. `engine_world.py` `_build_side_spec`. | E/H | **REACHABLE but vanishingly narrow.** The pool's only partial-trap move is `wrap`, on **Shuckle alone** (1 of 220). Bind, Clamp, Fire Spin, Whirlpool, Sand Tomb are all 0 of 220. | no |
| **G26** | **`approximate_hidden_duration_volatiles`: confusion and Yawn duration are guessed.** `engine_world.py` `_build_side_spec`. Note the comment's confusion half ("never expires it inside a search") is **stale** — `third_party/poke-engine-gen3-confusion-duration.patch` added `chance_confusion_ends`. | E/H | REACHABLE, but both halves are one-producer thin: confusion only via Signal Beam's 10 % secondary on 3 species (G34), Yawn only on Swalot (1 of 220). Consequently `hidden_counter_support:confusion` fired **1 time in dev and 0 in holdout** across 400 games and 32,123 full-round boundaries — the whole confusion-widening machinery rests on that one observation. | 1 dev event |
| **G27** | **`pending_hp_reading_move` is an enumeration that has been found short twice.** `reports/c96_unattributed_source_level_causes.json`, "recorded as an open question". | E | **SPLIT.** Pain Split (added in c128): REACHABLE, 4 species. Endeavor: **UNREACHABLE**, 0 of 220. Flail 1, Reversal 8, Substitute 75, Belly Drum 2 — all reachable. | no |
| **G28** | **`branch_events` panics `Invalid rest_turns value: 32`** — a `PanicException` out of Rust that kills the calling process rather than recording a divergence. `reports/c49_search_crate_rest_turns_panic.json`. | E | **UNREACHABLE from the harness, and settled statically rather than by sampling.** ⚠ An earlier draft rested this on "the current harness, which clamps at 3", which `engine_world.py::_rest_turns_from_row` contradicts verbatim: *"NOTHING CLAMPS IT — not here (the range check below is on the INPUT k, never on the returned counter) and not in the adapter (which validates only non-negativity)."* What actually bounds it is the earlier gate `refunded + skipped > attempts → None`. Since `rest_turns = 3 − k·(1 or 2) + refunded + skipped` and that gate forces `refunded + skipped <= k`, the value is `<= 3` on **both** the `fold_skipped` and non-fold paths, unconditionally. The source records the same conclusion from the other direction: an exhaustive sweep over `(k, refunded, skipped) × Early Bird` found **zero** inputs where the `1 <= rest_turns <= 3` backstop is the line doing the rejecting. This is a **stronger** settlement than the sweep the earlier draft proposed, since a sweep can only produce positive evidence. The panic remains a real engine robustness defect reachable from a hand-built or externally-constructed state — G50's item 4 is the same family of hazard. | no (`engine_error` is 0 in both windows) |
| **G29** | **Trace copies the ability field only; no Start event fires**, so a Traced Intimidate never activates and a Traced Flash Fire's volatile can be wrong. `engine_world.py` `_build_pokemon_spec`; matches `third_party/poke-engine-gen3-trace-no-activation.patch`. | E | REACHABLE. Trace on Gardevoir and Porygon2 (2 of 220); Intimidate on 11 species, Flash Fire on 4. | no |
| **G49** | **Trick can move White Herb onto a non-Deoxys, and Choice Band onto Deoxys — and G18's White Herb gap then travels with it.** ⚠ **This row exists because R26 was wrong in the first version of this document.** | E | **REACHABLE (narrow), by a cross-side check.** Trick is `target: normal` — the *opponent* uses it, so White Herb's holders never needed `trick` in their own movepool, which is exactly what the original R26 checked. `trick` is on Furret and Kecleon; `getItem` gives every Trick user Choice Band (`if (moves.has('trick')) return 'Choice Band';`, measured 944/944). Rate, measured across three independent seed schemes: P(team carries a Trick user) ≈ **2.4 %**, P(team carries White Herb) ≈ **3.0 %**, so P(a battle contains the cross-side pairing) ≈ **0.14 %**, about 1 battle in 700. **The marginals move in the second decimal across schemes (2.15–2.38 %) and the joint spans 0.123–0.137 %**, so the order of magnitude and the ~1-in-700 conclusion are robust and the third decimal is not — an earlier draft claimed third-decimal stability and overstated it. A sampling caveat that matters for anyone re-deriving this: the two teams must come from **disjoint** seed draws. Deriving them from a shared seed by bit-twiddling produces correlated streams and a materially different answer. That is well above the bar at which this ledger grants REACHABLE elsewhere — G25 is granted on Wrap-on-Shuckle alone. | no |
| **G50** | **Transform: max PP is not modelled, and an externally-constructed TRANSFORMED active cannot revert.** `third_party/poke-engine-gen3-transform.patch` approximations 3 and 4, verbatim: *"MAX PP IS NOT MODELLED because the engine has no max-PP field at all"*, and *"A state built from OUTSIDE the engine that carries the TRANSFORMED volatile without a snapshot (e.g. an engine-world constructor expressing an already-transformed Ditto) cannot be reverted, because its base form was never observed. That case degrades to 'drop the volatile, keep the copied form'."* | E | REACHABLE. `transform` on Ditto and Mew (2 of 220). Item 4 names *exactly* the shape `engine_world` produces — its `transform_unexpressible` refusals exist precisely because the constructor is the outside builder in question. | no; `skip:world_unsupported:transform_unexpressible` is 0 in both windows |
| **G51** | **Toxic's min-1 clamp sits inside the multiply, so a 1-HP defender takes `1 × stage` instead of 1.** `gen3/generate_instructions.rs`: `let per_stage = cmp::max(active_pkmn.maxhp / 16, 1); … cmp::min(per_stage as i32 * stage as i32, hp as i32)`. `third_party/poke-engine-gen3-residual-rounding.patch` names it and names the carrier: *"The min-1 clamp sits INSIDE the multiply, which is a second, opposite divergence at the bottom of the HP range: a 1 HP Shedinja (in the gen3 randbats pool) takes `1 * stage`, not 1."* | E | **REACHABLE** — the patch names a pool member, which is this ledger's own bar for a row. Shedinja is in the pool and is the **only** species with `maxhp <= 47` (measured: minimum maxhp across 60,000 generated Pokémon is 1, and Shedinja is the sole holder). Narrow in practice: Shedinja dies to the first stage regardless, so the divergence is in the *magnitude* of a lethal tick, not in whether it is lethal. | no |
| **G30** | **Pivot turns skip the turn's residuals entirely** (documented deviation). `rust/pokezero-search/src/events.rs` `finish_ply`. | E | REACHABLE via Baton Pass, **25 of 220** species — the pool's only pivot (U-turn is gen4). | no |

### 3.2 Renderer gaps — REACHABLE

| # | Gap | Class | Reachability evidence | Observed |
|---|---|---|---|---|
| **G31** | **Haze renders `-unboost` lines; Showdown emits `\|-clearallboost\|`.** `events.rs`, the five-producer note above `boost_may_be_a_switch_out_reset`: "No `clearallboost` exists anywhere in this crate, so the NAMED path is wrong for Haze too." Verified: `clearallboost` appears in the crate exactly twice, both inside that comment. The engine's gen3 `choice_effects.rs` `Choices::HAZE` emits two `state.reset_boosts` calls, i.e. `Boost` instructions. The gap is **one-sided**: the Python parser handles `-clearallboost` (`src/pokezero/showdown.py`, `tier2.py`, `silent_mutation_audit.py`), so the harness understands the line Showdown emits — only the crate cannot produce it. | R | **REACHABLE for direct Haze** — 4 of 220 species (Altaria, Crobat, Mantine, Weezing). The *Sleep-Talk-callee* half of the same note is **UNREACHABLE**: measured on this `sets.json`, **0 of 393 sets** pair `sleeptalk` with `haze` (also 0 with `psychup`, `roar`, `whirlwind`, `batonpass`) — independently reproducing the crate's "350 Sleep Talk sets, zero pair it" claim. | no |
| **G32** | **Psych Up: same shape** (`\|-copyboost\|`). | R | **UNREACHABLE.** `psychup` is 0 of 220. Both the direct path and the Sleep-Talk path are dead. | no |
| **G33** | **Known-open: the drain slot is booked from pre-residual HP.** `events.rs`, `#[ignore = "known open: drain slot booked from pre-residual HP; see doc comment"]` on `a_near_full_hp_seeder_still_over_books_the_drain_slot`. The comment itself says: "Zero occurrences in seeds 19000000-19000199, reachable in ordinary gen3 stall play." | R | REACHABLE (Leech Seed 12 species + Leftovers). | no |
| **G33b** | **The Leftovers heal slot is over-booked when the residual phase is truncated by battle end**, and the constant fallback then relabels the bare Leech Seed drain as `item: Leftovers`. Same `ResidualPlan` over-booking family as G33, different slot and a trigger no HP predicate can see. Mechanism, all from source: gen 3 inherits gen 4, so `data/mods/gen4/items.ts:231` puts `leftovers` at `onResidualOrder 10 / subOrder 4` and `data/mods/gen4/moves.ts:711-716` puts the `leechseed` condition at `10 / subOrder 5` — **one speed-sorted bucket per Pokemon, not two global phases** (the base `data/items.ts` 5 and `data/moves.ts` 8 predict the wrong order), and `sim/battle.ts:565-566` ends that bucket with `this.faintMessages(); if (this.ended) return;`. So a **slower** seeder whose capped drain kills the opponent's **last** Pokemon never reaches its own Leftovers slot; `ResidualPlan::build` books it anyway (deliberately — see the `NOTE:` at `events.rs:5123-5133`, where an `hp < maxhp` guard was measured to cost 5 rows), the count mismatch sets `plan.usable[side] = false`, and every heal on that side falls to `residual_heal_cause`, which since C131 change 3 tests Leftovers first. **The HP arithmetic is correct; only the attribution is wrong** — C131's finding one surface over. Reproduced on three GENERATED gen3 Custom Game boundaries with no Fire Blast, Flamethrower, burn or paralysis, single-variable: variants A and B differ only in whether a spare Pokemon sits behind the victim, and the engine renders the drain `[from] item: Leftovers` in A and, in B, emits **both** heal lines exactly as Showdown does (the rest of B's protocol still differs in two respects the matcher does not compare: the drain damage line omits `[of]`, and `\|faint\|` precedes the mirror heal). `reports/c143_heal_attribution_diagnosis.md` §1–2; artifact `reports/artifacts/c143_heal_attribution_probe.json`. **Not cosmetic:** with it in place the boundary matches **0 of 14** rolls at each of **fifteen** representatives — every member of the residual-lethal band plus the off-fan shipping value, 210 cells — so it is independently sufficient to keep a row divergent under the collapsed path. ⚠ **A gate closes nothing under the collapsed path and CLOSES THIS ROW UNDER ENUMERATION — measured; this cell said "unmeasured" in its first revision and that was the wrong path.** With `POKEZERO_ENUMERATE_ROLLS=1` the shipped renderer gives 416 branches / diverged / 12 misses, one of them `observed_only=[('heal', 36)] engine_only=[('itemleftovers', 36)]` at 0.2189 % — right magnitude, **wrong label only**; exactly 1 arm of the 416 reproduces the full observed HP trace. A modelled gate (drop side one's Leftovers-tagged heal to `[silent]` in arms truncated by the opposing active's faint) gives **matched, 0 misses**. Soundness control, since a relabelling can only widen what matches and "nothing opened" therefore proves nothing about it: 350 heals relabelled, every delta in [27, 47] — the exact range of the 14 achievable mirrors — and **none equal to 16** (`268//16`, a genuine Moltres tick), so no real tick was silenced. Since C137 made enumeration the oracle, that closure lands on the certifying path. **Still unshipped**: the measurement models the fix at the renderer's output rather than building it in `ResidualPlan`, and C133 §7's discipline (built gate, registered prediction with a "nothing opened" falsifier, dev **and** validation-holdout swept, never the final holdout) has not been run. Recommendation: **worth building and sweeping.** | R | REACHABLE (Leech Seed 12 species + Leftovers, 72 % of items) — and observed. | **yes** — `19200244/115`, superimposed on G8 |
| **G34** | **Confusion self-damage is not tagged**, so the source sets differ and exact-component comparison rejects. `reports/c81_small_pure_families.json`. | R | **REACHABLE through exactly one producer.** Enumerated over the pool's 125 moves against `Dex.mod('gen3')`: the only move that can inflict confusion is **`signalbeam`** (10 % secondary), on **3 of 220** species — Venomoth, Ariados, Yanma. Every classical gen3 confuser (`confuseray`, `supersonic`, `swagger`, `flatter`, `sweetkiss`, `dynamicpunch`, `psybeam`, `teeterdance`) and every self-confusing move (`outrage`, `petaldance`, `thrash`) is **0 of 220**. That single 10 % secondary on three species is why `hidden_counter_support:confusion` fired once in 400 games. | no |
| **G35** | **`\|-crit\|` is gated on an exact-value equality**, so the cross-check rejects an observed crit against the engine's own crit arm on identity rather than magnitude. `reports/c93_crit_tag_renderer_gap.json`, `"not_yet_implemented": true`; `reports/c77_i2_crit_arm_absence.json` notes 122 of 173 rows remain unexplained. | R | REACHABLE — crits occur on every damaging move at ≥ 1/16. | no |
| **G36** | **HP-*rise* direction is not rendered in the residual walk** ("DECREASES ONLY, deliberately"), leaving C52's impossible component alive in mirror image. `events.rs` `render_move_phase`. | R | REACHABLE (Leftovers, Wish, Leech Seed drain, `synthesis`/`morningsun`/`moonlight`). | no |
| **G37** | **Attract's empty-immobilization branch is indistinguishable from a fully-capped boost or a blocked stat drop** — 17 sub-cases. `events.rs` `volatile_empty_tail_ambiguous`; `reports/c56_excluded_branch_census.json` counts `attract_empty_tail_ambiguous` 123 and `attract_immobilization_source_unknown` 39. | R | **REACHABLE, but only via the ability route.** The move `attract` is **0 of 220**. Cute Charm is on **3 of 220** (Clefable, Delcatty, Wigglytuff) and applies `attract` on contact at 1/3 (`data/mods/gen3/abilities.ts` `cutecharm.onDamagingHit`). So Attract exists in this format solely as a Cute Charm proc. | no |
| **G38** | **Sleep Talk's unnamed callee: the largest world-level refusal channel.** `events.rs` `unrenderable_family_at`; era 59 measured `sleeptalk_called_unidentified:ambiguous_unrenderable` at 8,149 world failures, 51.6 % of the abort channel. Five of the six allowlist entries have **no fixture** ("admitted on a structural argument"). | R | REACHABLE. `sleeptalk` on **40 of 220** species. | **yes as annotation** — `strict:sleeptalk_union_branch` 126 dev / 105 holdout |
| **G39** | **The residual-order pin filters to `p1a`**, so a within-side sequence can be right while the cross-side interleaving is wrong. `events.rs`, test `end_of_turn_section_order_is_pinned_against_the_engine`. | R/X | REACHABLE — the residual phase is speed-major across both sides in every game. | n/a (test-coverage gap) |
| **G40** | **Five unpinned exemptions in `weather_chips`**: ROCK, GROUND, STEEL (sand), ICE (hail), and the `hp <= 0` gate — each verified to leave the suite green when deleted. `reports/c131_leechseed_heal_label.md` §5. | R/X | ROCK/GROUND/STEEL: REACHABLE (sand, see R4). ICE/hail: **UNREACHABLE** (R2). | n/a |

### 3.3 Harness / instrument gaps — REACHABLE

| # | Gap | Class | Reachability evidence | Observed |
|---|---|---|---|---|
| **H1** | **The differential measures ~87 % of boundaries, not the ~96.6 % it reports.** Single-seat plies are counted in `skip:single_seat_boundary` and never in `boundaries_full_round`; the two sets are disjoint. `reports/c132_single_seat_coverage_bound.md`. | H | REACHABLE by construction. | **yes** — 1,742 dev / 1,813 holdout (§2) |
| **H2** | **The single-seat pins cannot go red from a change to the counting logic** — they read four committed artifacts and import nothing from `scripts/`. The live-coupled pin is "filed, not done here." `c132` §5. | H | REACHABLE by construction. | n/a |
| **H3** | **A deferred residual phase has been compared only in the double-faint case.** `c132` §3. | H | REACHABLE — Explosion (25 species), Selfdestruct (3), Destiny Bond (4) and recoil KOs all produce it. | n/a |
| **H4** | **The differential compares *components*, not branch *masses*** — a whole defect class is structurally invisible to it. `reports/c115_program_state.md` §4. | H | REACHABLE by construction. **This is the channel G4 and G15 hide in.** | n/a |
| **H5** | **Struggle cannot be submitted to the engine**, so those boundaries are dropped. `reports/c44_struggle_repair_probe.json`: "a genuine harness limitation needing engine-side support … should not be ranked as cheap." Note G21b: since the engine never decrements PP above 10, its notion of when Struggle becomes forced is wrong in the first place. | H | REACHABLE. | **yes** — `skip:unmappable_choice:struggle_not_submittable` 118 dev / **233 holdout**. ⚠ It is **not** the largest exit after substitute health; the true ranking below `single_seat` is dev **volatile_unsupported 144 > substitute_health 131 > struggle 118**, and holdout **struggle 233 > substitute_health 139 > volatile_unsupported 127**. Struggle is third in dev and first in holdout |
| **H5b** | **`volatile_unsupported` is the largest non-single-seat exit in dev and was named only once in the first version of this document — inside an UNREACHABLE row.** `engine_world.py`: `unsupported = sorted(set(volatiles) - supported)` → `raise EngineWorldUnsupported("volatile_unsupported", …)`. The supported allowlist is small and documented ("Everything else fails closed"); Substitute, confusion "and kin" need duration state the public replay does not carry. | H | REACHABLE and live. R23 correctly rules out the eight *moves* it names as producers, but the exit itself fires **271 times across the two windows** from other volatiles, so R23 must not be read as retiring it. | **yes** — 144 dev / 127 holdout |
| **H5c** | **`materialization_blocker`: the payload producer's undischarged blockers drop the boundary.** `engine_world.py` `_undischarged_materialization_blockers` — a blocker is discharged only by the matching positive signal for the same species; "everything else … stays fail-closed, because nothing in the caller's signals expresses it and guessing would search a mechanically false world." Appeared **zero** times in the first version of this document. | H | REACHABLE and live. | **yes** — 18 dev / 8 holdout |
| **H6** | **`world_prestate_mismatch`: the constructed engine pre-state disagrees with Showdown's observed pre-state.** Re-derivation trap: the four sub-counters **sum to the parent**, so adding both double-counts. | H | REACHABLE. | **yes** — 39 dev / 68 holdout (`p1_hp` 6, `p1_status` 25, `p2_hp` 23, `p2_status` 14 in holdout) |
| **H7** | **`capped_lethal` overwrites Showdown's supplied `[from]` attribution** in the parser. `reports/c104_capped_lethal_drops_attribution.json`: "CAUSE STILL STANDS; FIX STILL NOT IMPLEMENTED as of 2026-08-04." | H | REACHABLE — any lethal residual tick. | no direct row; it is the label mechanism behind G6/G8's miss text |
| **H8** | **The comparator's fallback window `[0.92·eng − 1, 1.09·eng + 1]` carries unmeasured matched mass.** It applies whenever `pre_legal` is unavailable. `reports/c135` §6. | H | **UNKNOWN how much.** `strict:no_damage_rolls` — the counter that fires when `pre_legal` is None at the state level — is **0 in both windows**, which bounds the *state*-level fallback at zero but not the per-branch one. **Settling measurement:** count boundaries whose accept came from the window rather than exact fan membership, over a 200-game dev sweep. | partially — the counter is 0 |
| **H9** | **Per-slot HP comparison is invalid on any boundary where the active changed.** `reports/c61_empty_engine_arm_census.json`: "37 of 108 rows here are uncomparable … roughly a third of the residue." | H | REACHABLE — **4 of the 6** c136 divergence rows carry `active_changed: true` on one side (`19000074/27`, `19100170/71`, `19100170/72`, `19100191/5`). | **yes**, structurally |
| **H10** | **Repro retention caps at `keep_repro=25` and retains repros only for *divergent* boundaries**, so an adjacent matched boundary needed for a diagnosis is simply not in the artifact. `reports/c120_a1_marker_design.md` §2. | H | REACHABLE by construction. | n/a (both windows are under the cap: 2 and 4 retained) |
| **H11** | **`19100170/71` and `19100170/72` were open divergent rows.** Both `component_missing_in_engine:itemleftovers`, `branch_count: 1`, `pct=100.00`, `p1: protect` against a p2 switch. ⚠ **ADJUDICATED 2026-08-07 — `reports/c145_itemleftovers_row_adjudication.md`.** Class: **a world-construction fix in shipped Python, not a limit and not an engine fix.** Mechanism, measured end to end: `_build_side_spec` resolved the Encore lock — Showdown locks by move **id**, the engine by move **slot index** — against `_active_row_moves`, which is deliberately the **pre-Transform** snapshot (`local_showdown.actor_move_states_from_request_history` skips requests taken while transformed so PP stays honest). For a gen3 randbats Ditto that snapshot is the single move `transform`, so the self-seat rule "exactly one enabled move identifies the lock" was satisfied **spuriously** at index 0; `_apply_transform` then swapped the donor's moveset in underneath the surviving index. Showdown Encored **Protect** (donor slot 3); the world built `last_used_move=move:0` — **Body Slam**. Because an Encore lock is a *forced* choice, that phantom move is the only thing the engine can do, its damage is **lethal** to the switch-in (Spikes had taken Delcatty to 65 at step 71; Typhlosion was at 2 at step 72). The component is labelled `capped_lethal`, but its magnitude equalling remaining HP is **not** corroboration — `engine_transition_differential.py:551` constructs it as `-remaining`, so every lethal capped hit reads that way, the faint arms `end_of_turn_is_deferred`, and the entire residual block is deferred off the boundary. **BOTH** sides' Leftovers ticks are lost, not just p1's — the recorded miss names p1 only because `evaluate_boundary_strict` `break`s out of its `("p1","p2")` slot loop on the first failure, so p2's `itemleftovers +18` at step 71 was never compared. Sizing this class from the miss string undercounts it **by up to half, at boundaries where both sides tick** — 2 ticks against 1 named at step 71, but only 1 tick at step 72 (Typhlosion holds no item), where the miss is complete; **3 lost against 2 named across the class, a third rather than a half.** | H/E | REACHABLE — observed. **Settling measurement (the one specified here): RUN.** Replayed at `dc6e1e19` through `scripts/replay_residue.py`, which re-executes the same `pokezero_search.branch_events` call `evaluate_boundary_strict` makes on the retained candidate states. One branch, `pct=100.00`, `lossy=[]`; instruction stream ends at `ToggleSideTwoForceSwitch` with **no residual phase at all**. Rewriting **one field** of that state — side one's `last_used_move`, `move:0` → `move:3`, same byte-identical engine build — makes the render reproduce Showdown's `\|-heal\|p1a: Ditto\|161/258\|[from] item: Leftovers` and `\|-heal\|p2a: Delcatty\|83/290\|[from] item: Leftovers` **character-for-character**, and the component sets become exactly equal to the observed sets on both slots at both boundaries. Dump: `reports/artifacts/c145_settling_branch_dump.json`. Both rows carry `gating: exact`, so **none of this rides the Constraint-7 hidden-sleep-counter union** (22 of 81 measured boundaries in this game do). | **was yes**; ⚠ **CLOSED by `d27316b6` (#1148)** — **bisected, not inferred.** A one-game sweep (`--games 1 --seed-start 19100170`, strict, rebuilt and `--check`-verified engine at each point) gives 2 divergent rows at `2ec0cb13` (fp `907bea70…`) and at `dc6e1e19` (fp `fdbf5937…`), and **0** at `d27316b6` (fp `fdbf5937…`, unchanged) and on this branch at `662d9db8`. `boundaries_full_round` 88 / `boundaries_measured` 81 and the full skip and gating histograms are identical at all four, so the two boundaries became `matched` rather than skipped. `27609063` is unmeasured and cannot hold the transition: it lies between two measured reds. Artifacts `reports/artifacts/c145_g19100170_{2ec0cb13,dc6e1e19,d27316b6,head}.json`. ⚠ **RETRACTED from the previous revision of this cell:** "nothing in `reports/` still explains the original rows" and "no written cause anywhere in `reports/`". **Both were false when written.** `reports/c139_encore_transform_move_index_prediction.md` § Observation states this mechanism on these two boundaries by seed and step, and #1148 — the very commit this cell guessed at — is what merged it, so the diagnosis was already on `main` at `f876803e`. The negative was asserted over `reports/` from a search that missed a file two commits behind. The closure was also **not incidental**: #1148 registered that prediction, naming these two rows and this class, before measuring. |
| **H12** | **The skip counters do not sum to the coverage shortfall** — `reports/c43_coverage_shortfall_diagnosis.json` measured ~7,224 rows invisible to any skip counter: "no repair list built from them can be complete." | H | **UNKNOWN whether it still holds.** c43 is an older era. In the c136 windows the full-round path reconciles *exactly* (§2), which is evidence against a residual invisible population **within the full-round path** — but says nothing about the single-seat population. **Settling measurement:** instrument the single-seat arm with the same exit taxonomy and re-run. | n/a |
| **H13** | **`self_moveset_mismatch`, `transform_unexpressible`, `status_unsupported` and 33 other world-construction refusal reasons are defined and never fire in either window.** Full list in §3.5. | H | REACHABLE-in-principle for some (Transform is 2 of 220: Ditto, Mew), unreachable for others (Future Sight, 0 of 220 — see R1). | **no** — 36 of 40 `world_unsupported` reasons are 0 in both windows |
| **H14** | ⚠ **CORRECTED 2026-08-07 (C144).** This cell said "**`skip:strict_all_branches_lossy` has never fired**". **That is false, and the refuting artifacts were already committed when it was written:** `reports/c26_structural_probe_report.json` and `reports/c27_structural_probe_report.json` both carry it at **2** (seeds 17000000–17000059, strict matcher), and C141's final-holdout sweep carries it at **4**. On all three the two-term `matched + diverged == boundaries_measured` **fails**. The rest of the cell was right: it increments at `run_game` *after* `_prepare_boundary` has already incremented `boundaries_measured`, so C132's "not an exit" holds for the coverage denominator — but it *is* an exit from the **verdict** tally, which C132 does not say. The identity that actually holds is `boundaries_measured == matched + diverged + engine_error + skip:strict_all_branches_lossy`, and it is now mechanized (`verdict_partition_failures`, gated per shard in `cert_sweep_readout.py`, pinned in `tests/test_boundary_verdict_partition.py`). See `reports/c144_boundary_identity_correction.md`. | H | **REACHED, three times over** — not "in-principle". `strict:lossy_render` is the per-branch precursor and reaching 14 of it (C141 holdout) dropped every branch on 4 boundaries. | **no** — its own gap is closed by the mechanized identity; the *engine_error* term of that identity remains unexercised (0 on every committed artifact). |
| **H15** | **Seven of the eight `unmappable_choice` reasons never fire, and only 6 of the 19 `divergence_class` values have *ever* fired — so 13 have never fired.** ⚠ An earlier draft said 11 of 19, counting only the c136 pair. Re-derived across **all 31 committed sweep artifacts**: the classes that have ever fired are `component_missing_in_engine`, `component_magnitude`, `component_extra_in_engine`, `component_mismatch`, `roll_scaled_component` and `limit:roll_divergent_lethality` — six. (The 18 *distinct keys* observed are dynamic expansions of those six; the taxonomy is 19 static `return` sites in `classify_divergence`.) In c136 specifically only **3** fired: `component_magnitude` and `component_missing_in_engine` in dev, `component_missing_in_engine` and `limit:roll_divergent_lethality` in holdout. Two of the 13 are **structurally unreachable**: `mapper_lossy` (its verdict `continue`s before the classification line) and `no_usable_branch` (its trigger string exists nowhere in the repo). Four more (`boost_delta_support`, `status_support`, `faint_boundary`, `damage_band`) are reachable only through the `--matcher banded` path, which no committed artifact used. The remaining seven — `component_set_equal_but_unmatched`, `limit:world_sample_drag_target`, `evidence:faint_ply_no_upkeep`, `evidence:spikes_in_step`, `evidence:crit_in_step`, `no_miss_recorded`, `unclassified` — are strict-path classes the program has simply never produced. | H | mixed. | **no** |
| **H16** | **The dev window is overfit relative to holdout by 3.53×.** `reports/c117_validation_holdout_baseline.md` §1: "Any statement of the form 'the residue is 7' describes *one particular* 200-game window and must not be read as a fidelity rate." | X | REACHABLE by construction. | **yes** — dev 2 divergences vs holdout 4 on the same build |
| **H17** | **`reports/c119_phase2_scoping.md`, `reports/c134…` and `reports/c137_phase2_enumerate_decision.md` are cited by merged reports and absent from `reports/`.** They are load-bearing: c135 §5/§7 rests on C134 §3's freeze and c137's adopt-for-harness-only decision. | X | n/a. | **yes** (verified by `ls`) |
| **H18** | **Enumeration closes G6's rows but cannot be used in search**: depth-4/1024-sim throughput regresses 2.38 ms → 8,881.8 ms per decision, and the mass gate's `test_matrix_is_not_vacuous` fails under the flag. `reports/c135` §7. | X | REACHABLE. | n/a |
| **H19** | **Four families were never adjudicated**: `LS_capped_lethal_shape` (the largest unresolved), `I2_matcher_accounting`, `I3_roll_inherited`, `I5_boundary_truncation`. `reports/c86_current_era_family_adjudication.json`. | H | **UNKNOWN whether they survive into the current era.** They were defined on an older seed space and none of their labels appears in the c136 counters. **Settling measurement:** re-run `scripts/family_bucket_audit.py` against the c136 artifacts and report which families still have rows. | no |
| **H21** | **`--approximate-sleep`'s help string describes behaviour the tool has not had since hidden-counter support landed**, and it is the most likely origin of this document's own G24 error. Verbatim: *"(default: strict — a publicly-asleep mon with an unknown counter is a counted SKIP, never a guessed world)"*. With `hidden_counter_support` on — which **is** the default (`hidden_counter_support=not args.no_hidden_counter_support`) — such a mon is neither a counted skip nor a guessed world: it is an **enumerated widening** over up to 64 counter assignments, accepted if any matches, and tallied under `gating:support` rather than any `skip:` counter. The string is not merely incomplete; each of its two claims is now false. | X | n/a — a documentation defect in shipped code, on the flag governing the single largest caveat in §6. | n/a |
| **H20** | **`scripts/engine_behavioral_probes.py` states the f32 top-rung count as 174.** The correct figure is **173** — see §5, where I re-derive it three ways. `reports/c115_program_state.md` has it right; the probe comment does not. | X | n/a — a documentation defect in shipped code. | n/a |

### 3.4 Leaf / encoder gaps — from `reports/c112_leaf_state_divergence_ledger.md`

c112 groups 18 families / **138 rows** into **six** mechanisms, and the row counts below are that
one union across its three corpora: `P1 100 + P2 28 + P3 2 + P4 1 + P5 5 + P6 2 = 138`. ⚠ The
first version of this document carried only four of the six and gave P5 as 20 rows by summing
across corpora it was not counted in. Both are corrected: all six are listed, on one footing.

| # | Gap | Class | Reachability evidence | Observed |
|---|---|---|---|---|
| **G41** | **P1 — root-frozen metadata passthrough.** `leaf.rs` clones the root row and mutates metadata in place, so every key the leaf does not explicitly overwrite keeps the **root's** value at every depth. Affects `*_wish_turns`, `*_sleep_clause_blocks`, `*_stall_counter`, `*_encore_elapsed`. | H | REACHABLE — Wish 16 species, Protect 43, Encore 16. Three named members are **latent**: `confusion_elapsed`, `wrap_trap_elapsed` (Shuckle only, G25), `meanlook_trap` (Misdreavus + Ariados, 2 of 220). | 100 rows in the c112 corpus |
| **G42** | **P2 — the event renderer emits status-free condition strings**, so a Toxic stint is invisible to the leaf. `events.rs` `hp_condition` never appends a status token. c112 flags a downstream trap: `leaf.rs`'s "no status token means unchanged, not cured" rule becomes a stale-status bug the moment absence can mean "no status". | R | REACHABLE and pervasive — `toxic` is on **114 of 220** species, the most widespread move in the pool. | 28 rows |
| **G43** | **P5 — a recharge request carries no `disabled` bits, so the constructed world loses a Choice lock.** "the only [family] where the leaf is *permissive* rather than empty." c112's disposition table warns this one reads as test-only work but "P5's fix is in production world construction". | H | REACHABLE, but narrow: `hyperbeam` is on **Slaking alone** (1 of 220), and Choice Band is 12.4 % of items. | **5 rows** |
| **G43b** | **P3 — the self-side recharge root-freeze.** `CATEGORY_VOLATILE_OFFSET` (self); c112 disposition "production + gate fix → task 4", i.e. `engine_search.py::_recharging_slots` plus four gates. c112 attaches a row-level caveat to this mechanism specifically. | H | REACHABLE via the same Slaking-only Hyper Beam population as G43; c112's repro is `golden-scenario-hyperbeam_recharge-91000#[2,3]`. | 2 rows |
| **G43c** | **P4 — `recharging` is never seeded on a faint-replacement round.** `CATEGORY_VOLATILE_OFFSET` (opponent). The **only** one of the six c112 mechanisms that is harness-only (`scripts/leaf_vs_reality.py`); the other five touch the leaf encoding or production world construction. c112 notes an earlier revision mis-described this as "the opponent mirror of P3" and corrected it. | H | REACHABLE, same population. | 1 row |
| **G44** | **P6 — the gen3 Sleep-Talk turn refund (`time += skippedTime`) is not modelled at the leaf.** `LeafMeta.sleep` is `(started, cant_count)` with no skipped term. | H | REACHABLE — `sleeptalk` 40 species, `snore` 0. | 2 rows |
| **G45** | **The pyo3 `LeafEncoder` is not a byte-identical proxy for production** — it goes through `branch_context(lines)` with no transitions where production passes `rendered.active_status_transitions`. | H | REACHABLE by construction. | n/a |
| **G46** | **The scenarios corpus skips 273 of 369 boundaries on `encode_error:ValueError`, and the matchup gate then reports INERT** (0 of 369 same-seat boundaries compared). | H | REACHABLE by construction. | **yes** |
| **G47** | **Tier-2 `is_physical` heuristic diverges from the generator.** `src/pokezero/showdown.py` `_variant_has_physical_attack`: "measured against opening `\|request\|` truth over 60 games, **13 of 720 mons** get an `EXPECTED_ATK` band that excludes the engine's Atk — e.g. a Farfetch'd with `return` banded 133..133 against an engine value of 185." Mirrored deliberately in `encoder.rs::resolve_move_base_power`, so parity holds and both are wrong together. | H | REACHABLE — measured live at 1.8 % of mons. | **yes** |
| **G48** | **Opponent request order: the in-crate fallback is ~91 % wrong** beyond the opponent's first switch-in, and it corrupts action indices. `leaf.rs::root_opponent_order`; `scripts/foulplay_paired_eval.py`: "fail-closed in Python is fail-OPEN in the crate." | H | REACHABLE — any game where the opponent switches twice. | n/a |

### 3.5 Named unobserved coverage — every exit the code can emit and neither window did

Every one of these is a gap in coverage *by definition*: the code has an exit for it and the
program has never seen it fire. Listed rather than summarised, because "we have no rows for X"
is only meaningful if X is named.

**Never-fired static counters (10):** `abort:no_legal_action`, `skip:no_action_candidates`,
`skip:world_error:no_constructible_candidate`, `skip:strict_all_branches_lossy` (H14),
`strict:no_damage_rolls` (H8), `strict:branch_event_legal_error:BranchLegalRollError` — whose
raise-site messages are all unexercised and are not distinguishable in the key anyway —
`engine_error`, `world_prestate_mismatch:side_conditions`, and the two structurally
unreachable `divergence_class` values `mapper_lossy` and `no_usable_branch`.

**Never-fired dynamic families (6):** `skip:no_materialization:{Exc}`,
`skip:world_error:{Exc}`, `strict:branch_events_error:{Exc}`, `engine_error:{Exc}:{detail}`,
`engine_error_choice:{choice}`, `world_prestate_mismatch:weather_{WEATHER}`.

**`skip:unmappable_choice` — 7 of 8 unobserved:** `no_candidate_row`, `blank_move_id`,
`hidden_power_ambiguous`, `move_not_in_engine_set:{id}`, `blank_switch_species`,
`switch_species_not_in_party`, `unknown_kind:{kind}`.

**`skip:world_unsupported` — 36 of 40 reasons unobserved in both windows.** Two are
structurally diverted on the default flags and their absence is expected
(`status_unsupported` → `hidden_counter_support:sleep`; `substitute_health_unknown` →
`limit:world_substitute_health_unknown`). One is UNREACHABLE in this pool and should be read
as retired rather than untested: **`future_sight_pending`** — see R1. The remaining 33 are
unobserved exits, two of which (marked) are additionally unreachable in this pool:
`boost_unsupported`, `boundary_not_move_request`,
`deferred_opponent_action`, `hidden_power_iv_mismatch`, `item_state_conflict`, `move_unknown`,
`nature_not_neutral` (also UNREACHABLE — R7), `override_side_missing`, `payload_malformed`,
`pending_baton_pass`, `public_effect_blocked`, `public_species_not_in_world`,
`rest_sleep_attempt_unsettled`, `rest_sleep_provenance_unrepresentable`,
`rest_sleep_refund_pending_precounts_legacy`, `rest_sleep_refund_pending_unsplit_legacy`,
`self_maxhp_mismatch`, `self_moveset_mismatch`, `self_pp_unknown`, `self_world_mismatch`,
`side_condition_turns_inconsistent`, `side_condition_turns_unknown`,
`side_condition_unsupported`, `species_unknown`, `substitute_depletion_world_incompatible`,
`substitute_health_provenance_contradiction`, `toxic_stage_inconsistent`, `toxic_stage_unknown`,
`transform_unexpressible`, `weather_turns_inconsistent`, `weather_turns_unknown`,
`weather_unsupported` (also UNREACHABLE — R8), `wish_turns_inconsistent`.

**One-sided exits worth watching:** `skip:world_unsupported:self_request_state_unsupported` is
13 in dev and **absent** in holdout; `hidden_counter_support:confusion` is 1 in dev and 0 in
holdout, so the entire confusion hidden-counter machinery rests on a single observation across
400 games.

---

## 4. Dropped by the reachability filter — verified UNREACHABLE

This is the section item 14 exists for. Each of these is a real difference between the engine
and gen3 Showdown, or a real inexpressibility, that **cannot be reached in gen3 randbats**.
Carrying them in a ledger would inflate it and mislead a reader about where the risk is.

| # | Candidate | Why it is unreachable, measured |
|---|---|---|
| **R1** | **Future Sight / Doom Desire — residual order 11, and the `future_sight_pending` refusal** | `futuresight` and `doomdesire` are each **0 of 220** species. They are the whole of gen3's delayed-damage class, so residual order 11 is unreachable and `_reject_unsupported_globals`'s `future_sight_pending` raise is dead code in this format. (Re-derived; C125 reached the same verdict.) |
| **R2** | **Hail, and everything downstream of it** — the ICE branch of `weather_chips`, `Items`/ability hail interactions, `world_prestate_mismatch:weather_HAIL` | `hail` is **0 of 220** as a move, and the only ability that sets it — Snow Warning — reports `gen: 4, isNonstandard: "Future"` in `Dex.mod('gen3')`, i.e. it does not exist in gen3 at all. Two independent routes, both closed. |
| **R3** | **Rain Dish's `maxhp/16` rain heal, and its missing `ResidualPlan` slot** | Rain Dish is **0 of 393 sets**. Verified at the species level too: the only three gen3 species with Rain Dish are Lotad, Lombre and Ludicolo; Lotad and Lombre are **not in the pool at all**, and Ludicolo's single set lists only `["Swift Swim"]`. |
| **R4** | **Weather-expiry sand/hail chip truncation** | **0 of 220** species carry `sandstorm` or `hail` as a move (`raindance` 7 and `sunnyday` 4 exist and neither chips). So sand only ever comes from Sand Stream — **Tyranitar alone**, on all 3 of its sets — which sets `WEATHER_ABILITY_TURNS = -1` and never expires. The expiry path has no trigger. Corroborates `reports/c131` §2. |
| **R5** | **Dry Skin's rain heal at order 10.3** | `Dex.mod('gen3').abilities.get('dryskin')` reports `gen: 4, isNonstandard: "Future"` — it does not exist in gen3. Stronger than C115's "dead code for gen3 randbats". |
| **R6** | **Sitrus Berry, and the monotonicity break it causes in the residual mirror's bisection** | Not in the 13-item universe. `getItem` cannot return it: verified by reading every return path and by generating 24,000 Pokémon. `reports/c111`'s A2 addendum ("threshold berries break the monotonicity … Sitrus fires at `hp <= maxhp/2`") is therefore unreachable. The three reachable pinch berries (Salac, Petaya, Liechi) fire at `maxhp/4` and grant **stat boosts, not HP**, so they do not break HP monotonicity. |
| **R7** | **`nature_not_neutral`** | Generated sets carry **no nature field at all** — measured unset on **24,000 of 24,000** Pokémon (the single §1.3 census run; an earlier draft quoted a stale 12,000 denominator from a superseded run). The engine_world comment ("Gen 3 randbats sets are neutral") is correct and the refusal is unreachable. |
| **R8** | **`weather_unsupported`** | All four gen3 weathers are in `_WEATHER_IDS` (`rain`, `sun`, `sand`, `hail`). No gen3-legal weather can miss the map. |
| **R9** | **Liquid Ooze mislabelled by the residual-heal renderer** | Two things are true and they are different. Liquid Ooze *is* reachable — 2 of 220 species (Swalot, Tentacruel). But the **renderer path** is unreachable: in `events.rs` `render_residual_instruction`, an `Instruction::Heal` with `heal_amount < 0` is intercepted and rendered as `\\|-damage\\|…\\|[from] ability: Liquid Ooze` **before** `plan.take(side, true)` or `residual_heal_cause` is ever called. The Liquid Ooze guard inside `residual_heal_cause` is therefore dead code, exactly as `reports/c131` §5 says. On the mechanic side, note that both `leechseed` and `gigadrain` are `target: normal`, so the interaction that matters is a **seeder or drainer facing** a Liquid Ooze holder — cross-side, and reachable, since 12 and 2 species carry those moves against 2 Liquid Ooze carriers. The "0 of 393 sets pair them on one set" measurement is true but answers the wrong question and is not load-bearing here; what makes this row UNREACHABLE is the renderer interception alone. |
| **R10** | **Shell Bell, and the `heal_drain_or_shellbell` ambiguity** | Shell Bell is modelled in the engine (`src/gen3/items.rs` `SHELLBELL`) and is **not in the 13-item universe**. So the bucket the crate names for an ambiguity it "cannot resolve" is, in this format, unambiguous: the only reachable producer is a drain move, and the pool's only drain move is `gigadrain` (Exeggutor, Parasect). |
| **R11** | **The `TwoToFiveHits` flat-3.2 approximation** | Every `[2,5]`-hit move is **0 of 220**. The only multi-hit move in the pool is `bonemerang`, whose `multihit` is the scalar `2`. See G9 for what *is* reachable. |
| **R12** | **Rest's Insomnia / Vital Spirit fail clauses** | **0 of 393 sets** pair `rest` with either ability (independently reproducing the patch's "0 of the 55 Rest sets"). Comatose is gen7. |
| **R13** | **Belly Drum's Shedinja `maxhp === 1` fail clause** | Shedinja's single set is `agility / batonpass / hiddenpowerfighting / shadowball / silverwind / toxic`. No Belly Drum. It ships for source parity only. |
| **R14** | **N5 — the residual ceiling overshooting into a move-KO** | c129 measured 4,326 such states, **every one at `maxhp <= 47`**. Measured over 60,000 generated Pokémon, the minimum maxhp in the pool is **1** and **Shedinja is the only species at or below 47**. ⚠ The original evidence then said "and Shedinja carries no multi-hit move" — a *same-side* check, and therefore the wrong one, since the multi-hit move belongs to the **attacker**. The correct argument is stronger: N5 additionally needs `hit_count > 1`, the pool's only multi-hit move is Bonemerang (Ground), and Ground is **resisted** by Bug/Ghost (`getEffectiveness('Ground', ['Bug','Ghost']) = −1`), so **Wonder Guard blocks it outright**. Enumerated exhaustively: of the pool's 125 moves, **every** move that is super-effective against Bug/Ghost is single-hit. No multi-hit move can damage Shedinja at all. |
| **R15** | **Magic Coat / reflect path, and Ingrain blocking phazing** | `magiccoat` and `ingrain` are each **0 of 220**. The `reflectable: true` flag upstream puts on Roar/Whirlwind is inert. |
| **R16** | **Dragon Rage and Psywave emit no instructions** | Each **0 of 220**. (Night Shade, also 0, was fixed anyway by the fixed-damage-pipeline patch.) |
| **R17** | **Eruption / Water Spout one-ULP ordering divergence** | Each **0 of 220**. |
| **R18** | **Locked-continuation PP on Outrage / Petal Dance / Thrash** | Each **0 of 220**. The reachable half of that patch is Solar Beam (4 species) and Hyper Beam (Slaking). |
| **R19** | **Snore treated as not sleep-usable** | `snore` is **0 of 220**. |
| **R20** | **Low Kick's weight-based base power, which Transform does not copy** | `lowkick` is **0 of 220**. |
| **R21** | **Reflect / Light Screen keeping a trailing float position in the damage pipeline** | `reflect` and `lightscreen` are each **0 of 220**, as are `safeguard` and `mist`. No Pokémon in the pool can set a screen. (Flagged for re-checking because `engine_world` *can construct* screens as side conditions — but no battle path reaches that state, so the construction capability is not a reachability route.) |
| **R22** | **Mimic, Imprison, Psych Up, Metronome, Assist, Nature Power, Sketch, Mirror Move** | Each **0 of 220**. This closes `_HIDDEN_INFORMATION_REQUEST_FLAGS`'s `maybeDisabled`/`maybeLocked` (Imprison is their only producer), the `failencore` move-list edge cases, and G32. |
| **R23** | **Focus Energy, Mud Sport, Water Sport, Taunt, Torment, Disable, Nightmare, Foresight** | Each **0 of 220** as moves. The `volatile_unsupported` refusals keyed to them cannot fire from play. |
| **R24** | **Attract as a *move*** | `attract` is **0 of 220**. The volatile is reachable only through Cute Charm — see G37, which is the correctly-scoped version of the gap. |
| **R25** | **Sleep Talk calling Haze / Psych Up / Roar / Whirlwind / Baton Pass** | **0 of 393 sets** pair `sleeptalk` with any of the five. Measured on this `sets.json`, not carried from the crate's three-universe count. |
| ~~**R26**~~ | ~~**Trick-style item acquisition reaching White Herb**~~ | ❌ **WITHDRAWN — this verdict was WRONG, and it is the error this whole section is supposed to prevent.** The original evidence was "White Herb's only holders are `deoxys` and `deoxysattack`, and neither movepool contains `trick`". **Trick is `target: normal`** — the *opponent* uses it, so the holder's own movepool is irrelevant. Measured, the pairing occurs in roughly **1 battle in 700**. Reclassified as **REACHABLE** at **G49**. What survives of the original row: Transform does not copy items and Psych Up copies only boosts, so neither is an item-acquisition route — but Knock Off (4 species) is an item *removal* route on the same cross-side footing, handled by `removed_item_species` rather than here. |
| **R27** | **Quick Claw, King's Rock, Bright Powder, Lax Incense, Focus Band, Scope Lens, Berry Juice, Leppa/Oran/Chesto/Pecha/Rawst/Aspear/Persim/Cheri Berry, and every type-boosting item outside the 13** | None is in the 13-item universe. This is worth stating positively because of what it retires: **no priority randomness (Quick Claw), no item-sourced flinch (King's Rock), no evasion item, no crit item, no HP-restoring berry, and no item-sourced heal-on-damage (Shell Bell)** can occur in gen3 randbats. Any engine gap in those mechanics is unreachable here. |

**Two candidates resist the filter, and the second one caught the method out.**

Sandstorm chip damage is *not* reachable via a move (0 of 220) and *is* reachable via Sand
Stream (Tyranitar). It is observed — `divergence_class:component_missing_in_engine:sandstorm` is
the dev row `19000074/27`. A ledger that had checked only the move column would have dropped G7.

**And R26 is the case where exactly that happened, in the other direction.** Checking only the
holder's own movepool produced a confident UNREACHABLE for a shape that occurs about once in 700
battles. The failure is not that the method was unavailable — §4's own Sand Stream note
celebrates catching this same class of error one row earlier — it is that the method was not
*applied* to a row that looked settled. Three rows in this document now carry a ⚠ for the same
family of mistake (R26/G49, R14, G14), and §8 records the rules they produced.

---

## 5. The f32 comparator, re-derived rather than transcribed

C115 §5 states this gap with specific numbers. The instruction for this ledger was to verify it
from the shipped expression rather than copy the figure, so I did.

The shipped expression, from the patched engine source (cited by symbol:
`src/gen3/generate_instructions.rs`, `fn compare_health_with_damage_multiples`):

```rust
let increment = max_damage as f32 * 0.01;
let mut damage = max_damage_f32 * 0.85;
for _ in 0..16 {
    if damage < health_f32 { total_less_than += damage as i16; num_less_than += 1; }
    else { num_greater_than += 1; }
    damage += increment;
}
```

The truth it is meant to represent is the gen3 fan `floor(max * r / 100)` for `r = 85..=100`,
with a roll lethal iff it is `>= health`. Transcribed into exact IEEE-754 single precision
(`numpy.float32`, matching Rust's `f32` add and `as i16` truncation) and swept over
`max_damage` 1..400 and `threshold` 1..max:

```
A) max values in 1..400 whose TOP rung truncates below max ............ 173
B) kill-count mismatches over all (max, threshold) pairs .............. 195
      of which undercounts ............................................ 195
      of which overcounts .............................................   0
C) mismatches at INTERIOR thresholds (threshold != max) ...............  22
      by true rung: r=90 → 14, r=95 → 8
E) max=120, threshold=108: engine 10 kills vs true 11
   max=120, threshold=114: engine  5 kills vs true  6
```

**This reproduces C115 exactly** — 173 / 195 / 195 / 0 / 22 / {90: 14, 95: 8} — including its
two worked examples, and including its correction that the drift is *not* confined to
`threshold == max_damage`. The one-directional finding holds: the engine undercounts in all 195
cases and never overcounts, so the defect always **under**-partitions, which is the safe
direction.

Two things the re-derivation adds:

1. **`scripts/engine_behavioral_probes.py` states 174, not 173.** I checked whether the
   discrepancy was an off-by-one in the range or in the predicate; it is neither. `< max` and
   `== max - 1` give the same 173, and 1..400, 0..399 and 1..401 all give 173. The comment in
   shipped code is wrong; C115 is right. Filed as **H20**.
2. **The gap is reachable and invisible at once.** It is on the KO-threshold split path for
   every damaging move, so reachability is not in question — but it changes *branch masses*,
   and H4 records that the differential compares components rather than masses. A zero-row
   count for G4 is therefore not evidence of absence.

---

## 6. What this ledger implies for how a fidelity claim should be worded

Six constraints, each traceable to a row above.

1. **State the denominator.** Any figure of the form "N divergences over two 200-game windows"
   is a statement about **full-round boundaries**, which are 87.5 % (dev) and 86.7 % (holdout)
   of all boundaries. It must say so, or carry §2. (H1)
2. **State the window, not a rate.** Dev shows 2 divergences and holdout 4 on the same build
   and the same 200-game budget. A single window is not a fidelity rate. (H16)
3. **Say "components", not "transitions".** The comparison is over attributed HP components,
   not branch probability masses. A mass-only defect — which is what G4 and G15 are — produces
   a *match*, not a divergence. "Zero divergences" cannot be read as "zero defects", and the
   two largest registered engine gaps are precisely of the invisible kind. (H4, G4, G15)
4. **Do not launder reachability into correctness.** 27 candidate gaps — real differences from
   gen3 Showdown, or real inexpressibilities — are unreachable in this format (§4, where R27
   bundles a further class of items). The honest phrasing is "gen3 random battles as generated by
   Showdown at `f76228a1`", not "gen3". A fidelity claim scoped to gen3 randbats is defensible;
   the same claim scoped to gen3 is not, and §4 is the exact list of what would have to be
   re-opened.
5. **Do not claim a gap is closed on the strength of a closed row.** G2's two rows closed when
   the *attribution* was fixed; the cross-side modelling gap is unchanged. G9's row closed via
   the hit-count partition, not via the shared-roll defect. A row is evidence about a row.
6. **Name the unobserved exits rather than omitting them.** 36 of 40 world-construction refusal
   reasons, 7 of 8 unmappable-choice reasons and **13 of 19** divergence classes have never
   fired — the last across *all 31* committed artifacts, not just the c136 pair. Some are
   unreachable (R1, R7, R8) and should be retired; the rest are untested code paths sitting
   behind the measurement. §3.5 is the list. (H13, H15)
7. **Say that ~9 % of boundaries were judged under a weaker bar.** `gating:support` is 1,347 of
   15,503 in dev (**8.689 %**) and 1,431 of 15,579 in holdout (**9.185 %**). On those boundaries
   the harness enumerates every legal sleep-counter assignment and accepts if **any** of them
   matches. A widened accept set can only convert divergences into matches, never the reverse,
   so it is the one caveat that biases the headline in the flattering direction. (G24)

   > **The denominator is measured boundaries, not matches — and the two coincide only by
   > accident.** All six divergence repros across both windows carry `gating: exact`, so no
   > support-gated boundary happened to diverge, which makes "9 % of boundaries" and "9 % of
   > matches" numerically equal here. That equality is **contingent on this build and these
   > seeds**, not structural: one support-gated divergence would break it. Quote it as a share
   > of measured boundaries.
8. **A fidelity claim about transitions says nothing about terminal values.** G0 is a
   win/loss/tie error that no boundary comparison can ever surface, because a terminal is not a
   boundary. Any claim that reads as "the engine agrees with Showdown" must either exclude
   terminal adjudication or state G0. (G0)

A wording that satisfies all eight looks roughly like: *"On gen3 random battles as generated by
Showdown `f76228a1`, over two disjoint 200-game seed windows, the engine's attributed HP
components matched Showdown's on 15,501 of 15,503 measured full-round boundaries in dev and
15,575 of 15,579 in holdout — 87.5 % and 86.7 % of all boundaries respectively. Branch
probability masses were not compared, and 27 documented gap candidates are unreachable in this
pool and therefore untested by it. About 9 % of those matches were accepted under a widened
sleep-counter bar rather than an exact one, and terminal adjudication was not compared at all."*

**Six constraints became eight** after review. The two additions (7 and 8) are both cases where
the first version of this document stated a *number* correctly and drew no consequence from it:
`gating:support` was in the G24 row and G0 was missing entirely. A ledger that lists a caveat
without wiring it into the recommended wording has not actually protected the claim.

---

## 7. What I could not determine

Stated explicitly, because a ledger of blind spots is the worst place to imply completeness.

1. **Whether G1 (Stick) currently produces a differential row.** The pool reachability is
   settled and deterministic — `getItem`'s Farfetch'd branch is unconditional, so the rate is
   100 % by construction, and G1 explains why the raw count that once stood here was dropped as
   seed-scheme dependent. What is *not* settled is whether the differential ever
   hands the engine a Farfetch'd holding Stick — items are not public until revealed, and Stick
   has no reveal event. **Settling measurement:** a 200-game dev sweep with an assertion on
   `PokemonSpec.item` recording every distinct item string reaching `build_poke_engine_state`;
   if `stick` appears, the crit-rate divergence (1/16 vs 1/4 on Return) is live in the compared
   population, and if it does not, the gap is confined to search-side worlds. I did not build
   an engine for this document, so I did not run it.
2. **How much matched mass rides on the comparator's ±9 % fallback window** (H8). The
   state-level counter is 0 in both windows; the per-branch usage is not counted at all.
3. **Whether c43's ~7,224 invisible shortfall rows still exist** (H12). The full-round path now
   reconciles exactly, which is evidence against it *within that path*, but the single-seat arm
   carries no exit taxonomy so nothing can be said about it.
4. **Whether the four never-adjudicated families survive into the current era** (H19).
5. **Whether the generator can actually draw both moves** in any co-occurrence check that came
   back non-zero. Every *same-side* pairing verdict I relied on for an UNREACHABLE came back
   zero, which is decisive. The non-zero ones — `sleeptalk`+`rest` (44 sets), `wish`+`protect`
   (24), `leechseed`+`substitute` (6) — are upper bounds, not measured draw rates. **G14 is the
   row where I read such an upper bound as a reachability fact and was wrong**; it is corrected
   in place, but the same trap applies to any future row built this way.
6. **How much of G0's population is a *last-mon* double faint.** Double faints are reachable and
   demonstrated; the fraction where both sides are down to their final Pokémon is not measured.
   **Settling measurement:** count games in a 200-game dev sweep ending in a same-ply double
   faint with both parties at one remaining Pokémon. This bounds G0's incidence but not its
   severity, which is per-occurrence large.
7. **The incidence of the missing Protect `|-fail|` line** (C145 §4.5, added 2026-08-07 —
   `reports/c145_itemleftovers_row_adjudication.md`). Showdown's gen3 `protect.onPrepareHit` is
   `!!this.queue.willAct() && this.runEvent('StallMove', pokemon)`, so a Protect the opponent's
   switch pre-empted **fails on the first conjunct**; the engine renders no `|-fail|`. The mechanism
   is settled and the *cost on the observed shape is zero*, measured rather than argued: on
   `19100170/71` with the lock corrected, `generate_instructions` emits **no**
   `ApplyVolatileStatus PROTECT` and **no** `ChangeSideCondition Protect`, against a
   single-variable control (only p2's choice changed from a switch to a move) that emits both — so
   the stall ladder does not move and matches Showdown, and no HP changes either way. What is
   **not** measured is whether any shape exists where the omission costs more than a protocol line.
   **Settling measurement:** a dev sweep counting boundaries where a side's chosen or Encored move
   is a stalling move and every opposing action is a switch, cross-checked against
   `side_conditions.protect` at the following boundary. Filed here rather than in §3 because C125
   (§8) forbids a §3 row without a recorded pool-reachability check, which I have not run.
   ⚠ **This entry exists partly to keep §7 honest:** the same PR *removed* an item (H11's), and a
   list of blind spots that only ever shrinks is losing them by attrition rather than by resolution.
8. **Nothing here was measured on the reserved final holdout** (`19,200,000+`), deliberately.
   Per the §J.7 amendment it must appear in exactly one measurement in the whole record.
9. **No new sweep was run and no engine was built for this document.** Every "observed" column
   reads `reports/artifacts/c136_faintcancels_fix_{dev,holdout}_sweep.json`. A gap whose
   incidence changed since commit `aeaee2b1` would be mis-stated here.

> **RESOLVED and removed from this list, 2026-08-07.** Item 2 was *"the class of `19100170/71`
> and `19100170/72` (H11) — observed, open, and undiagnosed anywhere in `reports/`. I did not
> attempt a diagnosis because it needs a branch dump I would have had to regenerate."* The branch
> dump was regenerated and the class is settled: a world-construction fix, closed by `d27316b6`
> (#1148), diagnosed in the H11 cell above and in
> `reports/c145_itemleftovers_row_adjudication.md`. The items below it were renumbered; `§7.1` is
> the only one cross-referenced (from §1 and §8) and kept its number. The clause *"undiagnosed
> anywhere in `reports/`"* was already false when written — `reports/c139_encore_transform_move_index_prediction.md`
> had merged, in #1148, before this document did.

---

## 8. Standing rule, restated and widened

C125's rule was: no entry joins the ledger without a recorded pool-reachability check **and the
instrument that answered it**. That stands. This pass adds two clauses, both earned by a check
that would otherwise have gone wrong:

- **For a gap that needs two things at once, the instrument is per-set co-occurrence, not
  per-species presence.** Haze and Sleep Talk are each reachable; the Sleep-Talk-calls-Haze gap
  is not.
- **For a gap in a *renderer* or *classifier*, pool reachability of the mechanic is necessary
  but not sufficient.** Liquid Ooze is reachable and its residual-heal-renderer gap is not,
  because a negative `Heal` is intercepted before that code runs. The reachable thing must be
  the code path, not just the mechanic.
- **Check the move's `target` before writing a movepool-absence verdict.** For a `target: normal`
  move the user is the *opponent*, so "species X cannot have move Y" is not evidence that X
  cannot be *hit* by Y. A cross-side verdict is a product of two independent per-team rates, not
  a movepool lookup. This is the rule R26 was missing, and R14 needed it too.
- **A movepool is an upper bound on a draw.** A four-move draw from a six-move movepool does not
  realise every subset. Where a row needs a *combination* of chosen moves — as G14 does — the
  movepool proves possibility and only generation proves reachability.
- **Quote rates, not raw counts, for anything seed-dependent.** Two of this document's numbers
  failed to reproduce for a reviewer running a different seed scheme, both harmlessly. Presence,
  absence and 100 %-by-construction facts are seed-stable; per-species tallies are not. Where a
  rate is load-bearing it is now measured across three seed schemes.

**What this revision changed, so the diff is legible.** One UNREACHABLE was withdrawn as wrong
(R26 → G49); two UNREACHABLE verdicts survived with their evidence replaced (R14, R9); one
REACHABLE verdict survived with its reasoning replaced (G14); one UNKNOWN was settled by a
static argument stronger than the sweep it had asked for (G28); one row was materially
re-characterised against its own artifact (G24); four counts were corrected (H5's ranking, H15's
13-of-19, G43's 5 rows, §6's 15,501); and **nine** gaps were added that the first pass missed,
taking §3 from 68 rows to 77 — G0, G21b, G43b, G43c, G49, G50, G51, H5b, H5c. (An earlier draft
of this very sentence said "eight" and then listed nine, which is the smallest possible version
of the same failure the rest of the document is about.)

**Second review round.** Six numbers were fixed in place, none of them changing a verdict:
G21b's transposed `shadowball`/`substitute` PP (and "base" → "max"), §7.1's retracted 435/435,
R7's stale 12,000 denominator, §8's own miscount above, G49's overstated precision, and
Constraint 7's denominator. One gap was added — **H21**, taking §3 to 78 — because the review
traced Constraint 7 end to end and found the `--approximate-sleep` help string still describes
a strict-skip default that has not been the behaviour since hidden-counter support landed. That
stale string is the most plausible origin of G24's original error, which makes it a gap in its
own right rather than a typo. Review also verified Constraint 7 is **understated**, not
overstated, and refuted its own first estimate of G49's rate as having been drawn from
correlated seed streams; rebuilt from disjoint pairs it agrees with the figure recorded here.
