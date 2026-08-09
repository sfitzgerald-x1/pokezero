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

**Shape of the document.** §3 carries **82 rows** (58 engine/renderer/leaf, 24 harness/process);
§4 considers **27** candidates and drops **26** as verified unreachable (R26 is struck — it was
withdrawn as wrong and reclassified at G49, so it is not one of the drops); §3.5 names every exit
the differential can emit and neither window did. A handful of §3 rows carry an UNREACHABLE
verdict in place rather than moving to §4 — G28 and G32 outright, and the split rows G12, G16 and
G27 in half — because they are named, cited gaps whose *status* a reader needs at the place they
will look for them.

⚠ **The row count was stale by two, and both halves drifted the same way (C146, 2026-08-07).**
It read "78 rows (55 engine/renderer/leaf, 23 harness/process)". §8's last block records §3 at 78
after H21, and **two** rows have joined since without the header following: **G33b** (C143) and
**G37b** (#1166). Re-derived here by counting `| **<id>**` rows under §3.0–§3.4 rather than by
adding to 78 — §3.0 1 + §3.1 34 + §3.2 12 + §3.3 23 + §3.4 10 = **80**, cross-checked as 57
G-rows + 23 H-rows with no duplicate ids. ⚠ **80 → 81 on 2026-08-08 (C152)**: **G33c** joined
§3.2, taking it 12 → 13 and the G-rows 57 → 58. Re-derived the same way — by running
`tests/test_ledger_table_uniformity.py`'s own row selector over both trees, not by adding one —
and that module's exact row inventory is bumped in the same commit, which is the mechanism that
forces an author through the column beside a new gap. Deriving it instead of incrementing is what found the
second row: a count bumped from 78 to 79 for G37b alone would have been wrong in the same way and
indistinguishable in the diff.
⚠ **81 → 82 on 2026-08-08 (C153)**: **H22** joined §3.3, taking the H-rows 23 → **24** and leaving
the G-rows at 58. Re-derived by running the same selector over both trees — 81 at `7fcd9e19`, 82
here — and **the first draft of this very change got it wrong in exactly the forbidden way**: §8's
C153 block read *"H22 added, taking §3 to 79"*, incremented from the **stale 78** that the
paragraph above had already corrected, so the document briefly asserted three different counts at
once. Caught in review, not by the author. The lesson the paragraph above states is now enforced
rather than restated: `tests/test_ledger_table_uniformity.py::test_the_stated_row_count_matches_the_rows`
parses this sentence's three numbers and re-derives all of them from the tables, the way §1.1's
UNKNOWN count has been pinned since C152. **An unpinned count in this document has now gone stale
three times.** The §4 figure is corrected in the same spirit — 27 candidates were
*considered*, 26 are *drops*, because R26 was withdrawn. A document whose subject is uncounted
drift may not carry an uncounted count.

> ⚠ **Read §8's last block first if you are about to cite a negative from this document.** Five
> "never fired"-shaped claims here have been false, four of them corrected on 2026-08-07 (H14 via
> #1163, §3.5's count via #1162, H11 via #1165, and H13/H15/H17 via C146). The ones that survive
> now carry the glob that establishes them and are marked ✅; the ones that do not are marked ⚠ in
> place. `reports/c146_negative_claim_audit.md` is the full inventory of which is which, and
> `tests/test_never_fired_counter_census.py` re-derives the counter lists from the artifacts so
> the next recurrence is a red gate rather than a citation.

> **Revision note.** ⚠ **2026-08-09 (C154): all 26 §4 verdicts were re-adjudicated against C153's
> rule and all 26 survive; THIRTEEN of their stated mechanisms do not** — four false, nine
> incomplete, each corrected in place in the row's own cell and traced in
> `reports/artifacts/c154_unreachable_readjudication.json`. Nothing was closed and nothing opened.
> The paragraph below is the earlier post-review revision. One §4 verdict was **withdrawn as wrong**
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
classification. ⚠ **ZERO** rows below are UNKNOWN, as of 2026-08-08 (C152). It read **three** —
H8, H12 and H19 — and all three are settled in place, two of them by finding that the cell's own
stated basis was defective rather than by answering a hard question: H8's named mechanism
(`pre_legal` unavailable) contributes **0 of 371** measured window accepts, H12's named settling
measurement is scoped to a population c43 explicitly **excluded**, and H19's evidence was absence
from a counter vocabulary the family layer **never writes to**, behind a settling script that
raised `NameError` on every input. See `reports/c152_ledger_terminal_disposition.md`. (H11 was the
fourth until 2026-08-07, when `reports/c145_itemleftovers_row_adjudication.md` classified it; an
earlier draft said six: it counted G1, which is classified REACHABLE and merely carries an open
*follow-up* measurement in §7.1, and G28, which is now settled UNREACHABLE by a static argument.)

⚠ **A fourth verdict now exists in practice and is named here rather than left implicit:
RETIRED.** C152 retires exactly **two things: G8, and G33b's order-8 weather arm.** Each is a real,
measured mechanism whose observed consequence is zero **in the two permitted windows**, retained as
a named property rather than as an open gap. An audit has already reversed one "closed" that was
really a fourth category, so a retirement is labelled as one and always carries three things: the
measurement, the scope, and the condition under which it comes back.

⚠ **Two corrections to an earlier draft of this very paragraph, both caught in review, and both
are the failure modes the rest of this document is about.** It read *"C152 disposes G8 and both
open arms of G33b as consciously retired"* — **false, and it inverted the headline finding.**
G33b's **speed-tie arm is OPEN**, as its own cell says at `| **G33b** |` (*"THE OTHER MEASURED AND
STILL OPEN"*), as §3.5 says, and as §7.1 says. This is the sentence a reader uses to judge whether
the ledger is terminal, and it is the **same inversion C152 caught and corrected one draft earlier
inside the G33b cell**, reappearing in the summary. A correction made in a cell does not propagate
to the paragraph that summarises it. It also read *"zero in every population this program is
permitted to measure"* — **over-broad, by this document's own §8 rule, added in the same
revision.** Unregistered seeds are such a population, this pass measured **80,439** boundaries
there, and **G8's shape was never checked against them.** The scope is the two permitted windows,
and it now says so.

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
| **G8** | **The collapsed lethal arm mis-prices a roll-dependent Leech Seed drain** (`hp_after_move + leftovers < maxhp/8` clamp). `reports/c111_residue_row_causes.md` Addendum 2. ⚠ **DIAGNOSED 2026-08-07** — `reports/c140_last_dev_row_diagnosis.md`. Not a limit: the enumeration oracle emits the arm at the observed roll (−109, mirror heal 28) and the shipped comparator accepts it, measured. Over the 7-roll lethal band the drain is **injective** (108→29, 109→28, 110→27, 111→26, 112→25, 113→24, 115→22), so a fixed representative prices **exactly one** of the seven rolls and the shipping engine is already at that bound. ⚠ **CORRECTED 2026-08-07 (same PR):** an earlier version of this cell said "Not closable by a representative either." **That is false** — representative 109 closes it, measured on the full 7×7 representative-versus-thrown-roll matrix, which is the identity (c140 §6a). Re-pricing 108→109 is a **trade**, not a fix: it opens the boundaries where Showdown throws 108, at equal `1/16` mass, and it would be fitting an engine constant to a single dev observation. The engine's own `floor(mean(band))` convention — the rule that yields the `−101` survive representative — gives 111, which also diverges. Disposition: **closed by enumeration, retained under the collapsed path; this gap stays OPEN.** ⚠ **SECOND CONFIRMED INSTANCE 2026-08-07 — `19200244/115`, final holdout** (`reports/c143_heal_attribution_diagnosis.md`). Same arithmetic, second window, different HP configuration: maxhp 407 / hp-after-move 11 / Leftovers 25 / mirror 36, against c140's 235 / 14 / 14 / 28. The band is **14 rolls wide**, not 7, and the mirror is injective over all 14 — so the gap is not specific to one HP configuration and the "one arm prices one roll" bound is a property of the band, not of the row. **It also exhibits a sub-case c140 did not measure, and the shipping engine is BELOW c140's bound in it.** c140's arm was priced *at a residual threshold* (108, a fan member) via `residual_disjoint_bands`; here every residual threshold lies below the fan minimum, that function's `min_roll < threshold` guard cannot pass, and the single non-KO arm is priced at the **survive representative** — `sum(band)//len(band) = 145`, which is **not a member of its own 16-roll fan**. Its mirror 37 is therefore unachievable, and the row matches **0 of 14** rolls rather than 1 of 14 (measured over the **whole band plus the off-fan shipping value — 15 representatives × 14 columns, 210 cells** — through the unmodified shipped comparator). **c140 §6a's "exactly one" needs two scope conditions its own matrix satisfied without stating them**: the representative must be inside the fan, and its mirror must not saturate the seeder's max HP. Both excluded cases are exhibited here — a non-fan representative prices **zero**, and the two saturating ones price **two**. ⚠ **The rule doing that is NOT H8's proportional window, and an earlier revision of this cell said it was.** H8's `[0.92·eng − 1, 1.09·eng + 1]` around the engine's cap of 45 is `[40.40, 50.05]`, which would admit **five** achievable mirrors (47, 46, 44, 43, 41) against the **two** measured. The binding rule is the `_to_full` branch of `roll_components_agree` (`scripts/engine_transition_differential.py:984-1020`), which `continue`s before any later window: its magnitude bound `abs(abs(obs) − abs(eng)) > _damage_difference + 1` with `\|49 − 45\| = 4` gives `[40, 50]`, and its **direction** test `if _obs_damage > _eng_damage and abs(obs) < abs(eng): return False` forces the mirror `>= 45` — leaving exactly {47, 46}, confirmed by direct calls to the shipped function (44/43/41 rejected, 45/46/47 accepted). That tolerance is **adjudicated, not incidental**: `docs/engine_divergence_ledger_20260728.md` §B.4 (`:1005`) filed the `magnitude:heal` class UNRESOLVED and §C.2 (`:1278`) settled it as a matcher defect, prescribing that a capped heal is roll-scaled with *"only the magnitude relaxed, and only in the capped direction (clipping can only reduce, so the test is an **inequality, not a window**)"* — which is the direct refutation of the H8 window attribution, and whose worked case (seed 1310001/72, 251 from 2 HP against 247 from 6 HP) is the same one the code comment names. `I3_roll_inherited` (**H19**) is the still-unadjudicated remainder of the shape, tied to B.4 by `reports/c101_i3_painsplit_tolerance_derivation.json:43`. It is **not** H8's `pre_legal`-absent fallback; mis-attributing it inflated H8's reach, which matters because H8's own cell says "UNKNOWN how much" mass rides on it. **Saturation is a census, not a sample:** only representatives 135 and 136 make the repricer write a `268/268` line, so they are *the* saturating members of the band. All seven of c140's band values were fan members and none saturated, so its bound is sound within its scope; read it as scoped. **Consequence c140's row did not have:** because 145 prices zero rather than one, re-pricing this arm to a non-saturating fan member is a strict **0 → 1** gain here, not the even permutation c140 §6a(ii) analysed — so the "wash" argument against re-pricing does not apply to an off-fan representative, though c140's other two objections (fitted to the sample; the engine's own convention is what produces 145) do. Snapping a mean-priced representative to the nearest fan member is filed as a **candidate rule change, unbuilt and unswept** — and note the nearest are **144 and 146, a tie at distance 1, of which only 146 closes this row**, so the principled snap closes this boundary only if the tie-break is chosen to fit the sample. A **third** collapse sits on the same boundary, on a **different code path**: Fire Blast into Moltres is survivable across its whole fan, so it takes the `max_damage_dealt < defender_active.hp` branch, whose rule is `(max_damage_dealt as f32 * 0.925) as i16` = `49 × 0.925 → 45` (`generate_instructions.rs:4027`), against an observed 49. ⚠ An earlier revision called this a fan mean; `720//16` is also 45 here, which is a coincidence of this fan and not the rule. Either collapse alone would block an arm, so the collapsed path's failure here is over-determined. Second instance carries a companion renderer defect, filed separately as **G33b**; that one is what made the row's class string look novel. ⚠ **CLOSED ON THE DEV WINDOW BY AN ENGINE CHANGE, 2026-08-08 — `reports/c149_g8_leechseed_band_split.md`, prediction registered first at `7613f3e0`.** c140 §7a is now BUILT: at the two `residual_disjoint_bands` call sites whose `ceiling` argument is `i16::MAX` — the non-crit and crit sites where the fan cannot kill on the hit, verified by reading all four call sites rather than carried from a handoff — a residual-kill band whose defender carries `LEECHSEED` emits **one arm per roll of the band at `1/16` each** instead of one arm at the threshold. Roll values come from the EXACT INTEGER fan `floor(max * r / 100)`, not from `compare_health_with_damage_multiples`'s f32 accumulator, so no arm is ever priced at a damage Showdown cannot deal; the band's mass still comes from the comparator's own count, and when the two bases disagree about the band size the split **declines and keeps today's single arm**. That fallback is reachable and measured, not theoretical: **288 of 27,318 `(max, threshold)` windows disagree, 1.054 %**, over `max_damage` 10–600 and every threshold above the f32 fan floor (`reports/artifacts/c149_fan_basis_census.json`). The consequence is the property that distinguishes this from c140 §7c: **on any given band the split is a strict improvement or a no-op, never a trade -- given that `max_damage_dealt` equals Showdown's own fan maximum**, because a band that would lose its matching arm keeps it. That condition is load-bearing and the patch's own doc comment states it unconditionally: the threshold comes from HP arithmetic rather than from the fan, so the guarantee rests on the engine's max agreeing with Showdown's. **Measured at 0 instances, and settled structurally.** ⚠ **FIGURES REPLACED 2026-08-08 (C150).** This cell previously cited an independent-review scan of **124,188 fixtures** — "split fired 44,393, declined 79,795; of 25,728 collapsed arms kept and 18,901 dropped" — for which **no artifact was ever committed**, and whose numbers do not reconcile with each other (`25,728 + 18,901 = 44,629`, neither the 44,393 fires nor the 79,795 declines). The conclusion is unchanged and was never in doubt; the numbers were unauditable, and this is the document five false "never fired" claims propagated from precisely because citing it felt like verification. What settles it is **structural**: `ResidualThresholdLadder::insert` is an insertion sort with dedup, so thresholds strictly ascend, `band_window` therefore always yields `upper > lower`, and a threshold that IS a fan member always lies inside its own half-open window `[t_i, t_i+1)` — so if the collapsed arm sat on a damage Showdown could throw, the split emits an arm at exactly that damage. A dropped arm is therefore always at a non-fan damage, i.e. never a real trade. **Now artifacted** (`reports/artifacts/c150_band_split_trade_census.json`, reproducible via `scripts/c150_band_split_trade_census.py`): over `max_damage` 10–600 and every half-open window the shipped partition can emit an arm for — **838,560 windows examined, 796,878 non-empty bands** — the split fires on **783,903** and declines on **12,975**; **207,868** collapsed arms are kept and **589,010** dropped, and **real trades = 0**, with both closure identities checked (`kept + dropped == bands`, `fired + declined == bands`). The transcription is validated rather than trusted: it re-derives all three committed crate fixtures, including fixture C's decline (`max_damage` 30, threshold 27, comparator 10 rolls against the integer fan's 11) and the exact `105.859375 %` guard-deleted mass the crate test pins. **`19000191/63` closes**: 14 branches / `diverged` / 12 misses → **38 branches / `matched` / 0 misses**, mass 100.000000 % both sides, single-variable across two builds of the same tree (`de29e3dc79c80659` and `8e912b45544034e6`). The mirror-heal census goes `{29: 93.9062 %}` → `{22, 24, 25, 26, 27, 28: 5.7129 each; 29: 59.6289}` — **byte-identical to what the enumeration oracle emits on the same boundary**, measured on ONE build with `POKEZERO_ENUMERATE_ROLLS=1` at 1,015 branches, so 38 collapsed arms reproduce the oracle's mirror distribution exactly. **Both permitted windows swept on both builds and NOTHING OPENED**: dev 15,503 measured, `transitions_diverged` **1 → 0**, `transition:matched` 15,502 → 15,503, the three divergence-related counter keys removed and **every other counter identical**; holdout 15,579 measured, 0 → 0, the whole `counters` block **byte-identical** to the same-tree base. Zero holdout rows closed was the registered expectation — that window had no divergence left — so it is a safety measurement and cannot corroborate the mechanism. Pinned by **ten** crate tests. The gate deletion (`if defender_leech_seeded` → `if true`) reddens **three** -- both `a_poisoned_defender_*` controls and `the_split_conserves_the_bands_mass_and_prices_each_arm_at_one_sixteenth`, whose split-vs-collapsed sanity assertion fails once both fixtures split -- so the two controls are the only tests that STATE the gate as a property but are not its sole killers; an earlier revision of this cell said they were. A full revert to the 73-patch preimage reddens 5 of the 8 original tests. **The count guard needed its own fixture and had none:** review found all eight original tests, the whole crate suite, `--test test_gen3`, the engine lib suite, `test_branch_mass_reconstruction` and `test_collapsed_arm_mass_oracle` stayed GREEN with the guard deleted, because both fixtures sit on fans whose two bases agree. Two tests now reach the decline path (attack 66 / defense 300 / maxhp 160 / hp 47, threshold 27): head keeps the single arm at 27, and guard-deleted emits `[27, 28, 29, 30]` with total branch mass **105.859375 %** -- mass conjured from a basis mismatch, which `update_percentage` cannot see. Guard efficacy, re-measured and artifacted in C150 after the unartifacted "**0** on head against **115** on the guard-deleted mutant" figure was withdrawn with the rest of the 124,188-fixture scan: a declined band is *exactly* a band whose two fan bases disagree, so head has **0** band-mass disagreements by construction (it declines) against **12,975** on the guard-deleted mutant over the census's 796,878 bands (`reports/artifacts/c150_band_split_trade_census.json`). **THIS GAP STAYS OPEN.** The second instance `19200244/115` is untouched and unreachable by this change — its arm is priced at the **survive representative**, not at a residual threshold, because every threshold there lies below the fan minimum and `residual_disjoint_bands`'s `min_roll < threshold` guard cannot pass — and the two `defender_active.hp`-ceiling call sites were left alone for blast radius, **not** because they are immune to the same arithmetic. Both remainders are unmeasured, and the final holdout was not swept. ⚠ **RETIRED 2026-08-08 (C152) — `reports/c152_ledger_terminal_disposition.md` §2. Not closed, not fixed: consciously retired, with the first remainder measured.** Three things changed, each measured rather than argued. **(i) The unreachability is now instrumented, not read.** An `eprintln!` inside `residual_disjoint_bands`, placed BEFORE its `applicable_len == 0` early return so it fires on every call, was run against the row replayed from the committed C141 artifact (`reports/artifacts/c152_g8_call_site_trace.json`, throwaway build `89797289…`, head `bfdbe1c04876edcd`). Exactly three calls occur. The only one carrying a threshold is `max_damage 159 / min_roll 135 / thresholds [82] / ceiling 157`, where `min_roll < *threshold` is `135 < 82` — false — so `applicable_len == 0` and the function returns `None`. **And one thing the audit's static argument did not say:** the two call sites C149's split actually touches are the `i16::MAX`-ceiling ones (**`generate_instructions.rs:4300` and `:4456`**, against `defender_active.hp` at **`:4197`** and **`:4406`**, all four read — ⚠ **these four were 4208/4311/4417/4467 in an earlier revision, which are the INSTRUMENTED build's**, shifted +11 by the eleven-line `eprintln!` block the trace needed; re-derived from the SHIPPING vendored tree at fingerprint `bfdbe1c04876edcd` with each site's `ceiling` argument re-read at the corrected line), and on this boundary **both are reached with an EMPTY threshold slice** — so the split is unreachable here twice over, not merely filtered. The call that does carry a threshold is at an `hp` ceiling, one of the two sites C149 left alone. **(ii) The first remainder is measured** (`reports/artifacts/c152_g8_survive_representative_census.json`, reproducible via `scripts/c152_g8_survive_representative_census.py`, which exits 2 without writing unless its model first reproduces this row exactly — 14-roll band, minimum 135, representative 145, 145 off-fan). Over `max_damage` 10–600 and every `health` to the fan maximum: **180,592 windows examined, 27,655 carrying a survive band, of which the representative is OFF the exact integer fan `floor(max·r/100)` in 16,205 — 58.597 %** — and every one of those prices **zero** achievable rolls rather than one. `on_fan + off_fan == bands` checked. ⚠ **Read the scope or this number misleads:** it is an ARITHMETIC census over a synthetic uniformly-weighted plane, not an incidence over boundaries — and the health axis runs to `max(f32 accumulator rolls) + 1`, which is not `max_damage` (at `max_damage 159` the accumulator tops out at **158**), so "the fan maximum" is the accumulator's, not Showdown's. The same engine measures **0 divergences over 31,082 boundaries** across both permitted windows at head. ⚠ **AND THAT ZERO CARRIES TWO ACCEPT BARS, which §6 of this document now forbids omitting** — support-gated acceptance at **8.689 % dev / 9.185 % holdout**, and the ±9 % roll window at **167 dev / 140 holdout matched boundaries (1.077 % / 0.899 %)** measured in the same pass (H8). That second bar is not incidental to G8: the dominant class the window absorbs is `roll_scaled_component` (**158 dev / 129 holdout** of the 167 and 140), which is exactly what an off-fan survive representative produces. So some unknown part of this row's zero is the comparator's tolerance rather than agreement, and the honest reading is that the mechanism is arithmetically common, its consequence is **not separately measured beneath the tolerance**, and no divergent row survives it in the two permitted windows. The census sizes the mechanism; the sweeps size the consequence; quoting either alone misstates this row. **(iii) The instance can never be re-measured and already matches on the certifying path.** `19200244/115` sits inside `19,200,000`–`19,200,259`, which `_reject_burned_final_holdout` refuses **unconditionally** — `--final-holdout-i-mean-it` does not open it — on a window C151 demoted to dev-grade evidence. Replayed at head: **collapsed `diverged` / 9 branches / 9 misses; enumeration oracle `matched` / 416 branches / 0 misses**, which re-derives C147's closure at the current head after both merges. C141's OTHER row, `19200131/129`, now re-reads **`matched` / 0 misses / 4 branches on the collapsed path**, so neither C141 row is a live divergence. **Why retired rather than fixed.** The candidate is C150's filed snap-to-nearest-fan-member, and this census adds a third objection to C140 §6a(ii)'s two: at `max_damage 159 / health 157` the nearest fan members are **144 and 146, a tie at distance 1, of which only 146 closes this row** — so the principled snap closes it only if the tie-break is chosen to fit the sample. Building it would also be an engine change whose measured benefit is **zero rows in both permitted windows**, validated by sweeps that cannot see it. **Stated scope of the retirement:** the mechanism is real, reachable and measured; its observed consequence is 0 divergent rows in 400 games at head — on the two permitted windows, at the two accept bars above — and 0 under the oracle on the one historical instance. ⚠ **The scope is those two windows and nothing wider.** G8's shape was never checked against the 80,439 boundaries this pass measured on unregistered seeds; by this document's own §8 rule, added in the same revision, a negative measured only inside the permitted windows is a claim about those windows. Retired as a ledger-blocking entry, RETAINED as a named property of the collapsed path — **if the collapsed path is ever certified directly rather than through the enumeration oracle, this comes back.** The SECOND remainder — the two `defender_active.hp`-ceiling call sites — is still unmeasured and is now the only unmeasured part of this row. | E | REACHABLE (as G2, plus Leftovers, which is 72 % of all generated items). | **was yes**; ⚠ **RETIRED (C152)** — `19000191/63` closed by C149; `19200244/115` matches under the oracle, sits on a burned and demoted window, and cannot be re-measured |
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
| **G21b** | **The engine does not decrement PP above 10 at all — for either side, on every move.** This is the *engine* half that G21's harness half hides, and it is the wider of the two. `gen3/generate_instructions.rs`: `if active.moves[&choice.move_index].pp < 10 { … }`, guarded by the upstream comment *"most of the time pp decrement doesn't matter and just adds another instruction so we only decrement pp if the move is under 10 pp since that is when it starts to matter"*. Corroborated by `poke-engine-gen3-transform.patch` approximation 2, which calls it a "pre-existing cost-tracking threshold … it applies to every move equally". Directly undercuts the two mechanics G21 itself names: **Struggle onset** (a side is only Struggle-locked once every slot hits 0) and **Encore's PP-zero termination** (`move_fails_encore(...) \|\| move_slot.pp <= 0`). | E | REACHABLE — every battle, every move above 10 PP. Most of the pool sits above the threshold. Gen3 **max** PP at the default 3 PP Ups is **24** for `shadowball` (38 species, base 15) and **16** for `earthquake` (72), `rockslide` (42), `substitute` (75) and `toxic` (114) — all base 10. ⚠ An earlier draft transposed the `shadowball` and `substitute` figures and called them "base PP"; base is 15/10/10/10/10 and none of it is above the threshold. The verdict is untouched — all five max values exceed 10 — but a transposed number in a ledger invites exactly the re-derivation the ledger exists to prevent. | no |
| **G22** | **Encore's elapsed-turn counter is seeded at a floor of 1.** `engine_world.py` `_build_side_spec` (encore block): the true elapsed count is not observable from the request. Gen3 Encore runs `this.random(3,7)`. | E/H | REACHABLE. `encore` on **16 of 220** species. | no directly; `skip:world_unsupported:encore_move_unknown` fires 2 dev / 1 holdout |
| **G23** | **Substitute health after a surviving hit is unknowable, so the world fails closed.** `engine_world.py` `approximate_substitute_health`; `reports/c16_substitute_depletion_prediction.md`. | E/H | REACHABLE, and common. `substitute` on **75 of 220** species — the single most widespread non-`toxic` move in the pool. The exactly-derivable depletions it names (Seismic Toss reachable at 11 species; Night Shade, Dragon Rage, Sonic Boom all 0) are mostly unreachable. | **yes as an exit** — `limit:world_substitute_health_unknown` 131 dev / 139 holdout |
| **G24** | **A sleeping active cannot be expressed exactly, and the differential recovers by *weakening its own accept bar*.** `slp` is deliberately absent from `_STATUS_CODES` (`engine_world.py`: "guessing them biases wake-up odds (fail closed by default)"), so the strict build raises `status_unsupported`. `approximate_sleep_turns=True` — the "just fell asleep" approximation — is a **search-POC** opt-in and was **OFF** in both c136 sweeps (`approximate_sleep_turns: false` in the artifact). What the differential does instead, traced end to end: `--approximate-sleep` is `action="store_true"`, so it is opt-in and default **OFF**; with it off an underivable counter raises `status_unsupported`; `hidden_counter_recovery` maps that to `"sleep"`; the world is rebuilt with `approximate_sleep_turns=True`; and `_sleep_counter_variants` then enumerates `sleep_turns` 0..`MAX_SLEEP_TURNS`(4) plus `rest_turns` 1..`MAX_REST_TURNS`(2) — **7 variants per asleep side**, cross-product truncated at `MAX_HIDDEN_COUNTER_WORLDS = 64`. `evaluate_boundary_strict` then returns matched on the first passing branch of **any** variant. That is not a wrong guess, it is a **widened accept set** — the more dangerous shape, because it can only ever convert a divergence into a match. ⚠ An earlier draft of this row said the approximation was "default ON" and described a biased guess. Both were wrong: the flag's own default was misread, and the mechanism is enumeration, not approximation. | E/H | REACHABLE and dominant. `hypnosis` 11, `sleeppowder` 9, `spore` 3, `lovelykiss` 1, `yawn` 1 species. | **yes, and it is the largest single caveat on the match count** — `hidden_counter_support:sleep` 1,352 dev / 1,435 holdout, and `gating:support` 1,347 dev / 1,431 holdout against `gating:exact` 14,156 / 14,148. So **8.7 %** of dev and **9.2 %** of holdout matched boundaries were accepted under the widened bar, not the exact one |
| **G25** | **`approximate_partial_trap_turns`: the engine models PARTIALLYTRAPPED with no duration counter at all**, so the trap is *unbounded* and one-sided in the trapper's favour. `engine_world.py` `_build_side_spec`. | E/H | **REACHABLE but vanishingly narrow.** The pool's only partial-trap move is `wrap`, on **Shuckle alone** (1 of 220). Bind, Clamp, Fire Spin, Whirlpool, Sand Tomb are all 0 of 220. | no |
| **G26** | **`approximate_hidden_duration_volatiles`: confusion and Yawn duration are guessed.** `engine_world.py` `_build_side_spec`. Note the comment's confusion half ("never expires it inside a search") is **stale** — `third_party/poke-engine-gen3-confusion-duration.patch` added `chance_confusion_ends`. | E/H | REACHABLE, but both halves are one-producer thin: confusion only via Signal Beam's 10 % secondary on 3 species (G34), Yawn only on Swalot (1 of 220). Consequently `hidden_counter_support:confusion` fired **1 time in dev and 0 in holdout** across 400 games and 32,123 full-round boundaries — the whole confusion-widening machinery rests on that one observation. | 1 dev event |
| **G27** | **`pending_hp_reading_move` is an enumeration that has been found short twice.** `reports/c96_unattributed_source_level_causes.json`, "recorded as an open question". | E | **SPLIT.** Pain Split (added in c128): REACHABLE, 4 species. Endeavor: **UNREACHABLE**, 0 of 220. Flail 1, Reversal 8, Substitute 75, Belly Drum 2 — all reachable. | no |
| **G28** | **`branch_events` panics `Invalid rest_turns value: 32`** — a `PanicException` out of Rust that kills the calling process rather than recording a divergence. `reports/c49_search_crate_rest_turns_panic.json`. | E | **UNREACHABLE from the harness, and settled statically rather than by sampling.** ⚠ An earlier draft rested this on "the current harness, which clamps at 3", which `engine_world.py::_rest_turns_from_row` contradicts verbatim: *"NOTHING CLAMPS IT — not here (the range check below is on the INPUT k, never on the returned counter) and not in the adapter (which validates only non-negativity)."* What actually bounds it is the earlier gate `refunded + skipped > attempts → None`. Since `rest_turns = 3 − k·(1 or 2) + refunded + skipped` and that gate forces `refunded + skipped <= k`, the value is `<= 3` on **both** the `fold_skipped` and non-fold paths, unconditionally. The source records the same conclusion from the other direction: an exhaustive sweep over `(k, refunded, skipped) × Early Bird` found **zero** inputs where the `1 <= rest_turns <= 3` backstop is the line doing the rejecting. This is a **stronger** settlement than the sweep the earlier draft proposed, since a sweep can only produce positive evidence. The panic remains a real engine robustness defect reachable from a hand-built or externally-constructed state — G50's item 4 is the same family of hazard. | no (`engine_error` is 0 in both windows) |
| **G29** | **Trace copies the ability field only; no Start event fires**, so a Traced Intimidate never activates and a Traced Flash Fire's volatile can be wrong. `engine_world.py` `_build_pokemon_spec`; matches `third_party/poke-engine-gen3-trace-no-activation.patch`. | E | REACHABLE. Trace on Gardevoir and Porygon2 (2 of 220); Intimidate on 11 species, Flash Fire on 4. | no |
| **G49** | **Trick can move White Herb onto a non-Deoxys, and Choice Band onto Deoxys — and G18's White Herb gap then travels with it.** ⚠ **This row exists because R26 was wrong in the first version of this document.** | E | **REACHABLE (narrow), by a cross-side check.** Trick is `target: normal` — the *opponent* uses it, so White Herb's holders never needed `trick` in their own movepool, which is exactly what the original R26 checked. `trick` is on Furret and Kecleon; `getItem` gives every Trick user Choice Band (`if (moves.has('trick')) return 'Choice Band';`, measured 944/944). Rate, measured across three independent seed schemes: P(team carries a Trick user) ≈ **2.4 %**, P(team carries White Herb) ≈ **3.0 %**, so P(a battle contains the cross-side pairing) ≈ **0.14 %**, about 1 battle in 700. **The marginals move in the second decimal across schemes (2.15–2.38 %) and the joint spans 0.123–0.137 %**, so the order of magnitude and the ~1-in-700 conclusion are robust and the third decimal is not — an earlier draft claimed third-decimal stability and overstated it. A sampling caveat that matters for anyone re-deriving this: the two teams must come from **disjoint** seed draws. Deriving them from a shared seed by bit-twiddling produces correlated streams and a materially different answer. That is well above the bar at which this ledger grants REACHABLE elsewhere — G25 is granted on Wrap-on-Shuckle alone. | no |
| **G50** | **Transform: max PP is not modelled, and an externally-constructed TRANSFORMED active cannot revert.** `third_party/poke-engine-gen3-transform.patch` approximations 3 and 4, verbatim: *"MAX PP IS NOT MODELLED because the engine has no max-PP field at all"*, and *"A state built from OUTSIDE the engine that carries the TRANSFORMED volatile without a snapshot (e.g. an engine-world constructor expressing an already-transformed Ditto) cannot be reverted, because its base form was never observed. That case degrades to 'drop the volatile, keep the copied form'."* | E | REACHABLE. `transform` on Ditto and Mew (2 of 220). Item 4 names *exactly* the shape `engine_world` produces — its `transform_unexpressible` refusals exist precisely because the constructor is the outside builder in question. | no in the c136 windows — `skip:world_unsupported:transform_unexpressible` is 0 in both. ⚠ **But not never-fired** (C146): **23** in `reports/c32_fail_diagnosis.json` (`coverage_diagnosis.coverage_reducing_skips.*`), `decomposition.ranked[8]` in `reports/c43_coverage_shortfall_diagnosis.json`, and **208** in `docs/audit_artifacts/k0-depth-grid-20260729/results/k0g-{a,c}-d1-1.json` (`side 'p1' copied 'Deoxys', absent from the sampled opposing party`) — the refusal is *observed*, which corroborates this row rather than weakening it. See H13. ✅ **RE-MEASURED WIDE (C153): 6 firings over 10,000 games on unregistered seeds `1,001,000`–`1,010,999`** (`reports/artifacts/c153_*_sweep.json`), so the refusal is live on the SHIPPING build at head, not only in the c32/c43/k0 eras. The "0 in both windows" figure is unchanged and is a statement about those windows. |
| **G51** | **Toxic's min-1 clamp sits inside the multiply, so a 1-HP defender takes `1 × stage` instead of 1.** `gen3/generate_instructions.rs`: `let per_stage = cmp::max(active_pkmn.maxhp / 16, 1); … cmp::min(per_stage as i32 * stage as i32, hp as i32)`. `third_party/poke-engine-gen3-residual-rounding.patch` names it and names the carrier: *"The min-1 clamp sits INSIDE the multiply, which is a second, opposite divergence at the bottom of the HP range: a 1 HP Shedinja (in the gen3 randbats pool) takes `1 * stage`, not 1."* | E | **REACHABLE** — the patch names a pool member, which is this ledger's own bar for a row. Shedinja is in the pool and is the **only** species with `maxhp <= 47` (measured: minimum maxhp across 60,000 generated Pokémon is 1, and Shedinja is the sole holder). Narrow in practice: Shedinja dies to the first stage regardless, so the divergence is in the *magnitude* of a lethal tick, not in whether it is lethal. | no |
| **G30** | **Pivot turns skip the turn's residuals entirely** (documented deviation). `rust/pokezero-search/src/events.rs` `finish_ply`. | E | REACHABLE via Baton Pass, **25 of 220** species — the pool's only pivot (U-turn is gen4). | no |

### 3.2 Renderer gaps — REACHABLE

| # | Gap | Class | Reachability evidence | Observed |
|---|---|---|---|---|
| **G31** | **Haze renders `-unboost` lines; Showdown emits `\|-clearallboost\|`.** `events.rs`, the five-producer note above `boost_may_be_a_switch_out_reset`: "No `clearallboost` exists anywhere in this crate, so the NAMED path is wrong for Haze too." Verified: `clearallboost` appears in the crate exactly twice, both inside that comment. The engine's gen3 `choice_effects.rs` `Choices::HAZE` emits two `state.reset_boosts` calls, i.e. `Boost` instructions. The gap is **one-sided**: the Python parser handles `-clearallboost` (`src/pokezero/showdown.py`, `tier2.py`, `silent_mutation_audit.py`), so the harness understands the line Showdown emits — only the crate cannot produce it. | R | **REACHABLE for direct Haze** — 4 of 220 species (Altaria, Crobat, Mantine, Weezing). The *Sleep-Talk-callee* half of the same note is **UNREACHABLE**: measured on this `sets.json`, **0 of 393 sets** pair `sleeptalk` with `haze` (also 0 with `psychup`, `roar`, `whirlwind`, `batonpass`) — independently reproducing the crate's "350 Sleep Talk sets, zero pair it" claim. | no |
| **G32** | **Psych Up: same shape** (`\|-copyboost\|`). | R | **UNREACHABLE.** `psychup` is 0 of 220. Both the direct path and the Sleep-Talk path are dead. | no |
| **G33** | **Known-open: the drain slot is booked from pre-residual HP.** `events.rs`, `#[ignore = "known open: drain slot booked from pre-residual HP; see doc comment"]` on `a_near_full_hp_seeder_still_over_books_the_drain_slot`. The comment itself says: "Zero occurrences in seeds 19000000-19000199, reachable in ordinary gen3 stall play." | R | REACHABLE (Leech Seed 12 species + Leftovers). | no |
| **G33b** | **The Leftovers heal slot is over-booked when the residual phase is truncated by battle end**, and the constant fallback then relabels the bare Leech Seed drain as `item: Leftovers`. Same `ResidualPlan` over-booking family as G33, different slot and a trigger no HP predicate can see. Mechanism, all from source: gen 3 inherits gen 4, so `data/mods/gen4/items.ts:231` puts `leftovers` at `onResidualOrder 10 / subOrder 4` and `data/mods/gen4/moves.ts:711-716` puts the `leechseed` condition at `10 / subOrder 5` — **one speed-sorted bucket per Pokemon, not two global phases** (the base `data/items.ts` 5 and `data/moves.ts` 8 predict the wrong order), and `sim/battle.ts:565-566` ends that bucket with `this.faintMessages(); if (this.ended) return;`. So a **slower** seeder whose capped drain kills the opponent's **last** Pokemon never reaches its own Leftovers slot; `ResidualPlan::build` books it anyway (deliberately — see the `NOTE:` at `events.rs:5123-5133`, where an `hp < maxhp` guard was measured to cost 5 rows), the count mismatch sets `plan.usable[side] = false`, and every heal on that side falls to `residual_heal_cause`, which since C131 change 3 tests Leftovers first. **The HP arithmetic is correct; only the attribution is wrong** — C131's finding one surface over. Reproduced on three GENERATED gen3 Custom Game boundaries with no Fire Blast, Flamethrower, burn or paralysis, single-variable: variants A and B differ only in whether a spare Pokemon sits behind the victim, and the engine renders the drain `[from] item: Leftovers` in A and, in B, emits **both** heal lines exactly as Showdown does (the rest of B's protocol still differs in two respects the matcher does not compare: the drain damage line omits `[of]`, and `\|faint\|` precedes the mirror heal). `reports/c143_heal_attribution_diagnosis.md` §1–2; artifact `reports/artifacts/c143_heal_attribution_probe.json`. **Not cosmetic:** with it in place the boundary matches **0 of 14** rolls at each of **fifteen** representatives — every member of the residual-lethal band plus the off-fan shipping value, 210 cells — so it is independently sufficient to keep a row divergent under the collapsed path. ⚠ **A gate closes nothing under the collapsed path and CLOSES THIS ROW UNDER ENUMERATION — measured; this cell said "unmeasured" in its first revision and that was the wrong path.** With `POKEZERO_ENUMERATE_ROLLS=1` the shipped renderer gives 416 branches / diverged / 12 misses, one of them `observed_only=[('heal', 36)] engine_only=[('itemleftovers', 36)]` at 0.2189 % — right magnitude, **wrong label only**; exactly 1 arm of the 416 reproduces the full observed HP trace. A modelled gate (drop side one's Leftovers-tagged heal to `[silent]` in arms truncated by the opposing active's faint) gives **matched, 0 misses**. Soundness control, since a relabelling can only widen what matches and "nothing opened" therefore proves nothing about it: 350 heals relabelled, every delta in [27, 47] — the exact range of the 14 achievable mirrors — and **none equal to 16** (`268//16`, a genuine Moltres tick), so no real tick was silenced. Since C137 made enumeration the oracle, that closure lands on the certifying path. **Still unshipped**: the measurement models the fix at the renderer's output rather than building it in `ResidualPlan`, and C133 §7's discipline (built gate, registered prediction with a "nothing opened" falsifier, dev **and** validation-holdout swept, never the final holdout) has not been run. Recommendation: **worth building and sweeping.** ⚠ **BUILT, SWEPT AND SHIPPED 2026-08-07 — `reports/c147_g33b_residual_bucket_gate.md`.** The gate is now a real change in `ResidualPlan::build` (`leftovers_slot_truncated`), not a model applied to output: it walks the residual segment for the instruction that ends the battle and asks whether the winner's 10.4 slot was behind that point. Five arms, taken from the engine's own section order, of which two gate — the shared weather entry at order 8, and the loser's own order-10 bucket when the loser is strictly faster; Liquid Ooze recoil (10.5, the winner's own bucket), Future Sight (11) and Perish Song (12) all land after the winner's 10.4 and are excluded. **Closure, built rather than modelled:** on `19200244/115` replayed from the committed C141 artifact under the enumeration oracle, the shipped renderer goes `diverged` / 12 misses → **`matched` / 0 misses** over 416 branches, and the c143 model's relabel count falls **350 → 0**, so the built gate covers exactly the set the model covered. **Both permitted windows swept on both builds**, prediction registered first at `e0b8ca0f`: every verdict scalar unchanged and the whole `counters` block byte-identical to the same-build base on dev and on holdout. **Zero rows closed in-window and that was the registered expectation** — neither window contains a G33b row, so the sweeps are a safety measurement. **Reach measured separately, because a verdict-level sweep cannot see this gate fire** (at a firing site the baseline plan is already unusable, so any count-preserving mutation reproduces the baseline): **52 slot skips on dev and 56 on holdout**, 108 firings across 400 games, no verdict moved either way — re-derived at the final head after both merges and unchanged. Two arms of the family stay **OPEN**: a fatal weather chip when the winner is faster, and exact speed ties (the engine forks both orders, so one is mislabelled either way). Pinned by seven crate tests, one of them verified red on a separately-vendored `origin/main` worktree; the other six are green on `main` too and each is the sole killer of its own mutant. ⚠ **ONE OF THE TWO OPEN ARMS RETIRED, THE OTHER MEASURED AND STILL OPEN, 2026-08-08 (C152) — `reports/c152_ledger_terminal_disposition.md` §3, artifact `reports/artifacts/c152_g33b_open_arm_census.json`, parser `scripts/c152_g33b_open_arm_census.py`.** Retired, not closed: nothing was built. **The weather arm was measured, and it FIRES — C147's "unmeasured, not believed absent" was right about the belief and wrong to leave it unmeasured.** `leftovers_slot_truncated` was instrumented on a throwaway build to emit one line per battle-ending residual instruction it reaches, BEFORE any arm returns, so the census sees the whole family rather than the part the gate acts on. Whether the fatal instruction is the order-8 chip is decided by a STATE predicate, never by classifying the instruction (which the function's own comment correctly says is impossible, since a lethal residual always equals remaining HP): `weather_chips(state, loser)` is `Some` AND `loser_pre_hp <= max(1, loser_maxhp / 16)`, sound because order 8 is the first damaging residual phase so the loser's HP at its chip is still pre-residual. Corroborated rather than assumed — the terminating amount equals `min(chip, loser_pre_hp)` in **42 of 42** weather-fatal calls, the engine capping a lethal residual at remaining HP being why the naive `amount == chip` matches only 17. **Cross-check on the instrument itself:** its per-window `loser_first`-with-a-Leftovers-winner count is **52 dev and 56 holdout**, reproducing C147's independently-measured reach of 52 and 56 exactly. **The census, over 1,400 games:** 925 predicate calls at battle-ending residual instructions — 911 `order_le_10` and **14 `perish`**, the order-12 arm the gate correctly excludes — split **494 loser_first / 407 winner_first / 24 exact ties**. **42** calls have the loser dying to its own order-8 chip; **17 of those are winner-first, i.e. the un-gated half, and all 17 have a Leftovers winner**. So the arm is REACHED. **The tie arm is reached too — 24 calls, all with a Leftovers winner** — which is the first measurement of it; C147 recorded neither arm's incidence. **But the arm cannot mislabel a heal, and that is a structural demonstration rather than a count.** Read off the vendored engine's `add_end_of_turn_instructions` in emission order — class-0 side conditions, then **Wish**, then the weather block, then the speed-major order-10 buckets, with the engine's own comment stating Wish *"is in a class ahead of the weather chip at 8 no matter who is faster"* — the ONLY positive `Heal` the winner can emit inside a weather-truncated segment is a resolving Wish. And `residual_heal_cause` labels a resolving Wish **correctly without the plan**: it tests `matches!(next_ins, Some(Instruction::DecrementWish(d)) if d.side_ref == side)` first and returns `move: Wish` before reaching the Leftovers branch, on an adjacency the engine guarantees by emitting `Heal` then `DecrementWish`. So the over-booking sets `plan.usable[winner] = false` on a side whose only possible heal is fallback-correct — and in every one of the 17 weather-arm instances the winner emitted **zero** heals, so even that was not exercised. ⚠ **THE SAME DEMONSTRATION DOES NOT COVER THE TIE ARM, and an earlier draft of this cell said it did.** The weather arm is safe because order 8 precedes every order-10 heal slot, so the winner's Leech Seed **drain** heal — emitted at the LOSER's 10.5 — cannot have fired. A tie truncates INSIDE order 10, and in the fork where the loser's bucket runs first the winner's drain heal IS emitted while its own 10.4 is not. `plan.heal[winner]` then books Wish/Leftovers/drain against fewer emitted heals, the side goes unusable, and `residual_heal_cause` returns **`item: Leftovers`** for that drain — it tests the holder's item BEFORE its silent-drain empty-string branch, deliberately and for a recorded reason. That is exactly the mislabel this row is about. ⚠ **The exposed shape is 3 of 20, not 7 of 24 — corrected in review.** 7 of the 24 tie calls do carry a winner-side heal before the truncation, but **4 of those 7 are `perish`-arm calls**, and the predicate returns `NO_TRUNCATION` for the perish reason because Perish Song is order 12, AFTER all of order 10 — nothing in order 10 is skipped there, so those 4 cannot exhibit the mislabel at all, and their carrying an earlier heal is the expected consequence of the winner's bucket having already run. The exposable population is the `order_le_10` ties: **20 calls, of which 3 carry a winner-side heal**. Re-derived from the census artifact's `by_arm_and_order` (`order_le_10` ties 20, `perish` ties 4) and its stored tie rows. The shape is still not hypothetical; the number was asserted over a population wider than the mechanism can reach. Where the winner emitted a **damage** line instead (its own order-8 chip), the unusable plan sends it to `residual_damage_cause`, which returns `Sandstorm` from `state.weather.weather_type` — **identical to the planned label**. ⚠ **And the one shape where those labels WOULD differ is a shape this gate cannot fix**, which is why retiring the arm is not the same as closing the family: a winner that is also burned books `["Sandstorm", "brn"]` against one emitted instruction, so the DAMAGE count mismatches and the side is unusable whatever the heal slot does. That belongs to G33's damage-slot family, and it is named here rather than absorbed. **The tie arm's refusal was re-verified, not carried:** `residual_speed_order` returns `None` on an exact tie, and `add_end_of_turn_branches` builds `residual_orders` as both sides on `None`, keeping both candidates unless `same_residual_outcome` finds the instruction multisets equal — which a truncation makes unequal. So both orders really are live. ⚠ **One correction to the STATED reason, which does not move the disposition:** the doc comment says *"there is no single answer to give."* True of `residual_speed_order`, which sees only the state — but `leftovers_slot_truncated` is also handed the **segment**, and the two forks have different segments, so which order a branch took is recoverable from the order of the first instruction belonging to each side. The refusal is a **choice**, not an impossibility. **Disposition, split because the two arms are not alike.** The **weather arm is RETIRED**: reached 17 times in 1,400 games, and structurally unable to mislabel a heal, with the count and the mechanism both above. Its gate is buildable from the state predicate alone and is filed unbuilt, measured benefit zero. **The speed-tie arm STAYS OPEN**, and this is the first time it has been measured at all: 24 calls, all with a Leftovers winner, of which **20 are `order_le_10`** — the only ones a truncation can expose — and **3 of those 20** carry a winner-side heal that an over-booked plan sends to a fallback which answers `item: Leftovers`. No tie-arm divergence was OBSERVED in 1,400 games, but "not observed" is not "cannot happen", and this document's own standing rule is that those are different claims. Its fix is the segment-order inference of the correction above — the shipped comparator already has the segment, so the order the branch actually took is recoverable. ⚠ **And C152 found a THIRD arm, filed as G33c below**, which is why this row is not terminal. **Stated scope:** 1,400 games — the two permitted windows plus 1,000 games on unregistered seeds `1,000,000`–`1,000,999`, which are below `FIDELITY_SEED_FLOOR`, in no registered band and in no acceptance namespace. Nothing here is a claim about the reserved windows. | R | REACHABLE (Leech Seed 12 species + Leftovers, 72 % of items) — and observed. | **yes** — `19200244/115`, superimposed on G8 |
| **G33c** | ⚠ **NEW 2026-08-08 (C152), and it defeats C147's gate on the very boundaries the gate was built for.** **The same battle-end truncation strands the winner's order-10 DAMAGE bookings, and nothing un-books those.** `leftovers_slot_truncated` removes the winner's 10.4 Leftovers **heal** slot when the loser's own order-10 bucket ends the battle first. But the winner's `brn` / `psn` / `partiallytrapped` entries live in the SAME skipped bucket (10.6 and later) and `ResidualPlan::build` books them unconditionally, so `plan.damage[winner].len() != emitted_damage[winner]`, `plan.usable[winner]` goes false anyway, and every heal on that side still drops to `residual_heal_cause` — which tests the holder's item first and answers `item: Leftovers`. The heal gate is therefore INERT wherever the winner also carries a residual status, which is a large share of seeders. **Observed, diagnosed and reproducible:** `1000513/121` in `reports/artifacts/c152_wide_census_1000500_sweep.json`, `component_mismatch:heal` + `itemleftovers` at **pct 100.00**, `observed_only=[('heal', 36)] engine_only=[('itemleftovers', 36)]`. Protocol, both sides side by side: Showdown emits `\|-heal\|p1a: Tropius\|317/341 psn\|[silent]` and the engine emits `\|-heal\|p1a: Tropius\|317/341\|[from] item: Leftovers`. Tropius is the seeder, holds Leftovers, is **poisoned**, and is SLOWER than Kangaskhan (effective speed 151 against 188), so the gate fires (`order=loser_first`, confirmed by the C152 truncation instrumentation emitting exactly one `C152_TRUNC` line for this boundary) and the row diverges regardless. **Distinguish it from G33b:** G33b is the heal slot and is now gated; this is the damage slots of the same bucket and is not. Distinguish it from G33: G33 books the drain slot from pre-residual HP; this books a status tick that a truncation skipped. **Fix, unbuilt:** `leftovers_slot_truncated` already computes the flag; the same flag has to suppress the winner's post-10.4 damage entries as well as its heal entry. That is a `ResidualPlan` change under C133 §7 discipline, and unlike G33b's two open arms it has a **measured** benefit — this row. | R | REACHABLE — **observed**, on a live boundary with the mechanism read off both protocols. Leech Seed 12 species, Leftovers 72 % of generated items, and any residual status on the seeder. | **yes** — `1000513/121`, on unregistered seeds. ⚠ **Zero in both permitted windows**, which is the point: this shape was invisible to every sweep this program has run. |
| **G34** | **Confusion self-damage is not tagged**, so the source sets differ and exact-component comparison rejects. `reports/c81_small_pure_families.json`. | R | **REACHABLE through exactly one producer.** Enumerated over the pool's 125 moves against `Dex.mod('gen3')`: the only move that can inflict confusion is **`signalbeam`** (10 % secondary), on **3 of 220** species — Venomoth, Ariados, Yanma. Every classical gen3 confuser (`confuseray`, `supersonic`, `swagger`, `flatter`, `sweetkiss`, `dynamicpunch`, `psybeam`, `teeterdance`) and every self-confusing move (`outrage`, `petaldance`, `thrash`) is **0 of 220**. That single 10 % secondary on three species is why `hidden_counter_support:confusion` fired once in 400 games. | no |
| **G35** | **`\|-crit\|` is gated on an exact-value equality**, so the cross-check rejects an observed crit against the engine's own crit arm on identity rather than magnitude. `reports/c93_crit_tag_renderer_gap.json`, `"not_yet_implemented": true`; `reports/c77_i2_crit_arm_absence.json` notes 122 of 173 rows remain unexplained. | R | REACHABLE — crits occur on every damaging move at ≥ 1/16. | no |
| **G36** | **HP-*rise* direction is not rendered in the residual walk** ("DECREASES ONLY, deliberately"), leaving C52's impossible component alive in mirror image. `events.rs` `render_move_phase`. | R | REACHABLE (Leftovers, Wish, Leech Seed drain, `synthesis`/`morningsun`/`moonlight`). | no |
| **G37** | **Attract's empty-immobilization branch is indistinguishable from a fully-capped boost or a blocked stat drop** — 17 sub-cases. `events.rs` `volatile_empty_tail_ambiguous`; `reports/c56_excluded_branch_census.json` counts `attract_empty_tail_ambiguous` 123 and `attract_immobilization_source_unknown` 39. **CLOSED (immobilizer markers, this change).** `third_party/poke-engine-gen3-attract-marker.patch` adds `Instruction::MoveImmobilized { side_ref, reason }` (apply and reverse both no-ops, so masses are unchanged) and pushes it into BOTH gen3 move-time immobilizer branches — Attract AND full paralysis. `events.rs` reads the marker and renders `\|cant\|<ident>\|Attract` or `\|cant\|<ident>\|par` exactly; the `attract_empty_tail_ambiguous` refusal and its five sub-case literals are DELETED, as is the older probability-mass `\|cant\|..\|par\|` guess. BOTH markers were required: `reject_attribution_unsafe` aborts the whole WORLD rather than the branch, so marking Attract alone left the paralysis sibling refusing and every such world still fell back. Verified per sub-case in `tests/test_instruction_event_mapping.py` (`test_every_attract_empty_tail_ambiguity_is_resolved`, nine shapes, zero refusals) and over the attracted/paralyzed fan in `no_branch_of_an_attracted_or_paralyzed_fan_refuses_any_more`. The `123`/`39` counts cited above are the era-56 census and are NOT re-measured here; what is measured is that the refusal is unreachable, and the collateral effect on the `#1048` attribution oracle (branches 2614 -> 2720, agree 2377 -> 2483, both +106, from paralysis branches that `combine_duplicate_instructions` had been merging). STILL OPEN, split out rather than folded in: the engine does not track the infatuation SOURCE, so Showdown's companion `\|-activate\|..\|move: Attract\|[of] <source>` line stays unrenderable and the branch keeps the telemetry-only `attract_immobilization_source_unknown` tag — see G37b. | R | **REACHABLE, but only via the ability route.** The move `attract` is **0 of 220**. Cute Charm is on **3 of 220** (Clefable, Delcatty, Wigglytuff) and applies `attract` on contact at 1/3 (`data/mods/gen3/abilities.ts` `cutecharm.onDamagingHit`). So Attract exists in this format solely as a Cute Charm proc. | no |
| **G37b** | **NEW, opened by the immobilizer-marker change.** Searched worlds now emit `\|cant\|<ident>\|Attract`, which `src/pokezero/public_action_capture.py` keys as the public action `cant:attract` — while `src/pokezero/public_replay_materializer.py`'s `_SUPPORTED_CANT_REASONS` (`slp`/`frz`/`par`/`flinch`/`recharge`/`truant`) still REJECTS the identical event coming from a real replay, as `unsupported_public_event:cant:attract`. Pre-existing — the renderer already emitted this line on its uniquely-immobilized path, and a real Cute Charm proc already produces it in replay logs — but this change makes the two sides diverge in VOLUME, because the line is now emitted on every Attract-immobilized branch instead of only the sub-case where no other explanation existed. Deliberately NOT fixed in that change: admitting a new cant reason widens the materializer's public-action vocabulary, which is a corpus-compatibility decision with its own measurement. | R | REACHABLE via Cute Charm, as G37. | no |
| **G38** | **Sleep Talk's unnamed callee: the largest world-level refusal channel.** `events.rs` `unrenderable_family_at`; era 59 measured `sleeptalk_called_unidentified:ambiguous_unrenderable` at 8,149 world failures, 51.6 % of the abort channel. Five of the six allowlist entries have **no fixture** ("admitted on a structural argument"). ⚠ **PARTIAL DISPOSITION — a double `damage_dealt` reset is CLOSED at the engine, and the closure is measured ON THE IN-MEMORY FOLD PATH ONLY.** `generate_instructions_from_move` emitted the turn-start carry-over reset TWICE for every Sleep Talk turn carrying a non-default `damage_dealt`: the Sleep Talk block reverses `incoming_instructions` before recursing, restoring the pre-reset carry-over, so the callee's own call re-entered the same opening and pushed the reset again. `ChangeDamageDealtDamage` is a DELTA and `ToggleDamageDealtHitSubstitute` a TOGGLE, so neither is idempotent under doubling. Because `consume_move_prelude` eats every leading damage-dealt instruction — and walks PAST `SetSleepTurns`, so it eats both copies even though they are not adjacent — while `identify_sleep_talk_called` regenerates ONE, the divergence sits at INDEX 0, which is why the class registers `shape_length` and never `shape_branch_is_prefix_of_tail` or `shape_tail_is_prefix_of_branch`. Guard: `state.use_damage_dealt && !choice.sleep_talk_move` (`third_party/poke-engine-gen3-sleeptalk-damage-dealt-double-reset.patch`). **REACHABILITY BOUNDS THE CLAIM, and this cell's first revision overstated it.** `Side::serialize` (`third_party/poke-engine-src/src/state.rs:1292`) emits 29 fields and `damage_dealt` is not one of them; `Side::deserialize` (`:1394`) hardcodes `damage_dealt: DamageDealt::default()`. The field cannot survive a round trip, so any consumer handing the engine a SERIALIZED state presents a zero carry-over — that is `branch_events` and `env_step`, each opening with `parse_state` → `State::deserialize`, and `env_step` also returns `post_state: state.serialize()`, so the carry cannot survive between calls either. `State::deserialize` DOES call `set_conditional_mechanics`, so the FLAG is set from movesets while the VALUE is zeroed. **The differential has THREE engine entry points, not one — an earlier revision of this cell said `branch_events` was the only one, which was a false totality claim in a cell correcting a false totality claim.** Re-derived: (1) `engine_transition_differential.py:2057` `branch_events(state.to_string(), …)`, which deserializes, so the carry is zeroed by `Side::deserialize`; (2) `:2304` `poke_engine.generate_instructions(state, …)` on the LIVE pyo3-constructed `State` from `build_poke_engine_state` (`src/pokezero/poke_engine_adapter.py:298`) — `State::deserialize` is never involved here and it DOES reach the guarded function via `poke-engine-py/src/lib.rs:1067` → `:1092`, but the carry is zeroed by a different mechanism, `damage_dealt: Default::default()` at `poke-engine-py/src/lib.rs:263` with `use_damage_dealt: false` at `:84` before `set_conditional_mechanics()` at `:86`; (3) `:842`/`:2075` `poke_engine.calculate_damage` → `lib.rs:1099` → `calculate_both_damage_rolls`, which never calls `generate_instructions_from_move` at all. `DamageDealt::default()` is `{0, Physical, false}` (`state.rs:586`) and `reset_damage_dealt` tests against exactly those, so on a zero carry it emits nothing and emitting nothing twice is emitting nothing. **So the sweep corpus cannot observe this fix on ANY of its three entry points.** The one reachable path is the in-memory tree fold (`model.rs` / `tree.rs`, apply/reverse on a live `State`), at **depth ≥ 2**. **MEASURED** on that path by `rust/pokezero-search/examples/gen3_sleeptalk_none_matched_census.rs`, two builds whose vendored trees differ in exactly one line: over 80 cells / **2,025 branches**, `none_matched:shape_length` **2,025 → 0** and `none_matched:shape_structure` **954 → 0**, attributed callee **0 → 1,840**. The probe carries its own reachability control — the SAME population with the carry-over zeroed, which is what a deserialized boundary presents, is **identical between the two builds** (0 refused, 1,840 attributed, same end-state mass digest). Artifacts `reports/artifacts/c148_sleeptalk_double_reset_census_{base,gate}.json`. That control PREDICTED the sweep result: both permitted windows swept on both builds, `counters` **byte-identical** on each, every verdict scalar unchanged, **zero rows opened and zero closed** — `reports/artifacts/c148_sleeptalk_double_reset_{base,gate}_{dev,holdout}_sweep.json`. **End-state neutrality is recorded as a digest, not asserted**: all four arms return the same `(cell, serialized end state) → mass` digest over 1,899 distinct end states — but note that projection EXCLUDES `damage_dealt`, so it says the observable distribution does not move, not that the corruption is harmless. gen3 has exactly THREE readers of the field — Counter (`generate_instructions.rs:1498`), Mirror Coat (`:1507`), Focus Punch (`choice_effects.rs:266`); the `abilities.rs` hits are a local `i16` parameter. Mirror Coat gates on the category, an absolute set and the one idempotent sub-field; Focus Punch needs `damage > 0`, which both `-137` and `0` fail. Counter is the only channel that can carry it. **WITHDRAWN from the first revision of this cell:** the headline `none_matched` **8,613 → 0** over 8,903 branches, the 8,292/321/290 partition, and the 5,184-cell and 4,368-cell decompositions — no artifact ever supported them, `grep -rl none_matched reports/artifacts/*.json` returns **0 files**, and the 8,613 counted AGGREGATE `none_matched` rather than `shape_length`. **No campaign-era rate is claimed** — §3 of `reports/c148_sleeptalk_double_reset.md` is probe-population branch counts, not a fallback rate, and the era-61 `shape_length` figure (4,786 worlds, 33.3 %) is a source comment this change does not re-measure and cannot, per the reachability paragraph above. ⚠ **THE COMMIT TITLE IN `main` OVERSTATES THIS CELL.** #1170 squash-merged as `cf3c03d3` under the title *"the Sleep Talk double damage-dealt reset, which is all of `none_matched:shape_length`"*. That title is permanent in history and **is not what the body it merged establishes**, nor what this cell claims: the closure is measured on the in-memory fold path, `shape_length` on the sweep corpus was never measured at all, and the change closes `shape_structure` on the probe population too, so "all of `shape_length`" is wrong in both directions. A reader following `git log` should take the body and this cell as the claim, not the subject line. `reports/c148_sleeptalk_double_reset.md`. | R | REACHABLE. `sleeptalk` on **40 of 220** species. | **yes as annotation** — `strict:sleeptalk_union_branch` 126 dev / 105 holdout |
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
| **H8** | **The comparator's fallback window `[0.92·eng − 1, 1.09·eng + 1]` carries unmeasured matched mass.** It applies whenever `pre_legal` is unavailable. `reports/c135` §6. **Was: UNKNOWN how much.** `strict:no_damage_rolls` — the counter that fires when `pre_legal` is None at the state level — is **0 in both windows**, which was read as bounding the *state*-level fallback at zero but not the per-branch one. Named settling measurement: count boundaries whose accept came from the window rather than exact fan membership, over a 200-game dev sweep. | H | ⚠ **SETTLED 2026-08-08 (C152) — `reports/c152_ledger_terminal_disposition.md` §4, artifact `reports/artifacts/c152_h8_window_census.json`, script `scripts/c152_h8_window_census.py`.** The named settling measurement was run, on both windows rather than dev alone. **167 dev boundaries (1.077 % of 15,503 measured) and 140 holdout (0.899 % of 15,579) match ONLY because of this window** — measured as the difference in `transitions_matched` between the shipped comparator and a variant with the window accept removed, both arms on engine `bfdbe1c04876edcd` / 74 patches, sweeps `reports/artifacts/c152_h8_nowindow_{dev,holdout}_sweep.json` against `c152_head_{dev,holdout}_sweep.json`. Neither arm edits `scripts/engine_transition_differential.py`, which is under a certification pin: both are AST rewrites of the shipped `roll_components_agree` source obtained by `inspect.getsource`, and the rewrite refuses to run unless it finds exactly one window test, matched structurally rather than by text. ⚠ **THIS CELL'S STATED MECHANISM IS WRONG, and the measurement is what shows it.** It says the window *"applies whenever `pre_legal` is unavailable"*. That door was taken **0 times in 400 games**: the usage tally splits **190 dev / 181 holdout** component-level window accepts as `window_accept_legal_none` **0** and `window_accept_legal_miss` **190 / 181**. `strict:no_damage_rolls` being 0 was read here as bounding only *part* of the fallback; it is in fact the WHOLE of the door this cell names, and that door contributes nothing. What the window actually carries is boundaries where the fan **was** enumerated and the observed magnitude is simply not in it. **Usage is not dependence, and only the difference is quotable:** 190 accepts against 167 boundaries flipped, because a boundary matches if ANY branch matches. **What the window absorbs**, from the disabled arm's classes: dev 158 `roll_scaled_component` + 5 `limit:roll_divergent_lethality` + 4 component-level; holdout 129 + 9 + 2. **This is a second accept bar and §6 now quotes it** alongside support-gated acceptance. Scope: two 200-game windows, collapsed roll path, strict matcher; says nothing about other seed ranges or about the enumeration oracle. | **yes — 167 dev / 140 holdout**, and the counter is 0 |
| **H9** | **Per-slot HP comparison is invalid on any boundary where the active changed.** `reports/c61_empty_engine_arm_census.json`: "37 of 108 rows here are uncomparable … roughly a third of the residue." | H | REACHABLE — **4 of the 6** c136 divergence rows carry `active_changed: true` on one side (`19000074/27`, `19100170/71`, `19100170/72`, `19100191/5`). | **yes**, structurally |
| **H10** | **Repro retention caps at `keep_repro=25` and retains repros only for *divergent* boundaries**, so an adjacent matched boundary needed for a diagnosis is simply not in the artifact. `reports/c120_a1_marker_design.md` §2. | H | REACHABLE by construction. | n/a (both windows are under the cap: 2 and 4 retained) |
| **H11** | **`19100170/71` and `19100170/72` were open divergent rows.** Both `component_missing_in_engine:itemleftovers`, `branch_count: 1`, `pct=100.00`, `p1: protect` against a p2 switch. ⚠ **ADJUDICATED 2026-08-07 — `reports/c145_itemleftovers_row_adjudication.md`.** Class: **a world-construction fix in shipped Python, not a limit and not an engine fix.** Mechanism, measured end to end: `_build_side_spec` resolved the Encore lock — Showdown locks by move **id**, the engine by move **slot index** — against `_active_row_moves`, which is deliberately the **pre-Transform** snapshot (`local_showdown.actor_move_states_from_request_history` skips requests taken while transformed so PP stays honest). For a gen3 randbats Ditto that snapshot is the single move `transform`, so the self-seat rule "exactly one enabled move identifies the lock" was satisfied **spuriously** at index 0; `_apply_transform` then swapped the donor's moveset in underneath the surviving index. Showdown Encored **Protect** (donor slot 3); the world built `last_used_move=move:0` — **Body Slam**. Because an Encore lock is a *forced* choice, that phantom move is the only thing the engine can do, its damage is **lethal** to the switch-in (Spikes had taken Delcatty to 65 at step 71; Typhlosion was at 2 at step 72). The component is labelled `capped_lethal`, but its magnitude equalling remaining HP is **not** corroboration — `engine_transition_differential.py:551` constructs it as `-remaining`, so every lethal capped hit reads that way, the faint arms `end_of_turn_is_deferred`, and the entire residual block is deferred off the boundary. **BOTH** sides' Leftovers ticks are lost, not just p1's — the recorded miss names p1 only because `evaluate_boundary_strict` `break`s out of its `("p1","p2")` slot loop on the first failure, so p2's `itemleftovers +18` at step 71 was never compared. Sizing this class from the miss string undercounts it **by up to half, at boundaries where both sides tick** — 2 ticks against 1 named at step 71, but only 1 tick at step 72 (Typhlosion holds no item), where the miss is complete; **3 lost against 2 named across the class, a third rather than a half.** | H/E | REACHABLE — observed. **Settling measurement (the one specified here): RUN.** Replayed at `dc6e1e19` through `scripts/replay_residue.py`, which re-executes the same `pokezero_search.branch_events` call `evaluate_boundary_strict` makes on the retained candidate states. One branch, `pct=100.00`, `lossy=[]`; instruction stream ends at `ToggleSideTwoForceSwitch` with **no residual phase at all**. Rewriting **one field** of that state — side one's `last_used_move`, `move:0` → `move:3`, same byte-identical engine build — makes the render reproduce Showdown's `\|-heal\|p1a: Ditto\|161/258\|[from] item: Leftovers` and `\|-heal\|p2a: Delcatty\|83/290\|[from] item: Leftovers` **character-for-character**, and the component sets become exactly equal to the observed sets on both slots at both boundaries. Dump: `reports/artifacts/c145_settling_branch_dump.json`. Both rows carry `gating: exact`, so **none of this rides the Constraint-7 hidden-sleep-counter union** (22 of 81 measured boundaries in this game do). | **was yes**; ⚠ **CLOSED by `d27316b6` (#1148)** — **bisected, not inferred.** A one-game sweep (`--games 1 --seed-start 19100170`, strict, rebuilt and `--check`-verified engine at each point) gives 2 divergent rows at `2ec0cb13` (fp `907bea70…`) and at `dc6e1e19` (fp `fdbf5937…`), and **0** at `d27316b6` (fp `fdbf5937…`, unchanged) and on this branch at `662d9db8`. `boundaries_full_round` 88 / `boundaries_measured` 81 and the full skip and gating histograms are identical at all four, so the two boundaries became `matched` rather than skipped. `27609063` is unmeasured and cannot hold the transition: it lies between two measured reds. Artifacts `reports/artifacts/c145_g19100170_{2ec0cb13,dc6e1e19,d27316b6,head}.json`. ⚠ **RETRACTED from the previous revision of this cell:** "nothing in `reports/` still explains the original rows" and "no written cause anywhere in `reports/`". **Both were false when written.** `reports/c139_encore_transform_move_index_prediction.md` § Observation states this mechanism on these two boundaries by seed and step, and #1148 — the very commit this cell guessed at — is what merged it, so the diagnosis was already on `main` at `f876803e`. The negative was asserted over `reports/` from a search that missed a file two commits behind. The closure was also **not incidental**: #1148 registered that prediction, naming these two rows and this class, before measuring. |
| **H12** | **The skip counters do not sum to the coverage shortfall** — `reports/c43_coverage_shortfall_diagnosis.json` measured ~7,224 rows invisible to any skip counter: "no repair list built from them can be complete." **Was: UNKNOWN whether it still holds**, with the named settling measurement *"instrument the single-seat arm with the same exit taxonomy and re-run."* | H | ⚠ **CLOSED — MEASURABLY FALSE IN THE CURRENT ERA, 2026-08-08 (C152), and this cell's own named settling measurement was scoped to the wrong population.** `reports/c152_ledger_terminal_disposition.md` §5. **The mis-scoping first, because it is why this sat UNKNOWN.** c43's own `decomposition.note` reads, verbatim: *"skip:single_seat_boundary (89,887) is excluded -- a single-seat boundary is not a full round and is not in the denominator"*. Every term of the 7,224 is inside the **full-round** path — re-derived from the artifact rather than from this cell: `821,320 − 787,376 = 33,944` unmeasured, ten ranked counters summing to `26,720`, difference `7,224`. So *"instrument the single-seat arm"* would have measured a **disjoint** population and could never have settled the claim. ⚠ **And c43's arithmetic is itself wrong by 372, in the direction that understates it:** its ranked list counts `strict_all_branches_lossy` 372 as accounting for the shortfall, but that counter fires AFTER `boundaries_measured` increments — it is a post-measure verdict, pinned as such in `VERDICT_PARTITION_SKIP_COUNTERS` — so the era's real unaccounted residual is `33,944 − 26,348 = 7,596`, not 7,224. Recorded, not repaired; c43 is an artifact of its era. **The claim itself, re-derived at head** (74 patches, `bfdbe1c04876edcd`, `reports/artifacts/c152_head_{dev,holdout}_sweep.json`), with the exit set taken from `tests/test_single_seat_coverage_bound.py`'s own allowlist: dev `15,503 measured + 465 in-path exits = 15,968 = boundaries_full_round`, holdout `15,579 + 576 = 16,155`, **both exact**; `abort:no_legal_action` **0 in both**, so `full_round + single_seat` is 17,710 and 17,968, the totals §2 reports. There is no population invisible to a counter. ⚠ **What this does NOT close, said explicitly so the move cannot be read as progress it is not:** the single-seat population is **visible** but **uncompared** — 1,742 and 1,813 boundaries, 9.84 % and 10.09 %. That is **H1**, which stays open. C152 also found no committed census breaking the single-seat population into categories anywhere in `reports/` or `docs/`, so §7 keeps that as an open item rather than inheriting H12's closure. | n/a |
| **H13** | ⚠ **CORRECTED 2026-08-07 (C146).** This cell said "**`self_moveset_mismatch`, `transform_unexpressible`, `status_unsupported` and 33 other world-construction refusal reasons are defined and never fire in either window**". **All three named reasons have fired, and the first fired in these very windows.** `skip:world_unsupported:self_moveset_mismatch` is **75 in dev and 24 in holdout** on **27 committed sweep artifacts**, c121 through c133 — and those are not an older seed space: each carries `seeds: {min: 19000000, max: 19000199, distinct: 200}` and `{19100000, 19100199, 200}`, **byte-identical to the c136 pair this document reads**, at 200 games and `matcher: strict`. It is 0 from the c134/c136 generation onward because it was **closed**, by `29ca5697` ("Closes the dominant half of `self_moveset_mismatch`: 365 killed decisions in era 59") — and the closure reconciles: dev `boundaries_measured` 15,432 → 15,503 is +71, against −75 here and +4 `limit:world_substitute_health_unknown`, so the freed skips reappeared as measured boundaries rather than vanishing. A closed exit is not a never-fired one, and reading only "the newest committed post-fix pair" (§1.3) cannot tell the two apart. `transform_unexpressible` is **23** in `reports/c32_fail_diagnosis.json` under the differently-named field `coverage_diagnosis.coverage_reducing_skips.transform_unexpressible`, `decomposition.ranked[8]` in `reports/c43_coverage_shortfall_diagnosis.json`, and **208** in `docs/audit_artifacts/k0-depth-grid-20260729/results/k0g-{a,c}-d1-1.json`; `status_unsupported` is **2** in c32 and `ranked[9]` in c43, plus 9,071 and 3,453 in `docs/engine_divergence_ledger_20260728.md`. That is the **exact** C32/C43 shape H14 was corrected for two rows below, so this cell repeated an error the same document had already recorded. **Two of the four other reasons in §3.5's list of 33 have also fired**: `payload_malformed` 4 and `pending_baton_pass` 2–3 in the c112 leaf-state corpora. **What survives, and it is a real result:** the other **29** of the 33 have no nonzero record anywhere — measured over **347 committed JSON** (`reports/**/*.json` + `docs/**/*.json`, recursive), matching on the name as a path token *and* through the `{"counter": "<name>", "rows": N}` shape, not on a counter key. Now mechanized: `tests/test_never_fired_counter_census.py` derives the 40 reasons from `engine_world.py` by AST and asserts the fired set is **exactly** the 10 named there. | H | REACHABLE-in-principle for some (Transform is 2 of 220: Ditto, Mew), unreachable for others (Future Sight, 0 of 220 — see R1). | **yes for 10 of 40**, four of them in the c136 windows (`volatile_unsupported`, `materialization_blocker`, `encore_move_unknown`, `self_request_state_unsupported`) and six in earlier eras or the `docs/audit_artifacts` grids. **30 of 40 have never fired anywhere** |
| **H14** | ⚠ **CORRECTED 2026-08-07 (C144).** This cell said "**`skip:strict_all_branches_lossy` has never fired**". **That is false, and the refuting artifacts were already committed when it was written:** `reports/c26_structural_probe_report.json` and `reports/c27_structural_probe_report.json` both carry it at **2** (seeds 17000000–17000059, strict matcher), and C141's final-holdout sweep carries it at **4**. On all three the two-term `matched + diverged == boundaries_measured` **fails**. The rest of the cell was right: it increments at `run_game` *after* `_prepare_boundary` has already incremented `boundaries_measured`, so C132's "not an exit" holds for the coverage denominator — but it *is* an exit from the **verdict** tally, which C132 does not say. The identity that actually holds is `boundaries_measured == matched + diverged + engine_error + skip:strict_all_branches_lossy`, and it is now mechanized (`verdict_partition_failures`, gated per shard in `cert_sweep_readout.py`, pinned in `tests/test_boundary_verdict_partition.py`). See `reports/c144_boundary_identity_correction.md`. ⚠ **The arity above is stale as of C146:** the shipped identity is **five**-term, `+ skip:rump_branch_set` (`cert_sweep_readout.py:1451,1611`; the gate's own message names all five). The C144 correction is untouched — a four-term reading was right against a two-term one — but the number in this cell is not the number in the code, which is the kind of drift this row exists to punish. | H | **REACHED, three times over** — not "in-principle". `strict:lossy_render` is the per-branch precursor and reaching 14 of it (C141 holdout) dropped every branch on 4 boundaries. | **no** — its own gap is closed by the mechanized identity; **two** terms of that identity remain unexercised, `engine_error` and `skip:rump_branch_set`, both 0 across all 347 committed JSON under `reports/` and `docs/` (C146). |
| **H15** | **Seven of the eight `unmappable_choice` reasons never fire, and only ~~6~~ → **7** of the 19 `divergence_class` values have *ever* fired — so ~~13~~ → **12** have never fired.** ⚠ An earlier draft said 11 of 19, counting only the c136 pair; the correction to 13 was quoted as "re-derived across **all 31 committed sweep artifacts**". ⚠ **CORRECTED AGAIN 2026-08-07 (C146): that re-derivation was scoped to `reports/artifacts/`, and reported as though it were repo-wide.** Inside `reports/artifacts/` the figure is exactly right — 19 distinct keys, 6 static classes, reproducible with `reports/artifacts/*sweep*.json`. Over the whole of `reports/` + `docs/` it is **35 distinct keys and 7 static classes**: the seventh is **`limit:world_sample_drag_target`**, at **5** in `divergence_classes` / `counters.divergence_class:limit:world_sample_drag_target` in seven c10–c13 differential artifacts, **4** in `reports/c26_structural_probe_report.json`, and **271** observed in `reports/c14_cert_sweep_readout.json` `family_attribution` — and this cell listed it among the classes "the program has simply never produced". None of those files is under `reports/artifacts/`, which is the whole mechanism of the error and is the same defect §4 celebrates catching at Sand Stream. The classes that have ever fired are `component_missing_in_engine`, `component_magnitude`, `component_extra_in_engine`, `component_mismatch`, `roll_scaled_component`, `limit:roll_divergent_lethality` and `limit:world_sample_drag_target`. In c136 specifically only **3** fired: `component_magnitude` and `component_missing_in_engine` in dev, `component_missing_in_engine` and `limit:roll_divergent_lethality` in holdout. Two of the 12 are **structurally unreachable**, both re-verified: `mapper_lossy` (the `skip_lossy` verdict `continue`s before the classification line — `engine_transition_differential.py:2223` returns the trigger string on the skip path, never into `classify_divergence`) and `no_usable_branch` (its trigger `"mapper produced no usable branch"` appears at exactly one site in the repo, the classifier's own test of it, so nothing can produce it; the *identifier* appears in seven files, and one of them, `reports/c9_decomposition.json`'s `"basis"` narration, reads as a hit to any prose-matching search — it is not one). Four more (`boost_delta_support`, `status_support`, `faint_boundary`, `damage_band`) are reachable only through the `--matcher banded` path, which no committed artifact used. The remaining six — `component_set_equal_but_unmatched`, `evidence:faint_ply_no_upkeep`, `evidence:spikes_in_step`, `evidence:crit_in_step`, `no_miss_recorded`, `unclassified` — are strict-path classes the program has never produced, measured over the 347-file corpus. Now mechanized: the 19 comes from `classify_divergence`'s return sites by AST and the fired set is pinned at exactly ~~7~~ → **14** (`tests/test_never_fired_counter_census.py`). ⚠ **CORRECTED A THIRD TIME 2026-08-08 (C153), and this time the SPLIT is what was wrong, not the count.** **Seven of the twelve fire**, all of them on the 2,000-game `--matcher banded` arm of the C153 wide census (`reports/artifacts/c153_banded_census_*_sweep.json`, unregistered seeds `1,009,000`–`1,010,999`, 688 classified divergences): `damage_band` **375** (313 distinct seeds), `unclassified` **163** (144), `status_support` **84** (71), `faint_boundary` **30** (29), `evidence:faint_ply_no_upkeep` **30** (30), `evidence:crit_in_step` **3**, `evidence:spikes_in_step` **2**. Three of those seven are from this cell's own "reachable only through `--matcher banded`" list, and their firing **discharges** that scope caveat rather than refuting it — no committed artifact had ever used the banded path, and now twelve do. **The other four are the error.** This cell files `evidence:crit_in_step`, `evidence:faint_ply_no_upkeep`, `evidence:spikes_in_step` and `unclassified` among *"strict-path classes the program has never produced"*, and they are **fallback-tail** classes, not members of the strict component ladder: `classify_divergence` marks that whole tail *"Banded matcher (**or an unparsable miss**): fall back to protocol evidence"*, so they sit with `status_support` / `faint_boundary` / `damage_band` and were unmeasured for the same reason. ⚠ **Stated as "fallback-tail", not as "not strict-path" — the absolute form is false and a draft of this cell carried it.** The tail IS strict-reachable, on any miss the component regexes cannot parse; what is true is that its practical producer is the banded comparator, so grouping these four with `component_set_equal_but_unmatched` (a genuine strict-ladder class) made them look settled by 200-game strict sweeps that in practice never reach them. The absolute phrasing also discarded a result: all four are **zero across 641,866 strict boundaries and 261 strict divergences**, the first measurement of the fallback tail on the shipping matcher. **Five survive**, and now with a scope: `boost_delta_support`, `component_set_equal_but_unmatched` and `no_miss_recorded` are absent across **803,264 measured boundaries and 949 classified divergences** on unregistered seeds — a 95 % upper bound of **0.32 %** of divergences, which is the honest bound for a class rather than the per-boundary one — and `mapper_lossy` / `no_usable_branch` remain structurally unreachable. ⚠ **`unclassified` at 163 of 688 is 23.7 % of the banded arm's divergences, against `classify_divergence`'s own contract** *"No divergence may land in an unnamed bucket"* — filed as **H22**. `reports/c153_wide_seed_negative_census.md`. | H | mixed. | ⚠ **yes for 7 of the 12, all on the `--matcher banded` arm and none on the shipping strict matcher** — see H22 |
| **H16** | **The dev window is overfit relative to holdout by 3.53×.** `reports/c117_validation_holdout_baseline.md` §1: "Any statement of the form 'the residue is 7' describes *one particular* 200-game window and must not be read as a fidelity rate." | X | REACHABLE by construction. | **yes** — dev 2 divergences vs holdout 4 on the same build |
| **H17** | ⚠ **PARTLY RETRACTED 2026-08-07 (C146).** This cell said "**`reports/c119_phase2_scoping.md`, `reports/c134…` and `reports/c137_phase2_enumerate_decision.md` are cited by merged reports and absent from `reports/`**", Observed "**yes** (verified by `ls`)". One of the three holds. **`reports/c137_phase2_enumerate_decision.md` was already in the tree** — added by `dc6e1e19` (#1147), which `git merge-base --is-ancestor dc6e1e19 f876803e` confirms is an ancestor of `f876803e`, the commit that merged this very document. **That negative was false when written**, by the same mechanism as the H11 retraction two rows above: an `ls` that missed a file already on `main`. **`reports/c134_enumerate_rolls_oracle.md`** was genuinely absent at `f876803e` and arrived one commit later at `6be52191` (#1149), so that half was true when written and is now **stale** — the file exists. Only **`reports/c119_phase2_scoping.md`** is still absent, verified by `ls reports/ \| grep '^c119'` returning nothing. The load-bearing point survives on c119 alone: c135 §5/§7 rests on a C134 freeze whose *report* is now present, and on c137's adopt-for-harness-only decision, which is present. | X | n/a. | **1 of 3** — c119 absent; c134 and c137 present |
| **H18** | **Enumeration closes G6's rows but cannot be used in search**: depth-4/1024-sim throughput regresses 2.38 ms → 8,881.8 ms per decision, and the mass gate's `test_matrix_is_not_vacuous` fails under the flag. `reports/c135` §7. | X | REACHABLE. | n/a |
| **H19** | **Four families were never adjudicated**: `LS_capped_lethal_shape` (the largest unresolved), `I2_matcher_accounting`, `I3_roll_inherited`, `I5_boundary_truncation`. `reports/c86_current_era_family_adjudication.json`. **Was: UNKNOWN whether they survive into the current era**, on the evidence that *"none of their labels appears in the c136 counters"*, with the named settling measurement *"re-run `scripts/family_bucket_audit.py` against the c136 artifacts."* | H | ⚠ **CLOSED — MEASURED, 2026-08-08 (C152).** `reports/c152_ledger_terminal_disposition.md` §6, artifact `reports/artifacts/c152_h19_family_recensus.json`, script `scripts/c152_h19_family_recensus.py`. ⚠ **THE EVIDENCE THIS CELL RESTED ON IS VACUOUS.** No sweep has ever emitted a family label into a counter: `scripts/engine_transition_differential.py` neither imports `cert_sweep_readout` nor calls `classify_row` — re-derived on every run of the C152 script, not asserted. A divergent row today carries a `divergence_class`; the `I*`/`LS_*` names are a SECOND pass applied afterwards by `cert_sweep_readout.classify_row`. Absence from a vocabulary the layer never writes to is not evidence of anything. ⚠ **AND THE NAMED SETTLING MEASUREMENT CRASHED ON EVERY INPUT.** `scripts/family_bucket_audit.py:355` read `(ROOT / evidence).is_file()` with `ROOT` defined nowhere in the module; the line is reached **unconditionally**, because all five `ESTABLISHED` families are in the registered set, so `main()` did all the re-read work and raised `NameError`. `tests/test_family_bucket_audit.py` exercises `signatures()` and `bucket_from_signatures()` and never `main()`, which is how it survived from #1022 (2026-08-02). **Fixed in C152 and pinned** — `TheFamilyBucketAuditCanActuallyRunTests` in `tests/test_never_fired_counter_census.py` resolves every global name the module references and is verified red when the defect is reintroduced. **Measured three ways.** *(a) As recorded, over the widest glob available* — `reports/**/*.json` plus `docs/**/*.json`, **78 artifacts carrying `repros`, 1,167 divergent rows**, each classified by the shipped `classify_row` on its own artifact's recorded `divergence_class` and `branch_misses`. Two C152 artifacts are excluded from that history WITH the reason recorded in the artifact: the window-disabled `c152_h8_nowindow_*` pair is a MUTANT comparator run only for H8, and leaving it in inflated `I2_matcher_accounting` from 85 to 113 — a census must not absorb its own instrument. All four families have fired; their highest-C appearances are `LS_capped_lethal_shape` **181 rows in 55 artifacts**; `I2_matcher_accounting` 85 in 16, last in `c141_final_holdout_sweep.json` (`19200131/129`); `I5_boundary_truncation` 65 in 30, last in the `c137_*_holdout` sweeps (`19100180/24`); `I3_roll_inherited` 23 in 7, last in `reports/c13_batch_e_differential.json`. ⚠ **`LS_capped_lethal_shape` IS NOT EXTINCT, and C152's own wide census is what shows it:** its highest-C row is now `reports/artifacts/c152_wide_census_1000250_sweep.json`, one row on **unregistered seeds** `1,000,250`–`1,000,499`, measured on this same 74-patch engine. Zero in the two permitted windows is not zero everywhere. So **two of the four did survive into the c136 era** — `LS_capped_lethal_shape` at `19000074/27` and `19000191/63`, `I5_boundary_truncation` at `19100180/24` — and two did not. *(b) Re-read at head:* `19000074/27` and `19000191/63` (`LS_capped_lethal_shape`) **matched**; `19200131/129` (`I2_matcher_accounting`) **matched**, 0 misses, 4 branches; `19100180/24` (`I5_boundary_truncation`) **matched**, 0 misses; `19100107/135` and `19100191/5` (`limit:roll_divergent_lethality`) **matched**. Of the rows re-read, only `19100170/71`, `/72` and `19200244/115` still diverge — and `19200244/115` now classifies **`I3_roll_inherited`**, which is the family G8's cell already named as *"the still-unadjudicated remainder of the shape"*; that is now measured rather than asserted. ⚠ **A re-read is not a re-sweep**, and `19100170/71`+`/72` show why: `reread_row` replays the state RECORDED in the artifact, and their fix (`d27316b6`, #1148) changed world construction rather than the engine, so the recorded state still encodes the pre-fix world and still diverges while a fresh sweep of the seed produces no row. Both are `unattributed_generic`, neither is a family row. *(c) Live:* `family_bucket_audit.py --rows` over the 13 committed c136 divergent rows — this cell's own measurement, now that it runs — returns **0 rows for every one of the 21 registered families**; and the head sweeps measure **0 divergent rows in both windows**, so every family is 0 by construction. **Scope:** "current era" is the two 200-game permitted windows at fingerprint `bfdbe1c04876edcd`. It is NOT a claim about the seed space at large — C152 measured 1,000 games on unregistered seeds `1,000,000`–`1,000,999` and found divergences there, recorded in §7. | no |
| **H21** | **`--approximate-sleep`'s help string describes behaviour the tool has not had since hidden-counter support landed**, and it is the most likely origin of this document's own G24 error. Verbatim: *"(default: strict — a publicly-asleep mon with an unknown counter is a counted SKIP, never a guessed world)"*. With `hidden_counter_support` on — which **is** the default (`hidden_counter_support=not args.no_hidden_counter_support`) — such a mon is neither a counted skip nor a guessed world: it is an **enumerated widening** over up to 64 counter assignments, accepted if any matches, and tallied under `gating:support` rather than any `skip:` counter. The string is not merely incomplete; each of its two claims is now false. | X | n/a — a documentation defect in shipped code, on the flag governing the single largest caveat in §6. | n/a |
| **H22** | ⚠ **NEW 2026-08-08 (C153).** **`classify_divergence` leaves 23.7 % of banded-matcher divergences in the bucket its own docstring forbids.** The function opens *"Name every divergence. No divergence may land in an unnamed bucket"*, and records that an earlier revision *"left ~28 % of strict divergences `unclassified`"* — a defect it treats as fixed. It is fixed **on the strict path only**. Measured on 2,000 games of `--matcher banded` on unregistered seeds `1,009,000`–`1,010,999` (`reports/artifacts/c153_banded_census_*_sweep.json`): **163 of 688** classified divergences return `unclassified`, plus 35 more in the `evidence:*` protocol-evidence buckets, which name what the step CONTAINED rather than what went wrong — the function's own comment says so. **Mechanism, from source:** the component regexes `_MISS_COMPONENTS_RE` / `_MISS_SOURCE_RE` parse the miss text that `evaluate_boundary_strict` produces. `evaluate_boundary` — the banded path — produces `_transition_mismatch` text instead, which those regexes do not match, so control falls through the whole component ladder to the protocol-evidence tail and then to `unclassified`. **Why it matters and why it is H rather than E:** no engine behaviour is implicated. The banded comparator is *"kept for continuity with the pre-hardening numbers"*, and any such continuity comparison is now known to be **23.7 % unattributable** on its own terms — a divergence count nobody can decompose is the shape §6 of this document forbids for the strict path and had never checked for the legacy one. **Not observed before because nothing had ever run it:** H15 records that *no committed artifact used the banded path*, and that was true until C153's four shards. **Fix, unbuilt:** either give the banded path a component decomposition, or state in `--matcher`'s help that `banded` divergences are not attributable and gate `unclassified` to zero on the strict path only. `reports/c153_wide_seed_negative_census.md`. | H | REACHABLE — **observed**, 163 times over 144 distinct seeds, on the legacy comparator only. Pool reachability is not the operative filter here: the trigger is a MATCHER selection (`--matcher banded`), not a gen3 mechanic, so §1.2's instruments do not apply and the reachability question is answered by the flag's existence plus a measurement of it. Zero on the shipping strict matcher over 8,000 games and 641,866 boundaries. | ⚠ **yes on the banded arm, no in either permitted window** — no committed artifact had ever used this path |
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
| **G48** | **Opponent request order: the in-crate fallback is ~91 % wrong** beyond the opponent's first switch-in, and it corrupts action indices. `leaf.rs::root_opponent_order`; `scripts/foulplay_paired_eval.py`: "fail-closed in Python is fail-OPEN in the crate." ⚠ **CLOSED 2026-08-08** by #1194: the fallback is DELETED, not corrected. `root_opponent_order` returns `Option<&[String]>` — the caller's `ctx["opponent_request_order"]` or unknown — and an unknown order refuses the whole opponent action map, so the node keeps uniform priors and the refusal is counted in the existing `prior_fallbacks`. Both halves of the description above are now false: there is no in-crate fallback to be wrong, and the reachability claim describes the pre-fix state. **What closed is the fail-OPEN conversion, not the ~91 % measurement** — that figure is `scripts/measure_opponent_request_order.py::wrong_one_swap`'s own docstring, its harness is deliberately not in-repo (that file drives no games and reports no numbers by design), and no per-control per-row artifact exists; the file header records the three controls together at 81-96 %. The fix never depended on the number, only on the asymmetry: a refusal is counted and visible, a substituted order is neither. Verified on the committed corpus — with the order withheld, **0 of 25** opponent switch options resolve, against 25 of 25 before. The gap description above is the pre-fix state and is kept as the record of it. | H | REACHABLE — any game where the opponent switches twice. | n/a |

### 3.5 Named unobserved coverage — every exit the code can emit and neither window did

Every one of these is a gap in coverage *by definition*: the code has an exit for it and
**neither c136 window saw it fire**. Listed rather than summarised, because "we have no rows
for X" is only meaningful if X is named.

⚠ **CORRECTED 2026-08-07 (C146).** This paragraph said "the program has **never** seen it
fire", which is a strictly stronger claim than the section heading's "neither window did", and
the two were being read interchangeably. **Six names below have fired**, all of them
`skip:world_unsupported` reasons: `self_moveset_mismatch` (75 dev / 24 holdout on the same two
seed windows, c121–c133 — see H13), `transform_unexpressible` (23), `status_unsupported` (2),
`substitute_health_unknown` (12–14), `payload_malformed` (4) and `pending_baton_pass` (2–3).
Everything else in this section survives, and now survives a **stated and wider** glob than any
of it was first measured against: **347 committed JSON**, being every `.json` under `reports/`
*and* `docs/`, recursively. Two distinctions this section had been eliding, and which the
corrections below now mark in place:

- **"Zero in the two c136 windows" is not "never fired."** A closed exit reads as zero. The
  window figure is what §1.3's instrument can answer; the never-fired figure needs the archive.
- **A counter can be recorded under a name that is not its key.** `reports/c32_fail_diagnosis.json`
  files it under `coverage_diagnosis.coverage_reducing_skips.<reason>` and
  `reports/c43_coverage_shortfall_diagnosis.json` under `decomposition.ranked[i].counter` with
  the count in a sibling `rows` field. An audit keyed on `counters.skip:…` misses both — which
  is how H14 went wrong, and then H13.

Mechanized, because prose has now failed here four times:
**`tests/test_never_fired_counter_census.py`** derives the 40 refusal reasons from
`src/pokezero/engine_world.py` and the 19 divergence classes from `classify_divergence` by AST,
scans all 347 artifacts under both matching shapes, and asserts the fired/never-fired partition
as **exact set equality in both directions** — so a counter that starts firing is red, and so is
a scanner that stops finding one. Gated as its own step in
`.github/workflows/engine-fidelity-gates.yml`. See `reports/c146_negative_claim_audit.md`.

✅ **RE-MEASURED OUTSIDE THE WINDOWS 2026-08-08 (C153), and this is the instrument §8's newest
rule was missing.** Every list in this section was a *corpus* result: `test_never_fired_counter_census.py`
scans committed artifacts, and §8's own rule says a corpus scan cannot find the class of error it
is about. C153 is the other half — **10,000 games on unregistered seeds `1,001,000`–`1,010,999`**
(8,000 on the shipping `strict` matcher, 2,000 on `banded`), **803,264 measured boundaries** on
engine `bfdbe1c04876edcd` at harness `e0617d12`, against **31,082** in the two permitted windows:
a **25.8×** wider measurement, in a place the program has never tuned against.
`reports/c153_wide_seed_negative_census.md`; artifact
`reports/artifacts/c153_wide_negative_census.json`; twelve shards
`reports/artifacts/c153_{wide,banded}_census_*_sweep.json`; pinned by
`tests/test_wide_seed_negative_census.py`.

**The result for this section: all four lists survive, and now say at what scope.**
**Not one** of the 8 static counters, 6 dynamic families, 7 `unmappable_choice` reasons or 29
`world_unsupported` reasons fired. Every "still absent" sentence below is therefore a claim about
**803,264 boundaries outside both windows**, not about 31,082 inside them. Rule of three: zero in
8,000 strict games bounds a per-game rate at **3.75 × 10⁻⁴** (95 %) and a per-boundary rate at
**4.67 × 10⁻⁶** — so §1.4's own named blind spot, *"a shape with a 1-in-50,000 boundary incidence
… would show zero rows"*, is now **excluded** (16.1 expected hits, P(0) ≈ 10⁻⁷). Calibration
rather than analogy, and anchored on this census's OWN rates rather than C152's: the two names
C152 refuted fire here at **146** and **14** over 8,000 strict games, so a negative at either rate
would show zero with probability ≈ 0 and ≈ 8 × 10⁻⁷. **This instrument demonstrably detects a
per-boundary counter in that incidence class.** ⚠ A draft of this sentence used C152's 3 and 27
per 1,000 games, "expected ~24 and ~216", and called the shortfall seed-block variation;
P(X ≤ 146 | λ = 216) = **2.7 × 10⁻⁷**, −4.76σ, so it is not, and the cause is #1199 rewriting the
`local_showdown.py` fold both counters live in — i.e. the very pooling §7.3 of the C153 report
forbids. **And the in-family calibrators are better and were already here**: the census's four
anti-vacuity controls sit on the entries' **own emission statements** at 10²–10³ counts on this
build — `struggle_not_submittable` 7,410, `volatile_unsupported` 4,827, `world_prestate_mismatch`
2,624, `materialization_blocker` 327 — so the `UnmappableChoice`, `EngineWorldUnsupported` and
prestate paths are demonstrably live here and a zero on them is a measurement, not an unreached
exit. C152's two are cross-family (strict branch-legality/rump, different engine and harness) and
do only their own path's work. What a calibrator establishes is **emission-path liveness**, not
sample size — the rule of three does that and needs no calibrator.
⚠ **A draft of this sentence said "the 6 per-game abort/error counters have no in-family liveness
witness", and BOTH halves were untraced and wrong.** Resolved by AST over the innermost enclosing
scope: **only one** of the 46 is per-game (`abort:no_legal_action`, and not by loop depth — the
next statement returns out of `run_game`); the three `engine_error*` keys are per-boundary and
`strict:no_damage_rolls` / `strict:branch_events_error:` are per-state within a boundary, which
the differential's own comment at `:3134-3136` states verbatim. So **45 of the 46 carry the
per-boundary bound**, not six the per-game one — a factor of ~80, conservative in direction and
wrong in kind. And **38 of the 46 do have a witness**; the eight that do not span **six**
independent paths, of which the `engine_error` handler is one carrying three keys. Both figures
are now derived and pinned (`emission_granularity` and `liveness_witnesses` in
`reports/artifacts/c153_wide_negative_census.json`,
`tests/test_wide_seed_negative_census.py::TheEmissionGranularitySplitIsDerivedTests`).
The whole calibration still licenses **nothing** about a per-divergence class. What the census does not
exclude is anything below those bounds, anything behind a non-default flag, and the six entries §6
of the C153 report names as unreachable by this instrument.

⚠ **The row-level list in H15 did NOT survive — seven of its twelve fired.** See H15.

**Never-fired static counters (8, was 9):** `abort:no_legal_action`, `skip:no_action_candidates`,
`skip:world_error:no_constructible_candidate`,
`strict:no_damage_rolls` (H8),
`engine_error`, `world_prestate_mismatch:side_conditions`, and the two structurally
unreachable `divergence_class` values `mapper_lossy` and `no_usable_branch`.

⚠ **CORRECTED 2026-08-08 (C152) — the SIXTH and SEVENTH false "never fired" in this document.**
Two names left these lists because they **fire**:
`strict:branch_event_legal_error:BranchLegalRollError` (was in the nine above, at **18, 8 and 1**)
and `skip:rump_branch_set` (H14's second unexercised verdict term, at **2 and 1**), both in
`reports/artifacts/c152_wide_census_*_sweep.json`. So the nine become **eight**, and H14's closing
sentence — *"two terms of that identity remain unexercised, `engine_error` and
`skip:rump_branch_set`"* — is now **one**: `engine_error`.

✅ **BOTH RE-MEASURED ON A BUILD THAT CAN BE REBUILT, 2026-08-08 (C153).** C152 found them on the
**throwaway instrumented** engine `89797289…`, which
`tests/test_harness_digest_provenance.py` records as *"NOT reproducible from any committed tree, by
design"* — so until C153 the entire evidence that either counter fires **at all** sat on an engine
nobody could rebuild. On the shipping `bfdbe1c04876edcd`, over 8,000 strict games:
`strict:branch_event_legal_error:BranchLegalRollError` **146** and `skip:rump_branch_set` **14**,
both zero on the banded arm. The refutations stand, and now stand on a reproducible build.

**How they were refuted is the part worth keeping.** Not by re-reading the corpus. Over every
artifact committed *before* C152 both really are 0, and `tests/test_never_fired_counter_census.py`
re-derives that on every run — the C146 machinery worked exactly as designed and could not have
caught this. They were refuted by **measuring somewhere new**: 1,000 games on unregistered seeds
`1,000,000`–`1,000,999`, run for an unrelated purpose (G33b's open arms) on the same 74-patch
engine. Both fire immediately outside the two 200-game windows this program has iterated against
for its whole history. The corresponding standing rule is added to §8. Both names are now pinned
as **fired**, with the pin additionally asserting their evidence is still confined to the wide
census — so if either ever appears in a dev or holdout sweep, that is a new fact and goes red.

⚠ **CORRECTED 2026-08-07 (C142).** This list said **10** and included
`skip:strict_all_branches_lossy` (H14) — cross-referencing the very cell that, as of C144's
correction above, now reads **"REACHED, three times over"**. The list and H14 contradicted each
other two paragraphs apart, and #1163 fixed H14 without reaching here. Removed, and the count
re-derived rather than decremented on assertion: each of the other nine names was searched for a
nonzero value across all 260 committed JSONs under `reports/`, and all nine are still absent, so
10 − 1 = **9** is the measured figure. The refutation rests on hard counter values —
`reports/c26_structural_probe_report.json` and `reports/c27_structural_probe_report.json` carry
`skip:strict_all_branches_lossy` at **2** apiece — not on narration. (An older-era diagnosis,
`reports/c32_fail_diagnosis.json`, records the same phenomenon at **372** under the differently
named field `coverage_diagnosis.coverage_reducing_skips.strict_all_branches_lossy`; it is
corroborating rather than the counter itself.) `engine_error` remains genuinely never-fired,
which is what H14's own closing sentence says. See `reports/c142_rump_branch_adjudication.md`.

✅ **RE-VERIFIED 2026-08-07 (C146), and the number holds.** C142's glob was "all 260 committed
JSONs under `reports/`". Re-run over **347** — `reports/` *and* `docs/`, recursive, which adds
the 80 `docs/audit_artifacts` search-grid artifacts where three other refusal reasons reach the
hundreds — all nine are still absent, so **9** is measured against a strictly wider corpus than
the one that produced it. This is a **verified negative**, not an asserted one, and it is the
only one of §3.5's four lists that needed no correction. One trap worth recording, because
admitting it flipped two of the nine: `no_usable_branch` and `BranchLegalRollError` each appear
in prose inside a JSON *value* next to an unrelated number
(`reports/c9_decomposition.json` / `c12_decomposition.json` `"basis"`, and a c17 sentence). A
name in narration is not a counter value; the census pin carries
`test_prose_alone_is_not_evidence` as the control. ⚠ **`BranchLegalRollError` left that control on
2026-08-08 (C152)**, because it now has real counter evidence and can no longer distinguish "prose
was admitted" from "the counter fired"; keeping it would have turned a genuine refutation into a
red matcher pin and invited someone to loosen the matcher to make it green. `no_usable_branch`
still has both properties and carries the control alone, with an added anti-vacuity assertion that
the prose really is present.

**Never-fired dynamic families (6):** `skip:no_materialization:{Exc}`,
`skip:world_error:{Exc}`, `strict:branch_events_error:{Exc}`, `engine_error:{Exc}:{detail}`,
`engine_error_choice:{choice}`, `world_prestate_mismatch:weather_{WEATHER}`.
✅ **VERIFIED (C146):** no key under any of these six prefixes carries a nonzero value in any of
the 347 artifacts.

**`skip:unmappable_choice` — 7 of 8 unobserved:** `no_candidate_row`, `blank_move_id`,
`hidden_power_ambiguous`, `move_not_in_engine_set:{id}`, `blank_switch_species`,
`switch_species_not_in_party`, `unknown_kind:{kind}`.
✅ **VERIFIED (C146)** over the 347-file corpus: all seven absent, and the eighth
(`struggle_not_submittable`) present in 78 of them, which is the control that makes the seven
mean something — a scanner that found nothing would have "verified" all eight.

**`skip:world_unsupported` — 36 of 40 reasons unobserved in both windows.** That window figure
is correct and re-derived (the four that fire in c136 are `volatile_unsupported` 144/127,
`materialization_blocker` 18/8, `encore_move_unknown` 2/1 and `self_request_state_unsupported`
13/0). Two of the 36 are structurally diverted on the default flags and their absence *in the
windows* is expected (`status_unsupported` → `hidden_counter_support:sleep`;
`substitute_health_unknown` → `limit:world_substitute_health_unknown`). One is UNREACHABLE in
this pool and should be read as retired rather than untested: **`future_sight_pending`** — see
R1. The remaining 33 were listed here as "unobserved exits".

⚠ **CORRECTED 2026-08-07 (C146): four of the 33 have fired, and so have both diverted ones.**
Marked ⚠ **FIRED** in place below, with the value and the artifact. **Twenty-nine have no
nonzero record in any of the 347 committed JSON under `reports/` and `docs/`** — that is the
verified negative, and it is what this list is now good for. Two are (marked) additionally
unreachable in this pool:
`boost_unsupported`, `boundary_not_move_request`,
`deferred_opponent_action`, `hidden_power_iv_mismatch`, `item_state_conflict`, `move_unknown`,
`nature_not_neutral` (also UNREACHABLE — R7), `override_side_missing`,
`payload_malformed` (⚠ **FIRED** — 4, `reports/c112_leaf_state_scenarios.json`),
`pending_baton_pass` (⚠ **FIRED** — 3 / 3 / 2 in `reports/c112_leaf_state_golden_v2.json`,
`_v4.json` and `_scenarios.json`), `public_effect_blocked`, `public_species_not_in_world`,
`rest_sleep_attempt_unsettled`, `rest_sleep_provenance_unrepresentable`,
`rest_sleep_refund_pending_precounts_legacy`, `rest_sleep_refund_pending_unsplit_legacy`,
`self_maxhp_mismatch`,
`self_moveset_mismatch` (⚠ **FIRED, and in these windows** — 75 dev / 24 holdout across 27
sweep artifacts c121–c133, 108 in the c7–c13 era, 11 in `reports/c112_leaf_state_scenarios.json`,
**5,058** in `reports/c32_fail_diagnosis.json`, `ranked[2]` in
`reports/c43_coverage_shortfall_diagnosis.json`, up to **2,560** in
`docs/audit_artifacts/hc-depth-grid-20260729/hc-d1.json`; closed by `29ca5697` — see H13),
`self_pp_unknown`, `self_world_mismatch`,
`side_condition_turns_inconsistent`, `side_condition_turns_unknown`,
`side_condition_unsupported`, `species_unknown`, `substitute_depletion_world_incompatible`,
`substitute_health_provenance_contradiction`, `toxic_stage_inconsistent`, `toxic_stage_unknown`,
`transform_unexpressible` (⚠ **FIRED** — 23 in c32, `ranked[8]` in c43, 208 in
`docs/audit_artifacts/k0-depth-grid-20260729/results/k0g-{a,c}-d1-1.json`),
`weather_turns_inconsistent`, `weather_turns_unknown`,
`weather_unsupported` (also UNREACHABLE — R8), `wish_turns_inconsistent`.

And the two diverted ones, for completeness, since "expected absence" was doing double duty as
"never fired": `status_unsupported` ⚠ **FIRED** — 2 in c32, `ranked[9]` in c43, and 9,071 /
3,453 in `docs/engine_divergence_ledger_20260728.md`; `substitute_health_unknown` ⚠ **FIRED** —
12 and 14 in `reports/c112_leaf_state_golden_v{2,4}.json`.

**One-sided exits worth watching:** `skip:world_unsupported:self_request_state_unsupported` is
13 in dev and **absent** in holdout; `hidden_counter_support:confusion` is 1 in dev and 0 in
holdout, so the entire confusion hidden-counter machinery rests on a single observation across
400 games.
✅ **BOTH RE-MEASURED WIDE (C153), and neither is one-sided outside the windows.** Over the
10,000-game census: `skip:world_unsupported:self_request_state_unsupported` **902** and
`hidden_counter_support:confusion` **207**. The confusion machinery no longer rests on a single
observation — but note precisely what changed: the *dev/holdout* figures are unmoved and still 1
and 0, so this is evidence that the machinery is exercised **somewhere**, not that the two
permitted windows exercise it. `reports/c153_wide_seed_negative_census.md`.
(Both numbers were quoted as 1,510 and 113 in a draft of this line, taken from checkpoints while
the census was still running. They are re-derived from the twelve committed shards here — a figure
read off an unfinished measurement is the stale-denominator defect one turn earlier.)

---

## 4. Dropped by the reachability filter — verified UNREACHABLE

This is the section item 14 exists for. Each of these is a real difference between the engine
and gen3 Showdown, or a real inexpressibility, that **cannot be reached in gen3 randbats**.
Carrying them in a ledger would inflate it and mislead a reader about where the risk is.

| # | Candidate | Why it is unreachable, measured |
|---|---|---|
| **R1** | **Future Sight / Doom Desire — residual order 11, and the `future_sight_pending` refusal** | `futuresight` and `doomdesire` are each **0 of 220** species. They are the whole of gen3's delayed-damage class, so residual order 11 is unreachable and `_reject_unsupported_globals`'s `future_sight_pending` raise is dead code in this format. (Re-derived; C125 reached the same verdict.) ⚠ **C154 correction 2026-08-09 — the reason is INCOMPLETE and the closure is one keyword argument deep.** Two things this cell did not say. The payload ALWAYS CARRIES the key (`_public_materialization_payload` emits `"futureSight": dict(replay.future_sight)` unconditionally), so the closure is "the mapping is always EMPTY", never "the key is absent" — the distinction C153 drew for `deferred_opponent_action` and nobody carried here. And **the repo ships a Custom Game scenario that casts Future Sight** (`golden_corpus_scenarios.py`, spec `future_sight_pending`), so the raise is constructible from committed code; what keeps it unreached is that the spec sits in `interaction_registry_specs()` and not `scenario_specs()`, which every world-building harness defaults to. "Dead code in this format" holds; "retired" would not. Traced demonstration in `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R2** | **Hail, and everything downstream of it** — the ICE branch of `weather_chips`, `Items`/ability hail interactions, `world_prestate_mismatch:weather_HAIL` | `hail` is **0 of 220** as a move, and the only ability that sets it — Snow Warning — reports `gen: 4, isNonstandard: "Future"` in `Dex.mod('gen3')`, i.e. it does not exist in gen3 at all. Two independent routes, both closed. |
| **R3** | **Rain Dish's `maxhp/16` rain heal, and its missing `ResidualPlan` slot** | Rain Dish is **0 of 393 sets**. Verified at the species level too: the only three gen3 species with Rain Dish are Lotad, Lombre and Ludicolo; Lotad and Lombre are **not in the pool at all**, and Ludicolo's single set lists only `["Swift Swim"]`. |
| **R4** | **Weather-expiry sand/hail chip truncation** | **0 of 220** species carry `sandstorm` or `hail` as a move (`raindance` 7 and `sunnyday` 4 exist and neither chips). So sand only ever comes from Sand Stream — **Tyranitar alone**, on all 3 of its sets — which sets `WEATHER_ABILITY_TURNS = -1` and never expires. "The expiry path has no trigger." Corroborates `reports/c131` §2. ⚠ **C154 correction 2026-08-09 — the sentence quoted above is FALSE; the verdict survives.** The order-8 decrement-and-clear block fires on every Rain Dance (7 species) and Sunny Day (4), and `weather_survives_upkeep` evaluates false on their expiring turn: the expiry path is exercised in ordinary play. What has no trigger is the narrower thing this row is titled after — a CHIPPING weather holding a FINITE counter. Two neighbouring over-readings go with it: permanent (`-1`) weather is not Tyranitar-only (Kyogre's Drizzle and Groudon's Drought write it too), and the payload-seeding lane is closed by a mechanism this cell does not cite (`_weather_fields` returns `-1` only under `weatherFromAbility`; a prefix that saw only `[upkeep]` lines fails closed at `weather_turns_unknown`). Traced in `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R5** | **Dry Skin's rain heal at order 10.3** | `Dex.mod('gen3').abilities.get('dryskin')` reports `gen: 4, isNonstandard: "Future"` — it does not exist in gen3. Stronger than C115's "dead code for gen3 randbats". |
| **R6** | **Sitrus Berry, and the monotonicity break it causes in the residual mirror's bisection** | Not in the 13-item universe. `getItem` cannot return it: verified by reading every return path and by generating 24,000 Pokémon. `reports/c111`'s A2 addendum ("threshold berries break the monotonicity … Sitrus fires at `hp <= maxhp/2`") is therefore unreachable. The three reachable pinch berries (Salac, Petaya, Liechi) fire at `maxhp/4` and grant **stat boosts, not HP**, so they do not break HP monotonicity. ⚠ **C154 correction 2026-08-09 — the reason is a GENERATION-TIME argument and the row needs a RUNTIME one.** "`getItem` cannot return it" says what a team STARTS with and nothing about acquisition in play — the exact gap that made R26 wrong, in this document, about this mechanic, one row below. Closed here by measurement rather than by inference: the pool's only item-moving moves are `trick` (2 of 220) and `knockoff` (4), which SWAP and REMOVE; `thief`, `covet`, `recycle`, `switcheroo` and `bugbite` are each 0; no gen3 mechanism CREATES an item. A closed set of 13 stays closed under permutation and deletion. Re-derived over 24,000 Pokémon under a second seed scheme in `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R7** | **`nature_not_neutral`** | Generated sets carry **no nature field at all** — measured unset on **24,000 of 24,000** Pokémon (the single §1.3 census run; an earlier draft quoted a stale 12,000 denominator from a superseded run). The engine_world comment ("Gen 3 randbats sets are neutral") is correct and the refusal is unreachable. ⚠ **C154 correction 2026-08-09 — the reason is HALF the demonstration, and the missing half points the other way.** "No nature field at all" does not close the refusal on its own; it would OPEN it, because an absent field makes `mon.nature` falsy and the tested value the empty string. What closes it is that `""` is a MEMBER of `_NEUTRAL_NATURES`. Second, the generator is not the only producer that reaches the guard: `_build_pokemon_spec` has a caller in `engine_fidelity.py` that never touches a packed team, and `scenario_studio` parses `nature` from scenario JSON with no vocabulary check. Neither fires today, for reasons this cell does not state. Traced in `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R8** | **`weather_unsupported`** | All four gen3 weathers are in `_WEATHER_IDS` (`rain`, `sun`, `sand`, `hail`). No gen3-legal weather can miss the map. ⚠ **C154 correction 2026-08-09 — the reason names the wrong side of the mapping and survives by coincidence.** The four strings listed are the dict's VALUES; the lookup uses its KEYS. They happen to be keys as well — they are the engine-side aliases — so the membership sentence is technically true, and still not the demonstration: **three of the four strings named are never what the lookup receives.** Showdown emits the condition NAME, which normalises to `raindance` / `sunnyday` / `sandstorm` / `hail`, and the closure rests on the three ALIAS keys this cell does not mention. A map holding only the four it names would refuse every rain, sun and sand battle. "All four gen3 weathers" also over-reaches: hail is unreachable (R2), so only three can occur. Traced in `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R9** | **Liquid Ooze mislabelled by the residual-heal renderer** | Two things are true and they are different. Liquid Ooze *is* reachable — 2 of 220 species (Swalot, Tentacruel). But the **renderer path** is unreachable: in `events.rs` `render_residual_instruction`, an `Instruction::Heal` with `heal_amount < 0` is intercepted and rendered as `\|-damage\|…\|[from] ability: Liquid Ooze` **before** `plan.take(side, true)` or `residual_heal_cause` is ever called. "The Liquid Ooze guard inside `residual_heal_cause` is therefore dead code," exactly as `reports/c131` §5 said. On the mechanic side, note that both `leechseed` and `gigadrain` are `target: normal`, so the interaction that matters is a **seeder or drainer facing** a Liquid Ooze holder — cross-side, and reachable, since 12 and 2 species carry those moves against 2 Liquid Ooze carriers. The "0 of 393 sets pair them on one set" measurement is true but answers the wrong question and is not load-bearing here; "what makes this row UNREACHABLE is the renderer interception alone." ⚠ **C154 correction 2026-08-09 — the quoted sentence is FALSE and is a NON SEQUITUR from the one before it.** The guard is not in a negative-heal branch: it is the conjunct `ability != Abilities::LIQUIDOOZE` inside the LEECHSEED arm, which runs only on a POSITIVE heal — exactly the heals the interception lets through — and `residual_heal_cause` takes no heal amount, so the sign is not a fact it can see. It is not dead in the literal sense either: the crate PINS it with a positive heal and no Leftovers, and that pin exists because deleting the guard once left the suite green. What forecloses the conjunct IN THIS FORMAT is the two EARLIER returns plus the item universe — a resolving Wish returns first, a Leftovers holder returns first, and the only other gen3 residual positive heals are Sitrus (outside the 13, R6), Rain Dish (R3), Ingrain (0 of 220) and the drain itself, which under Liquid Ooze is negative and intercepted. There are also THREE ooze-aware sites, not one: the residual plan's own conjunct is live and pinned in both directions, and the MOVE-PHASE ooze renderer has its own interception and is genuinely REACHABLE (`gigadrain` on 2 species against Liquid Ooze on 2). `reports/c131` §5 is corrected in place with the same finding. Traced in `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R10** | **Shell Bell, and the `heal_drain_or_shellbell` ambiguity** | Shell Bell is modelled in the engine (`src/gen3/items.rs` `SHELLBELL`) and is **not in the 13-item universe**. So the bucket the crate names for an ambiguity it "cannot resolve" is, in this format, unambiguous: the only reachable producer is a drain move, and the pool's only drain move is `gigadrain` (Exeggutor, Parasect). ⚠ **C154 correction 2026-08-09 — the bucket is not merely unambiguous, it is UNEMITTABLE, and this cell reasoned from its NAME without tracing its caller.** Every production path into `heal_subcase` runs through `render_move_phase`'s Sleep Talk block, by **2 routes** — the `sleeptalk_refusal_is_unsafe_with_protect` predicate and the slug emit, of which only the second produces a key — so the tail is always a Sleep Talk CALLEE's tail and the bucket needs `sleeptalk` and a drain move on the SAME SET: 44 Sleep Talk sets, 3 drain sets, **0 of 393 pair them**. ⚠ **This sentence said "one non-test path" until 2026-08-09, and the graph it describes has two** — the fifth instance in this document of a correction not reaching the prose describing it, inside the correction filed against reasoning-without-tracing. The route count is now taken from the derived graph and pinned, not written. Two smaller upgrades: "the pool's only drain move is `gigadrain`" is now the `drain`-flagged subset of the pool's 125 moves derived from the dex, with the five gen3 drain moves it excludes named; and the Shell Bell half needs R6's runtime clause. One site this cell does not cover: the named renderer tags a heal `[from] drain` whenever `choice.drain.is_some()`, and Shell Bell sets that field. Traced in `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R11** | **The `TwoToFiveHits` flat-3.2 approximation** | Every `[2,5]`-hit move is **0 of 220**. The only multi-hit move in the pool is `bonemerang`, whose `multihit` is the scalar `2`. See G9 for what *is* reachable. |
| **R12** | **Rest's Insomnia / Vital Spirit fail clauses** | **0 of 393 sets** pair `rest` with either ability (independently reproducing the patch's "0 of the 55 Rest sets"). Comatose is gen7. |
| **R13** | **Belly Drum's Shedinja `maxhp === 1` fail clause** | Shedinja's single set is `agility / batonpass / hiddenpowerfighting / shadowball / silverwind / toxic`. No Belly Drum. It ships for source parity only. |
| **R14** | **N5 — the residual ceiling overshooting into a move-KO** | c129 measured 4,326 such states, **every one at `maxhp <= 47`**. Measured over 60,000 generated Pokémon, the minimum maxhp in the pool is **1** and **Shedinja is the only species at or below 47**. ⚠ The original evidence then said "and Shedinja carries no multi-hit move" — a *same-side* check, and therefore the wrong one, since the multi-hit move belongs to the **attacker**. The correct argument is stronger: N5 additionally needs `hit_count > 1`, the pool's only multi-hit move is Bonemerang (Ground), and Ground is **resisted** by Bug/Ghost (`getEffectiveness('Ground', ['Bug','Ghost']) = −1`), so **Wonder Guard blocks it outright**. Enumerated exhaustively: of the pool's 125 moves, **every** move that is super-effective against Bug/Ghost is single-hit. No multi-hit move can damage Shedinja at all. ⚠ **C154 correction 2026-08-09 — the corrected argument still had a hole in the same cross-side family, and the exhaustive enumeration it rests on is vacuous.** It turns entirely on Wonder Guard and never asks whether the OPPONENT can remove it: in gen3 the two moves that can are Skill Swap and Role Play (Gastro Acid, Worry Seed, Entrainment, Simple Beam and Mold Breaker are gen4+). Both are 0 of 220 — but nothing had closed the route until it was measured. Transform IS in the pool (2 species) and is not a route: it copies the ability, not the HP. "Every move super-effective against Bug/Ghost is single-hit" is true and adds nothing, because every pool move except Bonemerang is single-hit; it is kept as a control, not as the argument. The stronger argument, which c129 made and this row dropped, is `residual_disjoint_bands`'s `threshold < ceiling` admission gate: at `hit_count == 1` the arm deals strictly less than the defender's HP by construction, and Shedinja's `hp == 1` makes the gate unsatisfiable outright. Finally there are TWO N5 sites, not one — the same ceiling appears on the crit-straddle branch. Traced in `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R15** | **Magic Coat / reflect path, and Ingrain blocking phazing** | `magiccoat` and `ingrain` are each **0 of 220**. The `reflectable: true` flag upstream puts on Roar/Whirlwind is inert. |
| **R16** | **Dragon Rage and Psywave emit no instructions** | Each **0 of 220**. (Night Shade, also 0, was fixed anyway by the fixed-damage-pipeline patch.) |
| **R17** | **Eruption / Water Spout one-ULP ordering divergence** | Each **0 of 220**. |
| **R18** | **Locked-continuation PP on Outrage / Petal Dance / Thrash** | Each **0 of 220**. The reachable half of that patch is Solar Beam (4 species) and Hyper Beam (Slaking). |
| **R19** | **Snore treated as not sleep-usable** | `snore` is **0 of 220**. |
| **R20** | **Low Kick's weight-based base power, which Transform does not copy** | `lowkick` is **0 of 220**. |
| **R21** | **Reflect / Light Screen keeping a trailing float position in the damage pipeline** | `reflect` and `lightscreen` are each **0 of 220**, as are `safeguard` and `mist`. No Pokémon in the pool can set a screen. (Flagged for re-checking because `engine_world` *can construct* screens as side conditions — but no battle path reaches that state, so the construction capability is not a reachability route.) ⚠ **C154 correction 2026-08-09 — this row flagged itself for re-checking and then answered with an assertion; re-tracing it moves where the closure sits.** It is NOT `side_condition_unsupported`: that fires for a condition OUTSIDE `_SIDE_CONDITION_IDS`, and `reflect`, `lightscreen`, `safeguard` and `mist` are all INSIDE it with turn counters derived rather than copied — if a screen appeared, `engine_world` would build it. The only thing keeping it from appearing is the protocol: a mapped condition enters solely through a `\|-sidestart\|` line, no pool Pokémon can emit one for a screen, and there is no copier route (`metronome`, `assist`, `mirrormove`, `mimic`, `sketch`, `naturepower`, `magiccoat` each 0; Sleep Talk draws from the user's own slots). `spikes` is the live control on the same map. One capability not mentioned here and not a battle path: the scenario harness injects arbitrary side conditions, and no committed scenario sets one. Traced in `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R22** | **Mimic, Imprison, Psych Up, Metronome, Assist, Nature Power, Sketch, Mirror Move** | Each **0 of 220**. "This closes `_HIDDEN_INFORMATION_REQUEST_FLAGS`'s `maybeDisabled`/`maybeLocked` (Imprison is their only producer), the `failencore` move-list edge cases, and G32." ⚠ **C154 correction 2026-08-09 — the quoted sentence is FALSE in one clause and uses the wrong verb in the other.** (1) The `failencore` edge cases are NOT closed. `move_fails_encore` matches `ENCORE \| MIMIC \| MIRRORMOVE \| SKETCH \| STRUGGLE \| TRANSFORM`, and this row's eight names cover three of the six: `encore` is 16 of 220, `transform` is 2, and Struggle is reachable by PP exhaustion. **Nothing opens, and that was checked rather than hoped** — the crate's six are exactly the non-`Future` gen3 moves carrying Showdown's `failencore` flag, which is the condition `encore.condition.onStart` actually tests, so the shipped list is right for its reachable members too. The clause is withdrawn; the patch is not. (2) "Closes" is wrong for the flags: `_HIDDEN_INFORMATION_REQUEST_FLAGS` is a TOLERATE-list whose members are filtered OUT of the refusal binding, so `maybeDisabled` and `maybeLocked` never caused a refusal. What unreachability protects is a SILENT failure — under Imprison the singles request reports the blocked moves as `disabled: false`, `engine_world` tolerates the flag, and search plans a move Showdown rejects. Traced in `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R23** | **Focus Energy, Mud Sport, Water Sport, Taunt, Torment, Disable, Nightmare, Foresight** | Each **0 of 220** as moves. The `volatile_unsupported` refusals keyed to them cannot fire from play. ⚠ **C154 correction 2026-08-09 — "0 of 220 as moves" is not the whole producer set for two of the eight, and "cannot fire from play" is false at the scope written.** (1) The `foresight` VOLATILE has a second gen3 move producer, `odorsleuth` (`volatileStatus: 'foresight'`, no condition of its own), also 0 of 220 — one move id checked where two exist, the R26 shape. (2) `focusenergy` has a live gen3 NON-MOVE producer, **Lansat Berry**, whose `onEat` calls `addVolatile('focusenergy')`; a move census cannot see it and what forecloses it is the 13-item universe plus the absence of any item-creating move. (3) The refusal DOES fire, on this repo's own scenario corpus: `struggle_taunt_stall` is refused with `volatile_unsupported: side 'p1': ['taunt']`, recorded in `docs/belief_edge_case_matrix.md`. The correct scope is "cannot fire from SAMPLED RANDBATS play". ⚠ The enumeration that found (1) and (2) is itself a corrected instrument: a first version regexed `JSON.stringify` of each dex entry, which drops functions and therefore reported the same-named move as the only producer of all ten — a scan that could not fail. Both scans are recorded in `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R24** | **Attract as a *move*** | `attract` is **0 of 220**. The volatile is reachable only through Cute Charm — see G37, which is the correctly-scoped version of the gap. ⚠ **C154 correction 2026-08-09 — two scope notes this phrasing invites a reader to get wrong.** `attract` is a MEMBER of `_SUPPORTED_VOLATILES`, so the volatile is expressed and searched, not refused — unlike the eight in R23, the neighbouring row. And "reachable only through Cute Charm" is randbats-scoped: the scenario corpus produces it from the move itself (`attract_snorlax`). The gen3 producer set is now enumerated rather than asserted — move (0 of 220), Cute Charm (3 species), Destiny Knot (gen4), G-Max Cuddle (gen8) — and Baton Pass cannot move it (`noCopy: true`). See `reports/artifacts/c154_unreachable_readjudication.json`. |
| **R25** | **Sleep Talk calling Haze / Psych Up / Roar / Whirlwind / Baton Pass** | **0 of 393 sets** pair `sleeptalk` with any of the five. Measured on this `sets.json`, not carried from the crate's three-universe count. |
| ~~**R26**~~ | ~~**Trick-style item acquisition reaching White Herb**~~ | ❌ **WITHDRAWN — this verdict was WRONG, and it is the error this whole section is supposed to prevent.** The original evidence was "White Herb's only holders are `deoxys` and `deoxysattack`, and neither movepool contains `trick`". **Trick is `target: normal`** — the *opponent* uses it, so the holder's own movepool is irrelevant. Measured, the pairing occurs in roughly **1 battle in 700**. Reclassified as **REACHABLE** at **G49**. What survives of the original row: Transform does not copy items and Psych Up copies only boosts, so neither is an item-acquisition route — but Knock Off (4 species) is an item *removal* route on the same cross-side footing, handled by `removed_item_species` rather than here. |
| **R27** | **Quick Claw, King's Rock, Bright Powder, Lax Incense, Focus Band, Scope Lens, Berry Juice, Leppa/Oran/Chesto/Pecha/Rawst/Aspear/Persim/Cheri Berry, and every type-boosting item outside the 13** | None is in the 13-item universe. This is worth stating positively because of what it retires: **no priority randomness (Quick Claw), no item-sourced flinch (King's Rock), no evasion item, no crit item, no HP-restoring berry, and no item-sourced heal-on-damage (Shell Bell)** can occur in gen3 randbats. Any engine gap in those mechanics is unreachable here. ⚠ **C154 correction 2026-08-09 — the same missing runtime clause as R6 and R10, and it matters most here** because this is the row the ledger uses to retire a whole block of mechanics at once. A 13-item generation census bounds what a team STARTS with. It is closed in play as well, measured: `trick` (2 of 220) swaps and `knockoff` (4) removes, `thief`/`covet`/`recycle`/`switcheroo`/`bugbite` are each 0, and no gen3 mechanism creates an item — so no fourteenth item can appear mid-battle. The 13 reproduce under a second seed scheme over 24,000 Pokémon in `reports/artifacts/c154_unreachable_readjudication.json`. |

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
6. **Name the unobserved exits rather than omitting them — and say which corpus made them
   unobserved.** 36 of 40 world-construction refusal reasons are zero *in the two c136 windows*;
   **30 of 40** have never fired *anywhere* in the 347 committed JSON under `reports/` and
   `docs/`, and the gap between those two numbers is six reasons that fired and closed. 7 of 8
   unmappable-choice reasons and **12 of 19** divergence classes (⚠ not 13 — H15) have never
   fired repo-wide. Some are unreachable (R1, R7, R8) and should be retired; the rest are
   untested code paths sitting behind the measurement. §3.5 is the list, each entry now labelled
   ⚠ FIRED or verified absent, and re-derived by
   `tests/test_never_fired_counter_census.py` rather than restated. (H13, H15)
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

9. ⚠ **A SECOND accept bar, added 2026-08-08 (C152), and it is independent of the sleep bar.**
   `roll_components_agree`'s ±9 % fallback window is not a tie-breaker at the margin: removing it
   and changing nothing else turns **167 dev and 140 holdout matched boundaries into
   divergences** — 1.077 % and 0.899 % of measured. Those boundaries match on a proportional
   tolerance around the engine's representative roll, not on membership of Showdown's enumerated
   fan. Measured in `reports/artifacts/c152_h8_window_census.json`. A claim that says "matched"
   without this reads as exact agreement, and for about one boundary in a hundred it is not.
   (H8)

A wording that satisfies all nine looks roughly like: *"On gen3 random battles as generated by
Showdown `f76228a1`, over two disjoint 200-game seed windows, the engine's attributed HP
components matched Showdown's on 15,503 of 15,503 measured full-round boundaries in dev and
15,579 of 15,579 in holdout — 87.5 % and 86.7 % of all boundaries respectively. Branch
probability masses were not compared, and 27 documented gap candidates are unreachable in this
pool and therefore untested by it. About 9 % of those matches were accepted under a widened
hidden-counter support bar rather than an exact one, a further ~1 % on a ±9 % roll tolerance
rather than exact fan membership, and terminal adjudication was not compared at all."*

⚠ **"Widened sleep-counter bar" became "widened hidden-counter support bar" in the same pass, and
the reason is measured.** The gate is `gating:support`, and sleep is its dominant but not its only
component: dev's 1,347 support-gated boundaries sit alongside a `hidden_counter_support:confusion`
counter at **1** as well as `hidden_counter_support:sleep` at 1,352
(`reports/artifacts/c152_head_dev_sweep.json`; those two are per-state tallies, not a partition of
the 1,347, so they establish that the bar is not sleep-only and are **not** its composition).
Holdout carries no confusion key at all. Naming the widest accept bar after its narrowest
component is the same defect as a glob reported wider than it was run, one surface over — and this
row's own cell already used the precise form, so the document disagreed with itself.

⚠ **The two match counts in that sentence were 15,501 and 15,575 until 2026-08-08 (C152) and are
now 15,503 and 15,579** — re-derived from `reports/artifacts/c152_head_{dev,holdout}_sweep.json`
at engine `bfdbe1c04876edcd` / 74 patches, not carried. C149 closed the last dev row and the
holdout window has been at zero since C136. **Zero divergences is not zero gaps**, and this
document is the reason: the number moved because the measured population is 87 % of boundaries,
compared on a bar that includes two widenings, on a path whose oracle is enumeration.

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
2. ⚠ **RESOLVED 2026-08-08 (C152) and kept here as the record of what it was.** *"How much
   matched mass rides on the comparator's ±9 % fallback window (H8). The state-level counter is 0
   in both windows; the per-branch usage is not counted at all."* **Measured: 167 dev / 140
   holdout boundaries, 1.077 % / 0.899 % of measured**, as the difference between the shipped
   comparator and a window-disabled variant. The per-branch usage is now counted too — 190 / 181
   accepts, **all** through the fan-miss door and **none** through the `pre_legal`-absent door
   this item assumed. See the H8 cell.
3. ⚠ **RESOLVED 2026-08-08 (C152), and the item was asking the wrong question.** *"Whether c43's
   ~7,224 invisible shortfall rows still exist (H12)… the single-seat arm carries no exit taxonomy
   so nothing can be said about it."* c43 **excluded** the single-seat population from the 7,224
   by its own note, so the single-seat arm was never what would settle it. The full-round identity
   closes **exactly** in both windows at head, which is the whole claim. What survives is not
   H12 but **H1** — the single-seat population is counted and uncompared — plus the new item 10
   below.
4. ⚠ **RESOLVED 2026-08-08 (C152).** *"Whether the four never-adjudicated families survive into
   the current era (H19)."* Two of the four did survive into c136 (`LS_capped_lethal_shape`,
   `I5_boundary_truncation`) and two did not; all four are at **0 rows** in both windows at head,
   with each family's last row named and replayed. The item's own settling script raised
   `NameError` on every input until C152 fixed it. See the H19 cell.
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
   incidence changed since commit `aeaee2b1` would be mis-stated here. ⚠ **Partly repaired
   2026-08-08 (C152):** an engine was built at the shipping fingerprint `bfdbe1c04876edcd` / 74
   patches and both windows were swept — `reports/artifacts/c152_head_{dev,holdout}_sweep.json`,
   the first committed sweeps at that fingerprint. The rows dispositioned by C152 read those.
   **Every other "observed" column in this document still reads the c136 pair** and is therefore
   still as of `aeaee2b1`; the head sweeps say only that both windows are now at **0 divergences**,
   which cannot re-derive a per-row "observed" for a row that never had one.
10. ⚠ **The single-seat population has no taxonomy anywhere, and that is now a stated blind spot
   rather than a sentence inside H12.** Added 2026-08-08 (C152). Searched over all of `reports/`
   and `docs/`: no committed artifact breaks `skip:single_seat_boundary` into categories, no
   sub-keyed counter (`skip:single_seat_boundary:*`) exists, and nothing emits one.
   `reports/c132_single_seat_coverage_bound.md` §3 is the only mechanistic account and is argued
   from two hand-driven Showdown probes with no counts. So 1,742 and 1,813 boundaries — 9.84 %
   and 10.09 % — are a single undifferentiated lump. **Settling measurement:** a sub-key on the
   single-seat branch of `run_game` recording the ply's shape (post-faint replacement, phazing,
   Baton Pass, other), over a 200-game dev sweep.
11. ⚠ **Both permitted windows are at 0 divergences and the engine is not divergence-free.**
   Added 2026-08-08 (C152), because "0 and 0" is the most misreadable number in this record. A
   1,000-game census on unregistered seeds `1,000,000`–`1,000,999`, run for a different purpose
   on the same 74-patch engine, produced divergent rows. The zeros are a property of two
   200-game windows that the program has iterated against for its whole history, not of the
   engine. Figures and classes in `reports/c152_ledger_terminal_disposition.md` §7.

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
- **A negative claim carries its glob, or it is marked unverified.** Added 2026-08-07 (C146),
  after the *fifth* false "never fired" in this document. Three sub-clauses, one per way the
  previous four went wrong:
  - **"Zero in the current window" and "never fired" are different claims, and a closed exit
    reads as the first.** H13 called `self_moveset_mismatch` never-fired while 27 committed
    artifacts on the **same two seed windows** carry it at 75 and 24. Reading only "the newest
    committed post-fix pair" cannot distinguish a dead code path from a repaired one, and the
    ledger's job is to distinguish them.
  - **Match on the reason NAME, never only on a counter path.** `reports/c32_fail_diagnosis.json`
    files counters under `coverage_diagnosis.coverage_reducing_skips.<name>` and
    `reports/c43_coverage_shortfall_diagnosis.json` under `ranked[i].counter` with the count in a
    sibling field. Both H14 and H13 were refuted by exactly those two shapes.
  - **A glob scoped to one directory may not be reported as repo-wide.** H15's "6 of 19" is right
    under `reports/artifacts/` and wrong over `reports/` + `docs/`. `docs/audit_artifacts/**` is
    a committed measurement corpus and was invisible to every negative in this document until
    C146.
  A negative that satisfies all three is worth recording as **verified** — §3.5's nine static
  counters, six dynamic families and seven `unmappable_choice` reasons all are, and now say so.
  The rest of the inventory, sorted into measured and merely asserted, is in
  `reports/c146_negative_claim_audit.md`.
- **The reachability check must survive RENDERING, and that is now machine-checked.** Added
  2026-08-08 (C150). The rule above says the check must be recorded next to the entry — and it
  is possible to satisfy that in the bytes and fail it in the document, because GFM **drops**
  the surplus cells of a row that carries more delimiters than its header. That is not
  hypothetical here: at `a587e614^`, `G37` (20 pipes) and `G37b` (9) against this table's
  6-pipe / 5-column header both rendered with an **empty `Reachability evidence` cell**, their
  pool checks and `Observed` values gone from the document. #1166 fixed them, and the file has
  been clean at every commit since — re-derived at `f876803e` (0), `a587e614^` (2), `a587e614`
  (0) and `553cf2c3` (0). What #1166 left behind was prose, not a control, and its own note
  said so. `tests/test_ledger_table_uniformity.py` is now that control: every table exactly
  uniform, an exact per-table row inventory (so a gap cannot join without an author touching
  the pin and therefore the column beside it), a non-empty rendered **Reachability evidence**
  cell on all **82** gap rows — 80 until C152 added G33c, and the pin is an exact row
  inventory rather than a count, so it moved with the row — and a measurement on all 27 §4
  rows, plus an exact repo-wide
  inventory of the markdown rows whose cells GFM drops. 9 mutations applied, 9 caught; the two
  pre-#1166 rows are fixtures in the module. **A rule the ledger states and nothing re-derives
  is the same shape as a negative with no glob.**
- **A checker for a rendering defect is validated against a RENDERER, not against a rule you
  infer.** Added 2026-08-08 (C150), and earned the hard way in the same change. C150's first
  attempt used a delimiter rule read off CommonMark's backslash-escape section — odd runs of
  backslashes escape, even runs do not — and on that basis reported `G21b` and `R9` as having
  been dropping their reachability cells since #1151, and `reports/c146_negative_claim_audit.md`
  as having asserted uniformity falsely. **All of that was wrong.** GFM's table-cell scanner
  treats a pipe preceded by *any* backslash as escaped; `G21b` renders 5 cells with its
  reachability check intact, `R9` renders 3, the whole file renders 9 clean tables at
  `553cf2c3`, and C146's claim was true when written. Verified on local `cmarkgfm`, GitHub's
  `/markdown` API at `mode=gfm` and the same API at `mode=markdown`, over backslash runs of 1
  to 4, with a bare-pipe positive control proving each instrument detects real drops. The two
  rows were still edited, but only for what the change is: `\\|` renders a stray backslash
  inside the code span. **A plausible reading of a spec is not a measurement**, and a claim
  about what a document *looks like* has exactly one instrument.
- **A permanent-ledger cell may not cite a number with no committed artifact.** Added 2026-08-08
  (C150). `G8` cited a 124,188-fixture review scan (44,393 fired / 79,795 declined / 25,728 arms
  kept / 18,901 dropped / 115 mutant disagreements) that was never artifacted and whose own
  partitions do not reconcile — `25,728 + 18,901 = 44,629`, matching neither side of the first
  split. The property was never in doubt and is not withdrawn: it follows structurally from
  `ResidualThresholdLadder::insert` being an insertion sort with dedup. The *numbers* are
  replaced by `reports/artifacts/c150_band_split_trade_census.json`, reproducible from
  `scripts/c150_band_split_trade_census.py`, whose transcription is validated against the three
  committed crate fixtures and re-run against the artifact in CI. This is the document five false
  "never fired" claims propagated from, precisely because citing it felt like verification, so an
  uncheckable figure here is worse than no figure.

- ⚠ **A negative measured only inside the two permitted windows is a claim about those windows.**
  Added 2026-08-08 (C152), after the **sixth and seventh** false "never fired" in this document.
  C146's rule — *a negative claim carries its glob* — is about the **corpus**, and it is sound: the
  census pin re-derives every absence over 388 committed artifacts on every run. It cannot catch
  this class of error at all. `strict:branch_event_legal_error:BranchLegalRollError` and
  `skip:rump_branch_set` are 0 in every artifact committed before C152 **and fire immediately** on
  1,000 games of unregistered seeds `1,000,000`–`1,000,999`, on the same 74-patch engine. Every
  fidelity artifact this program has ever committed comes from two 200-game windows it has
  iterated against for its whole history, so "never fired" derived from them means "never fired
  in dev or holdout". Widening the corpus cannot find this; only widening the **measurement** can.
  The operational form: **before writing a never-fired claim, run the counter's own shape on seeds
  the program has not tuned against**, and if that is not affordable, scope the sentence to the
  windows in the sentence. This applies with full force to the two zeros in §2: dev and holdout
  are at 0 divergences at head, and the engine is not divergence-free — 12 divergent rows in
  80,439 boundaries on those unregistered seeds, including one `LS_capped_lethal_shape` and two of
  G33b's own `heal` / `itemleftovers` shape.
- ✅ **AND THAT RULE NOW HAS AN INSTRUMENT — added 2026-08-08 (C153), because it did not.**
  The clause above is sound and it shipped with nothing behind it, which its own reviewer said.
  `tests/test_never_fired_counter_census.py` re-derives every absence over 401 committed
  artifacts on every run, and the clause states in its own sentence that a corpus scan **cannot**
  catch this class of error. Nothing re-measured the affected negatives anywhere new, so §3.5's
  inventory was still asserted at a scope it had never been measured at — and that exposure was
  itself measurable: of 388 committed JSON, **103** carry a seed span and **83 of the 103 are the
  two permitted windows** (39 dev, 40 holdout, 4 single-seed holdout replays), 1 the burned block,
  4 C152's wide census, 15 obsolete-engine c6–c13 / c26–c27 artifacts.
  **The instrument is `tests/test_wide_seed_negative_census.py`**, over
  `reports/artifacts/c153_wide_negative_census.json` and twelve committed shards: a **10,000-game**
  census on unregistered seeds `1,001,000`–`1,010,999`, **803,264 measured boundaries**,
  **25.8×** the two windows, on the shipping build. Every entry of a derived 61-name inventory
  gets one of four verdicts, and **every verdict carries its scope in the sentence** — a bare
  "never fired" is a red gate in that module, not a terse row. Three things follow, and each is a
  rule rather than a result:
  - **The sample size is justified or it is not coverage.** Rule of three, at the denominator the
    claim actually uses. A per-boundary counter gets 3/803,264; a **`divergence_class` gets
    3/949**, because the classifier only runs on a boundary that already diverged — quoting the
    boundary bound for a class overstates the census by four orders of magnitude, and the pin
    asserts both separately.
  - **A second arm, because one matcher structurally cannot answer for the other.** H15's own cell
    scoped four classes to a path *"no committed artifact used"*. A strict-only census would have
    left that negative exactly where it was and reported it as measured.
  - **Name what the instrument cannot reach, beside the zero — and TRACE it to the call, not to a
    plausible sentence.** Six entries carry a `census_cannot_reach` note: the two structurally
    unreachable classes, two rest-sleep legacy canaries whose designed value *is* zero, and two
    override/payload shapes the differential never builds. A zero from an instrument that could
    never have produced a one is not the same measurement as a zero from one that could, and an
    audit has already reversed one "closed" that was a fourth category in disguise.
    ⚠ **A seventh was filed here and was wrong**: `public_effect_blocked` is REACHABLE — the
    differential passes `blocked_slots` from the production `_public_effect_signals` on a live
    observation, and the same scan drove six `transform_unexpressible` firings in this census — so
    it is re-filed as merely unobserved. Re-tracing the other four then found two more imprecise
    reasons. The rule the category now carries is the one that failure earned: **trace the raise
    site to the differential's actual call.**
  **What it found**: §3.5's four lists all survive at the new scope, and **H15's does not** — seven
  of its twelve fired, four of them because the cell had them in the wrong category. One new row,
  **H22**. `reports/c153_wide_seed_negative_census.md`.

- ⚠ **A gate that removes one over-booking does not close a family whose other bookings survive.**
  Added 2026-08-08 (C152), earned by **G33c**. C147 shipped `leftovers_slot_truncated` to un-book
  the winner's 10.4 Leftovers **heal** at a battle-end truncation, and `plan.usable` is set from
  the **damage** and the **heal** counts together — so a winner that also carries `brn`, `psn` or
  `partiallytrapped` in the same skipped bucket makes the side unusable regardless, and the gate
  is inert there. That was visible in C147's own result and was not read: 108 firings, no verdict
  moved either way. **When a gate's effect is mediated by a reconciliation, check every term of
  the reconciliation, not the one the gate touches.**

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

**Third round — the negative-claim audit (C146, 2026-08-07).** After #1163 (H14), #1162 (§3.5's
count) and #1165 (H11) each corrected a false negative in this document, the remaining
"never"-shaped claims were audited as a class rather than one at a time.
`reports/c146_negative_claim_audit.md` is the inventory; what changed here:

- **H13 corrected.** All three reasons it named by name have fired, and `self_moveset_mismatch`
  fired **in these two windows** at 75 dev / 24 holdout across 27 committed sweep artifacts
  before `29ca5697` closed it. This is the fourth instance of the same error shape in this
  document, and the second refuted by the C32/C43 differently-named-field artifacts specifically.
- **H15 corrected a second time.** 6 of 19 → **7 of 19** fired; 13 → **12** never. The seventh is
  `limit:world_sample_drag_target`, which the cell listed among classes "never produced". The
  cause was a glob scoped to `reports/artifacts/` and reported as repo-wide.
- **H17 partly retracted.** `reports/c137_phase2_enumerate_decision.md` was in the tree at
  `f876803e`, the commit that merged this document, so that negative was false when written —
  the same mechanism as H11's. `c134`'s report has since arrived; only `c119` is still absent.
- **G50 annotated.** Its "0 in both windows" is correct and is not never-fired.
- **§3.5 rewritten** to separate "zero in the c136 windows" from "never fired", with each of the
  four never-fired lists either ⚠ corrected or ✅ marked verified against a stated 347-file glob.
- **Six negatives promoted from asserted to verified**, which is a real result and recorded as
  one: the nine static counters (re-measured over a wider corpus than C142 used), the six dynamic
  families, the seven `unmappable_choice` reasons, `mapper_lossy` and `no_usable_branch`'s
  structural unreachability, and the 29 of §3.5's 33 that genuinely never fire.
- **Mechanized**, because five prose corrections in three days is the argument for it:
  `tests/test_never_fired_counter_census.py`, gated in
  `.github/workflows/engine-fidelity-gates.yml`. Battery: 10 mutations applied, 9 caught; the
  tenth is the documented fail-open (a subset assertion carrying H15's own wrong six-class
  expectation passes), which is why the partitions are asserted as set **equality**. No §3 row
  was added or removed, so §3 stays at 78.

**Fourth round — the wide-seed census (C153, 2026-08-08).** §8's newest standing rule shipped in
#1200 with no instrument, and its reviewer named that as the gap. `tests/test_wide_seed_negative_census.py`
is the instrument; `reports/c153_wide_seed_negative_census.md` is the measurement. What changed here:

- **§3.5's inventory re-derived and its own arithmetic corrected.** The four lists come to **50**,
  of which **46** — not 45 — rest purely on window-scoped measurement: `future_sight_pending` is
  usually counted among the measurement-independent names and is **not a member of the 50**, being
  retired under R1 before the corrections that take 33 to 29.
- **All 46 survive**, now at 803,264 boundaries on unregistered seeds instead of 31,082 inside the
  windows, with the scope written into each sentence and the rule-of-three bound stated.
- **H15 corrected a third time**, and the split rather than the count: 7 of its 12 fire, and four
  of the seven were filed as strict-path classes when they are protocol-evidence fallbacks. 12 →
  **5** never-fired, and `_FIRED_DIVERGENCE_CLASSES` moves 7 → 14.
- **H22 added**, taking §3 from **81 to 82** — H-rows 23 → 24, G-rows unchanged at 58.
  `classify_divergence` leaves 23.7 % of banded divergences in the bucket its own docstring
  forbids. ⚠ **This line first read "taking §3 to 79", incremented from the stale 78 recorded two
  blocks above** — the precise operation §1's shape paragraph forbids, committed inside the change
  whose subject is uncounted drift. Re-derived across both trees with the row selector, and the
  header sentence is now machine-checked rather than transcribed.
- **C152's two refutations re-measured on a build that can be rebuilt.** They had rested entirely
  on the throwaway instrumented `89797289…`; on the shipping `bfdbe1c04876edcd` they are 146 and 14.
- **Mechanized**, and the battery is the argument: **13 mutations applied, 13 caught**, enumerated
  in the pin module's docstring rather than merely counted — 12 of the author's plus a **13th found
  by review**, which was the important one: the `combined` per-divergence bound, the number this
  ledger quotes, was the one thing no pin recomputed. Two of the twelve were written expecting a
  pass and got one — a closure pin that read the artifact's own
  `agrees` flag walked straight through a perturbed sum, which is the self-certifying-field defect
  G8 was withdrawn for. Both sides are now recomputed from the shards. Pins verified under
  `python -m unittest` **and** direct execution, 24 tests either way, after #1200's five half-inert
  pins.

**Fifth round — section 4 re-adjudicated (C154, 2026-08-09).** C153's rule was written for
`CENSUS_CANNOT_REACH`'s seven entries and corrected three of them, and its own report says the
class recurs outside the map it was written for. §4's 26 UNREACHABLE verdicts had never been
through it: they are neither counters nor sweeps, so neither
`tests/test_never_fired_counter_census.py` nor `tests/test_wide_seed_negative_census.py` had ever
looked at one. `scripts/c154_unreachable_readjudication.py`,
`reports/artifacts/c154_unreachable_readjudication.json` and `tests/test_unreachable_readjudication.py`
are the third instrument; `reports/c154_unreachable_readjudication.md` is the pass.

- **All 26 verdicts SURVIVE, and THIRTEEN of the stated mechanisms do not** — four outright false,
  nine incomplete. That is the `deferred_opponent_action` shape thirteen times over and the
  `public_effect_blocked` shape zero times: no §4 row turned out to be reachable. **Nothing was
  closed and nothing new opened**; the ledger is exactly as open as it was, with thirteen fewer
  sentences a reader could rely on.
- ⚠ **C154 correction 2026-08-09 — the four false ones.** R4's "the expiry path has no trigger" — it fires on every Rain Dance
  and Sunny Day; what has no trigger is a *chipping* weather with a *finite* counter. R9's "the
  Liquid Ooze guard … is therefore dead code" — a non sequitur from its own previous sentence: the
  guard sits in a positive-heal branch the interception never touches, and the crate pins it. R22's
  "this closes … the `failencore` move-list edge cases" — three of the six members are reachable
  (`encore` 16 of 220, `transform` 2, Struggle by PP exhaustion); **nothing opens**, because the
  crate's list is exactly gen3's `failencore`-flagged move set, which was checked rather than
  hoped. R23's "cannot fire from play" — `volatile_unsupported: taunt` fires today on this repo's
  own scenario corpus; the true scope is *sampled randbats* play.
- **One error class accounts for three rows at once, and it is R26's.** R6, R10 and R27 all closed
  on "`getItem` cannot return it", a GENERATION-TIME argument that says nothing about acquisition
  in play — the exact gap that made R26 wrong, in this document, about this mechanic, in adjacent
  rows. The universe is closed at runtime too (`trick` swaps, `knockoff` removes, nothing creates),
  but nothing had said so.
- **Three rows rested on an enumeration that had not been run.** R23 named one producer of the
  `foresight` volatile where the dex has two (`odorsleuth`) and none of the `focusenergy` item
  producer that exists in gen3 (Lansat Berry). R14's Wonder Guard argument never asked whether the
  opponent can *remove* Wonder Guard (Skill Swap and Role Play, both 0 of 220 — closed, but by
  nothing until it was measured), and its exhaustive type enumeration is vacuous, because every
  pool move except Bonemerang is single-hit. R10 reasoned from a bucket's NAME and never traced its
  caller: the bucket is not merely unambiguous, it is unemittable.
- **⚠ A new standing rule, and it is the R26 rule's missing half.** *A whole-pool "0 of 220" is
  side-independent; a per-species one is not.* §8 has read since C138 as though every movepool
  check were suspect for a `target: normal` move. R26 was wrong because it scoped the check to TWO
  SPECIES, not because it used a movepool: if no species in the pool has the move, no side can use
  it and no side can be hit by it. Eighteen §4 rows are whole-pool absences and are safe for that
  reason, and saying so is what let this pass spend its attention on the eight that are not.
- **⚠ A second standing rule.** *A generation-time census bounds what a team STARTS with. A claim
  about what can exist mid-battle needs the runtime clause too.* R6/R10/R27, above.
- **⚠ And a third.** *Cite which guard.* R9 said "the Liquid Ooze guard inside
  `residual_heal_cause`" for a conjunct whose text occurs twice in the same file, and collapsed two
  distinct guards — one dead in this format, one live and pinned in both directions — into one
  sentence. The re-adjudication resolves it with `_anchor_after` relative to the enclosing
  function, because an occurrence index would be as brittle as a literal.
- **§8's own row count was stale.** This section read **81** while the table held **82** and §1's
  header sentence had already been machine-checked at 82 by C153. A fourth instance of "a
  correction applied to data does not propagate to the prose describing it", in the section that
  records the third. Fixed, and now pinned by
  `test_section_eight_states_the_derived_row_count` rather than by care.
- **Mechanized, and every demonstration is re-derived from source on every run** — that is the
  point of the module, not a feature of it. `_anchor` / `_anchor_after` / `_raise_line` are
  IMPORTED from C153's census rather than copied, so a stale anchor anywhere in the traced set is
  one loud failure. Battery: **21 mutations applied, 21 caught**, enumerated in the pin's
  docstring. No §3 row was added or removed, so §3 stays at **82** and §4 still considers 27
  candidates.
- ⚠ **REVIEW ROUND (2026-08-09). Three of this pass's own load-bearing sentences were the defect it
  was removing, and a fourth rule it wrote down was evaded three ways.** Recorded here because a
  pass whose subject is untraced claims may not quietly repair its own.
  - **R10's correction made R10's mistake in the sentence that names it.** It opened *"this cell
    reasoned from its NAME without tracing its caller"* and then asserted, untraced, that
    `heal_subcase` is reached only through `ambiguous_unrenderable_slug_with_protect`. There are two
    routes; the predicate at the head of the same Sleep Talk block is the other. The conclusion
    survives — both roots are `render_move_phase` and only one emits a key — and the caller graph is
    now derived by reverse reachability and pinned instead of asserted.
  - **The artifact was placed on a hazard nobody measured, with a control that could not fail.** It
    was written to `tests/data/` because its reason names beside pool counts would supposedly read
    to `tests/test_never_fired_counter_census.py` as four counters firing. Measured, that census
    reports `Ran 22 tests … OK` on it: the names occur only in prose, which its matcher excludes in
    terms. The placement bought nothing and cost the guard that census's own header warns about by
    name. It is in `reports/artifacts/` now, and the control feeds the matcher a counter-keyed copy
    and requires it to FIRE.
  - **The generator's docstring stated the corrections tally twice, as SEVEN and as TEN, against
    THIRTEEN.** The fifth and sixth instances of this section's own subject, inside the generator
    that produces the number. Derived now, and pinned against the marked cells.
  - **The phrase guard was evaded by markdown emphasis, U+00AD and U+200B — in a markdown file** —
    and by a re-assertion smuggled inside the correction's own cell, which the quoting rule admits
    and must. The normaliser folds all three, and an EXACT per-phrase occurrence inventory catches
    the fourth. Each of the guard's four normalisations was added after something got past the
    previous three: the class was never fixed, only the instance named.
  - ⚠ **ROUND TWO: fixing round one weakened the guard on the owner ratification and the burn.**
    Bumping this pass's own `Ran N tests` guard used an UNBOUNDED string replace over the workflow,
    and two steps carried `Ran 25 tests`. The other was `tests/test_final_holdout_guard.py`, whose
    step gates `OWNER_RATIFIED`, `BURNED_FINAL_HOLDOUT` and the `19,200,000`-`19,200,259` burn and
    whose module this change does not touch; it became `Ran 31 tests` for a suite that can only
    print 25, so it stopped failing closed. Re-derived by AST (14 + 11 = 25, no `subTest`, no
    `load_tests`) and restored. **The class fix is the durable part**: every `Ran N tests` guard in
    the workflow is now re-derived from its module's AST, because the guard had no guard. R10's cell
    in this section also still read "one non-test path" against the derived two, and is corrected
    and pinned to the graph.
  - Smaller, all corrected in place: `UNREACHABLE_TRACED` was documented as "cannot fire for any
    caller", which is false for R1, R23 and R24 — every row now carries an explicit `ALL_CALLERS`
    or `RANDBATS_POPULATION` foreclosure scope rather than the word being softened for all 26;
    R22's set equality was called machine-checked while the pin asserted six memberships, defeated
    by adding `Choices::TACKLE`; `_weather_fields` returns `-1` from two places, not only under
    `weatherFromAbility`; `SCENARIO_WEATHER_IDS` belongs to `scenario_studio`, not "the bridge" — a
    cite-which-module slip inside the correction that introduces *cite which guard*; `ceiling =
    defender.hp` holds at 2 of `residual_disjoint_bands`'s 4 call sites, which are the two the
    argument uses; and the fingerprint move was credited to #1202, which touches no `.rs` file, when
    it is #1197.
