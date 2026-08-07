# C143 — `19200244/115` is G8, plus one renderer defect that hid it behind a new class string

`19200244/115` (`component_mismatch:heal|itemleftovers`, final holdout, C141) is **not a new
shape.** It is a second confirmed instance of ledger **G8** — the collapsed lethal arm mis-pricing a
roll-dependent Leech Seed drain — on a second window with a different HP configuration. What made it
*look* new is a **separate renderer defect** that relabels the drain, flipping the class string from
`component_magnitude:heal` (c140's) to `component_mismatch:heal|itemleftovers`.

Two defects, superimposed. Both are measured below, and the measurement separates them: with the
renderer's label corrected the row is still divergent, and with the representative corrected it is
still divergent, so **neither is the whole cause and the renderer one is not merely cosmetic.**

> **The brief I was given framed this as a novel heal-attribution gap and asked for a novelty
> adjudication. It is not novel.** The correction came from independent review mid-diagnosis, and it
> was right: the mechanism is c140's, and the numbers below are the same arithmetic with 407/157/25
> in place of 235/123/14.

**Nothing in this report is a fix.** No engine or crate change ships. The only non-documentation
addition is `scripts/c143_heal_attribution_probe.py` and its artifact.

## 0. Provenance, and one honest gap in it

| | |
|---|---|
| replay build | fingerprint `c72e6523d8de6f64090c9d9160a493ce5253662a65debc7f4229e88d9bb23761`, **71 patches**, built by `scripts/build_search_crate_engine.sh` with `exit=0` captured directly |
| the row's own sweep | `c141_final_holdout_sweep.json`, engine fingerprint `44ee1430…` — **a different build.** `main` has since taken #1156/#1157/#1158, two of which touch the heal family in `events.rs` |
| the control that makes the replay usable anyway | `reread_row` on this build reproduces the recorded verdict and **all nine `branch_misses` byte-for-byte** (`matrix.control.misses_identical_to_recorded: true`). The replay is on a different fingerprint but is behaviourally identical *on this row* |
| every number below | `reports/artifacts/c143_heal_attribution_probe.json` |
| the artifact the row lives in | **not on `main`** — it lands with the C141 PR (`aa2f2d40`, branch `report-final-holdout-sweep`). The probe takes `--row` for that reason |

No sweep was run at or above seed 19,200,000. The row was read from the committed artifact and
replayed against the local engine; every Showdown call in this report is a **generated** gen3 Custom
Game fixture on fixture seed 7717.

**Showdown is cited first-hand here, unlike in c131 §4 and c140 §5.** Both of those flagged their
Showdown citations as second-hand on the grounds that Showdown is not vendored in this repository.
That is true, but `pokezero.local_showdown.default_showdown_root()` resolves to a checkout that *is*
on this machine — `~/workspace/pokerena/vendor/pokemon-showdown` — and it is the checkout the harness
actually drives. Every `sim/` and `data/` line below was opened in it. This does not overturn c131 or
c140; it upgrades the provenance of a claim they both had to hedge.

## 1. Why the seeder's own Leftovers tick is missing from the protocol

The recorded protocol is short one line that gen3 ought to contain, and the whole renderer defect
follows from that absence:

```
|-heal|p2a: Wigglytuff|36/407 brn|[from] item: Leftovers    ← +25 = 407//16
|-damage|p2a: Wigglytuff|0 fnt|[from] Leech Seed|[of] p1a: Moltres   ← −36, CAPPED by the 36 present
|-heal|p1a: Moltres|255/268 par|[silent]                    ← +36, the mirror. Bare, by design
|faint|p2a: Wigglytuff
|win|PokeZero p1
```

Moltres holds Leftovers and sits at 219/268, so a `+16` tick is owed and never appears; `observed.p1_hp`
is 255 = 219 + 36, independently confirming its absence. Three facts from source explain it:

* **Gen 3 has no separate "items phase".** Gen 3 inherits gen 4, and `data/mods/gen4/items.ts:231`
  re-points `leftovers` to `onResidualOrder: 10, onResidualSubOrder: 4` while
  `data/mods/gen4/moves.ts:711-716` puts the `leechseed` condition at `10 / subOrder 5`. One
  `order`-10 bucket, speed-sorted across Pokémon, Leftovers-then-Leech-Seed within a Pokémon. (The
  base `data/items.ts` value is 5 and the base `data/moves.ts` value is 8, which is why reading
  either alone predicts the wrong order.)
* **Moltres is the slower of the two.** Wigglytuff 135 against Moltres 185 quartered by paralysis, so
  Wigglytuff's whole block resolves first.
* **The bucket stops at battle end.** `sim/battle.ts:565-566`, inside the residual handler loop:
  `this.faintMessages(); if (this.ended) return;`. Wigglytuff is p2's **last** living Pokémon (the
  engine state has all five others at 0 HP), so the drain that kills it ends the battle and Moltres'
  slot is never reached.

### 1a. Measured on generated boundaries, with the trigger isolated to one bit

Three gen3 Custom Game fixtures, no randbats seed, **no Fire Blast, no Flamethrower, no burn and no
paralysis** — a Snorlax or Suicune seeding a Blissey with Seismic Toss and Splash. Predictions were
registered before the run and all three hold:

| variant | seeder | victim is opponent's last mon | predicted | **measured** |
|---|---|---|---|---|
| A | Snorlax, spe 96 (slower) | yes | seeder Leftovers absent | **absent** |
| B | Snorlax, spe 96 (slower) | no | present, after the mirror | **present, after the mirror** |
| C | Suicune, spe 186 (faster) | yes | present, before the drain | **present, before the drain** |

A and B differ in exactly **one bit** — whether a spare Misdreavus sits behind the victim — and A
reproduces the holdout row's protocol shape line for line. C shows a faster seeder is immune, which
is what makes the paralysis load-bearing in the holdout row and *only* in that indirect way. B
reproduces c140's shape, where the seeder's tick does appear after the faint.

So Fire Blast, Flamethrower and the burn are **incidental**. The burn does one thing: it makes side
two's *damage* plan over-book as well, which is why the engine renders the residual kill as
`[from] brn` in the holdout row and `[from] Leech Seed` in variant A. Both are absorbed by the
`capped_lethal` promotion (ledger H7), so that half changes no verdict.

## 2. Defect 1 — the renderer over-books the seeder's Leftovers slot

`ResidualPlan::build` (`rust/pokezero-search/src/events.rs`) books a heal slot for every side whose
active holds Leftovers, deliberately and with a measured justification (the `NOTE:` at
`events.rs:5123-5133`: adding an `hp < maxhp` guard costs 5 rows). It then disables the whole side if
the booked count and the emitted count disagree, and every label on that side drops to
`residual_heal_cause` — a **constant function of state** which, since C131 change 3, tests Leftovers
*before* the drain. So the bare silent drain comes back tagged `[from] item: Leftovers`.

An `hp < maxhp` guard would not fix this instance. Moltres ends at 260/268, not full: the slot goes
unfilled because the **residual phase was truncated by battle end**, which no HP predicate can see.

**Measured, on the same generated position as §1a, single-variable:**

| variant | Showdown | engine renderer |
|---|---|---|
| A (victim is last) | `\|-heal\|p1a: Snorlax\|407/461\|[silent]` | `\|-heal\|p1a: Snorlax\|407/461\|[from] item: Leftovers` ✗ |
| B (victim not last) | `\|-heal\|p1a: Snorlax\|407/461\|[silent]` then `\|435/461\|[from] item: Leftovers` | **identical** ✓ |

The **HP arithmetic is identical to Showdown's in both** (361 → 407 in A; 361 → 407 → 435 in B), and
the engine agrees with Showdown that the seeder's tick does not fire on battle end. Only the
attribution is wrong, and only when the phase is truncated — exactly C131's finding one surface over.

The holdout row carries its own internal control for this. On the same boundary and the same build,
the **1.33 %** arm — where Wigglytuff survives, so side one emits *two* heals and the plan
reconciles — renders the drain correctly:

```
|-heal|p1a: Moltres|227/268|[silent]                  ← the drain, correctly bare
|-heal|p1a: Moltres|243/268|[from] item: Leftovers     ← the tick, correctly tagged
```

Nothing distinguishes it from the 49.03 % arm except whether the side's booked count was filled.

## 3. Defect 2 — G8, and this instance is *worse* than c140's bound

Flamethrower into this Wigglytuff, from `poke_engine.calculate_damage`: max 159, fan

```
[135, 136, 138, 139, 141, 143, 144, 146, 147, 149, 151, 152, 154, 155, 157, 159]
```

**146 is a member** (the observed roll), which settles the bucket as engine-gap by
`scripts/family_bucket_audit.py`'s own definition. `157` and `159` are move-KOs; every one of the
other **14** rolls survives the move and dies to the residual, and the mirror
`min(407//8, hp_after + 407//16)` is **injective over all 14** (47, 46, 44, 43, 41, 39, 38, **36**,
35, 33, 31, 30, 28, 27).

The engine emits **145**, and `sum(band) // len(band) == 145` exactly — the survive representative,
`compare_health_with_damage_multiples`'s integer mean, not a threshold. This is a **different
sub-case of G8 from c140's.** c140's row went through `residual_disjoint_bands` and was priced *at
the threshold* (108, a fan member). Here every residual threshold lies below the fan minimum, so the
`min_roll < threshold` guard cannot pass, `residual_disjoint_bands` yields nothing, and the single
non-KO arm is mean-priced. *(The code-path attribution is inferred from the emitted value matching
the band mean exactly and matching no threshold; I did not trace it in a debugger.)*

**And 145 is not in its own fan, so its mirror 37 is not achievable by any roll.** Measured through
the unmodified shipped `evaluate_boundary_strict`, rows = representative, columns = the 14 rolls
Showdown can throw:

| representative | renderer as shipped | renderer with the drain rendered `[silent]` |
|---|---|---|
| **145 (shipping)** | 0 of 14 | **0 of 14** |
| 135, 141, 144, 146, 147, 155 (fan members) | 0 of 14 | **1 of 14 — its own column, each** |

Seven representatives, all 14 columns each: 98 cells. Rows are the engine's representative, columns
the roll Showdown threw; the diagonal under the fixed renderer is definitional, as c140 §6a notes.

Control, non-vacuous: the repricer at 145 with the shipping label reproduces the recorded misses
byte-for-byte, having touched 3 arms and rewritten 2 p1 heal lines. *(An earlier run of this matrix
reported all-zeros because it keyed the rewrite on `Moltres` while the replay path renders side one's
active as `unknown5`; the rewrite counters exist so that failure cannot recur silently.)*

Read the two columns together:

* **The renderer defect alone is sufficient to keep this row divergent** — 0 of 14 at every one of
  the **seven** representatives measured, which span the band (135, 141, 144, 145, 146, 147, 155).
  It is not cosmetic. Not tested at the seven band values not in that set.
* **The representative defect alone is sufficient too** — 0 of 14 with the label fixed.
* c140 §6a's bound ("any fixed representative prices exactly one") is sound **for representatives
  inside the fan**, which is the only kind its matrix tested — all seven of its band values were fan
  members. This instance exhibits the case it did not: a **non-fan representative prices zero**, and
  the shipping engine is in that case here. So the shipping engine is **below** c140's bound, not at
  it. That is a scoping correction to a merged claim, not a refutation of it.

## 4. The second miss (`engine_only=[]`, 10.74 %) is not a third defect

It is the **move-KO arm**: the engine's other Flamethrower outcome kills Wigglytuff with the move,
so there is no residual phase, no drain and no p1 heal at all.

```
|-damage|p2a: Wigglytuff|0 fnt
|faint|p2a: Wigglytuff
|
|upkeep
```

The 0.72 % miss is its crit twin. A boundary needs only one matching arm, so a non-matching
complementary arm of the same KO/non-KO partition is expected and carries no information. Together
with the 49.03 % arm it also confirms the partition has exactly two Flamethrower outcomes here.

## 5. Adjudication

| | |
|---|---|
| **magnitude, 36 vs 37** | **ENGINE gap — ledger G8, second confirmed instance.** Not a limit: the observed roll is in the engine's own fan and c140 measured the enumeration oracle accepting it. Both values are correct given each side's own HP; the engine's arm sits at a damage value Showdown cannot throw |
| **attribution, bare `heal` vs `itemleftovers`** | **RENDERER (harness) defect**, in `ResidualPlan::build` in `rust/pokezero-search/src/events.rs` — the Leftovers heal slot, over-booked on a residual phase truncated by battle end. Filed as **G33b**, adjacent to G33's over-booked *drain* slot |
| **`engine_only=[]` arm** | not a defect — the complementary move-KO arm |
| **novelty** | **none.** G8 for the magnitude; a new row only for the renderer half |

## 6. No fix ships, and the reason is measured rather than argued

The renderer fix is small and its predicate is exact: **do not book a side's Leftovers slot when the
residual phase will be truncated by battle end** — i.e. when the opposing side's active is the last
living Pokémon on that side and a booked residual will kill it. It is tempting.

**It closes nothing on this row: 0 of 14 before, 0 of 14 after.** Its only effect here is to move the
class string from `component_mismatch:heal|itemleftovers` to `component_magnitude:heal`, which is
where the row belonged all along. A change with a measured upside of zero rows and an unmeasured
downside is exactly the trade C133 §7 and c140 §7 refuse, so no prediction is registered and no
sweep was run.

What is **unmeasured** and would justify revisiting it: whether any *other* boundary of this shape
exists where the representative *is* a fan member, so the mirror magnitude already agrees and the
label is the only thing failing. On such a row the gate would close it outright. The measurement that
would settle it is the one C133 §7 prescribes — the gate built, a prediction registered with a
"nothing opened" falsifier, and both the dev (19,000,000) and validation-holdout (19,100,000) windows
swept. This report does not do it, and the reachability question is stated, not answered.

## 7. What was ruled out, and by what

Ruled out **by measurement**:

* **A Sitrus or other berry.** No berry appears in the protocol at all; the heal is the Leech Seed
  mirror, whose magnitude equals the capped residual damage on the other side in every arm.
* **Wish, Rain Dish, a drain move's mirror.** None is present in the protocol or in either side's
  moveset; the `[silent]` tag is reachable only from `leechseed` and `rest` in `sim/battle.ts:2293-2296`.
* **"A Leftovers tick Showdown rendered silently."** Showdown renders Leftovers as
  `[from] item: Leftovers` — it does so for Wigglytuff on this very line — and 268//16 = 16 ≠ 36.
* **The seeder's missing tick being a harness world-construction error.** It is real Showdown
  behaviour, reproduced on three generated boundaries, and the engine reproduces it too.
* **An `hp < maxhp` guard as the renderer fix.** Moltres ends at 260/268; the guard cannot see the
  truncation.
* **The renderer defect being cosmetic.** 0 of 14 columns at each of the seven representatives
  measured, with it in place.
* **The representative defect being sufficient on its own to explain the class string.** With the
  label fixed the class becomes c140's, which is the point.
* **Fire Blast / Flamethrower / burn / a burn interaction being required.** The generated
  reproduction uses none of them.
* **A limit.** c140 §6 recorded the falsifier firing for this family; nothing here re-opens it.

**Left uncertain, and stated as such:**

* Which engine code path prices the arm at 145 is **inferred** from the emitted value equalling the
  band mean exactly, not traced. If `residual_disjoint_bands` does fire here with a threshold I have
  not accounted for, the §3 sub-case story changes while the 0-of-14 measurement does not.
* Whether the G33b gate would close any row anywhere. Unmeasured — see §6.
* The replay is on `c72e6523…`, not the row's own `44ee1430…`. The nine-miss identity is strong
  evidence they agree on this row and says nothing about any other.
* The 14-row band and the 7-row band are two instances. Nothing here bounds how the band width
  distributes over the pool, so "1 of 14" and "1 of 7" are two observations, not a rate.
