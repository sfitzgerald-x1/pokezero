# C143 — `19200244/115` is G8, plus one renderer defect that hid it behind a new class string

`19200244/115` (`component_mismatch:heal|itemleftovers`, final holdout, C141) is **not a new
shape.** It is a second confirmed instance of ledger **G8** — the collapsed lethal arm mis-pricing a
roll-dependent Leech Seed drain — on a second window with a different HP configuration. What made it
*look* new is a **separate renderer defect** that relabels the drain, flipping the class string from
`component_magnitude:heal` (c140's) to `component_mismatch:heal|itemleftovers`.

Two defects, superimposed. Both are measured below, and the measurement separates them **per path**:

* **Under the collapsed path** (what the sweep runs) neither is the whole cause — correcting the label
  alone leaves the row divergent, and so does correcting the representative alone.
* **Under the enumeration oracle** (what C137 made the program's oracle) the engine emits the observed
  row exactly, at 0.2189 % mass, and **the label is the only thing failing** — so gating the renderer
  defect makes the row **match**.

The first revision of this report measured only the collapsed path and concluded from it that a
renderer gate "closes nothing". That was false; §6 carries the correction and the measurement.

> **The brief I was given framed this as a novel heal-attribution gap and asked for a novelty
> adjudication. It is not novel.** The correction came from independent review mid-diagnosis, and it
> was right: the mechanism is c140's, and the numbers below are the same arithmetic with the victim's
> maxhp/pre-move HP/Leftovers at **407/157/25** in place of c140's **235/123/14**. Where a cell
> elsewhere writes `407/11/25` it is quoting the *post*-move HP, 157 − 146 = 11; both are correct and
> neither is interesting without the label, which is why they are labelled here.

**Nothing in this report is a fix.** No engine or crate change ships. The only non-documentation
addition is `scripts/c143_heal_attribution_probe.py` and its artifact.

## 0. Provenance, and one honest gap in it

| | |
|---|---|
| replay build | fingerprint `c72e6523d8de6f64090c9d9160a493ce5253662a65debc7f4229e88d9bb23761`, **71 patches**, built by `scripts/build_search_crate_engine.sh` with `exit=0` captured directly |
| the row's own sweep | `c141_final_holdout_sweep.json`, engine fingerprint `44ee1430…` — **a different build.** `main` has since taken #1156/#1157/#1158, two of which touch the heal family in `events.rs` |
| the control that makes the replay usable anyway | `reread_row` on this build reproduces the recorded verdict and **all nine `branch_misses` byte-for-byte** (`matrix.control.misses_identical_to_recorded: true`). The replay is on a different fingerprint but is behaviourally identical *on this row* |
| every number below | `reports/artifacts/c143_heal_attribution_probe.json` (collapsed path) and `reports/artifacts/c143_heal_attribution_enumerated.json` (`POKEZERO_ENUMERATE_ROLLS=1`). **Two files because the flag is a Rust `OnceLock`** — one process is one engine, so the two paths cannot be measured in one run. Both are byte-reproducible (`cmp`-verified across re-runs) |
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
* **The burn cannot beat the drain to the kill.** `data/mods/gen4/conditions.ts` puts `brn` at
  `onResidualOrder 10 / subOrder 6`, *after* `leechseed`'s 5 — so within Wigglytuff's own block the
  drain resolves first, which is why the protocol has Leech Seed and not the burn land the faint. It
  also means the burn is not a competing explanation for the 36.
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

**The agreement in B is confined to those two heal lines, and the table above is deliberately narrow
about it.** The engine's full B protocol still differs from Showdown's in two ways this report does not
pursue: the drain damage line omits `|[of] p1a: Snorlax`, and `|faint|` precedes the mirror heal
instead of following it. Neither is a component the strict matcher compares, so neither changes a
verdict — but "the engine renders B byte-identically to Showdown" would be false, and an earlier draft
of the PR body said it.

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
sub-case of G8 from c140's**, and the path is **traced statically, in three steps that each name a
value**:

1. `compare_health_with_damage_multiples(159, 157)` in f32 returns **`(145, 2)`** — mean of the 14
   rolls below Wigglytuff's 157 HP, and the 2 move-KO rolls at or above it.
2. `residual_phase_final_hp`, bisected, gives `low = 76`, so the residual ladder on this boundary is
   the **single threshold 82**.
3. `residual_disjoint_bands`' guard `min_roll < threshold` is therefore **`135 < 82` → false**,
   `applicable_len == 0`, the function returns `None`, and `survive_representative` keeps **145**.
   The function's own doc comment names this behaviour: *"A threshold at or below `min_roll` is
   SKIPPED rather than clamped: no roll lies below it, so it has no survive side."*

c140's row took the other branch: its threshold **was** inside its fan, so the arm was priced *at the
threshold* (108, a fan member). Here the threshold is skipped and the arm is mean-priced.

**And 145 is not in its own fan, so its mirror 37 is not achievable by any roll.** Measured through
the unmodified shipped `evaluate_boundary_strict`, rows = representative, columns = the 14 rolls
Showdown can throw — eight representatives × 14 columns = 112 cells:

| representative | mirror | drain overflows the seeder's max HP | renderer as shipped | renderer with the drain rendered `[silent]` |
|---|---|---|---|---|
| **145 (shipping)** | 37 — unachievable | n/a | 0 of 14 | **0 of 14** |
| 141, 144, 146, 147, 155 | 41, 38, 36, 35, 27 | no | 0 of 14 | **1 of 14 — its own column, each** |
| 135, 136 | 47, 46 | **yes** | 0 of 14 | **2 of 14 — columns 135 *and* 136, both** |

Control, non-vacuous: the repricer at 145 with the shipping label reproduces the recorded misses
byte-for-byte, having touched 3 arms and rewritten 2 p1 heal lines. *(An earlier run of this matrix
reported all-zeros because it keyed the rewrite on `Moltres` while the replay path renders side one's
active as `unknown5`; the rewrite counters exist so that failure cannot recur silently.)*

**The last row of that table is a correction to my own first draft, and it came out of fixing the
probe.** The repricer originally did not clamp the mirror at max HP and synthesised an impossible
`270/268`. With the clamp in place, representatives 135 and 136 — the only two whose mirror overflows
Moltres' 268 from 223 — each price **two** columns, not one, because a heal saturating at max HP is
relabelled `heal_to_full` by `damage_component_events`, which moves it out of the exact bucket into
the tolerant roll-scaled one (ledger H8's `[0.92·eng − 1, 1.09·eng + 1]` window). So the exact
diagonal holds only where the mirror does **not** saturate.

Read the columns together:

* **The renderer defect alone is sufficient to keep this row divergent under the collapsed path** —
  0 of 14 at every one of the **eight** representatives measured (135, 136, 141, 144, 145, 146, 147,
  155). It is not cosmetic. Not tested at the six band values not in that set.
* **The representative defect alone is sufficient too** — 0 of 14 with the label fixed.
* c140 §6a's bound ("any fixed representative prices exactly one") needs **two** scope conditions,
  both of which its own matrix satisfied without stating: the representative must be **inside the
  fan**, and its mirror must **not saturate** the seeder's max HP. All seven of c140's band values
  were fan members and none saturated. This instance exhibits both excluded cases: a non-fan
  representative prices **zero**, and a saturating one prices **two**. The shipping engine is in the
  first, so it is **below** c140's bound rather than at it. That is a scoping correction to a merged
  claim, not a refutation of it.
* **A third consequence, which the first draft of this report failed to draw.** Because 145 prices
  **zero** rather than one, re-pricing this arm to any non-saturating fan member is a **strict gain
  here — 0 → 1** — not the even permutation c140 §6a(ii) analysed, where moving 108 → 109 closed one
  column and opened another at equal `1/16` mass. c140's refusal of re-pricing rested on that
  exchange being a wash; on an off-fan representative the wash argument does not apply. It does not
  change the refusal, for c140's *other* two reasons, which survive intact: it would be **fitted to
  the sample** (c140 §6a(iii), and this row is `n = 1` on a spent holdout), and the engine's own
  `floor(mean(band))` convention is what *produces* 145, so "use a fan member instead" is a new rule,
  not the existing one applied. What it does change is the **characterisation**: "any representative
  is as good as any other" is false, and an engine change that merely snapped the mean-priced
  representative to the nearest non-saturating fan member would be a rule change with a real, if
  unmeasured, upside. That is a candidate for the ledger, not for this PR.

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

### 4a. The 34.92 % of mass nothing above accounts for — a *third* roll collapse

Five of the nine recorded misses are discussed above (49.03, 10.74, 1.33, 3.27, 0.72 %). The other
four — **3.75, 9.23, 2.02 and 19.92 %, totalling 34.92 %** — fail on p1's *roll-scaled* list, before
any heal is compared, and the first revision of this report left them unmentioned. They sit on the
**other side of the same boundary**:

* **15.00 % exactly** (3.75 + 9.23 + 2.02) are arms where the engine's **Fire Blast missed** against a
  Showdown hit — `observed=[('', -49)] engine=[]`. Fire Blast is 85 % accurate and `100 − 85 = 15`, so
  this is the accuracy split, not a defect. It is the same shape as ledger **G3**: no rendering or
  pricing change can make a miss branch reproduce a hit.
* **19.92 %** is the Fire-Blast-hit, Moltres-fully-paralysed arm, and it fails because the engine's
  **Fire Blast representative is 45 against the observed 49**.

That last one is a **third roll collapse on this single boundary**, and it is the same mechanism as
§3 one field over. Fire Blast into Moltres kills nothing, so no residual threshold applies at all and
the entire 16-roll fan collapses to its integer mean: max 49, rolls
`[41,41,42,42,43,44,44,45,45,46,46,47,48,48,49,49]`, sum 720, `720 // 16 = 45`. **The observed 49 is
the fan's top value** and a member. So the boundary carries *two* independent mean-priced collapses —
Flamethrower at 145 and Fire Blast at 45 — and either alone would block an arm.

This does not change the diagnosis: a boundary needs one matching arm, and §6 shows the enumerated
oracle supplies one. It does mean the collapsed path's failure here is **over-determined**, which is
worth stating rather than leaving 35 % of the mass silently unexplained.

## 5. Adjudication

| | |
|---|---|
| **magnitude, 36 vs 37** | **ENGINE gap — ledger G8, second confirmed instance.** Not a limit: the observed roll is in the engine's own fan and c140 measured the enumeration oracle accepting it. Both values are correct given each side's own HP; the engine's arm sits at a damage value Showdown cannot throw |
| **attribution, bare `heal` vs `itemleftovers`** | **RENDERER (harness) defect**, in `ResidualPlan::build` in `rust/pokezero-search/src/events.rs` — the Leftovers heal slot, over-booked on a residual phase truncated by battle end. Filed as **G33b**, adjacent to G33's over-booked *drain* slot. **Under the enumeration oracle it is the row's only remaining failure, and gating it makes the row match** (§6) |
| **`engine_only=[]` arm** | not a defect — the complementary move-KO arm |
| **novelty** | **none.** G8 for the magnitude; a new row only for the renderer half |

## 6. The G33b gate closes nothing under the collapsed path and **closes this row under enumeration**

> ### Correction — the first revision of this section measured the wrong path, and its conclusion was false
>
> It refused the gate on the grounds that it has "a measured upside of zero rows". That is true of
> the **collapsed** path and only of it. G8's own merged disposition is *"closed by enumeration,
> retained under the collapsed path"*, and c140 §6 measured its row flag-off **versus flag-on**. I
> measured only flag-off, then generalised. Review ran the flag and the gate closes the row.
>
> Worse, §6 itself named the trigger for revisiting — *"a boundary of this shape where the mirror
> magnitude already agrees and the label is the only thing failing"* — and that condition is
> satisfied **by this very row** under enumeration. I wrote the falsifier and did not run it.

The gate's predicate: **do not book a side's Leftovers slot when the residual phase will be truncated
by the opposing active's faint** — read off the render, since an arm that ends the battle carries a
`|faint|p2a:` and no `|turn|` line.

Both paths, same build, same row, flag off versus on (the flag is a Rust `OnceLock`, so one process
is one engine and these are two processes):

| path | renderer | branches | verdict | misses |
|---|---|---|---|---|
| collapsed | as shipped | 9 | diverged | 9 |
| collapsed | modelled G33b gate | 9 | **diverged** | — 0 of 14 at every representative, §3 |
| **enumerated** | as shipped | 416 | diverged | **12** |
| **enumerated** | **modelled G33b gate** | 416 | **matched** | **0** |

Under enumeration the oracle **emits the observed row exactly** and fails on the label alone:

```
pct=0.22: p1 attributed components differ: observed_only=[('heal', 36)] engine_only=[('itemleftovers', 36)]
```

Right magnitude, wrong label. Exactly **1 arm** of the 416 reproduces the full observed HP trace —
Fire Blast 49 *and* Flamethrower 146 — at **0.2189 %** mass; 26 arms totalling 4.3945 % reproduce the
Flamethrower half, summed over the paralysis and crit splits on the other side of the field.

**Soundness control on the gate**, because a relabelling can only ever *widen* what matches and
"nothing opened" therefore proves nothing about it: across all 416 arms the gate relabelled **350**
heals, every delta in **[27, 47]** — the exact range of the 14 achievable mirrors — and **none equal
to 16**, which is `268//16`, a genuine Moltres Leftovers tick. So the gate silenced no real tick. It
does not show the gate is safe repo-wide; it shows it is not obviously over-broad on this boundary.

**Still no fix ships in this PR**, and the reason is now different and narrower than the first
revision's:

* The gate is a **crate change** to `ResidualPlan::build`, not a rewrite of rendered output. What is
  measured above is a *model* of it applied to the renderer's output — the method c140 §6a used, and
  faithful because the strict path compares rendered components only, but not the same thing as the
  built change.
* Its measured upside is **one row under the enumeration oracle**, and C137 already made enumeration
  the oracle while H18 records that enumeration cannot be used in search (2.38 ms → 8,881.8 ms per
  decision). So the closure is real and lands on the path the program certifies against.
* What is **still unmeasured** is the only thing that licenses shipping: both windows swept with the
  built gate. C133 §7's discipline applies — the gate built in the crate, a prediction registered with
  a "nothing opened" falsifier, dev (19,000,000) and validation-holdout (19,100,000) swept, and
  **never** the final holdout, which this row has already spent.

**Recommendation, changed from the first revision: this gate is worth building and sweeping.** It has
a measured closure on the certifying oracle, an exact and cheap predicate, and a soundness control
that did not fire. That is a materially stronger case than "zero upside", which is what I reported.

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
* **The renderer defect being cosmetic.** 0 of 14 columns at each of the eight representatives
  measured, with it in place — and under enumeration it is the row's *sole* remaining failure (§6).
* **The representative defect being sufficient on its own to explain the class string.** With the
  label fixed the class becomes c140's, which is the point.
* **Fire Blast / Flamethrower / burn / a burn interaction being required.** The generated
  reproduction uses none of them.
* **A limit.** c140 §6 recorded the falsifier firing for this family; nothing here re-opens it, and §6
  above adds a second firing on this row.
* **Which code path prices the arm at 145.** Traced statically in §3 — `(159, 157) → (145, 2)`,
  `low = 76` so the ladder is the single threshold 82, and `135 < 82` is false so
  `residual_disjoint_bands` returns `None`. The first revision hedged this as "inferred, not traced";
  the hedge is withdrawn.
* **"A G33b gate closes nothing."** False as the first revision stated it. It closes nothing under the
  collapsed path and **closes this row under enumeration** (§6).
* **"Any fan-member representative is as good as any other here."** False: the two whose mirror
  saturates max HP price two columns, the rest price one (§3).

**Left uncertain, and stated as such:**

* Whether the G33b gate is safe **repo-wide**. The soundness control (350 relabels, all deltas in
  [27, 47], none equal to 16) shows it silenced no genuine Leftovers tick *on this boundary*, and a
  relabelling can only widen what matches, so "nothing opened" cannot vindicate it. Only both windows
  swept with the built crate change will.
* Whether **snapping the mean-priced representative to the nearest non-saturating fan member** is a net
  gain. It is a strict 0 → 1 gain *on this boundary* (§3), which is new, but no rule was written, no
  build was made and no window was swept.
* The replay is on `c72e6523…`, not the row's own `44ee1430…`. The nine-miss identity is strong
  evidence they agree on this row and says nothing about any other.
* The 14-roll band and the 7-roll band are two instances. Nothing here bounds how band width
  distributes over the pool, so "1 of 14" and "1 of 7" are two observations, not a rate.
* Six of the fourteen band values were not used as representatives, and the two saturating ones are the
  only members of their class that were.
