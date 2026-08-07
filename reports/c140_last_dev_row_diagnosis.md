# C140 — the last dev row, `19000191/63`: an engine gap that enumeration closes and the collapsed path structurally cannot

After #1144, #1148 and #1152, `19000191/63` (`component_magnitude:heal`) is the only divergent
row on either window: dev 1/15,503, holdout 0/15,579
(`reports/artifacts/c138_collapsefix_merged_{dev,holdout}_sweep.json`).

**Disposition: ENGINE gap. Not a limit — the falsifier fires, measured.** The remedy that works is
enumeration, which was measured to close it on a build made for this report. The collapsed
representative path cannot close it without becoming per-roll enumeration inside the lethal band,
and the reason is a measured injectivity, not an argument. The recommended program disposition is
therefore **"closed by enumeration, retained under the collapsed path"**.

No engine or harness change ships with this report.

## 0. Provenance

| | |
|---|---|
| shipping build | fingerprint `e65044f6c8c3917acb9aae155898d6b4b78bcaf154615ce688f1ae77ed86c044`, 70 patches, source `eaef463a` |
| sweep the row comes from | `c138_collapsefix_merged_dev_sweep.json`, fingerprint `e65044f6…` — **the same fingerprint**, so the replay is on the engine that produced the row |
| enumeration build | fingerprint `e3f442a0183f370f382ecd596d90d6a3150a33ef8a41fa329ddda7e04c767a8f`, 71 patches, `POKEZERO_ENUMERATE_ROLLS=1` |
| every number below | `reports/artifacts/c140_last_dev_row_probe.json` |

Both builds were produced by `scripts/build_search_crate_engine.sh` with the exit code captured
directly (`exit=0` on both). Replays go through `cert_sweep_reread.reread_row`, which calls the
shipped `evaluate_boundary_strict` — nothing here reimplements the comparator.

## 1. The boundary, component by component

Seed 19000191, step 63, turn 59. Choices `p1=thunderbolt`, `p2=hiddenpowergrass70`.
Raichu (side_one) 123/235, Leftovers, no status, **Leech Seeded**. Tangela (side_two) 277/277,
Leftovers, no status.

What Showdown emitted:

```
|move|p1a: Raichu|Thunderbolt|p2a: Tangela
|-resisted|p2a: Tangela
|-damage|p2a: Tangela|201/277                     ← 76 (Thunderbolt, roll 100 of a 64..76 fan)
|-status|p2a: Tangela|par                         ← Thunderbolt's 10 % secondary
|move|p2a: Tangela|Hidden Power|p1a: Raichu
|-damage|p1a: Raichu|14/235                       ← 109 (HP Grass, roll 95 of a 97..115 fan)
|-heal|p1a: Raichu|28/235|[from] item: Leftovers   ← +14 = 235//16
|-damage|p1a: Raichu|0 fnt|[from] Leech Seed       ← −28, CAPPED by the 28 HP that were there
|-heal|p2a: Tangela|229/277 par|[silent]           ← +28, the MIRROR of that capped drain
|faint|p1a: Raichu
|-heal|p2a: Tangela|246/277 par|[from] item: Leftovers   ← +17 = 277//16
```

Extracted by the harness into

* p1 `[('', −109), ('itemleftovers', 14), ('capped_lethal', −28)]`
* p2 `[('', −76), ('heal', 28), ('itemleftovers', 17)]`

The shipping engine emits **14 arms**, summing to 100.000000 %. Their non-crit damage values on p1
are exactly two: **−101** (mass 51.416 %) and **−108** (mass 39.990 %), plus **−123** on the crit
arms (6.094 %) and no p1 damage at all on the 2.5 % full-paralysis arms. Twelve arms are reported as
misses; the list is truncated at `misses[:12]`, which is why 14 arms yield 12 lines.

Four arms — 34.61 %, 2.88 %, 2.31 %, 0.19 %, the residual-kill arms crossed with the paralysis
and Thunderbolt-crit splits — get **all the way past the p1 comparison** and fail on one component:

```
p2 attributed components differ: observed_only=[('heal', 28)] engine_only=[('heal', 29)]
```

On those arms the engine's p1 list is `[('', −108), ('itemleftovers', 14), ('capped_lethal', −29)]`
against the observed `[('', −109), …, ('capped_lethal', −28)]`, and that **passes**: the comparator's
cap-vs-cap identity holds exactly, `|obs_cap| − |eng_cap| = 28 − 29 = −1 = 108 − 109 =
eng_direct − obs_direct`. The other ten arms fail earlier, on p1's roll-scaled list.

> **A prior note in this program compared 109 against 101 and read Raichu at 14 versus 22.** That is
> the *surviving* representative — the arm whose p1 side fails outright. The arm that actually
> produces the residue is the residual-kill arm at **108**, where Raichu sits at 15, not 22. The
> arithmetic in the note is right for the arm it names; it names the wrong arm. Both readings give
> heal 29, so the conclusion survived, but the mechanism did not: the residue is not "the engine
> collapsed onto a surviving roll", it is "the engine emitted the kill arm at the bottom of the
> lethal band and the drain is not constant across that band".

## 2. Is the observed roll in the engine's own fan, and is the observed heal reachable?

`poke_engine.calculate_damage` returns fan **maxima**; the fan is `floor(max · r / 100)` for
`r ∈ 85..=100`. For Tangela's Hidden Power Grass into Raichu the max is 115, giving

```
[97, 98, 100, 101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 115]
```

**109 is a member — roll 95.** The engine prices it and emits no arm at it. By
`scripts/family_bucket_audit.py`'s own definition of engine-gap (the branch support does not contain
the observed transition) that settles the bucket.

The drain is `min(maxhp//8, hp_after_move + leftovers) = min(29, 137 − d)`. Over the lethal band:

| roll | d | HP after move | +Leftovers | drain / mirror heal |
|---|---|---|---|---|
| 94 | 108 | 15 | 29 | **29** |
| 95 | **109** | 14 | 28 | **28** ← observed |
| 96 | 110 | 13 | 27 | 27 |
| 97 | 111 | 12 | 26 | 26 |
| 98 | 112 | 11 | 25 | 25 |
| 99 | 113 | 10 | 24 | 24 |
| 100 | 115 | 8 | 22 | 22 |

**Seven rolls, seven distinct drains. The map is injective on the lethal band** (measured, not
derived by hand: `lethal_band.injective = true` in the artifact).

Census of what the shipping engine actually emits for the p2 `heal` component across all 14 arms:

```
{29: 93.9063 %}          — one value, plus "absent" on the crit arms where Raichu never reaches the residual
```

**28 is not reachable by any arm the shipping engine can emit on this boundary.**

## 3. Why #1152's two fixes are inert here — measured, not read

* **Crit-straddle sub-split.** The crit fan for Hidden Power Grass into Raichu is
  `[196 … 231]`; its **minimum, 196, exceeds Raichu's 123 HP**, so every crit roll is a move-KO.
  There is no surviving crit sub-fan for the sub-split to partition, and the observed protocol
  carries no `|-crit|` on p1 anyway.
* **Status-aware residual threshold.** The mon that dies to the residual is Raichu, and its attacker's
  move is `hiddenpowergrass70`. Across all 14 arms the only `|-status|` line the engine emits is
  `|-status|p2a: Tangela|par` — nothing is ever applied to Raichu, so the threshold ladder for
  Raichu has exactly one rung. That is visible in the output: the engine emits exactly **one**
  residual-kill damage value, 108, where a two-rung ladder would emit two.

Both fixes are correct and both are simply not on this boundary's path. That the full-enumeration
spike closed the row and these two did not is therefore not a coincidence about the fixes — it is
the tell that the mechanism is the *representative*, not the *threshold*.

## 4. Four counterfactuals that isolate the cause to a single number

Run through the unmodified, shipped `evaluate_boundary_strict`.

| # | change | verdict |
|---|---|---|
| — | control: unmodified row, unmodified harness | **diverged** (14 arms) |
| A | change **only** the observed damage roll 109 → 108 (and the HP trace it implies) | **matched** |
| A2 | change **only** the observed mirror heal 28 → 29, leaving the −109 roll in place | **matched** |
| B | change **only** the harness: add bare `heal` to `_ROLL_SCALED_SOURCES` | **matched** |
| C | change **only** the harness, narrowly: reclassify a bare `heal` as roll-scaled *iff* it equals a `capped_lethal` on the opposite slot | **matched** |

A2 is the decisive one. With the observed roll left at 109 and *only* the mirror heal moved, the
boundary matches — so the p1 side, roll difference and all, is already forgiven by the shipped
comparator, and the whole divergence is **one integer on the other side of the field**.

## 5. Root cause, stated exactly

`damage_component_events` contains, forty lines apart, the rule and its missing mirror:

```python
if tag == "-heal" and max_hp and new_hp >= max_hp:
    source = f"{source}_to_full"      # a heal capped AT MAX HP is roll-scaled
...
if fainted_here and tag == "-damage":
    source = "capped_lethal"          # a residual capped BY REMAINING HP is roll-scaled
```

The second rule's own comment says why: *"a residual that KILLS is capped by the HP that happened to
be left, so its magnitude inherits the roll of whatever damaged the mon earlier in the turn."*
Leech Seed transfers that capped amount to the other side, where it arrives as a bare
`|-heal| … |[silent]|`, gets `source = "heal"`, and is compared **exactly**. One physical quantity,
two buckets, depending on which side of the field it lands on.

So there are two true statements and they are not in conflict:

1. **The engine emits no arm at the observed roll**, and the observed roll is in its own fan. Engine
   gap by the repo's definition.
2. **The comparator is internally asymmetric** about the very quantity that fails.

§6 argues that (2) is not the one to fix.

## 6. Enumeration closes it; the collapsed path cannot

The enumeration build, single-variable (same build, same seeds, flag off versus on):

| | branches | mass sum | p2 `heal` magnitudes emitted | verdict |
|---|---|---|---|---|
| flag **off** (control) | 14 | 100.000000 % | `{29: 93.9062 %}` | **diverged** — misses identical to the shipping build |
| flag **on** | 1015 | 100.000000 % | `{22: 5.7129, 24: 5.7129, 25: 5.7129, 26: 5.7129, 27: 5.7129, 28: 5.7129, 29: 59.6289}` | **matched** |

The off-control matters: it shows the rebased patch is inert when the flag is clear, so the closure
is attributable to enumeration and not to the rebase. The emitted masses are coherent. 2.5 % of the
boundary is Tangela fully paralysed and never striking — exactly Thunderbolt's 10 % paralysis times
gen-3's 25 % immobilisation — leaving 97.5 % on which Hidden Power lands; `1/16` of that is the crit
arm at 6.0938 %, and the remaining `97.5 × 15/16 = 91.40625 %` splits into sixteen equal non-crit
rolls of **5.7129 %** each. Six of the seven lethal rolls therefore carry one roll's mass apiece,
and the 29-heal figure of 59.6289 % is the nine surviving rolls plus roll 108 plus the 2.5 %
full-paralysis arms — `10 × 5.7129 + 2.5 = 59.629`.

**The limit claim is dead.** C133 §5 states the falsifier for this family: *any arm the per-roll
enumerator can produce that the comparator accepts*. It fires — arm `−109 / heal 28`, mass 5.71 %,
verdict `matched`. Nothing in this row is a limit, and this report does not construct a
demonstration because none can be constructed.

**What the collapsed path cannot do**, stated so it can be attacked:

> Under the collapsed discipline — at most one arm per residual threshold — no choice of
> representative closes this row.
>
> Because: (a) the only failing component is the p2 mirror heal (counterfactual A2, measured);
> (b) the mirror heal is injective on the 7-roll lethal band, taking 7 distinct values (measured);
> (c) one arm carries exactly one value, and `heal` is compared for exact equality; (d) therefore at
> most 1 of the 7 lethal rolls can ever match, and Showdown rolled one of the other 6.
>
> Falsifier: exhibit a single representative the shipped comparator accepts against all seven
> lethal rolls. Impossible while the seven values are distinct and the comparison is exact.

Note carefully what this does **not** say. It does not say no engine change can close it — §7 gives
one. It says the *collapsed* discipline cannot, and the reason is arithmetic rather than an
implementation gap. The bound is 1/7 of the lethal band, and the shipping engine is already at that
bound: it prices the arm at 108, the one roll of the seven that would have matched.

## 7. The two fixable routes, and why neither should ship on the strength of one row

### 7a. Engine: split the residual-kill arm per roll inside the band

Concrete shape: where the partition emits a residual-kill arm and the killing residual **transfers
HP to the other side** (in gen 3, Leech Seed alone — Nightmare and Ghost-Curse are damage-only and
are absent from the randbats pool anyway), the single arm at the threshold must become one arm per
distinct drain across the band, each at mass `1/16`.

Blast radius, **arithmetic estimate, not measured**: on this boundary it replaces 4 arms with
4 × 7 = 28, taking the branch count 14 → 38. Generally it multiplies each affected residual-kill arm
by the band width, which is at most 16 and here is 7. It fires only where a residual-kill arm already
fires *and* the residual is Leech Seed.

Recommendation: **do not ship.** Restricted to the lethal band this is per-roll enumeration wearing a
narrower gate, so it buys the correctness of enumeration at the cost of a second partition mechanism
to maintain — in a family that has already produced three wrong hand-derived mass recipes (C134 §3)
and one fix that opened 78 rows against a single closure (C133 §7). The program already decided
(C137) that enumeration is the oracle. This row is evidence for that decision, not an exception to it.

### 7b. Harness: reclassify the mirror of a capped lethal residual

Two variants were built and measured, both of which close the row (counterfactuals B and C).
The mutation test is the non-vacuous falsifier — a comparator *widening* cannot open a row by
construction, so "nothing opened" proves nothing about it. Falsify the observed mirror heal to
values that are physically impossible given the observed −109 roll, and ask whether each comparator
still calls it a divergence:

| observed heal | truthful? | shipping | B (blanket) | C (narrow) |
|---|---|---|---|---|
| 28 | **yes** | diverged | matched | matched |
| 27 | no | diverged | **matched** | diverged |
| 26 | no | diverged | **matched** | diverged |
| 25, 24, 22, 20, 18 | no | diverged | diverged | diverged |

**Variant B over-accepts, measured.** Putting bare `heal` in `_ROLL_SCALED_SOURCES` gives it the
`[0.92·eng − 1, 1.09·eng + 1]` window — `[25.68, 32.61]` around 29 — and it then blesses mirrors of
27 and 26 that no roll can produce. It would also hand that window to Recover, Soft-Boiled and
Morning Sun, which are exact fractions of max HP. That is exactly the over-acceptance C133 §3
warned the harness route had to avoid.

**Variant C does not over-accept on this test**, because its gate is an identity rather than a
tolerance: the heal is reclassified only when it equals the `capped_lethal` the harness *derived
from the observed HP trace* on the opposite slot. Falsify the heal and the identity breaks, the
reclassification does not fire, and the exact comparison rejects it. Over the 68 distinct retained
transition repros committed under `reports/artifacts/` (deduplicated on `(seed, step)`, re-read on
`e65044f6`), C closes exactly this row and nothing else: 64 → 65 matched, 4 → 3 diverged, none
newly diverged. That census is small and historically biased and should not be read as strong
evidence of safety.

Recommendation: **do not ship.** Not because it is loose, but because of what it deletes. The p2
mirror heal is currently the **only** component on this boundary shape that pins the damage roll
exactly; the p1 side's roll difference is already forgiven by the cap-vs-cap identity. Reclassifying
the mirror removes the instrument's last ability to distinguish *"the engine chose a different
roll"* from *"the engine computed the drain wrong"* on residual-lethal Leech Seed boundaries — which
is precisely the class of defect this row is evidence of. C135 §4 rejected the analogous move for the
analogous reason: making the matcher accept the engine's answer here would not repair the instrument,
it would blind it.

## 8. Recommended disposition

**`19000191/63` — engine gap; closed by enumeration, retained under the collapsed path.**

Concretely, for the ledger: `reports/c138_known_gaps_ledger.md` entry **G8** describes this row as
*"the collapsed lethal arm mis-prices a roll-dependent Leech Seed drain"*, which is accurate, and it
should now also record that (i) the drain is injective over the lethal band so no representative can
price it, (ii) the enumeration oracle closes it, measured, and (iii) it is not a limit. The dev
window's steady state is **1 divergence in 15,503 boundaries under the collapsed path and 0 under
the oracle**, and that gap is a known, bounded, understood property of collapsing a fan — not an
open defect to keep chasing.

## 9. Operational finding: the enumerate spike patch no longer applies

`third_party/poke-engine-gen3-enumerate-damage-rolls.patch` **fails to apply on current main**:

```
ERROR: both strict applicators rejected poke-engine-gen3-enumerate-damage-rolls.patch
error: patch failed: src/gen3/generate_instructions.rs:3457
1 out of 3 hunks failed while patching 'src/gen3/generate_instructions.rs'
```

#1152's status-aware-threshold patch replaced `let residual_threshold_opt =
residual_lethality_threshold(...)` with `let residual_thresholds =
residual_lethality_thresholds(state, &attacking_side, choice)`, and the spike's second hunk is
authored against the old binding. Since C137 makes enumeration the program's *reference oracle*, an
oracle that cannot be built is a live problem, not a curiosity.

`third_party/poke-engine-gen3-enumerate-damage-rolls-rebased.patch` is added here with the rebase.
It is **not** in `third_party/poke-engine-gen3-patches.txt` and is therefore inert — exactly like its
un-rebased sibling. To build the oracle:

1. append the patch name to `third_party/poke-engine-gen3-patches.txt`;
2. set `PATCHED_TARGET_TREE_SHA256` in `scripts/apply_poke_engine_patches.py` to
   `8386e653da668cdcc76073d7ebff5631146b8a0f26cbe044ac36d26b005ec81c`;
3. build into a **separate** venv, then run with `POKEZERO_ENUMERATE_ROLLS=1` — the flag is a
   `OnceLock`, so one process is one engine.

Both changes are local to the oracle checkout and must not land on main, which is why they are not
in this branch.

## 10. What was ruled out, and what is left uncertain

Ruled out by measurement:

* **A limit.** The enumerated arm at −109 / heal 28 exists and the shipped comparator accepts it.
* **A crit route.** Every crit roll (fan minimum 196) kills Raichu on the move; the observation has
  no `|-crit|`.
* **A missing threshold.** The threshold is 108, correct, and the engine emits an arm at it.
* **A second threshold rung.** Nothing statuses Raichu on this boundary, so #1152's ladder is
  single-rung here.
* **The p1 side being at fault.** Counterfactual A2 matches with the −109 roll left in place.
* **"Blanket `heal` widening is harmless."** It blesses two impossible mirrors.

Left uncertain, and stated as such:

* The **per-roll band-split engine fix's cost is an arithmetic estimate** (14 → 38 arms on this
  boundary). No build was made and no sweep was run. If the program ever wants that fix, it needs a
  registered prediction with a "nothing opened" falsifier and both windows swept, per C133 §7.
* The **68-row retained census is weak**. It is capped at 25 repros per artifact and skewed toward
  rows recorded on older engines. It bounds nothing about future over-acceptance; the mutation test
  is what carries that argument.
* **How much matched mass rides on the `±9 %` fallback window** repo-wide is still unmeasured —
  C135 §6 raised it and this report does not close it.
* Variant C was measured only against mirror falsifications on **this** boundary. A compensating
  engine defect — a wrong drain that stays consistent with a wrong roll — would be invisible to it,
  and is invisible to the p1 cap-vs-cap identity today too.

The single measurement that would overturn this report's disposition: **a collapsed-path arm, from
any engine change that keeps one arm per residual threshold, that the shipped comparator accepts
against this boundary.** §6 argues that cannot exist; producing one refutes §6 directly.
