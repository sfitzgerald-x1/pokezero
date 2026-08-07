# C140 — the last dev row, `19000191/63`: an engine gap that enumeration closes and no fixed representative can do better than trade

After #1144, #1148 and #1152, `19000191/63` (`component_magnitude:heal`) is the only divergent
row on either window: dev 1/15,503, holdout 0/15,579
(`reports/artifacts/c134_collapsed_{dev,holdout}_sweep.json`, the sweeps #1149 committed).

**Disposition: ENGINE gap. Not a limit — the falsifier fires, measured.** The remedy that works is
enumeration, measured on main's shipped oracle at both the row and the sweep level. Under the
collapsed path the row is not closable *without a trade*: any fixed representative prices exactly
one of the seven lethal rolls, and the shipping engine is already at that bound. The recommended
program disposition is **"closed by enumeration, retained under the collapsed path"**.

> ### Correction, second revision — the first revision's central claim was FALSE
>
> §6 of the first revision asserted, in a box, that **no choice of representative closes this row**.
> A review overturned it with one monkeypatch, and §6 now carries the reproduction: re-pricing the
> residual-kill arm to **109** — still one arm per residual threshold, still 14 arms — **matches**.
>
> The defect was step (d). Steps (a)–(c) proved a *bound* (one arm carries one drain, the drains are
> distinct, so a representative prices exactly one roll). Step (d) then wrote "therefore at most 1 of
> the 7 can ever match, and Showdown rolled one of the other 6" — which silently assumes the
> representative is 108. It is 108 only because that is what the engine happens to emit today.
>
> Two smaller errors went with it. The box's falsifier — *"exhibit a single representative accepted
> against all seven lethal rolls"* — was a strawman, strictly harder than the claim it defended; the
> correctly scoped version carried in §9 and the PR body ("a collapsed-path arm, from any engine
> change keeping one arm per residual threshold, that the comparator accepts") **fired**. And the
> box closed by calling 108 "the one roll of the seven that would have matched". **109 is.**
>
> The bound survives, and §6 now states it with the measurement that makes it stronger than the
> universal it replaces.

## Why this boundary keeps getting a correction box: the count, enumerated

An earlier revision of this box wrote *"four confident diagnoses, three of them wrong"* and cited
`reports/c105_retract_limit_overclaim.json` for the 8-for-8. Both were sloppy, and a report whose
thesis is *derive the number, do not carry it* cannot carry an uncounted number in the paragraph
making that argument. Enumerated, with citations, every one resolvable:

| # | reading of `19000191/63` | where | retracted by |
|---|---|---|---|
| 1 | filed under `limit:` with no demonstration artifact | C99 — **second-hand**, see note | the C116 Phase 0 item 1 adjudication |
| 2 | "parser overwrites a supplied source" as the row's cause | `c103_limit_readjudication.json` → `FINAL_RESULT.c104_class_parser_overwrites_a_supplied_source` | c103's own `SUPERSESSION_2026_08_04`: *"SUPERSEDED IN ITS PER-ROW CAUSES"* |
| 3 | "a genuine comparison limit" | `c105_retract_limit_overclaim.json` → `so_three_of_the_seven_are_limits` | c105's own `SUPERSESSION_2026_08_04` |
| 4 | `limit:` again, re-adjudicated | C111 **v1** | v2, reduced to cause A7 (`c111_residue_row_causes.md:91`) |
| 5 | "the collapsed arm kept the surviving representative" (109 vs 101, Raichu at 22) | **c140 rev 1**, §1 | §1 of this revision — it named the survive arm, not the residual-kill arm at 108 |
| 6 | "no representative closes this row" | **c140 rev 1**, §6 box | §6a of this revision, by measurement |

**Six wrong readings, four of them by three separate earlier reports and two of them mine in this
one.** Two caveats, because the table is the argument:

- **#1 is second-hand.** No `c99_*` file exists under `reports/`. Membership is taken from
  `docs/engine_fidelity_program_20260801.md`, which states the eight-row set and names
  `19000008/54`, `19000191/63`, `19000198/33` as the three C111 v1 re-adjudicated — i.e. this row is
  in both sets. I could not open the primary source, and say so rather than counting it silently.
- **c104 is deliberately *not* in the table.** Its supersession reads *"CAUSE STILL STANDS"*, and the
  parser overwrite it identifies is real and still open (ledger H7). What this report adds is that
  the overwrite is **not the operative cause here**: it is happening on this very row — Showdown
  supplies `[from] Leech Seed` and the observation carries the bare `capped_lethal` — and the p1
  comparison **passes anyway** (§4, counterfactual A2). c104 was describing a real defect at an era
  when the engine's representative was `−106`; it is not what keeps this row open today.

**The correct citation for the 8-for-8** is the C116 Phase 0 item 1 adjudication, documented at
`docs/engine_fidelity_program_20260801.md` ("Established by the independent adjudication that closed
that report"). c105 only *records* it, in a supersession note, and c105's own conclusion was the
opposite — it called this row a genuine limit. Citing c105 for the retraction of a claim c105 made
was the wrong attribution even though the membership is right.

No engine or harness change ships with this report. The only non-documentation edit is two entries
added to `_MENTION_ALLOWLIST` in `tests/test_roll_enumeration_scope.py`, because that gate is a
ledger of every tracked file naming `POKEZERO_ENUMERATE_ROLLS` and this report and its artifact
both do. Neither is on an import path and neither sets the flag; the runtime-scope gate that
actually measures leakage is untouched and passes.

## 0. Provenance

| | |
|---|---|
| replay build | fingerprint `44ee1430708cbb55033f5c7f1234b4bf9699009e6ba6d9a972ba442df615d652`, **71 patches**, source `6be52191` (current main) |
| sweep the row comes from | `c134_collapsed_dev_sweep.json`, engine fingerprint `44ee1430…` — **the same fingerprint**, so the replay is on the engine that produced the row |
| both roll paths | **one build**. Since #1149 put `poke-engine-gen3-enumerate-damage-rolls.patch` in the manifest, `POKEZERO_ENUMERATE_ROLLS` selects the path at process start (the flag is a `OnceLock`, so one process is one engine). The collapsed and enumerated measurements below are therefore single-variable by construction. |
| every number below | `reports/artifacts/c140_last_dev_row_probe.json` |

The build was produced by `scripts/build_search_crate_engine.sh` with the exit code captured
directly (`exit=0`); all 71 patches applied via git-apply. Replays go through
`cert_sweep_reread.reread_row`, which calls the shipped `evaluate_boundary_strict` — nothing here
reimplements the comparator.

> **An earlier revision of this report was written against `eaef463a` and reported that the
> enumerate patch no longer applied, bundling a rebase as an inert unlisted patch file plus two
> local build steps. That was stale on arrival.** #1149 (`6be52191`) landed the same rebase and
> resolved the #1152 collision *better*: it **dropped** the `residual_threshold_opt` hunk rather
> than porting it, because that hunk disabled the residual mirror unconditionally while enumerating
> only under a narrower guard — giving a flag-on search process a third never-measured
> configuration, the defect c137 §2 named. The oracle is buildable on main today. The rebase, the
> extra steps and the digest bump are withdrawn. Every measurement below was re-run on `44ee1430`
> rather than carried forward, and the results are unchanged.

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
{29: 93.9062 %}          — one value, plus "absent" on the crit arms where Raichu never reaches the residual
```

**28 is not reachable by any arm the shipping engine can emit on this boundary** — with the
representative it currently uses. That qualifier is load-bearing and its absence is what wrecked the
first revision: a *different* representative in the same one-arm-per-threshold discipline does emit
28, and §6a measures it.

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

**And the obvious third route — re-label the mirror so it carries its cause — is closed at the
source.** Showdown emits **no `[from] Leech Seed` heal line at all**: the drain renders silently
(`sim/battle.ts:2293-2296`, `case 'leechseed'`, reached from `data/moves.ts:10218-10221`), which is
`reports/c131_leechseed_heal_label.md` change 4 — a report that records asserting the opposite label
as one of its own errors. So both sides render a bare `heal` **by design**, and no parser change can
recover the attribution from the protocol; the information is not in it. That completes the domain:
the mirror can be re-bucketed (§7b) or the arm can be re-priced (§6a, §7a, §7c), but it cannot be
re-labelled. *(Cited second-hand: Showdown is not vendored in this repo, so this rests on c131's
citation rather than on a file I opened.)*

So there are two true statements and they are not in conflict:

1. **The engine emits no arm at the observed roll**, and the observed roll is in its own fan. Engine
   gap by the repo's definition.
2. **The comparator is internally asymmetric** about the very quantity that fails.

§6 argues that (2) is not the one to fix.

## 6. Enumeration closes it; a fixed representative can only trade

Single-variable on **one** build (`44ee1430`), same row, flag off versus on:

| | branches | mass sum | p2 `heal` magnitudes emitted | verdict |
|---|---|---|---|---|
| flag **off** (control) | 14 | 100.000000 % | `{29: 93.9062 %}` | **diverged** — misses identical to the sweep's |
| flag **on** | 1015 | 100.000000 % | `{22: 5.7129, 24: 5.7129, 25: 5.7129, 26: 5.7129, 27: 5.7129, 28: 5.7129, 29: 59.6289}` | **matched** |

This is the per-row form of a closure #1149 already measured at sweep scale on the same
fingerprint — `c134_collapsed_dev_sweep.json` 1 diverged versus `c134_enumerated_dev_sweep.json`
**0**, both at 15,503 boundaries measured, holdout 0 → 0. The two measurements are independent in
method (per-row replay versus a 200-game sweep) and agree.

The emitted masses are coherent. 2.5 % of the
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

### 6a. What the collapsed path can and cannot do — the bound, measured

> **A first revision of this section claimed no representative closes this row. That is false.**
> Representative **109** closes it. What is true is a *bound*, and the bound is worth more than the
> false universal was.

The measurement. `pokezero_search.branch_events` is monkeypatched to re-price the residual-kill arms
to a different representative **inside** the band — one arm per residual threshold throughout,
**14 arms throughout, asserted per cell** — and the result is fed to the **unmodified** shipped
`evaluate_boundary_strict`. The strict path compares rendered components only, so an event rewrite
models the engine change faithfully. Rows are the engine's representative; columns are the roll
Showdown threw. (`observation(109)` reproduces the recorded protocol byte-for-byte, asserted.)

|  | 108 | 109 | 110 | 111 | 112 | 113 | 115 |
|---|---|---|---|---|---|---|---|
| **rep 108** *(shipping)* | **MATCH** | · | · | · | · | · | · |
| **rep 109** | · | **MATCH** | · | · | · | · | · |
| **rep 110** | · | · | **MATCH** | · | · | · | · |
| **rep 111** | · | · | · | **MATCH** | · | · | · |
| **rep 112** | · | · | · | · | **MATCH** | · | · |
| **rep 113** | · | · | · | · | · | **MATCH** | · |
| **rep 115** | · | · | · | · | · | · | **MATCH** |

**Two controls, without which the matrix proves less than it looks:**

* **The rewrite machinery is a no-op at 108.** Re-pricing to `rep = 108` reproduces the unpatched
  render **byte for byte** (asserted on the full payload). Without this, the diagonal cell at 108
  could be an artifact of the rewrite rather than of the engine.
* **No synthesised observation is malformed.** `observation(109)` reproduces the recorded protocol
  byte for byte, and a review ran all seven synthesised observations against the **enumerated** build
  and got seven matches — so every column is a boundary the engine can actually produce.

**The matrix is the identity** — but half of it is a tautology and should be labelled as such.

> **The diagonal is definitional, not a discovery.** With `rep == r` the engine arm's p1 components
> and its mirror heal are *identical to the observation by construction*, so the only live
> adjudication left is p2's direct damage roll, which the fan forgives. **All the informative content
> is in the 42 off-diagonal cells.**

And those 42 carry more than "diverged". In **every one** the residual-kill arm reaches the p2
comparison and fails there with exactly the predicted pair,
`observed_only=[('heal', 137−r)] engine_only=[('heal', 137−rep)]` — 42 of 42, asserted, 14 arms per
cell. Reaching p2 means the p1 comparison **passed**, so the cap-vs-cap identity
`(|obs_cap| − |eng_cap|) == (eng_direct − obs_direct)` holds for **every** `(rep, r)` pair in the
band, not merely at the shipping 108. That is what the bound actually needs: the mirror heal is the
**sole discriminator across the whole band**, provably, rather than at one point.

So:

**(i) The bound.** Any fixed representative prices **exactly one** of the seven lethal rolls —
asserted on all seven rows, not inferred. This is what steps (a)–(c) of the withdrawn box actually
proved: one arm carries one drain, the drains are distinct, `heal` is compared exactly.
The shipping engine is at that bound; it is not below it.

**(ii) Moving 108 → 109 is a trade, not a fix.** It closes this row and opens every boundary of this
shape where Showdown throws 108 — column 108, which representative 109 misses. Both rolls carry one
sixteenth of the fan, so the exchange is **even in expectation**. Nothing is gained; the covered
roll is permuted.

**(iii) Picking 109 would be fitting the representative to the sample.** The only reason 109 is
attractive is that the dev window happens to contain a boundary where Showdown threw it. Choosing an
engine constant because it closes an observed row is exactly what the dev/holdout split exists to
forbid — and this row is a single observation, `n = 1`.

**(iv) The principled rule does not close it either — measured, and confirmed at source.** The two
conventions are visible in `gen3/generate_instructions.rs`, so this is not fitted to the observed
`−101`:

* `residual_disjoint_bands` stores `(threshold, band)` and prices the band arm at the **threshold
  itself** — which is why the shipping representative is 108.
* `survive_representative = average_below`, and `compare_health_with_damage_multiples` returns
  `total_less_than / num_less_than` — an **integer mean of the sub-fan below the threshold**. That
  gives `916/9 = 101` exactly, reproducing the `−101` the engine emits.

Applying the survive arm's own rule to the lethal band gives
`floor((108+109+110+111+112+113+115)/7) = floor(778/7) = 111` → drain 26 → **diverged** (measured).
The result does not depend on which mean-like rule is chosen: `floor((min+max)/2) = floor(223/2)` is
also **111**. So the row is not closable by any rule the engine already follows — only by a
hand-picked constant, which (iii) rules out.

**The correctly scoped falsifier, and what happened to it.** §9 and the PR body of the first revision
named it properly: *a collapsed-path arm, from any engine change keeping one arm per residual
threshold, that the shipped comparator accepts against this boundary.* **It fired.** (The box's own
falsifier — "a representative accepted against all seven rolls" — was a strawman, strictly harder
than the claim it defended, and would never have fired. That is the lesson: a falsifier harder than
the claim is not a falsifier.)

What remains true, and is the operative point for the program: **under one arm per residual
threshold this boundary shape is matched with probability 1/7 whatever representative is chosen, and
under enumeration with probability 1.** That is the gap, and it is a property of collapsing a fan
rather than a defect to be tuned away.

## 7. The three routes that close it, and why none should ship on the strength of one row

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

**Variant C survives this particular test — but the test is weaker than it looks, and an earlier
revision of this paragraph oversold it.** C's gate keys on the *observed trace alone*: it
reclassifies the heal only when it equals the `capped_lethal` the harness derived from that same
trace. The mutation falsifies the mirror while leaving the drained side untouched, which makes the
observed protocol **internally inconsistent in a way Showdown cannot emit** — Showdown always
transfers exactly what it removed, so `mirror == drain` holds in every real protocol. Against real
input, therefore, **C's gate fires on every Leech-Seed mirror**, and what it applies past the gate is
the same `[0.92·eng − 1, 1.09·eng + 1]` window B uses. The rows in the table where C reads `diverged`
are rows Showdown could not have produced.

So C is **narrower in scope** than B — it touches only the Leech-Seed mirror rather than every bare
heal, which is a real difference and keeps Recover exact — but the evidence here does **not** show
that an identity gate is inherently safer than a tolerance. On the boundaries it actually governs, it
is a tolerance.

Over the 68 distinct retained transition repros committed under `reports/artifacts/` (deduplicated on
`(seed, step)`, re-read on `44ee1430`), C closes exactly this row and nothing else: 64 → 65 matched,
4 → 3 diverged, none newly diverged. That census is small and historically biased and should not be
read as strong evidence of safety either.

Recommendation: **do not ship.** Not because it is loose, but because of what it deletes. The p2
mirror heal is currently the **only** component on this boundary shape that pins the damage roll
exactly; the p1 side's roll difference is already forgiven by the cap-vs-cap identity. Reclassifying
the mirror removes the instrument's last ability to distinguish *"the engine chose a different
roll"* from *"the engine computed the drain wrong"* on residual-lethal Leech Seed boundaries — which
is precisely the class of defect this row is evidence of. C135 §4 rejected the analogous move for the
analogous reason: making the matcher accept the engine's answer here would not repair the instrument,
it would blind it.

### 7c. Engine: re-price the residual-kill representative from 108 to 109

This closes the row — §6a measured it — and it is a one-constant change with no new machinery, which
makes it the most tempting of the three and the one that most needs refusing.

Recommendation: **do not ship**, for the two reasons §6a establishes. It is a **wash**: the identity
matrix says representative 109 misses exactly the column 108 covers, and both rolls carry `1/16` of
the fan, so the expected number of matched boundaries of this shape is unchanged. And it would be
**fitted to the sample**: the sole evidence for 109 over 108 is that the dev window contains one
boundary where Showdown threw 109, `n = 1`. Meanwhile the engine's *own* convention — `floor` of the
band mean, which reproduces the `−101` survive representative exactly — gives 111, which diverges.
Changing a representative to close an observed row, against the rule the engine otherwise follows, is
the definition of what the holdout window exists to catch.

## 8. Recommended disposition

**`19000191/63` — engine gap; closed by enumeration, retained under the collapsed path.**

Concretely, for the ledger: `reports/c138_known_gaps_ledger.md` entry **G8** describes this row as
*"the collapsed lethal arm mis-prices a roll-dependent Leech Seed drain"*, which is accurate, and it
now also records that (i) the drain is injective over the lethal band, so a fixed representative
prices exactly one of the seven rolls and re-pricing only trades which one; (ii) the enumeration
oracle closes it, measured; and (iii) it is not a limit. **G8 stays open** — the row is unfixed under
the shipping configuration and the ledger should keep saying so.

> The first revision of this report also wrote the sentence *"Not closable by a representative
> either"* into G8, and that sentence merged. It is false — see §6a — and is corrected in this PR.
> A merged document must not be left carrying an overturned claim.

The dev window's steady state is **1 divergence in 15,503 boundaries under the collapsed path and 0
under the oracle**, and that gap is a known, bounded, understood property of collapsing a fan — not
an open defect to keep chasing.

This disposition does not conflict with C116 §5. That clause requires a written demonstration for a
`limit:` disposition, and this is explicitly **not** one: §6 records the falsifier firing.

## 9. What was ruled out, and what is left uncertain

Ruled out by measurement:

* **A limit.** The enumerated arm at −109 / heal 28 exists and the shipped comparator accepts it.
* **A crit route.** Every crit roll (fan minimum 196) kills Raichu on the move; the observation has
  no `|-crit|`.
* **A missing threshold.** The threshold is 108, correct, and the engine emits an arm at it.
* **A second threshold rung.** Nothing statuses Raichu on this boundary, so #1152's ladder is
  single-rung here.
* **The p1 side being at fault.** Counterfactual A2 matches with the −109 roll left in place.
* **"Blanket `heal` widening is harmless."** It blesses two impossible mirrors.
* **"No representative closes this row."** Overturned by measurement, in this report's own second
  revision. Representative 109 closes it. What survives is the bound in §6a.
* **Re-labelling the mirror.** Showdown emits no `[from] Leech Seed` heal line at all, so the
  attribution is not in the protocol to recover (§5; cited second-hand via c131 change 4).
* **The parser overwrite (c104 / ledger H7) as this row's cause.** The overwrite is happening here
  and the p1 comparison passes regardless — measured, §4 counterfactual A2. The defect is real and
  still open; it is not what keeps this row divergent.

Left uncertain, and stated as such:

* The **per-roll band-split engine fix's cost is an arithmetic estimate** (14 → 38 arms on this
  boundary). No build was made and no sweep was run. If the program ever wants that fix, it needs a
  registered prediction with a "nothing opened" falsifier and both windows swept, per C133 §7.
* The **68-row retained census is weak**. It is capped at 25 repros per artifact and skewed toward
  rows recorded on older engines. It bounds nothing about future over-acceptance; the mutation test
  is what carries that argument.
* **How much matched mass rides on the `±9 %` fallback window** repo-wide is still unmeasured —
  C135 §6 raised it and this report does not close it.
* Variant C was measured only against mirror falsifications on **this** boundary, and §7b now records
  that those falsifications are protocols Showdown cannot emit — so the test does not show C is safe
  against real input, only that it is narrower in scope than B. A compensating engine defect — a
  wrong drain that stays consistent with a wrong roll — is invisible to C, and is invisible to the p1
  cap-vs-cap identity today too.
* **(ii)'s "wash" is an expectation argument, not a sweep.** The identity matrix shows representative
  109 misses column 108, and both rolls are `1/16` of the fan; the claim that the exchange is even
  therefore rests on the fan being uniform, which it is, rather than on having measured a
  109-representative build over both windows. No such build was made.

What would overturn what, stated separately now that one of these has already fired:

* **The disposition** (engine gap, not a limit, closed by enumeration): an arm the enumerator
  produces that the comparator rejects, or a demonstration that the enumerated closure is an
  artifact of the replay rather than the engine. Neither is in evidence; #1149's sweep-level result
  on the same fingerprint is independent corroboration.
* **The §6a bound** (any fixed representative prices exactly one of the seven lethal rolls): a
  representative, or a rule producing one, that the shipped comparator accepts against **two or more**
  columns of the matrix. That is the correctly scoped falsifier. The first revision's version — "no
  representative closes this row" — was refuted by a single monkeypatch, which is why the bound is
  now stated as a bound and measured on all seven rows rather than argued.
