# C152 — disposing the last entries that keep the known-gaps ledger open

`RATIFIED_SWEEP_PRECONDITION` defers the program's terminal measurement until *"the ledger is
terminal and the engine fingerprint is declared frozen for the claim."* An adversarial audit
found `reports/c138_known_gaps_ledger.md` **not terminal by its own text** and named five
entries: **G8** stays OPEN, **G33b** carries two open arms, and **H8**, **H12** and **H19** are
UNKNOWN.

This document disposes four of the five, and **the ledger is NOT terminal.** Three entries close
on measurement, one arm of the fourth is retired with a written demonstration, and while measuring
it C152 found **two new open items** — the other arm of G33b turns out not to be covered by the
demonstration, and a **third arm nobody had filed** (**G33c**) defeats C147's gate on exactly the
boundaries it was built for, observed live. Two further "never fired" claims in the ledger were
refuted in passing, the sixth and seventh of that class. Nothing here changes the engine or the
renderer.

> **The honest headline.** Every entry the audit named is now disposed with a measurement instead
> of an UNKNOWN, and the document is materially more terminal than it was. It is not terminal.
> The remaining opens are listed exactly, with their evidence, in §7.

> **What this document does NOT do.** It does not sweep, and does not propose to sweep, any
> seed at or above `19,200,000`. `19,200,000`–`19,200,259` is burned in the guard;
> `19,300,000`–`19,300,199` is the owner-ratified window awaiting the deferred one-shot. The
> only contact with `19,200,244` here is a **replay of states already committed** in
> `reports/artifacts/c141_final_holdout_sweep.json`, which generates no game and consumes no
> seed.

---

## 0. The five dispositions, and what each rests on

| entry | was | now | what it rests on |
|---|---|---|---|
| **G8** | OPEN — second instance `19200244/115` unreachable by C149's split, "both remainders unmeasured" | **RETIRED**, with the remainder measured | direct engine instrumentation of all four `residual_disjoint_bands` call sites on the row; a 27,655-band arithmetic census of the survive representative; both windows at 0 divergences at head |
| **G33b** | two arms OPEN — order-8 weather with the winner faster; exact speed ties | **weather arm RETIRED; tie arm STILL OPEN and now measured** | a 1,400-game predicate census, 925 predicate calls; the engine's own residual section order; `residual_heal_cause`'s Wish adjacency test |
| **G33c** | *not filed* | ⚠ **NEW and OPEN** — the truncation strands the winner's order-10 **damage** bookings, so C147's heal gate is inert wherever the winner carries a residual status | `1000513/121`, observed at pct 100.00, both protocols read side by side, diagnosed to the line |
| **H8** | UNKNOWN how much matched mass rides on the ±9 % window | **CLOSED — measured, and the cell's mechanism was wrong** | two 200-game sweeps per window, shipped comparator against a window-disabled variant produced by an AST rewrite of the shipped source |
| **H12** | UNKNOWN whether c43's ~7,224 invisible rows still hold | **CLOSED — falsified, and H12's own settling measurement is mis-scoped** | c43's own exclusion note; the coverage identity re-derived at head in both windows |
| **H19** | UNKNOWN whether four families survive into the current era | **CLOSED — measured, and the cell's evidence was vacuous** | 1,167 committed divergent rows re-classified; the four families' last rows replayed at head; the settling measurement fixed so it can run at all |

Three of the five cells contained a **wrong statement**, not merely an unresolved one, and all
three are corrected in place: H8's mechanism (the window is not reached through the door the cell
names, 0 times in 400 games), H12's named settling measurement (scoped to a population c43
explicitly excluded), and H19's evidence (the family layer never writes to a counter, so its
absence from the counters is not evidence of anything).

---

## 1. Base state, re-derived rather than accepted

Every figure in the handoff was re-derived from its own instrument in a fresh worktree.

| quantity | stated | re-derived | instrument |
|---|---|---|---|
| `main` | `66ee1869` | **`d20cf840`** — `66ee1869` is its parent's ancestor; #1195 merged after | `git log --oneline -1 origin/main` |
| patch stack | 74 | **74** | `grep -vc '^#\|^$' third_party/poke-engine-gen3-patches.txt` |
| engine fingerprint | `bfdbe1c04876edcd` | **`bfdbe1c04876edcd1957e7a…`** | `scripts/build_search_crate_engine.sh`, step 8 |
| harness digest | `e3459e1f…` | **`e3459e1f2ce334848c27…`** | `harness_digest.harness_digest()` |
| harness closure | 73 files | **73** | `harness_digest.harness_files()` |
| `_EXPECTED_SWEEP_ARTIFACTS` | 95 | **95** at the base tree | `tests/test_boundary_verdict_partition.py::_sweep_reports` |
| `_EXPECTED_COUNTER_ARTIFACTS` | 375 | **375** at the base tree | `tests/test_never_fired_counter_census.py::counter_artifacts` |

**The head sweeps.** Both permitted windows, 200 games each, strict matcher, no
`--approximate-sleep`, no `--enumerate-rolls`, at fingerprint `bfdbe1c04876edcd` / 74 patches /
source commit `d20cf840`, source tree clean. These are the first committed sweeps at the
shipping fingerprint: every earlier pair, including C149's, was taken at `8e912b45544034e6` or
older.

| window | all boundaries | single-seat | full-round | in-path exits | measured | matched | diverged | coverage |
|---|---|---|---|---|---|---|---|---|
| dev `19,000,000–199` | 17,710 | 1,742 | 15,968 | 465 | 15,503 | 15,503 | **0** | **87.54 %** |
| holdout `19,100,000–199` | 17,968 | 1,813 | 16,155 | 576 | 15,579 | 15,579 | **0** | **86.70 %** |

Support-gated acceptance re-derives exactly on the stated bar: dev `1,347 / 15,503` = **8.689 %**,
holdout `1,431 / 15,579` = **9.185 %**. Coverage re-derives exactly against the self-reported
`measured_fraction_of_full_rounds` of 97.09 % / 96.43 %, whose denominator excludes the
single-seat population.

---

## 2. G8 — the second instance `19200244/115`

### 2.1 The mechanism, re-verified by instrumentation rather than by reading

The audit's claim is that C149's per-roll split **structurally cannot reach** this row. That was
argued from source. It is now measured: an `eprintln!` was placed inside `residual_disjoint_bands`
**before** its `applicable_len == 0` early return, so it fires on every call whatever the filter
decides, and the row was replayed from the committed C141 artifact through that throwaway build.

Three distinct calls, and only three (`reports/artifacts/c152_g8_call_site_trace.json`):

```
42×  max_damage=159  min_roll=135  thresholds=[82]  ceiling=157    above=2  applicable=[]
38×  max_damage=99   min_roll=84   thresholds=[]    ceiling=32767  above=0  applicable=[]
38×  max_damage=49   min_roll=41   thresholds=[]    ceiling=32767  above=0  applicable=[]
```

**Confirmed.** The only call carrying a threshold at all has `min_roll < *threshold` evaluate
`135 < 82`, which is false, so `applicable_len == 0` and the function returns `None`.

**And one thing the audit did not say.** The two call sites C149's split actually touches are the
ones whose `ceiling` argument is `i16::MAX` — verified by reading all four, at
`generate_instructions.rs:4197` (`defender_active.hp`), `:4300` (`i16::MAX`), `:4406`
(`defender_active.hp`), `:4456` (`i16::MAX`).

> ⚠ **Corrected in review.** These read 4208/4311/4417/4467 in the first revision, which are the
> **instrumented** build's line numbers — shifted +11 by the eleven-line `eprintln!` block the
> trace needed, inserted above all four. Re-derived from the **shipping** vendored tree at
> fingerprint `bfdbe1c04876edcd`, with each site's `ceiling` argument re-read at the corrected
> line. A line number measured on a throwaway build and quoted as the shipping tree's is the same
> defect as a figure quoted from a tree that has since moved.
On this boundary **both `i16::MAX` sites are reached with an empty threshold slice**. The split is
therefore unreachable here *twice over*: not merely filtered out, but with nothing to filter. The
one call that carries a threshold is at `ceiling = defender_active.hp = 157`, one of the two sites
C149 deliberately left alone for blast radius.

### 2.2 The remainder, measured — which is what the cell said was missing

G8's cell ends *"Both remainders are unmeasured, and the final holdout was not swept."* The first
remainder is now measured, and stated without reference to any boundary.

The engine prices a non-KO arm with `compare_health_with_damage_multiples(max, health).0`, an
**average** over the f32 accumulator `max*0.85 + k*max*0.01`. Showdown throws from the **exact
integer fan** `floor(max * r / 100)`, `r` in 85..=100 — the C149 patch says so itself, and takes
its own per-roll arm values from that fan precisely so that "no arm is ever priced at a damage
Showdown cannot deal". An average is under no obligation to be a fan member, and when it is not,
the arm reproduces **zero** rolls of its own band instead of one.

`scripts/c152_g8_survive_representative_census.py` measures that plane
(`reports/artifacts/c152_g8_survive_representative_census.json`). It refuses to run unless its
model first reproduces `19200244/115` exactly — 14-roll band, band minimum 135, representative
145, 145 off-fan — so a model that had drifted would exit 2 rather than produce a number.

Over `max_damage` 10–600 and every `health` from 1 to `max(f32 accumulator rolls) + 1`:

> ⚠ **Corrected in review — that bound is not "the fan maximum", and the difference is real.**
> The loop runs to the top of the **engine's f32 accumulator**, which is not `max_damage` and
> not the integer fan's top: at `max_damage 159` the accumulator tops out at **158** against
> an integer-fan maximum of **159**. Calling it "the fan maximum" names the wrong of the two
> quantities this census exists to distinguish.

| quantity | value |
|---|---|
| windows examined | 180,592 |
| windows with a survive band | 27,655 |
| representative **is** a fan member | 11,450 |
| representative is **off-fan** | **16,205 (58.597 %)** |
| arms pricing **zero** achievable rolls | 16,205 |

`on_fan + off_fan == bands` is checked in the artifact.

**Read the scope, because this number is easy to over-read.** It is an *arithmetic* census over a
synthetic `(max_damage, health)` plane with uniform weight. It is **not** an incidence over real
boundaries, and the two differ enormously: at head the same engine measures **0 divergences over
31,082 boundaries** across both permitted windows. The census sizes the *mechanism*, and the
sweeps size the *consequence*. Both are needed, and quoting either alone misstates G8.

⚠ **And that zero must be quoted with its two accept bars — item 9 of the ledger's §6, added by
this same pass, forbids the bare number.** Those 31,082 matches include **8.689 % dev / 9.185 %
holdout** accepted on the **hidden-counter support bar**, and **167 dev / 140 holdout (1.077 % /
0.899 %)** that match only because of the ±9 % roll window (§4).

> ⚠ **Two pointer fixes here, both caught in review, and both are this document's own subject.**
> The citation read *"§9 of the ledger"*, which **does not resolve** — the ledger has §1–§8, and
> the second accept bar went in as **item 9 inside §6**. A pointer that does not resolve, in the
> document arguing for scoped citation. (The ledger's own G8 cell cites *"§6 of this document"*
> correctly, so the two disagreed.) And the first bar read *"a widened sleep-counter bar"*, which
> is **narrower than the thing it names**: the gate is `gating:support`, and dev's 1,347
> support-gated boundaries sit alongside a `hidden_counter_support:confusion` counter at **1** as
> well as `hidden_counter_support:sleep` at 1,352 — re-derived from
> `reports/artifacts/c152_head_dev_sweep.json`, and note those two are per-state tallies rather
> than a partition of the 1,347, so they are quoted as evidence that the bar is not sleep-only
> and **not** as its composition. Holdout carries no confusion key at all. The precise name is the
> one the ledger's own cell uses: **support-gated acceptance**. The second bar is **not incidental to G8**: the
dominant class the window absorbs is `roll_scaled_component`, **158 dev / 129 holdout** of those
167 and 140, which is precisely what an off-fan survive representative produces. So an unknown
part of G8's zero is the comparator's tolerance rather than agreement. The honest statement is
narrower than "zero": the mechanism is arithmetically common, its consequence is **not separately
measured beneath the tolerance**, and no divergent row survives it in the two permitted windows.

### 2.3 The row itself, at head

Replayed from the committed C141 artifact at fingerprint `bfdbe1c04876edcd`:

| path | verdict | branches | misses |
|---|---|---|---|
| collapsed (what ships in search) | `diverged` | 9 | 9 |
| enumeration oracle (`POKEZERO_ENUMERATE_ROLLS=1`) | **`matched`** | 416 | **0** |

The oracle result re-derives C147's G33b closure at the current head, after both merges, which
the C147 report could not do because the head has moved twice since.

The **other** C141 row, `19200131/129`, now re-reads **`matched` / 0 misses / 4 branches on the
collapsed path**. Neither of C141's two rows is a live divergence at head.

### 2.4 Disposition: RETIRED, and why not "fix" and not "still open"

`19200244/115` is not a gap. It is a boundary, and it now has three properties it did not have
when the cell was written:

1. **It cannot be re-measured, ever.** It sits inside `19,200,000`–`19,200,259`, which the guard
   `_reject_burned_final_holdout` refuses **unconditionally** — `--final-holdout-i-mean-it` does
   not open it. Its window was demoted by C151 to dev-grade evidence: 200 then-fresh seeds, a
   71-patch engine, self-chosen, terminal for nothing. A row on a burned, demoted window cannot
   be the thing that keeps a ledger open, because no future measurement can close it.
2. **It matches on the certifying path.** C137 made enumeration the oracle. Under the oracle this
   row is `matched` at head, measured above.
3. **Its remaining collapsed-path defect is a documented structural property, now sized.** The
   collapsed cascade prices a fan by representatives; a representative need not be a fan member;
   that is exactly what §2.2 measures. It is not a row-specific bug and there is no row-specific
   fix for it.

**Why not build the fix.** The candidate is the one C150 already filed: snap an off-fan
representative to the nearest fan member. C140 §6a(ii) and C150 both record why it is not
obviously right, and this census adds a third reason: at `max_damage 159 / health 157` the nearest
fan members are **144 and 146, a tie at distance 1, of which only 146 closes this row**. A
principled snap closes this boundary only if the tie-break is chosen to fit the sample. Building
it would also be an engine change with a measured benefit of **zero rows in both permitted
windows**, validated by sweeps that cannot see it.

**Stated scope of the retirement.** The mechanism is real, reachable and measured at 58.597 % of
the synthetic band plane. Its observed consequence is **0 divergent rows in 400 games across the
two permitted windows at head**, and **0 divergent rows in the one historical instance under the
enumeration oracle**. It is retired as a *ledger-blocking* entry and retained as a named property
of the collapsed path. If the collapsed path is ever certified directly rather than via the
enumeration oracle, this comes back.

---

## 3. G33b — the two open arms

C147 shipped `leftovers_slot_truncated` and left two arms open: *"a fatal weather chip when the
winner is faster, and exact speed ties (the engine forks both orders, so one is mislabelled
either way)."* The weather arm was left **"unmeasured, not believed absent."**

### 3.1 The speed-tie refusal still holds — verified, not carried

Both halves of the stated justification re-read from source at head:

* `residual_speed_order` (`third_party/poke-engine-gen3-residual-speed-order.patch`) returns
  `Some(SideOne)` / `Some(SideTwo)` on a strict comparison of `get_effective_speed` and **`None`
  on an exact tie**.
* `add_end_of_turn_branches`, same patch, builds `residual_orders` as
  `vec![SideOne, SideTwo]` when `residual_speed_order` is `None`, generates a candidate per order,
  and keeps both unless `same_residual_outcome` finds the instruction multisets equal. A
  truncation makes the multisets differ, so on a tie **both orders are live branches**.

So the shipped predicate's `_ => NO_TRUNCATION` really is a refusal to guess between two live
orders, not an oversight.

⚠ **One correction to the reason, which does not change the disposition.** The function's doc
comment says *"there is no single answer to give."* That is true of `residual_speed_order`, which
sees only the state — but `leftovers_slot_truncated` is also handed the **segment**, and the two
forks have different segments. Which order a given branch took is recoverable from the order of
the first instruction belonging to each side. So the refusal is a **choice**, not an impossibility,
and the honest form of the comment is "declines to infer the order from the segment" rather than
"cannot know it". Filed as a candidate below; not built.

### 3.2 The weather arm, measured

`leftovers_slot_truncated` was instrumented to emit one line per battle-ending residual
instruction it reaches, **before any arm returns**, so the census sees the whole family rather
than only the part the shipped gate acts on. Whether the fatal instruction is the order-8 weather
chip is decided by a **state predicate**, not by classifying the instruction — which the
function's own comment says is impossible, correctly, because a lethal residual damage always
equals the victim's remaining HP:

> `weather_chips(state, loser)` is `Some` **and** `loser_pre_hp <= max(1, loser_maxhp / 16)`.

Order 8 is the first residual phase that damages, so the loser's HP when its chip fires is still
its pre-residual HP. Corroboration, recorded and not assumed: the terminating instruction's amount
equals `min(chip, loser_pre_hp)` in **23 of 23** weather-fatal calls — the engine caps a lethal
residual at remaining HP, which is why the naive `amount == chip` check matches only 8.

`reports/artifacts/c152_g33b_open_arm_census.json`, over the two permitted windows plus a
1,000-game census on unregistered seeds `1,000,000`–`1,000,999` (below `FIDELITY_SEED_FLOOR`, in
no registered band, and outside every acceptance namespace):

| quantity | dev + holdout (400 games) | all six windows (1,400 games) |
|---|---|---|
| predicate calls at a battle-ending residual instruction | 226 | **925** |
| `order_le_10` / `perish` arms | 226 / 0 | 911 / **14** |
| `loser_first` (the arm the gate acts on) | 112 | 494 |
| `winner_first` (gate declines) | 114 | 407 |
| **exact speed ties** (gate declines) | **0** | **24** |
| loser dies to its own order-8 chip | 23 | **42** |
| …of those, **not gated today** | 9 | **17** |
| …of those, winner holds Leftovers | 9 | **17** |
| **winner-side heals inside a weather-truncated segment** | 0 | **0** |

**Instrument validated against an independent measurement.** Per window, the census's
`loser_first`-with-a-Leftovers-winner count is **52 dev and 56 holdout** — exactly C147's
separately-measured 52 and 56 slot skips, taken on a different throwaway build months of commits
earlier. That is the check that the instrument is counting the thing C147 counted.

**Corroboration of the weather predicate**, recorded rather than assumed: the terminating
instruction's amount equals `min(chip, loser_pre_hp)` in **42 of 42** weather-fatal calls. The
naive `amount == chip` matches only 17, because the engine caps a lethal residual at remaining
HP — which is why the census does not use it as the test.

### 3.3 The weather arm cannot mislabel a heal — a structural demonstration, not a count

The count alone would be a weak retirement, so here is the mechanism.

At a weather-fatal truncation the winner's side can emit, in the truncated segment, only what the
engine emits **before** its weather block. Read off `add_end_of_turn_instructions` in the vendored
engine, in emission order: class 0 side conditions, then **Wish**, then the weather block, then
the speed-major order-10 buckets. The engine's own comment at the Wish block says it *"is in a
class ahead of the weather chip at 8 no matter who is faster"*, and quotes the sim.

So the only positive `Heal` the winner can emit is a **resolving Wish**. And a resolving Wish is
labelled correctly **without the plan**: `residual_heal_cause` tests
`matches!(next_ins, Some(Instruction::DecrementWish(d)) if d.side_ref == side)` first and returns
`move: Wish` before it ever reaches the Leftovers branch. The engine emits `Heal` immediately
followed by `DecrementWish`, which is the adjacency that test relies on.

Therefore the un-gated weather arm's over-booking sets `plan.usable[winner] = false` on a side
whose **only possible heal is fallback-correct**. It cannot produce a wrong heal label. That is
the whole of what G33b is about — the cell's own title is *"the Leftovers heal **slot** is
over-booked"*.

Two further facts, so the demonstration is not narrower than it sounds:

* In every measured instance the winner emitted **zero** heals in the truncated segment
  (`winner_heals_before == 0` in all of them), so the mechanism above was not even exercised.
* Where the winner did emit a **damage** instruction — its own order-8 chip, in the two holdout
  shapes with `seglen 2` — an unusable plan sends that line to `residual_damage_cause`, which
  returns `Sandstorm` from `state.weather.weather_type` for a mon that is not burned, poisoned,
  seeded or partially trapped. **Identical to the planned label.**
* ⚠ **And the case where those labels would differ is one the gate would not fix anyway.** If the
  winner is also burned, `plan.damage[winner]` books `["Sandstorm", "brn"]` against one emitted
  instruction, so the **damage** count mismatches and the side is unusable *whatever the heal slot
  does*. Un-booking the Leftovers heal cannot rescue it. That shape belongs to G33's damage-slot
  family, not to G33b, and saying so is the difference between retiring this arm and pretending
  the family is closed.

### 3.4 The gate is buildable — filed, not built

For the record, because "cannot be fixed" would be false: the weather arm has a clean state
predicate needing no instruction classification —

```
weather_chips(state, loser).is_some()
    && loser_pre_hp <= max(1, loser_maxhp / 16)
    && !has_reserve[loser]
```

— which is decidable before the segment walk and truncates the winner regardless of speed. The
tie arm has the segment-order inference of §3.1. Both are filed as **candidate changes, unbuilt
and unswept**, with a measured benefit of zero rows and, per §3.3, no reachable label consequence.
Shipping a renderer change under C133 §7 discipline requires a registered prediction and four
sweeps to demonstrate a null; that is the cost, and the benefit is measured at zero.

### 3.5 Disposition — and the two arms are not alike

**The weather arm is RETIRED.** Reached 17 times in 1,400 games, all with a Leftovers winner, and
structurally unable to mislabel a heal for the reason in §3.3. Its gate is buildable from the
state predicate in §3.4 and is filed unbuilt: measured benefit zero.

⚠ **The speed-tie arm STAYS OPEN, and §3.3's demonstration does not cover it.** An earlier draft
of this document said it did, and that was wrong. The weather arm is safe *because order 8
precedes every order-10 heal slot*, so the winner's Leech Seed **drain** heal — emitted at the
**loser's** 10.5 — cannot have fired. A tie truncates *inside* order 10, and in the fork where the
loser's bucket runs first the winner's drain heal **is** emitted while its own 10.4 is not.
`plan.heal[winner]` then books Wish/Leftovers/drain against fewer emitted heals, the side goes
unusable, and `residual_heal_cause` answers **`item: Leftovers`** for that drain — it tests the
holder's item *before* its silent-drain empty-string branch, deliberately and for a reason C131
recorded. That is precisely the mislabel G33b is about. ⚠ **The exposed shape is 3 of 20, not 7 of 24 — corrected in review, and the correction is a
scope error of exactly the kind this document is about.** 7 of the 24 tie calls do carry a
winner-side heal before the truncation, but **4 of those 7 are `perish`-arm calls**, and
`leftovers_slot_truncated` returns `NO_TRUNCATION` for the perish reason because Perish Song is
order 12 — *after* all of order 10. Nothing in order 10 is skipped there, so those 4 cannot exhibit
the mislabel at all; that all 4 carry an earlier heal is the expected consequence of the winner's
whole order-10 bucket having already run. The tie population that can be exposed is the
**`order_le_10` ties: 20 calls, of which 3 carry a winner-side heal.** Re-derived from
`reports/artifacts/c152_g33b_open_arm_census.json`'s `by_arm_and_order`
(`order_le_10|tie` 20, `perish|tie` 4) and its stored tie rows, not re-counted by hand. The
disposition is unchanged — the shape is still not hypothetical and no tie-arm divergence was
*observed* in 1,400 games, and "not observed" is not "cannot happen" — but the number was asserted
over a population wider than the one the mechanism can reach.

The tie arm's fix is §3.1's segment-order inference. It is filed unbuilt, with a measured benefit
of zero *observed* rows and a mechanism that is live.

### 3.6 ⚠ And a THIRD arm, which C152 found and which defeats the gate where it does fire

Filed as ledger row **G33c**. The wide census turned up
`1000513/121` — `component_mismatch:heal` / `itemleftovers` at **pct 100.00**,
`observed_only=[('heal', 36)] engine_only=[('itemleftovers', 36)]` — and it is a truncation the
shipped gate **does** fire on (`order=loser_first`, confirmed by a single `C152_TRUNC` line on
that boundary). Side by side:

```
Showdown  |-heal|p1a: Tropius|317/341 psn|[silent]
engine    |-heal|p1a: Tropius|317/341|[from] item: Leftovers
```

Tropius is the seeder, holds Leftovers, is **poisoned**, and is slower than Kangaskhan (effective
speed 151 against 188). The gate un-books its 10.4 Leftovers **heal** — but its `psn` tick at 10.6
sits in the **same skipped bucket** and `ResidualPlan::build` books that unconditionally. So
`plan.damage[winner].len() != emitted_damage[winner]`, `plan.usable[winner]` goes false anyway,
and the drain heal drops to the fallback regardless of what the heal gate did.

**The heal gate is therefore inert wherever the winner also carries a residual status**, which is
a large share of seeders — and that is consistent with C147's own finding that its 108 firings
moved no verdict either way. The fix is small: the flag `leftovers_slot_truncated` already
computes must suppress the winner's post-10.4 **damage** entries too. Unlike G33b's two open arms
it has a **measured** benefit — this row. Not built here.

---

## 4. H8 — the ±9 % fallback window

### 4.1 The measurement

Two arms on one build (fingerprint `bfdbe1c04876edcd`, 74 patches, `d20cf840`):

* **count** — the shipped comparator, behaviour unchanged, with a tally at the window accept;
* **disable** — the window accept removed. Exact equality or membership in the enumerated fan,
  nothing else.

Neither edits `scripts/engine_transition_differential.py`, which is under a certification pin
binding `successor_pending_identity.differential_sha256`. Both are AST rewrites of the shipped
`roll_components_agree` source obtained by `inspect.getsource`, and the rewrite **refuses to run**
unless it finds exactly one window test, matched by structure rather than by text; the count-mode
rewrite additionally re-dumps itself with the inserted tally removed and requires the result to be
`ast.dump`-identical to the original. `scripts/c152_h8_window_census.py`.

**Proved to fire before either sweep**, on five constructed cases, because an instrument that
silently never fires reports zero and looks like a clean result:

| case | shipped | count mode | tally | disable mode |
|---|---|---|---|---|
| fan enumerated, magnitude not in it | accept | accept | `window_accept_legal_miss` | **reject** |
| `pre_legal` unavailable (`legal is None`) | accept | accept | `window_accept_legal_none` | **reject** |
| exact fan membership | accept | accept | — | accept |
| exact equality | accept | accept | — | accept |
| magnitude far outside the window | reject | reject | — | reject |

| window | measured | matched, shipped | matched, window removed | **accepts that depended on the window** |
|---|---|---|---|---|
| dev | 15,503 | 15,503 | 15,336 | **167 — 1.077 %** |
| holdout | 15,579 | 15,579 | 15,439 | **140 — 0.899 %** |

### 4.2 The cell's stated mechanism is wrong, and that is the finding

H8 says the window *"applies whenever `pre_legal` is unavailable."* Measured usage, per component
comparison over the same 400 games:

| door | dev | holdout |
|---|---|---|
| `window_accept_legal_none` — `pre_legal` unavailable, the door the cell names | **0** | **0** |
| `window_accept_legal_miss` — fan enumerated, magnitude simply not in it | **190** | **181** |

`strict:no_damage_rolls` being 0 was read in the cell as bounding *only the state-level part* of
the fallback. It is not a partial bound: it is the **whole** of the door the cell names, and that
door contributes nothing. All 371 window accepts came through the other one.

**Usage is not dependence.** 190 accepts against 167 boundaries flipped, because a boundary
matches if *any* branch matches, so a window accept on a non-decisive branch costs nothing. The
quotable number is the difference, not the tally.

**What the window is absorbing**, from the disabled arm's classes: dev 158 `roll_scaled_component`
+ 5 `limit:roll_divergent_lethality` + 4 component-level; holdout 129 + 9 + 2.

### 4.3 Disposition

**CLOSED — measured.** And it produces a **second accept bar** for §6 of the ledger, alongside
support-gated acceptance at 8.689 % / 9.185 %: about **1 boundary in 100** matches on a
proportional tolerance rather than on exact fan membership. The ledger's recommended claim wording
now carries it. Scope: two 200-game windows, collapsed roll path, strict matcher; nothing about
other seed ranges and nothing about the enumeration oracle, where the fan is emitted per roll and
the window has less to do.

---

## 5. H12 — the skip counters and the coverage shortfall

### 5.1 The cell's own settling measurement is scoped to the wrong population

H12 cites `reports/c43_coverage_shortfall_diagnosis.json`: *"~7,224 rows invisible to any skip
counter … no repair list built from them can be complete."* Its named settling measurement is
*"instrument the single-seat arm with the same exit taxonomy and re-run."*

**That measurement cannot settle this claim, because c43 explicitly excluded the single-seat
population from the number.** From the artifact's own `decomposition.note`, quoted verbatim:

> `"skip:single_seat_boundary (89,887) is excluded -- a single-seat boundary is not a full round and is not in the denominator"`

Re-derived arithmetic, from the same file rather than from the ledger's summary of it:
`boundaries_full_round 821,320 − boundaries_measured 787,376 = 33,944`; the ten ranked counters
sum to `26,720`; `33,944 − 26,720 = 7,224`. Every term is inside the **full-round** path.
Instrumenting the single-seat arm would measure a disjoint population.

⚠ **And c43's own arithmetic is wrong by 372, in the direction that understates it.** Its ranked
list includes `strict_all_branches_lossy: 372` among the counters that "account for" the
shortfall. That counter fires **after** `boundaries_measured` increments — it is a post-measure
verdict, pinned as such in `tests/test_boundary_verdict_partition.py`'s
`VERDICT_PARTITION_SKIP_COUNTERS` — so it is not a coverage exit at all. Excluding it gives
`26,720 − 372 = 26,348` accounted and `33,944 − 26,348 = **7,596**` unaccounted for that era, not
7,224. This is recorded rather than repaired: c43 is an artifact of its era and is not edited.

### 5.2 The claim itself, re-derived at head

The claim H12 makes is that the counters **do not sum**. Measured at head, on the two sweeps of
§1, with the exit set taken from `tests/test_single_seat_coverage_bound.py`'s own allowlist:

| window | measured | in-path exits | sum | `boundaries_full_round` | closes? |
|---|---|---|---|---|---|
| dev | 15,503 | 465 | 15,968 | 15,968 | **exactly** |
| holdout | 15,579 | 576 | 16,155 | 16,155 | **exactly** |

and every ply outside the full-round path is counted: `skip:single_seat_boundary` 1,742 / 1,813,
`abort:no_legal_action` **0 / 0**, so `full_round + single_seat` = 17,710 / 17,968, the totals §1
reports. The verdict partition closes independently in both windows.

There is no population invisible to a counter. **H12 is CLOSED — measurably false in the current
era**, over the two 200-game windows at fingerprint `bfdbe1c04876edcd`.

### 5.3 What is NOT closed by this, said explicitly

The single-seat population is **visible** (it has its own counter and its own pinned reconciliation)
but **uncompared** — 1,742 and 1,813 boundaries, 9.84 % and 10.09 %. That is **H1**, the coverage
gap, and it is a different ledger row that stays open. Moving H12 to CLOSED must not be read as
progress on H1, and the ledger cell now says so.

---

## 6. H19 — the four never-adjudicated families

### 6.1 The cell's evidence is vacuous, and its settling measurement could not run

H19 records **UNKNOWN whether they survive into the current era**, on the evidence that *"none of
their labels appears in the c136 counters."*

**No sweep has ever emitted a family label into a counter.**
`scripts/engine_transition_differential.py` neither imports `cert_sweep_readout` nor calls
`classify_row` — measured, and re-derived by
`scripts/c152_h19_family_recensus.py --check` on every run. A divergent row gets a
`divergence_class`; the `I*`/`LS_*` families are a **second** classification pass applied
afterwards by `cert_sweep_readout.classify_row` to a row's recorded `divergence_class`, `protocol`
and `branch_misses`. Absence from a vocabulary the layer never writes to is not evidence.

**And the named settling measurement crashed on every input.**
`scripts/family_bucket_audit.py:355` read `(ROOT / evidence).is_file()`, and `ROOT` is defined
nowhere in that module — only `REPO_ROOT` is. The line is reached **unconditionally**, because all
five `ESTABLISHED` families are members of the registered set, so `main()` did all the re-read
work and then raised `NameError`. `tests/test_family_bucket_audit.py` exercises `signatures()` and
`bucket_from_signatures()` and never `main()`, which is why this survived from #1022 (2026-08-02)
to here. **Fixed in this branch**, and pinned so it cannot recur.

### 6.2 Measured three ways

`reports/artifacts/c152_h19_family_recensus.json`.

**(a) As recorded, over the widest glob available** — `reports/**/*.json` plus `docs/**/*.json`,
**78** artifacts carrying `repros`, **1,167** recorded divergent rows, each classified by the
shipped `cert_sweep_readout.classify_row` using its own artifact's recorded `divergence_class` and
`branch_misses`. Two artifacts are **excluded** from that history with the reason recorded in the
JSON: the window-disabled `c152_h8_nowindow_*` pair is a **mutant comparator** run only for H8, and
leaving it in inflated `I2_matcher_accounting` from 85 to 113 — a census must not absorb its own
instrument. All four families have fired. Their highest-C appearances:

| family | rows ever | artifacts | highest-C artifact | rows there |
|---|---|---|---|---|
| `LS_capped_lethal_shape` | **181** | 55 | `reports/artifacts/c152_wide_census_1000250_sweep.json` | 1 |
| `I2_matcher_accounting` | 85 | 16 | `reports/artifacts/c141_final_holdout_sweep.json` | 1 (`19200131/129`) |
| `I5_boundary_truncation` | 65 | 30 | `reports/artifacts/c137_encore_transform_holdout_sweep.json` | 1 (`19100180/24`) |
| `I3_roll_inherited` | 23 | 7 | `reports/c13_batch_e_differential.json` | 3 |

⚠ **CORRECTED IN REVIEW.** This block read *"74 artifacts … 1,156 … `LS_capped_lethal_shape` 180,
last in `c149_base_dev_sweep.json`"*. Those figures were **correct when taken and false once the
tree moved**: they came from a run made before this PR's own four wide-census shards were
committed, and the shards then joined the glob. Re-derived from
`reports/artifacts/c152_h19_family_recensus.json` rather than re-typed. §7.3 and the ledger's H19
cell already carried the corrected numbers, so this document disagreed with itself and with its
own artifact — exactly the drift the ledger's standing rule about re-deriving after a merge exists
to catch, hit by this document while writing about it.

⚠ **And `LS_capped_lethal_shape` is therefore NOT extinct.** Its highest-C row is now one row on
**unregistered seeds** `1,000,250`–`1,000,499`, measured on this same 74-patch engine. Zero in the
two permitted windows is not zero everywhere; see §7.3.

So the honest answer to "did they survive into the c136 era" is **two of the four did**:
`LS_capped_lethal_shape` carried `19000074/27` and `19000191/63` in both c136 dev sweeps, and
`I5_boundary_truncation` carried `19100180/24` in the c136 main holdout. `I2` and `I3` carried
nothing in c136.

**(b) Re-read at head.** Each of those last rows replayed through the current build:

| row | family as recorded | verdict at head |
|---|---|---|
| `19000191/63` | `LS_capped_lethal_shape` | closed by C149's split — dev `transitions_diverged` 1 → 0 |
| `19200131/129` | `I2_matcher_accounting` | **`matched`**, 0 misses, 4 branches |
| `19100180/24` | `I5_boundary_truncation` | **`matched`**, 0 misses, 1 branch |
| `19100107/135`, `19100191/5` | `limit:roll_divergent_lethality` | **`matched`** |

⚠ **A re-read is not a re-sweep, and two rows show why.** `19100170/71` and `/72` still re-read as
diverged, because `reread_row` replays the state **recorded in the artifact** and their fix
(`d27316b6`, #1148) changed **world construction**, not the engine — the recorded state still
encodes the pre-fix world. C145 closed them by bisected one-game sweeps, which is the right
instrument for a world-construction fix. Both facts are in the artifact; neither is a family row
(both classify `unattributed_generic`).

**(c) Live.** Running `scripts/family_bucket_audit.py --rows` over the 13 committed c136 divergent
rows — H19's named measurement, now that it runs — returns **0 rows for every one of the 21
registered families**, with 9 of 13 rows cleared on re-read and the remaining 4 all
`unattributed_generic`. And the head sweeps of §1 measure **0 divergent rows in both windows**, so
every registered family is 0 by construction.

### 6.3 Disposition

**H19 is CLOSED.** All four families are at **0 rows** in both permitted windows at head; the last
row of each is named, dated and re-read above. The cell's UNKNOWN was not a hard question — it was
an unrunnable script and a vacuous piece of evidence.

**Scope.** "Current era" here means the two 200-game permitted windows at fingerprint
`bfdbe1c04876edcd`. It is not a claim about the seed space at large — see §7.

---

## 7. What this document does not settle — and what it opened

### 7.1 Open after C152, exactly

1. **G33b's speed-tie arm.** Measured for the first time — 24 predicate calls in 1,400 games, all
   with a Leftovers winner. **20 of the 24 are `order_le_10`**, the only ones a truncation can
   expose, and **3 of those 20** carry a winner-side heal that an over-booked plan sends to a
   fallback answering `item: Leftovers`. (The other 4 are `perish`-arm calls, order 12, where
   nothing in order 10 is skipped — see §3.5.) No tie-arm divergence observed. Fix filed unbuilt
   (§3.1's segment-order inference).
2. **G33c, new.** The truncation strands the winner's order-10 **damage** bookings, so C147's heal
   gate is inert wherever the winner carries a residual status. Observed at `1000513/121`,
   diagnosed to the line, reproducible from `reports/artifacts/c152_wide_census_1000500_sweep.json`.
   Fix filed unbuilt, and unlike the two G33b arms it has a **measured** benefit.
3. **G8's second remainder.** The two `defender_active.hp`-ceiling `residual_disjoint_bands` call
   sites, left alone by C149 for blast radius. Still unmeasured; §2.2 measured the first remainder
   only.
4. **H1**, unchanged: 1,742 and 1,813 single-seat boundaries, counted and uncompared.
5. **New ledger §7 item 10:** the single-seat population has no taxonomy anywhere in `reports/` or
   `docs/` — no sub-keyed counter exists and nothing emits one.
6. **G0 and every other §3 row C152 did not touch.** 81 rows now; C152 disposed five of them.

### 7.2 ⚠ Two more false "never fired" claims, and what refuted them

`skip:rump_branch_set` and `strict:branch_event_legal_error:BranchLegalRollError` were both on the
ledger's never-fired lists and both pinned as absent by
`tests/test_never_fired_counter_census.py`. **Both are false.** They fire in the C152 wide census:
`rump_branch_set` at 2 and 1, `BranchLegalRollError` at 18, 8 and 1. That is the **sixth and
seventh** instance of this defect class in `reports/c138_known_gaps_ledger.md`, after the five
C146 inventoried.

**What refuted them matters more than that they fell.** Neither was refuted by re-reading the
existing corpus — over every artifact committed before C152 they really are 0, and the census pin
re-derives that on every run. They were refuted by **measuring somewhere new**: 1,000 games on
unregistered seeds, run for an unrelated purpose. So the standing rule *"a negative claim carries
its glob"* needs a companion, and C152 adds it to the ledger's §8:

> **A negative measured only inside the two permitted windows is a claim about those windows.**
> Widening the *corpus* cannot find this class of error. Only widening the *measurement* can.

Both names are moved to a `_FIRED_ONLY_OUTSIDE_THE_PERMITTED_WINDOWS` list with a pin asserting
they **do** fire and that their evidence is still confined to the wide census — so if either ever
appears in a dev or holdout sweep, that is a new fact and goes red.

### 7.3 The wide census, stated as what it is

1,000 games at seeds `1,000,000`–`1,000,999`, four 250-game shards, on the **throwaway
instrumented build** `89797289…` (the shipping tree plus two `eprintln!` blocks and nothing else).
Those seeds are below `FIDELITY_SEED_FLOOR`, in no registered band in
`docs/engine_divergence_ledger_20260728.md`'s partition, and in no acceptance namespace.

| shard | measured | matched | diverged | classes |
|---|---|---|---|---|
| `1,000,000` | 19,922 | 19,918 | 2 | `roll_scaled_component` 2 |
| `1,000,250` | 19,865 | 19,858 | 6 | `roll_scaled_component` 2, `component_extra_in_engine:itemleftovers,sandstorm` 1, `component_mismatch:heal`/`itemleftovers` 1, `component_missing_in_engine:leechseed` 1, `limit:roll_divergent_lethality` 1 |
| `1,000,500` | 20,114 | 20,111 | 3 | `roll_scaled_component` 1, `component_mismatch:heal`/`itemleftovers` 1, `limit:roll_divergent_lethality` 1 |
| `1,000,750` | 20,538 | 20,537 | 1 | `roll_scaled_component` 1 |
| **total** | **80,439** | **80,424** | **12** | 0.0149 % of measured |

⚠ **This is not fidelity evidence and must never be quoted as a divergence rate for the program.**
It is an unregistered band, on an instrumented build, with no registered prediction. What it is
good for is exactly what it did: it found G33c, it refuted two never-fired claims, and it shows
that **`LS_capped_lethal_shape` is not extinct** — its highest-C row is now
`reports/artifacts/c152_wide_census_1000250_sweep.json`, one row, on the same 74-patch engine that
measures zero in both permitted windows.

⚠ **So read the two zeros correctly.** Dev and holdout are at 0 divergences at head. The engine is
not divergence-free. Those windows are the two the program has iterated against for its entire
history, and a shape that is invisible in them is not absent.

### 7.4 Not measured here

* Nothing was measured at or above `19,200,000`. The only contact is the replay of §2.3.
* No engine or renderer change was built. Four fixes are filed unbuilt: G8's snap-to-fan (argued
  against), G33b's weather predicate and tie inference (measured benefit zero), and G33c's
  damage-slot suppression (measured benefit one row).
* The G33b census counts **predicate calls**, not distinct boundaries — `ResidualPlan::build` runs
  per branch, so one boundary contributes several. C147's reach census has the same property and
  the two are compared on the same basis.
* §2.2's census is arithmetic over a synthetic uniformly-weighted plane. It sizes a mechanism and
  says nothing about incidence.
