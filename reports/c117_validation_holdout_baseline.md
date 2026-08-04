# C117 — the validation holdout, and what it says about the 208 → 7 trajectory

C116 Phase 1 item 5. The first out-of-window measurement in this program's history.

Era: engine fingerprint `8a00d812b41566a0`, 58 patches, mass gate
(`tests/test_branch_mass_reconstruction.py`) green at 4 tests. Artifacts
`/tmp/sweep_holdout.json` (holdout, `source_commit 89bbabe4`) and `/tmp/sweep_ca.json`
(dev window, `source_commit 6ba1145b`).

**The two artifacts have different source commits and the same engine fingerprint.**
That is what makes the comparison valid, and it should be stated rather than left to a
reader: `git diff 6ba1145b 89bbabe4 -- scripts src rust third_party` shows
`engine_transition_differential.py` and the entire search crate untouched.

**But that diff is not only path helpers, and two earlier revisions of this paragraph said
it was.** Within the very pathspec this report invites the reader to run, it also carries
`third_party/poke-engine-gen3-residual-lethality-partition.patch` **+55** — the #1069 Case A
three-way arm — and `src/pokezero/paths.py` **+45**, a new file. `6ba1145b` is #1066, and
#1069 landed later at `795a1f90`, yet `sweep_ca.json` reports 7 divergences and the
post-#1069 fingerprint, because it ran from #1069's branch base with the patch present but
uncommitted. A reader running that command finds an engine change this sentence did not
mention and concludes the report is wrong.

**The conclusion survives on the other argument, which is the stronger one: `source_commit`
is not an engine identifier — the fingerprint is.** `compute_fingerprint()`
(`engine_build_fingerprint.py:145`) hashes patch bytes *and* crate sources, and each run
nulls its own stamp unless it equals its run-time recomputation. Both runs' stamps are
non-null and equal, so the engines were byte-identical regardless of what the commits differ
by. That is the sentence that was missing, and it makes §1 stronger rather than weaker. The
fingerprint is a real engine stamp, not the C111 v1 mix-up: the differential nulls the
venv stamp unless it equals `compute_fingerprint()` recomputed from tracked patch bytes
at run time (`scripts/engine_transition_differential.py:2493-2500`).

The holdout also carries `abort:max_steps: 1` — one of the 200 games was truncated
(`measured_fraction_of_full_rounds` 0.9530 against the dev window's 0.9534). Recorded
because "0 engine errors" alone reads as "nothing anomalous".

---

## 1. The headline: the dev window is flattering by 3.5×

| window | seeds | boundaries | matched | diverged | rate | errors |
|---|---|---|---|---|---|---|
| dev | 19000000–19000199 | 15,224 | 15,217 | **7** | 0.0460% | 0 |
| **validation holdout** | 19100000–19100199 | 15,396 | 15,371 | **25** | **0.1624%** | 0 |

`matched + diverged == boundaries_measured` on both. The holdout divergence rate is
**3.53×** the dev window's.

**This is C116's M6, confirmed rather than feared.** The entire 208 → 7 era iterated
against seeds 19000000–19000199, and no holdout existed anywhere in the program. The
plan said the stopping condition "can be satisfied by a window overfit"; it can, and
partially was. Any statement of the form "the residue is 7" describes *one particular
200-game window* and must not be read as a fidelity rate.

That is the honest headline and it is recorded before any mitigation below.

## 2. But the excess is concentrated, not diffuse

Two structural facts change the interpretation.

**Class concentration.** Eleven of the twenty-five rows — 44% — carry a single class:

| count | class |
|---|---|
| **11** | `limit:world_sample_drag_target` |
| 4 | `roll_scaled_component` |
| 2 | `component_missing_in_engine:sandstorm` |
| 2 | `limit:roll_divergent_lethality` |
| 1 each | `component_extra_in_engine:itemleftovers,psn,sandstorm`; `component_extra_in_engine:spikes`; `component_mismatch:itemleftovers|leechseed`; `component_missing_in_engine:itemleftovers,movewish`; `component_missing_in_engine:leechseed`; `component_missing_in_engine:psn` |

**Game concentration.** Four games produce fourteen of the twenty-five rows:
`19100122` alone produces **seven** (steps 13, 117, 147, 151, 164, 169, 181), plus
`19100180` three, and `19100072` and `19100142` two each. The 25 rows fall in **15
distinct seeds, so 185 of the 200 games produce no divergence at all** — confirmed by
`divergent_transitions_per_game: 0.125` = 25/200. (An earlier revision said "twenty-one
games", which was wrong and understated how clean the window is. Corrected by review.) So this is not a uniformly worse window; it is a window
containing a few games that exercise a mechanism the dev window barely touches.

## 3. The eleven drag rows are B1, not comparison limits — one verified at source

`19100122/13` replayed in full. Whirlwind drags p1's Sableye out and Shuckle in:

```
|-damage|p1a: Sableye|210/239|[from] Spikes
|drag|p1a: Shuckle|Shuckle, L98, F|172/198 slp
|-damage|p1a: Shuckle|148/198 slp|[from] Spikes
|-heal|p1a: Shuckle|160/198 slp|[from] item: Leftovers
observed p1: itemleftovers=+12, spikes=-29, spikes=-24
```

The engine emits **five** arms at 20% each — it *does* enumerate the candidate set —
and one of them carries the realised target with components
`spikes=-29, rolled move=-24`.

**So the divergence is not the drag target. It is that the second Spikes tick renders
as `move` (rolled) instead of `spikes` (exact).** The engine found the right
replacement; the renderer mis-tagged its Spikes damage. That is exactly cause **B1**,
which C111 v2 recorded from the dev window's `19000008/54`, where the same class name
was applied to a row in which "the engine dragged the *same* Pokémon and only a
component tag differed".

**Attribution status: the hypothesis is now DISCHARGED, 11 of 11.** An earlier revision
of this section recorded one verified row and ten owed replays. Review examined all
eleven and every one carries the identical signature — observed has an exact `spikes`
component for the dragged-in Pokémon, and some engine arm carries the same magnitude
untagged, with the magnitude matching that specific Pokémon's `maxhp/8` (Shuckle 24,
Donphan 285/8 = 35, Medicham 230/8 = 28, Cloyster 216/8 = 27, and Victreebel
275/6 = 45 at **two** layers — the two-layer divisor is 6, not 8, which an earlier
revision's blanket "maxhp/8" got wrong).

**Two of them falsify the limit reading by arithmetic alone.** `19100122/169` and
`/181` have `branch_count: 1` at 100% mass — there is no drag-target ambiguity for a
limit to be *about*. So the "if B1 covers all eleven" line below is the actual reading,
not the optimistic one.

Two precision corrections to this section: the mis-tag is an *empty / unattributed*
source (`source == ''`), not literally the string `move`; and the claim that the Shuckle
arm carries `spikes=-29` is not shown by that arm's output — its miss line breaks at the
roll check so its exact components were never printed. The `-29` is provable instead
from the Ho-Oh arm, whose `observed_only` omits it. Ho-Oh is Flying and therefore Spikes-
immune, which is why **one of the five** arms carries no chip. (An earlier revision said
this was "why only five arms carry chip". Ho-Oh's immunity explains the *missing* chip, not
the arm count — that is just the five alive reserves.)

The renderer site is now pinned: `rust/pokezero-search/src/events.rs:2383` renders a
defender `Damage` in the ordinary move path as a bare `|-damage|` with no `[from]`.
Both other paths already special-case Spikes (`events.rs:1055`, and the Sleep Talk walk
at `events.rs:1706-1741`); the phazing path was the one missed. And the classifier
really does short-circuit: `scripts/engine_transition_differential.py:1761` returns
`limit:world_sample_drag_target` on the mere presence of a `|drag|` line, before any
component test — so that class name can never distinguish a target limit from anything
merely co-occurring with a drag.

If B1 accounts for all eleven, the holdout residue is **14** and the rate 0.0909% —
still **2.31×** the dev window like-for-like, so the overfit finding survives the
mitigation entirely. (An earlier revision said 1.98×, comparing the post-B1 holdout against
the **pre**-B1 dev window. #1081 has since measured both — dev 7→6, holdout 25→14 — so the
honest ratio is 14/15396 ÷ 6/15224 = 2.31×. The error understated this report's own thesis.)

## 4. What this changes about the plan

**B1's priority inverts.** It was one row on the dev window and last in the Phase 3
queue. On the holdout it is the single largest class and 44% of all divergence. Ordered
by rows × search impact — the program's own ranking rule — it is now first, and it is
cheap: a renderer tag fix plus a classifier fix that stops applying the drag limit on
the mere presence of `|drag|`, measured separately per the program's rule.

**The remaining fourteen, re-filed from replays rather than from class names.** An
earlier revision assigned these families by reading the class strings, and review found
four of the assignments wrong — which is exactly the failure mode
`limit:world_sample_drag_target` has now caused twice.

| rows | class | actual cause |
|---|---|---|
| `19100002/53`, `19100154/75` | `missing:sandstorm` | **A1**, not A4/A7. `19100002/53` has `branch_count: 1` — there is no fan to partition. One side faints *to the sandstorm tick* and the engine omits the other side's tick. |
| `19100107/135`, `19100191/5` | `limit:roll_divergent_lethality` | **A8 — NEW CAUSE: the residual mirror reads status PRE-move.** Not A2. |
| `19100181/45` | `extra:itemleftovers,psn,sandstorm` | **A1**, not A2 — the engine ran a residual phase Showdown did not. |
| `19100193/46`, `19100014/35` | `mismatch/missing: leechseed` | a **renderer tag** defect like B1, not A2. The engine tags an 18 HP heal `leechseed`, but `290/16 = 18` is Leftovers and Miltank's sap would be `273/8 = 34`. |
| `19100148/76`, `19100179/21` | `roll_scaled_component` | **collapse tax** — Pain Split's amount is a function of the arm's representative roll, so a collapsed fan mis-prices it: 3 HP on `19100148/76` (−20 vs −23) and 5 HP on
`19100179/21` (−41 vs −36) — an earlier revision quoted the 3 for both. Absorbed by Phase 2 enumerate-then-merge. |
| `19100072/17`, `/19` | `roll_scaled_component` | **NEW CAUSE** — Belly Drum at +6, below. |
| `19100180/24` | `extra:spikes` | unowned. The engine applies `spikes -32` to p1 where Showdown applies none — a side-condition state divergence, not A2. |
| `19100113/62` | `missing:itemleftovers,movewish` | **A9 — NEW CAUSE: a planned Wish heal is never emitted.** Not A2. |
| `19100012/61` | `missing:psn` | **A5** — contact-ability trigger precedes the same-turn wake. Protocol carries `[from] ability: Poison Point` with `[from] move: Wrap`, the same signature as C111's `19000125/226`. |
| `19100122/13,117,147,151,164,169,181`, `19100142/21,/22`, `19100180/7,/40` | `limit:world_sample_drag_target` | **B1**, 11/11 (§3) |

### A8 — the residual mirror reads status from the PRE-move state

`19100107/135` and `19100191/5`. An earlier revision filed both under A2 as "burn-residual
kills, A2's explicitly named unmirrored-Burn case". **That is false, and the error is worse
than a misfiling: A2's mechanism no longer exists.** #1066 replaced
`pending_residual_damage` with `residual_phase_final_hp`, which *does* mirror the 10.6
status tick — `generate_instructions.rs:1521-1523`,
`PokemonStatus::BURN | PokemonStatus::POISON => hp -= cmp::max(maxhp / 8, 1)` — and Burn is
not among the gaps its own doc comment enumerates. So filing these to A2 pointed at a cause
whose fix had already shipped, which would let a reader conclude the rows were covered by
queued work. That is an M2 failure: transcribed from C111 rather than derived at this era.

The artifact hands over the real mechanism. In **both** rows the burn is inflicted **by the
very move being partitioned**: `19100107/135` is `|move| Sacred Fire → |-damage| →
|-status|p1a: Roselia|brn`, and `19100191/5` is `|move| Fire Blast → |-damage| →
|-status|p2a: Ninjask|brn` where Ninjask had just switched in at 225/225 with no status.
`residual_lethality_threshold` is called with `state` at `:3204`, i.e. before the move
applies its own secondary, so a status **the move itself inflicts** is invisible to the
threshold. The mirror is complete; the *timing of the read* is wrong.

This is adjacent to A3 (the pre-switch read) and the same shape — a threshold computed
against a state that the turn has not finished producing — but distinct: A3 was fixed by
moving the binding later within the call, whereas the move's own secondary lands later
still, inside `run_move`. Whether a correct threshold is even computable before the
secondary is decided is an open question, and it belongs in the Phase 2 decision: a
per-roll evaluator that runs the phase *after* the move would settle it by construction.

### A9 — a planned Wish heal is never emitted

`19100113/62`. An earlier revision filed this as "A2 — unmirrored Wish", and then said in
its own next clause that #1066 ordered Wish at step 7. Both cannot be true. Wish **is**
mirrored (`:1473-1477`), so the label was self-contradicting.

It is also not a threshold divergence at all. At 84.38% mass the roll check **passes**, and
the miss is `observed_only = [('itemleftovers', 15), ('movewish', 126)]` with
`engine_only = []` — the engine emitted **no** p1 end-of-turn heal, while the pre-state is
Registeel 253/253 with `side.wish == (1, 126)`. The engine *knows* about the Wish and does
not render its resolution. The renderer does plan both heals (`events.rs:3277`, `:3292`),
and the drag walk this report cites at `events.rs:1706-1741` is documented "DECREASES
ONLY, deliberately … an unrendered rise leaves the row divergent" — which is the first
place to look.

**A correction to this table, which an earlier revision of it got wrong.** That revision
filed `19100012/61` and `19100113/62` into the B1 row, which made the table's drag count
read **13** against §2's histogram of **11** — a contradiction inside the document that
is meant to be the authoritative record. Neither row carries a `|drag|` line, and that
is provable from the classifier rather than from a reading: `engine_transition_differential.py:1761`
short-circuits to `limit:world_sample_drag_target` on the *mere presence* of `|drag|`,
so any row with a different class cannot contain one. The same revision also said "the
`movewish` row is `19100181/45`", which was wrong — exactly one row carries `movewish`
and it is `19100113/62`. Both errors were introduced while fixing the previous round's
errors, which is its own lesson: a correction pass needs the same derivation discipline
as the original.

**A NEW CAUSE DID APPEAR, and an earlier revision of this section denied it.** That
revision said "no new mechanism has appeared". It was refuted by two rows of this
report's own artifact.

`19100072/17` and `/19`, both `roll_scaled_component` — the group this report called
unowned. Linoone at 131/261 uses Belly Drum; Showdown emits `|move|Belly Drum|[still]`
followed by `|-fail|`, while the engine pays `('', -130)` = `maxhp/2`. HP is *above*
half (131 > 130), so Showdown's failure cannot be the HP clause — it is the
`boosts.atk >= 6` clause, and the engine state confirms `attack_boost = 6`.

Verified at source: `third_party/poke-engine-src/src/gen3/choice_effects.rs:643-645`
computes `let boost_amount = 6 - attacking_side.attack_boost;` and then guards only on
`if attacker.hp > attacker.maxhp / 2`. There is **no** `attack_boost` guard, so at +6 the
engine pays half its HP for a boost of zero. Two rows, 8% of this window, and
search-relevant: the engine believes it loses 130 HP for nothing.

Extra sting, and an M2 instance in landed code:
`third_party/poke-engine-gen3-bellydrum-roll-gate.patch` quotes only *half* of
Showdown's predicate (`target.hp <= target.maxhp / 2`) and on that basis calls the
engine "faithful on both parities".

**The compression claim survives, and here is the honest count.** An earlier revision
said "23 of 25 attribute to existing causes", which the table directly above it
contradicted — the same optimistic-subtraction shape that produced "no new mechanism":
I subtracted only the two Belly Drum rows and ignored the five other rows the table
marks as unnamed or unowned.

Derived from the corrected filings: **15 of 25 rows land on the six causes named in
C111** — 11 B1, 3 A1, 1 A5 — and **10 do not**: 2 the Belly Drum cause (**A10**), 2 the
pre-move status read (**A8**), 1 the unrendered Wish (**A9**), 2 an unnamed renderer
mis-tag (leechseed-for-Leftovers), 2 Pain Split collapse tax, and 1 unowned
(`19100180/24`, a side-condition state divergence).

**A10 is Belly Drum**, named here because an earlier revision called it "A7b" — a label
that appeared once, was defined nowhere, and collided with C111's A7 (*a collapsed lethal arm
discards the clamped sap*), an unrelated mechanism. A dangling sub-letter under the wrong
cause is worse than no label. Its derivation is in §4 below.

**So the number of causes rose by three, not one**, and two earlier revisions of this
paragraph said one and then said it again after being corrected. The honest summary is
weaker than the one this report wanted: 25 holdout rows resolve into **nine buckets** — three of C111's six causes (B1, A1, A5;
A2, A4 and A6 have no holdout rows at all), three new ones (A8, A9, A10), and three sitting
outside both: the leechseed renderer mis-tag ×2, Pain Split collapse tax ×2, and one unowned
row. An earlier revision glossed this as "six existing causes plus three new ones", which
does not account for five of its own rows. That is still very far
from "25 investigations" — the compression is real — but "the number of causes rose by
one" was reached by filing three rows to a cause whose fix had already shipped.

## 4b. Era note — this baseline is no longer the current state

**#1081 has since closed all eleven B1 rows**: dev 7 → 6, holdout 25 → 14, zero newly
divergent on either, `limit:world_sample_drag_target` → 0 on both. So the 25 in §1 is **the
baseline as measured on 89bbabe4**, not `main`'s residue. The 3.53× overfit ratio is
unaffected — one engine, two windows — and the like-for-like post-fix ratio is 2.31×.

#1081 shipped only B1's **renderer** half; the classifier fix this report asks for is still
owed.

## 5. What is owed

1. ~~Replay the remaining ten drag rows~~ — **done**: B1 discharged 11 of 11 (§3), and
   #1081 has since closed all eleven.
2. ~~Replay the four `roll_scaled_component` rows~~ — **done**: 2 are the Belly Drum cause,
   2 are Pain Split collapse tax (§4). The only unowned row is `19100180/24`.
3. **B1's renderer half shipped in #1081** (dev 7→6, holdout 25→14). The **classifier**
   half is still owed: `engine_transition_differential.py:1761` short-circuits on the mere
   presence of `|drag|`, so the class can still mask a future defect the same way.
4. Investigate **A8** (is a correct threshold computable before the move's secondary is
   decided?) and **A9** (why the planned Wish heal is not rendered).
5. Leave `19,200,000+` untouched. Per the §J.7 amendment it must appear in **exactly
   one** measurement in the whole record; this report deliberately does not touch it.

## 6. A note on what this measurement cost

Nothing. It is 200 games and **13.6 minutes** (`elapsed_seconds: 817.91`; an earlier
revision said nine, which was the dev window's 611.84 s — itself 10.2 minutes, so "nine"
was not even the number being misremembered), and it should
have been run months ago —
before the first fix, as a baseline. The reason it was not is that the dev window's
counter was falling, and a falling number is a poor prompt to ask whether you are
measuring the right thing. The plan was right to make this Phase 1 rather than Phase 4.
