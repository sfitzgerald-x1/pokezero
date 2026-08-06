# C133 — the collapsed-roll residue: three sub-mechanisms, none of them a limit

C116 Phase 4 item 12 requires every remaining divergence to be disposed of as an engine fix, a
harness fix, or **a limit with a written demonstration**. This report disposes of the four
collapsed-roll rows. **None is a limit**, and the `limit:` prefix two of them carry is a classifier
label rather than an adjudication.

No code change ships with this report. It exists so the next attempt starts from measurement instead
of hypothesis — the previous engine fix in this family failed its own falsifier by 48 rows.

## 1. The `limit:` label is unearned

`19100107/135` and `19100191/5` are classified `limit:roll_divergent_lethality`. I have repeatedly
described that as disposing of them. **It does not.** Three of this repository's own records say the
label is wrong for this family:

- `scripts/family_bucket_audit.py:58` buckets it as **"engine-gap (partially resolved)"**, and `:12`
  defines engine-gap as the engine's branch support not containing the observed transition.
- `reports/c105_retract_limit_overclaim.json` records the limit label as **"8-for-8 falsified"** across
  the eight rows it was applied to — and `19000191/63` is explicitly one of the eight.
- The audit's engine-gap signature — *a legal roll range straddling a discrete threshold while the
  emitted arm sits on one side of it* — holds for all four rows.

## 2. The shared shape

In every case the observed damage roll **is a member of the engine's own 16-roll fan**, and it lies on
the far side of the residual-lethality threshold from the arm the engine emitted.

Worth stating because I got it wrong first: `calculate_damage(..., DamageRolls::Max)` returns
**maxima**, not roll endpoints; the fan is derived at `generate_instructions.rs:3343` as
`raw_damage * random / 100`. On `19000074/27` the crit fan is
`[214, 216, 219, 221, 224, 226, 229, 231, 234, 236, 239, 241, 244, 246, 249, 252]` and the observed
**241 is roll 96**. The engine's `227` is the *mean of the 12 non-KO rolls* — not a fan member at all;
`244` is the defender's HP. The engine prices the observed roll and emits no arm at it.

## 3. Three sub-mechanisms

| row | mechanism | disposition |
|---|---|---|
| `19000074/27` | the **crit**-straddle path has no residual sub-split | engine fix, ~15 lines, mirrors existing code |
| `19100107/135`, `19100191/5` | the threshold is read **before** the move's own secondary status (cause **A8**) | engine fix, but see §4 — deeper than it looks |
| `19000191/63` | the arm exists; its representative mis-prices a roll-dependent drain | needs enumeration or a matcher change |

**`19000074/27`.** The crit population straddles two thresholds — KO at 244 and sand-lethality at 229 —
and the code splits only on KO. The `ko_max_crit >= hp && ko_min_crit < hp` arm emits crit-kill and
crit-survive and never consults `residual_threshold_opt`; the residual treatment for crits lives only
in the sibling `else` reached when the crit fan *cannot* KO. So the residual split exists in two of
three places and is missing from the third. The threshold 229 is computable today; the code simply
does not ask.

**`19000191/63`.** The engine *does* emit the residual-kill arm, at the threshold 108. Showdown rolled
109. Over the 7-roll lethal band the Leech Seed drain is **injective** — 108→29, 109→28, 110→27,
111→26, 112→25, 113→24, 115→22 — so no single representative can cover it. The engine's HP arithmetic
is correct for its own roll; the residue is the choice of representative. A harness route also exists:
the drain is a heal capped by the victim's remaining HP, i.e. roll-inherited, which is the principle
the matcher already applies to `heal_to_full`, `capped_lethal` and `movepainsplit`.

## 4. Why A8 is not a threshold tweak — the finding that matters here

The obvious fix for `19100107/135` and `19100191/5` is to make `residual_lethality_threshold`
status-aware: take the minimum over the statuses the move can inflict. **That is wrong**, and the
reason is measurable.

Replaying `19100107/135` from its recorded state:

```
 41.748%  burn=False  damages=[25, 157]
 41.748%  burn=True   damages=[25, 157, 31]
```

The burn arm and the non-burn arm **carry the same damage, 157**. The status branch happens inside
`run_move`, *after* the damage representative was chosen. So a threshold that assumes the burn would
apply the burn-aware value to the non-burn arm as well — pricing a residual death on a branch where
no residual is lethal.

The correct fix has to partition **per status branch**, so the burn arm's representative can be 159
(matching the observation) while the non-burn arm's stays on the unburned threshold. That is a
restructuring of the order in which the partition and the secondary are resolved, not a change to the
threshold function.

This does answer c117's open question — *whether a correct threshold is computable before the
secondary is decided*. The thresholds are 159 and 211 and both are computable. But computing them is
necessary and not sufficient; the arm they belong on is the harder half.

## 5. What a limit claim in this family would need, and why none holds

Claim: *the observed roll is outside the engine's priced roll set, or the arm-to-observation map is
non-injective for a reason no enumeration can remove.* Measurement: the fan from `calculate_damage`,
the thresholds from `residual_lethality_threshold`, the emitted arms from a replay of the recorded
state, and the verdict from `roll_component_events_agree` on each arm. Falsifier: **any** arm the
per-roll enumerator can produce that the comparator accepts.

For all four rows the observed roll **is** in the engine's fan, so the falsifier fires immediately.
No limit can be asserted. For `19100107/135` and `19100191/5` the specific limit claim would be "no
correct threshold is computable before the secondary is decided" — and §4 shows the thresholds are
computable, so that claim is dead too.

## 6. Why widening the roll-fan gate is not the answer

The gate is `branch_on_damage && choice.first_move && pending_hp_reading_move(defender_choice) &&
fixed_damage.is_none()`. **All four rows fail the `first_move` clause**, because `handle_both_moves`
clears `first_move` on whichever choice resolves second, and a switch resolves before a move. So
widening `pending_hp_reading_move` is worth **zero** rows here.

Making any of them enumerate requires dropping `first_move` *and* the `defender_choice` clause — i.e.
enumerating every damaging move for both movers, which is the blanket enumeration `reports/c128`
rejected. c128's "no throughput cost" result does **not** cover it: that flag ORed into the
`defender_choice` clause only, and c128 says so. The engine's own comment measures the per-call cost
of one-sided enumeration at "12 branches to 144, ~8x slower per call"; two-sided composes
multiplicatively and is unmeasured.

The two targeted engine fixes add at most one arm per boundary, and only where the straddle already
fires. That is the cheaper path by a wide margin.

## 7. The warning this report exists to carry

The previous engine fix in this residue — a gen-3 faint cancelling a queued switch — had a correct
mechanism, a verified Showdown citation, a census, a red-on-main pin and green unit gates, **and was
still wrong by 48 rows** (dev 2 → 40, holdout 3 → 42), because its guard also cancelled forced
replacements. A synthetic pin did not reproduce the state space the sweep covers.

So for each fix above: register a prediction naming **"nothing opened"** as a falsifier, and sweep
both windows before believing it. Both remaining fixes re-price a survive representative — 227 → 220
and 203 → 197 — which shifts the acceptance window on arms that currently match. Whether that opens
rows is unmeasured and is exactly what the sweep is for.
