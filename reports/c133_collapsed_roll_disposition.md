# C133 — the collapsed-roll residue: three sub-mechanisms, none of them a limit

C116 Phase 4 item 12 requires every remaining divergence to be disposed of as an engine fix, a
harness fix, or **a limit with a written demonstration**. This report disposes of the four
collapsed-roll rows. **None is a limit**, and the `limit:` prefix two of them carry is a classifier
label rather than an adjudication.

No code change ships with this report. It exists so the next attempt starts from measurement instead
of hypothesis — the previous engine fix in this **residue** (not this family: it targeted
`19100180/24`) failed its own falsifier, opening 38 rows in dev and 40 in holdout.

## 1. The `limit:` label is unearned

`19100107/135` and `19100191/5` are classified `limit:roll_divergent_lethality`. I have repeatedly
described that as disposing of them. **It does not.** Three of this repository's own records say the
label is wrong for this family:

- `scripts/family_bucket_audit.py:58` buckets it as **"engine-gap (partially resolved)"**, and `:12`
  defines engine-gap as the engine's branch support not containing the observed transition.
- `reports/c105_retract_limit_overclaim.json` records the limit label as **"8-for-8 falsified"** across
  the eight rows it was applied to — and `19000191/63` is one of the eight — c105 defines the set by reference rather than enumerating it,
  so the membership is corroborated at `reports/c111_residue_row_causes.md:91` ("this was v1's 'genuine
  limit'"). Note the claim comes from c105's `SUPERSESSION_2026_08_04` field, which is the file's **first** key,
  while its body says the opposite — so a reader hits the retraction before the text it retracts.
- The audit's engine-gap signature — *a legal roll range straddling a discrete threshold while the
  emitted arm sits on one side of it* — holds for three of the four rows; `19000191/63` qualifies by
  the definition instead (see §2).

## 2. The shared shape

In every case the observed damage roll **is a member of the engine's own 16-roll fan**. In three of
the four it also lies on the far side of the residual-lethality threshold from the arm the engine
emitted.

`19000191/63` is the exception, and §3 says so: there the engine emits arms on **both** sides of
threshold 108 (survive at `-101`, kill at `-108`), so the straddle is already resolved and the observed
109 sits on the same side as the emitted 108. That row is an engine gap by the audit's **definition**
(the branch support lacks a transition enumeration would reach) rather than by its **signature**.

Worth stating because I got it wrong first: `calculate_damage(..., DamageRolls::Max)` returns
**maxima**, not roll endpoints; the fan is derived in the `for random in 85..=100` loop of
`generate_instructions_from_move` as `raw_damage * random / 100` (cited by symbol: that file is
gitignored and locally rebuilt, so line numbers into it do not resolve for a later reader). On `19000074/27` the crit fan is
`[214, 216, 219, 221, 224, 226, 229, 231, 234, 236, 239, 241, 244, 246, 249, 252]` and the observed
**241 is roll 96**. The engine's `227` is the *mean of the 12 non-KO rolls* — not a fan member at all;
`244` is the defender's HP. The engine prices the observed roll and emits no arm at it.

## 3. Three sub-mechanisms

| row | mechanism | disposition |
|---|---|---|
| `19000074/27` | the **crit**-straddle path has no residual sub-split | engine fix, ~15 lines, mirrors existing code |
| `19100107/135`, `19100191/5` | the threshold is read **before** the move's own secondary status (cause **A8**) | engine fix — make the threshold status-aware; see §4 |
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
is correct for its own roll; the residue is the choice of representative. A harness route also exists, but it must be
narrower than "treat the drain as roll-inherited". The drain is `min(29, 137 − d)`: roll-inherited only
while the cap binds (`d >= 108`), and a flat 29 below the band. Blanket roll-inheriting a `heal` would
over-accept in the uncapped regime and mask real drain defects, so the reclassification has to be
conditioned on the cap actually binding. The precedent is split across two mechanisms, and an earlier
revision of this line conflated them: `_ROLL_SCALED_SOURCES`
(`scripts/engine_transition_differential.py:330-332`) holds `capped_lethal`,
`move_unknown_callee` and `movepainsplit` **unconditionally**, while `heal_to_full` is not in that
frozenset at all — it is handled by the `source.endswith("_to_full")` clause and `capped_bases`, which
is the part that actually conditions on the cap binding. The drain belongs with the latter.

## 4. A8 *is* a threshold change — and an earlier revision of this section said the opposite

**An earlier revision of this report claimed, in capitals, that a min-over-statuses threshold would be
wrong and that the fix required "a restructuring of the order in which the partition and the secondary
are resolved". That was wrong, and a review disproved it by running the counterfactual. Corrected here
because a claim like that would send the next attempt into work it does not need.**

The premise was right and the inference was not. The premise, verified: on `19100107/135` the two
41.75 % arms are `burn=False damages=[25,157]` and `burn=True damages=[25,157,31]` — identical damage.
The same holds on `19100191/5`, where the non-burn (71.72 %) and burn (7.97 %) arms both carry `-203`,
and `7.97 / 79.69 = 10 %` is exactly Fire Blast's burn rate. So the damage representative really is
chosen before the status branch.

What I inferred from that — that a burn-aware threshold would "price a residual death on a branch
where no residual is lethal" — misreads what the partition emits. An `extra_branches` entry is a
**damage representative plus a mass**, not an asserted KO: `generate_instructions.rs` pushes
`(kill_ins, residual_threshold)` and the consuming loop passes that value into `run_move`, which then
computes the faint from the residual instructions *that branch actually emits*. Nothing in the
partition asserts a death.

Verified by replaying `19100107/135`'s recorded state with only Roselia's status field changed, so a
pre-move residual threshold exists:

```
BASELINE          41.75  [25, 157]           41.75  [25, 157, 31]   (burn)
COUNTERFACTUAL    46.97  [25, 150, 31]       36.53  [25, 159, 31]
```

The existing machinery, handed threshold 159, emits **an arm at exactly the observed 159**, at mass
`36.53 / (36.53 + 46.97) = 7/16` — the seven fan rolls at or above 159. And the composition is
multiplicative rather than conflicting: a review's sand counterfactual on `19100191/5` produced two
damage arms × two status arms = four arms, with the burn fraction exactly 10 % inside each, and each
arm's faint computed from its own residuals. On a non-burn sub-branch at 159 damage the defender sits
at `175 − 159 + 15 = 31` HP and lives; there is no mechanism by which the partition could kill it.

**So the fix is a threshold change** — make `residual_lethality_threshold` status-aware by passing the
`choice` in so it can see the secondaries. But the recipe is the **union of the distinct thresholds**,
one arm each, **not the minimum.**

> **An earlier revision of this section said "take the minimum", and called that safe. It is not, and
> I over-corrected twice: first that the fix needed a restructuring, then that the minimum was fine.**
> A review found the counterexample. When the defender already carries a **non-status** residual, the
> minimum can *destroy* an arm the engine emits today. On this same state with the defender's Leftovers
> cleared and sand up, the pre-move threshold is 160 and the engine correctly emits a residual-kill arm
> at 160 on both sub-branches (6/16 of the fan). Taking `min(160, 129)` — 129 being sand 15 plus burn
> 31 off 175 HP — drops the threshold *below the fan's lowest roll*, and all three partition sites gate
> on `<min_roll> < residual_threshold` (`generate_instructions.rs`, the `residual_min`,
> `min_damage_dealt` and `min_crit_roll` guards). With `144 < 129` false the partition does not fire at
> all, and the 160 arm vanishes. That is a regression, and its shape — a sand or Leech Seed
> residual-kill arm — appears in the **dev** sweep's divergence classes
> (`component_missing_in_engine:sandstorm`, `component_missing_in_engine:itemleftovers,leechseed`), so
> it is not hypothetical. An earlier revision said "both committed sweeps"; the holdout
> sweep's classes show neither.
>
> **That absence is real, and a further revision wrongly explained it away.** I wrote that the
> retained repros are "capped at 25 out of ~15.5 k measured boundaries" so the absence proved nothing.
> Two errors: the cap is on `repros` out of the window's **40 divergences**, not out of 15,432
> boundaries; and `divergence_classes` is **not capped at all** — it is built from the full `totals`
> counter, and its counts sum exactly to `transitions_diverged` (40/40 dev, 42/42 holdout). The class
> census is a complete enumeration. That caveat also contradicted §7, which relies on differencing
> `divergence_classes` *because* the repros list is capped; both cannot be true.
>
> What the absence is worth: dev saw the shape twice in 15,432 boundaries, so a window of comparable
> size would miss it by chance about **13 %** of the time (`λ = 2 × 15,551 / 15,432 = 2.02`,
> `e^−2.02 = 0.13`), and the two windows differ in composition. Weak evidence of rarity, not of
> impossibility.

**And the mass rule is disjoint bands, not `P(roll ≥ t)` per arm** — a third correction to this
paragraph, because my first union recipe double-counted. The thresholds are **nested**: a status only
adds a residual tick, so `t_status < t_premove`, and `{roll ≥ t_status} ⊃ {roll ≥ t_premove}`. Giving
each arm `P(roll ≥ tᵢ)` therefore counts every roll above the higher threshold twice. Verified on a
170-max fan with thresholds 145 and 169: `15/16 + 1/16 +` a survive arm of `1/16` = **17/16**, which
overflows, and `update_percentage(1.0 - branch_chance - residual_kill_chance)` has no room for it.
Worse than the overflow, the single roll ≥ 169 also sits inside the 145 arm, whose non-burn
sub-branch renders a *survival* — so the excess mass lands on the wrong outcome.

The correct rule: sort the distinct thresholds `t₁ < … < t_k`; arm `i` carries the **disjoint band**
`#{rolls ∈ [tᵢ, tᵢ₊₁)}/16`, the top arm carries `#{rolls ≥ t_k}/16`, and the survive arm keeps
`#{rolls < t₁}/16`. Each kill arm is priced at its own `tᵢ`, and the survive arm keeps
`average_surviving_damage` over the rolls below `t₁` — which is the engine's existing convention and
what §7's `157 → 150` re-pricing depends on. **This is a generalisation of a subtraction the engine
already performs**: `num_residual_only = num_at_or_above − num_kill_rolls` is exactly the band between
the residual threshold and the KO threshold. And it applies at all three partition sites, the crit
path included, since a status-aware threshold nests there too. On that fan: `145 → 14/16`, `169 → 1/16`, survive `→ 1/16`, totalling exactly
`16/16`.

The status-independence of `P(roll ≥ t)` is true, and it is what makes the *structure* sound; it is not
a licence for the per-arm mass. I conflated the two.

**Reachability of the nested case**, so this is not filed as a corner: it needs
`t_premove − t_status ≤` the fan width, i.e. roughly `maxhp/8 ≲ 0.15 · max_damage`. Burn and poison
make that window narrow but real — a near-full-HP defender against a near-OHKO move — and a **Toxic**
secondary (gen-3 Poison Fang) halves the gap to `maxhp/16` and opens it wide.

This also answers c117's open question — *whether a correct threshold is computable before the
secondary is decided*. It is: 159 and 211, both re-derived and both produced empirically.

The real risk is not correctness but re-pricing: see §7.

## 5. What a limit claim in this family would need, and why none holds

Claim: *the observed roll is outside the engine's priced roll set, or the arm-to-observation map is
non-injective for a reason no enumeration can remove.* Measurement: the fan from `calculate_damage`,
the thresholds from `residual_lethality_threshold`, the emitted arms from a replay of the recorded
state, and the verdict from `roll_component_events_agree` on each arm. Falsifier: **any** arm the
per-roll enumerator can produce that the comparator accepts.

For all four rows the observed roll **is** in the engine's fan, so the falsifier fires immediately.
No limit can be asserted. For `19100107/135` and `19100191/5` the specific limit claim would be "no
correct threshold is computable before the secondary is decided" — and §4 shows the thresholds are
computable *and* that the existing partition emits a matching arm from them, so that claim is dead
too.

## 6. Why widening the roll-fan gate is not the answer

The gate is `branch_on_damage && choice.first_move && pending_hp_reading_move(defender_choice) &&
fixed_damage.is_none()`. **All four rows fail the `first_move` clause**, because `handle_both_moves`
clears `first_move` on whichever choice resolves second, and a switch resolves before a move. So
widening `pending_hp_reading_move` is worth **zero** rows here.

Making any of them enumerate requires dropping `first_move` *and* the `defender_choice` clause — i.e.
enumerating every damaging move for both movers, which is the blanket enumeration `reports/c128`
rejected. c128's "no throughput cost" result does **not** cover it: that flag ORed into the
`defender_choice` clause only, and c128 says so. The engine's own comment gives a **per-call** cost for
one-sided enumeration — "12 branches to 144, ~8x slower per call" — and two-sided composition is
unmeasured.

**That per-call figure must not be read as a workload cost, and c128 §3(b) is an explicit retraction of
my doing exactly that**: measured at the differential's workload, one-sided enumeration cost nothing
(1,463.6 games/h with it on versus 1,427.8 with it off). So: the two targeted engine fixes add at most
one arm per boundary and only where the straddle already fires, which is cheaper *in branch count*;
the aggregate cost of two-sided enumeration is unmeasured, and the one aggregate measurement in
evidence found no cost at all. An earlier revision of this section concluded "cheaper by a wide
margin", which reuses the inference c128 withdrew.

## 7. The warning this report exists to carry

The previous engine fix in this residue — a gen-3 faint cancelling a queued switch — had a correct
mechanism, a verified Showdown citation, a census, a red-on-main pin and green unit gates, **and was
still wrong**: it opened **38 rows in dev and 40 in holdout** — 78 opened against a single closure,
its own target `19100180/24` — taking dev 2 → 40 and holdout 3 → 42, net **+77**, because its guard
also cancelled forced replacements. A synthetic pin did not reproduce the state space the sweep covers.

> **Two earlier revisions of this figure were wrong, the second in a way worth recording.** The first
> said "wrong by 48 rows" with no referent. The second said "24 rows in each window" — which is
> internally impossible, since 24 opened with nothing closed cannot give a net of +38. I had read the
> length of each artifact's `repros` list as the opened count, and **that list is capped at
> `keep_repro=25`**. The real counts come from differencing `divergence_classes`: dev gains 38 rows
> across `itemleftovers` (21), `itemleftovers,psn` (4), `itemleftovers,spikes` (4), `roll_scaled` (5)
> and others, closing none; holdout gains 40 and loses exactly `component_extra_in_engine:spikes`.

The two sweeps are committed as
`reports/artifacts/c133_withdrawn_switchcancel_{dev,holdout}_sweep.json` (fingerprint `b73929bd1e`,
`source_commit 0e4fc75a`), so the figure is checkable rather than asserted.

So for each fix above: register a prediction naming **"nothing opened"** as a falsifier, and sweep
both windows before believing it. The remaining fixes re-price **three** survive representatives — 227 → 220 on `19000074/27`,
203 → 197 on `19100191/5`, and **157 → 150 on `19100107/135`** (measured in §4's counterfactual; an
earlier revision listed only two) — each of which shifts the acceptance window on arms that currently
match. Whether that opens
rows is unmeasured and is exactly what the sweep is for.
