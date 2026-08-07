# C135 — `limit:roll_divergent_lethality` is not a limit. Both rows are engine gaps.

C116 Phase 4 item 12, for holdout rows `19100107/135` and `19100191/5`. These are the
last two rows in the residue carrying a `limit:` prefix, and the plan requires every
disposition to be an engine fix, a harness fix, or **a limit with a written
demonstration**. No demonstration is owed here, because none can be constructed: the
falsifier fires.

> **The `limit:` prefix is a classifier emission, not an adjudication.** The repo has
> already retracted the contrary reading — `scripts/family_bucket_audit.py` buckets this
> family as "engine-gap (partially resolved)", and
> `reports/c105_retract_limit_overclaim.json` records the limit claim as
> 8-for-8 falsified. This measurement corroborates that retraction rather than resting
> on it.

## 1. The falsifier, run first

The limit claim for this family would be: *no arm the engine can produce would be
accepted by `roll_component_events_agree`.*

Counterfactual: take each recorded state, flip **only the defender's status field** to
BURN so that a pre-move threshold exists, and feed the result through the real,
unmodified `evaluate_boundary_strict`.

| row | baseline | counterfactual |
|---|---|---|
| `19100107/135` | diverged (8 branches), misses identical to the artifact | **matched** (8 branches) |
| `19100191/5` | diverged (4 branches), misses identical to the artifact | **matched** (4 branches) |

The engine's existing partition machinery, once handed a threshold, emits a
residual-kill arm **at the threshold**. On `19100107/135` that threshold *is* the observed
roll (159). On `19100191/5` it is **not**: the arm sits at 211 and the observation is 215,
and the `matched` verdict there comes from the comparator's cap-vs-cap identity rather
than from an arm at the observed roll — see §3, which works that case through. Either way
the verdict is produced by the shipped comparator, not by an argument about one.

## 2. Both rows are one mechanism

A Fire move whose secondary burn is applied *after* the engine has chosen its damage
representative, where the burn tick then kills a defender the representative leaves
alive.

| | `19100191/5` | `19100107/135` |
|---|---|---|
| move | Fire Blast → Ninjask | Sacred Fire → Roselia |
| defender | 225/225, Leftovers, no status | 175/254, Leftovers, no status |
| observed | −215, brn, +14, capped −24, faint | −159, brn, +15, capped −31, faint |
| non-crit fan max | 220 | 170 |
| **observed roll** | **215 = roll 98** | **159 = roll 94** |
| engine representative | 203 = `(220 × 0.925) as i16` (truncated from 203.5) | 157 = `(170 × 0.925) as i16` (from 157.25) |
| crit fan minimum | 374 (≥ hp 225) | 290 (≥ hp 175) |

The observation is a **member of the engine's own priced fan** in both rows, and the
engine's representative is not a fan member at all — it is a collapsed mean. By this
repo's own definition (`scripts/family_bucket_audit.py`) that is an engine gap.

Neither observation is reachable on the crit arm: every crit roll is a move-KO, and the
observed protocol carries no `|-crit|`.

## 3. Root cause, in source

Both rows take Case B of the roll partition in `generate_instructions.rs`
(`branch_on_damage && fixed_damage.is_none() && max_damage_dealt < defender_active.hp`).
That case *does* contain a residual-lethality split, but it is gated on
`residual_threshold_opt`, computed from the defender's **pre-move** state — before the
move's own secondary is known.

`residual_lethality_threshold` declines whenever a 1-HP defender would survive the
residual phase. With status `NONE` and Leftovers it declines for both defenders, so the
split never fires and the entire non-crit mass collapses onto a surviving representative.

Re-derived by hand from the source, reproducing the engine's own numbers:

| defender | status NONE | status BURN |
|---|---|---|
| Roselia (175/254, Leftovers) | declines | **159** |
| Ninjask (225/225, Leftovers) | declines | **211** |

The documented identity holds — `hp − low + 1`, and a status only *lowers* the threshold
— here from declined (effectively infinite) to a finite value inside the fan.
`#{rolls ≥ 159} = 7/16` and `#{rolls ≥ 211} = 5/16`, and the observed rolls sit in those
bands.

The 211-vs-215 case passes for a structural reason worth recording: the matcher's
cap-vs-cap rule requires `(|obs_cap| − |eng_cap|) == (eng_direct − obs_direct)`, and here
`24 − 28 = −4` equals `211 − 215 = −4`. Both caps land at exactly 0 HP from the same
pre-HP, so the identity is exact rather than slack.

## 4. Why this is not a harness defect

The strongest case for a harness defect is that the crit arm's net HP equals the
observation's on both rows — the `instrument-artifact` signature in
`family_bucket_audit.py`. It fails on three measurements:

1. The crit arm carries a `|-crit|` event and the observed protocol has none.
2. The observed damage is not in the crit fan (minima 290 and 374, versus HP 175 and 225).
3. The signature requires the **majority** arm. The majority arms here end at 33/2 HP and
   36/8 HP — **alive** — so their nets are −142/−173 and −189/−217, not −175/−225.

No emitted arm expresses "survives the move, dies to the residual". Making the matcher
accept a surviving arm against a fainting observation would not repair the instrument, it
would delete its ability to see residual-lethality errors. And the renderer is
demonstrably adequate: it renders that exact transition correctly the moment the engine
emits it, which is what §1 shows.

## 5. Disposition, and what is deliberately not done here

**Both rows: engine gap.** That is the disposition. The fix is *not* implemented in this
change, for two reasons.

C134 §3 places a **freeze** on new partitions, sub-splits and mass recipes until the
enumeration spike's measurement exists, and it does so because this exact family has
already burned three wrong hand-derived mass recipes. This adjudication is evidence for
that decision, not an exception to it.

For the record, the shape the fix would take: make `residual_lethality_threshold`
status-aware by passing the attacker's `choice` in so secondaries are visible, then emit
one residual-kill arm per **distinct** threshold, priced at its own `tᵢ`, with
**disjoint-band** masses (`#{rolls ∈ [tᵢ, tᵢ₊₁)}/16`, top arm `#{rolls ≥ t_k}/16`,
survive arm keeping `average_surviving_damage` over rolls below `t₁`), applied at all
three partition sites. Explicitly **not** the minimum over statuses:
`reports/c133_collapsed_roll_disposition.md` §4 records a measured counterexample where
`min` drops the threshold below the fan's lowest roll and destroys an arm the engine
emits today.

## 6. Limits of this adjudication, stated

- No patched engine was built. The post-fix branch set is inferred from a
  status-substitution **proxy**, not from the fix. The proxy collapses the burn secondary
  where the real fix yields a 2×2 product of damage arm × status arm; the match should
  survive because `evaluate_boundary_strict` accepts if *any* branch matches, but that is
  **predicted, not measured**.
- The disjoint-band mass rule is not exercised by the proxy at all.
- The fix re-prices survive representatives (157 → 150 and 203 → 197 here; c133 names a
  third at 227 → 220). The comparator's primary accept path is exact fan membership,
  which is independent of the representative — but the fallback window
  `[0.92·eng − 1, 1.09·eng + 1]` is not, and it applies whenever `pre_legal` is
  unavailable. How much matched mass rides on that fallback is **unmeasured**.
- **`19100191/5`'s match rides on the cap-vs-cap identity, not on an exact arm.** §3
  derives it; it belongs in this list too, because it is the weaker of the two matches and
  a reader skimming §1 should not miss it.
- `19000074/27` and `19000191/63`, the other two rows in the collapsed-roll family, were
  outside this assignment and are not adjudicated here.

## 7. Update: the freeze has lifted, and enumeration closes both rows

The C134 §3 enumeration spike has since produced its measurement, which supersedes §5's
"the fix is deliberately not implemented here". Measured on both windows with the
enumerate flag on, same build, same seeds:

| window | collapsed | enumerated |
|---|---|---|
| dev `19,000,000–199` | 2 diverged | **0** |
| holdout `19,100,000–199` | 4 diverged | **2** |

Measured on the spike build (fingerprint `1807ce9590bbd5b2`, source `425bf220`), which is
*not* the `6b6fb368` build the rest of this report rests on — that is why the collapsed
holdout column reads 4 where §3's baseline has 5. The difference is exactly `19100180/24`,
closed in between by the faint-cancels-switch work, and not by enumeration.

The two rows adjudicated here — `19100107/135` and `19100191/5` — **both close**, and
nothing opened on either window. The stronger form of that check also holds:
`boundaries_measured` is identical off versus on (15,503 dev / 15,579 holdout),
`engine_errors` is 0 in all four runs, and every gating and skip counter is unchanged — so
enumeration removed divergences without shrinking or perturbing the measured population,
which is the check that would catch a fake closure. So the disposition stands as *engine gap*, and the
remedy is not the status-aware threshold sketched in §5 but enumeration itself, which
deletes the mirror rather than repairing it. C134 §3 anticipated exactly this: "enumeration
fixes them or *constitutes* the demonstration". The measurement says **fixes**.

> **SUPERSEDED, 2026-08-07.** The scoping paragraph below was written under the first
> version of the C116 Phase 2 decision, which chose harness-only adoption. That decision
> was reversed after review: harness-only would close these rows in the measuring
> instrument while leaving the defects in the shipping engine, and would stop the fidelity
> gate testing the path production runs. The current decision is
> `reports/c137_phase2_enumerate_decision.md`: enumeration ships as a flag-gated reference
> **oracle**, the differential keeps measuring the shipping configuration, and the §5
> sketch and the crit-straddle sub-split are **un-cancelled and should be implemented**,
> validated against that oracle. The adjudication above is unaffected — both rows remain
> engine gaps, not limits. Only the remedy changed.

**Scope: the differential harness only.** "The remedy is enumeration" here means for the
*fidelity comparison*, and must not be read as licensing enumeration in search. The other
two acceptance measurements C134 §3 required are decisive against that and are recorded in
`reports/c137_phase2_enumerate_decision.md`: search throughput at depth 4 / 1024 sims
regresses **2.38 ms → 8,881.8 ms per decision** on the production-representative position,
and the mass gate's `test_matrix_is_not_vacuous` fails under the flag — not a mass
disagreement, but because enumeration leaves no collapsed fan for the matrix's own negative
control. C137 takes the resulting decision (adopt for the harness only).

~~That also retires the sketch in §5 as the recommended fix. It should not be implemented;
the crit-straddle sub-split queued for `19000074/27` should not be written either, for the
same reason.~~ **Retracted — see the superseded notice above. Both fixes are back on.**

Whatever implements this must register **"nothing opened"** as an explicit falsifier and
sweep both windows before it is believed. The precedent is in c133 §7: the last engine
fix in this residue had a correct mechanism, a red-on-main pin and green unit gates, and
still opened 38 dev / 40 holdout rows.
