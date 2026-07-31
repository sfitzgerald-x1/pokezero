# C25 Toxic residual stage recovery prediction

## Scope

This prediction concerns only public replay recovery of a surviving Gen 3
badly-poisoned (`tox`) residual in
`ShowdownReplayState._reseed_toxic_stage_from_residual`. The original prediction
covered parser/world recovery only and did not anticipate an engine-rule change.
The final reviewed disposition below also includes a separate engine stage-cap
correction; it still does not infer hidden stages from capped or lethal residuals.

The retained identities use an absolute maximum HP of 239. Percentage-form
Showdown conditions use a `/100` denominator and round the hidden HP delta, so
they cannot recover an exact Gen 3 stage. They may preserve a stage already
established by public status/switch/turn history. A public switch/drag reset
does prove that the next rounded Toxic residual is stage 1; an incomplete prefix
without that reset must not synthesize one. The denominator alone is not a
representation marker because a real Pokemon can have exactly 100 maximum HP.

## Hypothesis

Gen 3 Toxic damage is `max(1, floor(max_hp / 16)) * stage`. The replay parser
currently reconstructs the stage with `round(16 * damage / max_hp)`, which is
wrong whenever the maximum HP is not divisible by 16. Recovering a surviving
residual as `damage // max(1, max_hp // 16)` should preserve the public stage
that `engine_world` later seeds as `stage - 1` at an ordinary action request.

## Predicted affected evidence

The retained current-source identity sequence `2900415/69-71` has maximum HP
239. Its public Toxic residuals are 126, 140, and 154 damage, respectively:

| Identity | Damage | Gen 3 unit | Expected parser stage | Current rounded stage |
| --- | ---: | ---: | ---: | ---: |
| `2900415/69` | 126 | 14 | 9 | 8 |
| `2900415/70` | 140 | 14 | 10 | 9 |
| `2900415/71` | 154 | 14 | 11 | 10 |

After the parser fix, these rows should seed the world with Toxic counts
8, 9, and 10 rather than 7, 8, and 9. The direct replay parser output should
be stages 9, 10, and 11.

## Discriminating controls

1. A divisible maximum HP (for example 240) gives the same stage under both
   formulae and must remain unchanged.
2. A low maximum HP below 16 has a Gen 3 Toxic unit of 1; surviving damage of
   1, 2, and 3 must recover stages 1, 2, and 3.
3. Ordinary `psn` remains excluded even though its residual line also carries
   `[from] psn`.
4. A re-entered `tox` Pokemon continues to re-seed from the first surviving
   residual after switch-in, using the floored unit rather than proportional
   rounding.
5. Percentage-form residuals may recover only stage 1 after a public reset;
   later stage progression comes from public turn chronology, not reverse
   rounding.
6. A real exact-100 HP Pokemon uses the exact Gen 3 unit of 6, established by
   private-request or omniscient-stream provenance rather than its denominator.
7. A capped or lethal residual remains fail-closed. A non-integral exact-HP
   difference after re-entry makes the zero-stage provenance unknown.

## Acceptance evidence

The implementation must add tests that fail before this change for the
239-HP sequence and controls above, preserve regular-poison and pivot behavior,
bound percentage recovery to a publicly reset stage 1, and leave the public
confidentiality invariant validation green.

## Recovery hardening result

The first implementation corrected the floor-before-multiply arithmetic but
still accepted three invalid provenance paths: it reverse-rounded `/100` public
HP into a hidden stage, allowed a benched `-curestatus` line to clear the
active counter, and kept a fainted active's counter until the replacement
switch. The recovery records explicit public-counter and HP-representation
provenance in replay snapshots. Legacy snapshots with active `tox` but no
provenance, condition-only residuals, and rounded residuals without a public
reset now fail closed at world construction. A percentage stream may establish
only the mechanically certain post-reset stage 1; an exact 100-HP stream remains
exact.

At an ordinary action request the parser feature is one residual ahead, so
materialization emits `stage - 1`. At a post-upkeep, pre-next-turn forced-switch
boundary it is the just-applied multiplier and materialization emits the stage
unchanged. Showdown caps the current stage at 15; internal parser value 16
preserves the distinction between "current 14, next 15" and "current 15,
remains 15", while the observation still clamps both to its public stage-15
maximum. The Rust engine then applies `min(15, toxic_count + 1)` at its next residual.
The real 316-HP capture and exact-100 HP scenario controls pin those conventions,
while lifecycle controls cover switch/drag, Baton Pass, status overwrite/Rest,
Natural Cure, reapplication, faint, and checkpoint resume. This section records
the parser/world provenance portion of the work. The final disposition also
includes the engine stage-cap correction documented in the review amendment below.

Caller provenance is explicit. Local Showdown and the controlled FoulPlay bridge
own full omniscient streams and mark both sides exact. The online client owns its
room buffer from battle start; its private request identifies exact own HP and
percentage opponent HP. The read-only sidecar strips requests and may reconnect
mid-battle, so it cannot prove either fact and deliberately keeps Toxic stage
unknown until public reset/status chronology establishes it; the sidecar does
not construct engine worlds from that value.

## Review amendment

Independent review found that the preceding prediction overstated the engine
behavior: the vendored Gen 3 engine used `toxic_count + 1` without a cap. Thus
materializing the raw parser saturation sentinel `16` as counter `15` produced
an illegal stage-16 residual. The repaired seam caps engine residual damage at
stage 15 and saturates the stored pre-tick counter at 14. Materialization now
maps ordinary raw `16` and post-upkeep stage `15` to counter `14`; inputs above
the parser representation fail closed. The correction is proven by advancing
two residual blocks at 640 max HP, each dealing exactly 600 (15/16), never 640.

Production leaf rendering does not invent `-status` or cure protocol lines for
these lifecycle paths. It supplies an internal ordered active-status transition
to `LeafMeta`, so Rest, Refresh, and Heal Bell clear stale Toxic metadata while
their externally rendered protocol remains faithful. Clean switch and drag
entries clear both the stage and active provenance before re-entry derivation.

## Post-upkeep replacement amendment

One active-Toxic zero is public rather than unknown. Showdown resets Toxic's
`statusState.stage` to zero on switch-in and increments it immediately before
the next residual. When `|upkeep|` is followed by a non-Baton-Pass `|switch|`
whose replacement condition retains `tox`, that replacement missed the
completed residual block. Its first pending engine residual therefore requires
the legitimate pre-tick `toxic_count = 0`; direct materialization now admits
only that snapshot shape.

The proof is snapshot-carried but construction-only: it changes neither the raw
public Toxic feature nor V2/V2.1/V2.2 observation bytes. It expires on the
first Toxic residual and is cleared by active status/cure/faint transitions or
a later switch/drag. A post-upkeep `|drag|` is rejected as synthetic chronology:
Gen 3 executes phazing during the move action, before its residual action emits
`|upkeep|`. Missing proof in a legacy snapshot remains fail-closed.
