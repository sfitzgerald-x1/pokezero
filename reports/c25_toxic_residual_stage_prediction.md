# C25 Toxic residual stage recovery prediction

## Scope

This prediction concerns only public replay recovery of a surviving Gen 3
badly-poisoned (`tox`) residual in
`ShowdownReplayState._reseed_toxic_stage_from_residual`. It does not alter the
engine's Toxic implementation or infer hidden stages from capped or lethal
residuals.

The retained identities use an absolute maximum HP of 239. Percentage-form
Showdown conditions use a `/100` denominator and round the hidden HP delta, so
they cannot recover an exact Gen 3 stage. They may preserve a stage already
established by public status/switch/turn history, but they must not synthesize
one for an incomplete prefix.

## Hypothesis

Gen 3 Toxic damage is `max(1, floor(max_hp / 16)) * stage`. The replay parser
currently reconstructs the stage with `round(16 * damage / max_hp)`, which is
wrong whenever the maximum HP is not divisible by 16. Recovering a surviving
residual as `damage // max(1, max_hp // 16)` should preserve the public stage
that `engine_world` later seeds as `stage - 1`.

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
5. A capped, percentage-only, or lethal residual must remain fail-closed: when
   the prior HP is unavailable, the current HP is zero, or an exact-HP observed
   difference is not a positive multiple of the Gen 3 unit, this change must
   not invent a stage.

## Acceptance evidence

The implementation must add tests that fail before this change for the
239-HP sequence and controls above, preserve regular-poison and pivot behavior,
reject percentage-only recovery, and leave the public confidentiality invariant
validation green.

## Recovery hardening result

The first implementation corrected the floor-before-multiply arithmetic but
still accepted three invalid provenance paths: it reverse-rounded `/100` public
HP into a hidden stage, allowed a benched `-curestatus` line to clear the
active counter, and kept a fainted active's counter until the replacement
switch. The recovery records explicit public-counter provenance in replay
snapshots. Legacy snapshots with active `tox` but no provenance, percentage-only
residuals, and condition-only residuals now fail closed at world construction.

The parser feature remains one residual ahead of the world request boundary;
materialization emits `stage - 1` exactly once, and the Rust engine applies
`toxic_count + 1` at its next residual. The real 316-HP capture control pins
Leftovers before stages 1 and 2, while lifecycle controls cover switch/drag,
Baton Pass, status overwrite/Rest, Natural Cure, reapplication, faint, and
checkpoint resume. This result is parser/world provenance recovery only; it
does not claim an engine-rule correction.
