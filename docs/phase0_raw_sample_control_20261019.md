# Raw argmax versus sampled champion control

October 19, 2026. Completed descriptive full-game control for the current
champion, independently reviewed across all 32 seed clusters and 64 games.

Raw argmax recorded **38 wins, 25 losses and one genuine draw** against the
sampled champion: **38.5/64 = 60.15625% score**. No games were capped or
missing. This is a fixed-roster observation, not a population confidence
interval, significance test or search-gain claim.

## Design and evidence

Both sides used the same unchanged checkpoint. One selected raw argmax;
the other sampled at temperature one, with epsilon zero, no family gating
and no history truncation. Each game used fresh, separate policy histories.
The rollout hid simultaneous opponent observations and legal masks.
Seeds 3829143000–3829143031 ran in fixed seed-major order, both raw seats,
on a separate cloud CPU with no search or Foul Play. The 250-decision-round
cap would have left a score unknown, not assigned a draw.

The [64-game ledger](audit_artifacts/phase0_raw_sample_control_20261019.json)
retains every outcome, journal hash and the exact entry, progress and terminal
bindings. Independent cloud-side review checked all full journals against
the simulator protocol, legal actions, mode flags and behavior probabilities.
It reproduced 3,021 raw and 3,047 sampled decisions. The draw was seed
3829143004 with raw in p2 and an actual `|tie` event. Collection exited zero,
with verified cleanup, in 175.364 seconds.

The full journals and independent review remain in the owner's local report
collection under `phase0-raw-vs-sampled-20261019-r2/`; observations and model
weights are not duplicated in Git. The ledger is a portable score record,
not a replacement for those journals or an independently runnable archive.

The initial attempt failed during duplicate thread initialization, before
checkpoint loading or any game. After the exact repair, the standing human
zero-root harness-failure rule permitted this separately reviewed fresh
entry. The failed original roster remains separate with all 64 cells unknown;
its missing cells were not filled or pooled into this result.

## Implication for the search goal

A search agent's win rate against sampled play does not alone establish
improvement over raw argmax: raw also scored above 50% in this observed panel.
Do not subtract this result from historical search win rates on different
cohorts. The full-game search panels still require their own matched raw
controls. This control is complete and needs no repeat; it does not qualify
a search winner, deployment or a training teacher.
