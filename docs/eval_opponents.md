# Evaluation opponent registry — gen3randombattle

Status: living registry, started 2026-07-03. Every strength claim should
name its opponent from this table and the settings row used. Companion to
[`recent_pokemon_agents_survey.md`](recent_pokemon_agents_survey.md).

## Why this exists

Win rates are meaningless without a fixed, documented opponent. Two things
prompted this registry: (a) the PokéAgent Challenge established that our
main external yardstick, foul-play, is the reigning NeurIPS-challenge Gen 9
OU champion — so absolute numbers against it need that context attached;
(b) we investigated the challenge's 30-checkpoint skill ladder as
intermediate rungs and found a format incompatibility worth recording so
it isn't re-litigated (below).

## Active gauntlet (gen3randombattle)

| opponent | kind | strength band | provenance / settings |
|---|---|---|---|
| `random-legal` | scripted | floor | uniform over legal actions |
| `simple-legal` | scripted | low | `policy.py` heuristic |
| `max-damage` | scripted | low-mid | greedy damage estimate (needs Showdown dex) |
| `scripted-teacher` | scripted | mid | curated branch logic incl. hazards/Rapid-Spin (`policy.py`) |
| **foul-play @ search-time-ms** | external search bot (poke-engine, MCTS) | **mid → SOTA, tunable** | `third_party/foul-play` pinned submodule + local patches; `--search-time-ms` sets strength |
| historical self checkpoints | frozen nets | matched | v2+ current-family only; enforce with `opponents.py` `current_family_*` helpers; curated milestones at `/shared` + `checkpoints/curated/` |

`random-legal` and `simple-legal` are plumbing checks only. They saturated by
the 1M no-belief family (~99% and ~95%), so they should not be used as strength
gradients or advancement evidence. Strength reads should use meaningful
opponents: max-damage, foul-play rungs, and frozen v2+ checkpoint pools.

Historical checkpoint opponents should also stay on the current comparison
family. In this registry, **v2+** means current-family checkpoints encoded with
observation-schema v2 or newer, not older no-belief/pre-v2 families and not
pool-version labels such as `pool-self-v1`. For new wave strength evaluation,
use v2+ checkpoints/frozen pools and matched milestones. Legacy-family
checkpoints at any milestone, including longer historical runs beyond 500k
games, are historical context, not opponents to evaluate against. Calibration
pools can still be used for value/diagnostic reads when their role is stated
explicitly. The `current_family_*` helpers enforce both sides of this rule:
schema v2+ is required, and no-belief/pre-v2 family markers in checkpoint
filenames or adjacent checkpoint metadata are rejected.

### foul-play as a graded ladder (the randbats-native rung system)

foul-play's `--search-time-ms` knob turns one integrated opponent into a
strength ladder with identical rules coverage and zero new integration:

| rung | search-time-ms | role |
|---|---|---|
| FP-10 | 10 | weak search — first search-bot rung above scripted |
| FP-50 | 50 | intermediate |
| FP-100 | 100 | cluster-standard rung (matches controller evals) |
| FP-1000 | 1000 | foul-play default; the E0-oracle setting |

Convention: report the rung in every read (e.g. "12.4% vs FP-1000").
Rung-to-rung win-rate curves also give a finer progress signal than a
single rung's band. Calibration of rung spacing (FP-10 vs scripted-teacher
etc.) is a cheap one-off worth running before relying on the middle rungs.

Context for absolute numbers: foul-play (successor lineage) **won the
Gen 9 OU tournament of the NeurIPS 2025 PokéAgent Challenge**
([paper](https://arxiv.org/abs/2603.15563)) — a fixed-rules variant of the
same engine + search stack. Reads against FP-1000 are reads against a
state-of-the-art search opponent.

## Investigated and excluded (for now): PokéAgent 30-checkpoint ladder

The challenge released checkpoints for ~30 agents spanning its practice
ladder (compact RNNs → 200M-param transformers), hosted at
[`jakegrigsby/metamon`](https://huggingface.co/jakegrigsby/metamon/tree/main)
(notable: Kakuna 142M, "best public metamon agent"; TaurosV0 62M Gen1OU;
Abra 57M Gen9OU). They would be ideal graded external rungs, **but they are
OU-only**: Metamon "initially focused on the most popular singles ruleset
('OverUsed') for Generations 1, 2, 3, and 4… recently expanded to include
Generation 9 OU," with fixed team sets — random battles are explicitly not
supported. Two consequences:

1. **Not valid gen3randombattle rungs as-is.** Running an OU-trained model
   on randbats is mechanically conceivable (poke-env supports the format;
   gen-3 dex is inside Metamon's gen 1–4 coverage) but out-of-distribution:
   the strength grading — the entire value of the ladder — would not
   transfer without recalibration.
2. **Possible future uses**, in preference order:
   - *Gen 3 OU side-yardstick*: our agent plays their home format with a
     fixed team set. Measures transfer, not the mainline objective; our
     agent is the OOD party there.
   - *OOD-calibrated randbats rungs*: run 2–3 of the smaller checkpoints on
     gen3randombattle, calibrate against the FP ladder, and admit them only
     if their ordering is stable. Integration cost: metamon venv +
     poke-env bridge to our local Showdown (same arms-length pattern as
     foul-play).

Neither is scheduled; recorded here so the compatibility finding isn't
rediscovered.

## Not opponents (quarantine note)

Per project direction, human-play datasets (e.g. PokéChamp's 3M-game
corpus) and human-imitation models are admissible as *evaluation material,
search-time opponent models, and exploiter opponents* — never as policy
teachers. Any addition to this registry from that lineage must state which
of those three roles it serves.
