# checkpoints

Curated **milestone** self-play checkpoints (each ~1.5 MB, self-describing) plus a sidecar
`<name>.json` recording provenance + lineage. Per-iteration checkpoints stay on the training run;
only milestones land here. See [`../docs/model_versioning.md`](../docs/model_versioning.md) for the
versioning policy (pin only at a breaking change) and the sidecar format.

## Models

| name | run | games | parent (lineage) | vs max-damage | vs simple | vs random |
|---|---|---|---|---|---|---|
| `pokezero-no-belief-gen3-500k` | `foundation-500k-20260629192858` | 500,800 | — (from scratch) | 51.5% | 92.3% | 98.7% |
| `pokezero-no-belief-gen3-1m` | `foundation-1m-20260630020847` | ~1,000,000 | **`pokezero-no-belief-gen3-500k`** | 62.5% | 95.5% | 98.75% |
| `pokezero-no-belief-gen3-1-5m` | `foundation-2m-20260630171151` | 1,500,800 | **`pokezero-no-belief-gen3-1m`** | 72.9% | 96.25% | 99.5% |

These are recipe-faithful (value-clip on, 1600-game cadence, MIT-thesis LR annealing), trained
**from scratch with no teacher.** The 500k crossed the imitation ceiling (>50% vs max-damage); the
1M is a **continuation of the 500k** that climbed to 62.5%, and the 1.5M checkpoint is a
continuation of the same foundation line. (The 500k -> 1M lineage link is currently *inferred* —
the 1M run used a 1,000,000-game LR denominator over its ~500k segment and continued the 500k's
win-rate curve; future runs should record `continued_from` explicitly.) The 1.5M sidecar records
the completed low-fidelity row plus high-fidelity standard opponents; high-fidelity foul-play is
pending and should be recorded in the follow-up progress documentation.

## Play a checkpoint
```sh
# local Showdown server (pinned commit), then:
python scripts/play_online.py --checkpoint checkpoints/pokezero-no-belief-gen3-1m.pt \
  --showdown-root /path/to/pokemon-showdown --username PokeZeroBot \
  --format gen3randombattle --accept --no-login
```
