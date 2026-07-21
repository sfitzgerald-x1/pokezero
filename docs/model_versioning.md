# Model & checkpoint versioning

## Principle: behavior = the whole inference stack, not just the weights
The same `state_dict` will play **differently** if any of these change: the observation schema,
the `CategoryVocabulary`, the gen3 dex data, the model/forward code, the action mapping, the pinned
Showdown commit, or torch/deps. Checkpoints are self-describing — they carry `schema_version`,
`training_schema_version`, and `model_config` — so the loader can detect an incompatible stack.

Observation schema `pokezero.observation.v3` (CLI `v3`, checkpoint-driven like v2.1/v2.2, not yet
the fresh default): v2.2's turn-merged surface plus six appended numeric columns — the `-fail`
transition-event bit per sub-block, the public sleep-clause block bits on the field token, the
public consecutive-stall counter on each side's active mon, and confusion turns-so-far on a
confused active mon (`docs/observation_v3_spec.md`); v3 is still pre-freeze, so the numeric feature
count is 161; v2/v2.1/v2.2 encodes stay byte-identical.

## Policy: do NOT pin active checkpoints — pin only at a breaking change
During normal development we **do not** freeze, tag, or pin checkpoints. Active checkpoints churn
with the recipe and are reproducible from the run + current code; pinning them continuously is
wasted effort. The checkpoints under [`checkpoints/`](../checkpoints/) are **current, unpinned
models** — convenient to play, not preservation releases.

**Trigger to preserve:** when we make a **breaking change** — anything that alters a model's
behavior or its ability to load (observation-schema bump, vocab/dex change, architecture change,
action-mapping change, Showdown-pin bump) — *and* we want to keep the behavior of a model trained
under the old recipe, we cut a **pinned, preserved release** of that specific model at that point.

## What a "pinned release" is (the lightweight freeze, only at a breaking change)
For each model we choose to preserve:
- **Release manifest** pinning the full stack: checkpoint SHA-256, pokezero git tag/commit, Showdown
  commit, `uv.lock` (deps), and the bundled `schema_version` / `model_config`.
- **Git tag** the release — the tag *is* the freeze; checking it out reproduces the behavior.
- *(optional)* **Golden behavior test**: a small `(battle state → chosen action)` fixture set in CI
  to detect drift the moment a future change would alter that model's decisions.

To replay a preserved model later: check out its tag (+ Showdown commit) and load its checkpoint.

## Lineage / provenance metadata (every committed checkpoint)
A checkpoint's weights don't record where they came from, and run names/timestamps are **not** a
reliable lineage source (we've already had to *infer* "1M continued from 500k" from LR denominators
and win-rate continuity). So every checkpoint in `checkpoints/` carries a sidecar
`checkpoints/<name>.json` with explicit provenance:

```json
{
  "name": "pokezero-no-belief-gen3-1m",
  "file": "pokezero-no-belief-gen3-1m.pt",
  "sha256": "<hash>",
  "games_trained": 1000000,
  "run_id": "foundation-1m-20260630020847",
  "parent": "pokezero-no-belief-gen3-500k",          // ← lineage link (null if from scratch)
  "input_family": "no-belief",
  "known_input_issue": "Belief-derived opponent set features were unavailable.",
  "schema_version": "<from checkpoint>",
  "recipe": {"value_clip": true, "cadence_games": 1600, "lr_schedule": "mit-thesis", "lr_total_games": 1000000},
  "winrates": {"max-damage": 0.625, "simple-legal": 0.955, "random-legal": 0.9875},
  "date": "2026-06-30"
}
```

The `parent` field chains the lineage (500k → 1M). `input_family` records meaningful observation
family differences, such as `no-belief` versus a future fixed belief-input run, so eval curves can be
compared without accidentally merging incompatible inputs. Going forward, the training side should
record `continued_from` in each run's summary so lineage is captured automatically rather than
reconstructed.

## Storage & retention
Checkpoints are ~1.5 MB and self-describing, so curated **milestone** checkpoints may be committed
directly to the repo (no LFS/external storage needed at this size). Per-iteration checkpoints stay
on the training run; only milestones land in `checkpoints/`.

**Capture milestones promptly:** runs prune all but the **last ~8 iterations'** checkpoints, so a
milestone must be copied out of the run **before retention deletes it.**

## Deferred until there's a concrete need
**Backward-compatible multi-schema inference** — making future code load and run *old* checkpoints
in one process (for a cross-version league / model zoo) — is heavy, ongoing maintenance. Defer it
until we actually want old + new models coexisting; until then, "check out the tagged stack" fully
preserves behavior at far lower cost.
