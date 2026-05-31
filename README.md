# PokeZero

PokeZero is an experiment to train a model to play Pokemon Showdown random battles through self-play.

The initial focus is Gen 3 random battles. The goal is to build a training loop where agents repeatedly battle each other, learn from the resulting games, and improve decision quality with fast learned policies.

This repo will hold the self-play, training, evaluation, and model artifacts for that work.

## Rollout Collection

Collect local random-vs-random self-play trajectories as JSONL:

```bash
python -m pokezero.rollout_cli collect \
  --games 10 \
  --out runs/random-vs-random.jsonl \
  --showdown-root /path/to/pokemon-showdown
```

The Showdown checkout must be built so `dist/sim/index.js` exists. Each JSONL row contains one battle trajectory plus seed, policy ids, terminal outcome, decision-round count, simulator turn count, and elapsed time.

The printed throughput metrics use wall-clock collection time, including JSONL serialization. Use `pokezero.collection.iter_rollout_records(path)` for streaming reads of large trajectory files.

## Gen 3 Belief Sidecar

The read-only sidecar can attach to a local Showdown battle room and display the public Gen 3 random-battle belief state:

```bash
python -m pokezero.sidecar serve \
  --room battle-gen3randombattle-123 \
  --showdown-root /path/to/pokemon-showdown \
  --showdown-url ws://localhost:8000/showdown/websocket
```

The Showdown checkout must be built so `dist/data/random-battles/gen3/teams.js` exists. The sidecar serves a local webview on `http://127.0.0.1:8010` and does not submit battle choices.
