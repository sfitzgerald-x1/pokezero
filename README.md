# PokeZero

PokeZero is a **work-in-progress** effort to train an agent that plays Pokémon Showdown **Gen 3
random battles** well enough to be **competitive on the live ladder** — learned entirely from
self-play, on CPU-first hardware. The approach is AlphaZero-style: improve a policy/value network
by having it play itself, here applied to an imperfect-information, simultaneous-move game.

> ⚠️ Active research. Encodings, APIs, and checkpoints change frequently. The neural policy below
> is the current frontier; the linear baseline and parts of the harness are earlier scaffolding
> kept for reference.

## How it works

- **Observation — raw facts only.** The battle state is encoded as per-entity tokens (active mons,
  team members, candidate moves, field), each carrying categorical ids plus numeric features. A
  **hard rule**: no precomputed type effectiveness, STAB, expected power, damage estimates, or
  matchup summaries — the model must learn these from raw observable facts.
  ([`docs/observation_input_shape.html`](docs/observation_input_shape.html))
- **Hidden information → belief.** A public belief engine tracks only what is observable about the
  opponent (revealed moves/ability/item, narrowed candidate sets) instead of leaking hidden state.
- **Model.** An entity-token transformer **encoder** that outputs a policy over legal actions **and**
  a value estimate — AlphaZero-style policy+value, *not* autoregressive next-token prediction. Gen 3
  dex data is loaded generation-correctly via `Dex.forGen(3)`.

## Quickstart

Prerequisites: a **built** Pokémon Showdown checkout (so `dist/sim/index.js` exists), passed as
`--showdown-root` on each command, plus the optional `neural` extra (PyTorch) that the transformer
policy needs:

```bash
pip install -e '.[neural]'
```

Collect self-play rollouts as JSONL:

```bash
python -m pokezero.rollout_cli collect --games 50 --out runs/rollouts.jsonl \
  --showdown-root /path/to/pokemon-showdown \
  --p1-policy scripted-teacher --p2-policy scripted-teacher
```

Train the neural (entity-token transformer) policy from rollout JSONL:

```bash
python -m pokezero.neural_cli train --data runs/rollouts.jsonl --out runs/policy.pt \
  --objective behavior-cloning --showdown-root /path/to/pokemon-showdown
```

For larger CPU self-play runs, prefer compact training caches over raw JSONL and process them in
bounded shards. The training cache stores array-backed examples directly. Cache creation and
training-cache consumption default to a 50GiB active-root cap; `train` deletes consumed cache shards
after the checkpoint is safely written unless `--keep-cache-after-read` is passed:

```bash
mkdir -p runs/cache-chunk-000

python -m pokezero.rollout_cli collect-training-cache --games 1000 \
  --out runs/cache-chunk-000/cache-000 --showdown-root /path/to/pokemon-showdown \
  --p1-policy random-legal --p2-policy random-legal --window-size 4

# Repeat cache-001, cache-002, ... until the current chunk is ready, then train/delete it
# before collecting the next chunk. Keep each shard small enough that collection memory stays
# bounded; the default 50GiB cap is the on-disk guardrail for the active cache root.
# The active-root cap counts all files under that root, so keep checkpoints/raw JSONL outside it.

python -m pokezero.neural_cli train --data runs/cache-chunk-000/cache-* \
  --out runs/policy.pt --objective ppo --showdown-root /path/to/pokemon-showdown \
  --max-cache-gb 50
```

Run neural self-play iterations (collect → train → benchmark each round):

```bash
python -m pokezero.neural_cli iterate --run-dir runs/selfplay --iterations 5 \
  --games-per-iteration 512 --evaluation-games 40 --initial-policy neural:runs/policy.pt \
  --showdown-root /path/to/pokemon-showdown
```

Benchmark a checkpoint against the fixed baselines:

```bash
python -m pokezero.neural_cli benchmark --checkpoint runs/policy.pt --games 50 \
  --showdown-root /path/to/pokemon-showdown
```

## Public Prior/Belief Profile

Capture a `pokezero.public-decision-corpus.v1` sidecar from controlled raw-policy FoulPlay games.
The sidecar retains only the acting player's encoded observation/history and legal mask, public
resolved action rounds, and public belief view. It never serializes opponent observations, request
payloads, or opponent legal masks. Capture another non-overlapping seed band with
`--append-public-decision-corpus` until the corpus has at least 2,000 valid `p1` decisions:

```bash
pokezero-foulplay-capture --checkpoint runs/policy.pt --out runs/foulplay-band-001.jsonl \
  --public-decision-corpus-out runs/public-decisions.jsonl --games 128 \
  --showdown-root /path/to/pokemon-showdown

pokezero-foulplay-capture --checkpoint runs/policy.pt --out runs/foulplay-band-002.jsonl \
  --public-decision-corpus-out runs/public-decisions.jsonl --append-public-decision-corpus \
  --games 128 --seed-start 129 --showdown-root /path/to/pokemon-showdown
```

Profile raw, untempered checkpoint priors and public-belief worlds. The command rejects smaller
corpora and privileged opponent-mask mode, disables root noise, and records checkpoint, corpus,
schema, and configuration hashes in the report:

```bash
pokezero-neural prior-belief-profile --corpus runs/public-decisions.jsonl \
  --checkpoint runs/policy.pt --showdown-root /path/to/pokemon-showdown \
  --out runs/prior-belief-profile.json
```

## Components & docs

- **Self-play environment** — `pokezero.local_showdown`: a Node BattleStream-backed Gen 3 env;
  observations are built incrementally from the protocol stream.
- **Baselines** — `random-legal`, `simple-legal`, `scripted-teacher` (Gen 3 heuristic, for
  bootstrap data), and `max-damage` / `aggressive-damage` (evaluation targets).
- **Linear baseline** — `pokezero.linear_cli`: the original dependency-free masked-softmax policy.
  Superseded by the neural policy; kept for plumbing and debugging.
- **Bootstrap & promotion** — `pokezero.bootstrap_cli`, `pokezero.selfplay_cli`, `pokezero.eval_cli`:
  scripted-teacher bootstrap, the linear self-play harness, and benchmark/health promotion gates.
  Operational flags are documented in each command's `--help`.
- **Belief sidecar** — `pokezero.sidecar`: a read-only webview of the public belief state for a live
  battle room.
- **Design & background** — [`docs/`](docs/): `goals.md`, `learning_architecture_exploration.md`,
  `bootstrap_strategy.md`, `cpu_self_play_roadmap.md`, `max_damage_exploration_learnings.md`,
  `observation_input_shape.html`.
