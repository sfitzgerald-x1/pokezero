# PokeZero

PokeZero is a **work-in-progress** experiment in training an agent to play Pokémon Showdown
**Gen 3 random battles** through self-play, on CPU-first hardware. The aim is AlphaZero-style —
learn from self-play with a policy/value network — applied to an imperfect-information,
simultaneous-move game.

**How we measure progress:** win rate against fixed baselines. Beating `random-legal` and
`simple-legal` is table stakes (~0.9–1.0); the real benchmark is the **`max-damage`** baseline,
which is the current open problem (see Status).

> ⚠️ Active research. Encodings, APIs, and checkpoints change frequently. The neural policy below
> is the current frontier; the linear baseline and parts of the harness are earlier scaffolding
> kept for reference.

## Status

- **Beats random/simple decisively; plateaus against max-damage.** Across ~20 training-method
  variants, win rate vs max-damage sits in a tight **~0.46–0.52 band** — a ceiling, not active
  climbing. (vs random/simple is table stakes, not a progress signal.)
- **Why:** the imitation ceiling. The scripted teacher itself only beats max-damage ~0.57, and you
  cannot out-clone your teacher — behavior cloning / DAgger caps the policy near (slightly below)
  the teacher. Exceeding max-damage requires RL that *discovers exploitation*, or search.
- Full analysis: [`docs/max_damage_exploration_learnings.md`](docs/max_damage_exploration_learnings.md).

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

## Direction (what we're working toward)

1. **Make the RL line measurable** — more PPO-from-DAgger iterations with a larger vs-max-damage
   eval (≥300–400 games) to tell a real climb from eval noise.
2. **If the single-pass policy stays capped, add search** — AlphaZero-style MCTS over the real
   Showdown simulator, using the belief engine to determinize hidden information. Higher ceiling
   than a feed-forward policy, and it reuses assets we already have (a fast sim + the belief engine).
3. **Broaden the opponent set** beyond a single baseline so training rewards general skill rather
   than exploiting one fixed opponent.

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
