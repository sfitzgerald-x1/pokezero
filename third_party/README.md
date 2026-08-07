# third_party

External dependencies pulled in as git submodules. Code here is **not** pokezero's and may carry
its own license — keep it at arm's length.

## foul-play (`third_party/foul-play`)

A competitive Pokémon Showdown battle bot (set prediction + eval-guided MCTS over `poke-engine`).
We use it as a **stronger-than-max-damage eval opponent** for Gen 3 random battles — a more
realistic strength yardstick than the max-damage baseline.

### License boundary — important
- **foul-play is GPL-3.0; pokezero is MIT.** To keep pokezero MIT-clean, foul-play is used
  **arms-length only**: it runs as its **own process in its own venv** and battles our bot over the
  local Showdown server. **Do not import foul-play (or its GPL modules) into pokezero code** — that
  would link GPL into an MIT codebase. Mere aggregation (two programs playing each other over a
  protocol) is fine; linking is not.
- `poke-engine` underneath is MIT, but foul-play's own code is GPL — the boundary is foul-play.

### Eval use (Gen 3 random battles)
foul-play is an opponent bot, not a library. To run it as an eval opponent:
1. In foul-play's own venv, install `poke-engine` built for **gen3** (`--features gen3`).
2. Run foul-play with `--pokemon-format gen3randombattle` pointed at the local Showdown server.
3. Have our bot challenge/accept it; score win rate over a milestone-size sample (≥300 games).

This is an **eval/sparring opponent only — never a training teacher.** Cloning it would relocate the
imitation ceiling to a higher floor and abandon the learn-from-scratch goal.

### Build + run the benchmark
```sh
# one-time: build foul-play's gen3 engine + venv (needs rustup/cargo + uv)
scripts/setup_foulplay_eval.sh

# run N games: our checkpoint vs foul-play on a local --no-security Showdown server
POKEZERO_SHOWDOWN_ROOT=/path/to/pokemon-showdown \
  scripts/benchmark_vs_foulplay.sh runs/<...>/transformer-policy.pt 50 1000
```
`setup` applies `foulplay-local-nosec.patch` — a small env-gated change so foul-play claims its
name with an empty assertion against a local `--no-security` server (it otherwise fetches a guest
assertion from the hosted login server, which is invalid for a local server's challstr). The patch
lives inside foul-play's tree (GPL); pokezero only ships the `.patch` and the arms-length runner.
Smoke-validated: ran two `gen3randombattle` games end-to-end (foul-play set prediction working).

### Updating the pin
Bump the pinned commit in a dedicated PR: `git submodule update --remote third_party/foul-play`.

## poke-engine gen3 patches (`third_party/poke-engine-gen3-*.patch`)

`third_party/poke-engine-gen3-patches.txt` is the **single source of truth** for which
patches ship and in what order. Both builders read it. A `.patch` file sitting in this
directory is *not* part of the engine unless that manifest names it.

### One patch here is deliberately NOT in the manifest

`poke-engine-gen3-enumerate-damage-rolls.patch` is a **reference oracle, not a shipped
fix**. It replaces the collapsed damage-roll partition with one arm per distinct
`floor(max * r / 100)` and is exact where the collapsed path is an approximation, so it
is used to validate the collapsed path's masses — never to run them.

Build it into a **separate venv on a separate checkout** and regenerate the pinned truth
with `scripts/collapsed_arm_mass_oracle.py --write`; see that module's docstring.
`tests/test_collapsed_arm_mass_oracle.py` then consumes the pin, so ordinary CI needs no
second build.

**Do not add it to the manifest.** `reports/c137_phase2_enumerate_decision.md` records why:
it changes the engine production searches with (the argmax moves), the differential must
keep measuring the shipping configuration, and the mass gate's own negative control has no
collapsed fan left to be a control over once enumeration is on.
