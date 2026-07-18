# pokezero-search

Native search crate skeleton for the engine swap (stream S1 of
`docs/test_time_search_plan_v3.md`, "Integration endgame: the native search
crate"). A PyO3 extension that links the vendored, gen3-patched poke-engine
directly, so the whole search loop — instruction generation, apply, leaf
evaluation, reverse — runs inside Rust with zero Python/FFI crossings per
simulation step.

Two things are proven here, and only these two:

1. **The throughput regime.** A native loop over engine primitives runs at
   the same order of magnitude as poke-engine's built-in MCTS, 13–27× the
   Python-FFI loop (numbers below).
2. **The eval-hook architecture.** Leaf values flow through the pluggable
   `LeafEval` trait — the hook poke-engine's own MCTS lacks and the reason a
   custom loop is required at all. The trivial `HpFractionEval` stands in
   where the learned model (batched TorchScript/ONNX per the plan) plugs in.

Search *quality* is explicitly not a goal: `puct_search` is one ply deep with
uniform priors. Do not read strength into its outputs.

## Building

The engine source is vendored, never committed (`third_party/poke-engine-src/`
is gitignored). From the repo root:

```sh
scripts/vendor_poke_engine_src.sh            # fetch sdist 0.0.47 + apply gen3 patches
uv pip install --python .venv/bin/python maturin   # once
cd rust/pokezero-search
.venv/bin/maturin build --release -i <venv-python>  # or: maturin build --release
uv pip install --python <venv-python> --force-reinstall target/wheels/pokezero_search-*.whl
```

(`maturin develop` does not reliably target a venv outside the crate dir;
build-then-install is the supported path.)

The Cargo `path` dependency points at the sdist's workspace root, which *is*
the engine crate (`src/gen3/` lives there; the `poke-engine-py` member is
ignored), with `default-features = false, features = ["gen3"]`. The same
residual-order patch used by `scripts/setup_poke_engine.sh` is applied with
`--fuzz=0`, so a future version bump fails loudly instead of silently
mis-patching.

## API

- `bench_apply_reverse(state_str, s1_move, s2_move, iterations, branch_on_damage=True) -> float`
  — parse once, then loop generate_instructions → apply first branch → cheap
  value read → reverse, entirely in Rust; returns steps/sec.
  `branch_on_damage=True` matches what poke-engine's MCTS does at the root ply
  only; pass `False` to price the deep-tree regime (all non-root plies).
- `puct_search(state_str, iterations, c_puct=1.4, seed=0) -> str` — minimal
  one-ply PUCT with decoupled simultaneous-move selection (side two maximizes
  `1 - value`), stochastic branch sampling by engine percentages, terminal
  detection, and leaf values through `LeafEval`. Returns JSON with visit
  counts and mean values per root move for both sides.

State strings come from the production adapter:
`pokezero.poke_engine_adapter.build_poke_engine_state(spec).to_string()`.

## Measured throughput

Apple M-series laptop, single thread, `--release` (lto), 1M iterations per
cell; states built via `pokezero.poke_engine_adapter` (minimal 1v1 fixture and
a 6v6 gen3-OU-style team). Python-FFI baseline measured in the same session:
a Python loop calling the poke-engine binding's `generate_instructions` on the
same state/move pair. 2026-07-18.

| Loop | 1v1 minimal | 6v6 OU-style |
|---|---|---|
| **In-Rust apply/reverse, deep-tree regime** (`branch_on_damage=False`) | **1,549,311 steps/s** | **668,677 steps/s** |
| In-Rust apply/reverse, root regime (`branch_on_damage=True`) | 537,718 steps/s | 189,522 steps/s |
| `puct_search` full loop (select + generate + sample + apply + eval + reverse + backprop) | 1,070,187 iters/s | 1,592,509 iters/s |
| Python-FFI loop, same state/moves (measured here) | 57,152 calls/s | 32,542 calls/s |

Reference baselines (`docs/engine_search_poc.md`, model-cost ladder):

| Baseline | Throughput |
|---|---|
| Native poke-engine MCTS (handcrafted eval, in-engine) | ~580k–1.08M sims/s |
| Python-driven loop over engine primitives (FFI, no NN) | ~33–46k sims/s |

Reading:

- The deep-tree in-Rust rate (0.67–1.55M steps/s) sits squarely in the native
  MCTS band — the native-loop regime is confirmed with our own tree code and
  eval hook in the loop. Speedup over the same-state Python-FFI loop: 20–27×
  (13–40× against the documented 33–46k band).
- Our measured Python-FFI baseline (32–57k) reproduces the documented one;
  the spread is state complexity (the 1v1 fixture is cheap to convert).
- The 6v6 `puct_search` rate exceeds its own bench rate because PUCT
  concentrates visits on cheap move pairs (e.g. `protect`, few branches);
  the bench rate prices a fixed, damage-branching move pair every step.
  Caveat it accordingly — it is not a like-for-like comparison.
- None of these numbers include model forwards. Per the model-cost ladder,
  a model-priced loop is bounded by inference (~168–309 evals/s CPU,
  ~1.7k MPS b=64); the native loop's job is to make everything around the
  forward free, which these numbers demonstrate.

## What is deliberately not here yet

- Multi-ply tree (the `puct_search` skeleton is root-only by design — the
  plan re-prices depth after the swap).
- In-tree leaf batching and native model inference (tch-rs / ONNX Runtime)
  behind `LeafEval`.
- The Rust v2.2 encoder (track B's deliverable; validated bit-exactly by the
  golden corpus before anything is trusted).

## Tests

`tests/test_pokezero_search_crate.py` — skips cleanly unless the built module
imports; otherwise smoke-tests both entry points, JSON shape, seed
determinism, and a loose regime floor (>100k steps/s).
