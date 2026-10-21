# GPU engine throughput spike

This isolated experiment asks whether a sampled-chance Gen 3 transition,
placeholder observation and the champion forward can sustain 100 completed
games per second on one GPU. It is not a production engine or training teacher.
Nothing here is imported by the existing battle collector or search runtime.

## Current implementation

Day 1: batched structure-of-arrays storage, per-battle PRNG streams, and the
nine-action layout. No transition, real-team pool, encoder, policy forward,
fidelity result or games-per-second benchmark is implemented yet. Empty states
are storage fixtures and are terminal by default; they are not benchmark games.

State has six party members per side; all fields are JAX arrays with a leading
batch dimension. Types, status, items, abilities and moves require the eventual
team pool's vocabulary. Bench actions enumerate party order excluding the
active slot. Move0 represents synthetic Struggle only when no move is usable.
The initial mask handles PP, externally resolved disabled/trapped flags,
forced replacement and fainted slots; Taunt, Encore, recharge and other
restrictions still require the transition and are not claimed as supported.
For unilateral forced replacement, the future transition must suppress the
other seat's action; this mask is not yet a complete request-phase protocol.

## Run the first device check

From the checkout root, in a fresh Python 3.12 environment on a supported NVIDIA GPU:

```sh
pip install -r gpu_engine/requirements-cuda.txt
python -m gpu_engine.probe
python -m unittest discover -s tests -p test_gpu_engine_state.py -v
```

Select the CUDA wheel against the installed driver using the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
The probe requires exactly one visible GPU and waits for a compiled operation.
It is a capability check, not an engine throughput measurement.

On October 21, 2026, JAX 0.12.0 on one NVIDIA GB200 (ARM64, driver 580.178.04)
passed the synchronized GPU probe and all nine state/action-mask tests. CUDA
reported an unavailable FABRIC memory-handle path and successfully retried with
simpler handle types; no permission or driver setting was changed. This proves
the first device capability, not engine speed or game correctness.

## Measurement contract

Use real Showdown-generated teams (at least 200,000), a frequency-led coverage
ledger and explicit placeholders. Target 85% move coverage before the final
throughput table. At batches 256, 1,024, 2,048 and 8,192 report synchronized engine,
placeholder encode, both-seat champion forward and end-to-end times separately;
games/second must use measured completed games/turns, never an assumed length.
Search-shaped expansion covers all 81 joint actions for 256, 1,024 and 4,096 nodes.

The predeclared go condition is at least 100 end-to-end games/second at batch 2,048,
engine step under 100 ms and fewer than 10% of 200 games diverging on covered
mechanics by turn 20. 30–100 games/second means reconsider; below 30 or no batch
scaling means no-go. These thresholds are not yet measured. Missing components
must remain missing in reports, not receive projected or placeholder timings.
No training, policy-quality or full-fidelity claim follows from this spike.
