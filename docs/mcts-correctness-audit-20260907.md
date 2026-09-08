# MCTS correctness audit — 2026-09-07

Base: `dacb6358d9b145ce069d6718662a38f581a38bc0` (`origin/main`).
Scope: local CPU verification and source changes only; no cluster changes,
checkpoint changes, deployments, or claims of increased playing strength.

## Demonstrated defect and repair

The real Rust batched tree permanently mixed temporary traversal reservations
into backed-up values. `tree::traverse` incremented a chance branch's visit
denominator before its result existed. `tree::finalize` then computed a permanent
chance expectation while other collected traversals still inflated that
denominator. This is not merely ordinary batched PUCT exploring differently:
even an exact terminal value became wrong.

New CPU tests drive the production `traverse`/`finalize` functions, with actual
engine-generated outcomes and deterministic collisions at the root. Before the
fix, the focused test command failed (exit **101**) with:

- Two visits to a guaranteed immediate side-one win: root Q **0.75**, not **1.0**.
- Two visits to a fixed nonterminal value **0.8**: root Q **0.6**, not **0.8**.

The repair moves the chance-branch visit increment to `finalize`, alongside its
real value. Decision-arm virtual loss is unchanged: each selecting seat still
gets a temporary loss to spread selections. Deferred initial leaf values still
resolve in collection order before dependent backups. No search parameter or
experimental feature switch was added.

Four new tests verify:

1. Exact terminal win/loss orientation, with seat-mirrored engine fixtures and
   batches 1, 2, 8, 64; no model evaluation or child expansion at terminal leaves.
2. Constant ready/deferred leaf values remain 0.8 at those batch sizes.
3. A forced 64-traversal, four-depth path preserves 0.8 at every node and chance
   branch, including multiple pending expansions in the same batch.
4. An already-expanded 25%/75% terminal lottery preserves its exact expectation
   after every backup, regardless of which chance branch was sampled.

Engine state is also checked for exact restoration after each fixture traversal.
The defect's general asymmetry was already discussed in the old collision-ledger
comments. This work provides a concrete red/green invariant and repairs it.
The model-priors campaign path is explicitly documented as using batch 64 in
`src/pokezero/engine_search.py`; this is not confined to the sequential rollout
path's separately acknowledged `leaf_batch > 1` mode.

## Other audit findings

- Existing Rust tests pass for side-one-absolute values, side-two minimization,
  same-expectation backup to both seats, terminal orientation, chance probability
  conservation, and depth/mirror checks. The model boundary converts self-relative
  values once in `model.rs`; no new sign or ply-parity defect was demonstrated.
- Python root-search tests pass. Its terminal values are self-relative +1/0/-1
  (win/tie/loss); the Rust tree uses side-one-relative [0,1]. Those are deliberately
  different contracts, not evidence of a missing per-ply negation.
- The engine's double-wipe verdict remains a **previously documented unresolved
  issue**, not a discovery or fix here: `tests/gen3_terminal_options.rs` records
  that simultaneous Gen 3 wipes resolve to a side-two win rather than a tie.
  Correcting the shared engine verdict contract is separate work.
- Decoupled simultaneous-move PUCT was not proven game-theoretically optimal.
  No claim is made that these tests establish equilibrium play or calibrated
  model values.

## Verification

Fresh engine vendoring used the repository script and verified the pinned
`poke-engine==0.0.47` archive SHA-256
`84a7dfad5ce4650a2cb9250999597c594385069eb33622c6a14bb1279694b434`.

Run from this worktree:

```sh
PYO3_PYTHON=/Users/scott/workspace/agents/pokezero-agent/hceval-venv/bin/python CARGO_BUILD_JOBS=4 cargo test --manifest-path rust/pokezero-search/Cargo.toml --lib colliding_batch -- --nocapture
PYO3_PYTHON=/Users/scott/workspace/agents/pokezero-agent/hceval-venv/bin/python CARGO_BUILD_JOBS=4 cargo test --manifest-path rust/pokezero-search/Cargo.toml -- --quiet
.venv/bin/python -m unittest tests.test_search tests.test_search_policy tests.test_search_benchmark -q
git diff --check
```

Observed results: focused tests **4 passed**; final broad default-feature Rust run
independently repeated by the primary agent **250 library + 334 integration tests
passed**, one pre-existing ignored renderer test; Python **170 passed** and
independently repeated; whitespace check passed. All these post-fix test commands
exited 0. Existing compiler warnings were not changed.

A separate read-only reviewer found no blocking issue in the final patch after
checking chance sampling, initial branch samples, collection-order finalization,
nested propagation, and unchanged decision-arm virtual loss. The reviewer did
not rerun the Rust suite; the independent reruns above were by the primary agent.

The additional model-feature build was attempted and **failed (exit 101)**.
The first invocation selected a Python without torch; retrying with the shared
venv's `bin` first on `PATH` reached C++ compilation, then failed because the
installed torch **2.13.0** no longer provides `Tensor::align_as` and
`torch::align_tensors` expected by the pinned `torch-sys 0.24.0`. The retry used
the repository recipe's `LIBTORCH_USE_PYTORCH=1` and
`LIBTORCH_BYPASS_VERSION_CHECK=1`; bypassing the version check did not establish
compatibility. No dependency, shared environment, or build source was changed.
The other known source venv had no torch installed (import check exited 1).

End-to-end Python/native model orientation, production checkpoint replay, and
matched-compute playing-strength tests therefore remain **unverified** in this
audit. The actual shared tree seam is tested without libtorch, not mocked, but
that does not replace a model-feature integration run on a compatible toolchain.
Batch > 1 still changes the selection schedule versus sequential PUCT; the fix
does not certify batch-size equivalence or erase existing fidelity safeguards.
Historical batch results should not be relabeled as corrected results.
