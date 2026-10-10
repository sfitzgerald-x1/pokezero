# Search over raw evidence and implementation

October 10, 2026

The new investigation asks whether either search arm improves on raw argmax,
including play against Foul Play. It starts with decision diagnostics and
held-out validation. The historical wider comparison and eight qualification
games are not confirmation evidence for this question.

## Preserved candidate and previous findings

The exact qualified candidate is published at
[`4e051535cc9aa0e6e7473bafdb48fdbca69bd189`](https://github.com/sfitzgerald-x1/pokezero/tree/4e051535cc9aa0e6e7473bafdb48fdbca69bd189)
in [draft PR 1507](https://github.com/sfitzgerald-x1/pokezero/pull/1507).
Publication preserves its original bytes and history; it does not establish
merge readiness, clean-build reproduction or strength. Its 58-commit source
line changes 145 files relative to the PR1504 merge. Do not rewrite this
qualified commit while repairing current-main integration.

The [October5 screen](paper_trajectory_reference_comparison_20261005.md)
reported no demonstrated action-quality gain at a nominal one second and
retains its STOP disposition. The later nominal ten-second full-game search
with persistent statistics tested dimensions outside that screen. It did not
reverse the earlier finding or establish significance.

The historical fixed roster remains 64 seeds and 256 games. Ninety accepted
games contain 33 reference wins and 10 losses, versus 30 incumbent wins and 17
losses. The remaining 166 cells are uncertain: four refusals, one unvalidated
complete terminal and 161 unstarted games. Full-roster identification bounds
are [-0.609375,0.6875]; making all 13 source-affected clusters uncertain gives
[-0.8125,0.875]. These are finite-ledger bounds, not confidence intervals.
The historical collection stopped on a refusal and its validation pipeline
failed with exit 1. Do not restart or pool it into the new investigation.

Separate technical qualification completed eight excluded games, 357 durable
boundaries and 140 reference decisions with 167108 new trajectories and 374727
transitions. Full games exercised no staged particle populations; evidence
for the difficult conditioning path remains the fixed-nine diagnostic.
No exact posterior, strict ten-second latency, equal-compute benefit or
playing-strength claim follows from qualification.

## Portable evidence identities

The original local archive is not yet replicated in this repository. Paths
below are relative to its report archive root, not promises that the artifacts
can currently be downloaded from GitHub. Hashes identify exact immutable bytes.
A portable archive export and verifier remain implementation work.

| Artifact within the report archive | SHA256 |
| --- | --- |
| wider-conditioning-guided-fullgame-qualification-20261008-r1/READOUT.json | 39a8fd03ee457306d21409d736c6cb7e55a41b5adaddbf56760474054c86af22 |
| wider-conditioning-guided-fullgame-qualification-20261008-r1/registration.json | 065b7716e13bcdaba08456981068d58827f4ea72107bc9483a4cf920274667db |
| wider-search-evidence-audit-20261006-r1/fullgame-conditioning-qualification-independent-review-r1.json | 5c3b69cb43a35552212863b0de785d3b7807f9c4b30b21341a26ec1dd78a5dff |
| wider-search-evidence-audit-20261006-r1/fullgame-reference-terminal-work-profile-r1.json | a900ba9df1858b57eac0006c43fc973b4fbce51e52b0490d9cbeb1b2b13d12a3 |
| search-statistical-amendment-20261009-r1/bounds/RESULT.json | a54e1b2f1fd01cc9e027289fe48f45b07fffc69baccca86d903a49fa6bb79b63 |
| search-statistical-amendment-20261009-r1/CALIBRATION_RESULTS.json | d2c21c9626c0fa54b555becf578fdae9f119617c34026edc5907cb51323e6d74 |
| search-statistical-amendment-20261009-r1/REFUSAL_POLICY_RESULTS.json | 8536e14a1b4e65f00217a8b9db19a7e6b034454203f11958baf7c77331188a94 |

## Native engine build requirements

Use a separate checkout at the exact published candidate commit and a new
runtime; do not install into or overwrite the retained qualification runtime.
The engine is based on poke-engine0.0.47, verified against its source archive,
with the ordered patches in `third_party/poke-engine-gen3-patches.txt`, including
the synthetic-Struggle patch. Both consumers must use that same patched tree:
the Python engine package and the native model-search crate.

The existing scripts provide the build path:

```sh
scripts/setup_poke_engine.sh /path/to/new/runtime/bin/python
scripts/vendor_poke_engine_src.sh /path/to/new/runtime/bin/python
scripts/build_search_crate_model.sh /path/to/new/runtime/bin/python
```

The model crate requires PyTorch2.13.0 with tch/torch-sys0.26.0 and
`LIBTORCH_USE_PYTORCH=1`; bypassing the version check is forbidden. Install
the repository's neural dependencies and build prerequisites in the new
runtime first. Verify both engine consumers, model-leaf parity and the patched
Struggle surface before qualification. Preserve the patch order, archive
verification and source hashes in the resulting receipt.

The old source-bound native build receipt reports exit0 and patched-tree
SHA256 `6d5dc090a74d4944e14c08aab858fbf99cc7e22282b58b6a9c7d9d4564709113`.
That retained build is evidence of the old execution environment, not proof
that the recipe above has just been reproduced from a clean checkout.
PR1507 currently has provenance and test-count CI failures; frozen patch
context also causes six diff-check whitespace findings. Those failures are
not waived by the qualification receipt.

## Phase A implementation status

`pokezero.mcts_eval.search_over_raw` implements disjoint prospective seed
panels, 400 registered root slots across 64 source seeds, an explicit 37-option
exploration roster, eight paired continuations per unique selected action,
and source-seed bootstrap summaries. Identical raw/search actions remain
zero-difference observations. Capped outcomes remain uncertain. Oracle
configurations cannot be selected for deployment, and a validation summary
must match the frozen exploration selection.

`scripts/prepare_search_over_raw.py` prepares a create-only contract and binds
the supplied exposure registrations and approved plan. Preparation alone is
not an execution-ready scientific registration. The module's bootstrap gate
is an approximate decision-level screen, not a full-game error-control
guarantee. It does not authorize Phase B.

`search_over_raw_source` collects raw-versus-raw source games through the
existing rollout driver. Its policy wrapper verifies every decision against
the checkpoint's masked probability row and rejects altered selection laws
or multi-snapshot checkpoints. Roots are sampled across the full completed
public request catalog, independently of the source winner. Short games leave
missing slots; capped games supply no replacement catalog. The collector
exports canonical public records, not trajectories or private snapshots.

`search_over_raw_ledger` binds the contract and its external input hashes,
records exploration evidence without overwriting it, and freezes selection
before claiming a single held-out attempt. The claim is persisted before the
worker starts. Failures and orphan claims do not permit a second attempt;
missing roots retain their full uncertainty. This is drift detection and
execution ordering, not authentication or runtime qualification of arbitrary
callbacks. Public search adapters and their qualification must enforce the
same boundaries before real validation.

Verification passed 37 focused tests and 24 adjacent rollout regression
tests. Source-collector tests exercise the production rollout driver with a
scripted environment and policy, not a real checkpoint or Showdown battle.
No scientific game or search has been launched by these tests.

Remaining Phase A work includes full exposure inventory, source/model/native
and simulator bindings, real-game root-collector qualification, public-only policy adapters,
the reference oracle diagnostic, evaluator ablations, timing benchmarks,
direct belief/value/fidelity checks, durable worker progress receipts and
integration of the held-out ledger with the real adapters. No new battle has
been launched by this preparation.

## Later gates and completion criteria

Both deployable search arms must be qualified against the pinned Foul Play
bridge before the full-game study. Two primary search-minus-raw contrasts
against Foul Play share the family error budget; raw-opponent contrasts are
secondary. Choose an explicit resource ceiling rather than claiming equal
CPU allocation from equal wall-clock time.

Calibrate candidate tests with the actual missing-data and interim policy.
Testing six simulated nulls is necessary stress evidence, not a universal
false-positive theorem. Worst-case endpoints are analytical uncertainty,
not historical refused games reclassified as losses. Their monotonicity for
positive-stake betting does not automatically validate a t-test. Worker-state
and provenance failures halt execution irrespective of isolated-game allowance.

Full-game collection depends on held-out deployable gain, valid analysis,
reliability evidence and measured cost. Training remains conditional on
direct evidence implicating the evaluator. The active implementation goal is
not complete until the applicable phases and their measured gates are resolved.
