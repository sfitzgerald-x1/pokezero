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

The new implementation branch integrates the exact candidate as a merge
parent in `d4156a07edbdd492bcacd601ffdeb14b2b0b5447`. Integration preserves
the current-main C153 and C154 historical artifacts. Its provenance pin
includes the reviewed public sleep and Yawn dependencies, and its grouped
reference test-count guard matches the actual 182 source tests. Local checks
passed all 41 existing count guards, 28 provenance tests and 131 focused
contracts. The six patch-context whitespace findings remain a diff-check
failure, not a native build failure.

Both native consumers were rebuilt in a new isolated artifact directory from
source commit `099e902ae6b53a3fcc44c9f84a894b0645161820`, using the 79 ordered
patches and PyTorch2.13.0 without a version bypass. The checked native build
fingerprint is `62dca60fde5d45ef2f6404436661465ca3f0b5bd00b86d3f76adcd8d651cca1b`.
The first model build failed with exit1 because torch-sys selected a Python
without torch. That receipt remains intact; explicitly binding PATH and PYTHON
produced exit0. Fresh imports resolve to the isolated packages, not the retained
qualification environment. Full-champion native parity has zero maximum
absolute difference at batch sizes1,8,16. All182 reference regression tests
passed; synthetic fixture vocabulary warnings remain disclosed. These are
build and regression checks, not full-game qualification or strength evidence.

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

`search_over_raw_adapters.PublicModelSearchAdapter` connects the raw control,
one-worker incumbent and twenty-worker reference through the existing runtime
APIs. It strips simultaneous opponent observations and extra actor metadata,
binds checkpoint and catalog identities, starts each root with independent
statistics, and refuses retries or raw fallback after a failed decision.
Reference receipts reconcile new trajectories, released world draws and
zero-backup deadline cancellations. The adapter records elapsed time and
nominal-ceiling overruns rather than promising a strict latency bound.
Runtime identities include the complete reference factory and dispatch width;
they must be bound by the admission controller before execution.

Thirteen adapter tests pass, including six with real Showdown opening boundaries
and instrumented selectors. They do not run actual checkpoint-backed search.
Their engineering fixture seed, 2026101009, must be excluded from prospective
panels. Preparation and direct contract generation now automatically exclude
that fixture and the twenty reference-worker startup seeds; this minimum
exclusion does not replace the full historical inventory. Oracle and incumbent
alternative-leaf configurations explicitly fail before runtime construction
until their dedicated adapters are implemented. CI runs all98 harness
contracts with a no-skips and exact-count guard.

### Reference evaluator ablations

The reference now supports HP-fraction and raw-policy terminal leaves with the
same trajectory tree, public belief sampler and champion action priors. HP
uses the signed difference between each hypothetical party's total remaining
HP divided by total maximum HP. Raw-policy leaves run masked argmax for both
seats with a separate deterministic chance domain, without consuming extra
belief-sampler RNG draws. They restore and release their owned hypothetical
snapshot before the tree continues. Source snapshots are never accepted.

A capped rollout is a refusal, not an HP or model-value fallback. An expired
rollout produces no value and backs up no partial trajectory; its worker
receipt must contain a finite, expired decision clock. The original model
leaf remains the default. Nineteen leaf tests cover signed terminals, priors,
world restoration, cleanup after failure, cancellation and receipt validation.
All182 reference regression tests pass with the existing synthetic vocabulary
warnings disclosed above. These tests do not qualify real alternative-leaf
search or admit either scientific panel.

The source qualification driver's optional `--reference-leaves` registration
adds a separate excluded ten-second experiment. It compares raw, incumbent,
reference model, reference HP and reference raw-terminal selectors at two
non-opening roots before paired terminal continuations. Its registration uses
a separate chance and root-sampling namespace, fixed250-boundary caps and
create-only attempts. Failed runs are retained and cannot be retried. The
reference variants use sequential twenty-worker pools, with construction
separately measured, rather than keeping sixty workers resident at once. The
unchanged default contract preserves the earlier source qualification's
scope; it does not relabel that earlier result as leaf qualification.

At source `ad6260f4679ef7ba46b24a015b2bbb91a1da0021`, the separately
registered benchmark completed with exit0 in 108.7286 seconds. The fixed
excluded source game supplied late requests 30 and 32. Every selector ran at
both roots; six reference selections each used 20 distinct workers. Model
leaves completed 2804 trajectories, HP leaves 2503 and raw-terminal leaves 1120.
The raw-terminal variant recorded 1575 completed and 21 deadline-cancelled
leaf evaluations, with 5585 rollout boundaries in total. Cancelled evaluations
supplied no backup value. HP produced 3802 completed leaf evaluations. No capped leaf or refusal was
accepted as a value.

At request 30, HP selected action0 while all other selectors selected action3;
at request 32, all selected action3. The 24 paired continuation outcomes across
125 boundaries were complete and every win-score contrast was zero. These
late positions from one exposed seed qualify real leaf execution and terminal
bookkeeping, not benefit on a prospective root population. Raw-terminal leaves
completed fewer trajectories under the same nominal clock; longer midgame
rollouts still require qualification. Reference selection times ranged from
10.0538 to 10.1945 seconds, so no strict ten-second ceiling is claimed.

The artifacts reside in the report archive under
`search-over-raw-reference-leaf-qualification-20261010-r1/`. Its registration
identity is
`c4165a14fc28ae1a74d232ab7174a6792fcc1f01e05c985e44f0499c81de0247`;
registration-file SHA256 is
`24f5926c31ae6fec6234421391906f84d4f8d3dd081204ad424f2e8e501c24c8`,
and terminal-file SHA256 is
`1d8f227591249f49f85d57d3affcfa1ea2eb3920699df3dcd2d7b671cce2ddaa`.
An additional read-only reconciliation passed all input, source-root,
configuration, runtime, worker-work, leaf, outcome and contrast checks while
the qualified source was still clean. Subsequent documentation changes do
not relabel the qualification source.

`scripts/qualify_search_over_raw_opening.py` registers an excluded opening
benchmark before runtime construction and permits one attempt only. It binds
clean source and Showdown commits, every native build input and artifact,
the unchanged champion, checkpoint-derived encoder tables and reviewed factory
options. Explicit hashed exports avoid writes beside the original checkpoint.
The benchmark exercises raw, incumbent and reference at one registered budget
and records actual work, construction time and selection latency. It neither
plays terminal games nor admits an outcome panel. Its ten contract tests cover
input drift, resources, exposure exclusion, one-attempt ordering and isolated
export reuse. Two older lattice fixtures required their missing request kind
to match the current action translator; production translation is unchanged.

The first real opening attempt at source `c6ad4e44` failed with exit1 after
raw selection succeeded: the incumbent root-allocation witness was not
authoritative. Reference construction was never reached. The terminal receipt
remains `FAILED_NO_RETRY`; it is not qualification success. A new boundary
regression reproduced a public transport mismatch: canonical switch candidates
carry `switched_species`, while the engine vocabulary expected a nested
`pokemon.species` row. The vocabulary now accepts the public semantic identity
without reconstructing private metadata. Unit verification of that repair does
not retroactively qualify the failed opening or authorize an outcome panel.

A separately registered check at repaired source `9ffd7d52` completed with
exit0. Raw, incumbent and reference all selected action7 at the same excluded
opening. Raw took 0.0131 seconds for one model forward. Incumbent took 0.9498
seconds, completing 854 native iterations and 1256 model evaluations without
fallbacks or prior fallbacks. Reference used 20 distinct workers and completed
65 new trajectories, 86 transitions and 76 world draws. Its selection took
1.4100 seconds against a nominal one-second budget, with 4.5132 seconds of
separately measured construction. The overrun is retained; this does not qualify
a strict one-second ceiling. No terminal game or continuation outcome was
measured, and shared selected actions provide no demonstrated improvement over
raw. This is one-opening technical evidence, not qualification across
reconstructed midgame requests, hidden-state diagnostics or alternative leaves.

### Excluded source and continuation qualification

`scripts/qualify_search_over_raw_source.py` registers one excluded raw source
game and two non-opening roots before runtime construction. It reuses the
unchanged champion and isolated native artifacts, retains source snapshots
only in the trusted auditor's memory, and selects roots by seed-derived
priorities over the completed source catalog. Search receives actor/public
contexts and public pending-transition certificates, never source snapshots
or the opponent's private observations or committed choices. The auditor
uses a snapshot only after all three arms have selected. Durable progress and
each completed continuation receipt are create-only; a failure stops without
retrying or replacing a slot.

The qualification at source `3cdc31d6018f4b1bb162f813a1e72291745f8793`
completed with exit0 in 20.8802 seconds. Its existing excluded seed2026101009
produced a complete raw-versus-raw game with 39 boundaries and 72 verified
raw decisions in 2.0856 seconds. The two selected source requests were3 and25.
At request3, raw and reference selected action6 while incumbent selected
action1. At request25, all selected action2. Eight continuations per unique
action produced24 complete outcomes and739 continuation boundaries, with no
capped or refused outcome. Identical actions reused the same outcome receipts
and remained zero-difference observations.

The changed incumbent action's eight-replicate continuation win-score delta
was +0.1875 at request3; all other contrasts were zero. These two roots belong
to one previously exposed engineering seed, not either prospective panel.
They exercise action comparison and terminal scoring but supply no population
effect estimate, held-out admission or playing-strength claim.

Incumbent selection took0.9528 and0.9481 seconds. Reference selection took
1.3323 and1.0306 seconds, retaining both overruns against the nominal one-second
budget. Reference completed193 new trajectories,287 transitions and217 world
draws across the two roots. This does not qualify a strict one-second ceiling,
the oracle/alternative-leaf paths or every difficult midgame reconstruction.
The separately preserved opening failure remains a failure.

Qualification artifacts are in the local report archive under
`search-over-raw-source-qualification-20261010-r1/`. The registration's
canonical identity is
`e3ff3e90a140b69720b2224ac76c52c9f3710c6585dd81b0e25e0e2d15869e89`.
Subsequent binding tests additionally reject altered actor history or belief
with recomputed valid checksums; the qualification receipt still names its
actual source commit, not the later test revision.

CI at source `099e902a` also failed the historical C153 live-line citation check:
three stored citations differ from current source locations. Historical C153
and C154 artifact bytes remain unchanged. A separate current-source citation
record and its validation are still needed; the failing check is not waived.

Remaining Phase A work includes full exposure inventory, complete source/model/native
and simulator admission, broader non-opening and difficult-state qualification,
the reference oracle diagnostic, incumbent evaluator ablations, broader
reference leaf qualification, timing benchmarks,
direct belief/value/fidelity checks, durable worker progress receipts and
integration of the held-out ledger with the real adapters. The excluded source
game and continuations do not open exploration or validation. Phase B also
requires a resolved both-arm admission rule, valid two-primary inference,
missing-data and stopping rules, and qualification against pinned Foul Play.

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
