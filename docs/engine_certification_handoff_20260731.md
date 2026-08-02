# Engine Certification Continuation Handoff

## Status update (successor pass, 2026-07-31)

Read this before acting on the Remaining Work below; most of it is done and one
item is deliberately held.

- **Remaining Work 1 — done.** C26 is registered:
  `reports/c26_current_engine_resweep_spec.json` and
  `reports/c26_current_engine_calibration.json` landed in one commit, the
  lifecycle is at `contract_registered_attestation_pending`, and C15/C25 bytes
  are unchanged. Eight fresh blocks from 16,000,000. The calibration is C26's
  own, produced by `scripts/c26_archival_recalibration.py`; nothing was copied
  from C15. The launcher now derives seed identity from the registered contract,
  rejects malformed / duplicate / overlapping reservations, and records the
  selected reservation in shard and aggregate provenance (private repo).
- **Remaining Work 2 — done for the contract attestation.**
  `reports/c26_cert_contract_attestation.json` is in a later commit than the
  registration, and the existing `engine-cert-contract-attestation/v1` schema
  fit C26 unchanged, so no historical check was touched or weakened. The
  *sweep-result* attestation schema is still open and belongs with the sweep.
- **Remaining Work 3 — HELD, and the reason matters.** A 60-game fresh probe of
  the frozen build fires the predicted-zero counter
  `structural_component_count_without_supported_sibling` on 29 of 64 divergent
  rows, projecting ~4,800 unattributed rows in the registered sweep. C26 would
  FAIL. See `docs/engine_c26_structural_residue_diagnosis_20260731.md`. The
  registered contract is untouched and its blocks are unspent.
- **Remaining Work 4 and 5 — blocked** on 3 by construction.

The next real question is the surviving residue (~670 rows per 10,000 games
after every defensible rule rescoping), whose shape is a damage component and a
cancelling `heal_to_full` / `itemleftovers_to_full` component decomposed
differently by the two sides. Any repair changes `cert_sweep_readout.py` or the
engine, which invalidates the frozen C26 source identity: freeze a new source,
re-register contract plus calibration, then sweep. Do not amend C26 in place.

## Purpose

Continue the engine-certification program from the immutable C26 build-source
stage. The objective is to establish whether the current Rust battle engine has
zero **unexplained** divergence against the reference simulator on a fresh,
pre-registered sample. Raw divergence is not required to be zero: documented
comparison limits may remain only when they satisfy their registered bounds.

This document intentionally contains no infrastructure configuration. Run
coordination details belong in the private deployment tooling; public evidence
and contracts belong in this repository.

## Current State

The repository is at the C26 `build_source` lifecycle stage:

- `reports/certification_contract_lifecycle.json` records `stage:
  "build_source"` and `launchable: false`.
- The frozen source contains 51 engine patches with fingerprint
  `776fa1e15cec731d3223b493fab992dc042fd3da8bdee6d4b8c9dc1a1d192c9c`.
- The certification readout and execution-manifest producer hashes are pinned
  in the lifecycle record.
- The following C26 artifacts are intentionally absent and must remain absent
  until they can be registered together:
  - `reports/c26_current_engine_resweep_spec.json`
  - `reports/c26_current_engine_calibration.json`
  - `reports/c26_current_engine_attestation.json`

Historical C15 and C25 artifacts have been restored as immutable evidence.
They are useful for calibration and regression context, but they are not
evidence that certifies the C26 source.

## Work Completed

1. The certification lifecycle now prevents a partial C26 registration. Tests
   require the source freeze to contain no successor contract, calibration, or
   attestation artifact.
2. The C26 source identity has been frozen after the current engine patch
   stack. Do not change the engine, reader, or manifest producer after
   registering C26 without starting a new source-freeze cycle.
3. The retained historical divergence population has been re-read against the
   C26 engine for calibration purposes. It contains 3,821 complete historical
   rows; the reread produced 3,496 divergent outcomes, 324 matched outcomes,
   one lossily stored historical row that cannot be reconstructed, and no
   reread errors. This is an archival calibration result, not a certification
   measurement.
4. The C26 execution design requires a selected fresh seed reservation to flow
   through shard and aggregate provenance, with malformed or overlapping
   reservations rejected before work starts. Verify that enforcement in the
   execution launcher before submitting the registered sweep.

## Remaining Work

### 1. Register C26 without mutating historical evidence

Create the C26 resweep specification and archival calibration artifact in the
**same commit**. The registration must:

- bind the C26 source commit, 51-patch fingerprint, readout hash, and
  execution-manifest producer hash from the lifecycle record;
- identify the archival source as historical calibration only;
- record the one lossily stored archival row as skipped, rather than silently
  treating it as a match;
- reserve eight fresh, non-overlapping 1,250-game blocks, distinct from all
  prior certification and probe reservations;
- verify that the execution launcher rejects malformed, duplicate, or
  overlapping reservations and records the selected reservation in shard and
  aggregate provenance;
- preserve the C15/C25 artifact contents and hashes unchanged; and
- change the lifecycle to `contract_registered_attestation_pending` only after
  both C26 contract and calibration files validate.

The existing C15 contract in
`reports/c15_resweep_spec.json` is the structural template. Do not copy its
source identity, seed blocks, or historical calibration claims into C26.

### 2. Keep attestation separate from registration

The C26 attestation must be produced in a later commit after the fresh sweep
has completed. If the current attestation schema is restricted to C15 paths,
generalize it with explicit historical-C15 protections or add a distinct C26
successor schema. Do not weaken the historical checks to make C26 fit.

### 3. Run the fresh C26 certification sweep

Use the registered contract to run eight resumable 1,250-game shards and a
provenance-validating aggregate, for 10,000 distinct fresh games total. The
execution must retain the registered repro population, run the required
behavioral probes, and write completion evidence only after each shard is
complete. All published readout inputs must identify the registered source,
contract, and seed reservation.

Treat failures honestly:

- A contract or provenance mismatch is terminal and must fail before any
  measurement is accepted.
- A transient execution failure may resume only after existing shard evidence
  passes the contract's provenance checks.
- An unattributed row, new unregistered mechanism, coverage shortfall,
  incomplete retention, duplicate or missing seed, or failed probe is a C26
  certification failure. Record it and return to diagnosis; do not broaden
  attribution after inspecting the result.

### 4. Produce and validate the C26 attestation

After a complete sweep, write the C26 attestation against the immutable C26
contract. The lifecycle can leave
`contract_registered_attestation_pending` only when this attestation exists
and validates. It must state both the raw divergence results and the
unexplained-divergence result.

### 5. Run the throughput smoke only after certification passes

The follow-on smoke measures wall-time behavior from 0 through 16,000 games.
It is gated on a successful C26 certification outcome. Record the per-iteration
wall time and the improvement against the declared baseline; a performance
result cannot substitute for correctness certification.

## Certification Standard

C26 passes only if all of the following hold on the registered fresh run:

- zero unattributed rows;
- zero engine errors;
- complete, non-truncated repro retention;
- all registered comparison-limit families remain at or below their bounds;
- all predicted-zero counters remain zero;
- the minimum full-round coverage and measured-coverage fraction are met;
- exactly 10,000 distinct games occur in the registered seed blocks; and
- every shard, aggregate, readout, and attestation agrees on the frozen source
  identity and contract provenance.

This is a strict correctness gate. A negative result is useful evidence and
should be reported as such rather than converted into a contract amendment.

## Key Public References

- Lifecycle and C26 source identity:
  `reports/certification_contract_lifecycle.json`
- Historical contract template: `reports/c15_resweep_spec.json`
- Certification reader: `scripts/cert_sweep_readout.py`
- Manifest producer: `scripts/cert_execution_manifest.py`
- Historical audit ledger: `docs/engine_divergence_ledger_20260728.md`
- Historical evidence protections:
  `tests/test_cert_historical_attestation.py`
- Source-freeze manifest protections:
  `tests/test_cert_execution_manifest.py`

## Scope Boundary

This handoff covers certification, not search strength, model training, or
engine feature expansion. If a new engine patch is necessary to resolve a C26
failure, freeze a new source identity and repeat the registration sequence;
do not amend a registered C26 contract in place.
