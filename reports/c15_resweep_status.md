# C15 Re-sweep Contract Status

`c15_resweep_spec.json` is an immutable historical registration for the
48-patch build attested by `c25_cert_contract_attestation.json`. It was never
executed, and it is not a launch contract for the current engine.

The contract's source commit, engine fingerprint, script hashes, calibration,
and eight reserved seed bands remain evidence of the original registration.
They must not be updated in place to describe a later build, and the seed
bands must not be reused.

A current-engine certification needs a separately named successor contract
with a fresh source snapshot, archival recalibration, attestation, and a new
seed reservation. The C15 contract remains available only for audit and
regression-reference purposes.
