# B3 owning Python versus JAX study

This is source preparation for the objective's end-to-end encoder comparison.
It does not authorize collection. The root workstream controls B2 admission,
independent scientific reviews, runtime/cost admission and the eventual launch.
A4 remains closed. Preparing a prespec consumes no games.

The study uses the unchanged raw champion iteration 9375 (file SHA256
`0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be`),
the pinned Showdown revision `f76228a1354b5d0f307ca2d16101294ad3a2308b`,
and time-budgeted FoulPlay at 1000 ms, one search worker/thread. There is no search
arm, value-head replacement, privileged input, retry, fallback or refill.

## Fixed roster and invocation order

There are 500 battle seeds, 30100000 through 30100499. Each is played from both
learner seats, once per encoder: 1000 planned games per arm, 2000 total. Each
matching seed-seat pair uses FoulPlay seed 40100000 plus the battle ordinal.
Both seats stay together inside the statistical seed cluster.

Four serial 500-game invocations run in this order, frozen before outcomes:

1. Python owning, learner p1.
2. JAX owning, learner p1.
3. JAX owning, learner p2.
4. Python owning, learner p2.

This ABBA order reverses encoder order between seats and balances the average
ordinal position of the arms. Time-budgeted FoulPlay remains nondeterministic;
matching seeds does not establish a deterministic counterfactual. No additional
blocking is introduced merely for ordering. Each invocation gets a fresh
factory and a single ordinary benchmark model load. Within its 500 games, the
factory shares only its catalog-bound pure JAX functions and creates fresh
provider leases, host/device state and raw-policy history for every game.

The seed-exclusion metadata covers original A2 (root
`a2-mirror-20261029-5e4338a880d9`, 256 games) and corrected A2 (root
`a2-corrected-owning-warmstart-20261030-r1`, 4096 games). Their reconstructed
cohort-file hashes exactly matched the saved completion metadata:
`3a3ac9cbb285140603304b349203f0cf52fde7c540c586c6b242b8296a8a9350` and
`7c7e1d624d2a83ac504c43ede598c4c9fddbc5cccdce06b4a6b0fcb395dd3355`.
The union has 4352 unique seeds; neither proposed band overlaps it or the other
band. A4's screen reused the original A2 collection; it added no game seeds.
The prespec retains the source-derived roster hashes and exclusion digest.

## Preparation and admission

`scripts/owning_encoder_study.py prepare` reads one compact JSON object with:

- `study_id`: a new immutable identifier.
- `runtime`: explicit `checkpoint`, `showdown_root`, `foulplay_root`,
  `foulplay_python` and `node_binary` paths.
- `max_decision_rounds`: prospective fixed cap (default 250); a cap is unknown.
- `owning_terminal_timeout_seconds`: fixed timeout, default 5, bounded to 30.
- `admission`: hash-bound `comparison` and `scientific_review` JSON references,
  each with `path` and `sha256`; plus `source_commit`, `selection_sha256`,
  `selection_metadata_sha256` and `collection_sha256`.

The compact independent B2 review must say `verdict: CLEAR` and bind
`comparison_sha256`, `reviewed_commit`, `checkpoint_sha256`,
`selection_sha256`, `selection_metadata_sha256` and `collection_sha256` to the
registered artifact and identities. This is a result-review binding, not a
claim that a source-only review establishes the numerical gate.

The runner verifies artifact hashes before precision setup, model loading or
game calls. It accepts only the actual-saved-owning-wire schema with original
consumed Python references, the exact champion contract, a complete fixed
roster of at least 10,000 views, all 132 numeric and 41 categorical columns,
zero refused/unprocessed cells, exact categorical/mask parity, numeric error
at most 1e-6, forwarded denominator equal to compared/expected, argmax agreement
at least 99.9%, and maximum value difference below 1e-4. Older projected results
cannot satisfy admission. Broader `full_encoder_admitted` and raw-chunk flags
remain honest metadata; false values are not additional numerical gate failures.

The explicit `run` subcommand accepts the prepared file and its byte SHA256,
and requires a fresh output directory. It configures CPU FP32, MHA/TF32 off,
highest Torch/JAX matmul precision, JAX x64 off, and matched one-thread CPU
settings. The factory retains its source/Node/checkpoint and drift refusals.
The output records observed precision/library versions. Node selected through
PATH must match the configured executable because the Dex loader uses PATH.

No command in this document is a launch request. The study wrapper does not
grant the separate governing offline cost/runtime admission or merge authority.

## Durable outcomes and failure semantics

Before precision setup, model loading or a benchmark call, the output receives the prespec and a
ledger with all 2,000 planned cells as `NOT_ATTEMPTED`. Successful chronological
progress is atomically published, with public terminal and owning actor
receipts, bound game hashes and compact outcome data. Prior completed prefixes
cannot be rewritten. Full observations, requests and model outputs are not
exported by this wrapper.

An encoder refusal, cap, transport/runtime fault, permission error, cancellation
or incomplete return stops the affected invocation and the rest of this study.
The next failed cell stays `FAILED` or `CAPPED`, with no score; remaining cells
stay `NOT_ATTEMPTED`. Wrapper admission, precision and factory failures leave
all game cells unattempted.
The failure records its stage/type and available owning-close receipt; permission
failures identify the denied path/errno. There is no resume, replay, replacement
or auto-retry. Durable I/O failures raise and require a nonzero external process
receipt; an older prefix is not a normal-completion attestation.

The ordinary benchmark supports both learner seats. Its p1-only capture and
raw-versus-search comparison helpers are not used.

## Frozen arithmetic and interpretation

Primary outcome is a win indicator; secondary outcome is score (win 1, genuine
owning tie 0.5, loss 0). Failed/capped/unattempted cells have bounds [0, 1], never
draw scores. Each arm keeps denominator 1,000; paired contrast bounds keep all 500
seed clusters and both seats. Identification bounds are not confidence intervals.

Only a normally completed full roster produces a two-sided nominal 90% paired
t interval (499 degrees of freedom), on each seed's mean of its two seat
differences, JAX minus Python. This is approximate mean-effect inference, not
finite-sample certification. A stopped/incomplete study has no confidence
interval, even if its observed prefix looks favorable. Zero cluster variance
leaves the interval unavailable rather than manufacturing significance.

`reduce` validates the hash-bound prespec/ledger and produces the same arithmetic.
Normal process completion and provenance still require external independent
review. There is no equivalence margin, automatic pass rule, or automatic B3
qualification from an interval crossing zero. The readout always leaves
`B3_qualified: false` and `equivalence_claim: false`.
