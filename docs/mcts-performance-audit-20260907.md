# Local MCTS performance audit — 2026-09-07

The paired microbenchmark below was measured at source
`dacb6358d9b145ce069d6718662a38f581a38bc0`; this narrow implementation was
then carried forward to `df4e3ce15ee69f922f6ae1b81c7b5e9861828319` before
opening a review. The work is CPU-only on the laptop. No Kubernetes objects,
shared artifacts, running studies, or published branches were changed. Local
encoder timing is **not** a playing-strength result or a current estimate of
the encoder's share of full MCTS wall time.

## What the current code already does

- `engine_search.py` concentrates identical belief completions into one tree with
  multiplicity-scaled simulations, retaining their belief weight. Skipping all
  but one independent search without reinvesting its simulations would change
  the estimator/compute budget; that is not a semantics-preserving cache.
- `tree.rs` reuses a deferred batch row when a selected depth-capped branch is
  still pending. Repeated selections are not automatically duplicate forwards.
  The collision ledger in `model.rs` counts repeats by decision-node/arm and
  seat. Changing selection to remove collisions is a search-mechanics experiment,
  not a safe inference-cache optimization.
- `model.rs` already skips encoding/model forwards for rollout-priced leaves
  that cannot supply child policy priors. Production model-valued leaves still
  need their value. The skip does not bypass branch rendering or safety checks.
- Leaf tensors depend on the branch's advanced public fold, evolved metadata,
  seat, action ordering, and engine state. A cache keyed only by serialized engine
  state would not prove equivalent observations.
- Tables are retained between encodes. `leaf.rs::leaf_row_inputs` still deep-clones
  the root JSON before rewriting it; the full leaf timing seam separately records
  row construction, fold products, and tensor writing. Those are good places for
  a future **current production** profile, but their August percentages must not
  be presented as measurements of today's build.

## Reproducible local measurement

Added `rust/pokezero-search/examples/bench_boundary_encoder.rs`. It loads the
five committed v4 golden rows, extracts only sanctioned observation metadata and
public-materialization inputs, warms the encoder, and emits timing plus a hash
of all five output arrays (including exact floating-point bits).

The measured path is `encode_row`: JSON parsing plus boundary tensor encoding.
It does **not** include engine leaf-row construction, fold advancement, tree
selection/backup, tensor-to-Torch conversion, or neural inference. It cannot
estimate whole-search speedup or validate a strength claim.

Environment: macOS 26.5.2 arm64; rustc 1.94.1; release profile, LTO enabled.
The vendored engine was fetched and hash-verified with the existing repository
script; no shared Python environment was modified.

Inputs:

- `tests/data/golden_corpus_sample/rows.jsonl` SHA-256
  `c1f2a681fcea4cf16f74f2aab73324cecce0caf6e2c1a0ecb4ec5d58a47f98c7`.
- Existing read-only v4 `encoder_tables.json` SHA-256
  `08f2e4464036602b26572873b2ae6cd31629b3abb8c2d93250fd126e02a113fa`.
  The original local path is intentionally omitted: it is deployment-local and
  not part of a portable source claim. Reproduce only with a table artifact
  independently validated against the checkpoint under study.
- Baseline `encoder.rs` SHA-256
  `ca92578ff56e2e57a965b030938f02dea0df28ec4d6636a5ddac5e59f31a7227`.
- Retained candidate `encoder.rs` SHA-256
  `c90a29a57357333933cdeed16c950da0ca0f9a4476bdec8ef3442ced9ce6fd80`.

From the source worktree, build with:

```sh
PYO3_PYTHON=/path/to/python-with-project-dependencies \
  cargo build --release --manifest-path rust/pokezero-search/Cargo.toml \
  --example bench_boundary_encoder
```

Invoke the executable with `TABLES ROWS 12000`: five rows times 12,000 repetitions
= 60,000 encodes per timing. Build the base and candidate separately and alternate
their order across paired runs. Preserve the baseline executable before rebuilding.
Do not benchmark concurrently with builds or treat an early noisy median as a win.

Every measured baseline/candidate output so far has the same BLAKE2b-256 hash:
`037c8cd488ba1c01bbd82c188f3153620dabc95fe70edb18ed983ea45adf7d94`.
That is baseline-vs-candidate agreement over this fixture, not a claim that this
boundary-only fixture reproduces the full history-derived golden surface.

## Rejected candidate: borrow already-normalized category strings

The first candidate returned a borrowed `Cow<str>` for trimmed lowercase ASCII
categories instead of allocating an identical lowercase String. Non-ASCII and
uppercase input kept the old normalization path. Three focused tests passed:
16,384 ASCII pairs, all valid Unicode scalar values, contextual/expanding Unicode
examples, and the borrowed-storage contract. The candidate's encoder suite passed
21 tests, zero failures/skips, and all five output arrays stayed bit-identical.

An initial noisy eight-pair run suggested 4.15% lower boundary time. A longer
six-pair run after builds settled did **not** reproduce it:

| Pair | Baseline µs/encode | Candidate µs/encode |
|---|---:|---:|
| 0 | 87.8177 | 88.3820 |
| 1 | 88.6767 | 88.0954 |
| 2 | 87.8340 | 88.8915 |
| 3 | 87.8582 | 86.8009 |
| 4 | 87.9886 | 87.8632 |
| 5 | 109.7261 | 87.0166 |
| Median | **87.9234** | **87.9793** |

Median difference: candidate **0.064% slower**; paired median reduction 0.40%.
Five pairs differ by at most 1.21%; the final baseline has a clear timing outlier.
Conclusion: no resolved wall-time benefit. **This production change and its
candidate-only tests were removed**, rather than promoting the noisy first read.

## Profile-directed follow-up

A three-second baseline CPU sample (2,500 stack samples) made allocation/free,
JSON deserialization, and identifier normalization visible. Top-of-stack counts
included 454 `_xzm_free`, 267 `_xzm_xzone_malloc_tiny`, 76 `to_lowercase`, and
56 `normalize_identifier`. Inlining/symbol merging and the boundary-only path
prevent interpreting these as production phase percentages.

Unlike category normalization, identifier normalization currently allocates a
lowercase intermediate and then grows a second filtered String. A bounded second
candidate combines lowercase/filtering into one pre-sized pass for ASCII input,
keeping the original Unicode path unchanged. This changes no vocabulary, feature,
state, cache key, search budget, or batching policy.

### Retained candidate: one-pass ASCII identifier normalization

After compilation ended, six fresh alternating pairs each measured 60,000 encodes:

| Pair | Baseline µs/encode | Candidate µs/encode | Reduction |
|---|---:|---:|---:|
| 0 | 87.63679445 | 81.07245903 | 7.49% |
| 1 | 88.40208750 | 82.11774513 | 7.11% |
| 2 | 88.10221667 | 81.84677362 | 7.10% |
| 3 | 91.23101458 | 81.72594513 | 10.42% |
| 4 | 87.34206805 | 81.07458682 | 7.18% |
| 5 | 88.94234583 | 81.14201945 | 8.77% |
| Median | **88.25215208** | **81.43398229** | **7.73%** |

Paired median reduction is 7.33%; all six pairs improve, unlike the rejected
category fast path. This supports retaining the small identifier optimization.
It remains a single-laptop, five-row, boundary-encoding microbenchmark with timing
noise, **not** a production-MCTS speedup estimate, throughput guarantee, or evidence
of better decisions. No production encoder-share percentage was refreshed.

Verification:

- Release example builds successfully; all paired runs exit 0.
- All five arrays match the baseline bit-for-bit through the hash above.
- `cargo test --release --lib encoder:: -- --nocapture`: **20 passed, 0 failed,
  0 ignored**. This includes the existing column-mapping/encoder tests plus two
  new identifier-equivalence tests: all 16,384 ASCII pairs, every valid Unicode
  scalar, and multi-character/context-sensitive examples. The tests compare
  against the previous production expression; the non-ASCII production path
  remains literally unchanged. Kelvin sign and dotted capital I have explicit
  expected outputs so an ASCII-only fallback cannot pass silently.
- Broader `cargo test --release --lib`: **248 passed, 0 failed, 1 ignored**.
  This is the non-model crate suite, not model-feature or live-runtime validation.
  The pre-existing ignored test is
  `events::tests::a_near_full_hp_seeder_still_over_books_the_drain_slot`; it was
  not enabled or modified by this optimization.
- The primary agent independently ran the final full default-feature suite
  twice: **248 library + 334 integration tests passed**, one pre-existing
  ignored test, exit 0. An independent read-only correctness review found no
  blocking issue, verified input hashes and the five-array hash coverage, and
  recomputed the paired timing summaries. It did not rerun the benchmark.
- `git diff --check` and formatting check for the new example pass. Initial
  formatter attempts exited 1 (first an incorrect relative path, then formatting
  differences); the path and formatting were corrected before the final check.
- No model-feature/full-MCTS benchmark, live leaf parity campaign, or playing-
  strength panel was run by this workstream. Those remain outside this local
  timing result. Any later deployment needs fresh source/image provenance; do
  not change source under an existing registered study.

## Recommendation

Keep the modest, semantics-preserving identifier improvement and the compact
benchmark. Prioritize the independently reproduced search-correctness defect
over more speculative encoder micro-optimizations. Before a larger typed-row or
cache redesign, measure the **full current native leaf path** on a representative
fixed workload. Do not reuse the old 77–80% encoding attribution, assume repeated
batch selections are duplicate model work, or spend another gate cycle chasing
the rejected category-normalization effect.
