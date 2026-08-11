# Plan: make v4 the default observation schema

Written 2026-08-11. Supersedes the problem statement in
`docs/observation_default_schema_problem_20260810.md`, which established *that* the constant is
stale and *why* nothing is mis-stamped. This is the execution plan.

## The change

```python
# src/pokezero/observation.py:77
-OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION_V2_2
+OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION_V4
```

One line. Everything below is the fallout.

## Measured baseline

Full suite with the rotation applied, on `main` at `b0d21647` (post-#1227, pre-#1228), in an
isolated worktree, 3 files excluded for the pre-existing Python 3.11 collection defect:

**94 failed / 5707 passed / 48 skipped.** 93 `FAILED` lines across 25 files.

| file | n | | file | n |
|---|---|---|---|---|
| `test_neural_policy` | 24 | | `test_remote_self_describe` | 4 |
| `test_region_trim` | 11 | | `test_observation_spec_v2_1` | 4 |
| `test_interaction_registry` | 10 | | `test_batch_replay` | 4 |
| `test_feature_mask_consistency` | 10 | | `test_neural_selfplay` | 2 |
| `test_fallback_replay_end_to_end` | 7 | | `test_distributed_training` | 2 |

plus 15 files at 1 each.

An earlier report of "6", then "~4 mechanical updates", then "97" were all denominators that had
not been established — the first two from a hand-picked file list, the third from a summary line
whose detail had been lost to a truncated output file. **94 is the first figure derived from a
complete, retained run.** The list lives at `/Users/scott/workspace/agents/rotation-enum/`.

## Root cause of the bulk: v4 has no transition region

The dominant failure is not stamping and not widths. It is:

```
ValueError: observation schema 'pokezero.observation.v4' carries no transition region,
so transition_token_budget must be 0; got 32.
```

v2/v2.1/v2.2/v3 all carry a transition-token region; **v4 does not**. So every test that
exercises history-window or region-trim machinery *through the process default* loses its
subject the moment the default becomes v4. `test_region_trim` and `test_feature_mask_consistency`
are almost entirely this.

This is the same coupling class as the `token_count` defect, one level up: code and tests reach
for the global default where they mean "a schema with the property I am testing".

## Measured again, with #1228 applied: no change

Same worktree, same exclusions, `fix/schema-aware-width-defaults` applied on top of the rotation:

**94 failed / 5710 passed** — the SAME count, and an identical per-file distribution.

This falsifies the expectation recorded in the first draft of this plan, that #1228 would shrink
the set materially because `test_neural_policy` (24) is its target. It shrank it by **zero**.
#1228 is a correct standalone fix for a real defect; it is not a rotation prerequisite in the
sense of reducing the blast radius, and the sequencing below no longer claims that it is.

## Failure taxonomy, derived from the retained run

Distinct error signatures across the 94:

| n | signature | bucket |
|---|---|---|
| 19 | `categorical_ids shape does not match TransformerPolicyConfig` | A |
| 14 | `schema v4 carries no transition region, so transition_token_budget must be 0` | A |
| 9 | `token_count 151 does not equal the fixed prefix (23) + transition_token_count (0)` | A |
| 7 | bare `AssertionError: X != X` (per-schema expected values) | A or C — inspect |
| 6 | `row_categorical_ids shape does not match TransformerPolicyConfig` | A |
| 5 | `IndexError: list index out of range` | **unclassified — inspect first** |
| 3 | `AssertionError: 0 != 128` (transition region width) | A |
| 2 | `schema v4 carries no transition region, so transition_token_count must be 0` | A |
| 2 | `AssertionError: 0 not greater than 0` | **unclassified — inspect first** |

**47 of 94 (50%) are the no-transition-region refusal in some form.** The single largest group
after that is fixtures that hardcode v2.2 shapes and then rely on the default schema to agree.

So roughly 80% of the work is one mechanical edit repeated: **a fixture names the schema it
means instead of taking the default**. The `IndexError` and `0 not greater than 0` groups
(7 total, in `test_fallback_replay_end_to_end` and neighbours) do not obviously reduce to
coupling and must be read individually before being assigned a bucket.

## Prerequisites

1. **#1227 — `token_count` resolves from the stamped schema.** MERGED (`b0d21647`).
2. **#1228 — the two feature widths resolve the same way.** OPEN, in review. `token_count` dying
   loudly had been masking these; after #1227 a v4 config built successfully while silently
   carrying v2.2's `51/155` against v4's `41/132`.

Both are correct standalone fixes and neither depends on the rotation.

## Classification of the remaining work

Three buckets. Each failure must land in exactly one, with the bucket named in the commit.

- **A — test reaches for the default where it means a specific schema.** Fix: pin the schema
  explicitly. The assertion's subject is unchanged. Confirmed instances:
  `test_observation_spec_v2::test_token_section_offsets` (a v2-family layout check reading
  `DEFAULT_REPLAY_OBSERVATION_SPEC`), three bare `self._config()` helpers in
  `test_observation_spec_v2_1`, and — by inspection — most of `test_region_trim` and
  `test_feature_mask_consistency`, which need a transition-carrying schema.
- **B — invariant that the rotation retires.** The three `..._but_not_the_default` tests. Fix:
  rotate the assertion and rewrite the rationale to say why the old rule is retired rather than
  violated. The old rule guarded arms mid-campaign against a silent rotation; it does not apply
  when v4 is already what every arm runs.
- **C — genuine behavior change.** Re-derive, never retype. Confirmed: the golden-corpus live
  smoke grid `(151,51) -> (23,41)`, and the linear-policy feature fingerprint, which hashes the
  schema and must rotate with it.

Unclassified and needing a look before assignment: `test_interaction_registry` (10 — encoding
assertions with per-schema column indices, e.g. `844 != 1163` on a forme-change retype) and
`test_fallback_replay_end_to_end` (7 — `0 not greater than 0`, `IndexError`, which do not
obviously reduce to schema coupling and may be real).

## Sequence

1. Land #1228. *(in review)*
2. ~~Re-enumerate with #1228 applied; the remaining set should shrink materially.~~ **DONE, and
   the prediction was wrong: 94 -> 94, zero change.** Recorded above rather than deleted, because
   the reasoning that produced it ("the biggest bucket is its target") is the same shortcut that
   produced the 6 and the ~4.
3. Classify every remaining failure into A/B/C. Publish the list before editing.
4. Land A and B as reviewed PRs, grouped by subsystem, rotation NOT yet applied — a bucket-A fix
   is correct on its own and should be green both before and after the rotation. This is the
   property that keeps the rotation itself a one-line diff.
5. Land C separately, each value re-derived from the tree.
6. Rotate the constant. By then the suite is green with it applied.

## Constraints

- Nothing lands without independent review.
- Every guard added or changed is kill-confirmed against a mutant. A green test is not a pin;
  #1227's review found a mutant that survived all 264 tests, and #1228 pins it.
- No value is copied from a CI log or a message. Re-derive from the tree.
- Three test files cannot be collected under the venv's Python 3.11
  (`scripts/c153_wide_negative_census.py:445`, backslash in an f-string; CI is on 3.12). They are
  excluded from every count above. This gap is how the c155 fingerprint breakage in #1221 reached
  CI unseen, and it should be fixed on its own PR.

## Non-goals

- No encoder or schema-definition changes. v4 is already defined, supported and running; this
  moves which schema is chosen when nobody names one.
- No opportunistic refactors. Bucket-A fixes correct the *coupling*, not the assertion.
