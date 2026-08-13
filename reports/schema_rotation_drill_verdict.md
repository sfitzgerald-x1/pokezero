# The schema-rotation drill's verdict, with its scope

Run on the union of #1244 and #1247 (the tree that exists once both land), full scope, three arms.
Command:

```sh
DRILL_REPO=<checkout> bash scripts/schema_rotation_drill.sh
```

## The verdict

**The class is NOT dead. It is much smaller than it was, and the residue is enumerated below.**

```
expected (class-iii, must break):   7
actual breakages:                  12
UNEXPECTED BREAKAGES:               5
```

The drill exits non-zero. That is the correct outcome: five sites break under a default rotation
that the rubric does not sanction, and each is named below with what it actually asserts.

### Denominators, and the arms they came from

| quantity | value | arm |
|---|---|---|
| tests collected | 6091 | rotated |
| raw failures, rotated run | 42 | rotated |
| stable baseline failures, subtracted | 12 | two baseline runs, intersected |
| unstable across the two baselines, NOT subtracted | 0 | a flake must not earn a permanent excuse |
| caused by ADDING a schema, subtracted | 8 | control arm (inject, do not rotate) |
| compiled-Rust dispatch, subtracted by cause | 10 | native-schema exclusion |
| source-mutation artifacts, subtracted by name | 1 | artifact list |
| **attributable to the rotation** | **12** | rotated − baseline − control − native |
| of those, sanctioned by the rubric | 7 | |
| **unexplained** | **5** | |

## The five, classified by what each asserts

**Two are class-(iii) sites MISSING FROM THE RUBRIC, not conflations.** They answer "what does a
fresh artifact get" — the legitimate question — and the rubric simply does not list them:

- `test_observation_spec_v2_1.py::ConfigDualSchemaTest::test_fresh_config_stamps_v2_2_and_its_width`
  asserts `config.observation_schema_version == OBSERVATION_SCHEMA_VERSION_V2_2` on a FRESH config,
  under a comment reading "The fresh-selection default flipped to v2.2 on 2026-07-08". Its subject is
  the default. It is also a **D2 defect**: it asserts `numeric_feature_count == 155` in the same
  test, so the dead-pin detector can never fire for it. Needs the same split as the five already
  done, then a rubric row.
- `test_observation_schema_flag.py::SchemaFlagEndToEndTest::test_collect_and_train_v2_2_end_to_end_and_cross_checks`
  asserts a collected cache's `metadata["observation_schema"]` equals v2.2, under a comment that says
  in as many words "(this assertion is deliberately about the default)". The test declares itself
  class (iii). Needs a rubric row.

**One is a genuine surviving instance of the class:**

- `test_golden_corpus.py::GoldenCorpusLiveSmokeTest::test_one_game_generates_a_verifiable_corpus`
  asserts `row.observation_schema_version == "pokezero.observation.v2.2"` — a HARDCODED string — on
  rows generated under whatever the default is. The corpus is collected at the default and then
  checked against a literal, so the two are coupled by nothing but coincidence. Either the collection
  should name its schema explicitly or the assertion should read what it collected under. This is the
  conflation, and it earns a ledger row.

**Two need individual adjudication, and I am not asserting a classification for them:**

- `test_foulplay_bridge.py::FoulPlayBridgeTest::test_capture_writes_p1_only_rollouts_and_preserves_partial_output`
  compares `observation.schema_version` against `spec.schema_version` — relative, not hardcoded — so
  it should be rotation-invariant, and is not. Something in the path binds a fixed spec while the
  capture uses the default. Worth reading before classifying.
- `test_observation_spec_v2_1.py::ConfigDualSchemaTest::test_observation_spec_from_model_config_resolves_schema_and_width`
  compares two whole `ObservationSpec`s that differ only in `schema_version`.

## What this claim does NOT cover

- **The shape half.** The synthetic schema is shape-identical to the OUTGOING DEFAULT by design, so a
  site hardcoding that schema's width sees no change in this arm. `DRILL_SHAPE=differ` is the arm for
  that half; its preconditions pass and `PRECONDITION 3` proves it discriminates (a width pin passes
  under `identical`, fails under `differ`), but a full scored `differ` run has not been done. Until it
  has, the shape half is **UNCOVERED**.
- **The compiled Rust encoder.** `rust/pokezero-search/src/encoder.rs` dispatches on schema version in
  compiled code. This drill edits Python in a worktree and never rebuilds the crate, so 10
  `EngineEnvTest` failures are excluded by cause. Anything the rotation would break *inside* the
  native encoder is invisible to this instrument.
- **The blind set.** 12 stable baseline failures plus 1 artifact are subtracted; a genuine breakage of
  any of them would be invisible. They are printed in full by every run rather than implied.
- **Runtime-assembled containers.** The container census finds literal containers of schema names or
  version constants (19 of them, across 6 files). A set built at runtime from a comprehension is not
  found by it and is covered only by `PRECONDITION 2`'s membership check.

## Why the earlier numbers were wrong

Six scored runs were discarded rather than reported. Every one failed for a defect in the
instrument, not the codebase, and the sequence is the reason this verdict is stated with scope
rather than as a headline:

| run | reported | actual defect |
|---|---|---|
| 1 | 19 unexpected | `_EXPORTABLE_TABLE_SCHEMAS` unregistered (10 of the 19); inventory pins charged to the rotation |
| 2 | — | the "identical" arm cloned **v4** while the outgoing default was v2.2, moving the shape 155/51/151 → 132/41/23. Its own precondition compared the clone against v4, so it passed by construction |
| 3 | 60 raw | `_norm_id` mangled 4 real subtest ids; its self-test printed a hardcoded `5/5` over 9 cases |
| 4 | 15 unexpected | the exporter's argparse `choices` unregistered |
| 5 | 15 unexpected | `OBSERVATION_SCHEMA_CLI_CHOICES`, the exporter's inline set, and the lattice's map all unregistered — then the last gate turned out to be compiled Rust |
| 6 | **5 unexpected** | this one |

The pattern, recorded because it is the transferable part: **nine** schema-keyed structures went
unregistered one at a time, each found by running the drill, reading a stack trace, and adding the
one it named. That loop does not converge — after each fix the instrument still had no way to say
what it had missed. Enumerating them took the count from 8 to 19.

## Next actions this verdict implies

1. Split `test_fresh_config_stamps_v2_2_and_its_width` (D2), add it and the
   `test_collect_and_train_...` pin to the rubric. Expected → 9.
2. Fix the `test_golden_corpus` hardcoded literal, or record it as a ledger row.
3. Read the two unadjudicated tests and classify them.
4. Run `DRILL_SHAPE=differ` to full completion, so the shape half stops being uncovered.
