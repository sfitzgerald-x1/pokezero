# The schema-rotation drill's verdict, with its scope

> **AUTHORITATIVE RUN: v8**, the first whose scorer has no known path to drop a genuine breakage.
> Two such paths (G1: the native exclusion matched any mention of the Rust message, including an
> assertion message; G2: the section splitter could not parse pytest-subtests' spaced headers, so an
> innocent test was excluded on its neighbour's exception) were found by independent review AFTER the
> earlier run and closed before this one.
>
> **v8 reproduces the earlier verdict exactly** -- 12 attributable, 7 expected, 5 unexpected, the same
> five ids, and no dead pins. So those two defects were LATENT in this tree: nothing here tripped
> them. The earlier numbers were right, and there was no way to know that until the paths were shut.
> That is the difference between a figure that happens to be true and one that has been measured.

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
| tests collected | 6092 rotated / 6088 baseline | both |
| raw failures, rotated run | 42 | rotated |
| *(the 6091 previously stated here was carried over from an earlier run and never re-derived against v8 -- corrected)* | | |
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

## The shape half: was one site wide, and is now ZERO

> **SUPERSEDED AS OF #1251 — read this before the section below.** The single site that gave the
> `differ` arm its coverage, `test_from_dict_width_defaults_are_schema_keyed`, no longer reads the
> default's width. #1251 named `TransformerPolicyConfig.observation_schema_version`
> (`neural_policy.py:271`), so the config it restores is v2.2-stamped whatever the global says, and
> the arm — which narrows the *outgoing* default's width by one — cannot move it.
>
> Confirmed by execution on the v4-rotated tree, and the confirmation is stronger than a differ run:
> the rotation moved the default's numeric width 155 → **132**, twenty-three columns where `differ`
> narrows by one, and the row stayed green.
>
> ```sh
> python -c "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC as D; \
>            from pokezero.observation import OBSERVATION_SCHEMA_VERSION as V; \
>            print(V, D.numeric_feature_count)"
> #   pokezero.observation.v4 132     <- the process default;  the row asserts 155
> python -m pytest -q -p no:randomly \
>   "tests/test_observation_spec_v2_1.py::ConfigDualSchemaTest::test_from_dict_width_defaults_are_schema_keyed"
> #   1 passed
> ```
>
> So `DRILL_SHAPE=differ` will now fire its dead-pin detector and exit non-zero, and **the shape half
> of this class has no coverage at all.** This is not a regression in #1251 — naming the version is
> the fix this programme exists to make. It is the instrument losing its last shape example *because*
> the codebase improved. The honest reading of "one site wide" was always that almost nothing in this
> suite pins the default's width; it is now zero sites wide, the same fact with the last example
> removed. A replacement site must be found or written, and deliberately has not been admitted by
> reasoning alone.
>
> Everything below this box describes the measurement as it stood BEFORE #1251 and is kept as the
> record of that run, not as a current claim.

## The shape half, as measured before #1251: COVERED, and one site wide

`DRILL_SHAPE=differ` has been run to completion on the same tree, full scope, three arms. It was
previously "constructed but unverified" and is no longer.

```
raw failures, rotated (differ)    43
subtracted, stable baseline       12   (0 unstable)
subtracted, ADDING a schema        8
subtracted, compiled Rust         10
attributable                      13
expected (shape rubric)            8   ->  0 MISSING: no dead pins
UNEXPECTED                         5
```

The two arms' attributable sets differ by **exactly one id**:

| arm | attributable | sanctioned | unexplained |
|---|---|---|---|
| identical | 12 | 7 | 5 |
| differ | 13 | 8 | 5 (the same 5) |

The one extra is
`test_observation_spec_v2_1.py::ConfigDualSchemaTest::test_from_dict_width_defaults_are_schema_keyed`,
which restores a config with `numeric_feature_count` REMOVED and asserts it takes its schema's width
default -- 155 for the fresh default, under the comment "The fresh default config is v2.2-stamped;
its schema-keyed width default is 155." Its subject is what WIDTH a fresh artifact gets: it ANSWERS
"nobody said" for the width, which is the shape analogue of class (iii). It breaks under `differ`
(154) and not under `identical` (155), so it is the one breakage the identical arm structurally
cannot produce.

**The honest reading of "one site wide" is not that the arm is powerful — it is that almost nothing
in this suite pins the default's width.** That is worth knowing either way, and it is the first time
it has been measured rather than assumed.

That the SAME five sites are unexplained in both arms is itself evidence: they break on a naming
change alone, so they are naming failures, not shape conflations, and they belong to the residue
above rather than to a second population.

The differ rubric was committed EMPTY until this run, and the run that produced it aborted at exit 7
("the expected-breakages file has no entries") for exactly that reason -- the emptiness working as
designed, refusing a scored verdict without a stated pass condition. It is populated per its own
admission rule: one row at a time, each with its demonstration recorded, and the five unexplained
sites deliberately NOT admitted, because an entry justified by "it broke" rather than by what it
asserts is the laundering the emptiness was protecting against.

## What this claim does NOT cover

- **The shape half has NO coverage as of #1251** -- see the superseded box above. It was one site wide;
  that site stopped reading the default's width when `TransformerPolicyConfig` named its schema, which
  is a fact about the suite rather than about the arm. Confirmed by execution on the v4-rotated tree.
- **The compiled Rust encoder.** `rust/pokezero-search/src/encoder.rs` dispatches on schema version in
  compiled code. This drill edits Python in a worktree and never rebuilds the crate, so 10
  `EngineEnvTest` failures are excluded by cause. Anything the rotation would break *inside* the
  native encoder is invisible to this instrument.
- **The blind sets -- ALL of them.** Four things are subtracted, not one, and an earlier version of
  this list named only the first, which is the omission a reviewer called out as a hole in this
  section's own honesty contract:
    - 12 stable baseline failures + 1 source-mutation artifact (printed in full every run);
    - 8 control-arm failures, i.e. caused by ADDING a schema. The control arm now runs TWICE and
      subtracts only the intersection, because a single run let a flake earn a permanent excuse --
      the same defect the baseline had fixed, reintroduced in the arm added later;
    - 10 native-Rust failures. Excluded only when the message is attached to an EXCEPTION TYPE on a
      pytest `E ` line, and resolved to FULL test ids -- a substring match over the section body,
      and a `Class.test` comparison that discarded the file, were both shown to drop a genuine
      naming breakage.
  A genuine breakage of anything in any of these four is invisible to the verdict.
- **Runtime-assembled containers.** The container census finds literal containers of schema names or
  version constants (19 of them, across 10 files). A set built at runtime from a comprehension is not
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
4. Close the two subtraction-chain defects a reviewer demonstrated (done: the control arm now
   runs twice and intersects; the native exclusion requires an exception-type attachment and
   matches full ids, aborting on an ambiguous `Class.test`).
