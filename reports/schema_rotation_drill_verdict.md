# The schema-rotation drill's verdict, with its scope

> ## ⚠ THIS VERDICT'S NUMBERS ARE PRE-ROTATION. THE SCORED v4 RUN IS NOT DONE.
>
> The default rotated to v4 on 2026-08-14 (#1253, `e496e9b8`). Everything below was measured with
> **v2.2 outgoing**, on a tree that no longer exists. It is kept because its scope statement and its
> discarded-run history are still the honest record of how it was obtained — not because its figures
> describe today.
>
> ### What HAS been established on the v4 tree, by execution
>
> ```sh
> DRILL_REPO=$PWD bash scripts/schema_rotation_drill.sh
> ```
>
> | | |
> |---|---|
> | every precondition (−1, 0, 1, 2, 3) | **pass** with v4 outgoing |
> | outgoing profile, derived not hardcoded | `transition_region:F turn_merged:F grouped_layout:T feature_pack:T` |
> | synthetic ≡ outgoing, spec AND membership | **verified** — v4's absence from `TURN_MERGED` yields no skew, because the mirror (`:363`) and the skew check (`:784`) both derive from the outgoing default |
> | shape canary | pinned **132** from the pristine tree — v4's width, not a hardcoded 155 |
> | arm discriminates | yes, under `shape=identical` |
> | rotated arm, RAW | **35** — `35 failed, 6012 passed, 48 skipped, 2 xfailed, 18597 subtests in 1123s`; denominator 6097 collected; 29 FAILED + 6 SUBFAILED, 0 ERROR |
>
> ### THE CLAIM, STATED WITH ITS SCOPE — and it does not depend on the drill's arms
>
> The honest claim is narrower than "the class is dead", and narrower than the drill was ever asked to
> prove. It rests on two tools that run in seconds, not on a scored rotation:
>
> **No test reads the default's version string where it means a SPECIFIC version.**
>
> `python scripts/schema_default_assertion_scan.py --texts` over **254 test files** reports **PIN 5 /
> MIXED 4**, and the five PINs are exactly the five class-(iii) rows in
> `tests/data/schema_drill_expected_breakages.txt`:
>
> ```
> test_observation.py::test_the_fresh_artifact_default_is_v4                       1/1 asserts
> test_observation_spec_v2_1.py::test_v4_IS_the_fresh_default                      2/2 asserts
> test_observation_spec_v3.py::test_v4_IS_the_default_not_v3                       1/1 asserts
> test_observation_spec_v4.py::test_v4_IS_the_default                              1/1 asserts
> test_turn_merged_encode.py::test_v4_IS_the_default_and_the_default_is_NOT_turn_merged  1/1
> ```
>
> Those five ANSWER "which schema does a fresh artifact get" — the legitimate question — rather than
> consuming the answer while meaning a fixed version. That agreement between the scan and the rubric is
> the check that the identity half is complete rather than merely non-empty.
>
> **The residue is 4 MIXED tests, not zero.** Each reads the default to BUILD something and also
> asserts on it, which blinds the dead-pin detector for its own read (D2). They are named by the scan on
> every run. Four is the honest number; "the class is dead" was never true.
>
> **Exposure: 59 sites across 21 files**, from `python scripts/schema_default_ledger.py` (scanning 530
> tracked `.py` files): 38 `default-spec`, 16 `bare-const`, 3 `implicit:LinearPolicyModel`, 2
> `implicit:OnlineBattleAgent`. Frozen by `tests/test_schema_default_ledger.py` at
> `HIGH_WATER_MARK = 59`, which only ever lowers — raised exactly once in this programme, deliberately,
> with the justification recorded at the constant.
>
> That is down from **391** at the start and from the **213** the plan quoted. Every reduction is a site
> that named its schema instead of reading the global, and each is enumerated in the allowlist by
> `file::owner::kind::unclosed` — deliberately without a line number, so a reformat cannot launder a new
> site in as an old one.
>
> **What the rotation itself proved, independently of all of the above:** the one-line change to v4
> breaks ZERO tests by set equality against main — baseline and rotated both 7 failed / 6034 passed /
> 18601 subtests, difference empty in both directions, re-derived by an independent reviewer on a fresh
> clone with a crate fingerprint-verified against the same tree.
>
> ### THE SCORED RUN WAS ATTEMPTED AND THE DRILL ABORTED. That abort is the verdict.
>
> A full run on this tree completed the rotated arm and BOTH baseline runs, then stopped itself:
>
> ```
> == rotated ==   raw total: 35   denominator 6095
> == baseline ==  stable across two runs (subtracted): 8   denominator 6089
> ABORT: rotated and baseline denominators differ by 6 (6095 vs 6089).
>        One run measured a different suite; the subtraction is meaningless.
> DRILL_EXIT=11
> ```
>
> **The guard is correct and the instrument is now self-blocking.** The drill works by INJECTING a
> synthetic sixth schema and rotating the default to it. This tree contains tests that ENUMERATE the
> schema table — `test_schema_property_membership.py` and `test_schema_with_selector.py`, both added by
> #1244 — so a sixth schema creates six additional collected cases. The rotated arm therefore always
> measures a slightly larger suite than the baseline, and the denominator guard (added because a
> scope-mismatched subtraction silently shrinks the residue) refuses to subtract across them.
>
> So the honest verdict for the v4 tree is: **there is no scored verdict, because the drill cannot
> produce one here.** Not "the run is too slow", not "it was not attempted" — attempted, and it
> correctly refused. Two of its own improvements now conflict: the injection design (#1247) and the
> tests that enumerate the schema table (#1244).
>
> Fixing it means one of: excluding schema-enumerating tests from the drill's target set; comparing
> failure SETS without requiring equal denominators (which is what `_norm_id` already makes possible,
> and which the guard was written to prevent for a different reason); or having the control arm supply
> the denominator delta it already measures. That is instrument work with its own review surface, and
> it should not be decided inside a docs PR.
>
> ### What has NOT been established, and must not be inferred
>
> **MY FIRST FIGURE HERE WAS 42 AND DID NOT REPRODUCE.** A reviewer re-ran the rotated arm on this
> exact commit and measured **35**, passing the drill's own `raw total == summary failures`
> cross-check. 42 is withdrawn. Likely mechanism, hypothesis not finding: 42 − 35 = 7, and
> `test_fallback_replay_end_to_end.py` contributes exactly 7 — the group this script's header
> names as the historical `FAILED -> ERROR` double-bucket, and whose `@requires_showdown` gate
> turns on whether the local checkout can actually play a game. I did not establish which, and
> a figure I cannot reproduce is not a figure.
>
> **35 is a raw count and attributable to nothing either.** The baseline (two runs, intersected), the control
> (two runs, intersected) and the native-Rust exclusion have not been run on this tree, so there is no
> attributable set, no sanctioned/unexplained split, and no dead-pin reading. Quoting 35 as a verdict
> would be precisely the error this programme spent eleven PRs retiring.
>
> To finish: FIVE suites — rotated x1 (:875), baseline x2 (:951), control x2 (:1140) — about 94
> minutes at the measured 18:43 per suite, and
> needs a session that outlives that. After one complete run, `DRILL_BASELINE_REUSE=1` makes
> subsequent runs shorter -- but only by 2 of 5 suites, leaving rotated + control x2 at roughly 56
> minutes; there is NO control cache. `DRILL_STOP_AFTER_PRECONDITIONS=1` (:866) is the supported
> ~1-minute instrument check. The stamp is `(SHA, scope, SHAPE, interpreter)`, so it refuses a stale
> reuse rather than subtract the wrong set.
>
> ### Two decays found in two days, and the reason
>
> - **#1251** killed the differ arm's last shape site, so the shape half now has **zero** coverage
>   (confirmed by execution: the rotation moved the default's width 155 → 132, twenty-three columns
>   where the arm narrows by one, and the row stayed green).
> - **#1253** shifted `neural_cli.py`'s unnamed container from line 1951 to 1952, and the census
>   aborted the first v4 run rather than guess.
>
> Neither was caught by review or CI — four review rounds and eight green checks passed over the
> second — **because the drill is not a CI job.** An instrument nobody runs decays silently. Wiring it
> into CI is worth more than any single defect on the follow-up list.

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

> **SCOPE IN TIME, not just in coverage.** Every number below was measured with **v2.2 as the outgoing
> default**, on a tree that predates #1251, #1252 and the v4 rotation. Three things have changed since,
> and one of them is known to have changed what the instrument can see:
>
> - **#1251** named `TransformerPolicyConfig.observation_schema_version`, which killed the differ arm's
>   only row — see the superseded box below. That is not a hypothetical: it is confirmed by execution.
> - **#1252** named `LocalShowdownConfig.observation_spec`.
> - **the rotation** makes v4 the outgoing default, so the synthetic clone, the mirrored property
>   memberships, and PRECONDITION 3's width canary all re-derive against v4 rather than v2.2.
>
> The drill's design survives that last one without changes — the mirror only copies memberships the
> outgoing default actually has, and PRECONDITION 2 compares memberships symmetrically, so v4's absence
> from `TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS` produces no skew. **SUPERSEDED: now verified by
> EXECUTION — see the v4 banner at the top of this file, which printed `premise verified` on a
> real run. The reading below was correct and is kept as the argument.** Originally verified by reading
> `schema_rotation_drill.sh:363` (`if _DRILL_OUTGOING in _dt:` — the mirror copies only memberships the
> outgoing default has) and `:784` (`skew = [n for n, t in tuples.items() if (ref in t) != (drill in t)]`
> — a symmetric comparison, so absent-in-both is not skew), not by running it.
>
> **What has NOT been done: the SCORED run with v4 outgoing.** (The drill HAS been run with v4
> outgoing through all five preconditions and a complete rotated arm — see the top banner. What is
> missing is the baseline, control and native arms, hence the subtractions.) Until that happens, the "12
> attributable / 7 expected / 5 unexpected" headline is a correct measurement of a tree that no longer
> exists. Treat it as the last known reading, not as the current state.

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
