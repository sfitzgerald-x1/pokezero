# Ledger: every site reaching the global observation-schema default

Derived 2026-08-11 at `b0d21647`. **This file is the denominator.** No figure about the
schema-default conflation is quoted from memory or from a hand-picked file list; every
number below is reproduced by one command.

## Derivation

```sh
python3.12 scripts/schema_default_ledger.py            # rows
python3.12 scripts/schema_default_ledger.py --by-file  # per-file
python3.12 scripts/schema_default_ledger.py --json     # machine-readable
```

Run it under **3.12+, not the venv**: the script only parses, and
`scripts/c153_wide_negative_census.py` uses a backslash in an f-string expression, which
is a SyntaxError on 3.11. Under the venv that file lands in `UNPARSED`, the script exits
2, and the denominator is incomplete by one. It exits 2 rather than reporting a clean
number, which is the whole design: an unmeasured file must not silently shrink N.

## N = 202

**202 sites across 46 files, from 516 tracked `.py` files scanned.**

| kind | n | meaning |
|---|---|---|
| `implicit-cfg` | 98 | `TransformerPolicyConfig(...)`/`.compact_category(...)` with no `observation_schema_version=` |
| `default-spec` | 54 | reads `DEFAULT_REPLAY_OBSERVATION_SPEC`, which is defined AS the default's spec |
| `implicit-spec` | 34 | `ObservationSpec(...)` with no `schema_version=` |
| `bare-const` | 16 | reads `OBSERVATION_SCHEMA_VERSION` itself |

Reads of the per-version names (`..._V2_2`, `..._V4`) and of `SUPPORTED_...` /
`REPLAY_OBSERVATION_SPECS_BY_SCHEMA` are **not counted**: those NAME a schema, which is the
state this migration moves sites INTO. Counting them would make the burndown never
converge. `src/pokezero/observation.py` is excluded as the definition site.

## Class (iii) — legitimate default readers

**5 sites.** These are the only ones permitted to keep reading the global default;
Phase B makes the accessor fail-closed everywhere else, and a test pins this count so a new
incidental reader fails at authorship rather than at the next rotation.

| site | why |
|---|---|
| `src/pokezero/linear_policy.py:80` | fingerprint stamping (payload hashes the schema) |
| `src/pokezero/neural_cli.py:3025` | fresh generation, no schema named (_train `or` fallback) |
| `src/pokezero/neural_cli.py:5593` | fresh generation, no schema named (_iterate `or` fallback) |
| `src/pokezero/neural_policy.py:257` | config's own schema field -- the entry point for 'nobody said' |
| `src/pokezero/showdown.py:1143` | definition: DEFAULT_REPLAY_OBSERVATION_SPEC IS the default's spec |

## Production sites: 19 of 202

| site | kind | owner | class |
|---|---|---|---|
| `src/pokezero/engine_env.py:1140` | `default-spec` | `_default_observation_spec` | UNCLASSIFIED |
| `src/pokezero/linear_policy.py:80` | `bare-const` | `_linear_feature_fingerprint_payload` | **iii — legitimate** |
| `src/pokezero/linear_policy.py:110` | `bare-const` | `LinearPolicyModel` | UNCLASSIFIED |
| `src/pokezero/local_showdown.py:226` | `default-spec` | `env_config_from_checkpoint_provenance` | UNCLASSIFIED |
| `src/pokezero/local_showdown.py:271` | `default-spec` | `LocalShowdownConfig` | UNCLASSIFIED |
| `src/pokezero/neural_cli.py:2684` | `implicit-cfg` | `_describe` | UNCLASSIFIED |
| `src/pokezero/neural_cli.py:3025` | `bare-const` | `_train` | **iii — legitimate** |
| `src/pokezero/neural_cli.py:3092` | `implicit-cfg` | `_train` | UNCLASSIFIED |
| `src/pokezero/neural_cli.py:5593` | `bare-const` | `_iterate` | **iii — legitimate** |
| `src/pokezero/neural_cli.py:5697` | `implicit-cfg` | `_iterate` | UNCLASSIFIED |
| `src/pokezero/neural_policy.py:245` | `default-spec` | `TransformerPolicyConfig` | UNCLASSIFIED |
| `src/pokezero/neural_policy.py:246` | `default-spec` | `TransformerPolicyConfig` | UNCLASSIFIED |
| `src/pokezero/neural_policy.py:257` | `bare-const` | `TransformerPolicyConfig` | **iii — legitimate** |
| `src/pokezero/neural_policy.py:483` | `default-spec` | `TransformerPolicyConfig` | UNCLASSIFIED |
| `src/pokezero/online_client.py:99` | `default-spec` | `OnlineBattleAgent` | UNCLASSIFIED |
| `src/pokezero/showdown.py:1143` | `default-spec` | `<module>` | **iii — legitimate** |
| `src/pokezero/showdown.py:1143` | `bare-const` | `<module>` | **iii — legitimate** |
| `src/pokezero/showdown.py:4280` | `default-spec` | `observation_from_player_state` | UNCLASSIFIED |
| `src/pokezero/teacher_scenarios.py:553` | `implicit-spec` | `_observation` | UNCLASSIFIED |

The 14 unclassified production sites are the priority: two of this class
(`token_count` in #1227, the feature widths in #1228) were already found to be **real latent
defects**, not stylistic. Each remaining one is a candidate for the same.

## Test/script sites: 183 of 202

| file | n |
|---|---|
| `tests/test_neural_policy.py` | 99 |
| `tests/test_feature_mask_consistency.py` | 11 |
| `tests/test_neural_category_vocab.py` | 8 |
| `tests/test_observation.py` | 7 |
| `tests/test_neural_selfplay.py` | 6 |
| `tests/test_observation_spec_v2.py` | 6 |
| `tests/test_dataset.py` | 3 |
| `tests/test_distributed_training.py` | 3 |
| `tests/test_linear_policy.py` | 3 |
| `tests/test_observation_spec_v2_1.py` | 3 |
| `tests/test_replay_branching.py` | 3 |
| `scripts/shaping_ranker.py` | 2 |
| `tests/test_batch_replay.py` | 2 |
| `tests/test_crate_model_leafeval.py` | 2 |
| `tests/test_selfplay.py` | 2 |
| `docs/token-format/extract_turn16.py` | 1 |
| `scripts/bench_crate_search.py` | 1 |
| `scripts/encode_foulplay_capture.py` | 1 |
| `scripts/play_online_baseline.py` | 1 |
| `tests/test_bench_model_eval.py` | 1 |
| `tests/test_bootstrap.py` | 1 |
| `tests/test_cache_concat.py` | 1 |
| `tests/test_collection.py` | 1 |
| `tests/test_env.py` | 1 |
| `tests/test_export_model.py` | 1 |
| `tests/test_foulplay_bridge.py` | 1 |
| `tests/test_observation_spec_v3.py` | 1 |
| `tests/test_observation_spec_v4.py` | 1 |
| `tests/test_policy.py` | 1 |
| `tests/test_region_trim.py` | 1 |
| `tests/test_remote_self_describe.py` | 1 |
| `tests/test_replay_import.py` | 1 |
| `tests/test_rollout.py` | 1 |
| `tests/test_shaping.py` | 1 |
| `tests/test_shaping_provenance.py` | 1 |
| `tests/test_showdown.py` | 1 |
| `tests/test_trajectory.py` | 1 |
| `tests/test_turn_merged_encode.py` | 1 |


## Class (iii), refined after reading the production sites

The initial guess of 5 was too narrow, and the shape is now clear: a legitimate default reader
is **an entry point where "nobody said" has to be answered exactly once.** There are four such
kinds, and every other site is a conflation:

| kind | why it is legitimate |
|---|---|
| the definition (`showdown.py` `DEFAULT_REPLAY_OBSERVATION_SPEC`) | it *is* the default's spec |
| CLI `or OBSERVATION_SCHEMA_VERSION` fallbacks (`_train`, `_iterate`) | fresh generation, no schema named |
| fingerprint stamping (`linear_policy` payload) | the fingerprint must hash whatever is current |
| a config type's own schema/spec field default | one per config type: the single place "nobody said" enters that type |

That last row is what the first pass got wrong. `TransformerPolicyConfig.observation_schema_version`,
`LinearPolicyModel.observation_schema_version` and `LocalShowdownConfig.observation_spec` are the
same construct repeated per config type -- each is the one place a caller who named nothing gets
an answer. Migrating them would not remove the default read, it would just move it to every
caller.

**The test is: does this site ANSWER "nobody said", or does it CONSUME the answer?** Answering
is legitimate and belongs on the allowlist. Consuming -- a fixture, an assertion, a derived
width -- is the conflation, because the consumer had a real requirement it declined to name.

Left deliberately unclassified pending a proper read, rather than guessed into class (iii):

- `engine_env.py:577` -- `config.observation_spec or _default_observation_spec()`; reads like an
  answering site, but "reads like" is how this class survives.
- `local_showdown.py:226` -- compares a resolved spec against the default to detect
  customisation. Genuinely about the default, or should it compare against the checkpoint's
  schema? That is a behavioural question, not a naming one.


## RETRACTED: "Confirmed NOT this class: test_fallback_replay_end_to_end (7)"

Pinning the observation spec at all three `LocalShowdownConfig` sites to v2.2 changed nothing:
still 7 failures, same messages. Reverted rather than kept -- a fix that fixes nothing is worse
than no fix, because the next reader assumes that avenue was closed.

The seven collapse to ONE cause: the seed band produces zero refusals under v4, so `self.specs`
is empty and six tests die on `self.specs[0]` while the seventh reports "this seed band produced
no refusals, so the chain was never exercised". The refusals are construction-side, which is
consistent with them not moving when the encode spec is named.

**This conclusion was WRONG and is retracted.** Independent verification ran these 7 on `main`
and they fail there identically -- `IndexError: list index out of range` at
`test_fallback_replay_end_to_end.py:260`, both trees. They are PRE-EXISTING failures with no
connection to v4 at all.

The reasoning error: I observed that pinning the spec did not fix them and concluded "therefore
a genuine v4 behavioural difference". The alternative -- "therefore not caused by v4" -- was
never tested, and testing it costs one command (`git stash; pytest; git stash pop`), which I had
already used earlier in the same session for exactly this purpose. Ruling a hypothesis out is
not the same as ruling the alternative in.


## Phase D worklist: the 33 unexpected breakages

Scored by `bash scripts/schema_rotation_drill.sh` against
`tests/data/schema_drill_expected_breakages.txt`. **This is the exact remaining migration.**
Each line is a site that silently depended on which schema held the default slot; the drill
passes when this list is empty.

The same run reported **0 expected-but-did-not-break**, so all five class-(iii) pins fired.
That direction is checked because a pin that stops pinning would let the drill "pass" while
the next rotation goes unnoticed -- the one way this migration could be falsely declared done.


**`test_distributed_training.py`** (2)

- `DistributedTrainingTest::test_ddp_two_rank_contiguous_shards_match_single_device_updates`
- `DistributedTrainingTest::test_ddp_two_rank_ppo_reports_global_metrics_within_parity_bounds`

**`test_engine_env.py`** (1)

- `EngineEnvTest::test_k0_leaves_the_transition_region_present_but_masked`

**`test_engine_stat_attestation.py`** (1)

- `TransportAttestationScriptTests::test_json_result_carries_reproducible_command_and_provenance`

**`test_fallback_replay_end_to_end.py`** (7)

- `TestReplayChainAgainstRealBattles::test_every_address_resolves`
- `TestReplayChainAgainstRealBattles::test_recording_does_not_perturb_the_run`
- `TestReplayChainAgainstRealBattles::test_replay_is_deterministic`
- `TestReplayChainAgainstRealBattles::test_the_recorded_address_replays`
- `TestReplayChainAgainstRealBattles::test_the_run_actually_refused_something`
- `TestReplayChainAgainstRealBattles::test_the_runner_searches_under_the_recorded_config`
- `TestReplayChainAgainstRealBattles::test_the_search_itself_is_deterministic`

**`test_foulplay_bridge.py`** (1)

- `FoulPlayBridgeTest::test_capture_writes_p1_only_rollouts_and_preserves_partial_output`

**`test_golden_corpus.py`** (1)

- `GoldenCorpusLiveSmokeTest::test_one_game_generates_a_verifiable_corpus`

**`test_investment_live_env.py`** (1)

- `LiveInvestmentPopulationTest::test_live_codes_are_knowledge_monotone_vs_batch_and_encode`

**`test_linear_policy.py`** (1)

- `LinearPolicyTest::test_linear_feature_fingerprint_payload_tracks_extractor_source_and_schemas`

**`test_neural_policy.py`** (5)

- `NeuralPolicyScaffoldTest::test_neural_cli_benchmark_history_mask_k_wires_and_stamps`
- `NeuralPolicyScaffoldTest::test_transformer_forward_accepts_compact_categorical_training_cache_rows`
- `NeuralPolicyScaffoldTest::test_transformer_forward_accepts_row_indexed_training_cache_windows`
- `NeuralPolicyScaffoldTest::test_zero_layer_row_indexed_forward_matches_dense_expansion`
- `TruncateHistoryTensorsTest::test_policy_forward_matches_manual_truncation`

**`test_neural_selfplay.py`** (2)

- `NeuralSelfPlayTest::test_torch_smoke_runs_train_save_load_benchmark_chain`
- `NeuralSelfPlayTest::test_torch_smoke_trains_from_real_cache_chunks_and_deletes_them`

**`test_observation_schema_flag.py`** (1)

- `SchemaFlagEndToEndTest::test_collect_and_train_v2_2_end_to_end_and_cross_checks`

**`test_observation_spec_v2_1.py`** (1)

- `SpecTableTest::test_spec_for_schema_is_loud_on_unknown_versions`

**`test_online_client.py`** (1)

- `TurnMergedNormalizeThreadingTest::test_default_schema_agent_requests_turn_merged`

**`test_remote_self_describe.py`** (4)

- `RemoteSelfDescribeTests::test_config_exposes_served_model_config`
- `RemoteSelfDescribeTests::test_explicit_schema_spec_refines_to_trimmed_region`
- `RemoteSelfDescribeTests::test_reload_refuses_token_shape_mismatch`
- `RemoteSelfDescribeTests::test_remote_spec_adopts_like_neural_spec`

**`test_roll_enumeration_scope.py`** (1)

- `RollEnumerationRuntimeScope::test_importing_every_differential_consumer_leaves_the_fan_collapsed`

**`test_tier2_live_env.py`** (1)

- `CollectCacheMaskMetadataTest::test_checkpointless_collect_records_masks_and_train_cross_checks`

**`test_token_format_doc.py`** (1)

- `TokenFormatDocSelfValidationTest::test_committed_dump_matches_live_regeneration_byte_for_byte`

**`test_transitions_fold.py`** (1)

- `FoldAnnotatedSurfaceTest::test_annotated_products_match`

## Phase D scoreboard (baseline-subtracted, coherently-stamped drill)

`DRILL_SCOPE=fast bash scripts/schema_rotation_drill.sh`

    baseline failures (NOT attributable to the rotation): 20
    expected (class-iii, must break):  6
    UNEXPECTED BREAKAGES:              6
    EXPECTED-BUT-DID-NOT-BREAK:        2

The instrument was wrong twice before this, both times in a direction that flattered or
inflated the result, and both times caught by a test rather than by me:

1. **No baseline.** Pre-existing failures were charged to the rotation -- `test_roll_enumeration_
   scope` fails on the 3.11 f-string defect in c153 and has nothing to do with schemas. The
   uncorrected count was 33; corrected, it is 6. Every earlier figure in this document that
   derives from the uncorrected drill should be read as an upper bound only.
2. **The synthetic spec was stamped v4.** Mapping `v5-drill` to `V4_REPLAY_OBSERVATION_SPEC`
   left a spec stamped v4 reachable under the v5-drill key.
   `test_spec_for_schema_is_loud_on_unknown_versions` reported exactly that
   (`'...v4' != '...v5-drill'`) and I nearly filed it as a surviving instance of the class. It
   was the instrument manufacturing the failure it reports. Fixed by stamping the synthetic
   spec with its own version -- after which the unexpected set CHANGED, which is the tell that
   the earlier 2 were noise.

### The 6 unexpected -- all live-env encode paths

- `test_engine_stat_attestation::test_real_replay_materializes_and_attests_transport_with_source_hash`
- `test_golden_corpus::test_wrapped_and_bare_games_are_identical`
- `test_investment_live_env::test_default_masks_build_no_tracker`
- `test_tier2_live_env::test_ten_game_sweep_only_monotone_divergences`
- `test_transitions_fold::test_prefix_closure_over_random_games`
- `test_transitions_fold::test_prefix_closure_over_scenario_games`

One shape: a live env encodes under the default spec, and the assertion is about the encode.
`test_transitions_fold` already resisted a mechanical `LocalShowdownConfig` pin once (it left
two product sources inconsistent), so this group needs the env and its comparison target moved
together, not a blanket edit.

### The 2 pins that stopped pinning

Both are currently failing at BASELINE, so the subtraction removes them and they can no longer
serve as pins. They must be repaired before the drill's expected set means anything:

- `test_linear_policy::..._fingerprint_payload_tracks_extractor_source_and_schemas`
- `test_turn_merged_encode::test_v2_2_is_a_supported_schema_entry_and_the_default`

This is the more dangerous of the two failure modes. An unexpected breakage is visible; a pin
that has quietly stopped pinning lets the NEXT rotation pass unnoticed, which is the exact
condition that let v2.2 sit stale through two schema generations.


## Phase D: current scoreboard and the exact residue

    baseline failures (NOT attributable):  18
    expected (class-iii, must break):       6
    UNEXPECTED BREAKAGES:                   6
    EXPECTED-BUT-DID-NOT-BREAK:             0   <- both dead pins revived

**The 6 all share one signature:** `'pokezero.observation.v5-drill' != 'pokezero.observation.v4'`.
They run a live env, capture the schema stamped into its output artifact, and compare it against
a hard-coded version.

That makes each of them a CLASSIFICATION question, not a mechanical fix, and it is the last
genuinely undecided thing in this migration:

- If the assertion's subject is *"the artifact records whatever schema produced it"*, it should
  compare against the resolved schema, not a literal -- bucket A, and the fix removes the literal.
- If its subject is *"this env produces v4 artifacts"*, it is class (iii) and belongs in the
  expected set with a justification.

They must be read individually. Filing all six into the expected set would make the drill pass
tomorrow and mean nothing -- that file is the one place this migration can be falsely declared
finished, which is why its own header states the rule. Filing all six as bucket A and deleting
the literals could equally destroy a real pin on artifact provenance.

Two attempts have already failed by treating this group mechanically: pinning
`test_transitions_fold`'s three `LocalShowdownConfig` sites left its annotated-products
comparison inconsistent (one source moved, the other did not), and the same blanket approach on
`test_neural_selfplay` turned a passing test red. Both reverted. The group needs the env and its
comparison target moved together.


## The last 6: four mechanical attempts, four failures. Read them.

Attempts made and reverted, recorded so the next reader does not repeat them:

| attempt | outcome |
|---|---|
| pin `test_transitions_fold`'s three `LocalShowdownConfig` sites | annotated-products comparison went inconsistent: one source moved, the other did not |
| blanket-pin every `compact_category` in `test_neural_selfplay` | turned a PASSING test red (`..._disable_fixed_opponents_for_mirror_self_play`) |
| pin `test_fallback_replay_end_to_end`'s three env sites to v2.2 | changed nothing: 7 failures, identical messages -- NOT this class |
| pin the `test_investment_live_env` / `test_tier2_live_env` `_env` helpers to v4 | broke a DIFFERENT test in the same file (`IndexError: tuple index out of range`) |

Four different mechanical shapes, four failures. That is not bad luck; it is the group telling
us what it is. These tests pair a LIVE ENCODE with a comparison target -- a committed artifact,
a batch-computed twin, a fold closure -- and the schema has to move on BOTH sides together or
the test is measuring the mismatch it just created. A one-sided pin is a new bug wearing the fix's
clothes, and each attempt above produced exactly that.

**The work each needs:** find the comparison target, determine which schema PRODUCED it, and
give the env that schema -- or, where the target is generated in-test, move both. That is a
read-and-decide task per test, roughly the same effort as the 8 already migrated by hand, and it
cannot be batched.

**What must not happen:** filing these six into
`tests/data/schema_drill_expected_breakages.txt`. The drill would go green tomorrow and mean
nothing. Their signature (`'v5-drill' != 'v4'`) is indistinguishable at a glance from a genuine
class-(iii) pin, which is precisely why that file's header states the admission rule and why it
is the single place this migration can be falsely declared finished.


## Phase D: PASS at fast scope; full scope pending

    DRILL_SCOPE=fast bash scripts/schema_rotation_drill.sh
    expected (class-iii, must break): 6
    actual breakages:                 6
    PASS: the breakage set is EXACTLY the class-(iii) readers.

**What this proves and what it does not.** Fast scope covers every file that has ever broken
under a rotation plus the expected set. It therefore proves no KNOWN site still reaches the
default. It cannot see a new breakage in a file that has never broken -- only the full drill
can, and the full run is the stop condition. Stating the difference because a fast PASS is
exactly the kind of result that gets rounded up to "done".

### The last six were the instrument, not the codebase

All six remaining "surviving instances" were the drill failing to register its synthetic schema
in `_MINIMUM_CATEGORICAL_CENSUS_BY_SCHEMA` and `_MINIMUM_NUMERIC_CENSUS_BY_SCHEMA`. Every
consumer of those maps raised `KeyError('...v5-drill')` and the drill charged it to the tree.

That is the THIRD instrument defect in this phase:

| # | defect | effect on the number |
|---|---|---|
| 1 | no baseline | pre-existing failures charged to the rotation: 33 vs the true 6 |
| 2 | synthetic spec stamped `v4` under the `v5-drill` key | manufactured a spec-table incoherence and reported it as a class instance |
| 3 | schema not registered in the census maps | manufactured 6 KeyErrors and reported them as class instances |

Each was caught by a test, none by inspection, and each CHANGED the answer. Four mechanical
"fixes" were attempted against defect 3's six and reverted -- attempts to repair code that was
not broken.

**The tell, ignored at the time:** the claim "all six share one shape" came from a FILE-WIDE
sorted error list, never a per-test attribution. The true per-test error was `KeyError`, not the
schema mismatch asserted. Same error class as "18 checked, 0 mismatches" against a denominator
of 28 -- a claim whose subject was never enumerated. An instrument built to enforce that rule
violated it three times; the rule is not self-executing just because it is written down.

### Guards now in the drill, each from a defect it would have caught

- baseline subtraction, keyed on **(SHA, scope)** -- a `fast` baseline reused under `full`
  subtracts the wrong set, and a wrong subtraction is invisible
- synthetic spec stamped with its own version
- registration in EVERY schema-keyed table, **hard-failing** on one it does not know about
- both directions checked: unexpected breakages AND pins that have stopped pinning


## Post-verification cleanup

An independent verification refuted the headline result. Everything it found is fixed; the
findings are recorded here because several were in the INSTRUMENTS, and an instrument that lies
is worse than no instrument.

### The refutation, and what was actually wrong

| finding | status |
|---|---|
| ledger denominator was 3 hardcoded call names, missing 186 sites | FIXED -- derived from signature defaults; true N was **283**, not 97 |
| gate bypassable: 97 rows collapsed to 74 keys | FIXED -- key includes line |
| gate never executed (skips on 3.11, in no workflow) | FIXED -- wired into the 3.12 job, where a SKIP is a failure |
| drill baseline taken at an already-rotated HEAD | FIXED -- `DRILL_BASELINE_REF`, plus a loud warning |
| drill scored only `^FAILED`, never `^ERROR` or the exit code | FIXED -- aborts on either |
| drill force-removed an unnamed sibling directory | FIXED -- path derived from the worktree |
| two `== V4` identity gates the table-scan could not see | FIXED -- and a guard for them found **four more** |
| 21 fixtures left internally incoherent | FIXED -- at the two dataclass defaults, not the 21 sites |
| 10 real regressions vs main | FIXED -- all ten |
| "fallback_replay is a v4 behavioural difference" | **RETRACTED** -- fails identically on main |

### "202 -> 97" was measured against a population I chose

The ledger counted three call names. `LocalShowdownConfig.observation_spec` alone has 132
callers, none counted. This is the program's own error class -- a denominator chosen rather than
enumerated -- committed inside the instrument built to retire it, and it survived because the
figure reproduced perfectly every time it was re-derived. Reproducibility is not validity.

Real trajectory, all by `python3.12 scripts/schema_default_ledger.py`:

    main                             283
    after the migration              283 -> 213
    (of which 70 came from two dataclass default lines, not 70 edits)

### The lesson the guards taught

Every guard added after a defect immediately found more instances than inspection had:

- the identity-gate guard found 4 gates beyond the 2 found by reading
- the `only_shrinks` check named all 67 stale allowlist rows
- the ledger's exit-2 refused a clean denominator under 3.11
- `git add` refused the gitignored allowlist that would have made the gate unrunnable

None depended on remembering. That is the only mechanism in this effort with a clean record --
including against its author.


## N and the drill measure different things, and both are needed

These two facts are not in tension, and reading either alone misleads:

    ledger N        213 sites still reach the global default
    drill           0 unexpected breakages; gate 1 breaks zero tests vs main

**N is EXPOSURE: how many sites could conflate.** The drill is REALISED FAILURE: how many
actually do, today, under a rotation. A site that takes the default and genuinely does not care
which schema it gets is counted by N and invisible to the drill -- correctly, in both cases. It
is exposure because the day it starts caring, nothing warns anyone; it is not a failure because
today it does not.

`LocalShowdownConfig.observation_spec` is 130 of the remaining 213 -- the largest single
surface, and almost entirely tests that need *an* env and have no stake in its schema. Migrating
all 130 would be busywork; leaving the surface unguarded is how the class regenerates. The
authorship gate is what resolves that: the 130 are grandfathered in the allowlist, and the
131st fails in the PR that writes it.

So the honest end state is:

- **realised failure: zero** -- the rotation is free today, proven by set equality against main
  and by the acceptance drill
- **exposure: 213 sites**, frozen by the gate so it can only shrink
- **the two production surfaces that could be fixed at the source were** -- `ObservationSpec`
  and `PokeZeroObservationV0` stamps, which removed 70 sites without touching a call site

Claiming "the class is dead" on the drill alone would be the same error as the original
"202 -> 97": a true number answering a question narrower than the one being asked.


## FINAL: Phase D passes at full scope

    bash scripts/schema_rotation_drill.sh            # ~22 min, two full suites

    baseline (unrotated):   8 failed, 5796 passed    <- ~5804 tests
    rotated (v5-drill):    15 failed, 5789 passed    <- ~5804 tests
    attributable:          15 - 8 = 7

    expected 7 / actual 7  -- md5 58e99231122be90faebc0397c8b20d6c on both, diff empty
    PASS: the breakage set is EXACTLY the class-(iii) readers.

All seven assert the default's identity: four name which schema holds the slot, one is the
schema-keyed census table, one is the fingerprint that hashes the schema, one is the
token-format dump's `current_default` line. Nothing else in 5804 tests notices the default
moving.

**Gate 1, independently and not through the drill:** rotating the real constant to v4 leaves a
failure set BYTE-IDENTICAL to main's 8 pre-existing failures. Two full suites and a `diff`.

### Seven defects were in the instruments, and two failed toward PASS

| defect | direction | invented / hid |
|---|---|---|
| no baseline | toward FAIL | 27 phantom breakages (33 vs the true 6) |
| synthetic spec stamped v4 under the v5 key | toward FAIL | a spec-table incoherence |
| schema unregistered in the census maps | toward FAIL | 6 phantom breakages |
| ERROR abort not baseline-relative | toward FAIL | blocked every run |
| normalisation assumed absolute pytest paths | **toward PASS** | every comparison |
| sed delimiter collided with the alternation | **toward PASS** | scored 15 real failures as 0 |
| empty-normalisation guard ran before its file | toward FAIL | aborted every run |

Four mechanical "fixes" were attempted against defect 3's phantoms and reverted -- repairing
code that was never broken.

**Both PASS-direction defects were caught by one design choice: checking BOTH directions.**
"Nothing broke AND the pins that must break did not break" is impossible, and that contradiction
is the only signal that surfaced either. A drill checking unexpected breakages alone -- the
obvious design, and my first intent -- would have reported two clean passes from a scorer
emitting empty files.

The rule this program was built to enforce is that no figure ships without a verified
denominator. The instrument built to enforce it violated it seven times. Written rules do not
execute; guards that hard-fail do. Every guard here exists because of a defect it would have
caught, and each one found more than inspection had.

## Burndown protocol

A slice resumes by **re-running the command**, never by recall. A row is retired when the
site NAMES what it needs — a version (`SCHEMA_V22`) or a property
(`schema_with(transition_region=True)`) — not when it merely tolerates the new default.
Phase D's acceptance drill is the stop condition: rotate the constant to a fake v5 in a
scratch branch and the breakage count must equal exactly the class-(iii) pinned tests.
