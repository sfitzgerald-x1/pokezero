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

## Burndown protocol

A slice resumes by **re-running the command**, never by recall. A row is retired when the
site NAMES what it needs — a version (`SCHEMA_V22`) or a property
(`schema_with(transition_region=True)`) — not when it merely tolerates the new default.
Phase D's acceptance drill is the stop condition: rotate the constant to a fake v5 in a
scratch branch and the breakage count must equal exactly the class-(iii) pinned tests.
