# Deep-Line Encoder Audit Report

**Status:** Complete. The deep-line phase and deterministic coverage-enumeration phase have both been executed, independently reviewed, and recorded below.
**Scope:** Read-only audit of the production Python observation encoder. This
branch contains audit tooling and regression coverage only; it does not modify
encoder, belief, transition, or engine-search behavior.

## Objective

Find state-accumulation and multi-mechanic observation errors that ordinary
single-event tests may miss. A confirmed issue is recorded here with a minimal
reproduction and an actionable, checkpoint-compatible follow-up fix. Findings
are not patched in this audit branch.

## Current Coverage

| Lane | Coverage | Current result |
| --- | --- | --- |
| Live decision differential | Four local random Gen 3 randbats plus a persistent run of 100 random games and seven protocol cuts: 17,963 decision boundaries total | 14,027 persistent-run boundaries were turn 20+. The shard reproduced the known self-fact, stale-cure, Transform, and candidate-pruning roots at scale, and found one new move-weather countdown defect below. |
| Incremental vs batch | Included at every audited boundary | No mismatch in the smoke run. |
| Snapshot vs live | Included at every audited boundary | No mismatch in the smoke run. |
| Perspective symmetry | Included whenever both seats request an action | No mismatch in the corrected random smoke. |
| Request/action oracle | Raw request independently rebuilds all 9 legal action slots and 4 move-PP fractions | Eight-turn live smoke passed; this does not call the production legality helper. |
| Field oracle | Bridge weather, Spikes layers, screens, and timed side-condition durations | Sand Stream plus Reflect/Light Screen scripted chains passed. Permanent ability weather is represented by bridge `duration: 0`, which the audit now handles explicitly. |
| Source-distribution manifest | Configured Gen 3 randbat universe versus components disclosed in sampled self requests; every public opponent candidate variant is source-checked | One-game smoke checked 155 candidate variants with no membership mismatch. It observed 12/220 species, 12/1,748 exact variants, 29/125 moves, 11/71 abilities, and 5/13 items: useful provenance and coverage accounting, explicitly not an exhaustive universe sweep. |
| Scripted mechanic chains | 18 existing `gen3customgame` scenarios, 405 decisions | 17 findings: 15 confirmed encoder divergences across Transform and Chesto-Rest; 2 perspective views of the same underlying defects. |
| Protocol co-occurrence census | Captured for every completed audited game plus seven public protocol cuts | Committed fold sample has 0 Intimidate, 0 Sand Stream, and 0 Baton Pass occurrences across five retained fold rows. New cuts cover all three ordered chains. |
| Deterministic source breadth | 220 source-derived `gen3customgame` 1v1 fixtures, encoded as `gen3randombattle` observations | All 220 species, all 235 reachable species-ability pairs, all 125 movepool moves, and all 13 source items exercised through the production encoder with no uncovered atom and no oracle finding. |
| Exact source variants | 874 source-derived `gen3customgame` 1v1 fixtures, encoded from both seats | All 1,748 complete `(species, role, level, moves, ability, item)` source tuples exercised as self-known and true opponent belief candidates, with no uncovered variant and no oracle finding. |
| Universal move mini-lane | Deterministic mechanics fixtures for `struggle`, `recharge`, and generic `hiddenpower` | All three surfaced their expected action-token identity; Struggle also executed as a protocol move and Recharge as a protocol `cant` event. |

## Commands And Evidence

```sh
uv run python scripts/deep_line_audit.py \
  --random-games 1 --max-rounds 8 \
  --json /tmp/pokezero-deep-line-smoke.json
```

Result: 16 decisions checked, 0 findings after the perspective-token alignment
correction. A later four-game depth shard reached 762 boundaries (606 at turn
20 or later), reproduced the stale-cure and false-pinch-pruning classes in
natural random battles, and exposed no snapshot or batch-fold divergence.

The audit artifact now includes `randbat_source_coverage`, built from the exact
local Gen 3 set universe. It records source metadata, catalog totals, sampled
self-request component totals, and the number of public belief candidates
confirmed as source members. This is deliberately a coverage manifest rather
than a claim that random play exercised every species, move, ability, item, or
variant. The one-game source smoke covered 155 candidate variants with zero
membership mismatches, while its self-request sample covered 12/220 species,
12/1,748 exact variants, 29/125 moves, 11/71 abilities, and 5/13 items.

The persistent long run completed 100 random games plus seven protocol cuts and
17,963 decision boundaries, including 14,027 at turn 20 or later. Its immutable image predated the later
source-coverage manifest and default finding suppression, so the long artifact
does not make a source-coverage claim and contains repeated copies of the
already-triaged signatures. Triage grouped those copies by root cause rather
than treating their raw count as separate defects. The next source-derived
enumeration audit is the required exhaustive breadth complement.

```sh
# Build a clean Showdown checkout first, then point the audit at that checkout.
node build
POKEZERO_SHOWDOWN_ROOT="$POKEZERO_SHOWDOWN_ROOT" \
  uv run python scripts/coverage_enumeration_audit.py \
    --json /tmp/pokezero-coverage-enumeration-audit.json \
    --coverage-json /tmp/pokezero-coverage-enumeration-ledger.json
```

Result: 220/220 planned fixture games completed against source hash
`754b71cfed643fa0`, with zero oracle findings. The execution ledger has empty
uncovered sets for 220 species, 235 reachable species-ability pairs, 125 source
movepool moves, and 13 source items. The bounded universal lane also executed
real Struggle and Recharge transitions and verified generic Hidden Power action
identity. The source plan used no gap-fill fixtures on this source revision:
the two deterministic variant passes covered all required atoms directly.
The committed machine-readable evidence is
`docs/audit_artifacts/coverage-enumeration-audit-754b71cfed643fa0.json` and
`docs/audit_artifacts/coverage-enumeration-ledger-754b71cfed643fa0.json`.
The live run also checked 3,495 public candidate-variant records for configured
source-universe membership.

```sh
POKEZERO_SHOWDOWN_ROOT="$POKEZERO_SHOWDOWN_ROOT" \
  uv run python scripts/coverage_enumeration_audit.py \
    --exact-variants \
    --json /tmp/pokezero-exact-variant-audit.json \
    --coverage-json /tmp/pokezero-exact-variant-ledger.json
```

Result: 874/874 exact-variant fixtures completed against the same source hash
with zero oracle findings. The v2 ledger records first coverage for all 1,748
source variants and has empty uncovered sets for variants as well as the 220
species, 235 reachable species-ability pairs, 125 moves, and 13 items. Every
fixture is audited from both player perspectives, so each exact tuple is tested
both as self-known state and as the opponent's true surviving candidate. The
committed evidence is
`docs/audit_artifacts/coverage-exact-variant-audit-754b71cfed643fa0.json` and
`docs/audit_artifacts/coverage-exact-variant-ledger-754b71cfed643fa0.json`.

```sh
uv run python scripts/deep_line_audit.py \
  --random-games 0 --scenario sand_shedinja --scenario screens_jirachi \
  --suppress-kind self_known_ability --suppress-kind self_known_item \
  --suppress-kind self_transform_identity --suppress-kind candidate_count_increased \
  --json /tmp/pokezero-deep-line-field-results.json
```

Result: 50 scripted decision boundaries, 0 new findings. The raw-request oracle
matched every policy legality bit, action-token legality bit, and request move
PP fraction. The raw bridge field oracle matched permanent Sand Stream weather
and the Reflect/Light Screen chain.

```sh
uv run python scripts/deep_line_audit.py \
  --random-games 0 --scenarios \
  --json /tmp/pokezero-deep-line-scenarios.json
```

Current result: 405 decisions checked, including 20 at turn 20 or later. The
identity-aware raw matcher now handles duplicate species, cosmetic Unown forms,
Transform, and force-switch request boundaries. The 17 surviving findings reduce
to the confirmed Transform and Chesto-Rest encoder divergences below.

```sh
uv run python scripts/deep_line_audit.py \
  --random-games 0 --protocol-fixtures \
  --json /tmp/pokezero-deep-line-protocol-fixtures.json
```

Result: the reusable public-only protocol cuts reproduce three parser/belief
findings: two `-cureteam` stale-status surfaces and one Forecast
`-formechange` identity surface. The fixture catalogue also preserves Color
Change and the Leech Seed pending-snapshot boundary for the corresponding fix
PRs. It now also covers Intimidate switch-in ordering, ability Sand Stream,
and Baton Pass replacement, all absent from the committed golden fold sample.

The committed `fold.jsonl.gz` contains five retained decision rows. A direct
event-slice census found 0 occurrences of `Intimidate`, `Sand Stream`, and
`Baton Pass`; its only action-side protocol events are ordinary `switch`,
`move`, `-damage`, `-heal`, and `-ability` records. It is therefore a useful
schema/parity smoke, not meaningful coverage for those stateful switch chains.

## Triage Log

| Status | Finding | Evidence | Action |
| --- | --- | --- | --- |
| Resolved audit false positive | Perspective symmetry compared fixed team-token offsets after a switch. Team order is player-relative, so the same active Pokemon can occupy different token positions. | Initial random smoke reported 9 mismatches; matching active tokens reduced the same run to 0. | Keep active-token matching in the audit harness. No production change. |
| Resolved audit false positive | `candidate_set_count >= 1` was applied to scripted `gen3customgame` fixtures. Those intentionally do not draw from the Gen 3 randbat set universe. | 510 of 574 initial scenario alerts were this invalid assertion. | Run candidate-count invariants only for `gen3randombattle`. No production change. |
| Resolved audit false positive | Raw bridge matching initially selected the wrong Pokemon for duplicate species, cosmetic Unown forms, transformed Pokemon, and force-switch request boundaries. | Identity-aware matching reduced scripted findings from 574 to 17, all attributable to confirmed defects below. | Preserve source-metadata matching and request-aware active handling in the audit harness. No production change. |
| Resolved audit false positive | `status:tox => toxic_stage >= 1` is not valid at the decision immediately after a poisoned Pokemon switches in. | Random seed 3 reached turn 88 with a freshly switched-in, poisoned Delcatty and a correct zero Toxic stage; Gen 3 increments after the next residual. | Retain numeric bounds, but do not infer a Toxic stage solely from current status. No production change. |
| Resolved audit-manifest defect | Self requests spell dynamic-power moves as `return102`/`frustrationNN`, while the set universe uses `return`/`frustration`. | Independent review reproduced a source variant that was not counted as observed when only its request-side Return spelling differed. | Normalize dynamic-power request IDs before source-component and exact-variant coverage accounting. No production change. |
| Confirmed at scale | The long shard emitted 421,594 raw findings because its image predated default suppression. | 420,691 are already-triaged direct signatures; the remaining bridge/status/species/boost/perspective rows collapse to the existing stale-cure, Forecast, and Transform roots, except for the four Rain Dance timing rows below. | Preserve root-cause deduplication; use the new runner defaults for any future depth shard. |
| Under investigation | Scripted custom-game moves and items can be outside the closed random-battle category vocabulary. | The scenario run emitted OOV-vocabulary warnings for custom-only fixtures such as `attract` and `safeguard`. | Keep this separate from the randbat encoder audit; confirm whether it affects only custom-game test fixtures before reporting a production issue. |
| Resolved coverage-audit false positive | Trace emits a public copied current ability at switch-in while retaining the native source ability in its candidate universe. | Four initial breadth alerts were Gardevoir/Porygon2 copying Keen Eye or Water Absorb. `candidate_variants` and `possible_abilities` correctly retained native Trace; only the visible current-ability bucket changed. | Audit native ability on the source candidate surface and the copied ability on the public categorical surface. No production change. |

## Confirmed Encoder Bugs

### 1. Self Transform Does Not Encode the Copied Battle Identity

| Property | Evidence |
| --- | --- |
| Trigger | `ditto_transform` scenario: `|move|p1a: Ditto|Transform|p2a: Snorlax` followed by `|-transform|p1a: Ditto|p2a: Snorlax`. |
| Divergent surface | The transforming player's self token encodes `species:ditto`; the opposing player's public opponent token encodes the same active mon as `species:snorlax`. |
| Independent evidence | Both player-relative belief views record `transformed: true` and `transform_species: Snorlax`; only the self-token encode path ignores that belief. |
| Training impact | Yes. The acting policy sees the wrong active identity, types, and base-stat surface during Transform. |
| Incidence | Every self-side Transform decision after the public `-transform` event until the user leaves the field. |
| Classification | Confirmed encoder bug. |
| Required fix | In `observation_from_player_state`, pass `self_exact_beliefs` as `beliefs_by_species` to the self `_encode_pokemon_tokens` call, not only as `exact_beliefs_by_species`, then add a self-Transform regression test. |

### 2. Snapshot Drops a Pending Leech Seed Source

| Property | Evidence |
| --- | --- |
| Trigger | Snapshot the replay after `|move|p1a: X|Leech Seed|p2a: Y` and before `|-start|p2a: Y|move: Leech Seed`; restore, then feed the `-start` line. |
| Divergent surface | Live parser resolves `leech_seed_source_sides` to `{'p2': 'p1'}`. The restored parser loses the pending source and records `leechseed-source-unknown` as a materialization blocker. |
| Root cause | `_ReplayParser` owns `_pending_leech_seed_source_sides`, but `ShowdownReplayState` does not serialize it and `_ReplayParser.from_snapshot()` cannot restore it. |
| Training impact | Indirect but real: snapshot/direct-materialization branches at this protocol cut fail closed even though the live state is reconstructible. Request-boundary incidence is expected to be low. |
| Classification | Confirmed snapshot/fold bug. |
| Required fix | Add the pending Leech Seed source map to `ShowdownReplayState`, copy it in `_ReplayParser.snapshot()` and `from_snapshot()`, and add a protocol-cut snapshot convergence regression test. |

### 3. Forecast Forme Changes Never Reach the Observation

| Property | Evidence |
| --- | --- |
| Trigger | Castform uses Rain Dance, then Showdown emits `|-formechange|...|Castform-Rainy|`. |
| Divergent surface | The next decision still encodes `species:castform` with Normal type categories rather than the public Castform-Rainy Water identity. |
| Root cause | `_ReplayParser._feed_line()` has no `-formechange` state update, so encoding continues from the original switch-line species. |
| Training impact | Yes. Active identity and type effectiveness are wrong until the Pokemon switches or the battle ends. |
| Classification | Confirmed encoder bug. |
| Required fix | Track active public form overrides in replay state, clear them on switch/replacement, serialize them in snapshots, and apply them before species/type/base-stat token encoding. Add a Forecast form-cycle regression. |

### 4. Color Change Type Overrides Never Reach the Observation

| Property | Evidence |
| --- | --- |
| Trigger | Kecleon is hit by Ice Beam and Showdown emits `|-start|...|typechange|Ice|[from] ability: Color Change`. |
| Divergent surface | The next decision retains Kecleon's Normal type categories instead of the public Ice override. |
| Root cause | The parser’s volatile updater intentionally ignores untracked `typechange` payloads and preserves no dynamic type state. |
| Training impact | Yes. The policy receives the wrong defensive/offensive type matchup until switch-out. |
| Classification | Confirmed encoder bug. |
| Required fix | Track public active type overrides from `-start typechange` and `-end`, clear them on switch/replacement, serialize them in snapshots, and apply overrides to the active Pokemon type categories. Add a Color Change switch-reset regression. |

### 5. Chesto-Rest Leaves the Opponent Encoded as Asleep

| Property | Evidence |
| --- | --- |
| Trigger | `berry_eat_chesto` scenario: Snorlax uses Rest, Showdown emits `-status slp`, `-heal ... slp`, `-enditem Chesto Berry`, then `-curestatus slp`. |
| Divergent surface | The simulator and public replay condition both show a healthy, status-free Snorlax after `-curestatus`; the opposing player's token still encodes `status:slp`. The self view correctly encodes `status:none`. |
| Root cause | `PublicBattleBeliefEngine` clears `belief.status` on `-curestatus`, but leaves the earlier `belief.condition` as `387/387 slp`. `_encode_pokemon_tokens` falls back to `condition.status` whenever `belief.status` is `None`, restoring the stale sleep category. |
| Training impact | Yes. Opponent-facing observations incorrectly treat a Chesto-cured Pokemon as asleep, affecting action ranking and any belief features that depend on current status. |
| Classification | Confirmed encoder/belief-fold bug. This is the same `ledger_skew` class previously documented in `docs/leaf_observation_column_map.md`; the audit establishes a direct live reproduction and shows that the bug is not limited to leaf replay. |
| Required fix | When `-curestatus` is processed, update both canonical status representations: clear the status suffix from `belief.condition` as well as `belief.status`, retaining HP/faint state. Add a cross-seat Chesto-Rest observation regression. |

### 6. Self-Known Ability And Item Do Not Reach Pokemon Tokens

| Property | Evidence |
| --- | --- |
| Trigger | Any normal decision boundary with a self Pokemon whose request supplies a known ability/item, such as seed 2 Sunflora (`Chlorophyll`, `Leftovers`). |
| Divergent surface | The self token has zero revealed ability/item flags and blank ability/item fact buckets even though the self request and simulator state disclose both values. Changing the self item/ability does not change those token facts. |
| Root cause | `observation_from_player_state()` passes self beliefs only as `exact_beliefs_by_species`. `_encode_pokemon_tokens()` reads the standard ability/item fields exclusively from `beliefs_by_species`, which is `None` for the self side. More importantly, that belief entry is a candidate summary, while the authoritative self request already carries exact `ShowdownPokemon.ability` and `.item`; the exact-state block does not encode either. |
| Training impact | Yes. The policy cannot condition on its own held item or ability except through incidental downstream effects, despite both being private information it is entitled to observe. This also explains the self Transform identity gap: the same missing `beliefs_by_species` path bypasses transform facts. |
| Classification | Confirmed encoder bug. |
| Required fix | For `role == self`, encode exact ability/item directly from the self request's `ShowdownPokemon` fields, including correct removal/current-item semantics. Separately pass self beliefs through `beliefs_by_species` to restore self Transform identity. Keep opponent epistemic buckets unchanged. Add a regression that varies only self ability/item and asserts changed self token facts with zero self uncertainty. |

### 7. `-cureteam` Leaves Benched Living Status And Toxic State Stale

| Property | Evidence |
| --- | --- |
| Trigger | A statused Pokemon switches out; its teammate uses Heal Bell/Aromatherapy and Showdown emits `|-cureteam|pN`; the statused Pokemon remains benched. |
| Divergent surface | The public replay and belief state retain the benched Pokemon's old condition/status (for example `300/300 tox`) after the team-wide cure. The next opponent token therefore encodes stale status and can retain stale toxic state. |
| Root cause | The replay public-condition updater, toxic-stage tracker, and public belief engine all handle `-curestatus` but not `-cureteam`. The existing cure-all counter supports Natural Cure inference but does not clear the affected members. |
| Training impact | Yes. Heal Bell/Aromatherapy can leave an entire bench represented with obsolete status facts until each Pokemon returns or another event overwrites it. |
| Classification | Confirmed replay/belief-fold bug. |
| Required fix | Handle `-cureteam` in the replay parser and belief engine: clear every living member's condition-status suffix, belief status/sleep bookkeeping, and toxic stage for that side; include it in pending-switch-boundary handling. Add an Aromatherapy benched-toxic cross-seat regression. |

### 8. Residual Toxic Damage Incorrectly Rules Out Pinch Berries

| Property | Evidence |
| --- | --- |
| Trigger | Random seed 3: Ludicolo holds Petaya Berry, switches in poisoned, then falls from `93/272` to `42/272 tox` due to end-of-turn Toxic damage. The bridge and next request still show the Petaya Berry held. |
| Divergent surface | At `|upkeep|`, the belief engine adds Petaya/Salac/Liechi to `ruled_out_items` because HP is under 25% and no berry was eaten. The rule-out leaves no compatible variant after the fourth revealed move, so the source degrades to the full two-variant pool and candidate count jumps from 1 to 2. |
| Root cause | `_sweep_end_of_turn_non_procs()` tests final HP, rather than the action-phase HP snapshot already captured in `_hp_after_actions`. In Gen 3, a Pokemon that crosses the pinch threshold only from residual Toxic did not receive a berry activation opportunity at that boundary. |
| Training impact | Yes. The model falsely discards the real held-item possibility, then sees an inconsistent full-pool fallback and inflated uncertainty/candidate count for the opponent. |
| Classification | Confirmed belief-pruning bug. The candidate-count increase is a downstream symptom, not a separate root cause. |
| Required fix | Apply pinch non-proc pruning only when the action-phase HP snapshot crossed or was already below the threshold at a point where the berry could have activated; do not prune solely from residual-end HP. Add a Toxic-residual Petaya regression that verifies no rule-out and monotone candidates. |

### 9. Move-Weather Counter Is One Turn Stale After Rain Dance

| Property | Evidence |
| --- | --- |
| Trigger | Rain Dance resolves, residual upkeep completes, and the next decision is requested. Reproduced locally from random seed 112 at turns 45 and 51. |
| Divergent surface | On the first post-resolution decision, the bridge reports four of five weather turns remaining (`duration: 4`), while `field.weather_turns` encodes the full five turns (`1.0`). A second Rain Dance reproduction encodes four turns while the bridge reports three. |
| Long-shard incidence | Four decision boundaries across three random seeds (112 twice, 124, and 152) in 17,963 audited boundaries. |
| Root cause | `_weather_duration_features()` subtracts only whole replay-turn differences from `weather_set_turn`. The observation boundary occurs after the set turn's upkeep, which already consumes the first duration tick. |
| Training impact | Yes, at low incidence. The policy receives an overlong move-weather horizon, which can distort timing-sensitive rain decisions and weather-dependent move/ability evaluation. |
| Classification | Confirmed encoder bug. |
| Required fix | Define move-weather remaining turns at the post-upkeep request boundary: consume the first tick when calculating the first observation after a fresh `-weather` line. Add start and recast Rain Dance regressions that compare encoded `weather_turns` with the bridge duration. |

## Deep-Line Disposition

The depth-focused audit is complete. Its persistent run covered 100 random
battles plus seven public protocol cuts, checking 17,963 decision boundaries
(14,027 at turn 20 or later). It verified no incremental-vs-batch or
snapshot-vs-live divergence in the focused local smoke, reproduced eight
pre-existing encoder/belief defects at real-game depth, and identified the new
post-resolution move-weather countdown defect above.

Protocol event-type coverage in the persistent artifact included 7,951
`switch`, 1,106 `-ability`, and 1,545 `-weather` events. Its event-type
co-occurrence census observed 67 `-ability`/`switch` and 62
`-weather`/`switch` turn groups. Event types alone cannot identify a particular
ability or move, so the explicit Intimidate, Sand Stream, and Baton Pass public
protocol cuts remain the semantic coverage for those chains; all three ran
without a new root cause.

Recommended permanent gate: retain the live differential, public bridge field
oracle, raw request action/PP oracle, candidate monotonicity, snapshot/live,
and perspective checks; run the deterministic mechanic and protocol cuts on
encoder/belief changes; and use a persistent turn-20+-weighted random shard
with known signatures suppressed for broad accumulation coverage. The runner
now applies that suppression by default for future shards. No production
encoder fix is included in this read-only audit branch.

The depth-only result did **not** claim exhaustive atom coverage. The completed
source-derived enumeration sweep now supplies that breadth evidence: every Gen
3 randbat species, reachable ability, source movepool move, and source item has
been exercised through the production encoder. It does not replace deep-line
coverage of multi-turn mechanic chains; the two lanes remain complementary.

## Coverage Enumeration Disposition

The deterministic breadth lane is implemented and locally validated against a
fresh Showdown build. Its planner fails closed when any expected source move or
item lacks a materializable source-variant carrier; its ledger records a
first-covering game and an explicit uncovered set for species, ability pairs,
moves, and items. The completed source revision required no gap-fill games, but
gap-fill remains mandatory behavior if a future source update leaves an atom
uncovered by the two main passes.

No new encoder or belief bug was confirmed by the exhaustive static source
sweep. The generated JSON evidence is committed, the coverage implementation
received an independent adversarial review, and the permanent-gate/report
completion audit is complete.

## Runner Provenance

Future persistent deep-line random shards suppress the seven signatures already
triaged above: self-known ability/item, self Transform identity, the two
`-cureteam` status surfaces, Forecast `-formechange` identity, and the
downstream candidate-count increase. This changes only audit-artifact signal to
noise; it does not hide the defects or alter the encoder. The immutable shard
already in flight retains its original configuration, while later shards will
continue to record newly discovered signatures without repeated copies of the
known set.

## Completion Audit

- Deep-line execution: 100 persistent random games, seven protocol cuts, and
  17,963 decision boundaries. Confirmed findings and their actionable fixes are
  recorded in the finding tables above; each newly confirmed bug was also
  reported through Slack during execution.
- Fresh-build enumeration: a clean Showdown build and source hash
  `754b71cfed643fa0` produced 220/220 completed fixtures, zero findings, and
  empty coverage-ledger sets for 220 species, 235 reachable species-ability
  pairs, 125 source moves, and 13 source items. The committed JSON artifacts
  preserve the machine-readable evidence and per-atom first coverage.
- Exact-tuple enumeration: the opt-in exact-variant lane completed 874/874
  fixtures, checked all 1,748 source tuples from both player perspectives, and
  produced an empty `uncovered.variants` set. This is stronger static coverage,
  but still does not claim exhaustive multi-turn interaction coverage.
- Verification: the merged-main focused audit suite passed 17 tests. Full
  `unittest` discovery with the fresh Showdown root runs 90 tests but retains
  one pre-existing, environment-sensitive bootstrap expectation: it assumes
  `dex unavailable`, while the deliberately built checkout makes the Dex
  available. This is unrelated to the audit changes and is not counted as a
  green full-suite result.
- Independent review: Claude Code adversarially reviewed PR #773, found no
  blockers, and its Trace public-versus-native ability regression-test request
  was added before merge.

REPORT FINISHED
