# Gen 3 Randbats V3 Observation Audit Findings

**Status:** V6 exact-universe, bounded-depth, silent-mutation, and curated
party lanes are clean. Collision capture and the canonical E/O/C differential
remain in progress; no schema-freeze recommendation has been made.

This is the durable, public findings ledger for the v3 observation-schema
freeze gate. It records completed audit layers and all omission candidates in
one place. The companion [plan](silent_noop_sweep_plan.md) defines the audit
methods and the [coverage plan](coverage_enumeration_audit_plan.md) defines
the exact-universe and bounded-depth fixtures.

## Evidence Rules

Every completed row below must link to a durable aggregate artifact and carry:

| Required provenance | Purpose |
| --- | --- |
| Public-repository commit | Pins encoder, belief, parser, and audit code. |
| Showdown/engine source hash | Pins the reachable Gen 3 universe and simulator behavior. |
| Observation schema | Must be `pokezero.observation.v3` for dynamic lanes. |
| Protocol-signature census schema (E/O/C only) | Must be `pokezero.protocol-signature-census.v2` for observed signatures and `pokezero.protocol-emission-inventory.v3` for the canonical E/C differential. |
| Immutable image digest | Pins the runtime and bundled dependencies. |
| Command, seed/shard range, and completion time | Makes the result reproducible and scoped. |

An artifact that lacks one of these fields, mixes fields across shards, or
fails its terminal validation is **not** audit evidence. It may be retained as
historical debugging context, but it must not be reported as a clean result or
used to close a layer.

The final tracked aggregate is generated with
`scripts/publish_v3_audit_evidence.py`. It validates every terminal input and
their common provenance tuple, then publishes a whitelist-only summary under
`docs/audit_artifacts/`. The publisher retains the immutable image digest but
does not copy private paths, fully qualified image names, or deployment data
into the public repository.

## Current V6 Results

The active V6 core wave was built from public commit
`cc1042b61222fcf0e8a608b3f1b762cbafc35891`, Showdown source
`9d01efb72af41473`, and `pokezero.observation.v3`. Its immutable runtime image
digest is `sha256:ba14c4bc80281f899ee6cfee71b9946660f9529c2f67b1fa1efb32384a2a3f2a`.
Its public-safe terminal summary is
[`v6-core-terminal-cc1042b.json`](audit_artifacts/v6-core-terminal-cc1042b.json).

| Layer | Result | Evidence boundary |
| --- | --- | --- |
| Exact source-universe | Clean: 1,682 decision boundaries, eight shards, no uncovered species, ability pairs, moves, items, or variants; zero failure artifacts. | The source universe has 1,748 final-set variants. The smaller decision count is the number of eligible decision boundaries exercised by the fixture, not a reduced variant universe. See the [core terminal summary](audit_artifacts/v6-core-terminal-cc1042b.json). |
| Bounded depth | Clean: 9,594 decision boundaries, depth eight, eight shards, zero findings and zero failure artifacts. | Covers the exact universe through the bounded multi-turn contract; it is not exhaustive arbitrary battle-state coverage. See the [core terminal summary](audit_artifacts/v6-core-terminal-cc1042b.json). |
| Curated party interactions | Clean in the original V6 registry: 489 decisions, zero findings. | This original registry predates the targeted Natural Cure fixture below. See the [core terminal summary](audit_artifacts/v6-core-terminal-cc1042b.json). |
| Silent engine mutations | Clean: 675 steps, zero unaccounted silent candidates. | Eight random games plus the documented interaction-registry scenarios; not a proof about every possible long-game state. See the [core terminal summary](audit_artifacts/v6-core-terminal-cc1042b.json). |
| Encoding collisions | Running: 20 resumable capture shards targeting at least 100,000 public decision records. | The audit hashes only model-visible arrays and masks, scopes comparisons by perspective and decision kind, and keeps compact public locators for later hydration. |
| Canonical E/O/C differential | Running. | The report will distinguish engine-emittable (`E`), observed (`O`), and consumed (`C`) canonical signatures. No learned schema-v3 policy capture exists, so that specific census will be an explicit limitation rather than a clean result. |
| Layer 4 harm probes | Pending shortlist. | Probes run only for non-whitelisted collisions or prioritized E/O/C candidates. |

### Natural Cure Addendum

Public commit `6b292aebdd66bca799fa419c8c58f5ba269abde5` added the
`natural_cure_switch` party fixture and its regression assertions. The diff
from the V6 core commit changes only the fixture and its test; it does not
change the engine, belief, parser, or observation implementation. A targeted
party-only rerun on that current commit, using Showdown source
`9d01efb72af41473`, schema v3, and image digest
`sha256:80c53db660591c6a6992efe030fed0af8b75d45b30e17cd6a1a6d5248a74fbdc`,
checked 495 decisions with zero findings. Its scenario registry explicitly
includes `natural_cure_switch`; see the [targeted party
summary](audit_artifacts/v6-natural-cure-party-6b292ae.json).

The fixture first proves that Toxic landed, then checks the public Natural
Cure status removal after switch-out and before re-entry. This prevents the
deterministic clean-reentry inference from masking a lost public cure line.
The separate unit regression for the out-of-format multi-active
`showCure=false` boundary remains required because Gen 3 random-battle
Singles emits the public `-curestatus` line for Natural Cure.

The final public aggregate will retain the V6 common-provenance bundle and
link this targeted current-source party addendum separately. It must not claim
that a single mixed-commit artifact is a uniform V6 result.

## Previous Cycle Detail

The final audit cycle uses source identity schema v6, which hashes the resolved
Gen 3 Dex metadata used to materialize variants plus the bounded Showdown
simulator surface that determines protocol emissions and state mutations. No v5
artifact below is current clean evidence: its source identity did not cover
that simulator surface, so a matching v5 hash could still describe a different
engine.

The current-source v5 diagnostic wave was rebuilt from public commit
`cb159b7d9fcf7bc4d18b14723a6498a5d87b07dc` after the Wish residual-order repair
and the count-only protocol-census contract landed. It reruns exact-universe,
bounded-depth, party-interaction, silent-mutation, collision, and E/O/C
inventory lanes with an immutable image and fresh source provenance. Its
terminal aggregate is retained as diagnostic evidence while the canonical
inventory, collision-hydration, and v6 engine-source contracts are hardened. A
rebuilt v6 wave after those contracts land is required before the schema-freeze
gate can close.

The replacement inventory is canonical-signature based: it distinguishes
payload-sensitive engine emissions and consumer patterns such as
`-activate:<effect>` and `cant:<reason>`, and it records dynamic or
unparseable source calls as unresolved evidence rather than silently reducing
them to a tag. A terminal inventory can be `clean` only when every tag and
canonical differential is resolved, both static surfaces are complete, and a
provenanced learned-v3 self-play O-census is present. Until then its terminal
status is `needs-triage`, even if fixture and fixed-opponent capture rows are
otherwise clean.

No trained v3 checkpoint exists yet. The learned-policy production-self-play
O-census is therefore deliberately **not** included in this wave: using a
legacy checkpoint would violate the strict v3 boundary. The inventory still
collects O from the fresh fixture and capture lanes, but this limitation must
remain explicit until the first v3-trained checkpoint can supply a separately
provenanced learned-policy capture. It is not a clean learned-policy result.

The prior v5 source-and-observation boundary was public commit
`a61ee3e710965fa114cf4889cbe33be09026ff34` (#820). It includes the v3 parser
and encoder additions for consecutive stall state, confusion turns-so-far,
Encore turns-so-far, Wrap elapsed turns, per-mon gender, Mean Look / Spider
Web trapping, and the per-side Wish turns-to-land clock. Its numeric width is
therefore **168**. The Wish repair also keeps a landing Wish heal out of the
belief layer's action-phase HP snapshot, so it cannot mask a pinch-item
non-proc. The first v5 wave from this source boundary is retained as diagnostic
evidence only. Follow-up audit-contract hardening in public commit
`250bdf5c533f44087c55aadfde28d50695092bf4` (#822) makes protocol backing
entity-specific in the silent-mutation lane and hashes collision numerics at
the model's float32 boundary. The diagnostic and replacement waves may still
contribute reproducible candidate evidence, but they cannot be cited as a
final clean result after the Wish-inclusive rebuild.

The in-flight 161-feature v5 wave remains useful regression evidence, but it
is historical and cannot close this gate because it predates the later v3
additions and the Wish repair.

| Cycle | Public revision | Observation schema | Protocol signature schema | Layer | Status | Aggregate artifact |
| --- | --- | --- | --- | --- | --- | --- |
| v5 Wish-inclusive diagnostic | `cb159b7d9fcf7bc4d18b14723a6498a5d87b07dc` | v3 | v2 | Exact universe fixtures | Submitted from the rebuilt image; diagnostic only while evidence contracts are hardened. | Terminal aggregate pending |
| v5 Wish-inclusive diagnostic | `cb159b7d9fcf7bc4d18b14723a6498a5d87b07dc` | v3 | v2 | Bounded-depth exact fixtures | Submitted behind the static gate; diagnostic only. | Terminal aggregate pending |
| v5 Wish-inclusive diagnostic | `cb159b7d9fcf7bc4d18b14723a6498a5d87b07dc` | v3 | v2 | Curated party interactions | Submitted behind the static gate; diagnostic only. | Terminal aggregate pending |
| v5 Wish-inclusive diagnostic | `cb159b7d9fcf7bc4d18b14723a6498a5d87b07dc` | v3 | N/A | Silent engine-mutation lane | Submitted with the repaired classifier; diagnostic only. | Terminal aggregate pending |
| v5 Wish-inclusive diagnostic | `cb159b7d9fcf7bc4d18b14723a6498a5d87b07dc` | v3 | v2 | E/O/C protocol inventory and census differential | Submitted behind terminal coverage and capture markers; no learned-policy v3 capture exists yet; diagnostic only until canonical E/C is enforced. | Terminal aggregate pending |
| v5 Wish-inclusive diagnostic | `cb159b7d9fcf7bc4d18b14723a6498a5d87b07dc` | v3 | N/A | Encoding-collision capture and audit | Submitted with model-float32 numeric hashing; diagnostic only until compact candidates are hydrated. | Terminal aggregate pending |
| v5 Wish-inclusive diagnostic | `cb159b7d9fcf7bc4d18b14723a6498a5d87b07dc` | v3 | N/A | Counterfactual harm probes | Runs only for the validated current-wave shortlist. | Not started |
| v5 diagnostic | `a61ee3e710965fa114cf4889cbe33be09026ff34` | v3 | v2 | Exact universe fixtures | Submitted; may produce diagnostic candidates only. | Terminal aggregate pending |
| v5 diagnostic | `a61ee3e710965fa114cf4889cbe33be09026ff34` | v3 | v2 | Bounded-depth exact fixtures | Submitted behind the static gate; diagnostic only. | Terminal aggregate pending |
| v5 diagnostic | `a61ee3e710965fa114cf4889cbe33be09026ff34` | v3 | v2 | Curated party interactions | Submitted behind the static gate; diagnostic only. | Terminal aggregate pending |
| v5 diagnostic | `a61ee3e710965fa114cf4889cbe33be09026ff34` | v3 | N/A | Silent engine-mutation lane | Submitted; diagnostic only under the pre-hardening classifier. | Terminal aggregate pending |
| v5 diagnostic | `a61ee3e710965fa114cf4889cbe33be09026ff34` | v3 | v2 | E/O/C protocol inventory and census differential | Submitted behind terminal coverage and capture markers; diagnostic only. | Terminal aggregate pending |
| v5 diagnostic | `a61ee3e710965fa114cf4889cbe33be09026ff34` | v3 | N/A | Encoding-collision capture and audit | Submitted; diagnostic only under pre-float32 hashing. | Terminal aggregate pending |
| v5 replacement | `250bdf5c533f44087c55aadfde28d50695092bf4` | v3 | v2 | Exact universe fixtures | Submitted as a final-eligible replacement; terminal aggregate pending. | Terminal aggregate pending |
| v5 replacement | `250bdf5c533f44087c55aadfde28d50695092bf4` | v3 | v2 | Bounded-depth exact fixtures | Submitted behind the static gate; terminal aggregate pending. | Terminal aggregate pending |
| v5 replacement | `250bdf5c533f44087c55aadfde28d50695092bf4` | v3 | v2 | Curated party interactions | Submitted behind the static gate; terminal aggregate pending. | Terminal aggregate pending |
| v5 replacement | `250bdf5c533f44087c55aadfde28d50695092bf4` | v3 | N/A | Silent engine-mutation lane | Submitted with entity-specific protocol backing; terminal aggregate pending. | Terminal aggregate pending |
| v5 replacement | `250bdf5c533f44087c55aadfde28d50695092bf4` | v3 | v2 | E/O/C protocol inventory and census differential | Submitted behind terminal coverage and capture markers; terminal aggregate pending. | Terminal aggregate pending |
| v5 replacement | `250bdf5c533f44087c55aadfde28d50695092bf4` | v3 | N/A | Encoding-collision capture and audit | Submitted with model-float32 numeric hashing; terminal aggregate pending. | Terminal aggregate pending |
| v5 replacement | `250bdf5c533f44087c55aadfde28d50695092bf4` | v3 | N/A | Counterfactual harm probes | Runs only for the validated replacement-wave shortlist. | Not started |

## Historical Triage Evidence

The pre-repair and v4-source-identity cycles remain useful for identifying what
the refreshed audit must confirm or minimize. They are not current clean
evidence because either the audit oracle/snapshot behavior changed or source
identity did not include every input that could change the materialized
universe.

| Cycle | Public revision | Schema | Layer | Status | Aggregate artifact |
| --- | --- | --- | --- | --- | --- |
| 2026-07-20 | `f54617455a28238d764d1bbb9ee0fb840092edf5` | v3 | Exact universe fixtures | Complete: clean | Private immutable aggregate, coverage complete |
| 2026-07-20 | `f54617455a28238d764d1bbb9ee0fb840092edf5` | v3 | Bounded-depth and party fixtures | Complete: needs audit-oracle triage | Private immutable aggregate, coverage complete |
| 2026-07-20 | `f54617455a28238d764d1bbb9ee0fb840092edf5` | v3 | Silent engine-mutation lane | Complete: needs triage | Private immutable artifact `sha256:0c2a59f3320abea600400f96d1174e84bd1abdf6649774a4057a537889e948f3` |
| 2026-07-20 | `f54617455a28238d764d1bbb9ee0fb840092edf5` | v3 | E/O/C protocol inventory and census differential | Depends on fresh coverage artifact | Pending |
| 2026-07-20 | `f54617455a28238d764d1bbb9ee0fb840092edf5` | v3 | Encoding-collision capture and audit | Checkpointless v3 capture support merged; awaiting fresh terminal artifact | Pending |
| 2026-07-20 | `f54617455a28238d764d1bbb9ee0fb840092edf5` | v3 | Counterfactual harm probes | Runs only for the validated shortlist | Not started |

The prior provenance-rejected coverage and silent-mutation attempts are not
listed as results. They did not prove a v3 audit outcome.

### Historical v4 Coverage and Fixture-Boundary Result

The v4 exact-universe audit reported **1,682** variants, zero findings, zero
failure-only artifacts, and no uncovered species, moves, items, abilities, or
exact variants. It is useful historical evidence, but must be re-run under v5
before it can support a schema-freeze recommendation.

The v4 bounded-depth lane completed all eight-round exact fixtures with
9,594 checked decisions and zero findings. The curated party lane completed
489 checked decisions and zero findings. These lanes together cover every
current source tuple at the static surface, bounded multi-turn behavior for
every tuple, and the defined party/switch/clause interaction registry. They
do **not** claim exhaustive arbitrary six-mon game-state coverage, every
chance outcome, or that every move is used in every possible tactical context.

The broader party fixtures intentionally exercise some legal Gen 3 mechanics
outside the current random-battle source universe. The warned Attract,
Safeguard, Taunt, Future Sight, Reflect, Light Screen, Confuse Ray, Chesto
Berry, and Quick Claw atoms are absent from both current `sets.json` and the
v5 materialized universe. They are explicit fixture-boundary limitations, not
reachable Gen 3 random-battle observation findings.

### Historical v4 Silent-Mutation Result

The v4 bounded instrumentation pass checked 675 transitions across random and
curated interactions. It found zero unaccounted silent mutations after the
audit surface was repaired to ignore inactive type resets and normalize fainted
state. The earlier 30-candidate result is historical oracle debugging context,
not an unresolved observation or belief omission; v5 must independently
confirm this clean result.

### Pre-Repair Coverage Result

The prior exact-universe lane exercised all 1,748 source variants across 841
games and 1,682 decision boundaries. It reported zero findings, zero failure
artifacts, and no uncovered species, moves, items, ability pairs, or variants.
It is historical coverage evidence, not a current clean result; the active
cycle re-runs the same source-universe and observation-surface gate on the
current canonical-signature revision. Neither result is a claim of exhaustive battle-state
coverage.

The bounded-depth lane reached coverage completion over 841 games and 9,594
decision boundaries, but reported 6,159 integrity divergences. The preliminary
breakdown is 6,111 incremental-versus-batch comparisons, 36
snapshot-versus-live comparisons, and 12 bridge-oracle comparisons. The
dominant class is currently explained by the batch oracle omitting the live
Tier-2/investment annotation overlay, so it must be repaired and re-run before
being interpreted as a production encoder defect.

The party fixtures reported six bridge-oracle divergences across 489 decision
boundaries. Four involve Castform Forecast, whose intentional base-species plus
live-type representation conflicts with an obsolete oracle that expects a form
species identity. The remaining two are Ditto Transform comparisons and remain
under triage. Neither class is a production finding until the oracle contract
is corrected and the fixtures are re-run.

### Pre-Repair Silent-Mutation Result

The completed historical lane ran eight random games (up to 120 decision rounds each)
plus the curated interaction registry. It audited 675 transitions across 7,755
public entities. The command selected v3 explicitly and recorded engine source
`754b71cfed643fa0` at `2026-07-20T23:56:45Z`.

| Classification | Count | Interpretation |
| --- | ---: | --- |
| Protocol-backed | 2,016 | The mutation coincided with an expected public protocol family. |
| Belief-inferred | 49 | The public belief state matched the resulting status without a direct status tag. |
| Silent candidate | 30 | Requires replay/minimization before it can be called an observation or belief defect. |

The 30 candidates are 27 status transitions and three Kecleon type
transitions. The artifact intentionally omits raw simulator values, so the
candidate count alone does not establish that any transition is publicly
observable or behaviorally harmful.

## Candidate Verdicts

Each candidate discovered by Layers 1--3 is appended here before any schema
change is proposed. Known targeted regressions are retained here as well, but
are explicitly labeled rather than credited to a broad audit layer. A candidate
is actionable only after its evidence row is complete; `COVERED`,
`UNREACHABLE`, and `ACCEPTED-LOSS` are explicit verdicts, not omissions from
this table.

| Canonical signature or public fact | Layer | Reachability evidence | Observed count and provenance | Consuming handler | Collision or mutation evidence | Harm probe | Verdict | Reproduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Status/type candidates from the pre-repair silent surface | Silent mutation | Fresh random plus curated recheck | Historical 30 candidates; current 675-transition aggregate has 0 | The repaired audit excludes inactive resets and normalizes fainted state before classification | No fresh candidate survived | Not needed | SUPERSEDED BY CLEAN RE-RUN | `python scripts/silent_mutation_audit.py --observation-schema v3 --random-games 8 --max-rounds 120 --interaction-registry --json audit.json` |
| Live Tier-2/investment annotations differed from a bare batch fold | Bounded depth | Fresh exact eight-round recheck | Historical 6,111 annotation-only divergences; current 9,594-decision lane has 0 findings | Batch-equivalent annotation overlay now matches the live path | No fresh divergence survived | Not needed | SUPERSEDED BY CLEAN RE-RUN | Fresh bounded-depth aggregate in `v3signature-coverage-r2` |
| Castform Forecast and Ditto Transform party-oracle differences | Party fixture | Fresh curated interaction recheck | Historical six divergences; current 489-decision lane has 0 findings | Oracle contract now validates base species with live type state | No fresh divergence survived | Not needed | SUPERSEDED BY CLEAN RE-RUN | Fresh party aggregate in `v3signature-coverage-r2` |
| Wish landing heal after an action-phase pinch-item non-proc | Targeted belief regression, not audit-discovered | Minimal public trace: action-phase damage crosses the pinch threshold, then `-heal ... [from] move: Wish` lands at residual | One synthetic public trace in `ExactStateLedgerTest`; no v5 broad-lane finding is claimed | `belief._RESIDUAL_HP_TAGS` keeps the residual Wish heal from replacing the action-phase HP snapshot | Not a collision candidate; the old failure was an over-retained source-valid candidate set | Directly asserts all incompatible pinch items are ruled out; no separate harm probe required | COVERED BY TARGETED REGRESSION; broad Wish and berry scenarios did not independently detect this ordering case | `python -m unittest tests.test_belief.ExactStateLedgerTest.test_wish_landing_heal_does_not_mask_the_action_phase_pinch_non_proc` |

## Completion Record

The schema-freeze recommendation is published only after every current-cycle
layer above has a validated artifact or an evidence-based limitation, every
candidate has a verdict, and all `ADD` candidates are grouped into one reviewed
implementation proposal. The remaining open gates are canonical E/O/C
enumeration with verdicts for all tag and signature E-O/C-E rows, hydrated 100k
collision evidence, the learned-policy O-census after a v3 checkpoint exists,
and any resulting focused harm probes. A clean layer is recorded as a completed
row with its full provenance; an empty table alone is never evidence of a clean
audit.
