# Gen 3 Randbats V3 Observation Audit Findings

**Status:** audit cycle in progress. No historical result is a current clean
result unless its complete provenance tuple matches the active source inputs.

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
| Protocol-signature census schema (E/O/C only) | Must be `pokezero.protocol-signature-census.v2` for artifacts consumed by the E/O/C differential. |
| Immutable image digest | Pins the runtime and bundled dependencies. |
| Command, seed/shard range, and completion time | Makes the result reproducible and scoped. |

An artifact that lacks one of these fields, mixes fields across shards, or
fails its terminal validation is **not** audit evidence. It may be retained as
historical debugging context, but it must not be reported as a clean result or
used to close a layer.

## Active Audit Cycle

The rows below are the only live evidence for the current canonical-signature
audit cycle. They remain `Running` until their persistent Jobs write validated
terminal aggregate artifacts; earlier artifacts are retained solely to explain
why this recheck is required. The current cycle uses protocol-signature census
schema `pokezero.protocol-signature-census.v2`; reports without that stamp are
not valid input to the current E/O/C differential.

| Cycle | Public revision | Observation schema | Protocol signature schema | Layer | Status | Aggregate artifact |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-20 | `996ea5489c48e8a8415057ce0258d8f144f65079` | v3 | v2 | Exact universe fixtures | Complete: clean. All 1,682 current exact variants covered, zero findings, and no uncovered source atoms. | [public evidence mirror](audit_artifacts/v3signature-coverage-r2-summary.json) |
| 2026-07-20 | `996ea5489c48e8a8415057ce0258d8f144f65079` | v3 | v2 | Bounded-depth exact fixtures | Complete: clean. Eight-round lane checked 9,594 decisions with zero findings and full coverage. | [public evidence mirror](audit_artifacts/v3signature-coverage-r2-summary.json) |
| 2026-07-20 | `996ea5489c48e8a8415057ce0258d8f144f65079` | v3 | v2 | Curated party interactions | Complete: clean. The defined multi-mon interaction suite checked 489 decisions with zero findings. | [public evidence mirror](audit_artifacts/v3signature-coverage-r2-summary.json) |
| 2026-07-20 | `996ea5489c48e8a8415057ce0258d8f144f65079` | v3 | N/A | Silent engine-mutation lane | Complete: clean. 675 bounded transitions, zero unaccounted candidates. | [public evidence mirror](audit_artifacts/v3signature-silent-r2-summary.json) |
| 2026-07-20 | `996ea5489c48e8a8415057ce0258d8f144f65079` | v3 | v2 | E/O/C protocol inventory and census differential | In progress. Fixture observations are available; retry-safe production-style count-only capture support landed in #802 before final aggregation. | `v3signature-inventory-r2` terminal artifact pending |
| 2026-07-20 | `996ea5489c48e8a8415057ce0258d8f144f65079` | v3 | N/A | Encoding-collision capture and audit | In progress. Persistent compact capture is collecting the 100k-decision corpus. | `v3signature-collision-r3` terminal artifact pending |
| 2026-07-20 | `996ea5489c48e8a8415057ce0258d8f144f65079` | v3 | N/A | Counterfactual harm probes | Runs only for the validated shortlist. | Not started |

## Historical Triage Evidence

The pre-repair cycle remains useful for identifying what the refreshed audit
must confirm or minimize. It is not current clean evidence because the audit
oracle and snapshot behavior changed in the parity repair.

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

### Current Clean Coverage Result

The fresh exact-universe audit contains **1,682** variants. This supersedes the
historical 1,748 count for this audit cycle; it is not a partial run or a
coverage regression. The shared Showdown source hash is unchanged, so the
count difference must not be read as evidence of an upstream source update.
The static lane covered all 1,682 current tuples and reported zero findings,
zero failure-only artifacts, and no uncovered species, moves, items,
abilities, or exact variants. Its immutable source hash is
`754b71cfed643fa0`; the runtime image is pinned by digest
`sha256:0a5de863336b035902bd1552b2247d9b4d350796b95a204ef71d488163993949`.

The fresh bounded-depth lane completed all eight-round exact fixtures with
9,594 checked decisions and zero findings. The curated party lane completed
489 checked decisions and zero findings. These lanes together cover every
current source tuple at the static surface, bounded multi-turn behavior for
every tuple, and the defined party/switch/clause interaction registry. They
do **not** claim exhaustive arbitrary six-mon game-state coverage, every
chance outcome, or that every move is used in every possible tactical context.

The broader party fixtures intentionally exercise some legal Gen 3 mechanics
outside the current random-battle source universe. Vocabulary warnings for
those fixture-only atoms are recorded as an explicit fixture-boundary
limitation, not as a reachable Gen 3 random-battle observation finding.

### Current Silent-Mutation Result

The fresh bounded instrumentation pass checked 675 transitions across random
and curated interactions. It found zero unaccounted silent mutations after
the audit surface was repaired to ignore inactive type resets and normalize
fainted state. The earlier 30-candidate result is therefore historical oracle
debugging context, not an unresolved observation or belief omission.

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
change is proposed. A candidate is actionable only after its evidence row is
complete; `COVERED`, `UNREACHABLE`, and `ACCEPTED-LOSS` are explicit verdicts,
not omissions from this table.

| Canonical signature or public fact | Layer | Reachability evidence | Observed count and provenance | Consuming handler | Collision or mutation evidence | Harm probe | Verdict | Reproduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Status/type candidates from the pre-repair silent surface | Silent mutation | Fresh random plus curated recheck | Historical 30 candidates; current 675-transition aggregate has 0 | The repaired audit excludes inactive resets and normalizes fainted state before classification | No fresh candidate survived | Not needed | SUPERSEDED BY CLEAN RE-RUN | `python scripts/silent_mutation_audit.py --observation-schema v3 --random-games 8 --max-rounds 120 --interaction-registry --json audit.json` |
| Live Tier-2/investment annotations differed from a bare batch fold | Bounded depth | Fresh exact eight-round recheck | Historical 6,111 annotation-only divergences; current 9,594-decision lane has 0 findings | Batch-equivalent annotation overlay now matches the live path | No fresh divergence survived | Not needed | SUPERSEDED BY CLEAN RE-RUN | Fresh bounded-depth aggregate in `v3signature-coverage-r2` |
| Castform Forecast and Ditto Transform party-oracle differences | Party fixture | Fresh curated interaction recheck | Historical six divergences; current 489-decision lane has 0 findings | Oracle contract now validates base species with live type state | No fresh divergence survived | Not needed | SUPERSEDED BY CLEAN RE-RUN | Fresh party aggregate in `v3signature-coverage-r2` |

## Completion Record

The schema-freeze recommendation is published only after every current-cycle
layer above has a validated artifact or an evidence-based limitation, every
candidate has a verdict, and all `ADD` candidates are grouped into one reviewed
implementation proposal. The remaining open gates are the E/O/C differential,
the 100k collision audit, and any resulting focused harm probes. A clean layer
is recorded as a completed row with its full provenance; an empty table alone
is never evidence of a clean audit.
