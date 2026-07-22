# V3 Observation Audit -- End-of-Cycle Evaluation Plan

Status: **planned; do not launch until the v3 token-format refactor has
merged.** This is the execution contract for the final observation/belief
audit. It complements [the layered silent-noop plan](silent_noop_sweep_plan.md)
and [the findings ledger](silent_noop_sweep_findings.md); it does not replace
their candidate-level evidence requirements.

## Decision Boundary

The final evaluation answers a narrow question: does the frozen v3 public
observation and belief implementation omit, conflate, or incorrectly infer a
reachable Gen 3 random-battle fact within the audited surface?

It is deliberately not a model-strength benchmark, an exhaustive enumeration
of every chance sequence, or evidence that a learned policy will encounter
every possible long-game history. A clean run establishes the stated source
universe, fixture depth, party-interaction, omission-detection, and
silent-mutation coverage. The final findings report must retain residual
limitations rather than overstate that result.

## Prerequisite: Token-Format Refactor

The pending v3 token-format refactor is a hard precondition. It changes
model-visible inputs, so static, bounded-depth, party, collision, and E/O/C
artifacts produced before that refactor are historical diagnostics only.

Before any final-evaluation job is submitted:

1. Merge the token-format refactor and update this plan's execution branch to
   the resulting `main` commit.
2. Rebase or rebuild the audit repair against that commit, including the
   canonical protocol-signature fixes.
3. Build the audit image in-region from the merged source and record its
   immutable digest.
4. Recompute all final artifacts from that one source identity. Do not relabel
   or reuse a prior V6 result whose model-visible signature schema differs.

No final-evaluation result may mix a pre-refactor token encoder with a
post-refactor parser, belief engine, source universe, or image.

## Source Identity And Provenance

Every shard and every merged artifact must share this provenance tuple:

| Field | Requirement |
| --- | --- |
| Public commit | Exact merged public-repository revision containing encoder, belief, parser, audit, and token-format code. |
| Showdown source hash | Exact local Showdown/engine source used for the Gen 3 universe and simulator. |
| Observation schema | `pokezero.observation.v3` and the final token-format revision. |
| Signature schemas | Final canonical protocol-signature and emission-inventory schema identifiers. |
| Image | Immutable image digest, not a mutable tag. |
| Execution | Command, seed range, shard range, timestamps, and terminal marker for each lane. |

The artifact publisher must reject mixed tuples, missing terminal markers, or
marker-less shards. It publishes only public-safe aggregate data: no private
paths, registry names, credentials, raw request payloads, or model tensors.

## Final Audit Lanes

### 1. Exact source-universe breadth

Materialize every exact Gen 3 randbat variant from the upstream source:
role, level, final move set, ability, item, and metadata. The expected
universe is the complete **1,748 exact variants**, evaluated from both seats.

This lane proves source-universe and observation-surface coverage. It does
not claim that every move is used in a battle, every interaction happens, or
every chance outcome is sampled. The final report must distinguish the exact
variant total from the number of decision boundaries exercised.

### 2. Bounded-depth legal-line audit

Advance legal, source-backed fixtures through depth 8 while checking public
observation/belief invariants at every decision boundary. The scripts must
include known multi-turn conflation triggers where reachable, including
Toxic into an already-statused target, Protect turns, Yawn, recharge, Curse,
and PP/replay-sensitive paths.

This lane targets single-mon temporal encoding and parser/order defects. It
does not substitute for party interactions or arbitrary deep stochastic play.

### 3. Curated party and switch interactions

Run the maintained 31-scenario party suite with both-seat coverage. It must
exercise the historic interaction class that 1v1 fixtures cannot observe:

- hazards and entry damage;
- Intimidate triggers and safe non-triggers;
- Baton Pass and volatile carry;
- status followed by switch-out and Natural Cure re-entry;
- Sleep Clause and a second sleep attempt;
- Trick/Knock Off item changes;
- Wish timing and landing;
- weather and party-state transitions.

The suite is targeted coverage, not a claim that all switch sequences are
exhausted. Natural Cure must remain a required regression scenario, but it is
not the only silent-transition case.

### 4. Canonical E/O/C omission differential

Run the protocol inventory and publish three canonical-signature sets:

- **E (engine):** Gen 3 reachable engine emissions. Source discovery must be
  limited to the Gen 3 simulator surface and actual randbat source universe.
  Generic all-generation move, ability, and item tables must not make a tag
  appear Gen 3 reachable.
- **O (observed):** dynamically canonicalized signatures from exact,
  bounded-depth, party, and production captures, with occurrence counts and
  provenance. A learned-v3 self-play census is included only when a genuine
  v3-trained capture exists; otherwise the final report must state that
  limitation explicitly.
- **C (consumed):** public-protocol handlers in the observation, replay, and
  belief paths. Private `request` payload dispatch is excluded from C and
  from all model-visible evidence.

The report must show and triage `O - C`, `E - O`, and `C - E`. Known
presentation-only lines such as `-anim` and `-message` require an explicit
non-model classification, not silent disappearance from the diff. Shared
post-Gen-3 tags (including Z, Tera, Mega, Primal, and Dynamax families) must
be source-evidenced and excluded from the primary Gen 3 E set.

Canonical signatures preserve strategically meaningful distinctions such as
`cant:slp`, `-activate:protect`, and `-start:futuresight`. `-sethp` is
**tag-only**: its raw HP/status payload must never create an unbounded
signature subtype.

### 5. Model-visible encoding-collision audit

Sample roughly 100,000 decision states across the final captures. Hash only
the model-visible input: categorical ids, float32-boundary numeric values,
token types, attention structure, and legal-action masks. Do not hash raw
logs, debug metadata, timestamps, player identifiers, private request data,
or hidden state.

Compare only states with the same side-relative perspective and decision kind.
For byte-identical model inputs, compare canonical public-state and public-log
fingerprints. Hydrate every surviving collision with deterministic replay
locators and exact public-field deltas. A versioned whitelist may suppress
only documented intentional abstractions, initially HP quantization, tendency
bucketing, and transition-window truncation.

Any remaining group is an omission/conflation candidate, not a clean result.
The report records its frequency, public difference, whitelist decision, and
provenance without publishing tensors or full raw captures.

### 6. Silent engine-mutation audit

Instrument the Gen 3 simulator in a bounded test-only pass and classify every
reachable state mutation as:

1. protocol-backed;
2. belief-tracked;
3. deterministically inferable from public information;
4. provably untrackable; or
5. accepted loss.

Every silent mutation must receive one of these classifications. Natural Cure
is a required check, but this lane also covers any other mutation for which a
public protocol line is absent or insufficient. A silent transition cannot be
cleared merely because the observation encoder agrees with itself.

### 7. Counterfactual harm probes

Run this lane only for the shortlist from the E/O/C, collision, and
silent-mutation lanes. Each probe constructs public-state pairs differing in
one candidate fact, verifies whether the policy input/logits are identical,
and measures the simulator's differing consequences or outcome distribution.

Each candidate receives exactly one final verdict:

- `ADD`;
- `COVERED`;
- `UNREACHABLE`; or
- `ACCEPTED-LOSS`.

`ADD` candidates block schema freeze until an owner-approved batch lands and
the affected final lanes are rerun. `ACCEPTED-LOSS` requires measured harm
evidence and rationale; it is never a default for an inconvenient finding.

## Execution Model

The audit is a persistent, idempotent cluster-owned wave, not a Codex
foreground session. Shards write atomically, emit their completion marker
only after validation, and are skipped on rerun when their marker and
provenance validate. The merged publisher writes its output atomically and
emits both a terminal marker and a stdout completion token.

Jobs send Slack only for a material candidate finding, a lane completion, or
terminal failure. Codex does not continuously poll; it uses completed markers
and job-produced notifications when prompted. New runs must use the current
locally configured Slack credential without recording its value in code,
artifacts, logs, or documentation.

## Outputs

The final wave produces:

1. Provenanced per-lane aggregates and one validated public-safe merged
   aggregate under `docs/audit_artifacts/`.
2. A complete candidate table in
   `docs/silent_noop_sweep_findings.md`, including source reachability,
   observed frequency, consumer mapping, collision and mutation evidence,
   harm-probe outcome, and verdict.
3. A final findings PR that updates the audit status, reports the exact
   coverage achieved, and names the residual limits.

## Acceptance And Stop Conditions

The end-of-cycle evaluation is complete only when all final lanes have valid,
same-identity terminal artifacts and every candidate has a verdict. A clean
result requires no unadjudicated reachable omission or conflation candidate.

The final report must explicitly retain these limits:

- the 1,748-variant breadth lane is not exhaustive battle-state coverage;
- bounded depth 8 does not exhaust long stochastic/chance histories;
- curated party scenarios cover named interaction classes rather than every
  legal team sequence; and
- a learned-policy O census is unavailable until a genuinely v3-trained
  policy exists, if such a capture is absent at execution time.

After the final report PR merges, the observation audit becomes maintenance:
only a source, parser, belief, or observation change that can alter the
covered surface reopens the relevant lane. No final job is launched before the
token-format refactor prerequisite above is satisfied.
