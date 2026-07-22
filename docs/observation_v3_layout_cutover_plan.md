# Observation v3 layout cutover — plan

Status: 2026-07-22, implemented and independently reviewed in the V3 layout-cutover
PR; Rust mirroring is complete, with fresh audit/corpus artifacts still pending. Companion to
`observation_v3_spec.md` (content) and `silent_noop_sweep_plan.md` (the
input-audit program). This doc governs the one-time DESTRUCTIVE
reorganization of the v3 observation layout before the schema freeze.

## Decision (and why there is no "legacy v3")

The v3 layout is reorganized IN PLACE — one v3, no v4, no frozen
"legacy v3". Rationale: a schema id exists to protect TRAINED ARTIFACTS,
and nothing has ever trained on v3; it is an identifier with zero
checkpoints behind it. A legacy v3 would be a stillborn schema maintained
forever (dispatch, count maps, latching, tests) for a version no model
will ever use. v2/v2.1/v2.2 are preserved because real lineages latch to
them; v3-as-appendix has no such claim.

**The v4 tripwire:** the moment anything real trains on v3-as-it-stands,
this plan expires and the reorg becomes v4 by the same logic that
protects v2.2. Do not start the generation run before the cutover.

## What the reorg is (and is not)

- **Is:** removal of evidence-backed dead columns (the screens class —
  features constant across the entire training distribution because the
  gen3 randbats pool cannot produce them); logical regrouping of the
  remaining columns (state / belief / actions / history) in place of four
  generations of additive appendices; preserve the categorical vocabulary
  unchanged because its globally sorted row identities are shared with V2.2;
  the
  approved input-audit ADDs (protect counter, wish counter, and the batch
  verdicts) landing INTO the new layout so content and layout freeze
  together as one owner decision.
- **Is not:** a model-strength change from reordering. Within a token the
  feature vector feeds a linear projection — the network is
  permutation-invariant to column order. The value is dead-weight removal
  (marginal), spec coherence, encoder maintainability, and a clean
  single-table implementation target for the Rust mirror. Claims beyond
  that are not honest and should not appear in changelogs.

## Method

1. **Dead-column evidence sweep (prerequisite, runs now).** Scan the
   golden corpus plus a large production self-play sample (millions of
   decisions) for zero-variance columns; each candidate then gets a
   reachability argument from the pool/ruleset (the established pattern:
   screens absent from the pool, Freeze Clause absent from the ruleset).
   Drop list requires BOTH the empirical zero-variance evidence AND the
   reachability argument — either alone is insufficient (a rare-but-live
   column must stay; a coincidentally-quiet sample must not kill one).
2. **Layout design.** Group by semantic region; document every column
   with its source and consumer. Deliverable: the v3 column table in
   `observation_v3_spec.md`, replacing the appendix history.
3. **The permutation map is the spec artifact.** For every carried
   column, the map declares old→new position. The core test: encode
   identical states under v2.2 and new-v3; assert
   `v3[new] == v2.2[old]` for every mapped column. New columns are
   covered by their own suites; dropped columns cite their evidence.
   This REPLACES the current byte-prefix (v2.2 ⊂ v3) invariant and its
   tests — strictly stronger against transcription errors, which are the
   dominant risk of any renumbering.
4. **Single cutover PR.** Renumbering, dead-column removal, ADD landing,
   permutation-map test, and the rewrite of position-pinning tests all in
   ONE review-gated PR (independent review at the #779 bar). No
   incremental layout changes before or after — layout drift across
   multiple PRs is how transcription errors hide.
5. **Freeze immediately after the cutover merges.** Then the Rust fold
   mirror implements the final table once, and the golden corpus
   regenerates at v3 once.

## Multi-lane coordination (this is the hard part)

Several sessions are actively landing v3 CONTENT changes. Rules:

- Content changes (emission logic, semantics, tests keyed on NAMED
  constants) continue freely until the announced cutover window — they
  survive renumbering untouched.
- No lane merges layout-adjacent work (column constants, count maps,
  position-pinning tests) during the cutover window (~1 day, announced
  in `observation_v3_spec.md` ahead of time).
- All new v3 tests written from now on MUST key on named constants, not
  integer positions; position-pinning is reserved for the cutover PR's
  own map test.
- v2.2 byte-identity remains absolute and untouched throughout — its
  suites are not modified by the cutover.

## Sequencing

Dead-column sweep now (parallel with the running 50k smoke arms and the
input-audit adjudication) → audit batch decision + layout design →
cutover PR + review → merge → FREEZE → Rust mirror + corpus regen →
generation launch (settings per the v3 run plan in the deploy repo).
The cutover inserts ~2–4 agent-days before the freeze; the smoke arms'
50k reads run on a similar clock, so the critical path moves little.

## Implementation disposition

- **Numeric layout:** implemented as a private legacy writer surface plus one
  declared V3 projection map. This preserves every frozen V2.x writer and makes
  the public V3 order inspectable in one place.
- **Dead-field removal:** implemented for the 14 mechanics listed in
  `dead_observation_fields.md`: screens, Future Sight, and the Gen3-unreachable
  hail reveal pair.
- **Compatibility oracle:** implemented as both a synthetic all-columns
  projection test and a real V2.2/V3 encode test. It checks every carried column
  through the old-to-new map, with the two confusion-self-hit damage corrections
  (first and second sub-block) as the documented semantic exceptions.
- **Token axis:** V3 uses the shared 23-row fixed prefix plus 64 turn-merged
  history rows, for 87 rows total. V2.2 remains frozen at 23 + 128 = 151.
- **Still required before EOC:** fresh V3 golden corpus/audit captures and the
  final EOC run described in
  `v3_end_of_cycle_evaluation_plan.md`. No historical V3 artifact may be reused.
