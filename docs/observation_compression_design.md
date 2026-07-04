# Observation compression: opponent-signal stats replace the 4-turn window

Status: design, 2026-07-03. Motivated by the width result (512d arm leading
its cohort on trajectory: 74.4% max-damage at 208k games) and the cost
structure that blocks scaling width further. Companion to
[`wse_shaping_and_coverage_design.md`](wse_shaping_and_coverage_design.md)
(separate arm; this doc changes the *observation*, not the reward) and
[`research_synthesis.md`](research_synthesis.md) (T2/T6).

## Motivation

The encoder consumes `window_size=4` complete observation copies
(`neural_policy.py`): each turn is ~46 tokens (1 field + 6 self-mon +
6 opponent-mon + 9 action candidates + **24 recent-event tokens**), so
~3/4 of the sequence is stale board state — and the 24 event tokens per
turn already duplicate most of what the history window provides, paying
twice for temporal context. Cutting to `window_size=1` plus a compact
opponent-signal block frees roughly 4× the sequence compute. That saving
is the feasibility budget for width: the point is not a smaller model, it
is **the same FLOPs spent on a wider trunk** (512d-class and beyond),
which current evidence says is the productive axis.

The bet: within-game temporal context in this game is compressible into
sufficient-statistic-style summaries (the poker-HUD precedent — aggregate
frequency stats as the standard compression of action history), and the
capacity freed is worth more than raw history. This is a hypothesis; the
ablation section makes it falsifiable.

**Evidence update (2026-07-03 evening, active-run curves):** the 256d
history-8 arm holds a consistent slight edge over 256d-4x at matched
milestones vs max-damage — temporal structure carries signal that pure
aggregates would destroy. This kills the strong "history is deadweight"
reading but not the encoding critique: snapshot replay makes the net
recover events by diffing 46-token state copies. The revised design
(below) keeps ordered history in a form ~23× cheaper per turn.

## Transition tokens: ordered history without snapshot replay

Replace history snapshots with **two transition tokens per past turn**
(one per side), each recording what that side did and what it caused:

- actor (species/slot), action (move id, or switch → incoming species)
- damage fraction dealt to the defender
- **damage residual vs the public calc's expected median roll** — one
  scalar carrying the set-inference content of the damage calc (positive
  ⇒ Choice Band / offensive investment evidence, negative ⇒ defensive
  investment; doubles as belief-engine evidence, same accumulator as the
  exact-state CB bit)
- flags: effectiveness class, crit, miss, KO

**The flags condition the residual; they are not merely beside it.** On
a `|-crit|` turn the residual is computed against the *crit-expected*
median — gen-3 crits are 2× and ignore attack drops, defense boosts, and
screens, so the crit median must drop those modifiers or a crit through
Reflect reads as a 3–4× outlier. Unconditioned, ~6% of damage events
(1/16 base, 1/8 high-crit moves) would inject 2× outliers into the
set-inference channel as manufactured Choice Band evidence. On a miss
the residual is masked (no damage ≠ weak-set evidence); multi-hit moves
carry an explicit `n_hits` field (from `|-hitcount|`) and normalize per
hit. The crit flag exists to explain the observation, not to learn from
past luck — crit-risk respect comes from outcomes shaping the value head
(per the luck-ledger decision).

### Implementation tiers (decided 2026-07-04 — complexity containment)

The modeling risk of this section lives entirely in the expected-damage
computation; everything else is protocol-announced fact. Tiered
accordingly:

- **Tier 1 (ships with the compression work; zero modeling risk):**
  transition tokens with actor, action, raw damage fraction, crit / miss
  / KO flags, `n_hits`, side-effect category — every field read straight
  from protocol events. No residual. The E/C ablation arms run on Tier 1
  alone; raw fractions + species embeddings are already strictly more
  legible than snapshot diffs.
- **Tier 2 (separate follow-up; all the modifier complexity, bounded three
  ways):** the damage residual and the conservative CB/investment bits.
  (a) Expected damage comes from **poke-engine queries** under the
  known-stats randbats world (fixed 85/31/neutral spread + public level
  make both sides' stats deterministic) — no hand-written calculator, so
  stat stages, screens, burn, weather, Explosion's defense-halving etc.
  are the engine's problem, not ours. (b) CB inference uses a
  **fixed-power physical move whitelist** (Hidden Power, Flail/Reversal,
  Rollout-class, Magnitude, and multi-hit moves never flip the bit).
  (c) The bit is **conservative by construction**: observed must exceed
  max-explainable = max over the species' candidate abilities × 1.1
  type-boost item × max roll × all public modifiers, by a margin, on
  **two independent events** (one calc edge case cannot flip it).
  Acceptance gate: precision ≈ 1.0 against ground-truth items extracted
  from the omniscient logs of controlled foul-play games (we own the
  BattleStream; true sets are recoverable) before any training run
  consumes the bit.
- **Tier 3 (not now):** evidence-weighted belief posteriors —
  ReBeL-direction work, out of scope for this doc.

### The item universe is 13 items, mostly species-locked (verified 2026-07-04)

From the generator (`teams.ts`): seven species-locked assignments
(Farfetch'd→Stick, Latias/Latios→Soul Dew, Marowak→Thick Club,
Pikachu→Light Ball, Deoxys→White Herb, Linoone-sweeper→Silk Scarf,
Unown→CB/Twisted Spoon by set), plus a general pool of just
{Leftovers (default), Choice Band, Lum Berry, Salac/Petaya/Liechi
Berry}. No RNG trick items exist in the pool. Consequences:

- The randbat source's candidate variants **already carry items** (the
  spread-reproduction work replicated the generator's item logic), and
  variant pruning is joint — revealing Substitute prunes to variants
  whose items come along (Sub→pinch-berry, RestTalk→Leftovers
  correlations are structural, not inferred). Species locks collapse
  `possible_items` on upsert with no special-casing.
- The CB max-explanation tightens: the only 1.1× items in the format
  are themselves species-locked, so for the general population the
  non-CB item universe contains **zero damage modifiers** — exceedance
  needs to clear only candidate-ability variance and roll variance.
- **New exact-state rule family — non-proc pruning.** Three items in
  the general pool announce themselves *automatically* when their
  trigger occurs, so a trigger without the proc is deterministic
  negative evidence (PP-ledger epistemic class, not the descoped
  behavioral kind):
  1. *No-Leftovers:* ended a turn active and damaged with no itemized
     Leftovers heal ⇒ prune Leftovers variants. Full-HP turns yield no
     evidence.
  2. *No-Lum:* a successful status application not **instantly** cured
     ⇒ prune Lum variants (gen-3 Lum procs immediately, incl. on Rest's
     self-sleep; Shed Skin's end-of-turn ability-tagged cure is
     timing-distinguishable; Safeguard/Substitute blocks apply no
     status and yield no evidence).
  3. *No-pinch-berry:* HP at or below 25% after an action with no berry
     activation ⇒ prune Salac/Petaya/Liechi variants.
  Because Leftovers is the default item, the family compounds fast: one
  damaged turn removes the modal candidate, and a stuck status or a big
  hit reduces the general pool to **Choice Band by elimination** —
  making the Tier-2 damage residual confirmation rather than primary
  evidence for most mons. Knock Off/Trick and all positive activations
  emit explicit item events the engine already consumes.

**Status chip damage (toxic/burn/sand/Spikes) is attribution hygiene,
not a feature.** Chip is fully determined by public state the
observation already tracks (toxic stage, statuses, hazard layers,
weather) and carries no hidden information; past chip is summarized in
the HP bar. The pipeline rule: residuals are computed from the move's
own itemized damage event (protocol `[from]` tags separate chip from
move damage), never from turn-level HP deltas. No chip history tokens.
- side-effect category (status inflicted / hazard set / weather set /
  boost used)

Cost: 2 tokens/turn vs 46 — **16–24 turns of ordered history for less
than one snapshot**, so coverage extends to effectively the whole game
where h4/h8 snapshot windows were the ceiling. The 24 raw
`recent_public_events` tokens are subsumed and dropped (they are the
crude 3–4-turn version of this idea, currently paid *in addition to*
snapshots). Precedent: Metamon's transformer consumes explicit
(previous-action, reward) sequences rather than repeated state — the
strongest published agent in this domain uses decision history, not
snapshot history.

The compression spectrum is then: full snapshots (structure + restated
state) → transition tokens (structure only) → aggregate stats (no
structure). The stats block (below) is retained alongside transition
tokens: cross-turn aggregates spare the trunk from recomputing counts
over the token sequence, and they remain the z-descriptor vocabulary.

## Design principle: evidence mass, not rates

Every tendency feature is a **(count, opportunity) pair**, never a bare
rate. A 100% switch-out rate over one observation and over eight are
different evidence; the net can compute the ratio but cannot recover the
mass. All features are player-relative and computed from public
information only (events, revealed moves, belief candidate sets, damage
estimates) — no oracle leakage.

## v1 feature set

### Exact-state features (belief-engine/state layer — computations, not tendencies)

These are deterministic bookkeeping the engine can do perfectly; they live
beside the PP ledger, not in the stats block. All approved 2026-07-03.

- **Speed brackets from turn order.** Every observed turn order is an exact
  inequality against a known stat of ours. Two scalars per opponent mon:
  best lower / upper bound on speed observed so far. Pins randbats sets
  fast; the belief engine does not consume turn order today.
- **Sleep clause consumed** (one bit per side): once a side has put an
  opposing mon to sleep, its remaining sleep moves are dead weight.
- **Sleep turn counters** per sleeping mon, both sides, with a
  **rest-sleep flag**: Rest is exactly 2 turns (wake turn *known*), natural
  sleep is 1–4 (wake is a hazard rate). The flag distinguishes "they know
  when they wake" from "they're gambling," which changes both sides' play
  around the sleeping mon.
- **Choice Band detection — damage-calc only.** One bit per opponent mon:
  observed damage exceeded the maximum non-CB roll for that move/matchup.
  This is deterministic proof (set-level variation cannot produce +50%
  damage). Repeat-move streaks are deliberately *not* used: repetition is
  consistent with CB but proves nothing.
- **Timed field-condition counters**: weather turns remaining (normalized
  /5) + permanent bit (see Weather section below), and the same for
  Reflect / Light Screen / Safeguard / Mist (deterministic 5-turn-class
  counters in gen 3).
- **Turns-in-battle counter** for the opponent's active mon. Cheap proxy
  with two readings: a fresh counter after a faint implies the KO context;
  a long counter signals commitment to the current position.

### PP-ledger subsumption principle

The exact PP ledger (feature 1 below) doubles as a **per-move usage
counter**, and revealed-move tokens already persist set knowledge. Together
they subsume, at zero extra cost, the whole family of reveal/usage-count
features considered and rejected as separate items: phazing usage, Pursuit
counts, Counter/Mirror Coat usage, recovery-move counts, Encore/Taunt
reveals, boost-history, Endure/reversal-kit alerts, Substitute counts, and
Protect patterns. Do not add dedicated features for these; if a later probe
shows the net failing to use one, the fix is representation/capacity, not
another counter. (Boom threat likewise: candidate-set collapse already
carries "Explosion is still possible," which is the decision-relevant
part.)

### Tendency features (stats block)

1. **Exact opponent PP ledger (belief engine — exact-state class).** For
   every revealed opponent move: remaining-PP fraction, maintained
   exactly — max PP from the randbats catalog, decrement per observed
   use, ×2 when our active mon has Pressure (our own ability is perfectly
   known, so the ledger is exact, not approximate). Exposed as a numeric
   feature on the opponent move slots, mirroring our own
   `NUMERIC_MOVE_PP_FRACTION`. Motivation: pp-stall is a real strategy in
   both directions, and per the subsumption principle above this single
   ledger carries the entire usage-count feature family.
2. **Global switch tendency:** (opponent switch count, opponent decision
   opportunities).
3. **Per-opponent-mon tendency triple** on the existing 6 opponent-mon
   tokens: (switched-out-before-attacking count, stayed-and-attacked
   count, turns-active). Deliberately *not* the full 6×6
   matchup-conditional matrix — sparse deadweight at battle length.
4. **Opponent weather reveals, with source.** Per weather type:
   {opponent set it this game, source was an ability}. The ability case
   (Drizzle/Drought/Sand Stream — all present in the gen3 randbats pool)
   is a double reveal: the mon's ability is confirmed (narrowing its set
   catalog) and the weather is **permanent** in gen 3, versus exactly 5
   turns for move weather (no extension items exist in gen 3). Source is
   parsed directly from the `|-weather|...|[from] ability:` protocol tag.
   Current gap being fixed: `_update_weather` keeps only a bare
   identifier — no duration, no source — so Tyranitar sand (plan the
   whole game around it) is indistinguishable from turn-4 Sandstorm
   (about to expire). Own-side weather usage is *not* tracked: the field
   token carries active state + the turns-remaining counter above, and
   our own past usage changes no decision.
5. **Midground/prediction signal**, over turns where *we* switched:
   (my-switch-turn count, opponent-move-better-vs-incoming count
   [margin-thresholded damage-calc comparison, to exclude
   good-vs-both moves], pivot-move count). High ratio = opponent predicts
   switches / midgrounds; low = they tunnel the active mon. Noisy by
   nature; the (count, rate) form lets the net weigh confidence.
6. ~~Opponent setup usage bits~~ — removed 2026-07-03: subsumed by
   revealed-move tokens + the PP ledger (see subsumption principle).

**Descoped — per-candidate-move declined-opportunity counters.** The
"they didn't click the KO move they might have" evidence is real, but
representing it requires per-candidate-move accumulators across the whole
candidate set — expanding an already expensive feature family that is
often deadweight on the net. Revisit only if the v1 stats show value and
a later probe shows the net misreading stay-in bluffs; the principled
version (behavior-likelihood belief updates) is ReBeL-direction work, not
an observation feature.

## Encoding

One new **stats token** (or two) per observation carrying the global
pairs (items 2, 4, 5), plus per-mon fields added to the existing opponent
tokens (items 1, 3, 6). New observation spec version with checkpoint
compatibility guards per `model_versioning.md`; `window_size=1` becomes a
config choice, not a new code path (the window machinery already
supports it).

## Interplay notes (both load-bearing)

- **These features are near-constant in mirror self-play** — against a
  copy of yourself, tendency stats carry almost no signal. They become
  informative exactly when opponents vary. Ship alongside the cross-arm
  opponent pools (#487 arms as each other's population); the pools make
  the features informative and the features let the net exploit what the
  pools expose.
- **The same statistics are the self-derived z-descriptor vocabulary**
  (switch rate, weather/setup usage, midground rate) for the
  conditioning mechanism in `research_synthesis.md` stack item 3. One
  instrumentation effort, two consumers.

## Ablation plan (500k, same seed bands and yardstick as the width arms)

| arm | history | stats block | width |
|---|---|---|---|
| A (control) | 4 snapshots | no | 256d |
| A8 (running) | 8 snapshots | no | 256d |
| B | none (window=1) | yes | 256d |
| **E (primary)** | **16-turn transition tokens** | yes | 256d |
| C (payoff) | 16-turn transition tokens | yes | 512d |
| D (optional) | GRU aggregator | no | 256d |

B vs E isolates the value of *ordered* history beyond its aggregates —
the exact question the h8 > h4 edge raises. E vs A/A8 tests whether
transition tokens preserve what snapshot history provides at a fraction
of the sequence cost. C spends the savings on width, which the active
runs say is the productive axis.

Reads: foul-play at matched milestones (primary), max-damage trajectory,
ΔV/behavior probes, and wall-clock/games-per-hour (the feasibility claim
is itself a measurable). Success for the compression hypothesis: B ≥ A on
foul-play at matched milestones with the expected sequence-compute
saving. Success for the program: C > both on trajectory at equal
wall-clock. D distinguishes "engineered stats" from "any temporal
compression."

## Explicit non-goals

- No reward changes (that is #487's arm).
- No cross-battle opponent memory — all counters reset per battle
  (randbats opponents are exchangeable across battles).
- No behavior-likelihood belief updates (descoped above).
