# Observation schema v4 — spec (the k0 feature pack)

Status: 2026-07-31, owner-directed. Successor to `pokezero.observation.v3`.
Implements Parts A and B of the k0 feature-pack plan.

**What this schema is for.** Five 10M v3 arms differ only in history budget
(`transition_token_budget` = k). Pooled hi-fi foul-play over 4.0–5.2M ordered them
**k1 > k0 > k8 > k64**, and history is also the instability axis: the k64 arm entered
runaway dilution→sharpening cycles and collapsed while k0/k1/k8 traversed the same counts
clean. The goal is **pure-Markov k0** for downstream simplification — full region trim, no
synthesized history in search worlds, the simplest observation contract — *if* an enumerated
set of current-state features captures what k1's single history row is actually providing.

The existence proof that sufficient statistics beat raw history is already in the tree: the
consecutive-stall counter (`NUMERIC_STALL_COUNTER`) encodes the double-protect decision
exactly, where raw history would spend a token row on it. V4 asks the same question of nine
more facts.

**The architectural principle this implements.** **ONE PARSER TRUTH, TWO CONSUMERS.** Every
parser-derived public fact should reach BOTH the search world and the observation. The engine
divergence ledger named the failure class from the world side — "a parser field with exactly
one consumer is a latent world gap" — and this campaign's fix wave closed those world-side
gaps (#942 stall, #958 Encore, #961 typechange, #967 Trace+toxic, #970 Truant). **V4 is the
same audit run in the opposite direction: world-seeded facts the observation never sees.**

## Census and layout

| | v3 | v4 |
|---|---|---|
| numeric | 155 | **132** |
| categorical | 51 | **41** |
| transition tokens | 64 | **0 — the region is gone** |
| tokens in the sequence | 87 | **23** |
| vocabulary rows | 1289 | **899** |

V4 is the v3 grouped layout with the pack columns **appended inside their semantic group**,
not bolted onto the end. The v3 table's own rule is that grouping follows the token encoder's
semantic surfaces rather than the chronology in which columns were introduced; a v4 appendix
would break exactly that rule for the pack it exists to carry. Consequence: v4's physical
positions diverge from v3's from `pokemon_state` onward. That is free — v4 is a new contract,
so no artifact is ever read under both layouts (see *Contract consequence* below).

The same drop set applies: the fourteen evidence-backed unreachable fields v3 removed
([dead observation fields](dead_observation_fields.md)) stay removed. No v4 column is
*rewritten* relative to the v3 writer surface, so every surviving v3 column carries the same
VALUE at a different index — asserted directly in `tests/test_observation_spec_v4.py`.

**Two live current-state columns are retired at v4 — the whole pinned Tier-2 pair:
`NUMERIC_TIER2_CB_PINNED` (writer 138) and `NUMERIC_TIER2_INVESTMENT_PINNED` (writer 139).**
Both conclusions now NARROW THE BELIEF CANDIDATE SET
(`ObservationFeatureMasks.investment_belief_narrowing`) rather than being projected onto a
reserved scalar:

- the attacker-side **Choice Band** conclusion narrows that mon to its candidate variants
  holding a Choice Band. `item` is already a candidate-set discriminator, so concluding "this
  mon holds a Choice Band" is a statement about a first-class belief field;
- the defender-side **investment** pin narrows to the variants matching its stat lattice.

Narrowing moves `NUMERIC_CANDIDATE_SET_COUNT` (5) and `NUMERIC_UNCERTAINTY` (6) — frozen legacy
positions present in *every* schema, on every opponent-mon token — plus the `possible_items` /
`possible_moves` / `possible_abilities` surfaces, and it sharpens every sampled search world.
Against that, 139 is a lossy ±1 / ±0.5 projection that carries the investment CLASS and
discards the integer and the axis, and 138 is a single bit that names the item while saying
nothing about which SETS remain — i.e. nothing about what else the mon is carrying.

The retirement is **v4-only**. v2.1 / v2.2 / v3 keep both columns intact — checkpoints trained
under those schemas have them in their input layout, and removing them would be a silent census
break for artifacts that exist. v4 is unlaunched and its censuses are EXACT-matched, so here
it is a clean census edit now and a loud schema break later.

The exclusion is surgical by necessity: `schema_v2_1` in the encoder means "carries the v2.1
blocks", and v4 inherits it for the others (PP-validity bits, sub HP), so 138 and 139 are
switched off by name rather than by turning that flag off.

The as-of-strike history twins (`NUMERIC_TT_CB_BIT` 119, `NUMERIC_TT_INVESTMENT_BIT` 120) need
no such treatment: they are `history`-group columns and the region trim below already removes
them, so under v4 the tier2 conclusions have **no** encoded column at all.

## The history region is REMOVED, not masked

V4 carries no transition history at all. This is the plan's stated end goal — "full region trim,
no synthesized history in search worlds, the simplest observation contract" — and it is a
stronger statement than a budget of zero over a region that still exists:

- the 64 transition tokens are gone from the sequence (87 → 23);
- the 34-column `history` numeric group is gone;
- the 12 turn-merged categorical columns (`CATEGORY_TM_*`) are gone, and with them the
  `tt_phase` / `tt2_*` vocabulary families (1289 → 899 rows).

Consequences that a mask could not give: every forward is ~3.8× shorter; there is no
`transition_token_budget` knob left to mis-set; and a v4 checkpoint cannot be fed synthesized
history, because there is nowhere to put it.

**TURN-MERGED and GROUPED-LAYOUT are separate axes.** v4 is v3-lineage for every current-state
writer while being not-turn-merged. Both the Python and the native encoder keep those flags
distinct (`schema_v3` / `schema_turn_merged`, `is_v3()` / `is_v4()`).

**What survives the trim.** The transition tokens are still EXTRACTED — the tendency aggregates
derive from that stream and live on real mon tokens as current state, and so do the per-mon
Tier-2 conclusions, which under v4 land in the BELIEF candidate sets rather than in a column.
Only the per-row encoding is gone. (Both pinned tier2 columns are retired at v4 for that
separate reason — see *Census and layout* — not because the region went away.)

`showdown.v4_numeric_index()` is the physical-layout authority. `NUMERIC_*` constants are
writer-semantic identifiers, never physical v4 positions.

## Part A — the pack (per-mon current state)

Every A-row is parser-derivable public information and **Markov-legal**: a function of the
current public state plus the immediately-preceding round's public record — the same window
the parser already holds.

### A1 — `NUMERIC_MUST_RECHARGE` (both sides' ACTIVE token, 0/1)

**The correction that motivated the pack.** An earlier status relay claimed mustrecharge "is a
current-state volatile in the obs." That is wrong on the side that matters:

- `mustrecharge` is **NOT in `TRACKED_VOLATILES`**, so no `volatile:mustrecharge` categorical
  can be emitted and no numeric column existed. There was **no parser tracker at all.**
- The protocol inventory classifies the `-mustrecharge` line as a **semantic alias** of the
  *following* turn's `cant:recharge` transition token. That surface is (a) history-region-only,
  i.e. invisible at k0, and (b) one decision too late — the row lands after the free turn has
  already resolved.
- **SELF side was covered by accident**: a recharging mon's request offers exactly one action,
  so the action tokens collapse to a lone legal `move:recharge`.
- **OPPONENT side: a k0 policy is blind**, at precisely the decision where it is choosing what
  to do against a mon that cannot act.
- k1 CAN infer it: the single most recent transition row carries the opponent's move identity
  plus the miss bit. This makes opponent-recharge a prime suspect for part of the k1-over-k0
  delta.

Contrast that closes the loop: the CHARGE half of the two-turn-move family IS tracked —
`solarbeam` is a tracked volatile precisely so mid-charge commitment is public state. The
recharge half was never given the same treatment.

**Parser rule (net-new tracker).** SET on `|-mustrecharge|SLOT`; CLEAR on
`|cant|SLOT|recharge` (the forced turn is spent), on faint, and on switch/drag out.

The line is a **strictly better source than the search lane's reconstruction** from the
round-indexed action record: a MISSED Hyper Beam never emits it (so gen3's "a miss does not
recharge" rule needs no special case), it names its own actor (no species-continuity anchor),
and it cannot scroll out of a rolling window (no fail-open branch). `engine_search`
`_recharging_slots` now **prefers this tracker** via the observation metadata, keeping the old
reconstruction only as a fallback for contexts that predate the pack — so the world and the
observation are one truth with two consumers. An explicit `False` from the tracker is treated
as a proof of no lock, not an absent signal.

**Self side, amended.** Because this block is published only under the v4 schemas, the SELF
lock was absent on v2.2/v3 and those recharge turns stayed unsearchable — Showdown sets
`trapped: true` on a recharge request, so the world carried no `mustrecharge` volatile and
`engine_world` refused it as `self_request_state_unsupported`. `_recharging_slots` now unions
this tracker with the request's own legal choice set, read off the UNGATED `action_candidates`
metadata (`engine_search.self_recharge_from_action_candidates`): `getMoveRequestData` collapses
the moveset to the lone `recharge` pseudo-move exactly when `mustrecharge` is held. Both inputs
are positive proofs about our own seat, so the union cannot manufacture a lock, and this pack's
gating is unchanged.

Written on BOTH sides' active tokens: the rule is side-symmetric and exact, and a one-sided
column would be the only asymmetric per-mon scalar in the layout.

### A2 — `CATEGORY_LAST_USED_MOVE` (both sides' ACTIVE token, categorical)

The largest single surface in the pack: what Encore locks, what a Choice-lock read
corroborates, and the cadence anchor a k1 row was implicitly providing. The parser already
tracked `last_used_move` for the WORLD (`local_showdown` → `engine_world`; "Encore's own
onStart READS it — a world that omits it makes Encore fail outright"). The observation had no
`NUMERIC_`/`CATEGORY_` write anywhere.

**Three states, all positive facts:**

| state | encoding |
|---|---|
| never executed a move | unwritten (padding row) |
| came in this turn | `lastmove:switch` — a DISTINCT sentinel |
| executed a move | `move:<id>`, reusing the EXISTING move family |

The switch sentinel is a **fact, not ignorance**: Encore correctly FAILS against a fresh
switch-in, and the engine models it as `LastUsedMove::Switch`. Collapsing it into the padding
state would relabel a fact as ignorance.

Reusing `move:<id>` (rather than a private `lastmove:<id>` family) shares an embedding row with
the same move on an action token; the token-type embedding supplies the context, and the
pokemon-token row's other move-ish labels are `belief:possible_move:<id>` — a different family
— so the categorical bag stays unambiguous. This is why the pack costs no per-move vocabulary
rows.

The parser's truth table (record on `|move|`, never on `|cant|`, never for a `[from]`-tagged
CALLED move) is transcribed from the same semantics the vendored engine patch obeys, so the
two consumers cannot disagree.

### A3 — `NUMERIC_TRUANT_LOAF` (both sides' ACTIVE token, 0/1)

1 = this mon loafs on its next move attempt. The parser already runs the exact gen3
free-running-toggle state machine (`truant_phase`: switch-in seed `this.turn !== 0`,
unconditional per-residual flip, post-upkeep replacement guard, Traced-Truant unknown state)
because the WORLD needs it; the observation never saw it.

**0 encodes both "no Truant holder" and "phase unknown"**, mirroring the world's `None`
fallback, which never asserts a phase it cannot prove. The ability itself is separately visible
through the ability channel, so the two zeros are distinguishable where it matters.

### A4 — `CATEGORY_TRACED_ABILITY` (ACTIVE token, `ability:<id>`)

The ability the mon is CURRENTLY borrowing via Trace, cleared on switch-out.

**Why the belief channel is the WRONG source** (and why this column exists): belief holds the
LAST ability that mon ever traced, and Trace re-fires on every switch-in. Seeding from the
belief once handed a Gardevoir `levitate` from an earlier switch-in — silently granting it
Spikes immunity. This column is the observation twin of the #967 world fix, reading the
parser's transient `traced_ability`, which is the current copy or nothing.

### A5 — `NUMERIC_LAST_DAMAGE_DEALT` / `NUMERIC_LAST_DAMAGE_TAKEN` (ACTIVE token)

Previous-round point evidence, per max-HP fraction, 0 on rounds with none.

- **DEALT** — move-attributed only: untagged `-damage` on the slot opposite the open move
  window's actor. Confusion self-hits and `[from]`-tagged chip are excluded.
- **TAKEN** — total: every `-damage` on the mon, tagged or not (move damage, residuals,
  hazards, recoil, confusion self-hit).

The pair is **deliberately not a mirror**: DEALT is move-attributed and TAKEN is total, and
both are keyed to the **MON**, so a mon that just switched in reads 0/0 even though its side
dealt and took damage last round.

Attribution is transcribed from the transitions fold's rules so the current-state pack and the
history region cannot disagree. The one attribution error this surface could make is the
confusion self-hit: a slower confused mon self-hits with an UNTAGGED `-damage` and no move line
of its own, so the `|-activate|SLOT|confusion` marker closes the move window — without that
latch the self-damage would be credited to whoever moved first.

**Point observation ONLY.** The range/stat/variant inference this evidence feeds is the belief
layer's job (Tier-2 residual lane) and is explicitly out of scope: these columns state what
happened, they do not conclude anything from it.

## Part B — hazard payoff + credit (field token)

The credit-assignment fix. Spikes pay off turns after they are laid, in nobody's visible state,
so the value head regresses on states that never contain the layers' realized payoff.

**Orientation, shared with `NUMERIC_SELF_HAZARDS`/`NUMERIC_OPP_HAZARDS`:** `SELF_*` is about
OUR OWN ground — layers on our side, damage our mons suffered, items of ours knocked off.
`OPP_*` is the opponent's ground, i.e. where OUR hazards' and OUR Knock Offs' payoff shows up.
Reading a column as "credit we earned" therefore means reading `OPP_*`, exactly as "layers we
laid" means `NUMERIC_OPP_HAZARDS`.

### B1 — `NUMERIC_{SELF,OPP}_HAZARD_CREDIT`

Cumulative `[from] Spikes` damage suffered by that side over the whole game, as a fraction of
the side's TOTAL team HP. The parser ledger accumulates each hit as a fraction of the struck
mon's own max HP; the encoder divides by six. Never reset — the point of a credit ledger is
that it accumulates.

### B2 — `NUMERIC_{SELF,OPP}_HAZARD_EXPECTED`

The forward-looking twin: `healthy GROUNDED bench count x current layer damage fraction`, on
the same team-HP denominator so B1 and B2 read as a matched (spent, remaining) pair.

Gen3 grounding rule as the ENGINE applies it (`engine_world`): Flying types and Levitate are
exempt. Spikes is the pool's only entry hazard, at 1/8, 1/6, 1/4 of max HP for 1/2/3 layers.
Grounding is evaluated from PUBLIC knowledge only and is **conservative by construction**: an
unrevealed bench mon, or one whose ability is still ambiguous, counts as grounded. The honest
failure direction for an expected-value column is to over-count, never to claim an immunity we
cannot see. Our own team is fully known, so the same code is exact on the self side.

The active mon is excluded from the bench count: it already paid, and will not pay again unless
it leaves and returns — which is precisely the future this column prices.

### B3 — opponent switch propensity: marginal already present; **conditional added**

The global (switch count, decision opportunities) pair plus the per-mon (switched-before-attack,
stayed-and-attacked, turns-active) triple, evidence-mass /64, never bare rates, are already
encoded and are unchanged. The triple is already keyed to **the mon that switched out**, which
is right.

What it lacked is the conditioning. The triple marginalises over the thing that actually drives
the behaviour: **what the mon was facing.** Switching in gen3 is almost entirely matchup-driven —
a mon stands its ground against one threat and bails from another — so "bailed 3 of 7" averages
over whatever matchups the game happened to present, and is a biased estimator of the only
quantity that matters at decision time: *will this mon bail against the mon I have out right now.*

**`NUMERIC_MON_SWITCHED_VS_ACTIVE` / `NUMERIC_MON_STAYED_VS_ACTIVE`** — two columns on **each of
the six** opponent mon tokens, conditioned on OUR CURRENT ACTIVE: the literal conditional form of
the triple's first two members. Written on all six rather than just their active so the pair
answers two questions at once — will the mon in front of me bail, and which of their mons has
historically been willing to face what I have out (i.e. what they will bring *in*).

Why this belongs in the k0 pack specifically: the marginal aggregate survives at k0 (a token
column, not a history row), but the matchup *context* of each switch is carried solely by the
transition rows. So a k0 policy sees "bailed 3 times" with no way to recover what from. And the
raw form was fully available at **k64 — the worst and least stable arm**, where the model could
not use it. That is the pack's thesis in miniature.

Chosen as (switched, stayed) rather than (switched, opportunities) for two reasons: it is the
stay-or-switch evidence exactly (a `cant` turn is an opportunity but not a stay-or-switch datum),
and both halves are accumulated at live hook points in **both** the batch and the incremental
fold, so the parity twins cannot drift — an opportunity count would have had to be reconstructed
from a turn map the incremental fold prunes.

Normalized **/8**, not /64: same principle, different range. A single (their mon x our mon) cell
is visited a handful of times per game. A cell with no history reads (0, 0) — "no history in
*this* matchup", not "no history at all" — and the marginal triple sits on the same token as the
fallback. Rides the same `opponent_tendency_stats_block` mask as the triple; it is the same
channel, conditioned.

**Known limit: sparsity.** A game visits maybe 10–25 of the 36 pairings at a few turns each, so
the cell is usually small and is empty early. The pair encoding makes that honest rather than
misleading — `(1, 0)` and `(1, 4)` stay distinguishable, which is exactly why the design doc
forbids bare rates.

### B4 — `NUMERIC_{SELF,OPP}_ITEMS_REMOVED_CREDIT`

Per-side count of held items publicly removed by the OTHER side's action
(`-enditem … [from] move: Knock Off`). Per-mon removal state is already encoded
(`NUMERIC_REVEALED_ITEM` goes to 0 while the named item bucket persists); what was missing is
the CREDIT AGGREGATE the value head needs to price a Knock Off whose payoff is spread over the
rest of the game.

Excluded on purpose: a bare `-enditem` (a berry the holder ate, White Herb) is self-consumption
and nobody's credit; `[from] move: Trick` is a SWAP whose giving half the belief layer
explicitly declines to model, so counting it as removal credit would price a trade as a theft.

**Normalized /6** (items per team), **not** the /64 evidence-mass convention the plan sketched.
Deliberate, flagged deviation: a team can lose at most six items, so /64 would pin this column
under 0.1 for its entire realistic range. /6 is the same team-fraction denominator the hazard
block uses, which is what makes all of Part B read on one scale.

**Adjudication caveat (binding, inherited from the plan).** FoulPlay's knock-off rate is NOT
automatically the target. Before any training reads this column as a deficiency signal, a
G4-style counterfactual probe (replay lost games, search the loser's seat, certify across
reseeds) must adjudicate whether self-play's lower usage is actually a deficiency.

## Already covered — cited, not duplicated (the negative controls)

| field | existing feature | note |
|---|---|---|
| consecutive stall (double-protect) | `NUMERIC_STALL_COUNTER` | the existence proof; both consumers live |
| toxic ramp | `NUMERIC_TOXIC_STAGE` | world twin in `local_showdown` |
| opponent mid-charge (Solar Beam) | `volatile:solarbeam` categorical | the charge half of A1's family |
| confusion / encore / wrap elapsed, Mean Look | `NUMERIC_CONFUSION_TURNS`, `NUMERIC_ENCORE_TURNS`, `NUMERIC_WRAP_TRAP_TURNS`, `NUMERIC_MEANLOOK_TRAP` | **k0 already sees these cadences.** The ledger's open candidates name the WORLD side, which approximates what the obs already has — world-side seeding work, not v4 columns |
| hazard CAUSE | `NUMERIC_SELF_HAZARDS`/`_OPP_HAZARDS` | layers visible; the PAYOFF was not — that is Part B |
| opponent switch tendency | `NUMERIC_STAT_OPP_SWITCH_COUNT`/`_OPPORTUNITIES` + per-mon triple | B3 |
| per-mon item removed | `belief.item_removed` → `NUMERIC_REVEALED_ITEM` | removal already distinguishable from never-revealed; B4 adds the aggregate |
| sleep / Rest wake bookkeeping | `NUMERIC_SLEEP_TURNS`/`_REST_SLEEP`/`_WAKE_KNOWN` | needs no new column; the bench-survival fix lane is world-side |
| substitute presence + initial size | `NUMERIC_SUB_HP_FRACTION` | depletion is KNOWLEDGE-LIMITED, not a gap: gen3 emits `[damage]` with no magnitude |

**Sweep residue, checked and not promoted:** rampage lock (`lockedmove`, Thrash family — no
parser field exists and pool reachability of the carriers must be checked first: "encode
reachable, not rare"); sleep `skippedTime` (deliberately belief-internal); `weather_upkeeps` and
`stall_move_pending` (transient bookkeeping, exempted by the ledger audit itself); fail/miss/KO,
sleep-clause blocks, wish clocks, gender, confusion-selfhit (all landed in the v3 writer surface).

## Contract consequence — new arms only, never mixed

Both v4 censuses differ from v3's, so a v4 checkpoint can never share a cache, an env, or a run
with a v3 one. The machinery refuses mixing at every layer, by design:

- schema mismatch raises at validation (`observation.py`), and legacy/unversioned schemas
  load-and-refuse;
- the encode-time census floors refuse narrowed hybrids (`showdown.py`);
- the spec is latched from the checkpoint (`neural_policy.observation_spec_from_model_config`)
  and the env from checkpoint provenance
  (`local_showdown.env_config_from_checkpoint_provenance`), which raises on mismatch;
- the search lane pins tables to the checkpoint contract (`mcts_eval/resolver.py`).

The parity-lineage incident is the standing reason this is a feature, not friction.

**The vocabulary is the same kind of latch.** The pack's two categorical families
(`lastmove:switch` and `ability:<id>`) are OPT-IN via
`gen3_category_vocabulary(include_feature_pack_v4=True)`, keyed off
`FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS`, for the same reason `include_turn_merged` is: extra
rows change the vocabulary SIZE and therefore every embedding table's shape. With the latch off
the enumeration is byte-for-byte what every v2.2/v3 checkpoint was trained against, which is
what makes v4 safe to land while those arms are still running. With it on, the sorted
enumeration renumbers rows after the additions — harmless, because a checkpoint stamps its own
`category_vocab` and resolves through that, never through a fresh build (the row-drift bug
`category_vocab_from_model_config` exists to prevent), and a v4 arm trains from game 0.

**v2.2 remains the fresh default.** Adding a schema never moves the default; every running arm
keeps collecting under the schema its checkpoints were trained on.

## Known follow-ups (NOT blockers for a v4 training arm)

1. **Rust leaf encoder / `mcts_eval` lane.** `mcts_eval/resolver.py`'s
   `SUPPORTED_OBSERVATION_SCHEMAS` still lists only v2.2 and v3, so a v4 checkpoint entering
   that lane is **refused loudly** rather than silently mis-encoded — the correct fail-closed
   behaviour until `rust/pokezero-search/src/encoder.rs` can express the pack columns. The
   Python-side exporter (`scripts/export_encoder_tables.py --observation-schema v4`) already
   emits v4 tables, so the Rust work is the remaining half. The foundation training + hi-fi
   foul-play lanes do not use this encoder and are unaffected.
2. **Golden-corpus regeneration at v4**, the same gate v3 carries.
3. **The §5 attribution probe** (plan Part C) is offline, needs no training, and is unchanged
   by this schema: it masks a k1 checkpoint's history and buckets the shifted decisions. Its
   §3.4 bucket table maps one-to-one onto the A-rows above, including the two negative controls
   (own-protect-chain and CB-lock inference, both already covered — a large shift in either
   means masked-forward pathology or a mis-specified bucket rule, and blocks pack conclusions
   until explained).

## Risks / honesty notes

- **Feature-pack scope creep.** A-rows are cheap individually; the discipline is that the pack
  is a hypothesis list, not a claim that all of it earns its slot. The probe scores which
  buckets are real.
- **The k0/k1 argmax tie.** At their respective argmaxes k0 and k1 are 0.454 vs 0.454 — a
  statistical tie (hi-fi n=2000 ⇒ ~±2.2pp Wilson half-width). Nobody should claim a strength
  ordering AT argmax while arguing this work; the pooled-window ordering is the finding that
  motivates it.
- **Correction propagation.** Any downstream doc or decision that inherited "k0 already sees
  Hyper Beam recharge" must be corrected against A1 above.
