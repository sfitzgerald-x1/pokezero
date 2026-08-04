"""Minimal Showdown replay normalization helpers.

This module is intentionally small: it is a testable boundary between raw
Showdown protocol seats (`p1`/`p2`) and PokeZero's player-relative model input.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import re
import warnings
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from .category_vocab import CategoryVocabulary
    from .dex import ShowdownDex
    from .transitions import OpponentMonTendency, TendencyStats, TransitionToken
    from .turn_merged import TurnMergedToken

from .actions import (
    ACTION_COUNT,
    MOVE_ACTION_COUNT,
    canonical_switch_action_map,
    is_move_action,
    is_switch_action,
)
from .belief import (
    PlayerBeliefView,
    PokemonSetSource,
    PublicBattleBeliefEngine,
    RevealedPokemonBelief,
    strip_condition_status,
)
from .dex import resolve_move_base_power, resolve_move_effect
from .observation import (
    ACTION_CANDIDATE_TOKEN_COUNT,
    DEFAULT_OBSERVATION_FEATURE_MASKS,
    FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
    FIELD_TOKEN_COUNT,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION_V2,
    OBSERVATION_SCHEMA_VERSION_V2_1,
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
    OBSERVATION_SCHEMA_VERSION_V4,
    OPPONENT_POKEMON_TOKEN_COUNT,
    OPPONENT_TENDENCY_STATS_TOKEN_COUNT,
    TRANSITION_TOKEN_COUNT,
    V3_TRANSITION_TOKEN_COUNT,
    V4_TRANSITION_TOKEN_COUNT,
    ObservationFeatureMasks,
    ObservationPerspective,
    ObservationSpec,
    PokeZeroObservationV0,
    SELF_POKEMON_TOKEN_COUNT,
    opponent_showdown_slot,
)
from .randbat import canonical_gen3_randbat_species_id

# Belief-fact columns are sized to the Gen 3 closed universe's max distinct values per species
# (measured from the randbat set universe): at most 2 abilities, 5 items, 14 possible moves. The
# values are placed positionally (sorted) into these columns — exact and collision-free.
BELIEF_ABILITY_BUCKET_COUNT = 2
BELIEF_ITEM_BUCKET_COUNT = 6
BELIEF_MOVE_BUCKET_COUNT = 16
BELIEF_FACT_BUCKET_COUNT = BELIEF_ABILITY_BUCKET_COUNT + BELIEF_ITEM_BUCKET_COUNT + BELIEF_MOVE_BUCKET_COUNT
# Fixed categorical columns (0-8), then belief-fact buckets, then active-mon volatile-status
# columns. Volatiles (confusion / leech seed / substitute / taunt / ...) are placed positionally
# like belief facts; 6 columns cover any realistic simultaneous set on one mon.
CATEGORY_FIXED_COUNT = 9
VOLATILE_BUCKET_COUNT = 6
# The replay observation specs are schema-keyed and constructed AFTER the numeric column
# constants below (see REPLAY_OBSERVATION_SPECS_BY_SCHEMA / DEFAULT_REPLAY_OBSERVATION_SPEC),
# so each census is derived from the last named column of its schema rather than a bare int.
CATEGORY_ID_BUCKETS = 1_000_000
CATEGORY_PRIMARY = 0
CATEGORY_SECONDARY = 1
CATEGORY_ROLE = 2
CATEGORY_SLOT = 3
# Raw mechanical type facts (dex-derived). For pokemon/switch tokens: the mon's two types
# (TYPE_2 padding if mono-type). For move tokens: the move's type in TYPE_1, its damage class
# (physical/special/status) in MOVE_CATEGORY. These let the type chart + effectiveness emerge
# in the embedding space rather than being hand-computed.
CATEGORY_TYPE_1 = 4
CATEGORY_TYPE_2 = 5
CATEGORY_MOVE_CATEGORY = 6
# Move-effect TYPE (move tokens): move_effect:<id> — the move's primary OR secondary effect as
# one label: a status (brn/par/frz/...), a volatile (substitute/leechseed/flinch/...), or a
# target-explicit, magnitude-enumerated stat change (lower_foe_def_sharply / raise_self_atk /
# raise_self_all / lower_self_atkdef / ...). NUMERIC_EFFECT_CHANCE carries its probability
# (1.0 = guaranteed), so the model can tell e.g. a 10% freeze from a guaranteed setup, and a
# foe-debuff from a self-drawback. NUMERIC_SELF_HP_COST carries the move's upfront HP price.
CATEGORY_MOVE_EFFECT = 7
# Move priority bracket (move tokens): move_priority:<n> for the integer priority (e.g. +1 Quick
# Attack, -3 Focus Punch). Priority is a discrete turn-order bracket — a higher bracket always
# moves first regardless of speed — so a per-bracket embedding captures it better than the scalar
# NUMERIC_PRIORITY (kept for ordinal grounding).
CATEGORY_MOVE_PRIORITY = 8
CATEGORY_BELIEF_ABILITY_OFFSET = CATEGORY_FIXED_COUNT
CATEGORY_BELIEF_ITEM_OFFSET = CATEGORY_BELIEF_ABILITY_OFFSET + BELIEF_ABILITY_BUCKET_COUNT
CATEGORY_BELIEF_MOVE_OFFSET = CATEGORY_BELIEF_ITEM_OFFSET + BELIEF_ITEM_BUCKET_COUNT
# Active-mon volatile-status columns follow the belief blocks (volatile:<name>, positional).
CATEGORY_VOLATILE_OFFSET = CATEGORY_BELIEF_MOVE_OFFSET + BELIEF_MOVE_BUCKET_COUNT
NUMERIC_HP_FRACTION = 0
NUMERIC_ACTIVE = 1
NUMERIC_LEGAL = 2
NUMERIC_PRESENT = 3
NUMERIC_REVEALED_MOVE_COUNT = 4
NUMERIC_CANDIDATE_SET_COUNT = 5
NUMERIC_UNCERTAINTY = 6
NUMERIC_POSSIBLE_ABILITY_COUNT = 7
NUMERIC_POSSIBLE_ITEM_COUNT = 8
NUMERIC_POSSIBLE_MOVE_COUNT = 9
NUMERIC_REVEALED_ABILITY = 10
NUMERIC_REVEALED_ITEM = 11
# Raw move mechanics (dex-derived), populated on move action tokens.
NUMERIC_BASE_POWER = 12  # normalized base power (bp/200, clamped)
NUMERIC_PRIORITY = 13  # move priority bracket (normalized)
NUMERIC_ACCURACY = 14  # accuracy [0,1]; 1.0 for never-miss
# Phase 2 — dynamic decision-critical state.
NUMERIC_LEVEL = 15  # per pokemon/switch token: level/100
# Species base stats (dex-derived, public, consistent scale stat/200) on every pokemon/switch
# token. With NUMERIC_LEVEL the model can reason about damage and turn order (speed).
NUMERIC_BASE_HP = 16
NUMERIC_BASE_ATK = 17
NUMERIC_BASE_DEF = 18
NUMERIC_BASE_SPA = 19
NUMERIC_BASE_SPD = 20
NUMERIC_BASE_SPE = 21
# Field token (global), player-relative: hazard layers + screen counts.
NUMERIC_SELF_HAZARDS = 22  # self-side entry-hazard layers (e.g. spikes) / 3
NUMERIC_OPP_HAZARDS = 23
NUMERIC_SELF_SCREENS = 24  # self-side screens active (reflect/lightscreen) / 2
NUMERIC_OPP_SCREENS = 25
# Current stat-boost stages (stage/6 in [-1, 1]) on the ACTIVE mon — the setup-sweep signal.
# Populated only on the active self/opponent pokemon token (boosts reset on switch).
NUMERIC_BOOST_ATK = 26
NUMERIC_BOOST_DEF = 27
NUMERIC_BOOST_SPA = 28
NUMERIC_BOOST_SPD = 29
NUMERIC_BOOST_SPE = 30
# Weather is encoded categorically on the field token's SECONDARY slot (weather:<id>).
# Per-move dynamic/mechanical facts on move action tokens (raw, not judgments).
NUMERIC_MOVE_PP_FRACTION = 31  # remaining PP / max PP from the request (1.0 = full; low = scarce)
NUMERIC_EFFECT_CHANCE = 32  # move-effect probability [0,1]; pairs with move_effect (1.0 = guaranteed)
NUMERIC_TURN_COUNT = 33  # field token: battle turn number / 1000 (clamped) — tempo / stall signal
# Move tokens: fraction of user max HP the move spends upfront (Belly Drum 0.5, Substitute 0.25,
# Explosion 1.0) — a deterrent the model weighs against the effect.
NUMERIC_SELF_HP_COST = 34
# Field token: a pending delayed attack (Future Sight / Doom Desire) landing on each side, as
# turns-remaining / 2. SELF = incoming to the player (a hit to brace/switch around); OPP = the
# player's own outgoing attack landing on the foe.
NUMERIC_SELF_FUTURE_SIGHT = 35
NUMERIC_OPP_FUTURE_SIGHT = 36
# Active mon token: badly-poisoned (tox) ramp stage / 15 — the escalating 1/16, 2/16, ... damage
# (0 if not badly poisoned). Distinct from the status:tox categorical, which only marks the type.
NUMERIC_TOXIC_STAGE = 37
# Actual computed stats (stat / 714, the Gen 3 max, so nothing saturates) on every self mon +
# switch token — free, exact knowledge from the request (EVs/nature/IVs baked in), unlike the
# species BASE stats which are all the model gets for the opponent. Left padding (0) for opponent
# mons, whose actual stats are hidden. HP is the actual max HP (from the request condition).
NUMERIC_ACTUAL_HP = 38
NUMERIC_ACTUAL_ATK = 39
NUMERIC_ACTUAL_DEF = 40
NUMERIC_ACTUAL_SPA = 41
NUMERIC_ACTUAL_SPD = 42
NUMERIC_ACTUAL_SPE = 43
# ---- observation spec v2 additions (exact-state layer + opponent-tendency-stats token + transition tokens). ----
# Field token — side-level exact state. Sleep-clause bits carry LIVE semantics (corrections
# item 8): 1 while the side currently has an opposing mon asleep from its own sleep move.
NUMERIC_SELF_SLEEP_CLAUSE = 44
NUMERIC_OPP_SLEEP_CLAUSE = 45
# Weather duration: turns remaining / 5 for move weather; ability weather is permanent in gen 3
# (permanent bit set, counter pinned at 1.0 so it never reads as decaying).
NUMERIC_WEATHER_TURNS = 46
NUMERIC_WEATHER_PERMANENT = 47
# Deterministic 5-turn side-condition counters (turns remaining / 5), per side.
NUMERIC_SELF_REFLECT_TURNS = 48
NUMERIC_SELF_LIGHT_SCREEN_TURNS = 49
NUMERIC_SELF_SAFEGUARD_TURNS = 50
NUMERIC_SELF_MIST_TURNS = 51
NUMERIC_OPP_REFLECT_TURNS = 52
NUMERIC_OPP_LIGHT_SCREEN_TURNS = 53
NUMERIC_OPP_SAFEGUARD_TURNS = 54
NUMERIC_OPP_MIST_TURNS = 55
# Pending Wish per side (latent state no rule can reconstruct — design doc pending-effect rule).
NUMERIC_SELF_WISH_PENDING = 56
NUMERIC_OPP_WISH_PENDING = 57
# Pokemon tokens — per-mon exact state (both sides where known). Sleep counter /5; wake-known
# distinguishes "they know when they wake" (Rest, Early Bird resolved per corrections item 8)
# from natural sleep's hazard rate. Turns-active is the current stint (reset on entry), /64.
NUMERIC_SLEEP_TURNS = 58
NUMERIC_REST_SLEEP = 59
NUMERIC_WAKE_KNOWN = 60
NUMERIC_TURNS_ACTIVE = 61
# Trapper-alive: this mon has a revealed trap ability (Shadow Tag / Arena Trap / Magnet Pull),
# is not fainted, and is benched — the persistent switch-threat flag from the WS-1 A corrective.
NUMERIC_TRAPPER_ALIVE = 62
# Opponent tokens — per-mon tendency triple (design doc stats item 3), evidence-mass counts /64.
NUMERIC_MON_SWITCHED_BEFORE_ATTACK = 63
NUMERIC_MON_STAYED_AND_ATTACKED = 64
NUMERIC_MON_TURNS_ACTIVE_TOTAL = 65
# Opponent tokens — computed expected stats (design doc exact-state; corrections item 1): the
# fixed four (def/spa/spd/spe) are exact from species+level+85 EV/31 IV/neutral; HP and Atk are
# variant-conditioned — the 85/31 baseline plus a [low, high] bound pair over candidate variants
# (Atk-zeroing on no-physical sets, HP-EV trim on Sub+Flail/Reversal / Sub+pinch-berry /
# Belly Drum sets) when a set source is attached, else baseline. All / 714 like actual stats.
NUMERIC_EXPECTED_HP = 66
NUMERIC_EXPECTED_HP_LOW = 67
NUMERIC_EXPECTED_HP_HIGH = 68
NUMERIC_EXPECTED_ATK = 69
NUMERIC_EXPECTED_ATK_LOW = 70
NUMERIC_EXPECTED_ATK_HIGH = 71
NUMERIC_EXPECTED_DEF = 72
NUMERIC_EXPECTED_SPA = 73
NUMERIC_EXPECTED_SPD = 74
NUMERIC_EXPECTED_SPE = 75
# Opponent tokens — exact PP ledger (design doc stats item 1): remaining-PP fraction per
# REVEALED move, positionally aligned with the belief-move bucket columns (same sorted order as
# CATEGORY_BELIEF_MOVE_OFFSET..+16). Max PP is the randbat catalog rule (3 PP Ups: floor(pp*8/5))
# from the dex; Pressure ×2 / Sleep-Talk-charges-caller / Transform scoping are already applied
# engine-side in move_uses. Unrevealed columns stay 0.0 (no knowledge claimed).
NUMERIC_OPP_MOVE_PP_OFFSET = 76  # ..91 (BELIEF_MOVE_BUCKET_COUNT columns)
# Stats token — global tendency (count, opportunity) pairs, evidence mass /64, never bare rates.
NUMERIC_STAT_OPP_SWITCH_COUNT = 92
NUMERIC_STAT_OPP_DECISION_OPPORTUNITIES = 93
NUMERIC_STAT_BLOCKED_ON_OUR_ATTACK = 94
NUMERIC_STAT_PURSUIT_INTERCEPT_PREDICT = 95
NUMERIC_STAT_MY_SWITCH_TURNS = 96
# Opponent weather reveals: per weather in _WEATHER_REVEAL_ORDER, a (set-this-game bit,
# source-was-ability bit) pair — ability weather is a double reveal + permanent (item 4).
NUMERIC_STAT_WEATHER_REVEAL_OFFSET = 97  # ..104 (4 weathers x 2)
# Transition tokens (corrections item 9 canonical schema; categoricals share the fixed columns).
NUMERIC_TT_DAMAGE_FRACTION = 105
NUMERIC_TT_N_HITS = 106  # /5 (gen 3 multi-hit max)
NUMERIC_TT_CALLED = 107  # Sleep Talk execution bit
NUMERIC_TT_TRANSFORMED = 108
NUMERIC_TT_CRIT = 109
NUMERIC_TT_MISS = 110
NUMERIC_TT_KO = 111
NUMERIC_TT_PURSUIT_INTERCEPT = 112
# Context trio numerics (weather is categorical on CATEGORY_MOVE_EFFECT).
NUMERIC_TT_OWN_SPIKES = 113  # /3
NUMERIC_TT_OPP_SPIKES = 114  # /3
# Positional pair (corrections item 11): absolute turn /1000 (matches NUMERIC_TURN_COUNT) +
# turns-ago /64 (the token-budget turn scale), both clamped.
NUMERIC_TT_ABS_TURN = 115
NUMERIC_TT_TURNS_AGO = 116
# Tier-2 slots (corrections item 9 reserves FOUR: residual scalar + validity bit, CB bit,
# investment bit — same spec version, no second break). Populated ONLY for tokens whose
# Tier-2 fields were filled by ``pokezero.tier2`` (``infer_tier2`` / ``apply_residuals`` /
# the live tracker) behind the #505 precision gate, all under the ONE
# ``ObservationFeatureMasks.tier2_residuals`` switch (one tier2 channel, one provenance
# story); tokens from the plain extraction path carry none, so all four stay 0.0 there.
NUMERIC_TT_RESIDUAL = 117
NUMERIC_TT_RESIDUAL_VALID = 118
# The two-strike Choice Band conclusion for the ACTING mon, as of this strike (monotone
# within a battle: once concluded, every later assessed strike token of that mon carries
# it). Set on opponent move tokens only — the same rows the residual channel annotates.
NUMERIC_TT_CB_BIT = 119
# Defender-side investment conclusion code for the STRUCK opponent mon, as of the
# strike (monotone within a battle; the mirror of the CB bit, set on assessed OWN move
# tokens only). This is the v2.1 window's batch-2 population of the former H3 reserve:
# written by ``pokezero.investment`` behind its precision gate
# (runs/investment-gate-2026-07-04) under masks.tier2_residuals AND the SEPARATE
# masks.tier2_investment switch (default False — checkpoints trained post-#505 but
# pre-investment latched residuals live over a constant-zero investment column, so the
# channels need independent provenance masks). Codes: +/-1 HP investment full/trimmed,
# +/-0.5 defensive stat full/reduced; 0 = no damage-evidence conclusion. The column
# number predates the v2.1 split (it sits below the v2 census end), but the WRITE is
# v2.1-schema-gated on top of the double mask: the legacy v2 encode path never
# populates it, even under a hand-crafted v2-schema config carrying the mask.
NUMERIC_TT_INVESTMENT_BIT = 120
# The v2 numeric census ends here.
_V2_NUMERIC_FEATURE_COUNT = NUMERIC_TT_INVESTMENT_BIT + 1

# ---- observation spec v2.1 additions (checkpoint-driven; written ONLY under a v2.1 spec —
# the v2 encode path never touches columns >= _V2_NUMERIC_FEATURE_COUNT, keeping v2-mode
# encodes byte-identical to the pre-v2.1 encoder). ----
# Opponent tokens — per-bucket REVEALED-move validity bits, positionally aligned with the
# PP-fraction columns (NUMERIC_OPP_MOVE_PP_OFFSET) and the belief-move categorical buckets:
# bit k = 1 iff bucket k's move is protocol-revealed, REGARDLESS of remaining PP. This closes
# the v2 revealed-at-0-PP collision (a revealed move ledgered to exactly 0 PP encoded 0.0,
# indistinguishable from an unrevealed bucket) and doubles as the explicit confirmed-move
# flag per bucket.
NUMERIC_OPP_MOVE_PP_VALID_OFFSET = 121  # ..136 (BELIEF_MOVE_BUCKET_COUNT columns)
# Pokemon tokens — the ACTIVE mon's substitute HP fraction while the volatile is up.
# ENGINE-VERIFIED (vendored pokemon-showdown, data/moves.ts substitute condition + the
# gen5/gen4 mod overrides gen3 inherits): sub HP = floor(maxhp/4) at creation
# (condition onStart, `effectState.hp = Math.floor(target.maxhp / 4)`), but chip against the
# sub is NOT protocol-derivable — a surviving hit emits only
# `|-activate|<target>|Substitute|[damage]` with no magnitude (the corpus confirms the bare
# gen3 form), and the break emits `|-end|<target>|Substitute`. The only magnitude leak is
# drain-vs-sub (attacker heal = ceil(damage/2), corrections item 3), which is Tier-2
# residual territory, not exact-state bookkeeping. So per the hard-rule asymmetry this
# column carries presence + the KNOWN INITIAL fraction: floor(maxhp/4)/maxhp exact for the
# self side (max HP from the request), the 0.25 baseline for the opponent (max HP hidden;
# floor error < 1%). 0.0 while no sub is up. Exact chip tracking can upgrade the value
# in-place later without a spec break (same column, tighter semantics). KNOWN LIMIT
# (#512 review note): a Baton-Passed substitute reads 0.0 after the pass — the parser's
# volatile tracker conservatively resets on every switch-in (pre-existing behavior,
# shared with the categorical volatile:substitute column), so the passed sub disappears
# from BOTH surfaces together; fixing that is a volatile-tracker change, not a column one.
NUMERIC_SUB_HP_FRACTION = NUMERIC_OPP_MOVE_PP_VALID_OFFSET + BELIEF_MOVE_BUCKET_COUNT  # 137
# Opponent tokens — per-mon PERSISTENT Tier-2 conclusions (design ruling: persistent
# conclusions belong on the OPP-MON token surface, the current-state belief channel, not
# only as as-of-strike history bits). Two surfaces now carry the CB conclusion:
#   - NUMERIC_TT_CB_BIT on move transition tokens: the as-of-strike HISTORY record (kept
#     as-is so the ordered stream stays self-describing under K-truncation);
#   - NUMERIC_TIER2_CB_PINNED here: the AUTHORITATIVE current-state form — 1.0 while the
#     tier2 two-strike + non-KO Choice Band conclusion stands for this mon, persistent
#     across switches (a per-mon fact, not a per-strike one).
# The value is derived at encode time from the tier2-annotated transition-token stream
# (any assessed strike token of this mon carrying cb_bit — exactly equivalent to
# Tier2LiveTracker.cb_bits / infer_tier2's per-mon cb_bits, since both sources express a
# conclusion solely by stamping the monotone as-of-strike bit onto the concluding strike
# and every later assessed strike), so the same ``masks.tier2_residuals`` gate + the
# belief-source double-gate govern it: pipelines that never ran the Tier-2 inference
# carry unannotated tokens and the column stays 0.0.
# LAYER SEPARATION, SUPERSEDED: this column was introduced under an invariant that Tier-2
# conclusions never mutate the belief engine's Tier-1 candidate sets. That invariant now holds
# only with ``masks.investment_belief_narrowing`` OFF (its default). With it on, the CB
# conclusion NARROWS the mon's candidate variants to those holding a Choice Band — the owner's
# ruling being that "this mon holds a Choice Band" is a statement about a first-class belief
# field, since ``item`` is already a candidate-set discriminator, and belongs there rather than
# in a reserved bit. The narrowing is monotone and refusal-asymmetric (see
# PublicBattleBeliefEngine.narrow_candidate_variants), never a widening.
# RETIRED AT V4 (and only at v4), for exactly that reason — see
# _V4_DROPPED_CURRENT_STATE_NUMERIC_INDICES. v2.1/v2.2/v3 keep it: their checkpoints have it
# in their input layout.
NUMERIC_TIER2_CB_PINNED = NUMERIC_SUB_HP_FRACTION + 1  # 138
# The per-mon twin of NUMERIC_TT_INVESTMENT_BIT — the AUTHORITATIVE current-state form of
# the defender-side investment conclusion (the CB_PINNED derivation mirrored to the
# defender): the code of the LAST tier2_investment-annotated own strike against this mon,
# switch-persistent, derived from the FULL untruncated token stream (robust to the
# K-budget truncation the tt-row history record is subject to). Same codes as the tt
# column (+/-1 HP full/trimmed, +/-0.5 defense full/reduced, 0 = no conclusion); gated by
# masks.tier2_residuals AND masks.tier2_investment (default OFF — see the tt column's
# provenance note) on top of the v2.1 schema this column only exists under.
NUMERIC_TIER2_INVESTMENT_PINNED = NUMERIC_TIER2_CB_PINNED + 1  # 139
# The v2.1 numeric census ends here.
_V2_1_NUMERIC_FEATURE_COUNT = NUMERIC_TIER2_INVESTMENT_PINNED + 1

_CATEGORICAL_FEATURE_COUNT = CATEGORY_FIXED_COUNT + BELIEF_FACT_BUCKET_COUNT + VOLATILE_BUCKET_COUNT
# Schema-keyed replay observation specs: BOTH schemas stay first-class encode modes during
# the dual-schema window. Which one an env/harness uses resolves from the loaded checkpoint's
# model_config (neural_policy.observation_spec_from_model_config through the
# env_config_from_checkpoint_provenance latch); DEFAULT_REPLAY_OBSERVATION_SPEC is only the
# checkpoint-free default (fresh trains, fresh encodes) and tracks the CURRENT schema.
V2_REPLAY_OBSERVATION_SPEC = ObservationSpec(
    categorical_feature_count=_CATEGORICAL_FEATURE_COUNT,
    numeric_feature_count=_V2_NUMERIC_FEATURE_COUNT,
    schema_version=OBSERVATION_SCHEMA_VERSION_V2,
)
V2_1_REPLAY_OBSERVATION_SPEC = ObservationSpec(
    categorical_feature_count=_CATEGORICAL_FEATURE_COUNT,
    numeric_feature_count=_V2_1_NUMERIC_FEATURE_COUNT,
    schema_version=OBSERVATION_SCHEMA_VERSION_V2_1,
)
# ---- observation spec v2.2 (checkpoint-driven; TURN-MERGED transition tokens). --------
# The transition block carries pokezero.turn_merged.TurnMergedToken rows: one per
# turn/lead/replacement phase, two ordered sub-blocks (first mover / second mover). The
# FIRST sub-block rides the existing per-action columns (PRIMARY=actor species,
# SECONDARY=action label, ROLE=transition:<role>, TYPE_1/TYPE_2/MOVE_CATEGORY=outcome/
# effectiveness/side-effect on move kinds, MOVE_PRIORITY=defender identity exactly as
# v2.1 uses it on per-action rows) and the existing NUMERIC_TT_* slots. Merged-mode
# re-purposing on transition rows: the SLOT column carries tt_phase:<phase>; the
# per-action tt_kind moves to an appended column. The whole SECOND sub-block + the
# chain-collapse fields live on appended columns. Categorical columns embed as an
# unordered bag per row, so second-mover labels use tt2_-prefixed vocabulary families
# (randbat_vocab.gen3_category_vocabulary(include_turn_merged=True)) to stay bound to
# their sub-block — the same precedent as v2.1's actor/defender sharing the species:
# family on per-action rows.
TURN_MERGED_CATEGORICAL_BASE = _CATEGORICAL_FEATURE_COUNT
CATEGORY_TM_FIRST_KIND = TURN_MERGED_CATEGORICAL_BASE + 0  # tt_kind:* (SLOT now holds the phase)
CATEGORY_TM_FIRST_CANT = TURN_MERGED_CATEGORICAL_BASE + 1  # cant:<reason> (RestTalk collapse)
CATEGORY_TM_FIRST_BP = TURN_MERGED_CATEGORICAL_BASE + 2  # species:<name> (Baton Pass follow-up)
CATEGORY_TM_SECOND_KIND = TURN_MERGED_CATEGORICAL_BASE + 3  # tt2_kind:* | tt2_status:negated/absent
CATEGORY_TM_SECOND_SPECIES = TURN_MERGED_CATEGORICAL_BASE + 4  # tt2_species:<actor>
CATEGORY_TM_SECOND_ACTION = TURN_MERGED_CATEGORICAL_BASE + 5  # tt2_move:/tt2_species:/tt2_cant:
CATEGORY_TM_SECOND_DEFENDER = TURN_MERGED_CATEGORICAL_BASE + 6  # tt2_species:<defender>
CATEGORY_TM_SECOND_OUTCOME = TURN_MERGED_CATEGORICAL_BASE + 7
CATEGORY_TM_SECOND_EFFECTIVENESS = TURN_MERGED_CATEGORICAL_BASE + 8
CATEGORY_TM_SECOND_SIDE_EFFECT = TURN_MERGED_CATEGORICAL_BASE + 9
CATEGORY_TM_SECOND_CANT = TURN_MERGED_CATEGORICAL_BASE + 10  # tt2_cant:<reason> (collapse)
CATEGORY_TM_SECOND_BP = TURN_MERGED_CATEGORICAL_BASE + 11  # tt2_species:<name> (collapse)
TURN_MERGED_CATEGORICAL_EXTRA = 12
_V2_2_CATEGORICAL_FEATURE_COUNT = TURN_MERGED_CATEGORICAL_BASE + TURN_MERGED_CATEGORICAL_EXTRA

# Second sub-block numerics, appended after the v2.1 census. NUMERIC_TM2_PRESENT is the
# second-half-is-an-executed-action bit (negated/absent rows keep 0.0 and are
# distinguished categorically via tt2_status). The first sub-block reuses NUMERIC_TT_*.
TURN_MERGED_NUMERIC_BASE = _V2_1_NUMERIC_FEATURE_COUNT
NUMERIC_TM2_DAMAGE_FRACTION = TURN_MERGED_NUMERIC_BASE + 0
NUMERIC_TM2_N_HITS = TURN_MERGED_NUMERIC_BASE + 1
NUMERIC_TM2_CALLED = TURN_MERGED_NUMERIC_BASE + 2
NUMERIC_TM2_TRANSFORMED = TURN_MERGED_NUMERIC_BASE + 3
NUMERIC_TM2_CRIT = TURN_MERGED_NUMERIC_BASE + 4
NUMERIC_TM2_MISS = TURN_MERGED_NUMERIC_BASE + 5
NUMERIC_TM2_KO = TURN_MERGED_NUMERIC_BASE + 6
NUMERIC_TM2_PURSUIT_INTERCEPT = TURN_MERGED_NUMERIC_BASE + 7
NUMERIC_TM2_RESIDUAL = TURN_MERGED_NUMERIC_BASE + 8
NUMERIC_TM2_RESIDUAL_VALID = TURN_MERGED_NUMERIC_BASE + 9
NUMERIC_TM2_CB_BIT = TURN_MERGED_NUMERIC_BASE + 10
NUMERIC_TM2_PRESENT = TURN_MERGED_NUMERIC_BASE + 11
# Second-sub-block twin of NUMERIC_TT_INVESTMENT_BIT (#513): the as-of-strike
# defender-side investment code when the second mover's strike carried one. Same
# double mask (tier2_residuals AND tier2_investment); v2.2-only column.
NUMERIC_TM2_INVESTMENT = TURN_MERGED_NUMERIC_BASE + 12
# SELF_HP_COST (v2.2-only pair, mirroring the first/second block layout): fraction of
# the ACTOR'S max HP lost to its OWN declared action within that action's chunk —
# recoil family, crash on miss, Substitute/Belly Drum cost, Ghost Curse, Pain Split
# down-side, and self-faint moves (= the actor's entire remaining fraction at strike).
# Source classification + rationale: transitions._SELF_COST_FROM_TAGS. The v2/v2.1
# encodes never touch these columns (they sit above the v2.1 census), keeping both
# legacy modes byte-frozen.
NUMERIC_TT_SELF_HP_COST = TURN_MERGED_NUMERIC_BASE + 13
NUMERIC_TM2_SELF_HP_COST = TURN_MERGED_NUMERIC_BASE + 14
TURN_MERGED_NUMERIC_EXTRA = 15
_V2_2_NUMERIC_FEATURE_COUNT = TURN_MERGED_NUMERIC_BASE + TURN_MERGED_NUMERIC_EXTRA

# ---- pre-cutover v3 writer surface ----------------------------------------------------------
#
# V3 was originally an append-only extension of v2.2. The in-place layout cutover keeps this
# surface private to the encoder: legacy schemas still write and emit these positions exactly,
# while v3 writes this complete internal surface and projects it through the declarative layout
# table below. Keeping one legacy writer surface avoids threading schema-specific offsets through
# every parser and token encoder, while the projection gives v3 a single grouped public layout.
V3_LEGACY_NUMERIC_BASE = _V2_2_NUMERIC_FEATURE_COUNT
# Change 1 — the ``-fail`` transition event, mirroring the miss bit's emission convention
# exactly (numeric 0/1 on the action transition row, one column per turn-merged sub-block,
# laid out as an adjacent first/second pair like the v2.2 SELF_HP_COST twins). Window-scoped
# (no side condition — the engine's ``-fail`` argument slot is effect-dependent); with the
# miss bit a silent no-op disambiguates: miss = accuracy miss, fail = move failed, neither =
# genuinely event-less resolution.
NUMERIC_TT_FAIL = V3_LEGACY_NUMERIC_BASE + 0
NUMERIC_TM2_FAIL = V3_LEGACY_NUMERIC_BASE + 1
# Change 2 — public sleep-clause block bits on the FIELD token (predictive current-state,
# SEPARATE from the change-1 history marker by owner decision — conflating them would make
# the fail marker wrong for most fails). BLOCKS_SELF: an opposing mon is currently asleep
# from a sleep OUR side induced, so our sleep-inducing moves will fail; BLOCKS_OPP is the
# symmetric bit (feeds the opponent-action head). Derived ONLY from public protocol lines
# (the _ReplayParser induced-sleep tracker — attribution rule: a ``-status … slp`` line
# without the ``[from] move: Rest`` tag is opponent-induced; cleared on ``-curestatus``/
# faint, NOT on switch-out), unlike the belief-engine-fed v2 bits at columns 44/45 which
# ride the checkpoint-latched exact_state mask. Gen3 Standard has no Freeze Clause Mod, so
# there is deliberately no freeze twin (it would be a dead column).
NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF = V3_LEGACY_NUMERIC_BASE + 2
NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP = V3_LEGACY_NUMERIC_BASE + 3
# Change 3 — the consecutive-stall counter on each side's ACTIVE pokemon token (predictive
# current-state, written like NUMERIC_TOXIC_STAGE — a per-slot public scalar on the active mon,
# NOT the field token). One per-side counter = consecutive SUCCESSFUL stall-move uses
# (Protect/Detect/Endure, which gen3 shares through a single ``stall`` volatile; engine ground
# truth data/conditions.ts:439-462, where a failed stall deletes the volatile). Incremented on
# the success-only ``-singleturn`` tag and reset on a failed stall / non-stall move / cant /
# switch-out / faint (the five public mirrors of the engine's volatile deletion). Value is
# ``min(1.0, count / 8.0)``; derived ONLY from public protocol lines, so both players compute
# both sides' counters. Schema >= v3 only; sits above the v2.2 census so legacy modes stay frozen.
NUMERIC_STALL_COUNTER = V3_LEGACY_NUMERIC_BASE + 4
# Change 4 — Confusion turns-so-far on the CONFUSED (active) mon's token, schema >= v3 only.
# Gen3 confusion runs ``this.random(2,6)`` = {2,3,4,5} turns (no gen3 override), so the encoded
# value is ``min(1, elapsed/5)`` with CAP = 5. The confusion PRESENCE is already the
# ``volatile:confusion`` categorical (TRACKED_VOLATILES); this is the turns-so-far counter only.
# Public trace: |-start (apply) / |-activate (each confused turn) / |-end (snap-out) confusion;
# elapsed is public, remaining hidden. Sits above the v2.2 census — legacy modes stay byte-frozen.
NUMERIC_CONFUSION_TURNS = V3_LEGACY_NUMERIC_BASE + 5
# Change 5 — Encore turns-so-far on the ENCORED (active) mon's token, schema >= v3 only, the
# sibling of change 4. Gen3 Encore runs the gen3 mod override (data/mods/gen3/moves.ts
# encore.condition.durationCallback → ``this.random(3, 7)`` = {3,4,5,6} turns), so the encoded
# value is ``min(1, elapsed/6)`` with CAP = 6. The encore PRESENCE is already the
# ``volatile:encore`` categorical (TRACKED_VOLATILES); this is the turns-so-far counter only.
# Public trace: |-start|SLOT|Encore (apply) / |-end|SLOT|Encore (expiry); elapsed is public,
# remaining hidden. Sits above the v2.2 census — legacy modes stay byte-frozen.
NUMERIC_ENCORE_TURNS = V3_LEGACY_NUMERIC_BASE + 6
# Change 6 — Wrap (partial-trap) turns-so-far on the TRAPPED (active) mon's token, schema >= v3
# only, the sibling of changes 4/5. Gen3 partial-trap (Wrap) lasts 2..5 turns (max 5): the base
# ``data/conditions.ts`` partiallytrapped ``duration``/``random(5,7)`` is the MODERN value and is
# NOT overridden by the gen3 mod, but the authoritative Gen II-IV binding mechanic is 2-5 turns;
# poke-engine models the trap as a boolean volatile with a flat maxhp/16 residual and NO duration
# counter, so the elapsed comes from the protocol, not the engine (see docs/observation_v3_spec.md).
# The encoded value is ``min(1, elapsed/5)`` with CAP = 5. The trap PRESENCE is already the
# ``volatile:partiallytrapped`` categorical (TRACKED_VOLATILES); this is the turns-so-far counter
# only. Public trace: |-activate|SLOT|move: Wrap (apply; no -start) / |-end|SLOT|Wrap
# |[partiallytrapped] (expiry); elapsed is public, remaining hidden. Wrap is the pool's SOLE
# partial-trap move (Shuckle, sole carrier). Sits above the v2.2 census — legacy modes stay
# byte-frozen.
NUMERIC_WRAP_TRAP_TURNS = V3_LEGACY_NUMERIC_BASE + 7
# Change 7 — per-mon GENDER, two 0/1 bits on EVERY mon token (self and opponent), schema >= v3
# only. A STATIC public attribute (no parser counter): male -> (MALE=1, FEMALE=0), female ->
# (0, 1), genderless -> (0, 0). SELF gender comes from the request/known set (candidate.details);
# OPPONENT gender from the ``details`` string revealed on switch-in (both parsed by the existing
# ``determinization._gender_from_details``, which reads the ``, M`` / ``, F`` token — genderless
# has no letter). An OPPONENT mon is 00 before it is ever seen (it is not in the revealed team) and
# the bits appear on the switch-in reveal. Motivation: gender is public but was unencoded, while
# the search engine already conditions on it (Cute Charm infatuation; pool carriers
# Clefable/Wigglytuff/Delcatty) — a policy/search asymmetry the Layer-3 collision audit found. Sits
# above the v2.2 census — legacy modes stay byte-frozen.
NUMERIC_GENDER_MALE = V3_LEGACY_NUMERIC_BASE + 8
NUMERIC_GENDER_FEMALE = V3_LEGACY_NUMERIC_BASE + 9
# Change 8 — Mean Look / Spider Web move-trap: one 0/1 bit on the TRAPPED (active) mon's token,
# schema >= v3 only, = "switch-locked by Mean Look / Spider Web". DISTINCT from the Wrap
# partial-trap column (+7 — chip + can't switch, a DIFFERENT volatile) and from NUMERIC_TRAPPER_ALIVE
# (ability traps Shadow Tag / Arena Trap / Magnet Pull, whose shape this mirrors). Gen3 Mean Look
# (Misdreavus) / Spider Web (Ariados) run ``target.addVolatile('trapped', source, move, 'trapper')``:
# the base ``trapped`` volatile is ``noCopy`` with NO onEnd, applied via ``|-activate|SLOT|trapped``
# (no ``[of]``), and is removed SILENTLY (no protocol line) when the source's linked ``trapper``
# volatile drops. poke-engine does not model move-traps at all (its gen3 ``trapped()`` covers only
# LockedMove / partiallytrapped / trap abilities), so this is a protocol-only signal. The trap ends
# when the trapper leaves the field, the trapped mon leaves, or either faints — see the parser.
# Sits above the v2.2 census — legacy modes stay byte-frozen.
NUMERIC_MEANLOOK_TRAP = V3_LEGACY_NUMERIC_BASE + 10
# Change 9 — Wish turns-to-land, two per-SIDE numeric columns on the FIELD token (like the
# sleep-clause pair, change 2), schema >= v3 only. A Wish is a per-side ``slotCondition`` (NOT a
# per-mon volatile), so the clock lives on the field token beside the v2.2 pending bits (56/57).
# ``self_wish_turns`` / ``opp_wish_turns`` = ``min(1, remaining / 2)`` where ``remaining =
# 2 - (turn - set_turn)`` is the turns until the Wish resolves: 2 on the declaration turn, 1 on the
# landing turn, 0 otherwise — so the column reads 2/2 then 1/2 across a Wish's life and returns to 0
# the turn it lands. Re-derived from the SAME ``wish_set_turns`` tracker the v2.2 pending bit reads
# (``_update_wish``); nonzero on EXACTLY the pending turns. Per-slot (keyed on side, not mon), so it
# survives a wish-pass switch: the incoming mon reads 1/2. Gen3 Wish heals the RECIPIENT's
# baseMaxhp/2 (the engine/materialization are already gen3-correct — no engine change, and NO
# heal-amount column: the heal is ½ the recipient's max HP, already derivable from its max-HP
# columns). Public-protocol-derived (declaration + landing heal lines), so gated on the schema alone
# (NOT masks.exact_state, which darkens the belief-fed layer where the v2.2 pending BIT lives). Sits
# above the v2.2 census — legacy modes stay byte-frozen; the v2.2 pending bits 56/57 are unchanged.
NUMERIC_SELF_WISH_TURNS = V3_LEGACY_NUMERIC_BASE + 11
NUMERIC_OPP_WISH_TURNS = V3_LEGACY_NUMERIC_BASE + 12
# Change 10 — confusion self-hit marker, one 0/1 bit on the OPPONENT's
# turn-merged move sub-block, schema >= v3 only. When a
# SLOWER confused mon self-hits, the sim emits ``|-activate|SLOT|confusion`` then an UNTAGGED
# ``|-damage|SLOT|…`` with no |move|/|cant| line; the fold keeps that self-damage separate
# from the opponent's still-open move window's damage and KO attribution. V3 sets this bit =
# "the defender self-hit from confusion after this move." A single column (not a
# first/second pair like the fail bit) because the correction always rides the FIRST sub-block in
# practice — the confused mon must be SLOWER, so the opponent moved first; the write is mirrored
# onto the second sub-block defensively. Attribution is corrected schema-agnostically at
# extraction; only V3 emits the additional marker.
NUMERIC_TT_CONFUSION_SELFHIT = V3_LEGACY_NUMERIC_BASE + 13
# EXTRA counts the stall-counter column (+4, change 3), the confusion column (+5, change 4), the
# encore column (+6, change 5), the Wrap partial-trap column (+7, change 6), the two gender bits
# (+8 / +9, change 7), the Mean Look move-trap bit (+10, change 8), the two Wish turns-to-land
# bits (+11 / +12, change 9), and the confusion self-hit flag (+13, change 10). This is the
# private, pre-cutover writer surface; the public v3 width is derived from the layout map below.
V3_LEGACY_NUMERIC_EXTRA = 14
V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT = V3_LEGACY_NUMERIC_BASE + V3_LEGACY_NUMERIC_EXTRA

# ---- v4 writer surface: the k0 FEATURE PACK (docs/observation_v4_spec.md, plan Parts A/B) ------
#
# Every column here is parser-derived PUBLIC information and Markov-legal: a function of the
# current public state plus the immediately-preceding round's public record — the same window the
# parser already holds. They exist because the campaign's world-side audit ran in the opposite
# direction and found facts the SEARCH WORLD is seeded with (or that only the history region
# carries) which the observation never sees. A pure-Markov k0 policy is blind to exactly those.
#
# The columns sit above the v3 writer census, so v3 (and every legacy mode) stays byte-frozen:
# the v3 projection table never names them and the v3 encode path never writes them.
V4_NUMERIC_BASE = V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT
# Pack A1 — FORCED RECHARGE. Encoded as ``volatile:mustrecharge`` in the ACTIVE mon's
# volatile bag (both sides), NOT as a numeric column.
#
# This was the pack's highest-priority gap and the plan's §2 correction: `mustrecharge` is
# NOT in TRACKED_VOLATILES, so no `volatile:mustrecharge` categorical could ever be emitted and
# no numeric column existed. The SELF side was covered by accident — a recharging mon's request
# offers exactly one action, so the action tokens collapse to a lone legal ``move:recharge`` —
# but a k0 policy is blind on the OPPONENT side, where it decides whether this is a free turn.
# The search lane already re-derives the fact and seeds the world's ``mustrecharge`` volatile;
# this is the observation twin, from the SAME parser tracker (ONE PARSER TRUTH, TWO CONSUMERS).
#
# WHY THE BAG AND NOT A DEDICATED COLUMN. The two are the same function: categorical columns are
# SUMMED into the token embedding (``category_embedding(ids).sum(dim=3)`` with padding_idx=0), so
# a present volatile contributes one learned vector exactly as a 0/1 numeric column would
# contribute one column of the numeric projection. Position in the bag is already semantically
# irrelevant. Given that, the bag wins on cost — no new numeric column on all 87 tokens — and on
# consistency: ``solarbeam``, the CHARGE half of this very move family, is already a tracked
# volatile precisely so mid-charge commitment is public state. The recharge half now matches.
#
# The one thing a dedicated column would have bought is immunity from bucket overflow, and that
# risk is measured, not assumed: over 160 random-legal self-play games (337,314 slot
# observations) the maximum simultaneous tracked-volatile count on one mon was TWO, against six
# buckets — and 23 of the 38 tracked volatiles have no carrier in the gen3 randbat pool at all.
# Overflow is nonetheless made LOUD rather than silent; see _encode_active_volatiles.
#
# NOT added to TRACKED_VOLATILES: that set is the closed list ``_update_volatiles`` accepts from
# ``|-start|``/``|-end|`` lines, and mustrecharge never arrives that way (the sim emits a bespoke
# ``|-mustrecharge|SLOT``). It is injected into the bag at encode time from the parser tracker,
# under schema v4 only, so v3's bag is untouched. Its vocabulary row rides the feature-pack latch.
MUST_RECHARGE_VOLATILE = "mustrecharge"
# Pack A3 — TRUANT LOAF PHASE on the ACTIVE mon (both sides), 1 = loafs on its next move attempt.
#
# The parser already runs the exact gen3 free-running-toggle state machine (``truant_phase``:
# switch-in seed ``this.turn !== 0``, unconditional per-residual flip, post-upkeep replacement
# guard, and the Traced-Truant unknown state) because the WORLD needs it. The observation never
# saw it. 0 encodes BOTH "no Truant holder" and "phase unknown" — mirroring the world's None
# fallback, which never asserts a phase it cannot prove; the ability itself is separately visible
# through the ability channel, so the model can tell the two zeros apart in the cases that matter.
NUMERIC_TRUANT_LOAF = V4_NUMERIC_BASE + 0
# Pack A5 — LAST-ROUND DAMAGE point evidence, two per-mon scalars on the ACTIVE mon (both sides).
#
# DEALT: the fraction of the DEFENDER's max HP this mon removed with its own move damage in the
# previous round (untagged ``-damage`` inside its own move window; confusion self-hits and
# ``[from]``-tagged chip are excluded, matching the transitions fold's attribution rules).
# TAKEN: the fraction of THIS mon's max HP it lost to ANY source in the previous round — move
# damage, residuals, hazards, recoil, confusion self-hit. The pair is deliberately not a mirror:
# DEALT is move-attributed and TAKEN is total, and both are keyed to the MON, so a mon that just
# switched in reads 0/0 even though its side dealt and took damage last round.
#
# Point observation ONLY. The range/stat/variant inference this evidence feeds is the belief
# layer's job (Tier-2 residual lane) and is explicitly out of scope for the pack: these columns
# state what happened, they do not conclude anything from it.
NUMERIC_LAST_DAMAGE_DEALT = V4_NUMERIC_BASE + 1
NUMERIC_LAST_DAMAGE_TAKEN = V4_NUMERIC_BASE + 2
# Part B1 — ENTRY-HAZARD CREDIT ACCRUED, per side, on the FIELD token.
#
# The credit-assignment fix. Spikes pay off turns after they are laid, in nobody's visible state,
# so the value head regresses on states that never contain the layers' realized payoff. These are
# the cumulative ``[from] Spikes`` damage totals, expressed as a fraction of that side's TOTAL
# team HP (the sum of per-mon max-HP fractions / 6 — six mons, and the opponent's real max HPs
# are hidden, so an equal-share denominator is the only public normalization).
#
# ORIENTATION (the whole hazard block shares it with NUMERIC_SELF_HAZARDS/NUMERIC_OPP_HAZARDS):
# SELF_* is about OUR OWN ground — layers on our side, damage our mons suffered. OPP_* is the
# opponent's ground, i.e. the payoff OUR Spikes have realized.
NUMERIC_SELF_HAZARD_CREDIT = V4_NUMERIC_BASE + 3
NUMERIC_OPP_HAZARD_CREDIT = V4_NUMERIC_BASE + 4
# Part B2 — EXPECTED REMAINING HAZARD VALUE, per side, on the FIELD token: the forward-looking
# twin of B1. ``healthy GROUNDED bench count x current layer damage fraction``, normalized by the
# same six-mon team-HP denominator so it is directly comparable with the credit columns.
#
# Gen3 grounding rule as the ENGINE applies it (engine_world): Flying types and Levitate are
# exempt. Spikes is the pool's only entry hazard, at 1/8, 1/6, 1/4 of max HP for 1/2/3 layers.
# Grounding is evaluated from PUBLIC knowledge only — for the opponent that means revealed
# species types plus a revealed/uniquely-implied Levitate, so an unrevealed bench mon counts as
# grounded (the encoder's conservative default; it never claims an immunity it cannot see).
NUMERIC_SELF_HAZARD_EXPECTED = V4_NUMERIC_BASE + 5
NUMERIC_OPP_HAZARD_EXPECTED = V4_NUMERIC_BASE + 6
# Part B4 — ITEMS-REMOVED CREDIT, per side, on the FIELD token: how many of that side's held items
# have been publicly removed by the OTHER side's actions (``-enditem … [from] move: Knock Off``).
# Per-mon removal state is already encoded (NUMERIC_REVEALED_ITEM goes to 0 while the named item
# bucket persists); what was missing is the CREDIT AGGREGATE — the side-level ledger the value head
# needs to price a Knock Off whose payoff is spread over the rest of the game.
#
# Normalized /6 (items per team), NOT the /64 evidence-mass convention used by the tendency
# (count, opportunity) pairs. Deliberate deviation from the plan's sketch: a team can lose at most
# six items, so /64 would pin this column under 0.1 for its entire realistic range. /6 is the same
# team-fraction denominator the hazard columns above use, which is what makes the whole Part-B
# block read on one scale. Trick is excluded: it is a SWAP, and the giving half is unmodeled
# (belief marks it item_mutated with no removal), so counting it as removal credit would be wrong.
#
# ADJUDICATION CAVEAT (plan §4 item 4, binding): FoulPlay's knock-off rate is NOT automatically
# the target. Before any training reads this column as a deficiency signal, a G4-style
# counterfactual probe must adjudicate whether self-play's lower usage is actually worse.
NUMERIC_SELF_ITEMS_REMOVED_CREDIT = V4_NUMERIC_BASE + 7
NUMERIC_OPP_ITEMS_REMOVED_CREDIT = V4_NUMERIC_BASE + 8
# Part B3+ — MATCHUP-CONDITIONAL switch tendency, two columns on EVERY opponent mon token,
# conditioned on OUR CURRENT ACTIVE: the literal conditional form of the marginal triple's
# (switched-out-before-attacking, stayed-and-attacked) pair — "in THIS matchup, how often did
# it bail and how often did it stand its ground". An evidence-mass pair, never a bare rate.
#
# The existing per-mon triple is keyed to the mon that switched out — correct — but it
# marginalises over the thing that actually drives the behaviour: WHAT it was facing.
# Switching in gen3 is almost entirely matchup-driven (a mon stays in on one threat and bails
# from another), so "bailed 3 of 7" is a biased estimator of the only quantity that matters at
# decision time: will this mon bail against the mon I have out RIGHT NOW.
#
# Why it belongs in the k0 pack: the marginal aggregate survives at k0 (it is a token column,
# not a history row), but the matchup CONTEXT of each switch is carried solely by the
# transition rows — so a k0 policy sees "bailed 3 times" with no way to recover what from. And
# the raw form was fully available at k64, the worst and least stable arm; the model could not
# use it. That is this pack's thesis in miniature: encode the sufficient statistic rather than
# widening the window.
#
# Written on ALL SIX opponent tokens, not just their active, so the pair answers two questions
# at once — will the mon in front of me bail, and which of their mons has historically been
# willing to face what I have out (i.e. what they will bring IN).
#
# Chosen over (switched, opportunities) for two reasons: it is the stay-or-switch evidence
# exactly (a ``cant`` turn is an opportunity but not a stay-or-switch datum), and both halves
# are already accumulated at live hook points in BOTH the batch and the incremental fold, so
# the parity twins cannot drift — an opportunity count would have had to be reconstructed from
# a turn map the incremental fold prunes.
#
# NORMALIZED /8, not the /64 the global tendency pairs use. Same principle, different range:
# /64 suits whole-game counts that reach tens, while a single (their mon x our mon) cell is
# visited a handful of times in one game. A cell with no history reads (0, 0) and the model
# falls back to the marginal triple on the same token — the two are side by side by design.
NUMERIC_MON_SWITCHED_VS_ACTIVE = V4_NUMERIC_BASE + 9
NUMERIC_MON_STAYED_VS_ACTIVE = V4_NUMERIC_BASE + 10
# Pack A4 addendum — the CHOICE LOCK and the item's provenance, on the ACTIVE mon (both sides).
#
# ``NUMERIC_CHOICE_LOCKED``: this mon publicly holds a choice item and has executed a move since
# acquiring it, so gen3's SILENT ``choicelock`` volatile is on it and it can use nothing else
# until it switches. Pairs with pack A2, which names the move it is stuck in — exactly the way
# ``volatile:encore`` pairs with A2 to specify an Encore. The two are the same mechanic from the
# model's point of view, and the asymmetry in how they were encoded is the leading explanation
# for the observed usage gap: Encore's lock is announced (``-start|SLOT|Encore``), tracked
# (``volatile:encore``), and timed (``NUMERIC_ENCORE_TURNS``), and its usage climbs generation
# over generation; Trick's lock emitted nothing at all, and its usage sits near zero.
#
# ``NUMERIC_ITEM_SWAPPED``: the currently-held item arrived via Trick rather than being the
# mon's own. In principle this is the sign discriminator — a NATIVE Choice Band is assigned to
# all-attacks sets and makes its holder stronger, while a Tricked one is a liability we
# inflicted, same item and opposite valence.
#
# HONEST LIMIT IN THIS POOL, do not read more into the pair than it carries: a native Band is
# never announced (gen3 emits no Frisk/held-item reveal, and the only ``isChoice`` item is
# Choice Band), so ``choice_item_public`` can only be set by a Trick's ``|-item|`` line. In
# gen3 randbats today that makes the two columns COLLINEAR — locked implies swapped — and the
# separate column earns its keep only if a native reveal surface ever appears. It is kept
# distinct rather than folded in because collapsing two facts into one column is the harder
# thing to undo later, and because the pair is what the model needs if that surface arrives.
#
# Both are raw facts, not judgements: neither says the lock is good or bad. The move identity
# (A2, sharing the action token's ``move:<id>`` embedding row, which carries the move's damage
# class) is what supplies that.
NUMERIC_CHOICE_LOCKED = V4_NUMERIC_BASE + 11
NUMERIC_ITEM_SWAPPED = V4_NUMERIC_BASE + 12
V4_NUMERIC_EXTRA = 13
V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT = V4_NUMERIC_BASE + V4_NUMERIC_EXTRA
# The writer columns that exist ONLY at v4. Asked about under v3 these are absent (not dropped,
# not invalid) — the schema-keyed index resolver answers None so cross-schema audit/export code
# can iterate every named column once and let each schema report what it carries.
V4_ONLY_NUMERIC_INDICES = frozenset(
    range(V4_NUMERIC_BASE, V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)
)

# ---- v4 categorical additions (the two pack rows that are identities, not scalars) ------------
#
# Categorical columns embed as an unordered BAG per row (the model sums the per-column embeddings),
# so every label must be self-describing within its row — the same constraint that gave the v2.2
# second sub-block its ``tt2_`` prefixes.
# The pack's categorical columns sit on the PRE-v2.2 base: v4 has no transition region, so the
# twelve turn-merged second-sub-block columns (CATEGORY_TM_*) describe rows that no longer exist
# and are dropped with them.
V4_CATEGORICAL_BASE = _CATEGORICAL_FEATURE_COUNT
# Pack A2 — the ACTIVE mon's LAST EXECUTED MOVE (gen3 ``Pokemon.lastMove``), one column per side's
# active token. The single largest surface in the pack: it is what Encore locks, what a Choice-lock
# read corroborates, and the cadence anchor a k1 row was implicitly providing.
#
# THREE states, all positive facts:
#   * unwritten (padding) — this mon has never executed a move (nothing is claimed);
#   * ``lastmove:switch`` — the DISTINCT sentinel: the mon came in this turn, so ``lastMove`` is
#     genuinely null. That is a FACT, not ignorance — Encore correctly FAILS against a fresh
#     switch-in, and the engine models it as ``LastUsedMove::Switch``. Collapsing the sentinel into
#     the padding state would relabel a fact as ignorance (the parser's own note on the field);
#   * ``move:<id>`` — the executed move, reusing the EXISTING move family rather than a private
#     ``lastmove:<id>`` one, so the identity shares an embedding row with the same move on an
#     action token. The token-type embedding supplies the context, and the pokemon-token row's
#     other move-ish labels are ``belief:possible_move:<id>`` — a different family — so the bag
#     stays unambiguous.
# The parser's truth table (record on ``|move|``, never on ``|cant|``, never for a ``[from]``-tagged
# CALLED move) is transcribed from the same semantics the vendored engine patch obeys, so the two
# consumers cannot disagree.
CATEGORY_LAST_USED_MOVE = V4_CATEGORICAL_BASE + 0
# Pack A4 — the ability the ACTIVE mon is CURRENTLY borrowing via Trace, ``ability:<id>``, cleared
# on switch-out.
#
# Deliberately NOT the belief's ability channel, which is the WRONG source for this: belief holds
# the LAST ability the mon ever traced, and Trace re-fires on every switch-in, so a stale entry
# once handed a Gardevoir ``levitate`` from an earlier switch-in — silently granting it Spikes
# immunity. This column is the observation twin of the world-side fix: the parser's transient
# ``traced_ability``, which is the current copy or nothing.
CATEGORY_TRACED_ABILITY = V4_CATEGORICAL_BASE + 1
V4_CATEGORICAL_EXTRA = 2
_V4_CATEGORICAL_FEATURE_COUNT = V4_CATEGORICAL_BASE + V4_CATEGORICAL_EXTRA
# The ``lastmove:switch`` sentinel string (enumerated in randbat_vocab so it never hashes OOV).
LAST_USED_MOVE_SWITCH_SENTINEL = "lastmove:switch"
# The Baton-Pass arrival's own sentinel — see the write site for why it is not folded into the
# plain switch sentinel.
LAST_USED_MOVE_BATON_PASS_SENTINEL = "lastmove:batonpass"

# Evidence-backed unreachable mechanics from docs/dead_observation_fields.md. These columns
# remain part of every legacy schema's frozen layout but are intentionally absent from v3.
V3_DROPPED_LEGACY_NUMERIC_INDICES = frozenset(
    (
        NUMERIC_SELF_SCREENS,
        NUMERIC_OPP_SCREENS,
        NUMERIC_SELF_FUTURE_SIGHT,
        NUMERIC_OPP_FUTURE_SIGHT,
        NUMERIC_SELF_REFLECT_TURNS,
        NUMERIC_SELF_LIGHT_SCREEN_TURNS,
        NUMERIC_SELF_SAFEGUARD_TURNS,
        NUMERIC_SELF_MIST_TURNS,
        NUMERIC_OPP_REFLECT_TURNS,
        NUMERIC_OPP_LIGHT_SCREEN_TURNS,
        NUMERIC_OPP_SAFEGUARD_TURNS,
        NUMERIC_OPP_MIST_TURNS,
        NUMERIC_STAT_WEATHER_REVEAL_OFFSET + 6,
        NUMERIC_STAT_WEATHER_REVEAL_OFFSET + 7,
    )
)

# The confusion self-hit repair intentionally changes v3's move-damage semantics relative to
# frozen v2.2. It is carried to a new position but excluded from byte-equality map assertions.
V3_REWRITTEN_LEGACY_NUMERIC_INDICES = frozenset(
    (NUMERIC_TT_DAMAGE_FRACTION, NUMERIC_TM2_DAMAGE_FRACTION)
)

# One table is the v3 numeric layout specification. Grouping follows the token encoder's
# semantic surfaces rather than the chronology in which columns were introduced. Every legacy
# v2.2 position is either carried, explicitly dropped above, or explicitly rewritten above;
# the former v3 appendix entries are v3-only additions.
_V3_NUMERIC_LAYOUT_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    (
        "core",
        (
            NUMERIC_HP_FRACTION,
            NUMERIC_ACTIVE,
            NUMERIC_LEGAL,
            NUMERIC_PRESENT,
            NUMERIC_LEVEL,
            NUMERIC_TURN_COUNT,
        ),
    ),
    (
        "pokemon_state",
        (
            NUMERIC_BASE_HP,
            NUMERIC_BASE_ATK,
            NUMERIC_BASE_DEF,
            NUMERIC_BASE_SPA,
            NUMERIC_BASE_SPD,
            NUMERIC_BASE_SPE,
            NUMERIC_BOOST_ATK,
            NUMERIC_BOOST_DEF,
            NUMERIC_BOOST_SPA,
            NUMERIC_BOOST_SPD,
            NUMERIC_BOOST_SPE,
            NUMERIC_TOXIC_STAGE,
            NUMERIC_ACTUAL_HP,
            NUMERIC_ACTUAL_ATK,
            NUMERIC_ACTUAL_DEF,
            NUMERIC_ACTUAL_SPA,
            NUMERIC_ACTUAL_SPD,
            NUMERIC_ACTUAL_SPE,
            NUMERIC_SLEEP_TURNS,
            NUMERIC_REST_SLEEP,
            NUMERIC_WAKE_KNOWN,
            NUMERIC_TURNS_ACTIVE,
            NUMERIC_TRAPPER_ALIVE,
            NUMERIC_SUB_HP_FRACTION,
            NUMERIC_TIER2_CB_PINNED,
            NUMERIC_TIER2_INVESTMENT_PINNED,
            NUMERIC_STALL_COUNTER,
            NUMERIC_CONFUSION_TURNS,
            NUMERIC_ENCORE_TURNS,
            NUMERIC_WRAP_TRAP_TURNS,
            NUMERIC_GENDER_MALE,
            NUMERIC_GENDER_FEMALE,
            NUMERIC_MEANLOOK_TRAP,
        ),
    ),
    (
        "belief",
        (
            NUMERIC_REVEALED_MOVE_COUNT,
            NUMERIC_CANDIDATE_SET_COUNT,
            NUMERIC_UNCERTAINTY,
            NUMERIC_POSSIBLE_ABILITY_COUNT,
            NUMERIC_POSSIBLE_ITEM_COUNT,
            NUMERIC_POSSIBLE_MOVE_COUNT,
            NUMERIC_REVEALED_ABILITY,
            NUMERIC_REVEALED_ITEM,
            NUMERIC_MON_SWITCHED_BEFORE_ATTACK,
            NUMERIC_MON_STAYED_AND_ATTACKED,
            NUMERIC_MON_TURNS_ACTIVE_TOTAL,
            NUMERIC_EXPECTED_HP,
            NUMERIC_EXPECTED_HP_LOW,
            NUMERIC_EXPECTED_HP_HIGH,
            NUMERIC_EXPECTED_ATK,
            NUMERIC_EXPECTED_ATK_LOW,
            NUMERIC_EXPECTED_ATK_HIGH,
            NUMERIC_EXPECTED_DEF,
            NUMERIC_EXPECTED_SPA,
            NUMERIC_EXPECTED_SPD,
            NUMERIC_EXPECTED_SPE,
            *tuple(range(NUMERIC_OPP_MOVE_PP_OFFSET, NUMERIC_OPP_MOVE_PP_OFFSET + BELIEF_MOVE_BUCKET_COUNT)),
            *tuple(
                range(
                    NUMERIC_OPP_MOVE_PP_VALID_OFFSET,
                    NUMERIC_OPP_MOVE_PP_VALID_OFFSET + BELIEF_MOVE_BUCKET_COUNT,
                )
            ),
        ),
    ),
    (
        "action",
        (
            NUMERIC_BASE_POWER,
            NUMERIC_PRIORITY,
            NUMERIC_ACCURACY,
            NUMERIC_MOVE_PP_FRACTION,
            NUMERIC_EFFECT_CHANCE,
            NUMERIC_SELF_HP_COST,
        ),
    ),
    (
        "field",
        (
            NUMERIC_SELF_HAZARDS,
            NUMERIC_OPP_HAZARDS,
            NUMERIC_SELF_SLEEP_CLAUSE,
            NUMERIC_OPP_SLEEP_CLAUSE,
            NUMERIC_WEATHER_TURNS,
            NUMERIC_WEATHER_PERMANENT,
            NUMERIC_SELF_WISH_PENDING,
            NUMERIC_OPP_WISH_PENDING,
            NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF,
            NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP,
            NUMERIC_SELF_WISH_TURNS,
            NUMERIC_OPP_WISH_TURNS,
        ),
    ),
    (
        "tendency",
        (
            NUMERIC_STAT_OPP_SWITCH_COUNT,
            NUMERIC_STAT_OPP_DECISION_OPPORTUNITIES,
            NUMERIC_STAT_BLOCKED_ON_OUR_ATTACK,
            NUMERIC_STAT_PURSUIT_INTERCEPT_PREDICT,
            NUMERIC_STAT_MY_SWITCH_TURNS,
            *tuple(range(NUMERIC_STAT_WEATHER_REVEAL_OFFSET, NUMERIC_STAT_WEATHER_REVEAL_OFFSET + 6)),
        ),
    ),
    (
        "history",
        (
            *tuple(range(NUMERIC_TT_DAMAGE_FRACTION, NUMERIC_TT_INVESTMENT_BIT + 1)),
            *tuple(range(TURN_MERGED_NUMERIC_BASE, _V2_2_NUMERIC_FEATURE_COUNT)),
            NUMERIC_TT_FAIL,
            NUMERIC_TM2_FAIL,
            NUMERIC_TT_CONFUSION_SELFHIT,
        ),
    ),
)
V3_NUMERIC_LAYOUT_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = _V3_NUMERIC_LAYOUT_GROUPS
V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX = tuple(
    legacy_index for _, indices in V3_NUMERIC_LAYOUT_GROUPS for legacy_index in indices
)
V3_NUMERIC_INDEX_BY_LEGACY_INDEX = {
    legacy_index: new_index
    for new_index, legacy_index in enumerate(V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX)
}

if len(V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX) != len(set(V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX)):
    raise AssertionError("v3 numeric layout maps a legacy column more than once")
if set(V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX) | V3_DROPPED_LEGACY_NUMERIC_INDICES != set(
    range(V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)
):
    raise AssertionError("v3 numeric layout must account for every legacy v3 writer column")

_V3_NUMERIC_FEATURE_COUNT = len(V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX)


def v3_numeric_index(legacy_index: int) -> int:
    """Physical v3 index for a named legacy writer column.

    The existing ``NUMERIC_*`` constants remain the frozen writer positions shared by v2,
    v2.1, and v2.2. Consumers that inspect the reorganized v3 tensor must use this mapping
    instead of assuming those legacy offsets are physical v3 positions.
    """

    try:
        return V3_NUMERIC_INDEX_BY_LEGACY_INDEX[legacy_index]
    except KeyError as exc:
        if legacy_index in V3_DROPPED_LEGACY_NUMERIC_INDICES:
            raise ValueError(f"legacy numeric column {legacy_index} was dropped from v3") from exc
        raise ValueError(f"legacy numeric column {legacy_index} is not part of v3") from exc
_V3_CATEGORICAL_FEATURE_COUNT = _V2_2_CATEGORICAL_FEATURE_COUNT

# ---- the v4 public layout ---------------------------------------------------------------------
#
# V4 is the v3 layout with the feature-pack columns APPENDED INSIDE their semantic group, not
# bolted onto the end. The v3 table's own rule is that grouping follows the token encoder's
# semantic surfaces rather than the chronology in which columns were introduced; a v4 appendix
# would break exactly that rule for the pack it exists to carry. The consequence — v4's physical
# positions diverge from v3's from the first extended group onward — is free: v4 is a new
# contract, so no artifact is ever read under both layouts (see the "new arms only" note below).
#
# The same drop set applies: the fourteen evidence-backed unreachable fields v3 removed stay
# removed. No v4 column is dropped or rewritten relative to the v3 writer surface.
_V4_NUMERIC_LAYOUT_ADDITIONS: Mapping[str, tuple[int, ...]] = {
    "pokemon_state": (
        NUMERIC_TRUANT_LOAF,
        NUMERIC_LAST_DAMAGE_DEALT,
        NUMERIC_LAST_DAMAGE_TAKEN,
        NUMERIC_CHOICE_LOCKED,
        NUMERIC_ITEM_SWAPPED,
    ),
    # The marginal tendency triple lives in "belief" (the per-opponent-mon surface), so its
    # matchup-conditional twin sits directly beside it.
    "belief": (
        NUMERIC_MON_SWITCHED_VS_ACTIVE,
        NUMERIC_MON_STAYED_VS_ACTIVE,
    ),
    "field": (
        NUMERIC_SELF_HAZARD_CREDIT,
        NUMERIC_OPP_HAZARD_CREDIT,
        NUMERIC_SELF_HAZARD_EXPECTED,
        NUMERIC_OPP_HAZARD_EXPECTED,
        NUMERIC_SELF_ITEMS_REMOVED_CREDIT,
        NUMERIC_OPP_ITEMS_REMOVED_CREDIT,
    ),
}
# V4 drops everything v3 dropped, PLUS the entire history group: the transition region is gone
# from the contract, so every per-strike / per-turn column that only ever described a history row
# has no surface left to sit on. What those rows were carrying is either named as current state
# by the feature pack (recharge, last move, last-round damage) or deliberately let go.
#
# The two per-mon PINNED tier2 columns (138/139) are not history columns — they live on the
# opponent MON token as the authoritative CURRENT-STATE form of those conclusions — so the
# history sweep does not reach them. V4 retires them anyway, for the separate reason spelled
# out below. Their derivation is unaffected either way: the transition stream is still
# EXTRACTED under v4, only its row ENCODING is gone.
_V4_HISTORY_GROUP_INDICES = frozenset(
    index for name, indices in _V3_NUMERIC_LAYOUT_GROUPS if name == "history" for index in indices
)
# V4-ONLY drops that are not history rows. Retiring a live current-state column is a schema
# break, which is why this set is empty for every schema that has shipped an artifact and why
# it is spelled separately from the history sweep above.
#
# BOTH pinned tier2 conclusions now NARROW THE BELIEF CANDIDATE SET
# (ObservationFeatureMasks.investment_belief_narrowing) instead of being projected onto a
# reserved scalar:
#
#   - NUMERIC_TIER2_INVESTMENT_PINNED (139), the defender-side investment pin;
#   - NUMERIC_TIER2_CB_PINNED (138), the attacker-side Choice Band conclusion, whose
#     survivors are that mon's candidate variants holding a Choice Band. ``item`` is already
#     a candidate-set discriminator, so "this mon holds a Choice Band" is a statement about
#     a first-class belief field, not about a reserved bit.
#
# A narrowing moves NUMERIC_CANDIDATE_SET_COUNT (5) and NUMERIC_UNCERTAINTY (6) — frozen
# legacy positions present in EVERY schema, on every opponent-mon token — plus the
# possible_items / possible_moves / possible_abilities surfaces, and it sharpens every sampled
# search world. Against that, 139 is a lossy +/-1 / +/-0.5 class projection that discards the
# integer and the axis, and 138 is a single bit that says "Choice Band" while the belief
# surface can say WHICH sets remain and therefore what this mon's other moves are.
#
# Dropped from V4 ONLY. v2.1/v2.2/v3 keep both columns intact: checkpoints trained under those
# schemas have them in their input layout, and removing them would be a silent census break for
# artifacts that exist. V4 is unlaunched and its censuses are EXACT-matched, so here it is a
# clean census edit now and a loud schema break later.
_V4_DROPPED_CURRENT_STATE_NUMERIC_INDICES = frozenset(
    (NUMERIC_TIER2_CB_PINNED, NUMERIC_TIER2_INVESTMENT_PINNED)
)
V4_DROPPED_LEGACY_NUMERIC_INDICES = (
    V3_DROPPED_LEGACY_NUMERIC_INDICES
    | _V4_HISTORY_GROUP_INDICES
    | _V4_DROPPED_CURRENT_STATE_NUMERIC_INDICES
)
_V4_NUMERIC_LAYOUT_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = tuple(
    (
        name,
        tuple(
            index
            for index in indices
            if index not in _V4_DROPPED_CURRENT_STATE_NUMERIC_INDICES
        )
        + _V4_NUMERIC_LAYOUT_ADDITIONS.get(name, ()),
    )
    for name, indices in _V3_NUMERIC_LAYOUT_GROUPS
    if name != "history"
)
V4_NUMERIC_LAYOUT_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = _V4_NUMERIC_LAYOUT_GROUPS
V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX = tuple(
    legacy_index for _, indices in V4_NUMERIC_LAYOUT_GROUPS for legacy_index in indices
)
V4_NUMERIC_INDEX_BY_LEGACY_INDEX = {
    legacy_index: new_index
    for new_index, legacy_index in enumerate(V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX)
}

if set(_V4_NUMERIC_LAYOUT_ADDITIONS) - {name for name, _ in _V3_NUMERIC_LAYOUT_GROUPS}:
    raise AssertionError("v4 layout additions name a group the v3 layout does not define")
if len(V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX) != len(set(V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX)):
    raise AssertionError("v4 numeric layout maps a writer column more than once")
if set(V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX) | V4_DROPPED_LEGACY_NUMERIC_INDICES != set(
    range(V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)
):
    raise AssertionError("v4 numeric layout must account for every v4 writer column")

_V4_NUMERIC_FEATURE_COUNT = len(V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX)
# v4 = the v3 public surface, MINUS the history group, MINUS the retired current-state
# columns, PLUS the feature pack.
_V4_NUMERIC_FEATURE_COUNT_EXPECTED = (
    _V3_NUMERIC_FEATURE_COUNT
    - len(_V4_HISTORY_GROUP_INDICES)
    - len(_V4_DROPPED_CURRENT_STATE_NUMERIC_INDICES)
    + V4_NUMERIC_EXTRA
)
if _V4_NUMERIC_FEATURE_COUNT != _V4_NUMERIC_FEATURE_COUNT_EXPECTED:
    raise AssertionError(
        "v4 must be the v3 public surface minus history minus the retired current-state "
        f"columns plus the feature pack ({_V4_NUMERIC_FEATURE_COUNT_EXPECTED} columns), "
        f"got {_V4_NUMERIC_FEATURE_COUNT}"
    )


def v4_numeric_index(legacy_index: int) -> int:
    """Physical v4 index for a named writer column (the v3 accessor's twin).

    The ``NUMERIC_*`` constants are writer positions, not physical v4 positions: v4 projects the
    private writer row through its own grouped layout, so a consumer inspecting a v4 tensor must
    resolve through this map. ``v3_numeric_index`` and this function disagree for every column at
    or after the first v4 addition — that is the point of a new contract, and why mixing the two
    is refused everywhere rather than coerced.
    """

    try:
        return V4_NUMERIC_INDEX_BY_LEGACY_INDEX[legacy_index]
    except KeyError as exc:
        if legacy_index in V4_DROPPED_LEGACY_NUMERIC_INDICES:
            raise ValueError(f"legacy numeric column {legacy_index} was dropped from v4") from exc
        raise ValueError(f"legacy numeric column {legacy_index} is not part of v4") from exc

V2_2_REPLAY_OBSERVATION_SPEC = ObservationSpec(
    categorical_feature_count=_V2_2_CATEGORICAL_FEATURE_COUNT,
    numeric_feature_count=_V2_2_NUMERIC_FEATURE_COUNT,
    schema_version=OBSERVATION_SCHEMA_VERSION_V2_2,
)
V3_REPLAY_OBSERVATION_SPEC = ObservationSpec(
    categorical_feature_count=_V3_CATEGORICAL_FEATURE_COUNT,
    numeric_feature_count=_V3_NUMERIC_FEATURE_COUNT,
    transition_token_count=V3_TRANSITION_TOKEN_COUNT,
    schema_version=OBSERVATION_SCHEMA_VERSION_V3,
)
V4_REPLAY_OBSERVATION_SPEC = ObservationSpec(
    categorical_feature_count=_V4_CATEGORICAL_FEATURE_COUNT,
    numeric_feature_count=_V4_NUMERIC_FEATURE_COUNT,
    transition_token_count=V4_TRANSITION_TOKEN_COUNT,
    schema_version=OBSERVATION_SCHEMA_VERSION_V4,
)
REPLAY_OBSERVATION_SPECS_BY_SCHEMA: Mapping[str, ObservationSpec] = {
    OBSERVATION_SCHEMA_VERSION_V2: V2_REPLAY_OBSERVATION_SPEC,
    OBSERVATION_SCHEMA_VERSION_V2_1: V2_1_REPLAY_OBSERVATION_SPEC,
    OBSERVATION_SCHEMA_VERSION_V2_2: V2_2_REPLAY_OBSERVATION_SPEC,
    OBSERVATION_SCHEMA_VERSION_V3: V3_REPLAY_OBSERVATION_SPEC,
    OBSERVATION_SCHEMA_VERSION_V4: V4_REPLAY_OBSERVATION_SPEC,
}
DEFAULT_REPLAY_OBSERVATION_SPEC = REPLAY_OBSERVATION_SPECS_BY_SCHEMA[OBSERVATION_SCHEMA_VERSION]
# Encode-time census FLOOR per schema (#512 review, MED-LOW defense-in-depth): a spec
# narrower than its schema's floor would make ``_set_numeric``'s bounds check silently
# drop that schema's own columns — encoding an undeclared v2/v2.1 hybrid stamped with the
# wider version (e.g. a v2.1@121 spec would emit v2 numerics + v2.1 defender identity
# with no refusal anywhere, since 121 == the model's width). No shipped path builds such
# a spec (from_dict width defaults are schema-keyed; fresh trains use the full census;
# resume carries stamps), so the encoder refuses it outright. v2's floor is 119 — the
# pre-CB/investment relic family whose narrowing is deliberate ("feed the model the
# shape it was trained on") and whose dropped tail columns are all-zero under those
# checkpoints' latched masks; v2.1 has NO narrowed family (born at the full census).
# Categorical twin of the numeric floor (review MED-3): v2.2 is the FIRST schema whose
# categorical width differs (39 -> 51 + the investment column round), and _set_category
# bounds-drops silently — a v2.2-stamped spec narrowed to 39 categorical columns would
# encode the whole second-sub-block categorical surface away while staying numerically
# byte-identical to full v2.2. No narrowed relic family exists on this axis, so every
# schema floors at its own census.
_MINIMUM_CATEGORICAL_CENSUS_BY_SCHEMA: Mapping[str, int] = {
    OBSERVATION_SCHEMA_VERSION_V2: _CATEGORICAL_FEATURE_COUNT,
    OBSERVATION_SCHEMA_VERSION_V2_1: _CATEGORICAL_FEATURE_COUNT,
    OBSERVATION_SCHEMA_VERSION_V2_2: _V2_2_CATEGORICAL_FEATURE_COUNT,
    OBSERVATION_SCHEMA_VERSION_V3: _V3_CATEGORICAL_FEATURE_COUNT,
    OBSERVATION_SCHEMA_VERSION_V4: _V4_CATEGORICAL_FEATURE_COUNT,
}
_MINIMUM_NUMERIC_CENSUS_BY_SCHEMA: Mapping[str, int] = {
    OBSERVATION_SCHEMA_VERSION_V2: 119,
    OBSERVATION_SCHEMA_VERSION_V2_1: _V2_1_NUMERIC_FEATURE_COUNT,
    OBSERVATION_SCHEMA_VERSION_V2_2: _V2_2_NUMERIC_FEATURE_COUNT,
    OBSERVATION_SCHEMA_VERSION_V3: _V3_NUMERIC_FEATURE_COUNT,
    OBSERVATION_SCHEMA_VERSION_V4: _V4_NUMERIC_FEATURE_COUNT,
}


# CLI short names for the schema-selection flag (--observation-schema). v2 is
# deliberately NOT offered for fresh selection: it exists only as a checkpoint-driven
# legacy mode.
OBSERVATION_SCHEMA_CLI_CHOICES: Mapping[str, str] = {
    "v2.1": OBSERVATION_SCHEMA_VERSION_V2_1,
    "v2.2": OBSERVATION_SCHEMA_VERSION_V2_2,
    "v3": OBSERVATION_SCHEMA_VERSION_V3,
    "v4": OBSERVATION_SCHEMA_VERSION_V4,
}


def observation_schema_version_from_choice(choice: str | None) -> str | None:
    """Full schema version string for a CLI --observation-schema choice (None passes through)."""
    if choice is None:
        return None
    version = OBSERVATION_SCHEMA_CLI_CHOICES.get(str(choice))
    if version is None:
        raise ValueError(
            f"unknown observation schema choice {choice!r}; expected one of "
            f"{', '.join(sorted(OBSERVATION_SCHEMA_CLI_CHOICES))}."
        )
    return version


def observation_spec_for_schema(schema_version: str) -> ObservationSpec:
    """The canonical replay observation spec for a supported schema version.

    Loud on anything else: an unsupported schema at spec-resolution time is the same
    train/eval mismatch class the census guard bounces at tensor time, caught earlier and
    with both supported versions named.
    """
    spec = REPLAY_OBSERVATION_SPECS_BY_SCHEMA.get(schema_version)
    if spec is None:
        supported = ", ".join(repr(version) for version in REPLAY_OBSERVATION_SPECS_BY_SCHEMA)
        raise ValueError(
            f"No replay observation spec for schema {schema_version!r}; supported schemas "
            f"are {supported}. Legacy artifacts replay from their pinned tag "
            "(docs/model_versioning.md)."
        )
    return spec


def numeric_index_for_schema(schema_version: str, legacy_index: int) -> int:
    """Physical numeric index for a named historical ``NUMERIC_*`` column.

    Named numeric constants are writer-semantic identifiers, not universally physical
    positions. V2-family layouts retain those historical positions; V3 projects them into
    semantic groups and drops unreachable fields. Public-tensor consumers must resolve through
    this function rather than indexing with a ``NUMERIC_*`` constant directly.
    """

    spec = observation_spec_for_schema(schema_version)
    if schema_version == OBSERVATION_SCHEMA_VERSION_V4:
        return v4_numeric_index(legacy_index)
    if schema_version == OBSERVATION_SCHEMA_VERSION_V3:
        return v3_numeric_index(legacy_index)
    if legacy_index < 0 or legacy_index >= spec.numeric_feature_count:
        raise ValueError(
            f"legacy numeric column {legacy_index} is outside the "
            f"{spec.numeric_feature_count}-column public layout for schema {schema_version!r}"
        )
    return legacy_index


def numeric_index_if_present_for_schema(
    schema_version: str, legacy_index: int
) -> int | None:
    """Physical numeric index, or ``None`` only for an explicitly omitted field.

    Invalid and out-of-range semantic indices still raise. This keeps audit code fail-closed
    while allowing one implementation to span schemas that intentionally omit a field.

    Two kinds of omission are legitimate, and both answer None: a field the schema explicitly
    DROPPED (v3's fourteen evidence-backed dead columns), and a field introduced by a LATER
    schema (the v4 feature-pack columns, asked about under v3). The v2 family needs no such
    case — every later column sits above its census, so the range check below already covers it.
    """

    if schema_version == OBSERVATION_SCHEMA_VERSION_V4:
        if legacy_index in V4_DROPPED_LEGACY_NUMERIC_INDICES:
            return None
    elif schema_version == OBSERVATION_SCHEMA_VERSION_V3:
        if (
            legacy_index in V3_DROPPED_LEGACY_NUMERIC_INDICES
            or legacy_index in V4_ONLY_NUMERIC_INDICES
        ):
            return None
    return numeric_index_for_schema(schema_version, legacy_index)


FIELD_TOKEN_OFFSET = 0
SELF_POKEMON_TOKEN_OFFSET = FIELD_TOKEN_OFFSET + FIELD_TOKEN_COUNT
OPPONENT_POKEMON_TOKEN_OFFSET = SELF_POKEMON_TOKEN_OFFSET + SELF_POKEMON_TOKEN_COUNT
ACTION_CANDIDATE_TOKEN_OFFSET = OPPONENT_POKEMON_TOKEN_OFFSET + OPPONENT_POKEMON_TOKEN_COUNT
OPPONENT_TENDENCY_STATS_TOKEN_OFFSET = ACTION_CANDIDATE_TOKEN_OFFSET + ACTION_CANDIDATE_TOKEN_COUNT
# Historical name consumed by the committed V2.2 token-format generator.
STATS_TOKEN_OFFSET = OPPONENT_TENDENCY_STATS_TOKEN_OFFSET
TRANSITION_TOKEN_OFFSET = OPPONENT_TENDENCY_STATS_TOKEN_OFFSET + OPPONENT_TENDENCY_STATS_TOKEN_COUNT

# Transition-token kind ids. Literal copies of transitions.TOKEN_KIND_* — showdown cannot import
# transitions at module level (transitions imports showdown's parse helpers); a unit test asserts
# the two sets stay identical.
_TT_KIND_MOVE = "move"
_TT_KIND_SWITCH = "switch"
_TT_KIND_CANT = "cant"
# Turn-merged sub-block status id: literal copy of turn_merged.SUB_BLOCK_ACTION under the
# same no-module-level-import constraint; lockstep-asserted in tests.
_TM_SUB_BLOCK_ACTION = "action"

# Evidence-mass normalization scale for tendency counts (turn-scale, matches the 64-turn
# transition budget); counts saturate at 64 rather than being encoded as rates.
_STAT_COUNT_DIVISOR = 64.0
# Fixed field order for the opponent-tendency-stats token's opponent weather-reveal pairs.
_WEATHER_REVEAL_ORDER = ("raindance", "sunnyday", "sandstorm", "hail")
# Deterministic gen 3 timed effects: 5 turns for move weather and for these side conditions.
_TIMED_CONDITION_DURATION = 5
_TIMED_SIDE_CONDITIONS = ("reflect", "lightscreen", "safeguard", "mist")
# Revealed trap abilities whose holder threatens switches while alive on the bench.
_TRAP_ABILITIES = frozenset({"shadowtag", "arenatrap", "magnetpull"})
# Pinch berries for the HP-EV-trim variant condition (corrections item 1).
_PINCH_BERRIES = frozenset({"salacberry", "petayaberry", "liechiberry"})


@dataclass(frozen=True)
class ShowdownPokemon:
    ident: str
    showdown_slot: str
    species: str
    condition: Optional[str] = None
    active: bool = False
    details: Optional[str] = None
    moves: tuple[str, ...] = ()
    ability: Optional[str] = None
    item: Optional[str] = None
    # Actual computed stats {hp, atk, def, spa, spd, spe} — known only for the player's own team
    # (from the request); None for opponent mons, whose actual stats are hidden.
    stats: Optional[Mapping[str, int]] = None
    # In-battle LIVE type override for an active mon whose type is retyped by an effect the
    # species token cannot express (Castform Forecast `-formechange`, Kecleon Color Change
    # `typechange`). Unresolved discriminated source: ``"type:<T>"`` (payload already a type)
    # or ``"forme:<Forme>"`` (resolve to the forme's type via the dex at encode time). None for
    # every mon at base type. Set only on the CURRENTLY-ACTIVE mon (reverts on switch-out); the
    # species token stays the base species (retyped formes are OOV for the species vocab).
    live_type_source: Optional[str] = None


def _is_current_public_active(pokemon: object | None) -> bool:
    """Whether a public record is explicitly the current active Pokemon.

    Toxic proof latches authorize simulator-private state, so truthiness is not
    sufficient here: snapshots can contain stale, partial, or malformed rows.
    """

    return getattr(pokemon, "active", None) is True


@dataclass(frozen=True)
class ShowdownReplayState:
    battle_id: str
    players: Mapping[str, str]
    requests: Mapping[str, Mapping[str, Any]]
    public_active: Mapping[str, ShowdownPokemon]
    public_revealed: Mapping[str, tuple[ShowdownPokemon, ...]]
    side_conditions: Mapping[str, tuple[str, ...]]
    side_condition_counts: Mapping[str, Mapping[str, int]]
    boosts: Mapping[str, Mapping[str, int]]
    volatiles: Mapping[str, tuple[str, ...]]
    direct_materialization_blockers: Mapping[str, tuple[str, ...]]
    future_sight: Mapping[str, int]
    # Public Toxic chronology for the active slot. Values 0..15 are the
    # model-facing multiplier; 16 is an internal saturation sentinel meaning
    # Showdown's current stage is already capped at 15 at an ordinary request.
    # Observation encoding still clamps this to the public maximum of 15.
    toxic_stage: Mapping[str, int]
    # Confusion turns-so-far per slot (spec v3 change 4, docs/observation_v3_spec.md): the
    # public elapsed-duration counter of the active mon's ``confusion`` volatile. Advances by 1
    # on each ``|turn|`` while the volatile is present (like the toxic ramp), resets to 0 on
    # ``-end confusion`` / switch-out / faint. Gen3 confusion runs ``this.random(2,6)`` = 2..5
    # turns (encoded ``min(1, elapsed/5)``); the raw counter is uncapped (a mon asleep while
    # confused can dwell past 5). Derived ONLY from public protocol lines.
    confusion_elapsed: Mapping[str, int]
    # Encore turns-so-far per slot (spec v3 change 5, docs/observation_v3_spec.md): the public
    # elapsed-duration counter of the active mon's ``encore`` volatile. Advances by 1 on each
    # ``|turn|`` while the volatile is present (like the toxic ramp / confusion counter), resets
    # to 0 on ``-end Encore`` / switch-out / drag / faint. Gen3 Encore runs the gen3 mod override
    # ``durationCallback() { return this.random(3,7) }`` = 3..6 turns (encoded ``min(1, elapsed/6)``,
    # CAP 6); the raw counter is uncapped (a mon asleep while encored can dwell past 6). Encore is
    # ``noCopy: true`` (not Baton-Pass-copied), so it always drops on switch. Derived ONLY from
    # public protocol lines.
    encore_elapsed: Mapping[str, int]
    # Wrap (partial-trap) turns-so-far per slot (spec v3 change 6, docs/observation_v3_spec.md):
    # the public elapsed-duration counter of the active mon's ``partiallytrapped`` volatile.
    # Advances by 1 on each ``|turn|`` while the volatile is present (like the toxic ramp /
    # confusion / encore counters), resets to 0 on ``-end <partial-trap move> [partiallytrapped]``
    # / switch-out / drag / faint. Gen3 partial-trap (Wrap) lasts 2..5 turns (the base condition's
    # modern ``random(5,7)`` is NOT overridden by the gen3 mod but is the wrong value; poke-engine
    # tracks no duration at all — see the spec), encoded ``min(1, elapsed/5)`` with CAP 5; the raw
    # counter is uncapped. Unlike encore, ``partiallytrapped`` IS Baton-Pass-copied, so the switch
    # reset is gated on the volatile being absent (parallel to confusion). Derived ONLY from public
    # protocol lines.
    wrap_trap_elapsed: Mapping[str, int]
    # Mean Look / Spider Web move-trap per slot (spec v3 change 8, docs/observation_v3_spec.md): a
    # public 0/1 flag = the mon in this slot is switch-locked by an opposing Mean Look / Spider Web.
    # Set on ``|-activate|SLOT|trapped`` (the base ``trapped`` volatile's onStart); the trapper is the
    # opposing active mon (singles). The ``trapped`` volatile is ``noCopy`` with NO onEnd, so no
    # protocol line marks the end; the parser clears the flag when the trapped mon leaves its slot
    # (switch/drag/faint) or the trapper leaves its slot (switch/drag/faint of the opposing slot —
    # the linked source-side volatile is what actually drops the trap). Kept DISTINCT from
    # partiallytrapped (Wrap) and from the trap-ability signal. Derived ONLY from public protocol
    # lines.
    meanlook_trap: Mapping[str, bool]
    public_events: tuple["ShowdownPublicEvent", ...]
    public_lines: tuple[str, ...]
    weather: Optional[str] = None
    turn_number: int = 0
    winner: Optional[str] = None
    # Weather duration/source tracking (exact-state layer): the turn the current weather was set
    # and whether it came from an ability (|-weather|...|[from] ability: — permanent in gen 3).
    weather_set_turn: Optional[int] = None
    weather_from_ability: bool = False
    # Count of end-of-turn ``|-weather|<id>|[upkeep]`` ticks observed since the current weather was
    # set. Move weather runs a 5-turn countdown and the first tick fires at the END of the set turn,
    # before the next request is issued, so the first post-resolution observation already reflects it
    # (deep-line audit #9). Remaining move-weather turns = 5 - this count. Reset on set/clear; unused
    # for permanent ability weather (which short-circuits to the pinned duration).
    weather_upkeeps: int = 0
    # Set-turn per side for the deterministic 5-turn side conditions (Reflect / Light Screen /
    # Safeguard / Mist), keyed by normalized condition id.
    side_condition_set_turns: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    # Pending Wish per side: the turn each side declared Wish (heals its slot end of next turn).
    wish_set_turns: Mapping[str, int] = field(default_factory=dict)
    # For an active Leech Seed target, the public side whose current active slot receives the
    # residual heal. The protocol exposes this through the original move declaration.
    leech_seed_source_sides: Mapping[str, str] = field(default_factory=dict)
    # Transient per-target Leech Seed source recorded on the ``|move|`` line and consumed by the
    # matching ``|-start|``. Serialized so a snapshot taken *between* those two adjacent messages
    # restores identically (snapshot-vs-live convergence). Empty at decision boundaries, where the
    # ``-start`` has already folded it into ``leech_seed_source_sides``.
    pending_leech_seed_source_sides: Mapping[str, str] = field(default_factory=dict)
    # A declared Baton Pass creates a public forced-switch boundary. The incoming Pokemon must
    # inherit boosts and transferable volatiles when that boundary is resolved.
    pending_baton_pass: tuple[str, ...] = ()
    # Per-side in-battle LIVE type override for the currently-active mon (Castform Forecast
    # `-formechange`, Kecleon Color Change `typechange`). Value is the unresolved discriminated
    # source (``"type:<T>"`` / ``"forme:<Forme>"``); None/absent means base type. Cleared on
    # switch-out/drag (both effects revert on leaving the field).
    live_type_override: Mapping[str, Optional[str]] = field(default_factory=dict)
    # Ability the ACTIVE mon is currently borrowing via Trace, or absent.
    #
    # Deliberately NOT `belief.revealed_ability`, which is a persistent fact about a mon and
    # is right for abilities a mon simply revealed. A TRACED ability is transient: Trace
    # re-fires on every switch-in and the copy is dropped on switch-out, so the belief holds
    # the LAST ability that mon ever traced, not the one it holds now. Seeding worlds from
    # the belief stamped a historical trace -- observed handing a Gardevoir `levitate` from
    # an earlier switch-in, which silently granted it Spikes immunity.
    traced_ability: Mapping[str, Optional[str]] = field(default_factory=dict)
    # gen3 Truant loaf parity for the active mon: True = this mon loafs on its next move
    # attempt, False = it acts, absent/None = no Truant holder or the phase is UNKNOWN.
    #
    # gen3 owns Truant outright (`data/mods/gen3/abilities.ts`, `onStart: undefined`) and
    # models it as a free-running boolean, NOT base's volatile:
    #
    #     onSwitchIn(p) { p.truantTurn = this.turn !== 0; }
    #     onBeforeMove(p) { if (p.truantTurn) { cant 'ability: Truant'; return false; } }
    #     onResidualOrder: 27
    #     onResidual(p) { p.truantTurn = !p.truantTurn; }
    #
    # The bit flips at EVERY residual unconditionally -- whether the mon moved, slept,
    # flinched, was paralyzed, recharged or did nothing. That is why "moved last round ->
    # loafs now" is a proxy rather than the bit: the first turn a holder fails to move for a
    # NON-Truant reason the two disagree, and the parity stays inverted thereafter.
    #
    # `this.turn !== 0` is the compensation for the extra residual a mid-battle switch-in
    # sees before its first move opportunity, so a lead and a switch-in both ACT first.
    truant_phase: Mapping[str, Optional[bool]] = field(default_factory=dict)
    # Public parser chronology needed across snapshots taken after ``|upkeep`` but before a
    # forced replacement and the following ``|turn|``. A replacement entered after that
    # residual, so its next turn-boundary flip must be skipped.
    post_upkeep_window: bool = False
    truant_skip_next_flip: tuple[str, ...] = ()
    # Public sleep-clause tracker (spec v3, docs/observation_v3_spec.md change 2): per INDUCING
    # side, the set of enemy victims it has publicly put to sleep (victim keys
    # ``<slot>:<normalized ident name>``). Attribution rule: a ``-status … slp`` line WITHOUT
    # the ``[from] move: Rest`` tag ⇒ induced by the opposing side (in gen3 singles sleep is
    # only ever opponent-induced or self-inflicted Rest, and Rest tags its line). Cleared on
    # ``-curestatus … slp`` and faint; switch-out does NOT clear (sleep persists and is public
    # on revealed mons). Derived ONLY from public protocol lines — no engine-side hidden state.
    induced_sleep_victims: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Rest-sleep provenance, per mon: victim key -> k, the raw number of public move attempts
    # observed since this Rest began. Same ``<slot>:<normalized ident
    # name>`` key as ``induced_sleep_victims`` (see ``_induced_sleep_victim_key`` for why the
    # ident NAME and not the species). Membership answers "is this slp Rest-inflicted"; k is
    # an ATTEMPT count, not elapsed timer units, because Early Bird burns two units per attempt.
    #
    # WHY AN ATTEMPT COUNT AND NOT ELAPSED TURNS — do not "simplify" this back. gen3's sleep
    # timer decrements ONLY inside ``slp.onBeforeMove`` (data/mods/gen3/conditions.ts:24-28),
    # so a BENCHED sleeper's Rest does not tick at all. Elapsed wall-clock turns therefore
    # over-count progress for exactly the population this exists to serve — a Rest-sleeper
    # sitting on the bench — and would build it as closer to waking than it is. ``|cant|SLOT|slp``
    # is emitted on precisely the attempts that DO tick, and on no others, so counting those
    # lines tracks the real timer across any number of switches.
    #
    # The raw count includes sleepUsable attempts too. ``rest_sleep_skipped_turns`` below
    # records the trailing Sleep Talk/Snore run that Gen 3 will refund on re-entry, so a
    # benched Rest sleeper can still be reconstructed exactly rather than discarded.
    rest_sleep_counts: Mapping[str, int] = field(default_factory=dict)
    # Per Rest sleeper, skippedTime refunds already applied by a public switch-in. Keep this
    # history separate from attempts: after the switch the live Showdown field is zero, but an
    # Early Bird world still needs the one-unit refund to reconstruct its timer exactly.
    rest_sleep_refunded_turns: Mapping[str, int] = field(default_factory=dict)
    # Per Rest sleeper, the trailing run of public Sleep Talk/Snore attempts currently stored
    # in Showdown's hidden ``statusState.skippedTime``. It is public because every relevant
    # attempt emits ``|cant|...|slp`` followed by the direct move line or a later blocking
    # event. The Rust world has no skippedTime field, so an ACTIVE sleeper with a nonzero value
    # remains fail-closed; a benched sleeper carries the count separately for exact rebuilding.
    rest_sleep_skipped_turns: Mapping[str, int] = field(default_factory=dict)
    # ``|cant|...|slp`` is observed before the direct Sleep Talk/Snore move line. A later
    # flinch/Truant/confusion/Attract event also proves the selected move was sleepUsable: the
    # Gen 3 sleep handler incremented skippedTime before those lower-priority checks. This
    # transient records the still-unclassified attempt and is snapshot-safe.
    rest_sleep_pending_attempt: Mapping[str, bool] = field(default_factory=dict)
    # Public consecutive-stall counter (spec v3, docs/observation_v3_spec.md change 3): per side,
    # the number of consecutive SUCCESSFUL stall-move uses (Protect/Detect/Endure — gen3 shares
    # one ``stall`` volatile; engine ground truth data/conditions.ts:439-462) by that side's
    # currently-active mon. Incremented on the success-only ``-singleturn`` tag; reset to 0 on a
    # failed stall, any non-stall move, ``cant``, switch-out/drag, or faint. ``stall_move_pending``
    # is the transient per-side "a stall move is in flight this action window" flag (set on a
    # stall ``|move|``, consumed by its ``-singleturn``/``-fail``) that distinguishes reset cause
    # (1) — a failed stall — from an unrelated ``-fail``; serialized so a mid-window resume
    # converges. Both derived ONLY from public protocol lines — no engine-side hidden state.
    stall_counter: Mapping[str, int] = field(default_factory=dict)
    stall_move_pending: Mapping[str, bool] = field(default_factory=dict)
    # Public per-side last EXECUTED move (gen3 ``Pokemon.lastMove``), as the id string, or
    # the sentinel ``"switch"`` for a mon that just came in, or absent for "never moved".
    #
    # Semantics are transcribed from the truth table that
    # ``third_party/poke-engine-gen3-lastmove-semantics.patch`` already made the ENGINE obey,
    # so the two halves cannot disagree: Showdown sets ``lastMove`` in ``Pokemon.moveUsed()``,
    # reached only AFTER the BeforeMove gate and the PP deduction, and BEFORE ``useMove``.
    # Consequences, and why each is publicly readable:
    #   * a move that MISSES, FAILS, or is blocked by Protect still counts as used — Showdown
    #     emits its ``|move|`` line either way, so recording on ``|move|`` is exactly right;
    #   * every immobilizer (par/slp/frz/flinch/confusion-self-hit/attract) returns false from
    #     onBeforeMove, so no ``|move|`` line is emitted at all — Showdown emits ``|cant|``
    #     instead, and this parser records nothing, matching by construction;
    #   * a CALLED move (Sleep Talk's callee) goes through ``useMove``, which never touches
    #     lastMove, while the CALLER does record. Called moves carry a ``[from]`` tag on their
    #     ``|move|`` line, which is the public discriminator used below.
    # Derived ONLY from public protocol lines — no engine-side hidden state.
    last_used_move: Mapping[str, str] = field(default_factory=dict)
    # Public Substitute-health provenance for the active slot. ``full`` is
    # exact immediately after ``-start Substitute``; ``exact`` means cumulative
    # fixed-damage depletion is known; ``unknown`` means a non-breaking hit was
    # announced without a public amount; ``broken`` follows ``-end Substitute``.
    # This stays distinct from volatile presence because an active Substitute
    # can exist while its remaining HP is not public.
    substitute_health_state: Mapping[str, str] = field(default_factory=dict)
    # Cumulative exact depletion since creation when public fixed-damage
    # chronology derives it. This invariant is portable across sampled max HP.
    substitute_depletion: Mapping[str, int | None] = field(default_factory=dict)
    # Whether ``toxic_stage`` is known from the public protocol. A zero alone is
    # ambiguous: it can mean a fresh Toxic/Switch-in counter, or an incomplete
    # prefix whose live toxic counter was never observed. The observation keeps
    # its legacy zero encoding, but direct world construction must reject the
    # latter rather than silently seed a false ``toxic_count = 0``.
    toxic_stage_known: Mapping[str, bool] = field(default_factory=dict)
    # A known stage-0 active ``tox`` counter is normally ambiguous at an
    # action boundary. This proof is set only when a non-Baton-Pass ``switch``
    # introduced the poisoned Pokemon after ``|upkeep``, after that turn's
    # residual had already run. It permits the engine's legitimate pre-tick
    # counter 0 without relaxing the fail-closed rule for every other stage-0
    # snapshot. It is retired by the first Toxic residual and every active
    # status/faint transition.
    toxic_stage_zero_after_upkeep: Mapping[str, bool] = field(default_factory=dict)
    # The next turn whose residual phase must contain the first Toxic tick for
    # a post-upkeep replacement proof. Keeping this deadline in the snapshot
    # makes a resumed parser reject a skipped residual just like a live fold.
    toxic_stage_zero_after_upkeep_expires_after_turn: Mapping[str, int | None] = field(
        default_factory=dict
    )
    # Exact active ident that entered under the bounded stage-zero proof. A
    # boolean alone cannot survive a snapshot safely: the proof belongs to one
    # canonical p1a/p2a occupant, not merely its side.
    toxic_stage_zero_after_upkeep_ident: Mapping[str, str | None] = field(default_factory=dict)
    # A public active faint is eligible to authorize exactly one same-seat
    # post-upkeep replacement. This stays distinct from the materialization
    # proof: a post-upkeep switch without a preceding same-seat faint is not
    # known to be a forced replacement.
    toxic_faint_replacement_pending: Mapping[str, bool] = field(default_factory=dict)
    # Exact active ident that fainted to open a replacement window. This is
    # separate from the pending bit so a restored parser can verify the same
    # outgoing occupant before accepting a replacement.
    toxic_faint_replacement_expected_ident: Mapping[str, str | None] = field(
        default_factory=dict
    )
    # Malformed faint/turn chronology is terminal until the next clean turn.
    # Retaining it across snapshots prevents a later duplicate faint from
    # re-arming a cleared boolean latch.
    toxic_faint_replacement_invalid: Mapping[str, bool] = field(default_factory=dict)
    # Provenance for HP numerators/denominators in this protocol stream. ``exact`` means the
    # denominator is the Pokemon's real max HP; ``percentage`` means Showdown's rounded /100
    # player view; absent/``unknown`` means residual magnitude must not distinguish the two.
    # Persisting this prevents a resumed incremental parser from reinterpreting an exact
    # 100-HP Pokemon as percentage-form (or vice versa).
    hp_visibility: Mapping[str, str] = field(default_factory=dict)
    # ---- v4 k0 feature pack trackers (spec v4, docs/observation_v4_spec.md) --------------------
    # Pack A1. Per slot: the mon in this slot is publicly FORCED to recharge — it spends its next
    # move opportunity on ``cant … recharge`` and cannot act.
    #
    # Derived from the ``|-mustrecharge|SLOT`` line, which the vendored sim emits the moment a
    # recharge move (Hyper Beam, the pool's only carrier) LANDS. That line is a strictly better
    # source than the search lane's reconstruction from the round-indexed action record: a MISSED
    # Hyper Beam never emits it (so the gen3 "a miss does not recharge" rule needs no special
    # case), it names the actor directly (no species-continuity anchor needed), and it cannot
    # scroll out of a rolling window (so there is no fail-open branch). The protocol inventory
    # classifies the line as a semantic alias of the FOLLOWING turn's ``cant:recharge`` transition
    # token — true for the history region, and exactly why the fact was invisible at k0: that row
    # lands one decision too late, after the free turn has already resolved.
    #
    # SET on ``-mustrecharge``; CLEARED when the forced turn is consumed (``|cant|SLOT|recharge``),
    # and on switch/drag out or faint (the volatile leaves with the mon).
    must_recharge: Mapping[str, bool] = field(default_factory=dict)
    # Pack A5. Per slot, for the mon CURRENTLY in it: HP fractions from the PREVIOUS round.
    # ``last_damage_dealt`` is move-attributed damage this mon inflicted on the opposing active mon
    # (fraction of the DEFENDER's max HP); ``last_damage_taken`` is everything this mon lost from
    # any source (fraction of its OWN max HP). The ``current_*`` pair is the in-flight accumulator
    # for the round in progress; the roll-over happens at ``|turn|``. All four reset to 0 on a
    # switch/drag into the slot — these are per-MON facts, and a fresh mon has no record.
    last_damage_dealt: Mapping[str, float] = field(default_factory=dict)
    last_damage_taken: Mapping[str, float] = field(default_factory=dict)
    current_damage_dealt: Mapping[str, float] = field(default_factory=dict)
    current_damage_taken: Mapping[str, float] = field(default_factory=dict)
    # Part B1. Per slot, cumulative entry-hazard damage SUFFERED by that side over the whole game,
    # in units of "one mon's max HP" (each ``[from] Spikes`` ``-damage`` line contributes its own
    # per-mon fraction). The encoder divides by the six-mon team to get a team-HP fraction. Never
    # reset — the point of a credit ledger is that it accumulates.
    hazard_damage_suffered: Mapping[str, float] = field(default_factory=dict)
    # Pack A2 addendum. Per slot: the mon currently in it arrived via BATON PASS rather than an
    # ordinary switch. Only meaningful while ``last_used_move`` is still the ``switch`` sentinel
    # (once the mon executes a move the sentinel is gone), and it exists because the two
    # arrivals are genuinely different facts: a Baton-Pass arrival inherits boosts and the
    # transferable volatiles. That difference IS partly recoverable from the boost columns, but
    # only when something was actually passed, and only as an inference — the explicit
    # ``SWITCH_REASON_BATON_PASS`` the transitions layer records is history-region-only and so
    # invisible at k0. Engine-wise both are ``LastUsedMove::Switch``; the observation is simply
    # allowed to be richer than the world here.
    arrived_by_baton_pass: Mapping[str, bool] = field(default_factory=dict)
    # Pack A4 addendum — the CHOICE LOCK, and where the item came from.
    #
    # ``choice_item_public``: this slot's occupant is publicly known to hold a choice item.
    # ``choice_locked``: it has executed a move since acquiring that item, so gen3's silent
    # ``choicelock`` volatile is now on it and it can use nothing else. That volatile emits NO
    # protocol line at any point (``data/conditions.ts`` choicelock has no ``add``), so this is
    # the only way the fact can reach either consumer.
    # ``item_from_trick``: the currently-held item was SWAPPED on by Trick rather than being the
    # mon's own. This is the valence discriminator: a native Choice Band is the holder's asset
    # (+50% Atk on an all-attacks set), while a Tricked one is a liability we inflicted. Both
    # produce the same "holds a Choice Band" reading without it.
    choice_item_public: Mapping[str, bool] = field(default_factory=dict)
    choice_locked: Mapping[str, bool] = field(default_factory=dict)
    item_from_trick: Mapping[str, bool] = field(default_factory=dict)
    # Part B4. Per slot, how many of that side's held items have been publicly removed by the
    # OTHER side's action (``-enditem … [from] move: Knock Off``). Self-consumed berries and
    # Trick swaps are excluded — see NUMERIC_SELF_ITEMS_REMOVED_CREDIT.
    items_removed: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ShowdownPublicEvent:
    event_type: str
    raw_line: str
    actor_slot: Optional[str] = None
    actor_ident: Optional[str] = None
    target_slot: Optional[str] = None
    target_ident: Optional[str] = None
    primary: Optional[str] = None
    secondary: Optional[str] = None


@dataclass(frozen=True)
class PlayerRelativePublicEvent:
    event_type: str
    raw_line: str
    actor_role: str = "none"
    target_role: str = "none"
    primary: Optional[str] = None
    secondary: Optional[str] = None
    relative_line: Optional[str] = None


@dataclass(frozen=True)
class ShowdownSubmission:
    showdown_slot: str
    choice: str


@dataclass(frozen=True)
class PlayerRelativeBattleState:
    battle_id: str
    player_id: str
    perspective: ObservationPerspective
    request: Mapping[str, Any] | None
    request_kind: str
    self_team: tuple[ShowdownPokemon, ...]
    opponent_team: tuple[ShowdownPokemon, ...]
    self_side_conditions: tuple[str, ...]
    opponent_side_conditions: tuple[str, ...]
    self_side_condition_counts: Mapping[str, int]
    opponent_side_condition_counts: Mapping[str, int]
    self_active_boosts: Mapping[str, int]
    opponent_active_boosts: Mapping[str, int]
    self_active_volatiles: tuple[str, ...]
    opponent_active_volatiles: tuple[str, ...]
    self_toxic_stage: int
    opponent_toxic_stage: int
    belief_view: PlayerBeliefView
    legal_action_mask: tuple[bool, ...]
    recent_events: tuple[PlayerRelativePublicEvent, ...]
    recent_public_events: tuple[str, ...]
    weather: Optional[str] = None
    turn_number: int = 0
    self_future_sight_turns: int = 0  # turns until a delayed attack lands on the player's side
    opponent_future_sight_turns: int = 0  # turns until the player's own delayed attack lands
    winner: Optional[str] = None
    # ---- spec v2: ordered history + tendency aggregates + side-level exact state. ----
    # One TransitionToken per declared action, whole game, within-turn resolution order
    # (oldest-truncation to the encode budget happens at encode time, not here).
    transition_tokens: tuple["TransitionToken", ...] = ()
    tendency_stats: "TendencyStats | None" = None
    # Turn-merged transition stream (spec v2.2): populated only when the caller asks
    # (normalize_for_player(include_turn_merged=True)); empty tuple otherwise so the
    # per-action hot path pays nothing.
    turn_merged_tokens: tuple["TurnMergedToken", ...] = ()
    weather_turns_remaining: int = 0
    weather_permanent: bool = False
    # Turns remaining per active timed side condition (reflect/lightscreen/safeguard/mist).
    self_timed_condition_turns: Mapping[str, int] = field(default_factory=dict)
    opponent_timed_condition_turns: Mapping[str, int] = field(default_factory=dict)
    self_wish_pending: bool = False
    opponent_wish_pending: bool = False
    # ---- spec v3 change 9: Wish turns-to-land per side (docs/observation_v3_spec.md). Turns until
    # a declared Wish resolves — 2 the declaration turn, 1 the landing turn, 0 otherwise — re-derived
    # from the SAME public ``wish_set_turns`` tracker the v2.2 pending bit reads; encoded on the
    # field token under schema >= v3 only as ``min(1, remaining / 2)``. Per-slot, so it survives a
    # wish-pass switch. Nonzero on exactly the turns the pending bit is set.
    self_wish_turns: int = 0
    opponent_wish_turns: int = 0
    # Live sleep-clause consumption per side (from the belief engine's holders).
    self_sleep_clause_used: bool = False
    opponent_sleep_clause_used: bool = False
    # ---- spec v3: public sleep-clause block bits (docs/observation_v3_spec.md change 2).
    # Derived ONLY from public protocol lines (the _ReplayParser induced-sleep tracker),
    # independent of the belief engine — encoded on the field token under schema >= v3 only.
    # self_sleep_clause_blocks: an opposing mon is currently asleep from a sleep OUR side
    # induced (our sleep-inducing moves will fail); opponent_* is the symmetric bit.
    self_sleep_clause_blocks: bool = False
    opponent_sleep_clause_blocks: bool = False
    # ---- spec v3 change 3: consecutive-stall counter (docs/observation_v3_spec.md). Public
    # count of consecutive SUCCESSFUL stall-move uses by each side's ACTIVE mon (Protect/Detect/
    # Endure), from the _ReplayParser tracker. Encoded on the active pokemon token (like
    # NUMERIC_TOXIC_STAGE) as min(1.0, count / 8.0) under schema >= v3 only.
    self_stall_counter: int = 0
    opponent_stall_counter: int = 0
    # ---- spec v3 change 4: confusion turns-so-far (docs/observation_v3_spec.md). Per-side
    # public elapsed-duration counter for the ACTIVE mon's confusion volatile, from the
    # _ReplayParser tracker; encoded on the confused mon's token under schema >= v3 only as
    # min(1, elapsed/5) (gen3 CAP = 5). 0 when the active mon is not confused.
    self_confusion_elapsed: int = 0
    opponent_confusion_elapsed: int = 0
    # ---- spec v3 change 5: encore turns-so-far (docs/observation_v3_spec.md). Per-side public
    # elapsed-duration counter for the ACTIVE mon's encore volatile, from the _ReplayParser
    # tracker; encoded on the encored mon's token under schema >= v3 only as min(1, elapsed/6)
    # (gen3 CAP = 6). 0 when the active mon is not encored.
    self_encore_elapsed: int = 0
    opponent_encore_elapsed: int = 0
    # ---- spec v3 change 6: Wrap (partial-trap) turns-so-far (docs/observation_v3_spec.md).
    # Per-side public elapsed-duration counter for the ACTIVE mon's partiallytrapped volatile,
    # from the _ReplayParser tracker; encoded on the trapped mon's token under schema >= v3 only
    # as min(1, elapsed/5) (gen3 CAP = 5). 0 when the active mon is not partially trapped.
    self_wrap_trap_elapsed: int = 0
    opponent_wrap_trap_elapsed: int = 0
    # ---- spec v3 change 8: Mean Look / Spider Web move-trap (docs/observation_v3_spec.md). Per-side
    # public 0/1 flag for the ACTIVE mon, from the _ReplayParser tracker; encoded on the trapped
    # mon's token under schema >= v3 only. False when the active mon is not switch-locked by a
    # Mean Look / Spider Web.
    self_meanlook_trap: bool = False
    opponent_meanlook_trap: bool = False
    # ---- spec v4: the k0 feature pack (docs/observation_v4_spec.md). Every field below is read
    # from a _ReplayParser tracker (public protocol only) and encoded under schema v4 only.
    # A1: the active mon is publicly locked into a recharge turn and cannot act.
    self_must_recharge: bool = False
    opponent_must_recharge: bool = False
    # A3: the active mon is a Truant holder whose next move attempt LOAFS. False covers both "not
    # a holder" and "phase unknown", mirroring the world's None fallback.
    self_truant_loaf: bool = False
    opponent_truant_loaf: bool = False
    # A2: the active mon's last EXECUTED move id, the ``"switch"`` sentinel for a mon that just
    # came in, or None for "never moved".
    self_last_used_move: Optional[str] = None
    opponent_last_used_move: Optional[str] = None
    # A2 addendum: the active mon arrived by Baton Pass rather than an ordinary switch. Only
    # meaningful while last_used_move is still the "switch" sentinel.
    self_arrived_by_baton_pass: bool = False
    opponent_arrived_by_baton_pass: bool = False
    # A4 addendum: the active mon is publicly choice-locked into its last executed move, and
    # whether the item that locked it was Tricked on rather than its own.
    self_choice_locked: bool = False
    opponent_choice_locked: bool = False
    self_item_swapped: bool = False
    opponent_item_swapped: bool = False
    # A4: the ability the active mon is CURRENTLY borrowing via Trace (transient, cleared on
    # switch-out), or None. Deliberately not the belief's persistent revealed-ability channel.
    self_traced_ability: Optional[str] = None
    opponent_traced_ability: Optional[str] = None
    # A5: previous-round damage for the mon currently in the slot — move-attributed damage it
    # DEALT (fraction of the defender's max HP) and total damage it TOOK (fraction of its own).
    self_last_damage_dealt: float = 0.0
    self_last_damage_taken: float = 0.0
    opponent_last_damage_dealt: float = 0.0
    opponent_last_damage_taken: float = 0.0
    # B1: cumulative entry-hazard damage SUFFERED by each side, in units of one mon's max HP.
    # Orientation matches self/opponent_side_conditions: self_* is damage OUR mons took.
    self_hazard_damage_suffered: float = 0.0
    opponent_hazard_damage_suffered: float = 0.0
    # B4: how many of each side's held items the OTHER side has publicly knocked off.
    self_items_removed: int = 0
    opponent_items_removed: int = 0
    # Matchup-conditional switch evidence, ALREADY conditioned on our current active:
    # normalized opponent species -> (switched-out-before-attacking, stayed-and-attacked)
    # observed while that mon was facing the mon we have out now. Absent species have no
    # history in this matchup and encode (0, 0). The second slot is the complementary COUNT,
    # not a denominator: an opportunities total cannot survive the incremental fold, which
    # prunes turn_start_occupants, so both live hook points increment one counter or the other.
    opponent_matchup_switch_evidence: Mapping[str, tuple[int, int]] = field(
        default_factory=dict
    )

    @property
    def self_active(self) -> ShowdownPokemon | None:
        return next((pokemon for pokemon in self.self_team if pokemon.active), None)

    @property
    def opponent_active(self) -> ShowdownPokemon | None:
        return next((pokemon for pokemon in self.opponent_team if pokemon.active), None)


class _ReplayParser:
    """Incremental fold of Showdown protocol lines into transport-level replay state.

    ``parse_showdown_replay`` is a thin batch wrapper around this. The local sim env keeps a
    persistent instance and ``feed()``s only newly-arrived lines, so each line is parsed once
    (O(n) per game) instead of the whole accumulated log being re-parsed on every observation
    (O(n^2)). ``snapshot()`` returns an immutable :class:`ShowdownReplayState` and copies the
    mutable accumulators, so a snapshot is unaffected by later ``feed()`` calls.
    """

    def __init__(
        self,
        battle_id: str = "replay",
        *,
        complete_prefix: bool = False,
        hp_visibility: Mapping[str, str] | None = None,
    ) -> None:
        self.battle_id = battle_id
        self.players: dict[str, str] = {}
        self.requests: dict[str, Mapping[str, Any]] = {}
        self.public_active: dict[str, ShowdownPokemon] = {}
        self.public_revealed: dict[str, list[ShowdownPokemon]] = {}
        self.side_condition_counts: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
        self.boosts: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
        self.volatiles: dict[str, set[str]] = {"p1": set(), "p2": set()}
        self.substitute_health_state: dict[str, str] = {"p1": "absent", "p2": "absent"}
        self.substitute_depletion: dict[str, int | None] = {"p1": None, "p2": None}
        self.direct_materialization_blockers: dict[str, set[str]] = {"p1": set(), "p2": set()}
        self.future_sight: dict[str, int] = {}
        self.toxic_stage: dict[str, int] = {"p1": 0, "p2": 0}
        # A fresh parser is safe for attach-midstream use: zero is not public proof of an
        # active Toxic counter unless the caller attests that the prefix starts at reset.
        complete_prefix_is_true = type(complete_prefix) is bool and complete_prefix
        self.toxic_stage_known: dict[str, bool] = {
            "p1": complete_prefix_is_true,
            "p2": complete_prefix_is_true,
        }
        self.toxic_stage_zero_after_upkeep: dict[str, bool] = {"p1": False, "p2": False}
        self.toxic_stage_zero_after_upkeep_expires_after_turn: dict[str, int | None] = {
            "p1": None,
            "p2": None,
        }
        self.toxic_stage_zero_after_upkeep_ident: dict[str, str | None] = {
            "p1": None,
            "p2": None,
        }
        self.toxic_faint_replacement_pending: dict[str, bool] = {"p1": False, "p2": False}
        self.toxic_faint_replacement_expected_ident: dict[str, str | None] = {
            "p1": None,
            "p2": None,
        }
        self.toxic_faint_replacement_invalid: dict[str, bool] = {"p1": False, "p2": False}
        self.hp_visibility: dict[str, str] = {"p1": "unknown", "p2": "unknown"}
        for slot, visibility in (hp_visibility or {}).items():
            if slot not in self.hp_visibility:
                raise ValueError(f"unknown Showdown slot in hp_visibility: {slot!r}")
            if visibility not in {"exact", "percentage", "unknown"}:
                raise ValueError(f"invalid HP visibility for {slot}: {visibility!r}")
            self.hp_visibility[slot] = visibility
        # Confusion turns-so-far per slot (spec v3 change 4). See ShowdownReplayState.confusion_elapsed.
        self.confusion_elapsed: dict[str, int] = {"p1": 0, "p2": 0}
        # Encore turns-so-far per slot (spec v3 change 5). See ShowdownReplayState.encore_elapsed.
        self.encore_elapsed: dict[str, int] = {"p1": 0, "p2": 0}
        # Wrap (partial-trap) turns-so-far per slot (spec v3 change 6). See
        # ShowdownReplayState.wrap_trap_elapsed.
        self.wrap_trap_elapsed: dict[str, int] = {"p1": 0, "p2": 0}
        # Mean Look / Spider Web move-trap per slot (spec v3 change 8). See
        # ShowdownReplayState.meanlook_trap.
        self.meanlook_trap: dict[str, bool] = {"p1": False, "p2": False}
        self.pending_baton_pass: set[str] = set()
        self.public_events: list[ShowdownPublicEvent] = []
        self.public_lines: list[str] = []
        self.weather: Optional[str] = None
        self.turn_number: int = 0
        self.winner: Optional[str] = None
        self.weather_set_turn: Optional[int] = None
        self.weather_from_ability: bool = False
        self.weather_upkeeps: int = 0
        self.side_condition_set_turns: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
        self.wish_set_turns: dict[str, int] = {}
        self.leech_seed_source_sides: dict[str, str] = {}
        self._pending_leech_seed_source_sides: dict[str, str] = {}
        # Per-side live type override for the active mon (Castform Forecast / Kecleon Color
        # Change). Unresolved discriminated source ("type:<T>" / "forme:<Forme>"); None = base.
        self.live_type_override: dict[str, Optional[str]] = {"p1": None, "p2": None}
        # See ShowdownReplayState.traced_ability: transient, cleared on switch-out.
        self.traced_ability: dict[str, Optional[str]] = {"p1": None, "p2": None}
        # See ShowdownReplayState.truant_phase. None = no holder / phase unknown.
        self.truant_phase: dict[str, Optional[bool]] = {"p1": None, "p2": None}
        # True between |upkeep and the next |turn| -- the window in which a faint
        # replacement enters AFTER that turn's residual has already run.
        self._post_upkeep_window: bool = False
        # Slots whose next |turn| flip must be skipped (see _TRUANT replacement guard).
        self._truant_skip_next_flip: set[str] = set()
        # Public sleep-clause tracker (spec v3): per INDUCING side, the set of enemy victims
        # it has publicly put to sleep. See ShowdownReplayState.induced_sleep_victims.
        self.induced_sleep_victims: dict[str, set[str]] = {"p1": set(), "p2": set()}
        # Rest-sleep provenance: victim key -> raw public attempts since this Rest began.
        # See ShowdownReplayState.rest_sleep_counts (and why it is attempts, not turns).
        self.rest_sleep_counts: dict[str, int] = {}
        self.rest_sleep_refunded_turns: dict[str, int] = {}
        self.rest_sleep_skipped_turns: dict[str, int] = {}
        self.rest_sleep_pending_attempt: dict[str, bool] = {}
        # Public consecutive-stall counter (spec v3 change 3) + its transient in-flight flag.
        # See ShowdownReplayState.stall_counter / stall_move_pending.
        self.stall_counter: dict[str, int] = {"p1": 0, "p2": 0}
        self.stall_move_pending: dict[str, bool] = {"p1": False, "p2": False}
        # See ShowdownReplayState.last_used_move for the transcribed truth table.
        self.last_used_move: dict[str, str | None] = {"p1": None, "p2": None}
        # ---- v4 k0 feature pack trackers. See the matching ShowdownReplayState fields. --------
        self.must_recharge: dict[str, bool] = {"p1": False, "p2": False}
        self.last_damage_dealt: dict[str, float] = {"p1": 0.0, "p2": 0.0}
        self.last_damage_taken: dict[str, float] = {"p1": 0.0, "p2": 0.0}
        self.current_damage_dealt: dict[str, float] = {"p1": 0.0, "p2": 0.0}
        self.current_damage_taken: dict[str, float] = {"p1": 0.0, "p2": 0.0}
        self.hazard_damage_suffered: dict[str, float] = {"p1": 0.0, "p2": 0.0}
        self.items_removed: dict[str, int] = {"p1": 0, "p2": 0}
        self.arrived_by_baton_pass: dict[str, bool] = {"p1": False, "p2": False}
        self.choice_item_public: dict[str, bool] = {"p1": False, "p2": False}
        self.choice_locked: dict[str, bool] = {"p1": False, "p2": False}
        self.item_from_trick: dict[str, bool] = {"p1": False, "p2": False}
        # Transient (NOT snapshotted, and deliberately so): which slot owns the move window the
        # next untagged ``-damage`` line belongs to. Set by ``|move|``, cleared by anything that
        # proves the damage is not the actor's move damage (a confusion self-hit marker, a
        # ``cant``, a turn boundary). A snapshot taken mid-window resumes with no attribution
        # rather than a guessed one: dropping one action's DEALT credit is a zero, whereas a
        # wrong actor would be a false fact.
        self._damage_window_actor: str | None = None

    @classmethod
    def from_snapshot(cls, snapshot: ShowdownReplayState) -> "_ReplayParser":
        """Hydrate parser state directly, without replaying its protocol prefix."""

        snapshot_post_upkeep_window = getattr(snapshot, "post_upkeep_window", None)
        post_upkeep_window_is_valid = type(snapshot_post_upkeep_window) is bool
        parser = cls(
            snapshot.battle_id,
            hp_visibility=getattr(snapshot, "hp_visibility", {}),
        )
        parser.players = dict(snapshot.players)
        parser.requests = dict(snapshot.requests)
        parser.public_active = dict(snapshot.public_active)
        parser.public_revealed = {
            slot: list(pokemon) for slot, pokemon in snapshot.public_revealed.items()
        }
        parser.side_condition_counts = {
            slot: dict(snapshot.side_condition_counts.get(slot, {})) for slot in ("p1", "p2")
        }
        parser.boosts = {slot: dict(snapshot.boosts.get(slot, {})) for slot in ("p1", "p2")}
        parser.volatiles = {
            slot: set(snapshot.volatiles.get(slot, ())) for slot in ("p1", "p2")
        }
        parser.substitute_health_state = {
            slot: str(snapshot.substitute_health_state.get(slot, "absent"))
            for slot in ("p1", "p2")
        }
        parser.substitute_depletion = {
            slot: snapshot.substitute_depletion.get(slot) for slot in ("p1", "p2")
        }
        parser.direct_materialization_blockers = {
            slot: set(snapshot.direct_materialization_blockers.get(slot, ()))
            for slot in ("p1", "p2")
        }
        parser.future_sight = dict(snapshot.future_sight)
        parser.toxic_stage = {slot: int(snapshot.toxic_stage.get(slot, 0)) for slot in ("p1", "p2")}
        snapshot_stage_known = getattr(snapshot, "toxic_stage_known", {})
        parser.toxic_stage_known = {}
        for slot in ("p1", "p2"):
            if isinstance(snapshot_stage_known, Mapping) and slot in snapshot_stage_known:
                parser.toxic_stage_known[slot] = snapshot_stage_known[slot] is True
                continue
            # Old snapshots did not preserve this provenance. A non-toxic active
            # has no counter to reconstruct; an active tox mon must fail closed.
            active = snapshot.public_active.get(slot)
            parser.toxic_stage_known[slot] = (
                _is_current_public_active(active)
                and not _condition_has_status(getattr(active, "condition", None), "tox")
            )
        for slot, known in parser.toxic_stage_known.items():
            if not known:
                parser.toxic_stage[slot] = 0
        snapshot_zero_after_upkeep = getattr(snapshot, "toxic_stage_zero_after_upkeep", {})
        snapshot_faint_replacement_pending = getattr(
            snapshot, "toxic_faint_replacement_pending", {}
        )
        snapshot_expected_ident = getattr(snapshot, "toxic_faint_replacement_expected_ident", {})
        snapshot_invalid = getattr(snapshot, "toxic_faint_replacement_invalid", None)
        authorization_maps_are_valid = all(
            isinstance(values, Mapping)
            and all(type(values.get(slot)) is bool for slot in ("p1", "p2"))
            for values in (
                snapshot_zero_after_upkeep,
                snapshot_faint_replacement_pending,
                snapshot_invalid,
            )
        )
        parser.toxic_stage_zero_after_upkeep = {
            slot: (
                post_upkeep_window_is_valid
                and authorization_maps_are_valid
                and snapshot_zero_after_upkeep.get(slot) is True
                if isinstance(snapshot_zero_after_upkeep, Mapping)
                else False
            )
            for slot in ("p1", "p2")
        }
        snapshot_zero_after_upkeep_deadline = getattr(
            snapshot, "toxic_stage_zero_after_upkeep_expires_after_turn", {}
        )
        if not isinstance(snapshot_zero_after_upkeep_deadline, Mapping):
            snapshot_zero_after_upkeep_deadline = {}
        parser.toxic_stage_zero_after_upkeep_expires_after_turn = {}
        snapshot_zero_after_upkeep_ident = getattr(snapshot, "toxic_stage_zero_after_upkeep_ident", {})
        if not isinstance(snapshot_zero_after_upkeep_ident, Mapping):
            snapshot_zero_after_upkeep_ident = {}
        for slot in ("p1", "p2"):
            deadline = snapshot_zero_after_upkeep_deadline.get(slot)
            proof_ident = snapshot_zero_after_upkeep_ident.get(slot)
            active = parser.public_active.get(slot)
            if (
                parser.toxic_stage_zero_after_upkeep[slot] is True
                and type(deadline) is int
                and isinstance(proof_ident, str)
                and _is_active_protocol_ident(proof_ident)
                and _is_current_public_active(active)
                and getattr(active, "ident", None) == proof_ident
                and _condition_has_status(getattr(active, "condition", None), "tox")
                and parser.toxic_stage_known[slot]
                and parser.toxic_stage[slot] == 0
            ):
                parser.toxic_stage_zero_after_upkeep_expires_after_turn[slot] = deadline
                parser.toxic_stage_zero_after_upkeep_ident[slot] = proof_ident
            else:
                # Older snapshots lack the deadline, so their proof cannot be
                # bounded to a single residual opportunity.
                parser.toxic_stage_zero_after_upkeep[slot] = False
                parser.toxic_stage_zero_after_upkeep_expires_after_turn[slot] = None
                parser.toxic_stage_zero_after_upkeep_ident[slot] = None
        if not isinstance(snapshot_faint_replacement_pending, Mapping):
            snapshot_faint_replacement_pending = {}
        if not isinstance(snapshot_expected_ident, Mapping):
            snapshot_expected_ident = {}
        if not isinstance(snapshot_invalid, Mapping):
            snapshot_invalid = {}
        for slot in ("p1", "p2"):
            expected_ident = snapshot_expected_ident.get(slot)
            active = parser.public_active.get(slot)
            parser.toxic_faint_replacement_invalid[slot] = (
                True
                if (
                    not post_upkeep_window_is_valid
                    or not authorization_maps_are_valid
                    or not isinstance(snapshot_invalid, Mapping)
                    or slot not in snapshot_invalid
                    or slot not in snapshot_expected_ident
                )
                else snapshot_invalid.get(slot) is not False
            )
            parser.toxic_faint_replacement_pending[slot] = (
                snapshot_faint_replacement_pending.get(slot) is True
                and parser.toxic_faint_replacement_invalid[slot] is False
                and snapshot_post_upkeep_window is False
                and isinstance(expected_ident, str)
                and _is_active_protocol_ident(expected_ident)
                and _is_current_public_active(active)
                and getattr(active, "ident", None) == expected_ident
            )
            parser.toxic_faint_replacement_expected_ident[slot] = (
                expected_ident if parser.toxic_faint_replacement_pending[slot] else None
            )
            if parser.toxic_stage_zero_after_upkeep[slot] is True and (
                snapshot_faint_replacement_pending.get(slot) is not False
                or snapshot_invalid.get(slot) is not False
                or snapshot_expected_ident.get(slot) is not None
            ):
                # A materializable zero is the state *after* its one pending
                # replacement has been consumed.  Never repair a conflicting
                # snapshot by silently dropping the conflicting latch.
                parser.toxic_stage_zero_after_upkeep[slot] = False
                parser.toxic_stage_zero_after_upkeep_expires_after_turn[slot] = None
                parser.toxic_stage_zero_after_upkeep_ident[slot] = None
                parser.toxic_faint_replacement_pending[slot] = False
                parser.toxic_faint_replacement_expected_ident[slot] = None
                parser.toxic_faint_replacement_invalid[slot] = True
        parser.confusion_elapsed = {
            slot: int(snapshot.confusion_elapsed.get(slot, 0)) for slot in ("p1", "p2")
        }
        parser.encore_elapsed = {
            slot: int(snapshot.encore_elapsed.get(slot, 0)) for slot in ("p1", "p2")
        }
        parser.wrap_trap_elapsed = {
            slot: int(snapshot.wrap_trap_elapsed.get(slot, 0)) for slot in ("p1", "p2")
        }
        parser.meanlook_trap = {
            slot: bool(snapshot.meanlook_trap.get(slot, False)) for slot in ("p1", "p2")
        }
        parser.public_events = list(snapshot.public_events)
        parser.public_lines = list(snapshot.public_lines)
        parser.weather = snapshot.weather
        parser.turn_number = snapshot.turn_number
        parser.winner = snapshot.winner
        parser.weather_set_turn = snapshot.weather_set_turn
        parser.weather_from_ability = snapshot.weather_from_ability
        parser.weather_upkeeps = snapshot.weather_upkeeps
        parser.side_condition_set_turns = {
            slot: dict(snapshot.side_condition_set_turns.get(slot, {})) for slot in ("p1", "p2")
        }
        parser.wish_set_turns = dict(snapshot.wish_set_turns)
        parser.leech_seed_source_sides = dict(snapshot.leech_seed_source_sides)
        parser._pending_leech_seed_source_sides = dict(snapshot.pending_leech_seed_source_sides)
        parser.pending_baton_pass = set(snapshot.pending_baton_pass)
        parser.live_type_override = {
            slot: snapshot.live_type_override.get(slot) for slot in ("p1", "p2")
        }
        parser.traced_ability = {
            slot: snapshot.traced_ability.get(slot) for slot in ("p1", "p2")
        }
        parser.truant_phase = {
            slot: snapshot.truant_phase.get(slot) for slot in ("p1", "p2")
        }
        # Toxic replacement proof is valid only at an exact protocol boundary.
        # Never coerce malformed snapshot data into a boundary authorization.
        parser._post_upkeep_window = (
            snapshot_post_upkeep_window if post_upkeep_window_is_valid else False
        )
        parser._truant_skip_next_flip = {
            slot for slot in snapshot.truant_skip_next_flip if slot in {"p1", "p2"}
        }
        parser.induced_sleep_victims = {
            slot: set(snapshot.induced_sleep_victims.get(slot, ())) for slot in ("p1", "p2")
        }
        parser.rest_sleep_counts = {
            key: int(count) for key, count in snapshot.rest_sleep_counts.items()
        }
        parser.rest_sleep_refunded_turns = {
            key: int(count) for key, count in snapshot.rest_sleep_refunded_turns.items()
        }
        parser.rest_sleep_skipped_turns = {
            key: int(count) for key, count in snapshot.rest_sleep_skipped_turns.items()
        }
        parser.rest_sleep_pending_attempt = {
            key: bool(pending) for key, pending in snapshot.rest_sleep_pending_attempt.items()
        }
        parser.stall_counter = {
            slot: int(snapshot.stall_counter.get(slot, 0)) for slot in ("p1", "p2")
        }
        parser.stall_move_pending = {
            slot: bool(snapshot.stall_move_pending.get(slot, False)) for slot in ("p1", "p2")
        }
        parser.last_used_move = {
            slot: (snapshot.last_used_move.get(slot) or None) for slot in ("p1", "p2")
        }
        # v4 feature-pack trackers. ``getattr`` defaults keep a v3-era snapshot loadable: an
        # older payload simply restores the zero state these counters start in, which is the
        # honest answer (no evidence recorded) rather than a fabricated one.
        parser.must_recharge = {
            slot: bool(getattr(snapshot, "must_recharge", {}).get(slot, False))
            for slot in ("p1", "p2")
        }
        for field_name in (
            "last_damage_dealt",
            "last_damage_taken",
            "current_damage_dealt",
            "current_damage_taken",
            "hazard_damage_suffered",
        ):
            restored = getattr(snapshot, field_name, {}) or {}
            setattr(
                parser,
                field_name,
                {slot: float(restored.get(slot, 0.0) or 0.0) for slot in ("p1", "p2")},
            )
        parser.items_removed = {
            slot: int(getattr(snapshot, "items_removed", {}).get(slot, 0) or 0)
            for slot in ("p1", "p2")
        }
        for field_name in (
            "arrived_by_baton_pass",
            "choice_item_public",
            "choice_locked",
            "item_from_trick",
        ):
            restored = getattr(snapshot, field_name, {}) or {}
            setattr(
                parser,
                field_name,
                {slot: bool(restored.get(slot, False)) for slot in ("p1", "p2")},
            )
        return parser

    def feed(self, lines: Sequence[str]) -> None:
        for raw_line in lines:
            self._feed_line(raw_line)

    def _sanitize_toxic_replacement_provenance(self) -> None:
        """Fail closed before a mutable proof latch can authorize a world zero.

        These maps are parser internals, but live callers can retain and mutate a
        parser between lines.  Do not let Python truthiness turn a forged ``1``
        or ``0`` into an authorization bit while a replacement is in flight.
        """

        sentinel = object()
        fields = (
            "toxic_stage_zero_after_upkeep",
            "toxic_faint_replacement_pending",
            "toxic_faint_replacement_invalid",
            "toxic_stage_zero_after_upkeep_expires_after_turn",
            "toxic_stage_zero_after_upkeep_ident",
            "toxic_faint_replacement_expected_ident",
        )
        values: dict[str, dict[str, object]] = {}
        for field_name in fields:
            source = getattr(self, field_name, None)
            values[field_name] = {
                slot: source.get(slot, sentinel) if isinstance(source, Mapping) else sentinel
                for slot in ("p1", "p2")
            }
            setattr(self, field_name, values[field_name])

        post_upkeep_window_is_valid = type(self._post_upkeep_window) is bool
        if not post_upkeep_window_is_valid:
            self._post_upkeep_window = False

        for slot in ("p1", "p2"):
            proof = values["toxic_stage_zero_after_upkeep"][slot]
            pending = values["toxic_faint_replacement_pending"][slot]
            invalid = values["toxic_faint_replacement_invalid"][slot]
            deadline = values["toxic_stage_zero_after_upkeep_expires_after_turn"][slot]
            proof_ident = values["toxic_stage_zero_after_upkeep_ident"][slot]
            expected_ident = values["toxic_faint_replacement_expected_ident"][slot]
            active = self.public_active.get(slot)
            proof_is_complete = (
                proof is True
                and type(deadline) is int
                and deadline >= 1
                and isinstance(proof_ident, str)
                and _is_active_protocol_ident(proof_ident)
                and _is_current_public_active(active)
                and getattr(active, "ident", None) == proof_ident
                and _condition_has_status(getattr(active, "condition", None), "tox")
                and isinstance(self.toxic_stage_known, Mapping)
                and self.toxic_stage_known.get(slot) is True
                and isinstance(self.toxic_stage, Mapping)
                and type(self.toxic_stage.get(slot)) is int
                and self.toxic_stage.get(slot) == 0
                and type(self.turn_number) is int
                and self.turn_number >= 0
                and deadline
                == self.turn_number + (1 if self._post_upkeep_window is True else 0)
            )
            pending_is_complete = (
                pending is True
                and isinstance(expected_ident, str)
                and _is_active_protocol_ident(expected_ident)
                and _is_current_public_active(active)
                and getattr(active, "ident", None) == expected_ident
            )
            clean_empty_proof = proof is False and deadline is None and proof_ident is None
            clean_empty_pending = pending is False and expected_ident is None
            authorization_bits_are_valid = all(
                type(values[field_name][slot]) is bool
                for field_name in (
                    "toxic_stage_zero_after_upkeep",
                    "toxic_faint_replacement_pending",
                    "toxic_faint_replacement_invalid",
                )
            )
            if (
                post_upkeep_window_is_valid
                and authorization_bits_are_valid
                and (proof_is_complete or clean_empty_proof)
                and (pending_is_complete or clean_empty_pending)
                and not (proof is True and pending is True)
                and not (invalid is True and (proof is True or pending is True))
            ):
                continue
            self.toxic_stage_zero_after_upkeep[slot] = False
            self.toxic_stage_zero_after_upkeep_expires_after_turn[slot] = None
            self.toxic_stage_zero_after_upkeep_ident[slot] = None
            self.toxic_faint_replacement_pending[slot] = False
            self.toxic_faint_replacement_expected_ident[slot] = None
            self.toxic_faint_replacement_invalid[slot] = True

    def _feed_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return
        if line.startswith(">"):
            return
        self._sanitize_toxic_replacement_provenance()
        parts = line.split("|")
        event_type = parts[1] if len(parts) > 1 else ""
        canonical_turn = _canonical_turn_number(raw_line) if event_type == "turn" else None
        canonical_upkeep = _canonical_upkeep_marker(raw_line)
        canonical_faint = _canonical_faint_marker(raw_line, parts)
        canonical_replacement = _canonical_replacement_marker(raw_line, event_type, parts)
        # BattleStream emits wall-clock timestamp lines (``|t:|...``). They are useful for raw
        # protocol debugging but are not battle state and would make replay-from-root observations
        # differ across otherwise identical deterministic simulations.
        if event_type == "t:":
            return
        self._update_toxic_faint_replacement_latch(
            event_type,
            parts,
            canonical_turn=canonical_turn,
            canonical_upkeep=canonical_upkeep,
            canonical_faint=canonical_faint,
            canonical_replacement=canonical_replacement,
        )
        if event_type == "player" and len(parts) >= 4:
            showdown_slot = parts[2]
            if showdown_slot in {"p1", "p2"}:
                self.players[showdown_slot] = parts[3]
            self.public_events.append(_public_event_from_line(line))
            self.public_lines.append(line)
            return
        if event_type == "request" and len(parts) >= 3:
            payload = _decode_request_payload(line)
            side = payload.get("side") if isinstance(payload.get("side"), Mapping) else {}
            showdown_slot = side.get("id") if isinstance(side, Mapping) else None
            if showdown_slot in {"p1", "p2"}:
                self.requests[showdown_slot] = payload
            return
        if event_type in {"switch", "drag", "replace"} and len(parts) >= 4:
            replacement_slot = _slot_from_ident(parts[2])
            pending_faint_replacement = (
                self.toxic_faint_replacement_pending.get(replacement_slot) is True
            )
            expected_fainted_ident = self.toxic_faint_replacement_expected_ident.get(
                replacement_slot
            )
            replacement_is_canonical = canonical_replacement
            replacement_active = self.public_active.get(replacement_slot)
            replacement_matches_pending = (
                pending_faint_replacement
                and self.toxic_faint_replacement_invalid.get(replacement_slot) is False
                and replacement_is_canonical
                and isinstance(expected_fainted_ident, str)
                and _is_current_public_active(replacement_active)
                and getattr(replacement_active, "ident", None) == expected_fainted_ident
            )
            # Consume before parsing: malformed or duplicate replacement
            # lines must never leave a faint proof reusable later.
            if replacement_slot in self.toxic_faint_replacement_pending:
                self.toxic_faint_replacement_pending[replacement_slot] = False
                self.toxic_faint_replacement_expected_ident[replacement_slot] = None
            # Switch, drag, and replace protocol lines name the active singles
            # seat as p1a/p2a. Do not fold a malformed bench ident as a new
            # active Pokemon, even though its side prefix is recognizable.
            pokemon = _pokemon_from_public_line(parts) if replacement_is_canonical else None
            if pokemon is not None:
                self._refund_rest_sleep_on_switch(pokemon)
                self.public_active[pokemon.showdown_slot] = pokemon
                _record_public_reveal(self.public_revealed, pokemon)
                # Native Slakoth and Slaking are mono-ability, so species proves that Truant's
                # switch-in hook ran and `truantTurn = this.turn !== 0` is an honest seed. A
                # post-upkeep forced replacement missed the residual that just ran, so its
                # first following `|turn|` must not flip the bit again. Trace acquisition is
                # different: retained current-source cases prove that identical public line
                # placement can produce opposite first phases, so the `-ability` handler leaves
                # traced Truant UNKNOWN until a public move or Truant `cant` anchors it.
                if _normalize_identifier(pokemon.species or "") in _TRUANT_SPECIES:
                    self.truant_phase[pokemon.showdown_slot] = self.turn_number != 0
                    if self._post_upkeep_window is True:
                        self._truant_skip_next_flip.add(pokemon.showdown_slot)
                    else:
                        self._truant_skip_next_flip.discard(pokemon.showdown_slot)
                else:
                    # Not a holder: nothing to carry, and distinct from False (which asserts
                    # a KNOWN acting phase).
                    self.truant_phase[pokemon.showdown_slot] = None
                # A new mon takes the slot with fresh (zero) stat-boost stages — UNLESS it came
                # in via Baton Pass, which carries the passer's boosts to the incoming mon. Only
                # a true |switch| can be a Baton Pass; a |drag| (Roar/Whirlwind) never is. We
                # detect it from the preceding |move|...|Baton Pass (the flag) or a "[from] Baton
                # Pass" tag on the switch line itself.
                is_baton_pass = event_type == "switch" and (
                    pokemon.showdown_slot in self.pending_baton_pass or _line_mentions_baton_pass(parts)
                )
                self.pending_baton_pass.discard(pokemon.showdown_slot)
                if not is_baton_pass:
                    self.boosts[pokemon.showdown_slot] = {}
                if is_baton_pass:
                    transferred_volatiles = (
                        self.volatiles[pokemon.showdown_slot]
                        & _BATON_PASS_TRANSFERRED_VOLATILES
                    )
                    self.volatiles[pokemon.showdown_slot] = transferred_volatiles
                    self.direct_materialization_blockers[pokemon.showdown_slot].intersection_update(
                        f"baton-pass:{name}" for name in transferred_volatiles
                    )
                    unsupported = transferred_volatiles - _DIRECT_MATERIALIZATION_VOLATILES
                    if unsupported:
                        self.direct_materialization_blockers[pokemon.showdown_slot].update(
                            f"baton-pass:{name}" for name in unsupported
                        )
                    if "leechseed" not in transferred_volatiles:
                        self.leech_seed_source_sides.pop(pokemon.showdown_slot, None)
                    elif pokemon.showdown_slot not in self.leech_seed_source_sides:
                        # Preserve the fail-closed marker when an incomplete protocol prefix
                        # carried Leech Seed through Baton Pass without its public source move.
                        self.direct_materialization_blockers[pokemon.showdown_slot].add(
                            "leechseed-source-unknown"
                        )
                else:
                    # Volatile statuses are tied to the Pokemon that left the field.
                    self.volatiles[pokemon.showdown_slot] = set()
                    self.direct_materialization_blockers[pokemon.showdown_slot].clear()
                    self.leech_seed_source_sides.pop(pokemon.showdown_slot, None)
                # The existing Baton-Pass path deliberately declines to
                # materialize a passed Substitute: its HP belongs to the
                # passer and cannot be reconstructed for the recipient. Keep
                # this provenance surface aligned with that fail-closed world.
                self.substitute_health_state[pokemon.showdown_slot] = "absent"
                self.substitute_depletion[pokemon.showdown_slot] = None
                # Gen 3 resets the toxic counter when a mon leaves the field.
                self.toxic_stage[pokemon.showdown_slot] = 0
                self.toxic_stage_known[pokemon.showdown_slot] = True
                # A normal switch-in will take this turn's residual before its
                # next action. Gen 3 writes ``|upkeep`` only after the residual
                # phase, then emits the faint replacement as ``|switch|``. That
                # replacement missed the just-finished residual, so its next
                # Toxic tick has the legitimate pre-tick counter zero. A
                # ``|drag|`` is an action-phase phaze, and Baton Pass is not a
                # faint replacement; neither may manufacture this narrow proof.
                self.toxic_stage_zero_after_upkeep[pokemon.showdown_slot] = (
                    event_type == "switch"
                    and self._post_upkeep_window is True
                    and replacement_matches_pending
                    and not is_baton_pass
                    and pokemon.ident == parts[2]
                    and pokemon.showdown_slot == replacement_slot
                    and _condition_has_status(pokemon.condition, "tox")
                )
                self.toxic_stage_zero_after_upkeep_expires_after_turn[
                    pokemon.showdown_slot
                ] = (
                    self.turn_number + 1
                    if self.toxic_stage_zero_after_upkeep[pokemon.showdown_slot]
                    else None
                )
                self.toxic_stage_zero_after_upkeep_ident[pokemon.showdown_slot] = (
                    pokemon.ident
                    if self.toxic_stage_zero_after_upkeep[pokemon.showdown_slot]
                    else None
                )
                # The stall streak belongs to the mon that left the slot (the ``stall`` volatile
                # clears on switch/faint); switch-out/drag is reset cause (4). Clear the in-flight
                # flag too so no stale stall move carries onto the replacement.
                self.stall_counter[pokemon.showdown_slot] = 0
                self.stall_move_pending[pokemon.showdown_slot] = False
                # ``Pokemon.clearVolatile()`` nulls lastMove on switch-out, so the mon coming
                # in genuinely has none. Recorded as the ``switch`` sentinel rather than
                # "unknown" because the engine distinguishes them: LastUsedMove::Switch is a
                # POSITIVE fact (Encore correctly fails against a fresh switch-in), whereas
                # None means the world never knew. Collapsing the two would relabel a fact as
                # ignorance.
                self.last_used_move[pokemon.showdown_slot] = "switch"
                # Confusion turns-so-far (spec v3) belong to the mon that just left. A plain
                # switch/drag drops the confusion volatile (reset); a Baton Pass that carried
                # confusion (it is a copied volatile) keeps the counter running on the inheritor,
                # so gate the reset on the volatile being absent from the finalized slot set.
                if "confusion" not in self.volatiles[pokemon.showdown_slot]:
                    self.confusion_elapsed[pokemon.showdown_slot] = 0
                # Encore turns-so-far (spec v3 change 5) belong to the mon that just left. Encore is
                # ``noCopy: true`` (not carried by Baton Pass), so the volatile is always absent from
                # the finalized slot set after a switch/drag and this reset is unconditional; the
                # volatile-absence gate is kept parallel to the confusion reset for consistency.
                if "encore" not in self.volatiles[pokemon.showdown_slot]:
                    self.encore_elapsed[pokemon.showdown_slot] = 0
                # Wrap (partial-trap) turns-so-far (spec v3 change 6) belong to the mon that just
                # left. Unlike encore, ``partiallytrapped`` IS a Baton-Pass-copied volatile, so — as
                # with confusion — a plain switch/drag drops it (reset) while a Baton Pass that
                # carried the trap keeps the counter running on the inheritor; gate the reset on the
                # volatile being absent from the finalized slot set.
                if "partiallytrapped" not in self.volatiles[pokemon.showdown_slot]:
                    self.wrap_trap_elapsed[pokemon.showdown_slot] = 0
                # Mean Look / Spider Web move-trap (spec v3 change 8): a switch/drag of THIS slot
                # ends both directions. If the mon leaving was the TARGET, its trap is over (the
                # ``trapped`` volatile is noCopy, so it never rides a Baton Pass). If the mon leaving
                # was the TRAPPER, the trap it held on the OPPOSING active mon ends too (the linked
                # source-side volatile drops when the trapper leaves). Cleared unconditionally on
                # both slots — in singles the trapper is always the opposing active mon.
                self.meanlook_trap[pokemon.showdown_slot] = False
                self.meanlook_trap[_OTHER_SLOT[pokemon.showdown_slot]] = False
                # A live type override (Castform Forecast forme / Kecleon Color Change) belongs to
                # the mon that just left the slot: both revert to base type on switch-out, and a
                # Baton Pass brings in a DIFFERENT mon at base type, so clear it unconditionally so
                # no stale override survives onto the replacement.
                self.live_type_override[pokemon.showdown_slot] = None
                # Trace drops its copy on switch-out and re-fires on the next
                # switch-in, so the borrowed ability belongs to the mon that just
                # left. Not clearing it is what let a stale trace leak.
                self.traced_ability[pokemon.showdown_slot] = None
                # v4 pack A1: the ``mustrecharge`` volatile leaves with the mon. A recharging mon
                # cannot switch voluntarily, but it CAN be dragged out (Roar/Whirlwind) or faint
                # and be replaced, and neither the replacement nor a later occupant inherits the
                # lock. Cleared unconditionally: it is ``noCopy``, so no Baton Pass carries it.
                self.must_recharge[pokemon.showdown_slot] = False
                # v4 pack A5: last-round damage is a per-MON fact. The incoming mon dealt and took
                # nothing, so both the settled pair and the in-flight accumulators reset — a
                # switch-in must not inherit the record of the mon it replaced.
                self.last_damage_dealt[pokemon.showdown_slot] = 0.0
                self.last_damage_taken[pokemon.showdown_slot] = 0.0
                self.current_damage_dealt[pokemon.showdown_slot] = 0.0
                self.current_damage_taken[pokemon.showdown_slot] = 0.0
                # Entry-hazard chip lands on the incoming mon AFTER this switch line, so the
                # cumulative side ledger (hazard_damage_suffered) is deliberately NOT touched
                # here: it is a per-SIDE credit total for the whole game, not a per-mon counter.
                # Any move window is closed by the switch: whatever damage follows (hazard chip on
                # the way in, the next mover's strike) belongs to a different attribution.
                self._damage_window_actor = None
                # v4 pack A2 addendum: HOW this mon arrived. Read alongside the ``switch``
                # sentinel that was just written to last_used_move.
                self.arrived_by_baton_pass[pokemon.showdown_slot] = bool(is_baton_pass)
                # v4 pack A4 addendum: the choice lock and the item's provenance both belong to
                # the mon that left. ``choicelock`` is noCopy so it never rides a Baton Pass, and
                # a Tricked item stays with its holder — but that holder is gone from this slot.
                self.choice_item_public[pokemon.showdown_slot] = False
                self.choice_locked[pokemon.showdown_slot] = False
                self.item_from_trick[pokemon.showdown_slot] = False
            self.public_events.append(_public_event_from_line(line))
            self.public_lines.append(line)
            return
        if event_type == "win" and len(parts) >= 3:
            self.winner = parts[2]
            self.public_events.append(_public_event_from_line(line))
            self.public_lines.append(line)
            return
        if event_type == "upkeep":
            if not canonical_upkeep:
                return
            # Residuals for this turn are done; anything switching in from here until the
            # next |turn| is a post-residual faint replacement.
            self._post_upkeep_window = True
            self._settle_pending_rest_sleep_attempts()
        if event_type == "turn" and len(parts) >= 3:
            next_turn = canonical_turn
            turn_is_ordered = bool(
                isinstance(next_turn, int)
                and next_turn >= 1
                and (
                    (
                        self.turn_number == 0
                        and not any(self.toxic_stage_zero_after_upkeep.values())
                        and not any(self.toxic_faint_replacement_pending.values())
                    )
                    or next_turn == self.turn_number + 1
                )
            )
            if not turn_is_ordered:
                # The latch already discarded proof provenance. Do not let an
                # unordered marker mutate the parser's turn chronology.
                return
            self.turn_number = next_turn
            # A successful Baton Pass is consumed by its same-turn forced switch. Anything still
            # pending at a fresh turn belongs to a failed or truncated protocol sequence.
            self.pending_baton_pass.clear()
            self.toxic_faint_replacement_pending = {"p1": False, "p2": False}
            self._settle_pending_rest_sleep_attempts()
            # Each turn a badly-poisoned mon stays in, its toxic damage escalates
            # through 15/16. Preserve 16 as an internal sentinel after the
            # simulator has already reached its stage-15 cap: raw 15 otherwise
            # ambiguously means either "current 14, next 15" or "current 15".
            for slot, stage in self.toxic_stage.items():
                if self.toxic_stage_known[slot] and stage:
                    self.toxic_stage[slot] = min(16, stage + 1)
            # gen3 Truant: `onResidual` flips the bit every turn UNCONDITIONALLY. It is
            # deliberately NOT gated on the mon having moved, having a volatile, or anything
            # else -- that gating is exactly the proxy this replaces.
            # Turn 1 is skipped on purpose: the flip mirrors the PREVIOUS turn's residual and
            # there is no end-of-turn-0 residual. Flipping there inverts a lead's parity for
            # the whole stint.
            if self.turn_number >= 2:
                for slot, phase in self.truant_phase.items():
                    if phase is None:
                        continue
                    if slot in self._truant_skip_next_flip:
                        # Replacement guard. A holder that entered as a POST-RESIDUAL faint
                        # replacement missed nothing: that turn's `onResidual` had already
                        # run before it arrived, so flipping here would double-count it.
                        # Sim-probed: a Slaking replacing a mon that fainted at upkeep LOAFS
                        # on its first move turn, whereas a Slaking switched in as the turn's
                        # ACTION acts on its first move turn. Same seed, opposite outcome, and
                        # the only difference is which side of the residual it entered on.
                        continue
                    self.truant_phase[slot] = not phase
            self._truant_skip_next_flip.clear()
            self._post_upkeep_window = False
            # Confusion turns-so-far (spec v3 change 4): each turn the confusion volatile is
            # publicly present on a slot's active mon, its elapsed-duration counter advances.
            # Left uncapped in the raw counter (a mon asleep-while-confused can dwell past the
            # 5-turn gen3 max without the hidden move-attempt clock ticking); the encode's
            # min(1, elapsed/5) caps the emitted value.
            for slot in self.confusion_elapsed:
                if "confusion" in self.volatiles.get(slot, ()):
                    self.confusion_elapsed[slot] += 1
            # Encore turns-so-far (spec v3 change 5): each turn the encore volatile is publicly
            # present on a slot's active mon, its elapsed-duration counter advances (same per-|turn|
            # point as the toxic ramp / confusion counter). Left uncapped in the raw counter; the
            # encode's min(1, elapsed/6) caps the emitted value at the gen3 6-turn max.
            for slot in self.encore_elapsed:
                if "encore" in self.volatiles.get(slot, ()):
                    self.encore_elapsed[slot] += 1
            # Wrap (partial-trap) turns-so-far (spec v3 change 6): each turn the partiallytrapped
            # volatile is publicly present on a slot's active mon, its elapsed-duration counter
            # advances (same per-|turn| point as the toxic ramp / confusion / encore counters).
            # Left uncapped in the raw counter; the encode's min(1, elapsed/5) caps the emitted
            # value at the gen3 5-turn max.
            for slot in self.wrap_trap_elapsed:
                if "partiallytrapped" in self.volatiles.get(slot, ()):
                    self.wrap_trap_elapsed[slot] += 1
            # v4 pack A5: settle the damage ledger at the turn boundary. Everything accumulated
            # since the previous ``|turn|`` — both players' actions AND the residual phase that
            # closed the turn — becomes "the previous round", and a fresh round starts at zero.
            # This is the same per-``|turn|`` point every other elapsed counter advances at, so
            # the pack's notion of "last round" agrees with the rest of the current-state layer.
            for slot in ("p1", "p2"):
                self.last_damage_dealt[slot] = self.current_damage_dealt.get(slot, 0.0)
                self.last_damage_taken[slot] = self.current_damage_taken.get(slot, 0.0)
                self.current_damage_dealt[slot] = 0.0
                self.current_damage_taken[slot] = 0.0
            self._damage_window_actor = None
        if event_type == "-fail" and len(parts) >= 3:
            # A failed Baton Pass emits its move declaration but no switch request. Do not let
            # that declaration turn a later ordinary switch into a phantom Baton Pass.
            self.pending_baton_pass.discard(_slot_from_ident(parts[2]))
        # Re-seed the toxic ramp from the PUBLIC end-of-turn residual BEFORE the condition update
        # overwrites the pre-damage HP (needed to measure the residual's magnitude).
        self._reseed_toxic_stage_from_residual(parts)
        # v4 pack A5 / part B1: the damage ledger and the hazard-credit ledger measure magnitudes
        # the same way, so they run in the same pre-update window — the public condition still
        # holds the PRE-damage HP here, and the delta against the line's new value is the amount.
        self._update_damage_ledgers(parts, line)
        _update_public_pokemon_condition(parts, self.public_active, self.public_revealed)
        _update_side_conditions(parts, self.side_condition_counts)
        self.weather = _update_weather(parts, self.weather)
        self._update_weather_meta(parts, line)
        self._update_timed_side_conditions(parts)
        self._update_wish(parts, line)
        _update_boosts(parts, self.boosts)
        _update_volatiles(parts, self.volatiles)
        self._update_substitute_health_state(parts)
        self._update_live_type_override(parts)
        self._update_traced_ability(parts, line)
        self._anchor_truant_phase(parts, line)
        self._update_leech_seed(parts)
        self._prune_direct_materialization_blockers()
        _update_future_sight(parts, self.future_sight, self.turn_number)
        _update_toxic_stage(
            parts,
            self.toxic_stage,
            self.toxic_stage_known,
            self.toxic_stage_zero_after_upkeep,
        )
        _update_confusion_elapsed(parts, self.confusion_elapsed)
        _update_encore_elapsed(parts, self.encore_elapsed)
        _update_wrap_trap_elapsed(parts, self.wrap_trap_elapsed)
        _update_meanlook_trap(parts, self.meanlook_trap)
        _flag_baton_pass(parts, self.pending_baton_pass)
        self._update_induced_sleep(parts, line)
        self._update_stall_counter(parts)
        _update_must_recharge(parts, self.must_recharge)
        self._update_items_removed(parts, line)
        self._update_choice_lock(parts, line)
        self.public_events.append(_public_event_from_line(line))
        self.public_lines.append(line)

    def _update_toxic_faint_replacement_latch(
        self,
        event_type: str,
        parts: Sequence[str],
        *,
        canonical_turn: int | None = None,
        canonical_upkeep: bool = False,
        canonical_faint: bool = False,
        canonical_replacement: bool = False,
    ) -> None:
        """Bind the stage-zero exception to one exact, ordered forced replacement."""

        def clear_zero_proof(slot: str) -> None:
            self.toxic_stage_zero_after_upkeep[slot] = False
            self.toxic_stage_zero_after_upkeep_expires_after_turn[slot] = None
            self.toxic_stage_zero_after_upkeep_ident[slot] = None

        def clear_pending(slot: str) -> None:
            self.toxic_faint_replacement_pending[slot] = False
            self.toxic_faint_replacement_expected_ident[slot] = None

        def invalidate_window(slot: str) -> None:
            clear_pending(slot)
            clear_zero_proof(slot)
            self.toxic_faint_replacement_invalid[slot] = True

        def clear_all_pending() -> None:
            for slot in self.toxic_faint_replacement_pending:
                clear_pending(slot)

        def invalidate_all_windows() -> None:
            for slot in ("p1", "p2"):
                invalidate_window(slot)

        if event_type == "turn":
            # A missing replacement is a truncated/terminal sequence, not
            # evidence for an ordinary switch on a later turn.
            clear_all_pending()
            next_turn = canonical_turn
            if next_turn is None:
                # Without an ordered turn boundary, a durable proof cannot be
                # bounded to its one expected residual opportunity.
                for slot in self.toxic_stage_zero_after_upkeep:
                    invalidate_window(slot)
                return
            turn_is_ordered = bool(
                next_turn >= 1
                and (
                    (
                        self.turn_number == 0
                        and not any(
                            proof is True
                            for proof in self.toxic_stage_zero_after_upkeep.values()
                        )
                        and not any(
                            pending is True
                            for pending in self.toxic_faint_replacement_pending.values()
                        )
                    )
                    or next_turn == self.turn_number + 1
                )
            )
            if not turn_is_ordered:
                for slot in self.toxic_stage_zero_after_upkeep:
                    invalidate_window(slot)
                return
            # A strictly ordered turn starts a clean replacement window. An
            # earlier malformed faint cannot poison subsequent real history.
            self.toxic_faint_replacement_invalid = {"p1": False, "p2": False}
            for slot, proof in self.toxic_stage_zero_after_upkeep.items():
                deadline = self.toxic_stage_zero_after_upkeep_expires_after_turn.get(slot)
                if proof is True and (type(deadline) is not int or next_turn != deadline):
                    clear_zero_proof(slot)
            return
        if event_type == "upkeep":
            if not canonical_upkeep:
                # A look-alike marker has no residual chronology.  It also
                # cannot coexist with a one-shot proof for either seat.
                invalidate_all_windows()
                return
            # A proof created after the prior upkeep expects its first Toxic
            # residual immediately before this marker. If it is still live,
            # that residual was absent and the proof is spent.
            for slot, proof in self.toxic_stage_zero_after_upkeep.items():
                if proof is True:
                    clear_zero_proof(slot)
            if self._post_upkeep_window is True:
                # A second upkeep cannot follow the one faint that is still
                # awaiting its replacement; reject the malformed chronology.
                for slot in self.toxic_faint_replacement_pending:
                    invalidate_window(slot)
            return
        if event_type == "faint":
            if not canonical_faint:
                invalidate_all_windows()
                return
            slot = _slot_from_ident(parts[2])
            if slot not in self.toxic_faint_replacement_pending:
                return
            active = self.public_active.get(slot)
            if (
                self._post_upkeep_window is False
                and _is_active_protocol_ident(parts[2])
                and _is_current_public_active(active)
                and getattr(active, "ident", None) == parts[2]
                and self.toxic_faint_replacement_pending[slot] is False
                and self.toxic_faint_replacement_invalid[slot] is False
            ):
                self.toxic_faint_replacement_pending[slot] = True
                self.toxic_faint_replacement_expected_ident[slot] = parts[2]
            else:
                # Reversed, repeated, or forged active idents are not proof
                # that this seat is awaiting a forced replacement. Keep the
                # invalidity until a clean turn so a third duplicate cannot
                # re-arm this replacement window.
                invalidate_all_windows()
            return
        if event_type in {"switch", "drag", "replace"}:
            # The replacement branch consumes this before parsing its payload,
            # including malformed lines.
            if not canonical_replacement:
                invalidate_all_windows()
            return
        if event_type == "win":
            clear_all_pending()
            for slot in self.toxic_stage_zero_after_upkeep:
                clear_zero_proof(slot)
            return
        if len(parts) >= 3:
            slot = _slot_from_ident(parts[2])
            if (
                slot in self.toxic_faint_replacement_pending
                and self.toxic_faint_replacement_pending[slot] is True
            ):
                # A same-seat state transition cannot belong to a pending
                # forced replacement; fail closed rather than retain history.
                invalidate_window(slot)

    def _update_substitute_health_state(self, parts: Sequence[str]) -> None:
        """Track canonical Substitute provenance and public exact HP cases."""

        if len(parts) < 3:
            return
        slot = _slot_from_ident(parts[2])
        if slot not in self.substitute_health_state:
            return
        event_type = parts[1]
        if event_type == "faint":
            # The force-switch snapshot is taken after this line. Retaining the
            # fainted mon's Substitute would construct a phantom active effect.
            self.volatiles[slot].discard("substitute")
            self.substitute_health_state[slot] = "absent"
            self.substitute_depletion[slot] = None
            return
        if len(parts) < 4 or _side_condition_identifier(parts[3]) != "substitute":
            return
        if event_type == "-start":
            self.substitute_health_state[slot] = "full"
            self.substitute_depletion[slot] = 0
            return
        if event_type == "-end":
            self.substitute_health_state[slot] = "broken"
            self.substitute_depletion[slot] = None
            return
        if event_type != "-activate":
            return

        # A non-breaking hit normally reveals only that the Substitute lived.
        # The immediately preceding move is sufficient only for these four
        # public, deterministic Gen 3 fixed-damage moves.
        damage = (
            self._fixed_substitute_damage_from_previous_move(slot)
            if any(part.strip() == "[damage]" for part in parts[4:])
            else None
        )
        current = self.substitute_depletion.get(slot)
        if damage is not None and current is not None and damage > 0:
            self.substitute_health_state[slot] = "exact"
            self.substitute_depletion[slot] = current + damage
        else:
            self.substitute_health_state[slot] = "unknown"
            self.substitute_depletion[slot] = None

    def _fixed_substitute_damage_from_previous_move(self, target_slot: str) -> int | None:
        if not self.public_lines:
            return None
        previous = self.public_lines[-1].split("|")
        if (
            len(previous) < 5
            or previous[1] != "move"
            or _slot_from_ident(previous[4]) != target_slot
        ):
            return None
        move = _normalize_identifier(previous[3])
        if move == "dragonrage":
            return 40
        if move == "sonicboom":
            return 20
        if move not in {"seismictoss", "nightshade"}:
            return None
        attacker = self.public_active.get(_slot_from_ident(previous[2]))
        return _level_from_details(attacker.details) if attacker is not None else None

    def _reseed_toxic_stage_from_residual(self, parts: Sequence[str]) -> None:
        """Recover the badly-poisoned (tox) ramp stage from the PUBLIC end-of-turn toxic residual.

        A ``tox`` mon that switches out has its counter reset to 0 (Gen 3, ``tox.onSwitchIn`` sets
        ``effectState.stage = 0``); on re-entry the ``tox`` rides only the switch-line condition
        string with no fresh ``|-status|``, so ``_update_toxic_stage`` never re-seeds and the
        per-``|turn|`` escalation (gated on ``if stage``) can never lift it off 0 — the encoder
        would emit the contradictory ``status:tox`` + ``toxic_stage == 0`` for the whole stint.

        Exact-HP streams can derive the counter from Gen 3's floored damage unit. Percentage
        streams cannot reverse-round an arbitrary stage, but a public switch/drag reset proves
        that the first subsequent Toxic residual is stage one. Unknown stream provenance fails
        closed instead of treating a /100 denominator as either representation. Regular
        (non-badly) poison also emits ``[from] psn`` but is gated out by the residual's ``tox``
        status token.
        """

        if (parts[1] if len(parts) > 1 else "") != "-damage" or len(parts) < 4:
            return
        # The tox clock's residual is tagged exactly ``[from] psn`` (no ``[of]`` source field).
        if not any(field.strip() == "[from] psn" for field in parts[4:]):
            return
        slot = _slot_from_ident(parts[2])
        if slot not in self.toxic_stage:
            return
        new_condition = parts[3]
        # Only a BADLY-poisoned residual ramps; a plain ``psn`` residual carries no ``tox`` token.
        if "tox" not in new_condition.split():
            return
        # This public residual consumes the one deferred first tick of a
        # post-upkeep replacement. It is no longer evidence for materializing
        # a stage-zero world, even if later exact recovery is impossible.
        self.toxic_stage_zero_after_upkeep[slot] = False
        active = self.public_active.get(slot)
        prev_condition = (
            getattr(active, "condition", None)
            if _is_current_public_active(active) and getattr(active, "ident", None) == parts[2]
            else None
        )
        prev_hp, prev_max = _hp_numerator_denominator(prev_condition)
        cur_hp, cur_max = _hp_numerator_denominator(new_condition)
        max_hp = prev_max or cur_max
        if prev_hp is None or cur_hp is None or cur_hp <= 0 or not max_hp:
            return
        damage = prev_hp - cur_hp
        if damage <= 0:
            return
        # Percentage-form protocol always uses /100. Any other denominator is therefore exact;
        # only /100 needs stream/request provenance to distinguish representation from a real
        # exact-100 HP Pokemon.
        visibility = "exact" if max_hp != 100 else self._hp_visibility_for_slot(slot)
        if visibility == "percentage":
            if self.toxic_stage_known[slot] and self.toxic_stage[slot] == 0:
                self.toxic_stage[slot] = 1
            return
        if visibility != "exact":
            if self.toxic_stage[slot] == 0:
                self.toxic_stage_known[slot] = False
            return
        unit = max(1, max_hp // 16)
        # A surviving exact-HP Toxic residual is a whole number of Gen 3 units. Do not infer a
        # hidden stage from capped or otherwise non-exact public damage.
        if damage % unit:
            if self.toxic_stage[slot] == 0:
                self.toxic_stage_known[slot] = False
            return
        stage = damage // unit
        if not 1 <= stage <= 15:
            if self.toxic_stage[slot] == 0:
                self.toxic_stage_known[slot] = False
            return
        self.toxic_stage[slot] = stage
        self.toxic_stage_known[slot] = True

    def _hp_visibility_for_slot(self, slot: str) -> str:
        """Resolve HP representation from explicit stream provenance or private requests."""

        explicit = self.hp_visibility.get(slot, "unknown")
        if explicit != "unknown":
            return explicit
        request_slots = {side for side in self.requests if side in {"p1", "p2"}}
        if len(request_slots) == 2 or slot in request_slots:
            return "exact"
        if len(request_slots) == 1:
            # A single private request identifies a player-perspective stream: own HP is exact,
            # while the opposing side's public HP is rounded to /100.
            return "percentage"
        return "unknown"

    def _prune_direct_materialization_blockers(self) -> None:
        """Keep Baton Pass blockers only while their public volatile still exists."""

        for slot, blockers in self.direct_materialization_blockers.items():
            has_unknown_leech_seed_source = (
                "leechseed-source-unknown" in blockers and "leechseed" in self.volatiles[slot]
            )
            active_markers = {f"baton-pass:{name}" for name in self.volatiles[slot]}
            blockers.intersection_update(active_markers)
            if "leechseed" not in self.volatiles[slot]:
                self.leech_seed_source_sides.pop(slot, None)
            elif has_unknown_leech_seed_source:
                blockers.add("leechseed-source-unknown")

    def _update_leech_seed(self, parts: Sequence[str]) -> None:
        """Track the public source side needed to reconstruct an active Leech Seed.

        The ``|-start|...|move: Leech Seed`` line names the target but not the source. Its
        preceding public ``|move|`` declaration does, so record the source until the start line
        confirms that the move hit. The simulator resolves the source through its active slot,
        which intentionally continues to work after that side switches.
        """

        event_type = parts[1] if len(parts) > 1 else ""
        if event_type == "move" and len(parts) >= 5 and _normalize_identifier(parts[3]) == "leechseed":
            source_slot = _slot_from_ident(parts[2])
            target_slot = _slot_from_ident(parts[4])
            if source_slot in {"p1", "p2"} and target_slot in {"p1", "p2"} and source_slot != target_slot:
                self._pending_leech_seed_source_sides[target_slot] = source_slot
            return
        if len(parts) < 4:
            return
        target_slot = _slot_from_ident(parts[2])
        if target_slot not in {"p1", "p2"} or _side_condition_identifier(parts[3]) != "leechseed":
            return
        if event_type == "-start":
            source_slot = self._pending_leech_seed_source_sides.pop(target_slot, None)
            if source_slot in {"p1", "p2"} and source_slot != target_slot:
                self.leech_seed_source_sides[target_slot] = source_slot
                self.direct_materialization_blockers[target_slot].discard("leechseed-source-unknown")
            else:
                self.leech_seed_source_sides.pop(target_slot, None)
                self.direct_materialization_blockers[target_slot].add("leechseed-source-unknown")
        elif event_type == "-end":
            self.leech_seed_source_sides.pop(target_slot, None)
            self.direct_materialization_blockers[target_slot].discard("leechseed-source-unknown")

    def _update_weather_meta(self, parts: Sequence[str], line: str) -> None:
        """Track the current weather's set turn + ability source from |-weather| lines.

        A ``[upkeep]``-tagged line continues the existing weather (set turn/source unchanged);
        a fresh ``|-weather|<id>|`` line (re)sets them; ``none`` clears them. Ability-sourced
        weather (``[from] ability:`` — Drizzle/Drought/Sand Stream) is permanent in gen 3;
        move weather runs exactly 5 turns (no extension items exist in gen 3).
        """
        if (parts[1] if len(parts) > 1 else "") != "-weather":
            return
        identifier = _normalize_identifier(parts[2].strip() if len(parts) > 2 else "")
        if not identifier or identifier == "none":
            self.weather_set_turn = None
            self.weather_from_ability = False
            self.weather_upkeeps = 0
            return
        if "[upkeep]" in line:
            # Each end-of-turn upkeep consumes one move-weather duration tick, mirroring
            # Showdown's weatherState.duration countdown. The first tick fires at the END of the
            # set turn (before the next request), so the first post-resolution observation must
            # already reflect it — otherwise the counter reads one turn stale (audit #9).
            self.weather_upkeeps += 1
            return
        self.weather_set_turn = self.turn_number
        self.weather_from_ability = "[from] ability:" in line
        self.weather_upkeeps = 0

    def _update_timed_side_conditions(self, parts: Sequence[str]) -> None:
        """Record the set turn of the deterministic 5-turn side conditions per side."""
        event_type = parts[1] if len(parts) > 1 else ""
        if event_type not in {"-sidestart", "-sideend"} or len(parts) < 4:
            return
        slot = _slot_from_ident(parts[2])
        if slot not in self.side_condition_set_turns:
            return
        condition = _side_condition_identifier(parts[3])
        if condition not in _TIMED_SIDE_CONDITIONS:
            return
        if event_type == "-sidestart":
            self.side_condition_set_turns[slot][condition] = self.turn_number
        else:
            self.side_condition_set_turns[slot].pop(condition, None)

    def _update_wish(self, parts: Sequence[str], line: str) -> None:
        """Track pending Wish per side: set on the |move| declaration, cleared when it lands.

        The landing heal arrives ``[from] move: Wish`` on the slot occupant end of the NEXT
        turn (a full-HP landing emits no heal and simply expires via the turn arithmetic in
        ``_wish_pending``). The heal-line clear covers mid-turn observations between the
        landing and the next |turn| boundary.
        """
        event_type = parts[1] if len(parts) > 1 else ""
        if event_type == "move" and len(parts) >= 4:
            slot = _slot_from_ident(parts[2])
            if slot in {"p1", "p2"} and _normalize_identifier(parts[3]) == "wish":
                # A Wish declared while one is already pending FAILS in gen 3; re-arming here
                # would wrongly extend the pending bit by a turn on a double-click.
                existing = self.wish_set_turns.get(slot)
                if existing is None or (self.turn_number - existing) > 1:
                    self.wish_set_turns[slot] = self.turn_number
            return
        if event_type in {"-heal", "-sethp"} and len(parts) > 2 and "[from] move: Wish" in line:
            slot = _slot_from_ident(parts[2])
            if slot is not None:
                self.wish_set_turns.pop(slot, None)

    def _slot_holds_truant(self, slot: str) -> bool:
        """Whether this slot's active mon has Truant, natively or by Trace.

        Species is decisive for the native case: `slakoth` and `slaking` are the only gen3
        Truant lines and both are mono-ability, so no reveal is needed.
        """
        if self.traced_ability.get(slot) == "truant":
            return True
        active = self.public_active.get(slot)
        species = getattr(active, "species", None) if active is not None else None
        return _normalize_identifier(species or "") in _TRUANT_SPECIES

    def _anchor_truant_phase(self, parts: Sequence[str], line: str) -> None:
        """Correct the Truant parity against the two facts the sim publishes.

        The switch-in seed plus the per-``|turn|`` flip reproduce gen3's toggle in principle,
        but they are a DERIVATION: they depend on having seen the switch-in, on the turn
        number being what we think, and on the flip landing on the right side of the residual.
        Any one of those being off inverts the parity for the rest of the stint, silently,
        because a boolean that is wrong looks exactly like a boolean that is right.

        These two lines are ground truth for the turn they appear in, so they are used as
        anchors rather than as evidence:

        * ``|cant|<mon>|ability: Truant``  -> the holder is LOAFING this turn;
        * ``|move|<mon>|...``              -> the holder ACTED this turn, so it is not loafing.

        The anchor sets the value for the CURRENT turn; the next ``|turn|`` flip carries it
        forward. A derivation that agrees is confirmed, and one that has drifted is corrected
        at the first public evidence instead of staying wrong until the mon switches out.
        """
        event_type = parts[1] if len(parts) > 1 else ""
        if event_type not in {"cant", "move"} or len(parts) < 3:
            return
        slot = _slot_from_ident(parts[2])
        if slot not in self.truant_phase:
            return
        if not self._slot_holds_truant(slot):
            return
        if event_type == "cant":
            if "Truant" in line:
                self.truant_phase[slot] = True
            return
        # A called move (Sleep Talk's callee) is not an independent action; the caller's own
        # ``|move|`` line already anchored the turn.
        if any(part.startswith("[from]") for part in parts[4:]):
            return
        self.truant_phase[slot] = False

    def _update_traced_ability(self, parts: Sequence[str], line: str) -> None:
        """Track the ability the active mon is CURRENTLY borrowing via Trace.

        Showdown announces the copy publicly::

            |-ability|p1a: Gardevoir|Insomnia|Trace|[from] ability: Trace|[of] p2a: Noctowl

        where the payload at index 3 is the ability being COPIED. The ``[from] ability: Trace``
        tag is the discriminator: a bare ``|-ability|`` line is an ordinary reveal of a mon's
        own ability, which is persistent and belongs in the belief engine, not here.

        Cleared on switch-out (see the switch block) because the copy does not survive it.
        """
        event_type = parts[1] if len(parts) > 1 else ""
        if event_type != "-ability" or len(parts) < 4:
            return
        if "ability: Trace" not in line:
            return
        slot = _slot_from_ident(parts[2])
        if slot not in self.traced_ability:
            return
        copied = parts[3].strip()
        if copied:
            normalized = _normalize_identifier(copied)
            self.traced_ability[slot] = normalized
            if normalized == "truant":
                # Do not derive a boolean phase from the Trace line. Current-source probes show
                # that publicly similar pre-upkeep acquisitions can start on opposite phases:
                # a one-sided action switch (3400443/2) loafs next turn, while a simultaneous
                # Porygon2/Slaking switch (2200291/41) acts. Event-queue membership, not just
                # line position, decides whether copied Truant receives that residual.
                #
                # Preserve only the public holder fact. The first own move or Truant ``cant``
                # line anchors the phase exactly, after which normal public turn flips apply.
                # This is the measured Z13.3 withdrawal: seeding here fixed selected rows but
                # created 2200291/41, while leaving it unknown created no new identity.
                self.truant_phase[slot] = None
                self._truant_skip_next_flip.discard(slot)
            elif self.truant_phase.get(slot) is not None:
                # Traced something else: the mon is no longer a Truant holder.
                self.truant_phase[slot] = None

    def _update_live_type_override(self, parts: Sequence[str]) -> None:
        """Track the active mon's LIVE type for retypes the species token cannot express.

        Two gen3 in-battle retypes are mono-type and revert on switch-out:
        - ``|-formechange|<ident>|<forme>|...`` — Castform Forecast (Sunny->Fire, Rainy->Water,
          Snowy->Ice, weather-clear->base Normal). Stored UNRESOLVED as ``forme:<forme>`` (the
          forme's type is resolved from the dex at encode time; a ``-formechange`` back to the
          base forme clears the override).
        - ``|-start|<ident>|typechange|<type>|...`` — Kecleon Color Change (payload IS the new
          type). Stored as ``type:<type>``; a matching ``|-end|<ident>|typechange`` clears it.

        Switch-out/drag clearing is handled in the switch block (both effects revert on leaving
        the field, and a Baton Pass brings in a different mon at base type).
        """
        event_type = parts[1] if len(parts) > 1 else ""
        if event_type == "-formechange" and len(parts) >= 4:
            slot = _slot_from_ident(parts[2])
            if slot not in self.live_type_override:
                return
            forme = parts[3].strip()
            active = self.public_active.get(slot)
            base_species = active.species if active is not None else _species_from_ident(parts[2])
            if _normalize_identifier(forme) == _normalize_identifier(base_species or ""):
                # Reverted to the base forme (Forecast drops the forme when weather clears).
                self.live_type_override[slot] = None
            else:
                self.live_type_override[slot] = f"forme:{forme}"
            return
        if event_type == "-start" and len(parts) >= 5 and _normalize_identifier(parts[3]) == "typechange":
            slot = _slot_from_ident(parts[2])
            if slot not in self.live_type_override:
                return
            type_payload = parts[4].strip()
            if type_payload:
                self.live_type_override[slot] = f"type:{type_payload}"
            return
        if event_type == "-end" and len(parts) >= 4 and _normalize_identifier(parts[3]) == "typechange":
            slot = _slot_from_ident(parts[2])
            if slot in self.live_type_override:
                self.live_type_override[slot] = None

    @staticmethod
    def _induced_sleep_victim_key(victim_slot: str, ident: str) -> str:
        """Stable per-mon victim key: side + the ident's (nick)name, normalized.

        The ident NAME (not the species) keys the victim because the clearing lines use it
        too — including Heal Bell's benched ``|-curestatus|p2: Name|slp|[silent]`` form,
        whose position-less ident cannot be species-resolved through ``public_active``.
        Showdown nicknames are unique per team, so the key is collision-free per side.
        """
        return f"{victim_slot}:{_normalize_identifier(_species_from_ident(ident))}"

    def _update_induced_sleep(self, parts: Sequence[str], line: str) -> None:
        """Public sleep-clause tracker (spec v3 change 2, docs/observation_v3_spec.md).

        Attribution rule (no move-window bookkeeping needed): in gen3 singles, sleep is only
        ever (a) induced by the opposing side's move or (b) self-inflicted Rest, and Rest tags
        its status line (``|-status|SLOT|slp|[from] move: Rest``) — so a ``-status … slp``
        line WITHOUT the Rest tag was induced by the opposing side. The tracked victim clears
        when it wakes (``-curestatus … slp``) or faints; switching out does NOT clear (sleep
        persists and is public on revealed mons — Natural Cure resolves via the same
        ``-curestatus`` line). Deliberately NO ability exclusion, per the spec: Showdown's
        Sleep Clause Mod counts any non-ally-sourced sleep, Effect Spore included.
        """
        event_type = parts[1] if len(parts) > 1 else ""
        if event_type == "-status" and len(parts) >= 4 and parts[3].strip() == "slp":
            victim_slot = _slot_from_ident(parts[2])
            if victim_slot in {"p1", "p2"}:
                key = self._induced_sleep_victim_key(victim_slot, parts[2])
                if "move: Rest" in line:
                    # Rest-inflicted: start the attempt counter. Absence from the opposing
                    # side's victim set already means "not induced", but the count is what
                    # lets the world builder rebuild rest_turns instead of guessing.
                    self.rest_sleep_counts[key] = 0
                    self.rest_sleep_refunded_turns.pop(key, None)
                    self.rest_sleep_skipped_turns.pop(key, None)
                    self.rest_sleep_pending_attempt.pop(key, None)
                else:
                    inducing_slot = opponent_showdown_slot(victim_slot)
                    self.induced_sleep_victims[inducing_slot].add(key)
            return
        if event_type == "cant" and len(parts) >= 4 and parts[3].strip() == "slp":
            # The ONLY public line emitted on an attempt that actually ticks the sleep timer
            # (gen3 slp.onBeforeMove). Benched turns emit nothing, which is precisely why the
            # counter tracks attempts rather than elapsed turns.
            victim_slot = _slot_from_ident(parts[2])
            if victim_slot in {"p1", "p2"}:
                key = self._induced_sleep_victim_key(victim_slot, parts[2])
                if key in self.rest_sleep_counts:
                    # Keep the raw attempt count until a later switch-in applies a public
                    # skippedTime refund. The third ordinary Rest attempt wakes and clears
                    # before a fourth count could be observed; malformed prefixes are
                    # rejected by the materialization range check rather than clamped.
                    self.rest_sleep_counts[key] += 1
                    self.rest_sleep_pending_attempt[key] = True
            return
        if event_type == "cant" and len(parts) >= 4:
            # ``slp.onBeforeMove`` has priority 10. A later same-actor cant (flinch,
            # Truant, paralysis, or Attract) proves a sleepUsable move crossed that
            # handler: an ordinary selected move would have stopped at sleep. Gen 3
            # incremented skippedTime before this later gate, even without |move|.
            actor_slot = _slot_from_ident(parts[2])
            if actor_slot in {"p1", "p2"}:
                self._mark_pending_rest_sleep_refundable(
                    self._induced_sleep_victim_key(actor_slot, parts[2])
                )
            return
        if event_type == "-activate" and len(parts) >= 4:
            # Confusion and Attract announce their lower-priority check before the
            # branch that decides whether the Pokemon can move. Either branch is
            # already enough to prove skippedTime was incremented.
            actor_slot = _slot_from_ident(parts[2])
            activation = _normalize_identifier(parts[3])
            if actor_slot in {"p1", "p2"} and activation in {"confusion", "moveattract"}:
                self._mark_pending_rest_sleep_refundable(
                    self._induced_sleep_victim_key(actor_slot, parts[2])
                )
            return
        if event_type == "move" and len(parts) >= 4:
            # Sleep Talk and Snore are the only moves a sleeping mon can act with, and gen3
            # REFUNDS their trailing run on the next switch-in:
            #
            #     slp.onSwitchIn: this.effectState.time += this.effectState.skippedTime
            #
            # The preceding ``|cant|`` is the public timer decrement. Mark it refundable only
            # when this is that same Rest sleeper's direct move, never for a called move.
            actor_slot = _slot_from_ident(parts[2])
            if actor_slot in {"p1", "p2"} and _normalize_identifier(parts[3]) in _SLEEP_USABLE_MOVES:
                key = self._induced_sleep_victim_key(actor_slot, parts[2])
                if self.rest_sleep_pending_attempt.pop(key, False):
                    self.rest_sleep_skipped_turns[key] = (
                        self.rest_sleep_skipped_turns.get(key, 0) + 1
                    )
            return
        if event_type == "-cureteam" and len(parts) >= 3:
            # Aromatherapy cures every living team member with a SINGLE ``|-cureteam|SOURCE``
            # line and NO per-mon ``-curestatus`` (gen3 inherits the gen4 mod's silent
            # clearStatus). The wake is still public, so clear every tracked victim on the
            # cured (actor's) side — the spec's clear-on-wake rule through the only line the
            # protocol emits for it.
            cured_slot = _slot_from_ident(parts[2])
            if cured_slot in {"p1", "p2"}:
                prefix = f"{cured_slot}:"
                for victims in self.induced_sleep_victims.values():
                    for key in [key for key in victims if key.startswith(prefix)]:
                        victims.discard(key)
                for key in [k for k in self.rest_sleep_counts if k.startswith(prefix)]:
                    self._clear_rest_sleep_state(key)
            return
        clearing = (
            event_type == "-curestatus" and len(parts) >= 4 and parts[3].strip() == "slp"
        ) or (event_type == "faint" and len(parts) >= 3)
        if clearing:
            victim_slot = _slot_from_ident(parts[2])
            if victim_slot in {"p1", "p2"}:
                key = self._induced_sleep_victim_key(victim_slot, parts[2])
                for victims in self.induced_sleep_victims.values():
                    victims.discard(key)
                self._clear_rest_sleep_state(key)

    def _mark_pending_rest_sleep_refundable(self, key: str) -> None:
        """Move a publicly proven sleepUsable attempt into skippedTime."""

        if self.rest_sleep_pending_attempt.pop(key, False):
            self.rest_sleep_skipped_turns[key] = (
                self.rest_sleep_skipped_turns.get(key, 0) + 1
            )

    def _settle_pending_rest_sleep_attempts(self) -> None:
        """Finish public sleep attempts that did not resolve through Sleep Talk/Snore.

        A ``|cant|...|slp`` line is emitted before the possible direct sleep-usable move.
        At upkeep (or the next turn in a compact replay) a still-pending attempt was an
        ordinary sleep turn, which clears Showdown's trailing ``skippedTime`` run.
        """

        for key in tuple(self.rest_sleep_pending_attempt):
            if self.rest_sleep_pending_attempt.pop(key, False):
                self.rest_sleep_skipped_turns.pop(key, None)

    def _refund_rest_sleep_on_switch(self, pokemon: ShowdownPokemon) -> None:
        """Apply Gen 3's public Sleep Talk/Snore refund when a Rest sleeper re-enters."""

        slot = pokemon.showdown_slot
        if slot not in {"p1", "p2"}:
            return
        key = self._induced_sleep_victim_key(slot, pokemon.ident)
        if key not in self.rest_sleep_counts:
            return
        if "slp" not in str(pokemon.condition or "").split():
            # A wake/cure line normally clears this first. If a truncated stream supplies an
            # awake switch without it, dropping the stale provenance is the only safe choice.
            self._clear_rest_sleep_state(key)
            return
        skipped = self.rest_sleep_skipped_turns.pop(key, 0)
        self.rest_sleep_pending_attempt.pop(key, None)
        if isinstance(skipped, bool) or not isinstance(skipped, int) or skipped < 0:
            self._clear_rest_sleep_state(key)
            return
        refunded = self.rest_sleep_refunded_turns.get(key, 0)
        if (
            isinstance(refunded, bool)
            or not isinstance(refunded, int)
            or refunded < 0
            or refunded + skipped > self.rest_sleep_counts[key]
        ):
            self._clear_rest_sleep_state(key)
            return
        self.rest_sleep_refunded_turns[key] = refunded + skipped

    def _clear_rest_sleep_state(self, key: str) -> None:
        self.rest_sleep_counts.pop(key, None)
        self.rest_sleep_refunded_turns.pop(key, None)
        self.rest_sleep_skipped_turns.pop(key, None)
        self.rest_sleep_pending_attempt.pop(key, None)

    def _update_stall_counter(self, parts: Sequence[str]) -> None:
        """Public consecutive-stall counter (spec v3 change 3, docs/observation_v3_spec.md).

        One per-side counter = consecutive SUCCESSFUL stall-move uses (Protect/Detect/Endure —
        gen3 shares a single ``stall`` volatile across all three; engine ground truth
        ``data/conditions.ts:439-462``, where ``onStallMove`` deletes the volatile on a
        ``randomChance`` failure, so the counter is a consecutive-success streak) by that side's
        currently-active mon. Reproduces that semantics from PUBLIC lines only:

        - INCREMENT on the success-only ``-singleturn`` tag. Protect/Detect share
          ``volatileStatus: 'protect'`` -> ``|-singleturn|SLOT|Protect``; Endure ->
          ``|-singleturn|SLOT|move: Endure``. These fire ONLY on success (a failed stall emits
          ``-fail`` and no ``-singleturn``). Focus Punch / Magic Coat / Snatch also use
          ``-singleturn`` but normalize to other names and are excluded.
        - RESET to 0 on the five public mirrors of the engine's volatile deletion: (1) a
          ``-fail`` closing a stall move's action window (``stall_move_pending`` set by that
          move's ``|move|`` line); (2) any non-stall ``|move|`` by the mon; (3) ``cant``;
          (4) switch-out/drag (handled in the switch block, mirroring the toxic-stage reset);
          (5) faint.
        """
        event_type = parts[1] if len(parts) > 1 else ""
        if event_type == "-singleturn" and len(parts) >= 4:
            slot = _slot_from_ident(parts[2])
            if slot in self.stall_counter and _is_stall_singleturn(parts[3]):
                self.stall_counter[slot] = min(_STALL_COUNTER_CAP, self.stall_counter[slot] + 1)
                self.stall_move_pending[slot] = False
            return
        if event_type == "move" and len(parts) >= 4:
            slot = _slot_from_ident(parts[2])
            if slot in self.last_used_move:
                # A ``|move|`` line IS the public mirror of ``Pokemon.moveUsed()`` -- with one
                # exception. A move CALLED by another (Sleep Talk's callee, and Metronome /
                # Mirror Move where reachable) runs through ``useMove``, which never touches
                # lastMove; only the caller records. Showdown tags the callee's line with
                # ``[from]``, so that tag is the discriminator. Getting this backwards would
                # make Encore lock the CALLED move -- the exact inversion the engine-side
                # lastmove-semantics patch called out as the naive mistake.
                called_by_another_move = any(part.startswith("[from]") for part in parts[4:])
                if not called_by_another_move:
                    self.last_used_move[slot] = _normalize_identifier(parts[3])
            if slot in self.stall_counter:
                if _normalize_identifier(parts[3]) in _STALL_MOVE_IDS:
                    # A stall move is in flight; its ``-singleturn`` (success) or ``-fail``
                    # (failure) resolves the counter. Do NOT reset here — that would zero a
                    # climbing streak on every successful consecutive Protect.
                    self.stall_move_pending[slot] = True
                else:
                    # Any non-stall move breaks the consecutive-stall streak (reset cause 2).
                    self.stall_counter[slot] = 0
                    self.stall_move_pending[slot] = False
            return
        if event_type == "-fail" and len(parts) >= 3:
            slot = _slot_from_ident(parts[2])
            if slot in self.stall_counter and self.stall_move_pending.get(slot):
                # The in-flight stall move failed (the ``randomChance`` miss that deletes the
                # ``stall`` volatile) — reset cause (1).
                self.stall_counter[slot] = 0
                self.stall_move_pending[slot] = False
            return
        if event_type in {"cant", "faint"} and len(parts) >= 3:
            slot = _slot_from_ident(parts[2])
            if slot in self.stall_counter:
                # cant / faint (reset causes 3 and 5).
                self.stall_counter[slot] = 0
                self.stall_move_pending[slot] = False

    def _update_damage_ledgers(self, parts: Sequence[str], line: str) -> None:
        """Per-mon last-round damage (v4 pack A5) + per-side hazard credit (v4 part B1).

        Both read the magnitude of a ``-damage`` line the same way, and both must read it BEFORE
        ``_update_public_pokemon_condition`` overwrites the pre-damage HP — the same ordering the
        toxic-stage reseed relies on. The magnitude is a fraction of the struck mon's max HP,
        which is exactly what the condition head gives on either stream form (exact ``170/362`` or
        the opponent's rounded ``47/100``).

        ATTRIBUTION, transcribed from the transitions fold's rules so the current-state pack and
        the history region cannot disagree about who did what:

        * DEALT is move damage only. An UNTAGGED ``-damage`` on the slot OPPOSITE the open move
          window's actor is that actor's strike. Every other damage surface carries a ``[from]``
          tag (residuals, hazards, recoil, items, drain) or has no window at all.
        * The window is opened by a ``|move|`` line and closed by anything proving the next damage
          is not the actor's strike: a confusion self-hit marker, a ``cant``, a switch, a turn
          boundary. A slower confused mon self-hits with an UNTAGGED ``-damage`` and NO move line
          of its own, so without the ``|-activate|SLOT|confusion`` latch that self-damage would be
          credited to whoever moved first — the one attribution error this surface can make.
        * TAKEN is total: every ``-damage`` on the mon counts, tagged or not. That is what makes
          the pair non-redundant — DEALT is move-attributed, TAKEN includes the chip.

        Self-damage (Substitute, Belly Drum, recoil, crash) lands on the ACTOR's own slot, so it
        never reaches DEALT (which requires the opposite slot) but does reach that mon's TAKEN,
        which is correct: it lost the HP.
        """

        event_type = parts[1] if len(parts) > 1 else ""
        # The confusion self-hit latch: ``|-activate|SLOT|confusion`` immediately precedes the
        # untagged self-damage. Closing the window here is what keeps that damage out of the
        # previous mover's DEALT column (spec v3 change 10 documents the same protocol shape).
        if (
            event_type == "-activate"
            and len(parts) >= 4
            and _side_condition_identifier(parts[3]) == "confusion"
        ):
            self._damage_window_actor = None
            return
        if event_type == "move" and len(parts) >= 4:
            slot = _slot_from_ident(parts[2])
            self._damage_window_actor = slot if slot in {"p1", "p2"} else None
            return
        if event_type == "cant":
            # No move executed, so no strike can follow from this seat.
            self._damage_window_actor = None
            return
        if event_type != "-damage" or len(parts) < 4:
            return
        slot = _slot_from_ident(parts[2])
        if slot not in self.current_damage_taken:
            return
        active = self.public_active.get(slot)
        prev_condition = (
            getattr(active, "condition", None)
            if _is_current_public_active(active) and getattr(active, "ident", None) == parts[2]
            else None
        )
        prev_hp, prev_max = _hp_numerator_denominator(prev_condition)
        cur_hp, cur_max = _hp_numerator_denominator(parts[3])
        max_hp = prev_max or cur_max
        if prev_hp is None or not max_hp:
            # A faint line reads ``0 fnt`` with no denominator; when the PREVIOUS condition is
            # unreadable too there is no public magnitude, so nothing is recorded rather than a
            # guessed one. (``0 fnt`` as the NEW value is handled below: cur_hp is None -> 0.)
            return
        remaining = cur_hp if cur_hp is not None else 0
        fraction = (prev_hp - remaining) / max_hp
        if fraction <= 0:
            return
        self.current_damage_taken[slot] = self.current_damage_taken.get(slot, 0.0) + fraction
        tagged = any(part.strip().startswith("[from]") for part in parts[4:])
        if not tagged:
            actor = self._damage_window_actor
            if actor is not None and actor != slot and actor in self.current_damage_dealt:
                self.current_damage_dealt[actor] = (
                    self.current_damage_dealt.get(actor, 0.0) + fraction
                )
        elif "[from] Spikes" in line:
            # Part B1: the entry-hazard credit ledger. Spikes is gen3's only entry hazard and the
            # only pool member that tags this way, so the tag alone identifies the source; the
            # credit belongs to the side that laid the layers, i.e. the OTHER slot, and is read
            # off the victim's ledger at encode time.
            self.hazard_damage_suffered[slot] = (
                self.hazard_damage_suffered.get(slot, 0.0) + fraction
            )

    def _update_choice_lock(self, parts: Sequence[str], line: str) -> None:
        """The public choice lock and the item's provenance (spec v4 pack A4 addendum).

        Gen3's ``choicelock`` volatile is entirely SILENT — ``data/conditions.ts`` gives it no
        ``add`` on start, end, or transfer — so no protocol line ever announces it and no
        volatile tracker can catch it. It is reconstructed here from the two public facts that
        determine it, both of which the sim does emit:

        * WHICH item the mon holds. ``choiceband`` is gen3's only ``isChoice`` item (Scarf and
          Specs are gen4+). It becomes public on an ``|-item|`` line — in practice a Trick, and
          in gen3 randbats a Trick carrier ALWAYS holds a Choice Band (``teams.ts``:
          ``if (moves.has('trick')) return 'Choice Band'``), so the strategy is deterministic.
        * WHETHER it has moved since acquiring it. Choice Band's ``onStart`` REMOVES any existing
          choicelock when the item arrives, and its ``onModifyMove`` re-adds one on the next move
          used. So the lock attaches to the first move executed AFTER acquisition — which is
          exactly what pack A2 (``CATEGORY_LAST_USED_MOVE``) names. Lock bit + last move fully
          specify the lock, the same way ``volatile:encore`` + last move specify an Encore.

        ``item_from_trick`` is the valence discriminator, and the reason a bare "holds a Choice
        Band" reading is not enough: a NATIVE Choice Band is assigned to all-attacks sets
        (``counter.get('Physical') >= 4``) and makes its holder stronger, whereas a Tricked one
        is a liability we inflicted — often locking a support mon into a status move. Identical
        item, opposite sign. The belief engine already audits this surface (it sets
        ``item_mutated`` / ``current_public_item`` on the same line); this is the parser-side
        twin so the fact reaches the observation without a belief round-trip.

        Cleared on ``-enditem`` (the item is gone, so the lock goes with it) and on switch-out
        (handled in the parse loop's switch block, where every per-mon tracker resets).
        """

        event_type = parts[1] if len(parts) > 1 else ""
        if len(parts) < 3:
            return
        slot = _slot_from_ident(parts[2])
        if slot not in self.choice_locked:
            return
        if event_type == "-item" and len(parts) >= 4:
            # A fresh item resets the lock even when the new item is also a choice item:
            # Choice Band's onStart deletes choicelock, so the holder is free until it moves.
            self.choice_item_public[slot] = _normalize_identifier(parts[3]) in _CHOICE_ITEMS
            self.choice_locked[slot] = False
            self.item_from_trick[slot] = "[from] move: Trick" in line
            return
        if event_type == "-enditem":
            self.choice_item_public[slot] = False
            self.choice_locked[slot] = False
            self.item_from_trick[slot] = False
            return
        if event_type == "move" and len(parts) >= 4:
            # ``onModifyMove`` adds the lock on the move actually used. A ``|move|`` line is the
            # public mirror of that, and a called move (Sleep Talk's callee) does not lock —
            # the same ``[from]`` discriminator the last_used_move truth table uses.
            if self.choice_item_public.get(slot) and not any(
                part.startswith("[from]") for part in parts[4:]
            ):
                self.choice_locked[slot] = True
            return
        if event_type == "faint":
            self.choice_item_public[slot] = False
            self.choice_locked[slot] = False
            self.item_from_trick[slot] = False

    def _update_items_removed(self, parts: Sequence[str], line: str) -> None:
        """Per-side count of held items removed by the OPPOSING side's action (v4 part B4).

        Knock Off only. The public surface is ``|-enditem|SLOT|ITEM|[from] move: Knock Off`` — the
        same discriminator the belief engine uses to set ``item_removed`` on that mon. Excluded on
        purpose: a bare ``-enditem`` (a berry the holder ate, White Herb) is self-consumption and
        nobody's credit, and ``[from] move: Trick`` is a SWAP whose giving half the belief layer
        explicitly declines to model — counting it as removal credit would price a trade as a
        theft. Counted on the VICTIM's slot; the encoder reads the opposite side's ledger as our
        credit (see NUMERIC_OPP_ITEMS_REMOVED_CREDIT's orientation note).
        """

        if (parts[1] if len(parts) > 1 else "") != "-enditem" or len(parts) < 3:
            return
        if "[from] move: Knock Off" not in line:
            return
        slot = _slot_from_ident(parts[2])
        if slot in self.items_removed:
            self.items_removed[slot] = self.items_removed.get(slot, 0) + 1

    def snapshot(self) -> ShowdownReplayState:
        # Do not serialize a malformed mutable latch as a future authorization.
        self._sanitize_toxic_replacement_provenance()
        return ShowdownReplayState(
            battle_id=self.battle_id,
            players=dict(self.players),
            requests=dict(self.requests),
            public_active=dict(self.public_active),
            public_revealed={slot: tuple(pokemon) for slot, pokemon in self.public_revealed.items()},
            side_conditions={slot: tuple(sorted(conditions)) for slot, conditions in _side_conditions_from_counts(self.side_condition_counts).items()},
            side_condition_counts={
                slot: dict(sorted(conditions.items()))
                for slot, conditions in self.side_condition_counts.items()
            },
            boosts={slot: dict(sorted(stages.items())) for slot, stages in self.boosts.items()},
            volatiles={slot: tuple(sorted(names)) for slot, names in self.volatiles.items()},
            substitute_health_state=dict(self.substitute_health_state),
            substitute_depletion=dict(self.substitute_depletion),
            direct_materialization_blockers={
                slot: tuple(sorted(blockers))
                for slot, blockers in self.direct_materialization_blockers.items()
            },
            future_sight=dict(self.future_sight),
            toxic_stage=dict(self.toxic_stage),
            toxic_stage_known=dict(self.toxic_stage_known),
            toxic_stage_zero_after_upkeep=dict(self.toxic_stage_zero_after_upkeep),
            toxic_stage_zero_after_upkeep_expires_after_turn=dict(
                self.toxic_stage_zero_after_upkeep_expires_after_turn
            ),
            toxic_stage_zero_after_upkeep_ident=dict(self.toxic_stage_zero_after_upkeep_ident),
            toxic_faint_replacement_pending=dict(self.toxic_faint_replacement_pending),
            toxic_faint_replacement_expected_ident=dict(
                self.toxic_faint_replacement_expected_ident
            ),
            toxic_faint_replacement_invalid=dict(self.toxic_faint_replacement_invalid),
            hp_visibility={
                slot: self._hp_visibility_for_slot(slot) for slot in ("p1", "p2")
            },
            confusion_elapsed=dict(self.confusion_elapsed),
            encore_elapsed=dict(self.encore_elapsed),
            wrap_trap_elapsed=dict(self.wrap_trap_elapsed),
            meanlook_trap=dict(self.meanlook_trap),
            public_events=tuple(self.public_events),
            public_lines=tuple(self.public_lines),
            weather=self.weather,
            turn_number=self.turn_number,
            winner=self.winner,
            weather_set_turn=self.weather_set_turn,
            weather_from_ability=self.weather_from_ability,
            weather_upkeeps=self.weather_upkeeps,
            side_condition_set_turns={
                slot: dict(turns) for slot, turns in self.side_condition_set_turns.items()
            },
            wish_set_turns=dict(self.wish_set_turns),
            leech_seed_source_sides=dict(self.leech_seed_source_sides),
            pending_leech_seed_source_sides=dict(self._pending_leech_seed_source_sides),
            pending_baton_pass=tuple(sorted(self.pending_baton_pass)),
            live_type_override=dict(self.live_type_override),
            traced_ability=dict(self.traced_ability),
            truant_phase=dict(self.truant_phase),
            post_upkeep_window=self._post_upkeep_window,
            truant_skip_next_flip=tuple(sorted(self._truant_skip_next_flip)),
            induced_sleep_victims={
                slot: tuple(sorted(victims))
                for slot, victims in self.induced_sleep_victims.items()
            },
            rest_sleep_counts=dict(self.rest_sleep_counts),
            rest_sleep_refunded_turns=dict(self.rest_sleep_refunded_turns),
            rest_sleep_skipped_turns=dict(self.rest_sleep_skipped_turns),
            rest_sleep_pending_attempt=dict(self.rest_sleep_pending_attempt),
            stall_counter=dict(self.stall_counter),
            last_used_move={
                slot: value for slot, value in self.last_used_move.items() if value
            },
            stall_move_pending=dict(self.stall_move_pending),
            must_recharge=dict(self.must_recharge),
            last_damage_dealt=dict(self.last_damage_dealt),
            last_damage_taken=dict(self.last_damage_taken),
            current_damage_dealt=dict(self.current_damage_dealt),
            current_damage_taken=dict(self.current_damage_taken),
            hazard_damage_suffered=dict(self.hazard_damage_suffered),
            items_removed=dict(self.items_removed),
            arrived_by_baton_pass=dict(self.arrived_by_baton_pass),
            choice_item_public=dict(self.choice_item_public),
            choice_locked=dict(self.choice_locked),
            item_from_trick=dict(self.item_from_trick),
        )


def parse_showdown_replay(
    lines: Sequence[str],
    *,
    battle_id: str = "replay",
    complete_prefix: bool = False,
    hp_visibility: Mapping[str, str] | None = None,
) -> ShowdownReplayState:
    """Parse compact Showdown protocol lines into transport-level state.

    ``complete_prefix`` must be asserted only when the caller owns a stream that starts at battle
    reset. Attach-midstream callers should keep the fail-closed default. ``hp_visibility`` records
    whether each side's HP condition is exact or Showdown's rounded player-view percentage.
    """
    parser = _ReplayParser(
        battle_id=battle_id,
        complete_prefix=complete_prefix,
        hp_visibility=hp_visibility,
    )
    parser.feed(lines)
    return parser.snapshot()


def detect_showdown_slot(
    replay: ShowdownReplayState,
    *,
    player_name: str | None = None,
    configured_showdown_slot: str | None = None,
) -> str:
    """Resolve the actual Showdown side for a player.

    Player name from public battle state wins over a stale configured default.
    """
    normalized_name = _normalize_name(player_name)
    if normalized_name:
        for showdown_slot, name in replay.players.items():
            if _normalize_name(name) == normalized_name:
                return showdown_slot
        for showdown_slot, request in replay.requests.items():
            side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
            side_name = side.get("name") if isinstance(side, Mapping) else None
            if _normalize_name(side_name) == normalized_name:
                return showdown_slot
    if configured_showdown_slot in {"p1", "p2"}:
        return configured_showdown_slot
    raise ValueError("Unable to detect Showdown slot from player_name or configured_showdown_slot.")


def _apply_live_type_override(
    team: tuple[ShowdownPokemon, ...], source: str | None
) -> tuple[ShowdownPokemon, ...]:
    """Stamp the active mon of ``team`` with a live type override source (no-op when None).

    Only the currently-active mon retypes (Castform Forecast / Kecleon Color Change revert on
    switch-out), so the override is applied to the ``active`` member only.
    """
    if not source:
        return team
    return tuple(
        replace(mon, live_type_source=source) if mon.active else mon for mon in team
    )


def normalize_for_player(
    replay: ShowdownReplayState,
    *,
    player_id: str,
    player_name: str | None = None,
    configured_showdown_slot: str | None = None,
    format_id: str | None = None,
    set_source: PokemonSetSource | None = None,
    recent_event_limit: int = 24,
    belief_engine: "PublicBattleBeliefEngine | None" = None,
    include_turn_merged: bool = False,
) -> PlayerRelativeBattleState:
    """Build a player-relative state view from raw Showdown transport state.

    ``belief_engine`` lets a caller pass a persistent engine fed incrementally (the local sim
    env), avoiding a from-scratch rebuild from ``replay.public_events`` on every call. When
    omitted, the engine is built batch-style from the replay (unchanged behavior).
    ``include_turn_merged`` additionally populates ``turn_merged_tokens`` from the same
    shared fold — required for the v2.2 (turn-merged) encode, off by default so the
    v2/v2.1 observe hot path is unchanged.
    """
    showdown_slot = detect_showdown_slot(
        replay,
        player_name=player_name,
        configured_showdown_slot=configured_showdown_slot,
    )
    opponent_slot = opponent_showdown_slot(showdown_slot)
    perspective = ObservationPerspective(
        player_id=player_id,
        showdown_slot=showdown_slot,
        opponent_showdown_slot=opponent_slot,
    )
    request = replay.requests.get(showdown_slot)
    self_team = _self_team_from_request(request, showdown_slot)
    opponent_team = _opponent_team_from_public_state(replay, opponent_slot)
    if belief_engine is None:
        belief_engine = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id=format_id,
            set_source=set_source,
        )
        belief_engine.resolve_pending_switches_at_boundary()
        belief_view = belief_engine.snapshot().for_player(showdown_slot)
    else:
        # Persistent engine fed incrementally: resolve+snapshot on a copy so its pending-switch
        # state survives for the next ingested event.
        belief_view = belief_engine.resolved_player_view(showdown_slot)
    opponent_team = _merge_opponent_belief_facts(opponent_team, belief_view)
    # Stamp the active mon on each side with any live type override (Castform Forecast forme /
    # Kecleon Color Change) so the encoder overrides its type slots. Keyed per showdown slot; the
    # override is cleared on switch-out so only the currently-active mon ever carries one.
    self_team = _apply_live_type_override(self_team, replay.live_type_override.get(showdown_slot))
    opponent_team = _apply_live_type_override(opponent_team, replay.live_type_override.get(opponent_slot))
    recent_events = tuple(
        _relative_public_event(event, self_slot=showdown_slot, opponent_slot=opponent_slot)
        for event in replay.public_events[-recent_event_limit:]
    )
    # Ordered transition history + tendency aggregates (PR B extraction functions), from a
    # single shared fold of the replay (folding twice doubled the per-observe history cost).
    # Local import: transitions.py imports this module's parse helpers, so a module-level
    # import would cycle.
    turn_merged_tokens: tuple["TurnMergedToken", ...] = ()
    if include_turn_merged:
        from .turn_merged import extract_transition_products

        transition_tokens, turn_merged_tokens, tendency_stats = extract_transition_products(
            replay, perspective_slot=showdown_slot
        )
    else:
        from .transitions import extract_transitions_and_tendencies

        transition_tokens, tendency_stats = extract_transitions_and_tendencies(
            replay, perspective_slot=showdown_slot
        )
    weather_turns_remaining, weather_permanent = _weather_duration_features(replay)
    sleep_clause_holders = belief_engine.sleep_clause_holders
    return PlayerRelativeBattleState(
        battle_id=replay.battle_id,
        player_id=player_id,
        perspective=perspective,
        request=request,
        request_kind=_request_kind(request),
        self_team=self_team,
        opponent_team=opponent_team,
        self_side_conditions=tuple(replay.side_conditions.get(showdown_slot, ())),
        opponent_side_conditions=tuple(replay.side_conditions.get(opponent_slot, ())),
        self_side_condition_counts=dict(replay.side_condition_counts.get(showdown_slot, {})),
        opponent_side_condition_counts=dict(replay.side_condition_counts.get(opponent_slot, {})),
        self_active_boosts=dict(replay.boosts.get(showdown_slot, {})),
        opponent_active_boosts=dict(replay.boosts.get(opponent_slot, {})),
        self_active_volatiles=tuple(replay.volatiles.get(showdown_slot, ())),
        opponent_active_volatiles=tuple(replay.volatiles.get(opponent_slot, ())),
        self_toxic_stage=int(replay.toxic_stage.get(showdown_slot, 0)),
        opponent_toxic_stage=int(replay.toxic_stage.get(opponent_slot, 0)),
        self_confusion_elapsed=int(replay.confusion_elapsed.get(showdown_slot, 0)),
        opponent_confusion_elapsed=int(replay.confusion_elapsed.get(opponent_slot, 0)),
        self_encore_elapsed=int(replay.encore_elapsed.get(showdown_slot, 0)),
        opponent_encore_elapsed=int(replay.encore_elapsed.get(opponent_slot, 0)),
        self_wrap_trap_elapsed=int(replay.wrap_trap_elapsed.get(showdown_slot, 0)),
        opponent_wrap_trap_elapsed=int(replay.wrap_trap_elapsed.get(opponent_slot, 0)),
        self_meanlook_trap=bool(replay.meanlook_trap.get(showdown_slot, False)),
        opponent_meanlook_trap=bool(replay.meanlook_trap.get(opponent_slot, False)),
        belief_view=belief_view,
        legal_action_mask=_legal_action_mask(request),
        recent_events=recent_events,
        recent_public_events=tuple(event.relative_line or event.raw_line for event in recent_events),
        weather=replay.weather,
        turn_number=replay.turn_number,
        self_future_sight_turns=_future_sight_turns_remaining(replay, showdown_slot),
        opponent_future_sight_turns=_future_sight_turns_remaining(replay, opponent_slot),
        winner=replay.winner,
        transition_tokens=transition_tokens,
        turn_merged_tokens=turn_merged_tokens,
        tendency_stats=tendency_stats,
        weather_turns_remaining=weather_turns_remaining,
        weather_permanent=weather_permanent,
        self_timed_condition_turns=_timed_condition_turns(replay, showdown_slot),
        opponent_timed_condition_turns=_timed_condition_turns(replay, opponent_slot),
        self_wish_pending=_wish_pending(replay, showdown_slot),
        opponent_wish_pending=_wish_pending(replay, opponent_slot),
        self_wish_turns=_wish_turns_remaining(replay, showdown_slot),
        opponent_wish_turns=_wish_turns_remaining(replay, opponent_slot),
        self_sleep_clause_used=sleep_clause_holders.get(showdown_slot) is not None,
        opponent_sleep_clause_used=sleep_clause_holders.get(opponent_slot) is not None,
        self_sleep_clause_blocks=bool(replay.induced_sleep_victims.get(showdown_slot)),
        opponent_sleep_clause_blocks=bool(replay.induced_sleep_victims.get(opponent_slot)),
        self_stall_counter=int(replay.stall_counter.get(showdown_slot, 0)),
        opponent_stall_counter=int(replay.stall_counter.get(opponent_slot, 0)),
        # ---- spec v4 k0 feature pack. Read straight off the parser trackers; the schema gate
        # lives at encode time, so these travel on every normalized state (and into the
        # observation metadata) regardless of which schema the caller ends up encoding.
        self_must_recharge=bool(replay.must_recharge.get(showdown_slot, False)),
        opponent_must_recharge=bool(replay.must_recharge.get(opponent_slot, False)),
        # ``truant_phase`` is tri-state (True loafs / False acts / None no-holder-or-unknown);
        # ``is True`` collapses the two non-assertions onto the same 0 the world falls back to.
        self_truant_loaf=replay.truant_phase.get(showdown_slot) is True,
        opponent_truant_loaf=replay.truant_phase.get(opponent_slot) is True,
        self_last_used_move=replay.last_used_move.get(showdown_slot),
        opponent_last_used_move=replay.last_used_move.get(opponent_slot),
        self_arrived_by_baton_pass=bool(
            replay.arrived_by_baton_pass.get(showdown_slot, False)
        ),
        opponent_arrived_by_baton_pass=bool(
            replay.arrived_by_baton_pass.get(opponent_slot, False)
        ),
        self_choice_locked=bool(replay.choice_locked.get(showdown_slot, False)),
        opponent_choice_locked=bool(replay.choice_locked.get(opponent_slot, False)),
        self_item_swapped=bool(replay.item_from_trick.get(showdown_slot, False)),
        opponent_item_swapped=bool(replay.item_from_trick.get(opponent_slot, False)),
        self_traced_ability=replay.traced_ability.get(showdown_slot),
        opponent_traced_ability=replay.traced_ability.get(opponent_slot),
        self_last_damage_dealt=float(replay.last_damage_dealt.get(showdown_slot, 0.0) or 0.0),
        self_last_damage_taken=float(replay.last_damage_taken.get(showdown_slot, 0.0) or 0.0),
        opponent_last_damage_dealt=float(replay.last_damage_dealt.get(opponent_slot, 0.0) or 0.0),
        opponent_last_damage_taken=float(replay.last_damage_taken.get(opponent_slot, 0.0) or 0.0),
        self_hazard_damage_suffered=float(
            replay.hazard_damage_suffered.get(showdown_slot, 0.0) or 0.0
        ),
        opponent_hazard_damage_suffered=float(
            replay.hazard_damage_suffered.get(opponent_slot, 0.0) or 0.0
        ),
        self_items_removed=int(replay.items_removed.get(showdown_slot, 0) or 0),
        opponent_items_removed=int(replay.items_removed.get(opponent_slot, 0) or 0),
        # Conditioned HERE rather than at encode time: our own active is known at
        # normalization, so the encoder receives one already-selected column of the
        # (their mon x our mon) table instead of the whole table.
        opponent_matchup_switch_evidence=_matchup_switch_evidence(
            tendency_stats, self_team
        ),
    )


def _matchup_switch_evidence(
    tendency_stats: "TendencyStats | None",
    self_team: Sequence[ShowdownPokemon],
) -> dict[str, tuple[int, int]]:
    """Per-opponent-mon (switched, stayed) against the mon WE currently have out.

    Selects one column of the fold's (their mon x our mon) table. An empty result — no
    active mon resolvable, or no tendency stats — means every opponent token encodes (0, 0),
    which is the honest reading: this matchup has no history yet. The marginal triple on the
    same token is the fallback the model already has.
    """

    if tendency_stats is None:
        return {}
    active = next((mon for mon in self_team if mon.active), None)
    if active is None or not active.species:
        return {}
    ours = _normalize_identifier(active.species)
    evidence: dict[str, tuple[int, int]] = {}
    for cell in tendency_stats.opponent_mon_matchups:
        if _normalize_identifier(cell.opposing_species) != ours:
            continue
        evidence[_normalize_identifier(cell.species)] = (
            int(cell.switched_out_before_attacking),
            int(cell.stayed_and_attacked),
        )
    return evidence


def _weather_duration_features(replay: ShowdownReplayState) -> tuple[int, bool]:
    """(turns remaining, permanent) for the active weather; (0, False) when clear.

    Ability weather is permanent in gen 3: the counter is pinned at the full 5 so it never reads
    as decaying. Move weather counts down deterministically from its set turn.
    """
    if not replay.weather:
        return 0, False
    if replay.weather_from_ability:
        return _TIMED_CONDITION_DURATION, True
    if replay.weather_set_turn is None:
        return 0, False
    # Move weather counts down one tick per end-of-turn upkeep. The observation boundary always
    # sits after the set turn's own upkeep, so the elapsed count must come from the upkeep ticks
    # actually observed rather than the whole-turn difference (which is one short at the set turn,
    # before |turn|N+1 is fed — audit #9). This matches the bridge weatherState.duration at every
    # boundary from set to expiry, including a mid-turn switch on the set turn (0 upkeeps → full 5).
    return max(0, _TIMED_CONDITION_DURATION - replay.weather_upkeeps), False


def _timed_condition_turns(replay: ShowdownReplayState, slot: str) -> dict[str, int]:
    """Turns remaining per ACTIVE timed side condition for one side (5-turn class, gen 3)."""
    set_turns = replay.side_condition_set_turns.get(slot, {})
    active_counts = replay.side_condition_counts.get(slot, {})
    remaining: dict[str, int] = {}
    for condition, set_turn in set_turns.items():
        if not active_counts.get(condition):
            continue
        remaining[condition] = max(0, _TIMED_CONDITION_DURATION - (replay.turn_number - set_turn))
    return remaining


def _wish_pending(replay: ShowdownReplayState, slot: str) -> bool:
    """True while a declared Wish has not yet landed on ``slot``'s side (lands end of next turn)."""
    set_turn = replay.wish_set_turns.get(slot)
    return set_turn is not None and (replay.turn_number - set_turn) <= 1


def _wish_turns_remaining(replay: ShowdownReplayState, slot: str) -> int:
    """Turns until a declared Wish resolves on ``slot``'s side: 2 the declaration turn, 1 the
    landing turn, 0 when none is pending (spec v3 change 9). Reads the same ``wish_set_turns``
    tracker as ``_wish_pending`` and is nonzero on exactly the turns that predicate is true, so the
    v3 turns column and the v2.2 pending bit never disagree about presence. Keyed on the SIDE, so a
    wish-pass switch keeps the clock running for the mon that inherits the slot."""
    set_turn = replay.wish_set_turns.get(slot)
    if set_turn is None:
        return 0
    remaining = 2 - (replay.turn_number - set_turn)
    return remaining if 1 <= remaining <= 2 else 0


def observation_from_player_state(
    state: PlayerRelativeBattleState,
    *,
    category_vocab: "CategoryVocabulary",
    spec: ObservationSpec = DEFAULT_REPLAY_OBSERVATION_SPEC,
    dex: "ShowdownDex | None" = None,
    feature_masks: ObservationFeatureMasks = DEFAULT_OBSERVATION_FEATURE_MASKS,
) -> PokeZeroObservationV0:
    """Encode normalized replay state into fixed-shape observation rows.

    Categorical slots are encoded as raw token strings and converted to compact embedding rows
    via ``category_vocab`` (required) in a single pass. When ``dex`` is supplied, raw mechanical
    facts (Pokemon types; move type / damage class / base power / priority / accuracy) are
    populated into the type/mechanic feature slots; without it those slots stay padding.
    ``feature_masks`` darkens ablation-arm blocks (zeroed + attention-masked) without changing
    shapes or the spec version.

    ``spec.schema_version`` selects the encode mode (dual-schema window): a v2 spec produces
    the v2 layout byte-identically to the pre-v2.1 encoder (no v2.1 column is even attempted);
    a v2.1 spec additionally writes defender identity on move transition tokens, the
    revealed-move PP-validity bits, the substitute HP fraction, and the per-mon pinned
    Tier-2 conclusions. A v2.2 spec keeps every v2.1 block and swaps the transition
    surface to TURN-MERGED tokens (state.turn_merged_tokens; one row per phase, two
    ordered sub-blocks — budget counts THESE rows, i.e. whole turns). A v3 spec keeps the v2.2
    semantic surface, adds v3 signals, removes evidence-backed dead fields, and projects private
    legacy writer rows into a grouped public layout (docs/observation_v3_spec.md). Anything else
    refuses loudly here rather than encoding an undeclared hybrid.
    """
    if spec.schema_version not in REPLAY_OBSERVATION_SPECS_BY_SCHEMA:
        supported = ", ".join(repr(version) for version in REPLAY_OBSERVATION_SPECS_BY_SCHEMA)
        raise ValueError(
            f"observation encode: unsupported spec schema {spec.schema_version!r}; supported "
            f"schemas are {supported}."
        )
    # V4 is the v3 surface plus the k0 feature pack, so ``schema_v3`` stays the gate for every
    # v3-era writer (it means "grouped-layout lineage", not "exactly v3") and ``schema_v4`` gates
    # the pack columns on top. A v3 spec therefore never touches a pack column, and a v4 spec
    # writes the complete v3 surface — the two projections differ, never the semantics they share.
    schema_v4 = spec.schema_version == OBSERVATION_SCHEMA_VERSION_V4
    schema_v3 = schema_v4 or spec.schema_version == OBSERVATION_SCHEMA_VERSION_V3
    if schema_v4 and spec.numeric_feature_count != _V4_NUMERIC_FEATURE_COUNT:
        raise ValueError(
            "observation encode: the grouped v4 layout requires exactly "
            f"{_V4_NUMERIC_FEATURE_COUNT} numeric columns, got {spec.numeric_feature_count}. "
            "Its projection map defines the complete public surface."
        )
    if (
        schema_v3
        and not schema_v4
        and spec.numeric_feature_count != _V3_NUMERIC_FEATURE_COUNT
    ):
        raise ValueError(
            "observation encode: the grouped v3 layout requires exactly "
            f"{_V3_NUMERIC_FEATURE_COUNT} numeric columns, got {spec.numeric_feature_count}. "
            "Its projection map defines the complete public surface."
        )
    # Census floor (#512 review MED-LOW): refuse a spec narrower than its schema's own
    # census rather than letting the bounds-checked writers silently drop the schema's
    # columns and emit an undeclared hybrid stamped with the wider version.
    census_floor = _MINIMUM_NUMERIC_CENSUS_BY_SCHEMA[spec.schema_version]
    if spec.numeric_feature_count < census_floor:
        raise ValueError(
            f"observation encode: spec schema {spec.schema_version!r} requires at least "
            f"{census_floor} numeric columns, got {spec.numeric_feature_count}. A narrower "
            "spec would silently bounds-drop this schema's own columns and encode an "
            "undeclared hybrid stamped with the wider version; the 119-column relic family "
            f"is a {OBSERVATION_SCHEMA_VERSION_V2!r}-only exception."
        )
    if schema_v4 and spec.categorical_feature_count != _V4_CATEGORICAL_FEATURE_COUNT:
        # EXACT, not a floor. Every earlier schema was wider than its predecessor, so "at least
        # my census" was sufficient; v4 is the first that SHRINKS (41 vs v3's 51), which makes
        # a stale 51 look like a legal over-wide spec and silently emit ten dead columns on a
        # tensor stamped v4.
        raise ValueError(
            "observation encode: the grouped v4 layout requires exactly "
            f"{_V4_CATEGORICAL_FEATURE_COUNT} categorical columns, got "
            f"{spec.categorical_feature_count}. v4 is NARROWER than v3 (the turn-merged block "
            "is gone), so a v3-width spec is a mismatch, not a permissible superset."
        )
    categorical_floor = _MINIMUM_CATEGORICAL_CENSUS_BY_SCHEMA[spec.schema_version]
    if spec.categorical_feature_count < categorical_floor:
        raise ValueError(
            f"observation encode: spec schema {spec.schema_version!r} requires at least "
            f"{categorical_floor} categorical columns, got {spec.categorical_feature_count}. "
            "The categorical census is schema-keyed "
            f"({OBSERVATION_SCHEMA_VERSION_V2!r} and {OBSERVATION_SCHEMA_VERSION_V2_1!r}: "
            f"{_CATEGORICAL_FEATURE_COUNT}; {OBSERVATION_SCHEMA_VERSION_V2_2!r} and "
            f"{OBSERVATION_SCHEMA_VERSION_V3!r}: "
            f"{_V2_2_CATEGORICAL_FEATURE_COUNT}; {OBSERVATION_SCHEMA_VERSION_V4!r}: "
            f"{_V4_CATEGORICAL_FEATURE_COUNT}); a narrower spec would silently "
            "bounds-drop the schema's own categorical surface (v2.2's whole second "
            "sub-block, v4's last-move / traced-ability pair) and encode an undeclared "
            "hybrid stamped with the wider version."
        )
    # V3 keeps the v2.2 turn-merged semantic surface but projects the private legacy writer
    # rows into its grouped public layout after all token writers complete.
    # TURN-MERGED is a property of the TRANSITION REGION, and v4 has none — so v4 is a
    # grouped-layout (v3-lineage) schema that is NOT turn-merged. Keeping these two axes
    # separate is what lets v4 write every v3 current-state signal while encoding no history.
    schema_turn_merged = (not schema_v4) and (
        schema_v3 or spec.schema_version == OBSERVATION_SCHEMA_VERSION_V2_2
    )
    # v2.2 carries every v2.1 block forward unchanged; only the transition surface differs. v4
    # keeps those blocks too (PP-validity bits, sub HP, the per-mon pinned Tier-2 conclusions —
    # all current-state surfaces that survive the region trim).
    schema_v2_1 = (
        schema_turn_merged
        or schema_v4
        or spec.schema_version == OBSERVATION_SCHEMA_VERSION_V2_1
    )
    if schema_turn_merged and state.transition_tokens and not state.turn_merged_tokens:
        raise ValueError(
            "observation encode: a v2.2 (turn-merged) spec requires the state's "
            "turn_merged_tokens — normalize with include_turn_merged=True."
        )
    if schema_turn_merged and not category_vocab.is_enumerated("tt_phase:turn"):
        raise ValueError(
            "observation encode: a v2.2 (turn-merged) spec requires a vocabulary built "
            "with include_turn_merged=True — this one lacks the tt_phase/tt2_* families, "
            "so every turn-merged label would silently hash into the OOV band and the "
            "encoded rows could never align with a v2.2 checkpoint's embedding "
            "(review MED-2: the vocabulary axis of the #492/#512 mismatch class)."
        )
    # Per-mon pinned Tier-2 CB conclusions (v2.1, NUMERIC_TIER2_CB_PINNED): derived from the
    # tier2-annotated token stream under the same tier2_residuals gate as the tt columns —
    # the monotone as-of-strike bit makes "any assessed strike of this mon carries it"
    # exactly the tracker's per-mon conclusion, and reading the FULL (untruncated) token
    # list here is what makes the pinned form robust to the K-budget truncation the
    # history surface is subject to.
    #
    # RETIRED AT V4 (and only at v4): the conclusion narrows the belief candidate set there
    # instead — a strictly richer surface than this one bit, and the reason the exclusion is
    # spelled by name (``and not schema_v4``) rather than by turning ``schema_v2_1`` off:
    # that flag means "carries the v2.1 blocks" and v4 inherits it for the PP-validity bits
    # and sub HP. See _V4_DROPPED_CURRENT_STATE_NUMERIC_INDICES.
    tier2_cb_pinned_species: frozenset[str] = frozenset()
    if schema_v2_1 and not schema_v4 and feature_masks.tier2_residuals:
        opponent_slot = state.perspective.opponent_showdown_slot
        tier2_cb_pinned_species = frozenset(
            _normalize_identifier(token.actor_species)
            for token in state.transition_tokens
            if token.cb_bit and token.kind == _TT_KIND_MOVE and token.actor_slot == opponent_slot
        )
    # Per-mon pinned investment conclusions (v2.1, NUMERIC_TIER2_INVESTMENT_PINNED): the
    # CB derivation's mirror, inverted to the defender — investment codes ride OUR
    # assessed move tokens and describe the STRUCK opponent mon (token.defender_species,
    # the #512 identity channel). The as-of-strike code is monotone (conclusions freeze;
    # an HP conclusion upgrades over a defense-only pin, never retracts), so the LAST
    # annotated strike of each defender carries the tracker's current per-mon conclusion,
    # and reading the FULL untruncated token list keeps the pinned form robust to
    # K-budget truncation. Triple-gated like the tt-row write (v2.1 schema + both masks).
    #
    # RETIRED AT V4 (and only at v4): the conclusion narrows the belief candidate set there
    # instead, which is a strictly richer surface than this lossy class projection — see
    # _V4_DROPPED_CURRENT_STATE_NUMERIC_INDICES. The exclusion has to be surgical because
    # ``schema_v2_1`` means "carries the v2.1 blocks" and v4 inherits it for every OTHER
    # block (PP-validity bits, sub HP), so the column is switched
    # off by name rather than by turning that flag off.
    tier2_investment_pinned: dict[str, float] = {}
    if (
        schema_v2_1
        and not schema_v4
        and feature_masks.tier2_residuals
        and feature_masks.tier2_investment
    ):
        self_slot = state.perspective.showdown_slot
        for token in state.transition_tokens:
            if (
                token.investment
                and token.kind == _TT_KIND_MOVE
                and token.actor_slot == self_slot
                and token.defender_species
            ):
                tier2_investment_pinned[_normalize_identifier(token.defender_species)] = max(
                    -1.0, min(1.0, token.investment)
                )
    categorical_ids = _blank_categorical_rows(spec)
    numeric_features = _blank_numeric_rows(
        spec,
        # The writer constants are the frozen v2.2-plus-v3-appendix positions. V3 projects
        # this internal row after encoding so its public 155-column layout can freely reorder
        # and drop evidence-backed dead fields without perturbing a legacy writer. V4 widens the
        # same internal row by the feature-pack columns and projects through its own map.
        internal_numeric_feature_count=(
            V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT
            if schema_v4
            else V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT
            if schema_v3
            else spec.numeric_feature_count
        ),
    )
    _encode_field_token(
        categorical_ids,
        numeric_features,
        state,
        masks=feature_masks,
        schema_v3=schema_v3,
        schema_v4=schema_v4,
        dex=dex,
    )
    # Exact-state per-mon fields come from the belief engine's ledgers for BOTH sides (it tracks
    # self and opponent); the opponent's belief-fact buckets keep their existing single source.
    self_exact_beliefs = {
        _normalize_identifier(belief.species): belief for belief in state.belief_view.self_pokemon
    }
    _encode_pokemon_tokens(
        categorical_ids,
        numeric_features,
        SELF_POKEMON_TOKEN_OFFSET,
        state.self_team,
        role="self",
        limit=SELF_POKEMON_TOKEN_COUNT,
        active_boosts=state.self_active_boosts,
        active_volatiles=state.self_active_volatiles,
        active_toxic_stage=state.self_toxic_stage,
        active_stall_counter=state.self_stall_counter,
        active_confusion_elapsed=state.self_confusion_elapsed,
        active_encore_elapsed=state.self_encore_elapsed,
        active_wrap_trap_elapsed=state.self_wrap_trap_elapsed,
        active_meanlook_trap=state.self_meanlook_trap,
        active_must_recharge=state.self_must_recharge,
        active_truant_loaf=state.self_truant_loaf,
        active_last_used_move=state.self_last_used_move,
        active_arrived_by_baton_pass=state.self_arrived_by_baton_pass,
        active_choice_locked=state.self_choice_locked,
        active_item_swapped=state.self_item_swapped,
        active_traced_ability=state.self_traced_ability,
        active_last_damage_dealt=state.self_last_damage_dealt,
        active_last_damage_taken=state.self_last_damage_taken,
        dex=dex,
        exact_beliefs_by_species=self_exact_beliefs,
        masks=feature_masks,
        schema_v2_1=schema_v2_1,
        schema_v3=schema_v3,
        schema_v4=schema_v4,
    )
    opponent_beliefs = state.belief_view.opponent_by_species()
    tendency_by_species = (
        {
            _normalize_identifier(tendency.species): tendency
            for tendency in state.tendency_stats.opponent_mon_tendencies
        }
        if state.tendency_stats is not None
        else {}
    )
    _encode_pokemon_tokens(
        categorical_ids,
        numeric_features,
        OPPONENT_POKEMON_TOKEN_OFFSET,
        state.opponent_team,
        role="opponent",
        limit=OPPONENT_POKEMON_TOKEN_COUNT,
        beliefs_by_species=opponent_beliefs,
        active_boosts=state.opponent_active_boosts,
        active_volatiles=state.opponent_active_volatiles,
        active_toxic_stage=state.opponent_toxic_stage,
        active_stall_counter=state.opponent_stall_counter,
        active_confusion_elapsed=state.opponent_confusion_elapsed,
        active_encore_elapsed=state.opponent_encore_elapsed,
        active_wrap_trap_elapsed=state.opponent_wrap_trap_elapsed,
        active_meanlook_trap=state.opponent_meanlook_trap,
        active_must_recharge=state.opponent_must_recharge,
        active_truant_loaf=state.opponent_truant_loaf,
        active_last_used_move=state.opponent_last_used_move,
        active_arrived_by_baton_pass=state.opponent_arrived_by_baton_pass,
        active_choice_locked=state.opponent_choice_locked,
        active_item_swapped=state.opponent_item_swapped,
        active_traced_ability=state.opponent_traced_ability,
        active_last_damage_dealt=state.opponent_last_damage_dealt,
        active_last_damage_taken=state.opponent_last_damage_taken,
        dex=dex,
        exact_beliefs_by_species=opponent_beliefs,
        tendency_by_species=tendency_by_species,
        # Transform copy targets: in singles an opponent Transform copies OUR mon; species
        # clause makes the by-species lookup unique within our team.
        transform_targets_by_species={
            _normalize_identifier(member.species): member for member in state.self_team
        },
        matchup_switch_evidence=state.opponent_matchup_switch_evidence,
        masks=feature_masks,
        schema_v2_1=schema_v2_1,
        schema_v3=schema_v3,
        schema_v4=schema_v4,
        tier2_cb_pinned_species=tier2_cb_pinned_species,
        tier2_investment_pinned=tier2_investment_pinned,
    )
    _encode_action_tokens(categorical_ids, numeric_features, state, dex=dex)
    _encode_stats_token(categorical_ids, numeric_features, state, masks=feature_masks)
    if schema_v4:
        # No transition region exists at v4 — nothing to encode, and no budget to honour. The
        # tokens are still EXTRACTED upstream (normalize_for_player), because the per-mon pinned
        # Tier-2 conclusions and the tendency aggregates are derived from that stream; only the
        # per-row ENCODING is gone.
        pass
    elif schema_turn_merged:
        _encode_turn_merged_transition_tokens(
            categorical_ids, numeric_features, state, spec, masks=feature_masks, schema_v3=schema_v3
        )
    else:
        _encode_transition_tokens(
            categorical_ids, numeric_features, state, spec, masks=feature_masks, schema_v2_1=schema_v2_1
        )
    if schema_v4:
        numeric_features = _project_v4_numeric_rows(numeric_features)
    elif schema_v3:
        numeric_features = _project_v3_numeric_rows(numeric_features)
    # Convert the raw category strings to compact embedding rows in one pass.
    categorical_rows = [[category_vocab.encode(value) for value in row] for row in categorical_ids]
    token_type_ids = _token_type_ids(spec)
    attention_mask = _attention_mask(state, spec, masks=feature_masks)
    return PokeZeroObservationV0(
        categorical_ids=tuple(tuple(row) for row in categorical_rows),
        numeric_features=tuple(tuple(row) for row in numeric_features),
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        legal_action_mask=state.legal_action_mask,
        perspective=state.perspective,
        metadata=_observation_metadata(
            state, dex=dex, schema_version=spec.schema_version
        ),
        schema_version=spec.schema_version,
    )


def stable_category_id(value: str, *, buckets: int = CATEGORY_ID_BUCKETS) -> int:
    """Map a category string to a deterministic positive id.

    This is a stable hash-bucket encoder for early experiments. Explicit
    vocabularies can replace it once the observation vocabulary is finalized.
    """
    normalized = str(value or "").strip().lower()
    if not normalized:
        return 0
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "big") % buckets) + 1


def showdown_choice_for_action(state: PlayerRelativeBattleState, action_index: int) -> str:
    """Translate a 0-8 policy action index to a Showdown choice string."""
    if action_index < 0 or action_index >= ACTION_COUNT:
        raise ValueError(f"action_index must be between 0 and {ACTION_COUNT - 1}.")
    if not state.legal_action_mask[action_index]:
        raise ValueError(
            f"action_index {action_index} is not legal for the current request "
            f"(request_kind={state.request_kind})."
        )
    if is_move_action(action_index):
        return f"move {action_index + 1}"
    if is_switch_action(action_index):
        active_team_index = _active_team_index(state.self_team)
        if active_team_index is None:
            raise ValueError("Cannot translate switch action without an active self Pokemon.")
        switch_slot = action_index - MOVE_ACTION_COUNT
        switch_targets = canonical_switch_action_map(active_team_index, team_size=len(state.self_team))
        if switch_slot >= len(switch_targets):
            raise ValueError(f"action_index {action_index} is outside the current switch target map.")
        return f"switch {switch_targets[switch_slot] + 1}"
    raise ValueError(f"Unsupported action_index: {action_index}.")


def showdown_submission_for_action(state: PlayerRelativeBattleState, action_index: int) -> ShowdownSubmission:
    """Translate a policy action into the protocol side and choice string."""
    return ShowdownSubmission(
        showdown_slot=state.perspective.showdown_slot,
        choice=showdown_choice_for_action(state, action_index),
    )


def _decode_request_payload(line: str) -> Mapping[str, Any]:
    payload_text = line[len("|request|") :]
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Showdown request payload: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Showdown request payload must be a JSON object.")
    return payload


def _pokemon_from_public_line(parts: Sequence[str]) -> ShowdownPokemon | None:
    ident = parts[2]
    showdown_slot = _slot_from_ident(ident)
    if showdown_slot is None:
        return None
    details = parts[3] if len(parts) > 3 else ""
    return ShowdownPokemon(
        ident=ident,
        showdown_slot=showdown_slot,
        species=_species_from_details(details) or _species_from_ident(ident),
        condition=parts[4] if len(parts) > 4 else None,
        active=True,
        details=details,
    )


def _update_public_pokemon_condition(
    parts: Sequence[str],
    public_active: dict[str, ShowdownPokemon],
    public_revealed: dict[str, list[ShowdownPokemon]],
) -> None:
    """Apply public HP/status protocol updates to the current revealed mon."""

    if len(parts) < 3:
        return
    event_type = parts[1] if len(parts) > 1 else ""
    if event_type == "-cureteam":
        # Aromatherapy's ``|-cureteam|SOURCE`` clears the non-volatile status of every
        # LIVING member on the source's side and carries no per-mon ``-curestatus`` — so
        # the active-only path below never fires. Strip the status suffix team-wide.
        _apply_public_cureteam_condition(parts, public_active, public_revealed)
        return
    if event_type not in {"-damage", "-heal", "-sethp", "-status", "-curestatus", "faint"}:
        return
    ident = parts[2]
    slot = _slot_from_ident(ident)
    if slot is None:
        return
    if event_type == "-curestatus" and not _ident_has_position(ident):
        # A benched ally cured by a team-wide effect (Heal Bell's per-mon ``[silent]``
        # -curestatus, ident ``p2: Snorlax``) serializes WITHOUT a field-position letter.
        # The active-only path below resolves ``public_active[slot]`` — the ACTIVE mon
        # (e.g. Miltank), whose ident never equals the benched ident — so it early-returns
        # and the benched ally's status suffix in ``public_revealed`` stays stale. Resolve
        # the benched mon by species instead and strip its suffix. This is the parser-surface
        # sibling of the belief engine's benched -curestatus handling (#771's
        # ``_benched_target_belief``); ACTIVE-target cures keep their position letter and take
        # the unchanged path below.
        _apply_public_benched_curestatus_condition(ident, slot, public_revealed)
        return
    active = public_active.get(slot)
    if active is None or active.ident != ident:
        return
    condition = _updated_public_condition(active.condition, event_type=event_type, parts=parts)
    if condition is None:
        return
    updated = replace(active, condition=condition)
    public_active[slot] = updated
    public_revealed[slot] = [
        updated if _same_public_pokemon(pokemon, updated) else pokemon
        for pokemon in public_revealed.get(slot, ())
    ]


def _updated_public_condition(
    condition: str | None,
    *,
    event_type: str,
    parts: Sequence[str],
) -> str | None:
    if event_type in {"-damage", "-heal", "-sethp"}:
        return parts[3] if len(parts) > 3 else None
    if event_type == "faint":
        return "0 fnt"
    current = str(condition or "").split()
    hp = current[0] if current else ""
    statuses = [status for status in current[1:] if status != "fnt"]
    if event_type == "-status" and len(parts) > 3:
        status = _normalize_identifier(parts[3])
        if hp:
            return " ".join((hp, status))
    if event_type == "-curestatus" and hp:
        return hp
    return None


def _apply_public_benched_curestatus_condition(
    ident: str,
    slot: str,
    public_revealed: dict[str, list[ShowdownPokemon]],
) -> None:
    """Clear the non-volatile status suffix of the single BENCHED ally named by a team-wide
    ``-curestatus`` (Heal Bell's per-mon ``[silent]`` form, ident ``p2: Snorlax``). The ident
    carries no field-position letter, so the active-only ``-curestatus`` path cannot resolve it;
    match by species in ``public_revealed`` and strip the suffix via ``strip_condition_status``
    (the same shared helper #771's ``-cureteam`` path uses), mirroring
    ``_apply_public_cureteam_condition``'s per-member strip. A fainted ally's ``0 fnt`` is
    preserved unchanged by ``strip_condition_status``; a healthy ally is left byte-identical.

    Species clause makes the name unique within a randbats side, so at most one row matches.
    A cosmetic-forme ally serializes under its BASE name in the cure ident (gen3 randbats name
    an Unown-Z simply ``Unown``) while the revealed row keeps the lettered forme — the
    base-name fallback keeps the parser surface in step with the belief engine's forme-tolerant
    benched resolution (#771's ``_base_species_id``)."""
    revealed = public_revealed.get(slot)
    if not revealed:
        return
    target = _normalize_name(_species_from_ident(ident))

    def _matches(species: str | None) -> bool:
        normalized = _normalize_name(species)
        return normalized == target or _normalize_name(str(species or "").split("-", 1)[0]) == target

    updated_list: list[ShowdownPokemon] = []
    changed = False
    for pokemon in revealed:
        if _matches(pokemon.species):
            stripped = strip_condition_status(pokemon.condition)
            if stripped != pokemon.condition:
                pokemon = replace(pokemon, condition=stripped)
                changed = True
        updated_list.append(pokemon)
    if changed:
        public_revealed[slot] = updated_list


def _apply_public_cureteam_condition(
    parts: Sequence[str],
    public_active: dict[str, ShowdownPokemon],
    public_revealed: dict[str, list[ShowdownPokemon]],
) -> None:
    """Aromatherapy's ``|-cureteam|SOURCE`` clears the status of EVERY living member on
    the source's side. The ident is the active user, so strip the non-volatile status
    suffix from that side's active mon AND every revealed benched mon (the team-wide
    analogue of the ``-curestatus`` active-only strip). A fainted mon's ``0 fnt``
    condition is preserved unchanged by ``strip_condition_status``."""
    slot = _slot_from_ident(parts[2]) if len(parts) > 2 else None
    if slot is None:
        return
    active = public_active.get(slot)
    if active is not None:
        stripped = strip_condition_status(active.condition)
        if stripped != active.condition:
            public_active[slot] = replace(active, condition=stripped)
    active_now = public_active.get(slot)
    revealed = public_revealed.get(slot)
    if not revealed:
        return
    updated_list: list[ShowdownPokemon] = []
    for pokemon in revealed:
        if active_now is not None and _same_public_pokemon(pokemon, active_now):
            updated_list.append(active_now)
            continue
        stripped = strip_condition_status(pokemon.condition)
        updated_list.append(replace(pokemon, condition=stripped) if stripped != pokemon.condition else pokemon)
    public_revealed[slot] = updated_list


def _side_conditions_from_counts(side_condition_counts: Mapping[str, Mapping[str, int]]) -> dict[str, set[str]]:
    return {
        slot: {condition for condition, count in conditions.items() if count > 0}
        for slot, conditions in side_condition_counts.items()
    }


def _update_side_conditions(parts: Sequence[str], side_conditions: dict[str, dict[str, int]]) -> None:
    event_type = parts[1] if len(parts) > 1 else ""
    if event_type not in {"-sidestart", "-sideend"} or len(parts) < 4:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in side_conditions:
        return
    condition = _side_condition_identifier(parts[3])
    if not condition:
        return
    if event_type == "-sidestart":
        side_conditions[slot][condition] = min(
            _side_condition_max_layers(condition),
            side_conditions[slot].get(condition, 0) + 1,
        )
    else:
        side_conditions[slot].pop(condition, None)


# The volatile statuses we surface (normalized ids). `-start`/`-end` carry many payloads (ability
# procs, type changes, internal markers); we track only this closed, decision-relevant set so every
# emitted volatile:<id> token has an enumerated vocab row (no OOV) and is a genuine status. This is
# the single source of truth — randbat_vocab enumerates volatile:<id> from it.
# The only gen3 Truant lines, and both are MONO-ability, so species alone identifies a
# holder with no reveal required. (Durant is the only other Truant carrier and is gen5.)
_TRUANT_SPECIES = frozenset({"slakoth", "slaking"})

TRACKED_VOLATILES = frozenset({
    "confusion", "leechseed", "substitute", "taunt", "encore", "disable", "torment", "attract",
    "nightmare", "curse", "ingrain", "foresight", "lockon", "mindreader", "destinybond", "grudge",
    "focusenergy", "charge", "yawn", "stockpile", "bide", "uproar", "imprison", "magiccoat",
    "snatch", "mudsport", "watersport", "defensecurl", "minimize", "rage", "partiallytrapped",
    "perishsong", "perish0", "perish1", "perish2", "perish3", "flashfire",
    # Mid-charge state of a two-turn move, keyed by the MOVE id (see
    # _CHARGE_MOVE_VOLATILES). Note "charge" three lines up is a different thing
    # entirely -- the Charge MOVE's Electric-damage doubler -- and conflating the
    # two would hand a Solar Beam user a phantom Electric boost.
    "solarbeam",
})

# Two-turn charge moves whose mid-charge state we surface, keyed by the MOVE id so it
# maps straight onto the engine's own charge volatile (gen3 choice_effects.rs
# `charge_choice_to_volatile`: SOLARBEAM -> PokemonVolatileStatus::SOLARBEAM). The
# volatile IS the commitment on both sides -- the engine's `active_is_charging_move`
# locks get_all_options to that move, and releases it in generate_instructions.
#
# Only Solar Beam is reachable. Of the 17 moves carrying the dex `charge: 1` flag, it is
# the only one in the gen3 randbats pool (4 sets: Exeggutor, Sunflora, Tangela,
# Victreebel). Read that flag, never a substring of the move data: `recharge: 1` is a
# DIFFERENT mechanic (Hyper Beam and 9 others) that a naive match sweeps in.
_CHARGE_MOVE_VOLATILES = frozenset({"solarbeam"})

# Pokemon Showdown's Gen 3 `Pokemon.copyVolatileFrom` carries conditions whose
# `noCopy` flag is false. This is the tracked subset. The parser preserves these
# public facts through Baton Pass, then direct materialization rejects any whose
# serialized state is not yet fully public and reconstructable.
# gen3's ``sleepUsable`` moves: the only two a sleeping Pokemon can act with, and so the
# only two that accrue the ``skippedTime`` a later switch-in refunds. Mirrors
# ``belief._SLEEP_USABLE_MOVES``; kept local rather than imported because belief.py imports
# this module.
_SLEEP_USABLE_MOVES = frozenset({"sleeptalk", "snore"})

_BATON_PASS_TRANSFERRED_VOLATILES = frozenset({
    "confusion", "leechseed", "substitute", "taunt", "curse", "ingrain", "lockon",
    "grudge", "focusenergy", "charge", "bide", "uproar", "magiccoat", "snatch",
    "mudsport", "watersport", "rage", "partiallytrapped", "perishsong",
    # Perish Song lives on the mon as its COUNTER (``perish3`` ... ``perish1``),
    # never as ``perishsong`` -- the sim announces ``|-start|<mon>|perishN``. The
    # id-only entry therefore matched nothing and the intersection below silently
    # DROPPED a Baton-Passed countdown, losing a public fact rather than blocking
    # on it. Misdreavus is the pool's Perish Song user and also its Mean Look
    # user, so the passed countdown is exactly the line that decides those games.
    "perish0", "perish1", "perish2", "perish3",
})

# Transferred volatiles that do NOT wall the direct construction. Anything
# transferred but absent here becomes a ``baton-pass:<id>`` blocker.
#
# NOT a claim that every member is fully materializable -- the engine-world
# constructor keeps its own, narrower allowlist and re-validates independently,
# so ``focusenergy``/``ingrain``/``mudsport``/``watersport`` (pre-existing
# entries) still raise ``volatile_unsupported`` downstream. This list only
# decides which BP transfers are worth attempting.
#
# Perish counters are exact: Showdown announces the post-decrement value and the
# engine faints on PERISH1, so a mon publicly showing perishN acts N more times
# in both. confusion and partial trap ride their existing named approximations.
#
# SUBSTITUTE IS DELIBERATELY ABSENT. It transfers in gen 3 -- copyVolatileFrom
# shallow-clones the volatile including its ``hp`` field -- but that HP is
# ``floor(PASSER.maxhp/4)`` minus everything it has already absorbed, and the
# constructor can only derive ``recipient.maxhp // 4``. That is not an
# approximation of the true value, it is a different Pokemon's number: wrong in
# both directions (Ninjask->Snorlax models 86 where reality is <=56), and for a
# Shedinja recipient it computes 0, which makes the engine absorb one hit of
# ARBITRARY size for zero damage. Fail closed instead.
_DIRECT_MATERIALIZATION_VOLATILES = frozenset({
    "focusenergy", "ingrain", "leechseed", "mudsport", "watersport",
    "confusion", "partiallytrapped",
    "perish1", "perish2", "perish3",
})


# Gen 3 partial-trap moves. The sim announces the volatile via
# ``|-activate|<target>|move: Wrap|[of] <source>`` (conditions.ts partiallytrapped.onStart)
# and ends it with ``|-end|<target>|Wrap|[partiallytrapped]`` — the move NAME, not the
# volatile id, so both arms need this normalization set (audit bug C2). Wrap is the pool's
# only member; the rest are defensive against set drift.
_PARTIAL_TRAP_MOVES = frozenset({"wrap", "bind", "clamp", "firespin", "whirlpool", "sandtomb"})
# Singles slot pairing: the mon in the other seat. Used by the Mean Look / Spider Web move-trap
# tracker (spec v3 change 8) — the trapper is always the OPPOSING active mon, so when either seat's
# occupant changes the trap between the two seats ends.
_OTHER_SLOT = {"p1": "p2", "p2": "p1"}
# ``|-singlemove|`` volatiles with until-the-mon's-next-move semantics: the sim removes
# them SILENTLY (onBeforeMove / onMoveAborted, no protocol line), so the parser clears
# them on the mon's next |move| or |cant| line (audit bug C3). Destiny Bond is the pool's
# only reachable member (Grudge/Rage are -singlemove emitters but their moves are not in
# the gen3 randbats pool); Focus Punch's focus is ``-singleturn`` and is NOT tracked here.
_SINGLEMOVE_VOLATILES = frozenset({"destinybond", "grudge"})
# Perish Song's countdown is ONE value announced as successive ``-start perishN``
# (Showdown never emits an ``-end`` between the ticks), so these are mutually
# exclusive and the newest replaces the rest.
_PERISH_COUNTERS = frozenset({"perish0", "perish1", "perish2", "perish3", "perish4"})
# Gen 3 stall moves (spec v3 change 3): all three set ``stallingMove: true`` and share the ONE
# ``stall`` volatile (``data/moves.ts`` protect 13960 / detect 3523 / endure 4802). Pool
# reachability in ``data/random-battles/gen3/sets.json``: protect (43 species) and endure (4)
# are reachable; detect is NOT (0 species) but shares the ``protect`` volatile, so it is handled
# for correctness. Used to decide, on a ``|move|`` line, whether the streak continues (stall) or
# breaks (non-stall reset).
_STALL_MOVE_IDS = frozenset({"protect", "detect", "endure"})
# Streak saturates the ``min(1.0, count / 8.0)`` encoding at 8; cap the stored value there so a
# pathological log cannot grow it without bound (mirrors the toxic-stage clamp).
_STALL_COUNTER_CAP = 8


def _is_stall_singleturn(tag: str) -> bool:
    """True for a stall move's success-only ``-singleturn`` tag.

    Protect/Detect emit ``|-singleturn|SLOT|Protect``; Endure emits
    ``|-singleturn|SLOT|move: Endure`` (verified in the vendored data/moves.ts onStart lines).
    Strip any ``move:`` prefix and normalize: ``Protect`` -> ``protect``, ``move: Endure`` ->
    ``endure``. Other ``-singleturn`` users (Focus Punch, Magic Coat, Snatch) normalize to other
    names and are correctly excluded.
    """
    return _normalize_identifier(tag.split(":", 1)[-1]) in {"protect", "endure"}


def _is_confusion_self_hit(parts: Sequence[str]) -> bool:
    """True for ``|-activate|<mon>|confusion``, the Gen 3 hit-yourself line.

    This is the only way a Gen 3 action is consumed without a ``move`` or
    ``cant`` line, which makes it the one blind spot in the single-move
    expiry rule.
    """

    return (
        len(parts) >= 4
        and parts[1] == "-activate"
        and _side_condition_identifier(parts[3]) == "confusion"
    )


def _update_volatiles(parts: Sequence[str], volatiles: dict[str, set[str]]) -> None:
    """Track active-mon volatile statuses per Showdown slot.

    Arms: ``-start``/``-end`` (the common family), ``-activate move: <partial-trap>`` /
    ``-end <partial-trap move> [partiallytrapped]`` (bug C2 — the sim never emits a
    ``-start`` for partial traps), ``-singlemove`` (bug C3 — Destiny Bond class), and
    ``move``/``cant`` lines, which silently expire single-move volatiles. Only names in
    TRACKED_VOLATILES are recorded, so every emitted token has an enumerated vocab row.
    """
    event_type = parts[1] if len(parts) > 1 else ""
    if len(parts) < 3:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in volatiles:
        return
    if event_type in {"move", "cant"} or _is_confusion_self_hit(parts):
        # The sim removes single-move volatiles silently before the mon's next action
        # (onBeforeMove / onMoveAborted); a successful re-click re-arms via the
        # following |-singlemove| line.
        #
        # ``move``/``cant`` are not the whole story: MoveAborted fires on EVERY
        # consumed action, and the Gen 3 confusion self-hit is the one path that
        # emits neither line — just ``-activate|MON|confusion`` plus the damage
        # (mods/gen4/conditions.ts). Without this arm a confused Destiny Bond
        # user keeps the volatile forever, and since destinybond became a
        # searchable volatile that phantom is seeded into every sampled world,
        # pricing a KO-revenge that cannot happen. Every other Gen 3 abort
        # (sleep, freeze, paralysis, flinch, recharge, Attract) does emit
        # ``cant`` and is already covered above.
        volatiles[slot] -= _SINGLEMOVE_VOLATILES
        # A two-turn charge ends on whichever action consumes it, and BOTH ways end
        # here. `|move|SLOT|Solar Beam|TARGET|[from] lockedmove` is the release.
        # `|cant|SLOT|...` is an outright CANCEL: gen3's `twoturnmove.onMoveAborted`
        # drops the volatile with no `-end` line, so the `cant` is the only
        # announcement the charge is over -- a mon fully paralysed on its release turn
        # loses the charge and re-charges from scratch afterwards (verified against the
        # sim). Without this arm the world carries a phantom charge forever.
        #
        # The CHARGE turn's own `|move|SLOT|Solar Beam||[still]` also lands here, and
        # harmlessly: the `|-prepare|` that immediately follows it re-arms the volatile.
        volatiles[slot] -= _CHARGE_MOVE_VOLATILES
        return
    if event_type == "-anim":
        # A charge move that skips its charge turn — Solar Beam in SUN — still emits
        # `|move|...||[still]` and `|-prepare|`, then fires in the SAME turn via
        # `|-anim|SLOT|<Move>|TARGET` + damage, with no second `|move|` line. So
        # `-anim` is the "it actually executed" signal, and without this arm a sunny
        # Solar Beam user carries a phantom charge until its NEXT action clears it —
        # the world would offer it only Solar Beam while it is in fact free.
        # (The ordinary two-turn release emits no `-anim`; it is a real `|move|`
        # line tagged `[from] lockedmove`, handled above.)
        volatiles[slot] -= _CHARGE_MOVE_VOLATILES
        return
    if len(parts) < 4:
        return
    name = _side_condition_identifier(parts[3])  # strips move:/ability:/item: prefix + normalizes
    if event_type == "-prepare":
        # `|-prepare|SLOT|<Move>` is the public charge announcement, emitted right after
        # the paired `|move|` line's `[still]`. It leaks nothing: it names only the move
        # the opponent just watched being charged, which is exactly what a human sees.
        if name in _CHARGE_MOVE_VOLATILES:
            volatiles[slot].add(name)
        return
    if event_type == "-singlemove":
        if name in TRACKED_VOLATILES:
            volatiles[slot].add(name)
        return
    if event_type == "-activate":
        if name in _PARTIAL_TRAP_MOVES:
            volatiles[slot].add("partiallytrapped")
        return
    if event_type not in {"-start", "-end"}:
        return
    if event_type == "-end" and name in _PARTIAL_TRAP_MOVES:
        volatiles[slot].discard("partiallytrapped")
        return
    if name not in TRACKED_VOLATILES:
        return
    if event_type == "-start":
        if name in _PERISH_COUNTERS:
            # Perish Song announces its countdown as successive ``-start perishN``
            # with no ``-end`` between them, so a plain add ACCUMULATES: a mon one
            # turn from fainting carried {perish3, perish2, perish1} at once. The
            # counter is a single value, so the newest announcement replaces the
            # previous one. (The engine happened to survive the pile-up by
            # decrementing every counter in lockstep and checking PERISH1 first,
            # but that is an ordering coincidence, not the contract.)
            volatiles[slot] -= _PERISH_COUNTERS
        volatiles[slot].add(name)
    else:
        volatiles[slot].discard(name)


# Delayed-damage moves (Future Sight / Doom Desire): used on one turn, they land on the target's
# side ~2 turns later. Tracked as a per-side landing turn so the model sees an incoming/outgoing hit.
_FUTURE_MOVES = frozenset({"futuresight", "doomdesire"})
_FUTURE_SIGHT_DELAY = 2


def _update_future_sight(parts: Sequence[str], future_sight: dict[str, int], turn_number: int) -> None:
    """Track pending delayed attacks per side from |-start| (use) / |-end| (land) lines.

    Showdown puts the |-start| on the USER and the |-end| on the side that takes the hit, so a use
    schedules a landing on the user's OPPONENT side; the landing |-end| clears it.
    """
    event_type = parts[1] if len(parts) > 1 else ""
    if event_type not in {"-start", "-end"} or len(parts) < 4:
        return
    if _side_condition_identifier(parts[3]) not in _FUTURE_MOVES:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in {"p1", "p2"}:
        return
    if event_type == "-start":
        target_side = "p2" if slot == "p1" else "p1"
        future_sight[target_side] = turn_number + _FUTURE_SIGHT_DELAY
    else:
        future_sight.pop(slot, None)


def _is_active_protocol_ident(ident: str) -> bool:
    """Whether a protocol ident names the current singles active slot.

    Per-mon cure lines can name a benched team member as ``p1: Name``. A
    Toxic counter belongs only to the active Pokemon, so treating every same-
    side cure as a reset corrupts the live active counter.
    """

    return bool(re.fullmatch(r"p[12]a: \S(?:.*\S)?", ident))


def _canonical_turn_number(line: str) -> int | None:
    """Return a strictly canonical positive ``|turn|N`` marker."""

    match = re.fullmatch(r"\|turn\|([1-9][0-9]*)", line)
    return int(match.group(1)) if match is not None else None


def _canonical_upkeep_marker(line: str) -> bool:
    """Whether ``line`` is the unique no-payload Showdown upkeep boundary."""

    return line == "|upkeep"


def _canonical_faint_marker(line: str, parts: Sequence[str]) -> bool:
    """Whether a faint line can open the one-seat replacement proof."""

    return (
        line == "|".join(parts)
        and len(parts) == 3
        and parts[0] == ""
        and parts[1] == "faint"
        and _is_active_protocol_ident(parts[2])
    )


def _canonical_replacement_marker(
    line: str,
    event_type: str,
    parts: Sequence[str],
) -> bool:
    """Whether a switch-family line has a canonical active-slot payload.

    Baton Pass is the only optional switch suffix accepted by this parser
    boundary.  It remains ineligible for the Toxic proof, but accepting its
    canonical wire form keeps ordinary switch accounting intact.
    """

    has_baton_pass_suffix = (
        event_type == "switch"
        and len(parts) == 6
        and parts[5] == "[from] Baton Pass"
    )
    if len(parts) != 5 and not has_baton_pass_suffix:
        return False
    return (
        line == "|".join(parts)
        and parts[0] == ""
        and parts[1] == event_type
        and event_type in {"switch", "drag", "replace"}
        and _is_active_protocol_ident(parts[2])
        and all(parts[index] for index in (3, 4))
    )


def _condition_has_status(condition: str | None, status: str) -> bool:
    return bool(condition and _normalize_identifier(status) in condition.split())


def _update_toxic_stage(
    parts: Sequence[str],
    toxic_stage: dict[str, int],
    toxic_stage_known: dict[str, bool] | None = None,
    toxic_stage_zero_after_upkeep: dict[str, bool] | None = None,
) -> None:
    """Track the badly-poisoned (tox) ramp stage per side from |-status| / |-curestatus| /
    |-cureteam| lines.

    A `tox` status starts the counter at 1 (per-turn escalation is applied on |turn|); a status
    replacement or cure on the ACTIVE mon clears it. Team-wide ``-cureteam`` clears the active
    mon too. The counter is also reset on switch and faint (Gen 3 behavior) in the parse loop.

    Every transition here is public protocol evidence. ``toxic_stage_known`` is optional for
    direct unit callers; when supplied it distinguishes a real public zero from an incomplete
    prefix that must never be materialized as a synthetic zero counter. The optional
    ``toxic_stage_zero_after_upkeep`` carries the still-pending, post-upkeep replacement proof;
    every active status/cure/faint transition retires it before changing the regular stage.
    """
    event_type = parts[1] if len(parts) > 1 else ""
    if len(parts) < 3:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in toxic_stage:
        return
    active_target = _is_active_protocol_ident(parts[2])
    if event_type == "faint" and active_target:
        if toxic_stage_zero_after_upkeep is not None:
            toxic_stage_zero_after_upkeep[slot] = False
        toxic_stage[slot] = 0
        if toxic_stage_known is not None:
            toxic_stage_known[slot] = True
    elif not active_target and event_type != "-cureteam":
        # E.g. Heal Bell's ``|-curestatus|p1: Bench|...|[silent]``. It cannot
        # alter the current active mon's statusState.stage.
        return
    elif event_type == "-status" and len(parts) >= 4 and _normalize_identifier(parts[3]) == "tox":
        if toxic_stage_zero_after_upkeep is not None:
            toxic_stage_zero_after_upkeep[slot] = False
        toxic_stage[slot] = 1
        if toxic_stage_known is not None:
            toxic_stage_known[slot] = True
    elif event_type == "-status" and len(parts) >= 4:
        # Any OTHER status replaces tox, and the ramp dies with it. `Pokemon.setStatus`
        # does `this.statusState = this.battle.initEffectState(...)` (sim/pokemon.ts:1733),
        # replacing the state object wholesale -- and the toxic counter lives in
        # `statusState.stage`. Crucially Showdown emits NO `-curestatus` for the status it
        # replaced, so the two reset arms below never see it.
        #
        # Rest is the reachable case and the one that bit: `|-status|<mon>|slp|[from] move:
        # Rest` on an already-toxed mon left the ramp standing at its old value, so a LATER
        # re-tox in the same stint was priced from a stage that no longer existed. Observed
        # as a stage-5 tick (-75) where Showdown ticked a fresh stage-1 (-15).
        if toxic_stage_zero_after_upkeep is not None:
            toxic_stage_zero_after_upkeep[slot] = False
        toxic_stage[slot] = 0
        if toxic_stage_known is not None:
            toxic_stage_known[slot] = True
    elif event_type in {"-curestatus", "-cureteam"}:
        # ``-cureteam`` (Aromatherapy) ident is the active source, which is itself cured,
        # so resetting the active slot's ramp matches the per-mon ``-curestatus`` reset.
        if toxic_stage_zero_after_upkeep is not None:
            toxic_stage_zero_after_upkeep[slot] = False
        toxic_stage[slot] = 0
        if toxic_stage_known is not None:
            toxic_stage_known[slot] = True


def _update_confusion_elapsed(parts: Sequence[str], confusion_elapsed: dict[str, int]) -> None:
    """Reset the confusion turns-so-far counter on snap-out / faint (spec v3 change 4).

    The per-``|turn|`` advance happens in the parse loop (gated on the public ``confusion``
    volatile, mirroring the toxic ramp). This handles the two RESET lines that are not a
    switch (which the parse loop resets directly): ``|-end|SLOT|confusion`` (the mon snapped
    out) and ``|faint|SLOT`` (the mon fainted while confused). The counter is also reset on
    switch-out in the parse loop (Gen 3 clears the volatile), so a stale value can never ride
    onto a replacement or survive past the volatile.
    """
    event_type = parts[1] if len(parts) > 1 else ""
    if len(parts) < 3:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in confusion_elapsed:
        return
    if event_type == "faint":
        confusion_elapsed[slot] = 0
    elif (
        event_type == "-end"
        and len(parts) >= 4
        and _side_condition_identifier(parts[3]) == "confusion"
    ):
        confusion_elapsed[slot] = 0


def _update_encore_elapsed(parts: Sequence[str], encore_elapsed: dict[str, int]) -> None:
    """Reset the encore turns-so-far counter on expiry / faint (spec v3 change 5).

    The per-``|turn|`` advance happens in the parse loop (gated on the public ``encore``
    volatile, mirroring the toxic ramp and the confusion counter). This handles the two RESET
    lines that are not a switch (which the parse loop resets directly): ``|-end|SLOT|Encore``
    (the lock wore off — vendored gen3 ``encore.condition.onEnd`` emits ``this.add('-end',
    target, 'Encore')``) and ``|faint|SLOT`` (the mon fainted while encored). The counter is also
    reset on switch-out/drag in the parse loop (Encore is ``noCopy: true``, so the volatile always
    clears), so a stale value can never ride onto a replacement or survive past the volatile.
    """
    event_type = parts[1] if len(parts) > 1 else ""
    if len(parts) < 3:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in encore_elapsed:
        return
    if event_type == "faint":
        encore_elapsed[slot] = 0
    elif (
        event_type == "-end"
        and len(parts) >= 4
        and _side_condition_identifier(parts[3]) == "encore"
    ):
        encore_elapsed[slot] = 0


def _update_wrap_trap_elapsed(parts: Sequence[str], wrap_trap_elapsed: dict[str, int]) -> None:
    """Reset the Wrap (partial-trap) turns-so-far counter on expiry / faint (spec v3 change 6).

    The per-``|turn|`` advance happens in the parse loop (gated on the public ``partiallytrapped``
    volatile, mirroring the toxic ramp and the confusion / encore counters). This handles the two
    RESET lines that are not a switch (which the parse loop resets directly): the partial-trap
    ``|-end|SLOT|Wrap|[partiallytrapped]`` (the pin wore off, or the vendored sim's silent
    ``[silent]`` end when the trapper left the field — base ``conditions.ts partiallytrapped.onEnd``
    emits ``this.add('-end', pokemon, sourceEffect, '[partiallytrapped]')``, where ``sourceEffect``
    is the MOVE, so ``parts[3]`` is the move NAME like ``Wrap``, NOT the volatile id — the same
    keying the ``_update_volatiles`` partial-trap arm uses) and ``|faint|SLOT`` (the mon fainted
    while trapped). The counter is also reset on switch-out/drag in the parse loop, so a stale value
    can never ride onto a replacement or survive past the volatile.
    """
    event_type = parts[1] if len(parts) > 1 else ""
    if len(parts) < 3:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in wrap_trap_elapsed:
        return
    if event_type == "faint":
        wrap_trap_elapsed[slot] = 0
    elif (
        event_type == "-end"
        and len(parts) >= 4
        and _side_condition_identifier(parts[3]) in _PARTIAL_TRAP_MOVES
    ):
        wrap_trap_elapsed[slot] = 0


def _update_meanlook_trap(parts: Sequence[str], meanlook_trap: dict[str, bool]) -> None:
    """Track the Mean Look / Spider Web move-trap flag per slot (spec v3 change 8).

    SET on ``|-activate|SLOT|trapped`` — the base ``trapped`` volatile's ``onStart`` emits
    ``this.add('-activate', target, 'trapped')`` with no ``[of]`` and no move prefix, so ``parts[3]``
    is exactly the volatile id ``trapped``. Gen3 Mean Look / Spider Web are the only movers of this
    volatile (ability traps use ``onFoeTrapPokemon`` and emit NO ``-activate|trapped``), so the line
    uniquely marks a move-trap. The trapper is the OPPOSING active mon (singles).

    RESET on ``|faint|SLOT``: the trapped mon fainting clears its own flag, and the fainting mon was
    the trapper for the other seat (the linked source-side volatile drops when the trapper faints),
    so BOTH seats clear. Switch/drag resets are handled in the parse loop (the ``trapped`` volatile
    is ``noCopy``, so it never rides a Baton Pass). There is no ``-end`` line for this volatile (it
    has no ``onEnd`` and the linked removal is silent), so faint + switch/drag are the only public
    end signals.
    """
    event_type = parts[1] if len(parts) > 1 else ""
    if len(parts) < 3:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in meanlook_trap:
        return
    if event_type == "-activate" and len(parts) >= 4 and _side_condition_identifier(parts[3]) == "trapped":
        meanlook_trap[slot] = True
    elif event_type == "faint":
        meanlook_trap[slot] = False
        meanlook_trap[_OTHER_SLOT[slot]] = False


# Gen3's only ``isChoice`` item (``data/items.ts`` choiceband, gen: 3). Choice Scarf and Choice
# Specs are gen4+ and are not in the pool, so the lock has exactly one source.
_CHOICE_ITEMS = frozenset({"choiceband"})


def _update_must_recharge(parts: Sequence[str], must_recharge: dict[str, bool]) -> None:
    """Track the public forced-recharge lock per slot (spec v4 pack A1).

    SET on ``|-mustrecharge|SLOT``. The vendored sim emits that line from the ``mustrecharge``
    volatile's ``onStart``, which runs only when a recharge move actually LANDS — a missed Hyper
    Beam never reaches it, so gen3's "a miss does not recharge" rule needs no special case here
    (it is the one rule the search lane's round-record reconstruction had to encode by hand).

    CLEARED on ``|cant|SLOT|recharge``: the forced turn has been spent, and the lock is gone
    before the next decision. Cleared on ``|faint|SLOT`` and on switch/drag out (handled in the
    parse loop's switch block, where every other per-mon volatile-backed tracker resets).

    Ordering note: the ``-mustrecharge`` line lands on the SAME turn the beam hit, and the
    ``cant`` lands on the NEXT one, so the flag is true across exactly one decision boundary —
    the one where the opponent is choosing what to do against a mon that cannot act. That is the
    decision a k0 policy was blind at, and why the ``cant:recharge`` transition token (the
    protocol inventory's "semantic alias" for this line) was one decision too late.
    """

    event_type = parts[1] if len(parts) > 1 else ""
    if len(parts) < 3:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in must_recharge:
        return
    if event_type == "-mustrecharge":
        must_recharge[slot] = True
    elif event_type == "faint":
        must_recharge[slot] = False
    elif (
        event_type == "cant"
        and len(parts) >= 4
        and _side_condition_identifier(parts[3]) == "recharge"
    ):
        must_recharge[slot] = False


def _future_sight_turns_remaining(replay: "ShowdownReplayState", slot: str) -> int:
    """Turns until a pending delayed attack lands on ``slot``'s side (0 if none/overdue)."""
    landing = replay.future_sight.get(slot)
    if landing is None:
        return 0
    return max(0, landing - replay.turn_number)


def _update_weather(parts: Sequence[str], weather: Optional[str]) -> Optional[str]:
    """Track the active weather from |-weather| lines ('none'/absent clears it)."""
    if (parts[1] if len(parts) > 1 else "") != "-weather":
        return weather
    raw = parts[2].strip() if len(parts) > 2 else ""
    identifier = _normalize_identifier(raw)
    if not identifier or identifier == "none":
        return None
    return identifier


def _flag_baton_pass(parts: Sequence[str], pending_baton_pass: set[str]) -> None:
    """Track whether a side is mid-Baton-Pass so the next switch-in inherits its boosts.

    A |move|...|Baton Pass sets the flag; any *other* move by that side clears a stale flag (so a
    failed/interrupted Baton Pass that never produced a switch can't carry boosts into a later
    unrelated switch). The flag is otherwise consumed by the following switch.
    """
    if (parts[1] if len(parts) > 1 else "") != "move" or len(parts) < 4:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in {"p1", "p2"}:
        return
    if _normalize_identifier(parts[3]) == "batonpass":
        pending_baton_pass.add(slot)
    else:
        pending_baton_pass.discard(slot)


def _line_mentions_baton_pass(parts: Sequence[str]) -> bool:
    """True if a switch line carries a '[from] Baton Pass' tag (trailing protocol fields)."""
    return any("baton pass" in part.lower() for part in parts[4:])


_BOOST_STAGE_LIMIT = 6


def _update_boosts(parts: Sequence[str], boosts: dict[str, dict[str, int]]) -> None:
    """Accumulate per-active-slot stat-boost stages from boost protocol lines."""
    event_type = parts[1] if len(parts) > 1 else ""
    if event_type == "-clearallboost":
        for slot in boosts:
            boosts[slot].clear()
        return
    if event_type in {"-copyboost", "-transform"} and len(parts) >= 4:
        # Psych Up and Transform both copy the target's public boost stages. The latter
        # is emitted as ``|-transform|SOURCE|TARGET`` rather than ``|-copyboost|``.
        source = _slot_from_ident(parts[2])
        target = _slot_from_ident(parts[3])
        if source in boosts and target in boosts:
            boosts[source] = dict(boosts[target])
        return
    if event_type not in {
        "-boost", "-unboost", "-setboost", "-clearboost",
        "-clearpositiveboost", "-clearnegativeboost", "-restoreboost",
    } or len(parts) < 3:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in boosts:
        return
    stages = boosts[slot]
    if event_type == "-clearboost" or event_type == "-restoreboost":
        stages.clear()
        return
    if event_type == "-clearpositiveboost":
        for stat in [s for s, stage in stages.items() if stage > 0]:
            stages.pop(stat, None)
        return
    if event_type == "-clearnegativeboost":
        for stat in [s for s, stage in stages.items() if stage < 0]:
            stages.pop(stat, None)
        return
    if len(parts) < 5:
        return
    stat = parts[3].strip()
    try:
        amount = int(parts[4])
    except (TypeError, ValueError):
        return
    if event_type == "-setboost":
        new_stage = amount
    elif event_type == "-unboost":
        new_stage = stages.get(stat, 0) - amount
    else:  # -boost
        new_stage = stages.get(stat, 0) + amount
    new_stage = max(-_BOOST_STAGE_LIMIT, min(_BOOST_STAGE_LIMIT, new_stage))
    if new_stage == 0:
        stages.pop(stat, None)
    else:
        stages[stat] = new_stage


def _side_condition_max_layers(condition: str) -> int:
    # Spikes is the only multi-layer side condition in Gen 3 (max 3 layers).
    if condition == "spikes":
        return 3
    return 1


def _side_condition_identifier(raw_condition: str) -> str:
    # Strip the source prefix Showdown attaches to some effects (e.g. "move: Leech Seed",
    # "ability: Flash Fire", "item: ...") so the normalized id is the bare effect name.
    condition = raw_condition.strip()
    if ":" in condition and condition.split(":", 1)[0].strip().lower() in {"move", "ability", "item"}:
        condition = condition.split(":", 1)[1].strip()
    return _normalize_identifier(condition)


def _public_event_from_line(line: str) -> ShowdownPublicEvent:
    parts = line.split("|")
    event_type = parts[1] if len(parts) > 1 and parts[1] else "unknown"
    actor_ident: Optional[str] = None
    actor_slot: Optional[str] = None
    target_ident: Optional[str] = None
    target_slot: Optional[str] = None
    primary: Optional[str] = None
    secondary: Optional[str] = None

    if event_type == "player" and len(parts) >= 4:
        actor_slot = parts[2] if parts[2] in {"p1", "p2"} else None
        primary = parts[3]
    elif event_type in {"switch", "drag", "replace"} and len(parts) >= 4:
        actor_ident = parts[2]
        actor_slot = _slot_from_ident(actor_ident)
        primary = _species_from_details(parts[3]) or _species_from_ident(actor_ident)
        secondary = parts[4] if len(parts) > 4 else None
    elif event_type == "move" and len(parts) >= 4:
        actor_ident = parts[2]
        actor_slot = _slot_from_ident(actor_ident)
        primary = parts[3]
        if len(parts) > 4:
            target_ident = parts[4]
            target_slot = _slot_from_ident(target_ident)
    elif event_type in {
        "-ability",
        "ability",
        "-activate",
        "-boost",
        "-curestatus",
        "-damage",
        "-heal",
        "-item",
        "-sideend",
        "-sidestart",
        "-status",
        "-unboost",
        "faint",
    } and len(parts) >= 3:
        target_ident = parts[2]
        target_slot = _slot_from_ident(target_ident)
        primary = parts[3] if len(parts) > 3 else None
        secondary = parts[4] if len(parts) > 4 else None
    elif event_type == "win" and len(parts) >= 3:
        primary = parts[2]
    else:
        actor_ident = parts[2] if len(parts) > 2 and _slot_from_ident(parts[2]) else None
        actor_slot = _slot_from_ident(actor_ident or "")
        primary = parts[3] if len(parts) > 3 else None
        secondary = parts[4] if len(parts) > 4 else None

    return ShowdownPublicEvent(
        event_type=event_type,
        raw_line=line,
        actor_slot=actor_slot,
        actor_ident=actor_ident,
        target_slot=target_slot,
        target_ident=target_ident,
        primary=primary,
        secondary=secondary,
    )


def _relative_public_event(
    event: ShowdownPublicEvent,
    *,
    self_slot: str,
    opponent_slot: str,
) -> PlayerRelativePublicEvent:
    return PlayerRelativePublicEvent(
        event_type=event.event_type,
        raw_line=event.raw_line,
        actor_role=_relative_role(event.actor_slot, self_slot=self_slot, opponent_slot=opponent_slot),
        target_role=_relative_role(event.target_slot, self_slot=self_slot, opponent_slot=opponent_slot),
        primary=event.primary,
        secondary=event.secondary,
        relative_line=_relative_public_line(event, self_slot=self_slot, opponent_slot=opponent_slot),
    )


def _relative_role(slot: str | None, *, self_slot: str, opponent_slot: str) -> str:
    if slot == self_slot:
        return "self"
    if slot == opponent_slot:
        return "opponent"
    return "none"


def _relative_public_line(
    event: ShowdownPublicEvent,
    *,
    self_slot: str,
    opponent_slot: str,
) -> str:
    parts = event.raw_line.split("|")
    if len(parts) < 3:
        return event.raw_line
    normalized = [
        _normalize_public_field(field, self_slot=self_slot, opponent_slot=opponent_slot)
        for field in parts
    ]
    return "|".join(normalized)


def _self_team_from_request(request: Mapping[str, Any] | None, showdown_slot: str) -> tuple[ShowdownPokemon, ...]:
    side = request.get("side") if isinstance(request, Mapping) and isinstance(request.get("side"), Mapping) else {}
    pokemon_rows = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(pokemon_rows, list):
        return ()
    active_moves = _active_request_moves(request)
    team: list[ShowdownPokemon] = []
    for row in pokemon_rows:
        if not isinstance(row, Mapping):
            continue
        ident = str(row.get("ident") or "")
        condition = str(row.get("condition")) if row.get("condition") is not None else None
        row_moves = _request_pokemon_moves(row)
        team.append(
            ShowdownPokemon(
                ident=ident,
                showdown_slot=_slot_from_ident(ident) or showdown_slot,
                species=_species_from_request_pokemon(row),
                condition=condition,
                active=bool(row.get("active")),
                details=str(row.get("details")) if row.get("details") is not None else None,
                moves=row_moves or (active_moves if row.get("active") else ()),
                ability=_request_pokemon_ability(row),
                item=_request_pokemon_item(row),
                stats=_actual_stats_from_request_row(row, condition),
            )
        )
    return tuple(team)


def _actual_stats_from_request_row(row: Mapping[str, Any], condition: str | None) -> dict[str, int] | None:
    """The player mon's actual computed stats from a request row: the 5 battle stats plus max HP.

    The request's ``stats`` object holds atk/def/spa/spd/spe; max HP is the denominator of the
    condition (e.g. "250/250"). Returns None when no stats are present (e.g. simplified payloads).
    """
    raw = row.get("stats")
    stats: dict[str, int] = {}
    if isinstance(raw, Mapping):
        for key in ("atk", "def", "spa", "spd", "spe"):
            value = raw.get(key)
            if isinstance(value, int):
                stats[key] = value
    max_hp = _max_hp_from_condition(condition)
    if max_hp is not None:
        stats["hp"] = max_hp
    return stats or None


def _max_hp_from_condition(condition: str | None) -> int | None:
    """Max HP (the denominator) from a request condition like '180/250'; None for '0 fnt'/absent."""
    if not condition:
        return None
    head = condition.split()[0]
    if "/" not in head:
        return None
    _, _, denominator = head.partition("/")
    return int(denominator) if denominator.isdigit() and int(denominator) > 0 else None


def _hp_numerator_denominator(condition: str | None) -> tuple[int | None, int | None]:
    """Current and max HP from a condition head like '180/250 tox'; (None, None) for '0 fnt'/absent.

    Works for both absolute HP (own/omniscient stream) and the percentage form (``85/100``).
    The latter is recognizable by its `/100` denominator; callers needing an exact Gen 3 damage
    unit must handle its rounded deltas separately.
    """
    if not condition:
        return None, None
    head = condition.split()[0]
    if "/" not in head:
        return None, None
    numerator, _, denominator = head.partition("/")
    current = int(numerator) if numerator.isdigit() else None
    maximum = int(denominator) if denominator.isdigit() and int(denominator) > 0 else None
    return current, maximum


def _opponent_team_from_public_state(
    replay: ShowdownReplayState,
    opponent_slot: str,
) -> tuple[ShowdownPokemon, ...]:
    return tuple(replay.public_revealed.get(opponent_slot, ()))


def _merge_opponent_belief_facts(
    opponent_team: tuple[ShowdownPokemon, ...],
    belief_view: "PlayerBeliefView",
) -> tuple[ShowdownPokemon, ...]:
    """Copy protocol-revealed facts (moves/ability/item) from the belief view onto public rows.

    The belief engine is the single accumulator of opponent reveals; without this merge the
    opponent rows' ``moves``/``ability``/``item`` fields stay permanently empty and metadata
    consumers (dataset shaping, probes) silently see nothing the encoder sees.

    Semantics for consumers (deliberately different from request-sourced self rows):
    - values are identifier-normalized (``leftovers``), not display form;
    - fields mean "ever revealed this game", not "currently held" — a consumed or Knocked-Off
      item stays recorded (that is the belief engine's evidence semantics);
    - ``moves`` lists revealed set members only (Struggle is excluded: it is forced, not a set
      slot) and replaces the public row's value wholesale.
    """
    facts_by_species = {
        _normalize_identifier(belief.species): belief for belief in belief_view.opponent_pokemon
    }
    merged: list[ShowdownPokemon] = []
    for pokemon in opponent_team:
        belief = facts_by_species.get(_normalize_identifier(pokemon.species))
        if belief is None:
            merged.append(pokemon)
            continue
        merged.append(
            replace(
                pokemon,
                moves=tuple(
                    _normalize_identifier(move)
                    for move in belief.revealed_moves
                    if _normalize_identifier(move) != "struggle"
                ),
                ability=(
                    _normalize_identifier(belief.revealed_ability)
                    if belief.revealed_ability
                    else pokemon.ability
                ),
                item=(
                    _normalize_identifier(belief.revealed_item)
                    if belief.revealed_item
                    else pokemon.item
                ),
            )
        )
    return tuple(merged)


def _blank_categorical_rows(spec: ObservationSpec) -> list[list[str]]:
    # Categorical slots hold the raw token *strings* during encoding; observation_from_player_
    # state converts them to compact embedding rows via the CategoryVocabulary in one pass.
    return [[""] * spec.categorical_feature_count for _ in range(spec.token_count)]


def _blank_numeric_rows(
    spec: ObservationSpec, *, internal_numeric_feature_count: int | None = None
) -> list[list[float]]:
    width = internal_numeric_feature_count or spec.numeric_feature_count
    return [[0.0] * width for _ in range(spec.token_count)]


def _project_v3_numeric_rows(legacy_rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """Project private legacy writer rows into the public grouped v3 layout."""

    projected: list[list[float]] = []
    for row_index, row in enumerate(legacy_rows):
        if len(row) != V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT:
            raise ValueError(
                "v3 numeric projection requires the complete legacy writer surface "
                f"({V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT} columns), got {len(row)} on row {row_index}."
            )
        projected.append([row[legacy_index] for legacy_index in V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX])
    return projected


def _project_v4_numeric_rows(writer_rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """Project private writer rows into the public grouped v4 layout (the v3 projection's twin)."""

    projected: list[list[float]] = []
    for row_index, row in enumerate(writer_rows):
        if len(row) != V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT:
            raise ValueError(
                "v4 numeric projection requires the complete writer surface "
                f"({V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT} columns), got {len(row)} on row "
                f"{row_index}."
            )
        projected.append([row[writer_index] for writer_index in V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX])
    return projected


def _encode_field_token(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    state: PlayerRelativeBattleState,
    *,
    masks: ObservationFeatureMasks = DEFAULT_OBSERVATION_FEATURE_MASKS,
    schema_v3: bool = False,
    schema_v4: bool = False,
    dex: "ShowdownDex | None" = None,
) -> None:
    _set_category(categorical_ids[FIELD_TOKEN_OFFSET], CATEGORY_PRIMARY, f"request_kind:{state.request_kind}")
    # Winner identity is deliberately NOT encoded: it is constant ("none") at every decision
    # point (the rollout records observations only while the game is live) and would otherwise
    # be the game outcome leaking into the model input. The SECONDARY slot stays padding.
    _set_category(categorical_ids[FIELD_TOKEN_OFFSET], CATEGORY_ROLE, "field")
    _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_PRESENT, 1.0)
    if state.weather:
        _set_category(categorical_ids[FIELD_TOKEN_OFFSET], CATEGORY_SECONDARY, f"weather:{state.weather}")
    self_haz, self_scr = _side_condition_features(state.self_side_condition_counts)
    opp_haz, opp_scr = _side_condition_features(state.opponent_side_condition_counts)
    _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_SELF_HAZARDS, self_haz)
    _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_OPP_HAZARDS, opp_haz)
    _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_SELF_SCREENS, self_scr)
    _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_OPP_SCREENS, opp_scr)
    if state.turn_number:
        _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_TURN_COUNT, min(1.0, state.turn_number / 1000.0))
    if state.self_future_sight_turns:
        _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_SELF_FUTURE_SIGHT, min(1.0, state.self_future_sight_turns / 2.0))
    if state.opponent_future_sight_turns:
        _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_OPP_FUTURE_SIGHT, min(1.0, state.opponent_future_sight_turns / 2.0))
    if masks.exact_state:
        _encode_field_exact_state(numeric_features[FIELD_TOKEN_OFFSET], state)
    # Spec v3 change 2: the public sleep-clause block bits. Gated ONLY on the schema (not on
    # masks.exact_state — that mask darkens the belief-engine-fed exact-state layer; these
    # bits are a separate, purely public-protocol surface). The columns sit above the v2.2
    # census, so every legacy mode stays byte-frozen.
    if schema_v3:
        if state.self_sleep_clause_blocks:
            _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF, 1.0)
        if state.opponent_sleep_clause_blocks:
            _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP, 1.0)
        # Spec v3 change 9: the per-side Wish turns-to-land clock, on the field token beside the
        # v2.2 pending bits. Public-protocol-derived (like the sleep-clause bits above), so gated on
        # the schema alone; value min(1, remaining/2) reads 2/2 then 1/2 across a Wish's life.
        if state.self_wish_turns:
            _set_numeric(
                numeric_features[FIELD_TOKEN_OFFSET],
                NUMERIC_SELF_WISH_TURNS,
                min(1.0, state.self_wish_turns / 2.0),
            )
        if state.opponent_wish_turns:
            _set_numeric(
                numeric_features[FIELD_TOKEN_OFFSET],
                NUMERIC_OPP_WISH_TURNS,
                min(1.0, state.opponent_wish_turns / 2.0),
            )
    # Spec v4 Part B: the entry-hazard credit / expected-value pair and the items-removed credit,
    # all per side on the field token beside the layer counts they are about. Public-protocol
    # derived, so gated on the schema alone (not masks.exact_state).
    if schema_v4:
        _encode_field_credit_features(numeric_features[FIELD_TOKEN_OFFSET], state, dex=dex)


# (condition id, self numeric slot, opponent numeric slot) for the timed side conditions.
_TIMED_CONDITION_SLOTS = (
    ("reflect", NUMERIC_SELF_REFLECT_TURNS, NUMERIC_OPP_REFLECT_TURNS),
    ("lightscreen", NUMERIC_SELF_LIGHT_SCREEN_TURNS, NUMERIC_OPP_LIGHT_SCREEN_TURNS),
    ("safeguard", NUMERIC_SELF_SAFEGUARD_TURNS, NUMERIC_OPP_SAFEGUARD_TURNS),
    ("mist", NUMERIC_SELF_MIST_TURNS, NUMERIC_OPP_MIST_TURNS),
)


def _encode_field_exact_state(num_row: list[float], state: PlayerRelativeBattleState) -> None:
    """Side-level exact-state features: sleep clause, timed durations, pending Wish."""
    if state.self_sleep_clause_used:
        _set_numeric(num_row, NUMERIC_SELF_SLEEP_CLAUSE, 1.0)
    if state.opponent_sleep_clause_used:
        _set_numeric(num_row, NUMERIC_OPP_SLEEP_CLAUSE, 1.0)
    if state.weather:
        _set_numeric(
            num_row,
            NUMERIC_WEATHER_TURNS,
            min(1.0, state.weather_turns_remaining / float(_TIMED_CONDITION_DURATION)),
        )
        if state.weather_permanent:
            _set_numeric(num_row, NUMERIC_WEATHER_PERMANENT, 1.0)
    for condition, self_slot, opp_slot in _TIMED_CONDITION_SLOTS:
        self_turns = state.self_timed_condition_turns.get(condition, 0)
        if self_turns:
            _set_numeric(num_row, self_slot, min(1.0, self_turns / float(_TIMED_CONDITION_DURATION)))
        opp_turns = state.opponent_timed_condition_turns.get(condition, 0)
        if opp_turns:
            _set_numeric(num_row, opp_slot, min(1.0, opp_turns / float(_TIMED_CONDITION_DURATION)))
    if state.self_wish_pending:
        _set_numeric(num_row, NUMERIC_SELF_WISH_PENDING, 1.0)
    if state.opponent_wish_pending:
        _set_numeric(num_row, NUMERIC_OPP_WISH_PENDING, 1.0)


# Gen 3 has a single entry hazard (Spikes, max 3 layers); Toxic Spikes / Stealth Rock are
# Gen 4+. Screens are Reflect + Light Screen.
_HAZARD_CONDITIONS = ("spikes",)
_SCREEN_CONDITIONS = ("reflect", "lightscreen")
# Boost stats encoded on the active mon, in (Showdown stat key, numeric slot) order.
_BOOST_STAT_SLOTS = (
    ("atk", NUMERIC_BOOST_ATK),
    ("def", NUMERIC_BOOST_DEF),
    ("spa", NUMERIC_BOOST_SPA),
    ("spd", NUMERIC_BOOST_SPD),
    ("spe", NUMERIC_BOOST_SPE),
)


# Gen 3 Spikes damage as a fraction of the incoming mon's max HP, indexed by layer count
# (0 layers = no damage). Engine ground truth, and the same ladder engine_world prices.
_SPIKES_DAMAGE_BY_LAYERS = (0.0, 1.0 / 8.0, 1.0 / 6.0, 1.0 / 4.0)
# Both Part-B credit families are normalized by the TEAM: six mons, so six items and six
# max-HP units. The opponent's real max HPs are hidden, so an equal-share denominator is the
# only public normalization available — and using the same one on both sides keeps the four
# hazard columns and the two item columns on one comparable scale.
_TEAM_SIZE = 6.0


def _is_grounded_for_spikes(
    pokemon: ShowdownPokemon,
    *,
    belief: "RevealedPokemonBelief | None",
    dex: "ShowdownDex | None",
) -> bool:
    """Whether Spikes would damage this mon on entry, from PUBLIC knowledge only.

    The gen3 grounding rule as the engine applies it (``engine_world``'s trap/hazard test):
    Flying types and Levitate are exempt, everything else takes the chip.

    Conservative by construction — this feeds an EXPECTED-value column, and the honest failure
    direction is to over-count rather than to claim an immunity we cannot see. A mon whose
    species is not yet revealed, or whose ability is still ambiguous, counts as grounded; only a
    revealed Flying type or a revealed (or uniquely-implied) Levitate removes it. Our own team is
    fully known, so the same code is exact on the self side.
    """

    species_info = dex.species_info(pokemon.species) if dex is not None else None
    if species_info is not None and any(
        _normalize_identifier(type_name) == "flying" for type_name in species_info.types
    ):
        return False
    ability = _normalize_identifier(pokemon.ability or "")
    if not ability and belief is not None:
        ability = _normalize_identifier(belief.revealed_ability or "")
        if not ability:
            possible = [
                _normalize_identifier(candidate)
                for candidate in (belief.possible_abilities or ())
                if _normalize_identifier(candidate)
            ]
            # A single remaining candidate is a public conclusion, not a guess: the belief layer
            # has already eliminated every other set variant.
            if len(possible) == 1:
                ability = possible[0]
    return ability != "levitate"


def _healthy_grounded_bench(
    team: Sequence[ShowdownPokemon],
    *,
    beliefs_by_species: Mapping[str, "RevealedPokemonBelief"] | None,
    dex: "ShowdownDex | None",
    unseen_slots: int = 0,
) -> int:
    """Count of BENCHED, unfainted, Spikes-grounded mons — the population a layer still bills.

    The active mon is excluded: it is already on the field and will not pay entry chip again
    unless it leaves and returns, which is precisely the future the value column is pricing.

    ``unseen_slots`` is the opponent's NOT-YET-REVEALED party members, and it is load-bearing.
    ``state.opponent_team`` holds only revealed mons, so counting that list alone made the
    column smallest exactly when Spikes are worth most — early, before reveals — and made its
    magnitude a proxy for how much of their team we have seen rather than for hazard exposure.
    It also inverted the documented conservative default (an unknown mon counts as GROUNDED,
    because we must not claim an immunity we cannot see) and broke comparability with the self
    side, which is always a complete six.
    """

    count = max(0, unseen_slots)
    for pokemon in team:
        if pokemon.active:
            continue
        if _condition_features(pokemon.condition).fainted:
            continue
        belief = (
            _belief_for_species(beliefs_by_species, pokemon.species)
            if beliefs_by_species
            else None
        )
        if _is_grounded_for_spikes(pokemon, belief=belief, dex=dex):
            count += 1
    return count


def field_credit_values(
    state: PlayerRelativeBattleState,
    *,
    dex: "ShowdownDex | None",
) -> dict[str, float]:
    """The six settled Part-B column values (spec v4), by metadata key.

    SINGLE DERIVATION, TWO CONSUMERS — the same rule the pack itself is built on. The Python
    encoder writes these, and ``_observation_metadata`` publishes the identical numbers so the
    native Rust leaf encoder can read them instead of re-implementing the gen3 grounding rule.
    A re-derivation is exactly the kind of thing that drifts silently between two languages.
    """

    values: dict[str, float] = {}
    values["self_hazard_credit"] = min(1.0, state.self_hazard_damage_suffered / _TEAM_SIZE)
    values["opponent_hazard_credit"] = min(
        1.0, state.opponent_hazard_damage_suffered / _TEAM_SIZE
    )
    self_layers = min(
        3, sum(int(state.self_side_condition_counts.get(n, 0)) for n in _HAZARD_CONDITIONS)
    )
    opp_layers = min(
        3, sum(int(state.opponent_side_condition_counts.get(n, 0)) for n in _HAZARD_CONDITIONS)
    )
    values["self_hazard_expected"] = (
        min(
            1.0,
            _healthy_grounded_bench(state.self_team, beliefs_by_species=None, dex=dex)
            * _SPIKES_DAMAGE_BY_LAYERS[self_layers]
            / _TEAM_SIZE,
        )
        if self_layers
        else 0.0
    )
    values["opponent_hazard_expected"] = (
        min(
            1.0,
            _healthy_grounded_bench(
                state.opponent_team,
                beliefs_by_species=state.belief_view.opponent_by_species(),
                dex=dex,
                # Every party slot we have not seen yet is a benched, living, presumed-grounded
                # mon. int(_TEAM_SIZE) is the gen3 singles party size.
                unseen_slots=int(_TEAM_SIZE) - len(state.opponent_team),
            )
            * _SPIKES_DAMAGE_BY_LAYERS[opp_layers]
            / _TEAM_SIZE,
        )
        if opp_layers
        else 0.0
    )
    values["self_items_removed_credit"] = min(1.0, state.self_items_removed / _TEAM_SIZE)
    values["opponent_items_removed_credit"] = min(
        1.0, state.opponent_items_removed / _TEAM_SIZE
    )
    return values


def _encode_field_credit_features(
    num_row: list[float],
    state: PlayerRelativeBattleState,
    *,
    dex: "ShowdownDex | None",
) -> None:
    """Part B credit + expected-value columns on the field token (spec v4).

    ORIENTATION, shared with NUMERIC_SELF_HAZARDS / NUMERIC_OPP_HAZARDS: ``self_*`` is about our
    own ground — layers on our side, damage our mons took, items of ours that were knocked off.
    ``opp_*`` is the opponent's ground, which is where OUR hazards' and OUR Knock Offs' payoff
    shows up. Reading them as "credit we earned" therefore means reading the ``opp_*`` column,
    exactly as "layers we laid" means NUMERIC_OPP_HAZARDS.
    """

    values = field_credit_values(state, dex=dex)
    for key, column in (
        ("self_hazard_credit", NUMERIC_SELF_HAZARD_CREDIT),
        ("opponent_hazard_credit", NUMERIC_OPP_HAZARD_CREDIT),
        ("self_hazard_expected", NUMERIC_SELF_HAZARD_EXPECTED),
        ("opponent_hazard_expected", NUMERIC_OPP_HAZARD_EXPECTED),
        ("self_items_removed_credit", NUMERIC_SELF_ITEMS_REMOVED_CREDIT),
        ("opponent_items_removed_credit", NUMERIC_OPP_ITEMS_REMOVED_CREDIT),
    ):
        value = values[key]
        if value:
            _set_numeric(num_row, column, value)


def _side_condition_features(counts: Mapping[str, int]) -> tuple[float, float]:
    """(hazard layers /3, screens active /2) for one side's condition counts."""
    hazards = sum(int(counts.get(name, 0)) for name in _HAZARD_CONDITIONS)
    screens = sum(1 for name in _SCREEN_CONDITIONS if counts.get(name))
    return min(1.0, hazards / 3.0), min(1.0, screens / 2.0)


def _encode_active_boosts(num_row: list[float], boosts: Mapping[str, int] | None) -> None:
    """Set the five stat-boost-stage slots (stage/6, clamped to [-1, 1]) for an active mon."""
    if not boosts:
        return
    for stat_key, slot in _BOOST_STAT_SLOTS:
        stage = boosts.get(stat_key)
        if stage:
            _set_numeric(num_row, slot, max(-1.0, min(1.0, float(stage) / 6.0)))


class VolatileBucketOverflowWarning(UserWarning):
    """A mon carried more tracked volatiles than the bag has buckets.

    Non-fatal by design: the encode still produces a valid, correctly-shaped observation with
    the overflow truncated, so no run, cache, or sample can be broken by it. But it is a real
    loss of public state, so it is announced rather than swallowed — the silent-truncation
    failure class the divergence ledger exists to eliminate.
    """


# Count of truncated volatile-bag overflows since process start. A warning is emitted once per
# call site under Python's default filter, which is the right volume for a signal that should
# never fire; this counter is what a long fleet run can actually poll to prove it did not.
VOLATILE_BUCKET_OVERFLOWS = 0


def _encode_active_volatiles(cat_row: list[str], volatiles: Sequence[str]) -> None:
    """Place active-mon volatile statuses (sorted) positionally into the volatile columns.

    OVERFLOW IS LOUD BUT NEVER FATAL. Six buckets cover every reachable simultaneous set: over
    160 random-legal self-play games (337,314 slot observations) the observed maximum was TWO,
    and 23 of the 38 tracked volatiles have no carrier in the gen3 randbat pool at all. If that
    ever stops being true the excess is still truncated — an encode must not be able to abort a
    collection run or corrupt a sample — but it warns and increments a counter so the condition
    surfaces instead of silently dropping public state.

    Position within the bag carries no meaning: the model SUMS the categorical embeddings, so a
    volatile contributes the same vector from any column. Only membership matters, which is why
    truncation (not ordering) is the whole risk here.
    """

    global VOLATILE_BUCKET_OVERFLOWS
    names = sorted(set(volatiles))
    if len(names) > VOLATILE_BUCKET_COUNT:
        VOLATILE_BUCKET_OVERFLOWS += 1
        dropped = names[VOLATILE_BUCKET_COUNT:]
        # Wrapped: a `-W error` profile would otherwise turn this deliberately NON-FATAL
        # diagnostic into an encode exception, breaking the very collection run the truncation
        # exists to protect. The counter above is the signal that always survives.
        try:
            warnings.warn(
                f"volatile bag overflow: {len(names)} tracked volatiles on one mon exceeds the "
                f"{VOLATILE_BUCKET_COUNT} buckets; dropping {dropped}. The observation is "
                "still valid and the run continues, but this is public state the model cannot "
                "see — raise VOLATILE_BUCKET_COUNT (a schema change) if it recurs.",
                VolatileBucketOverflowWarning,
                stacklevel=2,
            )
        except Exception:  # pragma: no cover - a warning must never abort an encode
            pass
    for index, name in enumerate(names[:VOLATILE_BUCKET_COUNT]):
        column = CATEGORY_VOLATILE_OFFSET + index
        if column >= len(cat_row):
            break
        cat_row[column] = f"volatile:{_normalize_identifier(name)}"


def _species_info_base_fallback(dex: "ShowdownDex | None", species: str | None):
    """dex.species_info with a cosmetic-forme fallback to the base species.

    gen3 randbats emit Unown as lettered cosmetic formes (Unown-C, Unown-Z,
    Unown-Exclamation, ...) that are NOT separate Pokedex entries, so the direct dex
    lookup misses and the mon encodes with blank types + zero base stats. When the
    direct lookup misses, retry with the canonical base-species id
    (``canonical_gen3_randbat_species_id`` from randbat.py — the same collapse the
    world/belief path uses). That function only collapses genuine Unown cosmetic
    suffixes; real distinct dex formes (Deoxys-Attack/Defense/Speed, Castform,
    Nidoran-F/M, ...) resolve on the direct lookup and never reach the fallback, so
    they are left untouched.
    """
    if dex is None or not species:
        return None
    info = dex.species_info(species)
    if info is not None:
        return info
    canonical = canonical_gen3_randbat_species_id(species)
    if canonical and canonical != species:
        return dex.species_info(canonical)
    return None


# Explicit forme->type fallback for `-formechange` retypes whose forme is ABSENT from the dex
# (the Unown-cosmetic situation). Castform's weather formes ARE present in the gen3 dex
# (Castform-Sunny=Fire, -Rainy=Water, -Snowy=Ice), so the dex path is taken and this map is a
# fail-safe only; base Castform is Normal.
_FORMECHANGE_TYPE_FALLBACK = {
    "castformsunny": "Fire",
    "castformrainy": "Water",
    "castformsnowy": "Ice",
    "castform": "Normal",
}


def _resolve_live_type_slots(
    source: str | None, dex: "ShowdownDex | None"
) -> tuple[str, str | None] | None:
    """Resolve a ``ShowdownPokemon.live_type_source`` discriminant to (type1, type2 or None).

    ``type:<T>`` payloads (Color Change ``typechange``) already carry the type. ``forme:<Forme>``
    payloads (Castform Forecast) resolve to the forme's type from the dex first (Castform formes
    are real dex entries, like Deoxys), falling back to the explicit map for a dex-absent forme.
    Returns None when unresolvable (leaves the base dex type untouched). Both live retypes are
    mono-type; the ``/``-split tolerates a hypothetical dual-type payload defensively.
    """
    if not source:
        return None
    kind, _, payload = source.partition(":")
    payload = payload.strip()
    if not payload:
        return None
    if kind == "type":
        types = [segment.strip() for segment in payload.split("/") if segment.strip()]
        if not types:
            return None
        return types[0], (types[1] if len(types) > 1 else None)
    if kind == "forme":
        info = _species_info_base_fallback(dex, payload) if dex is not None else None
        if info is not None and info.types:
            return info.types[0], (info.types[1] if len(info.types) > 1 else None)
        mapped = _FORMECHANGE_TYPE_FALLBACK.get(_normalize_identifier(payload))
        if mapped:
            return mapped, None
    return None


def _encode_species_type_categories(row: list[int], dex: "ShowdownDex | None", species: str | None) -> None:
    """Set the two type slots for a Pokemon token from the dex (no-op without a dex)."""
    if dex is None or not species:
        return
    info = _species_info_base_fallback(dex, species)
    if info is None:
        return
    if len(info.types) >= 1:
        _set_category(row, CATEGORY_TYPE_1, f"type:{info.types[0]}")
    if len(info.types) >= 2:
        _set_category(row, CATEGORY_TYPE_2, f"type:{info.types[1]}")


def _level_from_details(details: str | None) -> int | None:
    """Extract the level from a details string like 'Charizard, L83, M'.

    Showdown OMITS the level token from a Pokemon's details string when — and only
    when — the level is exactly 100 (vendored ``sim/pokemon.ts::getUpdatedDetails``:
    ``name + (level === 100 ? '' : `, L${level}`)``; the ``, L<level>`` token is
    present for every level != 100 and absent only at 100). So a details string that
    carries a species name but no ``L`` token means level 100, not "unknown". Returns
    None only when there is no details string at all (genuinely no level information).
    """
    if not details:
        return None
    for part in details.split(","):
        token = part.strip()
        if token.startswith("L") and token[1:].isdigit():
            return int(token[1:])
    return 100


_BASE_STAT_SLOTS = (
    ("hp", NUMERIC_BASE_HP),
    ("atk", NUMERIC_BASE_ATK),
    ("def", NUMERIC_BASE_DEF),
    ("spa", NUMERIC_BASE_SPA),
    ("spd", NUMERIC_BASE_SPD),
    ("spe", NUMERIC_BASE_SPE),
)


_ACTUAL_STAT_SLOTS = (
    ("hp", NUMERIC_ACTUAL_HP),
    ("atk", NUMERIC_ACTUAL_ATK),
    ("def", NUMERIC_ACTUAL_DEF),
    ("spa", NUMERIC_ACTUAL_SPA),
    ("spd", NUMERIC_ACTUAL_SPD),
    ("spe", NUMERIC_ACTUAL_SPE),
)
# Gen 3 maximum possible stat (Blissey HP at level 100); normalizing by it keeps every actual
# stat in [0, 1] with no saturation.
_ACTUAL_STAT_DIVISOR = 714.0


def _encode_pokemon_stats(
    num_row: list[float], dex: "ShowdownDex | None", species: str | None, details: str | None
) -> None:
    """Set level + species base stats (dex-derived, public) for a pokemon/switch token."""
    level = _level_from_details(details)
    if level is not None:
        _set_numeric(num_row, NUMERIC_LEVEL, min(1.0, level / 100.0))
    if dex is None or not species:
        return
    info = _species_info_base_fallback(dex, species)
    if info is None:
        return
    for stat_key, slot in _BASE_STAT_SLOTS:
        value = info.base_stats.get(stat_key)
        if value:
            _set_numeric(num_row, slot, min(1.0, float(value) / 200.0))


def _encode_actual_stats(num_row: list[float], stats: Mapping[str, int] | None) -> None:
    """Set the player mon's actual computed stats (known only for the self team; no-op otherwise)."""
    if not stats:
        return
    for stat_key, slot in _ACTUAL_STAT_SLOTS:
        value = stats.get(stat_key)
        if value:
            _set_numeric(num_row, slot, min(1.0, float(value) / _ACTUAL_STAT_DIVISOR))


def _encode_move_mechanics(
    cat_row: list[int],
    num_row: list[float],
    dex: "ShowdownDex | None",
    move_name: str,
    user_types: Sequence[str] = (),
    user_hp_fraction: float | None = None,
) -> None:
    """Set move type / damage class (categorical) + base power / priority / accuracy + effect.

    ``user_types`` and ``user_hp_fraction`` are the acting (self active) mon's types and current HP
    fraction, used to resolve type-dependent effects (Curse) and HP-variable base power
    (Reversal / Flail / Eruption / Water Spout) at encode time.
    """
    if dex is None:
        return
    move = dex.move_info(move_name)
    if move is None:
        return
    base_power = resolve_move_base_power(move, user_hp_fraction)
    # Struggle is TYPELESS from Generation II onward: neutral vs every type (it
    # HITS Ghosts) and grants no STAB. The Showdown dex still records Struggle as
    # Normal-type, so emit the enumerated typeless token `type:???` directly —
    # mirroring gen3 Curse, whose dex type is already "???". Category (Physical)
    # and base power (50) are unchanged. This aligns the SELF forced-Struggle
    # action token with the engine fix that makes gen3 Struggle PokemonType::TYPELESS
    # (third_party/poke-engine-gen3-struggle-typeless.patch).
    move_type = "???" if move.id == "struggle" else move.type
    _set_category(cat_row, CATEGORY_TYPE_1, f"type:{move_type}")
    _set_category(cat_row, CATEGORY_MOVE_CATEGORY, f"move_category:{move.gen3_category}")
    _set_category(cat_row, CATEGORY_MOVE_PRIORITY, f"move_priority:{move.priority}")
    _set_numeric(num_row, NUMERIC_BASE_POWER, min(1.0, float(base_power) / 200.0))
    _set_numeric(num_row, NUMERIC_PRIORITY, max(-1.0, min(1.0, float(move.priority) / 5.0)))
    _set_numeric(num_row, NUMERIC_ACCURACY, (float(move.accuracy) / 100.0) if move.accuracy else 1.0)
    effect_label, effect_chance, self_hp_cost = resolve_move_effect(move, user_types)
    if effect_label:
        _set_category(cat_row, CATEGORY_MOVE_EFFECT, f"move_effect:{effect_label}")
    _set_numeric(num_row, NUMERIC_EFFECT_CHANCE, min(1.0, float(effect_chance) / 100.0))
    _set_numeric(num_row, NUMERIC_SELF_HP_COST, max(0.0, min(1.0, float(self_hp_cost))))


def _encode_active_feature_pack(
    cat_row: list[str],
    num_row: list[float],
    *,
    truant_loaf: bool,
    last_used_move: str | None,
    arrived_by_baton_pass: bool,
    choice_locked: bool,
    item_swapped: bool,
    traced_ability: str | None,
    last_damage_dealt: float,
    last_damage_taken: float,
) -> None:
    """The per-mon half of the v4 k0 feature pack, written on a side's ACTIVE token.

    Every value here is a CURRENT-STATE fact the search world (or the history region) already
    had and the observation did not — see the column comments for each one's provenance. Unset
    stays 0 / padding throughout: a mon that never moved writes no last-move label, a
    non-Trace-holder writes no ability, and a quiet round writes no damage.
    """

    if truant_loaf:
        _set_numeric(num_row, NUMERIC_TRUANT_LOAF, 1.0)
    if last_damage_dealt > 0.0:
        _set_numeric(num_row, NUMERIC_LAST_DAMAGE_DEALT, min(1.0, last_damage_dealt))
    if last_damage_taken > 0.0:
        _set_numeric(num_row, NUMERIC_LAST_DAMAGE_TAKEN, min(1.0, last_damage_taken))
    if choice_locked:
        _set_numeric(num_row, NUMERIC_CHOICE_LOCKED, 1.0)
    if item_swapped:
        _set_numeric(num_row, NUMERIC_ITEM_SWAPPED, 1.0)
    if last_used_move:
        # The parser stores the ``"switch"`` sentinel in the same field as move ids; it maps to a
        # DISTINCT label so the bag can tell "came in this turn" (a fact Encore keys off) from a
        # move identity, and both from the padding state "has never moved". A Baton-Pass arrival
        # gets its OWN sentinel: it is a different arrival — boosts and the transferable
        # volatiles came with it — and the explicit switch-reason that records this lives only
        # in the history region, so at k0 the distinction would otherwise be lost.
        if _normalize_identifier(last_used_move) == "switch":
            label = (
                LAST_USED_MOVE_BATON_PASS_SENTINEL
                if arrived_by_baton_pass
                else LAST_USED_MOVE_SWITCH_SENTINEL
            )
        else:
            label = f"move:{_normalize_identifier(last_used_move)}"
        _set_category(cat_row, CATEGORY_LAST_USED_MOVE, label)
    if traced_ability:
        _set_category(cat_row, CATEGORY_TRACED_ABILITY, f"ability:{_normalize_identifier(traced_ability)}")


def _encode_pokemon_tokens(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    offset: int,
    pokemon: Sequence[ShowdownPokemon],
    *,
    role: str,
    limit: int,
    beliefs_by_species: Mapping[str, RevealedPokemonBelief] | None = None,
    active_boosts: Mapping[str, int] | None = None,
    active_volatiles: Sequence[str] = (),
    active_toxic_stage: int = 0,
    active_stall_counter: int = 0,
    active_confusion_elapsed: int = 0,
    active_encore_elapsed: int = 0,
    active_wrap_trap_elapsed: int = 0,
    dex: "ShowdownDex | None" = None,
    exact_beliefs_by_species: Mapping[str, RevealedPokemonBelief] | None = None,
    tendency_by_species: Mapping[str, "OpponentMonTendency"] | None = None,
    transform_targets_by_species: Mapping[str, ShowdownPokemon] | None = None,
    masks: ObservationFeatureMasks = DEFAULT_OBSERVATION_FEATURE_MASKS,
    schema_v2_1: bool = False,
    schema_v3: bool = False,
    schema_v4: bool = False,
    tier2_cb_pinned_species: frozenset[str] = frozenset(),
    tier2_investment_pinned: Mapping[str, float] | None = None,
    active_meanlook_trap: bool = False,
    # ---- spec v4 k0 feature pack, all describing the side's ACTIVE mon. -----------------------
    active_must_recharge: bool = False,
    active_truant_loaf: bool = False,
    active_last_used_move: str | None = None,
    active_arrived_by_baton_pass: bool = False,
    active_choice_locked: bool = False,
    active_item_swapped: bool = False,
    active_traced_ability: str | None = None,
    active_last_damage_dealt: float = 0.0,
    active_last_damage_taken: float = 0.0,
    # Per-opponent-mon (switched, stayed) against OUR current active; opponent
    # tokens only, and empty under every schema below v4.
    matchup_switch_evidence: Mapping[str, tuple[int, int]] | None = None,
) -> None:
    # Spec v3 change 7: reuse the determinization gender parser (single source of truth for the
    # ``, M`` / ``, F`` details convention). Imported lazily to avoid a module-load cycle
    # (determinization imports the observation stack that imports this module) and only when the v3
    # gender columns actually exist.
    gender_from_details = None
    if schema_v3:
        from .determinization import _gender_from_details as gender_from_details
    for slot_index, candidate in enumerate(pokemon[:limit]):
        token_index = offset + slot_index
        belief = _belief_for_species(beliefs_by_species, candidate.species)
        condition = _condition_features(belief.condition if belief is not None else candidate.condition)
        revealed_moves = belief.revealed_moves if belief is not None else ()
        revealed_ability = belief.revealed_ability if belief is not None else None
        revealed_item = belief.revealed_item if belief is not None else None
        # CURRENT-held: True once the mon has publicly parted with its item — Knock Off /
        # a Trick that returned nothing / a consumed berry or White Herb. belief.item_removed
        # is the audited "holds nothing now" flag (belief.py sets it on every such surface).
        # revealed_item keeps NAMING the (now-gone) item so the possible_item set-identity
        # columns below still narrow the opponent's set; only the current-possession signal
        # (NUMERIC_REVEALED_ITEM) reflects the removal. Unaudited 0-occurrence mutations
        # (Thief/Covet: item_mutated without item_removed) stay fail-closed as still-held.
        item_removed = belief.item_removed if belief is not None else False
        possible_abilities = belief.possible_abilities if belief is not None else ()
        possible_items = belief.possible_items if belief is not None else ()
        possible_moves = belief.possible_moves if belief is not None else ()
        # Own mons carry no belief entry (they are fully known by design), so the belief-derived
        # reveals above are empty and the self-token item/ability buckets would encode nothing —
        # the policy could not condition on its OWN current item or ability. Populate them straight
        # from ``candidate``, which holds the request's CURRENT-held item + current ability (this is
        # exactly how self stats/details already flow — direct from the request row, not through the
        # belief engine). Zero uncertainty: the singleton collapses possible_items/possible_abilities
        # to the known value (NUMERIC_UNCERTAINTY is already forced to 0.0 for self above). CURRENT-
        # held semantics are honored for free: ``candidate.item`` is empty once the request shows the
        # mon holding nothing (Knock Off / Trick / consumed berry / White Herb), so a stripped mon
        # encodes not-currently-held (revealed_item -> None -> NUMERIC_REVEALED_ITEM 0.0, empty
        # bucket). ``item_removed`` stays False because the removal already surfaced as an empty
        # item — the opponent-side ever-revealed/current-held split does not apply to the self side,
        # where the request never names a parted-with item. Nothing not request-known is exposed.
        if role == "self":
            revealed_ability = candidate.ability or None
            revealed_item = candidate.item or None
            possible_abilities = (revealed_ability,) if revealed_ability else ()
            possible_items = (revealed_item,) if revealed_item else ()
            item_removed = False
        ability_feature_values = _known_or_possible_values(revealed_ability, possible_abilities)
        item_feature_values = _known_or_possible_values(revealed_item, possible_items)
        candidate_set_count = belief.candidate_set_count if belief is not None else None
        # Own mons are fully known (their belief entry is None by design): uncertainty
        # is 0.0, not the max-entropy default — the previous constant 1.0 was
        # semantically inverted (audit section 6 wart; cosmetic, constant either way).
        if role == "self":
            uncertainty = 0.0
        else:
            uncertainty = belief.uncertainty if belief is not None else 1.0
        # A transformed mon (Ditto) fights as its target: encode species, types and base stats from
        # the copied identity so the model sees the effective battler, not Ditto's base 48-across.
        # Transform copies everything EXCEPT HP and level, so base HP stays the original's (a
        # transformed Ditto is still frail) and level comes from the original's details.
        #
        # The Transform flag lives in whichever per-mon ledger tracks this side's exact state. The
        # OPPONENT passes its set-source belief as ``beliefs_by_species`` (carrying the flag), but
        # the SELF side passes only ``exact_beliefs_by_species`` — its ``beliefs_by_species`` is
        # None by design — so ``belief`` is None for our own transformed Ditto and the copied
        # identity would never surface (self token stuck on ditto/Normal/48-across while the belief
        # engine correctly holds transform_species). Fall back to the exact belief when the
        # set-source belief lacks the flag. For the opponent both maps resolve to the same object,
        # so this is a no-op there; a non-transformed self mon is likewise unchanged.
        transform_belief = belief
        if not (transform_belief is not None and transform_belief.transformed):
            transform_belief = _belief_for_species(exact_beliefs_by_species, candidate.species)
        transformed = (
            transform_belief is not None
            and transform_belief.transformed
            and bool(transform_belief.transform_species)
        )
        enc_species = transform_belief.transform_species if transformed else candidate.species
        _set_category(categorical_ids[token_index], CATEGORY_PRIMARY, f"species:{enc_species}")
        _encode_species_type_categories(categorical_ids[token_index], dex, enc_species)
        # In-battle LIVE retype (Castform Forecast forme / Kecleon Color Change): override ONLY the
        # type slots from the retype payload, keeping the base species token (retyped formes are
        # OOV for the species vocab). Set only on the active mon (see _apply_live_type_override).
        if candidate.live_type_source:
            resolved = _resolve_live_type_slots(candidate.live_type_source, dex)
            if resolved is not None:
                override_type1, override_type2 = resolved
                _set_category(categorical_ids[token_index], CATEGORY_TYPE_1, f"type:{override_type1}")
                _set_category(
                    categorical_ids[token_index],
                    CATEGORY_TYPE_2,
                    f"type:{override_type2}" if override_type2 else "",
                )
        _encode_pokemon_stats(numeric_features[token_index], dex, enc_species, candidate.details)
        if transformed and dex is not None:
            original = dex.species_info(candidate.species)
            original_hp = original.base_stats.get("hp") if original is not None else None
            if original_hp:
                _set_numeric(numeric_features[token_index], NUMERIC_BASE_HP, min(1.0, float(original_hp) / 200.0))
        _encode_actual_stats(numeric_features[token_index], candidate.stats)
        # Spec v3 change 7: per-mon gender on EVERY token, schema >= v3 only. Two 0/1 bits from the
        # mon's TRUE details (``candidate.details`` — Transform copies species/stats but NOT gender,
        # so this stays the real mon's sex): male -> (MALE, FEMALE) = (1, 0), female -> (0, 1),
        # genderless / not-yet-revealed -> (0, 0). Above the v2.2 census, so legacy modes stay
        # byte-frozen.
        if schema_v3:
            gender = gender_from_details(candidate.details)
            if gender == "M":
                _set_numeric(numeric_features[token_index], NUMERIC_GENDER_MALE, 1.0)
            elif gender == "F":
                _set_numeric(numeric_features[token_index], NUMERIC_GENDER_FEMALE, 1.0)
        if candidate.active:
            _encode_active_boosts(numeric_features[token_index], active_boosts)
            # Spec v4 pack A1: mustrecharge joins the volatile bag from the parser tracker.
            # Injected HERE rather than in the state field so v3's bag is untouched — the
            # label has no v3 vocabulary row and would hash into the OOV band there.
            bag = active_volatiles
            if schema_v4 and active_must_recharge:
                bag = tuple(bag) + (MUST_RECHARGE_VOLATILE,)
            _encode_active_volatiles(categorical_ids[token_index], bag)
            if active_toxic_stage:
                _set_numeric(numeric_features[token_index], NUMERIC_TOXIC_STAGE, min(1.0, active_toxic_stage / 15.0))
            # Spec v3 change 3: the public consecutive-stall counter, written on the ACTIVE mon
            # like the toxic stage above. Schema-gated so the column does not even exist below the
            # v3 census, keeping v2.2 output byte-identical.
            if schema_v3 and active_stall_counter:
                _set_numeric(numeric_features[token_index], NUMERIC_STALL_COUNTER, min(1.0, active_stall_counter / 8.0))
            # Spec v3 change 4: confusion turns-so-far on the confused (active) mon's token,
            # schema >= v3 only. Gen3 confusion maxes at 5 turns, so CAP = 5 and the ramp
            # saturates at 1.0. The column sits above the v2.2 census, so legacy modes stay
            # byte-frozen; the counter is 0 (unwritten) whenever the active mon is not confused.
            if schema_v3 and active_confusion_elapsed:
                _set_numeric(
                    numeric_features[token_index],
                    NUMERIC_CONFUSION_TURNS,
                    min(1.0, active_confusion_elapsed / 5.0),
                )
            # Spec v3 change 5: encore turns-so-far on the encored (active) mon's token,
            # schema >= v3 only. Gen3 Encore maxes at 6 turns (gen3 mod random(3,7)), so CAP = 6
            # and the ramp saturates at 1.0. The column sits above the v2.2 census, so legacy
            # modes stay byte-frozen; the counter is 0 (unwritten) whenever the active mon is not
            # encored.
            if schema_v3 and active_encore_elapsed:
                _set_numeric(
                    numeric_features[token_index],
                    NUMERIC_ENCORE_TURNS,
                    min(1.0, active_encore_elapsed / 6.0),
                )
            # Spec v3 change 6: Wrap (partial-trap) turns-so-far on the TRAPPED (active) mon's
            # token, schema >= v3 only. Gen3 partial-trap (Wrap) maxes at 5 turns, so CAP = 5 and
            # the ramp saturates at 1.0. The column sits above the v2.2 census, so legacy modes stay
            # byte-frozen; the counter is 0 (unwritten) whenever the active mon is not partially
            # trapped.
            if schema_v3 and active_wrap_trap_elapsed:
                _set_numeric(
                    numeric_features[token_index],
                    NUMERIC_WRAP_TRAP_TURNS,
                    min(1.0, active_wrap_trap_elapsed / 5.0),
                )
            # Spec v3 change 8: Mean Look / Spider Web move-trap on the TRAPPED (active) mon's
            # token, schema >= v3 only — a 0/1 "switch-locked by Mean Look / Spider Web" flag,
            # DISTINCT from the Wrap partial-trap column above and from the ability-trap signal.
            # The column sits above the v2.2 census, so legacy modes stay byte-frozen; the bit is
            # 0 (unwritten) whenever the active mon is not move-trapped.
            if schema_v3 and active_meanlook_trap:
                _set_numeric(numeric_features[token_index], NUMERIC_MEANLOOK_TRAP, 1.0)
            if schema_v4:
                _encode_active_feature_pack(
                    categorical_ids[token_index],
                    numeric_features[token_index],
                    truant_loaf=active_truant_loaf,
                    # A2 is separately maskable: the plan's arm pair is k0+pack vs
                    # k0+pack+lastmove, differing in exactly this column.
                    last_used_move=(
                        active_last_used_move if masks.feature_pack_last_move else None
                    ),
                    arrived_by_baton_pass=active_arrived_by_baton_pass,
                    choice_locked=active_choice_locked,
                    item_swapped=active_item_swapped,
                    traced_ability=active_traced_ability,
                    last_damage_dealt=active_last_damage_dealt,
                    last_damage_taken=active_last_damage_taken,
                )
        status = belief.status if belief is not None and belief.status is not None else condition.status
        _set_category(categorical_ids[token_index], CATEGORY_SECONDARY, f"status:{status}")
        _set_category(categorical_ids[token_index], CATEGORY_ROLE, f"pokemon:{role}")
        # The party-slot index (self_slot/opponent_slot) is intentionally NOT encoded: team order
        # is arbitrary in random battles, so the index carries no actionable signal, and the
        # token's position in the sequence + token_type already identify which team slot it is.
        # (The SLOT column stays in use on action tokens for move_slot/switch_slot.)
        _encode_belief_fact_categories(categorical_ids[token_index], "possible_ability", ability_feature_values)
        _encode_belief_fact_categories(categorical_ids[token_index], "possible_item", item_feature_values)
        # Moves mirror ability/item: revealed moves are ground truth (protocol-observed, no belief
        # set source required) and must always be encoded; possible_moves from the set source
        # augment them. Revealed take priority and are never evicted by the sort/truncate.
        # The final sorted bucket list is materialized here so the PP-ledger numeric columns can
        # align positionally with the belief-move categorical columns.
        bucket_moves = _compact_belief_values(
            _prioritized_belief_moves(revealed_moves, possible_moves, BELIEF_MOVE_BUCKET_COUNT),
            limit=BELIEF_MOVE_BUCKET_COUNT,
        )
        _encode_belief_fact_categories(categorical_ids[token_index], "possible_move", bucket_moves)
        _set_numeric(numeric_features[token_index], NUMERIC_HP_FRACTION, condition.hp_fraction or 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_ACTIVE, 1.0 if candidate.active else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_LEGAL, 0.0 if condition.fainted else 1.0)
        _set_numeric(numeric_features[token_index], NUMERIC_PRESENT, 1.0)
        _set_numeric(numeric_features[token_index], NUMERIC_REVEALED_MOVE_COUNT, float(len(revealed_moves)))
        _set_numeric(numeric_features[token_index], NUMERIC_CANDIDATE_SET_COUNT, float(candidate_set_count or 0))
        _set_numeric(numeric_features[token_index], NUMERIC_UNCERTAINTY, uncertainty)
        _set_numeric(numeric_features[token_index], NUMERIC_POSSIBLE_ABILITY_COUNT, float(len(ability_feature_values)))
        _set_numeric(numeric_features[token_index], NUMERIC_POSSIBLE_ITEM_COUNT, float(len(item_feature_values)))
        _set_numeric(numeric_features[token_index], NUMERIC_POSSIBLE_MOVE_COUNT, float(len(possible_moves)))
        _set_numeric(numeric_features[token_index], NUMERIC_REVEALED_ABILITY, 1.0 if revealed_ability else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_REVEALED_ITEM, 1.0 if (revealed_item and not item_removed) else 0.0)
        # ---- spec v2 per-mon blocks. ----
        exact = _belief_for_species(exact_beliefs_by_species, candidate.species)
        if masks.exact_state:
            _encode_mon_exact_state(
                numeric_features[token_index],
                candidate,
                exact,
                role=role,
                status=status,
                fainted=condition.fainted,
            )
            if schema_v2_1 and candidate.active and _has_substitute(active_volatiles):
                _set_numeric(
                    numeric_features[token_index],
                    NUMERIC_SUB_HP_FRACTION,
                    _substitute_hp_fraction(candidate),
                )
            if role == "opponent":
                _encode_opponent_move_pp_fractions(
                    numeric_features[token_index],
                    exact,
                    bucket_moves,
                    dex=dex,
                    write_validity=schema_v2_1,
                )
                _encode_expected_stats(
                    numeric_features[token_index],
                    dex,
                    base_species=candidate.species,
                    battle_species=enc_species,
                    details=candidate.details,
                    belief=exact,
                    transformed=transformed,
                    transform_target=(
                        (transform_targets_by_species or {}).get(_normalize_identifier(enc_species))
                        if transformed
                        else None
                    ),
                    # v4 ONLY. The corrected values differ from the legacy approximations, and
                    # these are frozen legacy positions shared with v2.1/v2.2/v3 -- three lineages
                    # are training against those encodes right now. v4 is unlaunched, so it can
                    # start correct; the older schemas keep bug-compatible values rather than
                    # having their input distribution shifted mid-run.
                    exact_spreads=schema_v4,
                )
        if masks.opponent_tendency_stats_block and role == "opponent" and tendency_by_species:
            tendency = tendency_by_species.get(_normalize_identifier(candidate.species))
            if tendency is not None:
                _encode_mon_tendency(numeric_features[token_index], tendency)
        # v4: the matchup-conditional twin of the triple above, on the same token and under
        # the SAME tendency mask (it is the same channel, conditioned). Absent cells stay
        # (0, 0) — no history in this matchup — and the marginal triple beside it carries on.
        if (
            schema_v4
            and masks.opponent_tendency_stats_block
            and role == "opponent"
            and matchup_switch_evidence
        ):
            cell = matchup_switch_evidence.get(_normalize_identifier(candidate.species))
            if cell is not None:
                switched, stayed = cell
                if switched:
                    _set_numeric(
                        numeric_features[token_index],
                        NUMERIC_MON_SWITCHED_VS_ACTIVE,
                        min(1.0, switched / _MATCHUP_COUNT_DIVISOR),
                    )
                if stayed:
                    _set_numeric(
                        numeric_features[token_index],
                        NUMERIC_MON_STAYED_VS_ACTIVE,
                        min(1.0, stayed / _MATCHUP_COUNT_DIVISOR),
                    )
        # v2.1 pinned Tier-2 conclusions (current-state surface; the tt cb_bit and
        # tt investment code stay the as-of-strike history records). Gated upstream:
        # the CB set is empty unless the spec is v2.1, masks.tier2_residuals is on,
        # AND the tokens were tier2-annotated (the belief-source double-gate); the
        # investment map additionally requires masks.tier2_investment (its separate
        # provenance switch). Keyed on BASE species (Transform identity rule) and
        # persistent across switches — per-mon facts, not per-strike ones.
        if (
            role == "opponent"
            and tier2_cb_pinned_species
            and _normalize_identifier(candidate.species) in tier2_cb_pinned_species
        ):
            _set_numeric(numeric_features[token_index], NUMERIC_TIER2_CB_PINNED, 1.0)
        if role == "opponent" and tier2_investment_pinned:
            investment_code = tier2_investment_pinned.get(_normalize_identifier(candidate.species))
            if investment_code:
                _set_numeric(
                    numeric_features[token_index], NUMERIC_TIER2_INVESTMENT_PINNED, investment_code
                )


def _encode_mon_exact_state(
    num_row: list[float],
    candidate: ShowdownPokemon,
    exact: RevealedPokemonBelief | None,
    *,
    role: str,
    status: str,
    fainted: bool,
) -> None:
    """Per-mon exact-state features from the belief engine's ledgers (both sides).

    Sleep fields populate only while asleep. ``wake-known`` semantics (corrections item 8):
    for our own mons the wake turn is always known (our ability is known); for opponent mons a
    Rest wake is known-2 iff Early Bird is absent from the live candidate abilities (ambiguous
    {1, 2} otherwise; a revealed ability restores determinism either way). Natural sleep is a
    hazard rate — never wake-known.
    """
    if exact is not None:
        if status == "slp":
            _set_numeric(num_row, NUMERIC_SLEEP_TURNS, min(1.0, exact.sleep_turns / 5.0))
            if exact.rest_sleep:
                _set_numeric(num_row, NUMERIC_REST_SLEEP, 1.0)
                if role == "self" or _opponent_rest_wake_known(exact):
                    _set_numeric(num_row, NUMERIC_WAKE_KNOWN, 1.0)
        if candidate.active and exact.turns_active:
            _set_numeric(num_row, NUMERIC_TURNS_ACTIVE, min(1.0, exact.turns_active / _STAT_COUNT_DIVISOR))
    ability = (
        candidate.ability
        if role == "self"
        else (_certain_opponent_ability(exact) if exact is not None else None)
    )
    if (
        ability
        and _normalize_identifier(ability) in _TRAP_ABILITIES
        and not fainted
        and not candidate.active
    ):
        _set_numeric(num_row, NUMERIC_TRAPPER_ALIVE, 1.0)


def _certain_opponent_ability(exact: RevealedPokemonBelief) -> str | None:
    """The opponent mon's ability when CERTAIN: protocol-revealed, or a singleton live
    candidate set (possible minus ruled-out) — the same known-or-singleton standard the
    belief categoricals expose. Gen 3 trap abilities are never protocol-revealed, but all
    four pool trappers (Wobbuffet/Dugtrio/Magneton/Nosepass) are single-ability species, so under
    belief-on this is exact knowledge the encoder must not ignore (audit bug C1)."""
    if exact.revealed_ability:
        return exact.revealed_ability
    ruled_out = {_normalize_identifier(ability) for ability in exact.ruled_out_abilities}
    live = [
        ability
        for ability in exact.possible_abilities
        if _normalize_identifier(ability) not in ruled_out
    ]
    if len(live) == 1:
        return live[0]
    return None


def _opponent_rest_wake_known(exact: RevealedPokemonBelief) -> bool:
    """Whether an opponent Rest sleeper's wake turn is deterministic to us (Early Bird resolved)."""
    if exact.revealed_ability:
        return True
    candidates = {
        _normalize_identifier(ability) for ability in exact.possible_abilities
    } - {_normalize_identifier(ability) for ability in exact.ruled_out_abilities}
    if not candidates:
        # No candidate information (set source off, nothing revealed): cannot assert Early Bird
        # absent, so the wake stays ambiguous.
        return False
    return "earlybird" not in candidates


def _encode_opponent_move_pp_fractions(
    num_row: list[float],
    exact: RevealedPokemonBelief | None,
    bucket_moves: Sequence[str],
    *,
    dex: "ShowdownDex | None",
    write_validity: bool = False,
) -> None:
    """Remaining-PP fraction per REVEALED opponent move, aligned with the belief-move buckets.

    Max PP is the randbat catalog rule (3 PP Ups) from the dex; ``move_uses`` already carries the
    engine-side charging rules (Pressure x2, Sleep-Talk-charges-caller, Transform scoping).
    Unrevealed bucket columns stay 0.0 — no PP knowledge is claimed for merely-possible moves.

    The v2 revealed-at-0-PP collision (a REVEALED move ledgered to exactly 0 PP encoded 0.0,
    indistinguishable in this channel from an unrevealed bucket — "confirmed empty" vs "no
    knowledge", which matters in pp-stall endgames) is CLOSED under spec v2.1: with
    ``write_validity`` (v2.1 specs only) the bucket-aligned NUMERIC_OPP_MOVE_PP_VALID_OFFSET
    column carries 1.0 for every protocol-revealed bucket move, regardless of remaining PP —
    the explicit confirmed-move flag per bucket. Under a v2 spec the collision stands exactly
    as before (byte-identical v2 encodes; no epsilon floor).
    """
    if exact is None or dex is None:
        return
    revealed_keys = {
        _normalize_identifier(move) for move in exact.revealed_moves if _normalize_identifier(move)
    }
    if not revealed_keys:
        return
    uses_by_move = {key: uses for key, uses in exact.move_uses}
    for index, move in enumerate(bucket_moves[:BELIEF_MOVE_BUCKET_COUNT]):
        key = _normalize_identifier(move)
        if key not in revealed_keys:
            continue
        # Revealed is protocol ground truth: the validity bit does not depend on the dex
        # carrying a max PP for the move (the PP fraction below still does).
        if write_validity:
            _set_numeric(num_row, NUMERIC_OPP_MOVE_PP_VALID_OFFSET + index, 1.0)
        info = dex.move_info(key)
        max_pp = info.max_pp if info is not None else 0
        if max_pp <= 0:
            continue
        remaining = max(0, max_pp - int(uses_by_move.get(key, 0)))
        _set_numeric(num_row, NUMERIC_OPP_MOVE_PP_OFFSET + index, remaining / float(max_pp))


def _has_substitute(active_volatiles: Sequence[str]) -> bool:
    """Whether the active mon's tracked volatiles include a live Substitute."""
    return any(_normalize_identifier(name) == "substitute" for name in active_volatiles)


def _substitute_hp_fraction(candidate: ShowdownPokemon) -> float:
    """The KNOWN INITIAL substitute HP fraction for a mon with a sub up (v2.1 column).

    Gen 3 sub HP = floor(maxhp/4) (engine-verified; see NUMERIC_SUB_HP_FRACTION). Exact for
    the self side, whose max HP comes from the request; the 0.25 baseline for the opponent
    (hidden max HP; floor error < 1%). Chip against the sub is not protocol-derivable, so
    the value is presence + initial size, not a running ledger.
    """
    max_hp = candidate.stats.get("hp") if candidate.stats else None
    if isinstance(max_hp, int) and max_hp > 0:
        return (max_hp // 4) / float(max_hp)
    return 0.25


def _gen3_stat(base: int, level: int, *, ev: int, iv: int, hp: bool) -> int:
    """Gen 3 stat formula at a neutral nature (the randbats generator's spread family)."""
    core = ((2 * base + iv + ev // 4) * level) // 100
    return core + level + 10 if hp else core + 5



# Legal EV/IV values the gen3 randbats generator can produce, measured across all 1682 pool
# variants. The whole degree of freedom is a handful of discrete values, which is what makes the
# check worth having: a wrong STAT is plausible and invisible, a wrong EV is provably illegal.
# The bug this replaces derived trimmed HP at ev=0 -- and 0 is NOT a legal HP EV, the generator
# can never strip the stat -- so this assertion would have caught it at write time.
_LEGAL_HP_EVS = frozenset({85, 81, 77, 73, 69})
_LEGAL_ATK_EVS = frozenset({85, 0})
_LEGAL_HP_IVS = frozenset({31, 30})
# Def/SpA/SpD/Spe IVs are 31, or 30 where the carried Hidden Power type's ``HPivs`` entry
# lowers them (``data/typechart.ts``). Measured over every pool set x both physical-attack
# arms x every listed item: exactly {30, 31}, no third value. Policed for the same reason as
# the HP/Atk axes -- these four columns are now derived from the spread core, so a generator
# drift that moved an IV would otherwise surface as a silently wrong stat.
_LEGAL_NON_HP_IVS = frozenset({31, 30})
_NON_HP_SPREAD_STATS = ("def", "spa", "spd", "spe")


def _variant_has_physical_attack(dex: "ShowdownDex", variant: "Mapping[str, Any]") -> bool:
    """The generator's ``counter.get('Physical')`` truthiness for one candidate variant."""
    return any(
        _is_physical_attack(dex, _normalize_identifier(str(move)))
        for move in _as_sequence(variant.get("moves"))
    )


def _variant_spread_stats(
    base_stats: "Mapping[str, int]", level: int, variant: "Mapping[str, Any]", has_physical: bool
) -> "dict[str, int] | None":
    """Generator-exact stats for one candidate variant, or None if not derivable.

    Degrades to None rather than raising: a malformed candidate must not take down an encode.
    An ILLEGAL spread is different and does raise -- that means the generator drifted or the
    core was mis-called, and silently emitting a plausible-but-wrong stat is the failure mode
    this whole change exists to remove.
    """

    from .gen3_damage import randbats_spread_details

    raw_moves = variant.get("moves")
    if not isinstance(raw_moves, (list, tuple)):
        return None
    try:
        spread = randbats_spread_details(
            base_stats,
            level=level,
            moves=tuple(str(move) for move in raw_moves),
            item=str(variant.get("item") or ""),
            has_physical_attack=has_physical,
        )
    except Exception:  # noqa: BLE001 - a bad candidate degrades, it does not break the encode
        return None
    hp_ev = int(spread.evs.get("hp", 85))
    atk_ev = int(spread.evs.get("atk", 85))
    hp_iv = int(spread.ivs.get("hp", 31))
    if hp_ev not in _LEGAL_HP_EVS or atk_ev not in _LEGAL_ATK_EVS or hp_iv not in _LEGAL_HP_IVS:
        raise ValueError(
            "randbats spread outside the generator's legal set "
            f"(hp_ev={hp_ev}, atk_ev={atk_ev}, hp_iv={hp_iv}); the generator has drifted or the "
            "spread core was mis-called -- refusing to emit a plausible-but-wrong stat"
        )
    for stat_key in _NON_HP_SPREAD_STATS:
        stat_iv = int(spread.ivs.get(stat_key, 31))
        stat_ev = int(spread.evs.get(stat_key, 85))
        if stat_iv not in _LEGAL_NON_HP_IVS or stat_ev != 85:
            raise ValueError(
                "randbats spread outside the generator's legal set "
                f"({stat_key}_iv={stat_iv}, {stat_key}_ev={stat_ev}); the generator has drifted "
                "or the spread core was mis-called -- refusing to emit a plausible-but-wrong stat"
            )
    stats = {"hp": int(spread.stats["hp"]), "atk": int(spread.stats["atk"])}
    stats.update({stat_key: int(spread.stats[stat_key]) for stat_key in _NON_HP_SPREAD_STATS})
    return stats


def _encode_expected_stats(
    num_row: list[float],
    dex: "ShowdownDex | None",
    *,
    base_species: str,
    battle_species: str,
    details: str | None,
    belief: RevealedPokemonBelief | None,
    transformed: bool = False,
    transform_target: ShowdownPokemon | None = None,
    exact_spreads: bool = False,
) -> None:
    """Deterministic opponent stat block from species + level + the generator's spread family.

    Every stat is variant-conditioned, not just HP and Atk. Def/SpA/SpD/Spe were long documented
    here as "exact (the generator never varies them)"; that was FALSE -- Hidden Power sets carry
    the HP type's ``HPivs`` override, which drops one or more of the four to IV 30 on 205 of 393
    pool sets (see the block below). Under ``exact_spreads`` all four now come from the generator
    core, maxed over the surviving candidates.

    HP and Atk are variant-conditioned (corrections item 1): baseline 85/31 plus a [low, high] bound pair over
    the candidate variants — Atk-zeroing (0 EV / 0 IV) on no-physical-attack variants, HP-EV trim
    (0 EV lower bound) on Sub+Flail/Reversal, Sub+pinch-berry, and Belly Drum variants. Without
    an attached set source the bounds collapse to the baseline.

    Transform rule (ENGINE-VERIFIED against the vendored pokemon-showdown checkout,
    ``sim/pokemon.ts`` ``transformInto``; no gen3 mod override): Transform copies the TARGET's
    stored stat VALUES for every non-HP stat (``this.storedStats[statName] =
    pokemon.storedStats[statName]``) — i.e. the target's own spread at the TARGET's level —
    and never copies HP. In singles the copy target is OUR active mon at transform time, whose
    actual stats are player-known from the request, so a transformed opponent's non-HP expected
    stats are the target's EXACT values (bounds collapse); HP stays the actor's own species at
    the actor's level. The actor's variant conditioning must NOT be applied to copied stats
    (a Transform-only Ditto has no physical attack, but the copied Atk is the target's real
    Atk). If the copy target cannot be identified, the whole block stays ZERO: per the
    asymmetry principle, an unknown hard-state feature beats a deterministically wrong one.
    """
    if dex is None:
        return
    if transformed:
        _encode_transformed_expected_stats(
            num_row,
            dex,
            base_species=base_species,
            details=details,
            transform_target=transform_target,
        )
        return
    # Belt-and-suspenders: a missing level means L100 (see _level_from_details). The root
    # fix already returns 100 for a token-less details string; treating a None level as 100
    # here also covers any other caller that passes details=None for a level-100 mon, rather
    # than silently zeroing this otherwise-deterministic block.
    level = _level_from_details(details)
    if level is None:
        level = 100
    battle_info = _species_info_base_fallback(dex, battle_species)
    hp_info = _species_info_base_fallback(dex, base_species)
    if battle_info is None or hp_info is None:
        return
    base = battle_info.base_stats
    hp_base = hp_info.base_stats.get("hp")
    variants = belief.candidate_variants if belief is not None else ()
    # Def/SpA/SpD/Spe under the generator core, when it is available (v4). These four are NOT
    # spread-invariant, contrary to what this function claimed for three schema generations:
    # the generator overwrites IVs from the carried Hidden Power type's ``HPivs`` entry
    # (``data/random-battles/gen3/teams.ts``: ``for (iv in HPivs) ivs[iv] = HPivs[iv]``), which
    # lowers one or more of these four to 30 on 205 of the 393 pool sets. Emitting the flat
    # iv=31 value was wrong on 25%/29%/24%/9% of sets (def/spa/spd/spe) by one point each --
    # the same fork-the-generator defect as C1, in the block C1's fix did not reach.
    #
    # MAX over the surviving candidates: exact whenever they agree on the stat (the common
    # case, and always once the set is pinned), and otherwise the iv=31 no-override value,
    # which is a true upper bound over the candidates rather than a fabricated midpoint. The
    # column stays single-valued -- adding a [low, high] band here would add v4 inputs, which
    # is an owner call, and the residual spread is one point.
    exact_variant_spreads: list[dict[str, int]] = []
    if exact_spreads and variants:
        for variant in variants:
            spread = _variant_spread_stats(
                base, level, variant, _variant_has_physical_attack(dex, variant)
            )
            if spread is None:
                # One unevaluable candidate makes the whole set unusable, for the same reason
                # the HP/Atk band abandons narrowing below: a max taken over a partial set
                # would report a bound derived from candidates that are not all of them.
                exact_variant_spreads = []
                break
            exact_variant_spreads.append(spread)
    for stat_key, slot in (
        ("def", NUMERIC_EXPECTED_DEF),
        ("spa", NUMERIC_EXPECTED_SPA),
        ("spd", NUMERIC_EXPECTED_SPD),
        ("spe", NUMERIC_EXPECTED_SPE),
    ):
        value = base.get(stat_key)
        if not value:
            continue
        if exact_variant_spreads:
            emitted = max(spread[stat_key] for spread in exact_variant_spreads)
        else:
            emitted = _gen3_stat(value, level, ev=85, iv=31, hp=False)
        _set_numeric(num_row, slot, min(1.0, emitted / _ACTUAL_STAT_DIVISOR))
    atk_base = base.get("atk")
    if not atk_base or not hp_base:
        return
    atk_baseline = _gen3_stat(atk_base, level, ev=85, iv=31, hp=False)
    hp_baseline = _gen3_stat(hp_base, level, ev=85, iv=31, hp=True)
    atk_low = atk_high = atk_baseline
    hp_low = hp_high = hp_baseline
    if variants:
        atk_values: list[int] = []
        hp_values: list[int] = []
        for index, variant in enumerate(variants):
            moves = {
                _normalize_identifier(str(move)) for move in _as_sequence(variant.get("moves"))
            }
            item = _normalize_identifier(str(variant.get("item") or ""))
            has_physical = any(_is_physical_attack(dex, move) for move in moves)
            if exact_spreads:
                # Ask the GENERATOR's own spread core rather than re-deriving its rules. The
                # approximations below are both wrong: the trimmed-HP bound jumps to ev=0 (a full
                # 85-EV strip) where the generator's `while evs.hp > 1` loop removes 4 at a time
                # and stops at the first value satisfying its modular condition -- measured wrong
                # on 100% of trim-eligible variants by +14..+17 HP; and the zeroed-Atk bound
                # hardcodes iv=0, missing `ivs.atk = hasHiddenPower ? (ivs.atk||31) - 28`, which
                # leaves IV 3 -- wrong on 43% of Atk-zeroed variants, every one a Hidden Power set.
                #
                # Worse than a loose band: because the band is min/max over survivors, a PERFECTLY
                # PINNED set still reported the wrong HP. randbats_spread_details is the same core
                # the investment inference uses and is cross-checked against server-computed stats
                # by its gate harness, so this stops the encoder being a fork of it.
                # Already computed above for the Def/SpA/SpD/Spe block, which needs the same
                # per-candidate spreads; an empty list there means some candidate was
                # unevaluable, which lands on the identical fallback either way.
                spread = (
                    exact_variant_spreads[index]
                    if index < len(exact_variant_spreads)
                    else None
                )
                if spread is None:
                    # An unevaluable candidate makes the whole BAND unsound, and substituting
                    # the baseline for it would be worse than emitting nothing: min/max would
                    # then report a bound partly derived from a value no real variant has, and
                    # the model would read it as confidently as a true one. A wrong belief costs
                    # more than an absent one, so abandon the narrowing entirely and fall back
                    # to the documented no-set-source state (low == high == baseline), which is
                    # an honest "unknown" rather than a fabricated range.
                    atk_values = []
                    hp_values = []
                    break
                atk_values.append(spread["atk"])
                hp_values.append(spread["hp"])
            else:
                atk_values.append(
                    atk_baseline if has_physical else _gen3_stat(atk_base, level, ev=0, iv=0, hp=False)
                )
                hp_trimmed = "bellydrum" in moves or (
                    "substitute" in moves and (bool(moves & {"flail", "reversal"}) or item in _PINCH_BERRIES)
                )
                hp_values.append(
                    _gen3_stat(hp_base, level, ev=0, iv=31, hp=True) if hp_trimmed else hp_baseline
                )
        if atk_values and hp_values:
            atk_low, atk_high = min(atk_values), max(atk_values)
            hp_low, hp_high = min(hp_values), max(hp_values)
    for slot, value in (
        (NUMERIC_EXPECTED_HP, hp_baseline),
        (NUMERIC_EXPECTED_HP_LOW, hp_low),
        (NUMERIC_EXPECTED_HP_HIGH, hp_high),
        (NUMERIC_EXPECTED_ATK, atk_baseline),
        (NUMERIC_EXPECTED_ATK_LOW, atk_low),
        (NUMERIC_EXPECTED_ATK_HIGH, atk_high),
    ):
        _set_numeric(num_row, slot, min(1.0, value / _ACTUAL_STAT_DIVISOR))


def _encode_transformed_expected_stats(
    num_row: list[float],
    dex: "ShowdownDex",
    *,
    base_species: str,
    details: str | None,
    transform_target: ShowdownPokemon | None,
) -> None:
    """Expected stats for a transformed opponent: copied non-HP values are the target's actual
    stats (exact, player-known); HP is the actor's own baseline. Unidentifiable target => the
    block stays zero (see the Transform rule in ``_encode_expected_stats``)."""
    target_stats = transform_target.stats if transform_target is not None else None
    if not target_stats:
        return
    if any(key not in target_stats for key in ("atk", "def", "spa", "spd", "spe")):
        return
    for stat_key, slot in (
        ("def", NUMERIC_EXPECTED_DEF),
        ("spa", NUMERIC_EXPECTED_SPA),
        ("spd", NUMERIC_EXPECTED_SPD),
        ("spe", NUMERIC_EXPECTED_SPE),
    ):
        _set_numeric(num_row, slot, min(1.0, float(target_stats[stat_key]) / _ACTUAL_STAT_DIVISOR))
    atk_value = min(1.0, float(target_stats["atk"]) / _ACTUAL_STAT_DIVISOR)
    for slot in (NUMERIC_EXPECTED_ATK, NUMERIC_EXPECTED_ATK_LOW, NUMERIC_EXPECTED_ATK_HIGH):
        _set_numeric(num_row, slot, atk_value)
    # HP is never copied: the actor's own species at the actor's own level. Transform carriers
    # (Ditto, Mew) have no HP-trim variants, so the baseline with collapsed bounds is exact
    # to within the HP-IV point.
    level = _level_from_details(details)
    hp_info = dex.species_info(base_species)
    hp_base = hp_info.base_stats.get("hp") if hp_info is not None else None
    if level is None or not hp_base:
        return
    hp_value = min(1.0, _gen3_stat(hp_base, level, ev=85, iv=31, hp=True) / _ACTUAL_STAT_DIVISOR)
    for slot in (NUMERIC_EXPECTED_HP, NUMERIC_EXPECTED_HP_LOW, NUMERIC_EXPECTED_HP_HIGH):
        _set_numeric(num_row, slot, hp_value)


def _is_physical_attack(dex: "ShowdownDex", move_id: str) -> bool:
    info = dex.move_info(move_id)
    return info is not None and info.gen3_category == "Physical" and info.base_power > 0


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return ()


# Evidence-mass divisor for the matchup-conditional pair. Deliberately NOT the /64 the
# whole-game tendency counts use: a single (their mon x our mon) cell is visited a handful of
# times per game, so /64 would pin the pair into the bottom few percent of its column for its
# entire realistic range. Same principle as /64, matched to this quantity's actual scale.
_MATCHUP_COUNT_DIVISOR = 8.0


def _encode_mon_tendency(num_row: list[float], tendency: "OpponentMonTendency") -> None:
    """Per-opponent-mon tendency triple (counts /64 — evidence mass, never rates)."""
    if tendency.switched_out_before_attacking:
        _set_numeric(
            num_row,
            NUMERIC_MON_SWITCHED_BEFORE_ATTACK,
            min(1.0, tendency.switched_out_before_attacking / _STAT_COUNT_DIVISOR),
        )
    if tendency.stayed_and_attacked:
        _set_numeric(
            num_row,
            NUMERIC_MON_STAYED_AND_ATTACKED,
            min(1.0, tendency.stayed_and_attacked / _STAT_COUNT_DIVISOR),
        )
    if tendency.turns_active:
        _set_numeric(
            num_row,
            NUMERIC_MON_TURNS_ACTIVE_TOTAL,
            min(1.0, tendency.turns_active / _STAT_COUNT_DIVISOR),
        )


def _encode_stats_token(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    state: PlayerRelativeBattleState,
    *,
    masks: ObservationFeatureMasks = DEFAULT_OBSERVATION_FEATURE_MASKS,
) -> None:
    """The opponent-tendency-stats token: (count, opportunity) pairs + opponent weather reveals."""
    stats = state.tendency_stats
    if stats is None or not masks.opponent_tendency_stats_block:
        return
    cat_row = categorical_ids[OPPONENT_TENDENCY_STATS_TOKEN_OFFSET]
    num_row = numeric_features[OPPONENT_TENDENCY_STATS_TOKEN_OFFSET]
    _set_category(cat_row, CATEGORY_ROLE, "stats")
    _set_numeric(num_row, NUMERIC_PRESENT, 1.0)
    for slot, count in (
        (NUMERIC_STAT_OPP_SWITCH_COUNT, stats.opponent_switch_count),
        (NUMERIC_STAT_OPP_DECISION_OPPORTUNITIES, stats.opponent_decision_opportunities),
        (NUMERIC_STAT_BLOCKED_ON_OUR_ATTACK, stats.blocked_on_our_attack_count),
        (NUMERIC_STAT_PURSUIT_INTERCEPT_PREDICT, stats.pursuit_intercept_predict_count),
        (NUMERIC_STAT_MY_SWITCH_TURNS, stats.my_switch_turn_count),
    ):
        if count:
            _set_numeric(num_row, slot, min(1.0, count / _STAT_COUNT_DIVISOR))
    reveals_by_weather = {reveal.weather: reveal for reveal in stats.opponent_weather_reveals}
    for index, weather in enumerate(_WEATHER_REVEAL_ORDER):
        reveal = reveals_by_weather.get(weather)
        if reveal is None:
            continue
        _set_numeric(num_row, NUMERIC_STAT_WEATHER_REVEAL_OFFSET + (2 * index), 1.0)
        if reveal.from_ability:
            _set_numeric(num_row, NUMERIC_STAT_WEATHER_REVEAL_OFFSET + (2 * index) + 1, 1.0)


def _encode_transition_tokens(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    state: PlayerRelativeBattleState,
    spec: ObservationSpec,
    *,
    masks: ObservationFeatureMasks = DEFAULT_OBSERVATION_FEATURE_MASKS,
    schema_v2_1: bool = False,
) -> None:
    """Encode the ordered transition-token block (corrections item 9 schema).

    Slots fill chronologically (oldest first) with the most recent ``budget`` tokens —
    oldest-first truncation, since the truncated prefix is exactly what the unbounded aggregates
    have absorbed. Unfilled slots stay zeroed and attention-masked. Categorical fields ride the
    shared fixed columns with transition-specific vocab families; the action column branches on
    ``kind`` (move id / incoming species / cant reason — deliberately unmerged vocabularies).
    ``NUMERIC_TT_RESIDUAL``/``NUMERIC_TT_RESIDUAL_VALID`` fill only from tokens whose Tier-2
    fields were populated (``pokezero.tier2``), gated by ``masks.tier2_residuals``; they stay
    0.0 for the plain extraction path.

    v2.1 defender identity: move tokens carry the defender's base species in the
    CATEGORY_MOVE_PRIORITY column — unused on transition tokens under v2 (the priority
    bracket is an action-candidate-token fact; transition rows never set it, verified by the
    v2 byte-identity gate), so reusing it costs no new column. The defender shares the
    ``species:`` vocabulary family. Rationale on record: the defender is inferable from the
    interleaved switch tokens EXCEPT when K-truncation drops the anchoring switch, and
    ``damage_fraction`` is defender-relative — the anchor must survive truncation.
    """
    budget = min(masks.transition_token_budget, spec.transition_token_count)
    tokens = state.transition_tokens[-budget:] if budget else ()
    self_slot = state.perspective.showdown_slot
    for index, token in enumerate(tokens):
        cat_row = categorical_ids[TRANSITION_TOKEN_OFFSET + index]
        num_row = numeric_features[TRANSITION_TOKEN_OFFSET + index]
        actor_role = "self" if token.actor_slot == self_slot else "opponent"
        _set_category(cat_row, CATEGORY_PRIMARY, f"species:{token.actor_species}")
        if token.kind == _TT_KIND_MOVE:
            action_label = f"move:{token.action}"
        elif token.kind == _TT_KIND_SWITCH:
            action_label = f"species:{token.action}"
        else:
            action_label = f"cant:{token.action}"
        _set_category(cat_row, CATEGORY_SECONDARY, action_label)
        _set_category(cat_row, CATEGORY_ROLE, f"transition:{actor_role}")
        _set_category(cat_row, CATEGORY_SLOT, f"tt_kind:{token.kind}")
        if token.kind == _TT_KIND_MOVE:
            _set_category(cat_row, CATEGORY_TYPE_1, f"tt_outcome:{token.damage_outcome}")
            _set_category(cat_row, CATEGORY_TYPE_2, f"tt_effectiveness:{token.effectiveness}")
            _set_category(cat_row, CATEGORY_MOVE_CATEGORY, f"tt_side_effect:{token.side_effect}")
            if schema_v2_1 and token.defender_species:
                _set_category(
                    cat_row, CATEGORY_MOVE_PRIORITY, f"species:{token.defender_species}"
                )
        if token.weather:
            _set_category(cat_row, CATEGORY_MOVE_EFFECT, f"weather:{token.weather}")
        _set_numeric(num_row, NUMERIC_PRESENT, 1.0)
        # Internal transition semantics exclude confusion self-damage. This
        # writer serves only frozen V2/V2.1, so reconstruct their historical
        # aggregate at the schema boundary.
        legacy_damage = token.damage_fraction
        if token.confusion_selfhit:
            legacy_damage += token.confusion_selfhit_fraction
        if legacy_damage:
            _set_numeric(num_row, NUMERIC_TT_DAMAGE_FRACTION, min(1.0, legacy_damage))
        if token.kind == _TT_KIND_MOVE:
            # n_hits is a move-token field; switch/cant rows keep 0.0 (not a constant 1/5).
            _set_numeric(num_row, NUMERIC_TT_N_HITS, min(1.0, token.n_hits / 5.0))
        for slot, flag in (
            (NUMERIC_TT_CALLED, token.called),
            (NUMERIC_TT_TRANSFORMED, token.transformed),
            (NUMERIC_TT_CRIT, token.crit),
            (NUMERIC_TT_MISS, token.miss),
            (NUMERIC_TT_KO, token.ko),
            (NUMERIC_TT_PURSUIT_INTERCEPT, token.pursuit_intercept),
        ):
            if flag:
                _set_numeric(num_row, slot, 1.0)
        if token.own_spikes_layers:
            _set_numeric(num_row, NUMERIC_TT_OWN_SPIKES, min(1.0, token.own_spikes_layers / 3.0))
        if token.opp_spikes_layers:
            _set_numeric(num_row, NUMERIC_TT_OPP_SPIKES, min(1.0, token.opp_spikes_layers / 3.0))
        _set_numeric(num_row, NUMERIC_TT_ABS_TURN, min(1.0, token.turn / 1000.0))
        turns_ago = max(0, state.turn_number - token.turn)
        _set_numeric(num_row, NUMERIC_TT_TURNS_AGO, min(1.0, turns_ago / _STAT_COUNT_DIVISOR))
        if masks.tier2_residuals and token.residual_valid and token.residual is not None:
            _set_numeric(num_row, NUMERIC_TT_RESIDUAL, max(-1.0, min(1.0, token.residual)))
            _set_numeric(num_row, NUMERIC_TT_RESIDUAL_VALID, 1.0)
        if masks.tier2_residuals and token.cb_bit:
            _set_numeric(num_row, NUMERIC_TT_CB_BIT, 1.0)
        # Investment column: double-masked (the tier2 channel gate AND its own
        # provenance switch — see NUMERIC_TT_INVESTMENT_BIT's comment) AND schema-gated:
        # the column physically sits below the v2 census end, but no v2 checkpoint was
        # ever trained on a populated 120, so the LEGACY encode path never writes it —
        # a (hand-crafted) v2-schema config carrying tier2_investment=True is a no-op
        # here (review MED-2a), keeping v2-mode encodes byte-identical to the
        # pre-investment encoder unconditionally. Tokens from the plain extraction
        # path carry 0.0, so pre-investment pipelines are byte-identical regardless.
        if schema_v2_1 and masks.tier2_residuals and masks.tier2_investment and token.investment:
            _set_numeric(num_row, NUMERIC_TT_INVESTMENT_BIT, max(-1.0, min(1.0, token.investment)))


def _encode_turn_merged_transition_tokens(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    state: PlayerRelativeBattleState,
    spec: ObservationSpec,
    *,
    masks: ObservationFeatureMasks = DEFAULT_OBSERVATION_FEATURE_MASKS,
    schema_v3: bool = False,
) -> None:
    """Encode the TURN-MERGED transition block (spec v2.2).

    One row per turn/lead/replacement phase from ``state.turn_merged_tokens``: the first
    sub-block on the per-action columns (SLOT re-purposed to tt_phase:<phase>, tt_kind
    moved to CATEGORY_TM_FIRST_KIND, defender identity in CATEGORY_MOVE_PRIORITY exactly
    as the v2.1 per-action rows carry it), collapse fields + the whole second sub-block
    on the appended TURN_MERGED_* columns (tt2_ vocab families for bag binding). Fill and
    truncation semantics match the per-action encoder: most recent ``budget`` rows,
    oldest-first, rest zeroed + attention-masked.

    K BUDGET UNIT CHANGE (loud): ``masks.transition_token_budget`` counts THESE rows — a
    whole turn each. The v2/v2.1 K=64 horizon (~32 turns) is budget=32 here; an unchanged
    K=64 config roughly doubles its temporal horizon. The per-mon pinned Tier-2 bits are
    derived from the FULL per-action stream and survive this truncation regardless.
    """
    budget = min(masks.transition_token_budget, spec.transition_token_count)
    tokens = state.turn_merged_tokens[-budget:] if budget else ()
    self_slot = state.perspective.showdown_slot
    for index, token in enumerate(tokens):
        cat_row = categorical_ids[TRANSITION_TOKEN_OFFSET + index]
        num_row = numeric_features[TRANSITION_TOKEN_OFFSET + index]
        first = token.first
        actor_role = "self" if first.actor_slot == self_slot else "opponent"
        _set_category(cat_row, CATEGORY_PRIMARY, f"species:{first.actor_species}")
        _set_category(cat_row, CATEGORY_SECONDARY, _tm_first_action_label(first.kind, first.action))
        _set_category(cat_row, CATEGORY_ROLE, f"transition:{actor_role}")
        _set_category(cat_row, CATEGORY_SLOT, f"tt_phase:{token.phase}")
        _set_category(cat_row, CATEGORY_TM_FIRST_KIND, f"tt_kind:{first.kind}")
        if first.kind == _TT_KIND_MOVE:
            _set_category(cat_row, CATEGORY_TYPE_1, f"tt_outcome:{first.damage_outcome}")
            _set_category(cat_row, CATEGORY_TYPE_2, f"tt_effectiveness:{first.effectiveness}")
            _set_category(cat_row, CATEGORY_MOVE_CATEGORY, f"tt_side_effect:{first.side_effect}")
            if first.defender_species:
                _set_category(cat_row, CATEGORY_MOVE_PRIORITY, f"species:{first.defender_species}")
        if token.weather:
            _set_category(cat_row, CATEGORY_MOVE_EFFECT, f"weather:{token.weather}")
        if first.cant_reason:
            _set_category(cat_row, CATEGORY_TM_FIRST_CANT, f"cant:{first.cant_reason}")
        if first.baton_pass_species:
            _set_category(cat_row, CATEGORY_TM_FIRST_BP, f"species:{first.baton_pass_species}")
        _set_numeric(num_row, NUMERIC_PRESENT, 1.0)
        # Internal transition semantics are corrected. Reconstruct the frozen
        # V2.2 aggregate only when dispatching the legacy merged layout.
        first_damage = first.damage_fraction
        if not schema_v3 and first.confusion_selfhit:
            first_damage += first.confusion_selfhit_fraction
        if first_damage:
            _set_numeric(num_row, NUMERIC_TT_DAMAGE_FRACTION, min(1.0, first_damage))
        if first.kind == _TT_KIND_MOVE:
            _set_numeric(num_row, NUMERIC_TT_N_HITS, min(1.0, first.n_hits / 5.0))
        for slot, flag in (
            (NUMERIC_TT_CALLED, first.called),
            (NUMERIC_TT_TRANSFORMED, first.transformed),
            (NUMERIC_TT_CRIT, first.crit),
            (NUMERIC_TT_MISS, first.miss),
            (NUMERIC_TT_KO, first.ko),
            (NUMERIC_TT_PURSUIT_INTERCEPT, first.pursuit_intercept),
        ):
            if flag:
                _set_numeric(num_row, slot, 1.0)
        if token.own_spikes_layers:
            _set_numeric(num_row, NUMERIC_TT_OWN_SPIKES, min(1.0, token.own_spikes_layers / 3.0))
        if token.opp_spikes_layers:
            _set_numeric(num_row, NUMERIC_TT_OPP_SPIKES, min(1.0, token.opp_spikes_layers / 3.0))
        _set_numeric(num_row, NUMERIC_TT_ABS_TURN, min(1.0, token.turn / 1000.0))
        turns_ago = max(0, state.turn_number - token.turn)
        _set_numeric(num_row, NUMERIC_TT_TURNS_AGO, min(1.0, turns_ago / _STAT_COUNT_DIVISOR))
        if masks.tier2_residuals and first.residual_valid and first.residual is not None:
            _set_numeric(num_row, NUMERIC_TT_RESIDUAL, max(-1.0, min(1.0, first.residual)))
            _set_numeric(num_row, NUMERIC_TT_RESIDUAL_VALID, 1.0)
        if masks.tier2_residuals and first.cb_bit:
            _set_numeric(num_row, NUMERIC_TT_CB_BIT, 1.0)
        # Investment column (#513): double-masked like the per-action write; the v2.2
        # schema gate is satisfied by construction (this encoder only runs under v2.2,
        # a v2.1 superset).
        if masks.tier2_residuals and masks.tier2_investment and first.investment:
            _set_numeric(num_row, NUMERIC_TT_INVESTMENT_BIT, max(-1.0, min(1.0, first.investment)))
        if first.self_hp_cost:
            _set_numeric(num_row, NUMERIC_TT_SELF_HP_COST, min(1.0, first.self_hp_cost))
        # Spec v3 change 1: the fail bit mirrors the miss bit. Its legacy writer position is
        # projected into v3's grouped history region after encoding.
        if schema_v3 and first.fail:
            _set_numeric(num_row, NUMERIC_TT_FAIL, 1.0)
        # Spec v3 change 10: the confusion self-hit flag on the preceding
        # opponent move sub-block.
        if schema_v3 and first.confusion_selfhit:
            _set_numeric(num_row, NUMERIC_TT_CONFUSION_SELFHIT, 1.0)

        second = token.second
        if second.status != _TM_SUB_BLOCK_ACTION:
            # NEGATED (declared, consumed with no protocol trace — the hazard-sack free
            # pivot) vs ABSENT (no declaration expected): categorical status, plus the
            # consumed mon's identity when the fold knows it. All TM2 numerics stay 0.0.
            _set_category(cat_row, CATEGORY_TM_SECOND_KIND, f"tt2_status:{second.status}")
            if second.actor_species:
                _set_category(cat_row, CATEGORY_TM_SECOND_SPECIES, f"tt2_species:{second.actor_species}")
            continue
        _set_category(cat_row, CATEGORY_TM_SECOND_KIND, f"tt2_kind:{second.kind}")
        _set_category(cat_row, CATEGORY_TM_SECOND_SPECIES, f"tt2_species:{second.actor_species}")
        _set_category(cat_row, CATEGORY_TM_SECOND_ACTION, _tm_second_action_label(second.kind, second.action))
        if second.kind == _TT_KIND_MOVE:
            _set_category(cat_row, CATEGORY_TM_SECOND_OUTCOME, f"tt2_outcome:{second.damage_outcome}")
            _set_category(
                cat_row, CATEGORY_TM_SECOND_EFFECTIVENESS, f"tt2_effectiveness:{second.effectiveness}"
            )
            _set_category(cat_row, CATEGORY_TM_SECOND_SIDE_EFFECT, f"tt2_side_effect:{second.side_effect}")
            if second.defender_species:
                _set_category(cat_row, CATEGORY_TM_SECOND_DEFENDER, f"tt2_species:{second.defender_species}")
        if second.cant_reason:
            _set_category(cat_row, CATEGORY_TM_SECOND_CANT, f"tt2_cant:{second.cant_reason}")
        if second.baton_pass_species:
            _set_category(cat_row, CATEGORY_TM_SECOND_BP, f"tt2_species:{second.baton_pass_species}")
        _set_numeric(num_row, NUMERIC_TM2_PRESENT, 1.0)
        # Symmetric legacy reconstruction for the second action sub-block.
        second_damage = second.damage_fraction
        if not schema_v3 and second.confusion_selfhit:
            second_damage += second.confusion_selfhit_fraction
        if second_damage:
            _set_numeric(num_row, NUMERIC_TM2_DAMAGE_FRACTION, min(1.0, second_damage))
        if second.kind == _TT_KIND_MOVE:
            _set_numeric(num_row, NUMERIC_TM2_N_HITS, min(1.0, second.n_hits / 5.0))
        for slot, flag in (
            (NUMERIC_TM2_CALLED, second.called),
            (NUMERIC_TM2_TRANSFORMED, second.transformed),
            (NUMERIC_TM2_CRIT, second.crit),
            (NUMERIC_TM2_MISS, second.miss),
            (NUMERIC_TM2_KO, second.ko),
            (NUMERIC_TM2_PURSUIT_INTERCEPT, second.pursuit_intercept),
        ):
            if flag:
                _set_numeric(num_row, slot, 1.0)
        if masks.tier2_residuals and second.residual_valid and second.residual is not None:
            _set_numeric(num_row, NUMERIC_TM2_RESIDUAL, max(-1.0, min(1.0, second.residual)))
            _set_numeric(num_row, NUMERIC_TM2_RESIDUAL_VALID, 1.0)
        if masks.tier2_residuals and second.cb_bit:
            _set_numeric(num_row, NUMERIC_TM2_CB_BIT, 1.0)
        if masks.tier2_residuals and masks.tier2_investment and second.investment:
            _set_numeric(num_row, NUMERIC_TM2_INVESTMENT, max(-1.0, min(1.0, second.investment)))
        if second.self_hp_cost:
            _set_numeric(num_row, NUMERIC_TM2_SELF_HP_COST, min(1.0, second.self_hp_cost))
        # Spec v3 change 1: the second-mover fail twin (mirrors NUMERIC_TM2_MISS's write).
        if schema_v3 and second.fail:
            _set_numeric(num_row, NUMERIC_TM2_FAIL, 1.0)
        # Spec v3 change 10: the confusion self-hit flag rides the same single column as the
        # first sub-block (one per-turn bit).
        if schema_v3 and second.confusion_selfhit:
            _set_numeric(num_row, NUMERIC_TT_CONFUSION_SELFHIT, 1.0)


def _tm_first_action_label(kind: str, action: str) -> str:
    if kind == _TT_KIND_MOVE:
        return f"move:{action}"
    if kind == _TT_KIND_SWITCH:
        return f"species:{action}"
    return f"cant:{action}"


def _tm_second_action_label(kind: str, action: str) -> str:
    if kind == _TT_KIND_MOVE:
        return f"tt2_move:{action}"
    if kind == _TT_KIND_SWITCH:
        return f"tt2_species:{action}"
    return f"tt2_cant:{action}"


def _self_active_types(state: PlayerRelativeBattleState, dex: "ShowdownDex | None") -> tuple[str, ...]:
    """Types of the acting (self active) mon, for resolving type-dependent move effects."""
    if dex is None or state.self_active is None:
        return ()
    info = dex.species_info(state.self_active.species)
    return tuple(info.types) if info is not None else ()


def _self_active_hp_fraction(state: PlayerRelativeBattleState) -> float | None:
    """Current HP fraction of the acting mon, for resolving HP-variable base power."""
    if state.self_active is None:
        return None
    return _condition_features(state.self_active.condition).hp_fraction


def _encode_action_tokens(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    state: PlayerRelativeBattleState,
    *,
    dex: "ShowdownDex | None" = None,
) -> None:
    active_request = _active_request(state.request)
    moves = active_request.get("moves") if isinstance(active_request, Mapping) else None
    # The acting mon's types + HP fraction, to resolve type-dependent effects (Curse) and
    # HP-variable base power (Reversal / Flail / Eruption / Water Spout) on its moves.
    user_types = _self_active_types(state, dex)
    user_hp_fraction = _self_active_hp_fraction(state)
    # The acting mon's own typed move ids ("hiddenpowerfighting", ...) — the request-side fallback
    # for resolving generic Hidden Power's real type/base power (see _self_move_mechanics_id).
    own_move_ids = state.self_active.moves if state.self_active is not None else ()
    for move_index in range(MOVE_ACTION_COUNT):
        token_index = ACTION_CANDIDATE_TOKEN_OFFSET + move_index
        move = moves[move_index] if isinstance(moves, list) and move_index < len(moves) else None
        move_name = _request_move_name(move) if isinstance(move, Mapping) else f"slot:{move_index + 1}"
        disabled = bool(move.get("disabled")) if isinstance(move, Mapping) else True
        # The token's move IDENTITY stays the request-keyed name (generic "hiddenpower" for HP:
        # checkpoint-stable). Only the MECHANICS lookup resolves HP's typed variant so its true
        # type / base power / damage class reach the acting mon's decision surface.
        _set_category(categorical_ids[token_index], CATEGORY_PRIMARY, f"move:{move_name}")
        _set_category(categorical_ids[token_index], CATEGORY_SECONDARY, "action:move")
        _set_category(categorical_ids[token_index], CATEGORY_ROLE, "action")
        _set_category(categorical_ids[token_index], CATEGORY_SLOT, f"move_slot:{move_index + 1}")
        if isinstance(move, Mapping):
            mechanics_name = _self_move_mechanics_id(move, move_name, own_move_ids)
            _encode_move_mechanics(
                categorical_ids[token_index], numeric_features[token_index], dex, mechanics_name,
                user_types, user_hp_fraction,
            )
            _set_numeric(numeric_features[token_index], NUMERIC_MOVE_PP_FRACTION, _move_pp_fraction(move))
        _set_numeric(numeric_features[token_index], NUMERIC_LEGAL, 1.0 if state.legal_action_mask[move_index] else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_PRESENT, 1.0 if isinstance(move, Mapping) else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_ACTIVE, 0.0 if disabled else 1.0)

    active_team_index = _active_team_index(state.self_team)
    switch_targets = (
        canonical_switch_action_map(active_team_index, team_size=len(state.self_team))
        if active_team_index is not None and len(state.self_team) >= 2
        else ()
    )
    for switch_slot in range(ACTION_CANDIDATE_TOKEN_COUNT - MOVE_ACTION_COUNT):
        action_index = MOVE_ACTION_COUNT + switch_slot
        token_index = ACTION_CANDIDATE_TOKEN_OFFSET + action_index
        team_index = switch_targets[switch_slot] if switch_slot < len(switch_targets) else None
        pokemon = state.self_team[team_index] if team_index is not None and team_index < len(state.self_team) else None
        condition = _condition_features(pokemon.condition if pokemon is not None else None)
        species = pokemon.species if pokemon is not None else f"slot:{switch_slot + 1}"
        _set_category(categorical_ids[token_index], CATEGORY_PRIMARY, f"species:{species}")
        if pokemon is not None:
            _encode_species_type_categories(categorical_ids[token_index], dex, pokemon.species)
            _encode_pokemon_stats(numeric_features[token_index], dex, pokemon.species, pokemon.details)
            _encode_actual_stats(numeric_features[token_index], pokemon.stats)
        _set_category(categorical_ids[token_index], CATEGORY_SECONDARY, "action:switch")
        _set_category(categorical_ids[token_index], CATEGORY_ROLE, "action")
        _set_category(categorical_ids[token_index], CATEGORY_SLOT, f"switch_slot:{switch_slot + 1}")
        _set_numeric(numeric_features[token_index], NUMERIC_HP_FRACTION, condition.hp_fraction or 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_ACTIVE, 1.0 if pokemon is not None and pokemon.active else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_LEGAL, 1.0 if state.legal_action_mask[action_index] else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_PRESENT, 1.0 if pokemon is not None else 0.0)


def _observation_metadata(
    state: PlayerRelativeBattleState,
    *,
    dex: "ShowdownDex | None" = None,
    # No default: this argument decides whether the v4 feature pack is disclosed, and a gate
    # that defaults to the disclosing side fails open. Callers name the schema explicitly.
    schema_version: str,
) -> dict[str, Any]:
    # The v4 pack block is SCHEMA-GATED, not unconditional. Publishing it on every schema was
    # not merely wasteful (field_credit_values walks the bench on each encode): the search lane
    # PREFERS metadata["opponent_must_recharge"] over its own reconstruction, so an always-
    # present key silently changed world seeding for the v2.2/v3 arms currently in flight.
    # Tensor bytes were frozen either way; behaviour was not, and mid-run behaviour changes to
    # a live arm are exactly what the contract discipline exists to prevent.
    pack: dict[str, Any] = {}
    if schema_version in FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS:
        pack = _feature_pack_metadata(state, dex=dex)
    return {
        **pack,
        "battle_id": state.battle_id,
        "player_id": state.player_id,
        "request_kind": state.request_kind,
        "showdown_slot": state.perspective.showdown_slot,
        "opponent_showdown_slot": state.perspective.opponent_showdown_slot,
        "self_side_conditions": list(state.self_side_conditions),
        "opponent_side_conditions": list(state.opponent_side_conditions),
        "self_side_condition_counts": dict(state.self_side_condition_counts),
        "opponent_side_condition_counts": dict(state.opponent_side_condition_counts),
        "weather": state.weather,
        "turn_number": state.turn_number,
        "self_active_boosts": dict(state.self_active_boosts),
        "opponent_active_boosts": dict(state.opponent_active_boosts),
        "self_active_volatiles": list(state.self_active_volatiles),
        "opponent_active_volatiles": list(state.opponent_active_volatiles),
        "self_future_sight_turns": state.self_future_sight_turns,
        "opponent_future_sight_turns": state.opponent_future_sight_turns,
        "self_toxic_stage": state.self_toxic_stage,
        "opponent_toxic_stage": state.opponent_toxic_stage,
        "self_active": _pokemon_metadata(state.self_active),
        "opponent_active": _pokemon_metadata(state.opponent_active),
        "self_team": [_pokemon_metadata(pokemon) for pokemon in state.self_team],
        "opponent_team": [_pokemon_metadata(pokemon) for pokemon in state.opponent_team],
        "action_candidates": _action_candidate_metadata(state),
        "recent_public_events": list(state.recent_public_events),
        "transition_token_count": len(state.transition_tokens),
        "self_sleep_clause_used": state.self_sleep_clause_used,
        "opponent_sleep_clause_used": state.opponent_sleep_clause_used,
        "weather_turns_remaining": state.weather_turns_remaining,
        "weather_permanent": state.weather_permanent,
        "self_wish_pending": state.self_wish_pending,
        "opponent_wish_pending": state.opponent_wish_pending,
        # V3 public-state inputs. These remain metadata-only under V2.x and let the
        # schema-bound Rust/golden encoders reproduce V3 without replaying private data.
        "self_sleep_clause_blocks": state.self_sleep_clause_blocks,
        "opponent_sleep_clause_blocks": state.opponent_sleep_clause_blocks,
        "self_wish_turns": state.self_wish_turns,
        "opponent_wish_turns": state.opponent_wish_turns,
        "self_stall_counter": state.self_stall_counter,
        "opponent_stall_counter": state.opponent_stall_counter,
        "self_confusion_elapsed": state.self_confusion_elapsed,
        "opponent_confusion_elapsed": state.opponent_confusion_elapsed,
        "self_encore_elapsed": state.self_encore_elapsed,
        "opponent_encore_elapsed": state.opponent_encore_elapsed,
        "self_wrap_trap_elapsed": state.self_wrap_trap_elapsed,
        "opponent_wrap_trap_elapsed": state.opponent_wrap_trap_elapsed,
        "self_meanlook_trap": state.self_meanlook_trap,
        "opponent_meanlook_trap": state.opponent_meanlook_trap,
    }


def _action_candidate_metadata(state: PlayerRelativeBattleState) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    active_request = _active_request(state.request)
    moves = active_request.get("moves") if isinstance(active_request, Mapping) else None
    for move_index in range(MOVE_ACTION_COUNT):
        move = moves[move_index] if isinstance(moves, list) and move_index < len(moves) else None
        move_name = _request_move_name(move) if isinstance(move, Mapping) else f"slot:{move_index + 1}"
        candidates.append(
            {
                "action_index": move_index,
                "kind": "move",
                "legal": bool(state.legal_action_mask[move_index]),
                "move_slot": move_index + 1,
                "move_id": _normalize_identifier(move_name),
                "move_name": move_name,
                "disabled": bool(move.get("disabled")) if isinstance(move, Mapping) else True,
                "target_species": state.opponent_active.species if state.opponent_active is not None else None,
            }
        )

    active_team_index = _active_team_index(state.self_team)
    switch_targets = (
        canonical_switch_action_map(active_team_index, team_size=len(state.self_team))
        if active_team_index is not None and len(state.self_team) >= 2
        else ()
    )
    for switch_slot in range(ACTION_CANDIDATE_TOKEN_COUNT - MOVE_ACTION_COUNT):
        action_index = MOVE_ACTION_COUNT + switch_slot
        team_index = switch_targets[switch_slot] if switch_slot < len(switch_targets) else None
        pokemon = state.self_team[team_index] if team_index is not None and team_index < len(state.self_team) else None
        candidates.append(
            {
                "action_index": action_index,
                "kind": "switch",
                "legal": bool(state.legal_action_mask[action_index]),
                "switch_slot": switch_slot + 1,
                "team_index": team_index,
                "pokemon": _pokemon_metadata(pokemon),
            }
        )
    return candidates


def _feature_pack_metadata(
    state: PlayerRelativeBattleState, *, dex: "ShowdownDex | None"
) -> dict[str, Any]:
    """V4 public-state inputs (the k0 feature pack), for v4 encodes only.

    Metadata-only under every earlier schema in the sense that it is simply ABSENT there: it
    lets the schema-bound Rust/golden encoders reproduce V4 without replaying private data, and
    it is the surface the SEARCH lane reads for ``must_recharge`` so the world and the
    observation share one parser truth — but only for the schema that has those columns.
    """

    return {
        # Part B's settled column values, published so the native leaf encoder reads the same
        # numbers this encoder writes rather than re-deriving the grounding rule in Rust.
        **field_credit_values(state, dex=dex),
        "self_must_recharge": state.self_must_recharge,
        "opponent_must_recharge": state.opponent_must_recharge,
        "self_truant_loaf": state.self_truant_loaf,
        "opponent_truant_loaf": state.opponent_truant_loaf,
        "self_last_used_move": state.self_last_used_move,
        "opponent_last_used_move": state.opponent_last_used_move,
        "self_traced_ability": state.self_traced_ability,
        "opponent_traced_ability": state.opponent_traced_ability,
        "self_last_damage_dealt": state.self_last_damage_dealt,
        "self_last_damage_taken": state.self_last_damage_taken,
        "opponent_last_damage_dealt": state.opponent_last_damage_dealt,
        "opponent_last_damage_taken": state.opponent_last_damage_taken,
        "self_hazard_damage_suffered": state.self_hazard_damage_suffered,
        "opponent_hazard_damage_suffered": state.opponent_hazard_damage_suffered,
        "self_items_removed": state.self_items_removed,
        "opponent_items_removed": state.opponent_items_removed,
        "self_arrived_by_baton_pass": state.self_arrived_by_baton_pass,
        "opponent_arrived_by_baton_pass": state.opponent_arrived_by_baton_pass,
        "self_choice_locked": state.self_choice_locked,
        "opponent_choice_locked": state.opponent_choice_locked,
        "self_item_swapped": state.self_item_swapped,
        "opponent_item_swapped": state.opponent_item_swapped,
        "opponent_matchup_switch_evidence": {
            species: list(pair)
            for species, pair in state.opponent_matchup_switch_evidence.items()
        },
    }


def _pokemon_metadata(pokemon: ShowdownPokemon | None) -> dict[str, Any] | None:
    if pokemon is None:
        return None
    condition = _condition_features(pokemon.condition)
    return {
        "ident": pokemon.ident,
        "showdown_slot": pokemon.showdown_slot,
        "species": pokemon.species,
        "condition": pokemon.condition,
        "hp_fraction": condition.hp_fraction,
        "status": condition.status,
        "fainted": condition.fainted,
        "active": pokemon.active,
        "details": pokemon.details,
        "moves": list(pokemon.moves),
        "ability": pokemon.ability,
        "item": pokemon.item,
        "stats": dict(pokemon.stats) if pokemon.stats is not None else None,
        "live_type_source": pokemon.live_type_source,
    }


@dataclass(frozen=True)
class _ConditionFeatures:
    hp_fraction: Optional[float]
    status: str
    fainted: bool


def _condition_features(condition: str | None) -> _ConditionFeatures:
    parts = str(condition or "").split()
    hp_fraction: Optional[float] = None
    if parts and "/" in parts[0]:
        numerator, _, denominator = parts[0].partition("/")
        try:
            hp_fraction = max(0.0, min(1.0, float(numerator) / float(denominator)))
        except (TypeError, ValueError, ZeroDivisionError):
            hp_fraction = None
    elif parts and parts[0] == "0":
        hp_fraction = 0.0
    fainted = "fnt" in parts
    status = next((part for part in parts[1:] if part != "fnt"), "none")
    return _ConditionFeatures(hp_fraction=hp_fraction, status=status, fainted=fainted)


def _set_category(row: list[str], index: int, value: str) -> None:
    if index < len(row):
        row[index] = value


def _set_numeric(row: list[float], index: int, value: float) -> None:
    if index < len(row):
        row[index] = float(value)


def _known_or_possible_values(known: str | None, possible: Sequence[str]) -> tuple[str, ...]:
    if known:
        return (known,)
    return _compact_belief_values(possible)


def _prioritized_belief_moves(
    revealed_moves: Sequence[str], possible_moves: Sequence[str], limit: int
) -> tuple[str, ...]:
    """Revealed moves (ground truth) first and never evicted; fill the rest with possible_moves.

    ``_encode_belief_fact_categories`` sorts its values alphabetically and truncates to the bucket
    count, so passing ``revealed + possible`` unbounded could drop an alphabetically-late REVEALED
    move once the union exceeds ``limit`` (reachable off-script, where a revealed move is not in
    possible_moves). Cap the union here — revealed kept in full — so the downstream sort/truncate
    can never evict a ground-truth reveal."""
    values = list(revealed_moves)
    seen = {_normalize_identifier(move) for move in revealed_moves if _normalize_identifier(move)}
    for move in possible_moves:
        if len(seen) >= limit:
            break
        key = _normalize_identifier(move)
        if key and key not in seen:
            values.append(move)
            seen.add(key)
    return tuple(values)


def _encode_belief_fact_categories(row: list[str], fact_kind: str, values: Sequence[str]) -> None:
    offset, bucket_count = _belief_bucket_range(fact_kind)
    # Place the (sorted, deduped) belief values positionally into this fact's columns. The bucket
    # counts are sized to the Gen 3 closed universe's per-species maxima (2 abilities / 5 items /
    # 14 moves), so positional placement is exact and collision-free — no hashing needed. The
    # stored value is the category string, converted to a vocab row later.
    for index, value in enumerate(_compact_belief_values(values, limit=bucket_count)):
        column = offset + index
        if column >= len(row):
            break
        row[column] = f"belief:{fact_kind}:{_normalize_identifier(value)}"


def _belief_bucket_range(fact_kind: str) -> tuple[int, int]:
    if fact_kind == "possible_ability":
        return CATEGORY_BELIEF_ABILITY_OFFSET, BELIEF_ABILITY_BUCKET_COUNT
    if fact_kind == "possible_item":
        return CATEGORY_BELIEF_ITEM_OFFSET, BELIEF_ITEM_BUCKET_COUNT
    if fact_kind == "possible_move":
        return CATEGORY_BELIEF_MOVE_OFFSET, BELIEF_MOVE_BUCKET_COUNT
    raise ValueError(f"unsupported belief fact kind: {fact_kind!r}")


def _compact_belief_values(values: Sequence[str], *, limit: int | None = None) -> tuple[str, ...]:
    compact_by_key: dict[str, str] = {}
    for raw_value in values:
        value = str(raw_value or "").strip()
        if not value:
            continue
        key = _normalize_identifier(value)
        if not key or key in compact_by_key:
            continue
        compact_by_key[key] = value
    compact = tuple(value for _, value in sorted(compact_by_key.items()))
    if limit is None:
        return compact
    return compact[:limit]


def _belief_for_species(
    beliefs_by_species: Mapping[str, RevealedPokemonBelief] | None,
    species: str,
) -> RevealedPokemonBelief | None:
    if not beliefs_by_species:
        return None
    return beliefs_by_species.get(_normalize_identifier(species))


def _legal_action_mask(request: Mapping[str, Any] | None) -> tuple[bool, ...]:
    mask = [False] * ACTION_COUNT
    if not isinstance(request, Mapping) or request.get("wait"):
        return tuple(mask)

    force_switch = request.get("forceSwitch")
    force_switch_requested = isinstance(force_switch, list) and any(bool(slot) for slot in force_switch)
    if not force_switch_requested:
        active_rows = request.get("active")
        active = active_rows[0] if isinstance(active_rows, list) and active_rows and isinstance(active_rows[0], Mapping) else None
        moves = active.get("moves") if isinstance(active, Mapping) else None
        if isinstance(moves, list):
            for move_index, move in enumerate(moves[:MOVE_ACTION_COUNT]):
                if isinstance(move, Mapping) and not move.get("disabled", False):
                    mask[move_index] = True

    if force_switch_requested or _switching_allowed(request):
        active_team_index = _active_team_index(_self_team_from_request(request, _request_side_id(request) or "p1"))
        team_size = _team_size_from_request(request)
        if active_team_index is not None and team_size >= 2:
            for switch_slot, team_index in enumerate(canonical_switch_action_map(active_team_index, team_size=team_size)):
                pokemon = _request_pokemon_at(request, team_index)
                if pokemon is not None and _can_switch_to(pokemon):
                    mask[MOVE_ACTION_COUNT + switch_slot] = True
    return tuple(mask)


def _request_kind(request: Mapping[str, Any] | None) -> str:
    if not isinstance(request, Mapping):
        return "none"
    if request.get("wait"):
        return "wait"
    if request.get("teamPreview"):
        return "team_preview"
    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list) and any(bool(slot) for slot in force_switch):
        return "force_switch"
    if request.get("active"):
        return "move"
    return "unknown"


def _switching_allowed(request: Mapping[str, Any]) -> bool:
    active_rows = request.get("active")
    active = active_rows[0] if isinstance(active_rows, list) and active_rows and isinstance(active_rows[0], Mapping) else None
    if isinstance(active, Mapping) and (active.get("trapped") is True or active.get("maybeTrapped") is True):
        return False
    return _request_kind(request) == "move"


def _request_pokemon_at(request: Mapping[str, Any], team_index: int) -> Mapping[str, Any] | None:
    side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
    pokemon = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(pokemon, list) or team_index < 0 or team_index >= len(pokemon):
        return None
    candidate = pokemon[team_index]
    return candidate if isinstance(candidate, Mapping) else None


def _active_request(request: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    active_rows = request.get("active") if isinstance(request, Mapping) else None
    if isinstance(active_rows, list) and active_rows and isinstance(active_rows[0], Mapping):
        return active_rows[0]
    return None


def _active_request_moves(request: Mapping[str, Any] | None) -> tuple[str, ...]:
    active = _active_request(request)
    moves = active.get("moves") if isinstance(active, Mapping) else None
    if not isinstance(moves, list):
        return ()
    return tuple(
        _request_move_name(move)
        for move in moves
        if isinstance(move, Mapping)
    )


def _request_pokemon_moves(row: Mapping[str, Any]) -> tuple[str, ...]:
    moves = row.get("moves")
    if not isinstance(moves, list):
        return ()
    return tuple(str(move).strip() for move in moves if str(move).strip())


def _request_pokemon_ability(row: Mapping[str, Any]) -> str | None:
    for key in ("ability", "baseAbility"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _request_pokemon_item(row: Mapping[str, Any]) -> str | None:
    value = row.get("item")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _move_pp_fraction(move: Mapping[str, Any]) -> float:
    """Remaining PP as a fraction of max PP from a request move (1.0 if PP data is absent)."""
    pp = move.get("pp")
    maxpp = move.get("maxpp")
    if isinstance(pp, (int, float)) and isinstance(maxpp, (int, float)) and maxpp:
        return max(0.0, min(1.0, float(pp) / float(maxpp)))
    return 1.0


def _request_move_name(move: Mapping[str, Any]) -> str:
    for key in ("id", "move"):
        value = move.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


_HIDDEN_POWER_TYPES = frozenset(
    {
        "bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost",
        "grass", "ground", "ice", "poison", "psychic", "rock", "steel", "water",
    }
)


def _hidden_power_variant_from_name(display_name: Any) -> str | None:
    """Typed Hidden Power id from a request's display move name.

    "Hidden Power Fighting 70" -> "hiddenpowerfighting". Returns None if the name carries no
    recognizable HP type (leaving the caller to fall back)."""
    if not isinstance(display_name, str):
        return None
    for token in re.findall(r"[a-z]+", display_name.lower()):
        if token in _HIDDEN_POWER_TYPES:
            return f"hiddenpower{token}"
    return None


def _self_move_mechanics_id(
    move: Mapping[str, Any], move_name: str, own_move_ids: Sequence[str] = ()
) -> str:
    """Move id to look up for SELF action-token MECHANICS (type / base power / damage class).

    Hidden Power's request keys ``id`` to the generic family ("hiddenpower"), whose dex entry is a
    0-power Normal placeholder — so the acting mon would encode its single most common coverage move
    as a Normal, 0-BP no-op. The real typed identity is self-observable two ways: authoritatively
    from the display ``move`` field ("Hidden Power Fighting 70"), and, as a fallback, from the mon's
    own typed move id in the request side list ("hiddenpowerfighting", which Showdown derives from
    its IVs). Resolve the typed variant for the mechanics lookup ONLY; the action token's move
    IDENTITY (CATEGORY_PRIMARY = ``move:hiddenpower``) stays generic and checkpoint-stable. Every
    non-Hidden-Power move passes straight through."""
    if _normalize_identifier(move_name) != "hiddenpower":
        return move_name
    typed = _hidden_power_variant_from_name(move.get("move"))
    if typed is not None:
        return typed
    for candidate in own_move_ids:
        normalized = _normalize_identifier(candidate)
        if normalized.startswith("hiddenpower") and len(normalized) > len("hiddenpower"):
            return normalized
    return move_name


def _request_side_id(request: Mapping[str, Any]) -> str | None:
    side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
    side_id = side.get("id") if isinstance(side, Mapping) else None
    return side_id if side_id in {"p1", "p2"} else None


def _team_size_from_request(request: Mapping[str, Any]) -> int:
    side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
    pokemon = side.get("pokemon") if isinstance(side, Mapping) else None
    return len(pokemon) if isinstance(pokemon, list) else 0


def _can_switch_to(pokemon: Mapping[str, Any]) -> bool:
    if pokemon.get("active"):
        return False
    condition = str(pokemon.get("condition") or "")
    return not condition.startswith("0 ")


def _active_team_index(team: Sequence[ShowdownPokemon]) -> int | None:
    for index, pokemon in enumerate(team):
        if pokemon.active:
            return index
    return None


def _species_from_request_pokemon(row: Mapping[str, Any]) -> str:
    details = row.get("details")
    ident = row.get("ident")
    if isinstance(details, str) and details.strip():
        return _species_from_details(details)
    if isinstance(ident, str):
        return _species_from_ident(ident)
    return "unknown"


def _species_from_details(details: str) -> str:
    return details.split(",", 1)[0].strip()


def _species_from_ident(ident: str) -> str:
    return ident.split(":", 1)[-1].strip() or "unknown"


def _slot_from_ident(ident: str) -> str | None:
    match = re.match(r"^(p[12])", ident.strip())
    return match.group(1) if match else None


def _ident_has_position(ident: str | None) -> bool:
    """True for an ACTIVE-slot ident (``p2a: Snorlax``); False for a benched ident (``p2: Snorlax``).

    Showdown appends a field-position letter (``a`` in singles) only to on-field Pokemon; a benched
    mon referenced by a team-wide effect (Heal Bell curing every ally) carries just ``pN:``. Mirrors
    ``belief._ident_has_position`` so the parser and belief surfaces classify cure idents identically."""
    return bool(re.match(r"^p[12][a-z]", str(ident or "")))


def _normalize_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _normalize_public_field(field: str, *, self_slot: str, opponent_slot: str) -> str:
    field = re.sub(rf"^{self_slot}([a-z]?):", r"self\1:", field)
    return re.sub(rf"^{opponent_slot}([a-z]?):", r"opponent\1:", field)


def _record_public_reveal(
    public_revealed: dict[str, list[ShowdownPokemon]],
    pokemon: ShowdownPokemon,
) -> None:
    current = public_revealed.setdefault(pokemon.showdown_slot, [])
    next_revealed: list[ShowdownPokemon] = []
    matched = False
    for existing in current:
        if _same_public_pokemon(existing, pokemon):
            next_revealed.append(pokemon)
            matched = True
        else:
            next_revealed.append(replace(existing, active=False))
    if not matched:
        next_revealed.append(pokemon)
    public_revealed[pokemon.showdown_slot] = next_revealed


def _same_public_pokemon(left: ShowdownPokemon, right: ShowdownPokemon) -> bool:
    return left.showdown_slot == right.showdown_slot and left.species == right.species


def _token_type_ids(spec: ObservationSpec) -> tuple[int, ...]:
    # Type id 4 (the v1 recent-event section) is retired, not reused: 5 = stats, 6 = transition.
    token_types: list[int] = []
    token_types.extend([0])
    token_types.extend([1] * 6)
    token_types.extend([2] * 6)
    token_types.extend([3] * ACTION_COUNT)
    token_types.extend([5] * spec.opponent_tendency_stats_token_count)
    token_types.extend([6] * spec.transition_token_count)
    return tuple(token_types)


def _attention_mask(
    state: PlayerRelativeBattleState,
    spec: ObservationSpec,
    *,
    masks: ObservationFeatureMasks = DEFAULT_OBSERVATION_FEATURE_MASKS,
) -> tuple[bool, ...]:
    mask: list[bool] = []
    mask.extend([True])
    mask.extend(index < len(state.self_team) for index in range(6))
    mask.extend(index < len(state.opponent_team) for index in range(6))
    mask.extend([True] * ACTION_COUNT)
    opponent_tendency_stats_visible = masks.opponent_tendency_stats_block and state.tendency_stats is not None
    mask.extend([opponent_tendency_stats_visible] * spec.opponent_tendency_stats_token_count)
    transition_stream = (
        state.turn_merged_tokens
        if spec.schema_version in (OBSERVATION_SCHEMA_VERSION_V2_2, OBSERVATION_SCHEMA_VERSION_V3)
        else state.transition_tokens
    )
    filled = min(
        len(transition_stream), masks.transition_token_budget, spec.transition_token_count
    )
    mask.extend(index < filled for index in range(spec.transition_token_count))
    return tuple(mask)
