"""Minimal Showdown replay normalization helpers.

This module is intentionally small: it is a testable boundary between raw
Showdown protocol seats (`p1`/`p2`) and PokeZero's player-relative model input.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import re
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
    FIELD_TOKEN_COUNT,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION_V2,
    OBSERVATION_SCHEMA_VERSION_V2_1,
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
    OPPONENT_POKEMON_TOKEN_COUNT,
    OPPONENT_TENDENCY_STATS_TOKEN_COUNT,
    TRANSITION_TOKEN_COUNT,
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
# LAYER SEPARATION (architectural invariant): Tier-2 conclusions NEVER mutate the belief
# engine's Tier-1 candidate sets — the exact/protocol layer stays inference-free; this is
# a parallel tier2-layer feature carried on the same token, not a belief-fact write.
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
# env_config_with_checkpoint_masks latch); DEFAULT_REPLAY_OBSERVATION_SPEC is only the
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
# Change 10 — confusion self-hit damage-attribution correction, one 0/1 bit on the OPPONENT's
# turn-merged move sub-block (the token whose damage was polluted), schema >= v3 only. When a
# SLOWER confused mon self-hits, the sim emits ``|-activate|SLOT|confusion`` then an UNTAGGED
# ``|-damage|SLOT|…`` with no |move|/|cant| line; the fold folds that self-damage into the
# opponent's still-open move window's ``damage_fraction`` (correct for the v2.2 field, which is
# left FROZEN). Under v3 the encode writes the move's damage-fraction column as
# ``damage_fraction - confusion_selfhit_fraction`` (the move's own damage, self-hit removed) and
# sets this bit = "the defender self-hit from confusion after this move." A single column (not a
# first/second pair like the fail bit) because the correction always rides the FIRST sub-block in
# practice — the confused mon must be SLOWER, so the opponent moved first; the write is mirrored
# onto the second sub-block defensively. Additive/schema-agnostic at extraction (the token fields
# are always populated); only a v3 encode reads them, so v2/v2.1/v2.2 output stays byte-identical.
NUMERIC_TT_CONFUSION_SELFHIT = V3_LEGACY_NUMERIC_BASE + 13
# EXTRA counts the stall-counter column (+4, change 3), the confusion column (+5, change 4), the
# encore column (+6, change 5), the Wrap partial-trap column (+7, change 6), the two gender bits
# (+8 / +9, change 7), the Mean Look move-trap bit (+10, change 8), the two Wish turns-to-land
# bits (+11 / +12, change 9), and the confusion self-hit flag (+13, change 10). This is the
# private, pre-cutover writer surface; the public v3 width is derived from the layout map below.
V3_LEGACY_NUMERIC_EXTRA = 14
V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT = V3_LEGACY_NUMERIC_BASE + V3_LEGACY_NUMERIC_EXTRA

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

V2_2_REPLAY_OBSERVATION_SPEC = ObservationSpec(
    categorical_feature_count=_V2_2_CATEGORICAL_FEATURE_COUNT,
    numeric_feature_count=_V2_2_NUMERIC_FEATURE_COUNT,
    schema_version=OBSERVATION_SCHEMA_VERSION_V2_2,
)
V3_REPLAY_OBSERVATION_SPEC = ObservationSpec(
    categorical_feature_count=_V3_CATEGORICAL_FEATURE_COUNT,
    numeric_feature_count=_V3_NUMERIC_FEATURE_COUNT,
    schema_version=OBSERVATION_SCHEMA_VERSION_V3,
)
REPLAY_OBSERVATION_SPECS_BY_SCHEMA: Mapping[str, ObservationSpec] = {
    OBSERVATION_SCHEMA_VERSION_V2: V2_REPLAY_OBSERVATION_SPEC,
    OBSERVATION_SCHEMA_VERSION_V2_1: V2_1_REPLAY_OBSERVATION_SPEC,
    OBSERVATION_SCHEMA_VERSION_V2_2: V2_2_REPLAY_OBSERVATION_SPEC,
    OBSERVATION_SCHEMA_VERSION_V3: V3_REPLAY_OBSERVATION_SPEC,
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
}
_MINIMUM_NUMERIC_CENSUS_BY_SCHEMA: Mapping[str, int] = {
    OBSERVATION_SCHEMA_VERSION_V2: 119,
    OBSERVATION_SCHEMA_VERSION_V2_1: _V2_1_NUMERIC_FEATURE_COUNT,
    OBSERVATION_SCHEMA_VERSION_V2_2: _V2_2_NUMERIC_FEATURE_COUNT,
    OBSERVATION_SCHEMA_VERSION_V3: _V3_NUMERIC_FEATURE_COUNT,
}


# CLI short names for the schema-selection flag (--observation-schema). v2 is
# deliberately NOT offered for fresh selection: it exists only as a checkpoint-driven
# legacy mode.
OBSERVATION_SCHEMA_CLI_CHOICES: Mapping[str, str] = {
    "v2.1": OBSERVATION_SCHEMA_VERSION_V2_1,
    "v2.2": OBSERVATION_SCHEMA_VERSION_V2_2,
    "v3": OBSERVATION_SCHEMA_VERSION_V3,
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
    if schema_version == OBSERVATION_SCHEMA_VERSION_V3:
        return v3_numeric_index(legacy_index)
    if legacy_index < 0 or legacy_index >= spec.numeric_feature_count:
        raise ValueError(
            f"legacy numeric column {legacy_index} is outside the "
            f"{spec.numeric_feature_count}-column public layout for schema {schema_version!r}"
        )
    return legacy_index


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
    # Public sleep-clause tracker (spec v3, docs/observation_v3_spec.md change 2): per INDUCING
    # side, the set of enemy victims it has publicly put to sleep (victim keys
    # ``<slot>:<normalized ident name>``). Attribution rule: a ``-status … slp`` line WITHOUT
    # the ``[from] move: Rest`` tag ⇒ induced by the opposing side (in gen3 singles sleep is
    # only ever opponent-induced or self-inflicted Rest, and Rest tags its line). Cleared on
    # ``-curestatus … slp`` and faint; switch-out does NOT clear (sleep persists and is public
    # on revealed mons). Derived ONLY from public protocol lines — no engine-side hidden state.
    induced_sleep_victims: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
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

    def __init__(self, battle_id: str = "replay") -> None:
        self.battle_id = battle_id
        self.players: dict[str, str] = {}
        self.requests: dict[str, Mapping[str, Any]] = {}
        self.public_active: dict[str, ShowdownPokemon] = {}
        self.public_revealed: dict[str, list[ShowdownPokemon]] = {}
        self.side_condition_counts: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
        self.boosts: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
        self.volatiles: dict[str, set[str]] = {"p1": set(), "p2": set()}
        self.direct_materialization_blockers: dict[str, set[str]] = {"p1": set(), "p2": set()}
        self.future_sight: dict[str, int] = {}
        self.toxic_stage: dict[str, int] = {"p1": 0, "p2": 0}
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
        # Public sleep-clause tracker (spec v3): per INDUCING side, the set of enemy victims
        # it has publicly put to sleep. See ShowdownReplayState.induced_sleep_victims.
        self.induced_sleep_victims: dict[str, set[str]] = {"p1": set(), "p2": set()}
        # Public consecutive-stall counter (spec v3 change 3) + its transient in-flight flag.
        # See ShowdownReplayState.stall_counter / stall_move_pending.
        self.stall_counter: dict[str, int] = {"p1": 0, "p2": 0}
        self.stall_move_pending: dict[str, bool] = {"p1": False, "p2": False}

    @classmethod
    def from_snapshot(cls, snapshot: ShowdownReplayState) -> "_ReplayParser":
        """Hydrate parser state directly, without replaying its protocol prefix."""

        parser = cls(snapshot.battle_id)
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
        parser.direct_materialization_blockers = {
            slot: set(snapshot.direct_materialization_blockers.get(slot, ()))
            for slot in ("p1", "p2")
        }
        parser.future_sight = dict(snapshot.future_sight)
        parser.toxic_stage = {slot: int(snapshot.toxic_stage.get(slot, 0)) for slot in ("p1", "p2")}
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
        parser.induced_sleep_victims = {
            slot: set(snapshot.induced_sleep_victims.get(slot, ())) for slot in ("p1", "p2")
        }
        parser.stall_counter = {
            slot: int(snapshot.stall_counter.get(slot, 0)) for slot in ("p1", "p2")
        }
        parser.stall_move_pending = {
            slot: bool(snapshot.stall_move_pending.get(slot, False)) for slot in ("p1", "p2")
        }
        return parser

    def feed(self, lines: Sequence[str]) -> None:
        for raw_line in lines:
            self._feed_line(raw_line)

    def _feed_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return
        if line.startswith(">"):
            return
        parts = line.split("|")
        event_type = parts[1] if len(parts) > 1 else ""
        # BattleStream emits wall-clock timestamp lines (``|t:|...``). They are useful for raw
        # protocol debugging but are not battle state and would make replay-from-root observations
        # differ across otherwise identical deterministic simulations.
        if event_type == "t:":
            return
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
            pokemon = _pokemon_from_public_line(parts)
            if pokemon is not None:
                self.public_active[pokemon.showdown_slot] = pokemon
                _record_public_reveal(self.public_revealed, pokemon)
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
                # Gen 3 resets the toxic counter when a mon leaves the field.
                self.toxic_stage[pokemon.showdown_slot] = 0
                # The stall streak belongs to the mon that left the slot (the ``stall`` volatile
                # clears on switch/faint); switch-out/drag is reset cause (4). Clear the in-flight
                # flag too so no stale stall move carries onto the replacement.
                self.stall_counter[pokemon.showdown_slot] = 0
                self.stall_move_pending[pokemon.showdown_slot] = False
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
            self.public_events.append(_public_event_from_line(line))
            self.public_lines.append(line)
            return
        if event_type == "win" and len(parts) >= 3:
            self.winner = parts[2]
            self.public_events.append(_public_event_from_line(line))
            self.public_lines.append(line)
            return
        if event_type == "turn" and len(parts) >= 3:
            try:
                self.turn_number = int(parts[2])
            except (TypeError, ValueError):
                pass
            # A successful Baton Pass is consumed by its same-turn forced switch. Anything still
            # pending at a fresh turn belongs to a failed or truncated protocol sequence.
            self.pending_baton_pass.clear()
            # Each turn a badly-poisoned mon stays in, its toxic damage escalates (1/16, 2/16, ...).
            for slot, stage in self.toxic_stage.items():
                if stage:
                    self.toxic_stage[slot] = min(15, stage + 1)
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
        if event_type == "-fail" and len(parts) >= 3:
            # A failed Baton Pass emits its move declaration but no switch request. Do not let
            # that declaration turn a later ordinary switch into a phantom Baton Pass.
            self.pending_baton_pass.discard(_slot_from_ident(parts[2]))
        # Re-seed the toxic ramp from the PUBLIC end-of-turn residual BEFORE the condition update
        # overwrites the pre-damage HP (needed to measure the residual's magnitude).
        self._reseed_toxic_stage_from_residual(parts)
        _update_public_pokemon_condition(parts, self.public_active, self.public_revealed)
        _update_side_conditions(parts, self.side_condition_counts)
        self.weather = _update_weather(parts, self.weather)
        self._update_weather_meta(parts, line)
        self._update_timed_side_conditions(parts)
        self._update_wish(parts, line)
        _update_boosts(parts, self.boosts)
        _update_volatiles(parts, self.volatiles)
        self._update_live_type_override(parts)
        self._update_leech_seed(parts)
        self._prune_direct_materialization_blockers()
        _update_future_sight(parts, self.future_sight, self.turn_number)
        _update_toxic_stage(parts, self.toxic_stage)
        _update_confusion_elapsed(parts, self.confusion_elapsed)
        _update_encore_elapsed(parts, self.encore_elapsed)
        _update_wrap_trap_elapsed(parts, self.wrap_trap_elapsed)
        _update_meanlook_trap(parts, self.meanlook_trap)
        _flag_baton_pass(parts, self.pending_baton_pass)
        self._update_induced_sleep(parts, line)
        self._update_stall_counter(parts)
        self.public_events.append(_public_event_from_line(line))
        self.public_lines.append(line)

    def _reseed_toxic_stage_from_residual(self, parts: Sequence[str]) -> None:
        """Recover the badly-poisoned (tox) ramp stage from the PUBLIC end-of-turn toxic residual.

        A ``tox`` mon that switches out has its counter reset to 0 (Gen 3, ``tox.onSwitchIn`` sets
        ``effectState.stage = 0``); on re-entry the ``tox`` rides only the switch-line condition
        string with no fresh ``|-status|``, so ``_update_toxic_stage`` never re-seeds and the
        per-``|turn|`` escalation (gated on ``if stage``) can never lift it off 0 — the encoder
        would emit the contradictory ``status:tox`` + ``toxic_stage == 0`` for the whole stint.

        The exact counter is hidden, but it is publicly derivable: Gen 3 badly-poison damage is
        ``clampIntRange(maxhp/16, 1) * stage`` (the sim ``stage++``s to 1 on the first residual
        after re-entry, so the ramp restarts at 1 and climbs 1, 2, 3 …), so the observed residual
        fraction gives ``stage = round(16 * damage / maxhp)``. Re-deriving here fixes the pivot,
        the forced re-entry (Roar/Whirlwind ``|drag|``), and a mon first observed already-``tox``
        (replay import / mid-battle observe start) uniformly, for both seats. Regular (non-badly)
        poison also emits ``[from] psn`` but is a flat 1/8 with no ramp — gated out by the
        residual's own status token, which is ``tox`` only for badly-poisoned mons.
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
        active = self.public_active.get(slot)
        prev_condition = active.condition if active is not None and active.ident == parts[2] else None
        prev_hp, prev_max = _hp_numerator_denominator(prev_condition)
        cur_hp, cur_max = _hp_numerator_denominator(new_condition)
        max_hp = prev_max or cur_max
        if prev_hp is None or cur_hp is None or not max_hp:
            return
        damage = prev_hp - cur_hp
        if damage <= 0:
            return
        # round(16 * damage_fraction) recovers the sim's stage for every reachable stage (1..14;
        # a mon never survives to stage 15). Clamp to [1, 15]: a tox residual is always >= stage 1.
        self.toxic_stage[slot] = min(15, max(1, round(16 * damage / max_hp)))

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
            if victim_slot in {"p1", "p2"} and "move: Rest" not in line:
                inducing_slot = opponent_showdown_slot(victim_slot)
                self.induced_sleep_victims[inducing_slot].add(
                    self._induced_sleep_victim_key(victim_slot, parts[2])
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

    def snapshot(self) -> ShowdownReplayState:
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
            direct_materialization_blockers={
                slot: tuple(sorted(blockers))
                for slot, blockers in self.direct_materialization_blockers.items()
            },
            future_sight=dict(self.future_sight),
            toxic_stage=dict(self.toxic_stage),
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
            induced_sleep_victims={
                slot: tuple(sorted(victims))
                for slot, victims in self.induced_sleep_victims.items()
            },
            stall_counter=dict(self.stall_counter),
            stall_move_pending=dict(self.stall_move_pending),
        )


def parse_showdown_replay(lines: Sequence[str], *, battle_id: str = "replay") -> ShowdownReplayState:
    """Parse compact Showdown protocol lines into transport-level state."""
    parser = _ReplayParser(battle_id=battle_id)
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
    )


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
    schema_v3 = spec.schema_version == OBSERVATION_SCHEMA_VERSION_V3
    if schema_v3 and spec.numeric_feature_count != _V3_NUMERIC_FEATURE_COUNT:
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
    categorical_floor = _MINIMUM_CATEGORICAL_CENSUS_BY_SCHEMA[spec.schema_version]
    if spec.categorical_feature_count < categorical_floor:
        raise ValueError(
            f"observation encode: spec schema {spec.schema_version!r} requires at least "
            f"{categorical_floor} categorical columns, got {spec.categorical_feature_count}. "
            "The categorical census is schema-keyed "
            f"({OBSERVATION_SCHEMA_VERSION_V2!r} and {OBSERVATION_SCHEMA_VERSION_V2_1!r}: "
            f"{_CATEGORICAL_FEATURE_COUNT}; {OBSERVATION_SCHEMA_VERSION_V2_2!r} and "
            f"{OBSERVATION_SCHEMA_VERSION_V3!r}: "
            f"{_V2_2_CATEGORICAL_FEATURE_COUNT}); a narrower spec would silently "
            "bounds-drop the schema's own categorical surface (v2.2's whole second "
            "sub-block) and encode an undeclared hybrid stamped with the wider version."
        )
    # V3 keeps the v2.2 turn-merged semantic surface but projects the private legacy writer
    # rows into its grouped public layout after all token writers complete.
    schema_v2_2 = schema_v3 or spec.schema_version == OBSERVATION_SCHEMA_VERSION_V2_2
    # v2.2 carries every v2.1 block forward unchanged; only the transition surface differs.
    schema_v2_1 = schema_v2_2 or spec.schema_version == OBSERVATION_SCHEMA_VERSION_V2_1
    if schema_v2_2 and state.transition_tokens and not state.turn_merged_tokens:
        raise ValueError(
            "observation encode: a v2.2 (turn-merged) spec requires the state's "
            "turn_merged_tokens — normalize with include_turn_merged=True."
        )
    if schema_v2_2 and not category_vocab.is_enumerated("tt_phase:turn"):
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
    # history surface is subject to. Tier-2 conclusions never touch the Tier-1 belief
    # candidate sets (layer separation; see the column comment).
    tier2_cb_pinned_species: frozenset[str] = frozenset()
    if schema_v2_1 and feature_masks.tier2_residuals:
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
    # K-budget truncation. Triple-gated like the tt-row write (v2.1 schema + both
    # masks); layer separation holds — Tier-1 candidate sets are never touched.
    tier2_investment_pinned: dict[str, float] = {}
    if schema_v2_1 and feature_masks.tier2_residuals and feature_masks.tier2_investment:
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
        # and drop evidence-backed dead fields without perturbing a legacy writer.
        internal_numeric_feature_count=(
            V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT if schema_v3 else spec.numeric_feature_count
        ),
    )
    _encode_field_token(
        categorical_ids, numeric_features, state, masks=feature_masks, schema_v3=schema_v3
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
        dex=dex,
        exact_beliefs_by_species=self_exact_beliefs,
        masks=feature_masks,
        schema_v2_1=schema_v2_1,
        schema_v3=schema_v3,
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
        dex=dex,
        exact_beliefs_by_species=opponent_beliefs,
        tendency_by_species=tendency_by_species,
        # Transform copy targets: in singles an opponent Transform copies OUR mon; species
        # clause makes the by-species lookup unique within our team.
        transform_targets_by_species={
            _normalize_identifier(member.species): member for member in state.self_team
        },
        masks=feature_masks,
        schema_v2_1=schema_v2_1,
        schema_v3=schema_v3,
        tier2_cb_pinned_species=tier2_cb_pinned_species,
        tier2_investment_pinned=tier2_investment_pinned,
    )
    _encode_action_tokens(categorical_ids, numeric_features, state, dex=dex)
    _encode_stats_token(categorical_ids, numeric_features, state, masks=feature_masks)
    if schema_v2_2:
        _encode_turn_merged_transition_tokens(
            categorical_ids, numeric_features, state, spec, masks=feature_masks, schema_v3=schema_v3
        )
    else:
        _encode_transition_tokens(
            categorical_ids, numeric_features, state, spec, masks=feature_masks, schema_v2_1=schema_v2_1
        )
    if schema_v3:
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
        metadata=_observation_metadata(state),
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
TRACKED_VOLATILES = frozenset({
    "confusion", "leechseed", "substitute", "taunt", "encore", "disable", "torment", "attract",
    "nightmare", "curse", "ingrain", "foresight", "lockon", "mindreader", "destinybond", "grudge",
    "focusenergy", "charge", "yawn", "stockpile", "bide", "uproar", "imprison", "magiccoat",
    "snatch", "mudsport", "watersport", "defensecurl", "minimize", "rage", "partiallytrapped",
    "perishsong", "perish0", "perish1", "perish2", "perish3", "flashfire",
})

# Pokemon Showdown's Gen 3 `Pokemon.copyVolatileFrom` carries conditions whose
# `noCopy` flag is false. This is the tracked subset. The parser preserves these
# public facts through Baton Pass, then direct materialization rejects any whose
# serialized state is not yet fully public and reconstructable.
_BATON_PASS_TRANSFERRED_VOLATILES = frozenset({
    "confusion", "leechseed", "substitute", "taunt", "curse", "ingrain", "lockon",
    "grudge", "focusenergy", "charge", "bide", "uproar", "magiccoat", "snatch",
    "mudsport", "watersport", "rage", "partiallytrapped", "perishsong",
})
_DIRECT_MATERIALIZATION_VOLATILES = frozenset({
    "focusenergy", "ingrain", "leechseed", "mudsport", "watersport",
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
    if event_type in {"move", "cant"}:
        # The sim removes single-move volatiles silently before the mon's next action
        # (onBeforeMove / onMoveAborted); a successful re-click re-arms via the
        # following |-singlemove| line.
        volatiles[slot] -= _SINGLEMOVE_VOLATILES
        return
    if len(parts) < 4:
        return
    name = _side_condition_identifier(parts[3])  # strips move:/ability:/item: prefix + normalizes
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


def _update_toxic_stage(parts: Sequence[str], toxic_stage: dict[str, int]) -> None:
    """Track the badly-poisoned (tox) ramp stage per side from |-status| / |-curestatus| /
    |-cureteam| lines.

    A `tox` status starts the counter at 1 (per-turn escalation is applied on |turn|); any cured
    status — per-mon (`-curestatus`) or team-wide (`-cureteam`/Aromatherapy) — clears it. The
    counter is also reset on switch (Gen 3 behavior) in the parse loop.
    """
    event_type = parts[1] if len(parts) > 1 else ""
    if len(parts) < 3:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in toxic_stage:
        return
    if event_type == "-status" and len(parts) >= 4 and _normalize_identifier(parts[3]) == "tox":
        toxic_stage[slot] = 1
    elif event_type in {"-curestatus", "-cureteam"}:
        # ``-cureteam`` (Aromatherapy) ident is the active source, which is itself cured,
        # so resetting the active slot's ramp matches the per-mon ``-curestatus`` reset.
        toxic_stage[slot] = 0


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

    Works for both absolute HP (own/omniscient stream) and the percentage form (``85/100``); the
    caller derives the toxic-residual fraction from the pair, so either scale recovers the stage.
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


def _encode_field_token(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    state: PlayerRelativeBattleState,
    *,
    masks: ObservationFeatureMasks = DEFAULT_OBSERVATION_FEATURE_MASKS,
    schema_v3: bool = False,
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


def _encode_active_volatiles(cat_row: list[str], volatiles: Sequence[str]) -> None:
    """Place active-mon volatile statuses (sorted) positionally into the volatile columns."""
    for index, name in enumerate(sorted(set(volatiles))[:VOLATILE_BUCKET_COUNT]):
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
    tier2_cb_pinned_species: frozenset[str] = frozenset(),
    tier2_investment_pinned: Mapping[str, float] | None = None,
    active_meanlook_trap: bool = False,
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
            _encode_active_volatiles(categorical_ids[token_index], active_volatiles)
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
                )
        if masks.opponent_tendency_stats_block and role == "opponent" and tendency_by_species:
            tendency = tendency_by_species.get(_normalize_identifier(candidate.species))
            if tendency is not None:
                _encode_mon_tendency(numeric_features[token_index], tendency)
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
) -> None:
    """Deterministic opponent stat block from species + level + the fixed 85/31/neutral spread.

    Def/SpA/SpD/Spe are exact (the generator never varies them). HP and Atk are
    variant-conditioned (corrections item 1): baseline 85/31 plus a [low, high] bound pair over
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
    for stat_key, slot in (
        ("def", NUMERIC_EXPECTED_DEF),
        ("spa", NUMERIC_EXPECTED_SPA),
        ("spd", NUMERIC_EXPECTED_SPD),
        ("spe", NUMERIC_EXPECTED_SPE),
    ):
        value = base.get(stat_key)
        if value:
            _set_numeric(
                num_row, slot, min(1.0, _gen3_stat(value, level, ev=85, iv=31, hp=False) / _ACTUAL_STAT_DIVISOR)
            )
    atk_base = base.get("atk")
    if not atk_base or not hp_base:
        return
    atk_baseline = _gen3_stat(atk_base, level, ev=85, iv=31, hp=False)
    hp_baseline = _gen3_stat(hp_base, level, ev=85, iv=31, hp=True)
    atk_low = atk_high = atk_baseline
    hp_low = hp_high = hp_baseline
    variants = belief.candidate_variants if belief is not None else ()
    if variants:
        atk_values: list[int] = []
        hp_values: list[int] = []
        for variant in variants:
            moves = {
                _normalize_identifier(str(move)) for move in _as_sequence(variant.get("moves"))
            }
            item = _normalize_identifier(str(variant.get("item") or ""))
            has_physical = any(_is_physical_attack(dex, move) for move in moves)
            atk_values.append(
                atk_baseline if has_physical else _gen3_stat(atk_base, level, ev=0, iv=0, hp=False)
            )
            hp_trimmed = "bellydrum" in moves or (
                "substitute" in moves and (bool(moves & {"flail", "reversal"}) or item in _PINCH_BERRIES)
            )
            hp_values.append(
                _gen3_stat(hp_base, level, ev=0, iv=31, hp=True) if hp_trimmed else hp_baseline
            )
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
        if token.damage_fraction:
            _set_numeric(num_row, NUMERIC_TT_DAMAGE_FRACTION, min(1.0, token.damage_fraction))
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
        # Spec v3 change 10: under v3 subtract a folded-in confusion self-hit back out of the
        # move's damage; under v2.2 first_damage IS first.damage_fraction (the frozen field),
        # so the write is byte-identical.
        first_damage = first.damage_fraction
        if schema_v3 and first.confusion_selfhit:
            first_damage = max(0.0, first_damage - first.confusion_selfhit_fraction)
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
        # Spec v3 change 10: the confusion self-hit flag on the (opponent's) move sub-block
        # whose damage was just corrected above.
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
        # Spec v3 change 10 (second-mover mirror; the confused mon is normally SLOWER so this
        # rarely fires, but the correction is symmetric). v2.2 uses the frozen field.
        second_damage = second.damage_fraction
        if schema_v3 and second.confusion_selfhit:
            second_damage = max(0.0, second_damage - second.confusion_selfhit_fraction)
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
        # first sub-block (one per-turn bit; the corrected damage is TM2's above).
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


def _observation_metadata(state: PlayerRelativeBattleState) -> dict[str, Any]:
    return {
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
