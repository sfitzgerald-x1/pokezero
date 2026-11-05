// Layout contract of the D3 device world preparation kernel (V2ePrepWorlds).
//
// gpu_engine/device_public_prep.py defines the same enumerations in the same
// order (its CPU test parses this header). Every table and record is int32
// unless noted; refusal codes index the host string table built by
// gpu_engine/device_public_source_tables.py (0 = no refusal; the static codes
// below come first, dynamic strings follow from R_STATIC_COUNT).
#pragma once
#include <cstdint>

namespace v2e {
namespace prep {

constexpr int kSlots = 6;
constexpr int kMoveSlots = 4;
constexpr int kBoosts = 7;
constexpr int kFeatures = 8;
constexpr int kSeedWords = 8;
constexpr int kMaxUniverses = 1024;
constexpr int kUniverseWords = kMaxUniverses / 32;

// Universe table [U, U_COLUMNS], universes in set_source.universes.values() order.
enum Uni { U_SPECIES = 0, U_VAR_OFF, U_VAR_CNT, U_GENDER, U_COLUMNS };

// Variant table [2, Vn, V_COLUMNS]: case 0 = hidden backline fixture, case 1 = revealed fixture.
enum Var {
  V_LEVEL = 0,
  V_MOVE0, V_MOVE1, V_MOVE2, V_MOVE3,   // raw variant moves (engine ids, 0 = none/unknown): revealed matching
  V_HAS_HP,                             // any raw move normalizes to hiddenpower*
  V_ABILITY, V_ITEM,                    // engine ids (0 = none/unknown)
  V_SPECIES,                            // written species id (hidden case); 0 in the revealed case (view species)
  V_STAT0, V_STAT1, V_STAT2, V_STAT3, V_STAT4, V_STAT5,   // [max hp, atk, def, spa, spd, spe]
  V_WMOVE0, V_WMOVE1, V_WMOVE2, V_WMOVE3,                 // written fixture moves (engine ids)
  V_WNMOVES,                                              // len(fixture.moves)
  V_PP0, V_PP1, V_PP2, V_PP3,
  V_TYPE0, V_TYPE1,
  V_HP_TYPE, V_HP_POWER,                // values after the hidden power write (0 / 70 when no hidden power)
  V_CHOICEBAND,                         // normalize_id(fixture.item) == 'choiceband'
  V_REFUSAL,                            // materialization refusal string index (0 = ok)
  V_FIXTURE_NONE,                       // revealed case: _fixture_from_variant_payload returned None
  V_COLUMNS
};

// Misc table: header scalars, then the volatile table [V], then feature-unavailable strings [kFeatures].
enum Misc {
  MISC_STRUGGLE = 0, MISC_TRUANT, MISC_UNIVERSES, MISC_VARIANTS, MISC_SPECIES, MISC_MOVES, MISC_ABILITIES, MISC_ITEMS,
  MISC_VOLATILES, MISC_HEADER
};
// Volatile table entries: -1 leechseed, -2 ingrain, -3 trapped, otherwise the refusal string index.
constexpr int VOL_LEECH_SEED = -1, VOL_INGRAIN = -2, VOL_TRAPPED = -3;

// mon_features rows passed to the kernel, in this order.
enum Feature { FT_TOXIC_STAGE = 0, FT_TURNS_ACTIVE, FT_STALL_COUNTER, FT_CONFUSION_TURNS, FT_ENCORE_TURNS, FT_SUB_PRESENT, FT_MEANLOOK_TRAP, FT_ITEM_SWAPPED };

// Presence flags (attribute `flags`).
enum Flags {
  F_BORROWED = 1 << 0, F_REMOVED = 1 << 1, F_PP_USED = 1 << 2, F_LAST_MOVES = 1 << 3, F_VOLATILES = 1 << 4,
  F_MON_FEATURES = 1 << 5, F_OWN_MEANLOOK = 1 << 6, F_FIELDS = 1 << 7, F_FEATURE_BASE = 8   // feature f present: 1 << (F_FEATURE_BASE + f)
};

// Kernel input order (RemainingArgs).
enum Input {
  IN_MISC = 0, IN_UNI, IN_VAR, IN_SPECIES_TO_UNIVERSE, IN_MOVE_IS_HP,
  IN_OWNER, IN_SPECIES, IN_PRESENT, IN_ACTIVE, IN_BELIEF, IN_REVEALED_MOVES, IN_ABILITY, IN_ITEM, IN_GENDER, IN_LEVEL,
  IN_ERROR, IN_HP_FRACTION, IN_HP_PERCENT, IN_STATUS, IN_BOOSTS, IN_TYPES, IN_BORROWED, IN_REMOVED, IN_PP_USED,
  IN_LAST_MOVES, IN_VOLATILES, IN_WEATHER, IN_TURN, IN_SPIKES, IN_OWN_ORDER, IN_FEATURES, IN_OWN_MEANLOOK, IN_FIELDS,
  IN_OWN_ACTIVE, IN_OWN_TRAP, IN_OWN_FLAGS, IN_SEEDS, IN_COUNT, IN_COUNT_ARGS
};

// Output records.
enum Mon {
  M_SPECIES = 0, M_LEVEL, M_HP, M_MAX_HP, M_GENDER, M_ABILITY, M_BASE_ABILITY, M_ITEM, M_STATUS, M_STATUS_COUNTER,
  M_HP_TYPE, M_HP_POWER, M_CHOICE_MOVE, M_LAST_MOVE, M_PROTECT_COUNTER, M_TRAP_SOURCE, M_VOLATILE,
  M_TYPE0, M_TYPE1, M_BTYPE0, M_BTYPE1,
  M_MOVE0, M_MOVE1, M_MOVE2, M_MOVE3, M_PP0, M_PP1, M_PP2, M_PP3,
  M_STAT0, M_STAT1, M_STAT2, M_STAT3, M_STAT4, M_STAT5,
  M_BOOST0, M_BOOST1, M_BOOST2, M_BOOST3, M_BOOST4, M_BOOST5, M_BOOST6,
  M_COLUMNS
};
enum Side {
  S_ACTIVE = 0, S_ACTIVE_TURNS, S_LAST_KIND, S_LAST_INDEX, S_SPIKES0, S_SPIKES1, S_WEATHER, S_WEATHER_COUNTER, S_TURN,
  S_OWN_TRAP0, S_OWN_TRAP1, S_OWN_TRAP2, S_OWN_TRAP3, S_OWN_TRAP4, S_OWN_TRAP5,
  S_COLUMNS
};
enum Fix { FX_UNIVERSE = 0, FX_VARIANT, FX_GENDER, FX_SLOT, FX_COLUMNS };
enum Code { C_PLAN = 0, C_MATERIALIZE, C_COLUMNS };

// Static refusal strings (see device_public_source_tables.STATIC_REFUSALS, same order).
enum Refusal {
  R_NONE = 0,
  R_TRACKER_ERROR, R_UNREVEALED_EXPOSES, R_NS_SPECIES, R_NO_SOURCE, R_DUPLICATE_SPECIES, R_BITS_OUTSIDE, R_ZERO_MASS,
  R_NS_MOVES, R_NS_ABILITIES, R_NS_ITEMS, R_ITEM_SWAP_UNAVAILABLE, R_NO_CONSISTENT, R_GENDER_NS,
  R_REVEALED_FIXTURE, R_HIDDEN_SOURCE,
  R_BOUNDARY, R_ENCOUNTER_DIFFER, R_ONE_ACTIVE, R_OWNER_TRAP, R_HP_PERCENT_UNAVAILABLE, R_HP_MALFORMED, R_SLEEP,
  R_ITEM_REMOVED_UNAVAILABLE, R_PP_UNAVAILABLE, R_PP_JOIN, R_CLOCK_CONFUSION, R_CLOCK_ENCORE, R_SUBSTITUTE,
  R_VOLATILE_BAG, R_LAST_MOVES_UNAVAILABLE, R_LAST_CALLER, R_TRUANT, R_FAINT, R_FIELDS, R_WISH, R_WEATHER_NS,
  R_WEATHER_DURATION,
  R_STATIC_COUNT
};

}  // namespace prep
}  // namespace v2e
