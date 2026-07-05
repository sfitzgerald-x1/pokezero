"""Regression guard: every observation feature slot must be exercised by real play.

Plays a fixed set of deterministic seeded games through the PRODUCTION encoding path
(``LocalShowdownEnv.observe`` with the belief set source ON) using a coverage-seeking driver, and
asserts that no numeric feature index is constant and no categorical column is padding-only — except
a small, documented allowlist of structurally-unreachable or situational slots.

This is the test that would have caught the dead belief-move buckets (the encoder never emitted
revealed opponent moves) and the dead candidate-set slots (the RL env never wired the set source):
if a future change silently stops populating an input, this fails and names the dead slot.

The seeds + driver ARE the stored "game": deterministic, so coverage is reproducible. We use the
live env (not a replay fixture) because the batch replay path does not reconstruct candidate-set
beliefs, so it cannot exercise the belief slots this test exists to guard.
"""
from __future__ import annotations
import os
import random
from pathlib import Path
import unittest

from pokezero.local_showdown import LocalShowdownEnv, LocalShowdownConfig

SHOWDOWN_ROOT = Path(
    os.environ.get("POKEZERO_SHOWDOWN_ROOT", "/Users/scott/workspace/pokerena/vendor/pokemon-showdown")
)
SEEDS = tuple(range(1, 13))  # fixed -> reproducible coverage

NUMERIC_LABELS = {
    0: "HP_FRACTION", 1: "ACTIVE", 2: "LEGAL", 3: "PRESENT", 4: "REVEALED_MOVE_COUNT",
    5: "CANDIDATE_SET_COUNT", 6: "UNCERTAINTY", 7: "POSSIBLE_ABILITY_COUNT", 8: "POSSIBLE_ITEM_COUNT",
    9: "POSSIBLE_MOVE_COUNT", 10: "REVEALED_ABILITY", 11: "REVEALED_ITEM", 12: "BASE_POWER",
    13: "PRIORITY", 14: "ACCURACY", 15: "LEVEL", 16: "BASE_HP", 17: "BASE_ATK", 18: "BASE_DEF",
    19: "BASE_SPA", 20: "BASE_SPD", 21: "BASE_SPE", 22: "SELF_HAZARDS", 23: "OPP_HAZARDS",
    24: "SELF_SCREENS", 25: "OPP_SCREENS", 26: "BOOST_ATK", 27: "BOOST_DEF", 28: "BOOST_SPA",
    29: "BOOST_SPD", 30: "BOOST_SPE", 31: "MOVE_PP_FRACTION", 32: "EFFECT_CHANCE", 33: "TURN_COUNT",
    34: "SELF_HP_COST", 35: "SELF_FUTURE_SIGHT", 36: "OPP_FUTURE_SIGHT", 37: "TOXIC_STAGE",
    38: "ACTUAL_HP", 39: "ACTUAL_ATK", 40: "ACTUAL_DEF", 41: "ACTUAL_SPA", 42: "ACTUAL_SPD",
    43: "ACTUAL_SPE",
    # ---- spec v2 (exact-state layer + stats token + transition tokens). ----
    44: "SELF_SLEEP_CLAUSE", 45: "OPP_SLEEP_CLAUSE", 46: "WEATHER_TURNS", 47: "WEATHER_PERMANENT",
    48: "SELF_REFLECT_TURNS", 49: "SELF_LIGHT_SCREEN_TURNS", 50: "SELF_SAFEGUARD_TURNS",
    51: "SELF_MIST_TURNS", 52: "OPP_REFLECT_TURNS", 53: "OPP_LIGHT_SCREEN_TURNS",
    54: "OPP_SAFEGUARD_TURNS", 55: "OPP_MIST_TURNS", 56: "SELF_WISH_PENDING", 57: "OPP_WISH_PENDING",
    58: "SLEEP_TURNS", 59: "REST_SLEEP", 60: "WAKE_KNOWN", 61: "TURNS_ACTIVE", 62: "TRAPPER_ALIVE",
    63: "MON_SWITCHED_BEFORE_ATTACK", 64: "MON_STAYED_AND_ATTACKED", 65: "MON_TURNS_ACTIVE_TOTAL",
    66: "EXPECTED_HP", 67: "EXPECTED_HP_LOW", 68: "EXPECTED_HP_HIGH", 69: "EXPECTED_ATK",
    70: "EXPECTED_ATK_LOW", 71: "EXPECTED_ATK_HIGH", 72: "EXPECTED_DEF", 73: "EXPECTED_SPA",
    74: "EXPECTED_SPD", 75: "EXPECTED_SPE",
    **{76 + i: f"OPP_MOVE_PP[{i}]" for i in range(16)},
    92: "STAT_OPP_SWITCH_COUNT", 93: "STAT_OPP_DECISION_OPPORTUNITIES",
    94: "STAT_BLOCKED_ON_OUR_ATTACK", 95: "STAT_PURSUIT_INTERCEPT_PREDICT",
    96: "STAT_MY_SWITCH_TURNS",
    97: "STAT_WEATHER_RAIN_SET", 98: "STAT_WEATHER_RAIN_ABILITY",
    99: "STAT_WEATHER_SUN_SET", 100: "STAT_WEATHER_SUN_ABILITY",
    101: "STAT_WEATHER_SAND_SET", 102: "STAT_WEATHER_SAND_ABILITY",
    103: "STAT_WEATHER_HAIL_SET", 104: "STAT_WEATHER_HAIL_ABILITY",
    105: "TT_DAMAGE_FRACTION", 106: "TT_N_HITS", 107: "TT_CALLED", 108: "TT_TRANSFORMED",
    109: "TT_CRIT", 110: "TT_MISS", 111: "TT_KO", 112: "TT_PURSUIT_INTERCEPT",
    113: "TT_OWN_SPIKES", 114: "TT_OPP_SPIKES", 115: "TT_ABS_TURN", 116: "TT_TURNS_AGO",
    117: "TT_RESIDUAL", 118: "TT_RESIDUAL_VALID", 119: "TT_CB_BIT", 120: "TT_INVESTMENT_BIT",
    # ---- spec v2.1 (defender identity rides categorical MOVE_PRIORITY; numerics below). ----
    **{121 + i: f"OPP_MOVE_PP_VALID[{i}]" for i in range(16)},
    137: "SUB_HP_FRACTION", 138: "TIER2_CB_PINNED", 139: "TIER2_INVESTMENT_PINNED",
}

def categorical_label(col: int) -> str:
    if col <= 8:
        return {0: "PRIMARY", 1: "SECONDARY", 2: "ROLE", 3: "SLOT", 4: "TYPE_1", 5: "TYPE_2",
                6: "MOVE_CATEGORY", 7: "MOVE_EFFECT", 8: "MOVE_PRIORITY"}[col]
    if col <= 10:
        return f"BELIEF_ABILITY[{col - 9}]"
    if col <= 16:
        return f"BELIEF_ITEM[{col - 11}]"
    if col <= 32:
        return f"BELIEF_MOVE[{col - 17}]"
    return f"VOLATILE[{col - 33}]"

# Slots that a bounded set of games legitimately cannot guarantee, documented so the test stays
# strict everywhere else. Two kinds:
#   * structural  — cannot ever be populated given Gen 3 per-species maxima (the belief buckets are
#                   sized 2 abilities / 6 items / 16 moves, but the real caps are 2 / 5 / 14).
#   * situational — a mechanic not triggered by SEEDS (identical uncovered set at 12 and 24 seeds).
#                   These ARE wired; they are exercised by mechanic-specific tests, not this one.
# The important slots this test exists to guard — every reachable belief-move/-item/-ability bucket,
# the candidate-set counts, and revealed move/ability/item — are NOT allowlisted and ARE verified.
ALLOW_NUMERIC: set[int] = {
    24, 25,   # SELF/OPP_SCREENS   — situational (Reflect/Light Screen users absent from SEEDS)
    35, 36,   # SELF/OPP_FUTURE_SIGHT — situational (Future Sight rare in Gen 3 randbats sets)
    # ---- spec v2 allowances (verified against the 12-seed sweep; everything else covers). ----
    48, 49, 50, 51, 52, 53, 54, 55,  # timed screen/Safeguard/Mist counters — situational
                                     # (same mechanic class as 24/25: no setters in SEEDS)
    87, 88, 89,  # OPP_MOVE_PP[11..13] — situational (needs 12+ occupied belief-move buckets
                 # with a REVEALED move that deep in the sorted order)
    90, 91,      # OPP_MOVE_PP[14..15] — structural (mirrors BELIEF_MOVE[14..15]: 16 buckets,
                 # Gen 3 cap is 14 moves/species)
    95,          # STAT_PURSUIT_INTERCEPT_PREDICT — situational (doubled Pursuit is rare;
                 # exercised by transitions fixtures)
    98,          # STAT_WEATHER_RAIN_ABILITY — situational (Drizzle = Kyogre only in the pool)
    103,         # STAT_WEATHER_HAIL_SET — situational (Hail users rare in gen 3 randbats)
    104,         # STAT_WEATHER_HAIL_ABILITY — structural (no hail ability exists in gen 3)
    108,         # TT_TRANSFORMED — situational (Ditto/Mew only; exercised by unit fixtures)
    112,         # TT_PURSUIT_INTERCEPT — situational (see 95)
    117, 118,    # TT_RESIDUAL / TT_RESIDUAL_VALID — structural BY DESIGN in Tier-1 corpora:
                 # populated only by pokezero.tier2 behind the #505 gate + tier2_residuals mask
    119,         # TT_CB_BIT — structural BY DESIGN in Tier-1 corpora (same channel as 117/118)
    120,         # TT_INVESTMENT_BIT — structural BY DESIGN everywhere: a true always-zero
                 # reserve held for the H3 defender-side/investment inference (carried
                 # forward into the v2.1 census, still constant zero; batch 2 populates it)
    # ---- spec v2.1 allowances (12-seed sweep: validity bits 0..10 and SUB_HP_FRACTION
    # all cover; the deep buckets mirror the OPP_MOVE_PP allowances exactly). ----
    132, 133, 134,  # OPP_MOVE_PP_VALID[11..13] — situational (mirrors 87..89: needs a
                    # REVEALED move that deep in the sorted bucket order)
    135, 136,       # OPP_MOVE_PP_VALID[14..15] — structural (mirrors 90..91: 16 buckets,
                    # Gen 3 cap is 14 moves/species)
    138,            # TIER2_CB_PINNED — situational (the two-strike CB conclusion is rare
                    # in a 12-seed sweep; same channel class as 117..119, exercised by the
                    # tier2 fixtures incl. the CB-Pidgeot game)
    139,            # TIER2_INVESTMENT_PINNED — structural BY DESIGN everywhere: the
                    # per-mon twin of 120, a true always-zero reserve until batch 2
}
ALLOW_CATEGORICAL: set[int] = {
    16,               # BELIEF_ITEM[5]  — structural (6 buckets, Gen 3 cap is 5 items/species)
    31, 32,           # BELIEF_MOVE[14..15] — structural (16 buckets, Gen 3 cap is 14 moves/species)
    34, 35, 36, 37, 38,  # VOLATILE[1..5] — situational (a mon rarely stacks 2+ volatiles at once)
}

_INTERESTING = (
    "dragondance", "swordsdance", "calmmind", "bulkup", "spikes", "reflect", "lightscreen",
    "raindance", "sunnyday", "sandstorm", "toxic", "willowisp", "thunderwave", "leechseed",
    "substitute", "futuresight", "batonpass", "rest", "curse", "agility",
)


def _interesting(name: str | None) -> bool:
    n = "".join(c for c in (name or "").lower() if c.isalnum())
    return any(k in n for k in _INTERESTING)


def _coverage_seeking_action(obs, rng: random.Random) -> int | None:
    mask = obs.legal_action_mask
    legal = [i for i, m in enumerate(mask) if m]
    if not legal:
        return None
    cands = obs.metadata.get("action_candidates", [])
    moves = [i for i in legal if i < len(cands) and cands[i].get("move_name")]
    switches = [i for i in legal if i not in moves]
    interesting = [i for i in moves if _interesting(cands[i].get("move_name"))]
    r = rng.random()
    if interesting and r < 0.6:
        return rng.choice(interesting)
    if switches and r < 0.2:
        return rng.choice(switches)
    return rng.choice(legal)


def _aggregate_coverage() -> tuple[dict[int, set], dict[int, set]]:
    num_values: dict[int, set] = {}
    cat_ids: dict[int, set] = {}
    for seed in SEEDS:
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=str(SHOWDOWN_ROOT), set_belief_source=True))
        try:
            env.reset(seed=seed)
            rng = random.Random(seed)
            for _ in range(400):
                if env.terminal() is not None:
                    break
                requested = env.requested_players()
                if not requested:
                    break
                actions = {}
                for player in requested:
                    obs = env.observe(player)
                    for row in obs.numeric_features:
                        for j, v in enumerate(row):
                            num_values.setdefault(j, set()).add(round(float(v), 4))
                    for row in obs.categorical_ids:
                        for j, v in enumerate(row):
                            cat_ids.setdefault(j, set()).add(v)
                    action = _coverage_seeking_action(obs, rng)
                    if action is not None:
                        actions[player] = action
                if not actions:
                    break
                env.step(actions)
        finally:
            env.close()
    return num_values, cat_ids


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout + node bridge",
)
class InputCoverageTest(unittest.TestCase):
    def test_every_input_slot_is_exercised(self) -> None:
        num_values, cat_ids = _aggregate_coverage()
        self.assertTrue(num_values, "no observations were encoded")
        dead_numeric = [
            f"{j}:{NUMERIC_LABELS.get(j, '?')}"
            for j in sorted(num_values)
            if j not in ALLOW_NUMERIC and len(num_values[j]) < 2
        ]
        dead_categorical = [
            f"{j}:{categorical_label(j)}"
            for j in sorted(cat_ids)
            if j not in ALLOW_CATEGORICAL and len(cat_ids[j]) < 2
        ]
        self.assertEqual(dead_numeric, [], f"constant/unused numeric inputs: {dead_numeric}")
        self.assertEqual(dead_categorical, [], f"padding-only categorical inputs: {dead_categorical}")


if __name__ == "__main__":
    num_values, cat_ids = _aggregate_coverage()
    dead_n = sorted(j for j in num_values if len(num_values[j]) < 2)
    dead_c = sorted(j for j in cat_ids if len(cat_ids[j]) < 2)
    print("UNCOVERED numeric:", [f"{j}:{NUMERIC_LABELS.get(j, chr(63))}" for j in dead_n])
    print("UNCOVERED categorical:", [f"{j}:{categorical_label(j)}" for j in dead_c])
