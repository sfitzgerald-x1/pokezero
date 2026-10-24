"""Production Protect/absorption reads with explicit Sleep Talk callee lineage.

The former 13-test suite exercised unnamed, merged zero-heal tails. The called-move
marker now separates those histories, so the old single-branch/refusal expectations
are no longer the production contract. These controls retain the actual HP, ability,
callee-order and Protect axes, and verify each named branch and its probability.
They do not relabel historical counters or outcomes. Missing/malformed lineage still
refuses in gen3_sleeptalk_called_identity's direct renderer regressions.
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import pokezero_search
except ImportError:
    pokezero_search = None

from pokezero.poke_engine_adapter import (
    BattleSpec, MoveSpec, PokemonSpec, SideSpec, build_poke_engine_state,
)

CTX = json.dumps({"p1": ["Registeel"], "p2": ["Mantine"], "turn": 1})
PROTECT_LINE = "|-activate|p2a: Mantine|Protect"
WATER_IMMUNE = "|-immune|p2a: Mantine|[from] ability: Water Absorb"
WATER_HEAL = "|-heal|p2a: Mantine|252/252|[from] ability: Water Absorb|[of] p1a: Registeel"
UNCONVERTIBLE = ("tackle", "scratch")
CONVERTIBLE = ("watersport", "tackle")
CONVERTIBLE_LAST = ("tackle", "scratch", "watersport")
RETIRED_COUNTERS = {
    "sleeptalk_called_unidentified:protect_marker_rendered",
    "sleeptalk_called_unidentified:protect_marker_rendered_absorb_headroom",
    "sleeptalk_called_unidentified:protect_marker_rendered_absorb_full_hp",
}


def _sleeper(callees=UNCONVERTIBLE):
    return PokemonSpec(
        id="registeel", level=100, types=("steel",), hp=200, maxhp=200,
        attack=100, defense=100, special_attack=100, special_defense=100, speed=500,
        status="sleep", ability=None, item=None, sleep_turns=3,
        moves=(MoveSpec(id="sleeptalk", pp=16),)
        + tuple(MoveSpec(id=move, pp=32) for move in callees),
    )


def _defender(ability, hp):
    return PokemonSpec(
        id="mantine", level=100, types=("water",), hp=hp, maxhp=252,
        attack=100, defense=100, special_attack=100, special_defense=100, speed=100,
        status="none", ability=ability, item=None, sleep_turns=0,
        moves=(MoveSpec(id="splash", pp=32),),
    )


@unittest.skipIf(pokezero_search is None, "pokezero_search native module not built")
class ProtectMarkerStateReadTests(unittest.TestCase):
    def _branches(self, ability, hp, callees=UNCONVERTIBLE, *, protected=True):
        spec = BattleSpec(
            side_one=SideSpec(pokemon=(_sleeper(callees),),
                volatile_statuses=(), side_conditions={}, boosts={}),
            side_two=SideSpec(pokemon=(_defender(ability, hp),),
                volatile_statuses=("protect",) if protected else (),
                side_conditions={}, boosts={}),
        )
        state = build_poke_engine_state(spec).to_string()
        return json.loads(pokezero_search.branch_events(
            state, "sleeptalk", "splash", CTX, True, True))["branches"]

    def _named(self, ability, hp, callees=UNCONVERTIBLE, *, protected=True):
        branches = self._branches(ability, hp, callees, protected=protected)
        self.assertAlmostEqual(math.fsum(b["percentage"] for b in branches), 100, places=4)
        named = {}
        waking = []
        for branch in branches:
            calls = [line for line in branch["events"]
                     if line.startswith("|move|p1a: Registeel|")
                     and "[from] Sleep Talk" in line]
            self.assertFalse(branch["attribution_unsafe"], branch)
            self.assertEqual(branch["lossy"], [], branch)
            if not calls:
                waking.append(branch)
                self.assertEqual(branch["post"]["p1"]["active_status"], "none", branch)
                continue
            self.assertEqual(len(calls), 1, branch)
            callee = calls[0].split("|")[3]
            self.assertIn(callee, callees, branch)
            self.assertNotIn(callee, named, "called histories must not duplicate")
            self.assertEqual(branch["post"]["p1"]["active_status"], "sleep", branch)
            self.assertAlmostEqual(branch["percentage"], 50 / len(callees), places=4)
            named[callee] = branch
        self.assertEqual(set(named), set(callees), "no callee may disappear or merge")
        self.assertEqual(len(waking), 1)
        self.assertAlmostEqual(waking[0]["percentage"], 50, places=4)
        return named

    def _assert_protected(self, named, hp):
        for branch in named.values():
            self.assertIn(PROTECT_LINE, branch["events"], branch)
            self.assertNotIn(WATER_IMMUNE, branch["events"], branch)
            self.assertNotIn(WATER_HEAL, branch["events"], branch)
            self.assertEqual(branch["post"]["p2"]["active_hp"], hp)

    def test_without_protect_the_same_callees_damage_instead_of_rendering_protect(self):
        branches = self._branches(None, 200, protected=False)
        self.assertAlmostEqual(math.fsum(b['percentage'] for b in branches), 100, places=4)
        called, called_mass = set(), 0.0
        for branch in branches:
            self.assertFalse(branch['attribution_unsafe'], branch)
            self.assertEqual(branch['lossy'], [], branch)
            self.assertNotIn(PROTECT_LINE, branch['events'])
            calls = [line for line in branch['events'] if '[from] Sleep Talk' in line]
            if calls:
                self.assertEqual(len(calls), 1)
                called.add(calls[0].split('|')[3])
                called_mass += branch['percentage']
                self.assertLess(branch['post']['p2']['active_hp'], 200)
        self.assertEqual(called, set(UNCONVERTIBLE))
        self.assertAlmostEqual(called_mass, 50, places=4)

    def test_no_absorb_ability_renders_each_protect_marker(self):
        self._assert_protected(self._named(None, 200), 200)

    def test_absorb_ability_with_headroom_does_not_convert_normal_callees(self):
        self._assert_protected(self._named("waterabsorb", 192), 192)

    def test_named_histories_do_not_redefine_old_ambiguous_counter_series(self):
        for ability in (None, "waterabsorb", "voltabsorb"):
            for hp in (192, 252):
                for branch in self._named(ability, hp).values():
                    self.assertFalse(RETIRED_COUNTERS.intersection(branch["lossy_subcases"]))

    def test_full_hp_absorber_with_unconvertible_callees_renders_both(self):
        self._assert_protected(self._named("waterabsorb", 252), 252)

    def test_full_hp_convertible_callee_is_immune_not_protect(self):
        named = self._named("waterabsorb", 252, CONVERTIBLE)
        self._assert_protected({"tackle": named["tackle"]}, 252)
        water = named["watersport"]
        self.assertIn(WATER_IMMUNE, water["events"])
        self.assertNotIn(PROTECT_LINE, water["events"])
        self.assertNotIn(WATER_HEAL, water["events"])
        self.assertEqual(water["post"]["p2"]["active_hp"], 252)

    def test_convertible_last_callee_keeps_its_own_identity_and_mass(self):
        named = self._named("waterabsorb", 252, CONVERTIBLE_LAST)
        self._assert_protected({m: named[m] for m in UNCONVERTIBLE}, 252)
        self.assertIn(WATER_IMMUNE, named["watersport"]["events"])
        self.assertNotIn(PROTECT_LINE, named["watersport"]["events"])

    def test_one_hp_below_full_has_real_heal_and_full_hp_has_immunity(self):
        for hp in (251, 252):
            named = self._named("waterabsorb", hp, CONVERTIBLE)
            self._assert_protected({"tackle": named["tackle"]}, hp)
            water = named["watersport"]
            self.assertNotIn(PROTECT_LINE, water["events"])
            self.assertEqual(water["post"]["p2"]["active_hp"], 252)
            self.assertIn(WATER_HEAL if hp == 251 else WATER_IMMUNE, water["events"])
            self.assertNotIn(WATER_IMMUNE if hp == 251 else WATER_HEAL, water["events"])

    def test_volt_absorb_does_not_convert_normal_callees_at_either_hp(self):
        for hp in (192, 252):
            self._assert_protected(self._named("voltabsorb", hp), hp)

    def test_volt_absorb_status_gate_has_water_absorb_positive_control(self):
        for move in ("charge", "magnetrise", "electricterrain", "iondeluge", "magneticflux"):
            named = self._named("voltabsorb", 252, (move, "tackle"))
            self._assert_protected({"tackle": named["tackle"]}, 252)
            bypass = named[move]
            self.assertEqual(bypass["post"]["p2"]["active_hp"], 252)
            self.assertFalse(any("ability: Volt Absorb" in line for line in bypass["events"]))
            self.assertNotIn(PROTECT_LINE, bypass["events"])
        water = self._named("waterabsorb", 252, CONVERTIBLE)["watersport"]
        self.assertIn(WATER_IMMUNE, water["events"])
        self.assertNotIn(PROTECT_LINE, water["events"])

    def test_absorption_without_protect_is_named_heal_or_full_hp_immunity(self):
        for hp in (192, 252):
            named = self._named("waterabsorb", hp, ("surf", "watergun"), protected=False)
            for branch in named.values():
                self.assertEqual(branch["post"]["p2"]["active_hp"], 252)
                self.assertNotIn(PROTECT_LINE, branch["events"])
                self.assertIn(WATER_HEAL if hp == 192 else WATER_IMMUNE, branch["events"])
                self.assertNotIn(WATER_IMMUNE if hp == 192 else WATER_HEAL, branch["events"])

    def test_non_absorb_ability_is_not_treated_as_absorption(self):
        for hp in (192, 252):
            self._assert_protected(self._named("flashfire", hp), hp)

    def test_identical_physical_successors_keep_distinct_public_calls(self):
        named = self._named(None, 200)
        self.assertEqual(named["tackle"]["post_state"], named["scratch"]["post_state"])
        self.assertNotEqual(named["tackle"]["events"], named["scratch"]["events"])
        self.assertAlmostEqual(sum(b["percentage"] for b in named.values()), 50)


if __name__ == "__main__":
    unittest.main()
