"""Unit tests for the fidelity differential's pure parts (no live bridge)."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.engine_fidelity import (  # noqa: E402
    TurnFeatures,
    match_branch,
    showdown_turn_features,
)
from pokezero.engine_world import EngineWorldUnsupported, hidden_power_engine_id  # noqa: E402
from pokezero.showdown_fixture import OneTurnFixtureResult  # noqa: E402


def _result(lines: list[str]) -> OneTurnFixtureResult:
    return OneTurnFixtureResult(
        format_id="gen3customgame", seed=1, choices={"p1": "move x", "p2": "move y"},
        protocol_lines=tuple(lines), p1_request=None, p2_request=None, terminal=False,
    )


class ProtocolFeatureTests(unittest.TestCase):
    def test_damage_status_faint_weather_sidestart(self) -> None:
        features = showdown_turn_features(_result([
            "|switch|p1a: Swampert|Swampert, L84|301/301",
            "|switch|p2a: Snorlax|Snorlax, L80|401/401",
            "|-damage|p2a: Snorlax|322/401",
            "|-status|p2a: Snorlax|par",
            "|-damage|p1a: Swampert|240/301",
            "|-sidestart|p1: PokeZero p1|Spikes",
            "|-weather|Sandstorm",
            "|-damage|p1a: Swampert|221/301 tox",
        ]))
        self.assertEqual(features.p1_hp, 221)
        self.assertEqual(features.p2_hp, 322)
        self.assertEqual(features.p2_status, "PARALYZE")
        self.assertEqual(features.weather, "SAND")
        self.assertEqual(features.presence()["p1"], ("spikes",))

    def test_faint_zeroes_hp(self) -> None:
        features = showdown_turn_features(_result([
            "|switch|p1a: Gengar|Gengar, L80|261/261",
            "|switch|p2a: Snorlax|Snorlax, L80|401/401",
            "|-damage|p2a: Snorlax|0 fnt",
            "|faint|p2a: Snorlax",
            "|faint|p1a: Gengar",
        ]))
        self.assertEqual(features.fainted, frozenset({"p1", "p2"}))
        self.assertEqual(features.p1_hp, 0)
        self.assertEqual(features.p2_hp, 0)

    def test_benched_damage_does_not_clobber_active(self) -> None:
        features = showdown_turn_features(_result([
            "|switch|p1a: Swampert|Swampert, L84|301/301",
            "|switch|p2a: Snorlax|Snorlax, L80|401/401",
            "|-damage|p1b: Starmie|100/240",
        ]))
        self.assertEqual(features.p1_hp, 301)


class MatchBranchTests(unittest.TestCase):
    def _branch(self, **kwargs):
        base = dict(p1_hp=280, p2_hp=200, p1_status="NONE", p2_status="NONE",
                    fainted=frozenset(), weather="NONE", side_conditions={})
        base.update(kwargs)
        return {"percentage": 50.0, "features": TurnFeatures(**base), "raw": ""}

    def test_hp_within_roll_tolerance_matches(self) -> None:
        observed = TurnFeatures(p1_hp=280, p2_hp=188, p1_status="NONE", p2_status="NONE",
                                fainted=frozenset(), weather="NONE", side_conditions={})
        row, _ = match_branch(observed, [self._branch()], p1_start_hp=280, p2_start_hp=300)
        self.assertIsNotNone(row)  # dealt 112 vs branch 100: within 16%

    def test_wrong_effect_magnitude_diverges(self) -> None:
        observed = TurnFeatures(p1_hp=280, p2_hp=100, p1_status="NONE", p2_status="NONE",
                                fainted=frozenset(), weather="NONE", side_conditions={})
        row, misses = match_branch(observed, [self._branch()], p1_start_hp=280, p2_start_hp=300)
        self.assertIsNone(row)
        self.assertIn("outside tolerance", misses[0])

    def test_status_must_match_exactly(self) -> None:
        observed = TurnFeatures(p1_hp=280, p2_hp=200, p1_status="NONE", p2_status="TOXIC",
                                fainted=frozenset(), weather="NONE", side_conditions={})
        row, _ = match_branch(observed, [self._branch()], p1_start_hp=280, p2_start_hp=300)
        self.assertIsNone(row)
        row, _ = match_branch(observed, [self._branch(p2_status="TOXIC")], p1_start_hp=280, p2_start_hp=300)
        self.assertIsNotNone(row)


class HiddenPowerIdTests(unittest.TestCase):
    _GRASS_IVS = {"hp": 31, "atk": 30, "def": 31, "spa": 30, "spd": 31, "spe": 31}

    def test_typed_id_gains_gen3_base_power(self) -> None:
        self.assertEqual(hidden_power_engine_id("hiddenpowergrass", self._GRASS_IVS), "hiddenpowergrass70")

    def test_bare_id_derives_type_from_ivs(self) -> None:
        self.assertEqual(hidden_power_engine_id("hiddenpower", self._GRASS_IVS), "hiddenpowergrass70")

    def test_all_31_ivs_are_dark(self) -> None:
        self.assertEqual(hidden_power_engine_id("hiddenpower", None), "hiddenpowerdark70")

    def test_type_iv_mismatch_fails_closed(self) -> None:
        with self.assertRaises(EngineWorldUnsupported) as caught:
            hidden_power_engine_id("hiddenpowerfire", self._GRASS_IVS)
        self.assertEqual(caught.exception.reason, "hidden_power_iv_mismatch")


if __name__ == "__main__":
    unittest.main()
