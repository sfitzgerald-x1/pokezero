"""The foul-play capture must record abilities, not just species and moves.

trait_extract resolves ability-gated traits via GameParse.ability_of(), which looks the species
up in the captured movesets and returns "" when the field is absent. A missing ability therefore
does not raise -- it silently reports every carrier denominator as zero, which reads in the report
as "no data" rather than as a bug. This pins the field so that cannot recur.
"""
import importlib.util
import json
import os
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "scripts", "trait_foulplay.py")
_spec = importlib.util.spec_from_file_location("trait_foulplay", _SRC)
TF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TF)


def _request(seat_mons):
    return "|request|" + json.dumps({"side": {"pokemon": seat_mons}})


class MovesetAbilityCapture(unittest.TestCase):
    def test_ability_is_captured(self):
        rh = [("p1", _request([
            {"details": "Zapdos, L77", "moves": ["thunderbolt"], "baseAbility": "pressure"},
        ]))]
        ms = TF._movesets(rh)
        self.assertEqual(ms["p1"][0]["ability"], "pressure",
                         "ability must be captured or every carrier denominator reads zero")

    def test_falls_back_to_ability_key(self):
        rh = [("p1", _request([
            {"details": "Muk, L83", "moves": ["toxic"], "ability": "stickyhold"},
        ]))]
        self.assertEqual(TF._movesets(rh)["p1"][0]["ability"], "stickyhold")

    def test_missing_ability_is_empty_string_not_absent(self):
        # ability_of() expects a string; a missing KEY is the failure mode we are guarding against.
        rh = [("p1", _request([{"details": "Ditto, L90", "moves": ["transform"]}]))]
        entry = TF._movesets(rh)["p1"][0]
        self.assertIn("ability", entry)
        self.assertEqual(entry["ability"], "")

    def test_species_and_moves_still_captured(self):
        rh = [("p1", _request([
            {"details": "Snorlax, L74, M", "moves": ["bodyslam", "rest"], "baseAbility": "immunity"},
        ]))]
        e = TF._movesets(rh)["p1"][0]
        self.assertEqual(e["species"], "Snorlax")
        self.assertEqual(e["moves"], ["bodyslam", "rest"])
        self.assertEqual(e["ability"], "immunity")


if __name__ == "__main__":
    unittest.main()
