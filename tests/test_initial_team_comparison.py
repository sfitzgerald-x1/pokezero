"""Synthetic packed teams only; no simulator, model, source or saved outcomes."""
from dataclasses import replace
import unittest

from pokezero.mcts_eval.search_over_raw_initial_teams import compare_initial_teams
from pokezero.showdown_fixture import FixturePokemon, pack_team


class InitialTeamComparisonTests(unittest.TestCase):
    def setUp(self):
        self.team = [FixturePokemon(species=s, moves=("Tackle", "Protect", "Return", "Hidden Power Ice"),
                                    ability="Pressure", item="Leftovers")
                     for s in ("Pikachu", "Snorlax", "Mew", "Lugia", "Blissey", "Celebi")]

    def compare(self, sampled, truth=None):
        return compare_initial_teams(pack_team(sampled), pack_team(self.team if truth is None else truth))

    def test_team_and_move_order_are_not_mistaken_for_identity_errors(self):
        sampled = [replace(mon, moves=tuple(reversed(mon.moves))) for mon in reversed(self.team)]
        row = self.compare(sampled)
        self.assertEqual(row["species"]["matched_multiset_members"], 6)
        self.assertEqual(row["joint_selected_fields"]["agreement_bounds"], [1., 1.])
        self.assertEqual([r["sampled_index"] for r in row["rows"]], list(reversed(range(6))))
        for flag in ("current_materialization_compared", "consumed_world_validated",
                     "source_truth_validated", "calibration_certified", "scientific_strength_claim"):
            self.assertIs(row[flag], False)

    def test_each_selected_field_mismatch_is_reported_separately(self):
        sampled = list(self.team)
        sampled[0] = replace(sampled[0], moves=("Thunderbolt",), ability="Static", item="Choice Band")
        row = self.compare(sampled)
        for value in row["fields"].values():
            self.assertEqual((value["denominator"], value["matches"], value["mismatches"]), (6, 5, 1))
        self.assertEqual(row["joint_selected_fields"]["mismatches"], 1)

    def test_missing_species_keeps_unpaired_fields_in_the_denominator(self):
        sampled = list(self.team)
        sampled[0] = replace(sampled[0], species="Moltres")
        row = self.compare(sampled)
        self.assertEqual(row["species"]["missing"], {"pikachu": 1})
        self.assertEqual(row["species"]["extra"], {"moltres": 1})
        self.assertEqual(row["species"]["coverage"], 5 / 6)
        for value in row["fields"].values():
            self.assertEqual(value["denominator"], 6)
            self.assertEqual(value["unknown"], 1)
            self.assertEqual(value["agreement_bounds"], [5 / 6, 1.])
        self.assertEqual(row["joint_selected_fields"]["agreement_bounds"], [5 / 6, 5 / 6])

    def test_duplicate_species_never_get_greedily_paired(self):
        sampled = list(self.team)
        sampled[1] = replace(sampled[1], species="Pikachu", item="Choice Band")
        row = self.compare(sampled)
        self.assertEqual(row["rows"][0]["identity_status"], "AMBIGUOUS_SPECIES_IDENTITY")
        self.assertIsNone(row["rows"][0]["sampled_index"])
        self.assertIsNone(row["rows"][0]["joint_selected_fields"])
        truth = list(self.team)
        truth[1] = replace(truth[1], species="Pikachu")
        row = self.compare(sampled, truth)
        self.assertEqual(row["species"]["matched_multiset_members"], 6)
        self.assertEqual(row["fields"]["item"]["unknown"], 2)

    def test_ability_slots_and_omissions_are_unknown_not_equal(self):
        for ability in (None, "0", "1", "H", "S"):
            team = [replace(mon, ability=ability) for mon in self.team]
            row = self.compare(team, team)
            self.assertEqual(row["fields"]["ability"]["agreement_bounds"], [0., 1.])
            self.assertEqual(row["joint_selected_fields"]["agreement_bounds"], [0., 1.])
        no_items = [replace(mon, item=None) for mon in self.team]
        self.assertEqual(self.compare(no_items, no_items)["fields"]["item"]["matches"], 6)

    def test_existing_canonicalizers_preserve_hidden_power_type(self):
        truth = [replace(self.team[0], species="Unown-B")]
        sampled = [replace(truth[0], species="Unown", moves=("tackle", "protect", "return102", "hiddenpowerice70"))]
        row = compare_initial_teams(pack_team(sampled), pack_team(truth), team_size=1)
        self.assertEqual(row["joint_selected_fields"]["matches"], 1)
        sampled[0] = replace(sampled[0], moves=("tackle", "protect", "return", "hiddenpowerfire"))
        row = compare_initial_teams(pack_team(sampled), pack_team(truth), team_size=1)
        self.assertEqual(row["fields"]["moves"]["mismatches"], 1)

    def test_generic_hidden_power_does_not_prove_typed_agreement(self):
        generic = [replace(mon, moves=("Tackle", "Hidden Power")) for mon in self.team]
        typed = [replace(mon, moves=("Tackle", "Hidden Power Ice")) for mon in self.team]
        for sampled, truth in ((generic, generic), (generic, typed), (typed, generic)):
            self.assertEqual(self.compare(sampled, truth)["fields"]["moves"]["unknown"], 6)
        wrong_other = [replace(mon, moves=("Protect", "Hidden Power")) for mon in self.team]
        self.assertEqual(self.compare(wrong_other, typed)["fields"]["moves"]["mismatches"], 6)

    def test_unselected_fields_do_not_become_current_state_fidelity_claims(self):
        sampled = [replace(mon, level=50, nature="Timid", gender="F", evs={"spe": 252}) for mon in self.team]
        row = self.compare(sampled)
        self.assertEqual(row["joint_selected_fields"]["matches"], 6)
        self.assertIn("current_battle_state", row["excluded_fields"])
        self.assertFalse(row["move_slot_order_compared"])

    def test_malformed_or_changed_roster_size_refuses(self):
        for sampled, truth, size in ((None, "bad", 6), ("bad", "bad", 6),
                                    (pack_team(self.team[:5]), pack_team(self.team), 6),
                                    (pack_team(self.team), pack_team(self.team), True)):
            with self.assertRaises(ValueError):
                compare_initial_teams(sampled, truth, team_size=size)
        duplicate_moves = [replace(self.team[0], moves=("Return", "return102"))]
        with self.assertRaises(ValueError):
            compare_initial_teams(pack_team(duplicate_moves), pack_team(self.team[:1]), team_size=1)


if __name__ == "__main__":
    unittest.main()
