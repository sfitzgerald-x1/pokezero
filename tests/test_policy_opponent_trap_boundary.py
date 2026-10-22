"""Engineering-only native/request contract diagnosis; no historical replay."""
import json
import os
from pathlib import Path
import subprocess
import traceback
import unittest

try:
    import pokezero_search
except ImportError:
    pokezero_search = None

ROOT = Path(__file__).resolve().parents[1]


class TrapBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("POKEZERO_SHOWDOWN_ROOT"), "requires pinned source oracle")
    def test_gen3_public_shadow_tag_differs_from_hidden_trap_rechoice(self):
        output = subprocess.run([os.environ.get("POKEZERO_NODE_BINARY", "node"),
            str(ROOT / "tests/policy_opponent_trap_oracle.cjs"), os.environ["POKEZERO_SHOWDOWN_ROOT"]],
            capture_output=True, text=True, check=True, timeout=30)
        result = json.loads(output.stdout)
        self.assertEqual(result["schema"], "pokezero.authored_trap_request_probe.v1")
        self.assertFalse(result["scientific_measurement"])
        self.assertFalse(result["historical_cause_established"])
        self.assertEqual([(r["case"], r["seat"]) for r in result["rows"]],
            [(kind, seat) for kind in ("shadowtag", "arenatrap", "magnetpull") for seat in ("p1", "p2")])
        for row in result["rows"]:
            with self.subTest(case=row["case"], seat=row["seat"]):
                public = row["case"] == "shadowtag"
                self.assertEqual(row["actual_trapped"], True if public else "hidden")
                self.assertEqual(row["before"].get("trapped", False), public)
                self.assertEqual(row["before"].get("maybeTrapped", False), not public)
                self.assertFalse(row["switch_accepted"])
                self.assertEqual(row["choice_error"], "Can't switch: The active Pokémon is trapped")
                self.assertTrue(row["after"]["trapped"])
                self.assertNotIn("maybeTrapped", row["after"])
                self.assertEqual((row["turn_before"], row["turn_after"]), (1, 1))
                self.assertEqual(row["before"]["moves"], row["after"]["moves"])

    @unittest.skipUnless(pokezero_search is not None and hasattr(pokezero_search, "sampled_policy_request"),
                         "requires fresh native constructor")
    def test_real_constructor_refusal_survives_python_note_without_claiming_historical_cause(self):
        from pokezero.engine_search import _note_policy_opponent_refusal
        source = (ROOT / "rust/pokezero-search/src/test_fixtures/minimal.state").read_text().strip().split("/")
        own = source[0].split("=")
        for position, species in ((1, "BULBASAUR"), (2, "SQUIRTLE")):
            mon = own[0].split(","); mon[0] = species; own[position] = ",".join(mon)
        source[0] = "=".join(own)
        names = ["Charmander", "Bulbasaur", "Squirtle"]
        maximum = {"ember": 40, "tackle": 56}
        for slot in ("p1", "p2"):
            for root in (True, False):
                state = source.copy()
                if slot == "p2": state[0], state[1] = state[1], state[0]
                ordinary = json.loads(pokezero_search.sampled_policy_request(
                    "/".join(state), slot, names, names, maximum, root=root))
                self.assertEqual(ordinary["native_action_indices"], [0, 1, 4, 5])
                # A native-only trapping ability is deliberately NOT a public
                # trap certificate. Do not call this an actual Showdown view.
                opposite = 1 if slot == "p1" else 0
                fields = state[opposite].split(",")
                fields[8] = fields[9] = "SHADOWTAG"; state[opposite] = ",".join(fields)
                with self.assertRaisesRegex(ValueError, "native and private-knowledge legal surfaces differ") as caught:
                    pokezero_search.sampled_policy_request("/".join(state), slot, names, names, maximum, root=root)
                error = caught.exception
                diagnostic = json.loads(error.policy_opponent_diagnostic)
                self.assertEqual(diagnostic["native_action_indices"], [0, 1])
                self.assertEqual(diagnostic["private_action_indices"], [0, 1, 4, 5])
                self.assertEqual((diagnostic["seat"], diagnostic["root"]), (slot, root))
                _note_policy_opponent_refusal(error)
                note, = error.__notes__
                facts = json.loads(note.split(": ", 1)[1])
                self.assertEqual(facts["native_action_indices"], [0, 1])
                self.assertEqual(facts["private_action_indices"], [0, 1, 4, 5])
                self.assertNotIn("node_depth", facts)  # standalone constructor, not a traversal annotation
                for private in names + ["SHADOWTAG", "request_order", "native_options"]:
                    self.assertNotIn(private, note)
                try:
                    raise RuntimeError("authored constructor refused") from error
                except RuntimeError as wrapped:
                    rendered = "".join(traceback.format_exception(wrapped))
                self.assertIn(note, rendered)


if __name__ == "__main__":
    unittest.main()
