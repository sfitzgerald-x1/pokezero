"""Exercise real Struggle through the compiled option, event, and step APIs."""
from __future__ import annotations

import json
import unittest

try:
    import pokezero_search
except ImportError:
    pokezero_search = None

from pokezero.poke_engine_adapter import (
    BattleSpec, MoveSpec, PokemonSpec, SideSpec, build_poke_engine_state,
)


def fixture(*, subject: str = "p1", bench: bool = False):
    attacker = PokemonSpec(
        id="charizard", level=100, types=("fire", "flying"), hp=300,
        maxhp=300, attack=180, defense=140, special_attack=100,
        special_defense=130, speed=200, status="none", ability=None,
        item="choiceband", moves=(MoveSpec(id="tackle", pp=0),
                                  MoveSpec(id="growl", pp=0)),
    )
    opponent = PokemonSpec(
        id="gengar", level=100, types=("ghost", "poison"), hp=300,
        maxhp=300, attack=100, defense=140, special_attack=100,
        special_defense=130, speed=100, status="none", ability=None,
        item=None, moves=(MoveSpec(id="splash", pp=10),),
    )
    own_party = (attacker, opponent) if bench else (attacker,)
    own = SideSpec(pokemon=own_party, volatile_statuses=(), side_conditions={}, boosts={})
    other = SideSpec(pokemon=(opponent,), volatile_statuses=(), side_conditions={}, boosts={})
    sides = (own, other) if subject == "p1" else (other, own)
    state = build_poke_engine_state(BattleSpec(side_one=sides[0], side_two=sides[1]))
    own_names = ["Charizard", "Gengar"] if bench else ["Charizard"]
    names = {subject: own_names, "p2" if subject == "p1" else "p1": ["Gengar"], "turn": 1}
    return state.to_string(), json.dumps(names)


@unittest.skipIf(pokezero_search is None, "fresh native module not available")
class NativeSyntheticStruggleTest(unittest.TestCase):
    def test_options_include_real_struggle_and_switches_on_both_seats(self):
        for subject in ("p1", "p2"):
            with self.subTest(subject=subject):
                state, _ = fixture(subject=subject, bench=True)
                options = json.loads(pokezero_search.env_options(state, True))[subject]
                self.assertIn("struggle", options)
                self.assertTrue(any(option.startswith("switch ") for option in options), options)
                self.assertNotIn("none", options)
                self.assertNotIn("tackle", options)

    def test_branches_render_attack_and_damage_based_recoil_on_both_seats(self):
        for subject in ("p1", "p2"):
            with self.subTest(subject=subject):
                other = "p2" if subject == "p1" else "p1"
                state, context = fixture(subject=subject)
                moves = ("struggle", "splash") if subject == "p1" else ("splash", "struggle")
                report = json.loads(pokezero_search.branch_events(state, *moves, context, True, False))
                self.assertAlmostEqual(sum(b["percentage"] for b in report["branches"]), 100.0)
                self.assertTrue(report["branches"])
                for branch in report["branches"]:
                    self.assertFalse(branch["attribution_unsafe"], branch)
                    self.assertEqual(branch["lossy"], [], branch)
                    self.assertIn(f"|move|{subject}a: Charizard|struggle|{other}a: Gengar", branch["events"])
                    damage = 300 - branch["post"][other]["active_hp"]
                    recoil = 300 - branch["post"][subject]["active_hp"]
                    self.assertGreater(damage, 0, "Struggle must hit a Ghost")
                    self.assertEqual(recoil, max(1, damage // 4))
                    self.assertTrue(branch["turn_completed"])

    def test_seeded_step_keeps_exhausted_real_slots_and_next_struggle_option(self):
        state, context = fixture()
        first = pokezero_search.env_step(state, "struggle", "splash", context, 12345, True)
        self.assertEqual(first, pokezero_search.env_step(state, "struggle", "splash", context, 12345, True))
        post = json.loads(first)["post_state"]
        self.assertIn("struggle", json.loads(pokezero_search.env_options(post, True))["p1"])
        # The serialized real move bank is unchanged; a synthetic attack must
        # not consume another move's PP or overwrite it as a fake move slot.
        import poke_engine
        restored = poke_engine.State.from_string(post)
        original = poke_engine.State.from_string(state)
        self.assertEqual(
            [(m.id, m.pp, m.disabled) for m in restored.side_one.pokemon[0].moves],
            [(m.id, m.pp, m.disabled) for m in original.side_one.pokemon[0].moves],
        )


if __name__ == "__main__":
    unittest.main()
