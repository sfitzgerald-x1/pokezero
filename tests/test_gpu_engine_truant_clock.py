"""Finite Gen3 clock boundaries, not full ability or recharge qualification.

The accompanying live Showdown trace defines the expectations. JAX has no
mustrecharge implementation: that native-only boundary is deliberately absent.
Trace/Skill Swap and root phase inference are outside this repair.
"""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated JAX")
class TruantClockTests(unittest.TestCase):
    def setUp(self):
        import jax.numpy as j
        from gpu_engine import effects, entry, transition, mechanics, pursuit
        from gpu_engine.state import empty_state
        self.j, self.t, self.m, self.empty = j, transition, mechanics, empty_state
        meta = {"vocabulary": {
            "moves": ["_none", "splash", "slackoff", "struggle", "pursuit"],
            "abilities": ["_none", "truant"],
            "types": ["_none", "normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost",
                "steel", "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark"],
            "items": ["_none"], "species": ["_none", "slaking"],
        }}
        self.table = transition.MoveTable(
            j.array([0, 0, 0, 50, 40]), j.array([0, 1, 1, 0, 17]),
            j.array([False, False, False, True, False]), j.full(5, 100),
            j.ones(5, bool), j.zeros(5, j.int32), j.ones((18, 18), j.int32),
            j.ones((18, 18), j.int32), j.array(3),
            effects=effects.from_metadata(meta), entry=entry.from_metadata(meta),
            pursuit=pursuit.from_metadata(meta))

    def live(self):
        j = self.j
        s = self.empty(1, seed=71)
        return s._replace(hp=j.full_like(s.hp, 1000).at[0, 0, 0].set(100),
            max_hp=j.full_like(s.max_hp, 1000), level=j.full_like(s.level, 100),
            stats=j.full_like(s.stats, 200).at[:, 0, :, 5].set(301),
            pp=j.full_like(s.pp, 8), moves=j.ones_like(s.moves).at[0, 0, 0, 0].set(2),
            ability=s.ability.at[0, 0, 0].set(1),
            types=s.types.at[..., 0].set(1), done=j.zeros_like(s.done),
            entry_pending=j.ones_like(s.entry_pending))

    def play(self, s, actions=((0, 0),)):
        result = self.t.step(s, self.j.array(actions), self.table)
        self.assertFalse(bool(result.invalid_actions[0]))
        return result.state

    def test_startup_slackoff_and_clock_do_not_depend_on_a_damaging_hit(self):
        s = self.live()
        for hp, pp, loaf in ((600, 7, True), (600, 7, False),
                             (1000, 6, True), (1000, 6, False)):
            s = self.play(s)
            self.assertEqual((int(s.hp[0, 0, 0]), int(s.pp[0, 0, 0, 0]),
                              bool(s.truant_loaf[0, 0, 0])), (hp, pp, loaf))

    def test_sleep_advances_above_truant_and_confusion_does_not(self):
        s = self.live()._replace(entry_pending=self.j.zeros((1, 2), bool))
        s = s._replace(status=s.status.at[0, 0, 0].set(self.m.SLP),
            status_counter=s.status_counter.at[0, 0, 0].set(4),
            truant_loaf=s.truant_loaf.at[0, 0, 0].set(True),
            confusion_counter=s.confusion_counter.at[0, 0, 0].set(3))
        s = self.play(s)
        self.assertEqual((int(s.status_counter[0, 0, 0]), int(s.confusion_counter[0, 0, 0]),
                          int(s.pp[0, 0, 0, 0]), bool(s.truant_loaf[0, 0, 0])), (3, 3, 8, False))

    def test_full_paralysis_does_not_stop_residual_clock(self):
        s = self.live()._replace(status=self.live().status.at[0, 0, 0].set(self.m.PAR))
        # This tests the fixed supplied-roll gate, not a sampled game outcome.
        before = self.m.before_move(self.m.PAR, 0, paralysis_roll=0)
        self.assertFalse(bool(before.can_act))
        s = self.play(s)
        self.assertTrue(bool(s.truant_loaf[0, 0, 0]))
        s = self.play(s)
        self.assertFalse(bool(s.truant_loaf[0, 0, 0]))

    def replacement(self, *, pending):
        s = self.live()._replace(entry_pending=self.j.zeros((1, 2), bool),
            turn=self.j.array([1]), pending_residual=self.j.array([pending]))
        return s._replace(hp=s.hp.at[0, 0, 0].set(0).at[0, 0, 1].set(100),
            ability=s.ability.at[0, 0, 1].set(1),
            moves=s.moves.at[0, 0, 1, 0].set(2))

    def test_midturn_replacement_gets_pending_residual_and_first_move_acts(self):
        s = self.play(self.replacement(pending=True), ((4, 0),))
        self.assertFalse(bool(s.truant_loaf[0, 0, 1]))
        s = self.play(s)
        self.assertEqual((int(s.hp[0, 0, 1]), int(s.pp[0, 0, 1, 0])), (600, 7))

    def test_postresidual_replacement_misses_tick_and_first_move_loafs(self):
        s = self.play(self.replacement(pending=False), ((4, 0),))
        self.assertTrue(bool(s.truant_loaf[0, 0, 1]))
        s = self.play(s)
        self.assertEqual((int(s.hp[0, 0, 1]), int(s.pp[0, 0, 1, 0])), (100, 8))
        self.assertFalse(bool(s.truant_loaf[0, 0, 1]))

    def test_ordinary_first_turn_switch_is_not_startup(self):
        s = self.live()._replace(ability=self.live().ability.at[0, 0, 0].set(0).at[0, 0, 1].set(1),
            hp=self.live().hp.at[0, 0, 1].set(100),
            moves=self.live().moves.at[0, 0, 1, 0].set(2))
        s = self.play(s, ((4, 0),))
        self.assertFalse(bool(s.truant_loaf[0, 0, 1]))
        s = self.play(s)
        self.assertEqual(int(s.hp[0, 0, 1]), 600)

    def test_pursuit_registration_excludes_loaf_but_keeps_act_interception(self):
        for loaf in (False, True):
            s = self.live()._replace(entry_pending=self.j.zeros((1, 2), bool))
            s = s._replace(moves=s.moves.at[0, 0, 0, 0].set(4),
                truant_loaf=s.truant_loaf.at[0, 0, 0].set(loaf),
                status=s.status.at[0, 0, 0].set(self.m.PAR),
                confusion_counter=s.confusion_counter.at[0, 0, 0].set(3))
            s = self.play(s, ((0, 4),))
            self.assertEqual(int(s.pp[0, 0, 0, 0]), 8 if loaf else 7)
            self.assertEqual(int(s.hp[0, 1, 1]), 1000)
            self.assertEqual(int(s.confusion_counter[0, 0, 0]), 3)
            self.assertEqual(bool(s.truant_loaf[0, 0, 0]), not loaf)
            if loaf:
                self.assertEqual(int(s.hp[0, 1, 0]), 1000)
            else:
                self.assertLess(int(s.hp[0, 1, 0]), 1000)


if __name__ == "__main__":
    unittest.main()
