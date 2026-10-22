"""Direct Baton continuation fixtures, not full-game or coverage qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE = os.environ.get("NODE") or shutil.which("node")
SHOWDOWN = os.environ.get("POKEZERO_SHOWDOWN_ROOT")


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated spike JAX")
class BatonTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import baton, combat, effects, entry, items, mechanics, pursuit, transition
        from gpu_engine.state import empty_state, LEECH_SEED
        self.jax, self.j, self.tr, self.m = jax, j, transition, mechanics
        self.empty, self.seeded = empty_state, LEECH_SEED
        self.names = ["_none", "batonpass", "pursuit", "splash", "struggle", "protect", "endure", "thunderwave"]
        types = ["_none", "normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark"]
        meta = {"vocabulary": {"moves": self.names, "types": types,
            "abilities": ["_none", "shadowtag", "naturalcure", "intimidate", "pressure"],
            "items": ["_none", "lumberry", "choiceband"], "species": ["_none"]}}
        n = len(self.names)
        self.table = transition.MoveTable(j.array([0, 0, 40, 0, 50, 0, 0, 0]),
            j.array([0, 1, 17, 1, 0, 1, 1, 13]), j.zeros(n, bool), j.full(n, 100),
            j.ones(n, bool), j.zeros(n, j.int32), j.ones((18, 18), j.int32),
            j.ones((18, 18), j.int32), j.array(4), effects=effects.from_metadata(meta),
            entry=entry.from_metadata(meta), items=items.from_metadata(meta), combat=combat.from_metadata(meta),
            pursuit=pursuit.from_metadata(meta), baton=baton.from_metadata(meta))

    def live(self, batch=1, own="batonpass", foe="pursuit", fast=True):
        j = self.j; s = self.empty(batch, seed=71)
        return s._replace(hp=j.full_like(s.hp, 1000), max_hp=j.full_like(s.hp, 1000),
            stats=j.full_like(s.stats, 200).at[:, 0, :, 5].set(300 if fast else 100),
            level=j.full_like(s.level, 73), pp=j.full_like(s.pp, 10),
            moves=j.full_like(s.moves, 3).at[:, 0, 0, 0].set(self.names.index(own)).at[:, 1, 0, 0].set(self.names.index(foe)),
            types=s.types.at[..., 0].set(1), done=j.zeros_like(s.done))

    def play(self, s, actions=None):
        return self.tr.step(s, self.j.zeros(s.active.shape, self.j.int32) if actions is None else self.j.array(actions), self.table)

    def test_fast_baton_pauses_pursuit_then_targets_recipient_once_at_ordinary_power(self):
        j = self.j; s = self.live(); paused = self.play(s).state
        self.assertEqual((paused.forced_switch.tolist(), paused.baton_pass.tolist(), int(paused.queued_actor[0])),
            ([[True, False]], [[True, False]], 1))
        self.assertEqual(paused.pp[0, :, 0, 0].tolist(), [9, 10])
        self.assertEqual(paused.hp[0, :, 0].tolist(), [1000, 1000])
        resumed = self.play(paused, [[4, 3]]).state
        ordinary = self.play(self.live(own="splash")).state
        self.assertEqual(int(resumed.hp[0, 0, 1]), int(ordinary.hp[0, 0, 0]))
        self.assertEqual((int(resumed.hp[0, 0, 0]), int(resumed.pp[0, 1, 0, 0]), int(resumed.queued_actor[0])), (1000, 9, -1))
        self.assertEqual((int(resumed.turn[0]), resumed.active_turns.tolist(), bool(resumed.pending_residual[0])), (1, [[0, 1]], False))
        self.assertTrue(bool(j.array_equal(paused.keys, resumed.keys)))
        self.assertFalse(bool(j.any(resumed.baton_pass)))

    def test_slow_baton_does_not_replay_already_executed_pursuit(self):
        paused = self.play(self.live(fast=False)).state
        self.assertLess(int(paused.hp[0, 0, 0]), 1000)
        self.assertEqual((int(paused.queued_actor[0]), int(paused.pp[0, 1, 0, 0])), (-1, 9))
        resumed = self.play(paused, [[4, 0]]).state
        self.assertEqual((int(resumed.hp[0, 0, 1]), int(resumed.pp[0, 1, 0, 0]), int(resumed.turn[0])), (1000, 9, 1))

    def test_both_baton_actions_pause_separately_then_finish_original_turn(self):
        j = self.j; first = self.play(self.live(foe="batonpass")).state
        middle = self.play(first, [[4, 0]]).state
        self.assertEqual((middle.active.tolist(), middle.baton_pass.tolist(), int(middle.queued_actor[0])),
            ([[1, 0]], [[False, True]], -1))
        self.assertTrue(bool(middle.pending_residual[0]))
        last = self.play(middle, [[0, 4]]).state
        self.assertEqual((last.active.tolist(), int(last.turn[0]), bool(last.pending_residual[0])), ([[1, 1]], 1, False))
        self.assertEqual(last.pp[0, :, 0, 0].tolist(), [9, 9])
        self.assertTrue(bool(j.array_equal(first.keys, last.keys)))

    def test_copy_supported_fields_not_recipient_owned_nonvolatiles_or_nocopy_flags(self):
        j = self.j; s = self.live(foe="splash")
        stages = j.array([2, -1, 3, 1, 2, -2, 1])
        s = s._replace(boosts=s.boosts.at[0, 0, 0].set(stages),
            volatile=s.volatile.at[0, 0, 0].set(self.seeded | 2),
            substitute_hp=s.substitute_hp.at[0, 0, 0].set(100),
            status=s.status.at[0, 0, 1].set(self.m.PSN), hp=s.hp.at[0, 0, 1].set(800),
            item=s.item.at[0, 0, 1].set(2))
        paused = self.play(s).state
        # Confusion is installed on the suspended outgoing mon to isolate copy
        # from the ordinary before-move random self-hit gate.
        paused = paused._replace(confusion_counter=paused.confusion_counter.at[0, 0, 0].set(3))
        resumed = self.play(paused, [[4, 0]]).state
        self.assertEqual(resumed.boosts[0, 0, 1].tolist(), stages.tolist())
        self.assertEqual((int(resumed.substitute_hp[0, 0, 1]), int(resumed.confusion_counter[0, 0, 1]),
            int(resumed.volatile[0, 0, 1])), (100, 3, self.seeded))
        self.assertEqual((int(resumed.status[0, 0, 1]), int(resumed.item[0, 0, 1]), int(resumed.choice_move[0, 0, 1])), (self.m.PSN, 2, 0))
        self.assertTrue(bool(j.all(resumed.boosts[0, 0, 0] == 0)))
        self.assertEqual(int(resumed.hp[0, 0, 1]), 675)  # Poison, not outgoing HP.

    def test_trapped_actor_can_pass_but_no_bench_and_failed_before_move_do_not_pause(self):
        s = self.live(foe="splash"); s = s._replace(trapped=self.j.array([[True, False]]), ability=s.ability.at[0, 1, 0].set(1))
        paused = self.play(s).state
        self.assertTrue(bool(self.tr.action_mask(paused, self.table)[0, 0, 4]))
        self.assertEqual(int(self.play(paused, [[4, 0]]).state.active[0, 0]), 1)
        no_bench = self.live(foe="splash"); no_bench = no_bench._replace(hp=no_bench.hp.at[0, 0, 1:].set(0))
        result = self.play(no_bench).state
        self.assertEqual((bool(result.pending_residual[0]), int(result.pp[0, 0, 0, 0])), (False, 9))
        asleep = self.live(foe="splash"); asleep = asleep._replace(status=asleep.status.at[0, 0, 0].set(self.m.SLP), status_counter=asleep.status_counter.at[0, 0, 0].set(3))
        result = self.play(asleep).state
        self.assertEqual((bool(result.pending_residual[0]), int(result.pp[0, 0, 0, 0])), (False, 10))

    def test_hazard_ko_cancels_queued_move_without_copying_again(self):
        s = self.live(); s = s._replace(hp=s.hp.at[0, 0, 1].set(1), spikes=s.spikes.at[0, 0].set(3), boosts=s.boosts.at[0, 0, 0, 0].set(2))
        paused = self.play(s).state; middle = self.play(paused, [[4, 0]]).state
        self.assertEqual((int(middle.hp[0, 0, 1]), int(middle.queued_actor[0]), bool(middle.pending_residual[0])), (0, -1, True))
        self.assertFalse(bool(self.j.any(middle.baton_pass)))
        result = self.play(middle, [[5, 0]]).state
        self.assertEqual((int(result.hp[0, 0, 2]), int(result.pp[0, 1, 0, 0]), int(result.boosts[0, 0, 2, 0])), (750, 10, 0))

    def test_saved_last_protect_and_endure_fail_and_lum_runs_after_saved_status(self):
        for foe in ("protect", "endure"):
            paused = self.play(self.live(foe=foe)).state
            resumed = self.play(paused, [[4, 0]]).state
            self.assertEqual((int(resumed.protect_counter[0, 1, 0]), int(resumed.pp[0, 1, 0, 0])), (0, 9))
        s = self.live(foe="thunderwave"); s = s._replace(item=s.item.at[0, 0, 1].set(1))
        result = self.play(self.play(s).state, [[4, 0]]).state
        self.assertEqual((int(result.status[0, 0, 1]), int(result.item[0, 0, 1])), (0, 0))

    def test_resumed_move_ko_waits_for_new_replacement_without_replay(self):
        s = self.live(); s = s._replace(hp=s.hp.at[0, 0, 1].set(1))
        middle = self.play(self.play(s).state, [[4, 0]]).state
        self.assertEqual((int(middle.hp[0, 0, 1]), int(middle.queued_actor[0]), bool(middle.pending_residual[0])), (0, -1, True))
        result = self.play(middle, [[5, 0]]).state
        self.assertEqual((int(result.hp[0, 0, 2]), int(result.pp[0, 1, 0, 0]), int(result.turn[0])), (1000, 9, 1))

    def test_jit_batch_prefix_and_invalid_pending_identity(self):
        j = self.j; s = self.live(3); paused = self.play(s).state
        actions = j.array([[4, 0], [4, 0], [4, 0]])
        eager = self.tr.step(paused, actions, self.table)
        compiled = self.jax.jit(self.tr.step)(paused, actions, self.table)
        one = self.tr.step(self.jax.tree.map(lambda x: x[:1], paused), actions[:1], self.table)
        for a, b, c in zip(eager.state, compiled.state, one.state):
            self.assertTrue(bool(j.array_equal(a, b))); self.assertTrue(bool(j.array_equal(a[:1], c)))
        frozen = self.play(paused, [[0, 0], [0, 0], [0, 0]])
        self.assertTrue(bool(j.all(frozen.invalid_actions)))
        for a, b in zip(paused, frozen.state): self.assertTrue(bool(j.array_equal(a, b)))

    @unittest.skipUnless(NODE and SHOWDOWN, "requires pinned source oracle")
    def test_pinned_direct_baton_suspends_queue_and_pursuit_is_not_intercepted(self):
        p = subprocess.run([NODE, str(Path(__file__).with_name("gpu_baton_oracle.cjs")), str(Path(SHOWDOWN).resolve())], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr); data = json.loads(p.stdout)
        fast, slow = data["fast"], data["slow"]
        self.assertEqual(fast["before"], dict(request="switch", hits=[], pp=[63, 32]))
        self.assertEqual(fast["after"], dict(request="move", hits=[dict(source="p2", target="ditto", power=40)], oldHP=1000, newHP=960, pp=31))
        self.assertEqual(slow["after"], dict(request="move", hits=[dict(source="p2", target="mew", power=40)], oldHP=960, newHP=1000, pp=31))
        self.assertEqual(data["twice"]["middle"], dict(request="switch", active=["ditto", "mew"]))
        self.assertEqual(data["twice"]["after"], dict(request="move", active=["ditto", "ditto"], turn=2))
        self.assertEqual(data["copied"], dict(atk=2, spd=1, sub=100, confusion=3,
            seed=True, flash=False, choice=False, status="psn", item="choiceband", hp=550))


if __name__ == "__main__": unittest.main()
