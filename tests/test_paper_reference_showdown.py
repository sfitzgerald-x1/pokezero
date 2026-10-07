"""Private-safe simulator-adapter contracts, not paper-sampler qualification."""

from copy import deepcopy
import random
from types import SimpleNamespace
import unittest

from pokezero.mcts_eval.paper_reference import Evaluation, ReferenceRefusal, Terminal
from pokezero.mcts_eval.paper_reference_showdown import (
    ShowdownTrajectoryWorld, decision_state,
)


def observation(player="p1"):
    return SimpleNamespace(schema_version="pokezero.observation.v4",
        categorical_ids=((1,),), numeric_features=((.5,),), token_type_ids=(0,),
        attention_mask=(True,), legal_action_mask=(True, True) + (False,) * 8,
        metadata={"self_team": [{"species": "Pikachu", "fainted": False}],
                  "opponent_team": [{"species": "Snorlax", "fainted": False}],
                  "belief_view": {"self_slot": player,
                                  "opponent_slot": "p2" if player == "p1" else "p1"}})


class FakeEnv:
    _search_snapshot_permitted = True
    def __init__(self):
        self.requested = ("p1", "p2")
        self.calls = []
        self.outcome = None

    def terminal(self):
        return None

    def requested_players(self):
        return self.requested

    def observe(self, player):
        return observation(player)

    def legal_actions(self, player):
        return self.observe(player).legal_action_mask

    def reseed_simulator_rng(self, seed):
        self.calls.append(("chance", seed))

    def step(self, actions):
        self.calls.append(("step", actions))
        return SimpleNamespace(terminal=self.outcome)


def evaluator(obs):
    return (0, 1), Evaluation((.4, .6), .3)


class ShowdownAdapterTests(unittest.TestCase):
    def test_hidden_extra_metadata_cannot_change_information_state_key(self):
        own = observation()
        poisoned = deepcopy(own)
        poisoned.metadata["opponent_private_request"] = {"moves": ["thunderbolt"]}
        poisoned.metadata["opponent_team"][0]["moves"] = ["thunderbolt"]
        self.assertEqual(decision_state(own, player="p1"), decision_state(poisoned, player="p1"))

    def test_hidden_field_inside_public_belief_is_refused(self):
        own = observation()
        own.metadata["belief_view"]["opponent_private_request"] = {"moves": ["thunderbolt"]}
        with self.assertRaises(ValueError):
            decision_state(own, player="p1")

    def test_numeric_state_legal_support_and_perspective_change_keys(self):
        own = observation()
        baseline = decision_state(own, player="p1")
        changed = deepcopy(own)
        changed.numeric_features = ((.6,),)
        self.assertNotEqual(baseline.key, decision_state(changed, player="p1").key)
        changed = deepcopy(own)
        changed.legal_action_mask = (True,) + (False,) * 9
        self.assertNotEqual(baseline.key, decision_state(changed, player="p1").key)
        with self.assertRaisesRegex(ReferenceRefusal, "actor-relative"):
            decision_state(own, player="p2")

    def test_faint_count_is_public_not_a_simulator_serialization(self):
        own = observation()
        own.metadata["opponent_team"][0]["fainted"] = True
        self.assertEqual(decision_state(own, player="p1").faint_count, 1)

    def test_live_hidden_environment_is_not_a_sampled_world(self):
        env = FakeEnv()
        env._search_snapshot_permitted = False
        with self.assertRaisesRegex(ReferenceRefusal, "sampled"):
            ShowdownTrajectoryWorld(env, subject="p1", evaluator=evaluator, release=lambda: None)

    def test_opponent_policy_uses_its_own_sampled_observation(self):
        seen = []
        def record(obs):
            seen.append(obs.metadata["belief_view"]["self_slot"])
            return evaluator(obs)
        world = ShowdownTrajectoryWorld(FakeEnv(), subject="p1", evaluator=record, release=lambda: None)
        frame = world.frame()
        self.assertEqual(seen, ["p2"])
        world.evaluate(frame.subject)
        self.assertEqual(seen, ["p2", "p1"])

    def test_owned_sampled_actions_and_chance_go_through_actual_env_api(self):
        env = FakeEnv()
        world = ShowdownTrajectoryWorld(env, subject="p1", evaluator=evaluator, release=lambda: None)
        world.advance("action:1", "action:0", random.Random(7))
        self.assertEqual([row[0] for row in env.calls], ["chance", "step"])
        self.assertEqual(env.calls[1][1], {"p1": 1, "p2": 0})

    def test_asymmetric_boundary_does_not_issue_a_fake_wait(self):
        env = FakeEnv()
        env.requested = ("p2",)
        world = ShowdownTrajectoryWorld(env, subject="p1", evaluator=evaluator, release=lambda: None)
        self.assertIsNone(world.frame().subject)
        world.advance(None, "action:0", random.Random(0))
        self.assertEqual(env.calls[-1][1], {"p2": 0})
        with self.assertRaisesRegex(ReferenceRefusal, "fabricated"):
            world.advance("action:0", "action:0", random.Random(0))

    def test_illegal_action_refuses_before_seeding_or_stepping(self):
        env = FakeEnv()
        world = ShowdownTrajectoryWorld(env, subject="p1", evaluator=evaluator, release=lambda: None)
        with self.assertRaisesRegex(ReferenceRefusal, "illegal"):
            world.advance("action:9", "action:0", random.Random(0))
        self.assertEqual(env.calls, [])

    def test_terminal_is_signed_for_subject_and_caps_refuse(self):
        for player, expected in (("p1", 1), ("p2", -1), (None, 0)):
            env = FakeEnv()
            env.outcome = SimpleNamespace(capped=False, winner=player)
            world = ShowdownTrajectoryWorld(env, subject="p1", evaluator=evaluator, release=lambda: None)
            self.assertEqual(world.advance("action:0", "action:1", random.Random(0)), Terminal(expected))
        env.outcome = SimpleNamespace(capped=True, winner=None)
        with self.assertRaisesRegex(ReferenceRefusal, "cap"):
            world.advance("action:0", "action:1", random.Random(0))

    def test_released_world_cannot_be_reused_or_released_twice(self):
        released = []
        world = ShowdownTrajectoryWorld(FakeEnv(), subject="p1", evaluator=evaluator,
            release=lambda: released.append(True))
        world.close()
        world.close()
        self.assertEqual(released, [True])
        with self.assertRaises(ReferenceRefusal):
            world.frame()


if __name__ == "__main__":
    unittest.main()
