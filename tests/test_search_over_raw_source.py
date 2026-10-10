import json
import random
from types import SimpleNamespace
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.env import StepResult, TerminalState
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.policy import PolicyDecision
from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest, phase_a_contract
from pokezero.mcts_eval.search_over_raw_source import AuditedRawPolicy, collect_raw_source


SHA = "a" * 64
NAMESPACE = "a267ac75-c45d-476a-8071-d29648191825"


class FakeChampion:
    def __init__(self):
        self.weights_sha256 = SHA
        self.result = SimpleNamespace(model_config=SimpleNamespace(window_size=1))
        self.deterministic = True
        self.exploration_epsilon = 0
        self.sampling_temperature = 1
        self.family_gated_selection = False
        self.history_mask_k = None
        self.forward_fn = self._default_forward
        self.distribution = [.4, .6] + [0.] * (ACTION_COUNT - 2)
        self.action = 1

    def _default_forward(self, _):
        pass

    def reset(self):
        pass

    def select_action(self, observation, *, rng):
        return PolicyDecision(self.action, "fixture", metadata={"policy_distribution": self.distribution})


class SourceEnv:
    def __init__(self, rounds=12, winner="p1"):
        self.rounds = rounds
        self.winner = winner
        self.n = 0
        self.calls = []

    def reset(self, *, seed, format_id):
        self.n = 0
        self.calls = []

    def requested_players(self):
        return ("p1", "p2") if self.n < self.rounds else ()

    def observe(self, seat):
        spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
        # The opponent's observation is distinctly marked. It must never be
        # copied into a p1 public root or its history.
        marker = 989898. if seat == "p2" else float(self.n)
        return PokeZeroObservationV0(
            categorical_ids=tuple((0,) for _ in range(spec.token_count)),
            numeric_features=tuple((marker,) for _ in range(spec.token_count)),
            token_type_ids=(0,) * spec.token_count, attention_mask=(True,) * spec.token_count,
            legal_action_mask=(True, True) + (False,) * (ACTION_COUNT - 2),
            metadata={"belief_view": {"self_slot": seat, "opponent_slot": "p2" if seat == "p1" else "p1",
                "self_pokemon": [], "opponent_pokemon": []},
                "action_candidates": [{"action_index": 0, "kind": "move", "move_id": "tackle"},
                                      {"action_index": 1, "kind": "move", "move_id": "protect"}]})

    def step(self, actions):
        self.calls.append(dict(actions))
        self.n += 1
        return StepResult(observations={s: self.observe(s) for s in ("p1", "p2")},
            rewards={"p1": 0., "p2": 0.}, terminal=self.terminal(), requested_players=self.requested_players())

    def terminal(self):
        return TerminalState(self.winner, self.n) if self.n >= self.rounds else None


class SourceCollectorTest(unittest.TestCase):
    def setUp(self):
        self.contract = phase_a_contract(NAMESPACE, excluded_seeds=[], configurations=[
            SearchConfiguration("raw"), SearchConfiguration("incumbent"),
            SearchConfiguration("reference", workers=20)])
        self.seed = self.contract["panels"]["exploration"]["seeds"][0]

    def collect(self, env, **kwargs):
        policies = {s: AuditedRawPolicy(FakeChampion(), checkpoint_sha256=SHA) for s in ("p1", "p2")}
        return collect_raw_source(self.contract, panel="exploration", source_seed=self.seed,
            env=env, policies=policies, **kwargs)

    def test_actual_rollout_driver_collects_raw_game_and_public_catalog(self):
        env = SourceEnv()
        result = self.collect(env)
        self.assertEqual(result["eligible_requests"], 12)
        self.assertEqual([r["source_request_index"] for r in result["eligible_public_records"]], list(range(12)))
        self.assertEqual(digest(result["eligible_public_records"]), result["eligible_catalog_sha256"])
        catalog = {r["source_request_index"]: r["public_record_sha256"] for r in result["eligible_public_records"]}
        self.assertTrue(all(catalog[r["source_request_index"]] == r["public_record_sha256"] for r in result["roots"]))
        self.assertEqual(result["verified_raw_decisions"], 24)
        self.assertEqual(len(result["roots"]), 7)
        self.assertEqual(result["missing_root_ids"], [])
        self.assertTrue(all(actions == {"p1": 1, "p2": 1} for actions in env.calls))
        self.assertFalse(result["search_invoked"])
        self.assertNotIn("989898", json.dumps(result))
        self.assertNotIn("snapshot", json.dumps(result))

    def test_source_winner_does_not_affect_selection_or_public_roots(self):
        won, lost = self.collect(SourceEnv(winner="p1")), self.collect(SourceEnv(winner="p2"))
        self.assertEqual(won["roots"], lost["roots"])
        self.assertEqual(won["eligible_catalog_sha256"], lost["eligible_catalog_sha256"])

    def test_excluded_nonopening_rule_does_not_change_default_panel(self):
        self.contract["exclude_opening_requests"] = True
        result = self.collect(SourceEnv(rounds=12))
        self.assertEqual(result["eligible_requests"], 11)
        self.assertTrue(all(root["source_request_index"] > 0 for root in result["roots"]))

    def test_sealed_source_hook_is_explicit_and_not_exported(self):
        env = SourceEnv(rounds=2)
        private = object()
        env.snapshot_actionable_boundary = lambda: private
        boundaries = []
        result = self.collect(env, sealed_pre_step_sink=boundaries.append)
        self.assertEqual(len(boundaries), 2)
        self.assertTrue(all(b.snapshot is private for b in boundaries))
        self.assertNotIn("snapshot", json.dumps(result))

    def test_context_sink_does_not_give_context_to_champion(self):
        from pokezero.policy import PolicyContext
        from pokezero.trajectory import BattleTrajectory
        seen = []
        policy = AuditedRawPolicy(FakeChampion(), checkpoint_sha256=SHA,
            public_context_sink=lambda context, action: seen.append((context, action)))
        env = SourceEnv()
        own, other = env.observe("p1"), env.observe("p2")
        context = PolicyContext("p1", 0, "fixture", "gen3randombattle", 1,
            own, ("p1", "p2"), BattleTrajectory("fixture", "gen3randombattle", 1),
            requested_observations={"p1": own, "p2": other})
        decision = policy.select_action_with_context(context, rng=random.Random(0))
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(seen[0][1], 1)
        self.assertEqual(set(seen[0][0].requested_observations), {"p1"})

    def test_short_game_keeps_missing_registered_slots(self):
        result = self.collect(SourceEnv(rounds=2))
        self.assertEqual(len(result["roots"]), 2)
        self.assertEqual(len(result["missing_root_ids"]), 5)
        self.assertEqual(result["replacement_seeds"], [])

    def test_capped_game_is_not_a_replacement_catalog(self):
        result = self.collect(SourceEnv(rounds=12), max_decision_rounds=2)
        self.assertEqual(result["status"], "UNCERTAIN_SOURCE")
        self.assertEqual(result["roots"], [])
        self.assertEqual(len(result["missing_root_ids"]), 7)

    def test_unregistered_seed_fails_before_reset(self):
        env = SourceEnv()
        with self.assertRaisesRegex(ValueError, "unregistered"):
            collect_raw_source(self.contract, panel="exploration", source_seed=7, env=env, policies={})
        self.assertEqual(env.calls, [])

    def test_modified_checkpoint_or_selection_law_is_rejected(self):
        policy = FakeChampion()
        with self.assertRaisesRegex(ValueError, "binding"):
            AuditedRawPolicy(policy, checkpoint_sha256="b" * 64)
        policy.result.model_config.window_size = 2
        with self.assertRaisesRegex(ValueError, "one-snapshot"):
            AuditedRawPolicy(policy, checkpoint_sha256=SHA)

    def test_non_argmax_and_illegal_probability_mass_are_rejected(self):
        env = SourceEnv()
        policy = FakeChampion()
        audited = AuditedRawPolicy(policy, checkpoint_sha256=SHA)
        policy.action = 0
        with self.assertRaisesRegex(ValueError, "differs"):
            audited.select_action(env.observe("p1"), rng=random.Random(0))
        policy.action = 1
        policy.distribution[-1] = .1
        with self.assertRaisesRegex(ValueError, "distribution"):
            audited.select_action(env.observe("p1"), rng=random.Random(0))


if __name__ == "__main__":
    unittest.main()
