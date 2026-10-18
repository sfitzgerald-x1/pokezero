"""Main-only authored callbacks and rollout fixtures; no native search or model."""
import hashlib
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.search_over_raw_initial_team_capture import InitialTeamSourceCheck, digest
from pokezero.policy import PolicyDecision
from pokezero.rollout import RolloutConfig, RolloutDriver, RolloutSealedPreStepBoundary
from pokezero.showdown_fixture import FixturePokemon, pack_team
from tests.test_rollout import ScriptedEnv, observation


class FakePublicPolicy:
    """Emit the existing production observer protocol, without a native call."""

    def __init__(self, owner):
        self.owner = owner
        self._config = SimpleNamespace(worlds=1, search_sims=1024, search_batch=16,
            search_depth=6, threads=1, model_world_workers=1, leaf_eval="model",
            model_leaf_override=None, early_stop=False, model_decision_time_ms=None,
            depth_min=None, worlds_min=None, strict_fallbacks=True)
        self._fixed_override = None
        self._world_attempt_observer = self._world_observer = None
        self.stats = SimpleNamespace(worlds_attempted=0, worlds_constructed=0,
                                     worlds_searched=0, total_iterations=0)
        self.calls = 0
        self.missing = self.bad_observer = self.search_failure = False
        self.refused_first = self.short_work = False

    def select_action_with_context(self, context, *, rng):
        self.calls += 1
        self.owner.assertIsNone(self._fixed_override)
        self.owner.assertFalse(hasattr(context, "snapshot"))
        if self.calls == 1:
            self.owner.truth_reader.assert_not_called()
        self.stats.worlds_attempted += 1 + int(self.refused_first)
        self.stats.worlds_constructed += 1
        if not self.missing:
            if self.refused_first:
                self._world_attempt_observer(context, 0, None)
            self._world_attempt_observer(context, int(self.refused_first),
                SimpleNamespace(player_teams={"p1": self.owner.packed, "p2": self.owner.packed}))
            world_context = context if not self.bad_observer else SimpleNamespace(
                seed=context.seed, battle_id=context.battle_id, player_id="p1",
                decision_round_index=context.decision_round_index + 99)
            try:
                self._world_observer(world_context,
                    SimpleNamespace(slot_sides={"p1": "side_one", "p2": "side_two"},
                        party_species={"p1": ["pikachu"], "p2": ["snorlax"]}),
                    SimpleNamespace(to_string=lambda: "authored-state"))
            except ValueError:
                pass  # Production swallows telemetry failures, never redraws.
        if self.search_failure:
            raise RuntimeError("authored selector failure")
        self.stats.worlds_searched += 1
        self.stats.total_iterations += 1023 if self.short_work else 1024
        return PolicyDecision(action_index=0, policy_id="authored-search")


class InitialTeamSourceCaptureTests(unittest.TestCase):
    def setUp(self):
        self.context = SimpleNamespace(seed=7, battle_id="authored", decision_round_index=0,
                                       player_id="p1")
        self.raw = PolicyDecision(action_index=1, policy_id="authored-raw")
        self.packed = pack_team([FixturePokemon(species=species, moves=("Tackle",),
            ability="Pressure", item="Leftovers")
            for species in ("Pikachu", "Snorlax", "Mew", "Lugia", "Blissey", "Celebi")])
        self.emitted = []
        self.check = InitialTeamSourceCheck(source_seed=7,
            publish=lambda kind, index, row: self.emitted.append((kind, index, row)),
            boundary_key=lambda context: "a" * 64)
        self.policy = FakePublicPolicy(self)
        self.truth_patch = patch("pokezero.golden_corpus._true_teams_from_bridge_snapshot",
            return_value={seat: {"packed": self.packed} for seat in ("p1", "p2")})
        self.truth_reader = self.truth_patch.start()
        self.addCleanup(self.truth_patch.stop)

    def context_at(self, index):
        return SimpleNamespace(**(vars(self.context) | {"decision_round_index": index}))

    def boundary(self, context=None, **changes):
        context = context or self.context
        values = dict(seed=context.seed, battle_id=context.battle_id,
            decision_round_index=context.decision_round_index, requested_players=("p1", "p2"),
            snapshot=SimpleNamespace(bridge_snapshot={"authored_round": context.decision_round_index}),
            decisions={"p1": self.raw, "p2": self.raw})
        return RolloutSealedPreStepBoundary(**(values | changes))

    def select(self, context=None):
        return self.check.select(self.policy, context or self.context,
                                 rng=random.Random(0), raw_action=self.raw.action_index)

    def invoke(self, context=None):
        context = context or self.context
        self.select(context)
        self.check.capture_private(self.boundary(context))

    def test_public_call_precedes_truth_restores_hooks_and_preserves_raw_action(self):
        selected = self.select()
        self.truth_reader.assert_not_called()
        self.assertEqual((selected.action_index, self.raw.action_index), (0, 1))
        self.assertEqual(self.policy.calls, 1)
        self.assertIsNone(self.policy._world_attempt_observer)
        self.assertIsNone(self.policy._world_observer)
        self.check.capture_private(self.boundary())
        row = self.check.rows[0]
        self.assertEqual(row["status"], "COMPARED")
        self.assertFalse(row["sides_pooled"])
        self.assertEqual((row["opponent_primary"], row["own_positive_control"]), ("p2", "p1"))
        for seat in ("p1", "p2"):
            self.assertEqual(row["comparisons"][seat]["joint_selected_fields"]["matches"], 6)
        self.assertFalse(row["original_source_transport_validated"])
        self.assertFalse(row["scientific_strength_claim"])

    def test_opening_truth_binds_once_and_duplicate_selection_or_join_refuses(self):
        self.invoke()
        self.invoke(self.context_at(1))
        self.truth_reader.assert_called_once()
        manifest = self.check.manifest()
        self.assertEqual([row["round"] for row in manifest["invocations"]], [0, 1])
        self.assertFalse(manifest["source_truth_sent_to_public"])
        self.assertFalse(manifest["source_play_uses_search"])
        with self.assertRaisesRegex(ValueError, "retry"):
            self.select()
        with self.assertRaisesRegex(ValueError, "one original"):
            self.check.capture_private(self.boundary())
        self.assertEqual(self.policy.calls, 2)

    def test_missing_and_swallowed_callbacks_stay_unknown(self):
        for index, condition in enumerate(("missing", "bad_observer")):
            with self.subTest(condition=condition):
                setattr(self.policy, condition, True)
                self.invoke(self.context_at(index))
                self.assertEqual(self.check.attempts[index]["status"], "UNAVAILABLE")
                self.assertEqual(self.check.rows[index]["comparisons"], {})
                setattr(self.policy, condition, False)
        self.assertEqual(self.policy.calls, 2)

    def test_refused_draw_is_retained_and_only_constructed_draw_is_compared(self):
        self.policy.refused_first = True
        self.invoke()
        captured = self.check.attempts[0]["capture"]
        self.assertEqual(captured["work"]["attempts"], 2)
        self.assertIsNone(captured["attempts"][0]["initial_packed_teams"])
        self.assertEqual(captured["constructed"][0]["attempt_ordinal"], 1)
        self.assertEqual(self.check.rows[0]["status"], "COMPARED")
        self.assertEqual(self.policy.calls, 1)

    def test_selector_failure_publishes_unavailable_attempt_without_truth_or_retry(self):
        self.policy.search_failure = True
        with self.assertRaisesRegex(RuntimeError, "selector failure"):
            self.select()
        self.assertEqual(self.emitted[0][0], "attempt")
        self.assertEqual(self.emitted[0][2]["status"], "UNAVAILABLE")
        self.assertIsNone(self.policy._world_attempt_observer)
        self.assertIsNone(self.policy._world_observer)
        self.truth_reader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "complete source"):
            self.check.manifest()
        with self.assertRaisesRegex(ValueError, "retry"):
            self.select()
        self.assertEqual(self.policy.calls, 1)

    def test_existing_observers_and_fixed_truth_override_refuse_before_work(self):
        for name in ("_world_observer", "_world_attempt_observer", "_fixed_override"):
            with self.subTest(name=name):
                original = Mock()
                setattr(self.policy, name, original)
                with self.assertRaisesRegex(ValueError, "observer boundary"):
                    self.select()
                self.assertIs(getattr(self.policy, name), original)
                setattr(self.policy, name, None)
        self.assertEqual(self.policy.calls, 0)
        self.assertEqual(self.check.attempts, {})

    def test_truth_and_publication_are_copies_and_wrong_source_join_refuses(self):
        self.select()
        with self.assertRaisesRegex(ValueError, "identity/raw action"):
            self.check.capture_private(self.boundary(battle_id="other"))
        with self.assertRaisesRegex(ValueError, "source pre-step"):
            self.check.capture_private(self.boundary(seed=8))
        boundary = self.boundary()
        self.check.capture_private(boundary)
        boundary.snapshot.bridge_snapshot["authored_round"] = 22
        self.assertEqual(self.check.truth["opening_snapshot"], {"authored_round": 0})
        emitted = next(row for kind, _, row in self.emitted if kind == "truth")
        emitted["initial_packed_teams"]["p1"] = "changed"
        self.assertEqual(self.check.truth["initial_packed_teams"]["p1"], self.packed)
        attempt = next(row for kind, _, row in self.emitted if kind == "attempt")
        attempt["capture"]["attempts"].clear()
        self.assertEqual(len(self.check.attempts[0]["capture"]["attempts"]), 1)

    def test_short_production_work_remains_unknown(self):
        self.policy.short_work = True
        self.invoke()
        self.assertEqual(self.check.attempts[0]["work"]["iterations"], 1023)
        self.assertEqual(self.check.attempts[0]["status"], "UNAVAILABLE")
        self.assertEqual(self.check.rows[0]["comparisons"], {})

    def test_rollout_keeps_original_raw_steps_and_uses_only_owned_pre_step_snapshots(self):
        env = ScriptedEnv(requested_sequence=[("p1", "p2")] * 2, terminal_after_steps=2)
        env.default_observation = observation((True, True, False, False, False, False, False, False, False))
        env.snapshot = Mock(side_effect=AssertionError("no general snapshot command"))
        env.snapshot_actionable_boundary = Mock(side_effect=lambda: SimpleNamespace(
            bridge_snapshot={"authored_round": len(env.step_calls)}))
        owner = self

        class RawPolicy:
            policy_id = "authored-raw"

            def select_action(self, observation, *, rng):
                return owner.raw

            def select_action_with_context(self, context, *, rng):
                if context.player_id == "p1":
                    owner.check.select(owner.policy, context, rng=rng, raw_action=owner.raw.action_index)
                return owner.raw

        RolloutDriver(env=env, policies={seat: RawPolicy() for seat in ("p1", "p2")},
            config=RolloutConfig(hide_opponent_legal_action_masks=True,
                sealed_pre_step_sink=self.check.capture_private)).run(seed=7, battle_id="authored")
        self.assertEqual(env.step_calls, [{"p1": 1, "p2": 1}] * 2)
        env.snapshot.assert_not_called()
        self.assertEqual(env.snapshot_actionable_boundary.call_count, 2)
        self.assertEqual(self.policy.calls, 2)
        self.assertEqual([kind for kind, _, _ in self.emitted],
                         ["attempt", "truth", "comparison", "attempt", "comparison"])
        self.assertEqual(len(self.check.manifest()["invocations"]), 2)

    def test_local_digest_matches_original_strict_json_law(self):
        expected = hashlib.sha256(b'{"a":[1,2],"x":3}').hexdigest()
        self.assertEqual(digest({"x": 3, "a": [1, 2]}), expected)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                digest({"invalid": value})

    def test_failed_publication_propagates_after_hook_restoration(self):
        self.check.publish = Mock(side_effect=OSError("durable sink unavailable"))
        with self.assertRaisesRegex(OSError, "durable sink"):
            self.select()
        self.assertIsNone(self.policy._world_attempt_observer)
        self.assertIsNone(self.policy._world_observer)
        self.truth_reader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "retry"):
            self.select()
        self.assertEqual(self.policy.calls, 1)

    def test_late_first_truth_cannot_replace_original_opening(self):
        context = self.context_at(1)
        self.select(context)
        with self.assertRaisesRegex(ValueError, "original opening"):
            self.check.capture_private(self.boundary(context))
        self.truth_reader.assert_not_called()
        self.assertIsNone(self.check.truth)
        self.assertEqual(self.check.rows, {})


if __name__ == "__main__":
    unittest.main()
