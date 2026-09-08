from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.engine_search import opponent_request_order
from pokezero.mcts_eval.lattice import _LiveEngineTimingDecider, time_lattice_cell
from pokezero.mcts_eval.manifest import SearchConfig
from pokezero.mcts_eval.resolver import CheckpointContract
from tests.test_mcts_eval_timing_corpus import _record
from pokezero.public_decision_corpus import PublicActionIdentifier, PublicResolvedActionRound
from pokezero.public_replay_materializer import replay_public_action_rounds

C = CheckpointContract(checkpoint_path="/c.pt", checkpoint_sha256="a"*64, policy_id="p",
    schema_version="pokezero.observation.v3", token_count=87, categorical_feature_count=51,
    numeric_feature_count=155, transition_token_count=64, category_vocab=("alpha", "beta"),
    architecture={}, feature_masks={},
    model_device="cpu")

class T(unittest.TestCase):
    def test_gate_failure_is_a_result_with_partial_sample(self):
        recs = [_record(i) for i in range(200)]
        slow = lambda rec, cfg: {"max_depth_reached": 3, "root_action": "move 1"}
        import time as _t
        def slow_decide(rec, cfg):
            _t.sleep(0.001)
            return {"max_depth_reached": 3, "root_action": "move 1"}
        row = time_lattice_cell(SearchConfig(depth=10, sims=8192), records=recs,
                                contract=C, decide=slow_decide, gate_s=0.0005)
        self.assertTrue(row.gate_failed)
        self.assertFalse(row.eligible)
        self.assertEqual(row.decisions_timed, 64)   # stops at the plan's minimum
        self.assertGreater(row.mean_wall_s, 0)

    def test_fast_cell_is_eligible_and_reports_phases(self):
        recs = [_record(i) for i in range(64)]
        def fast(rec, cfg):
            return {"max_depth_reached": cfg.depth-1, "root_action": "move 2",
                    "encode_s": 0.01, "model_s": 0.02, "tree_s": 0.005}
        row = time_lattice_cell(SearchConfig(depth=4, sims=512), records=recs, contract=C, decide=fast)
        self.assertTrue(row.eligible)
        self.assertEqual(row.decisions_timed, 64)
        self.assertEqual(row.cap_hit_rate, 1.0)
        self.assertAlmostEqual(row.model_s, 0.02*64, places=5)
        self.assertEqual(len(row.root_argmax_by_decision), 64)

    def test_fallbacks_make_cell_ineligible(self):
        recs = [_record(i) for i in range(64)]
        def fb(rec, cfg): return {"max_depth_reached": 2, "fallbacks": 1, "root_action": "x"}
        row = time_lattice_cell(SearchConfig(depth=4, sims=512), records=recs, contract=C, decide=fb)
        self.assertFalse(row.eligible)
        self.assertEqual(row.fallbacks, 64)

    def test_prior_fallbacks_are_reported_and_make_cell_ineligible(self):
        recs = [_record(i) for i in range(64)]

        def fallback_prior(rec, cfg):
            return {"max_depth_reached": 2, "prior_fallbacks": 1, "root_action": "move 1"}

        row = time_lattice_cell(
            SearchConfig(depth=4, sims=512), records=recs, contract=C, decide=fallback_prior
        )
        self.assertFalse(row.eligible)
        self.assertEqual(row.prior_fallbacks, 64)

    def test_preparation_is_not_charged_to_the_decision_window(self):
        recs = [_record(i) for i in range(64)]
        import time as _t

        def prepare(rec, cfg):
            # Prefix replay/fold warm-up can be slow. It must finish before
            # the decision stopwatch starts, otherwise the A3 timing row
            # measures a fresh battle reconstruction rather than a decision.
            _t.sleep(0.001)
            return lambda: {
                "max_depth_reached": cfg.depth - 1,
                "root_action": "move 3",
                "total_iterations": 7,
                "model_evals": 2,
                "fold_clone_s": 0.002,
                "row_input_s": 0.001,
            }

        row = time_lattice_cell(
            SearchConfig(depth=4, sims=512),
            records=recs,
            contract=C,
            prepared_decider=prepare,
            gate_s=0.0005,
        )
        self.assertFalse(row.gate_failed)
        self.assertTrue(row.eligible)
        self.assertEqual(row.total_iterations, 64 * 7)
        self.assertEqual(row.model_evals, 64 * 2)
        self.assertAlmostEqual(row.fold_clone_s, 64 * 0.002, places=5)
        self.assertAlmostEqual(row.row_input_s, 64 * 0.001, places=5)

    def test_live_adapter_keeps_public_history_and_times_showdown_choice(self):
        """The default path must not silently weaken native prior semantics.

        The p2 switch actions below only decode to the expected production
        request order when the replay adapter supplies both sampled-world action
        indexes and p1's prior public observations.  The same timed callback
        must also return the real Showdown command, rather than a raw index.
        """

        party = [
            "typhlosion",
            "smeargle",
            "absol",
            "vaporeon",
            "sharpedo",
            "deoxysdefense",
        ]
        mask = (True, False, False, False, False, False, False, False, False)

        def observation(active_species):
            return SimpleNamespace(
                legal_action_mask=mask,
                metadata={
                    "action_candidates": ({"kind": "move", "move_id": "surf", "slot": 1},),
                    "opponent_active": {"species": active_species},
                },
            )

        rounds = (
            PublicResolvedActionRound(
                turn_index=0,
                actions={
                    "p1": PublicActionIdentifier(kind="move", move_id="surf"),
                    "p2": PublicActionIdentifier(kind="switch", switched_species="Absol"),
                },
            ),
            PublicResolvedActionRound(
                turn_index=1,
                actions={
                    "p1": PublicActionIdentifier(kind="move", move_id="surf"),
                    "p2": PublicActionIdentifier(
                        kind="switch", switched_species="Deoxys-Defense"
                    ),
                },
            ),
        )
        record = _record(
            2,
            seat="p1",
            turn_index=2,
            legal_action_mask=mask,
            public_resolved_action_rounds=rounds,
        )
        replayed = SimpleNamespace(
            terminal=None,
            requested_players=("p1", "p2"),
            replay_actions={0: {"p1": 0, "p2": 5}, 1: {"p1": 0, "p2": 8}},
            replay_observations={
                0: {"p1": observation("Typhlosion"), "p2": object()},
                1: {"p1": observation("Absol"), "p2": object()},
            },
        )
        state = SimpleNamespace(legal_action_mask=mask)
        public_state = SimpleNamespace(replay=SimpleNamespace(public_lines=record.event_prefix))
        case = self

        class FakeEnv:
            def observe(self, player):
                case.assertEqual(player, "p1")
                return observation("Deoxys-Defense")

            def public_materialization_state(self, player):
                case.assertEqual(player, "p1")
                return public_state

            def _state_for_player(self, player):
                case.assertEqual(player, "p1")
                return state

        class FakePolicy:
            def __init__(self):
                self.stats = SimpleNamespace(
                    fallback_decisions=0,
                    prior_fallbacks=0,
                    total_iterations=0,
                    model_evals=0,
                    encode_wall_seconds=0.0,
                    model_wall_seconds=0.0,
                    tree_wall_seconds=0.0,
                    fold_clone_wall_seconds=0.0,
                    render_wall_seconds=0.0,
                    fold_advance_wall_seconds=0.0,
                    tensor_wall_seconds=0.0,
                    action_map_wall_seconds=0.0,
                    row_input_wall_seconds=0.0,
                    products_wall_seconds=0.0,
                    row_write_wall_seconds=0.0,
                    depth_reached_histogram={},
                )
                self.warmed = False

            def warm_public_prefix_for_replay(self, **kwargs):
                self.warmed = True

            def select_action_with_context(self, context, *, rng):
                case.assertTrue(self.warmed)
                case.assertEqual(
                    opponent_request_order(context, party),
                    [
                        "deoxysdefense",
                        "smeargle",
                        "typhlosion",
                        "vaporeon",
                        "sharpedo",
                        "absol",
                    ],
                )
                case.assertIsNone(
                    next(step.observation for step in context.trajectory.steps if step.player_id == "p2")
                )
                return SimpleNamespace(action_index=0)

        env = FakeEnv()
        policy = FakePolicy()
        decider = object.__new__(_LiveEngineTimingDecider)
        decider._closed = False
        decider._env = env

        def policy_for(config):
            return policy

        decider._policy_for = policy_for
        with patch("pokezero.public_replay_materializer.replay_public_action_rounds", return_value=replayed):
            telemetry = decider.prepare(record, SearchConfig(depth=4, sims=512))()
        self.assertEqual(telemetry["root_action"], "move 1")
        self.assertEqual(telemetry["prior_fallbacks"], 0)

    def test_public_replay_retains_ephemeral_request_history(self):
        """The live adapter's history must come from the actual replay, not a fixture."""

        mask = (True, False, False, False, False, False, False, False, False)
        p1_observation = SimpleNamespace(
            legal_action_mask=mask,
            metadata={"action_candidates": ({"action_index": 0, "kind": "move", "move_id": "surf"},)},
        )
        p2_observation = SimpleNamespace(
            legal_action_mask=mask,
            metadata={"action_candidates": ({"action_index": 0, "kind": "move", "move_id": "surf"},)},
        )

        class FakeEnv:
            def requested_players(self):
                return ("p1", "p2")

            def observe(self, player):
                return {"p1": p1_observation, "p2": p2_observation}[player]

            def step(self, actions):
                self.actions = actions

            def terminal(self):
                return None

        rounds = (
            PublicResolvedActionRound(
                turn_index=0,
                actions={
                    "p1": PublicActionIdentifier(kind="move", move_id="surf"),
                    "p2": PublicActionIdentifier(kind="move", move_id="surf"),
                },
            ),
        )
        with patch("pokezero.public_replay_materializer.replay_action_rounds"):
            replayed = replay_public_action_rounds(
                FakeEnv(),
                seed=1,
                format_id="gen3randombattle",
                public_action_rounds=rounds,
                start_override=None,
            )
        self.assertIs(replayed.replay_observations[0]["p1"], p1_observation)
        self.assertIs(replayed.replay_observations[0]["p2"], p2_observation)
        self.assertEqual(replayed.replay_actions[0], {"p1": 0, "p2": 0})

if __name__ == "__main__":
    unittest.main()
