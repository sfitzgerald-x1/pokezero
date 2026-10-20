"""Authored bridge integration; no Showdown games, champion or strength claim."""
from contextlib import ExitStack
from dataclasses import replace
import asyncio
import hashlib
import contextlib
import io
import json
from pathlib import Path
import pickle
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from pokezero.actions import ACTION_COUNT
from pokezero.belief import PublicBattleBeliefEngine
import pokezero.foulplay_bridge as bridge
from pokezero.local_showdown import PublicBattleMaterializationState
from pokezero.mcts_eval.paper_reference import Evaluation, ReferenceRefusal
from pokezero.mcts_eval.paper_reference_parallel import ParallelTrajectorySearch, PreparedDecision
import pokezero.mcts_eval.paper_reference_policy as reference
from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest, ShowdownWorkerFactory
from pokezero.mcts_eval.paper_reference_showdown import decision_state
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyContext
from pokezero.trajectory import BattleTrajectory
from test_paper_reference import ToyWorld


CATALOG = "a" * 16
SOURCE = SimpleNamespace(metadata=SimpleNamespace(source_hash=CATALOG))


def context(*, battle_id="owned-1", player="p1", legal=(0, 2), private="PRIVATE_CANARY"):
    opponent = "p2" if player == "p1" else "p1"
    observation = PokeZeroObservationV0(
        categorical_ids=((1,),), numeric_features=((.5,),), token_type_ids=(0,),
        attention_mask=(True,), legal_action_mask=tuple(i in legal for i in range(ACTION_COUNT)),
        metadata={"self_team": [{"species": "Pikachu", "fainted": False}],
            "opponent_team": [{"species": "Snorlax", "fainted": False, "moves": [private]}],
            "belief_view": {"self_slot": player, "opponent_slot": opponent},
            "private_canary": private})
    # Replay fold is authored, never a simulator snapshot.
    public = PublicBattleMaterializationState(player, "gen3randombattle", "gen3randombattle",
        SimpleNamespace(battle_id=battle_id, requests={}),
        PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=SOURCE),
        {"side": {"id": player}})
    return PolicyContext(player, 1, battle_id, "gen3randombattle", 1, observation,
        (player, opponent), BattleTrajectory(battle_id, "gen3randombattle", 1,
            metadata={"private_canary": private}),
        requested_legal_action_masks={player: observation.legal_action_mask,
                                     opponent: (True,) * ACTION_COUNT},
        requested_observations={player: observation, opponent: SimpleNamespace(private=private)},
        public_materialization_state=public)


class AuthoredRuntime:
    def __init__(self, index):
        self.index = index

    def prepare(self, request):
        assert isinstance(request, PublicRootRequest)
        assert request.state.replay.requests == {}
        assert request.state.belief_engine.set_source is None
        assert b"PRIVATE_CANARY" not in pickle.dumps(request)
        root = decision_state(request.observation, player=request.state.player_id)
        forwards = 0
        draws = 0
        def evaluate(_):
            nonlocal forwards
            forwards += 1
            return Evaluation((.5,) * len(root.actions), 0.)
        def world(rng):
            nonlocal draws
            draws += 1
            time.sleep(.01)
            return ToyWorld(root=root, value=1.)
        def evidence():
            nonlocal forwards, draws
            payload = {"neural_forwards": forwards, "draws": [{"status": "AUTHORED"}] * draws}
            forwards = draws = 0
            return payload
        return PreparedDecision(root, evaluate, world, evidence)

    def close(self):
        pass


class RefusedRuntime(AuthoredRuntime):
    def prepare(self, request):
        raise ReferenceRefusal("PRIVATE_CANARY refused")


class RefusedStartupRuntime(AuthoredRuntime):
    def __init__(self, index):
        raise ReferenceRefusal("PRIVATE_CANARY worker startup refused")


class DelayedRuntime(AuthoredRuntime):
    def prepare(self, request):
        if request.observation.numeric_features[0][0] == .9:
            time.sleep(.12)
        return super().prepare(request)


class FakeEndpoint:
    def __init__(self, **kwargs):
        self.uri = "ws://127.0.0.1:1/showdown/websocket"
        self.closed = False

    async def start(self):
        pass

    async def close(self):
        self.closed = True


class TrajectoryReferenceBridgeTests(unittest.TestCase):
    def config(self, **kwargs):
        args = dict(checkpoint=Path("checkpoint.pt"), showdown_root=Path("/showdown"),
            policy_mode="trajectory-reference", reference_seconds=.06, reference_workers=1,
            belief_set_source=True, allow_search_fallback=False, opponent_policy_mode="raw")
        args.update(kwargs)
        return bridge.ControlledFoulPlayConfig(**args)

    def policy(self, **kwargs):
        return reference.TrajectoryReferencePolicy(config=reference.ReferencePolicyConfig(.06, workers=1),
            factory=ShowdownWorkerFactory("/checkpoint", "b" * 64, "/showdown", CATALOG),
            policy_id="authored")

    def patch_entry(self, stack, cfg, *, runtime=AuthoredRuntime):
        """Keep the actual builder and game-owner path; replace all playing surfaces."""
        result = SimpleNamespace(model_config=SimpleNamespace(policy_id="authored"))
        for name, value in (
            ("_validate_external_paths", None), ("_warn_on_belief_provenance_mismatch", None),
            ("load_transformer_checkpoint", (object(), result)),
            ("observation_spec_from_model_config", object()),
            ("category_vocab_from_model_config", object()), ("load_showdown_dex_cached", object()),
            ("feature_masks_from_model_config", object()),
            ("load_gen3_randbat_source_cached", SOURCE),
            ("TransformerSoftmaxPolicy", SimpleNamespace(policy_id="authored-raw")),
        ):
            stack.enter_context(patch.object(bridge, name, return_value=value))
        stack.enter_context(patch("pokezero.mcts_eval.paper_reference_showdown.ChampionEvaluator"))
        stack.enter_context(patch.object(bridge, "_FoulPlayWebsocketServer", FakeEndpoint))
        stack.enter_context(patch.object(bridge, "_BattleBridge", FakeEndpoint))
        stack.enter_context(patch.object(bridge, "_probe_foulplay_start_method",
                                        new=AsyncMock(return_value={"status": "AUTHORED"})))
        spawn = stack.enter_context(patch.object(bridge, "_spawn_foulplay",
            new=AsyncMock(return_value=SimpleNamespace(stdout=None, stderr=None))))
        stack.enter_context(patch.object(bridge, "_wait_for_foulplay_challenge_or_exit", new=AsyncMock()))
        stack.enter_context(patch.object(bridge, "_stop_foulplay_process", new=AsyncMock()))
        pools = []
        def owned_pool(config, factory, **args):
            self.assertEqual(factory.checkpoint_sha256, hashlib.sha256(b"authored-checkpoint").hexdigest())
            self.assertEqual(factory.set_source_hash, CATALOG)
            self.assertEqual(factory.node_binary, cfg.node_binary)
            pool = ParallelTrajectorySearch(config, runtime, **args)
            pools.append(pool)
            return pool
        stack.enter_context(patch.object(reference, "ParallelTrajectorySearch", side_effect=owned_pool))
        return pools, spawn

    def game(self, cfg, decision):
        return bridge.ControlledFoulPlayGameResult(battle_id="owned-1", seed=cfg.seed_start,
            winner=cfg.pokezero_username, pokezero_won=True, decision_rounds=1,
            pokezero_decisions=1, root_puct_searches=0, root_puct_fallbacks=0)

    def test_cli_explicit_recipe_and_opponent_clearance(self):
        parser = bridge.build_arg_parser()
        cfg = bridge._config_from_args(parser.parse_args([
            "--checkpoint", "checkpoint.pt", "--showdown-root", "/showdown",
            "--policy-mode", "trajectory-reference", "--reference-seconds", "3",
            "--belief-set-source", "on", "--no-search-fallback", "--opponent-policy-mode", "raw"]))
        recipe = bridge._reference_policy_config(cfg).to_dict()
        self.assertEqual((recipe["workers"], recipe["alpha"], recipe["beta"], recipe["batch_size"]),
                         (20, .5, 1., 10))
        self.assertFalse(recipe["scientifically_qualified"])
        opponent = bridge._opponent_seat_config(cfg)
        self.assertEqual(opponent.policy_mode, "raw")
        self.assertIsNone(opponent.reference_seconds)
        self.assertIsNone(opponent.reference_workers)
        self.assertNotIn("--reference-seconds", bridge.build_comparison_arg_parser().format_help())
        self.assertEqual(parser.parse_args(["--checkpoint", "c"]).policy_mode, "root-puct")

    def test_configuration_refuses_ignored_or_invalid_overrides(self):
        for values in (
            {"reference_seconds": None}, {"reference_seconds": float("nan")},
            {"reference_seconds": True}, {"reference_workers": 21}, {"reference_workers": True},
            {"reference_max_transitions": 0}, {"reference_alpha": 2},
            {"reference_beta": -.1}, {"temperature": .5}, {"device": "cuda"},
            {"value_checkpoint": Path("other.pt")}, {"belief_set_source": False},
            {"allow_search_fallback": True}, {"format_id": "gen4randombattle"},
            {"opponent_policy_mode": "root-puct"},
            {"reference_allow_earlier_compatible_template": 1},
            {"policy_mode": "raw", "opponent_policy_mode": "foul-play"},
        ):
            with self.subTest(values=values), self.assertRaises((ValueError, ReferenceRefusal)):
                self.config(**values)

    def test_real_owned_bridge_entry_public_transport_stats_reuse_and_cleanup(self):
        progress, decisions, policies = [], [], []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"authored-checkpoint")
            cfg = self.config(checkpoint=checkpoint, games=2, node_binary="custom-node")
            pools, spawn = self.patch_entry(stack, cfg)
            async def authored_game(**args):
                policy = args["policy"]
                policies.append(policy)
                self.assertEqual(args["opponent_policy"].policy_id, "authored-raw")
                first = policy.select_action_with_context(context(), rng=random.Random(1))
                second = policy.select_action_with_context(context(), rng=random.Random(2))
                third = policy.select_action_with_context(context(battle_id="owned-2"), rng=random.Random(3))
                decisions.extend((first, second, third))
                return self.game(cfg, first)
            stack.enter_context(patch.object(bridge, "_run_single_game", side_effect=authored_game))
            result = asyncio.run(bridge.run_controlled_foulplay_benchmark(cfg, progress_callback=progress.append))
        self.assertEqual(len(pools), 1)
        self.assertEqual(len(set(id(p) for p in policies)), 1)
        self.assertTrue(all(not p.is_alive() for p in pools[0]._processes))
        spawn.assert_not_awaited()
        for decision in decisions:
            self.assertIn(decision.action_index, (0, 2))
            self.assertFalse(decision.metadata["trajectory_reference"]["raw_fallback"])
        first, second, third = [d.metadata["trajectory_reference"] for d in decisions[:3]]
        self.assertEqual(sum(second["root_visits"]), sum(first["root_visits"]) + second["new_trajectories"])
        self.assertEqual(sum(third["root_visits"]), third["new_trajectories"])
        self.assertEqual(first["worker_pids"], third["worker_pids"])
        self.assertEqual([p.completed_games for p in progress], [1, 2])
        for snapshot in [*progress, result]:
            summary = snapshot.to_dict()["trajectory_reference"]
            self.assertGreater(summary["policy_stats"]["trajectories"], 0)
            self.assertEqual(summary["workers"], 1)
            self.assertFalse(summary["atomic_work_instrumented"])
            self.assertIsNone(summary["cpu_seconds"])
            self.assertGreater(summary["policy_stats"]["pool_startup_seconds"], 0)
        self.assertLess(progress[0].policy_stats["trajectories"], result.policy_stats["trajectories"])

    def test_forced_action_uses_no_worker_and_is_not_raw_fallback(self):
        policy = self.policy()
        with patch.object(reference, "ParallelTrajectorySearch") as create:
            decision = policy.select_action_with_context(context(legal=(2,)), rng=random.Random(1))
            create.assert_not_called()
        self.assertEqual(decision.action_index, 2)
        self.assertEqual(policy.stats.forced_decisions, 1)
        self.assertEqual(decision.metadata["trajectory_reference"]["status"], "FORCED")
        policy.close()

    def test_mismatched_actor_catalog_replay_and_format_refuse_without_pool(self):
        own = context()
        states = (
            replace(own, player_id="p2"), replace(own, battle_id="other"),
            replace(own, public_materialization_state=None),
            replace(own, public_materialization_state=replace(own.public_materialization_state,
                format_id="gen4randombattle")),
            replace(own, public_materialization_state=replace(own.public_materialization_state,
                replay=SimpleNamespace(battle_id=own.battle_id, requests={"p2": {"private": "CANARY"}}))),
            replace(own, public_materialization_state=replace(own.public_materialization_state,
                belief_engine=PublicBattleBeliefEngine(set_source=SimpleNamespace(
                    metadata=SimpleNamespace(source_hash="c" * 16))))),
        )
        with patch.object(reference, "ParallelTrajectorySearch") as create:
            for state in states:
                policy = self.policy()
                with self.assertRaises(ReferenceRefusal):
                    policy.select_action_with_context(state, rng=random.Random(1))
                with self.assertRaisesRegex(ReferenceRefusal, "no restart"):
                    policy.select_action_with_context(own, rng=random.Random(1))
                policy.close()
            create.assert_not_called()

    def test_actual_worker_refusal_emits_partial_failure_summary_and_closes(self):
        progress = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"authored-checkpoint")
            cfg = self.config(checkpoint=checkpoint)
            pools, _ = self.patch_entry(stack, cfg, runtime=RefusedRuntime)
            async def authored_game(**args):
                return args["policy"].select_action_with_context(context(), rng=random.Random(1))
            stack.enter_context(patch.object(bridge, "_run_single_game", side_effect=authored_game))
            with self.assertRaises(ReferenceRefusal):
                asyncio.run(bridge.run_controlled_foulplay_benchmark(cfg, progress_callback=progress.append))
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0].completed_games, 0)
        stats = progress[0].to_dict()["trajectory_reference"]["policy_stats"]
        self.assertEqual(stats["refusals"], 1)
        self.assertEqual(stats["binding"]["set_source_hash"], CATALOG)
        self.assertEqual(stats["binding"]["checkpoint_sha256"], hashlib.sha256(b"authored-checkpoint").hexdigest())
        self.assertEqual(stats["last_decision"]["status"], "REFUSED")
        failures = stats["last_decision"]["worker_failures"]
        self.assertEqual(failures[0]["phase"], "public_root_preparation")
        self.assertTrue(failures[0]["error_code_sites"][0]["frames"])
        self.assertEqual(failures[0]["error_code_sites"][0]["category"], "ReferenceRefusal")
        self.assertNotIn("PRIVATE_CANARY", str(stats))
        self.assertTrue(all(not p.is_alive() for p in pools[0]._processes))

    def test_benchmark_closes_candidate_when_opponent_construction_fails(self):
        policy = self.policy()
        pool = ParallelTrajectorySearch(reference.ReferenceConfig(.5, 1.), AuthoredRuntime, workers=1)
        policy._pool = pool
        self.addCleanup(pool.close)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"authored-checkpoint")
            cfg = self.config(checkpoint=checkpoint)
            self.patch_entry(stack, cfg)
            stack.enter_context(patch.object(bridge, "_build_policy", side_effect=[policy, RuntimeError("opponent")]))
            with self.assertRaisesRegex(RuntimeError, "opponent"):
                asyncio.run(bridge.run_controlled_foulplay_benchmark(cfg))
        self.assertTrue(policy._closed)
        self.assertTrue(all(not p.is_alive() for p in pool._processes))

    def test_constructor_worker_refusal_retains_safe_sites_before_pool_assignment(self):
        policy = self.policy()
        self.addCleanup(policy.close)
        observed = []
        def make_pool(config, factory, **kwargs):
            try:
                return ParallelTrajectorySearch(config, RefusedStartupRuntime, **kwargs)
            except reference.ParallelRefusal as error:
                observed.append(error)
                raise
        with patch.object(reference, "ParallelTrajectorySearch", side_effect=make_pool):
            with self.assertRaises(reference.ParallelRefusal):
                policy.select_action_with_context(context(), rng=random.Random(1))
        self.assertEqual(len(observed), 1)
        self.assertIsNone(policy._pool)
        last = policy.stats.last_decision
        self.assertEqual(last["phase"], "pool_startup")
        self.assertEqual(last["worker_failures"][0]["phase"], "startup")
        self.assertTrue(last["worker_failures"][0]["error_code_sites"][0]["frames"])
        self.assertNotIn("PRIVATE_CANARY", json.dumps(policy.stats.to_dict()))
        with self.assertRaisesRegex(ReferenceRefusal, "no restart"):
            policy.select_action_with_context(context(), rng=random.Random(1))

    def test_malformed_diagnostic_cannot_make_refusal_summary_unserializable(self):
        policy = self.policy()
        self.addCleanup(policy.close)
        error = reference.ParallelRefusal("original refusal", {"errors": [(
            "error", 0, 1 << 65536, "trajectory_batch", "PRIVATE_CANARY", "PRIVATE_CANARY",
            {"error_code_sites": [{"category": "ReferenceRefusal", "frames": []}]},
        )]})
        policy.record_failure("search", error)
        self.assertEqual(policy.stats.refusals, 1)
        self.assertEqual(policy.stats.last_decision["status"], "REFUSED")
        self.assertNotIn("worker_failures", policy.stats.last_decision)
        self.assertNotIn("PRIVATE_CANARY", json.dumps(policy.stats.to_dict()))
        self.assertEqual(str(error), "original refusal")

    def test_zero_new_work_refuses_after_previous_work_without_stale_or_forced_fallback(self):
        policy = self.policy()
        self.addCleanup(policy.close)
        with patch.object(reference, "ParallelTrajectorySearch",
                side_effect=lambda config, factory, **args: ParallelTrajectorySearch(config, AuthoredRuntime, **args)):
            policy.select_action_with_context(context(), rng=random.Random(1))
            # A real owning pool deadline, not a fabricated zero-result object.
            with patch.object(reference, "decision_state", side_effect=lambda obs, **kw: (
                    time.sleep(.08) or decision_state(obs, **kw))):
                with self.assertRaisesRegex(ReferenceRefusal, "exhausted"):
                    policy.select_action_with_context(context(), rng=random.Random(2))
            with self.assertRaisesRegex(ReferenceRefusal, "no restart"):
                policy.select_action_with_context(context(legal=(2,)), rng=random.Random(3))
        self.assertEqual(policy.stats.searches, 1)
        self.assertEqual(policy.stats.refusals, 1)
        self.assertGreaterEqual(policy.stats.decision_wall_seconds, .08)

    def test_actual_worker_zero_new_work_does_not_select_previous_tree(self):
        policy = self.policy()
        self.addCleanup(policy.close)
        with patch.object(reference, "ParallelTrajectorySearch",
                side_effect=lambda config, factory, **args: ParallelTrajectorySearch(config, DelayedRuntime, **args)):
            own = context()
            policy.select_action_with_context(own, rng=random.Random(1))
            delayed = replace(own, observation=replace(own.observation, numeric_features=((.9,),)))
            with self.assertRaisesRegex(ReferenceRefusal, "zero NEW"):
                policy.select_action_with_context(delayed, rng=random.Random(2))
            with self.assertRaisesRegex(ReferenceRefusal, "no restart"):
                policy.select_action_with_context(own, rng=random.Random(3))
        last = policy.stats.last_decision
        self.assertEqual(last["observed_partial_trajectories"], 0)
        self.assertFalse(last["refused_work_accounting_complete"])

    def test_bad_returned_selector_is_refused_before_play(self):
        for visits, action in (((1, 5), "action:0"), ((1, 1), "action:2"), ((1, 1), "action:9")):
            with self.subTest(visits=visits, action=action):
                policy = self.policy()
                self.addCleanup(policy.close)
                def pool_factory(config, factory, **args):
                    pool = ParallelTrajectorySearch(config, AuthoredRuntime, **args)
                    real_search = pool.search
                    def bad_search(*a, **kw):
                        value = real_search(*a, **kw)
                        return replace(value, result=replace(value.result, action=action, root_visits=visits))
                    pool.search = bad_search
                    return pool
                with patch.object(reference, "ParallelTrajectorySearch", side_effect=pool_factory):
                    with self.assertRaises(ReferenceRefusal):
                        policy.select_action_with_context(context(), rng=random.Random(1))
                self.assertEqual(policy.stats.searches, 0)
                self.assertEqual(policy.stats.refusals, 1)
                policy.close()

    def test_bridge_failure_preserves_completed_games_and_closes_owned_pool(self):
        progress = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"authored-checkpoint")
            cfg = self.config(checkpoint=checkpoint, games=2)
            pools, _ = self.patch_entry(stack, cfg)
            async def authored_game(**args):
                if args["seed"] == cfg.seed_start + 1:
                    raise RuntimeError("PRIVATE_CANARY bridge loss")
                decision = args["policy"].select_action_with_context(context(), rng=random.Random(1))
                return self.game(cfg, decision)
            stack.enter_context(patch.object(bridge, "_run_single_game", side_effect=authored_game))
            with self.assertRaisesRegex(RuntimeError, "bridge loss"):
                asyncio.run(bridge.run_controlled_foulplay_benchmark(cfg, progress_callback=progress.append))
        self.assertEqual([p.completed_games for p in progress], [1, 1])
        stats = progress[-1].to_dict()["trajectory_reference"]["policy_stats"]
        self.assertEqual(stats["last_decision"]["phase"], "controlled_bridge")
        self.assertNotIn("PRIVATE_CANARY", str(stats))
        self.assertEqual(stats["searches"], 1)
        self.assertTrue(all(not p.is_alive() for p in pools[0]._processes))

    def test_external_foulplay_route_and_p2_subject(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"authored-checkpoint")
            cfg = self.config(checkpoint=checkpoint, opponent_policy_mode="foul-play", pokezero_player="p2")
            pools, spawn = self.patch_entry(stack, cfg)
            async def authored_game(**args):
                self.assertIsNone(args["opponent_policy"])
                decision = args["policy"].select_action_with_context(context(player="p2"), rng=random.Random(1))
                return self.game(cfg, decision)
            stack.enter_context(patch.object(bridge, "_run_single_game", side_effect=authored_game))
            result = asyncio.run(bridge.run_controlled_foulplay_benchmark(cfg))
        spawn.assert_awaited_once()
        self.assertEqual(result.completed_games, 1)
        self.assertTrue(all(not p.is_alive() for p in pools[0]._processes))

    def test_legacy_summary_has_no_reference_block(self):
        cfg = bridge.ControlledFoulPlayConfig(checkpoint=Path("c"), showdown_root=Path("/showdown"), policy_mode="raw")
        self.assertNotIn("trajectory_reference", bridge.ControlledFoulPlayBenchmarkResult(cfg, "raw", ()).to_dict())

    def test_pool_cleanup_failure_cannot_leave_a_success_summary(self):
        progress = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"authored-checkpoint")
            cfg = self.config(checkpoint=checkpoint)
            pools, _ = self.patch_entry(stack, cfg)
            async def authored_game(**args):
                policy = args["policy"]
                decision = policy.select_action_with_context(context(), rng=random.Random(1))
                real_close = policy.close
                def failed_close():
                    real_close()
                    raise RuntimeError("cleanup")
                policy.close = failed_close
                return self.game(cfg, decision)
            stack.enter_context(patch.object(bridge, "_run_single_game", side_effect=authored_game))
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                asyncio.run(bridge.run_controlled_foulplay_benchmark(cfg, progress_callback=progress.append))
        self.assertEqual(progress[-1].completed_games, 1)
        self.assertEqual(progress[-1].to_dict()["trajectory_reference"]["status"], "REFUSED")
        self.assertEqual(progress[-1].policy_stats["last_decision"]["phase"], "pool_cleanup")
        self.assertTrue(all(not p.is_alive() for p in pools[0]._processes))

    def test_cli_owns_the_reference_run_and_writes_its_recipe_and_stats(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"authored-checkpoint")
            summary_path = Path(directory) / "summary.json"
            cfg = self.config(checkpoint=checkpoint)
            pools, _ = self.patch_entry(stack, cfg)
            async def authored_game(**args):
                decision = args["policy"].select_action_with_context(context(), rng=random.Random(1))
                return self.game(cfg, decision)
            stack.enter_context(patch.object(bridge, "_run_single_game", side_effect=authored_game))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = asyncio.run(bridge.async_main([
                    "--checkpoint", str(checkpoint), "--showdown-root", "/showdown",
                    "--policy-mode", "trajectory-reference", "--reference-seconds", ".06",
                    "--reference-workers", "1", "--belief-set-source", "on",
                    "--no-search-fallback", "--opponent-policy-mode", "raw",
                    "--summary-out", str(summary_path)]))
            summary = json.loads(summary_path.read_text())["trajectory_reference"]
        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["decision_seconds"], .06)
        self.assertEqual(summary["policy_stats"]["searches"], 1)
        self.assertFalse(summary["scientifically_qualified"])
        self.assertTrue(all(not p.is_alive() for p in pools[0]._processes))

    def test_noncanonical_model_refuses_before_worker_construction(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"authored-checkpoint")
            cfg = self.config(checkpoint=checkpoint)
            result = SimpleNamespace(model_config=SimpleNamespace(window_size=2, value_activation="tanh"))
            stack.enter_context(patch.object(bridge, "TransformerSoftmaxPolicy",
                                             return_value=SimpleNamespace(result=result)))
            created = stack.enter_context(patch.object(reference, "ParallelTrajectorySearch"))
            with self.assertRaisesRegex(ReferenceRefusal, "one-snapshot"):
                bridge._build_policy(config=cfg, model=object(), result=result,
                    value_model=object(), value_result=result, env_config=object(),
                    rollout_config=object(), policy_id="authored")
            created.assert_not_called()

    def test_failed_startup_reports_its_wall_cost_and_never_restarts(self):
        policy = self.policy()
        def failed_startup(*args, **kwargs):
            time.sleep(.05)
            raise ReferenceRefusal("startup failed")
        with patch.object(reference, "ParallelTrajectorySearch", side_effect=failed_startup) as create:
            with self.assertRaisesRegex(ReferenceRefusal, "startup failed"):
                policy.select_action_with_context(context(), rng=random.Random(1))
            with self.assertRaisesRegex(ReferenceRefusal, "no restart"):
                policy.select_action_with_context(context(), rng=random.Random(2))
        create.assert_called_once()
        self.assertGreaterEqual(policy.stats.pool_startup_seconds, .05)
        self.assertEqual(policy.stats.last_decision["phase"], "pool_startup")
        self.assertEqual(policy.stats.trajectories, 0)
        policy.close()


if __name__ == "__main__":
    unittest.main()
