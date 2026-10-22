"""Authored liveness checks only: no checkpoint, native search or simulator."""

import asyncio
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from pokezero import foulplay_bridge as bridge
from pokezero.fullgame_phase_progress import FullgamePhaseProgress, SCHEMA_VERSION
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyDecision
from pokezero.trajectory import BattleTrajectory


class PhaseProgressTests(unittest.TestCase):
    def test_initial_publication_is_complete_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase.json"
            link = os.link
            observed = []

            def publish(source, destination):
                self.assertFalse(path.exists())
                staged = json.loads(Path(source).read_text())
                self.assertEqual(staged["current"]["status"], "started")
                observed.append(staged)
                return link(source, destination)

            with patch("pokezero.fullgame_phase_progress.os.link", side_effect=publish):
                FullgamePhaseProgress(path)
            self.assertEqual(json.loads(path.read_text()), observed[0])
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_bounded_atomic_snapshot_preserves_last_return_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase.json"
            rng = random.getstate()
            progress = FullgamePhaseProgress(path)
            progress.emit("candidate_policy", "returned", seed=7, seat="p2", decision_round=1)
            returned = json.loads(path.read_text())["current"]
            for index in range(100):
                progress.emit("candidate_policy", "started", seed=7, seat="p2", decision_round=index)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["schema_version"], SCHEMA_VERSION)
            self.assertEqual(saved["sequence"], 101)
            self.assertEqual(saved["last_returned"], returned)
            self.assertGreaterEqual(saved["current"]["monotonic_ns"], returned["monotonic_ns"])
            self.assertFalse(saved["score_bearing"])
            self.assertLess(path.stat().st_size, 1200)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertEqual(random.getstate(), rng)

    def test_existing_artifacts_and_symlinks_are_never_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text("historical evidence\n")
            with self.assertRaises(FileExistsError):
                FullgamePhaseProgress(path)
            link = Path(directory) / "link.json"
            link.symlink_to(path)
            with self.assertRaises(FileExistsError):
                FullgamePhaseProgress(link)
            self.assertEqual(path.read_text(), "historical evidence\n")

    def test_changed_owner_fails_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase.json"
            progress = FullgamePhaseProgress(path)
            path.write_text('{"schema_version": "different-owner"}\n')
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "ownership"):
                progress.emit("game", "started", seed=1)
            self.assertEqual(path.read_bytes(), before)

    def test_only_allowlisted_public_phase_fields_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase.json"
            progress = FullgamePhaseProgress(path)
            before = path.read_bytes()
            for kwargs in ({"phase": "private-state"}, {"status": "won"},
                           {"seat": "secret"}, {"seed": True}, {"decision_round": -1}):
                values = dict(phase="game", status="started", seed=1)
                values.update(kwargs)
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    progress.emit(**values)
            with self.assertRaises(TypeError):
                progress.emit("game", "started", action_index=3)
            self.assertEqual(path.read_bytes(), before)

    def test_cli_default_off_and_comparison_does_not_offer_shared_snapshot(self):
        args = bridge.build_arg_parser().parse_args(["--checkpoint", "champion.pt"])
        self.assertIsNone(args.phase_progress_out)
        self.assertNotIn("--phase-progress-out", bridge.build_comparison_arg_parser().format_help())

    def test_cli_rejects_result_alias_before_starting(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                bridge, "run_controlled_foulplay_benchmark", new_callable=AsyncMock) as run:
            path = Path(directory) / "summary.json"
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as caught:
                asyncio.run(bridge.async_main([
                    "--checkpoint", "champion.pt", "--showdown-root", "showdown",
                    "--summary-out", str(path), "--phase-progress-out", str(path)]))
            self.assertEqual(caught.exception.code, 2)
            run.assert_not_awaited()
            self.assertFalse(path.exists())

    def test_cli_routes_opt_in_writer_and_leaves_default_off(self):
        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "phase.json"
                args = ["--checkpoint", "champion.pt", "--showdown-root", "showdown", "--json"]
                if enabled:
                    args.extend(["--phase-progress-out", str(path)])
                result = SimpleNamespace(to_dict=lambda: {})

                async def benchmark(config, **kwargs):
                    writer = kwargs["phase_progress"]
                    self.assertEqual(writer is not None, enabled)
                    if enabled:
                        writer.emit("game", "started", seed=7)
                    return result

                with patch.object(bridge, "run_controlled_foulplay_benchmark", side_effect=benchmark), \
                        patch("sys.stdout", io.StringIO()):
                    self.assertEqual(asyncio.run(bridge.async_main(args)), 0)
                if enabled:
                    current = json.loads(path.read_text())["current"]
                    self.assertEqual((current["phase"], current["status"]), ("run", "returned"))
                else:
                    self.assertFalse(path.exists())


class PhaseBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def boundary(self, seat, progress):
        observation = PokeZeroObservationV0(
            categorical_ids=(), numeric_features=(), token_type_ids=(), attention_mask=(),
            legal_action_mask=(True, True) + (False,) * 7, metadata={"private": "never-write"})
        state = bridge._ControlledBattleState(
            battle_id="authored-7", seed=7, format_id="gen3randombattle",
            trajectory=BattleTrajectory(battle_id="authored-7", format_id="gen3randombattle", seed=7))
        config = bridge.ControlledFoulPlayConfig(
            checkpoint=Path("champion.pt"), showdown_root=Path("showdown"),
            policy_mode="engine-mcts", opponent_policy_mode="raw", pokezero_player=seat,
            engine_model_path=Path("model.pt"), engine_tables_path=Path("tables.json"))
        wire = SimpleNamespace(send=AsyncMock())
        kwargs = dict(config=config, bridge=wire, server=object(), state=state,
                      policy=object(), opponent_policy=object(), vocab=object(), dex=object(),
                      observation_spec=SimpleNamespace(schema_version="v2.2"),
                      decision_round=4, requested_players=("p1", "p2"),
                      foulplay_process=None, foulplay_logs=object(), phase_progress=progress)
        stack = ExitStack()
        for name, replacement in {
            "_player_state": lambda _state, player, **kw: SimpleNamespace(slot=player),
            "observation_from_player_state": lambda *a, **kw: observation,
            "_observation_with_search_metadata": lambda value, state: value,
            "_context_for_seat": lambda **kw: SimpleNamespace(player_id=kw["seat"]),
            "showdown_choice_for_action": lambda player, action: f"{player.slot}:{action}",
        }.items():
            stack.enter_context(patch.object(bridge, name, side_effect=replacement))
        return stack, kwargs, wire, state

    async def test_both_seats_enabled_and_disabled_preserve_calls_choices_and_timings(self):
        for seat in ("p1", "p2"):
            outputs = []
            for enabled in (False, True):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "phase.json"
                    progress = FullgamePhaseProgress(path) if enabled else None
                    stack, kwargs, wire, state = self.boundary(seat, progress)
                    calls = []

                    def select(policy, observation, context, *, seed):
                        calls.append((context.player_id, seed))
                        if enabled:
                            current = json.loads(path.read_text())["current"]
                            self.assertEqual(current["phase"],
                                             "candidate_policy" if len(calls) == 1 else "opponent_policy")
                            self.assertEqual(current["status"], "started")
                            self.assertEqual((current["seed"], current["seat"], current["decision_round"]),
                                             (7, context.player_id, 4))
                        return PolicyDecision(action_index=len(calls) - 1, policy_id="authored",
                                              metadata={"private": "never-write"})

                    with stack, patch.object(bridge, "_select_policy_decision", side_effect=select), \
                            patch.object(bridge.time, "perf_counter", side_effect=(1., 2., 3., 5.)):
                        await bridge._handle_decision_boundary(**kwargs)
                    outputs.append((calls, wire.send.call_args.args, state.decisions,
                                    state.opponent_decisions, state.pokezero_submitted_choice_players))
                    if enabled:
                        saved = json.loads(path.read_text())
                        self.assertEqual(saved["current"]["phase"], "submission")
                        self.assertEqual(saved["current"]["status"], "returned")
                        self.assertNotIn("never-write", path.read_text())
                        self.assertNotIn("action_index", path.read_text())
                    else:
                        self.assertFalse(path.exists())
            self.assertEqual(outputs[0], outputs[1])

    async def test_pending_policy_call_is_visible_before_it_returns_or_is_cancelled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase.json"
            progress = FullgamePhaseProgress(path)
            stack, kwargs, wire, _ = self.boundary("p1", progress)
            entered = asyncio.Event()

            async def pending(*a, **kw):
                entered.set()
                await asyncio.Event().wait()

            with stack, patch.object(bridge.asyncio, "to_thread", side_effect=pending):
                task = asyncio.create_task(bridge._handle_decision_boundary(**kwargs))
                try:
                    await asyncio.wait_for(entered.wait(), timeout=1)
                    current = json.loads(path.read_text())["current"]
                    self.assertEqual((current["phase"], current["status"]), ("candidate_policy", "started"))
                    wire.send.assert_not_awaited()
                finally:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                self.assertEqual(json.loads(path.read_text())["current"], current)

    async def test_policy_failure_keeps_started_marker_and_original_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase.json"
            progress = FullgamePhaseProgress(path)
            stack, kwargs, wire, _ = self.boundary("p2", progress)
            error = RuntimeError("private branch must not enter phase snapshot")
            with stack, patch.object(bridge, "_select_policy_decision", side_effect=error):
                with self.assertRaises(RuntimeError) as caught:
                    await bridge._handle_decision_boundary(**kwargs)
            self.assertIs(caught.exception, error)
            wire.send.assert_not_awaited()
            self.assertEqual(json.loads(path.read_text())["current"]["status"], "started")
            self.assertNotIn("private branch", path.read_text())

    async def test_bridge_wait_is_distinct_from_a_policy_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase.json"
            progress = FullgamePhaseProgress(path)
            config = bridge.ControlledFoulPlayConfig(checkpoint=Path("c.pt"), showdown_root=Path("showdown"))
            error = RuntimeError("authored bridge wait failure")
            wire = SimpleNamespace(send=AsyncMock(), next_event=AsyncMock(side_effect=error))
            with self.assertRaises(RuntimeError) as caught:
                await bridge._run_single_game(
                    config=config, bridge=wire, server=SimpleNamespace(send_room_lines=AsyncMock()),
                    policy=object(), vocab=object(), dex=object(), observation_spec=object(),
                    seed=7, foulplay_process=None, foulplay_logs=object(), phase_progress=progress)
            self.assertIs(caught.exception, error)
            current = json.loads(path.read_text())["current"]
            self.assertEqual((current["phase"], current["status"]), ("bridge_wait", "started"))

    async def test_continuation_predecode_marks_the_real_external_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase.json"
            progress = FullgamePhaseProgress(path)
            stack, kwargs, wire, _ = self.boundary("p1", progress)
            stop = RuntimeError("authored stop after predecode; no continuation runs")
            probe = SimpleNamespace(capture_boundary_async=AsyncMock(side_effect=stop))
            kwargs["live_continuation_probe"] = probe

            async def wait(**kw):
                current = json.loads(path.read_text())["current"]
                self.assertEqual((current["phase"], current["status"], current["seat"]),
                                 ("foulplay_choice", "started", "p2"))
                return "move 1"

            with stack, patch.object(bridge, "_wait_for_foulplay_choice_or_exit", side_effect=wait), \
                    patch.object(bridge, "action_index_from_choice_string", return_value=0), \
                    patch.object(bridge, "_foulplay_think_observation", new_callable=AsyncMock, return_value={}), \
                    patch.object(progress, "emit", wraps=progress.emit) as emit:
                with self.assertRaises(RuntimeError) as caught:
                    await bridge._handle_decision_boundary(**kwargs)
            self.assertIs(caught.exception, stop)
            self.assertEqual([call.args for call in emit.call_args_list],
                             [("foulplay_choice", "started"), ("foulplay_choice", "returned")])
            self.assertEqual(json.loads(path.read_text())["current"]["status"], "returned")
            wire.send.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
