"""Contract tests for the fixed-root opponent-prior ablation runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _runner():
    path = SCRIPTS / "run_source_root_opponent_prior_ablation.py"
    spec = importlib.util.spec_from_file_location("source_root_opponent_prior_ablation_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _selection() -> dict[str, object]:
    return {
        "root_action": "move 1",
        "total_iterations": 4096,
        "model_evals": 1024,
        "max_depth_reached": 6,
        "root_allocation": {"worlds": 4, "arms": []},
        "rollout_leaf": None,
        "live_branch_prior": {"prior_fallbacks": 0},
    }


def _telemetry(*, enabled: bool) -> dict[str, object]:
    return {
        "engine_mcts": {
            "opponent_prior_application": {
                "enabled": enabled,
                "root_applied": 1 if enabled else 0,
                "branch_applied": 2 if enabled else 0,
                "total_applied": 3 if enabled else 0,
                "digest": "a" * 64 if enabled else None,
            },
        },
    }


class OpponentPriorAblationRunnerTest(unittest.TestCase):
    def test_roster_is_the_exact_source_replayable_override_panel(self) -> None:
        runner = _runner()
        self.assertEqual(len(runner.TARGETS), 11)
        self.assertEqual(len(set(runner.TARGETS)), 11)
        self.assertEqual(
            [(item.seed, item.seat, item.turn_index) for item in runner.TARGETS],
            [
                (2026092004, "p1", 19), (2026092004, "p1", 20),
                (2026092005, "p1", 21), (2026092005, "p1", 25),
                (2026092005, "p2", 1), (2026092005, "p2", 6),
                (2026092006, "p1", 2), (2026092006, "p1", 3),
                (2026092007, "p1", 9), (2026092007, "p1", 19),
                (2026092007, "p2", 9),
            ],
        )
        self.assertEqual(
            [(item.seed, item.seat, item.turn_index) for item, _ in runner.EXCLUDED_UNREPLAYABLE_ROOTS],
            [
                (2026092004, "p2", 6), (2026092004, "p2", 10),
                (2026092006, "p2", 10), (2026092006, "p2", 29),
                (2026092007, "p2", 19),
            ],
        )

    def test_only_the_treatment_arm_enables_opponent_priors(self) -> None:
        runner = _runner()
        calls: list[bool] = []

        class _Decider:
            def __init__(self, enabled: bool) -> None:
                self.enabled = enabled

            def prepare_public_decision(self, *args, **kwargs):
                return lambda: _telemetry(enabled=self.enabled)

            def close(self) -> None:
                return None

        record = SimpleNamespace(
            decision_id="d" * 64,
            to_dict=lambda: {"decision_id": "d" * 64},
        )
        prefix = SimpleNamespace(public_action_rounds=(), repairs=())
        with (
            mock.patch.object(
                runner,
                "_new_decider",
                side_effect=lambda *args, use_opponent_priors: (
                    calls.append(use_opponent_priors) or _Decider(use_opponent_priors)
                ),
            ),
            mock.patch.object(runner.base, "source_bound_replay_prefix", return_value=prefix),
            mock.patch.object(runner.base, "_selection_witness", return_value=_selection()),
        ):
            result = runner._run_root(
                root=runner.TARGETS[0],
                record=record,
                source_records=(),
                historical_fallback=None,
                contract=object(),
                args=SimpleNamespace(),
                manifest_sha256="a" * 64,
            )
        self.assertEqual(calls, [False, False, True])
        self.assertEqual(tuple(result["arms"]), runner.ARMS)
        self.assertEqual(result["schema_version"], runner.SCHEMA_VERSION)
        self.assertEqual(
            result["arms"]["opponent_priors"]["selection"]["opponent_prior_application"]["total_applied"],
            3,
        )

    def test_treatment_without_an_applied_native_vector_is_rejected(self) -> None:
        runner = _runner()
        with self.assertRaisesRegex(runner.base.AblationError, "not applicable"):
            runner._validate_opponent_prior_application(
                {
                    "enabled": True,
                    "root_applied": 0,
                    "branch_applied": 0,
                    "total_applied": 0,
                    "digest": None,
                },
                enabled=True,
            )

    def test_control_with_an_applied_native_vector_is_rejected(self) -> None:
        runner = _runner()
        with self.assertRaisesRegex(runner.base.AblationError, "unexpectedly applied"):
            runner._validate_opponent_prior_application(
                {
                    "enabled": False,
                    "root_applied": 1,
                    "branch_applied": 0,
                    "total_applied": 1,
                    "digest": "a" * 64,
                },
                enabled=False,
            )

    def test_resume_rejects_a_treatment_with_different_fixed_work(self) -> None:
        runner = _runner()
        control = _selection()
        control["opponent_prior_application"] = runner._validate_opponent_prior_application(
            _telemetry(enabled=False)["engine_mcts"]["opponent_prior_application"], enabled=False
        )
        treatment = _selection()
        treatment["total_iterations"] = 1
        treatment["opponent_prior_application"] = runner._validate_opponent_prior_application(
            _telemetry(enabled=True)["engine_mcts"]["opponent_prior_application"], enabled=True
        )
        payload = {
            "arms": {
                "model_control_a": {"selection": control},
                "model_control_b": {"selection": dict(control)},
                "opponent_priors": {"selection": treatment},
            },
        }
        with (
            mock.patch.object(runner, "_BASE_VALIDATE_COMPLETED_ROOT"),
            self.assertRaisesRegex(runner.base.AblationError, "fixed native search work"),
        ):
            runner._validate_completed_root(
                payload,
                root=runner.TARGETS[0],
                record=object(),
                historical_fallback=None,
                manifest_sha256="a" * 64,
            )


if __name__ == "__main__":
    unittest.main()
