from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.env import TerminalState
from pokezero.observation import PokeZeroObservationV0
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "live_foulplay_continuation_smoke.py"


def _module():
    spec = importlib.util.spec_from_file_location("live_foulplay_continuation_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation() -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=tuple(index == 0 for index in range(ACTION_COUNT)),
    )


def _trajectory(*, metadata: dict[str, object], turn_index: int = 1, capped: bool = False) -> BattleTrajectory:
    observation = _observation()
    trajectory = BattleTrajectory(
        battle_id="controlled-foulplay-108000000",
        format_id="gen3randombattle",
        seed=108_000_000,
        metadata={"controlled_foulplay_bridge": True},
    )
    trajectory.append(
        TrajectoryStep(
            player_id="p1",
            turn_index=turn_index,
            observation=observation,
            legal_action_mask=tuple(observation.legal_action_mask),
            action_index=0,
            metadata=metadata,
        )
    )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=turn_index + 1, capped=capped))
    return trajectory


class LiveFoulPlayContinuationSmokeTest(unittest.TestCase):
    def test_extracts_a_nonfallback_midgame_continuation_proof(self) -> None:
        module = _module()
        trajectory = _trajectory(
            metadata={
                "policy_family": "root-puct-search",
                "root_puct_fallback": False,
                "root_puct_total_visits": 9,
                "root_puct_leaf_rollout_rounds": 1,
                "root_puct_leaf_actual_rollout_rounds": {"1": 3},
            }
        )

        proof = module.continuation_proof_from_trajectory(
            trajectory,
            pokezero_player="p1",
            minimum_live_decision_round=1,
        )

        self.assertEqual(proof["live_decision_round"], 1)
        self.assertEqual(proof["actual_leaf_continuation_decision_rounds"], 3)
        self.assertEqual(proof["actual_leaf_rollout_rounds"], {"1": 3})

    def test_rejects_the_failing_zero_continuation_input(self) -> None:
        module = _module()
        trajectory = _trajectory(
            metadata={
                "policy_family": "root-puct-search",
                "root_puct_fallback": False,
                "root_puct_total_visits": 1,
                "root_puct_leaf_rollout_rounds": 1,
                "root_puct_leaf_actual_rollout_rounds": {"0": 4},
            }
        )

        with self.assertRaisesRegex(module.ContinuationSmokeError, "no non-fallback Root-PUCT continuation"):
            module.continuation_proof_from_trajectory(
                trajectory,
                pokezero_player="p1",
                minimum_live_decision_round=1,
            )

    def test_rejects_an_opening_only_or_capped_trajectory(self) -> None:
        module = _module()
        metadata = {
            "policy_family": "root-puct-search",
            "root_puct_fallback": False,
            "root_puct_total_visits": 1,
            "root_puct_leaf_rollout_rounds": 1,
            "root_puct_leaf_actual_rollout_rounds": {"1": 1},
        }
        with self.assertRaises(module.ContinuationSmokeError):
            module.continuation_proof_from_trajectory(
                _trajectory(metadata=metadata, turn_index=0),
                pokezero_player="p1",
                minimum_live_decision_round=1,
            )
        with self.assertRaisesRegex(module.ContinuationSmokeError, "capped"):
            module.continuation_proof_from_trajectory(
                _trajectory(metadata=metadata, capped=True),
                pokezero_player="p1",
                minimum_live_decision_round=1,
            )

    def test_refuses_to_replace_an_existing_receipt(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            module._write_new_json(path, {"first": True})
            with self.assertRaises(FileExistsError):
                module._write_new_json(path, {"second": True})


if __name__ == "__main__":
    unittest.main()
