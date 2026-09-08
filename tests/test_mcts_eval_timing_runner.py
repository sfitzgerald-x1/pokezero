"""Runner-level durability checks for the timing-corpus launch gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from pokezero.mcts_eval.controller import Stage, write_marker
from pokezero.mcts_eval.manifest import MatrixManifest, ResourceProfile, SearchConfig
from pokezero.mcts_eval.resolver import CheckpointContract
from pokezero.mcts_eval.timing_corpus import (
    TimingDecisionRecord,
    build_corpus,
    label_strata,
    write_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mcts_timing_runner_under_test", ROOT / "scripts" / "run_mcts_depth_eval.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _contract() -> CheckpointContract:
    return CheckpointContract(
        checkpoint_path="/checkpoint.pt",
        checkpoint_sha256="a" * 64,
        policy_id="timing-runner-test",
        schema_version="pokezero.observation.v3",
        token_count=1,
        categorical_feature_count=1,
        numeric_feature_count=1,
        transition_token_count=0,
        category_vocab=("test",),
        architecture={},
        feature_masks={},
        model_device="cpu",
    )


def _homogeneous_record(index: int) -> TimingDecisionRecord:
    mask = (True, True, False, False, False, False, False, False, False)
    return TimingDecisionRecord(
        decision_id=f"one-battle-{index:03d}",
        battle_id="one-battle",
        seat="p1" if index % 2 == 0 else "p2",
        turn_index=0,
        team_seed=1,
        battle_seed=2,
        bot_rng_seed=3,
        event_prefix=("|start|",),
        public_resolved_action_rounds=(),
        action_candidates=({"kind": "move", "move_id": "surf", "slot": 1},),
        legal_action_mask=mask,
        public_belief_inputs={},
        strata=label_strata(
            remaining=6,
            team_hp_fraction=1.0,
            boosts=None,
            forced_switch=False,
            hidden_world_count=0,
            turn_index=0,
            legal_action_count=sum(mask),
        ),
    )


class TimingRunnerDurabilityTest(unittest.TestCase):
    def _write_nonrepresentative_corpus(self, out_root: Path) -> Path:
        corpus_path = out_root / Stage.BUILD_TIMING_CORPUS.value / "timing-corpus.jsonl"
        manifest, records = build_corpus(
            [_homogeneous_record(index) for index in range(16)],
            held_out_seed_start=1,
            held_out_seed_end=1,
            count=16,
        )
        write_corpus(corpus_path, manifest, records)
        return corpus_path

    @staticmethod
    def _ids() -> tuple[str, str]:
        contract = _contract()
        configs = tuple(
            SearchConfig(depth=depth, sims=sims)
            for depth in (2, 4, 6, 8, 10)
            for sims in (512, 1024, 2048, 4096, 8192)
        )
        manifest = MatrixManifest(
            checkpoint_manifest=contract.to_manifest(),
            configs=configs,
            resource_profile=ResourceProfile(concurrency=1, torch_threads=1),
            worlds=4,
            seed_band="default",
            corpus_decisions=256,
        )
        return manifest.experiment_id, manifest.execution_id

    def test_nonrepresentative_corpus_persists_terminal_failure_before_timing(self) -> None:
        """A launch refusal must never strand the resumable status as running."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_root = Path(temp_dir)
            self._write_nonrepresentative_corpus(out_root)
            fake_search = types.ModuleType("pokezero_search")
            fake_search.NativeLeafModel = object

            with (
                mock.patch.object(runner, "resolve_checkpoint_contract", return_value=_contract()),
                mock.patch.dict(sys.modules, {"pokezero_search": fake_search}),
            ):
                exit_code = runner.main(
                    [
                        "--checkpoint", "/checkpoint.pt",
                        "--out-root", str(out_root),
                        "--stage-through", Stage.BUILD_TIMING_CORPUS.value,
                    ]
                )

            self.assertEqual(exit_code, 2)
            status = json.loads((out_root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["stage"], Stage.BUILD_TIMING_CORPUS.value)
            self.assertIn("timing corpus rejected", status["terminal_failure"])
            self.assertIn("not representative", status["terminal_failure"])
            self.assertFalse((out_root / Stage.RUN_TIMING_LATTICE.value).exists())

    def test_cached_corpus_is_rejected_and_persisted_before_lattice_timing(self) -> None:
        """A stale completion marker cannot bypass the immediate launch recheck."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_root = Path(temp_dir)
            corpus_path = self._write_nonrepresentative_corpus(out_root)
            experiment_id, execution_id = self._ids()
            for stage in (
                Stage.MATERIALIZE_CHECKPOINT,
                Stage.VALIDATE_CONTRACT,
                Stage.MECHANICS_SMOKE,
            ):
                write_marker(
                    out_root,
                    stage,
                    experiment_id=experiment_id,
                    execution_id=execution_id,
                )
            write_marker(
                out_root,
                Stage.BUILD_TIMING_CORPUS,
                experiment_id=experiment_id,
                execution_id=execution_id,
                artifacts=(str(corpus_path),),
            )

            with mock.patch.object(runner, "resolve_checkpoint_contract", return_value=_contract()):
                exit_code = runner.main(
                    [
                        "--checkpoint", "/checkpoint.pt",
                        "--out-root", str(out_root),
                        "--stage-through", Stage.RUN_TIMING_LATTICE.value,
                    ]
                )

            self.assertEqual(exit_code, 2)
            status = json.loads((out_root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["stage"], Stage.RUN_TIMING_LATTICE.value)
            self.assertIn("timing corpus rejected", status["terminal_failure"])
            self.assertIn("not representative", status["terminal_failure"])
            lattice_dir = out_root / Stage.RUN_TIMING_LATTICE.value
            self.assertFalse(any(lattice_dir.glob("timing-*.json")))


if __name__ == "__main__":
    unittest.main()
