"""Controller resumability + report selection/frontier (plan D6, D8)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pokezero.mcts_eval.controller import (
    EXPERIMENT_SCOPED,
    STAGE_ORDER,
    RetryableFailure,
    Stage,
    TerminalFailure,
    next_incomplete_stage,
    run_pipeline,
    stage_is_complete,
    write_marker,
)
from pokezero.mcts_eval.report import (
    StrengthRow,
    TimingRow,
    pareto_frontier,
    render_markdown_table,
    render_report,
    root_action_agreement,
    select_candidates,
)
from pokezero.mcts_eval.scoring import Interval

EXP = "e" * 64
EXE = "x" * 64


def _timing(depth: int, sims: int, wall: float, *, depth_max: int | None = None, **kw) -> TimingRow:
    values = dict(
        config_id=f"d{depth}-s{sims}-b16-w4-local",
        depth=depth,
        sims=sims,
        decisions_timed=256,
        mean_wall_s=wall,
        median_wall_s=wall,
        p95_wall_s=wall * 1.5,
        max_wall_s=wall * 2,
        realized_depth_mean=float(min(depth, 6)),
        realized_depth_max=depth_max if depth_max is not None else min(depth, 6),
        cap_hit_rate=0.1,
    )
    values.update(kw)
    return TimingRow(**values)


class StageScopeTest(unittest.TestCase):
    def test_first_four_stages_are_experiment_scoped(self) -> None:
        self.assertEqual(
            EXPERIMENT_SCOPED,
            frozenset(
                {
                    Stage.MATERIALIZE_CHECKPOINT,
                    Stage.VALIDATE_CONTRACT,
                    Stage.MECHANICS_SMOKE,
                    Stage.BUILD_TIMING_CORPUS,
                }
            ),
        )

    def test_concurrency_change_preserves_experiment_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for stage in STAGE_ORDER:
                write_marker(temp_dir, stage, experiment_id=EXP, execution_id=EXE)
            new_execution = "y" * 64
            # Experiment-scoped stages survive; execution-scoped ones do not.
            self.assertTrue(
                stage_is_complete(
                    temp_dir, Stage.BUILD_TIMING_CORPUS, experiment_id=EXP, execution_id=new_execution
                )
            )
            self.assertFalse(
                stage_is_complete(
                    temp_dir, Stage.RUN_TIMING_LATTICE, experiment_id=EXP, execution_id=new_execution
                )
            )
            self.assertEqual(
                next_incomplete_stage(temp_dir, experiment_id=EXP, execution_id=new_execution),
                Stage.RUN_TIMING_LATTICE,
            )

    def test_checkpoint_change_invalidates_everything(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for stage in STAGE_ORDER:
                write_marker(temp_dir, stage, experiment_id=EXP, execution_id=EXE)
            self.assertEqual(
                next_incomplete_stage(temp_dir, experiment_id="z" * 64, execution_id=EXE),
                Stage.MATERIALIZE_CHECKPOINT,
            )

    def test_missing_artifact_invalidates_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_marker(
                temp_dir,
                Stage.MATERIALIZE_CHECKPOINT,
                experiment_id=EXP,
                execution_id=EXE,
                artifacts=[str(Path(temp_dir) / "gone.pt")],
            )
            self.assertFalse(
                stage_is_complete(
                    temp_dir, Stage.MATERIALIZE_CHECKPOINT, experiment_id=EXP, execution_id=EXE
                )
            )


class PipelineTest(unittest.TestCase):
    def _handlers(self, calls: list[str], *, failing: Stage | None = None, error=None):
        def make(stage: Stage):
            def handler(directory: Path):
                calls.append(stage.value)
                if stage is failing:
                    raise error
                artifact = directory / "out.json"
                artifact.write_text("{}", encoding="utf-8")
                return [str(artifact)]

            return handler

        return {stage: make(stage) for stage in STAGE_ORDER}

    def test_full_run_then_resume_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            calls: list[str] = []
            status = run_pipeline(
                temp_dir, experiment_id=EXP, execution_id=EXE, handlers=self._handlers(calls)
            )
            self.assertEqual(len(calls), len(STAGE_ORDER))
            self.assertEqual(status["state"], "complete")
            calls.clear()
            run_pipeline(
                temp_dir, experiment_id=EXP, execution_id=EXE, handlers=self._handlers(calls)
            )
            self.assertEqual(calls, [])  # every stage reused

    def test_crash_midway_resumes_without_losing_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            calls: list[str] = []
            with self.assertRaises(TerminalFailure):
                run_pipeline(
                    temp_dir,
                    experiment_id=EXP,
                    execution_id=EXE,
                    handlers=self._handlers(
                        calls, failing=Stage.RUN_TIMING_LATTICE, error=TerminalFailure("boom")
                    ),
                )
            done_before = list(calls)
            calls.clear()
            run_pipeline(
                temp_dir, experiment_id=EXP, execution_id=EXE, handlers=self._handlers(calls)
            )
            # Stages completed before the failure are not re-run.
            self.assertNotIn(Stage.MATERIALIZE_CHECKPOINT.value, calls)
            self.assertIn(Stage.RUN_TIMING_LATTICE.value, calls)
            self.assertIn(Stage.MATERIALIZE_CHECKPOINT.value, done_before)

    def test_retryable_failure_retries_then_becomes_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            calls: list[str] = []
            with self.assertRaisesRegex(TerminalFailure, "exhausted"):
                run_pipeline(
                    temp_dir,
                    experiment_id=EXP,
                    execution_id=EXE,
                    handlers=self._handlers(
                        calls, failing=Stage.MECHANICS_SMOKE, error=RetryableFailure("flaky")
                    ),
                    max_retries=3,
                )
            self.assertEqual(calls.count(Stage.MECHANICS_SMOKE.value), 3)
            status = json.loads(Path(temp_dir, "status.json").read_text())
            self.assertEqual(status["state"], "failed")
            self.assertIn("retry budget", status["terminal_failure"])

    def test_terminal_failure_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            calls: list[str] = []
            with self.assertRaises(TerminalFailure):
                run_pipeline(
                    temp_dir,
                    experiment_id=EXP,
                    execution_id=EXE,
                    handlers=self._handlers(
                        calls, failing=Stage.VALIDATE_CONTRACT, error=TerminalFailure("drift")
                    ),
                )
            self.assertEqual(calls.count(Stage.VALIDATE_CONTRACT.value), 1)


class EligibilityAndSelectionTest(unittest.TestCase):
    def test_gate_and_zero_fallback_requirements(self) -> None:
        self.assertTrue(_timing(4, 1024, 3.0).eligible)
        self.assertFalse(_timing(4, 1024, 15.0).eligible)  # not strictly below
        self.assertFalse(_timing(4, 1024, 3.0, fallbacks=1).eligible)
        self.assertFalse(_timing(4, 1024, 3.0, invalid_actions=1).eligible)
        self.assertFalse(_timing(4, 1024, 3.0, provenance_exact=False).eligible)
        self.assertFalse(_timing(4, 1024, 3.0, gate_failed=True).eligible)

    def test_selection_caps_at_seven_and_prefers_largest_sims(self) -> None:
        rows = [_timing(d, s, wall=0.5 * d + s / 4096) for d in (2, 4, 6, 8, 10) for s in (512, 8192)]
        chosen = select_candidates(rows)
        self.assertLessEqual(len(chosen), 7)
        by_depth = {row.depth: row for row in chosen if row.sims == 8192}
        self.assertTrue(by_depth, "largest eligible sims per depth must be retained")

    def test_gate_failed_cells_never_selected(self) -> None:
        rows = [_timing(2, 512, 1.0), _timing(10, 8192, 40.0, gate_failed=True)]
        self.assertEqual([row.config_id for row in select_candidates(rows)], ["d2-s512-b16-w4-local"])

    def test_dominated_cell_pruned(self) -> None:
        # Same breadth: slower AND no deeper -> dominated.
        fast = _timing(6, 2048, 2.0, depth_max=5)
        slow = _timing(8, 2048, 5.0, depth_max=5)
        chosen = {row.config_id for row in select_candidates([fast, slow])}
        self.assertIn(fast.config_id, chosen)


class FrontierTest(unittest.TestCase):
    def _strength(self, config_id: str, score: float, wall: float) -> StrengthRow:
        return StrengthRow(
            config_id=config_id,
            foulplay_rung="FP-1000",
            record={"win": 50, "tie": 0, "cap": 0, "loss": 50},
            score=Interval(score, score - 0.07, score + 0.07),
            delta_vs_raw=Interval(0.02, -0.03, 0.07),
            timing=_timing(4, 1024, wall),
        )

    def test_frontier_excludes_slower_and_weaker(self) -> None:
        rows = [
            self._strength("fast-strong", 0.60, 2.0),
            self._strength("slow-weak", 0.50, 9.0),
            self._strength("slow-stronger", 0.66, 9.0),
        ]
        frontier = {row.config_id for row in pareto_frontier(rows)}
        self.assertIn("fast-strong", frontier)
        self.assertIn("slow-stronger", frontier)
        self.assertNotIn("slow-weak", frontier)

    def test_report_and_table_render(self) -> None:
        rows = [self._strength("d4-s1024-b16-w4-local", 0.55, 3.0)]
        payload = render_report(rows, timing_rows=[_timing(4, 1024, 3.0)])
        self.assertEqual(payload["pareto_frontier"], ["d4-s1024-b16-w4-local"])
        self.assertIn("screening sample", " ".join(payload["notes"]))
        table = render_markdown_table(rows)
        self.assertIn("d4-s1024-b16-w4-local", table)
        self.assertIn("parity", table)

    def test_root_action_agreement(self) -> None:
        left = _timing(4, 1024, 1.0, root_argmax_by_decision=("a", "b", "c"))
        right = _timing(6, 1024, 2.0, root_argmax_by_decision=("a", "b", "z"))
        self.assertAlmostEqual(root_action_agreement(left, right), 2 / 3)
        self.assertIsNone(root_action_agreement(left, _timing(8, 1024, 3.0)))


if __name__ == "__main__":
    unittest.main()
