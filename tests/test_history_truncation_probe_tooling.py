from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HistorySlotCensusTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import numpy  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("requires numpy")
        self.census = _load("history_slot_census")

    def test_counts_fill_and_drops_padding(self) -> None:
        import numpy as np

        offset = self.census.TRANSITION_TOKEN_OFFSET
        token_count = offset + self.census.TRANSITION_TOKEN_COUNT
        rows = [np.zeros(token_count, dtype=bool)]  # pad row -> dropped
        for filled in range(41):
            row = np.zeros(token_count, dtype=bool)
            row[:offset] = True  # non-transition prefix always attended
            row[offset : offset + filled] = True
            rows.append(row)
        counts = self.census.transition_fill_counts(np.stack(rows), np)
        # 41 real rows with 0..40 filled; the all-False pad row is excluded.
        self.assertEqual(int(counts.size), 41)
        self.assertEqual(sorted(counts.tolist()), list(range(41)))

    def test_summary_matches_expected_stats(self) -> None:
        import numpy as np

        counts = np.arange(0, 41)
        summary = self.census._summary(counts, np)
        self.assertEqual(summary["n"], 41)
        self.assertEqual(summary["mean"], 20.0)
        self.assertEqual(summary["max"], 40)
        # 8 of 41 rows (33..40) exceed 32.
        self.assertAlmostEqual(summary["pct_gt_32"], 100 * 8 / 41, places=1)


class AnalyzeHistoryTruncationProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.analyze = _load("analyze_history_truncation_probe")

    @staticmethod
    def _cell(checkpoint: str, k: int, win_rates: dict[str, float]) -> dict:
        head_to_heads = []
        for opponent, win_rate in win_rates.items():
            head_to_heads.append(
                {
                    "first_policy_id": checkpoint,
                    "second_policy_id": opponent,
                    "games": 1000,
                    "first_policy_wins": round(win_rate * 1000),
                    "second_policy_wins": round((1 - win_rate) * 1000),
                    "ties": 0,
                    "first_policy_win_rate": win_rate,
                    "second_policy_win_rate": 1 - win_rate,
                }
            )
        payload: dict = {"head_to_heads": head_to_heads}
        if k != 128:
            payload["history_mask_k"] = k
        return payload

    def _verdict(self, cells: list[dict]) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, cell in enumerate(cells):
                path = Path(directory) / f"cell{index}.json"
                path.write_text(json.dumps(cell))
                paths.append(path)
            return self.analyze.analyze(self.analyze._load_cells(paths))

    def test_flat_curve_recommends_truncation(self) -> None:
        base = {"max-damage": 0.72, "simple-legal": 0.96, "random-legal": 0.99}
        cells = [self._cell("m50", k, base) for k in (128, 64, 32, 16)]
        report = self._verdict(cells)
        self.assertEqual(report["verdict"], "flat")
        self.assertEqual(report["k_stars_by_checkpoint"]["m50"], 16)

    def test_small_k_degradation_keeps_128(self) -> None:
        cells = [
            self._cell("m50", 128, {"max-damage": 0.72, "simple-legal": 0.96, "random-legal": 0.99}),
            self._cell("m50", 64, {"max-damage": 0.71, "simple-legal": 0.96, "random-legal": 0.99}),
            self._cell("m50", 32, {"max-damage": 0.66, "simple-legal": 0.95, "random-legal": 0.99}),
            self._cell("m50", 16, {"max-damage": 0.60, "simple-legal": 0.94, "random-legal": 0.98}),
        ]
        report = self._verdict(cells)
        self.assertEqual(report["verdict"], "degraded")
        self.assertIn(16, report["checkpoints"]["m50"]["degraded_ks"])

    def test_class_dependent_is_mixed(self) -> None:
        base = {"max-damage": 0.72, "simple-legal": 0.96, "random-legal": 0.99}
        cells = [self._cell("m50", k, base) for k in (128, 64, 32, 16)]
        cells += [
            self._cell("S", 128, base),
            self._cell("S", 64, base),
            self._cell("S", 32, {"max-damage": 0.66, "simple-legal": 0.96, "random-legal": 0.99}),
            self._cell("S", 16, {"max-damage": 0.60, "simple-legal": 0.95, "random-legal": 0.99}),
        ]
        report = self._verdict(cells)
        self.assertEqual(report["verdict"], "mixed")

    def test_infers_checkpoint_from_common_policy(self) -> None:
        base = {"max-damage": 0.72, "simple-legal": 0.96, "random-legal": 0.99}
        cells = self.analyze._load_cells(
            [self._write(self._cell("m50", 128, base))]
        )
        self.assertEqual(cells[0]["checkpoint_id"], "m50")
        self.assertEqual(cells[0]["k"], 128)

    def _write(self, cell: dict) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "cell.json"
        path.write_text(json.dumps(cell))
        return path


if __name__ == "__main__":
    unittest.main()
