"""Pure fixtures for the raw PokeZero / FoulPlay 1000-ms vs 760-ms reader."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import foulplay_paired_eval as _E  # noqa: E402
from pokezero.foulplay_bridge import (  # noqa: E402
    FOULPLAY_THINK_SCHEMA_VERSION,
    foulplay_think_reading_status,
)

_SPEC = importlib.util.spec_from_file_location(
    "foulplay_budget_calibration_test",
    REPO_ROOT / "scripts" / "foulplay_budget_calibration.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_C = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_C)

CALIBRATION_ID = "ceiling-24pct"
LAYOUT = "cpu1-c3"
CHECKPOINT = "/shared/checkpoints/k0.pt"
CHECKPOINT_TAG = "k0"


def scores(value: float, *, start: int = 11, pairs: int = 2) -> dict[tuple[int, str], float]:
    return {
        (seed, seat): value
        for seed in range(start, start + pairs)
        for seat in ("p1", "p2")
    }


def shard(
    condition: str,
    values: dict[tuple[int, str], float],
    *,
    start: int = 11,
    pairs: int = 2,
    layout: str = LAYOUT,
) -> dict:
    budget = {"baseline": 1000, "reduced": 760}[condition]
    config_id = _E.foulplay_budget_calibration_config_id(
        checkpoint=CHECKPOINT,
        checkpoint_tag_value=CHECKPOINT_TAG,
        calibration_id=CALIBRATION_ID,
        layout=layout,
        foulplay_search_time_ms=budget,
    )
    metadata = {
        "schema_version": _E.FOULPLAY_BUDGET_CALIBRATION_SCHEMA_VERSION,
        "calibration_id": CALIBRATION_ID,
        "resource_layout": layout,
        "checkpoint_tag": CHECKPOINT_TAG,
        "condition": condition,
        "foulplay_search_time_ms": budget,
        "baseline_foulplay_search_time_ms": 1000,
        "reduced_foulplay_search_time_ms": 760,
        "budget_cut_fraction": 0.24,
        "thread_pin": dict(sorted(_E.THREAD_PIN_ENV.items())),
        "config_id": config_id,
    }
    think = {
        "schema_version": FOULPLAY_THINK_SCHEMA_VERSION,
        "mean_iterations_per_budget_second": 500.0,
        "iterations_measured_decisions": 100,
        "iterations_coverage": 1.0,
        "iterations_observable": True,
        "record_failures": 0,
    }
    return {
        "schema_version": "pokezero.foulplay-paired-shard.v1",
        "arm": "raw",
        "config_id": config_id,
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": "c" * 64,
        "engine_fingerprint": "f" * 64,
        "commit": "a" * 40,
        "opponent_policy_mode": "foul-play",
        "foulplay_search_time_ms": budget,
        "foulplay_budget_calibration": metadata,
        "seed_start": start,
        "pairs": pairs,
        "per_seat": {
            seat: {
                "foulplay_think": dict(think),
                "foulplay_think_reading": foulplay_think_reading_status(think),
            }
            for seat in ("p1", "p2")
        },
        "rows": [
            {"seed": seed, "seat": seat, "score": value}
            for (seed, seat), value in sorted(values.items())
        ],
    }


def run(shards: list[dict], *, min_pairs: int = 1) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for index, payload in enumerate(shards):
            path = Path(directory) / f"shard-{index}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(str(path))
        out = Path(directory) / "report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            _C.main(
                paths
                + [
                    "--calibration-id",
                    CALIBRATION_ID,
                    "--resource-layout",
                    LAYOUT,
                    "--min-pairs",
                    str(min_pairs),
                    "--out",
                    str(out),
                ]
            )
        return json.loads(out.read_text(encoding="utf-8"))


class FoulPlayBudgetCalibrationTest(unittest.TestCase):
    def test_valid_mirrored_conditions_score_the_named_24_percent_cut(self) -> None:
        report = run([
            shard("baseline", scores(0.0)),
            shard("reduced", scores(1.0)),
        ])
        self.assertEqual(report["schema_version"], _C.SCHEMA_VERSION)
        self.assertEqual(
            report["calibration"]["foulplay_budget_ms"],
            {"baseline": 1000, "reduced": 760, "cut_fraction": 0.24},
        )
        comparison = report["paired_comparison"]
        self.assertEqual(comparison["pairs"], 4)
        self.assertEqual(comparison["reduced_minus_baseline"]["point"], 1.0)
        self.assertEqual(comparison["discordant"]["reduced_higher_score"], 4)
        self.assertIn("760 ms minus score at 1000 ms", comparison["estimand"])

    def test_default_minimum_refuses_a_small_fixture_instead_of_calling_it_a_verdict(self) -> None:
        with self.assertRaisesRegex(SystemExit, "mirrored pairs < required minimum"):
            run([
                shard("baseline", scores(0.0)),
                shard("reduced", scores(1.0)),
            ], min_pairs=_C.MIN_PAIRS)

    def test_a_wrong_budget_cannot_be_relabelled_as_the_named_cut(self) -> None:
        reduced = shard("reduced", scores(1.0))
        reduced["foulplay_search_time_ms"] = 750
        reduced["foulplay_budget_calibration"]["foulplay_search_time_ms"] = 750
        with self.assertRaisesRegex(SystemExit, "does not match its named budget"):
            run([shard("baseline", scores(0.0)), reduced])

    def test_missing_or_duplicate_pairs_are_terminal_not_dropped(self) -> None:
        incomplete = shard("reduced", scores(1.0))
        incomplete["rows"].pop()
        with self.assertRaisesRegex(SystemExit, "exact mirrored seed band"):
            run([shard("baseline", scores(0.0)), incomplete])

        baseline = shard("baseline", scores(0.0))
        with self.assertRaisesRegex(SystemExit, "duplicate calibration pair"):
            run([baseline, copy.deepcopy(baseline), shard("reduced", scores(1.0))])

    def test_layout_and_run_identity_must_match_between_conditions(self) -> None:
        with self.assertRaisesRegex(SystemExit, "one non-empty resource_layout"):
            run([
                shard("baseline", scores(0.0)),
                shard("reduced", scores(1.0), layout="cpu2-c3"),
            ])

        changed_build = shard("reduced", scores(1.0))
        changed_build["engine_fingerprint"] = "e" * 64
        with self.assertRaisesRegex(SystemExit, "one non-empty engine_fingerprint"):
            run([shard("baseline", scores(0.0)), changed_build])

    def test_unmeasured_foulplay_think_is_refused(self) -> None:
        reduced = shard("reduced", scores(1.0))
        reduced["per_seat"]["p2"]["foulplay_think"]["iterations_measured_decisions"] = 0
        with self.assertRaisesRegex(SystemExit, "admissible FoulPlay think reading"):
            run([shard("baseline", scores(0.0)), reduced])

    def test_forged_ok_reading_cannot_replace_the_producer_derived_witness(self) -> None:
        reduced = shard("reduced", scores(1.0))
        reduced["per_seat"]["p2"]["foulplay_think_reading"] = {"usable": True}
        with self.assertRaisesRegex(SystemExit, "admissible FoulPlay think reading"):
            run([shard("baseline", scores(0.0)), reduced])
