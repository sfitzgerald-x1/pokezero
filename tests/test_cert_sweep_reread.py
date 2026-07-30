"""Fail-closed input-contract tests for scripts/cert_sweep_reread.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cert_sweep_reread.py"


def _load():
    spec = importlib.util.spec_from_file_location("cert_sweep_reread", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


REREAD = _load()


def _shard(rows: list[dict[str, object]], *, complete: bool = True,
           retained: int | None = None, diverged: int | None = None) -> dict[str, object]:
    count = len(rows)
    return {
        "repro_retention": {
            "repros_complete": complete,
            "repros_retained": count if retained is None else retained,
            "transitions_diverged": count if diverged is None else diverged,
        },
        "repros": rows,
    }


class RetainedInputContractTests(unittest.TestCase):
    def _write_shard(self, directory: Path, name: str, payload: dict[str, object]) -> Path:
        path = directory / name
        path.write_text(json.dumps(payload))
        return path

    def test_default_contract_targets_the_certification_archive(self) -> None:
        self.assertEqual(REREAD.DEFAULT_EXPECTED_ROWS, 3821)

    def test_empty_glob_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "no shard files matched"):
                REREAD.load_retained_rows(f"{temp}/cert_shard_*.json", expected_rows=3821)

    def test_main_fails_before_writing_output_for_an_empty_glob(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            output = directory / "reread.json"
            code = REREAD.main([
                "--shards", f"{directory}/cert_shard_*.json",
                "--json", str(output),
            ])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())

    def test_declared_complete_input_with_a_missing_row_is_rejected(self) -> None:
        rows = [{"kind": "transition_diverged"}]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._write_shard(
                directory,
                "cert_shard_01.json",
                _shard(rows, retained=2, diverged=2),
            )
            with self.assertRaisesRegex(ValueError, "1 transition rows != 2 declared"):
                REREAD.load_retained_rows(f"{directory}/cert_shard_*.json", expected_rows=2)

    def test_complete_input_matching_the_expected_population_is_accepted(self) -> None:
        rows = [
            {"kind": "transition_diverged", "seed": 1, "step": 2},
            {"kind": "transition_diverged", "seed": 3, "step": 4},
        ]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._write_shard(directory, "cert_shard_01.json", _shard(rows))
            loaded = REREAD.load_retained_rows(
                f"{directory}/cert_shard_*.json", expected_rows=2,
            )
        self.assertEqual(loaded, rows)

    def test_duplicate_transition_identity_cannot_satisfy_expected_count(self) -> None:
        rows = [
            {"kind": "transition_diverged", "seed": 1, "step": 2},
            {"kind": "transition_diverged", "seed": 1, "step": 2},
        ]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._write_shard(directory, "cert_shard_01.json", _shard(rows))
            with self.assertRaisesRegex(ValueError, "duplicate transition identity"):
                REREAD.load_retained_rows(
                    f"{directory}/cert_shard_*.json", expected_rows=2,
                )

    def test_main_fails_on_reread_error_without_expect_flag(self) -> None:
        rows = [{"kind": "transition_diverged", "seed": 1, "step": 2}]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            output = directory / "reread.json"
            self._write_shard(directory, "cert_shard_01.json", _shard(rows))
            with patch.object(REREAD, "reread_row", side_effect=ValueError("broken row")):
                code = REREAD.main([
                    "--shards", f"{directory}/cert_shard_*.json",
                    "--json", str(output),
                    "--expected-rows", "1",
                ])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.read_text())["tally"], {"reread_error": 1})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
