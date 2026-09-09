"""Regression coverage for the MCTS-vs-MCTS process-level handoff."""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mcts_h2h_durable_launcher.py"
SPEC = importlib.util.spec_from_file_location("mcts_h2h_durable_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
_LAUNCHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _LAUNCHER
SPEC.loader.exec_module(_LAUNCHER)


class DurableLauncherTest(unittest.TestCase):
    def _runner(self, directory: Path, *, exit_code: int) -> Path:
        script = directory / "runner.py"
        script.write_text(
            "import sys\n"
            "print('wrapped runner output')\n"
            f"raise SystemExit({exit_code})\n",
            encoding="utf-8",
        )
        return script

    def _argv(self, out_dir: Path, runner: Path, *extra: str) -> list[str]:
        return [
            "--out-dir",
            str(out_dir),
            "--runner-python",
            sys.executable,
            "--runner-script",
            str(runner),
            "--",
            *extra,
        ]

    def test_success_writes_one_parseable_terminal_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "out"
            out_dir.mkdir()
            runner = self._runner(root, exit_code=0)

            self.assertEqual(
                _LAUNCHER.main(self._argv(out_dir, runner, "--checkpoint", "fixture")),
                0,
            )

            terminal_path = out_dir / "runner-terminal.json"
            raw_terminal = terminal_path.read_text(encoding="utf-8")
            terminal = json.loads(raw_terminal)
            log_path = out_dir / "runner-mcts-h2h.log"
            self.assertEqual(
                terminal["schema_version"], _LAUNCHER.RUNNER_TERMINAL_SCHEMA_VERSION
            )
            self.assertIsNotNone(
                datetime.datetime.fromisoformat(terminal["completed_at_utc"])
            )
            self.assertEqual(terminal["status"], "COMPLETE")
            self.assertEqual(terminal["exit_code"], 0)
            self.assertEqual(terminal["runner_log"], str(log_path.resolve()))
            self.assertEqual(terminal["runner_log_bytes"], log_path.stat().st_size)
            self.assertEqual(
                terminal["runner_log_sha256"],
                hashlib.sha256(log_path.read_bytes()).hexdigest(),
            )
            self.assertTrue(raw_terminal.endswith("\n"))
            self.assertNotIn("\\n", raw_terminal)

    def test_failure_is_terminal_and_preserves_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "out"
            out_dir.mkdir()
            runner = self._runner(root, exit_code=7)

            self.assertEqual(_LAUNCHER.main(self._argv(out_dir, runner)), 7)

            terminal = json.loads(
                (out_dir / "runner-terminal.json").read_text(encoding="utf-8")
            )
            self.assertEqual(terminal["status"], "FAILED")
            self.assertEqual(terminal["exit_code"], 7)

    def test_existing_terminal_refuses_without_running_or_replacing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "out"
            out_dir.mkdir()
            runner = self._runner(root, exit_code=0)
            terminal_path = out_dir / "runner-terminal.json"
            terminal_path.write_text('{"preserve":"me"}\n', encoding="utf-8")

            self.assertEqual(_LAUNCHER.main(self._argv(out_dir, runner)), 2)

            self.assertEqual(
                terminal_path.read_text(encoding="utf-8"), '{"preserve":"me"}\n'
            )
            self.assertFalse((out_dir / "runner-mcts-h2h.log").exists())

    def test_rejects_runner_out_dir_override_and_abbreviations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "out"
            out_dir.mkdir()
            runner = self._runner(root, exit_code=0)

            for override in ("--out-dir", "--out", "--out-d", "--out=/redirected"):
                with (
                    self.subTest(override=override),
                    self.assertRaises(SystemExit) as raised,
                ):
                    _LAUNCHER._parse_args(
                        self._argv(out_dir, runner, override, "wrong")
                    )

                self.assertEqual(raised.exception.code, 2)

    def test_rejects_custom_runner_log_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "out"
            out_dir.mkdir()
            runner = self._runner(root, exit_code=0)

            with self.assertRaises(SystemExit) as raised:
                _LAUNCHER._parse_args(
                    self._argv(
                        out_dir,
                        runner,
                        "--runner-log",
                        str(out_dir / "runner-terminal.json"),
                    )
                )

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
