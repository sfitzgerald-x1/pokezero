"""Regression coverage for the MCTS-vs-MCTS process-level handoff."""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import subprocess
from pathlib import Path
import sys
import tempfile
import time
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

    def _gated_runner(
        self,
        directory: Path,
        started_marker: Path,
        release_marker: Path,
        finished_marker: Path,
    ) -> Path:
        script = directory / "gated_runner.py"
        script.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "import time\n"
            f"Path({str(started_marker)!r}).write_text('started\\n', encoding='utf-8')\n"
            "deadline = time.monotonic() + 5\n"
            f"while not Path({str(release_marker)!r}).exists():\n"
            "    if time.monotonic() >= deadline:\n"
            "        raise SystemExit('timed out waiting for test release')\n"
            "    time.sleep(0.01)\n"
            f"Path({str(finished_marker)!r}).write_text('finished\\n', encoding='utf-8')\n"
            "print('wrapped runner output')\n",
            encoding="utf-8",
        )
        return script

    def _argv(
        self, out_dir: Path, runner: Path, *extra: str, attempt_id: str = "attempt-a"
    ) -> list[str]:
        return [
            "--out-dir",
            str(out_dir),
            "--runner-python",
            sys.executable,
            "--runner-script",
            str(runner),
            "--attempt-id",
            attempt_id,
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
            attempt_path = out_dir / "launcher-attempts" / "attempt-a.json"
            log_path = out_dir / "launcher-attempts" / "attempt-a.log"
            self.assertEqual(
                terminal["schema_version"], _LAUNCHER.RUNNER_TERMINAL_SCHEMA_VERSION
            )
            self.assertIsNotNone(
                datetime.datetime.fromisoformat(terminal["completed_at_utc"])
            )
            self.assertEqual(terminal["status"], "COMPLETE")
            self.assertEqual(terminal["exit_code"], 0)
            self.assertEqual(terminal["attempt_id"], "attempt-a")
            self.assertEqual(terminal["attempt_receipt"], str(attempt_path.resolve()))
            self.assertEqual(
                terminal["attempt_receipt_sha256"],
                hashlib.sha256(attempt_path.read_bytes()).hexdigest(),
            )
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
            self.assertFalse((out_dir / "launcher-attempts").exists())

    def test_incomplete_prior_attempt_is_preserved_and_does_not_block_new_attempt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "out"
            attempts = out_dir / "launcher-attempts"
            attempts.mkdir(parents=True)
            old_receipt = attempts / "interrupted.json"
            old_log = attempts / "interrupted.log"
            old_receipt.write_text('{"preserve":"attempt"}\n', encoding="utf-8")
            old_log.write_text("partial runner output\n", encoding="utf-8")
            runner = self._runner(root, exit_code=0)

            self.assertEqual(
                _LAUNCHER.main(
                    self._argv(out_dir, runner, attempt_id="replacement-attempt")
                ),
                0,
            )

            self.assertEqual(
                old_receipt.read_text(encoding="utf-8"), '{"preserve":"attempt"}\n'
            )
            self.assertEqual(
                old_log.read_text(encoding="utf-8"), "partial runner output\n"
            )
            self.assertTrue((attempts / "replacement-attempt.json").is_file())
            self.assertTrue((attempts / "replacement-attempt.log").is_file())

    def test_killed_launcher_cannot_be_recovered_while_child_still_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "out"
            out_dir.mkdir()
            started_marker = root / "child-started"
            release_marker = root / "allow-child-exit"
            finished_marker = root / "child-finished"
            runner = self._gated_runner(
                root, started_marker, release_marker, finished_marker
            )
            first = subprocess.Popen(
                [sys.executable, str(SCRIPT), *self._argv(out_dir, runner)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 5
                while not started_marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(started_marker.is_file())
                self.assertIsNone(first.poll())

                first.terminate()
                first.wait(timeout=5)
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        _LAUNCHER.main(
                            self._argv(out_dir, runner, attempt_id="recovery-attempt")
                        ),
                        2,
                    )
                self.assertFalse((out_dir / "runner-terminal.json").exists())

                release_marker.write_text("release\n", encoding="utf-8")
                deadline = time.monotonic() + 5
                while not finished_marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(finished_marker.is_file())
                while time.monotonic() < deadline:
                    try:
                        with _LAUNCHER._writer_lock(out_dir):
                            break
                    except _LAUNCHER.LauncherError:
                        time.sleep(0.01)
                else:
                    self.fail("orphaned scorer did not release its writer lock")
                self.assertEqual(
                    _LAUNCHER.main(
                        self._argv(out_dir, runner, attempt_id="recovery-attempt")
                    ),
                    0,
                )
            finally:
                if first.poll() is None:
                    first.kill()
                    first.wait(timeout=5)

    def test_reused_attempt_id_refuses_without_replacing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "out"
            attempts = out_dir / "launcher-attempts"
            attempts.mkdir(parents=True)
            receipt = attempts / "used.json"
            receipt.write_text('{"preserve":"attempt"}\n', encoding="utf-8")
            runner = self._runner(root, exit_code=0)

            self.assertEqual(
                _LAUNCHER.main(self._argv(out_dir, runner, attempt_id="used")), 2
            )

            self.assertEqual(
                receipt.read_text(encoding="utf-8"), '{"preserve":"attempt"}\n'
            )
            self.assertFalse((attempts / "used.log").exists())

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
