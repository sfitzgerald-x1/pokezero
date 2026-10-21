"""Unit pins for the image-time FoulPlay poke-engine capability check."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_foulplay_poke_engine", ROOT / "scripts" / "verify_foulplay_poke_engine.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Record:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _Engine:
    Pokemon = _Record
    Move = _Record
    Side = _Record
    State = _Record

    def __init__(self, *, visits: int) -> None:
        self.visits = visits
        self.calls: list[tuple[object, int, int, int]] = []

    def monte_carlo_tree_search(
        self, state: object, duration_ms: int, iterations: int, threads: int
    ) -> object:
        self.calls.append((state, duration_ms, iterations, threads))
        return SimpleNamespace(
            total_visits=self.visits,
            side_one=("sleeptalk",),
            side_two=("tackle",),
        )


class FoulPlayPokeEngineVerifierTest(unittest.TestCase):
    def test_fixture_reaches_the_exhausted_ordinary_sleep_counter(self) -> None:
        engine = _Engine(visits=128)
        MODULE.verify(engine, iterations=128)

        state, duration_ms, iterations, threads = engine.calls[0]
        self.assertEqual((duration_ms, iterations, threads), (1, 128, 1))
        sleeper = state.side_one.pokemon[0]
        self.assertEqual((sleeper.status, sleeper.rest_turns, sleeper.sleep_turns), ("sleep", 0, 5))
        self.assertEqual([move.id for move in sleeper.moves], ["sleeptalk", "rest"])

    def test_empty_search_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not complete any visits"):
            MODULE.verify(_Engine(visits=0), iterations=128)

    def test_invalid_iteration_count_is_refused_before_search(self) -> None:
        engine = _Engine(visits=0)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    MODULE.verify(engine, iterations=value)
        self.assertEqual(engine.calls, [])


class FoulPlaySetupTest(unittest.TestCase):
    def _setup(self, *, build_exit: int = 0, verify_exit: int = 0) -> tuple[int, str, list[str]]:
        """Execute the real installer with only external commands/builds stubbed."""
        with tempfile.TemporaryDirectory(prefix="foulplay setup ") as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copyfile(ROOT / "scripts/setup_foulplay_eval.sh", scripts / "setup_foulplay_eval.sh")
            commands = root / "commands"
            commands.mkdir()
            log = root / "calls"
            venv = root / "separate opponent venv"
            (venv / "bin").mkdir(parents=True)
            stubs = {
                commands / "git": '#!/bin/sh\nexit 0\n',
                commands / "uv": '#!/bin/sh\necho "uv:$*" >> "$CALLS"\n',
                scripts / "setup_poke_engine.sh": (
                    '#!/bin/sh\nprintf "build:%s\\n" "$1" >> "$CALLS"\nexit "$BUILD_EXIT"\n'
                ),
                venv / "bin/python": (
                    '#!/bin/sh\nprintf "verify:%s:%s:%s\\n" "$1" "$2" "$3" >> "$CALLS"\n'
                    'exit "$VERIFY_EXIT"\n'
                ),
            }
            for path, source in stubs.items():
                path.write_text(source, encoding="utf-8")
                path.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
                "FOULPLAY_VENV": str(venv),
                "CALLS": str(log),
                "BUILD_EXIT": str(build_exit),
                "VERIFY_EXIT": str(verify_exit),
            }
            result = subprocess.run(
                ["bash", str(scripts / "setup_foulplay_eval.sh")],
                env=env, text=True, capture_output=True, check=False,
            )
            calls = log.read_text(encoding="utf-8").splitlines()
            # Normalize the temporary root, retaining spaces to exercise quoting.
            calls = [line.replace(str(root), "ROOT") for line in calls]
            return result.returncode, result.stdout + result.stderr, calls

    def test_installs_canonical_patched_engine_then_checks_its_own_interpreter(self) -> None:
        code, output, calls = self._setup()
        self.assertEqual(code, 0, output)
        self.assertEqual(calls[-2:], [
            "build:ROOT/separate opponent venv/bin/python",
            "verify:ROOT/scripts/verify_foulplay_poke_engine.py:--iterations:128",
        ])
        self.assertEqual(len(calls), 4)  # venv, dependencies, canonical build, fixture
        self.assertNotIn("poke-engine==", "\n".join(calls))
        self.assertIn("gen3 eval ready", output)

    def test_builder_failure_stops_before_fixture_and_readiness(self) -> None:
        code, output, calls = self._setup(build_exit=23)
        self.assertEqual(code, 23)
        self.assertTrue(calls[-1].startswith("build:"))
        self.assertFalse(any(line.startswith("verify:") for line in calls))
        self.assertNotIn("gen3 eval ready", output)

    def test_fixture_failure_stops_before_readiness(self) -> None:
        code, output, calls = self._setup(verify_exit=24)
        self.assertEqual(code, 24)
        self.assertTrue(calls[-1].startswith("verify:"))
        self.assertNotIn("gen3 eval ready", output)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
