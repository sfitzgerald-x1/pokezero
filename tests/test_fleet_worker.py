"""Persistent collector-fleet worker: queue protocol + recycle bounds.

All torch-free — the collect function is stubbed; what is under test is the
claim/commit/revocation/failed transport (which must stay byte-compatible with
the shell fleet worker) and the OOM/task recycle bounds.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokezero import fleet_worker  # noqa: E402
from pokezero.fleet_worker import claim_next_task, run_worker  # noqa: E402


def _make_queue(root: Path) -> Path:
    queue = root / "collect-queue"
    for sub in ("pending", "claimed", "done", "failed"):
        (queue / sub).mkdir(parents=True)
    return queue


def _manifest(queue: Path, base: str, *, out: Path, iteration: int = 7,
              count: int = 2, seed: int = 4321, policy: str = "remote:http://svc:8600") -> Path:
    path = queue / "pending" / base
    path.write_text(
        f'a_iter={iteration}\na_offset=0\na_count={count}\n'
        f'a_seed={seed}\na_out="{out}"\na_policy={policy}\n',
        encoding="utf-8",
    )
    return path


class ClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.queue = _make_queue(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_claim_parses_and_renames(self) -> None:
        out = self.root / "cache" / "shard-f0"
        _manifest(self.queue, "i7-s0.env", out=out)
        task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(task)
        self.assertEqual(task.base, "i7-s0.env")
        self.assertEqual((task.iteration, task.count, task.seed), (7, 2, 4321))
        self.assertEqual(task.out, out)  # quoted value unwrapped
        self.assertEqual(task.policy, "remote:http://svc:8600")
        self.assertFalse((self.queue / "pending" / "i7-s0.env").exists())
        self.assertTrue(task.claim_path.exists())

    def test_empty_queue_returns_none(self) -> None:
        self.assertIsNone(claim_next_task(self.queue, "w1"))

    def test_malformed_manifest_parks_in_failed(self) -> None:
        (self.queue / "pending" / "i7-s9.env").write_text("garbage\n", encoding="utf-8")
        self.assertIsNone(claim_next_task(self.queue, "w1"))
        self.assertTrue((self.queue / "failed" / "i7-s9.env.w1.failed").exists())


class WorkerLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.queue = _make_queue(self.root)
        self.calls: list[list[str]] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _stub(self, *, returncode: int = 0, revoke: bool = False):
        queue = self.queue

        def collect_fn(argv: list[str]) -> int:
            self.calls.append(argv)
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "metadata.json").write_text("{}", encoding="utf-8")
            if revoke:  # controller revoked the claim mid-task
                for claim in (queue / "claimed").iterdir():
                    claim.unlink()
            return returncode

        return collect_fn

    def _run(self, collect_fn, **kwargs) -> int:
        defaults = dict(worker_id="w1", static_argv=["--format", "gen3randombattle"],
                        collect_fn=collect_fn, max_rss_mb=None, idle_exit_seconds=0.0,
                        sleep_seconds=0.0)
        defaults.update(kwargs)
        return run_worker(self.queue, **defaults)

    def test_success_commits_out_and_done_marker(self) -> None:
        out = self.root / "cache" / "shard-f0"
        _manifest(self.queue, "i7-s0.env", out=out)
        rc = self._run(self._stub())
        self.assertEqual(rc, 0)
        self.assertTrue((out / "metadata.json").exists())
        self.assertTrue((self.queue / "done" / "i7-s0.env").exists())
        self.assertFalse(list((self.queue / "claimed").iterdir()))
        self.assertFalse(list((self.queue / "failed").iterdir()))

    def test_per_task_argv_is_static_then_overrides(self) -> None:
        out = self.root / "cache" / "shard-f0"
        _manifest(self.queue, "i7-s0.env", out=out, count=2, seed=99)
        self._run(self._stub())
        argv = self.calls[0]
        self.assertEqual(argv[:2], ["--format", "gen3randombattle"])
        # Per-task values appended AFTER static flags (argparse last-wins).
        self.assertGreater(argv.index("--games"), argv.index("--format"))
        self.assertEqual(argv[argv.index("--games") + 1], "2")
        self.assertEqual(argv[argv.index("--seed-start") + 1], "99")
        self.assertTrue(argv[argv.index("--out") + 1].endswith(".tmp.w1"))

    def test_revocation_discards_the_work(self) -> None:
        out = self.root / "cache" / "shard-f0"
        _manifest(self.queue, "i7-s0.env", out=out)
        rc = self._run(self._stub(revoke=True))
        self.assertEqual(rc, 0)
        self.assertFalse(out.exists())
        self.assertFalse((self.queue / "done" / "i7-s0.env").exists())
        # tmp cleaned up too
        self.assertFalse(list(out.parent.glob("*.tmp.*")) if out.parent.exists() else [])

    def test_failure_moves_claim_to_failed(self) -> None:
        out = self.root / "cache" / "shard-f0"
        _manifest(self.queue, "i7-s0.env", out=out)
        rc = self._run(self._stub(returncode=1))
        self.assertEqual(rc, 0)
        self.assertFalse(out.exists())
        self.assertTrue((self.queue / "failed" / "i7-s0.env.w1.failed").exists())

    def test_exception_is_a_failure_not_a_crash(self) -> None:
        _manifest(self.queue, "i7-s0.env", out=self.root / "cache" / "shard-f0")

        def boom(argv: list[str]) -> int:
            raise RuntimeError("kaboom")

        rc = self._run(boom)
        self.assertEqual(rc, 0)
        self.assertTrue((self.queue / "failed" / "i7-s0.env.w1.failed").exists())

    def test_max_tasks_recycles_cleanly(self) -> None:
        _manifest(self.queue, "i7-s0.env", out=self.root / "cache" / "shard-f0")
        _manifest(self.queue, "i7-s1.env", out=self.root / "cache" / "shard-f1")
        rc = self._run(self._stub(), max_tasks=1, idle_exit_seconds=None)
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue((self.queue / "pending" / "i7-s1.env").exists())  # left for the next life

    def test_rss_bound_recycles_cleanly(self) -> None:
        _manifest(self.queue, "i7-s0.env", out=self.root / "cache" / "shard-f0")
        _manifest(self.queue, "i7-s1.env", out=self.root / "cache" / "shard-f1")
        original = fleet_worker._rss_mb
        fleet_worker._rss_mb = lambda: 99999.0
        try:
            rc = self._run(self._stub(), max_rss_mb=1000.0, idle_exit_seconds=None)
        finally:
            fleet_worker._rss_mb = original
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.calls), 1)  # recycled after the first task

    def test_drains_multiple_tasks_in_one_life(self) -> None:
        for i in range(3):
            _manifest(self.queue, f"i7-s{i}.env", out=self.root / "cache" / f"shard-f{i}")
        rc = self._run(self._stub())
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(list((self.queue / "done").iterdir())), 3)


class CliWiringTests(unittest.TestCase):
    def test_subcommand_dispatches_with_static_remainder(self) -> None:
        import tempfile

        from pokezero import rollout_cli

        with tempfile.TemporaryDirectory() as tmp:
            queue = _make_queue(Path(tmp))
            rc = rollout_cli.main([
                "collect-selfplay-worker",
                "--task-queue", str(queue),
                "--worker-id", "wtest",
                "--idle-exit-seconds", "0",
                "--",
                "--format", "gen3randombattle",
            ])
            self.assertEqual(rc, 0)  # empty queue + idle-exit → clean recycle


if __name__ == "__main__":
    unittest.main()
