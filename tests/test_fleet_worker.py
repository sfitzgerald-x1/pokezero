"""Persistent collector-fleet worker: queue protocol + recycle bounds.

All torch-free — the collect function is stubbed; what is under test is the
claim/commit/revocation/failed transport (which must stay byte-compatible with
the shell fleet worker) and the OOM/task recycle bounds.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokezero import fleet_worker  # noqa: E402
from pokezero.fleet_worker import (  # noqa: E402
    FanInQueueContract,
    FanInTask,
    FanInValidationError,
    claim_next_task,
    read_fanin_inventory,
    run_worker,
)


def _make_queue(root: Path) -> Path:
    queue = root / "collect-queue"
    for sub in ("pending", "claimed", "done", "failed"):
        (queue / sub).mkdir(parents=True)
    return queue


def _manifest(queue: Path, base: str, *, out: Path, iteration: int = 7, offset: int = 0,
              count: int = 2, seed: int = 4321, policy: str = "remote:http://svc:8600") -> Path:
    path = queue / "pending" / base
    path.write_text(
        f'a_iter={iteration}\na_offset={offset}\na_count={count}\n'
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


class FanInTests(unittest.TestCase):
    """Shard fan-in: one worker-owned versioned shard per window."""

    def setUp(self) -> None:
        import tempfile

        from tests.test_cache_concat import NUMPY

        if not NUMPY:
            self.skipTest("requires numpy")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.queue = _make_queue(self.root)
        self.cache_dir = self.root / "cache" / "iteration-0001"
        self.cache_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _real_collect(self, argv: list[str]) -> int:
        """Write a REAL single-record training cache at --out, seeded by --seed-start."""
        from pokezero.dataset import TrajectoryDatasetConfig
        from tests.test_cache_concat import rollout_record, write_cache

        out = Path(argv[argv.index("--out") + 1])
        seed = int(argv[argv.index("--seed-start") + 1])
        staging = out.parent / f".collect-staging-{out.name}"
        staging.mkdir(parents=True, exist_ok=True)
        cache = write_cache(staging, "c", [rollout_record(seed)], config=TrajectoryDatasetConfig(window_size=1))
        import shutil

        shutil.move(str(cache), str(out))
        shutil.rmtree(staging, ignore_errors=True)
        return 0

    def _manifests(self, count: int) -> None:
        for index in range(count):
            _manifest(
                self.queue, f"i1-s{index}.env",
                out=self.cache_dir / f"slice-{index}", iteration=1, offset=index,
                count=1, seed=100 + index,
            )

    def _read_meta(self, cache: Path) -> dict:
        import json

        return json.loads((cache / "metadata.json").read_text(encoding="utf-8"))

    def _contract(self, *, tasks: int, games: int, seed_start: int = 100) -> FanInQueueContract:
        return FanInQueueContract(
            iteration=1,
            expected_task_count=tasks,
            expected_game_count=games,
            offset_start=0,
            seed_start=seed_start,
        )

    def _write_version(self, name: str, tasks: list[FanInTask]) -> Path:
        from pokezero.dataset import TrajectoryDatasetConfig
        from tests.test_cache_concat import rollout_record, write_cache

        staging = self.root / f"staging-{len(list(self.root.glob('staging-*')))}-{name}"
        staging.mkdir()
        cache = write_cache(
            staging, name, [rollout_record(task.seed) for task in tasks],
            config=TrajectoryDatasetConfig(window_size=1),
        )
        import shutil

        target = self.cache_dir / name
        shutil.move(str(cache), str(target))
        fleet_worker._write_fanin_manifest(target, tasks)
        return target

    def _requeue_claim(self, worker: str, base: str) -> None:
        claim = self.queue / "claimed" / f"{base}.{worker}"
        self.assertTrue(claim.exists())
        claim.rename(self.queue / "pending" / base)

    def test_tasks_fan_into_one_versioned_shard(self) -> None:
        self._manifests(3)
        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 0)
        shards = sorted(p.name for p in self.cache_dir.glob("shard-w*"))
        self.assertEqual(shards, ["shard-ww1-v3"])
        self.assertEqual(self._read_meta(self.cache_dir / "shard-ww1-v3")["record_count"], 3)
        self.assertEqual(len(list((self.queue / "done").iterdir())), 3)
        self.assertFalse(list(self.cache_dir.glob("slice-*")))  # no per-task shards

    def test_select_fanin_shards_picks_highest_version(self) -> None:
        from pokezero.fleet_worker import select_fanin_shards

        self._manifests(2)
        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        # Simulate a stale lower version surviving a crash.
        stale = self.cache_dir / "shard-ww1-v1"
        stale.mkdir()
        selected = select_fanin_shards(self.cache_dir)
        self.assertEqual([p.name for p in selected], ["shard-ww1-v2"])

    def test_strict_inventory_uses_only_selected_highest_versions(self) -> None:
        self._manifests(2)
        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        # A legacy lower version is stale evidence and must neither be selected
        # nor make a valid selected cumulative version unsafe.
        (self.cache_dir / "shard-ww1-v1").mkdir()
        inventory = read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=2))
        self.assertEqual([shard.path.name for shard in inventory.shards], ["shard-ww1-v2"])
        self.assertEqual([task.task_id for task in inventory.tasks], ["i1-s0.env", "i1-s1.env"])

    def test_crash_before_target_publication_leaves_no_visible_version(self) -> None:
        self._manifests(1)

        def crash(boundary: str, _task) -> None:
            if boundary == "fanin-before-target-publication":
                raise SystemExit("simulated process death")

        with self.assertRaises(SystemExit):
            run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
                crash_inject=crash,
            )
        abandoned = list(self.cache_dir.iterdir())
        self.assertEqual(len(abandoned), 1)
        self.assertTrue(abandoned[0].name.startswith(".shard-ww1-v1.tmp."))
        self._requeue_claim("w1", "i1-s0.env")
        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual([path.name for path in self.cache_dir.iterdir()], ["shard-ww1-v1"])
        inventory = read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))
        self.assertEqual(inventory.total_games, 1)

    def test_crash_after_target_publication_recovers_on_another_worker_without_duplication(self) -> None:
        self._manifests(1)
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        def crash(boundary: str, _task) -> None:
            if boundary == "fanin-after-target-publication":
                raise SystemExit("simulated process death")

        with self.assertRaises(SystemExit):
            run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
                crash_inject=crash,
            )
        self.assertTrue((self.cache_dir / "shard-ww1-v1").exists())
        self.assertFalse((self.queue / "done" / "i1-s0.env").exists())
        self._requeue_claim("w1", "i1-s0.env")
        run_worker(
            self.queue, worker_id="w2", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertTrue((self.queue / "done" / "i1-s0.env").exists())
        self.assertEqual(self._read_meta(self.cache_dir / "shard-ww1-v1")["record_count"], 1)
        self.assertEqual(read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1)).total_games, 1)

    def test_crash_after_recovery_keeps_done_and_output_once(self) -> None:
        self._manifests(1)

        def crash_after_publish(boundary: str, _task) -> None:
            if boundary == "fanin-after-target-publication":
                raise SystemExit("simulated process death")

        with self.assertRaises(SystemExit):
            run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
                crash_inject=crash_after_publish,
            )
        self._requeue_claim("w1", "i1-s0.env")
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        def crash_after_recovery(boundary: str, _task) -> None:
            if boundary == "fanin-after-recovery":
                raise SystemExit("simulated process death")

        with self.assertRaises(SystemExit):
            run_worker(
                self.queue, worker_id="w2", static_argv=[], collect_fn=counted_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
                crash_inject=crash_after_recovery,
            )
        self.assertTrue((self.queue / "done" / "i1-s0.env").exists())
        self.assertEqual(calls, [])
        self.assertEqual(read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1)).total_games, 1)

    def test_crash_after_done_link_recovers_same_inode_without_failed_marker(self) -> None:
        self._manifests(1)
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        def crash_after_publish(boundary: str, _task) -> None:
            if boundary == "fanin-after-target-publication":
                raise SystemExit("simulated process death")

        with self.assertRaises(SystemExit):
            run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
                crash_inject=crash_after_publish,
            )
        self._requeue_claim("w1", "i1-s0.env")

        def crash_after_link(boundary: str, _task) -> None:
            if boundary == "fanin-after-done-link":
                raise SystemExit("simulated process death")

        with self.assertRaises(SystemExit):
            run_worker(
                self.queue, worker_id="w2", static_argv=[], collect_fn=counted_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
                crash_inject=crash_after_link,
            )
        claim = self.queue / "claimed" / "i1-s0.env.w2"
        done = self.queue / "done" / "i1-s0.env"
        self.assertTrue(os.path.samestat(claim.stat(), done.stat()))

        self._requeue_claim("w2", "i1-s0.env")
        run_worker(
            self.queue, worker_id="w3", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual([path.name for path in (self.queue / "done").iterdir()], ["i1-s0.env"])
        self.assertFalse(list((self.queue / "claimed").iterdir()))
        self.assertFalse(list((self.queue / "failed").iterdir()))

    def test_unrelated_legacy_shard_preserves_untried_claim_and_stops(self) -> None:
        (self.cache_dir / "shard-wlegacy-v1").mkdir()
        self._manifests(2)
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])
        self.assertTrue((self.queue / "claimed" / "i1-s0.env.w1").exists())
        self.assertTrue((self.queue / "pending" / "i1-s1.env").exists())
        self.assertFalse(list((self.queue / "failed").iterdir()))
        self.assertFalse(list((self.queue / "done").iterdir()))

    def test_unrelated_malformed_shard_preserves_untried_claim_and_stops(self) -> None:
        malformed = self.cache_dir / "shard-wmalformed-v1"
        malformed.mkdir()
        (malformed / "fanin-manifest.json").write_text("{", encoding="utf-8")
        self._manifests(1)
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(calls, [])
        self.assertTrue((self.queue / "claimed" / "i1-s0.env.w1").exists())
        self.assertFalse(list((self.queue / "failed").iterdir()))

    def test_inventory_becoming_malformed_after_collection_preserves_claim(self) -> None:
        self._manifests(2)
        calls: list[list[str]] = []

        def collect_then_add_malformed_shard(argv: list[str]) -> int:
            calls.append(argv)
            result = self._real_collect(argv)
            (self.cache_dir / "shard-wmalformed-v1").mkdir()
            return result

        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=collect_then_add_malformed_shard,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertTrue((self.queue / "claimed" / "i1-s0.env.w1").exists())
        self.assertTrue((self.queue / "pending" / "i1-s1.env").exists())
        self.assertFalse(list((self.queue / "failed").iterdir()))
        self.assertFalse(list((self.queue / "done").iterdir()))

    def test_current_task_metadata_conflict_fails_only_that_claim(self) -> None:
        self._write_version("shard-wother-v1", [FanInTask("i1-s0.env", 1, 0, 1, 999)])
        self._manifests(2)
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, max_tasks=1, idle_exit_seconds=None, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(calls, [])
        self.assertTrue((self.queue / "failed" / "i1-s0.env.w1.failed").exists())
        self.assertTrue((self.queue / "pending" / "i1-s1.env").exists())

    def test_different_inode_done_marker_is_a_task_conflict(self) -> None:
        self._write_version("shard-wother-v1", [FanInTask("i1-s0.env", 1, 0, 1, 100)])
        self._manifests(1)
        (self.queue / "done" / "i1-s0.env").write_text("different claim", encoding="utf-8")
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, max_tasks=1, idle_exit_seconds=None, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(calls, [])
        self.assertTrue((self.queue / "done" / "i1-s0.env").exists())
        self.assertTrue((self.queue / "failed" / "i1-s0.env.w1.failed").exists())

    def test_revoked_claim_cannot_publish_a_target(self) -> None:
        self._manifests(1)

        def revoke(boundary: str, task) -> None:
            if boundary == "fanin-before-target-publication":
                task.claim_path.unlink()

        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            crash_inject=revoke,
        )
        self.assertFalse(list(self.cache_dir.iterdir()))
        self.assertFalse((self.queue / "done" / "i1-s0.env").exists())

    def test_strict_inventory_rejects_malformed_or_missing_manifest(self) -> None:
        missing = self._write_version("shard-ww1-v1", [FanInTask("i1-s0.env", 1, 0, 1, 100)])
        (missing / "fanin-manifest.json").unlink()
        with self.assertRaisesRegex(FanInValidationError, "no valid"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))
        (missing / "fanin-manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(FanInValidationError, "malformed manifest envelope"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))

    def test_strict_inventory_rejects_cache_game_count_mismatch(self) -> None:
        # _write_version writes one record; the manifest's two-game claim must
        # be rejected before a launcher can include it in training.
        self._write_version("shard-ww1-v1", [FanInTask("i1-s0.env", 1, 0, 2, 100)])
        with self.assertRaisesRegex(FanInValidationError, "cache games disagree"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=2))

    def test_strict_inventory_rejects_conflicts_ranges_and_contract_totals(self) -> None:
        self._write_version("shard-ww1-v1", [FanInTask("i1-s0.env", 1, 0, 1, 100)])
        self._write_version("shard-ww2-v1", [FanInTask("i1-s0.env", 1, 1, 1, 101)])
        with self.assertRaisesRegex(FanInValidationError, "conflicting metadata"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=2))

        import shutil

        shutil.rmtree(self.cache_dir / "shard-ww2-v1")
        self._write_version("shard-ww2-v1", [FanInTask("i1-s0.env", 1, 0, 1, 100)])
        with self.assertRaisesRegex(FanInValidationError, "repeats task id"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=2))

        shutil.rmtree(self.cache_dir / "shard-ww2-v1")
        self._write_version("shard-ww2-v1", [FanInTask("i1-s2.env", 1, 2, 1, 102)])
        with self.assertRaisesRegex(FanInValidationError, "gap or overlap in queue offsets"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=2))

        shutil.rmtree(self.cache_dir / "shard-ww2-v1")
        self._write_version("shard-ww2-v1", [FanInTask("i1-s1.env", 1, 1, 1, 102)])
        with self.assertRaisesRegex(FanInValidationError, "gap or overlap in queue seed ranges"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=2))
        with self.assertRaisesRegex(FanInValidationError, "expected 3"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=3))
        with self.assertRaisesRegex(FanInValidationError, "wrong iteration"):
            read_fanin_inventory(
                self.cache_dir,
                FanInQueueContract(2, expected_task_count=2, expected_game_count=2, seed_start=100),
            )

    def test_selected_version_vanishing_reselects_new_highest_version(self) -> None:
        from unittest.mock import patch
        import shutil

        tasks = [FanInTask("i1-s0.env", 1, 0, 1, 100)]
        v4 = self._write_version("shard-ww1-v4", tasks)
        original_read = fleet_worker._read_fanin_manifest
        raced = False

        def publish_v5_while_reading(path: Path):
            nonlocal raced
            if path == v4 and not raced:
                raced = True
                self._write_version("shard-ww1-v5", tasks)
                shutil.rmtree(v4)
            return original_read(path)

        with patch.object(fleet_worker, "_read_fanin_manifest", new=publish_v5_while_reading):
            inventory = read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))
        self.assertTrue(raced)
        self.assertEqual([shard.path.name for shard in inventory.shards], ["shard-ww1-v5"])

    def test_adopts_existing_version_and_sweeps_stale(self) -> None:
        from pokezero.dataset import TrajectoryDatasetConfig
        from tests.test_cache_concat import rollout_record, write_cache

        staging = self.root / "staging"
        staging.mkdir()
        v1 = write_cache(staging, "v1", [rollout_record(1)], config=TrajectoryDatasetConfig(window_size=1))
        v2 = write_cache(staging, "v2", [rollout_record(2), rollout_record(3)], config=TrajectoryDatasetConfig(window_size=1))
        import shutil

        shutil.move(str(v1), str(self.cache_dir / "shard-ww1-v1"))
        shutil.move(str(v2), str(self.cache_dir / "shard-ww1-v2"))
        fleet_worker._write_fanin_manifest(
            self.cache_dir / "shard-ww1-v2",
            [
                FanInTask("prior-s0.env", 1, 0, 1, 98),
                FanInTask("prior-s1.env", 1, 1, 1, 99),
            ],
        )
        self._manifests(1)
        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        shards = sorted(p.name for p in self.cache_dir.glob("shard-w*"))
        self.assertEqual(shards, ["shard-ww1-v3"])  # v1 swept, v2 adopted + extended
        self.assertEqual(self._read_meta(self.cache_dir / "shard-ww1-v3")["record_count"], 3)

    def test_revocation_creates_no_shard_version(self) -> None:
        queue = self.queue

        def revoking_collect(argv: list[str]) -> int:
            rc = self._real_collect(argv)
            for claim in (queue / "claimed").iterdir():
                claim.unlink()
            return rc

        self._manifests(1)
        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=revoking_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertFalse(list(self.cache_dir.iterdir()))
        self.assertFalse(list((self.queue / "done").iterdir()))

    def test_log_dir_persists_commit_lines(self) -> None:
        log_dir = self.root / "worker-logs"
        self._manifests(2)
        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            log_dir=log_dir,
        )
        content = (log_dir / "w1.log").read_text(encoding="utf-8")
        self.assertEqual(content.count("commit-fanin"), 2)
        self.assertIn("concat=", content)
        self.assertIn("games=1", content)


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
