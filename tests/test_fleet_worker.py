"""Persistent collector-fleet worker: queue protocol + recycle bounds.

All torch-free — the collect function is stubbed; what is under test is the
claim/commit/revocation/failed transport (which must stay byte-compatible with
the shell fleet worker) and the OOM/task recycle bounds.
"""

from __future__ import annotations

import json
import errno
import os
import subprocess
import sys
import time
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

    def test_claim_parses_shell_quoted_routes_without_expansion(self) -> None:
        sentinel = self.root / "must-not-exist"
        out = self.root / "cache with space" / "it's-output"
        policy = f"remote:http://svc/a path 'quoted' $(touch {sentinel})"
        manifest = self.queue / "pending" / "i7-s1.env"
        manifest.write_text(
            "a_iter=7\na_offset=0\na_count=2\na_seed=4321\n"
            f"a_out={fleet_worker.shlex.quote(str(out))}\n"
            f"a_policy={fleet_worker.shlex.quote(policy)}\n",
            encoding="utf-8",
        )
        task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(task)
        self.assertEqual(task.out, out)
        self.assertEqual(task.policy, policy)
        self.assertFalse(sentinel.exists())

    def test_reused_claim_path_does_not_revive_the_prior_generation(self) -> None:
        _manifest(self.queue, "i7-s0.env", out=self.root / "out")
        first = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(first)
        first.claim_path.rename(self.queue / "pending" / first.base)
        second = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(second)
        self.assertNotEqual(first.claim_token, second.claim_token)
        self.assertFalse(fleet_worker._claim_token_is_current(first))
        self.assertTrue(fleet_worker._claim_token_is_current(second))

    def test_empty_queue_returns_none(self) -> None:
        self.assertIsNone(claim_next_task(self.queue, "w1"))

    def test_malformed_manifest_parks_in_failed(self) -> None:
        (self.queue / "pending" / "i7-s9.env").write_text("garbage\n", encoding="utf-8")
        self.assertIsNone(claim_next_task(self.queue, "w1"))
        self.assertEqual(len(list((self.queue / "failed").glob("i7-s9.env.w1.*.failed"))), 1)

    def test_claim_initialization_cannot_quarantine_a_successor_generation(self) -> None:
        from unittest.mock import patch

        _manifest(self.queue, "i7-s0.env", out=self.root / "out")
        original = fleet_worker._write_claim_token
        successor = None
        interleaved = False

        def replace_then_fail(claim: Path) -> str:
            nonlocal successor, interleaved
            if not interleaved:
                interleaved = True
                claim.rename(self.queue / "pending" / "i7-s0.env")
                successor = claim_next_task(self.queue, "w1")
                raise ValueError("simulated lease initialization failure")
            return original(claim)

        with patch.object(fleet_worker, "_write_claim_token", new=replace_then_fail):
            self.assertIsNone(claim_next_task(self.queue, "w1"))
        self.assertTrue(interleaved)
        self.assertIsNotNone(successor)
        self.assertTrue(successor.claim_path.exists())
        self.assertFalse(list((self.queue / "failed").iterdir()))

    def test_prior_generation_completion_cannot_acknowledge_or_delete_successor(self) -> None:
        _manifest(self.queue, "i7-s0.env", out=self.root / "route-a", policy="policy-a")
        prior = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(prior)
        prior.claim_path.rename(self.queue / "pending" / prior.base)
        _manifest(self.queue, prior.base, out=self.root / "route-b", policy="policy-b")
        successor = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(successor)

        fleet_worker._complete_claim(prior, self.queue, crash_inject=None)

        done = self.queue / "done" / prior.base
        self.assertTrue(successor.claim_path.exists())
        self.assertTrue(fleet_worker._done_marker_matches_task(done, prior))
        self.assertFalse(fleet_worker._done_marker_matches_task(done, successor))

    def test_malformed_fanin_route_provenance_fails_closed(self) -> None:
        _manifest(self.queue, "i7-s0.env", out=self.root / "route-a")
        task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(task)
        record = fleet_worker._fanin_route_record(self.queue, task.base)
        record.write_text("{", encoding="utf-8")

        with self.assertRaises(FanInValidationError):
            fleet_worker._resolve_fanin_route(self.queue, task)


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
        self.assertEqual(len(list((self.queue / "failed").glob("i7-s0.env.w1.*.failed"))), 1)

    def test_exception_is_a_failure_not_a_crash(self) -> None:
        _manifest(self.queue, "i7-s0.env", out=self.root / "cache" / "shard-f0")

        def boom(argv: list[str]) -> int:
            raise RuntimeError("kaboom")

        rc = self._run(boom)
        self.assertEqual(rc, 0)
        self.assertEqual(len(list((self.queue / "failed").glob("i7-s0.env.w1.*.failed"))), 1)

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

    def _fanin_task(
        self,
        task_id: str,
        iteration: int,
        offset: int,
        count: int,
        seed: int,
        *,
        out: Path | None = None,
        policy: str = "remote:http://svc:8600",
    ) -> FanInTask:
        return FanInTask(
            task_id,
            iteration,
            offset,
            count,
            seed,
            str(out or self.cache_dir / f"route-{task_id}"),
            policy,
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
        claim = self._claim(worker, base)
        claim.rename(self.queue / "pending" / base)

    def _claim(self, worker: str, base: str) -> Path:
        claims = list((self.queue / "claimed").glob(f"{base}.{worker}.*"))
        self.assertEqual(len(claims), 1)
        return claims[0]

    def _failed_claims(self, worker: str, base: str) -> list[Path]:
        return list((self.queue / "failed").glob(f"{base}.{worker}.*.failed"))

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
        abandoned = list(self.cache_dir.glob(".shard-ww1-v1.tmp.*"))
        staging = next(path for path in abandoned if path.is_dir())
        sidecar = fleet_worker._fanin_staging_owner_path(staging)
        self.assertTrue(staging.name.startswith(".shard-ww1-v1.tmp."))
        self.assertTrue(sidecar.exists())
        self.assertEqual(
            fleet_worker._read_fanin_staging_owner(staging),
            self._fanin_task("i1-s0.env", 1, 0, 1, 100, out=self.cache_dir / "slice-0"),
        )
        self._requeue_claim("w1", "i1-s0.env")
        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual([path.name for path in fleet_worker.select_fanin_shards(self.cache_dir)], ["shard-ww1-v1"])
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
        self.assertEqual(len(list(self.cache_dir.glob(".shard-ww1-v1.tmp.*.owner.json"))), 1)
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

    def test_post_publication_policy_change_is_terminal_without_recollection(self) -> None:
        out = self.cache_dir / "slice-0"
        _manifest(self.queue, "i1-s0.env", out=out, iteration=1, offset=0, count=1, seed=100, policy="policy-a")
        calls: list[list[str]] = []

        def collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        def crash(boundary: str, _task) -> None:
            if boundary == "fanin-after-target-publication":
                raise SystemExit("simulated process death")

        with self.assertRaises(SystemExit):
            run_worker(self.queue, worker_id="w1", static_argv=[], collect_fn=collect, max_rss_mb=None,
                       idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True, crash_inject=crash)
        self._claim("w1", "i1-s0.env").unlink()
        _manifest(self.queue, "i1-s0.env", out=out, iteration=1, offset=0, count=1, seed=100, policy="policy-b")
        rc = run_worker(self.queue, worker_id="w2", static_argv=[], collect_fn=collect, max_rss_mb=None,
                        idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True)
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self._claim("w2", "i1-s0.env").exists())
        self.assertFalse((self.queue / "done" / "i1-s0.env").exists())
        self.assertEqual(fleet_worker._read_fanin_manifest(self.cache_dir / "shard-ww1-v1")[0].policy, "policy-a")

    def test_post_publication_output_change_is_terminal_without_recollection(self) -> None:
        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-a", iteration=1, offset=0, count=1, seed=100)
        calls: list[list[str]] = []

        def collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        def crash(boundary: str, _task) -> None:
            if boundary == "fanin-after-target-publication":
                raise SystemExit("simulated process death")

        with self.assertRaises(SystemExit):
            run_worker(self.queue, worker_id="w1", static_argv=[], collect_fn=collect, max_rss_mb=None,
                       idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True, crash_inject=crash)
        self._claim("w1", "i1-s0.env").unlink()
        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-b", iteration=1, offset=0, count=1, seed=100)
        rc = run_worker(self.queue, worker_id="w2", static_argv=[], collect_fn=collect, max_rss_mb=None,
                        idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True)
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self._claim("w2", "i1-s0.env").exists())
        self.assertFalse((self.queue / "done" / "i1-s0.env").exists())
        self.assertEqual(fleet_worker._read_fanin_manifest(self.cache_dir / "shard-ww1-v1")[0].out, str(self.cache_dir / "slice-a"))

    def test_post_publication_output_parent_change_is_terminal_without_recollection(self) -> None:
        first_out = self.cache_dir / "slice-a"
        other_root = self.root / "other-cache" / "iteration-0001"
        _manifest(self.queue, "i1-s0.env", out=first_out, iteration=1, offset=0, count=1, seed=100)
        calls: list[list[str]] = []

        def collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        def crash(boundary: str, _task) -> None:
            if boundary == "fanin-after-target-publication":
                raise SystemExit("simulated process death")

        with self.assertRaises(SystemExit):
            run_worker(self.queue, worker_id="w1", static_argv=[], collect_fn=collect, max_rss_mb=None,
                       idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True, crash_inject=crash)
        self._claim("w1", "i1-s0.env").unlink()
        _manifest(self.queue, "i1-s0.env", out=other_root / "slice-b", iteration=1, offset=0, count=1, seed=100)
        rc = run_worker(self.queue, worker_id="w2", static_argv=[], collect_fn=collect, max_rss_mb=None,
                        idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True)
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self._claim("w2", "i1-s0.env").exists())
        self.assertFalse(other_root.exists())
        self.assertEqual([path.name for path in fleet_worker.select_fanin_shards(self.cache_dir)], ["shard-ww1-v1"])

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

    def test_recreated_done_marker_with_matching_metadata_is_idempotent(self) -> None:
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
        done = self.queue / "done" / "i1-s0.env"
        # A reaper may copy/recreate this marker rather than preserve the
        # link inode. Canonical queue metadata, not inode identity, is the
        # idempotence contract.
        copied = done.read_text(encoding="utf-8")
        done.unlink()
        done.write_text(copied, encoding="utf-8")

        self._requeue_claim("w2", "i1-s0.env")
        run_worker(
            self.queue, worker_id="w3", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual([path.name for path in (self.queue / "done").iterdir()], ["i1-s0.env"])
        self.assertFalse(list((self.queue / "claimed").iterdir()))
        self.assertFalse(list((self.queue / "failed").iterdir()))

    def test_unrelated_legacy_shard_preserves_claim_and_returns_terminal_status(self) -> None:
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
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])
        self.assertTrue(self._claim("w1", "i1-s0.env").exists())
        self.assertTrue((self.queue / "pending" / "i1-s1.env").exists())
        self.assertFalse(list((self.queue / "failed").iterdir()))
        self.assertFalse(list((self.queue / "done").iterdir()))

    def test_unrelated_malformed_shard_preserves_claim_and_returns_terminal_status(self) -> None:
        malformed = self.cache_dir / "shard-wmalformed-v1"
        malformed.mkdir()
        (malformed / "fanin-manifest.json").write_text("{", encoding="utf-8")
        self._manifests(1)
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])
        self.assertTrue(self._claim("w1", "i1-s0.env").exists())
        self.assertFalse(list((self.queue / "failed").iterdir()))

    def test_inventory_becoming_malformed_after_collection_preserves_claim_and_fails_terminally(self) -> None:
        self._manifests(2)
        calls: list[list[str]] = []

        def collect_then_add_malformed_shard(argv: list[str]) -> int:
            calls.append(argv)
            result = self._real_collect(argv)
            (self.cache_dir / "shard-wmalformed-v1").mkdir()
            return result

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=collect_then_add_malformed_shard,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self._claim("w1", "i1-s0.env").exists())
        self.assertTrue((self.queue / "pending" / "i1-s1.env").exists())
        self.assertFalse(list((self.queue / "failed").iterdir()))
        self.assertFalse(list((self.queue / "done").iterdir()))

    def test_current_task_metadata_conflict_fails_only_that_claim(self) -> None:
        self._write_version("shard-wother-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 999)])
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
        self.assertEqual(len(self._failed_claims("w1", "i1-s0.env")), 1)
        self.assertTrue((self.queue / "pending" / "i1-s1.env").exists())

    def test_duplicate_selected_task_is_terminal_and_preserves_claim(self) -> None:
        task = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        self._write_version("shard-wother-v1", [task])
        self._write_version("shard-wother2-v1", [task])
        self._manifests(1)
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])
        self.assertTrue(self._claim("w1", "i1-s0.env").exists())
        self.assertFalse(list((self.queue / "failed").iterdir()))

    def test_invalid_task_metadata_fails_only_each_claim_and_drains_following_task(self) -> None:
        _manifest(
            self.queue, "i1-s0.env", out=self.cache_dir / "slice-invalid", iteration=1,
            offset=-1, count=1, seed=100,
        )
        _manifest(
            self.queue, "i1-s1.env", out=self.cache_dir / "slice-zero", iteration=1,
            offset=0, count=0, seed=101,
        )
        _manifest(
            self.queue, "i1-s2.env", out=self.cache_dir / "slice-healthy", iteration=1,
            offset=2, count=1, seed=102,
        )
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(self._failed_claims("w1", "i1-s0.env")), 1)
        self.assertEqual(len(self._failed_claims("w1", "i1-s1.env")), 1)
        self.assertTrue((self.queue / "done" / "i1-s2.env").exists())

    def test_short_collection_fails_only_that_claim_and_drains_following_task(self) -> None:
        _manifest(
            self.queue, "i1-s0.env", out=self.cache_dir / "slice-short", iteration=1,
            offset=0, count=2, seed=100,
        )
        _manifest(
            self.queue, "i1-s1.env", out=self.cache_dir / "slice-healthy", iteration=1,
            offset=2, count=1, seed=102,
        )
        calls: list[list[str]] = []

        def short_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)  # intentionally writes one record for both tasks

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=short_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self._failed_claims("w1", "i1-s0.env")), 1)
        self.assertTrue((self.queue / "done" / "i1-s1.env").exists())

    def test_missing_or_garbled_temporary_metadata_fails_only_that_claim(self) -> None:
        self._manifests(3)
        calls: list[int] = []

        def malformed_then_healthy(argv: list[str]) -> int:
            out = Path(argv[argv.index("--out") + 1])
            seed = int(argv[argv.index("--seed-start") + 1])
            calls.append(seed)
            if seed == 100:
                out.mkdir(parents=True, exist_ok=True)
                return 0  # missing metadata.json
            if seed == 101:
                out.mkdir(parents=True, exist_ok=True)
                (out / "metadata.json").write_text("{", encoding="utf-8")
                return 0
            return self._real_collect(argv)

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=malformed_then_healthy,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [100, 101, 102])
        self.assertEqual(len(self._failed_claims("w1", "i1-s0.env")), 1)
        self.assertEqual(len(self._failed_claims("w1", "i1-s1.env")), 1)
        self.assertTrue((self.queue / "done" / "i1-s2.env").exists())
        self.assertFalse(list((self.queue / "claimed").iterdir()))

    def test_prepublication_matching_done_marker_is_terminal_and_retains_claim(self) -> None:
        pending = _manifest(
            self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1,
            offset=0, count=1, seed=100,
        )
        declared = fleet_worker._parse_manifest(pending, "i1-s0.env")
        (self.queue / "done" / "i1-s0.env").write_text(
            fleet_worker._task_manifest_text(declared), encoding="utf-8",
        )
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])
        self.assertTrue((self.queue / "done" / "i1-s0.env").exists())
        self.assertTrue(self._claim("w1", "i1-s0.env").exists())
        self.assertFalse(list((self.queue / "failed").iterdir()))
        self.assertFalse(list(self.cache_dir.glob("shard-w*")))

    def test_done_marker_reconstruction_is_shell_safe_and_binds_routes(self) -> None:
        sentinel = self.root / "must-not-exist"
        task = fleet_worker.TaskManifest(
            base="i1-s0.env",
            claim_path=self.queue / "claimed" / "i1-s0.env.w1",
            iteration=1,
            offset=0,
            count=1,
            seed=100,
            out=self.root / "out dir" / "it's $literal",
            policy=f"remote:http://svc/a path 'quoted' $(touch {sentinel})",
        )
        done = self.queue / "done" / task.base
        self.assertTrue(fleet_worker._write_done_marker(task, done))
        sourced = subprocess.run(
            ["sh", "-c", 'set -eu; . "$1"; printf "%s\\n%s" "$a_out" "$a_policy"', "sh", str(done)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(sourced.stdout, f"{task.out}\n{task.policy}")
        self.assertFalse(sentinel.exists())
        self.assertTrue(fleet_worker._done_marker_matches_task(done, task))
        other_route = fleet_worker.TaskManifest(**{**task.__dict__, "policy": "remote:http://different"})
        self.assertFalse(fleet_worker._done_marker_matches_task(done, other_route))

    def test_non_ascii_version_suffix_is_not_selected(self) -> None:
        self._write_version("shard-ww1-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 100)])
        (self.cache_dir / "shard-ww1-v１２").mkdir()
        self.assertEqual(
            [path.name for path in fleet_worker.select_fanin_shards(self.cache_dir)],
            ["shard-ww1-v1"],
        )

    def test_conflicting_done_marker_is_a_task_conflict(self) -> None:
        self._write_version("shard-wother-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 100)])
        self._manifests(1)
        (self.queue / "done" / "i1-s0.env").write_text("different claim", encoding="utf-8")
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, max_tasks=1, idle_exit_seconds=None, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])
        self.assertTrue((self.queue / "done" / "i1-s0.env").exists())
        self.assertTrue(self._claim("w1", "i1-s0.env").exists())
        self.assertFalse(list((self.queue / "failed").iterdir()))

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
        self.assertFalse(fleet_worker.select_fanin_shards(self.cache_dir))
        self.assertFalse((self.queue / "done" / "i1-s0.env").exists())

    def test_post_publication_revocation_recovers_done_marker_without_discarding_input(self) -> None:
        self._manifests(1)

        def revoke_after_publication(boundary: str, task) -> None:
            if boundary == "fanin-after-target-publication":
                task.claim_path.unlink()

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            crash_inject=revoke_after_publication,
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.cache_dir / "shard-ww1-v1").exists())
        self.assertTrue((self.queue / "done" / "i1-s0.env").exists())
        self.assertFalse(list((self.queue / "claimed").iterdir()))
        self.assertEqual(read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1)).total_games, 1)

    def test_post_publication_crash_after_revocation_recovers_done_on_recreated_claim(self) -> None:
        self._manifests(1)
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        def revoke_then_crash(boundary: str, task) -> None:
            if boundary == "fanin-after-target-publication":
                task.claim_path.unlink()
                raise SystemExit("simulated pod death after accepted publication")

        with self.assertRaises(SystemExit):
            run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
                crash_inject=revoke_then_crash,
            )
        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        rc = run_worker(
            self.queue, worker_id="w2", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue((self.queue / "done" / "i1-s0.env").exists())

    def test_changed_worker_reclaims_only_staging_whose_owner_claim_is_gone(self) -> None:
        abandoned = self.cache_dir / ".shard-wold-v1.tmp.123.456"
        abandoned.mkdir()
        fleet_worker._write_fanin_staging_owner(
            abandoned, self._fanin_task("i1-sstale.env", 1, 9, 1, 109),
        )
        self._manifests(1)
        rc = run_worker(
            self.queue, worker_id="new", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 0)
        self.assertFalse(abandoned.exists())
        self.assertTrue((self.queue / "done" / "i1-s0.env").exists())

    def test_changed_worker_never_reclaims_staging_with_a_live_owner_claim(self) -> None:
        abandoned = self.cache_dir / ".shard-wold-v1.tmp.123.456"
        abandoned.mkdir()
        owner = self._fanin_task("i1-sstale.env", 1, 9, 1, 109)
        fleet_worker._write_fanin_staging_owner(abandoned, owner)
        _manifest(
            self.queue, owner.task_id, out=self.cache_dir / "slice-stale", iteration=1,
            offset=owner.offset, count=owner.count, seed=owner.seed,
        )
        self.assertIsNotNone(claim_next_task(self.queue, "old"))
        self._manifests(1)
        rc = run_worker(
            self.queue, worker_id="new", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertEqual(rc, 0)
        self.assertTrue(abandoned.exists())

    def test_crash_during_concat_leaves_sidecar_for_cross_worker_dead_owner_reclaim(self) -> None:
        from unittest.mock import patch

        self._manifests(2)
        run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, max_tasks=1, idle_exit_seconds=None, sleep_seconds=0.0, shard_fanin=True,
        )

        def die_during_concat(_inputs, staging: Path) -> None:
            staging.mkdir(parents=True, exist_ok=True)
            raise SystemExit("simulated pod death during concat")

        with patch("pokezero.dataset.concat_training_caches", new=die_during_concat):
            with self.assertRaises(SystemExit):
                run_worker(
                    self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                    max_rss_mb=None, max_tasks=1, idle_exit_seconds=None, sleep_seconds=0.0, shard_fanin=True,
                )
        staging = next(path for path in self.cache_dir.glob(".shard-ww1-v2.tmp.*") if path.is_dir())
        sidecar = fleet_worker._fanin_staging_owner_path(staging)
        self.assertTrue(sidecar.exists())
        self.assertEqual(
            fleet_worker._read_fanin_staging_owner(staging),
            self._fanin_task("i1-s1.env", 1, 1, 1, 101, out=self.cache_dir / "slice-1"),
        )
        stale_claim = self._claim("w1", "i1-s1.env")
        stale_claim.rename(self.queue / "failed" / f"{stale_claim.name}.failed")
        _manifest(
            self.queue, "i1-s2.env", out=self.cache_dir / "slice-2", iteration=1,
            offset=2, count=1, seed=102,
        )
        # A vanished claim is not enough to reclaim a still-live producer.
        run_worker(
            self.queue, worker_id="w2", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, max_tasks=1, idle_exit_seconds=None, sleep_seconds=0.0, shard_fanin=True,
        )
        self.assertTrue(staging.exists())
        lease = fleet_worker._fanin_staging_lease_path(staging)
        payload = json.loads(lease.read_text(encoding="utf-8"))
        payload["renewed_at"] = 0
        lease.write_text(json.dumps(payload), encoding="utf-8")
        fleet_worker._sweep_abandoned_fanin_staging(self.cache_dir, self.queue)
        self.assertFalse(staging.exists())
        self.assertFalse(sidecar.exists())

    def test_same_worker_id_twin_publish_recovers_without_duplicate_manifest_entry(self) -> None:
        from unittest.mock import patch

        _manifest(
            self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1,
            offset=0, count=1, seed=100,
        )
        task = claim_next_task(self.queue, "same")
        self.assertIsNotNone(task)
        first_tmp = self.cache_dir / "first.tmp"
        twin_tmp = self.cache_dir / "twin.tmp"
        self._real_collect(["--out", str(first_tmp), "--seed-start", "100"])
        self._real_collect(["--out", str(twin_tmp), "--seed-start", "100"])
        original_read_current = fleet_worker._read_current_worker_shard
        interleaved = False

        def twin_while_first_holds_lock(*args, **kwargs):
            nonlocal interleaved
            if not interleaved:
                interleaved = True
                with self.assertRaises(fleet_worker._FanInTransientError):
                    fleet_worker._publish_fanin_task(task, twin_tmp, self.queue, "same", crash_inject=None)
            return original_read_current(*args, **kwargs)

        with patch.object(fleet_worker, "_read_current_worker_shard", new=twin_while_first_holds_lock):
            target, recovered = fleet_worker._publish_fanin_task(
                task, first_tmp, self.queue, "same", crash_inject=None,
            )
        self.assertTrue(interleaved)
        self.assertFalse(recovered)
        self.assertIsNotNone(target)
        _, recovered = fleet_worker._publish_fanin_task(task, twin_tmp, self.queue, "same", crash_inject=None)
        self.assertTrue(recovered)
        tasks = fleet_worker._read_fanin_manifest(target)
        self.assertEqual(tasks, (self._fanin_task("i1-s0.env", 1, 0, 1, 100, out=self.cache_dir / "slice-0"),))
        inventory = read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))
        self.assertEqual([entry.task_id for entry in inventory.tasks], ["i1-s0.env"])

    def test_cross_worker_requeue_cannot_overtake_live_task_publication_fence(self) -> None:
        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        first = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(first)
        first_tmp = self.cache_dir / "first.tmp"
        second_tmp = self.cache_dir / "second.tmp"
        self._real_collect(["--out", str(first_tmp), "--seed-start", "100"])
        first_fanin = fleet_worker._fanin_task_from_task_manifest(first)
        lease = fleet_worker._acquire_fanin_task_publication_lease(self.cache_dir, first, first_fanin)
        try:
            first.claim_path.unlink()
            _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
            second = claim_next_task(self.queue, "w2")
            self.assertIsNotNone(second)
            self._real_collect(["--out", str(second_tmp), "--seed-start", "100"])
            with self.assertRaises(fleet_worker._FanInTransientError):
                fleet_worker._publish_fanin_task(second, second_tmp, self.queue, "w2", crash_inject=None)
        finally:
            fleet_worker._release_fanin_task_publication_lease(lease)
        with self.assertRaises(fleet_worker._ClaimRevokedError):
            fleet_worker._publish_fanin_task(first, first_tmp, self.queue, "w1", crash_inject=None)
        target, recovered = fleet_worker._publish_fanin_task(second, second_tmp, self.queue, "w2", crash_inject=None)
        self.assertFalse(recovered)
        self.assertIsNotNone(target)
        self.assertEqual(
            [task.task_id for task in fleet_worker._read_fanin_manifest(target)],
            ["i1-s0.env"],
        )

    def test_task_fence_heartbeat_keeps_foreign_cleanup_off_long_concat(self) -> None:
        from unittest.mock import patch

        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(task)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(task)
        staging = self.cache_dir / ".shard-ww1-v1.tmp.heartbeat"
        staging.mkdir()
        fleet_worker._write_fanin_staging_owner(staging, fanin_task, producer_token=task.claim_token)
        with (
            patch.object(fleet_worker, "_FANIN_PRODUCER_LEASE_SECONDS", 0.04),
            patch.object(fleet_worker, "_FANIN_HEARTBEAT_INTERVAL_SECONDS", 0.005),
        ):
            lease = fleet_worker._acquire_fanin_task_publication_lease(self.cache_dir, task, fanin_task)
            try:
                task.claim_path.unlink()
                time.sleep(0.12)  # Longer than the old one-shot 60 s-equivalent lease.
                fleet_worker._sweep_abandoned_fanin_staging(self.cache_dir, self.queue)
                self.assertTrue(staging.exists())
            finally:
                fleet_worker._release_fanin_task_publication_lease(lease)
            fleet_worker._sweep_abandoned_fanin_staging(self.cache_dir, self.queue)
        self.assertFalse(staging.exists())

    def test_initial_guard_acquirers_never_observe_uninitialized_guard(self) -> None:
        from unittest.mock import patch

        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(task)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(task)
        fence_path = fleet_worker._fanin_task_fence_path(self.cache_dir, task.base)
        fence_path.parent.mkdir(parents=True, exist_ok=True)
        original_write = fleet_worker._write_fanin_fence
        winner = None
        interleaved = False

        def publish_second_after_first_is_initialized(path, *args):
            nonlocal interleaved, winner
            original_write(path, *args)
            if path != fence_path and not interleaved:
                interleaved = True
                winner = fleet_worker._acquire_fanin_guard(fence_path, task, fanin_task)

        with patch.object(fleet_worker, "_write_fanin_fence", new=publish_second_after_first_is_initialized):
            with self.assertRaises(fleet_worker._FanInTransientError):
                fleet_worker._acquire_fanin_guard(fence_path, task, fanin_task)
        self.assertTrue(interleaved)
        self.assertIsNotNone(winner)
        try:
            self.assertIsNotNone(fleet_worker._read_fanin_fence(fence_path))
            self.assertTrue(fleet_worker._fanin_fence_is_current(winner))
            self.assertFalse(list(fence_path.parent.glob(f".{fence_path.name}.tmp.*")))
        finally:
            fleet_worker._release_fanin_guard(winner)

    def test_guard_owner_write_failure_removes_only_attempt_temporary(self) -> None:
        from unittest.mock import patch

        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(task)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(task)
        fence_path = fleet_worker._fanin_task_fence_path(self.cache_dir, task.base)
        fence_path.parent.mkdir(parents=True, exist_ok=True)
        with patch.object(fleet_worker, "_write_fanin_fence", side_effect=OSError(errno.EIO, "owner write failed")):
            with self.assertRaises(OSError):
                fleet_worker._acquire_fanin_guard(fence_path, task, fanin_task)
        self.assertFalse(fence_path.exists())
        self.assertFalse(list(fence_path.parent.glob(f".{fence_path.name}.tmp.*")))

    def test_stale_fence_releaser_cannot_remove_new_reclaimer_fence(self) -> None:
        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        first = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(first)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(first)
        fence_path = fleet_worker._fanin_task_fence_path(self.cache_dir, first.base)
        fence_path.parent.mkdir(parents=True, exist_ok=True)
        first_fence = fleet_worker._acquire_fanin_guard(fence_path, first, fanin_task)

        def expire(fence) -> None:
            fence.stop.set()
            fence.heartbeat.join(timeout=1)
            payload = json.loads((fence.path / "owner.json").read_text(encoding="utf-8"))
            payload["renewed_at"] = 0
            (fence.path / "owner.json").write_text(json.dumps(payload), encoding="utf-8")

        try:
            expire(first_fence)
            first.claim_path.unlink()
            _manifest(self.queue, first.base, out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
            second = claim_next_task(self.queue, "w2")
            self.assertIsNotNone(second)
            second_fence = fleet_worker._acquire_fanin_guard(fence_path, second, fanin_task)
            try:
                expire(second_fence)
                second.claim_path.unlink()
                _manifest(self.queue, first.base, out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
                third = claim_next_task(self.queue, "w3")
                self.assertIsNotNone(third)
                third_fence = fleet_worker._acquire_fanin_guard(fence_path, third, fanin_task)
                try:
                    fleet_worker._release_fanin_guard(second_fence)
                    self.assertTrue(fleet_worker._fanin_fence_is_current(third_fence))
                finally:
                    fleet_worker._release_fanin_guard(third_fence)
            finally:
                fleet_worker._release_fanin_guard(second_fence)
        finally:
            fleet_worker._release_fanin_guard(first_fence)

    def test_guard_release_marker_cannot_remove_a_successor_between_check_and_mutation(self) -> None:
        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        first = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(first)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(first)
        fence_path = fleet_worker._fanin_task_fence_path(self.cache_dir, first.base)
        fence_path.parent.mkdir(parents=True, exist_ok=True)
        first_fence = fleet_worker._acquire_fanin_guard(fence_path, first, fanin_task)
        try:
            first_fence.stop.set()
            first_fence.heartbeat.join(timeout=1)
            payload = json.loads((fence_path / "owner.json").read_text(encoding="utf-8"))
            payload["renewed_at"] = 0
            (fence_path / "owner.json").write_text(json.dumps(payload), encoding="utf-8")
            first.claim_path.unlink()
            _manifest(self.queue, first.base, out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
            second = claim_next_task(self.queue, "w2")
            self.assertIsNotNone(second)
            second_fence = fleet_worker._acquire_fanin_guard(fence_path, second, fanin_task)
            try:
                # This is the former check-then-rename interleaving: the old
                # generation releases only its inode-keyed marker.
                fleet_worker._release_fanin_guard(first_fence)
                self.assertTrue(fleet_worker._fanin_fence_is_current(second_fence))
            finally:
                fleet_worker._release_fanin_guard(second_fence)
        finally:
            fleet_worker._release_fanin_guard(first_fence)

    def test_same_task_retry_reclaims_stale_task_and_publish_records(self) -> None:
        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        first = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(first)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(first)
        first_lease = fleet_worker._acquire_fanin_task_publication_lease(self.cache_dir, first, fanin_task)
        base = self.cache_dir / "shard-ww1"
        first_lock, first_identity = fleet_worker._acquire_fanin_publish_lock(base, first, fanin_task)
        try:
            first_lease.guard.stop.set()
            first_lease.guard.heartbeat.join(timeout=1)
            payload = json.loads((first_lease.guard.path / "owner.json").read_text(encoding="utf-8"))
            payload["renewed_at"] = 0
            (first_lease.guard.path / "owner.json").write_text(json.dumps(payload), encoding="utf-8")
            first.claim_path.unlink()

            _manifest(self.queue, first.base, out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
            second = claim_next_task(self.queue, "w2")
            self.assertIsNotNone(second)
            second_lease = fleet_worker._acquire_fanin_task_publication_lease(self.cache_dir, second, fanin_task)
            try:
                second_lock, second_identity = fleet_worker._acquire_fanin_publish_lock(base, second, fanin_task)
                self.assertNotEqual(first_identity, second_identity)
                self.assertEqual(fleet_worker._read_fanin_task_lock(second_lease.record)[2], second.claim_token)
                fleet_worker._release_fanin_publish_lock(first_lock, first_identity)
                self.assertEqual(second_lock.stat().st_ino, second_identity[1])
            finally:
                fleet_worker._release_fanin_publish_lock(second_lock, second_identity)
                fleet_worker._release_fanin_task_publication_lease(second_lease)
        finally:
            fleet_worker._release_fanin_publish_lock(first_lock, first_identity)
            fleet_worker._release_fanin_task_publication_lease(first_lease)

    def test_target_collision_classifies_enotempty_oserror(self) -> None:
        self.assertTrue(fleet_worker._is_fanin_target_collision(OSError(errno.ENOTEMPTY, "target not empty")))
        self.assertTrue(fleet_worker._is_fanin_target_collision(FileExistsError(errno.EEXIST, "target exists")))
        self.assertFalse(fleet_worker._is_fanin_target_collision(OSError(errno.EIO, "I/O error")))

    def test_cleanup_keeps_concurrently_published_higher_cumulative_version(self) -> None:
        self._manifests(1)

        def publish_higher(boundary: str, _task) -> None:
            if boundary == "fanin-before-stale-cleanup":
                self._write_version(
                    "shard-ww1-v2",
                    [
                        self._fanin_task("i1-s0.env", 1, 0, 1, 100),
                        self._fanin_task("i1-s1.env", 1, 1, 1, 101),
                    ],
                )

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            crash_inject=publish_higher,
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.cache_dir / "shard-ww1-v2").exists())
        inventory = read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=2))
        self.assertEqual([shard.path.name for shard in inventory.shards], ["shard-ww1-v2"])
        self.assertEqual([task.task_id for task in inventory.tasks], ["i1-s0.env", "i1-s1.env"])

    def test_task_lookup_tolerates_unrelated_selected_publish(self) -> None:
        from unittest.mock import patch

        self._write_version("shard-wother-v1", [self._fanin_task("prior-s0.env", 1, 10, 1, 110)])
        self._manifests(1)
        original_read = fleet_worker._read_fanin_manifest
        published = False

        def publish_unrelated(path: Path):
            nonlocal published
            if path.name == "shard-wother-v1" and not published:
                published = True
                self._write_version("shard-wother2-v1", [self._fanin_task("prior-s1.env", 1, 11, 1, 111)])
            return original_read(path)

        with patch.object(fleet_worker, "_read_fanin_manifest", new=publish_unrelated):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )
        self.assertEqual(rc, 0)
        self.assertTrue(published)
        self.assertTrue((self.queue / "done" / "i1-s0.env").exists())

    def test_strict_inventory_rejects_malformed_or_missing_manifest(self) -> None:
        missing = self._write_version("shard-ww1-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 100)])
        (missing / "fanin-manifest.json").unlink()
        with self.assertRaisesRegex(FanInValidationError, "no valid"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))
        (missing / "fanin-manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(FanInValidationError, "malformed manifest envelope"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))

    def test_manifest_schema_rejects_legacy_partial_and_non_route_writer_entries(self) -> None:
        path = self.cache_dir / "shard-ww1-v1"
        path.mkdir()
        legacy = {"schema_version": 1, "kind": "pokezero-fanin-shard", "tasks": []}
        (path / "fanin-manifest.json").write_text(json.dumps(legacy), encoding="utf-8")
        with self.assertRaisesRegex(FanInValidationError, "unsupported manifest schema"):
            fleet_worker._read_fanin_manifest(path)
        partial = {
            "schema_version": fleet_worker.FANIN_MANIFEST_SCHEMA_VERSION,
            "kind": "pokezero-fanin-shard",
            "tasks": [{"task_id": "i1-s0.env", "iteration": 1, "offset": 0, "count": 1, "seed": 100, "out": "out"}],
        }
        (path / "fanin-manifest.json").write_text(json.dumps(partial), encoding="utf-8")
        with self.assertRaisesRegex(FanInValidationError, "malformed task entry"):
            fleet_worker._read_fanin_manifest(path)
        with self.assertRaisesRegex(FanInValidationError, "requires non-empty"):
            fleet_worker._write_fanin_manifest(path, [FanInTask("i1-s0.env", 1, 0, 1, 100, "", "policy")])

    def test_strict_inventory_rejects_cache_game_count_mismatch(self) -> None:
        # _write_version writes one record; the manifest's two-game claim must
        # be rejected before a launcher can include it in training.
        self._write_version("shard-ww1-v1", [self._fanin_task("i1-s0.env", 1, 0, 2, 100)])
        with self.assertRaisesRegex(FanInValidationError, "cache games disagree"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=2))

    def test_strict_inventory_rejects_conflicts_ranges_and_contract_totals(self) -> None:
        self._write_version("shard-ww1-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 100)])
        self._write_version("shard-ww2-v1", [self._fanin_task("i1-s0.env", 1, 1, 1, 101)])
        with self.assertRaisesRegex(FanInValidationError, "conflicting metadata"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=2))

        import shutil

        shutil.rmtree(self.cache_dir / "shard-ww2-v1")
        self._write_version("shard-ww2-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 100)])
        with self.assertRaisesRegex(FanInValidationError, "repeats task id"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=2))

        shutil.rmtree(self.cache_dir / "shard-ww2-v1")
        self._write_version("shard-ww2-v1", [self._fanin_task("i1-s2.env", 1, 2, 1, 102)])
        with self.assertRaisesRegex(FanInValidationError, "gap or overlap in queue offsets"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=2))

        shutil.rmtree(self.cache_dir / "shard-ww2-v1")
        self._write_version("shard-ww2-v1", [self._fanin_task("i1-s1.env", 1, 1, 1, 102)])
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

        tasks = [self._fanin_task("i1-s0.env", 1, 0, 1, 100)]
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

        with (
            patch.object(fleet_worker, "_read_fanin_manifest", new=publish_v5_while_reading),
            patch.object(fleet_worker.time, "sleep") as sleep,
        ):
            inventory = read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))
        self.assertTrue(raced)
        sleep.assert_called_with(fleet_worker._SELECTED_FANIN_RETRY_SECONDS)
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
                self._fanin_task("prior-s0.env", 1, 0, 1, 98),
                self._fanin_task("prior-s1.env", 1, 1, 1, 99),
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
