"""Persistent collector-fleet worker: queue protocol + recycle bounds.

All torch-free — the collect function is stubbed; what is under test is the
claim/commit/revocation/failed transport (which must stay byte-compatible with
the shell fleet worker) and the OOM/task recycle bounds.
"""

from __future__ import annotations

import json
import errno
import hashlib
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
        _manifest(self.queue, "i7-s0.env", out=self.root.resolve() / "route-a")
        task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(task)
        record = fleet_worker._fanin_route_record(self.queue, task.base)
        record.write_text("{", encoding="utf-8")

        with self.assertRaises(FanInValidationError):
            fleet_worker._resolve_fanin_route(self.queue, task)

    def test_fanin_relative_route_fails_before_claim_from_every_working_directory(self) -> None:
        _manifest(self.queue, "i7-s0.env", out=Path("cache") / "slice")
        original_cwd = Path.cwd()
        try:
            for name in ("worker-a", "worker-b"):
                cwd = self.root / name
                cwd.mkdir()
                os.chdir(cwd)
                rc = run_worker(
                    self.queue, worker_id=name, static_argv=[], collect_fn=lambda _argv: 0,
                    max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
                )
                self.assertEqual(rc, 2)
        finally:
            os.chdir(original_cwd)
        self.assertTrue((self.queue / "pending" / "i7-s0.env").exists())
        self.assertFalse(list((self.queue / "claimed").iterdir()))
        self.assertFalse(list((self.queue / "failed").iterdir()))
        self.assertFalse((self.queue / fleet_worker._FANIN_ROUTE_DIRECTORY).exists())

    def test_persisted_fanin_manifest_rejects_relative_output_route(self) -> None:
        shard = self.root.resolve() / "cache" / "shard-wtest-v1"
        shard.mkdir(parents=True)
        (shard / fleet_worker.FANIN_MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "schema_version": fleet_worker.FANIN_MANIFEST_SCHEMA_VERSION,
                    "kind": "pokezero-fanin-shard",
                    "tasks": [
                        {
                            "task_id": "i7-s0.env",
                            "iteration": 7,
                            "offset": 0,
                            "count": 1,
                            "seed": 100,
                            "out": "relative-cache/slice",
                            "policy": "policy",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(FanInValidationError, "non-canonical output route"):
            fleet_worker._read_fanin_manifest(shard)

    def test_guard_generation_symlink_and_owner_replacement_fail_closed(self) -> None:
        from unittest.mock import patch

        cache_dir = self.root.resolve() / "cache"
        guard = fleet_worker._fanin_task_fence_path(cache_dir, "i7-s0.env")
        guard.parent.mkdir(parents=True)
        guard.mkdir()
        task = FanInTask("i7-s0.env", 7, 0, 1, 100, str(cache_dir / "slice"), "policy")
        fleet_worker._write_fanin_fence(guard, task, "i7-s0.env.w1", "guard-token")
        current = fleet_worker._read_current_fanin_fence(guard)
        self.assertIsNotNone(current)

        saved_guard = guard.with_name(f"saved-{guard.name}")
        guard.rename(saved_guard)
        guard.symlink_to(saved_guard, target_is_directory=True)
        with self.assertRaisesRegex(FanInValidationError, "not a real directory"):
            fleet_worker._read_current_fanin_fence(guard)

        guard.unlink()
        saved_guard.rename(guard)
        original_open = fleet_worker.os.open
        replaced = False

        def replace_owner_after_open(name, flags, mode=0o777, *, dir_fd=None):
            nonlocal replaced
            descriptor = original_open(name, flags, mode, dir_fd=dir_fd)
            if (
                name == "owner.json"
                and dir_fd is not None
                and flags & os.O_ACCMODE == os.O_RDONLY
                and not replaced
            ):
                replaced = True
                owner = guard / "owner.json"
                prior = guard / ".owner.prior"
                owner.rename(prior)
                owner.write_bytes(prior.read_bytes())
            return descriptor

        with patch.object(fleet_worker.os, "open", new=replace_owner_after_open):
            with self.assertRaisesRegex(FanInValidationError, "changed (while opening|during read)"):
                fleet_worker._read_current_fanin_fence(guard)
        self.assertTrue(replaced)

    def test_route_record_rejects_symlink_special_file_and_replacement_race(self) -> None:
        from unittest.mock import patch

        _manifest(self.queue, "i7-s0.env", out=self.root.resolve() / "cache" / "slice")
        task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(task)
        fleet_worker._resolve_fanin_route(self.queue, task)
        record = fleet_worker._fanin_route_record(self.queue, task.base)
        backing = record.with_name("route-backing.json")
        record.rename(backing)
        record.symlink_to(backing)
        with self.assertRaisesRegex(FanInValidationError, "regular non-symlink"):
            fleet_worker._resolve_fanin_route(self.queue, task)

        record.unlink()
        alternate = record.with_name("route-alternate.json")
        alternate.write_text('{"replaced":true}\n', encoding="utf-8")
        record.symlink_to(alternate)
        with self.assertRaisesRegex(FanInValidationError, "regular non-symlink"):
            fleet_worker._resolve_fanin_route(self.queue, task)

        record.unlink()
        os.mkfifo(record)
        with self.assertRaisesRegex(FanInValidationError, "regular non-symlink"):
            fleet_worker._resolve_fanin_route(self.queue, task)

        record.unlink()
        record.write_bytes(backing.read_bytes())
        original_open = fleet_worker.os.open
        replaced = False

        def replace_after_open(name, flags, mode=0o777, *, dir_fd=None):
            nonlocal replaced
            descriptor = original_open(name, flags, mode, dir_fd=dir_fd)
            if (
                name == record.name
                and dir_fd is not None
                and flags & os.O_ACCMODE == os.O_RDONLY
                and not replaced
            ):
                replaced = True
                previous = record.with_name("route-prior.json")
                record.rename(previous)
                record.write_bytes(previous.read_bytes())
            return descriptor

        with patch.object(fleet_worker.os, "open", new=replace_after_open):
            with self.assertRaisesRegex(FanInValidationError, "changed (while opening|during read)"):
                fleet_worker._resolve_fanin_route(self.queue, task)
        self.assertTrue(replaced)

    def test_fanin_post_preflight_manifest_replacement_is_terminal_and_retains_claim(self) -> None:
        from unittest.mock import patch

        pending = _manifest(
            self.queue, "i7-s0.env", out=self.root.resolve() / "cache" / "slice", policy="policy-a",
        )
        original_preflight = fleet_worker._preflight_fanin_filesystems
        calls: list[list[str]] = []

        def replace_after_preflight(queue: Path, cache_dir: Path) -> None:
            original_preflight(queue, cache_dir)
            pending.write_text(
                pending.read_text(encoding="utf-8").replace("policy-a", "policy-b"),
                encoding="utf-8",
            )

        with patch.object(fleet_worker, "_preflight_fanin_filesystems", new=replace_after_preflight):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=calls.append,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])
        self.assertFalse((self.queue / "pending" / "i7-s0.env").exists())
        self.assertEqual(
            len(list((self.queue / "claimed").glob("i7-s0.env.w1.*"))),
            1,
        )
        self.assertFalse(list((self.queue / "failed").iterdir()))
        self.assertFalse((self.queue / fleet_worker._FANIN_ROUTE_DIRECTORY).exists())

    def test_fanin_claim_lease_binds_the_manifest_generation_that_was_parsed(self) -> None:
        from unittest.mock import patch

        _manifest(
            self.queue, "i7-s0.env", out=self.root.resolve() / "cache" / "slice", policy="policy-a",
        )
        original_write = fleet_worker._write_claim_token
        replaced = False

        def replace_after_parse(claim: Path, **kwargs) -> str:
            nonlocal replaced
            replaced = True
            claim.write_text(
                claim.read_text(encoding="utf-8").replace("policy-a", "policy-b"),
                encoding="utf-8",
            )
            return original_write(claim, **kwargs)

        with patch.object(fleet_worker, "_write_claim_token", new=replace_after_parse):
            with self.assertRaises(FanInValidationError):
                claim_next_task(self.queue, "w1", fanin=True)
        self.assertTrue(replaced)
        self.assertFalse((self.queue / "pending" / "i7-s0.env").exists())
        self.assertEqual(len(list((self.queue / "claimed").glob("i7-s0.env.w1.*"))), 1)
        self.assertFalse(list((self.queue / "done").iterdir()))
        self.assertFalse(list((self.queue / "failed").iterdir()))

    def test_fanin_pending_parent_replacement_after_preflight_is_terminal(self) -> None:
        import shutil

        _manifest(self.queue, "i7-s0.env", out=self.root.resolve() / "cache" / "slice")
        pending = self.queue / "pending"
        replacement = self.root / "replacement-pending"
        saved = self.root / "saved-pending"

        def replace_parent(_candidate: Path, _preview) -> None:
            shutil.copytree(pending, replacement)
            pending.rename(saved)
            replacement.rename(pending)

        with self.assertRaises(FanInValidationError):
            claim_next_task(self.queue, "w1", fanin=True, before_claim=replace_parent)
        self.assertTrue((pending / "i7-s0.env").exists())
        self.assertFalse(list((self.queue / "done").iterdir()))
        self.assertFalse(list((self.queue / "failed").iterdir()))

    def test_fanin_claim_token_requires_leased_manifest_generation(self) -> None:
        from unittest.mock import patch
        import shutil

        _manifest(self.queue, "i7-s0.env", out=self.root.resolve() / "cache" / "slice")
        original_move = fleet_worker._move_fanin_manifest_into_claim
        saved = self.root / "saved-claim"

        def move_then_replace(candidate: Path, claim: Path, preview):
            task = original_move(candidate, claim, preview)
            claim.rename(saved)
            shutil.copy2(saved, claim)
            return task

        with patch.object(
            fleet_worker, "_move_fanin_manifest_into_claim", new=move_then_replace,
        ):
            with self.assertRaises(FanInValidationError):
                claim_next_task(self.queue, "w1", fanin=True)
        self.assertTrue(saved.exists())
        self.assertTrue((self.queue / "claimed").exists())
        self.assertFalse(list((self.queue / "done").iterdir()))

    def test_fanin_claim_token_rejects_claimed_parent_replacement(self) -> None:
        import shutil

        _manifest(self.queue, "i7-s0.env", out=self.root.resolve() / "cache" / "slice")
        task = claim_next_task(self.queue, "w1", fanin=True)
        self.assertIsNotNone(task)
        claimed = self.queue / "claimed"
        replacement = self.root / "replacement-claimed"
        saved = self.root / "saved-claimed"
        shutil.copytree(claimed, replacement)
        claimed.rename(saved)
        replacement.rename(claimed)

        with self.assertRaises(FanInValidationError):
            fleet_worker._fanin_claim_manifest_is_current(task)
        self.assertFalse(list((self.queue / "done").iterdir()))

    def test_fanin_route_record_replacement_after_snapshot_fails_closed(self) -> None:
        from unittest.mock import patch

        _manifest(self.queue, "i7-s0.env", out=self.root.resolve() / "cache" / "slice")
        task = claim_next_task(self.queue, "w1", fanin=True)
        self.assertIsNotNone(task)
        fleet_worker._resolve_fanin_route(self.queue, task)
        record = fleet_worker._fanin_route_record(self.queue, task.base)
        original_verify = fleet_worker._verify_fanin_authoritative_directory_identity
        replaced = False

        def replace_after_directory_check(path: Path, expected, label: str) -> None:
            nonlocal replaced
            original_verify(path, expected, label)
            if path == record.parent and label == "route provenance directory" and not replaced:
                replaced = True
                prior = record.with_name("route-before-final-check.json")
                record.rename(prior)
                record.write_bytes(prior.read_bytes())

        with patch.object(
            fleet_worker, "_verify_fanin_authoritative_directory_identity",
            new=replace_after_directory_check,
        ):
            with self.assertRaises(FanInValidationError):
                fleet_worker._resolve_fanin_route(self.queue, task)
        self.assertTrue(replaced)

    def test_fanin_route_record_replacement_after_parse_fails_closed(self) -> None:
        from unittest.mock import patch

        _manifest(self.queue, "i7-s0.env", out=self.root.resolve() / "cache" / "slice")
        task = claim_next_task(self.queue, "w1", fanin=True)
        self.assertIsNotNone(task)
        fleet_worker._resolve_fanin_route(self.queue, task)
        record = fleet_worker._fanin_route_record(self.queue, task.base)
        original_parse = fleet_worker._fanin_route_from_payload
        replaced = False

        def parse_then_replace(payload):
            nonlocal replaced
            result = original_parse(payload)
            if result is not None and not replaced:
                replaced = True
                prior = record.with_name("route-before-return.json")
                record.rename(prior)
                record.write_bytes(prior.read_bytes())
            return result

        with patch.object(fleet_worker, "_fanin_route_from_payload", new=parse_then_replace):
            with self.assertRaises(FanInValidationError):
                fleet_worker._resolve_fanin_route(self.queue, task)
        self.assertTrue(replaced)

    def test_route_directory_preflight_failure_leaves_queue_and_route_state_untouched(self) -> None:
        from unittest.mock import patch

        _manifest(self.queue, "i7-s0.env", out=self.root.resolve() / "cache" / "slice")
        route_directory = self.queue / fleet_worker._FANIN_ROUTE_DIRECTORY
        original_link = fleet_worker.os.link

        def unsupported_route_link(source, destination, *args, **kwargs):
            if Path(destination).parent == route_directory:
                raise OSError(errno.EOPNOTSUPP, "route CAS unsupported")
            return original_link(source, destination, *args, **kwargs)

        with patch.object(fleet_worker.os, "link", new=unsupported_route_link):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=lambda _argv: 0,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )
        self.assertEqual(rc, 2)
        self.assertTrue((self.queue / "pending" / "i7-s0.env").exists())
        self.assertFalse(list((self.queue / "claimed").iterdir()))
        self.assertFalse(list((self.queue / "failed").iterdir()))
        self.assertFalse(route_directory.exists())


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


class FanInFixture(unittest.TestCase):
    """Temp queue + cache dir and the shard/task builders every fan-in test uses.

    Held apart from the tests so other fan-in suites can reuse the fixture
    without inheriting (and re-running) this one's cases.
    """

    def setUp(self) -> None:
        import tempfile

        from tests.test_cache_concat import NUMPY

        if not NUMPY:
            self.skipTest("requires numpy")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
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

    def _mutate_cache_payload_in_place(self, cache: Path) -> None:
        """Change array bytes without replacing the accepted payload pathname."""
        payload = next(cache.glob("*.npy"))
        with payload.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            original = handle.read(1)
            self.assertEqual(len(original), 1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([original[0] ^ 1]))
            handle.flush()
            os.fsync(handle.fileno())

    def _replace_cache_payload_identically(self, cache: Path) -> None:
        """Keep bytes stable while replacing the accepted payload inode."""
        payload = next(cache.glob("*.npy"))
        replacement = payload.with_name(f".{payload.name}.replacement")
        with replacement.open("xb") as handle:
            handle.write(payload.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(replacement, payload)
        fleet_worker._fsync_directory(payload.parent)

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

    def _route_witness(self, task: FanInTask) -> fleet_worker._FanInRouteResolution:
        """Create the same immutable route record production acceptance retains."""
        record = fleet_worker._fanin_route_record(self.queue, task.task_id)
        payload = {
            "schema_version": 1,
            "cache_dir": str(self.cache_dir),
            "task": fleet_worker._fanin_task_payload(task),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            with record.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            fleet_worker._fsync_directory(record.parent)
        except FileExistsError:
            self.assertEqual(record.read_bytes(), encoded)
        contents, observed, parent_identity = fleet_worker._read_fanin_file_with_parent_snapshot(
            record, "route provenance record",
        )
        self.assertEqual(contents, encoded)
        self.assertIsNotNone(observed)
        root = fleet_worker._fanin_authoritative_directory_stat(self.cache_dir, "fan-in cache root")
        return fleet_worker._FanInRouteResolution(
            root=self.cache_dir,
            root_identity=fleet_worker._fanin_directory_identity(root),
            record=record,
            record_parent_identity=parent_identity,
            record_identity=fleet_worker._fanin_durable_stat_snapshot(observed),
            record_sha256=hashlib.sha256(encoded).hexdigest(),
            task=task,
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
        lineage, _, version_text = name.rpartition("-v")
        final_acceptance = None
        for task_index, task in enumerate(tasks):
            root = fleet_worker._fanin_task_fence_path(self.cache_dir, task.task_id)
            root.parent.mkdir(parents=True, exist_ok=True)
            current = fleet_worker._read_current_fanin_fence(root)
            if current is None:
                token = f"fixture-{task.task_id}-{name}"
                root.mkdir()
                fleet_worker._write_fanin_fence(root, task, f"{task.task_id}.fixture", token)
                stat = root.stat()
                record = fleet_worker._read_fanin_fence(root)
                self.assertIsNotNone(record)
                current = fleet_worker._FanInFenceGeneration(
                    root, (stat.st_dev, stat.st_ino), record,
                )
                acceptance = fleet_worker._FanInAcceptance(
                    task=task,
                    guard_root=root.name,
                    guard_generation=root.name,
                    claim_name=f"{task.task_id}.fixture",
                    claim_token=token,
                    lineage=lineage,
                    target=name,
                    version=int(version_text),
                    task_index=task_index,
                    prefix_sha256=fleet_worker._fanin_manifest_prefix_sha256(tasks, task_index),
                    manifest_sha256=fleet_worker._sha256_file(target / fleet_worker.FANIN_MANIFEST_NAME),
                    metadata_sha256=fleet_worker._sha256_file(target / "metadata.json"),
                    content_sha256=fleet_worker._fanin_content_sha256(target),
                    record_count=fleet_worker._cache_record_count(target),
                    payload_files=fleet_worker._snapshot_fanin_payload_files(target),
                    route=self._route_witness(task),
                )
                self.assertTrue(
                    fleet_worker._publish_initialized_fanin_acceptance(
                        fleet_worker._fanin_fence_successor_path(root, current), acceptance,
                    )
                )
                current = fleet_worker._read_current_fanin_fence(root)
            self.assertIsNotNone(current)
            self.assertIsNotNone(current.acceptance)
            final_acceptance = current.acceptance
        self.assertIsNotNone(final_acceptance)
        fleet_worker._write_fanin_publication(target, final_acceptance)
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


class FanInTests(FanInFixture):
    """Shard fan-in: one worker-owned versioned shard per window."""

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

    def test_target_first_recovery_rejects_identical_route_record_replacement(self) -> None:
        self._manifests(1)
        calls: list[list[str]] = []

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        def crash_after_target(boundary: str, _task) -> None:
            if boundary == "fanin-after-target-publication":
                raise SystemExit("simulated target-first crash")

        with self.assertRaises(SystemExit):
            run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
                crash_inject=crash_after_target,
            )
        target = self.cache_dir / "shard-ww1-v1"
        self.assertTrue(target.exists())
        record = fleet_worker._fanin_route_record(self.queue, "i1-s0.env")
        previous = record.with_name("route-before-target-recovery.json")
        record.rename(previous)
        record.write_bytes(previous.read_bytes())
        fleet_worker._fsync_directory(record.parent)
        self._requeue_claim("w1", "i1-s0.env")

        rc = run_worker(
            self.queue, worker_id="w2", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )

        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self._claim("w2", "i1-s0.env").exists())
        self.assertFalse((self.queue / "done" / "i1-s0.env").exists())
        root = fleet_worker._fanin_task_fence_path(self.cache_dir, "i1-s0.env")
        current = fleet_worker._read_current_fanin_fence(root)
        self.assertIsNotNone(current)
        self.assertIsNone(current.acceptance)

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
        # The pre-claim cache probe creates the candidate cache root before
        # route provenance rejects the changed parent.
        self.assertTrue(other_root.exists())
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
        self._write_version(
            "shard-wother-v1",
            [self._fanin_task("i1-s0.env", 1, 0, 1, 999, out=self.cache_dir / "slice-0")],
        )
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

    def test_duplicate_target_without_matching_publication_fails_closed(self) -> None:
        task = self._fanin_task(
            "i1-s0.env", 1, 0, 1, 100, out=self.cache_dir / "slice-0",
        )
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

    def test_task_fence_heartbeat_keeps_real_long_concat_and_publication_live(self) -> None:
        from unittest.mock import patch
        from pokezero.dataset import concat_training_caches as real_concat

        prior = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        self._write_version("shard-ww1-v1", [prior])
        _manifest(
            self.queue, "i1-s1.env", out=self.cache_dir / "slice-1", iteration=1,
            offset=1, count=1, seed=101,
        )
        task_fence = fleet_worker._fanin_task_fence_path(self.cache_dir, "i1-s1.env")
        original_write = fleet_worker._write_fanin_fence
        renewals = 0

        def record_renewal(path, *args):
            nonlocal renewals
            original_write(path, *args)
            if path == task_fence:
                renewals += 1

        def slow_concat(parts, staging):
            # Keep real path-based concat active long enough for many owner
            # replacements, rather than exercising only the liveness helper.
            for _ in range(16):
                time.sleep(0.01)
            return real_concat(parts, staging)

        with (
            patch.object(fleet_worker, "_FANIN_PRODUCER_LEASE_SECONDS", 0.04),
            patch.object(fleet_worker, "_FANIN_HEARTBEAT_INTERVAL_SECONDS", 0.005),
            patch.object(fleet_worker, "_write_fanin_fence", new=record_renewal),
            patch("pokezero.dataset.concat_training_caches", new=slow_concat),
        ):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )

        self.assertEqual(rc, 0)
        self.assertGreaterEqual(renewals, 2)
        self.assertTrue((self.queue / "done" / "i1-s1.env").exists())
        self.assertEqual(
            read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=2)).total_games, 2,
        )

    def test_heartbeat_guard_directory_replacement_fails_closed(self) -> None:
        import shutil
        from unittest.mock import patch

        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(task)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(task)
        original_write = fleet_worker._write_fanin_fence
        renewals = 0

        def record_renewal(path, *args):
            nonlocal renewals
            original_write(path, *args)
            if path == fleet_worker._fanin_task_fence_path(self.cache_dir, task.base):
                renewals += 1

        with (
            patch.object(fleet_worker, "_FANIN_PRODUCER_LEASE_SECONDS", 0.04),
            patch.object(fleet_worker, "_FANIN_HEARTBEAT_INTERVAL_SECONDS", 0.005),
            patch.object(fleet_worker, "_write_fanin_fence", new=record_renewal),
        ):
            lease = fleet_worker._acquire_fanin_task_publication_lease(self.cache_dir, task, fanin_task)
            try:
                time.sleep(0.03)
                lease.guard.stop.set()
                lease.guard.heartbeat.join(timeout=1)
                replacement = lease.guard.path.with_name(f"replacement-{lease.guard.path.name}")
                saved = lease.guard.path.with_name(f"saved-{lease.guard.path.name}")
                shutil.copytree(lease.guard.path, replacement)
                lease.guard.path.rename(saved)
                replacement.rename(lease.guard.path)
                self.assertGreaterEqual(renewals, 1)
                self.assertFalse(fleet_worker._fanin_fence_is_current(lease.guard))
                with self.assertRaises(FanInValidationError):
                    fleet_worker._verify_fanin_directory_identity(
                        lease.guard.path, lease.guard.identity, "publication fence directory",
                    )
            finally:
                fleet_worker._release_fanin_task_publication_lease(lease)

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

    def test_two_reclaimers_publish_only_one_deterministic_successor(self) -> None:
        from unittest.mock import patch

        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        predecessor_task = claim_next_task(self.queue, "w0")
        self.assertIsNotNone(predecessor_task)
        predecessor_fanin = fleet_worker._fanin_task_from_task_manifest(predecessor_task)
        fence_path = fleet_worker._fanin_task_fence_path(self.cache_dir, predecessor_task.base)
        fence_path.parent.mkdir(parents=True, exist_ok=True)
        predecessor = fleet_worker._acquire_fanin_guard(fence_path, predecessor_task, predecessor_fanin)
        winner = None
        interleaved = False
        try:
            predecessor.stop.set()
            predecessor.heartbeat.join(timeout=1)
            payload = json.loads((predecessor.path / "owner.json").read_text(encoding="utf-8"))
            payload["renewed_at"] = 0
            (predecessor.path / "owner.json").write_text(json.dumps(payload), encoding="utf-8")
            predecessor_task.claim_path.unlink()

            _manifest(self.queue, "i1-s1.env", out=self.cache_dir / "slice-1", iteration=1, offset=1, count=1, seed=101)
            _manifest(self.queue, "i1-s2.env", out=self.cache_dir / "slice-2", iteration=1, offset=2, count=1, seed=102)
            task_a = claim_next_task(self.queue, "a")
            task_b = claim_next_task(self.queue, "b")
            self.assertIsNotNone(task_a)
            self.assertIsNotNone(task_b)
            fanin_a = fleet_worker._fanin_task_from_task_manifest(task_a)
            fanin_b = fleet_worker._fanin_task_from_task_manifest(task_b)
            original_publish = fleet_worker._publish_initialized_fanin_generation

            def publish_a_after_b_authorizes(target: Path, task, fanin_task):
                nonlocal winner, interleaved
                if task.claim_token == task_b.claim_token and not interleaved:
                    interleaved = True
                    winner = fleet_worker._acquire_fanin_guard(fence_path, task_a, fanin_a)
                return original_publish(target, task, fanin_task)

            with patch.object(
                fleet_worker, "_publish_initialized_fanin_generation", new=publish_a_after_b_authorizes,
            ):
                with self.assertRaises(fleet_worker._FanInTransientError):
                    fleet_worker._acquire_fanin_guard(fence_path, task_b, fanin_b)

            self.assertTrue(interleaved)
            self.assertIsNotNone(winner)
            current = fleet_worker._read_current_fanin_fence(fence_path)
            self.assertIsNotNone(current)
            self.assertEqual(current.path, winner.path)
            self.assertTrue(fleet_worker._fanin_fence_is_current(winner))
            self.assertFalse(fleet_worker._fanin_fence_is_current(predecessor))
        finally:
            if winner is not None:
                fleet_worker._release_fanin_guard(winner)
            fleet_worker._release_fanin_guard(predecessor)

    def test_malformed_successor_generation_fails_closed(self) -> None:
        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(task)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(task)
        fence_path = fleet_worker._fanin_task_fence_path(self.cache_dir, task.base)
        fence_path.parent.mkdir(parents=True, exist_ok=True)
        fence = fleet_worker._acquire_fanin_guard(fence_path, task, fanin_task)
        try:
            fleet_worker._release_fanin_guard(fence)
            current = fleet_worker._read_current_fanin_fence(fence_path)
            successor = fleet_worker._fanin_fence_successor_path(fence_path, current)
            successor.mkdir()
            (successor / "owner.json").write_text("{", encoding="utf-8")
            with self.assertRaises(fleet_worker.FanInInventoryValidationError):
                fleet_worker._acquire_fanin_guard(fence_path, task, fanin_task)
            self.assertTrue(successor.exists())
        finally:
            fleet_worker._release_fanin_guard(fence)

    def test_guard_chain_traversal_is_bounded(self) -> None:
        from unittest.mock import patch

        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(task)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(task)
        fence_path = fleet_worker._fanin_task_fence_path(self.cache_dir, task.base)
        fence_path.parent.mkdir(parents=True, exist_ok=True)
        first = fleet_worker._acquire_fanin_guard(fence_path, task, fanin_task)
        second = None
        try:
            fleet_worker._release_fanin_guard(first)
            second = fleet_worker._acquire_fanin_guard(fence_path, task, fanin_task)
            with patch.object(fleet_worker, "_FANIN_GUARD_CHAIN_MAX_GENERATIONS", 1):
                with self.assertRaises(fleet_worker.FanInInventoryValidationError):
                    fleet_worker._read_current_fanin_fence(fence_path)
        finally:
            if second is not None:
                fleet_worker._release_fanin_guard(second)
            fleet_worker._release_fanin_guard(first)

    def test_missing_middle_generation_with_surviving_descendant_fails_closed(self) -> None:
        import shutil

        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        first_task = claim_next_task(self.queue, "w1")
        self.assertIsNotNone(first_task)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(first_task)
        fence_path = fleet_worker._fanin_task_fence_path(self.cache_dir, first_task.base)
        fence_path.parent.mkdir(parents=True, exist_ok=True)
        fences = []

        def expire(fence, task) -> None:
            fence.stop.set()
            fence.heartbeat.join(timeout=1)
            payload = json.loads((fence.path / "owner.json").read_text(encoding="utf-8"))
            payload["renewed_at"] = 0
            (fence.path / "owner.json").write_text(json.dumps(payload), encoding="utf-8")
            task.claim_path.unlink()

        try:
            first = fleet_worker._acquire_fanin_guard(fence_path, first_task, fanin_task)
            fences.append(first)
            expire(first, first_task)
            _manifest(self.queue, first_task.base, out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
            second_task = claim_next_task(self.queue, "w2")
            self.assertIsNotNone(second_task)
            second = fleet_worker._acquire_fanin_guard(fence_path, second_task, fanin_task)
            fences.append(second)
            expire(second, second_task)
            _manifest(self.queue, first_task.base, out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
            third_task = claim_next_task(self.queue, "w3")
            self.assertIsNotNone(third_task)
            third = fleet_worker._acquire_fanin_guard(fence_path, third_task, fanin_task)
            fences.append(third)

            shutil.rmtree(second.path)
            with self.assertRaisesRegex(
                fleet_worker.FanInInventoryValidationError, "unreachable generations",
            ):
                fleet_worker._read_current_fanin_fence(fence_path)
            with self.assertRaises(fleet_worker.FanInInventoryValidationError):
                fleet_worker._acquire_fanin_guard(fence_path, third_task, fanin_task)
            self.assertFalse(second.path.exists())
            self.assertTrue(third.path.exists())
        finally:
            for fence in reversed(fences):
                fleet_worker._release_fanin_guard(fence)

    def test_successor_acceptance_excludes_predecessor_target_after_stale_check(self) -> None:
        from unittest.mock import patch

        _manifest(self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1, offset=0, count=1, seed=100)
        predecessor = claim_next_task(self.queue, "predecessor")
        self.assertIsNotNone(predecessor)
        predecessor_tmp = self.cache_dir / "predecessor.tmp"
        self._real_collect(["--out", str(predecessor_tmp), "--seed-start", "100"])
        task_root = fleet_worker._fanin_task_fence_path(self.cache_dir, predecessor.base)
        captured_fence = None
        successor_target = None
        interleaved = False
        original_start = fleet_worker._start_fanin_guard_heartbeat

        def capture_task_fence(root, path, task, fanin_task):
            nonlocal captured_fence
            fence = original_start(root, path, task, fanin_task)
            if root == task_root and task.claim_token == predecessor.claim_token:
                captured_fence = fence
            return fence

        def publish_successor_after_predecessor_check(boundary: str, _task) -> None:
            nonlocal interleaved, successor_target
            if boundary != "fanin-after-publication-fence-check" or interleaved:
                return
            interleaved = True
            self.assertIsNotNone(captured_fence)
            captured_fence.stop.set()
            captured_fence.heartbeat.join(timeout=1)
            owner = captured_fence.path / "owner.json"
            payload = json.loads(owner.read_text(encoding="utf-8"))
            payload["renewed_at"] = 0
            owner.write_text(json.dumps(payload), encoding="utf-8")
            predecessor.claim_path.unlink()
            _manifest(
                self.queue, predecessor.base, out=self.cache_dir / "slice-0",
                iteration=1, offset=0, count=1, seed=100,
            )
            successor = claim_next_task(self.queue, "successor")
            self.assertIsNotNone(successor)
            successor_tmp = self.cache_dir / "successor.tmp"
            self._real_collect(["--out", str(successor_tmp), "--seed-start", "100"])
            successor_target, recovered = fleet_worker._publish_fanin_task(
                successor, successor_tmp, self.queue, "successor", crash_inject=None,
            )
            self.assertFalse(recovered)

        with patch.object(
            fleet_worker, "_start_fanin_guard_heartbeat", new=capture_task_fence,
        ):
            target, recovered = fleet_worker._publish_fanin_task(
                predecessor, predecessor_tmp, self.queue, "predecessor",
                crash_inject=publish_successor_after_predecessor_check,
            )

        self.assertTrue(interleaved)
        self.assertIsNone(target)
        self.assertTrue(recovered)
        self.assertIsNotNone(successor_target)
        self.assertTrue((self.cache_dir / "shard-wpredecessor-v1").exists())
        self.assertTrue((self.cache_dir / "shard-wsuccessor-v1").exists())
        inventory = read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))
        self.assertEqual([shard.path.name for shard in inventory.shards], ["shard-wsuccessor-v1"])
        self.assertEqual([task.task_id for task in inventory.tasks], ["i1-s0.env"])

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
                        self._fanin_task(
                            "i1-s0.env", 1, 0, 1, 100, out=self.cache_dir / "slice-0",
                        ),
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
        original_read = fleet_worker._read_fanin_manifest_snapshot
        published = False

        def publish_unrelated(path: Path):
            nonlocal published
            if path.name == "shard-wother-v1" and not published:
                published = True
                self._write_version("shard-wother2-v1", [self._fanin_task("prior-s1.env", 1, 11, 1, 111)])
            return original_read(path)

        with patch.object(fleet_worker, "_read_fanin_manifest_snapshot", new=publish_unrelated):
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

    def test_strict_inventory_rejects_missing_publication_provenance(self) -> None:
        target = self._write_version(
            "shard-ww1-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 100)],
        )
        (target / fleet_worker.FANIN_PUBLICATION_NAME).unlink()
        with self.assertRaisesRegex(FanInValidationError, "missing or malformed publication"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))

    def test_content_hash_accepts_honest_directories_and_files(self) -> None:
        root = self.root / "honest-cache"
        nested = root / "nested"
        nested.mkdir(parents=True)
        (root / "root.bin").write_bytes(b"root")
        (nested / "child.bin").write_bytes(b"child")

        first = fleet_worker._fanin_content_sha256(root)
        self.assertEqual(first, fleet_worker._fanin_content_sha256(root))
        (nested / "later.bin").write_bytes(b"later")
        self.assertNotEqual(first, fleet_worker._fanin_content_sha256(root))

    def test_content_hash_rejects_injected_directory_symlink(self) -> None:
        target = self._write_version(
            "shard-ww1-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 100)],
        )
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "escaped.bin").write_bytes(b"outside")
        (target / "directory-link").symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(FanInValidationError, "symlinked cache entry"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))

    def test_content_hash_rejects_file_symlink(self) -> None:
        target = self._write_version(
            "shard-ww1-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 100)],
        )
        outside = self.root / "outside.bin"
        outside.write_bytes(b"outside")
        (target / "file-link").symlink_to(outside)

        with self.assertRaisesRegex(FanInValidationError, "symlinked cache entry"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))

    def test_selected_root_directory_symlink_fails_closed(self) -> None:
        target = self._write_version(
            "shard-ww1-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 100)],
        )
        replacement = target.with_name("saved-shard-w1-v1")
        target.rename(replacement)
        target.symlink_to(replacement, target_is_directory=True)

        with self.assertRaisesRegex(FanInValidationError, "not a real directory"):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))

    def test_renamed_guard_and_outcome_symlink_substitutions_fail_closed(self) -> None:
        task = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        self._write_version("shard-ww1-v1", [task])
        root = fleet_worker._fanin_task_fence_path(self.cache_dir, task.task_id)
        saved_root = root.with_name(f"saved-{root.name}")
        root.rename(saved_root)
        root.symlink_to(saved_root, target_is_directory=True)
        with self.assertRaisesRegex(FanInValidationError, "not a real directory"):
            fleet_worker._read_current_fanin_fence(root)

        root.unlink()
        saved_root.rename(root)
        current = fleet_worker._read_current_fanin_fence(root)
        self.assertIsNotNone(current)
        outcome = fleet_worker._fanin_fence_successor_path(root, current)
        saved_outcome = outcome.with_name(f"saved-{outcome.name}")
        outcome.rename(saved_outcome)
        outcome.symlink_to(saved_outcome, target_is_directory=True)
        with self.assertRaisesRegex(FanInValidationError, "not a real directory"):
            fleet_worker._read_current_fanin_fence(root)

    def test_guard_owner_and_acceptance_replacement_races_fail_closed(self) -> None:
        from unittest.mock import patch

        task = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        self._write_version("shard-ww1-v1", [task])
        root = fleet_worker._fanin_task_fence_path(self.cache_dir, task.task_id)
        current = fleet_worker._read_current_fanin_fence(root)
        self.assertIsNotNone(current)
        outcome = fleet_worker._fanin_fence_successor_path(root, current)

        for name, parent in (("owner.json", root), (fleet_worker._FANIN_ACCEPTANCE_NAME, outcome)):
            with self.subTest(name=name):
                original_open = fleet_worker.os.open
                replaced = False

                def replace_after_open(opened_name, flags, mode=0o777, *, dir_fd=None):
                    nonlocal replaced
                    descriptor = original_open(opened_name, flags, mode, dir_fd=dir_fd)
                    if (
                        opened_name == name
                        and dir_fd is not None
                        and flags & os.O_ACCMODE == os.O_RDONLY
                        and not replaced
                    ):
                        replaced = True
                        record = parent / name
                        prior = parent / f".{name}.prior"
                        record.rename(prior)
                        record.write_bytes(prior.read_bytes())
                    return descriptor

                with patch.object(fleet_worker.os, "open", new=replace_after_open):
                    # Any no-follow identity check may notice this replacement;
                    # the invariant is rejection, not its exact phase.
                    with self.assertRaises(FanInValidationError):
                        fleet_worker._read_current_fanin_fence(root)
                self.assertTrue(replaced)

    def test_guard_chain_rechecks_earlier_generation_identity_before_returning(self) -> None:
        from unittest.mock import patch
        import shutil

        _manifest(
            self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1,
            offset=0, count=1, seed=100,
        )
        task = claim_next_task(self.queue, "w1", fanin=True)
        self.assertIsNotNone(task)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(task)
        root = fleet_worker._fanin_task_fence_path(self.cache_dir, task.base)
        root.parent.mkdir(parents=True, exist_ok=True)
        first = fleet_worker._acquire_fanin_guard(root, task, fanin_task)
        second = None
        try:
            fleet_worker._release_fanin_guard(first)
            task.claim_path.unlink()
            _manifest(
                self.queue, task.base, out=self.cache_dir / "slice-0", iteration=1,
                offset=0, count=1, seed=100,
            )
            retry = claim_next_task(self.queue, "w2", fanin=True)
            self.assertIsNotNone(retry)
            second = fleet_worker._acquire_fanin_guard(root, retry, fanin_task)
            original_read = fleet_worker._read_fanin_guard_directory
            replaced = False

            def replace_root_after_successor(path: Path):
                nonlocal replaced
                result = original_read(path)
                if path == second.path and not replaced:
                    replaced = True
                    prior = root.with_name(f"prior-{root.name}")
                    root.rename(prior)
                    shutil.copytree(prior, root)
                return result

            with patch.object(
                fleet_worker, "_read_fanin_guard_directory", new=replace_root_after_successor,
            ):
                with self.assertRaises(FanInValidationError):
                    fleet_worker._read_current_fanin_fence(root)
            self.assertTrue(replaced)
        finally:
            if second is not None:
                fleet_worker._release_fanin_guard(second)
            fleet_worker._release_fanin_guard(first)

    def test_guard_chain_carries_successor_identity_to_next_step(self) -> None:
        from unittest.mock import patch
        import shutil

        _manifest(
            self.queue, "i1-s0.env", out=self.cache_dir / "slice-0", iteration=1,
            offset=0, count=1, seed=100,
        )
        first_task = claim_next_task(self.queue, "w1", fanin=True)
        self.assertIsNotNone(first_task)
        fanin_task = fleet_worker._fanin_task_from_task_manifest(first_task)
        root = fleet_worker._fanin_task_fence_path(self.cache_dir, first_task.base)
        root.parent.mkdir(parents=True, exist_ok=True)
        first = fleet_worker._acquire_fanin_guard(root, first_task, fanin_task)
        second = None
        try:
            fleet_worker._release_fanin_guard(first)
            first_task.claim_path.unlink()
            _manifest(
                self.queue, first_task.base, out=self.cache_dir / "slice-0", iteration=1,
                offset=0, count=1, seed=100,
            )
            second_task = claim_next_task(self.queue, "w2", fanin=True)
            self.assertIsNotNone(second_task)
            second = fleet_worker._acquire_fanin_guard(root, second_task, fanin_task)
            original_read = fleet_worker._read_fanin_guard_directory
            replaced = False

            def replace_successor_after_read(path: Path):
                nonlocal replaced
                result = original_read(path)
                if path == second.path and not replaced:
                    replaced = True
                    prior = second.path.with_name(f"prior-{second.path.name}")
                    second.path.rename(prior)
                    shutil.copytree(prior, second.path)
                return result

            with patch.object(
                fleet_worker, "_read_fanin_guard_directory", new=replace_successor_after_read,
            ):
                with self.assertRaises(FanInValidationError):
                    fleet_worker._read_current_fanin_fence(root)
            self.assertTrue(replaced)
        finally:
            if second is not None:
                fleet_worker._release_fanin_guard(second)
            fleet_worker._release_fanin_guard(first)

    def test_committed_lookup_rejects_manifest_changed_after_selection(self) -> None:
        from unittest.mock import patch

        task = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        target = self._write_version("shard-ww1-v1", [task])
        manifest = target / fleet_worker.FANIN_MANIFEST_NAME
        original_select = fleet_worker._select_accepted_fanin_shards
        replaced = False

        def select_then_replace(cache_dir: Path):
            nonlocal replaced
            selected = original_select(cache_dir)
            if not replaced:
                replaced = True
                contents = manifest.read_bytes()
                with manifest.open("r+b") as handle:
                    handle.write(contents)
                    handle.flush()
                    os.fsync(handle.fileno())
            return selected

        with patch.object(fleet_worker, "_select_accepted_fanin_shards", new=select_then_replace):
            with self.assertRaises(FanInValidationError):
                fleet_worker._find_committed_fanin_task(self.cache_dir, task)
        self.assertTrue(replaced)

    def test_selected_shard_rejects_acceptance_record_replacement_after_selection(self) -> None:
        task = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        self._write_version("shard-ww1-v1", [task])
        selected = fleet_worker._select_accepted_fanin_shards(self.cache_dir)
        self.assertEqual(len(selected), 1)
        record = selected[0].acceptances[-1].acceptance_path
        self.assertIsNotNone(record)
        contents = record.read_bytes()
        with record.open("r+b") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())

        with self.assertRaises(FanInValidationError):
            fleet_worker._verify_selected_fanin_shard(selected[0])

    def test_selection_rejects_payload_mutation_between_acceptance_and_snapshot(self) -> None:
        from unittest.mock import patch

        task = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        target = self._write_version("shard-ww1-v1", [task])
        original_snapshot = fleet_worker._snapshot_fanin_payload_files
        mutated = False

        def snapshot_then_mutate(path: Path):
            nonlocal mutated
            result = original_snapshot(path)
            if path == target and not mutated:
                mutated = True
                self._mutate_cache_payload_in_place(path)
            return result

        with patch.object(fleet_worker, "_snapshot_fanin_payload_files", new=snapshot_then_mutate):
            with self.assertRaises(FanInValidationError):
                fleet_worker._select_accepted_fanin_shards(self.cache_dir)
        self.assertTrue(mutated)

    def test_identical_payload_replacement_after_acceptance_is_terminal_before_training(self) -> None:
        prior = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        accepted = self._write_version("shard-ww1-v1", [prior])
        self._replace_cache_payload_identically(accepted)
        _manifest(
            self.queue, "i1-s1.env", out=self.cache_dir / "slice-1", iteration=1,
            offset=1, count=1, seed=101,
        )
        calls: list[list[str]] = []

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=calls.append,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
        )

        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])
        self.assertTrue(self._claim("w1", "i1-s1.env").exists())
        self.assertFalse((self.queue / "done" / "i1-s1.env").exists())
        self.assertFalse((self.cache_dir / "shard-ww1-v2").exists())

    def test_recovery_ack_rejects_cache_root_replacement_after_lookup(self) -> None:
        from unittest.mock import patch
        import shutil

        task = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        self._write_version("shard-ww1-v1", [task])
        _manifest(
            self.queue, task.task_id, out=Path(task.out), iteration=1,
            offset=0, count=1, seed=100, policy=task.policy,
        )
        original_find = fleet_worker._find_committed_fanin_task_evidence
        replacement = self.root / "replacement-cache-root"
        saved = self.root / "saved-cache-root"
        replaced = False

        def find_then_replace(cache_dir: Path, candidate: FanInTask):
            nonlocal replaced
            result = original_find(cache_dir, candidate)
            if result is not None and not replaced:
                replaced = True
                shutil.copytree(self.cache_dir, replacement)
                self.cache_dir.rename(saved)
                replacement.rename(self.cache_dir)
            return result

        with patch.object(fleet_worker, "_find_committed_fanin_task_evidence", new=find_then_replace):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )
        self.assertTrue(replaced)
        self.assertEqual(rc, 2)
        self.assertTrue(self._claim("w1", task.task_id).exists())
        self.assertFalse((self.queue / "done" / task.task_id).exists())

    def test_recovery_ack_rejects_in_place_selected_payload_mutation_after_lookup(self) -> None:
        from unittest.mock import patch

        task = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        self._write_version("shard-ww1-v1", [task])
        _manifest(
            self.queue, task.task_id, out=Path(task.out), iteration=1,
            offset=0, count=1, seed=100, policy=task.policy,
        )
        original_find = fleet_worker._find_committed_fanin_task_evidence
        mutated = False

        def find_then_mutate(cache_dir: Path, candidate: FanInTask):
            nonlocal mutated
            result = original_find(cache_dir, candidate)
            if result is not None and not mutated:
                mutated = True
                self._mutate_cache_payload_in_place(result[1].path)
            return result

        with patch.object(fleet_worker, "_find_committed_fanin_task_evidence", new=find_then_mutate):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )
        self.assertTrue(mutated)
        self.assertEqual(rc, 2)
        self.assertTrue(self._claim("w1", task.task_id).exists())
        self.assertFalse((self.queue / "done" / task.task_id).exists())

    def test_route_witness_rejects_repeated_identical_record_replacement(self) -> None:
        from unittest.mock import patch

        _manifest(
            self.queue, "i1-s0.env", out=self.cache_dir / "slice-0",
            iteration=1, offset=0, count=1, seed=100,
        )
        original_resolve = fleet_worker._resolve_fanin_route
        replacements = 0
        calls: list[list[str]] = []

        def resolve_then_replace(queue: Path, task):
            nonlocal replacements
            resolved = original_resolve(queue, task)
            record = fleet_worker._fanin_route_record(queue, task.base)
            for index in range(3):
                prior = record.with_name(f"route-prior-{index}.json")
                record.rename(prior)
                record.write_bytes(prior.read_bytes())
                replacements += 1
            return resolved

        with patch.object(fleet_worker, "_resolve_fanin_route", new=resolve_then_replace):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=calls.append,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )
        self.assertEqual(replacements, 3)
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])
        self.assertTrue(self._claim("w1", "i1-s0.env").exists())
        self.assertFalse((self.queue / "done" / "i1-s0.env").exists())
        self.assertFalse((self.cache_dir / "shard-ww1-v1").exists())

    def test_concat_rejects_in_place_selected_payload_mutation(self) -> None:
        from unittest.mock import patch

        prior = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        self._write_version("shard-ww1-v1", [prior])
        _manifest(
            self.queue, "i1-s1.env", out=self.cache_dir / "slice-1", iteration=1,
            offset=1, count=1, seed=101,
        )
        from pokezero.dataset import concat_training_caches as real_concat

        mutated = False

        def mutate_then_concat(parts, staging):
            nonlocal mutated
            self._mutate_cache_payload_in_place(Path(parts[0]))
            mutated = True
            return real_concat(parts, staging)

        with patch("pokezero.dataset.concat_training_caches", new=mutate_then_concat):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )
        self.assertTrue(mutated)
        self.assertEqual(rc, 2)
        self.assertTrue(self._claim("w1", "i1-s1.env").exists())
        self.assertFalse((self.queue / "done" / "i1-s1.env").exists())
        self.assertFalse((self.cache_dir / "shard-ww1-v2").exists())

    def test_concat_rejects_current_shard_cache_root_replacement(self) -> None:
        from unittest.mock import patch
        import shutil

        prior = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        self._write_version("shard-ww1-v1", [prior])
        _manifest(
            self.queue, "i1-s1.env", out=self.cache_dir / "slice-1", iteration=1,
            offset=1, count=1, seed=101,
        )
        from pokezero.dataset import concat_training_caches as real_concat

        replacement = self.root / "replacement-cache-root"
        saved = self.root / "saved-cache-root"
        replaced = False

        def concat_then_replace(parts, staging):
            nonlocal replaced
            result = real_concat(parts, staging)
            if not replaced:
                replaced = True
                shutil.copytree(self.cache_dir, replacement)
                self.cache_dir.rename(saved)
                replacement.rename(self.cache_dir)
            return result

        with patch("pokezero.dataset.concat_training_caches", new=concat_then_replace):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )
        self.assertTrue(replaced)
        self.assertEqual(rc, 2)
        self.assertTrue(self._claim("w1", "i1-s1.env").exists())
        self.assertFalse((self.queue / "done" / "i1-s1.env").exists())
        self.assertFalse((self.cache_dir / "shard-ww1-v2").exists())

    def test_staging_mutation_after_fence_check_never_becomes_visible_or_accepted(self) -> None:
        self._manifests(1)
        calls: list[list[str]] = []
        mutated = False

        def mutate_staging_after_fence(boundary: str, _task) -> None:
            nonlocal mutated
            if boundary != "fanin-after-publication-fence-check":
                return
            staging = next(
                path for path in self.cache_dir.glob(".shard-ww1-v1.tmp.*") if path.is_dir()
            )
            self._mutate_cache_payload_in_place(staging)
            mutated = True

        def counted_collect(argv: list[str]) -> int:
            calls.append(argv)
            return self._real_collect(argv)

        rc = run_worker(
            self.queue, worker_id="w1", static_argv=[], collect_fn=counted_collect,
            max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            crash_inject=mutate_staging_after_fence,
        )

        self.assertTrue(mutated)
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self._claim("w1", "i1-s0.env").exists())
        self.assertFalse((self.queue / "done" / "i1-s0.env").exists())
        self.assertFalse((self.cache_dir / "shard-ww1-v1").exists())
        root = fleet_worker._fanin_task_fence_path(self.cache_dir, "i1-s0.env")
        current = fleet_worker._read_current_fanin_fence(root)
        self.assertIsNotNone(current)
        self.assertIsNone(current.acceptance)

    def test_finalization_mutation_removes_its_done_marker_and_invalidates_publication(self) -> None:
        from unittest.mock import patch

        self._manifests(1)
        original_write_done = fleet_worker._write_done_marker
        mutated = False

        def write_done_then_mutate(task, done):
            nonlocal mutated
            created = original_write_done(task, done)
            if created and not mutated:
                self._mutate_cache_payload_in_place(self.cache_dir / "shard-ww1-v1")
                mutated = True
            return created

        with patch.object(fleet_worker, "_write_done_marker", new=write_done_then_mutate):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )

        self.assertTrue(mutated)
        self.assertEqual(rc, 2)
        self.assertTrue(self._claim("w1", "i1-s0.env").exists())
        self.assertFalse((self.queue / "done" / "i1-s0.env").exists())
        with self.assertRaises(FanInValidationError):
            read_fanin_inventory(self.cache_dir, self._contract(tasks=1, games=1))

    def test_target_validation_rejects_entry_replacement_during_hashing(self) -> None:
        from unittest.mock import patch

        target = self._write_version(
            "shard-ww1-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 100)],
        )
        tasks = fleet_worker._read_fanin_manifest(target)
        acceptance = fleet_worker._read_fanin_publication(target)
        self.assertIsNotNone(acceptance)
        original_open = fleet_worker.os.open
        replaced = False

        def replace_metadata(name, flags, mode=0o777, *, dir_fd=None):
            nonlocal replaced
            descriptor = original_open(name, flags, mode, dir_fd=dir_fd)
            if name == "metadata.json" and dir_fd is not None and not replaced:
                replaced = True
                prior = target / ".metadata.prior"
                metadata = target / "metadata.json"
                metadata.rename(prior)
                metadata.write_bytes(prior.read_bytes())
            return descriptor

        with patch.object(fleet_worker.os, "open", new=replace_metadata):
            with self.assertRaises(FanInValidationError):
                fleet_worker._validate_fanin_target_acceptance(target, tasks, acceptance)
        self.assertTrue(replaced)

    def test_target_validation_rejects_root_replacement_during_validation(self) -> None:
        from unittest.mock import patch
        import shutil

        target = self._write_version(
            "shard-ww1-v1", [self._fanin_task("i1-s0.env", 1, 0, 1, 100)],
        )
        tasks = fleet_worker._read_fanin_manifest(target)
        acceptance = fleet_worker._read_fanin_publication(target)
        self.assertIsNotNone(acceptance)
        replacement = self.root / "replacement-shard"
        shutil.copytree(target, replacement)
        original = target.with_name("original-shard-w1-v1")
        original_record_count = fleet_worker._cache_record_count
        replaced = False

        def replace_root(path: Path) -> int:
            nonlocal replaced
            if path == target and not replaced:
                replaced = True
                target.rename(original)
                replacement.rename(target)
            return original_record_count(path)

        with patch.object(fleet_worker, "_cache_record_count", new=replace_root):
            with self.assertRaisesRegex(FanInValidationError, "root identity changed during validation"):
                fleet_worker._validate_fanin_target_acceptance(target, tasks, acceptance)
        self.assertTrue(replaced)

    def test_fanin_preflight_runs_before_claim_and_route_binding(self) -> None:
        from unittest.mock import patch

        self._manifests(1)
        events: list[str] = []
        original_preflight = fleet_worker._preflight_fanin_filesystems
        original_route = fleet_worker._resolve_fanin_route

        def preflight(queue: Path, cache_dir: Path) -> None:
            self.assertFalse(list((queue / "claimed").iterdir()))
            self.assertEqual(cache_dir, self.cache_dir)
            events.append("preflight")
            original_preflight(queue, cache_dir)

        def resolve(queue: Path, task) -> Path:
            self.assertIn("preflight", events)
            events.append("route")
            return original_route(queue, task)

        with (
            patch.object(fleet_worker, "_preflight_fanin_filesystems", new=preflight),
            patch.object(fleet_worker, "_resolve_fanin_route", new=resolve),
        ):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(events[0], "preflight")
        self.assertIn("route", events)

    def test_fanin_preflight_failure_preserves_pending_task_before_route_binding(self) -> None:
        from unittest.mock import patch

        self._manifests(1)
        with patch.object(
            fleet_worker,
            "_preflight_fanin_filesystems",
            side_effect=fleet_worker.FanInInventoryValidationError("hard-link probe failed"),
        ):
            rc = run_worker(
                self.queue, worker_id="w1", static_argv=[], collect_fn=self._real_collect,
                max_rss_mb=None, idle_exit_seconds=0.0, sleep_seconds=0.0, shard_fanin=True,
            )
        self.assertEqual(rc, 2)
        self.assertTrue((self.queue / "pending" / "i1-s0.env").exists())
        self.assertFalse(list((self.queue / "claimed").iterdir()))
        self.assertFalse((self.queue / fleet_worker._FANIN_ROUTE_DIRECTORY).exists())

    def test_fanin_preflight_rejects_nonatomic_hardlink_collision(self) -> None:
        from unittest.mock import patch

        original_link = fleet_worker.os.link

        def hide_collision(source, destination, *args, **kwargs) -> None:
            try:
                original_link(source, destination, *args, **kwargs)
            except FileExistsError:
                pass

        with patch.object(fleet_worker.os, "link", new=hide_collision):
            with self.assertRaisesRegex(FanInValidationError, "atomic hard-link CAS/collision"):
                fleet_worker._preflight_fanin_filesystems(self.queue, self.cache_dir)

    def test_ambiguous_acceptance_and_successor_outcome_fails_closed(self) -> None:
        import shutil

        task = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        self._write_version("shard-ww1-v1", [task])
        root = fleet_worker._fanin_task_fence_path(self.cache_dir, task.task_id)
        current = fleet_worker._read_current_fanin_fence(root)
        self.assertIsNotNone(current)
        self.assertIsNotNone(current.acceptance)
        outcome = fleet_worker._fanin_fence_successor_path(root, current)
        shutil.copy2(root / "owner.json", outcome / "owner.json")
        with self.assertRaisesRegex(
            FanInValidationError, "outcome is malformed or ambiguous",
        ):
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
        with self.assertRaisesRegex(FanInValidationError, "content or lineage disagrees"):
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
        next_tasks = [*tasks, self._fanin_task("i1-s1.env", 1, 1, 1, 101)]
        v4 = self._write_version("shard-ww1-v4", tasks)
        original_read = fleet_worker._read_fanin_manifest_snapshot
        raced = False

        def publish_v5_while_reading(path: Path):
            nonlocal raced
            if path == v4 and not raced:
                raced = True
                self._write_version("shard-ww1-v5", next_tasks)
                shutil.rmtree(v4)
            return original_read(path)

        with (
            patch.object(fleet_worker, "_read_fanin_manifest_snapshot", new=publish_v5_while_reading),
            patch.object(fleet_worker.time, "sleep") as sleep,
        ):
            inventory = read_fanin_inventory(self.cache_dir, self._contract(tasks=2, games=2))
        self.assertTrue(raced)
        sleep.assert_called_with(fleet_worker._SELECTED_FANIN_RETRY_SECONDS)
        self.assertEqual([shard.path.name for shard in inventory.shards], ["shard-ww1-v5"])

    def test_adopts_existing_version_and_sweeps_stale(self) -> None:
        (self.cache_dir / "shard-ww1-v1").mkdir()
        self._write_version(
            "shard-ww1-v2",
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


# A fleet worker driven from a child interpreter, so a whole fan-in fleet can be
# raced against one shared cache the way it is deployed. ``dev_offset`` and
# ``time_offset`` model a *second client* of that shared filesystem: the same
# files, observed through a different superblock and a separately revalidated
# attribute cache. Each identity helper is wrapped rather than reimplemented, so
# the shim stays neutral to how those helpers are defined.
_FLEET_CHILD = '''
import os, sys
sys.path.insert(0, {root_src!r})
sys.path.insert(0, {root!r})
from pathlib import Path
from pokezero import fleet_worker

dev_offset, time_offset = int(sys.argv[3]), int(sys.argv[4])


class ForeignStat:
    __slots__ = ("_observed",)

    def __init__(self, observed):
        self._observed = observed

    def __getattr__(self, name):
        if name == "st_dev":
            return self._observed.st_dev + dev_offset
        if name in ("st_mtime_ns", "st_ctime_ns"):
            return getattr(self._observed, name) + time_offset
        return getattr(self._observed, name)


if dev_offset or time_offset:
    for _name in (
        "_fanin_directory_identity", "_fanin_stat_identity",
        "_fanin_stat_snapshot", "_fanin_durable_stat_snapshot",
    ):
        _original = getattr(fleet_worker, _name, None)
        if _original is not None:
            def _wrapper(observed, _original=_original):
                return _original(ForeignStat(observed))
            setattr(fleet_worker, _name, _wrapper)


def collect(argv):
    import shutil
    from pokezero.dataset import TrajectoryDatasetConfig
    from tests.test_cache_concat import rollout_record, write_cache

    out = Path(argv[argv.index("--out") + 1])
    seed = int(argv[argv.index("--seed-start") + 1])
    staging = out.parent / (".collect-staging-%s.%d" % (out.name, os.getpid()))
    staging.mkdir(parents=True, exist_ok=True)
    cache = write_cache(
        staging, "c", [rollout_record(seed)],
        config=TrajectoryDatasetConfig(window_size=1),
    )
    shutil.move(str(cache), str(out))
    shutil.rmtree(staging, ignore_errors=True)
    return 0


raise SystemExit(fleet_worker.run_worker(
    Path(sys.argv[1]), worker_id=sys.argv[2], static_argv=[], collect_fn=collect,
    max_rss_mb=None, idle_exit_seconds=4.0, sleep_seconds=0.05, shard_fanin=True,
    log_dir=Path(sys.argv[5]),
))
'''


class FanInSharedFilesystemFleetTests(FanInFixture):
    """A whole fan-in fleet against one shared cache, as deployed.

    Everything below the concurrency is inherited from ``FanInTests``. What these
    add is the condition the single-process tests structurally cannot reach: more
    than one worker, and more than one *client* of the shared filesystem.
    """

    WORKERS = 4
    TASKS = 8

    def _run_fleet(self, *, dev_offset: int = 0, time_offset: int = 0) -> list[int]:
        """Race WORKERS child workers over TASKS tasks; return their exit codes.

        Odd-numbered workers observe the shared files as a foreign client would.
        """
        child = self.root / "fleet_child.py"
        child.write_text(
            _FLEET_CHILD.format(root=str(ROOT), root_src=str(ROOT / "src")),
            encoding="utf-8",
        )
        self._manifests(self.TASKS)
        processes = []
        for index in range(self.WORKERS):
            foreign = index % 2 == 1
            processes.append(subprocess.Popen(
                [
                    sys.executable, str(child), str(self.queue), f"w{index}",
                    str(dev_offset if foreign else 0),
                    str(time_offset if foreign else 0),
                    str(self.root / "worker-logs"),
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            ))
        codes = []
        for process in processes:
            try:
                _out, err = process.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                process.kill()
                _out, err = process.communicate()
                self.fail(f"fan-in worker hung: {err.decode('utf-8', 'replace')[-2000:]}")
            codes.append(process.returncode)
            if process.returncode not in (0, 2):
                self.fail(
                    "fan-in worker crashed instead of exiting cleanly:\n"
                    + err.decode("utf-8", "replace")[-2000:]
                )
        return codes

    def _terminal_log_lines(self) -> list[str]:
        lines = []
        for log in sorted((self.root / "worker-logs").glob("*.log")):
            lines += [
                line for line in log.read_text(encoding="utf-8").splitlines()
                if "TERMINAL" in line
            ]
        return lines

    def _assert_fleet_drained(self, codes: list[int]) -> None:
        done = sorted(path.name for path in (self.queue / "done").iterdir())
        failed = sorted(path.name for path in (self.queue / "failed").iterdir())
        self.assertEqual(failed, [])
        # rc=2 is the terminal "selected inventory is corrupt, preserve the
        # claim and stop" exit. Concurrency alone must never produce it, so the
        # log line that caused it is the whole diagnosis -- report it.
        self.assertEqual(
            [code for code in codes if code != 0], [],
            "worker(s) exited terminally:\n  " + "\n  ".join(self._terminal_log_lines()),
        )
        self.assertEqual(len(done), self.TASKS, f"only {len(done)}/{self.TASKS} tasks completed")
        inventory = read_fanin_inventory(
            self.cache_dir, self._contract(tasks=self.TASKS, games=self.TASKS),
        )
        self.assertEqual(
            sorted(task.task_id for task in inventory.tasks),
            sorted(f"i1-s{index}.env" for index in range(self.TASKS)),
        )

    def test_a_concurrent_fleet_drains_its_queue_on_one_shared_cache(self) -> None:
        self._assert_fleet_drained(self._run_fleet())

    def test_workers_on_different_mounts_of_the_shared_cache_all_collect(self) -> None:
        """The production failure: fan-in collected 0 of 800 slices across pods.

        Every worker validates *other* workers' published shards, against an
        identity witness those workers wrote into the shared cache. A witness
        carrying the writer's ``st_dev`` cannot be re-checked from any other
        mount, so the first published shard makes every other worker exit rc=2.
        """
        self._assert_fleet_drained(self._run_fleet(dev_offset=4096))

    def test_a_revalidated_attribute_cache_is_not_shard_corruption(self) -> None:
        """Same, for timestamps: st_mtime/st_ctime are not content signals.

        A remote client revalidates cached attributes against the server
        independently of the writer, and ``st_ctime`` moves on metadata changes
        that leave the bytes alone. Whether these are the certified bytes is
        settled by the SHA-256 recorded beside the identity, not by a timestamp.
        """
        self._assert_fleet_drained(self._run_fleet(time_offset=1_000_000))


class FanInDurableWitnessTests(FanInFixture):
    """What a published witness is allowed to contain."""

    def test_a_published_acceptance_records_no_client_local_identity(self) -> None:
        task = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
        target = self._write_version("shard-wtest-v1", [task])
        published = json.loads(
            (target / fleet_worker.FANIN_PUBLICATION_NAME).read_text(encoding="utf-8")
        )
        payload_files = published["payload_files"]
        self.assertTrue(payload_files)
        for entry in payload_files:
            observed = os.lstat(target.joinpath(*entry["relative"]))
            # The witness must be exactly the durable projection: reproducible
            # from any client. st_dev and both timestamps are not.
            self.assertEqual(
                tuple(entry["identity"]),
                fleet_worker._fanin_durable_stat_snapshot(observed),
            )
            self.assertNotIn(observed.st_dev, entry["identity"])
            self.assertNotIn(observed.st_mtime_ns, entry["identity"])
            self.assertNotIn(observed.st_ctime_ns, entry["identity"])
            self.assertTrue(fleet_worker._is_sha256(entry["sha256"]))

    def test_the_witness_still_rejects_a_tampered_or_replaced_payload(self) -> None:
        """Dropping client-local fields must not cost any integrity."""
        for mutate in (self._mutate_cache_payload_in_place, self._replace_cache_payload_identically):
            with self.subTest(mutate=mutate.__name__):
                self.setUp()  # a fresh cache per mutation
                task = self._fanin_task("i1-s0.env", 1, 0, 1, 100)
                target = self._write_version("shard-wtest-v1", [task])
                fleet_worker._select_accepted_fanin_shards(self.cache_dir)  # accepted as published
                mutate(target)
                with self.assertRaises(FanInValidationError):
                    fleet_worker._select_accepted_fanin_shards(self.cache_dir)


if __name__ == "__main__":
    unittest.main()
