"""The train agent's reliability contract, asserted case by case against the Job it replaces.

Every test here maps to a guarantee `backoffLimit: 2` / `activeDeadlineSeconds` / `restartPolicy:
Never` gave for free when training was a per-iteration Kubernetes Job. Losing any of them silently
is the failure this whole module exists to avoid, so they are tested as behaviour rather than
asserted in a comment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokezero.train_agent import (  # noqa: E402
    claim_next_request,
    _process_group_is_alive,
    record_running_trainer,
    release_stale_claims,
    run_training_attempt,
    serve_queue,
)


def write_request(queue: Path, iteration: int, script: str, *, deadline: int = 60,
                  max_attempts: int = 3) -> Path:
    pending = queue / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    path = pending / f"train-{iteration:04d}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pokezero.foundation-train-request.v1",
                "iteration": iteration,
                "script": script,
                "deadline_seconds": deadline,
                "max_attempts": max_attempts,
            }
        ),
        encoding="utf-8",
    )
    return path


class TrainAgentQueueTest(unittest.TestCase):
    def test_a_successful_request_publishes_a_done_result(self) -> None:
        with TemporaryDirectory() as root:
            queue = Path(root) / "train-queue"
            marker = Path(root) / "ran"
            write_request(queue, 7, f"touch {marker}")
            self.assertEqual(serve_queue(queue, once=True), 1)
            self.assertTrue(marker.exists(), "the training script did not run")
            result = json.loads((queue / "done" / "train-0007.json").read_text())
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["iteration"], 7)
            self.assertEqual([a["exit_code"] for a in result["attempts"]], [0])
            self.assertFalse(
                list((queue / "claimed").glob("*.claim.json")),
                "the claim outlived the result",
            )

    def test_a_failing_request_retries_to_the_attempt_bound_then_fails(self) -> None:
        """The `backoffLimit: 2` equivalent: three attempts total, then a durable failure.

        Asserted as a COUNT, not just a final status. An agent that gave up after one attempt would
        still produce a failed result, and the run would still die -- but a transient preempt or
        node blip, the exact case the Job's retries existed for, would now end a two-week run.
        """
        with TemporaryDirectory() as root:
            queue = Path(root) / "train-queue"
            counter = Path(root) / "count"
            write_request(queue, 3, f"echo x >> {counter}; exit 9", max_attempts=3)
            serve_queue(queue, once=True)
            self.assertEqual(counter.read_text().count("x"), 3, "wrong number of attempts")
            result = json.loads((queue / "failed" / "train-0003.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual([a["exit_code"] for a in result["attempts"]], [9, 9, 9])

    def test_a_transient_failure_recovers_within_the_attempt_bound(self) -> None:
        """The case the retries exist FOR: fail once, then succeed, and the iteration is fine."""
        with TemporaryDirectory() as root:
            queue = Path(root) / "train-queue"
            counter = Path(root) / "count"
            # Fails on the first attempt and succeeds on the second: each attempt appends a
            # 2-byte line, so the >=4 test is false once and true thereafter.
            script = f"echo x >> {counter}; [ $(wc -c < {counter}) -ge 4 ]"
            write_request(queue, 4, script)
            serve_queue(queue, once=True)
            result = json.loads((queue / "done" / "train-0004.json").read_text())
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual([a["exit_code"] for a in result["attempts"]], [1, 0])

    def test_the_deadline_kills_the_whole_process_tree(self) -> None:
        """`activeDeadlineSeconds`, and the part that is easy to get wrong.

        Training spawns helpers. Signalling only the direct child leaves those orphaned, holding
        GPU memory that the NEXT attempt then fails to allocate -- turning one timeout into a
        run-ending cascade. So the grandchild must be dead too, which is why the child runs in its
        own process group.
        """
        with TemporaryDirectory() as root:
            pid_file = Path(root) / "grandchild.pid"
            # The grandchild TRAPS SIGTERM, so only the SIGKILL escalation can end it. With a
            # plain `sleep` this test passed even with the escalation deleted entirely -- the
            # grandchild died politely on TERM and the assertion never exercised the code it
            # claimed to. A trainer that ignores TERM while holding GPU memory is the real case.
            script = (
                f"( trap '' TERM; sleep 120 & echo $! > {pid_file}; wait )"
            )
            started = time.monotonic()
            code = run_training_attempt(script, deadline_seconds=2)
            elapsed = time.monotonic() - started
            self.assertEqual(code, 124, "a timeout must be distinguishable from a train exit code")
            self.assertLess(elapsed, 30, "the deadline did not fire promptly")
            grandchild = int(pid_file.read_text().strip())
            deadline = time.monotonic() + 10
            alive = True
            while time.monotonic() < deadline:
                try:
                    os.kill(grandchild, 0)
                except OSError:
                    alive = False
                    break
                time.sleep(0.2)
            self.assertFalse(alive, f"grandchild {grandchild} survived the deadline kill")

    def test_a_timed_out_request_is_retried_and_then_recorded_failed(self) -> None:
        with TemporaryDirectory() as root:
            queue = Path(root) / "train-queue"
            write_request(queue, 5, "sleep 30", deadline=1, max_attempts=2)
            serve_queue(queue, once=True)
            result = json.loads((queue / "failed" / "train-0005.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual([a["exit_code"] for a in result["attempts"]], [124, 124])


class TrainAgentRestartTest(unittest.TestCase):
    """The kill matrix: what must survive an agent restart or a pod reschedule."""

    def test_an_agent_restart_releases_its_own_abandoned_claim(self) -> None:
        """Agent killed mid-train, supervisor restarts it: the iteration must be runnable again.

        Without this the request sits in claimed/ forever -- the controller waits out its deadline
        and the run dies of a bookkeeping artefact rather than a real failure.
        """
        with TemporaryDirectory() as root:
            queue = Path(root) / "train-queue"
            write_request(queue, 11, "true")
            claimed = claim_next_request(queue, "boot-A")
            self.assertIsNotNone(claimed)
            self.assertFalse(list((queue / "pending").glob("*.json")))

            released = release_stale_claims(queue, "boot-A")
            self.assertEqual(released, ["train-0011.json"])
            self.assertTrue((queue / "pending" / "train-0011.json").exists())
            # And it is genuinely runnable again, not merely moved.
            self.assertEqual(serve_queue(queue, owner="boot-A", once=True), 1)

    def test_an_orphaned_trainer_is_killed_before_its_claim_is_released(self) -> None:
        """The reproduced double-trainer, closed at the level that actually holds it: the GROUP.

        The rendered train script defines a shell function before invoking python, so bash cannot
        exec-replace itself -- the real tree is agent -> `bash -c` -> python trainer. An earlier fix
        tracked the direct child, i.e. the WRAPPER, and review reproduced the hole: kill the wrapper
        and the trainer is reparented and keeps writing the checkpoint, while a liveness check on
        the wrapper pid reports dead, so the claim is released and a second trainer starts.

        THE SCRIPT SHAPE IS THE TEST. A single simple command is exec-replaced by bash, making
        wrapper and trainer the same process and hiding the bug -- which is exactly why the previous
        tests passed against broken code. This one contains a function definition, so bash forks.
        """
        with TemporaryDirectory() as root:
            queue = Path(root) / "train-queue"
            started = Path(root) / "trainer.pid"
            # A function definition forces bash to fork rather than exec-replace.
            script = f"run_it() {{ sleep 120; }}; run_it & echo $! > {started}; wait"
            write_request(queue, 41, script)
            self.assertIsNotNone(claim_next_request(queue, "boot-A"))

            proc = subprocess.Popen(
                ["bash", "-c", script], preexec_fn=os.setsid,
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not started.exists():
                time.sleep(0.1)
            self.assertTrue(started.exists(), "the stand-in trainer never started")
            grandchild = int(started.read_text().strip())
            pgid = os.getpgid(proc.pid)
            self.assertNotEqual(
                grandchild, proc.pid,
                "the script did not fork, so this test cannot see the wrapper/trainer distinction",
            )
            record_running_trainer(queue, "train-0041", pgid)

            # Model what actually happens when the agent dies: PDEATHSIG kills the WRAPPER and it is
            # reaped by init, leaving the trainer orphaned but still running in the same group.
            # This is the state release_stale_claims has to handle, and the state in which a
            # wrapper-pid liveness check reports "dead" while a writer is very much alive.
            proc.kill()
            proc.wait(timeout=5)
            self.assertTrue(
                _process_group_is_alive(pgid),
                "the orphaned trainer is not visible via its group, so this test proves nothing",
            )

            # The claim is from a previous life of this pod, so the trainer under it is unowned:
            # release must KILL it, not leave it racing whoever picks the claim up next.
            released = release_stale_claims(queue, "boot-A")
            self.assertEqual(released, ["train-0041.json"])
            for _ in range(50):
                try:
                    os.kill(grandchild, 0)
                except OSError:
                    break
                time.sleep(0.2)
            else:
                os.killpg(pgid, 9)
                self.fail(f"grandchild trainer {grandchild} survived release; two writers possible")
            proc.wait(timeout=5)

    def test_liveness_is_measured_on_the_group_not_the_wrapper(self) -> None:
        """The predicate itself, isolated. Pointing it at the wrapper pid is what made the previous
        fix ineffective, and the difference is only visible when the script forks."""
        with TemporaryDirectory() as root:
            started = Path(root) / "pid"
            script = f"run_it() {{ sleep 60; }}; run_it & echo $! > {started}; wait"
            proc = subprocess.Popen(["bash", "-c", script], preexec_fn=os.setsid)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not started.exists():
                time.sleep(0.1)
            pgid = os.getpgid(proc.pid)
            grandchild = int(started.read_text().strip())
            try:
                os.kill(proc.pid, 9)   # exactly what PDEATHSIG delivers to the wrapper
                proc.wait(timeout=5)
                self.assertTrue(
                    _process_group_is_alive(pgid),
                    "the group reads dead while the trainer is still running, so a claim would be "
                    "released with a live writer under it",
                )
            finally:
                try:
                    os.killpg(pgid, 9)
                except OSError:
                    pass
            del grandchild

    def test_the_forked_trainer_is_death_linked_to_the_agent(self) -> None:
        """PR_SET_PDEATHSIG, which closes the double-trainer at its source rather than papering
        over it: an orphaned trainer must not outlive the agent that owns its claim.

        Skipped off Linux, where prctl does not exist and the claim liveness record is the only
        layer -- which is precisely why both layers exist.
        """
        if not sys.platform.startswith("linux"):
            self.skipTest("PR_SET_PDEATHSIG is Linux-only")
        with TemporaryDirectory() as root:
            pid_file = Path(root) / "child.pid"
            stand_in = Path(root) / "agent_stand_in.py"
            src = str(Path(__file__).resolve().parents[1] / "src")
            stand_in.write_text(
                "import sys, time, pathlib, threading\n"
                "sys.path.insert(0, %r)\n"
                "from pokezero.train_agent import run_training_attempt\n"
                "def note(pid):\n"
                "    pathlib.Path(%r).write_text(str(pid))\n"
                "threading.Thread(\n"
                "    target=run_training_attempt, args=('sleep 60', 60),\n"
                "    kwargs=dict(on_start=note), daemon=True).start()\n"
                "time.sleep(30)\n" % (src, str(pid_file)),
                encoding="utf-8",
            )
            agent = subprocess.Popen([sys.executable, str(stand_in)])
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not pid_file.exists():
                time.sleep(0.1)
            self.assertTrue(pid_file.exists(), "the trainer never started")
            child = int(pid_file.read_text().strip())
            agent.kill()
            agent.wait()
            gone_by = time.monotonic() + 15
            alive = True
            while time.monotonic() < gone_by:
                try:
                    os.kill(child, 0)
                except OSError:
                    alive = False
                    break
                time.sleep(0.2)
            if alive:
                try:
                    os.kill(child, 9)
                except OSError:
                    pass
            self.assertFalse(alive, f"trainer {child} outlived the agent that owned its claim")

    def test_a_claim_held_by_another_pod_is_never_reaped_here(self) -> None:
        """The dangerous direction, and the reason release is scoped to our own boot identity.

        Reaping a live sibling's claim would let two agents train the same iteration into the same
        output directory concurrently -- two writers, one checkpoint path. Only the controller can
        know whether the other pod is gone, so the agent must refuse.
        """
        with TemporaryDirectory() as root:
            queue = Path(root) / "train-queue"
            write_request(queue, 12, "true")
            claim_next_request(queue, "boot-OTHER")
            released = release_stale_claims(queue, "boot-MINE")
            self.assertEqual(released, [], "reaped a claim belonging to another pod")
            self.assertFalse((queue / "pending" / "train-0012.json").exists())

    def test_only_one_of_two_racing_agents_claims_a_request(self) -> None:
        """Atomic rename is the entire concurrency story; prove it rather than assume it."""
        with TemporaryDirectory() as root:
            queue = Path(root) / "train-queue"
            write_request(queue, 13, "true")
            first = claim_next_request(queue, "boot-A")
            second = claim_next_request(queue, "boot-B")
            self.assertIsNotNone(first)
            self.assertIsNone(second, "two agents claimed the same iteration")

    def test_a_malformed_request_is_parked_not_looped_on(self) -> None:
        """A request the agent cannot parse must not spin forever; the controller's bound decides."""
        with TemporaryDirectory() as root:
            queue = Path(root) / "train-queue"
            pending = queue / "pending"
            pending.mkdir(parents=True)
            (pending / "train-0014.json").write_text("{not json", encoding="utf-8")
            self.assertIsNone(claim_next_request(queue, "boot-A"))
            self.assertTrue((queue / "failed" / "train-0014.json.malformed").exists())

    def test_results_are_published_indivisibly(self) -> None:
        """The controller polls for these files; a truncated one would be read as authoritative.

        Asserted through os.replace, not by checking for leftover dotfiles. The dotfile check alone
        is satisfied by a plain open()+write, so it passed with the atomic publish removed entirely.
        What must be true is that the target only ever appears complete, i.e. it arrives by rename
        from a fully written temp.
        """
        import pokezero.train_agent as agent_module

        renames: list[tuple[str, str]] = []
        real_replace = os.replace

        def recording_replace(src, dst):
            renames.append((str(src), str(dst)))
            return real_replace(src, dst)

        with TemporaryDirectory() as root:
            queue = Path(root) / "train-queue"
            write_request(queue, 15, "true")
            agent_module.os.replace = recording_replace
            try:
                serve_queue(queue, once=True)
            finally:
                agent_module.os.replace = real_replace
            result = queue / "done" / "train-0015.json"
            payload = json.loads(result.read_text())
            self.assertEqual(payload["schema_version"], "pokezero.foundation-train-result.v1")
            self.assertTrue(
                any(dst == str(result) for _, dst in renames),
                "the result was not published by rename, so a reader can observe it half-written",
            )
            for stray in (queue / "done").glob(".*"):
                self.fail(f"a temporary result file was left behind: {stray}")


class TrainAgentEntrypointTest(unittest.TestCase):
    def test_the_module_runs_as_a_command(self) -> None:
        """The pod invokes this as `python -m pokezero.train_agent`; that path must work."""
        with TemporaryDirectory() as root:
            queue = Path(root) / "train-queue"
            marker = Path(root) / "ran"
            write_request(queue, 21, f"touch {marker}")
            completed = subprocess.run(
                [sys.executable, "-m", "pokezero.train_agent", "--queue", str(queue), "--once"],
                capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(marker.exists())


if __name__ == "__main__":
    unittest.main()
