"""Focused contracts for the source-isolated MCTS policy transport."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

from pokezero.mcts_eval.head_to_head import MctsPolicySpec
from pokezero.mcts_eval.isolated_policy import (
    IsolatedMctsPolicy,
    IsolatedPolicyError,
    IsolatedPolicyLaunch,
    IsolatedPolicyStats,
    read_frame,
    snapshot_annotation_source,
    source_neutral_context_payload,
    write_frame,
)
from pokezero.policy import PolicyContext


REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec() -> MctsPolicySpec:
    return MctsPolicySpec(
        config_id="fixed",
        policy_id="fixed-policy",
        source_commit="a" * 40,
        source_tree_sha256="b" * 64,
        engine_fingerprint="c" * 16,
        checkpoint_sha256="d" * 64,
        showdown_source_sha256="e" * 64,
        config={"leaf_eval": "model", "strict_fallbacks": True},
    )


def _context(*, padding_bytes: int = 0) -> PolicyContext:
    p1_observation = SimpleNamespace(
        legal_action_mask=(True, False), private="p1", padding=b"x" * padding_bytes
    )
    p2_observation = SimpleNamespace(legal_action_mask=(False, True), private="p2")
    trajectory = SimpleNamespace(
        battle_id="battle",
        format_id="gen3randombattle",
        seed=17,
        terminal=None,
        metadata={
            "private": "must not reach worker",
            "public_resolved_action_rounds": [{"turn_index": 0, "actions": {}}],
        },
        steps=(
            SimpleNamespace(player_id="p1", turn_index=0, action_index=0, observation=p1_observation),
            SimpleNamespace(player_id="p2", turn_index=0, action_index=1, observation=p2_observation),
        ),
    )
    return PolicyContext(
        player_id="p1",
        decision_round_index=1,
        battle_id="battle",
        format_id="gen3randombattle",
        seed=17,
        observation=p1_observation,
        requested_players=("p1", "p2"),
        trajectory=trajectory,
        requested_legal_action_masks={"p1": (True, False), "p2": (False, True)},
        requested_observations={"p1": p1_observation, "p2": p2_observation},
        public_materialization_state=SimpleNamespace(replay=SimpleNamespace(public_events=())),
    )


class _Annotations:
    def active(self) -> bool:
        return True

    def overlay_for(self, player_id: str):
        self.player_id = player_id
        return {2: ("residual", True, False, 0.25)}


def _fake_worker_script(root: Path) -> Path:
    script = root / "fake_worker.py"
    script.write_text(
        f"""import pickle
import struct
import sys
sys.path.insert(0, {str(REPO_ROOT / "src")!r})

def exact(stream, count):
    chunks = []
    while count:
        chunk = stream.read(count)
        if not chunk:
            raise EOFError()
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)

def read(stream):
    size = struct.unpack(">Q", exact(stream, 8))[0]
    return pickle.loads(exact(stream, size))

def write(stream, value):
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">Q", len(data)))
    stream.write(data)
    stream.flush()

stdin = sys.stdin.buffer
stdout = sys.stdout.buffer
start = read(stdin)
policy = start["policy"]
mode = start["worker_config"].get("mode", "ok")
if mode == "bad-receipt":
    write(stdout, {{"type": "hello", "receipt": {{"policy": {{}}}}}})
    raise SystemExit(0)
if mode == "silent-hello":
    import time
    time.sleep(5)
    raise SystemExit(0)
if mode == "partial-hello":
    stdout.write(b"\\x00\\x00\\x00\\x00")
    stdout.flush()
    import time
    time.sleep(5)
    raise SystemExit(0)
write(stdout, {{"type": "hello", "receipt": {{"policy": policy, "worker_pid": 999}}}})
if mode == "stop-after-hello":
    import time
    time.sleep(5)
    raise SystemExit(0)
if mode == "close-stdin-after-hello":
    # Close the inherited descriptor itself: closing BufferedReader alone can
    # retain a read buffer on some Python/platform combinations.
    import os
    os.close(stdin.fileno())
    import time
    time.sleep(5)
    raise SystemExit(0)
if mode == "ignore-term-after-hello":
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    import time
    time.sleep(5)
    raise SystemExit(0)
while True:
    message = read(stdin)
    if message["type"] == "close":
        write(stdout, {{"type": "close"}})
        raise SystemExit(0)
    if message["type"] == "reset":
        write(stdout, {{"type": "reset"}})
        continue
    if message["type"] != "decide":
        write(stdout, {{"type": "error", "message": "unexpected request"}})
        continue
    context = message["context"]
    if not isinstance(context, dict):
        write(stdout, {{"type": "error", "message": "context is not source-neutral"}})
        continue
    player_id = context["player_id"]
    if set(context["requested_observations"]) != {{player_id}}:
        write(stdout, {{"type": "error", "message": "private request leaked"}})
        continue
    if set(context["requested_legal_action_masks"]) != {{player_id}}:
        write(stdout, {{"type": "error", "message": "private mask leaked"}})
        continue
    trajectory = context["trajectory"]
    if not isinstance(trajectory, dict):
        write(stdout, {{"type": "error", "message": "trajectory is not source-neutral"}})
        continue
    if any(step["player_id"] != player_id and step["observation"] is not None for step in trajectory["steps"]):
        write(stdout, {{"type": "error", "message": "historic private observation leaked"}})
        continue
    annotation = message["annotation"]
    if annotation != {{"active": True, "overlay": {{2: ("residual", True, False, 0.25)}}}}:
        write(stdout, {{"type": "error", "message": "annotation snapshot changed"}})
        continue
    action = 1 if mode == "illegal" else 0
    stats = {{"decisions": 1, "searched_decisions": 1, "fallback_decisions": 0, "model_evals": 4, "total_iterations": 8, "worlds_constructed": 1, "worlds_searched": 1, "prior_fallbacks": 0, "root_prior_fallbacks": 0, "branch_prior_fallbacks": 0, "decision_wall_seconds": 0.25}}
    if mode == "legacy-stats":
        del stats["root_prior_fallbacks"]
        del stats["branch_prior_fallbacks"]
    write(stdout, {{
        "type": "decision",
        "decision": {{"action_index": action, "policy_id": policy["policy_id"], "metadata": {{"worker": "fake"}}}},
        "stats": stats,
    }})
""",
        encoding="utf-8",
    )
    return script


class FrameTest(unittest.TestCase):
    def test_frames_round_trip_exact_mapping(self) -> None:
        stream = BytesIO()
        write_frame(stream, {"kind": "frame", "count": 3})
        stream.seek(0)
        self.assertEqual(read_frame(stream), {"kind": "frame", "count": 3})


class AnnotationSnapshotTest(unittest.TestCase):
    def test_active_overlay_is_copied_at_the_host_boundary(self) -> None:
        annotations = _Annotations()
        snapshot = snapshot_annotation_source(annotations, player_id="p1")

        self.assertEqual(annotations.player_id, "p1")
        self.assertEqual(
            snapshot, {"active": True, "overlay": {2: ("residual", True, False, 0.25)}}
        )

    def test_annotation_snapshot_refuses_text_in_place_of_a_feature_tuple(self) -> None:
        class BadAnnotations:
            def active(self) -> bool:
                return True

            def overlay_for(self, _player_id: str):
                return {2: "not-a-feature-tuple"}

        with self.assertRaisesRegex(IsolatedPolicyError, "invalid entry"):
            snapshot_annotation_source(BadAnnotations(), player_id="p1")

    def test_context_payload_uses_only_a_mapping_for_host_adapter_trajectory(self) -> None:
        payload = source_neutral_context_payload(_context())

        self.assertIsInstance(payload, dict)
        self.assertIsInstance(payload["trajectory"], dict)
        self.assertIsInstance(payload["trajectory"]["steps"][0], dict)
        self.assertEqual(set(payload["requested_observations"]), {"p1"})
        self.assertIsNone(payload["trajectory"]["steps"][1]["observation"])


class IsolatedPolicyTest(unittest.TestCase):
    def _policy(
        self, directory: Path, *, mode: str = "ok", response_timeout_seconds: float = 5
    ) -> IsolatedMctsPolicy:
        return IsolatedMctsPolicy(
            IsolatedPolicyLaunch(
                policy=_spec(),
                command=(sys.executable, str(_fake_worker_script(directory))),
                worker_config={"mode": mode},
                response_timeout_seconds=response_timeout_seconds,
                stderr_path=directory / "worker.stderr",
            ),
            annotation_source=_Annotations(),
        )

    def test_worker_receives_only_public_context_and_returns_monotonic_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self._policy(Path(directory))
            try:
                decision = policy.select_action_with_context(_context(), rng=__import__("random").Random(7))
                self.assertEqual(decision.action_index, 0)
                self.assertEqual(decision.policy_id, "fixed-policy")
                self.assertEqual(decision.metadata, {"worker": "fake"})
                self.assertEqual(policy.stats.model_evals, 4)
                self.assertEqual(policy.stats.decision_wall_seconds, 0.25)
                self.assertEqual(policy.worker_receipt["policy"], _spec().to_payload())
                policy.reset()
            finally:
                policy.close()

    def test_illegal_child_action_refuses_before_a_battle_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self._policy(Path(directory), mode="illegal")
            try:
                with self.assertRaisesRegex(IsolatedPolicyError, "not legal"):
                    policy.select_action_with_context(_context(), rng=__import__("random").Random(7))
            finally:
                policy.close()

    def test_legacy_source_stats_cannot_be_treated_as_scoped_zero(self) -> None:
        """A v2 pilot must fail closed before scoring an uninstrumented source."""
        with tempfile.TemporaryDirectory() as directory:
            policy = self._policy(Path(directory), mode="legacy-stats")
            try:
                with self.assertRaisesRegex(IsolatedPolicyError, "root_prior_fallbacks"):
                    policy.select_action_with_context(_context(), rng=__import__("random").Random(7))
            finally:
                policy.close()

    def test_receipt_mismatch_refuses_the_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self._policy(Path(directory), mode="bad-receipt")
            try:
                with self.assertRaisesRegex(IsolatedPolicyError, "provenance receipt"):
                    policy.select_action_with_context(_context(), rng=__import__("random").Random(7))
            finally:
                policy.close()

    def test_partial_worker_frame_cannot_bypass_the_response_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self._policy(
                Path(directory), mode="partial-hello", response_timeout_seconds=0.1
            )
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(
                    IsolatedPolicyError,
                    "partial isolated policy worker response header: received 4 of 8 bytes",
                ):
                    policy.select_action_with_context(_context(), rng=__import__("random").Random(7))
            finally:
                policy.close()
            self.assertLess(time.monotonic() - started, 1.0)

    def test_silent_worker_timeout_is_not_misreported_as_a_partial_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self._policy(
                Path(directory), mode="silent-hello", response_timeout_seconds=0.1
            )
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(
                    IsolatedPolicyError,
                    "timed out before receiving isolated policy worker response header",
                ):
                    policy.select_action_with_context(_context(), rng=__import__("random").Random(7))
            finally:
                policy.close()
            self.assertLess(time.monotonic() - started, 1.0)

    def test_large_request_to_a_nonreading_worker_cannot_block_past_the_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self._policy(
                Path(directory), mode="stop-after-hello", response_timeout_seconds=0.1
            )
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(
                    IsolatedPolicyError,
                    "timed out sending partial isolated policy worker request payload",
                ):
                    policy.select_action_with_context(
                        _context(padding_bytes=1024 * 1024), rng=__import__("random").Random(7)
                    )
            finally:
                policy.close()
            self.assertLess(time.monotonic() - started, 1.0)

    def test_live_worker_with_closed_input_is_terminated_before_close_can_wait(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self._policy(
                Path(directory), mode="close-stdin-after-hello", response_timeout_seconds=0.1
            )
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(IsolatedPolicyError, "protocol failed"):
                    policy.select_action_with_context(_context(), rng=__import__("random").Random(7))
            finally:
                policy.close()
            self.assertLess(time.monotonic() - started, 1.0)

    def test_sigterm_ignoring_worker_cannot_start_a_second_deadline_during_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self._policy(
                Path(directory), mode="ignore-term-after-hello", response_timeout_seconds=0.1
            )
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(
                    IsolatedPolicyError,
                    "timed out before receiving isolated policy worker response header",
                ):
                    policy.select_action_with_context(_context(), rng=__import__("random").Random(7))
            finally:
                policy.close()
            self.assertLess(time.monotonic() - started, 1.0)


class StatsTest(unittest.TestCase):
    @staticmethod
    def _payload() -> dict[str, object]:
        return {
            "decisions": 3,
            "searched_decisions": 3,
            "fallback_decisions": 0,
            "model_evals": 4,
            "total_iterations": 8,
            "worlds_constructed": 1,
            "worlds_searched": 1,
            "prior_fallbacks": 0,
            "root_prior_fallbacks": 0,
            "branch_prior_fallbacks": 0,
            "decision_wall_seconds": 0.3,
        }

    def test_telemetry_regression_is_refused(self) -> None:
        stats = IsolatedPolicyStats()
        payload = self._payload()
        stats.update(payload)
        payload["model_evals"] = 3
        with self.assertRaisesRegex(IsolatedPolicyError, "regressed"):
            stats.update(payload)

    def test_scope_counters_are_required_from_an_isolated_source(self) -> None:
        stats = IsolatedPolicyStats()
        payload = self._payload()
        del payload["root_prior_fallbacks"]
        with self.assertRaisesRegex(IsolatedPolicyError, "root_prior_fallbacks"):
            stats.update(payload)

    def test_scope_counter_aggregate_mismatch_is_refused(self) -> None:
        stats = IsolatedPolicyStats()
        payload = self._payload()
        payload["prior_fallbacks"] = 1
        with self.assertRaisesRegex(IsolatedPolicyError, "aggregate must equal root plus branch"):
            stats.update(payload)

    def test_non_finite_wall_time_is_refused(self) -> None:
        stats = IsolatedPolicyStats()
        payload = self._payload()
        payload.update(
            {
                "decisions": 0,
                "searched_decisions": 0,
                "model_evals": 0,
                "total_iterations": 0,
                "worlds_constructed": 0,
                "worlds_searched": 0,
                "decision_wall_seconds": float("nan"),
            }
        )
        with self.assertRaisesRegex(IsolatedPolicyError, "invalid decision wall time"):
            stats.update(payload)


if __name__ == "__main__":
    unittest.main()
