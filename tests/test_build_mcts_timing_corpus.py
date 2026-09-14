"""Regression tests for public-prefix admission in the timing corpus builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

from pokezero.public_replay_materializer import PublicReplayError


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_mcts_timing_corpus_under_test", ROOT / "scripts" / "build_mcts_timing_corpus.py"
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class _Replay:
    def __init__(self, lines: tuple[str, ...]) -> None:
        self.public_lines = lines


class _State:
    def __init__(self, lines: tuple[str, ...]) -> None:
        self.replay = _Replay(lines)


class _Env:
    def __init__(self, lines: tuple[str, ...]) -> None:
        self._lines = lines

    def public_materialization_state(self, _player: str) -> _State:
        return _State(self._lines)


class TimingCorpusReplayAdmissionTest(unittest.TestCase):
    def test_rejects_public_protocol_divergence_after_replay(self) -> None:
        with mock.patch.object(builder, "replay_public_action_rounds"):
            reason = builder._prefix_replayability_reason(
                _Env(("|start", "|turn|2")),
                seed=7,
                public_rounds=(),
                expected_public_lines={"p1": ("|start", "|turn|1")},
            )
        self.assertEqual(reason, "public_replay:public_prefix_mismatch")

    def test_preserves_named_request_shape_rejection(self) -> None:
        with mock.patch.object(
            builder,
            "replay_public_action_rounds",
            side_effect=PublicReplayError("sampled_world_request_shape_mismatch"),
        ):
            reason = builder._prefix_replayability_reason(
                _Env(("|start",)),
                seed=7,
                public_rounds=(),
                expected_public_lines={"p1": ("|start",)},
            )
        self.assertEqual(reason, "public_replay:sampled_world_request_shape_mismatch")


if __name__ == "__main__":
    unittest.main()
