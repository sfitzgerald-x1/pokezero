from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pokezero.env import StepResult, TerminalState
from pokezero.policy import PolicyDecision
from pokezero.rollout import RolloutConfig
from pokezero.selfplay_protocol_capture import (
    SELFPLAY_PROTOCOL_CAPTURE_SCHEMA_VERSION,
    _capture_provenance,
    _validate_existing_capture,
    capture_selfplay_protocol_signatures,
)


class _Policy:
    def __init__(self, policy_id: str) -> None:
        self.policy_id = policy_id

    def select_action(self, observation: object, *, rng: object) -> PolicyDecision:
        return PolicyDecision(action_index=0, policy_id=self.policy_id)


class _Env:
    def __init__(self, *, capped: bool = False) -> None:
        self._terminal: TerminalState | None = None
        self._lines: tuple[str, ...] = ()
        self._capped = capped

    @property
    def protocol_lines(self) -> tuple[str, ...]:
        return self._lines

    def reset(self, *, seed: int, format_id: str) -> None:
        self._terminal = None
        self._lines = ("|player|p1|private-name", f"|move|p1a: Alpha|Protect|[from] seed: {seed}")

    def observe(self, player: str) -> object:
        return SimpleNamespace(legal_action_mask=(True,) * 9)

    def requested_players(self) -> tuple[str, ...]:
        return () if self._terminal is not None else ("p1", "p2")

    def step(self, actions: dict[str, int]) -> StepResult:
        self._lines += ("|-activate|p1a: Alpha|move: Protect", "|win|PokeZero p1")
        self._terminal = TerminalState(winner="p1", turn_count=1, capped=self._capped)
        return StepResult(observations={}, rewards={"p1": 1.0, "p2": -1.0}, terminal=self._terminal)

    def terminal(self) -> TerminalState | None:
        return self._terminal

    def close(self) -> None:
        pass


class SelfplayProtocolCaptureTests(unittest.TestCase):
    def test_capture_keeps_only_canonical_count_only_protocol_evidence(self) -> None:
        result = capture_selfplay_protocol_signatures(
            games=2,
            env_factory=_Env,
            policies={"p1": _Policy("learned-a"), "p2": _Policy("learned-b")},
            rollout_config=RolloutConfig(max_decision_rounds=5),
            seed_start=40,
        )

        self.assertEqual(result.completed_games, 2)
        self.assertEqual(result.decision_rounds, 2)
        self.assertEqual(result.capped_games, 0)
        self.assertEqual(result.protocol_signatures["move:protect"], 2)
        self.assertEqual(result.protocol_signatures["-activate:protect"], 2)
        self.assertNotIn("request", result.protocol_signatures)
        self.assertEqual(len(result.protocol_signature_game_ids), 2)
        self.assertEqual(len(set(result.protocol_signature_game_ids)), 2)

    def test_existing_capture_requires_identical_source_policy_and_seed_scope(self) -> None:
        provenance = _capture_provenance(
            source_hash="source-hash",
            command_arguments=("--example",),
            seed_start=10,
            games=1,
            capture_label="production-line",
            max_decision_rounds=250,
            p1_policy_identity={"policy_id": "learned-a", "policy_spec_sha256": "p1", "weights_sha256": "weights-a"},
            p2_policy_identity={"policy_id": "learned-b", "policy_spec_sha256": "p2", "weights_sha256": "weights-b"},
        )
        payload = {
            "schema_version": SELFPLAY_PROTOCOL_CAPTURE_SCHEMA_VERSION,
            "protocol_signature_schema_version": "pokezero.protocol-signature-census.v2",
            "protocol_signatures": {"move:tackle": 1},
            "protocol_signature_game_ids": ["locator"],
            "selfplay_protocol_capture": {"completed_games": 1, "capped_games": 0},
            "audit_provenance": provenance,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                _validate_existing_capture(path, expected_provenance=provenance, expected_games=1)["protocol_signatures"],
                {"move:tackle": 1},
            )
            changed = {**provenance, "execution_scope": {**provenance["execution_scope"], "seed_range": {"start": 11, "end": 11, "count": 1}}}
            with self.assertRaisesRegex(ValueError, "policy or seed scope"):
                _validate_existing_capture(path, expected_provenance=changed, expected_games=1)

    def test_rejects_capped_rollouts_without_writing_partial_census_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "refuses capped rollouts"):
            capture_selfplay_protocol_signatures(
                games=1,
                env_factory=lambda: _Env(capped=True),
                policies={"p1": _Policy("learned-a"), "p2": _Policy("learned-b")},
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=50,
            )

    def test_provenance_redacts_policy_specifications_in_the_stored_command(self) -> None:
        provenance = _capture_provenance(
            source_hash="source-hash",
            command_arguments=(
                "--p1-policy", "neural:/private/checkpoint-a.pt?token=secret",
                "--p2-policy=neural:/private/checkpoint-b.pt?token=secret",
            ),
            seed_start=10,
            games=1,
            capture_label="production-line",
            max_decision_rounds=250,
            p1_policy_identity={"policy_id": "learned-a", "policy_spec_sha256": "p1", "weights_sha256": "weights-a"},
            p2_policy_identity={"policy_id": "learned-b", "policy_spec_sha256": "p2", "weights_sha256": "weights-b"},
        )

        command = provenance["command"]
        self.assertIsInstance(command, list)
        self.assertNotIn("checkpoint-a.pt", " ".join(command))
        self.assertNotIn("checkpoint-b.pt", " ".join(command))
        self.assertIn("sha256:", " ".join(command))


if __name__ == "__main__":
    unittest.main()
