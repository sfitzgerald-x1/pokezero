"""Contract tests for the create-only root-action audit readout."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "root_action_audit_readout_test", ROOT / "scripts" / "root_action_audit_readout.py"
)
assert _SPEC is not None and _SPEC.loader is not None
READOUT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = READOUT
_SPEC.loader.exec_module(READOUT)


def _sidecar(*, seed: int, seat: str, round_index: int, capped: bool = False) -> dict[str, object]:
    actions = [
        {"action_label": "raw_policy", "action_index": 2},
        {"action_label": "mcts_selected", "action_index": 4},
    ]
    targets = []
    for target in ("policy_consistent", "uniform_own"):
        trials = []
        for trial_seed in range(16):
            winner = seat if trial_seed % 2 else ("p2" if seat == "p1" else "p1")
            trials.append({
                "continuation_rng_seed": trial_seed,
                "outcomes": [
                    {
                        "action_label": "raw_policy", "action_index": 2,
                        "continuation": {
                            "decision_round_count": 2, "terminal_after_fixed_joint_step": False,
                            "initial_max_continuation_decision_rounds": 250,
                            "effective_max_continuation_decision_rounds": 250,
                            "cap_retry": False,
                            "terminal": {"winner": winner, "turn_count": 3, "capped": capped},
                        },
                    },
                    {
                        "action_label": "mcts_selected", "action_index": 4,
                        "continuation": {
                            "decision_round_count": 2, "terminal_after_fixed_joint_step": False,
                            "initial_max_continuation_decision_rounds": 250,
                            "effective_max_continuation_decision_rounds": 250,
                            "cap_retry": False,
                            "terminal": {"winner": seat, "turn_count": 3, "capped": capped},
                        },
                    },
                ],
            })
        targets.append({"target": target, "trials": trials})
    evidence = {
        "model_argmax": 2, "search_argmax": 4, "model_override": True,
        "root_q_gap": 0.1, "root_visit_gap": 0.2, "root_gap_action_indices": [4, 2],
        "root_allocation": {
            "worlds": 1, "prior_authority": True, "prior_cause": None,
            "arms": [
                {"action_index": 2, "visit_share": 0.4, "q": 0.1, "reported_prior": 0.5, "model_prior": 0.5},
                {"action_index": 4, "visit_share": 0.6, "q": 0.2, "reported_prior": 0.4, "model_prior": 0.4},
            ],
        },
    }
    return {
        "schema_version": READOUT.WRAPPER_SCHEMA,
        "candidate_provenance_sha256": "a" * 64,
        "raw_provenance_sha256": "b" * 64,
        "candidate_seat": seat,
        "readout": {
            "schema_version": READOUT.READOUT_SCHEMA,
            "seed": seed,
            "battle_id": f"mcts-h2h-{seed}-{seat}",
            "candidate_seat": seat,
            "decision_round_index": round_index,
            "audit_status": "PAIRED",
            "audit": {
                "schema_version": READOUT.GRID_SCHEMA,
                "source_battle_id": f"mcts-h2h-{seed}-{seat}",
                "source_seed": seed,
                "source_decision_round": round_index,
                "subject_player": seat,
                "opponent_player": "p2" if seat == "p1" else "p1",
                "opponent_action_held_fixed": True,
                "actions": actions,
                "search_evidence": evidence,
                "continuation_targets": targets,
            },
        },
    }


class RootActionAuditReadoutTest(unittest.TestCase):
    def _write_complete_root(self, root: Path, *, capped: bool = False) -> None:
        for source_seed in range(4):
            for seat in ("p1", "p2"):
                for round_index in (3, 7):
                    directory = root / "seeds" / f"seed-{source_seed}" / "sealed-root-action-audits" / f"seed-{source_seed}-{seat}"
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / f"round-{round_index}.json").write_text(json.dumps(_sidecar(
                        seed=source_seed, seat=seat, round_index=round_index, capped=capped
                    )), encoding="utf-8")
            seed_root = root / "seeds" / f"seed-{source_seed}"
            input_root = seed_root / "input"
            input_root.mkdir(parents=True, exist_ok=True)
            (input_root / "MANIFEST.json").write_text(json.dumps({
                "seeds": [source_seed],
                "sealed_root_action_audit": {
                    "schema_version": "pokezero.mcts-guided-vs-raw-sealed-root-action-audit.v2",
                    "targets": [
                        {"seed": source_seed, "candidate_seat": seat, "decision_round_index": round_index}
                        for seat in ("p1", "p2") for round_index in (3, 7)
                    ],
                    "continuation_rng_seeds": list(range(16)),
                    "max_continuation_decision_rounds": 250,
                    "expanded_max_continuation_decision_rounds": 1024,
                    "continuation_targets": {
                        "policy_consistent": {"subject": "sampled_raw_transformer", "opponent": "sampled_raw_transformer"},
                        "uniform_own": {"subject": "uniform_legal", "opponent": "sampled_raw_transformer"},
                    },
                },
            }), encoding="utf-8")
            (seed_root / "COMPLETE.json").write_text(json.dumps({
                "schema_version": "test-seed", "status": "COMPLETE", "pairs": 1, "games": 2,
                "candidate_provenance_sha256": "a" * 64, "raw_provenance_sha256": "b" * 64,
            }), encoding="utf-8")
        (root / "COMPLETE.json").write_text(json.dumps({
            "schema_version": READOUT.ROOT_COMPLETE_SCHEMA,
            "status": "COMPLETE", "seeds": list(range(4)), "pairs": 4, "games": 8,
        }, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    def test_summarize_requires_a_complete_comparable_root_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_complete_root(root)
            summary = READOUT.summarize(root, expected_roots=16)
            self.assertEqual(summary["measurement_unit"], "source_root")
            self.assertEqual(summary["targets"]["policy_consistent"]["root_count"], 16)
            self.assertEqual(summary["targets"]["policy_consistent"]["correlation_clusters"], {
                "source_seed_count": 4, "source_game_count": 8,
            })
            self.assertEqual(summary["targets"]["policy_consistent"]["mcts_better_roots"], 16)
            self.assertEqual(
                [row["root_count"] for row in summary["targets"]["policy_consistent"]["by_source_seed"]],
                [4, 4, 4, 4],
            )
            self.assertEqual(
                [row["root_count"] for row in summary["targets"]["policy_consistent"]["by_source_game"]],
                [2] * 8,
            )

    def test_summarize_refuses_capped_continuations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_complete_root(root, capped=True)
            with self.assertRaisesRegex(READOUT.ReadoutError, "capped continuation"):
                READOUT.summarize(root, expected_roots=16)

    def test_summarize_refuses_a_complete_sidecar_set_without_root_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_complete_root(root)
            (root / "COMPLETE.json").unlink()
            with self.assertRaisesRegex(READOUT.ReadoutError, "root COMPLETE receipt"):
                READOUT.summarize(root, expected_roots=16)

    def test_summarize_refuses_sidecar_provenance_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_complete_root(root)
            path = root / "seeds" / "seed-0" / "sealed-root-action-audits" / "seed-0-p1" / "round-3.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["candidate_provenance_sha256"] = "c" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(READOUT.ReadoutError, "provenance"):
                READOUT.summarize(root, expected_roots=16)

    def test_summarize_refuses_unregistered_root_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_complete_root(root)
            old = root / "seeds" / "seed-0" / "sealed-root-action-audits" / "seed-0-p1" / "round-3.json"
            new = old.with_name("round-999.json")
            payload = json.loads(old.read_text(encoding="utf-8"))
            payload["readout"]["decision_round_index"] = 999
            payload["readout"]["audit"]["source_decision_round"] = 999
            old.unlink()
            new.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(READOUT.ReadoutError, "registered root roster"):
                READOUT.summarize(root, expected_roots=16)

    def test_summarize_refuses_impossible_cap_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_complete_root(root)
            path = root / "seeds" / "seed-0" / "sealed-root-action-audits" / "seed-0-p1" / "round-3.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            continuation = payload["readout"]["audit"]["continuation_targets"][0]["trials"][0]["outcomes"][0]["continuation"]
            continuation["cap_retry"] = True
            continuation["effective_max_continuation_decision_rounds"] = 1024
            with self.assertRaisesRegex(READOUT.ReadoutError, "incomplete or capped continuation"):
                path.write_text(json.dumps(payload), encoding="utf-8")
                READOUT.summarize(root, expected_roots=16)

    def test_create_only_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "readout.json"
            READOUT._write_create_only(output, {"schema_version": "test"})
            with self.assertRaisesRegex(READOUT.ReadoutError, "refusing to replace"):
                READOUT._write_create_only(output, {"schema_version": "test"})
