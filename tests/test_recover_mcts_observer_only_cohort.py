"""Contract tests for the mixed-source observer-only recovery fan-in."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "recover_mcts_observer_only_cohort.py"
SPEC = importlib.util.spec_from_file_location("recover_mcts_observer_only_cohort", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


OLD_COMMIT = "a" * 40
NEW_COMMIT = "b" * 40
ENGINE = "c" * 64
SHOWDOWN = "d" * 64
CHECKPOINT = "e" * 64


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _policy(commit: str, tree: str, policy_id: str) -> dict[str, object]:
    return {
        "config_id": policy_id,
        "policy_id": policy_id,
        "source_commit": commit,
        "source_tree_sha256": tree,
        "engine_fingerprint": ENGINE,
        "checkpoint_sha256": CHECKPOINT,
        "showdown_source_sha256": SHOWDOWN,
        "config": {"search_sims": 4096, "model_priors": True},
    }


def _seed_dir(root: Path, seed: int, *, original: bool) -> Path:
    if original:
        shard = "s0" if seed < 2026093100 else "s1"
        return root / "shards" / shard / "seeds" / f"seed-{seed}"
    lane = "s0" if seed in MODULE.DEFAULT_REPAIR_SEEDS[:3] else "s1"
    return root / "lanes" / lane / "seeds" / f"seed-{seed}"


def _write_terminal(root: Path, seed: int, *, commit: str, original: bool) -> None:
    directory = _seed_dir(root, seed, original=original)
    tree = "1" * 64 if commit == OLD_COMMIT else "2" * 64
    candidate = _policy(commit, tree, "candidate")
    raw = _policy(commit, tree, "raw")
    manifest = {
        "active_source": {"commit": commit, "tree_sha256": tree},
        "active_engine_fingerprint": ENGINE,
        "active_showdown_source": {"content_sha256": SHOWDOWN},
        "candidate": candidate,
        "raw": raw,
        "declared_manifest": {
            "seeds": [seed],
            "candidate": candidate,
            "raw": raw,
            "checkpoint_sha256": CHECKPOINT,
            "showdown_source_sha256": SHOWDOWN,
            "max_decision_rounds": 250,
            "bootstrap": {"resamples": 100, "seed": 7, "confidence_level": 0.95},
        },
    }
    _write_json(directory / "manifest.json", manifest)
    summary = directory / "summary.json"
    _write_json(
        summary,
        {
            "seeds": [seed],
            "pair_scores": [0.5],
            "candidate_prior_fallbacks": 0,
            "candidate_root_prior_fallbacks": 0,
            "candidate_branch_prior_fallbacks": 0,
            "candidate_opponent_request_order_root_fallback_statuses": {},
            "candidate_opponent_request_order_root_omission_statuses": {},
            "incumbent_prior_fallbacks": 0,
            "incumbent_root_prior_fallbacks": 0,
            "incumbent_branch_prior_fallbacks": 0,
            "incumbent_opponent_request_order_root_fallback_statuses": {},
            "incumbent_opponent_request_order_root_omission_statuses": {},
        },
    )
    candidate_sha = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    raw_sha = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write_json(
        directory / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "pairs": 1,
            "games": 2,
            "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
            "candidate_provenance_sha256": candidate_sha,
            "raw_provenance_sha256": raw_sha,
        },
    )
    _write_json(directory / "runner-terminal.json", {"status": "COMPLETE", "exit_code": 0})


def _fake_pair_games() -> dict[str, object]:
    return {
        "p1": SimpleNamespace(
            candidate_seat="p1",
            result=SimpleNamespace(score=1.0, outcome="win"),
            terminal_capped=False,
        ),
        "p2": SimpleNamespace(
            candidate_seat="p2",
            result=SimpleNamespace(score=0.0, outcome="loss"),
            terminal_capped=False,
        ),
    }


class ObserverRecoveryTest(unittest.TestCase):
    def _roots(self, directory: Path) -> tuple[Path, Path]:
        original = directory / "original"
        replay = directory / "replay"
        repair = set(MODULE.DEFAULT_REPAIR_SEEDS)
        for seed in MODULE.DEFAULT_SEEDS:
            if seed not in repair:
                _write_terminal(original, seed, commit=OLD_COMMIT, original=True)
        for seed in MODULE.DEFAULT_REPAIR_SEEDS:
            _write_terminal(replay, seed, commit=NEW_COMMIT, original=False)
        return original, replay

    def test_recovers_exactly_195_original_pairs_and_five_replays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original, replay = self._roots(Path(temporary))
            games = _fake_pair_games()
            with patch.object(MODULE, "load_pair", return_value=games), patch.object(
                MODULE, "complete_pair", return_value=tuple(games.values())
            ):
                manifest = MODULE.build_recovery_manifest(
                    original_root=original,
                    replay_root=replay,
                    original_commit=OLD_COMMIT,
                    replay_commit=NEW_COMMIT,
                )
        self.assertEqual(manifest["status"], "RECOVERED_COMPLETE")
        self.assertEqual(manifest["pair_count"], 200)
        self.assertEqual(manifest["game_count"], 400)
        self.assertEqual(manifest["recovery_scope"]["replayed_games"], 10)
        self.assertEqual(manifest["candidate_margin_over_neutral_95ci"]["low"], 0.0)
        self.assertEqual(manifest["seat_sensitivity"]["p1_minus_p2_score_95ci"]["point"], 1.0)
        self.assertEqual(manifest["cap_sensitivity"]["capped_games"], 0)
        self.assertEqual([item["seed"] for item in manifest["replayed_pairs"]], list(MODULE.DEFAULT_REPAIR_SEEDS))

    def test_rejects_a_missing_retained_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original, replay = self._roots(Path(temporary))
            missing = _seed_dir(original, 2026093000, original=True) / "COMPLETE.json"
            missing.unlink()
            with self.assertRaisesRegex(MODULE.RecoveryError, "retained seed 2026093000"):
                MODULE.build_recovery_manifest(
                    original_root=original,
                    replay_root=replay,
                    original_commit=OLD_COMMIT,
                    replay_commit=NEW_COMMIT,
                )

    def test_rejects_fallback_tainted_pair_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original, replay = self._roots(Path(temporary))
            summary = _seed_dir(replay, MODULE.DEFAULT_REPAIR_SEEDS[0], original=False) / "summary.json"
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["candidate_root_prior_fallbacks"] = 1
            _write_json(summary, payload)
            complete = summary.with_name("COMPLETE.json")
            complete_payload = json.loads(complete.read_text(encoding="utf-8"))
            complete_payload["summary_sha256"] = hashlib.sha256(summary.read_bytes()).hexdigest()
            _write_json(complete, complete_payload)
            games = _fake_pair_games()
            with patch.object(MODULE, "load_pair", return_value=games), patch.object(
                MODULE, "complete_pair", return_value=tuple(games.values())
            ), self.assertRaisesRegex(MODULE.RecoveryError, "candidate_root_prior_fallbacks"):
                MODULE.build_recovery_manifest(
                    original_root=original,
                    replay_root=replay,
                    original_commit=OLD_COMMIT,
                    replay_commit=NEW_COMMIT,
                )

    def test_rejects_a_replay_that_changes_search_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original, replay = self._roots(Path(temporary))
            changed = _seed_dir(replay, MODULE.DEFAULT_REPAIR_SEEDS[0], original=False) / "manifest.json"
            payload = json.loads(changed.read_text(encoding="utf-8"))
            payload["declared_manifest"]["candidate"]["config"]["search_sims"] = 1
            _write_json(changed, payload)
            games = _fake_pair_games()
            with patch.object(MODULE, "load_pair", return_value=games), patch.object(
                MODULE, "complete_pair", return_value=tuple(games.values())
            ), self.assertRaisesRegex(MODULE.RecoveryError, "battle/search setting"):
                MODULE.build_recovery_manifest(
                    original_root=original,
                    replay_root=replay,
                    original_commit=OLD_COMMIT,
                    replay_commit=NEW_COMMIT,
                )

    def test_rejects_a_replay_seed_present_in_two_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original, replay = self._roots(Path(temporary))
            seed = MODULE.DEFAULT_REPAIR_SEEDS[0]
            duplicate = replay / "lanes" / "s1" / "seeds" / f"seed-{seed}"
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            _write_json(duplicate / "sentinel.json", {"unexpected": True})
            games = _fake_pair_games()
            with patch.object(MODULE, "load_pair", return_value=games), patch.object(
                MODULE, "complete_pair", return_value=tuple(games.values())
            ), self.assertRaisesRegex(MODULE.RecoveryError, "exactly one durable lane"):
                MODULE.build_recovery_manifest(
                    original_root=original,
                    replay_root=replay,
                    original_commit=OLD_COMMIT,
                    replay_commit=NEW_COMMIT,
                )


if __name__ == "__main__":
    unittest.main()
