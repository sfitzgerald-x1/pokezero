"""Contract tests for the R=2048-only Phase-3 OI scorer configuration."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oi1_r2048", REPO / "scripts" / "value_head_ordering_auc_r2048.py")
assert SPEC is not None and SPEC.loader is not None
oi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oi)


def pairs(n: int = 1800, *, corrupt: int = 0) -> list[dict]:
    """A powered confirmation-shaped bank; corruption is a demonstrated failing input."""
    rows = []
    for index in range(n):
        true_gap = 0.20 if index % 2 else -0.20
        head_gap = -true_gap if index < corrupt else true_gap
        level = 0.1 if index % 3 else -0.1
        rows.append({
            "seed": 28_400_000 + index // 6,
            "prefix": index % 6,
            "seat": "p1",
            "arm_a": index % 4,
            "arm_b": (index + 1) % 4,
            "head_a": level + head_gap,
            "head_b": level - head_gap,
            "head_gap": head_gap,
            "true_gap": true_gap,
            "true_a": 0.5 + true_gap / 2,
            "true_b": 0.5 - true_gap / 2,
            "noise_var": 0.0002,
            "rollouts_a": 2048,
            "rollouts_b": 2048,
            "capped_a": 0,
            "capped_b": 0,
            "failed_a": [],
            "failed_b": [],
            "pairing_intact": True,
        })
    return rows


def write_cell(tmp_path: Path, name: str, rows: list[dict], **meta_override: object) -> Path:
    keyed = {oi.base.pair_key(row): row for row in rows}
    meta = {
        "schema": oi.SCHEMA,
        "contract_sha256": oi.CONTRACT_SHA256,
        "stage": "confirmation",
        "rollouts_per_arm": oi.ROLLOUTS,
        "screen_rows_included": False,
        "corpus_sha256": "a" * 64,
        "confirmation_shard_sha256": {"confirm-00.json": "b" * 64},
        "pair_set_sha256": oi.pair_set_sha256(keyed),
        "source_checkpoint_sha256": "c" * 64,
        **meta_override,
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"pairs": rows, "oi1_targeted_gap_r2048": meta}), encoding="utf-8")
    return path


def test_powered_r2048_confirmation_bank_scores_and_the_corruption_control_regresses(tmp_path):
    reference = write_cell(tmp_path, "base", pairs())
    # Flip enough actual orderings to demonstrate that the configuration can read False.
    corrupted = write_cell(tmp_path, "corrupt", pairs(corrupt=180))
    base_rows, base_meta = oi.load_cell(reference, "base")
    arm_rows, arm_meta = oi.load_cell(corrupted, "corrupt")
    result = oi.score("base", {"base": base_rows, "corrupt": arm_rows},
                      {"base": base_meta, "corrupt": arm_meta}, bootstrap_reps=2)
    primary = result["cells"]["corrupt"]["0.1"]
    assert primary["delta_c_gate"] <= -0.02
    assert primary["verdict"] == "REGRESSED"
    assert result["label_se_vs_tau"]["condition_label_se_much_less_than_tau_met"] is True
    assert result["r2048_contract"]["se_over_tau_primary"] == 0.15625


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"rollouts_per_arm": 64}, "rollouts_per_arm"),
        ({"stage": "screen"}, "stage"),
        ({"screen_rows_included": True}, "screen_rows_included"),
        ({"contract_sha256": "d" * 64}, "contract_sha256"),
    ],
)
def test_refuses_the_screen_r64_or_foreign_contract_substitution(tmp_path, override, message):
    path = write_cell(tmp_path, "bad", pairs(), **override)
    with pytest.raises(oi.Refusal, match=message):
        oi.load_cell(path, "bad")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"rollouts_a": 64}, "pinned at 2048"),
        ({"capped_a": 1}, "uncapped"),
        ({"failed_b": [3]}, "complete"),
        ({"capped_b": True}, "uncapped"),
        ({"failed_a": "[]"}, "complete"),
        ({"pairing_intact": False}, "incomplete"),
    ],
)
def test_refuses_capped_failed_or_unpaired_confirmation_truth(tmp_path, mutation, message):
    rows = pairs()
    rows[0].update(mutation)
    path = write_cell(tmp_path, "bad", rows)
    with pytest.raises(oi.Refusal, match=message):
        oi.load_cell(path, "bad")


@pytest.mark.parametrize(
    ("field", "message"),
    [("failed_b", "complete")],
)
def test_refuses_missing_required_rollout_completion_telemetry(tmp_path, field, message):
    rows = pairs()
    del rows[0][field]
    path = write_cell(tmp_path, "bad", rows)
    with pytest.raises(oi.Refusal, match=message):
        oi.load_cell(path, "bad")


def test_refuses_missing_per_pair_r2048_count(tmp_path):
    rows = pairs()
    del rows[0]["rollouts_b"]
    path = write_cell(tmp_path, "bad", rows)
    with pytest.raises(oi.Refusal, match="complete R=2048"):
        oi.load_cell(path, "bad")


def test_accepts_the_registered_source_cap_zero_encoding(tmp_path):
    rows = pairs()
    del rows[0]["capped_a"]
    path = write_cell(tmp_path, "canonical", rows)
    loaded, _ = oi.load_cell(path, "canonical")
    assert len(loaded) == 1800


def test_refuses_mismatched_pair_or_corpus_provenance(tmp_path):
    base = write_cell(tmp_path, "base", pairs())
    other = write_cell(tmp_path, "other", pairs(), corpus_sha256="z" * 64)
    base_rows, base_meta = oi.load_cell(base, "base")
    other_rows, other_meta = oi.load_cell(other, "other")
    with pytest.raises(oi.Refusal, match="corpus_sha256"):
        oi.score("base", {"base": base_rows, "other": other_rows},
                 {"base": base_meta, "other": other_meta}, bootstrap_reps=2)


def test_output_is_create_only(tmp_path):
    output = tmp_path / "score.json"
    oi._write_new(output, {"valid": True})
    with pytest.raises(oi.Refusal, match="create-only"):
        oi._write_new(output, {"valid": False})
