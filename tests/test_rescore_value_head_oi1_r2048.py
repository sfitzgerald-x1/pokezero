"""Adversarial contract tests for the Phase-3 R=2048 rescore boundary."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oi1_rescore", REPO / "scripts" / "rescore_value_head_oi1_r2048.py")
assert SPEC is not None and SPEC.loader is not None
rescore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rescore)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(n: int = 1800) -> list[dict]:
    out = []
    for index in range(n):
        true_gap = 0.2 if index % 2 else -0.2
        head_gap = true_gap / 2
        out.append({
            "seed": 28_400_000 + index // 6,
            "prefix": index % 6,
            "seat": "p1",
            "arm_a": index % 5,
            "arm_b": (index + 1) % 5,
            "true_a": 0.5 + true_gap / 2,
            "true_b": 0.5 - true_gap / 2,
            "true_gap": true_gap,
            "head_a": head_gap,
            "head_b": -head_gap,
            "head_gap": head_gap,
            "rollouts_a": 2048,
            "rollouts_b": 2048,
            "capped_a": 0,
            "capped_b": 0,
            "failed_a": [],
            "failed_b": [],
            "pairing_intact": True,
        })
    return out


def bank(tmp_path: Path, pairs: list[dict]) -> tuple[Path, Path, Path]:
    directory = tmp_path / "confirmation"
    directory.mkdir()
    shard = directory / "confirm-00.json"
    shard.write_text(json.dumps({
        "config": {"rollouts": 2048, "rollout_seed_salt": "oi1-targeted-gap-confirm-v1",
                   "device": "cpu"},
        "pairs": pairs,
    }), encoding="utf-8")
    corpus = tmp_path / "oi1-targeted-gap-corpus.json"
    corpus.write_text(json.dumps({
        "schema_version": rescore.CORPUS_SCHEMA,
        "contract_sha256": rescore.oi.CONTRACT_SHA256,
        "confirmation_shard_sha256": {shard.name: digest(shard)},
        "complete_confirmation_pairs": len(pairs),
        "primary_tau_eligible_pairs": len(pairs),
    }), encoding="utf-8")
    return corpus, directory, shard


def test_loads_only_the_sealed_complete_r2048_confirmation_inventory(tmp_path):
    corpus, directory, shard = bank(tmp_path, rows())
    loaded, sealed, corpus_sha, shards = rescore.load_confirmation(corpus, directory)
    assert len(loaded) == 1800
    assert sealed["complete_confirmation_pairs"] == 1800
    assert corpus_sha == digest(corpus)
    assert shards == {shard.name: digest(shard)}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"rollouts_a": 64}, "complete R=2048"),
        ({"capped_a": 1}, "capped_a"),
        ({"capped_b": True}, "capped_b"),
        ({"failed_a": [2]}, "failed_a"),
        ({"pairing_intact": False}, "paired rollout"),
        ({"seat": "p2"}, "nonregistered seat"),
        ({"head_gap": 0.9}, "head_gap disagrees"),
    ],
)
def test_refuses_invalid_confirmation_evidence_before_rescoring(tmp_path, mutation, message):
    bad = rows()
    bad[0].update(mutation)
    corpus, directory, _ = bank(tmp_path, bad)
    with pytest.raises(rescore.Refusal, match=message):
        rescore.load_confirmation(corpus, directory)


def test_accepts_the_registered_sparse_zero_cap_encoding(tmp_path):
    good = rows()
    del good[0]["capped_a"]
    corpus, directory, _ = bank(tmp_path, good)
    loaded, *_ = rescore.load_confirmation(corpus, directory)
    assert len(loaded) == 1800


def test_refuses_missing_or_changed_confirmation_inventory(tmp_path):
    corpus, directory, shard = bank(tmp_path, rows())
    changed = directory / "confirm-extra.json"
    changed.write_text("{}", encoding="utf-8")
    with pytest.raises(rescore.Refusal, match="exactly match"):
        rescore.load_confirmation(corpus, directory)
    changed.unlink()
    shard.write_text(shard.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(rescore.Refusal, match="differs from the corpus digest"):
        rescore.load_confirmation(corpus, directory)


def test_create_only_cells_bind_the_r2048_corpus_and_identical_keys(tmp_path):
    pairs = rows()
    corpus, directory, shard = bank(tmp_path, pairs)
    loaded, _, corpus_sha, shards = rescore.load_confirmation(corpus, directory)
    output = tmp_path / "cells"
    rescore.write_cells_new(
        output, {"control": loaded, "v1": loaded}, corpus_sha256=corpus_sha,
        shard_sha256=shards, source_checkpoint_sha256="a" * 64,
        checkpoint_sha256={"control": "b" * 64, "v1": "c" * 64},
        source_reproduction={"n": 1800, "tol": 1e-4, "max_abs_delta": 0.0},
    )
    control, control_meta = rescore.oi.load_cell(output / "control.json", "control")
    v1, v1_meta = rescore.oi.load_cell(output / "v1.json", "v1")
    assert rescore.oi.align_metadata({"control": control_meta, "v1": v1_meta})
    assert len(control) == len(v1) == 1800
    assert control_meta["phase3_rescore"]["candidate_checkpoint_sha256"] == "b" * 64
    with pytest.raises(rescore.Refusal, match="create-only"):
        rescore.write_cells_new(
            output, {"control": loaded}, corpus_sha256=corpus_sha, shard_sha256=shards,
            source_checkpoint_sha256="a" * 64, checkpoint_sha256={"control": "b" * 64},
            source_reproduction={"n": 1800},
        )


def test_parse_refuses_ambiguous_candidate_identity():
    with pytest.raises(rescore.Refusal, match="--head"):
        rescore._parse("checkpoint.pt")
    with pytest.raises(rescore.Refusal, match="reserves"):
        rescore._parse(f"{rescore.SOURCE_NAME}=checkpoint.pt")
    assert rescore._parse("v2=/tmp/v2.pt") == ("v2", Path("/tmp/v2.pt"))


def test_source_checkpoint_identity_is_an_immutable_digest_not_a_path_claim():
    assert rescore.SOURCE_CHECKPOINT_SHA256 == (
        "0897676c295a79bac0b24c347b8f6a72e1359b98c37327bc5639fb6229005937")
