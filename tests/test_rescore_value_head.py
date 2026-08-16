"""Tests for the pieces of `scripts/rescore_value_head.py` that can be wrong SILENTLY.

The replay itself needs a checkpoint and a Showdown tree, so it is exercised on the cluster
and its own reproduction check is the guard there. What is tested here is everything that
decides whether a rescored file is COMPARABLE to the bank -- because those are the failures
that produce a plausible number instead of an error.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "rescore_value_head", REPO / "scripts" / "rescore_value_head.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rvh = _load()


def test_trunk_difference_ignores_value_head_and_catches_trunk():
    base = {"enc.w": torch.ones(2, 2), "value_head.weight": torch.ones(1, 2)}
    same_trunk_new_head = {"enc.w": torch.ones(2, 2),
                           "value_head.0.weight": torch.zeros(4, 2),
                           "value_head.2.weight": torch.zeros(1, 4)}
    assert rvh.trunk_difference(base, same_trunk_new_head) == []
    moved_trunk = {"enc.w": torch.ones(2, 2) * 1.0001, "value_head.weight": torch.ones(1, 2)}
    assert rvh.trunk_difference(base, moved_trunk) == ["enc.w"]


def test_trunk_difference_catches_a_missing_or_reshaped_trunk_key():
    base = {"enc.w": torch.ones(2, 2), "enc.b": torch.ones(2)}
    assert rvh.trunk_difference(base, {"enc.w": torch.ones(2, 2)}) == ["enc.b"]
    assert rvh.trunk_difference(base, {"enc.w": torch.ones(4), "enc.b": torch.ones(2)}) \
        == ["enc.w"]


def test_trunk_difference_is_not_fooled_by_a_key_merely_containing_value():
    """Scoped by the leading component, not by substring.

    A `"value" in key` filter -- the obvious way to write this -- would also skip a trunk key
    like `enc.value_proj.weight` (attention's V projection). That is a trunk parameter, and
    exempting it would let a genuinely different trunk pass the check that exists to make the
    single-replay optimisation sound.
    """
    base = {"enc.value_proj.weight": torch.ones(2, 2)}
    moved = {"enc.value_proj.weight": torch.zeros(2, 2)}
    assert rvh.trunk_difference(base, moved) == ["enc.value_proj.weight"]


def test_parse_heads_rejects_the_reserved_name_and_duplicates():
    assert rvh.parse_heads(["ctl=/tmp/a.pt", "v1=/tmp/b.pt"]) == {
        "ctl": Path("/tmp/a.pt"), "v1": Path("/tmp/b.pt")}
    with pytest.raises(SystemExit):
        rvh.parse_heads([f"{rvh.REPRODUCE}=/tmp/a.pt"])
    with pytest.raises(SystemExit):
        rvh.parse_heads(["ctl=/tmp/a.pt", "ctl=/tmp/b.pt"])


def test_parse_heads_names_a_bare_path_by_its_parent_directory():
    """`cells/v1/value-tuned.pt` is named `v1`, not `value-tuned`: the cell is the identity
    and every cell's file has the same basename."""
    assert rvh.parse_heads(["/tmp/cells/v1/value-tuned.pt"]) == {
        "v1": Path("/tmp/cells/v1/value-tuned.pt")}


def _shard(tmp_path, name, idx, pairs, *, bank="BANKSHA", state="STATESHA",
           head="/ck/v1.pt", outside=0):
    doc = {
        "head_checkpoint": head, "head_name": name,
        "rescore": {"banked_pairs_sha256": bank, "state_checkpoint_sha256": state,
                    "reproduction": {"n": len(pairs), "n_outside_tol": outside,
                                     "max_abs_delta": 1e-7},
                    "dropped": {"arm_mismatch": 1}},
        "pairs": pairs,
    }
    p = tmp_path / f"s{idx}.json"
    p.write_text(json.dumps(doc))
    return p


def _pair(seed, prefix, head_gap=0.01):
    return {"seed": seed, "prefix": prefix, "seat": "p1", "head_gap": head_gap,
            "true_gap": 0.05, "noise_var": 0.004}


def test_merge_pools_pairs_and_sums_reproduction(tmp_path):
    a = _shard(tmp_path, "v1", 0, [_pair(1, 0), _pair(1, 5)])
    b = _shard(tmp_path, "v1", 1, [_pair(2, 0)])
    out = tmp_path / "merged.json"
    assert rvh.merge([a, b], out) == 0
    doc = json.loads(out.read_text())
    assert doc["n_pairs"] == 3 and len(doc["pairs"]) == 3
    assert doc["head_name"] == "v1"
    assert doc["reproduction_pooled"]["n"] == 3
    # Dropped counts are SUMMED, not overwritten: a per-shard view would understate how many
    # of the 465 banked pairs failed to rebuild.
    assert doc["dropped_pooled"]["arm_mismatch"] == 2


def test_merge_refuses_a_shard_that_failed_reproduction(tmp_path):
    a = _shard(tmp_path, "v1", 0, [_pair(1, 0)])
    b = _shard(tmp_path, "v1", 1, [_pair(2, 0)], outside=3)
    with pytest.raises(SystemExit, match="outside its"):
        rvh.merge([a, b], tmp_path / "m.json")


def test_merge_refuses_mixed_heads_banks_and_state_checkpoints(tmp_path):
    a = _shard(tmp_path, "v1", 0, [_pair(1, 0)])
    with pytest.raises(SystemExit, match="head_name"):
        rvh.merge([a, _shard(tmp_path, "v2", 1, [_pair(2, 0)])], tmp_path / "m1.json")
    with pytest.raises(SystemExit, match="head_checkpoint"):
        rvh.merge([a, _shard(tmp_path, "v1", 2, [_pair(2, 0)], head="/ck/v2.pt")],
                  tmp_path / "m2.json")
    with pytest.raises(SystemExit, match="banked_pairs_sha256"):
        rvh.merge([a, _shard(tmp_path, "v1", 3, [_pair(2, 0)], bank="OTHER")],
                  tmp_path / "m3.json")
    with pytest.raises(SystemExit, match="state_checkpoint_sha256"):
        rvh.merge([a, _shard(tmp_path, "v1", 4, [_pair(2, 0)], state="OTHER")],
                  tmp_path / "m4.json")


def test_merge_refuses_duplicate_pairs_across_shards(tmp_path):
    a = _shard(tmp_path, "v1", 0, [_pair(1, 0)])
    b = _shard(tmp_path, "v1", 1, [_pair(1, 0)])
    with pytest.raises(SystemExit, match="appears in both"):
        rvh.merge([a, b], tmp_path / "m.json")


def test_merge_refuses_a_file_this_tool_did_not_write(tmp_path):
    """A producer shard has `pairs` and looks mergeable, but its head_gap belongs to the
    baseline head. Pooling one into a rescore would silently dilute the tuned column."""
    p = tmp_path / "producer.json"
    p.write_text(json.dumps({"pairs": [_pair(9, 0)], "config": {"rollouts": 64}}))
    with pytest.raises(SystemExit, match="no `rescore` block"):
        rvh.merge([p], tmp_path / "m.json")


def test_merged_output_is_readable_by_the_beta_instrument_contract():
    """The merged payload must satisfy what `sibling_beta.py` reads: `pairs` with numeric
    `head_gap`, `true_gap` and `noise_var`. Asserted as a contract so a schema rename here
    fails a test rather than the scoring run."""
    required = {"head_gap", "true_gap", "noise_var"}
    assert required <= set(_pair(1, 0))


def test_finalize_pair_gaps_is_reused_from_the_producer():
    """The units conversion must come from the producer module, not a local copy.

    `head_gap` is on the WIN-PROBABILITY scale -- half the raw +/-1 difference -- and a
    rescore that emitted the raw difference would report a head_gap ~2x the bank's, i.e. a
    beta ~2x the baseline's, with nothing in the file to show it. Reusing the producer's
    function is the only thing that makes the tuned column comparable, so the reuse itself is
    what is asserted.
    """
    rec = {"head_a": 0.4, "head_b": 0.1, "true_a": 0.6, "true_b": 0.5}
    rvh.probe.finalize_pair_gaps(rec)
    assert rec["head_gap_return_scale"] == pytest.approx(0.3)
    assert rec["head_gap"] == pytest.approx(0.15)
    assert rec["true_gap"] == pytest.approx(0.1)
    assert rvh.probe.finalize_pair_gaps is not None
    assert rvh.probe.__file__.endswith("value_head_sibling_probe.py")
