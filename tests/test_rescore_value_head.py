"""Tests for the pieces of `scripts/rescore_value_head.py` that can be wrong SILENTLY.

Two halves.

The first is the pure logic -- trunk comparison, head naming, shard pooling -- which decides
whether a rescored file is COMPARABLE to the bank, and those are the failures that produce a
plausible number instead of an error.

The second is the REPLAY, and it used to be absent on the grounds that it needs a checkpoint
and a Showdown tree. It does not. What the replay loop actually needs is a value function, an
observation history and a branch, and every one of those can be a deterministic stub -- so
the loop is driven here end to end with the head replaced by a pure function of the
observation history (`_head_value`). That is enough to pin the FIDELITY MACHINERY: the
comparison the tool draws between replayed and banked head values, and the three refusals
that act on it.

  * the reproduction refusal (rebuilt states must be the banked states),
  * the recomputed-`true_gap` check (the reused ground truth must be the file's),
  * arm agreement (a pair whose arms re-derive differently is dropped, never rescored).

Deleting any one of those left the suite green before these tests existed, which is the same
shape of hole the units blocker sat in twice.

WHAT THIS STILL DOES NOT SHOW, stated because the distinction is the whole point of the
instrument: that the REAL replay reproduces the REAL bank. That is a property of the env, the
checkpoint and the device, it can only be measured on the cluster, and the tool's own
reproduction check is the measurement. What is established here is that the check exists, is
reached, is exact when the states agree, and refuses when they do not -- i.e. that the cluster
run's green is worth something.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
import types
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
           head="/ck/v1.pt", outside=0, tol=1e-4):
    doc = {
        "head_checkpoint": head, "head_name": name,
        "rescore": {"banked_pairs_sha256": bank, "state_checkpoint_sha256": state,
                    "reproduction": {"n": len(pairs), "n_outside_tol": outside,
                                     "tol": tol, "max_abs_delta": 1e-7},
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


def test_merged_output_is_readable_by_the_beta_instrument_contract(tmp_path):
    """The MERGED FILE must satisfy what `sibling_beta.py` reads: `pairs` with numeric
    `head_gap`, `true_gap` and `noise_var`.

    Asserted against `merge()`'s actual output, because the earlier version of this test
    asserted the keys on the fixture helper that the test file itself wrote -- a tautology
    that never called `merge` at all and would have stayed green while the merger dropped
    every column the scoring run reads.
    """
    a = _shard(tmp_path, "v1", 0, [_pair(1, 0, head_gap=0.013), _pair(1, 5, head_gap=-0.021)])
    b = _shard(tmp_path, "v1", 1, [_pair(2, 0, head_gap=0.004)])
    out = tmp_path / "merged.json"
    assert rvh.merge([a, b], out) == 0

    doc = json.loads(out.read_text())
    required = {"head_gap", "true_gap", "noise_var"}
    for row in doc["pairs"]:
        missing = required - set(row)
        assert not missing, f"merge dropped {sorted(missing)} from a pooled pair"
        for key in required:
            assert isinstance(row[key], float), f"{key} is {type(row[key]).__name__}"
    # Values, in order, not merely key presence: pooling must not reorder or rewrite the
    # column the regression is taken over.
    assert [r["head_gap"] for r in doc["pairs"]] == [0.013, -0.021, 0.004]
    assert [r["true_gap"] for r in doc["pairs"]] == [0.05, 0.05, 0.05]
    assert [r["noise_var"] for r in doc["pairs"]] == [0.004, 0.004, 0.004]


def test_pooled_reproduction_reports_the_loosest_tolerance_and_no_dead_counter(tmp_path, capsys):
    """`reproduction_pooled` must not carry a failure count, because it could only read zero.

    A shard reporting any pair outside its own tolerance is REFUSED above, so an accumulator
    summing that field is reachable only when every term is 0 -- a field that reads like
    evidence and carries none. What IS reported instead is the loosest tolerance any pooled
    shard was checked at, which a reader cannot otherwise recover and which genuinely varies:
    shards may be run with different `--reproduce-tol`, and the pooled column is only as
    strong as the weakest one.
    """
    a = _shard(tmp_path, "v1", 0, [_pair(1, 0)], tol=1e-4)
    b = _shard(tmp_path, "v1", 1, [_pair(2, 0)], tol=1e-3)
    c = _shard(tmp_path, "v1", 2, [_pair(3, 0)], tol=1e-5)
    out = tmp_path / "merged.json"
    assert rvh.merge([a, b, c], out) == 0
    pooled = json.loads(out.read_text())["reproduction_pooled"]
    assert "n_outside_tol" not in pooled
    assert pooled["n"] == 3
    # The LOOSEST, not the first shard's and not the last shard's.
    assert pooled["tol_max"] == 1e-3
    printed = capsys.readouterr().out
    assert "outside tol" not in printed
    assert "loosest 0.001" in printed


def test_finalize_pair_gaps_is_reused_from_the_producer():
    """The units conversion must come from the producer module, not a local copy.

    `head_gap` is on the WIN-PROBABILITY scale -- half the raw +/-1 difference -- and a
    rescore that emitted the raw difference would report a head_gap ~2x the bank's, i.e. a
    beta ~2x the baseline's, with nothing in the file to show it. Reusing the producer's
    function is the only thing that makes the tuned column comparable, so the reuse itself is
    what is asserted.

    This test pins the FUNCTION only. The producer's own docstring says why that is not
    enough -- reverting the CALL SITE left the whole suite green last time -- so the call site
    is pinned separately, by
    `test_the_replay_stamps_head_gap_through_the_producers_units_conversion`.
    """
    rec = {"head_a": 0.4, "head_b": 0.1, "true_a": 0.6, "true_b": 0.5}
    rvh.probe.finalize_pair_gaps(rec)
    assert rec["head_gap_return_scale"] == pytest.approx(0.3)
    assert rec["head_gap"] == pytest.approx(0.15)
    assert rec["true_gap"] == pytest.approx(0.1)
    assert rvh.probe.finalize_pair_gaps is not None
    assert rvh.probe.__file__.endswith("value_head_sibling_probe.py")


def test_the_private_producer_helpers_keep_the_signatures_this_tool_calls_them_with():
    """Both cross-module private helpers are called POSITIONALLY, so a reorder in the producer
    would pass the wrong arguments rather than raise.

    `_top_two_and_opponent(traj, seat, prefix, ...)` and `_post_branch_history(br, seat,
    prefix_history)` are private to `value_head_sibling_probe.py` and imported anyway --
    deliberately, because restating them is the thing that makes a rescore incomparable. The
    cost of that choice is that the producer has no way to know it is an interface, so the
    interface is pinned here.
    """
    assert list(inspect.signature(rvh.probe._top_two_and_opponent).parameters) == [
        "traj", "seat", "prefix", "model", "result", "device",
        "evaluate_transformer_action_priors", "history"]
    assert list(inspect.signature(rvh.probe._post_branch_history).parameters) == [
        "branch_result", "seat", "prefix_history"]
    assert list(inspect.signature(rvh.probe.finalize_pair_gaps).parameters) == ["rec"]


# ---------------------------------------------------------------------------------------
# THE REPLAY, driven end to end against a deterministic stand-in for the value head.
#
# The tool's headline claim is REPLAY FIDELITY: that the states it rebuilds are the states
# the banked ground truth was measured at, and that it refuses to certify when they are not.
# Nothing measured that. Deleting the reproduction refusal, the recomputed-true_gap check or
# the whole arm-agreement branch each left the suite green.
#
# None of the machinery those guards defend needs a GPU, a checkpoint, a bank or a Showdown
# tree. It needs a value function, an observation history and a branch. So the value function
# becomes a pure function of the observation history, the histories become tuples of strings,
# and the bank is BUILT FROM THE SAME FUNCTION -- which is what makes "reproduction delta is
# exactly 0" a real assertion rather than a tolerance that passes on noise.
# ---------------------------------------------------------------------------------------

STATE_SCALE = 1.0        # the head that produced the bank, scored as `__reproduce__`
TUNED_SCALE = 0.5        # a different value head on a bit-identical trunk
BELIEF_HASH = "belief-set-source-hash-of-the-fixture"


def _obs_history(seed, seat, prefix):
    """What the stubbed `player_observation_history` returns for a (seed, seat, prefix).

    The bank builder and the stub both go through this, which is what makes the banked
    `head_a` the value the replay recomputes -- bit for bit, not within a tolerance.
    """
    return (f"traj{seed}", seat, f"round{prefix}")


def _branch_history(seed, seat, prefix, arm):
    """The successor history the head is scored at: the prefix plus the branched action."""
    return (*_obs_history(seed, seat, prefix), f"arm{arm}")


def _head_value(observations, scale):
    """A deterministic, history-sensitive stand-in for the value head.

    History-SENSITIVE is the load-bearing property: a constant would make both arms tie and
    every gap zero, which is precisely the bug `_post_branch_history` was written to avoid.
    """
    digest = hashlib.sha256("|".join(str(o) for o in observations).encode()).hexdigest()
    return scale * ((int(digest[:8], 16) % 2001) / 1000.0 - 1.0)


def _banked_pair(seed, prefix, *, seat="p1", replay_arms=(0, 1), banked_arms=None,
                 true_a=0.625, true_b=0.5, **over):
    """One banked pair, with head values the STATE head will reproduce exactly.

    `replay_arms` are the arms the stubbed arm-selection will re-derive; `banked_arms` is what
    the record CLAIMS, defaulting to the same. Separating them is what lets one fixture perturb
    the arms and nothing else.
    """
    arm_a, arm_b = replay_arms
    head_a = _head_value(_branch_history(seed, seat, prefix, arm_a), STATE_SCALE)
    head_b = _head_value(_branch_history(seed, seat, prefix, arm_b), STATE_SCALE)
    claimed_a, claimed_b = banked_arms if banked_arms is not None else replay_arms
    rec = {
        "seed": seed, "prefix": prefix, "seat": seat,
        "arm_a": claimed_a, "arm_b": claimed_b,
        "true_a": true_a, "true_b": true_b, "true_gap": true_a - true_b,
        "head_a": head_a, "head_b": head_b,
        "head_gap": (head_a - head_b) / 2.0,
        "head_gap_return_scale": head_a - head_b,
        "noise_var": 0.0043, "rollouts_a": 64, "rollouts_b": 64, "pairing_intact": True,
    }
    rec.update(over)
    return rec


class _FakeModel:
    def __init__(self, scale, head_fill):
        self.scale = scale
        # A bit-identical trunk and a DIFFERENT value head: the exact shape the tool's
        # trunk-identity check exists to admit.
        self._sd = {"enc.w": torch.ones(2, 2), "enc.b": torch.zeros(2),
                    "value_head.weight": torch.full((1, 2), head_fill)}

    def state_dict(self):
        return self._sd


class _FakeResult:
    def __init__(self):
        self.model_config = {"value_head_hidden": 256}
        self.belief_set_source_hash = BELIEF_HASH
        self.value_calibration_transform = None


class _FakeEnv:
    def __init__(self):
        self.belief_set_source_hash = BELIEF_HASH
        self.seed = None

    def reset(self, seed=None):
        self.seed = seed

    def observe(self, seat):
        return f"observe-fallback-{seat}"


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _run_replay(monkeypatch, tmp_path, banked, *, arms=None, argv_extra=()):
    """Run `rescore_value_head.main()` over `banked` with every heavy dependency stubbed.

    Returns the status (0, or 1 for any SystemExit with a message), that message, and the
    output directory. `arms` overrides the re-derived arms per (seed, seat, prefix).
    """
    arms = dict(arms or {})
    root = tmp_path / "showdown"
    root.mkdir()
    state_ckpt = tmp_path / "state.pt"
    state_ckpt.write_bytes(b"state-checkpoint-bytes")
    head_ckpt = tmp_path / "cells" / "v1" / "value-tuned.pt"
    head_ckpt.parent.mkdir(parents=True)
    head_ckpt.write_bytes(b"tuned-checkpoint-bytes")
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({
        "schema": "pokezero.phase0.vhprobe-pairs.v1",
        "provenance": [{"checkpoint": str(state_ckpt), "belief_set_source_hash": BELIEF_HASH}],
        "config": {"max_decision_rounds": 250},
        "n_pairs": len(banked), "pairs": banked,
    }))
    out_dir = tmp_path / "out"

    models = {str(state_ckpt): (_FakeModel(STATE_SCALE, 0.25), _FakeResult()),
              str(head_ckpt): (_FakeModel(TUNED_SCALE, 0.75), _FakeResult())}

    def load_transformer_checkpoint(path, map_location=None):
        return models[str(path)]

    def observation_value(*, model, result, observations, device):
        return _head_value(observations, model.scale)

    def top_two_and_opponent(traj, seat, prefix, model, result, device, priors_fn, history):
        assert history == _obs_history(traj.seed, seat, int(prefix))
        return arms.get((traj.seed, seat, int(prefix)), (0, 1, 2))

    def post_branch_history(branch_result, seat, prefix_history):
        return (*prefix_history, f"arm{branch_result['actions'][seat]}"), None

    def replay_trajectory_branch(env, traj, *, prefix_decision_round_count, branch_actions,
                                 check_prefix_observations):
        assert check_prefix_observations is True
        return {"traj": traj, "prefix": prefix_decision_round_count,
                "actions": dict(branch_actions)}

    def continue_rollout_from_current_state(*, env, policies, config, seed, battle_id,
                                           starting_decision_round_index):
        return types.SimpleNamespace(
            trajectory=types.SimpleNamespace(seed=seed), decision_round_count=9,
            terminal=types.SimpleNamespace(winner="p1"))

    def player_observation_history(traj, *, player_id, through_decision_round):
        return _obs_history(traj.seed, player_id, int(through_decision_round))

    monkeypatch.setattr(rvh.probe, "_top_two_and_opponent", top_two_and_opponent)
    monkeypatch.setattr(rvh.probe, "_post_branch_history", post_branch_history)
    monkeypatch.setitem(sys.modules, "pokezero.local_showdown", _module(
        "pokezero.local_showdown",
        LocalShowdownConfig=lambda **kw: types.SimpleNamespace(**kw),
        LocalShowdownEnv=lambda cfg: _FakeEnv(),
        env_config_from_checkpoint_provenance=lambda base, masks, **kw: base))
    monkeypatch.setitem(sys.modules, "pokezero.neural_policy", _module(
        "pokezero.neural_policy",
        TransformerSoftmaxPolicy=lambda **kw: types.SimpleNamespace(**kw),
        category_vocab_from_model_config=lambda cfg, root: ("VOCAB",),
        evaluate_transformer_action_priors=lambda **kw: [1.0],
        evaluate_transformer_observation_value=observation_value,
        feature_masks_from_model_config=lambda cfg: ("MASKS",),
        load_transformer_checkpoint=load_transformer_checkpoint,
        observation_spec_from_model_config=lambda cfg: ("SPEC",)))
    monkeypatch.setitem(sys.modules, "pokezero.replay_branching", _module(
        "pokezero.replay_branching", replay_trajectory_branch=replay_trajectory_branch))
    monkeypatch.setitem(sys.modules, "pokezero.rollout", _module(
        "pokezero.rollout", RolloutConfig=lambda **kw: types.SimpleNamespace(**kw),
        continue_rollout_from_current_state=continue_rollout_from_current_state))
    monkeypatch.setitem(sys.modules, "pokezero.search", _module(
        "pokezero.search", player_observation_history=player_observation_history))
    monkeypatch.setattr(sys, "argv", [
        "rescore_value_head.py", "--pairs", str(bank), "--showdown-root", str(root),
        "--out-dir", str(out_dir), "--head", f"v1={head_ckpt}", "--device", "cpu",
        *argv_extra])

    try:
        status, message = rvh.main(), ""
    except SystemExit as exc:
        status, message = (1 if exc.code else 0), ("" if exc.code is None else str(exc.code))
    return types.SimpleNamespace(status=status, message=message, out_dir=out_dir)


def _rows(result, head="v1"):
    return json.loads((result.out_dir / f"pairs-{head}.json").read_text())


def test_a_clean_replay_reproduces_the_bank_exactly_and_certifies(monkeypatch, tmp_path,
                                                                  capsys):
    """REPLAY FIDELITY, measured. The states rebuilt here are the states the bank was
    measured at, so the reproduction delta is exactly zero and the run certifies.

    This is the positive control the three refusal tests below are read against: without it, a
    tool that refused everything would pass all of them.
    """
    banked = [_banked_pair(101, 4), _banked_pair(102, 7), _banked_pair(103, 11, seat="p2")]
    got = _run_replay(monkeypatch, tmp_path, banked)
    out = capsys.readouterr().out
    assert got.status == 0, got.message
    assert "=== REPRODUCTION: 3 pairs rebuilt, 0 outside tol" in out
    assert "CERTIFIED: 3 states rebuilt and reproduced within" in out
    assert "REPRODUCTION MISS" not in out

    repro = _rows(got, "reproduce")
    assert repro["rescore"]["reproduction"]["n"] == 3
    assert repro["rescore"]["reproduction"]["max_abs_delta"] == 0.0
    assert repro["rescore"]["dropped"] == {}
    assert repro["head_name"] == rvh.REPRODUCE

    tuned = _rows(got)
    assert tuned["n_pairs"] == 3 and len(tuned["pairs"]) == 3
    assert {r["rescored_by"] for r in tuned["pairs"]} == {"v1"}
    # The ground truth is COPIED, never recomputed from the replay.
    assert [r["true_gap"] for r in tuned["pairs"]] == [b["true_gap"] for b in banked]
    assert [r["noise_var"] for r in tuned["pairs"]] == [b["noise_var"] for b in banked]


def test_a_single_perturbed_banked_head_value_refuses_to_certify(monkeypatch, tmp_path,
                                                                capsys):
    """Perturb ONE banked `head_a` by 1e-3 -- 10x the default tolerance -- and the run must
    refuse, with the reason, and with the files marked diagnosis-only.

    Deleting the refusal leaves a tool that writes a confident, incomparable beta column and
    exits 0. That is the failure this whole instrument was built to make impossible, and it
    was the failure nothing tested.
    """
    banked = [_banked_pair(101, 4), _banked_pair(102, 7), _banked_pair(103, 11)]
    banked[1]["head_a"] = banked[1]["head_a"] + 1e-3
    got = _run_replay(monkeypatch, tmp_path, banked)
    out = capsys.readouterr().out

    assert got.status == 1
    assert "REFUSING to certify" in got.message
    assert "1 pairs reproduced outside tol" in got.message
    assert "CERTIFIED" not in out
    assert "REPRODUCTION MISS" in out
    assert "=== REPRODUCTION: 3 pairs rebuilt, 1 outside tol" in out
    # Written anyway, for diagnosis: the refusal is what says not to score them.
    assert _rows(got)["rescore"]["reproduction"]["n_outside_tol"] == 1


def test_a_reproduction_miss_can_be_allowed_but_only_on_purpose(monkeypatch, tmp_path):
    """The refusal is a budget, not a constant: `--max-reproduce-failures 1` admits exactly
    the run above. Pinned so the guard cannot be satisfied by hard-refusing everything."""
    banked = [_banked_pair(101, 4), _banked_pair(102, 7)]
    banked[1]["head_b"] = banked[1]["head_b"] - 1e-3
    got = _run_replay(monkeypatch, tmp_path, banked,
                      argv_extra=("--max-reproduce-failures", "1"))
    assert got.status == 0, got.message
    assert _rows(got)["rescore"]["reproduction"]["n_outside_tol"] == 1


def test_the_replay_stamps_head_gap_through_the_producers_units_conversion(monkeypatch,
                                                                          tmp_path):
    """THE CALL SITE, not the function. The units blocker's third appearance.

    `finalize_pair_gaps` exists because pinning `head_gap_win_prob` was not enough: reverting
    the CALL SITE to `rec["head_gap"] = rec["head_a"] - rec["head_b"]` left the suite green.
    Pinning `finalize_pair_gaps` itself repeated that mistake one level up -- the function was
    exercised, and nothing asserted `main()` called it. So this asserts BOTH: that the
    producer's function was invoked during a replay, and that the emitted `head_gap` is the
    halved value rather than the raw difference it provably differs from.
    """
    calls = []
    real = rvh.probe.finalize_pair_gaps

    def spy(rec):
        stamped = real(rec)
        calls.append(dict(stamped))
        return stamped

    monkeypatch.setattr(rvh.probe, "finalize_pair_gaps", spy)
    banked = [_banked_pair(101, 4), _banked_pair(102, 7)]
    got = _run_replay(monkeypatch, tmp_path, banked)
    assert got.status == 0, got.message

    # Two heads (`__reproduce__` and `v1`) x two pairs. Zero would mean main() stopped
    # routing through the producer.
    assert len(calls) == 4

    tuned = _rows(got)["pairs"]
    assert len(tuned) == 2
    for row, want in zip(tuned, banked):
        raw = row["head_a"] - row["head_b"]
        assert raw != 0.0, "the fixture must make the two scales distinguishable"
        assert row["head_gap_return_scale"] == raw
        assert row["head_gap"] == raw / 2.0
        assert row["head_gap"] != raw
        # And recomputed for THIS head, not carried through from the bank's column.
        assert row["head_gap"] != want["head_gap"]
        assert row["head_gap"] == pytest.approx(want["head_gap"] * TUNED_SCALE, rel=1e-12)


def test_a_bank_whose_true_gap_is_not_its_own_true_a_minus_true_b_is_refused(monkeypatch,
                                                                            tmp_path):
    """`finalize_pair_gaps` recomputes `true_gap` from the COPIED `true_a`/`true_b`, so it must
    land on the banked value. If it does not, the schema is not what the tool believes and
    every ground-truth reuse is suspect -- so the run refuses rather than emitting a pairs file
    whose truth column silently came from somewhere else.
    """
    banked = [_banked_pair(101, 4, true_gap=0.5)]        # true_a - true_b is 0.125
    got = _run_replay(monkeypatch, tmp_path, banked)
    assert got.status == 1
    assert "recomputed true_gap" in got.message
    assert "not the ground truth in the file" in got.message


def test_a_pair_whose_replayed_arms_differ_from_the_bank_is_dropped_not_rescored(monkeypatch,
                                                                                tmp_path,
                                                                                capsys):
    """The middle pair CLAIMS arms (3, 5); the replay re-derives (0, 1).

    Its banked head values are the ones the replay produces, so the pair reproduces perfectly
    and nothing else in the run distinguishes it -- only the arm check does. Without that
    check the tool attaches the head's opinion of one pair of siblings to the rollouts of
    another and certifies the result.
    """
    banked = [_banked_pair(101, 4),
              _banked_pair(102, 7, banked_arms=(3, 5)),
              _banked_pair(103, 11)]
    got = _run_replay(monkeypatch, tmp_path, banked)
    out = capsys.readouterr().out
    assert got.status == 0, got.message
    assert "ARM MISMATCH replay (0,1) vs banked (3,5)" in out

    tuned = _rows(got)
    assert tuned["n_pairs"] == 2
    assert {(r["seed"], r["prefix"]) for r in tuned["pairs"]} == {(101, 4), (103, 11)}
    assert tuned["rescore"]["dropped"] == {"arm_mismatch": 1}
    assert "=== REPRODUCTION: 2 pairs rebuilt" in out


def test_a_decision_with_fewer_than_two_legal_actions_gets_its_own_counter(monkeypatch,
                                                                          tmp_path):
    """`_top_two_and_opponent` returns (None, None, 0) when there is no sibling pair to score.

    That was counted as `arm_mismatch`, conflating "the replay ranked two other moves" -- which
    a wrong `--device` causes and re-running on the bank's device fixes -- with "there is no
    second legal action here", which it does not. One counter for two causes with different
    remedies is a counter you cannot act on.
    """
    banked = [_banked_pair(101, 4), _banked_pair(102, 7)]
    got = _run_replay(monkeypatch, tmp_path, banked,
                      arms={(102, "p1", 7): (None, None, 0)})
    assert got.status == 0, got.message
    dropped = _rows(got)["rescore"]["dropped"]
    assert dropped == {"no_sibling_pair_fewer_than_two_legal_actions": 1}
    assert "arm_mismatch" not in dropped


def test_a_run_that_rebuilt_no_pair_refuses_to_certify(monkeypatch, tmp_path):
    """An empty rescore is not a clean one. Every pair dropped means nothing was verified, and
    the pairs file would still be readable by the beta instrument."""
    banked = [_banked_pair(101, 4)]
    got = _run_replay(monkeypatch, tmp_path, banked,
                      arms={(101, "p1", 4): (None, None, 0)})
    assert got.status == 1
    assert "no pair was rebuilt" in got.message
