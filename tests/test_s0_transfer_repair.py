"""Focused deferred route tests. Executing this file exercises a full synthetic look."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets
import tempfile
import time

import numpy as np
import pytest

from pokezero import s0_transfer_recompute as scorer
from pokezero import v5_deferred_evaluation as deferred, v5_evaluation as ev, v5_run as run
from pokezero.s0_transfer_harness import fused_snapshots
from pokezero.s0_transfer_roster import (
    FP, H2H, SCHEMA, Shard, canonical, cell_key, digest, expected_cells, read_terminals,
    terminal_writer, write_json,
)


def fake_fused(root, replica, policy, key, *, mount_alias=False):
    root.mkdir()
    model = {"replicate": replica}
    declarations = [[{"games": [0] * 20_000}]] * 50
    declaration = dict(model=model, evaluation=policy.metadata(), iterations=declarations)
    write_json(root / "declaration.json", declaration)
    trainer = deferred.MilestoneDispatch(root, policy, {49: 1_000_000}, digest(declaration), stop_key=key)
    snapshots = []
    for i in range(-1, 50):
        prefix = f"artifacts/{i}"
        checkpoint, archive = prefix + ".pt", prefix + ".npz"
        snapshots.append(run.Snapshot(i, checkpoint, digest([replica, checkpoint]), archive,
                                      digest([replica, archive]), digest(model)))
    t = snapshots[-1]
    (root / "artifacts").mkdir()
    for relative in (t.checkpoint, t.archive):
        (root / relative).write_bytes(canonical([replica, relative]))
    state = dict(snapshots=list(map(asdict, snapshots)), declaration_sha256=digest(declaration),
                 checkpoint_encoding_receipts={}, reports=[dict(iteration=i, games=20_000) for i in range(50)])
    write_json(root / "state.json", state)
    if mount_alias:
        # Both trainers see their respective volume at /out. Write those wire
        # bytes once at the mount used by the evaluator.
        alias = Path("/out")
        request = run.EvaluationRequest(1_000_000, digest(declaration), policy.protocol_sha256,
                                        t, alias, alias / t.checkpoint, alias / t.archive)
        record = deferred._milestone_document(request, policy.route_declaration_sha256, None)
        ref = deferred._publish_equal(trainer.path(request.declared_games), record)
    else:
        receipt = trainer.publish_committed(t, report_error=lambda reason: (_ for _ in ()).throw(ValueError(reason)))
        ref = ev.Artifact(str(root / receipt.request), receipt.request_sha256)
    entry = dict(dispatch=asdict(ref), run=str(root))
    _, _, b, trained, _ = fused_snapshots(entry, 50, policy.route_declaration_sha256)
    assert b == snapshots[0] and trained == t
    return trainer, ref, asdict(t)


def synthetic(root, n, key, *, mount_alias=False, second_key=None):
    route = digest("synthetic route")
    policy = run.EvaluationProtocol(digest("synthetic protocol"), 1_000_000, "deferred", route)
    banks, starts, actors, refs, trainers = {}, {}, {}, {}, []
    for i in range(2):
        replica = str(i)
        trainer, ref, actor = fake_fused(root / replica, i, policy, second_key if i and second_key else key,
                                        mount_alias=mount_alias)
        refs[replica], actors[replica] = asdict(ref), {"T": actor}
        trainers.append(trainer)
        banks[replica] = {}
        for j, assay in enumerate((FP, H2H)):
            start = 100_000 + 20_000*i + 10_000*j
            bank = list(range(start, start+n))
            banks[replica][assay] = bank
            starts.update({str(c): dict(engine_birth=c, showdown_birth=c) for c in bank})
    manifest = dict(schema=SCHEMA, dry_run=False, banks=banks, bootstrap_seed=17,
                    actors=actors, dispatches=refs, route_declaration_sha256=route, starts=starts,
                    config=dict(control_cycles=50))
    write_json(root / "manifest.json", manifest)
    return manifest, trainers


def raw(cell, *, failure=True):
    # Vary individual outcomes across rosters. A large, prearranged E/S gap exercises FAIL.
    win = cell["cluster"] % 3 == 0
    if failure and cell["role"] == "T" and cell["environment"] == "engine":
        win = True
    winner = cell["seat"] if win else 1 - cell["seat"]
    evidence = dict(birth=cell["cluster"])
    if cell["environment"] == "engine":
        evidence.update(winner=winner, status="COMPLETE")
    else:
        evidence.update(marker=f"|win|p{winner}", players=["p0", "p1"],
                        seed=cell["chance"], fp_seed=cell["action"])
    return cell, "COMPLETE", evidence


def synthetic_panel(root, n, key, *, leave_one=False, **fixture):
    manifest, trainers = synthetic(root, n, key, **fixture)
    roster = expected_cells(manifest["banks"], dry_run=False)
    for side, shard in (("engine", Shard()), ("showdown", Shard(0, 2)), ("showdown", Shard(1, 2))):
        expected = {k: c for k, c in roster.items() if c["environment"] == side and shard.owns(c)}
        for resume in (False, True):
            with terminal_writer(root, side, shard, expected, digest(manifest), resume=resume) as (cells, publish):
                if resume and leave_one and side == "engine":
                    cells = cells[:-1]
                for cell in cells if resume else cells[:3]:
                    publish(raw(cell))
    del roster
    started = time.monotonic()
    report = scorer.score_directory(root, publish_stop=True)
    assert report["verdict"] == "FAIL" and report["missing_cells"] == report["invalid_rows"] == 0
    assert report["complete_cells"] == 32*n
    assert all(t.stop_at_boundary(50)["verdict"] == "FAIL" for t in trainers)
    return dict(cells=report["complete_cells"], replicas=2, clusters_per_assay_replica=n,
                draws=report["resampler"]["draws"], verdict=report["verdict"], trainer_stops=2,
                score_and_publish_seconds=round(time.monotonic()-started, 3))


def test_joint_bootstrap_preserves_paired_indices():
    matrices = [np.random.default_rng(i).integers(-2, 3, size=(97, 4)) / 2 for i in (11, 42)]
    joint = scorer.joint_bootstrap(matrices, 2111, 177, .01)
    for i in (0, 2):
        old = scorer.bootstrap([(v[:, i], v[:, i+1]) for v in matrices], draws=2111, seed=177, quantile=.01)
        np.testing.assert_array_equal(joint[:, i:i+2], old)


def test_sharded_fused_stop(tmp_path, monkeypatch):
    key = secrets.token_bytes(32)
    monkeypatch.setenv(deferred.STOP_KEY_ENV, key.hex())
    monkeypatch.setattr(scorer, "CLUSTERS_PER_REPLICA", 20)
    monkeypatch.setattr(scorer, "REGISTERED_DRAWS", 256)
    assert synthetic_panel(tmp_path, 20, key)["trainer_stops"] == 2
    manifest = json.loads((tmp_path/"manifest.json").read_bytes())
    ref = ev.Artifact(**manifest["dispatches"]["0"])
    path = tmp_path / "route-verdict.json"
    with pytest.raises(ValueError, match="signer differs"):
        deferred.publish_route_failure(ref, ev.Artifact(str(path), run._file_sha(path)), key=secrets.token_bytes(32))
    state_path = tmp_path/"0/state.json"
    state = json.loads(state_path.read_bytes())
    state["reports"][-1]["games"] = 19_999
    write_json(state_path, state)
    with pytest.raises(ValueError, match="committed fused milestone"):
        fused_snapshots(dict(run=str(tmp_path/"0"), dispatch=asdict(ref)), 50, manifest["route_declaration_sha256"])


def test_mount_aliases_and_two_signer_preflight(tmp_path, monkeypatch):
    key = secrets.token_bytes(32)
    monkeypatch.setenv(deferred.STOP_KEY_ENV, key.hex())
    monkeypatch.setattr(scorer, "CLUSTERS_PER_REPLICA", 20)
    monkeypatch.setattr(scorer, "REGISTERED_DRAWS", 256)
    with pytest.raises(ValueError, match="signer differs"):
        synthetic_panel(tmp_path, 20, key, mount_alias=True, second_key=secrets.token_bytes(32))
    assert not any((tmp_path / str(i) / "evaluation-stop.json").exists() for i in range(2))
    valid = tmp_path / "valid"
    valid.mkdir()
    assert synthetic_panel(valid, 20, key, mount_alias=True)["trainer_stops"] == 2


def test_stop_delivery_recovers_original_decision_after_resume(tmp_path, monkeypatch):
    key = secrets.token_bytes(32)
    monkeypatch.setenv(deferred.STOP_KEY_ENV, key.hex())
    monkeypatch.setattr(scorer, "CLUSTERS_PER_REPLICA", 20)
    monkeypatch.setattr(scorer, "REGISTERED_DRAWS", 256)
    original = deferred._publish_equal
    def interrupt(path, value):
        if path == tmp_path / "1/evaluation-stop.json":
            raise OSError("interrupted second write")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(deferred, "_publish_equal", interrupt)
        with pytest.raises(OSError, match="interrupted second write"):
            synthetic_panel(tmp_path, 20, key, leave_one=True)
    first = json.loads((tmp_path / "0/evaluation-stop.json").read_bytes())
    sha = first["payload"]["independent_verdict_sha256"]
    assert not (tmp_path / "1/evaluation-stop.json").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    expected = {k: c for k, c in expected_cells(manifest["banks"], dry_run=False).items()
                if c["environment"] == "engine"}
    with terminal_writer(tmp_path, "engine", Shard(), expected, digest(manifest), resume=True) as (cells, publish):
        assert len(cells) == 1
        publish(raw(cells[0]))
    assert scorer.score_directory(tmp_path)["missing_cells"] == 0
    assert run._file_sha(tmp_path / "route-verdict.json") != sha

    assert scorer.score_directory(tmp_path, publish_stop=True)["missing_cells"] == 1
    second = json.loads((tmp_path / "1/evaluation-stop.json").read_bytes())
    assert second["payload"]["independent_verdict_sha256"] == run._file_sha(tmp_path / "route-verdict.json") == sha


def test_verdict_floor_invalid_rows_and_boundaries(tmp_path, monkeypatch):
    n = 10
    monkeypatch.setattr(scorer, "CLUSTERS_PER_REPLICA", n)
    monkeypatch.setattr(scorer, "REGISTERED_DRAWS", 256)
    manifest, _ = synthetic(tmp_path, n, secrets.token_bytes(32))
    rows = [dict(c, status=status, evidence=witness) for c, status, witness in
            (raw(c, failure=False) for c in expected_cells(manifest["banks"], dry_run=False).values())]
    assert scorer.recompute(manifest, rows)["verdict"] == "NOT-FAILED"
    bad = [r for r in rows if r["replica"] == "0" and r["role"] == "T" and r["assay"] == FP and r["seat"] == 0]
    bad[0]["evidence"].pop("players")
    report = scorer.recompute(manifest, rows)
    assert report["invalid_rows"] == report["missing_cells"] == 1 and report["known_floor"]
    report = scorer.recompute(manifest, [*rows, None])
    assert report["integrity_errors"] == 0 and report["invalid_rows"] == 2 and report["known_floor"]
    bad[1]["evidence"]["marker"] = []
    report = scorer.recompute(manifest, [*rows, None])
    assert not report["known_floor"] and report["invalid_rows"] == 3 and report["verdict"] == "VOID"


def test_unbound_authority_is_explicit(tmp_path):
    policy = run.EvaluationProtocol(digest("generic deferred panel"), 20_000, "deferred")
    deferred.MilestoneDispatch(tmp_path, policy, {}, digest("plan"))
    assert json.loads((tmp_path/"route-stop-authority.json").read_bytes()) == dict(
        enabled=False, route_declaration_sha256=None, key_fingerprint=None)


if __name__ == "__main__":
    key = secrets.token_bytes(32)
    os.environ[deferred.STOP_KEY_ENV] = key.hex()
    with tempfile.TemporaryDirectory() as temporary:
        report = synthetic_panel(Path(temporary), scorer.CLUSTERS_PER_REPLICA, key)
    print(json.dumps(report), flush=True)
