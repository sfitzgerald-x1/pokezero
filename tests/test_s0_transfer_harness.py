import json
from types import SimpleNamespace

import numpy as np
import pytest

from pokezero import s0_transfer_recompute as scorer
from pokezero.s0_transfer_roster import (
    FP, H2H, SCHEMA, actor_bindings, cell_key, cell_record, digest, dry_run_clusters,
    expected_cells, journal_path, write_once,
)


def scratch_rows():
    banks = {"scratch": dry_run_clusters()}
    cells = expected_cells(banks, dry_run=True)
    starts = {str(c["cluster"]): {"engine_birth": [], "showdown_birth": []} for c in cells.values()}
    manifest = dict(schema=SCHEMA, dry_run=True, banks=banks, starts=starts, bootstrap_seed=917,
                    actors={"scratch": {role: dict(checkpoint_sha256="a"*64, archive_sha256="b"*64,
                                                   model_sha256="c"*64, self_moves=False) for role in ("T", "B", "C")}},
                    fp_revision="b"*40, fp_budget_ms=1000,
                    expected_cells_sha256=digest(list(cells.values())))
    results = []
    for cell in cells.values():
        # T wins only in the engine; all other primitive games end in a tie.
        winner = cell["seat"] if cell["environment"] == "engine" and cell["role"] == "T" else 2
        witness = dict(status="COMPLETE", winner=winner, players=["p0", "p1"], marker="|tie",
                       birth=[], seed=cell["chance"], fp_seed=cell["action"])
        if cell["assay"] == FP:
            witness["source"] = {"terminal_witness": dict(
                status="PUBLIC_TIE", capped=False, public_terminal_line="|tie",
                players={"p1": "p0", "p2": "p1"}, winner_player=None,
            )}
        score = .5 if winner == 2 else 1.0
        results.append(dict(cell, manifest_sha256=digest(manifest), start_sha256=digest(starts[str(cell["cluster"])]),
                            actor_bindings=actor_bindings(manifest, cell), status="COMPLETE", evidence=witness,
                            evidence_sha256=digest(witness), score=score))
    return manifest, results


def test_banks_preserve_physical_seats_and_explicit_rng_domains():
    banks = {"scratch": dry_run_clusters()}
    roster = expected_cells(banks, dry_run=True)
    assert len(roster) == 280
    c = banks["scratch"][H2H][0]
    engine = [r for r in roster.values() if r["environment"] == "engine" and r["cluster"] == c]
    showdown = [r for r in roster.values() if r["environment"] == "showdown" and r["cluster"] == c]
    assert {(r["role"], r["seat"]) for r in engine} == {("T", 0), ("T", 1), ("C", 0), ("C", 1)}
    assert len({(r["action"], r["chance"]) for r in engine}) == 1
    assert len({engine[0]["action"], engine[0]["chance"], showdown[0]["action"], showdown[0]["chance"]}) == 4
    assert not any(r["environment"] == "engine" for r in roster.values() if r["assay"] == FP)
    with pytest.raises(ValueError, match="scratch band"):
        expected_cells({"scratch": {H2H: [1600000000], FP: [1500000000]}}, dry_run=True)


def test_scratch_labels_bind_public_source_without_a_training_plan():
    from pokezero.randbat import Gen3RandbatSource, Gen3RandbatSpeciesUniverse, Gen3RandbatVariant, RandbatSourceMetadata
    from pokezero.s0_transfer_harness import scratch_label_vocabulary
    from pokezero.v5_contract import source_variant_census

    binding = "f5a5265143d423af"
    metadata = RandbatSourceMetadata("gen3randombattle", 3, None, None, None, binding)
    variant = Gen3RandbatVariant("variant-1", "set", "Swalot", "role", 80,
                               ("Hidden Power Ice", "Hidden Power Fire", "Toxic"), "Sticky Hold", "Leftovers")
    source = Gen3RandbatSource(metadata, {"swalot": Gen3RandbatSpeciesUniverse("Swalot", 80, (variant,))})
    labels = scratch_label_vocabulary(source, binding)
    assert labels.moves == ("hiddenpower", "struggle", "toxic")
    assert labels.species_variants == (("swalot", 1),)
    assert labels.source_binding_sha256 == source_variant_census(source)["source_rows_sha256"]
    with pytest.raises(ValueError, match="source hash differs"):
        scratch_label_vocabulary(source, "b" * 16)


def test_recompute_retains_unknown_cells_and_does_not_issue_dry_verdict():
    manifest, rows = scratch_rows()
    report = scorer.recompute(manifest, rows)
    assert report["primitive_cells"] == report["complete_cells"] == 280
    assert report["verdict"] is None and report["status"] == "UNSCORED"
    assert report["contrasts"]["G"]["identification_bounds_pp"] == [50.0, 50.0]
    assert report["contrasts"]["J"]["identification_bounds_pp"] == [50.0, 50.0]
    missing = next(r for r in rows if r["role"] == "T" and r["environment"] == "engine")
    missing.update(status="CAPPED", evidence={"reason": "request cap"}, score=None)
    missing["evidence_sha256"] = digest(missing["evidence"])
    report = scorer.recompute(manifest, rows)
    assert report["contrasts"]["G"]["identification_bounds_pp"] == [47.5, 50.0]
    assert report["contrasts"]["G"]["by_replica"]["scratch"]["known_clusters"] == 19
    assert report["contrasts"]["Q_H2H"]["by_replica"]["scratch"]["known_clusters"] == 20
    with pytest.raises(ValueError, match="missing expected cell"):
        scorer.recompute(manifest, rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        scorer.recompute(manifest, rows + [rows[-1]])


def test_recompute_rejects_reattributed_score_and_changed_seal():
    manifest, rows = scratch_rows()
    rows[0]["score"] = .25
    with pytest.raises(ValueError, match="producer score"):
        scorer.collect_scores(manifest, rows)
    rows[0]["evidence"]["winner"] = 0
    with pytest.raises(ValueError, match="terminal evidence changed"):
        scorer.collect_scores(manifest, rows)


def test_rejection_boundaries_are_strict_and_directional():
    assert scorer.rejection("G", np.array([[3.0, 3.0], [4., 4.]])) == (False, False)
    assert scorer.rejection("G", np.array([[2.0, 3.1], [3.1, 4.]])) == (False, True)
    assert scorer.rejection("D", np.array([[-4., -3.], [-3.2, -2.1]])) == (True, True)
    assert scorer.rejection("D", np.array([[20., 20.], [20., 20.]])) == (False, False)


def test_immutable_artifact_refuses_changed_publication(tmp_path):
    dest = tmp_path / "receipt.json"
    assert write_once(dest, {"status": "finished"}) == write_once(dest, {"status": "finished"})
    with pytest.raises(ValueError, match="immutable"):
        write_once(dest, {"status": "failed"})
    assert json.loads(dest.read_bytes()) == {"status": "finished"}


def test_supervisor_seals_finished_showdown_and_fp_rows_after_worker_timeout(tmp_path):
    from pokezero.s0_transfer_supervise import seal_side

    manifest, rows = scratch_rows()
    write_once(tmp_path / "manifest.json", manifest)
    retained = [next(row for row in rows if row["environment"] == "showdown" and row["assay"] == assay)
                for assay in (H2H, FP)]
    for row in retained:
        write_once(journal_path(tmp_path, row), row)
    assert not seal_side(tmp_path, "showdown", 124, 5400)
    terminal = json.loads((tmp_path / "showdown.terminal.json").read_bytes())
    assert terminal["status"] == "PARTIAL" and terminal["worker_exit_code"] == 124
    assert terminal["status_counts"] == {"COMPLETE": 2, "FAILED": 198}
    archived = json.loads((tmp_path / "showdown.cells.json").read_bytes())
    by_key = {cell_key(r): r for r in archived}
    assert [by_key[cell_key(row)] for row in retained] == retained
    scores = scorer.collect_scores(manifest, archived + [r for r in rows if r["environment"] == "engine"])
    assert scores[cell_key(retained[0])] == scores[cell_key(retained[1])] == .5
    assert len(scores) == 280 and sum(score is None for score in scores.values()) == 198


def test_resume_preserves_existing_cells_and_reopens_only_deadline_placeholders(tmp_path):
    from pokezero.s0_transfer_harness import resume_scratch

    manifest, rows = scratch_rows()
    prior, target = tmp_path / "r2", tmp_path / "r3"
    engine = [row for row in rows if row["environment"] == "engine"]
    showdown = [row for row in rows if row["environment"] == "showdown"]
    expected = expected_cells(manifest["banks"], dry_run=True)
    dropped, actual_failure = (expected[cell_key(row)] for row in showdown[:2])
    showdown[0] = cell_record(manifest, dropped, "FAILED", None,
                             {"reason": "worker exit 124 before a durable cell result"})
    showdown[1] = cell_record(manifest, actual_failure, "FAILED", None, {"reason": "actual worker exception"})
    manifest_sha = write_once(prior / "manifest.json", manifest)
    engine_sha = write_once(prior / "engine.cells.json", engine)
    showdown_sha = write_once(prior / "showdown.cells.json", showdown)

    with pytest.raises(ValueError, match="previous scratch artifact changed"):
        resume_scratch(target, prior, manifest_sha=manifest_sha, engine_sha=engine_sha, showdown_sha="0"*64)
    assert not target.exists()
    resume_scratch(target, prior, manifest_sha=manifest_sha, engine_sha=engine_sha, showdown_sha=showdown_sha)
    assert (target / "engine.cells.json").read_bytes() == (prior / "engine.cells.json").read_bytes()
    assert not journal_path(target, dropped).exists()
    assert json.loads(journal_path(target, actual_failure).read_bytes()) == showdown[1]
    assert json.loads(journal_path(target, showdown[2]).read_bytes()) == showdown[2]
    receipt = json.loads((target / "showdown.resume.json").read_bytes())
    assert receipt["retained_showdown_cells"] == 199 and receipt["pending_showdown_cells"] == 1


def test_engine_batch_deadline_preserves_completed_book_entries():
    pytest.importorskip("jax")
    from gpu_engine import s0_transfer_engine, v2_collector, v5_device_rollout

    class Collector:
        def run(self, params, *, observe_book):
            book = SimpleNamespace(
                status=np.array([v5_device_rollout.COMPLETE, v2_collector.NOT_ENDED]),
                winner=np.array([1, -1]), terminal_frame=np.array([4, 0]),
                terminal_hp=np.zeros((2, 2, 6), dtype=int),
                terminal_max_hp=np.ones((2, 2, 6), dtype=int),
            )
            assert observe_book(book)

    cells = [dict(cluster=91, seat=1), dict(cluster=92, seat=1)]
    finished = []
    s0_transfer_engine.play_batch(
        Collector(), (), SimpleNamespace(streams=[91, 92]), cells, ["birth-91", "birth-92"],
        publish=finished.append, deadline=0,
    )
    assert [(cell["cluster"], status, score) for cell, status, score, _ in finished] == [
        (91, "COMPLETE", 1.), (92, "FAILED", None),
    ]


def test_evaluation_draw_uses_exact_engine_key_and_masked_law():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from gpu_engine.v2_collector import sample_keys
    from gpu_engine.v5_evaluation_sampling import EvaluationDraw

    logits = jnp.array([.2, 9, .2, -1, 2, 3, -.4, -.5, .4], dtype=jnp.float32)
    legal = jnp.array([True, False, True, False, False, True, False, True, False])
    for frame in (0, 1, 81):
        for seat in (0, 1):
            keys = sample_keys(jnp.array([2080000000]), jnp.array([frame]), jnp.array([2**32 - 1], jnp.uint32))
            masked = jnp.where(legal, logits, -jnp.inf)
            expected = int(jax.random.categorical(keys[0, seat], masked))
            actual, p = EvaluationDraw(2**32 - 1, 2080000000, frame, seat).choose(logits, legal)
            assert actual == expected
            np.testing.assert_allclose(p, jax.nn.softmax(masked), rtol=0, atol=0)
    with pytest.raises(ValueError, match="nonempty"):
        EvaluationDraw(1, 2, 3, 0).choose(logits, np.zeros(9, dtype=bool))
    with pytest.raises(ValueError, match="finite"):
        EvaluationDraw(1, 2, 3, 0).choose(logits.at[1].set(jnp.nan), legal)


def test_control_lifecycle_verifies_separate_reload_and_detects_changed_receipt(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    from pokezero import v5_training
    from pokezero.s0_transfer_control import shadow, write_actor
    from pokezero.s0_transfer_harness import fixed_fixture
    from pokezero.v5_loss import V5LossConfig
    from pokezero.v5_model import V5ModelConfig
    from test_v5_contract import vocabulary

    model = V5ModelConfig(128, "b"*64, vocabulary(), width=8, layers=1, attention_heads=2,
                          expert_width=8, value_hidden=(8, 8), value_bins=5)
    initial = v5_training.create(model, V5LossConfig(), v5_training.TrainConfig(71), enable_v5=True)
    baseline = write_actor(tmp_path / "B", initial)
    control, receipt = shadow(tmp_path / "C", baseline, [], dry_run=True, fixture=fixed_fixture)
    assert baseline["tensor_sha256"] == control["tensor_sha256"]
    assert baseline["archive"] != control["archive"]
    manifest = dict(dry_run=True, banks={"scratch": {}}, actors={"scratch": {"B": baseline, "C": control}},
                    control={"scratch": receipt})
    scorer.verify_control(manifest)
    (tmp_path / "C" / "001" / "cycle.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        scorer.verify_control(manifest)
