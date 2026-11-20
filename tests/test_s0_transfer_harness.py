import json
from types import SimpleNamespace

import numpy as np
import pytest

from pokezero import s0_transfer_recompute as scorer
from pokezero.s0_transfer_roster import (
    FP, H2H, SCHEMA, canonical, cell_key, dry_run_clusters, expected_cells, read_terminals,
)


def scratch_rows():
    banks = {"scratch": dry_run_clusters()}
    cells = expected_cells(banks, dry_run=True)
    starts = {str(c["cluster"]): {"engine_birth": [], "showdown_birth": []} for c in cells.values()}
    manifest = dict(schema=SCHEMA, dry_run=True, banks=banks, starts=starts, bootstrap_seed=917)
    results = []
    for cell in cells.values():
        winner = cell["seat"] if cell["environment"] == "engine" and cell["role"] == "T" else 2
        witness = dict(status="COMPLETE", winner=winner, players=["p0", "p1"], marker="|tie",
                       birth=[], seed=cell["chance"], fp_seed=cell["action"])
        results.append(dict(cell, status="COMPLETE", evidence=witness))
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
    assert report["contrasts"]["G"]["gap_pp"] == 50.0
    assert report["contrasts"]["G"]["paired_ci90_pp"] == [50.0, 50.0]
    assert report["contrasts"]["G"]["threshold_pp"] == 3
    assert not report["contrasts"]["G"]["bounded_ci90_within_threshold"]
    missing = next(r for r in rows if r["role"] == "T" and r["environment"] == "engine")
    missing.update(status="CAPPED", evidence={"reason": "request cap"})
    report = scorer.recompute(manifest, rows)
    assert report["contrasts"]["G"]["identification_bounds_pp"] == [47.5, 50.0]
    assert report["contrasts"]["G"]["by_replica"]["scratch"]["known_clusters"] == 19
    assert report["contrasts"]["Q_H2H"]["by_replica"]["scratch"]["known_clusters"] == 20
    bounded = report["contrasts"]["G"]["missingness_bounded_ci90_pp"]
    assert bounded[0] < 50.0 == bounded[1]
    assert scorer.recompute(manifest, rows[:-1])["missing_cells"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        scorer.recompute(manifest, rows + [rows[-1]])


def test_recompute_uses_literal_winner_and_rejects_wrong_seat_or_roster():
    manifest, rows = scratch_rows()
    cell = next(r for r in rows if r["environment"] == "showdown" and r["assay"] == H2H and r["seat"] == 1)
    cell["evidence"]["marker"] = "|win|p1"
    assert scorer.collect_scores(manifest, rows)[cell_key(cell)] == 1.
    cell["evidence"]["marker"] = "|win|unrecognized"
    with pytest.raises(ValueError, match="literal Showdown terminal"):
        scorer.collect_scores(manifest, rows)
    cell["evidence"].update(marker="|tie", birth=["changed"])
    with pytest.raises(ValueError, match="declared physical start"):
        scorer.collect_scores(manifest, rows)
    cell["chance"] += 1
    with pytest.raises(ValueError, match="changed roster cell"):
        scorer.collect_scores(manifest, rows)


def test_rejection_boundaries_are_strict_and_directional():
    assert scorer.rejection("G", np.array([[3.0, 3.0], [4., 4.]])) == (False, False)
    assert scorer.rejection("G", np.array([[2.0, 3.1], [3.1, 4.]])) == (False, True)
    assert scorer.rejection("D", np.array([[-4., -3.], [-3.2, -2.1]])) == (True, True)
    assert scorer.rejection("D", np.array([[20., 20.], [20., 20.]])) == (False, False)


def test_interrupted_worker_retains_raw_terminals(tmp_path):
    manifest, rows = scratch_rows()
    raw = tmp_path / "showdown.jsonl"
    completed = [r for r in rows if r["environment"] == "showdown"][:4]
    raw.write_bytes(b"".join(canonical(row) + b"\n" for row in completed) + b'{"incomplete":')
    assert list(read_terminals(raw)) == completed
    score = scorer.recompute(manifest, read_terminals(raw))
    assert score["dispositions"] == {"COMPLETE": 4, "NOT_ATTEMPTED": 276}
    assert score["missing_cells"] == 276


def test_engine_publishes_terminal_lanes_incrementally():
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
            observe_book(book)
            assert len(finished) == 1
            book.status[1] = v5_device_rollout.CAPPED
            observe_book(book)

    cells = [dict(cluster=91, seat=1), dict(cluster=92, seat=1)]
    finished = []
    s0_transfer_engine.play_batch(
        Collector(), (), SimpleNamespace(streams=[91, 92]), cells, ["birth-91", "birth-92"],
        publish=finished.append,
    )
    assert [(cell["cluster"], status) for cell, status, _ in finished] == [(91, "COMPLETE"), (92, "CAPPED")]


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


def test_control_reloads_final_actor_with_the_same_policy(tmp_path):
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
    control = shadow(tmp_path / "C", baseline, 2, fixture=fixed_fixture)
    assert baseline["tensor_sha256"] == control["tensor_sha256"]
    assert baseline["archive"] != control["archive"]
    assert set(path.name for path in (tmp_path / "C").iterdir()) == {"checkpoint.pt", "actor.npz", "checkpoint.pt.encoding.json"}


def test_sampled_showdown_keeps_playing_until_its_terminal(monkeypatch):
    from gpu_engine import v5_live_protocol
    from pokezero import s0_transfer_showdown

    class Actor:
        generation = 0
        prefix_sha256 = "initial"

        def begin_game(self, *, owner, **_):
            self.owner = owner

        def accept_owner_stream(self, **_):
            pass

        def snapshot_boundary(self):
            return None

        def decide(self, boundary, *, rng):
            return SimpleNamespace(action_index=0)

        def choice(self, boundary):
            return "move 1"

        def close_game(self):
            return {"status": "COMPLETE"}

    class Factory:
        deterministic = False
        compute_dtype = "bfloat16"
        config = "same v5 config"

        def create_actor(self):
            return Actor()

    class Env:
        def __init__(self, config):
            self.owner_lines = {
                seat: ["|player|p1|alice", "|player|p2|bob"] for seat in ("p1", "p2")
            }
            self.frames = 0

        def reset_with_start_override(self, **_):
            pass

        def snapshot(self):
            return SimpleNamespace(bridge_snapshot="physical start")

        def terminal(self):
            return SimpleNamespace(capped=False) if self.frames == 210 else None

        def requested_players(self):
            return ("p1", "p2")

        def _choices_for_actions(self, actions):
            return {seat: "move 1" for seat in actions}

        def step(self, actions):
            self.frames += 1
            if self.terminal():
                for lines in self.owner_lines.values():
                    lines.append("|win|bob")

        def close(self):
            pass

    monkeypatch.setattr(v5_live_protocol, "check_factory", lambda _: None)
    monkeypatch.setattr(s0_transfer_showdown, "OwningEnv", Env)
    monkeypatch.setattr(s0_transfer_showdown, "_terminal_winner", lambda *_: 1)

    cell = dict(role="T", cluster=10, chance=11, action=12, seat=1)
    status, terminal = s0_transfer_showdown.evaluate_game(
        None, cell, factories=(Factory(), Factory()), packed_start=None,
        check_start=lambda start: start, max_requests=250,
    )
    assert status == "COMPLETE"
    assert terminal["frames"] == 210
    assert scorer.terminal_score(dict(cell, environment="showdown", status=status, evidence=terminal)) == 1.
