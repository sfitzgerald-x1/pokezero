import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import os
import unittest
from unittest.mock import patch

import numpy as np

from pokezero.category_vocab import CategoryVocabulary
from pokezero.mirror_value_probe import (CHAMPION_SHA, cohort, collect_battle, digest,
    file_sha, paired_battle_interval, swap_own_view, terminal_labels, write_new)
from pokezero.observation import ObservationPerspective, PokeZeroObservationV0
from pokezero.showdown import CATEGORY_ROLE


def pool_fixture_initialize(*arguments):
    pass


def pool_fixture_worker(arguments):
    row, output, _ = arguments
    if row["index"] == 0:
        (Path(output) / "first-failure.txt").write_text("synthetic worker failure")
        raise RuntimeError("synthetic first-game failure")
    # Queued work must be terminated before this side effect, rather than
    # draining the remaining scientific roster after a failed first game.
    time.sleep(10)
    (Path(output) / ("completed-" + str(row["index"]))).write_text("unexpected")
    return {}


def pool_fixture_raising_runtime(checkpoint, *arguments):
    marker = Path(checkpoint) / ("init-" + str(os.getpid()))
    with marker.open("x") as stream:
        stream.write("one failed initialization")
    raise ValueError("synthetic checkpoint binding failure")


def pool_fixture_failed_initialize(*arguments):
    import run_mirror_value_probe as runner
    with patch.object(runner, "load_runtime", pool_fixture_raising_runtime):
        runner.initialize(*arguments)


def observation(seat, marker):
    cat = np.zeros((23, 41), np.int64)
    cat[1:7, CATEGORY_ROLE] = 1
    cat[7:13, CATEGORY_ROLE] = 2
    return PokeZeroObservationV0(tuple(tuple(map(int, r)) for r in cat), tuple(tuple(map(float, r)) for r in np.full((23, 132), marker, np.float32)),
        tuple([0] + [1] * 6 + [2] * 6 + [3] * 9 + [5]), tuple([True] * 23),
        tuple([True] * 9), perspective=ObservationPerspective.from_showdown_slot(seat, seat),
        schema_version="pokezero.observation.v4", metadata={"turn_number": 1})


class MirrorContractTests(unittest.TestCase):
    def test_champion_warmstart_preserves_exact_masked_mean_predictions(self):
        import torch
        from types import SimpleNamespace
        from pokezero.mirror_value_probe import champion_linear_head
        from pokezero.neural_policy import _masked_mean
        torch.manual_seed(19)
        model = SimpleNamespace(value_head=torch.nn.Linear(512, 1),
            config=SimpleNamespace(value_activation="tanh"))
        tokens = torch.randn(17, 23, 512)
        valid = torch.rand(17, 23) > .2
        head = champion_linear_head(model)
        expected = torch.tanh(model.value_head(_masked_mean(tokens, valid)).squeeze(-1))
        self.assertTrue(torch.equal(head(tokens, valid)[0], expected))
        self.assertEqual(sum(p.numel() for p in head.parameters()), 513)
        self.assertTrue(torch.equal(head.network.weight, model.value_head.weight))
        self.assertTrue(torch.equal(head.network.bias, model.value_head.bias))

    def test_true_moves_canonical_pp_pressure_and_transform_scope(self):
        from types import SimpleNamespace as NS
        from pokezero.mirror_value_probe import true_move_facts
        initial = {"side": {"pokemon": [dict(ident="p2: " + str(i), moves=["transform"] if i == 0 else
            ["hiddenpowerice", "return102"] if i == 1 else ["surf"]) for i in range(6)]}}
        current = json.loads(json.dumps(initial["side"]["pokemon"]))
        current[0].update(active=True, moves=["surf", "icebeam"])
        team = [NS(ident=r["ident"], species="species" + str(i)) for i, r in enumerate(current)]
        ledger = [NS(species="species" + str(i), transformed=(i == 0), move_uses=
            (("hiddenpowerice", 2), ("return", 3)) if i == 1 else (("transform", 1),) if i == 0 else ()) for i in range(6)]
        request = dict(side=dict(pokemon=current), active=[dict(moves=[dict(id="surf", pp=3, maxpp=5), dict(id="icebeam", pp=4, maxpp=5)])])
        state = NS(request=request, self_team=team, belief_view=NS(self_pokemon=ledger))
        dex = NS(move_info=lambda _: NS(max_pp=16))
        facts = true_move_facts(state, initial, dex)
        self.assertEqual(facts[0], dict(icebeam=.8, surf=.6))
        self.assertEqual(facts[1], dict(hiddenpowerice=14/16, **{"return": 13/16}))
        # A copied set in an old side snapshot cannot survive switch-out.
        current[0]["active"] = False
        facts = true_move_facts(state, initial, dex)
        self.assertEqual(facts[0], dict(transform=15/16))
        # Current active Hidden Power base id maps back to its typed true set.
        current[1]["active"] = True
        state.request["active"] = [dict(moves=[dict(id="hiddenpower", pp=7, maxpp=24), dict(id="return", pp=2, maxpp=32)])]
        facts = true_move_facts(state, initial, dex)
        self.assertEqual(facts[1], dict(hiddenpowerice=7/24, **{"return": 2/32}))

    def test_actual_owning_wire_fold_has_no_other_request_or_omniscient_hp(self):
        from pokezero.local_showdown import LocalShowdownConfig
        from pokezero.mirror_value_probe import make_owning_wire_env
        env = make_owning_wire_env(LocalShowdownConfig(set_belief_source=False))
        env._battle_token = "b1"
        try:
            env._apply_event(dict(type="stream", battleId="b1", stream="omniscient", lines=["|-damage|p2a: Test|277/333"]))
            for seat in ("p1", "p2"):
                request = "|request|" + json.dumps(dict(side=dict(id=seat, pokemon=[])))
                env._apply_event(dict(type="stream", battleId="b1", stream=seat, lines=["", "|turn|1", request]))
            for seat, observer in env.public_observers.items():
                self.assertEqual(observer._lines, env.owning_lines[seat])
                self.assertEqual(set(observer._latest_requests), {seat})
                self.assertFalse(any("277/333" in line for line in observer._lines))
                self.assertEqual(observer._parser.hp_visibility[seat], "exact")
                self.assertEqual(observer._parser.hp_visibility["p2" if seat == "p1" else "p1"], "percentage")
        finally:
            env.close()

    def test_failed_initializer_surfaces_once_per_worker_without_pool_restarts(self):
        scripts = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import run_mirror_value_probe as runner
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "initialize", pool_fixture_failed_initialize):
            tmp = Path(tmp)
            config = dict(root="synthetic-initializer-failure", battles=8, workers=2, threads=1,
                checkpoint=str(tmp), showdown="fixture", node="fixture", max_rounds=1)
            dest = tmp / "collection"
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "mirror runtime initialization failed: ValueError: synthetic checkpoint binding failure"):
                runner.collect(config, dest)
            self.assertLess(time.monotonic() - started, 8)
            self.assertGreaterEqual(len(list(tmp.glob("init-*"))), 1)
            self.assertLessEqual(len(list(tmp.glob("init-*"))), 2)
            self.assertFalse(list(dest.glob("battle-*")))
            self.assertFalse((dest / "collection.json").exists())
            with self.assertRaisesRegex(ValueError, "no resume/refill"):
                runner.collect(config, dest)

    def test_first_pool_failure_terminates_active_and_queued_games_without_refill(self):
        repo = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("mirror_failure_runner", repo / "scripts/run_mirror_value_probe.py")
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        config = dict(root="synthetic-pool-failure-fixture", battles=8, workers=2, threads=1,
            checkpoint="fixture", showdown="fixture", node="fixture", max_rounds=1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "initialize", pool_fixture_initialize), patch.object(runner, "worker", pool_fixture_worker):
            dest = Path(tmp) / "collection"
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "synthetic first-game failure"):
                runner.collect(config, dest)
            self.assertLess(time.monotonic() - started, 8)
            self.assertTrue((dest / "first-failure.txt").exists())
            self.assertFalse(list(dest.glob("completed-*")))
            self.assertFalse((dest / "collection.json").exists())
            frozen = json.loads((dest / "cohort.json").read_text())
            self.assertEqual(len(frozen["battles"]), 8)
            with self.assertRaisesRegex(ValueError, "no resume/refill"):
                runner.collect(config, dest)

    def test_refit_parameters_do_not_depend_on_heldout_test_outcomes(self):
        import torch
        from types import SimpleNamespace
        repo = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("mirror_runner", repo / "scripts/run_mirror_value_probe.py")
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        torch.manual_seed(4)
        model = torch.nn.Module()
        model.value_head = torch.nn.Linear(4, 1)
        model.config = SimpleNamespace(value_activation="tanh")
        def fixture(root, flip, checkpoint):
            root.mkdir()
            config = dict(root="synthetic-refit-fixture", battles=16, champion_sha256=CHAMPION_SHA, checkpoint=str(checkpoint), probe_contract="owning-wire-true-moves-champion-linear-v2")
            rows = cohort(config["root"], 16)
            receipts, hashes = [], {}
            for row in rows:
                dest = root / f"battle-{row['index']:04d}"; dest.mkdir()
                winner = "p2" if flip and row["split"] == "test" else "p1"
                y, status = terminal_labels(dict(winner=winner, capped=False), ["p1", "p2"])
                rng = np.random.default_rng(row["index"])
                x = rng.normal(size=(2, 23, 4)).astype(np.float32)
                pooled = x.mean(1)
                with torch.no_grad():
                    values = torch.tanh(model.value_head(torch.from_numpy(pooled)).squeeze(-1)).numpy()
                arrays = dict(outcomes=y, label_status=status)
                for arm in ("public", "privileged"):
                    arrays.update({arm + "_tokens": x, arm + "_pooled": pooled,
                        arm + "_mask": np.ones((2, 23), bool), arm + "_values": values})
                np.savez(dest / "features.npz", **arrays)
                (dest / "boundaries.jsonl.gz").write_bytes(b"synthetic fixture, no games")
                receipt = dict(**row, paired_rows=2, actual_terminal=dict(winner=winner, capped=False),
                    contract=dict(width=4), identities=[dict(root_id="fixture", seat=t, turn_number=1, decision_round_index=0) for t in ("p1", "p2")],
                    files={p.name:file_sha(p) for p in dest.iterdir()})
                write_new(dest / "battle.json", receipt)
                hashes[f"battle-{row['index']:04d}/battle.json"] = file_sha(dest / "battle.json")
                receipts.append({k:v for k,v in receipt.items() if k != "identities"})
            manifest = dict(status="COMPLETE_FIXED_COHORT", config=config, battles=receipts, battle_receipt_hashes=hashes,
                new_games=0, paired_rows=32, known_rows=32, unknown_rows=0)
            write_new(root / "collection.json", manifest)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); checkpoint = tmp / "synthetic-champion.pt"; checkpoint.write_bytes(b"fixture")
            with patch("pokezero.neural_policy.load_transformer_checkpoint", return_value=(model, None)), patch.object(
                    runner, "file_sha", side_effect=lambda p: CHAMPION_SHA if Path(p) == checkpoint else file_sha(p)), patch(
                    "torch.set_num_interop_threads"):
                for label, flip in (("normal", False), ("flipped", True)):
                    root, out = tmp / label, tmp / (label + "-fit")
                    fixture(root, flip, checkpoint)
                    runner.fit(root, out, collection_sha=file_sha(root / "collection.json"), epochs=2, threads=1)
            for arm in ("public", "privileged"):
                a = torch.load(tmp / "normal-fit" / (arm + "-head.pt"), weights_only=True)
                b = torch.load(tmp / "flipped-fit" / (arm + "-head.pt"), weights_only=True)
                self.assertEqual(a["selected_epoch"], b["selected_epoch"])
                for key in a["state_dict"]:
                    self.assertTrue(torch.equal(a["state_dict"][key], b["state_dict"][key]), key)

    def test_swap_preserves_actor_and_uses_peer_raw_rows(self):
        vocab = CategoryVocabulary(("pokemon:self", "pokemon:opponent"))
        actor, peer = observation("p1", 3), observation("p2", 7)
        before = np.array(actor.numeric_features, copy=True)
        swapped = swap_own_view(actor, peer, vocab)
        np.testing.assert_array_equal(swapped.numeric_features[7:13], peer.numeric_features[1:7])
        np.testing.assert_array_equal(swapped.numeric_features[:7], actor.numeric_features[:7])
        np.testing.assert_array_equal(swapped.numeric_features[13:], actor.numeric_features[13:])
        np.testing.assert_array_equal(np.array(swapped.categorical_ids)[7:13, CATEGORY_ROLE], [2] * 6)
        np.testing.assert_array_equal(actor.numeric_features, before)
        self.assertEqual(swapped.token_type_ids, actor.token_type_ids)
        with self.assertRaises(ValueError):
            swap_own_view(actor, actor, vocab)

    def test_cohort_has_exact_frozen_battle_split_and_fresh_identities(self):
        a, b = cohort("fresh-root-a"), cohort("fresh-root-a")
        self.assertEqual(a, b)
        self.assertEqual([sum(r["split"] == s for r in a) for s in ("train", "validation", "test")], [192, 32, 32])
        self.assertFalse({r["battle_id"] for r in a} & {r["battle_id"] for r in cohort("fresh-root-b")})
        self.assertEqual(len({r["seed"] for r in a}), 256)

    def test_actual_terminal_draw_is_known_caps_and_absence_are_unknown(self):
        for terminal in (None, {"winner": "p1", "capped": True}):
            labels, status = terminal_labels(terminal, ["p1", "p2"])
            self.assertTrue(np.isnan(labels).all()); self.assertTrue((status == 1).all())
        y, s = terminal_labels({"winner": None, "capped": False}, ["p1", "p2"])
        np.testing.assert_array_equal(y, [0, 0]); np.testing.assert_array_equal(s, [0, 0])
        y, _ = terminal_labels({"winner": "p2", "capped": False}, ["p1", "p2"])
        np.testing.assert_array_equal(y, [-1, 1])

    def test_battle_cluster_interval_has_paired_sign(self):
        r = paired_battle_interval([.5, .7, .9], [.3, .5, .7])
        self.assertAlmostEqual(r["privileged_minus_public_mse"], -.2)
        self.assertTrue(all(v < 0 for v in r["interval90"]))

    def test_collector_captures_before_action_and_keeps_asymmetric_and_final_wire(self):
        import torch
        from pokezero.env import TerminalState
        from pokezero.policy import PolicyDecision
        class Env:
            def __init__(self, config):
                self.round = 0
                self._latest_requests = {s: {"side": {"pokemon": [{"ident": s + str(i), "moves": ["surf"],
                    "ability": "torrent", "item": "leftovers"} for i in range(6)]}} for s in ("p1", "p2")}
            def reset(self, **kw): pass
            def terminal(self): return TerminalState("p2", 2) if self.round == 2 else None
            def requested_players(self): return ("p1", "p2") if self.round == 0 else ("p1",)
            def observe(self, seat): return observation(seat, 3 if seat == "p1" else 7)
            @property
            def protocol_lines(self): return ("|turn|1",) + (() if self.round == 0 else ("|turn|2",)) + (() if self.round < 2 else ("|win|p2",))
            def step(self, actions): self.round += 1
            def close(self): pass
            @property
            def owning_lines(self): return {s: list(self.protocol_lines) for s in ("p1", "p2")}
            def peer_state(self, seat): return None
        class Policy:
            def __init__(self, *a, **kw): pass
            def select_action(self, obs, **kw): return PolicyDecision(0, "fixture")
        def features(model, tensors):
            n = tensors["numeric_features"].shape[0]
            return torch.zeros((n, 23, 4)), type("Output", (), {"value": torch.zeros(n)})()
        runtime = (torch.nn.Linear(1, 1), None, CategoryVocabulary(("pokemon:self", "pokemon:opponent", "belief:possible_move:surf")), type("Config", (), {"resolved_showdown_root": lambda self: "fixture"})(), {"width": 4})
        with tempfile.TemporaryDirectory() as tmp, patch("pokezero.mirror_value_probe.make_owning_wire_env", Env), patch("pokezero.mirror_value_probe.true_move_facts", return_value=[{"surf": .5}] * 6), patch("pokezero.dex.load_showdown_dex_cached"), patch(
                "pokezero.neural_policy.TransformerSoftmaxPolicy", Policy), patch("pokezero.value_precision_probe.frozen_token_features", features):
            receipt = collect_battle(cohort("fixture", 8)[0], tmp, runtime)
            self.assertEqual(receipt["paired_rows"], 2)
            dest = Path(tmp) / "battle-0000"
            with gzip.open(dest / "boundaries.jsonl.gz", "rt") as stream:
                boundaries = [json.loads(line) for line in stream]
            self.assertEqual([b.get("paired") for b in boundaries], [True, False, None])
            self.assertEqual(boundaries[0]["protocol_delta"], ["|turn|1"])
            self.assertEqual(boundaries[-1]["protocol_delta"][-1], "|win|p2")
            self.assertEqual(boundaries[0]["observations"]["p1"]["numeric_features"][0][0], 3)
            with np.load(dest / "features.npz") as data:
                np.testing.assert_array_equal(data["outcomes"], [-1, 1])
            self.assertFalse(receipt["observation_roots_are_siblings"])


if __name__ == "__main__":
    unittest.main()
