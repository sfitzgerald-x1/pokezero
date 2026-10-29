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
        repo = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("mirror_runner", repo / "scripts/run_mirror_value_probe.py")
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        def fixture(root, flip):
            root.mkdir()
            config = dict(root="synthetic-refit-fixture", battles=16, champion_sha256=CHAMPION_SHA)
            rows = cohort(config["root"], 16)
            receipts, hashes = [], {}
            for row in rows:
                dest = root / f"battle-{row['index']:04d}"; dest.mkdir()
                winner = "p2" if flip and row["split"] == "test" else "p1"
                y, status = terminal_labels(dict(winner=winner, capped=False), ["p1", "p2"])
                rng = np.random.default_rng(row["index"])
                x = rng.normal(size=(2, 23, 4)).astype(np.float32)
                arrays = dict(outcomes=y, label_status=status)
                for arm in ("public", "privileged"):
                    arrays.update({arm + "_tokens": x, arm + "_mask": np.ones((2, 23), bool), arm + "_values": np.zeros(2, np.float32)})
                np.savez(dest / "features.npz", **arrays)
                (dest / "boundaries.jsonl.gz").write_bytes(b"synthetic fixture, no games")
                receipt = dict(**row, paired_rows=2, actual_terminal=dict(winner=winner, capped=False),
                    contract=dict(width=4), identities=[dict(root_id="fixture", seat=s, turn_number=1, decision_round_index=0) for s in ("p1", "p2")],
                    files={p.name:file_sha(p) for p in dest.iterdir()})
                write_new(dest / "battle.json", receipt)
                hashes[f"battle-{row['index']:04d}/battle.json"] = file_sha(dest / "battle.json")
                receipts.append({k:v for k,v in receipt.items() if k != "identities"})
            manifest = dict(status="COMPLETE_FIXED_COHORT", config=config, battles=receipts, battle_receipt_hashes=hashes,
                new_games=0, paired_rows=32, known_rows=32, unknown_rows=0)
            write_new(root / "collection.json", manifest)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for label, flip in (("normal", False), ("flipped", True)):
                root, out = tmp / label, tmp / (label + "-fit")
                fixture(root, flip)
                command = [sys.executable, "-c", "import importlib.util,pathlib; p=pathlib.Path('scripts/run_mirror_value_probe.py'); s=importlib.util.spec_from_file_location('runner',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.fit(pathlib.Path(__import__('sys').argv[1]),pathlib.Path(__import__('sys').argv[2]),collection_sha=__import__('sys').argv[3],epochs=2,threads=1)", str(root), str(out), file_sha(root / "collection.json")]
                completed = subprocess.run(command, cwd=repo, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
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
        class Policy:
            def __init__(self, *a, **kw): pass
            def select_action(self, obs, **kw): return PolicyDecision(0, "fixture")
        def features(model, tensors):
            n = tensors["numeric_features"].shape[0]
            return torch.zeros((n, 23, 4)), type("Output", (), {"value": torch.zeros(n)})()
        runtime = (torch.nn.Linear(1, 1), None, CategoryVocabulary(("pokemon:self", "pokemon:opponent")), None, {"width": 4})
        with tempfile.TemporaryDirectory() as tmp, patch("pokezero.local_showdown.LocalShowdownEnv", Env), patch(
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
