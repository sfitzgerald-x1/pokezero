import copy
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pokezero.mirror_value_probe import CHAMPION_SHA, cohort, digest, file_sha
from pokezero.recorded_mirror_roots import (feed_recorded_boundary, freeze_cost_roots,
    reconstruct_cost_root, select_recorded_boundary, select_training_battles)


def observation(marker=0):
    return dict(categorical_ids=np.zeros((23, 41), int).tolist(),
        numeric_features=np.full((23, 132), marker, float).tolist(),
        token_type_ids=[0]*23, attention_mask=[True]*23,
        legal_action_mask=[True]*9, turn_number=3)


class OfflineEnv:
    def __init__(self):
        self._process = None
        self._battle_token = None
        self._latest_requests = {}
        self.lines = []
        self.closed = False
        self.calls = []

    @property
    def protocol_lines(self):
        return tuple(self.lines)

    def _apply_event(self, event):
        self.calls.append(event)
        self.lines.extend(event["lines"])
        if event["stream"] in ("p1", "p2"):
            self._latest_requests[event["stream"]] = json.loads(event["lines"][0][len("|request|"):])

    def requested_players(self):
        return tuple(self._latest_requests)

    def observe(self, seat):
        value = observation()
        return SimpleNamespace(**{k:v for k,v in value.items() if k != "turn_number"},
                               metadata={"turn_number":3})

    def close(self):
        self.closed = True


def boundary(battle, index=0):
    requests = {seat: {"side": {"id": seat}} for seat in ("p1", "p2")}
    lines = ["|turn|3"] + ["|request|" + json.dumps(requests[seat]) for seat in requests]
    prefix = digest(lines)
    return dict(schema="pokezero.mirror_boundary.v1", battle_id=battle["battle_id"], seed=battle["seed"],
        decision_round_index=index, requested_players=["p1","p2"], protocol_start=0,
        protocol_end=len(lines), protocol_delta=lines, public_prefix_sha256=prefix,
        requests=requests, root_id=digest([battle["battle_id"], index, prefix, requests]),
        observations={seat:observation() for seat in requests}, actions={"p1":0,"p2":1})


class SelectionTests(unittest.TestCase):
    def test_distinct_balanced_train_selection_ignores_outcomes(self):
        roster = cohort("fixture", 32)
        selected = select_training_battles(roster)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len({r["battle_id"] for r in selected}), 20)
        self.assertEqual(sum(r["cost_seat"] == "p1" for r in selected), 10)
        self.assertEqual({r["split"] for r in selected}, {"train"})
        changed = copy.deepcopy(roster)
        for row in changed:
            row.update(actual_terminal={"winner":"p1"}, cost=1e99, value=-999)
        self.assertEqual([r["battle_id"] for r in selected],
                         [r["battle_id"] for r in select_training_battles(changed[::-1])])
        with self.assertRaises(ValueError):
            select_training_battles([r for r in roster if r["split"] != "train"])

    def test_boundary_hash_selection_no_skip_on_source_mismatch(self):
        battle = select_training_battles(cohort("fixture",32))[0]
        rows = [boundary(battle, i) for i in range(3)]
        chosen = select_recorded_boundary(rows, battle)
        for row in rows:
            row.update(outcome=-999, cost=1e99, head_value=999)
        self.assertEqual(chosen["root_id"], select_recorded_boundary(rows[::-1], battle)["root_id"])
        rows[0]["battle_id"] = "different"
        with self.assertRaisesRegex(ValueError, "identity"):
            select_recorded_boundary(rows, battle)

    def test_freeze_complete_hash_roster_and_changed_source_refusal(self):
        roster = cohort("fixture",32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(schema="pokezero.mirror_collection.v1", status="COMPLETE_FIXED_COHORT",
                config={"champion_sha256":CHAMPION_SHA,"root":"fixture","battles":32},
                battle_receipt_hashes={})
            (root/"cohort.json").write_text(json.dumps({"battles":roster,"roster_sha256":digest(roster)}))
            manifest["cohort_sha256"] = file_sha(root/"cohort.json")
            for battle in select_training_battles(roster):
                dest = root/f"battle-{battle['index']:04d}"
                dest.mkdir()
                with gzip.open(dest/"boundaries.jsonl.gz","wt") as stream:
                    stream.write(json.dumps(boundary(battle))+"\n")
                    stream.write(json.dumps({"event":"end","actual_terminal":{"winner":"p1"}})+"\n")
                receipt = dict(**{k:battle[k] for k in ("index","battle_id","seed","split")},
                    files={"boundaries.jsonl.gz":file_sha(dest/"boundaries.jsonl.gz")})
                (dest/"battle.json").write_text(json.dumps(receipt))
                manifest["battle_receipt_hashes"][str((dest/"battle.json").relative_to(root))] = file_sha(dest/"battle.json")
            path = root/"collection.json"
            path.write_text(json.dumps(manifest))
            binding = file_sha(path)
            frozen = freeze_cost_roots(root, expected_sha256=binding)
            self.assertEqual(frozen["count"],20)
            self.assertEqual(frozen["native_search_calls"],0)
            selected = frozen["roots"][0]
            path = root/selected["boundaries_relative_path"]
            original = path.read_bytes()
            path.write_bytes(original+b"drift")
            with self.assertRaisesRegex(ValueError,"boundary artifact SHA"):
                freeze_cost_roots(root, expected_sha256=binding)


class ReconstructionTests(unittest.TestCase):
    def test_production_event_routing_exact_observations_and_fail_closed(self):
        battle = select_training_battles(cohort("fixture",32))[0]
        row = boundary(battle)
        env = OfflineEnv()
        observations = feed_recorded_boundary(env, row)
        self.assertEqual(set(observations), {"p1","p2"})
        self.assertEqual([e["stream"] for e in env.calls], ["omniscient","p1","p2"])
        self.assertIsNone(env._process)
        for mutate, message in ((lambda r:r.update(protocol_start=1), "continuity"),
                (lambda r:r.update(root_id="wrong"), "root SHA"),
                (lambda r:r["observations"]["p1"]["numeric_features"][0].__setitem__(0,1), "observation differs"),
                (lambda r:r.update(requests={}), "prefix/request")):
            changed = copy.deepcopy(row);mutate(changed)
            with self.assertRaisesRegex(ValueError,message):
                feed_recorded_boundary(OfflineEnv(),changed)
        env = OfflineEnv();env._process = object()
        with self.assertRaisesRegex(ValueError,"forbids a simulator"):
            feed_recorded_boundary(env,row)

    def test_frozen_root_missing_or_changed_does_not_skip_and_closes_env(self):
        battle = select_training_battles(cohort("fixture",32))[0]
        row = boundary(battle)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"boundaries.jsonl.gz"
            with gzip.open(path,"wt") as stream:
                stream.write(json.dumps(row)+"\n")
                stream.write(json.dumps({"event":"end"})+"\n")
            root = dict(battle,boundaries_relative_path=path.name,boundaries_file_sha256=file_sha(path),
                decision_round_index=0, root_id=row["root_id"], boundary_sha256=digest(row),
                requests_sha256=digest(row["requests"]),observation_sha256=digest(row["observations"][battle["cost_seat"]]))
            env = OfflineEnv()
            with patch("pokezero.recorded_mirror_roots.new_recorded_environment",return_value=env):
                actual, _ = reconstruct_cost_root(None,directory,root)
            self.assertIs(actual,env)
            for changed in (dict(root,root_id="wrong"),dict(root,decision_round_index=1)):
                env = OfflineEnv()
                with patch("pokezero.recorded_mirror_roots.new_recorded_environment",return_value=env), self.assertRaises(ValueError):
                    reconstruct_cost_root(None,directory,changed)
                self.assertTrue(env.closed)
            with self.assertRaisesRegex(ValueError,"TRAIN"):
                reconstruct_cost_root(None,directory,dict(root,split="test"))


if __name__ == "__main__":
    unittest.main()
