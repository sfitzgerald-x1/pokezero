import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
NODE = os.environ.get("NODE") or shutil.which("node")


def fixture():
    header = dict(kind="header", schema="pokezero.gpu-team-stream.v1", team_count=2, seed_start=100,
                  vocabulary=dict(moves=["_none", "tackle", "protect"],
                    species=["_none", "snorlax"], types=["_none", "normal", "fighting"],
                    items=["_none", "leftovers"], abilities=["_none", "immunity"],
                    natures=["_none", "serious"]))
    mon = dict(species=1, level=80, types=[1,0], stats=[300,100,100,100,100,100],
               moves=[1,2,1,2], pp=[8,16,8,16], item=1, ability=1, nature=1,
               ivs=[31]*6, evs=[84]*6, hidden_power_type=2, hidden_power_power=70)
    rows = [dict(kind="team", index=i, seed=100+i, members=[copy.deepcopy(mon) for _ in range(6)]) for i in range(2)]
    return [header, *rows, dict(kind="terminal", team_count=2)]


@unittest.skipUnless(NUMPY_AVAILABLE, "requires the isolated spike NumPy environment")
class PoolTests(unittest.TestCase):
    def setUp(self):
        from gpu_engine.team_pool import pack_stream
        self.pack = pack_stream

    def run_fixture(self, records):
        return self.pack(io.StringIO("".join(json.dumps(r)+"\n" for r in records)), count=2, seed_start=100)

    def test_shapes_numeric_metadata_and_measured_move_frequencies(self):
        arrays, header = self.run_fixture(fixture())
        self.assertEqual(arrays["stats"].shape, (2,6,6))
        self.assertEqual(arrays["moves"].shape, (2,6,4))
        self.assertEqual(arrays["metadata_utf8"].dtype.kind, "u")
        self.assertEqual(header["move_slots"], 48)
        self.assertEqual(header["target85_move_count"], 2)
        self.assertEqual([r["count"] for r in header["move_frequency"]], [24,24])
        self.assertEqual(header["implemented_move_coverage"], 0)

    def test_wrong_seed_and_extra_rows_are_refused(self):
        records = fixture(); records[1]["seed"] = 101
        with self.assertRaisesRegex(ValueError, "chronological"):
            self.run_fixture(records)
        with self.assertRaisesRegex(ValueError, "trailing"):
            self.run_fixture(fixture()+[{}])

    def test_invalid_shapes_ranges_and_ids_are_refused(self):
        for field, value in [("species", 99), ("level", 101), ("ivs", [32]*6),
                             ("evs", [252]*6), ("pp", [0]*4), ("types", [1]),
                             ("moves", [True]*4), ("species", 0), ("moves", [0]*4),
                             ("types", [0,1]), ("ability", 0), ("nature", 0),
                             ("hidden_power_type", 0), ("hidden_power_type", 1),
                             ("hidden_power_power", 29),
                             ("hidden_power_power", 71)]:
            with self.subTest(field=field):
                records = fixture(); records[1]["members"][0][field] = value
                with self.assertRaises(ValueError):
                    self.run_fixture(records)

    def test_missing_or_wrong_terminal_never_completes(self):
        records = fixture(); records[-1]["team_count"] = 1
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.run_fixture(records)
        with self.assertRaises(StopIteration):
            self.run_fixture(fixture()[:-1])


@unittest.skipUnless(NODE, "requires Node for the offline generator helper")
class StatsTests(unittest.TestCase):
    def test_curse_retains_explicit_typeless_catalog_type(self):
        helper = ROOT / "gpu_engine/generate_teams.cjs"
        code = ("const {moveTypeId:f}=require(process.argv[1]);"
                "const types=['_none','normal','fighting'];"
                "if(f('???',types)!==0 || f('Fighting',types)!==2)process.exit(1);"
                "try {f('Fairy',types);process.exit(2)} catch(e) {}")
        subprocess.run([NODE,"-e",code,str(helper)], check=True)

    def test_gen3_stat_rounding_nature_and_shedinja_hp(self):
        helper = ROOT / "gpu_engine/generate_teams.cjs"
        code = (
            "const {calculatedStats:f}=require(process.argv[1]);"
            "const b={hp:100,atk:100,def:100,spa:100,spd:100,spe:100};"
            "const iv=Object.fromEntries(Object.keys(b).map(s=>[s,31]));"
            "const ev=Object.fromEntries(Object.keys(b).map(s=>[s,252]));"
            "console.log(JSON.stringify([f(b,iv,ev,100,{plus:'atk',minus:'spa'}),"
            "f({...b,hp:1},iv,ev,100,{})[0]]));"
        )
        result = subprocess.run([NODE,"-e",code,str(helper)], check=True, text=True, capture_output=True)
        stats, shedinja_hp = json.loads(result.stdout)
        self.assertEqual(stats, [404,328,299,269,299,299])
        self.assertEqual(shedinja_hp, 1)


if __name__ == "__main__":
    unittest.main()
