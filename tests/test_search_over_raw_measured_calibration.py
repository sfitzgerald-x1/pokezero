import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measured_calibration_test",
    ROOT / "scripts/calibrate_search_over_raw_measured.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


class MeasuredCalibrationTests(unittest.TestCase):
    def test_finite_roster_dispersion_boxes_preserve_uncertainty_and_point_cases(self):
        self.assertEqual(cal.interval_variance_bounds([[-1,1],[-1,1]],[.5,.5]),[0.,1.])
        point=cal.interval_variance_bounds([[-.5,-.5],[.5,.5]],[.5,.5])
        self.assertEqual(point,[.25,.25])
        lo,hi=cal.interval_variance_bounds([[-1,-.5],[.5,1]],[.5,.5])
        self.assertLessEqual(lo,.25);self.assertGreater(lo,.249999999)
        self.assertEqual(hi,1.)
        for intervals,weights in (([[0,0]],[1.1]),([[1,-1]],[1]),([[0,2]],[1])):
            with self.assertRaises(ValueError):cal.interval_variance_bounds(intervals,weights)

    def fixed_fixture(self,path,*,complete=True,compact=False,unknown=False):
        from pokezero.mcts_eval import search_over_raw_fixed_exploration as fixed
        c=fixed.contract("51f8beec-d601-4fdc-ab73-f0517cd05001",list(range(32)),[100],
            dict(path="/authored",sha256="a"*64),4096 if compact else 1024,compact=compact)
        rows=[]
        for i,slot in enumerate(c["panels"]["exploration"]["root_slots"] if complete else []):
            interval=[-.25,-.25] if slot["source_seed"]%2 else [.25,.25]
            row=dict(root_id=slot["root_id"],source_seed=slot["source_seed"],
                source_pipeline_validated=True,actions=dict(raw=0,model=1),intervals=dict(model=interval))
            if unknown and i==0:
                row.pop("source_pipeline_validated");row.update(actions={},
                    input_unavailable={"source":"missing_fixed_slot"},intervals=dict(model=[-1.,1.]))
            rows.append(row)
        terminal=dict(supervisor_exit_code=0,worker_exit_code=0,status="STAGED_EXPLORATION_RUN_COMPLETE",
            elapsed_wall_seconds=1.,cpu_seconds_including_reaped_children=.8)
        (path/"completed-roots").mkdir();(path/"auditor").mkdir()
        for i,row in enumerate(rows):(path/"completed-roots"/f"{i}.json").write_text(json.dumps(row))
        (path/"entry.json").write_text(json.dumps(dict(contract=c)))
        (path/"auditor/terminal.json").write_text(json.dumps(terminal))
        (path/"readout.json").write_text(json.dumps(fixed.summarize(dict(contract=c),rows,terminal)))
        return path/"readout.json"

    def test_fixed_formats_keep_shared_control_vectors_and_exact_full_roster_variance(self):
        for compact in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder);readout=self.fixed_fixture(path,compact=compact)
                meta,groups=cal.load_panel(readout)
                self.assertEqual(meta["uncertain_config_root_contrasts"],0)
                self.assertEqual(len(groups),32)
                self.assertEqual(sum(len(g) for g in groups),200)
                self.assertEqual(cal.variance_components(groups)["pooled_within_seed_covariance"],[[0.]])
                self.assertEqual(meta["finite_roster_dispersion_bounds"][0]["seed_mean_population_dispersion_bounds"],[.0625,.0625])

    def test_fixed_missing_slot_or_incomplete_roster_never_fits_complete_cases(self):
        for complete,unknown in ((True,True),(False,False)):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder);readout=self.fixed_fixture(path,complete=complete,unknown=unknown)
                meta,groups=cal.load_panel(readout)
                self.assertIsNone(groups)
                self.assertEqual(meta["uncertain_config_root_contrasts"],1 if complete else 200)
                self.assertTrue(meta["fixed_denominator"])

    def test_missing_worker_exit_only_allows_zero_root_operational_stop(self):
        for complete in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder);readout=self.fixed_fixture(path,complete=complete)
                terminal_path=path/"auditor/terminal.json"
                terminal=json.loads(terminal_path.read_text())
                terminal.update(status="OPERATIONAL_STOP",supervisor_exit_code=1,worker_exit_code=None)
                terminal_path.write_text(json.dumps(terminal))
                record=json.loads(readout.read_text())
                record.update(terminal_status=terminal["status"],supervisor_exit_code=1,worker_exit_code=None)
                readout.write_text(json.dumps(record))
                if complete:
                    with self.assertRaises(ValueError):cal.load_panel(readout)
                else:
                    metadata,groups=cal.load_panel(readout)
                    self.assertIsNone(groups)
                    self.assertEqual(metadata["uncertain_config_root_contrasts"],200)
                    self.assertEqual(metadata["readout_identification_bounds"]["model"],[-1.,1.])

    def test_fixed_analysis_rejects_open_run_missing_validation_and_inconsistent_readout(self):
        for mode in ("open","source","bounds","counts","encoding","running","wrong_exit","boolean_exit","boolean_interval"):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder);readout=self.fixed_fixture(path,compact=True)
                if mode=="open":(path/"auditor/terminal.json").unlink()
                elif mode=="source":
                    p=path/"completed-roots/0.json";row=json.loads(p.read_text())
                    row.pop("source_pipeline_validated");p.write_text(json.dumps(row))
                elif mode=="boolean_interval":
                    p=path/"completed-roots/0.json";row=json.loads(p.read_text())
                    row["intervals"]["model"]=[True,True];p.write_text(json.dumps(row))
                elif mode in ("running","wrong_exit","boolean_exit"):
                    p=path/"auditor/terminal.json";terminal=json.loads(p.read_text())
                    record=json.loads(readout.read_text())
                    if mode=="running":terminal["status"]=record["terminal_status"]="RUNNING"
                    else:terminal["worker_exit_code"]=record["worker_exit_code"]=1 if mode=="wrong_exit" else False
                    p.write_text(json.dumps(terminal));readout.write_text(json.dumps(record))
                else:
                    record=json.loads(readout.read_text())
                    if mode=="bounds":record["effects"]["model"]["fixed_seed_weighted_bounds"]=[-1,1]
                    elif mode=="counts":record["effects"]["model"]["identified_roots"]-=1
                    else:record["native_atomic_encoding"]="v1"
                    readout.write_text(json.dumps(record))
                with self.subTest(mode=mode),self.assertRaises((ValueError,FileNotFoundError)):
                    cal.load_panel(readout)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);readout=self.fixed_fixture(path)
            record=json.loads(readout.read_text());record["native_atomic_encoding"]="native-atomic-backups-compact-v2"
            readout.write_text(json.dumps(record))
            with self.assertRaises(ValueError):cal.load_panel(readout)

    def test_constant_panel_has_zero_variance_without_pseudoreplicating(self):
        groups = [np.full((7 if i < 8 else 6, 2), .125) for i in range(32)]
        result = cal.variance_components(groups)
        self.assertEqual(sum(result["seed_root_counts"]), 200)
        np.testing.assert_array_equal(result["pooled_within_seed_covariance"], np.zeros((2, 2)))
        np.testing.assert_array_equal(result["between_seed_covariance_mom_psd"], np.zeros((2, 2)))
        self.assertEqual(result["mean"], [.125, .125])

    def test_perfect_within_seed_dependence_stays_between_seeds(self):
        groups = [np.full((7 if i < 8 else 6, 1), .25 if i % 2 else -.25) for i in range(32)]
        result = cal.variance_components(groups)
        self.assertEqual(result["pooled_within_seed_covariance"], [[0.]])
        self.assertAlmostEqual(result["observed_seed_mean_covariance"][0][0], .0625 * 32 / 31)
        self.assertAlmostEqual(result["equal_seed_mean_sampling_covariance"][0][0], .0625 / 31)

    def test_missing_roots_do_not_yield_complete_case_variance(self):
        cfg = dict(arm="incumbent", belief="public", leaf="model", seconds=1., workers=1)
        slots = [dict(root_id=f"r{i}", source_seed=i % 32) for i in range(200)]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / "completed-roots").mkdir()
            (p / "entry.json").write_text(json.dumps(dict(contract=dict(configurations=[cfg],
                panels=dict(exploration=dict(seeds=list(range(32)), root_slots=slots))))))
            (p / "readout.json").write_text(json.dumps(dict(recorded_roots=0, rejected_root_receipts=[],
                leaves=dict(model=dict(fixed_seed_weighted_mean_bounds=[-1, 1])))))
            metadata, groups = cal.load_panel(p / "readout.json")
            self.assertIsNone(groups)
            self.assertEqual(metadata["uncertain_config_root_contrasts"], 200)
            self.assertTrue(metadata["fixed_denominator"])

    def test_selection_uses_independent_holdout_and_declares_missing_reference(self):
        groups = [np.zeros((7 if i < 8 else 6, 2)) for i in range(32)]
        configs = [dict(arm="incumbent", leaf=leaf) for leaf in ("model", "hp_fraction")]
        result = cal.simulate(cal.variance_components(groups), configs, replicates=2, draws=20)
        self.assertEqual(result["results"][0]["per_arm"]["incumbent"]["rate"], 0.)
        self.assertEqual(result["results"][1]["per_arm"]["incumbent"]["rate"], 1.)
        self.assertFalse(result["both_plan_arms_measured"])
        self.assertFalse(result["holdout_opened"])
        self.assertTrue(result["independent_exploration_and_holdout"])
        self.assertIn("does not calibrate the revised fixed-work selection procedure",result["scope"])


if __name__ == "__main__":
    unittest.main()
