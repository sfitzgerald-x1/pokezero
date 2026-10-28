"""Outcome-only fitting, split isolation and unknown-label accounting."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pokezero.neural_policy import torch_available
from pokezero.value_precision_probe import battle_split


def runner():
    path=Path(__file__).resolve().parents[1]/"scripts/fit_value_probe_heads.py"
    spec=importlib.util.spec_from_file_location("fit_value_probe_heads",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch_available(),"PyTorch required")
class ValueProbeFitTests(unittest.TestCase):
    def fixture(self,root,*,flip_test=False):
        module=runner();root.mkdir();shard=root/"shard-00000";shard.mkdir()
        groups={name:[] for name in ("train","validation","test")}
        seed=0
        while any(len(v)<4 for v in groups.values()):
            key=battle_split(seed,lineage="fixture",salt="split")
            if len(groups[key])<4:groups[key].append(seed)
            seed+=1
        y=np.array([1,-1,1,np.nan]*3,np.float32)
        if flip_test:y[8:11]*=-1
        status=np.array([0,0,0,2]*3,np.uint8);status[-1]=1
        tokens=np.random.default_rng(4).normal(0,.1,(12,23,8)).astype(np.float32)
        # Input signals stay fixed when test outcomes are flipped below.
        tokens[:,:,0]=np.array([1,-1,1,0]*3,np.float32)[:,None]
        arrays={"tokens":tokens,"attention_mask":np.ones((12,23),bool),"outcomes":y,
            "label_status":status,"split":np.repeat(np.arange(3,dtype=np.uint8),4),
            "seeds":np.array(sum(groups.values(),[]),np.uint64),"turn_indices":np.arange(12,dtype=np.uint32),
            "champion_values":np.zeros(12,np.float32)}
        for name,array in arrays.items():np.save(shard/(name+".npy"),array,allow_pickle=False)
        counts={"rows":12,"known":9,"capped":1,"unresolved_zero":2}
        manifest={"schema":"pokezero.value_probe_features.v1","status":"FROZEN_FP32_FEATURES_COMPLETE_LABEL_ZEROS_UNRESOLVED",
            "trunk_unchanged":True,"model_fits":0,"checkpoint_contract":{"width":8},
            "counts":counts,"lineage":"fixture","split_salt":"split","prior_exposure":"fixture only",
            "policy_distribution":"generated fixture","shards":[{"path":str(shard),"counts":counts,
                "hashes":{p.name:module.digest_file(p) for p in shard.glob("*.npy")}}]}
        path=root/"features-manifest.json";path.write_text(json.dumps(manifest))
        return path,module.digest_file(path)

    def test_three_heads_fit_known_only_and_select_validation_before_test(self):
        from pokezero.neural_policy import require_torch
        torch=require_torch();module=runner()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,sha=self.fixture(root/"features")
            report=module.fit(manifest,root/"heads",manifest_sha256=sha,epochs=3,batch_size=2,seed=31)
            self.assertEqual(report["model_fits"],3);self.assertEqual(report["new_games"],0)
            self.assertEqual(report["baseline_test"]["known_rows"],3)
            self.assertEqual(report["baseline_test"]["capped_rows"],1)
            for kind in module.KINDS:
                row=report["heads"][kind]
                self.assertEqual([e["training_examples"] for e in row["epochs"]],[3]*3)
                best=min(row["epochs"],key=lambda e:e["validation"]["mse"])
                self.assertEqual(row["selected_epoch"],best["epoch"])
                self.assertEqual(row["test"]["total_rows"],4)
                self.assertTrue(np.isfinite(np.load(root/"heads"/(kind+"-test-predictions.npy"))).all())
            # Test-label changes must not alter any selected fitted parameter.
            other,other_sha=self.fixture(root/"flipped",flip_test=True)
            second=module.fit(other,root/"other-heads",manifest_sha256=other_sha,epochs=3,batch_size=2,seed=31)
            for kind in module.KINDS:
                a=torch.load(root/"heads"/(kind+".pt"),weights_only=True)
                b=torch.load(root/"other-heads"/(kind+".pt"),weights_only=True)
                self.assertEqual(a["selected_epoch"],b["selected_epoch"])
                for key in a["state_dict"]:self.assertTrue(torch.equal(a["state_dict"][key],b["state_dict"][key]))
            self.assertNotEqual(report["heads"]["linear"]["test"]["mse"],second["heads"]["linear"]["test"]["mse"])

    def test_unknown_bounds_and_changed_member_or_seed_refuse(self):
        module=runner()
        result=module.metrics(np.array([.5,-.5,1.5]),np.array([1,-1,np.nan]),np.array([0,0,2]))
        self.assertAlmostEqual(result["mse"],.25)
        np.testing.assert_allclose(result["full_roster_mse_identification_bounds"],[.25,2.25])
        large=module.metrics(np.array([1e20,1e20],np.float32),np.array([1,np.nan],np.float32),np.array([0,2]))
        self.assertTrue(np.isfinite(large["full_roster_mse_identification_bounds"]).all())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,sha=self.fixture(root/"features")
            with self.assertRaisesRegex(ValueError,"manifest binding"):module.load_features(manifest,"0"*64)
            member=manifest.parent/"shard-00000/split.npy";split=np.load(member);split[0]=2;np.save(member,split)
            with self.assertRaisesRegex(ValueError,"member changed"):module.load_features(manifest,sha)
            data=json.loads(manifest.read_text());data["shards"][0]["hashes"]["split.npy"]=module.digest_file(member)
            manifest.write_text(json.dumps(data));sha=module.digest_file(manifest)
            with self.assertRaisesRegex(ValueError,"battle split changed"):module.load_features(manifest,sha)

    def test_nonfinite_unknown_features_and_non_fp32_baseline_refuse(self):
        module=runner()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ("tokens","champion_values"):
                manifest,_=self.fixture(root/name)
                member=manifest.parent/"shard-00000"/(name+".npy")
                array=np.load(member)
                if name=="tokens":array[3,0,0]=np.nan  # unresolved training row
                else:array=array.astype(np.float64)
                np.save(member,array)
                data=json.loads(manifest.read_text());data["shards"][0]["hashes"][member.name]=module.digest_file(member)
                manifest.write_text(json.dumps(data))
                with self.assertRaises(ValueError):module.load_features(manifest,module.digest_file(manifest))


if __name__=="__main__":unittest.main()
