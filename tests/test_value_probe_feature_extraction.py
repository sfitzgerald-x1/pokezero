"""Small generated cohort through the actual frozen-trunk extraction runner."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pokezero.neural_policy import torch_available


def script(name):
    path=Path(__file__).resolve().parents[1]/"scripts"/(name+".py")
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch_available(), "PyTorch required")
class FeatureExtractionTests(unittest.TestCase):
    def fixture(self, root):
        from pokezero.neural_policy import (EntityTokenTransformerPolicy, TransformerPolicyConfig, require_torch,
                                           neural_policy_schema_version_for_model_config)
        from pokezero.value_precision_probe import checkpoint_contract
        torch=require_torch();torch.manual_seed(15)
        config=TransformerPolicyConfig.compact_category(category_vocab=["kind:a", "kind:b"],
            window_size=1, embedding_dim=16, attention_heads=4, feedforward_dim=32,
            transformer_layers=1, dropout=0.0, observation_schema_version="pokezero.observation.v4")
        model=EntityTokenTransformerPolicy(config)
        checkpoint=root/"champion.pt"
        torch.save({"schema_version":neural_policy_schema_version_for_model_config(config),
                    "model_config":config.to_dict(),"state_dict":model.state_dict(),
                    "belief_set_source_hash":"fixture-source"},checkpoint)
        contract=checkpoint_contract(checkpoint)
        cache=root/"cache";shard=cache/"iteration-1/shard-000";shard.mkdir(parents=True)
        rows=6
        arrays={"numeric_features":np.random.default_rng(3).normal(size=(rows+1,23,132)).astype(np.float16),
            "categorical_ids":np.ones((rows+1,23,5),np.uint16),
            "token_type_ids":np.zeros((rows+1,23),np.uint8),
            "attention_mask":np.ones((rows+1,23),bool),
            "returns":np.array([-1,1,0,.8,1,.2],np.float32),
            "terminal_capped":np.array([False]*5+[True]),
            "seeds":np.array([7,7,8,9,9,10],np.uint64),
            "turn_indices":np.arange(rows,dtype=np.uint32),
            "window_indices":np.arange(1,rows+1,dtype=np.uint32)[:,None]}
        arrays["attention_mask"][:,10:13]=False
        metadata={"schema_version":"pokezero.training_cache.v2","observation_schema":"pokezero.observation.v4",
            "collection_env":"showdown","belief_fidelity":"full","belief_set_source_hash":"fixture-source",
            "feature_masks":contract["feature_masks"],"example_count":rows,
            "observation_shapes":{"numeric_features":[23,132],"categorical_ids":[23,5],
                "token_type_ids":[23],"attention_mask":[23],"window_size":1},
            "categorical_storage":{"mode":"compact-nonzero","original_feature_count":41},
            "dataset_config":{"discount":.9999}}
        (shard/"metadata.json").write_text(json.dumps(metadata))
        for key,array in arrays.items():np.save(shard/(key+".npy"),array,allow_pickle=False)
        cohort=root/"cohort"
        script("freeze_value_probe_cohort").freeze(cache,cohort,contract,target=rows,
            lineage="fixture",selection_salt="selection",split_salt="split")
        manifest=cohort/"cohort-manifest.json"
        return checkpoint,cache,manifest,model,arrays,shard

    def test_streamed_fp32_tokens_values_membership_and_unknown_coverage(self):
        from pokezero.neural_policy import require_torch
        from pokezero.value_precision_probe import battle_split, frozen_token_features
        torch=require_torch();module=script("extract_value_probe_features")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint,cache,manifest,model,arrays,_=self.fixture(root)
            result=module.extract(manifest,checkpoint,cache,root/"features",
                manifest_sha256=module.digest_file(manifest),batch_size=2)
            self.assertEqual(result["counts"],{"rows":6,"known":4,"capped":1,"unresolved_zero":1})
            self.assertTrue(result["trunk_unchanged"]);self.assertEqual(result["model_fits"],0)
            out=Path(result["shards"][0]["path"])
            features=np.load(out/"tokens.npy");self.assertEqual(features.dtype,np.float32)
            tensors={name:torch.as_tensor(np.array(arrays[name][1:],copy=True),
                dtype=torch.float32 if name=="numeric_features" else torch.bool if name=="attention_mask" else torch.long)[:,None]
                for name in ("numeric_features","categorical_ids","attention_mask","token_type_ids")}
            tensors["history_mask"]=torch.ones(6,1,dtype=torch.bool)
            expected,value=frozen_token_features(model,tensors)
            np.testing.assert_allclose(features,expected.numpy(),rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(np.load(out/"champion_values.npy"),value.value.numpy(),atol=1e-6)
            self.assertEqual(np.load(out/"label_status.npy").tolist(),[0,0,2,0,0,1])
            outcomes=np.load(out/"outcomes.npy");self.assertTrue(np.isnan(outcomes[[2,5]]).all())
            split=np.load(out/"split.npy")
            for i,seed in enumerate(arrays["seeds"]):
                self.assertEqual(int(split[i]),{"train":0,"validation":1,"test":2}[battle_split(int(seed),lineage="fixture",salt="split")])
            for name,sha in result["shards"][0]["hashes"].items():self.assertEqual(module.digest_file(out/name),sha)
            with self.assertRaises(ValueError):module.extract(manifest,checkpoint,cache,root/"features",
                manifest_sha256=module.digest_file(manifest))

    def test_manifest_checkpoint_and_frozen_member_mismatches_refuse(self):
        module=script("extract_value_probe_features")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint,cache,manifest,_,arrays,shard=self.fixture(root)
            with self.assertRaisesRegex(ValueError,"manifest binding"):
                module.extract(manifest,checkpoint,cache,root/"wrong-sha",manifest_sha256="0"*64)
            self.assertFalse((root/"wrong-sha").exists())
            with self.assertRaisesRegex(ValueError,"outside the read-only cache"):
                module.extract(manifest,checkpoint,cache,cache/"new-output",manifest_sha256=module.digest_file(manifest))
            self.assertFalse((cache/"new-output").exists())
            missing=json.loads(manifest.read_text());del missing["shards"][0]["hashes"]["returns.npy"]
            bad=root/"missing-hash.json";bad.write_text(json.dumps(missing))
            with self.assertRaisesRegex(ValueError,"requires a frozen hash"):
                module.extract(bad,checkpoint,cache,root/"missing-hash",manifest_sha256=module.digest_file(bad))
            # A stable filename cannot admit changed source contents.
            altered=arrays["returns"].copy();altered[0]=1
            np.save(shard/"returns.npy",altered)
            with self.assertRaisesRegex(ValueError,"source hash changed"):
                module.extract(manifest,checkpoint,cache,root/"changed-member",manifest_sha256=module.digest_file(manifest))
            self.assertFalse((root/"changed-member/features-manifest.json").exists())
            # No re-selection or fallback checkpoint follows an input failure.
            np.save(shard/"returns.npy",arrays["returns"])
            data=json.loads(manifest.read_text());data["contract"]["checkpoint_sha256"]="0"*64
            changed=root/"wrong-contract.json";changed.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"champion/cohort contract"):
                module.extract(changed,checkpoint,cache,root/"wrong-contract",manifest_sha256=module.digest_file(changed))


if __name__=="__main__":unittest.main()
