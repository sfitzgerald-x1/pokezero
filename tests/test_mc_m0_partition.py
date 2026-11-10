"""Qualification/assignment guards; no simulator, checkpoint or Torch required."""
import json
from pathlib import Path
import tempfile
import unittest
from gpu_engine import mc_m0_partition as partition
from gpu_engine import mc_m0 as core, mc_root_labeller as lab, fp_reply_data as fp


class PartitionGuards(unittest.TestCase):
    def test_changed_prefix_is_rejected_even_when_encoded_root_is_identical(self):
        root=dict(seed=1,original_observation_sha256='same',actions=[0,1])
        frozen=dict(root=root,source_hashes={'telemetry/1.jsonl':'1'*64})
        reconstructed=dict(root=root,source_hashes={'telemetry/1.jsonl':'2'*64})
        with self.assertRaisesRegex(ValueError,'source inputs'):
            partition.require_same_root(frozen,reconstructed,root)
        reconstructed['source_hashes']=frozen['source_hashes']
        partition.require_same_root(frozen,reconstructed,root)

    def test_assignments_cannot_repeat_or_escape_frozen_roster(self):
        self.assertEqual(partition.assignment('1,3,49'),(1,3,49))
        for bad in ('','1,1','-1','50','1.5','true'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):partition.assignment(bad)

    def test_qualification_binds_config_core_and_all_fifty_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=dict(roots=[dict(seed=i) for i in range(50)])
            core.write(root/'config.json',config)
            manifest=dict(config_sha256=core.sha(root/'config.json'),source_sha256={
                'mc_m0.py':core.sha(core.__file__),'mc_root_labeller.py':core.sha(lab.__file__),'fp_reply_data.py':core.sha(fp.__file__)})
            core.write(root/'manifest.json',manifest)
            smoke=[dict(kind=kind,status='PASS') for kind in ('self','foulplay')]
            core.write(root/'runtime-smoke.json',smoke)
            ready=[dict(index=i,status='READY',root=r) for i,r in enumerate(config['roots'])]
            core.write(root/'prepared.json',ready)
            actual,refs,_=partition.qualify(root)
            self.assertEqual(actual,config);self.assertEqual(set(refs),set(range(50)))
            (root/'prepared.json').write_text(json.dumps(ready[:-1]))
            with self.assertRaisesRegex(ValueError,'50 READY'):partition.qualify(root)
            (root/'prepared.json').write_text(json.dumps(ready))
            smoke[0]['status']='FAILED';(root/'runtime-smoke.json').write_text(json.dumps(smoke))
            with self.assertRaisesRegex(ValueError,'qualifications'):partition.qualify(root)
            smoke[0]['status']='PASS';(root/'runtime-smoke.json').write_text(json.dumps(smoke))
            manifest['source_sha256']['mc_root_labeller.py']='0'*64
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'core changed'):partition.qualify(root)
