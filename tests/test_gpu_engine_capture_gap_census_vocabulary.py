"""Original checked request metadata contract, without constructor/model/game calls."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import numpy as np
from gpu_engine import benchmark_capture_gap_census as census

class VocabularyContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import test_gpu_engine_request_source as fixture
  fixture.RequestSourceTests.setUpClass()
  from gpu_engine import engine_request_source as requests
  cls.record=requests._checked(fixture.RequestSourceTests.compiled)
  cls.metadata=fixture.RequestSourceTests.metadata
 def test_original_compiled_json_move_order_and_geometry(self):
  self.assertIs(type(self.record['metadata']),str)
  with self.assertRaises(TypeError):self.record['metadata']['vocabulary']['moves']
  names=census._bound_engine_move_names(self.record)
  self.assertEqual(names,tuple(self.metadata['vocabulary']['moves']))
  self.assertEqual(names[2],'hiddenpowerice')
  self.assertEqual(len(names),len(self.record['engine_table'].power))
  for bad in ('{',json.dumps(dict(vocabulary=dict(moves='surf'))),json.dumps(dict(vocabulary=dict(moves=['surf'])))):
   with self.subTest(bad=bad),self.assertRaises((ValueError,TypeError,KeyError)):
    census._bound_engine_move_names(dict(self.record,metadata=bad))
 def test_envelope_uses_original_record_before_runtime_construction(self):
  from gpu_engine import benchmark_retained_games as retained,source_provenance
  plan=census.Plan();runtime=SimpleNamespace(binding='bound-fake-runtime')
  pool=SimpleNamespace(arrays={'species':np.zeros((4096,6),np.int32)})
  with tempfile.TemporaryDirectory()as tmp:
   path=Path(tmp)/'progress.json'
   def stopped(*args):
    # A real header is present before constructor entry; no engine is invoked.
    actual=json.loads(path.read_bytes())
    self.assertEqual(actual['status'],'ATTEMPTED_CONSTRUCTION_UNKNOWN')
    self.assertEqual(actual['engine_move_names'],self.metadata['vocabulary']['moves'])
    self.assertEqual(sum(row['executed_steps']for row in actual['cohort']),0)
    raise RuntimeError('deliberate fake boundary before any constructor')
   with mock.patch.object(retained,'_source_snapshot',return_value={'fake':'source'}),mock.patch.object(source_provenance,'_pool',return_value={'fake':'pool'}),mock.patch.object(retained,'_checked',return_value={'prepared':'original-contract-fixture'}),mock.patch.object(retained.driver,'_checked',return_value={'request':self.record}),mock.patch.object(census,'_Runtime',side_effect=stopped)as constructor:
    with self.assertRaisesRegex(RuntimeError,'deliberate fake boundary'):
     census.measure(runtime,pool,np.arange(4096,dtype=np.int64).reshape(2048,2),stream_ids=np.arange(1,2049,dtype=np.int32),plan=plan,constructor_seed=1,sampling_seed=2,enable_diagnostic=True,journal_path=path)
    self.assertEqual(constructor.call_count,1)
   final=json.loads(path.read_bytes());self.assertEqual(final['actual_exit'],1)
   self.assertEqual(final['engine_move_names'],self.metadata['vocabulary']['moves'])
   self.assertEqual(sum(row['executed_steps']for row in final['cohort']),0)

if __name__=='__main__':unittest.main()
