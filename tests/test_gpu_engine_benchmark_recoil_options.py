"""Ordinary named-option wiring only; no weights/model/device invocation."""
import ast
from pathlib import Path
import subprocess,unittest
from unittest.mock import patch
from types import SimpleNamespace
from gpu_engine import benchmark_retained_games as benchmark,engine_retained_encoder_driver as driver

class BenchmarkRecoilOptionsTests(unittest.TestCase):
 def test_explicit_original_recoil_and_direct_handles_forward_before_weights(self):
  request=object();own=object();recoil=object();direct=object();record=dict(request_compiled=request,catalog=SimpleNamespace(checkpoint_sha256='a'*64))
  with patch.object(benchmark.engine_own_facts,'_checked',return_value=record),patch.object(benchmark.driver,'prepare',side_effect=ValueError('DEVICE_BOUNDARY')) as prepare,patch.object(benchmark.precision_benchmark,'digest',side_effect=AssertionError('weights touched')):
   with self.assertRaisesRegex(ValueError,'DEVICE_BOUNDARY'):
    benchmark.prepare(request,own,'/missing/weights',expected_weights_sha256='b'*64,expected_checkpoint_sha256='a'*64,
      enable_unqualified=True,direct_hit_compiled=direct,direct_hit_mode='kernel-event',recoil_compiled=recoil,recoil_mode='kernel-event',frame_input_mode='source-witness')
  kwargs=prepare.call_args.kwargs;self.assertIs(kwargs['recoil_compiled'],recoil);self.assertIs(kwargs['direct_hit_compiled'],direct)
  self.assertEqual(kwargs['recoil_mode'],'kernel-event');self.assertTrue(kwargs['account_source_sites']);self.assertTrue(kwargs['track_kernel_risks'])
 def test_invalid_closed_mode_and_missing_actual_capability_refuse_at_host(self):
  for mode in (True,'bad'):
   with self.assertRaisesRegex(ValueError,'recoil mode'):
    driver.prepare(None,None,fake_forward=lambda x:x,recoil_mode=mode)
  for kwargs in ({'recoil_compiled':object()},{'recoil_mode':'kernel-event'},{'recoil_mode':'kernel-event','recoil_compiled':object(),'direct_hit_compiled':object(),'direct_hit_mode':'kernel-event','account_source_sites':True}):
   with self.assertRaises(ValueError):driver.prepare(None,None,fake_forward=lambda x:x,**kwargs)
 def test_all_measurement_and_model_functions_ast_unchanged(self):
  def funcs(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name!='prepare'}
  old=subprocess.check_output(['git','show','58ec3459:gpu_engine/benchmark_retained_games.py'],text=True)
  self.assertEqual(funcs(old),funcs(Path('gpu_engine/benchmark_retained_games.py').read_text()))
 def test_fresh_benchmark_import_and_all_frozen_source_pins(self):
  import os,sys
  run=subprocess.run([sys.executable,'-c','from gpu_engine import benchmark_retained_games; from gpu_engine import engine_source_frame_witness as w; w._source(); print(len(w.SOURCE_PINS))'],text=True,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(run.stdout.strip(),'50')
