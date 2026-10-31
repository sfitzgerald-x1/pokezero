"""Named ordinary host option forwarding; no model or device work."""
import ast,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from gpu_engine import benchmark_retained_games as benchmark,engine_retained_encoder_driver as driver

class BenchmarkSpikesOptionsTests(unittest.TestCase):
 def test_explicit_capability_reaches_driver_before_any_weights(self):
  request=object();own=object();spikes=object();record=dict(request_compiled=request,catalog=SimpleNamespace(checkpoint_sha256='a'*64))
  with patch.object(benchmark.engine_own_facts,'_checked',return_value=record),patch.object(benchmark.driver,'prepare',side_effect=ValueError('DEVICE_BOUNDARY')) as prepare,patch.object(benchmark.precision_benchmark,'digest',side_effect=AssertionError('weights touched')):
   with self.assertRaisesRegex(ValueError,'DEVICE_BOUNDARY'):
    benchmark.prepare(request,own,'/missing/weights',expected_weights_sha256='b'*64,expected_checkpoint_sha256='a'*64,enable_unqualified=True,spikes_compiled=spikes,spikes_mode='kernel-event',frame_input_mode='source-witness')
  kwargs=prepare.call_args.kwargs;self.assertIs(kwargs['spikes_compiled'],spikes);self.assertEqual(kwargs['spikes_mode'],'kernel-event');self.assertTrue(kwargs['account_source_sites']);self.assertTrue(kwargs['track_kernel_risks'])
 def test_closed_invalid_and_missing_actual_runtime_refuse(self):
  for mode in (True,'bad'):
   with self.assertRaisesRegex(ValueError,'Spikes mode'):driver.prepare(None,None,fake_forward=lambda x:x,spikes_mode=mode)
  for kwargs in ({'spikes_compiled':object()},{'spikes_mode':'kernel-event'},{'spikes_mode':'kernel-event','spikes_compiled':object(),'account_source_sites':True,'public_mode':'retained-full'}):
   with self.assertRaises(ValueError):driver.prepare(None,None,fake_forward=lambda x:x,**kwargs)
 def test_measurement_and_model_functions_unchanged(self):
  def functions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name!='prepare'}
  self.assertEqual(functions(Path('gpu_engine/benchmark_retained_games.py').read_text()),functions(subprocess.check_output(['git','show','ea7c5059:gpu_engine/benchmark_retained_games.py'],text=True)))
