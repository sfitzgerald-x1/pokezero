"""Named ordinary host option forwarding; no model or device work."""
import ast,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from gpu_engine import benchmark_retained_games as benchmark,engine_retained_encoder_driver as driver

class BenchmarkThunderWaveOptionsTests(unittest.TestCase):
 def test_explicit_capability_reaches_driver_before_any_weights(self):
  request=object();own=object();thunder_wave=object();record=dict(request_compiled=request,catalog=SimpleNamespace(checkpoint_sha256='a'*64))
  with patch.object(benchmark.engine_own_facts,'_checked',return_value=record),patch.object(benchmark.driver,'prepare',side_effect=ValueError('DEVICE_BOUNDARY')) as prepare,patch.object(benchmark.precision_benchmark,'digest',side_effect=AssertionError('weights touched')):
   with self.assertRaisesRegex(ValueError,'DEVICE_BOUNDARY'):
    benchmark.prepare(request,own,'/missing/weights',expected_weights_sha256='b'*64,expected_checkpoint_sha256='a'*64,enable_unqualified=True,thunder_wave_compiled=thunder_wave,thunder_wave_mode='kernel-event',frame_input_mode='source-witness')
  kwargs=prepare.call_args.kwargs;self.assertIs(kwargs['thunder_wave_compiled'],thunder_wave);self.assertEqual(kwargs['thunder_wave_mode'],'kernel-event');self.assertTrue(kwargs['account_source_sites']);self.assertTrue(kwargs['track_kernel_risks'])
 def test_closed_invalid_and_missing_actual_runtime_refuse(self):
  for mode in (True,'bad'):
   with self.assertRaisesRegex(ValueError,'Thunder Wave mode'):driver.prepare(None,None,fake_forward=lambda x:x,thunder_wave_mode=mode)
  for kwargs in ({'thunder_wave_compiled':object()},{'thunder_wave_mode':'kernel-event'},{'thunder_wave_mode':'kernel-event','thunder_wave_compiled':object(),'account_source_sites':True,'public_mode':'retained-full'}):
   with self.assertRaises(ValueError):driver.prepare(None,None,fake_forward=lambda x:x,**kwargs)
 def test_measurement_and_model_functions_unchanged(self):
  def functions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name not in ('prepare','measure')}
  self.assertEqual(functions(Path('gpu_engine/benchmark_retained_games.py').read_text()),functions(subprocess.check_output(['git','show','e5d38e3c:gpu_engine/benchmark_retained_games.py'],text=True)))


class ReviewedTimingCompositionTests(unittest.TestCase):
 def test_historical_composition_retains_reviewed_measure(self):
  # Bind this historical equality to the actual Thunder Wave handler change;
  # later capture additions are outside this unchanged-timing assertion.
  current=subprocess.check_output(["git","show","75ddb822d6fc3ecc4e9281b74735cab4578711dc:gpu_engine/benchmark_retained_games.py"],text=True)
  reviewed=subprocess.check_output(["git","show","4baa33e96d8d0c944374c81fd8cd44cc8b6bd26a:gpu_engine/benchmark_retained_games.py"],text=True)
  def measure(text):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="measure"),include_attributes=False)
  self.assertEqual(measure(current),measure(reviewed))
