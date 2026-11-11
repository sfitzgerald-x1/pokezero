"""Named source-mode wiring only; no weights/model/device invocation."""
import ast,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from gpu_engine import benchmark_retained_games as benchmark,engine_retained_encoder_driver as driver

class BenchmarkCalmMindOptionsTests(unittest.TestCase):
 def test_explicit_handle_and_mode_forward_before_weights(self):
  request=object();own=object();helper=object();record=dict(request_compiled=request,catalog=SimpleNamespace(checkpoint_sha256='a'*64))
  with patch.object(benchmark.engine_own_facts,'_checked',return_value=record),patch.object(benchmark.driver,'prepare',side_effect=ValueError('SOURCE_BOUNDARY')) as prepare,patch.object(benchmark.precision_benchmark,'digest',side_effect=AssertionError('weights touched')):
   with self.assertRaisesRegex(ValueError,'SOURCE_BOUNDARY'):benchmark.prepare(request,own,'/missing',expected_weights_sha256='b'*64,expected_checkpoint_sha256='a'*64,enable_unqualified=True,calm_mind_compiled=helper,calm_mind_mode='kernel-event',frame_input_mode='source-witness')
  kwargs=prepare.call_args.kwargs;self.assertIs(kwargs['calm_mind_compiled'],helper);self.assertEqual(kwargs['calm_mind_mode'],'kernel-event');self.assertTrue(kwargs['account_source_sites']);self.assertTrue(kwargs['track_kernel_risks'])
 def test_default_path_and_invalid_modes_preserve_original_boundary(self):
  request=object();record=dict(request_compiled=request,catalog=SimpleNamespace(checkpoint_sha256='a'*64))
  with patch.object(benchmark.engine_own_facts,'_checked',return_value=record),patch.object(benchmark.driver,'prepare',side_effect=AssertionError('unexpected early source prepare')),patch.object(benchmark.precision_benchmark,'digest',side_effect=ValueError('WEIGHTS_BOUNDARY')):
   with self.assertRaisesRegex(ValueError,'WEIGHTS_BOUNDARY'):benchmark.prepare(request,object(),'/missing',expected_weights_sha256='b'*64,expected_checkpoint_sha256='a'*64,enable_unqualified=True)
  for kwargs in ({'calm_mind_mode':True},{'calm_mind_mode':'bad'},{'calm_mind_compiled':object()},{'calm_mind_mode':'kernel-event'},{'calm_mind_mode':'kernel-event','calm_mind_compiled':object(),'account_source_sites':True}):
   with self.assertRaises(ValueError):driver.prepare(None,None,fake_forward=lambda x:x,**kwargs)
 def test_model_measurement_admission_functions_unchanged(self):
  def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name not in ('prepare','measure')}
  old=subprocess.check_output(['git','show','7bc45a72:gpu_engine/benchmark_retained_games.py'],text=True);self.assertEqual(functions(old),functions(Path('gpu_engine/benchmark_retained_games.py').read_text()))


class ReviewedTimingCompositionTests(unittest.TestCase):
 def test_historical_composition_retains_reviewed_measure(self):
  # This assertion concerns the recorded handler/timing merge, before later
  # capture work. It does not claim the current capture producer is unchanged.
  current=subprocess.check_output(["git","show","e5d38e3cf802c355ee944c440666dd84b4948f3c:gpu_engine/benchmark_retained_games.py"],text=True)
  reviewed=subprocess.check_output(["git","show","4baa33e96d8d0c944374c81fd8cd44cc8b6bd26a:gpu_engine/benchmark_retained_games.py"],text=True)
  def measure(text):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="measure"),include_attributes=False)
  self.assertEqual(measure(current),measure(reviewed))
