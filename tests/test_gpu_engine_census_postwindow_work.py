"""Whole finite orchestration with inert fakes; no engine/model/chance calls."""
import ast,copy,importlib.util,itertools,json,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
from gpu_engine import benchmark_capture_gap_census as c
BASE='ad123786c29ef068b53ca6f707db0313098856a6'
AUDITOR=Path(__file__).resolve().parents[1]/'scripts/audit_capture_gap_census_postwindow_stored.py'

class Fake:
 jax=SimpleNamespace(block_until_ready=lambda x:x)
 def __init__(self,streams,late):
  self.streams=streams;self.late=late;self.calls=[];self.actions=[];self.scored=[];self.engine=SimpleNamespace(done=np.zeros(2048,bool),turn=np.zeros(2048,np.int32),step=0);self.last_packet=[{}for _ in streams]
 def cache_snapshot(self):return dict(frame=1,encode_fast=1)
 def mask(self,s):
  self.calls.append(('legal',s.step));out=np.zeros((2048,2,9),bool);out[:,:,[1,7]]=True;return out
 def requested(self):self.calls.append(('requested',self.engine.step));return np.ones((2048,2),bool)
 def check_physical_source(self):self.calls.append(('source_guard',self.engine.step))
 def policy(self,active,ledger,step,seed):
  self.check_physical_source()
  self.calls.append(('policy',step,seed));bad=np.ones(2048,bool)
  if self.late and step<33:bad[0]=False
  ledger.mark_capture(bad,step=step,packets=lambda row:dict(kind='literal-fake-capture',row=row))
  admitted=2*int((active&ledger.capture_good).sum())
  return np.full((2048,2),7,np.int32),dict(bf16_physical_owner_rows=4096,admitted_policy_owner_rows=admitted,shadow_discarded_owner_rows=4096-admitted,bf16_compile_ms=1. if step==0 else 0.,bf16_forward_ms=2.,shadow_actual_encoder_ms=[1.,1.],shadow_actual_encoder_rows=2048,shadow_scope='fake-executed')
 def advance(self,actions,active,scored,chance):
  step=self.engine.step;self.calls.append(('advance',step,chance));self.actions.append(actions.copy());self.scored.append(scored.copy());self.engine.turn+=active;self.engine.done|=active&(step==34);self.engine.step+=1
  self.last_packet=[dict(unhandled_sites=[10]if row<512 else[5],raw_unsupported_kinds=[1],raw_unsupported_moves=[dict(move_id=17,name='literal')],fresh_call_risk=0,engine_failures=[])for row in range(2048)]
  return active.copy(),np.zeros(2048,bool),dict(original_request_rows=int(scored.sum()))
 def capture_failure(self,active):self.calls.append(('capture_check',self.engine.step));return np.zeros(2048,bool),np.zeros((2048,2),np.int32)
 def terminal(self,row):return dict(done=True,winner=0,hp=[[1]*6,[0]*6],max_hp=[[1]*6,[1]*6],stream_id=int(self.streams[row]),turn=int(self.engine.turn[row]))

class PostwindowTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  node=next(n for n in ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/benchmark_capture_gap_census.py'],text=True)).body if isinstance(n,ast.FunctionDef)and n.name=='_run');node.name='_original_run';env=dict(vars(c));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'original-census-run','exec'),env);cls.original=staticmethod(env['_original_run'])
  spec=importlib.util.spec_from_file_location('postwindow_auditor',AUDITOR);cls.auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.auditor)
 def run_fake(self,run,late):
  streams=np.arange(880001,882049,dtype=np.int32);backend=Fake(streams,late);plan=c.Plan(max_steps=40,fallback_seed=732110001);ticks=itertools.count();chance=[]
  def sample(engine):chance.append(engine.step);return ('fake-original-chance',engine.step)
  with mock.patch('gpu_engine.transition.sample_chance_inputs',side_effect=sample),mock.patch.object(c.time,'time',return_value=12345.),mock.patch.object(c.time,'perf_counter_ns',return_value=777):
   report=run(backend,plan,streams,sampling_seed=729110001,clock=lambda:next(ticks)*.01)
  return backend,report,chance
 def test_whole_ledgers_actions_packets_chance_equal_and_first32_retained(self):
  for late in(False,True):
   old,before,oldchance=self.run_fake(self.original,late);new,after,newchance=self.run_fake(c._run,late)
   for key in ('cohort','census','timing_only','unsupported','unsupported_by_source','per_step','callback_counts_complete','warm_launch_shape','stop','plan','full_B4_qualified','fidelity_qualified','strength_qualified'):
    self.assertEqual(before[key],after[key],key)
   self.assertEqual(oldchance,newchance);self.assertEqual(newchance,list(range(35)))
   np.testing.assert_array_equal(old.actions,new.actions);np.testing.assert_array_equal(old.scored,new.scored)
   self.assertEqual(old.last_packet,new.last_packet)
   keep=lambda calls:[x for x in calls if x[0]not in('policy','capture_check')]
   self.assertEqual(keep(old.calls),keep(new.calls))
   executed=34 if late else 32
   self.assertEqual([x[1]for x in new.calls if x[0]=='policy'],list(range(executed)))
   self.assertEqual(len([x for x in new.calls if x[0]=='capture_check']),executed)
   # Full new stored-checker path on this inert complete envelope, not merely
   # the new branch predicate. All numbers/terminals remain synthetic tests.
   package={'fake.py':'b'*64};data=copy.deepcopy(after)
   data.update(source=dict(commit='a'*40,package_sha256=package),pool_receipt=dict(pool_sha256='c'*64),constructor_seed=1,sampling_seed=729110001,original_stream_ids=new.streams.tolist(),original_team_indices=np.arange(4096).reshape(2048,2).tolist())
   prespec=dict(batch=2048,census_games=512,timing_only_rows=[512,2048],source_commit='a'*40,actual_source_map_sha256=self.auditor.canonical(package),team_seed_start=726110000,pool_generation=dict(new_output_only=True,no_reuse=True,producer='gpu_engine.team_pool',teams=4096,seed_start=726110000),constructor_seed=1,sampling_seed=729110001,stream_start=int(new.streams[0]),fallback_seed=after['plan']['fallback_seed'])
   audit=self.auditor.check(data,prespec);self.assertEqual(audit['timing']['postwindow_skipped_sample_steps'],list(range(executed,35)));self.assertEqual(audit['sum_executed_row_steps'],sum(row['executed_steps']for row in after['cohort']))
   json.dumps(data,allow_nan=False)
   for i,sample in enumerate(after['samples']):
    skipped=i>=executed;self.assertEqual(sample['policy_work_skipped'],skipped);self.assertEqual(self.auditor.policy_work(sample,i),skipped)
    if skipped:
     self.assertEqual(sample['bf16_physical_owner_rows'],0);self.assertEqual(sample['shadow_actual_encoder_rows'],0);self.assertEqual(sample['shadow_actual_encoder_ms'],[]);self.assertNotIn('bf16_forward_ms',sample);self.assertNotIn('bf16_compile_ms',sample);self.assertFalse(sample['warm_candidate'])
    else:
     for key in ('bf16_physical_owner_rows','admitted_policy_owner_rows','shadow_actual_encoder_rows','shadow_discarded_owner_rows','bf16_forward_ms','bf16_compile_ms','shadow_actual_encoder_ms'):self.assertEqual(sample[key],before['samples'][i][key])
 def test_skipped_work_retains_exact_runtime_source_guard(self):
  runtime=object();session=object();seen=[]
  fake=SimpleNamespace(runtime=runtime,session=session,retained=SimpleNamespace(_checked=lambda r,session:seen.append((r,session))))
  c._Runtime.check_physical_source(fake);self.assertEqual(seen,[(runtime,session)])
  fake.retained._checked=mock.Mock(side_effect=ValueError('source changed'))
  with self.assertRaisesRegex(ValueError,'source changed'):c._Runtime.check_physical_source(fake)
  fake.retained._checked.assert_called_once_with(runtime,session=session)

 def test_stored_contract_refuses_premature_skips_positive_rows_and_fake_zero_time(self):
  normal=dict(active_engine_rows=512,policy_input_capture_valid_rows=0,policy_work_skipped=False,policy_work_skip_reason=None,bf16_physical_owner_rows=4096,admitted_policy_owner_rows=0,shadow_discarded_owner_rows=4096,shadow_actual_encoder_rows=2048,bf16_forward_ms=2.,bf16_compile_ms=0.,shadow_actual_encoder_ms=[1.,1.],phases_ms=dict(original_request_rows=0))
  skipped=dict(normal,policy_work_skipped=True,policy_work_skip_reason='POST_TIMING_WINDOW_NO_ACTIVE_CAPTURE_VALID_GAME',bf16_physical_owner_rows=0,shadow_discarded_owner_rows=0,shadow_actual_encoder_rows=0,shadow_actual_encoder_ms=[]);skipped.pop('bf16_forward_ms');skipped.pop('bf16_compile_ms')
  self.assertFalse(self.auditor.policy_work(normal,31));self.assertTrue(self.auditor.policy_work(skipped,32))
  for step,change in ((31,{}),(32,dict(policy_input_capture_valid_rows=1)),(32,dict(bf16_forward_ms=0.)),(32,dict(bf16_compile_ms=None)),(32,dict(shadow_actual_encoder_rows=1)),(32,dict(admitted_policy_owner_rows=1)),(32,dict(policy_work_skip_reason='none')),(32,dict(phases_ms=dict(original_request_rows=1)))):
   with self.subTest(step=step,change=change),self.assertRaises(ValueError):self.auditor.policy_work(dict(skipped,**change),step)
  with self.assertRaises(ValueError):self.auditor.policy_work(normal,32)

if __name__=='__main__':unittest.main()
