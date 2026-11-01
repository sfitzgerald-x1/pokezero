"""Finite census orchestration fakes only; no engine/model/Native/scientific chance sampling."""
from types import SimpleNamespace
from unittest import mock
import unittest
import numpy as np
from gpu_engine import benchmark_capture_gap_census as census

class CensusTests(unittest.TestCase):
 def test_declared_plan_and_independent_uniform_legal_fallback(self):
  p=census.Plan(fallback_seed=732110001);self.assertIs(p.validate(),p)
  for kw in (dict(batch=512),dict(census_games=True),dict(max_steps=2049),dict(max_turn=1001),dict(timing_steps=1),dict(fallback='slot0'),dict(fallback_seed=True)):
   with self.subTest(kw=kw),self.assertRaises(ValueError):census.Plan(**kw).validate()
  legal=np.zeros((8,2,9),bool);legal[:,:,1]=True;legal[:,:,7]=True;requested=np.ones((8,2),bool);active=np.ones(8,bool);streams=np.arange(801,809,dtype=np.int32)
  original=legal.copy()
  first,bad=census.legal_fallback(legal,requested,active,seed=p.fallback_seed,streams=streams,step=3)
  repeated,_=census.legal_fallback(legal,requested,active,seed=p.fallback_seed,streams=streams,step=3)
  np.testing.assert_array_equal(first,repeated);self.assertFalse(bad.any());self.assertTrue(np.isin(first,[1,7]).all());np.testing.assert_array_equal(legal,original)
  # Reordering/compacting another row cannot change one stream's fallback.
  other,_=census.legal_fallback(legal[::-1],requested,active,seed=p.fallback_seed,streams=streams[::-1],step=3)
  np.testing.assert_array_equal(first,other[::-1])
  legal[2,1]=False;requested[4,1]=False
  result,bad=census.legal_fallback(legal,requested,active,seed=p.fallback_seed,streams=streams,step=4)
  self.assertEqual(np.flatnonzero(bad).tolist(),[2]);self.assertEqual(int(result[4,1]),0)
 def test_permanent_gap_and_fresh_sites_count_occurrences_and_unique_games(self):
  p=census.Plan();ids=np.arange(10001,12049,dtype=np.int32);l=census.Ledger(p,ids);mask=np.zeros(2048,bool);mask[[0,1,600]]=True
  l.mark_capture(mask,step=0,packets=lambda r:dict(row=r));l.mark_capture(np.zeros(2048,bool),step=1,packets=lambda r:{})
  self.assertFalse(l.capture_good[[0,1,600]].any());self.assertTrue(l.capture_good[2])
  for step in(0,1):
   l.observe(step=step,active=mask,executed=mask,engine_failed=np.zeros(2048,bool),done=np.zeros(2048,bool),turns=np.ones(2048,np.int32),packet=lambda r:dict(unhandled_sites=[5,5]if r==0 else[9],raw_unsupported_kinds=[],raw_unsupported_moves=[dict(move_id=17,name='literalboundmove')]if r==0 else[],engine_failures=[]),terminal=lambda r:{})
  summary=l.summary();by={(x['role'],x['kind']):x for x in summary['unsupported']}
  self.assertEqual(by[('census','site:5')]['occurrences'],4);self.assertEqual(by[('census','site:5')]['games_affected'],1)
  self.assertEqual(by[('timing-only','site:9')]['occurrences'],2);self.assertEqual(summary['census']['original_denominator'],512);self.assertEqual(summary['timing_only']['original_row_denominator'],1536)
  self.assertTrue(l.active(31)[600]);self.assertFalse(l.active(32)[600]);self.assertTrue(l.active(32)[0])
  self.assertLessEqual(len(by[('census','site:5')]['examples']),p.max_examples_per_kind)
  self.assertEqual(by[('census','raw-move:17:literalboundmove')]['occurrences'],2);self.assertEqual(by[('census','raw-move:17:literalboundmove')]['games_affected'],1)
 def test_fake_run_keeps_sampling_once_no_postgap_policy_credit_and_all_denominators(self):
  p=census.Plan();ids=np.arange(20001,22049,dtype=np.int32);calls=[]
  class Fake:
   jax=SimpleNamespace(block_until_ready=lambda x:x)
   def __init__(self):
    self.engine=SimpleNamespace(done=np.zeros(2048,bool),turn=np.zeros(2048,np.int32));self.last_packet=[{}for _ in ids]
   def cache_snapshot(self):return dict(frame=1,encode_fast=1)
   def requested(self):return np.ones((2048,2),bool)
   def mask(self,s):return np.ones((2048,2,9),bool)
   def policy(self,active,l,step,seed):
    capture=np.zeros(2048,bool);capture[0]=True;l.mark_capture(capture,step=step,packets=lambda r:{'first':True})
    calls.append(('policy',step,int((active&l.capture_good).sum())))
    return np.full((2048,2),8,np.int32),dict(bf16_physical_owner_rows=4096,admitted_policy_owner_rows=int((active&l.capture_good).sum())*2,bf16_compile_ms=1 if step==0 else 0)
   def advance(self,actions,active,scored,chance):
    calls.append(('advance',int(scored.sum()),int(active.sum()),chance));self.engine.turn+=active;self.engine.done|=active
    self.last_packet=[dict(unhandled_sites=[22]if r==0 else[],raw_unsupported_kinds=[],fresh_call_risk=0,engine_failures=[])for r in range(2048)]
    self.assert_actions=actions[0].copy();return active.copy(),np.zeros(2048,bool),{}
   def capture_failure(self,active):return np.zeros(2048,bool),np.zeros((2048,2),np.int64)
   def terminal(self,row):return dict(done=bool(self.engine.done[row]),winner=0,hp=[[1]*6,[0]*6])
  fake=Fake()
  with mock.patch('gpu_engine.transition.sample_chance_inputs',return_value='literal-fake-once')as sample:
   report=census._run(fake,p,ids,sampling_seed=729110001)
  self.assertEqual(sample.call_count,1);self.assertEqual(report['census']['engine_terminal_games'],512)
  self.assertEqual(report['timing_only']['engine_terminal_rows'],1536);self.assertEqual(report['census']['capture_gap_games'],1)
  self.assertEqual(calls[1][1:3],(2047,2048));self.assertEqual(calls[1][3],'literal-fake-once')
  self.assertFalse(report['cohort'][0]['observation_scoring_valid']);self.assertIsNone(report['production_fullgame_throughput']);self.assertFalse(report['full_B4_qualified'])
 def test_caps_engine_failure_and_timing_rows_end_unknown(self):
  p=census.Plan();l=census.Ledger(p,np.arange(30001,32049,dtype=np.int32));active=np.zeros(2048,bool);active[:3]=True
  bad=active.copy();bad[1:]=False;done=np.zeros(2048,bool);done[1]=True;turn=np.zeros(2048,np.int32);turn[2]=1000
  l.observe(step=0,active=active,executed=active,engine_failed=bad,done=done,turns=turn,packet=lambda r:dict(engine_failures=['INVALID']if r==0 else[]),terminal=lambda r:dict(done=True,winner=1,hp='actual-literal-fixture'))
  s=l.summary();self.assertEqual(s['census']['engine_refused_games'],1);self.assertEqual(s['census']['engine_terminal_games'],1);self.assertEqual(s['census']['capped_games'],1)
  self.assertEqual(s['census']['unknown_terminal_games'],511);self.assertFalse(l.active(1)[:3].any());self.assertIsNone(s['cohort'][0]['terminal']);self.assertEqual(s['cohort'][1]['terminal']['winner'],1)
class WitnessPredicateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import test_gpu_engine_intimidate_owning_event as fixture
  fixture.InitialIntimidateOwningTests.setUpClass();cls.f=fixture.InitialIntimidateOwningTests()
 def test_exact_original_predicate_padding_context_and_old_report_constructor(self):
  import ast,subprocess
  from pathlib import Path
  import jax.numpy as j
  from gpu_engine import engine_source_frame_witness as witness,engine_lifecycle_callbacks as life,engine_execution_completeness as ex
  path='gpu_engine/engine_source_frame_witness.py'
  old=ast.parse(subprocess.check_output(['git','show','414d57a8:'+path],text=True));new=ast.parse(Path(path).read_text())
  select=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
  original=select(old,'_raw_complete');factored=select(new,'_raw_complete_words')
  self.assertEqual(ast.dump(ast.Module(body=original.body[:-1],type_ignores=[])),ast.dump(ast.Module(body=factored.body[:-1],type_ignores=[])))
  self.assertEqual(ast.dump(original.body[-1].value.args[0]),ast.dump(factored.body[-1].value))
  original.name='_original_raw_complete';module=ast.fix_missing_locations(ast.Module(body=[original],type_ignores=[]));env=dict(vars(witness));exec(compile(module,'original-frozen-raw-predicate','exec'),env)
  f=self.f
  for b in(1,8):
   ids=j.arange(81001,81001+b,dtype=j.int32);cursor=life.register(ids,**f.kw);t=life._tape(cursor,f.producer,64);ledger=ex.initialize(ids,compiled=f.account,capacity=16,receipt_capacity=64);t=t._replace(execution=ex.Context(ledger,f.account.device))
   tables=SimpleNamespace(risk=f.risk.device,secondary_weather=j.zeros(len(f.table.power),bool),direct_hit_binding=j.zeros(8,j.uint32),substitute_binding=j.zeros(8,j.uint32),toxic_binding=j.zeros(8,j.uint32))
   cases=[t,t._replace(count=j.ones(b,j.int32),words=t.words.at[:,0,0].set(99)),t._replace(count=j.ones(b,j.int32),words=t.words.at[:,0,0].set(1).at[:,0,5].set(1)),t._replace(words=t.words.at[:,20,0].set(99))]
   for case in cases:
    words=witness._raw_complete_words(case,tables);self.assertEqual(words.shape,(b,64));self.assertEqual(words.dtype,j.bool_)
    np.testing.assert_array_equal(env['_original_raw_complete'](case,tables),witness._raw_complete(case,tables));np.testing.assert_array_equal(j.all(words,axis=1),witness._raw_complete(case,tables))
    self.assertTrue(bool(words[:,1:].all()))
  legacy=witness.Report(*([None]*(len(witness.Report._fields)-1)));self.assertIsNone(legacy.raw_complete_words)
  # No after-ledger reconstruction: SAME source mask feeds both original
  # reduction and appended data at prejoin source location.
  function=select(new,'consume_pair_device');text=ast.unparse(function)
  self.assertIn('raw_good_words = _raw_complete_words(tape, tables)',text);self.assertIn('raw = jnp.all(raw_good_words, axis=1)',text)
  self.assertLess(text.index('raw_good_words ='),text.index('execution.join_consumed_owners'))
  self.assertIn('raw_good_words)',text)

class JournalTests(unittest.TestCase):
 def test_atomic_exclusive_cap_and_nonfinite_refusal(self):
  import tempfile,json
  from pathlib import Path
  with tempfile.TemporaryDirectory()as tmp:
   path=Path(tmp)/'owned.json';journal=census.Journal(path,max_bytes=1024);journal.write({'status':'RUNNING','unknown':2048})
   before=path.read_bytes()
   with self.assertRaises(FileExistsError):census.Journal(path,max_bytes=1024)
   with self.assertRaises(ValueError):journal.write({'oversize':'x'*2000})
   self.assertEqual(path.read_bytes(),before)
   with self.assertRaises(ValueError):journal.write({'bad':float('nan')})
   self.assertEqual(json.loads(path.read_text())['unknown'],2048)

if __name__=='__main__':unittest.main()
