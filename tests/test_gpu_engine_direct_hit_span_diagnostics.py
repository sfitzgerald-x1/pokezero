"""Original emitter/span branches + literal transport; no engine/FULL/model."""
import ast,subprocess,sys,os,unittest
from copy import deepcopy
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_direct_hit_controls as direct,engine_execution_completeness as ex
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public,engine_outcome_diagnostics as d
import test_gpu_engine_outcome_diagnostics as fixture
BASE='de4180168925e163cb87d3220d58975086f8e96f'
def old_functions(module,path):
 ns=dict(vars(module));tree=ast.parse(subprocess.check_output(['git','show','b8092a85:'+path],text=True))
 for n in tree.body:
  if isinstance(n,ast.FunctionDef):exec(compile(ast.Module(body=[n],type_ignores=[]),'<frozen original>','exec'),ns)
 return ns
class DirectSpanDiagnosticTests(unittest.TestCase):
 def test_literal_transport_two_phases_missing_malformed_inactive_and_capacity(self):
  t,_=fixture.OutcomeDiagnosticTests().tape();b=8;z=j.zeros(b,j.int32);active=j.array([True,False]*4);values=d.direct_meta(t);values.update(move=z+250,requested=active,valid=j.ones(b,bool),write_old_hp=z+322,write_new_hp=z+19,registered_bound=None,effective_eligible=j.ones(b,j.float32))
  a=d.record_direct_on_tape(t,values,phase=0,active=active);a=d.record_direct_on_tape(a,dict(values,semantic=~active),phase=1,active=active)
  self.assertIs(a.words,t.words);self.assertIs(a.refusal,t.refusal);self.assertIs(a.execution.ledger,t.execution.ledger);self.assertIs(a.execution.risks,t.execution.risks)
  for x,y in zip(a.execution.diagnostics[:25],t.execution.diagnostics[:25],strict=True):np.testing.assert_array_equal(x,y)
  q=d.refused_rows(a.execution.diagnostics,np.array([True]+[False]*7))['rows'][0]['direct_hit'];self.assertEqual(q['capacity_per_phase'],2);self.assertEqual(q['emitter']['count'],1);self.assertEqual(q['span_capture']['count'],1)
  r=q['emitter']['records'][0];self.assertEqual((r['move'],r['write_old_hp'],r['write_new_hp'],r['stream_id'],r['frame']),(250,322,19,12001,5));self.assertIsNone(r['registered_bound']);self.assertNotIn('registered_bound',r['malformed_fields']);self.assertIsNone(r['effective_eligible']);self.assertIn('effective_eligible',r['malformed_fields'])
  full=d.record_direct_on_tape(a,values,phase=0,active=active);over=d.record_direct_on_tape(full,values,phase=0,active=active);self.assertEqual(over.execution.diagnostics.direct_count[:,0].tolist(),[2,0]*4);self.assertEqual(over.execution.diagnostics.direct_overflow[:,0].tolist(),active.tolist());np.testing.assert_array_equal(over.execution.diagnostics.direct_words,full.execution.diagnostics.direct_words)
  for bad in (None,d.Capture(*t.execution.diagnostics[:25]),t.execution.diagnostics._replace(direct_words=j.zeros((b,2,0,len(d.DIRECT_FIELDS)),j.int32)),t.execution.diagnostics._replace(direct_known=j.zeros((b,2,2,1),bool))):
   before=t._replace(execution=t.execution._replace(diagnostics=bad));self.assertIs(d.record_direct_on_tape(before,values,phase=0,active=active),before)
  self.assertNotIn('direct_hit',d.refused_rows(d.Capture(*a.execution.diagnostics[:25]),np.array([True]+[False]*7))['rows'][0])
  class HostOnly:
   def __init__(self,v):self.value=np.asarray(v);self.shape=self.value.shape;self.dtype=self.value.dtype;self.copies=0
   def __array__(self,dtype=None,copy=None):self.copies+=1;return np.asarray(self.value,dtype=dtype)
   def __getitem__(self,x):raise AssertionError('device row slice')
  wrapped=d.Capture(*(HostOnly(v)for v in a.execution.diagnostics));out=d.refused_rows(wrapped,np.array([True,False,True,False,False,False,False,False]));self.assertEqual([r['row']for r in out['rows']],[0,2]);self.assertTrue(all(v.copies==1 for v in wrapped))
  C=namedtuple('CurrentDiagnosticCursor','turn phase generation');L=namedtuple('CurrentDiagnosticLedger','refusal frame');RK=namedtuple('CurrentDiagnosticRisk','carry');RC=namedtuple('CurrentDiagnosticCarry','risk')
  t=t._replace(cursor=C(t.cursor.turn,t.cursor.phase,t.cursor.generation),execution=t.execution._replace(ledger=L(t.execution.ledger.refusal,t.execution.ledger.frame),risks=RK(RC(t.execution.risks.carry.risk))))
  capture=a.execution.diagnostics;fast=jax.jit(lambda t:d.record_direct_on_tape(t,values,phase=0,active=active));got=fast(t);np.testing.assert_array_equal(got.execution.diagnostics.direct_words[:,0,0],capture.direct_words[:,0,0])
 def test_unsigned_literal_signed_negative_and_nondevice_legacy_storage(self):
  t,_=fixture.OutcomeDiagnosticTests().tape();active=j.ones(8,bool);expected=2**31+1
  a=d.record_direct_on_tape(t,dict(stream_id=j.full(8,expected,j.uint32),ledger_start=j.full(8,-1,j.int32)),phase=0,active=active)
  record=d.refused_rows(a.execution.diagnostics,np.array([True]+[False]*7))['rows'][0]['direct_hit']['emitter']['records'][0]
  self.assertEqual((record['stream_id'],record['ledger_start']),(expected,-1))
  for capture in (d.Capture(*t.execution.diagnostics[:31]),t.execution.diagnostics._replace(**{name:np.asarray(getattr(t.execution.diagnostics,name))for name in d.Capture._fields[25:32]})):
   before=t._replace(execution=t.execution._replace(diagnostics=capture));self.assertFalse(d.direct_enabled(before));self.assertIs(d.record_direct_on_tape(before,dict(move=j.ones(8,j.int32)),phase=0,active=active),before)
  # Read-only host snapshots remain serializable, retaining original dtype bits.
  host=a.execution.diagnostics._replace(**{name:np.asarray(getattr(a.execution.diagnostics,name))for name in d.Capture._fields[25:32]})
  record=d.refused_rows(host,np.array([True]+[False]*7))['rows'][0]['direct_hit']['emitter']['records'][0];self.assertEqual((record['stream_id'],record['ledger_start']),(expected,-1))
 def actual_fixture(self):
  b=8;rows=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);ns=j.arange(8,dtype=j.uint32)+11;streams=z+204001;actor=rows%2;physical=j.stack((z+2,z+4),axis=1);mapping=j.full((b,2,6),-1,j.int32).at[rows[:,None],j.arange(2)[None,:],physical].set(0)
  cursor=life.Cursor(streams,j.broadcast_to(ns,(b,8)),j.zeros((b,2),j.int32),z+3,z,yes,yes,physical,j.zeros((b,2),j.int32),j.ones((b,2),j.int32),j.ones((b,2),bool),j.ones((b,2),j.int32),mapping,j.ones((b,2),j.int32))
  tape=life.LifecycleTape(j.zeros((b,4,10),j.int32),z,z+1,z+3,ns,ns,ns,ns,streams,j.zeros((b,4),j.int32),j.zeros((b,4),j.int32),j.zeros((b,4,2),j.int32),j.zeros((b,4,6),j.int32),cursor)
  arrays=[j.zeros((b,8),j.int32)for _ in range(8)];ledger=ex.Ledger(streams,ns,z+5,yes,z,z,z,z,*arrays,j.zeros((b,8,2),j.int32),j.zeros((b,4),j.int32),j.zeros((b,4,10),j.int32),j.zeros((b,4),j.int32),j.zeros((b,4),j.int32),j.zeros((b,4,2),j.int32),j.zeros((b,8),j.int32),j.zeros((b,8,6),j.int32),j.zeros((b,8,8),j.uint32))
  tables=ex.Tables(ns,ns,ns,ns);tape=tape._replace(execution=ex.Context(ledger,tables,NS(carry=NS(risk=z)),d.empty(b)))
  data=direct.Data(ns,ns,ns,ns,ns,j.array(10,j.int32),j.array(2,j.int32),j.zeros(10,bool),None,return_ko_move=j.array(0,j.int32))
  move=(z+3).at[7].set(2);ap=physical[rows,actor];tp=physical[rows,1-actor];eligible=(rows!=6)&(rows!=7);requested=rows!=6
  tape=public.move(tape,actor,ap,tp,move,eligible=requested)
  # A full row reaches its last MOVE slot before a positive annotation need.
  tape=tape._replace(words=tape.words.at[4,3].set(tape.words[4,0]),event_turn=tape.event_turn.at[4,3].set(tape.event_turn[4,0]),event_phase=tape.event_phase.at[4,3].set(tape.event_phase[4,0]),event_generation=tape.event_generation.at[4,3].set(tape.event_generation[4,0]),count=tape.count.at[4].set(4))
  tape,ticket=ex.open_attempt(tape,ex.Site.DIRECT_HIT,1-actor,active=requested)
  tape=tape._replace(switch_details=tape.switch_details.at[3,1,0].set(1),words=tape.words.at[5,0,5].set(9))
  write=direct.HitWrite((z+102).at[7].set(0),rows==1,yes,~yes,j.where((rows==1)|(rows==3)|(rows==4),2,1),z+1,~yes,~yes,~yes,z+322,j.where(rows==7,0,19),z+322)
  return tape,ticket,data,actor,ap,tp,move,write,eligible,requested
 def test_actual_original_emitter_span_capture_branch_identity_mixed_and_disabled(self):
  original=old_functions(direct,'gpu_engine/engine_direct_hit_controls.py');original['_check']=lambda *a:True;old_ex=old_functions(ex,'gpu_engine/engine_execution_completeness.py')
  tape,ticket,data,actor,ap,tp,move,w,eligible,requested=self.actual_fixture()
  with patch.object(direct,'_check',return_value=True):
   a,av=direct._record(tape,actor,ap,tp,move,w,data,eligible=eligible,_diagnostic_requested=eligible,_diagnostic_bound=j.ones(8,bool));b,bv=original['_record'](tape,actor,ap,tp,move,w,data,eligible=eligible)
  np.testing.assert_array_equal(av,bv)
  for name in tape._fields:
   if name=='execution':continue
   for x,y in zip(jax.tree.leaves(getattr(a,name)),jax.tree.leaves(getattr(b,name)),strict=True):np.testing.assert_array_equal(x,y)
  a=public.condition(a,1-actor,tp,w.new_hp,w.maximum,j.zeros(8,j.int32),eligible=requested);b=public.condition(b,1-actor,tp,w.new_hp,w.maximum,j.zeros(8,j.int32),eligible=requested)
  actual=ex._direct_hit_span_matches(a,ticket,actor,ap,tp,move,w,data=data);before=old_ex['_direct_hit_span_matches'](b,ticket,actor,ap,tp,move,w,data=data);np.testing.assert_array_equal(actual,before)
  pre_capture=b
  a=ex.capture_direct_hit_write(a,ticket,actor,ap,tp,move,w,requested=eligible,data=data);b=old_ex['capture_direct_hit_write'](b,ticket,actor,ap,tp,move,w,requested=eligible,data=data)
  for name in a.execution.ledger._fields:np.testing.assert_array_equal(getattr(a.execution.ledger,name),getattr(b.execution.ledger,name))
  np.testing.assert_array_equal(a.words,b.words);np.testing.assert_array_equal(a.refusal,b.refusal)
  RK=namedtuple('SpanTraceRisk','carry');RC=namedtuple('SpanTraceCarry','risk')
  trace=pre_capture._replace(execution=pre_capture.execution._replace(risks=RK(RC(j.zeros(8,j.int32)))))
  traced=jax.jit(lambda t:ex.capture_direct_hit_write(t,ticket,actor,ap,tp,move,w,requested=eligible,data=data))(trace)
  for field in b.execution.ledger._fields:np.testing.assert_array_equal(getattr(traced.execution.ledger,field),getattr(b.execution.ledger,field))
  self.assertEqual(traced.execution.diagnostics.direct_count[:,1].tolist(),[1]*6+[0,1])
  out=d.refused_rows(a.execution.diagnostics,np.array([True]*8))['rows'];r=out[0]['direct_hit'];self.assertEqual(r['emitter']['records'][0]['need'],0);self.assertEqual(r['span_capture']['records'][0]['semantic'],1);self.assertEqual(r['span_capture']['records'][0]['ledger_start'],1)
  r=out[3]['direct_hit']['emitter']['records'][0];self.assertEqual((r['valid_before_details'],r['valid'],r['detail0_zero']),(1,0,0))
  r=out[4]['direct_hit']['emitter']['records'][0];self.assertEqual((r['room'],r['overflow']),(0,1))
  r=out[5]['direct_hit']['span_capture']['records'][0];self.assertEqual((r['declared_header'],r['semantic']),(0,0))
  self.assertEqual(out[6]['direct_hit']['emitter']['count'],0);self.assertEqual(out[6]['direct_hit']['span_capture']['count'],0)
  r=out[7]['direct_hit'];self.assertEqual(r['emitter']['count'],0);self.assertEqual((r['span_capture']['records'][0]['requested'],r['span_capture']['records'][0]['write_valid']),(0,0))
  legacy=tape._replace(execution=tape.execution._replace(diagnostics=d.Capture(*tape.execution.diagnostics[:25])))
  with patch.object(direct,'_check',return_value=True):oldresult,oldvalid=original['_record'](legacy,actor,ap,tp,move,w,data,eligible=eligible);result,valid=direct._record(legacy,actor,ap,tp,move,w,data,eligible=eligible)
  self.assertEqual(result.execution.diagnostics.direct_words,None);np.testing.assert_array_equal(valid,oldvalid);np.testing.assert_array_equal(result.words,oldresult.words)
  original['_check']=direct._check
  bare=public.CallbackTape(tape.words,tape.count,tape.refusal,tape.turn,tape.engine_binding,tape.source_binding,tape.move_binding,tape.producer_binding)
  for current in (bare,tape):
   outcomes=[]
   for fn in (original['_record'],direct._record):
    try:
     r,v=fn(current,actor,ap,tp,move,w,None,eligible=eligible);outcomes.append(('returned',r is current,v.tolist()))
    except Exception as error:outcomes.append((type(error).__name__,str(error)))
   self.assertEqual(outcomes[0],outcomes[1])
 def test_reverse_ast_physical_scope_and_cold_imports(self):
  class Strip(ast.NodeTransformer):
   def visit_NamedExpr(self,n):return self.visit(n.value)
   def visit_Call(self,n):
    n=self.generic_visit(n);n.keywords=[k for k in n.keywords if k.arg not in('_diagnostic_sink','_diagnostic_requested','_diagnostic_bound')];return n
   def visit_If(self,n):
    if '_diagnostic_sink' in ast.unparse(n.test)or ast.unparse(n.test)=='diagnostics.direct_enabled(tape)':return None
    return self.generic_visit(n)
   def visit_ImportFrom(self,n):return None if any(a.asname=='diagnostics'for a in n.names)else n
   def visit_Assign(self,n):
    if any(isinstance(t,ast.Name)and t.id.startswith('_diagnostic')for t in n.targets):return None
    if isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='record_direct_on_tape':return None
    return self.generic_visit(n)
   def visit_Expr(self,n):return None if ast.unparse(n).startswith('_diagnostic_details.append')else self.generic_visit(n)
   def visit_FunctionDef(self,n):
    keep=[i for i,a in enumerate(n.args.kwonlyargs)if not a.arg.startswith('_diagnostic')];n.args.kwonlyargs=[n.args.kwonlyargs[i]for i in keep];n.args.kw_defaults=[n.args.kw_defaults[i]for i in keep];return self.generic_visit(n)
  for path,owned in [('gpu_engine/engine_direct_hit_controls.py',{'_record','record_registered_direct_hit_controls_device'}),('gpu_engine/engine_execution_completeness.py',{'_direct_hit_span_matches','capture_direct_hit_write'})]:
   old=ast.parse(subprocess.check_output(['git','show','b8092a85:'+path],text=True));new=ast.parse(Path(path).read_text());functions=lambda t:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)};a,b=functions(old),functions(new)
   for name in a:
    self.assertEqual(ast.dump(b[name]),ast.dump(a[name]),name)
    self.assertEqual(ast.dump(Strip().visit(deepcopy(b[name]))if name in owned else b[name]),ast.dump(Strip().visit(deepcopy(a[name]))if name in owned else a[name]),name)
  physical='action_history absorption baton combat critical damage_moves effects encore entry focus_punch items linked_traps mechanics phazing pressure pursuit secondary self_destruct sleep_talk solar_beam state sustain transition variable_moves weather'.split()
  for name in physical+['engine_retained_encoder_driver','engine_kernel_frame_risk','engine_full_v4_event_device','engine_lifecycle_event_device','benchmark_capture_gap_census']:
   path='gpu_engine/'+name+'.py';self.assertEqual(Path(path).read_bytes(),subprocess.check_output(['git','show','b8092a85:'+path]))
  from gpu_engine import engine_source_frame_witness as witness
  witness._source()
  for names in [('gpu_engine.benchmark_retained_games','gpu_engine.engine_outcome_diagnostics','gpu_engine.engine_full_v4_event_device'),('gpu_engine.engine_full_v4_event_device','gpu_engine.engine_outcome_diagnostics','gpu_engine.benchmark_retained_games')]:
   r=subprocess.run([sys.executable,'-B','-c','import importlib;'+''.join('importlib.import_module('+repr(n)+');'for n in names)],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr)
if __name__=='__main__':unittest.main()
