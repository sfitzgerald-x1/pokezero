"""Call-local immutable tuple visits; no model/kernel/game/benchmark runs."""
import ast
import builtins
from contextvars import copy_context
import jax.numpy as j
from collections import namedtuple
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import subprocess
import unittest
from unittest.mock import patch
import jax
import numpy as np
from gpu_engine import v4_encoder as e,engine_retained_encoder_driver as d

BASE='5914e1b70080dff891a21d3de2d6121a167bbe5d'

@dataclass
class HostFacts:
    tokens:object
    metadata:object
    sequence:object

class GuardPassTests(unittest.TestCase):
 def setUp(self):
    self.first=10**50;self.second=self.first+1
    self.tokens=tuple('guard-token-'+str(i)for i in range(8))
    self.facts=HostFacts(self.tokens,{'source':self.first},[self.first])
    self.catalog=object.__new__(e.Catalog);object.__setattr__(self.catalog,'vocabulary',self.facts)
    self.record=(('vocabulary',e._static_snapshot(self.facts,set())),)
    e._CATALOGS[self.catalog]=self.record;e._STATIC_PLANS[self.catalog]=e._compile_static_plan(self.record)
 def tearDown(self):
    e._CATALOGS.pop(self.catalog,None);e._STATIC_PLANS.pop(self.catalog,None)
    self.assertIsNone(e._GUARD_PASS.get())
 def original_checked(self):
    source=subprocess.check_output(['git','show',BASE+':gpu_engine/engine_retained_encoder_driver.py'],text=True)
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='_checked');scope=dict(vars(d));exec(compile(ast.Module(body=[fn],type_ignores=[]),'original-checked','exec'),scope);return scope['_checked']
 def prepared(self,*,execution=None):
    prepared=d.Prepared('original-pass-binding');r={name:None for name in ('execution_compiled','no_output_compiled','kernel_risk_compiled','calm_mind_compiled','swords_dance_compiled','recoil_compiled','toxic_compiled','substitute_compiled','direct_hit_compiled','spikes_compiled','thunder_wave_compiled','sleep_powder_compiled','speed_profile','leftovers_compiled','status_residual_compiled')}
    r.update(binding=prepared.binding,own_compiled=object(),request_compiled=object(),public=object(),check_public=lambda _:e.check_catalog(self.catalog),execution_compiled=execution)
    d._PREPARED[prepared]=r;return prepared,r
 def test_only_immutable_descendants_reuse_and_scope_never_escapes(self):
    visits=[]
    def observed_zip(*args,**kw):
        if args and args[0]is self.tokens:visits.append('tokens')
        return builtins.zip(*args,**kw)
    with patch.object(e,'zip',observed_zip,create=True):
        e.check_catalog(self.catalog);e.check_catalog(self.catalog);self.assertEqual(visits,['tokens']*2);visits.clear()
        with e._driver_guard_pass()as capability:
            self.assertIsNone(capability);outer=e._GUARD_PASS.get()
            e.check_catalog(self.catalog);e.check_catalog(self.catalog);self.assertEqual(visits,['tokens'])
            with e._driver_guard_pass():
                self.assertIsNot(e._GUARD_PASS.get(),outer);e.check_catalog(self.catalog)
            self.assertIs(e._GUARD_PASS.get(),outer)
        self.assertIsNone(e._GUARD_PASS.get());self.assertEqual(visits,['tokens']*2)
        with self.assertRaisesRegex(RuntimeError,'expire'):
            with e._driver_guard_pass():e.check_catalog(self.catalog);raise RuntimeError('expire')
        self.assertIsNone(e._GUARD_PASS.get());e.check_catalog(self.catalog);self.assertEqual(visits,['tokens']*4)
 def test_current_mutable_and_replaced_edges_are_still_checked_inside_pass(self):
    cases=[(lambda:self.facts.metadata.__setitem__('source',self.second),lambda:self.facts.metadata.__setitem__('source',self.first)),(lambda:self.facts.metadata.__setitem__('extra',self.first),lambda:self.facts.metadata.pop('extra')),(lambda:self.facts.sequence.append(self.first),lambda:self.facts.sequence.pop()),(lambda:setattr(self.facts,'tokens',tuple(list(self.tokens))),lambda:setattr(self.facts,'tokens',self.tokens))]
    for mutate,restore in cases:
        with e._driver_guard_pass():
            e.check_catalog(self.catalog);mutate()
            try:
                with self.assertRaises(ValueError):e.check_catalog(self.catalog)
            finally:restore()
            e.check_catalog(self.catalog)
    with e._driver_guard_pass():
        e.check_catalog(self.catalog);saved=e._CATALOGS.pop(self.catalog)
        try:
            with self.assertRaisesRegex(ValueError,'original compiled'):e.check_catalog(self.catalog)
        finally:e._CATALOGS[self.catalog]=saved
    # Mutable aliases stored as leaf snapshots cannot confer immutability.
    mutable=HostFacts((),{},[]);root=HostFacts((),{'first':mutable,'alias':(mutable,)},[])
    record=(('child',e._static_snapshot(root,set())),);plan=e._compile_static_plan(record)
    with e._driver_guard_pass():
        eligible=e._GUARD_PASS.get().immutable_nodes(plan)
        alias=root.metadata['alias'];idx=next(i for i,x in enumerate(plan.nodes)if x[0]is alias)
        self.assertFalse(eligible[idx])
 def test_old_new_driver_guard_outcomes_and_source_checks_match(self):
    original=self.original_checked();prepared,r=self.prepared();visits=[]
    with TemporaryDirectory()as directory:
        path=Path(directory,'source');path.write_text('original')
        for change in ('none','metadata','source'):
            results=[]
            for guard in (original,d._checked):
                self.facts.metadata['source']=self.first;path.write_text('original');visits.clear()
                def own(_):
                    e.check_catalog(self.catalog);visits.append('own')
                    if change=='metadata':self.facts.metadata['source']=self.second
                    if change=='source':path.write_text('changed')
                def source(_):
                    visits.append('source')
                    if path.read_text()!='original':raise ValueError('source bytes changed')
                    e.check_catalog(self.catalog)
                with patch.object(d.own,'_checked',side_effect=own),patch.object(d.source,'_checked',side_effect=source):
                    try:self.assertIs(guard(prepared),r);out='accepted'
                    except ValueError as err:out=str(err)
                results.append((out,tuple(visits)));self.assertIsNone(e._GUARD_PASS.get())
            self.assertEqual(results[0],results[1]);self.assertEqual(results[0][0]=='accepted',change=='none')
        self.facts.metadata['source']=self.first
 def test_real_driver_scope_closes_before_fake_phase_callback_and_failure(self):
    Core=namedtuple('Core','engine frame_number admitted requests orders unsafe originals constructors weather_profile execution risks cursor');State=namedtuple('State','core observers')
    yes=np.ones(1,bool);engine=NS(done=~yes);core=Core(engine,np.zeros(1,np.int32),yes,(),(),(),(),(),None,None,None,None)
    observers=tuple(NS(retained='public',retained_healthy=yes,projection=NS(causal=NS(admitted=yes)))for _ in range(2));state=State(core,observers);result=d.Result(state,());prepared,r=self.prepared(execution=object());session=d.Session(prepared,state,result);record=dict(prepared=prepared,state=state,result=result,constructed=object(),advancing=False);d._SESSIONS[session]=record
    r.update(retain_speed=None,public_mode='retained-full',direct_hit_mode='kernel-event',direct_hit_context=None)
    def static(_):e.check_catalog(self.catalog)
    def frame(*args):
        self.assertIsNone(e._GUARD_PASS.get());return NS(state=engine,cursor=NS(admitted=yes),callbacks=NS(refusal=np.zeros(1,np.int32),execution=None),weather_profile=None,invalid_actions=~yes,placeholder_moves=np.zeros(1,np.int32))
    r['frame']=frame
    def observer(name,ms):
        self.assertIsNone(e._GUARD_PASS.get())
        with self.assertRaisesRegex(ValueError,'not reentrant'):d.advance(session,np.zeros((1,2),np.int32),chance_inputs='supplied')
        self.facts.metadata['source']=self.second
    with ExitStack()as stack:
        for obj,name,value in ((d.own,'_checked',static),(d.source,'_checked',static),(d.execution,'_checked',static),(d.source,'_constructed',lambda _:None),(d,'_checked_witness',lambda *a,**k:None),(d,'_leech_history',lambda *a,**k:None),(d.jax,'block_until_ready',lambda x:x)):
            stack.enter_context(patch.object(obj,name,side_effect=value))
        stack.enter_context(patch.object(d.time,'perf_counter_ns',side_effect=count(1)))
        with self.assertRaises(ValueError):d.advance(session,np.zeros((1,2),np.int32),active_rows=yes,chance_inputs='supplied',timing_callback=observer)
    self.assertIs(session.state,state);self.assertIs(session.result,result);self.assertFalse(record['advancing']);self.assertIsNone(e._GUARD_PASS.get());self.facts.metadata['source']=self.first
 def test_public_trace_rejection_and_whole_driver_guard_program_unchanged(self):
    # Even a previously visited immutable subtree cannot cache tracing success.
    with e._driver_guard_pass():
        e.check_catalog(self.catalog)
        with self.assertRaisesRegex(ValueError,'cannot be traced'):jax.jit(lambda:e.check_catalog(self.catalog))()
    with self.assertRaisesRegex(ValueError,'cannot be traced'):jax.jit(lambda:e.check_catalog(self.catalog))()
    old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_retained_encoder_driver.py'],text=True));new=ast.parse(Path(d.__file__).read_text())
    fn=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    for name in ('advance','_checked_witness','_consume_accounted','_session','_timed','latest_diagnostics'):
        self.assertEqual(ast.dump(fn(old,name),include_attributes=False),ast.dump(fn(new,name),include_attributes=False))
    checked=fn(new,'_checked');self.assertEqual(len(checked.body),1);self.assertIsInstance(checked.body[0],ast.With);self.assertEqual(ast.unparse(checked.body[0].items[0].context_expr),'v4_encoder._driver_guard_pass()');checked.body=checked.body[0].body
    self.assertEqual(ast.dump(fn(old,'_checked'),include_attributes=False),ast.dump(checked,include_attributes=False))

 def test_copied_context_cannot_reuse_expired_lexical_guard(self):
   visits=[]
   def observed_zip(*args,**kw):
    if args and args[0]is self.tokens:visits.append('tuple')
    return builtins.zip(*args,**kw)
   with patch.object(e,'zip',observed_zip,create=True):
    for exceptional in (False,True):
     with self.subTest(exceptional=exceptional):
      visits.clear()
      try:
       with e._driver_guard_pass():
        e.check_catalog(self.catalog);saved=copy_context()
        if exceptional:raise RuntimeError('expected-exit')
      except RuntimeError:pass
      self.assertIsNone(e._GUARD_PASS.get())
      saved.run(e.check_catalog,self.catalog)
      self.assertEqual(visits,['tuple','tuple'],'copied Context must not retain prior completed visits after scope exit')
 def test_original_mutation_decisions_and_deleted_leaf_policy_match(self):
   source=subprocess.check_output(['git','show',BASE+':gpu_engine/v4_encoder.py'],text=True)
   fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='_check_catalog_plan');ns=dict(vars(e));exec(compile(ast.Module(body=[fn],type_ignores=[]),'original-flat-validator','exec'),ns);old=ns['_check_catalog_plan']
   plan=e._STATIC_PLANS[self.catalog]
   def result(f):
    try:f(self.catalog,plan);return 'accept'
    except ValueError as x:return str(x)
   def current(c,p):return e._check_catalog_plan(c,p)
   changes=[(lambda:self.facts.metadata.__setitem__('source',self.second),lambda:self.facts.metadata.__setitem__('source',self.first)),(lambda:self.facts.sequence.append(self.second),lambda:self.facts.sequence.pop()),(lambda:setattr(self.facts,'tokens',tuple(list(self.tokens))),lambda:setattr(self.facts,'tokens',self.tokens))]
   for mutate,restore in changes:
    with e._driver_guard_pass():
     self.assertEqual(result(old),result(current));e.check_catalog(self.catalog);mutate()
     try:self.assertEqual(result(old),result(current));self.assertNotEqual(result(current),'accept')
     finally:restore()
   # Array availability policy is not silently expanded: original flat plan
   # compares immutable leaf identity; all other source validators stay intact.
   array=j.array([2],j.int32);self.facts.tokens=(array,);record=(('vocabulary',e._static_snapshot(self.facts,set())),)
   e._CATALOGS[self.catalog]=record;plan=e._compile_static_plan(record);e._STATIC_PLANS[self.catalog]=plan
   with e._driver_guard_pass():
    self.assertEqual(result(old),result(current));array.delete()
    self.assertEqual(result(old),result(current))

if __name__=='__main__':unittest.main()
