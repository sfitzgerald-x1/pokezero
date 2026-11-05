"""SAME queue operands + original writers only; no engine step/game/model."""
from _showdown_root import showdown_root_str
import ast,json,subprocess,unittest
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_status_residual_callbacks as status,engine_lifecycle_callbacks as life
from gpu_engine import engine_request_source as source,engine_weather_speed_profile as profile
from gpu_engine import engine_execution_completeness as ex,engine_public_events as events
from gpu_engine import engine_full_v4_event_device as full,v4_encoder as v4,public_wire
from gpu_engine.state import empty_state
import test_gpu_engine_engine_weather_public as fixture
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected

ROOT=showdown_root_str()
class SingletonStatusQueueTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.WeatherPublicTests.setUpClass();cls.f=fixture.WeatherPublicTests
  cls.rules=source.load_source_rules(ROOT,cls.f.metadata)
  cls.helper=status.compile_status_residual(cls.f.metadata,engine_catalog=cls.f.producer,source_rules=cls.rules)
  cls.legacy=status.compile_status_residual(cls.f.metadata,engine_catalog=cls.f.producer)
  cls.profile=profile.compile_profile(cls.f.metadata,catalog=cls.f.catalog,engine_catalog=cls.f.producer,rules=cls.rules)
  cls.account=ex.compile_accounting(cls.f.compiled)
 def initial(self,b=8,consume=False):
  rows=j.arange(b);side=rows%2;s=empty_state(b,seed=0);ability=self.f.metadata['vocabulary']['abilities'].index('pressure');item=self.f.metadata['vocabulary']['items'].index('leftovers')
  s=s._replace(done=j.zeros(b,bool),active=j.broadcast_to(j.array([3,5],j.int32),(b,2)),species=j.ones_like(s.species).at[:,1].set(2),hp=j.full_like(s.hp,341),max_hp=j.full_like(s.max_hp,341),stats=j.full_like(s.stats,151),level=j.full_like(s.level,80),gender=j.ones_like(s.gender),ability=j.full_like(s.ability,ability),base_ability=j.full_like(s.base_ability,ability),item=j.full_like(s.item,item))
  streams=j.arange(5801,5801+b,dtype=j.int32);record=profile.qualify_birth(s,streams,profile=self.profile)
  self.assertTrue(bool(record.qualified_birth.all()))
  s=s._replace(status=s.status.at[rows,side,s.active[rows,side]].set(4))
  record=profile.retain_device(record,s,streams,self.profile.device)
  kw=dict(catalog=self.f.producer,expected_binding=self.f.producer.binding,expected_engine_binding=np.asarray(self.f.producer.engine_binding,np.uint32));c=life.register(streams,**kw);before=life.snapshot(c);t=life._tape(c,self.f.producer,64)
  for owner in (0,1):t=life._reveal(t,s,j.full(b,owner,j.int32),j.full(b,-1,j.int32),s.active[:,owner],eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
  t=t._replace(cursor=t.cursor._replace(initialized=j.ones(b,bool)));t=life._turn(t,j.ones(b,bool));consumers=[]
  if consume:
   frame=life.LifecycleResult(s,t.cursor,t,before,life.snapshot(t.cursor),j.zeros(b,bool),j.zeros(b,j.int32));fold=jax.jit(lambda stream,pub:full.project_and_fold(stream,frame,self.f.compiled.device,pub,self.f.catalog))
   for owner in (0,1):
    stream=full.register(streams,j.full(b,owner,j.int32),compiled=self.f.compiled);pub=v4.initialize(self.f.catalog,stream_ids=streams,owner_sides=j.full(b,owner,j.int32));got=fold(stream,pub);self.assertEqual(got.projection.error.tolist(),[0]*b);consumers.append((got.projection.state,got.public))
  return s,t.cursor,record,tuple(consumers)
 def queue(self,b=8,*,s=None,c=None,eligible=None):
  if s is None:s,c,_,_=self.initial(b)
  active=j.ones(b,bool)if eligible is None else eligible;t=life._tape(c,self.f.producer,64);t=ex.begin_on_tape(t,ex.initialize(c.stream_id,compiled=self.account),self.account.device,active=j.ones(b,bool));t=life.action_end(t,j.ones(b,bool));rows=j.arange(b)[:,None];sides=j.arange(2)[None,:]
  hp=s.hp[rows,sides,s.active];st=s.status[rows,sides,s.active];speeds=j.full((b,2),151,j.int32)
  q=status.capture_queue_device(t,self.helper.device,hp,st,speeds,eligible=active)
  side=j.arange(b)%2;physical=s.active[j.arange(b),side];t,ticket=ex.open_attempt(t,ex.Site.STATUS_RESIDUAL,side,active=active)
  return t,q,side,physical,ticket
 def matches(self,t,q,side,physical,*,data=None,hp=None,kind=None):
  b=len(side)
  return status.singleton_queue_matches(t,self.helper.device if data is None else data,q,side,physical,j.full(b,341,j.int32)if hp is None else hp,j.full(b,4,j.int32)if kind is None else kind)
 def emit(self,t,q,side,physical,ticket,*,new=320):
  b=len(side);good=self.matches(t,q,side,physical)
  out=status.record_status_residual_device(t,side,physical,j.full(b,341,j.int32),j.full(b,new,j.int32),j.full(b,341,j.int32),j.full(b,4,j.int32),j.zeros(b,j.int32),j.ones(b,j.int32),self.helper.device,eligible=j.ones(b,bool),_defer_own_unknown=good)
  ledger=ex.unhandled(out.execution.ledger,ticket,active=~good,tables=self.account.device);out=ex._put_context(out,ledger,out.execution)
  return ex.capture_status_residual_write(out,ticket,requested=good,positive_write=j.full(b,new>0,bool))
 def test_same_queue_singleton_guards_global_lifetime_and_legacy_closed(self):
  t,q,side,physical,ticket=self.queue();self.assertTrue(bool(self.matches(t,q,side,physical).all()));self.assertFalse(bool(self.matches(t,q,side,physical,data=self.legacy.device).any()))
  out=self.emit(t,q,side,physical,ticket);self.assertEqual(out.refusal.tolist(),[1]*8);self.assertEqual(out.execution.ledger.stage[:,0].tolist(),[2]*8)
  self.assertEqual(out.words[0,1].tolist(),[20,0,1,0,3,341,320,341,4,3])
  # Both tied source status handlers remain unknown, even if one would be skipped later.
  both=q._replace(status=q.status.at[j.arange(8),1-side].set(2));self.assertFalse(bool(self.matches(t,both,side,physical).any()))
  for key,value in [('frame',q.frame+1),('turn',q.turn+1),('phase',q.phase+1),('generation',q.generation.at[0,0].add(1)),('physical',q.physical.at[0,0].set(0)),('eligible',q.eligible.at[0].set(False)),('status',q.status.at[0,0].set(0)),('hp',q.hp.at[0,0].set(0)),('speeds',q.speeds.at[0,0].set(152)),('binding',q.binding.at[0].add(j.uint32(1)))]:
   bad=q._replace(**{key:value});self.assertFalse(bool(self.matches(t,bad,side,physical)[0]),key)
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))});self.assertFalse(bool(self.matches(bad,q,side,physical).any()),name)
   with self.assertRaises(ValueError):self.matches(t._replace(**{name:j.tile(getattr(t,name),(8,1))}),q,side,physical)
  self.assertFalse(bool(self.matches(t,q,side,physical,kind=j.ones(8,j.int32)).any()));self.assertFalse(bool(self.matches(t,q,side,physical,hp=j.zeros(8,j.int32)).any()))
  lethal=self.emit(t,q,side,physical,ticket,new=0);self.assertTrue(bool((lethal.refusal&256).all()));self.assertTrue(bool((lethal.execution.ledger.refusal!=0).all()));self.assertEqual(lethal.execution.ledger.stage[:,0].tolist(),[1]*8)
  sticky=t._replace(refusal=t.refusal|128);out=self.emit(sticky,q,side,physical,ticket);self.assertTrue(bool((out.refusal&128).all()))
 def test_both_original_full_consumers_exact_toxic_fields_and_stage3(self):
  s,c,record,consumers=self.initial(2,consume=True);t,q,side,physical,ticket=self.queue(2,s=s,c=c);out=self.emit(t,q,side,physical,ticket)
  after=s._replace(hp=s.hp.at[j.arange(2),side,physical].set(320),status_counter=s.status_counter.at[j.arange(2),side,physical].set(1));record=profile.retain_device(record,after,c.stream_id,self.profile.device)
  frame=life.LifecycleResult(after,out.cursor,out,life.snapshot(c),life.snapshot(out.cursor),j.zeros(2,bool),j.zeros(2,j.int32));consume=jax.jit(lambda stream,pub:ex.consume_frame_device(out.execution.ledger,frame,stream,pub,active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.f.compiled.device,catalog=self.f.catalog,weather_hp=(record,self.profile.device),status_residual=self.helper.device));closed=[]
  for owner,(stream,pub)in enumerate(consumers):
   end,got=consume(stream,pub);closed.append(end);self.assertEqual(got.projection.error.tolist(),[0,0]);self.assertTrue(bool(ex.report(end,tables=self.account.device).scoped_drained.all()))
   for row in range(2):
    names={i:self.f.catalog.species.names[int(self.f.compiled.device.causal.species[i+1])]for i in (0,1)};initial=[]
    for i in (0,1):initial.append(f'|switch|p{i+1}a: {names[i]}|{names[i]}, L80, M|'+('341/341'if i==owner else '100/100')+(' tox'if i==row else ''))
    wire,_=public_wire.project_lines(public_wire.WireState(),initial+['|turn|1'],**v4.wire_kwargs(self.f.catalog));hp='320/341'if row==owner else '94/100';_,packets=public_wire.project_lines(wire,['|',f'|-damage|p{row+1}a: {names[row]}|{hp} tox|[from] psn'],**v4.wire_kwargs(self.f.catalog))
    expected=np.asarray([e.words()for packet in packets for e in packet.events],np.int32);actual=np.asarray(got.projection.frames.words).transpose(1,0,2,3)[row][np.asarray(got.projection.frames.valid).transpose(1,0,2)[row]];np.testing.assert_array_equal(actual,expected)
  _,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
 def test_original_sort_singleton_no_shuffle_true_tie_and_default_program(self):
  script=r"""const path=require('path'),r=process.argv[1],{Battle}=require(path.join(r,'dist/sim/battle.js')),d=require(path.join(r,'dist/sim/dex.js')).Dex.mod('gen3');const results=[];for(const statuses of [['tox'],['tox','brn']]){let calls=0;let a=statuses.map((n,i)=>{const x=d.conditions.get(n);return {side:i,order:x.onResidualOrder,priority:x.onResidualPriority||0,speed:151,subOrder:x.onResidualSubOrder}});Battle.prototype.speedSort.call({comparePriority:Battle.prototype.comparePriority,prng:{shuffle(){calls++}}},a);results.push(calls)}let damages=[],state={stage:0};d.conditions.get('tox').onResidual.call({effectState:state,clampIntRange:(x,min)=>Math.max(Math.trunc(x),min),damage:n=>damages.push(n)},{baseMaxhp:341});console.log(JSON.stringify({results,damages,stage:state.stage}));"""
  got=json.loads(subprocess.check_output(['node','-e',script,ROOT],text=True));self.assertEqual(got,dict(results=[0,1],damages=[21],stage=1))
  old=ast.parse(subprocess.check_output(['git','show','1376098:gpu_engine/transition.py'],text=True));new=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(old);self.assertEqual(baseline,dump_default_program(new));assert_hp_rng_mutations_detected(self,new,baseline)
  # The old strict unequal-speed guard remains byte/AST-identical.
  get=lambda tree:ast.dump(next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_status_cached_speed_ok'for t in n.targets)))
  self.assertEqual(get(old),get(new))
  tree=ast.parse(Path('src/pokezero/v5_worker.py').read_text());call=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='compile_status_residual');env=dict(status=type('Spy',(),{'compile_status_residual':staticmethod(lambda *args,**kwargs:(args,kwargs))}),metadata=object(),lifecycle=object(),rules=object());args,kw=eval(compile(ast.Expression(call),'<original-worker-call>','eval'),env);self.assertIs(args[0],env['metadata']);self.assertIs(kw['engine_catalog'],env['lifecycle']);self.assertIs(kw['source_rules'],env['rules'])
