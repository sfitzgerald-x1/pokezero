"""Actual initialization/entry/FULL components; no world/model/Native/game."""
from _showdown_root import showdown_root_str
import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json,subprocess,shutil,unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import entry,transition,engine_public_events as events,engine_lifecycle_callbacks as life
from gpu_engine import engine_intimidate_callbacks as intimidate,engine_execution_completeness as ex,engine_kernel_frame_risk as risks
from gpu_engine import engine_public_event_device as base,engine_full_v4_event_device as full,public_belief,v4_encoder as v4,public_wire
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex,SpeciesInfo
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
import test_gpu_engine_pressure_entry_owning as fixtures
from gpu_engine_default_observer_ast import dump_default_program

class InitialIntimidateOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.PressureEntryOwningTests.setUpClass();cls.f=fixtures.PressureEntryOwningTests()
  old=cls.f.catalog;cls.metadata=deepcopy(cls.f.metadata);cls.metadata['vocabulary']['species']=['_none','arbok','celebi'];cls.metadata['vocabulary']['abilities']=['_none','intimidate','naturalcure','clearbody']
  source=public_belief.compile_source(Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-intimidate'),{name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('literal-'+name,'fixture',name,'fixture',80,('Surf',),ability,'Leftovers'),)) for name,ability in (('Arbok','Intimidate'),('Celebi','Natural Cure'))}))
  species=dict(old.dex.species)
  for name in ('Arbok','Celebi'):species[name.lower()]=replace(next(iter(old.dex.species.values())),id=name.lower(),name=name)
  dex=ShowdownDex(old.dex.moves,species,old.dex.type_chart)
  vocab=CategoryVocabulary(tuple(sorted(set(old.vocabulary.tokens)|{'species:arbok','species:celebi','ability:intimidate','belief:possible_ability:intimidate'})))
  cls.catalog=v4.compile_encoder(source,dex,vocab,checkpoint_sha256='a'*64,feature_masks=fixtures.encoder_fixture.V4EncoderTests.masks)
  cls.table=cls.f.engine._replace(entry=entry.from_metadata(cls.metadata));cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=fixtures.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog);cls.account=ex.compile_accounting(cls.adapter);cls.risk=risks.compile_risks(cls.producer,cls.account)
  cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
 def same(self,a,b):
  for x,z in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,z)
 def state(self,b=2):
  s=self.f.state(b)
  return s._replace(active=j.zeros((b,2),j.int32),entry_pending=j.ones((b,2),bool),done=j.zeros(b,bool),
      ability=j.ones_like(s.ability).at[:,1].set(2),base_ability=j.ones_like(s.base_ability).at[:,1].set(2),stats=j.full_like(s.stats,142).at[:,1,:,5].set(131))
 def initial(self,state=None,*,prior=False):
  s=self.state() if state is None else state;b=s.hp.shape[0];cursor=life.register(j.arange(1201,1201+b,dtype=j.int32),**self.kw)
  return life.initialize_with_lifecycle(s,self.table,cursor=cursor,**self.kw,tie_coin=j.zeros(b,j.int32),execution_compiled=self.account,kernel_risk_compiled=self.risk)
 def test_same_actual_init_write_exact_two_original_lines_and_genuine_both_full_drain(self):
  s=self.state();actual=self.initial(s);plain=transition.initialize(s,self.table,tie_coin=j.zeros(2,j.int32));self.same(actual.state,plain)
  self.assertEqual(actual.state.boosts[:,1,0,0].tolist(),[-1,-1]);self.assertEqual(actual.callbacks.refusal.tolist(),[1,1])
  self.assertEqual(actual.callbacks.words[:,2,0].tolist(),[29,29]);self.assertEqual(actual.callbacks.words[0,2].tolist(),[29,0,0,1,0,1,0,-1,0,0])
  self.assertEqual(actual.callbacks.execution.ledger.stage[:,0].tolist(),[2,2]);self.assertEqual(actual.callbacks.execution.risks.carry.risk.tolist(),[0,0])
  ledger=actual.callbacks.execution.ledger;closed=[]
  consume=jax.jit(lambda stream,public:ex.consume_frame_device(ledger,actual,stream,public,active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog))
  for owner in (0,1):
   stream=full.register(actual.cursor.stream_id,j.full(2,owner,j.int32),compiled=self.adapter);public=v4.initialize(self.catalog,stream_ids=actual.cursor.stream_id,owner_sides=j.full(2,owner,j.int32))
   account,result=consume(stream,public);closed.append(account);self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertEqual(account.stage[:,0].tolist(),[3,3])
   lines=['|switch|p1a: Arbok|Arbok, L80, M|100/100','|switch|p2a: Celebi|Celebi, L80, M|100/100','|-ability|p1a: Arbok|Intimidate|boost','|-unboost|p2a: Celebi|atk|1','|turn|1']
   _,packets=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog));expected=np.array([e.words() for packet in packets for e in packet.events],np.int32)
   for row in (0,1):np.testing.assert_array_equal(self.f.words(result.projection,row),expected)
   self.assertEqual(result.public.mons.boosts[:,1,0].tolist(),[-1,-1]);self.assertFalse(bool(result.projection.B4_qualified.any()))
  _,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
 def test_actual_mirrored_and_both_start_abilities_preserve_pair_order_both_owners(self):
  s=self.state(3);abilities=s.ability.at[1,0].set(2).at[1,1].set(1).at[2,1].set(1)
  s=s._replace(ability=abilities,base_ability=abilities,species=s.species.at[1,0].set(2).at[1,1].set(1).at[2,1].set(1))
  actual=self.initial(s);self.same(actual.state,transition.initialize(s,self.table,tie_coin=j.zeros(3,j.int32)))
  self.assertEqual(actual.callbacks.execution.ledger.count.tolist(),[1,1,2]);self.assertEqual(actual.callbacks.refusal.tolist(),[1]*3)
  self.assertEqual(actual.callbacks.words[:,2,1].tolist(),[0,1,0]);self.assertEqual(int(actual.callbacks.words[2,3,1]),1)
  closed=[]
  for owner in (0,1):
   stream=full.register(actual.cursor.stream_id,j.full(3,owner,j.int32),compiled=self.adapter);public=v4.initialize(self.catalog,stream_ids=actual.cursor.stream_id,owner_sides=j.full(3,owner,j.int32))
   account,result=ex.consume_frame_device(actual.callbacks.execution.ledger,actual,stream,public,active=j.ones(3,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog);closed.append(account)
   self.assertEqual(result.projection.error.tolist(),[0]*3)
   for row,names in enumerate((('Arbok','Celebi'),('Celebi','Arbok'),('Arbok','Arbok'))):
    lines=[f'|switch|p{side+1}a: {name}|{name}, L80, M|100/100' for side,name in enumerate(names)]
    for side in ((0,) if row==0 else (1,) if row==1 else (0,1)):
     lines += [f'|-ability|p{side+1}a: Arbok|Intimidate|boost',f'|-unboost|p{2-side}a: {names[1-side]}|atk|1']
    lines+=['|turn|1'];_,packets=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog))
    expected=np.array([event.words() for packet in packets for event in packet.events],np.int32);np.testing.assert_array_equal(self.f.words(result.projection,row),expected)
    self.assertEqual(account.stage[row,:int(account.count[row])].tolist(),[3]*int(account.count[row]))
  _,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
 def test_blocked_capped_nonzero_and_later_entry_stay_unknown_without_empty_span(self):
  s=self.state(8);bad=s._replace(substitute_hp=s.substitute_hp.at[0,1,0].set(5),ability=s.ability.at[1,1,0].set(3),boosts=s.boosts.at[2,1,0,0].set(-6).at[3,1,0,0].set(1))
  actual=self.initial(bad);self.assertTrue(bool(((actual.callbacks.refusal[:4]&~1)!=0).all()));self.assertEqual(actual.callbacks.execution.ledger.stage[:4,0].tolist(),[1]*4);self.assertEqual(actual.callbacks.execution.ledger.site[:4,0].tolist(),[int(ex.Site.INTIMIDATE_ATTEMPT)]*4)
  self.assertEqual(actual.callbacks.execution.ledger.stage[4:,0].tolist(),[2]*4);self.assertEqual(actual.callbacks.execution.ledger.site[4:,0].tolist(),[int(ex.Site.INTIMIDATE_INITIAL)]*4)
  later=actual.callbacks._replace(cursor=actual.cursor,words=j.zeros_like(actual.callbacks.words),count=j.zeros(8,j.int32),turn=actual.cursor.turn)
  # An ordinary entry retains no fresh-constructor exemption, even 0->-1.
  ledger=ex.initialize(later.stream_id,compiled=self.account);later=ex.begin_on_tape(later,ledger,self.account.device,active=j.ones(8,bool))
  _,_,out=entry.enter(self.state(8),0,self.table.entry,eligible=j.ones(8,bool),hazards=False,_weather_trace=later)
  self.assertTrue(bool((out.refusal&512).all()));self.assertEqual(out.count.tolist(),[0]*8)
 def before_entry(self,b=8):
  s=self.state(b);cursor=life.register(j.arange(1201,1201+b,dtype=j.int32),**self.kw)
  tape=life._tape(cursor,self.producer,64)._replace(refusal=j.ones(b,j.int32))
  tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=self.account),self.account.device,active=j.ones(b,bool))
  carry=risks.initialize_device(tape.stream_id,self.risk.device);carry=risks.begin_frame_device(carry,tape,self.risk.device,active=j.ones(b,bool))
  carry=risks.observe_constructor_device(carry,tape,s,self.risk.device,active=j.ones(b,bool))
  tape=tape._replace(execution=tape.execution._replace(risks=risks.Context(carry,self.risk.device)))
  for side in (0,1):
   tape=life._reveal(tape,s,j.full(b,side,j.int32),j.full(b,-1,j.int32),s.active[:,side],eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
  return s,tape
 def test_b8_prior_overflow_stale_stamp_and_inactive_exact_retention(self):
  s,tape=self.before_entry();mask=j.array([True]*4+[False]*4)
  tape=tape._replace(refusal=tape.refusal.at[1].set(257),count=tape.count.at[2].set(64),
      switch_details=tape.switch_details.at[2,63,0].set(1),event_generation=tape.event_generation.at[3,0,1].set(1))
  _,_,out=jax.jit(lambda t:entry.enter(s,0,self.table.entry,eligible=mask,hazards=False,_weather_trace=t))(tape)
  self.assertEqual(out.count.tolist(),[3,3,64,2,2,2,2,2]);self.assertTrue(bool(out.refusal[1]&256));self.assertTrue(bool(out.refusal[2]&events.OVERFLOW));self.assertTrue(bool(out.refusal[3]&512))
  for before,after in zip(jax.tree.leaves(tape),jax.tree.leaves(out)):
   if getattr(before,'ndim',0) and before.shape[0]==8:np.testing.assert_array_equal(before[4:],after[4:])
  for name,bad in (('global',tape._replace(execution=tape.execution._replace(risks=tape.execution.risks._replace(carry=tape.execution.risks.carry._replace(binding=tape.execution.risks.carry.binding.at[0].add(1)))))),
                   ('constructor',tape._replace(execution=tape.execution._replace(risks=tape.execution.risks._replace(carry=tape.execution.risks.carry._replace(counts=tape.execution.risks.carry.counts.at[:,0].set(0)))))),
                   ('generation',tape._replace(cursor=tape.cursor._replace(generation=j.ones((8,2),j.int32))))):
   with self.subTest(name=name):
    _,_,rejected=entry.enter(s,0,self.table.entry,eligible=mask,hazards=False,_weather_trace=bad)
    self.assertEqual(rejected.count.tolist(),bad.count.tolist());self.assertTrue(bool((rejected.refusal[:4]&512).all()))
 def test_default_entry_ast_unchanged_and_original_gen3_callback_pair(self):
  def body(text):return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='enter')
  before=subprocess.check_output(['git','show','ad6f1d9e:gpu_engine/entry.py'],text=True);self.assertEqual(dump_default_program(body(before)),dump_default_program(body(Path('gpu_engine/entry.py').read_text())))
  node=shutil.which('node')
  if node is None:self.skipTest('node unavailable')
  script="const {Dex}=require(process.argv[1]+'/dist/sim/dex.js');const a=Dex.mod('gen3').abilities.get('intimidate');const lines=[];const q={volatiles:{},toString:()=> 'p2a: Celebi'};const p={adjacentFoes:()=>[q],toString:()=> 'p1a: Arbok'};a.onStart.call({add:(...x)=>lines.push(x.map(String).join('|')),boost:(x,t,s,e,f)=>{if(x.atk!==-1||t!==q||s!==p||e!==null||f!==true)throw Error('wrong args');lines.push('-unboost|'+q+'|atk|1')}},p);process.stdout.write(JSON.stringify(lines));"
  self.assertEqual(json.loads(subprocess.check_output([node,'-e',script,showdown_root_str()],text=True)),['-ability|p1a: Arbok|Intimidate|boost','-unboost|p2a: Celebi|atk|1'])
