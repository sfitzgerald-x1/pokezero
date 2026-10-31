"""Prospective measured-family fixtures only; no saved-root replay or games."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import entry, engine_public_events as events, public_belief, public_wire, v4_encoder as v4
from gpu_engine import engine_lifecycle_callbacks as life, engine_execution_completeness as ex
from gpu_engine import engine_weather_speed_profile as speed, engine_public_event_device as base
from gpu_engine import engine_full_v4_event_device as full, engine_trace_callbacks as trace
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from pokezero.category_vocab import CategoryVocabulary
import test_gpu_engine_trace_owning_event as old
from gpu_engine_default_observer_ast import dump_default_program


class MeasuredCopies(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  old.TraceOwningEventTests.setUpClass();f=old.TraceOwningEventTests
  cls.f=f.f;cls.metadata=deepcopy(f.metadata)
  cls.metadata['vocabulary']['abilities']=['_none','pressure','trace','sturdy','naturalcure','oblivious','levitate']
  variants={}
  for species in ('snorlax','absol','gardevoir'):
   abilities=('Sturdy','Natural Cure','Oblivious') if species=='absol' else ('Trace',)
   variants[species]=Gen3RandbatSpeciesUniverse(species,80,tuple(Gen3RandbatVariant(
    'literal-'+species+'-'+ability,'fixture',species,'fixture',80,('Surf',),ability,'Leftovers')for ability in abilities))
  source=public_belief.compile_source(Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-measured-trace'),variants))
  vocabulary=CategoryVocabulary(tuple(sorted(set(f.catalog.vocabulary.tokens)|{
   'ability:naturalcure','belief:possible_ability:naturalcure','ability:oblivious','belief:possible_ability:oblivious'})))
  cls.catalog=v4.compile_encoder(source,f.catalog.dex,vocabulary,checkpoint_sha256='a'*64,feature_masks=old.fixture.encoder_fixture.V4EncoderTests.masks)
  cls.table=entry.from_metadata(cls.metadata);cls.engine=cls.f.engine._replace(entry=cls.table)
  cls.producer=events.compile_catalog(cls.metadata,cls.engine,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=old.fixture.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog);cls.account=ex.compile_accounting(cls.adapter)

 fixture=old.TraceOwningEventTests.fixture
 same=old.TraceOwningEventTests.same

 def case(self,b=2,*,constructor=False):
  s,t,before,d,_=self.fixture(b);rows=j.arange(b);donor=4+(rows%2)
  s=s._replace(ability=s.ability.at[:,1,5].set(donor),base_ability=s.base_ability.at[:,1,5].set(donor))
  d=d._replace(abilities=j.ones(7,bool))
  if constructor:
   s=s._replace(active=j.zeros((b,2),j.int32),entry_pending=j.ones((b,2),bool),
    ability=s.ability.at[:,0,3].set(3).at[:,0,0].set(2).at[:,1,0].set(donor),
    base_ability=s.base_ability.at[:,0,3].set(3).at[:,0,0].set(2).at[:,1,0].set(donor))
  return s,t,before,d,speed.qualify_birth_device(s,t.stream_id,d)

 def test_actual_constructor_two_copies_both_full_owners_and_original_words(self):
  s,t,_,d,birth=self.case(constructor=True);fresh=life.register(t.stream_id,catalog=self.producer,
   expected_binding=self.producer.binding,expected_engine_binding=np.array(self.producer.engine_binding,np.uint32))
  frame=life.initialize_with_lifecycle(s,self.engine,cursor=fresh,catalog=self.producer,
   expected_binding=self.producer.binding,expected_engine_binding=np.array(self.producer.engine_binding,np.uint32),
   tie_coin=j.zeros(2,j.int32),execution_compiled=self.account)
  plain=old.fixture.engine_fixture.EnginePublicEventsTests.t.initialize(s,self.engine,tie_coin=j.zeros(2,j.int32))
  self.same(frame.state,plain);self.assertEqual((frame.callbacks.refusal&~1).tolist(),[0,0])
  self.assertEqual(frame.state.ability[:,0,0].tolist(),[4,5]);self.assertEqual(frame.state.base_ability[:,0,0].tolist(),[2,2])
  self.assertFalse(bool(speed.retain_device(birth,frame.state,t.stream_id,d).qualified_birth.any()))
  retained=speed.retain_frame_device(birth,frame.state,t.stream_id,d,frame.callbacks)
  self.assertTrue(bool(retained.qualified_birth.all()));self.assertEqual(retained.current_ability[:,0,0].tolist(),[4,5])
  ledger=frame.callbacks.execution.ledger;owners=[]
  consume=jax.jit(lambda stream,public:ex.consume_frame_device(ledger,frame,stream,public,active=j.ones(2,bool),
   tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog))
  for owner in (0,1):
   stream=full.register(t.stream_id,j.full(2,owner,j.int32),compiled=self.adapter)
   public=v4.initialize(self.catalog,stream_ids=t.stream_id,owner_sides=j.full(2,owner,j.int32))
   account,consumed=consume(stream,public);owners.append(account)
   self.assertEqual(consumed.projection.error.tolist(),[0,0]);self.assertEqual(account.stage[:,0].tolist(),[3,3])
   for row,ability in enumerate(('Natural Cure','Oblivious')):
    actor=self.catalog.species.names[int(self.adapter.device.causal.species[1])]
    donor=self.catalog.species.names[int(self.adapter.device.causal.species[2])]
    lines=[f'|switch|p1a: {actor}|{actor}, L80, M|100/100',f'|switch|p2a: {donor}|{donor}, L80, M|100/100',
     f'|-ability|p1a: {actor}|{ability}|Trace|[from] ability: Trace|[of] p2a: {donor}','|turn|1']
    _,packets=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog))
    expected=np.array([e.words()for p in packets for e in p.events],np.int32)
    np.testing.assert_array_equal(self.f.words(consumed.projection,row),expected)
   self.assertFalse(bool(consumed.projection.B4_qualified.any()))
  closed,good=ex.join_consumed_owners(*owners,tables=self.account.device);self.assertTrue(bool(good.all()))
  # Genuine departure and installation restore this member's original Trace.
  departed=entry.depart(frame.state,0,self.table,eligible=j.ones(2,bool));departed=departed._replace(active=departed.active.at[:,0].set(1))
  after=life._tape(frame.cursor,self.producer,64);after=ex.begin_on_tape(after,closed,self.account.device,active=j.ones(2,bool))
  after=life.installed(after,departed,j.zeros(2,j.int32),j.zeros(2,j.int32),j.ones(2,j.int32),eligible=j.ones(2,bool),replacement=j.zeros(2,bool),baton=j.zeros(2,bool))
  reset=speed.retain_frame_device(retained,departed,t.stream_id,d,after)
  self.assertTrue(bool(reset.qualified_birth.all()));self.assertEqual(reset.current_ability[:,0,0].tolist(),[2,2])
  self.assertFalse(bool(speed.retain_frame_device(retained,departed,t.stream_id,d,frame.callbacks).qualified_birth.any()))

 def test_mixed_b8_later_entry_unknown_sticky_and_source_profile_guards(self):
  s,t,_,d,birth=self.case(8);mask=j.array([True]*6+[False]*2)
  # Copies2/3 unsupported Pressure/Levitate; inactive6/7 retain every leaf.
  ids=j.array([4,5,1,6,4,5,4,5],j.int32)
  s=s._replace(ability=s.ability.at[:,1,5].set(ids),base_ability=s.base_ability.at[:,1,5].set(ids))
  birth=speed.qualify_birth_device(s,t.stream_id,d);t=t._replace(refusal=t.refusal.at[4].set(65))
  run=jax.jit(lambda tape:entry.enter(s,0,self.table,eligible=mask,hazards=False,_weather_trace=tape))
  after,_,out=run(t);plain,_=entry.enter(s,0,self.table,eligible=mask,hazards=False);self.same(after,plain)
  self.assertEqual(out.count.tolist(),[3,3,2,2,3,3,2,2]);self.assertTrue(bool(out.refusal[2]&256));self.assertTrue(bool(out.refusal[3]&256));self.assertTrue(bool(out.refusal[4]&64))
  retained=speed.retain_frame_device(birth,after,t.stream_id,d,out)
  self.assertEqual(retained.qualified_birth.tolist(),[True,True,False,False,True,True,True,True])
  for a,b in zip(jax.tree.leaves(t),jax.tree.leaves(out)):
   if getattr(a,'ndim',0) and a.shape[0]==8:np.testing.assert_array_equal(a[6:],b[6:])
  for mutated in (out._replace(event_generation=out.event_generation.at[0,2,0].add(1)),
   out._replace(words=out.words.at[0,2,7].set(5)),
   out._replace(source_binding=j.zeros(8,j.uint32)),
   out._replace(execution=out.execution._replace(ledger=out.execution.ledger._replace(claimed=out.execution.ledger.claimed.at[0,2].set(0))))):
   self.assertFalse(bool(speed.retain_frame_device(birth,after,t.stream_id,d,mutated).qualified_birth[0]))
  poisoned=birth._replace(qualified_birth=birth.qualified_birth.at[0].set(False))
  self.assertFalse(bool(speed.retain_frame_device(poisoned,after,t.stream_id,d,out).qualified_birth[0]))
  wrong=t._replace(cursor=t.cursor._replace(physical_active=t.cursor.physical_active.at[0,1].set(4)))
  _,_,bad=run(wrong);self.assertTrue(bool(bad.refusal[0]&256));self.assertTrue(bool(bad.refusal[0]&512))

 def test_closed_compiler_vocabulary_default_math_and_pinned_callback(self):
  self.assertEqual(self.table.trace_copy.tolist(),[False,False,False,True,True,True,False])
  mapping=np.asarray(self.adapter.device.causal.trace_public_ability)
  self.assertEqual(np.nonzero(mapping)[0].tolist(),[3,4,5])
  for engine_id,name in ((3,'sturdy'),(4,'naturalcure'),(5,'oblivious')):self.assertEqual(int(mapping[engine_id]),self.catalog.abilities.ids[name])
  self.assertEqual(np.nonzero(np.asarray(self.adapter.device.trace_copy))[0].tolist(),[3,4,5])
  for file in ('transition.py','mechanics.py','engine_execution_completeness.py','engine_weather_speed_profile.py'):
   self.assertEqual(Path('gpu_engine',file).read_bytes(),subprocess.check_output(['git','show','552bd326:gpu_engine/'+file]))
  old_entry=subprocess.check_output(['git','show','552bd326:gpu_engine/entry.py'],text=True)
  enter=lambda source:next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='enter')
  self.assertEqual(dump_default_program(enter(old_entry)),dump_default_program(enter(Path('gpu_engine/entry.py').read_text())))
  root=Path(os.environ['POKEZERO_SHOWDOWN_ROOT']);self.assertEqual(subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'f76228a1354b5d0f307ca2d16101294ad3a2308b')
  script=r'''
const path=require('node:path');const root=process.argv[1];const {Dex}=require(path.join(root,'dist/sim/dex.js'));const {Pokemon}=require(path.join(root,'dist/sim/pokemon.js'));
const dex=Dex.mod('gen3');const trace=dex.abilities.get('trace');const results=[];
for(const id of ['sturdy','naturalcure','oblivious']){const lines=[],calls=[];const donor={getAbility:()=>dex.abilities.get(id),toString:()=> 'p2a: Donor'};
const battle={gen:3,dex,effect:trace,runEvent:()=>true,singleEvent:id=>calls.push(id),initEffectState:x=>x,add:(...xs)=>lines.push(xs.map(String).join('|'))};
const p={hp:267,ability:'trace',abilityState:{},battle,side:{randomFoe:()=>donor},getAbility(){return dex.abilities.get(this.ability)},setAbility:Pokemon.prototype.setAbility,toString:()=> 'p1a: Actor'};
trace.onStart.call(battle,p);results.push({ability:p.ability,lines,calls,update:typeof trace.onUpdate});}process.stdout.write(JSON.stringify(results));
'''
  actual=json.loads(subprocess.check_output([shutil.which('node'),'-e',script,str(root)],text=True))
  for value,name in zip(actual,('Sturdy','Natural Cure','Oblivious')):
   self.assertEqual(value,dict(ability=name.lower().replace(' ',''),lines=[f'-ability|p1a: Actor|{name}|Trace|[from] ability: Trace|[of] p2a: Donor'],calls=['End'],update='undefined'))


if __name__=='__main__':unittest.main()
