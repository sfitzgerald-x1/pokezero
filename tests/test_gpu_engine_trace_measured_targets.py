"""New literal constructor components for measured Trace targets; no games/replay."""
from _showdown_root import showdown_root_str
import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import entry,engine_trace_callbacks as trace,engine_public_events as events
from gpu_engine import engine_lifecycle_callbacks as life,engine_execution_completeness as ex
from gpu_engine import engine_full_v4_event_device as full,engine_public_event_device as base
from gpu_engine import public_belief,v4_encoder as v4,public_wire,engine_source_frame_witness as witness
from pokezero.dex import ShowdownDex,SpeciesInfo
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
import test_gpu_engine_pressure_entry_owning as fixture
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected

NAMES=('porygon2','gengar','articuno')
BASES=((85,80,90,105,95,60),(60,65,60,130,75,110),(90,85,100,95,125,85))
ABILITIES=('_none','trace','levitate','pressure','sturdy','naturalcure','oblivious','forecast','unmappedfixture')

class TraceMeasuredTargetsTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.PressureEntryOwningTests.setUpClass();cls.f=fixture.PressureEntryOwningTests()
  cls.metadata=deepcopy(cls.f.metadata);cls.metadata['vocabulary'].update(species=['_none',*NAMES],abilities=list(ABILITIES))
  original=cls.f.catalog;stats_order=('hp','atk','def','spa','spd','spe')
  additions={name:SpeciesInfo(name,label,types,dict(zip(stats_order,stats)),weight) for name,label,types,stats,weight in zip(NAMES,('Porygon2','Gengar','Articuno'),(('Normal',),('Ghost','Poison'),('Ice','Flying')),BASES,(32.5,40.5,55.4))}
  dex=ShowdownDex(original.dex.moves,dict(original.dex.species,**additions),original.dex.type_chart)
  # Protect is a legal Gen3 TM for all three named species; no move is executed.
  variants={name:Gen3RandbatSpeciesUniverse(name,level,(Gen3RandbatVariant('literal-'+name,'fixture',name,'fixture',level,('Protect',),ability,'Leftovers'),)) for name,level,ability in zip(NAMES,(80,74,80),('Trace','Levitate','Pressure'))}
  source=public_belief.compile_source(Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-measured-trace-targets'),variants))
  labels=set(original.vocabulary.tokens)
  labels|={'species:'+n for n in NAMES}
  for ability in ABILITIES[1:-1]:labels|={'ability:'+ability,'belief:possible_ability:'+ability}
  cls.catalog=v4.compile_encoder(source,dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=fixture.encoder_fixture.V4EncoderTests.masks)
  cls.table=entry.from_metadata(cls.metadata);cls.engine=cls.f.engine._replace(entry=cls.table)
  cls.producer=events.compile_catalog(cls.metadata,cls.engine,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=fixture.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog);cls.account=ex.compile_accounting(cls.adapter)
 def same(self,a,b):
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
 def state(self):
  s=self.f.state(2);species=np.ones((2,2,6),np.int32);species[0,1]=2;species[1,1]=3
  levels=np.full_like(species,80);levels[0,1]=74;stats=np.zeros((2,2,6,6),np.int32)
  for row in range(2):
   for side in (0,1):
    base_stats=BASES[species[row,side,0]-1];level=int(levels[row,side,0])
    values=[((2*v+31+85//4)*level)//100+(level+10 if k==0 else 5) for k,v in enumerate(base_stats)]
    stats[row,side]=values
  abilities=np.ones_like(species);abilities[0,1]=2;abilities[1,1]=3
  protect=self.metadata['vocabulary']['moves'].index('protect');moves=np.zeros((2,2,6,4),np.int32);moves[...,0]=protect
  pp=np.zeros_like(moves);pp[...,0]=16
  genders=np.zeros_like(species);genders[0,1]=1
  types=np.zeros((*species.shape,2),np.int32)
  for row in range(2):
   for side in (0,1):
    for k,name in enumerate(self.catalog.dex.species[NAMES[species[row,side,0]-1]].types):types[row,side,:,k]=self.metadata['vocabulary']['types'].index(name.lower())
  return s._replace(species=j.array(species),level=j.array(levels),gender=j.array(genders),types=j.array(types),stats=j.array(stats),
    hp=j.array(stats[...,0]),max_hp=j.array(stats[...,0]),ability=j.array(abilities),base_ability=j.array(abilities),
    active=j.zeros((2,2),j.int32),moves=j.array(moves),pp=j.array(pp),entry_pending=j.ones((2,2),bool),done=j.zeros(2,bool))
 def fresh(self,state):
  streams=j.array([99101,99102],j.int32)
  return life.register(streams,catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.array(self.producer.engine_binding,np.uint32))
 def test_exact_new_targets_and_original_three_keep_closed_masks(self):
  self.assertEqual(self.table.trace_copy.tolist(),[False,False,False,False,True,True,True,False,False])
  self.assertEqual((self.table.trace_copy|self.table.levitate|self.table.pressure).tolist(),[False,False,True,True,True,True,True,False,False])
  for idx in (2,3):
   self.assertGreater(int(self.adapter.device.trace_copy[idx]),0,msg=ABILITIES[idx]);self.assertGreater(int(self.adapter.device.causal.trace_public_ability[idx]),0,msg=ABILITIES[idx])
  for idx in (0,1,7,8):self.assertEqual(int(self.adapter.device.trace_copy[idx]),0)
 def test_actual_constructor_measured_two_same_numerics_and_both_original_full_drains(self):
  state=self.state();cursor=self.fresh(state)
  actual=life.initialize_with_lifecycle(state,self.engine,cursor=cursor,catalog=self.producer,expected_binding=self.producer.binding,
    expected_engine_binding=np.array(self.producer.engine_binding,np.uint32),tie_coin=j.zeros(2,j.int32),execution_compiled=self.account)
  plain=fixture.engine_fixture.EnginePublicEventsTests.t.initialize(state,self.engine,tie_coin=j.zeros(2,j.int32));self.same(actual.state,plain)
  self.assertEqual(actual.state.ability[:,0,0].tolist(),[2,3]);self.assertEqual(actual.state.base_ability[:,0,0].tolist(),[1,1])
  self.assertEqual((actual.callbacks.refusal&~1).tolist(),[0,0]);tape=actual.callbacks
  trace_at=[int(np.flatnonzero(np.asarray(tape.words[row,:,0])==27)[0]) for row in range(2)]
  for row,idx in enumerate(trace_at):
   self.assertEqual(tape.words[row,idx].tolist(),[27,1,2+row,0,0,1,0,2+row,0,0])
  # Only the actual Articuno emits Pressure's owner-only Start; its copied
  # Pressure never starts again on Porygon2 under the original Gen3 law.
  self.assertEqual([sum(np.asarray(tape.words[row,:int(tape.count[row]),0])==24) for row in range(2)],[0,1])
  ledger=tape.execution.ledger
  consume=jax.jit(lambda stream,public:ex.consume_frame_device(ledger,actual,stream,public,active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog))
  owners=[]
  for owner in (0,1):
   stream=full.register(tape.stream_id,j.full(2,owner,j.int32),compiled=self.adapter)
   public=v4.initialize(self.catalog,stream_ids=tape.stream_id,owner_sides=j.full(2,owner,j.int32))
   account,consumed=consume(stream,public);owners.append(account)
   self.assertEqual(consumed.projection.error.tolist(),[0,0]);self.assertEqual(account.stage[:,0].tolist(),[3,3])
   for row,donor in enumerate(('Gengar','Articuno')):
    level=74 if row==0 else 80;hp=int(state.hp[row,1,0]);own_hp=267
    lines=[f'|switch|p1a: Porygon2|Porygon2, L80|{own_hp if owner==0 else 100}/{own_hp if owner==0 else 100}',
      f'|switch|p2a: {donor}|{donor}, L{level}'+(', M' if row==0 else '')+f'|{hp if owner==1 else 100}/{hp if owner==1 else 100}']
    if row==1 and owner==1:lines+=['|-ability|p2a: Articuno|Pressure|[silent]']
    lines+=[f'|-ability|p1a: Porygon2|{("Levitate","Pressure")[row]}|Trace|[from] ability: Trace|[of] p2a: {donor}']
    lines+=['|turn|1']
    _,packets=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog))
    expected=np.array([event.words() for packet in packets for event in packet.events],np.int32)
    np.testing.assert_array_equal(self.f.words(consumed.projection,row),expected)
  closed,good=ex.join_consumed_owners(*owners,tables=self.account.device);self.assertEqual(good.tolist(),[True,True])
 def test_mixed_unsupported_forged_actor_and_prior_refusal_stay_unknown(self):
  state=self.state();cursor=self.fresh(state);tape=life._tape(cursor,self.producer,64)
  tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=self.account),self.account.device,active=j.ones(2,bool))
  for side in (0,1):tape=life._reveal(tape,state,j.full(2,side,j.int32),j.full(2,-1,j.int32),j.zeros(2,j.int32),eligible=j.ones(2,bool),replacement=j.zeros(2,bool),baton=j.zeros(2,bool),initial=True)
  # Existing false capability, mapped-but-source-banned Forecast, stays UNKNOWN.
  state=state._replace(ability=state.ability.at[0,1,0].set(7),base_ability=state.base_ability.at[0,1,0].set(7))
  _,_,out=entry.enter(state,0,self.table,eligible=j.ones(2,bool),hazards=False,_weather_trace=tape)
  self.assertTrue(int(out.refusal[0])&trace.UNKNOWN_TRACE);self.assertFalse(bool(out.words[0,:,0].tolist().count(27)))
  self.assertEqual(out.execution.ledger.stage[:,0].tolist(),[1,2])
  forged=tape._replace(cursor=tape.cursor._replace(physical_active=tape.cursor.physical_active.at[1,0].set(1)),refusal=tape.refusal.at[0].set(64))
  _,_,bad=entry.enter(state,0,self.table,eligible=j.ones(2,bool),hazards=False,_weather_trace=forged)
  self.assertTrue(int(bad.refusal[1])&trace.UNKNOWN_TRACE);self.assertTrue(int(bad.refusal[0])&64)
 def test_default_core_and_pinned_setability_source(self):
  for name in ('entry.py','transition.py','mechanics.py','state.py'):
   before=subprocess.check_output(['git','show','27fa443a:'+str(Path('gpu_engine')/name)])
   after=subprocess.check_output(['git','show','4d294c5d:'+str(Path('gpu_engine')/name)])
   if name=='transition.py':
    original=ast.parse(before);current=ast.parse(after)
    self.assertEqual(dump_default_program(current),dump_default_program(original))
    assert_hp_rng_mutations_detected(self,current,dump_default_program(original))
   else:self.assertEqual(after,before)
   # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
   # entry.py and state.py are byte-certified against the reviewed SOURCE_PINS below and by transition_observer_wiring.
   if name not in('transition.py','entry.py','state.py'):
    self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show','b8092a85:'+str(Path('gpu_engine')/name)]))
  for name,expected in witness.SOURCE_PINS.items():self.assertEqual(hashlib.sha256(Path('gpu_engine',name).read_bytes()).hexdigest(),expected)
  root=Path(showdown_root_str())
  for name in ('sim/pokemon.ts','data/mods/gen3/abilities.ts'):
   self.assertEqual((root/name).read_bytes(),subprocess.check_output(['git','show','f76228a:'+name],cwd=root))
  self.assertIn('this.battle.gen > 3',(root/'sim/pokemon.ts').read_text())

if __name__=='__main__':unittest.main()
