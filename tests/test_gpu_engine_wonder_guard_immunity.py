"""Original zero-ratio Wonder Guard attribution; typed components only."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,subprocess,unittest
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached,normalize_id
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_no_damage_callbacks as nd,engine_direct_hit_controls as hit,engine_kernel_frame_risk as risk
from gpu_engine import engine_execution_completeness as ex,engine_lifecycle_callbacks as life,engine_public_events as events
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_leftovers_callbacks as leftovers,engine_full_v4_event_device as full
from gpu_engine import public_belief,v4_encoder as v4,public_wire,public_inputs as inputs,transition,absorption
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixture

class WonderGuardImmunityTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.NoDamageOwningTests.setUpClass();f=fixture.NoDamageOwningTests;cls.f=f
  cls.metadata=deepcopy(f.metadata);cls.metadata['vocabulary']['abilities'].append('wonderguard');cls.wg=cls.metadata['vocabulary']['abilities'].index('wonderguard')
  dex=load_showdown_dex_cached(showdown_root());species=dict(f.dex.species,shedinja=dex.species_info('shedinja'))
  cls.metadata['vocabulary']['species'].append('shedinja');cls.dex=ShowdownDex(f.dex.moves,species,dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant(name+str(i),'authored',name,'authored',80,f.names,abi,'Leftovers')
      for i,abi in enumerate(('Pressure','Wonder Guard'))))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-wonder-guard-type-immunity'),variants)
  labels=set(f.catalog.vocabulary.tokens)|{'ability:wonderguard','belief:possible_ability:wonderguard','species:shedinja'}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  cls.table=transition.tables_from_metadata(cls.metadata);cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,
      exposure=events.HealthExposure(events.SHOWDOWN,'gen3randombattle',True,False))
  rules=request.load_source_rules(showdown_root(),cls.metadata)
  req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=cls.producer)
  facts=own.compile_own_facts(req,cls.catalog);cls.helper=hit.compile_direct_hit_controls(engine_catalog=cls.producer)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',
      leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',
      fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared)

 def component(self,b=8,*,plain=False,inactive=False):
  rows=j.arange(b);side=rows%2;z=j.zeros(b,j.int32);yes=j.ones(b,bool);no=j.zeros(b,bool);active=yes.at[-1].set(False)if inactive else yes
  physical=z+3;target=z+4;move=z+self.metadata['vocabulary']['moves'].index('earthquake')
  cursor=life.register(rows+19801,catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32))
  phys=j.zeros((b,2),j.int32).at[rows,side].set(physical).at[rows,1-side].set(target)
  mapping=j.full((b,2,6),-1,j.int32).at[:,:,0].set(0).at[:,:,1].set(1).at[rows,side,physical].set(2).at[rows,1-side,target].set(2)
  cursor=cursor._replace(generation=j.full((b,2),3,j.int32),turn=z+1,initialized=yes,physical_active=phys,public_slot=j.full((b,2),2,j.int32),
      encounter=j.full((b,2),3,j.int32),established=j.ones((b,2),bool),physical_to_public=mapping,next_slot=j.full((b,2),3,j.int32))
  tape=life._tape(cursor,self.producer,64);account=self.runtime['execution_compiled'];r=self.runtime['kernel_risk_compiled']
  tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes)
  carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,r.device),tape,r.device,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,r.device)))
  tape=events.move(tape,side,physical,target,move,eligible=active);tape,ticket=ex.open_attempt(tape,ex.Site.DIRECT_HIT,1-side,active=active)
  abi=z+(self.metadata['vocabulary']['abilities'].index('pressure')if plain else self.wg)
  before=nd.Before(yes,z+self.metadata['vocabulary']['types'].index('ground'),abi,z,z+100,z+100,j.zeros(b,j.uint32),z,z+100,no,z,z+70)
  p=nd.prepare(tape,ticket,side,physical,target,move,before,self.runtime['direct_hit_context'],requested=active)
  actual=absorption.try_hit(self.table.absorption,before.ability,move,before.move_type,before.status,j.full((b,2),1,j.int32),before.hp,before.maximum,before.sub_hp,before.volatile,eligible=active)
  p=nd.observed_absorption(p,actual)
  w=hit.HitWrite(z+100,no,yes,yes,z,z+1,no,no,no,before.hp,before.hp,before.maximum)
  o=risk.Outcome(w.power,w.critical,w.hit,w.immune,w.numerator,w.denominator,p.absorbed.blocked,no,no,no,z,no)
  tail=(z+100,before.volatile,z,before.actor_hp_type,before.actor_hp_power)
  out=nd.finish(tape,p,w,o,*tail)
  return out,p,w,o,tail

 def test_original_gen3_handler_and_default_physics_bytes(self):
  script=r'''const root=process.argv[1],{Dex}=require(root+'/dist/sim/dex.js');const d=Dex.mod('gen3'),handler=d.abilities.get('wonderguard').onTryHit,out=[];
for(const [category,type,eff,imm]of [['Physical','Ground',0,false],['Special','Psychic',0,false],['Physical','Fire',1,true],['Status','Psychic',0,false],['Physical','???',0,false]]){
const lines=[],source={toString:()=> 'p1a: Actor'},target={toString:()=> 'p2a: Shedinja',runEffectiveness:()=>eff,runImmunity:()=>imm};
const result=handler.call({debug:()=>{},add:(...v)=>lines.push(v.map(String).join('|'))},target,source,{id:'hiddenpower',category,type});out.push({lines,blocked:result===null});}
process.stdout.write(JSON.stringify({out,sha:require('crypto').createHash('sha256').update(handler.toString()).digest('hex')}));'''
  result=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True))
  self.assertEqual([x['blocked']for x in result['out']],[True,True,False,False,False]);self.assertEqual(result['out'][0]['lines'],['-immune|p2a: Shedinja|[from] ability: Wonder Guard'])
  for n in ('transition.py','mechanics.py','secondary.py','entry.py','state.py','engine_no_damage_callbacks.py','engine_direct_hit_controls.py'):
   self.assertEqual(Path('gpu_engine',n).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+n]))
  self.assertEqual(int(self.runtime['public'].device.causal.wonder_guard_engine_ability),self.wg)
  self.assertEqual(int(self.runtime['public'].device.causal.wonder_guard_public_ability),self.catalog.abilities.ids['wonderguard'])

 def test_source_snapshot_namespace_and_zero_ratio_scope(self):
  t,p,w,o,tail=self.component(inactive=True);self.assertEqual(t.count.tolist(),[3]*7+[0]);self.assertEqual(t.words[:7,1,6].tolist(),[nd.TYPE_IMMUNE]*7)
  captured=nd.Captured(p,w,*tail);self.assertEqual(nd.captured_matches(t,captured,o).tolist(),[True]*7+[False])
  self.assertEqual(nd.owned_pair(t,j.array(1,j.int32),self.helper.device.no_damage)[0].tolist(),[True]*7+[False])
  for label,bad in (('ability',t._replace(words=t.words.at[0,2,6].set(0))),('clock',t._replace(event_turn=t.event_turn.at[0,2].add(1))),
      ('claim',t._replace(execution=t.execution._replace(ledger=t.execution.ledger._replace(claimed=t.execution.ledger.claimed.at[0,2].set(0)))))):
   with self.subTest(label=label):self.assertFalse(bool(nd.owned_pair(bad,j.array(1,j.int32),self.helper.device.no_damage)[0][0]))
  # This attribution repair cannot create the unimplemented nonzero-ratio WG physics.
  self.assertFalse(bool(nd.scheduled(p,w._replace(numerator=j.ones(8,j.int32))).any()))
  prior=t._replace(refusal=t.refusal.at[0].set(65));self.assertEqual(int(prior.refusal[0])&65,65)
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   foreign=t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))})
   self.assertFalse(bool(nd.captured_matches(foreign,captured,o).any()))

 def test_both_full_consumers_exact_attributed_words_and_revealed_ability(self):
  kw=v4.wire_kwargs(self.catalog);ns={k:np.asarray(getattr(self.runtime['public'].device.namespaces,k))for k in full._NAMES}
  fold=jax.jit(lambda stream,frame,pub:full.project_and_fold(stream,frame,self.runtime['public'].device,pub,self.catalog))
  engine_ids=self.metadata['vocabulary']['species'];move=self.dex.move_info('earthquake').name
  for plain in (False,True):
   t,p,w,o,tail=self.component(b=2,plain=plain);base=t.cursor
   # Three distinct literal public encounters make current slot2/encounter3.
   species_names=[n for n in self.catalog.species.names[1:]if normalize_id(n)!='shedinja'][:3]
   initial=[];expected=[];wires=[]
   for row in (0,1):
    side=row;names=[species_names[2],species_names[2]];names[1-side]=self.catalog.species.names[self.catalog.species.ids['shedinja']];ident=[f'p{i+1}a: '+names[i].title()for i in (0,1)]
    lines=[]
    for earlier in species_names[:2]:
     for owner in (0,1):lines.append(f'|switch|p{owner+1}a: {earlier}|{earlier}, L80, M|100/100')
    for owner in (0,1):lines.append(f'|switch|{ident[owner]}|{names[owner]}, L80, M|100/100')
    lines.append('|turn|1');wire,packets=public_wire.project_lines(public_wire.WireState(),lines,**kw);wires.append(wire);initial.append([e for q in packets for e in q.events])
    ending=''if plain else '|[from] ability: Wonder Guard'
    _,packets=public_wire.project_lines(wire,[f'|move|{ident[side]}|{move}|{ident[1-side]}',f'|-immune|{ident[1-side]}'+ending],**kw)
    expected.append(np.array([e.words()for q in packets for e in q.events],np.int32))
   for owner in (0,1):
    pub=v4.initialize(self.catalog,stream_ids=t.stream_id,owner_sides=j.full(2,owner,j.int32))
    for chunk in inputs.pack_chunks(np.asarray(t.stream_id),initial,**ns):pub,error=v4.fold_chunk(pub,chunk,self.catalog);self.assertFalse(bool(error.any()))
    current_species=np.array([[engine_ids.index(normalize_id(self.catalog.species.names[int(pub.mons.species[r,s,2])]))for s in (0,1)]for r in (0,1)],np.int32)
    cursor=base._replace(revealed_species_id=j.array(current_species));t=t._replace(cursor=cursor)
    stream=full.register(t.stream_id,j.full(2,owner,j.int32),compiled=self.runtime['public'])
    stream=stream._replace(causal=stream.causal._replace(cursor=cursor,sequence=j.full(2,7,j.int32)),sequence=pub.cursor.sequence,
        live=j.ones((2,2),bool),species=pub.mons.species,status=pub.mons.status,hp=pub.mons.hp_numerator,maximum=pub.mons.hp_denominator)
    frame=life.LifecycleResult(None,cursor,t,life.snapshot(cursor),life.snapshot(cursor),j.zeros(2,bool),j.zeros(2,j.int32))
    got=fold(stream,frame,pub);self.assertEqual(got.projection.error.tolist(),[0,0])
    for row in (0,1):
     actual=np.asarray(got.projection.frames.words).transpose(1,0,2,3)[row][np.asarray(got.projection.frames.valid).transpose(1,0,2)[row]]
     np.testing.assert_array_equal(actual,expected[row])
     ability=int(got.public.context.abilities.original[row,1-row,2]);self.assertEqual(ability,0 if plain else self.catalog.abilities.ids['wonderguard'])
    if not plain:
     missing=self.runtime['public'].device._replace(causal=self.runtime['public'].device.causal._replace(wonder_guard_public_ability=j.array(0,j.int32)))
     refused=full.project_device(stream,t,life.snapshot(cursor),life.snapshot(cursor),missing)
     self.assertFalse(bool(refused.component_admitted.any()))
