"""Literal observed R10 Pressure naming plus prospective mixed-name controls.

No stored engine root is replayed. Static original Dex rows and tiny entry/FULL
components exercise original writer words; no Battle/Native/model is created.
"""
from copy import deepcopy
from pathlib import Path
import ast,json,subprocess,shutil,unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import entry,engine_public_events as events,engine_lifecycle_callbacks as life
from gpu_engine import engine_public_event_device as base,engine_full_v4_event_device as full,v4_encoder as v4,public_wire
from pokezero.dex import ShowdownDex,SpeciesInfo,normalize_id
from pokezero.category_vocab import CategoryVocabulary
import test_gpu_engine_trace_owning_event as fixture

class AbilityIdentLexicalTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.TraceOwningEventTests.setUpClass();cls.f=fixture.TraceOwningEventTests()
  node=shutil.which('node')
  if node is None:raise unittest.SkipTest('node unavailable')
  cls.root=Path('/Users/scott/workspace/agents/pokemon-showdown-timing-smoke-f76228a')
  script="const {Dex}=require(process.argv[1]+'/dist/sim/dex.js');const d=Dex.mod('gen3');process.stdout.write(JSON.stringify(['deoxys','deoxysattack','deoxysdefense','deoxysspeed','hooh','nidoranf','nidoranm','marowak'].map(x=>{let s=d.species.get(x);return {id:s.id,name:s.name,baseSpecies:s.baseSpecies,types:s.types,baseStats:s.baseStats,weightkg:s.weightkg}})))"
  cls.rows=json.loads(subprocess.check_output([node,'-e',script,str(cls.root)],text=True))
  species=dict(cls.f.catalog.dex.species)
  for row in cls.rows:species[row['id']]=SpeciesInfo(row['id'],row['name'],tuple(row['types']),row['baseStats'],row['weightkg'])
  old=cls.f.catalog;dex=ShowdownDex(old.dex.moves,species,old.dex.type_chart)
  vocab=CategoryVocabulary(tuple(sorted(set(old.vocabulary.tokens)|{'species:'+r['id'] for r in cls.rows})))
  cls.catalog=v4.compile_encoder(old.source,dex,vocab,checkpoint_sha256='a'*64,feature_masks=fixture.fixture.encoder_fixture.V4EncoderTests.masks)
  cls.metadata=deepcopy(cls.f.metadata);cls.metadata['vocabulary']['species']+= [r['id'] for r in cls.rows]
  cls.ids={name:i for i,name in enumerate(cls.metadata['vocabulary']['species'])}
  cls.table=entry.from_metadata(cls.metadata);cls.engine=cls.f.engine._replace(entry=cls.table)
  cls.producer=events.compile_catalog(cls.metadata,cls.engine,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=fixture.fixture.engine_fixture.EnginePublicEventsTests.exposure)
  cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog)
  cls.fast=staticmethod(jax.jit(lambda stream,t,a,z:full.project_device(stream,t,a,z,cls.adapter.device)))
 def packet(self,*,trace=False):
  names=['deoxysspeed','deoxysattack','deoxysdefense','hooh','snorlax','nidoranf','nidoranm','gardevoir'];b=len(names)
  s=self.f.f.state(b);side=j.arange(b,dtype=j.int32)%2 if trace else j.zeros(b,j.int32);rows=j.arange(b);foe=1-side
  s=s._replace(done=j.zeros(b,bool),ability=j.zeros_like(s.ability),base_ability=j.zeros_like(s.base_ability))
  actor=s.active[rows,side];donor=s.active[rows,foe]
  if trace:
   s=s._replace(species=s.species.at[rows,side,actor].set(self.ids['gardevoir']).at[rows,foe,donor].set(j.array([self.ids[n] for n in names],j.int32)),
     ability=s.ability.at[rows,side,actor].set(2).at[rows,foe,donor].set(3),base_ability=s.base_ability.at[rows,side,actor].set(2).at[rows,foe,donor].set(3))
  else:
   s=s._replace(species=s.species.at[rows,side,actor].set(j.array([self.ids[n] for n in names],j.int32)).at[rows,foe,donor].set(self.ids['marowak']),
     ability=s.ability.at[rows,side,actor].set(1),base_ability=s.base_ability.at[rows,side,actor].set(1))
   # Exact observed R10 p1 source details/condition on the measured forme.
   s=s._replace(level=s.level.at[0,0,3].set(76).at[0,1,5].set(83),gender=s.gender.at[0,0,3].set(0).at[0,1,5].set(2),hp=s.hp.at[0,0,3].set(201),max_hp=s.max_hp.at[0,0,3].set(201))
  cursor=life.register(j.arange(1101,1101+b,dtype=j.int32),**self.kw);before=life.snapshot(cursor);t=life._tape(cursor,self.producer,64)
  for seat in (0,1):t=life._reveal(t,s,j.full(b,seat,j.int32),j.full(b,-1,j.int32),s.active[:,seat],eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
  after,_,t=entry.enter(s,side,self.table,eligible=j.ones(b,bool),hazards=False,_weather_trace=t)
  plain,_=entry.enter(s,side,self.table,eligible=j.ones(b,bool),hazards=False)
  for a,z in zip(jax.tree.leaves(after),jax.tree.leaves(plain)):np.testing.assert_array_equal(a,z)
  t=t._replace(cursor=t.cursor._replace(initialized=j.ones(b,bool)));t=life._turn(t,j.ones(b,bool))
  return s,t,before,side
 def expected(self,s,row,owner,actor,*,trace=False):
  names=[self.metadata['vocabulary']['species'][int(s.species[row,side,int(s.active[row,side])])] for side in (0,1)]
  data={r['id']:r for r in self.rows};idents={};lines=[]
  for side,name in enumerate(names):
   detail=self.catalog.species.names[self.catalog.species.species_id(name)];ident=data.get(name,dict(baseSpecies=detail))['baseSpecies'];idents[side]=f'p{side+1}a: '+ident
   phys=int(s.active[row,side]);level=int(s.level[row,side,phys]);gender={0:'',1:', M',2:', F'}[int(s.gender[row,side,phys])]
   hp=int(s.hp[row,side,phys]);maximum=int(s.max_hp[row,side,phys])
   if side!=owner:hp=100;maximum=100
   lines.append(f'|switch|{idents[side]}|{detail}, L{level}{gender}|{hp}/{maximum}')
  if trace:lines.append(f'|-ability|{idents[actor]}|Sturdy|Trace|[from] ability: Trace|[of] {idents[1-actor]}')
  elif owner==actor:lines.append(f'|-ability|{idents[actor]}|Pressure|[silent]')
  lines.append('|turn|1')
  if row==0 and owner==0 and not trace:
   self.assertEqual(lines[0],'|switch|p1a: Deoxys|Deoxys-Speed, L76|201/201');self.assertEqual(lines[2],'|-ability|p1a: Deoxys|Pressure|[silent]')
  _,packets=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog))
  return np.array([e.words() for p in packets for e in p.events],np.int32)
 def check_words(self,trace):
  s,t,before,side=self.packet(trace=trace)
  for owner in (0,1):
   stream=full.register(t.stream_id,j.full(8,owner,j.int32),compiled=self.adapter)
   result=self.fast(stream,t,before,life.snapshot(t.cursor));self.assertEqual(result.error.tolist(),[0]*8)
   for row in range(8):np.testing.assert_array_equal(self.f.f.words(result,row),self.expected(s,row,owner,int(side[row]),trace=trace))
   self.assertFalse(bool(result.B4_qualified.any()))
 def test_saved_deoxys_speed_pressure_and_mixed_names_both_owners(self):self.check_words(False)
 def test_trace_retains_real_donor_side_encounter_and_mixed_ident_names_both_owners(self):self.check_words(True)
 def test_pinned_default_naming_only_maps_confirmed_deoxys_formes(self):
  source=(self.root/'sim/pokemon.ts').read_text();self.assertIn('if (set.name === set.species || !set.name)',source);self.assertIn('set.name = this.baseSpecies.baseSpecies;',source)
  for row in self.rows:
   public=self.catalog.species.species_id(row['name']);mapped=int(self.adapter.device.ident_species[public]);base_id=self.catalog.species.species_id(row['baseSpecies'])
   self.assertEqual(mapped,base_id)
   if row['id'] in ('hooh','nidoranf','nidoranm'):self.assertEqual(public,mapped)
  from gpu_engine_default_observer_ast import dump_default_program
  from test_gpu_engine_v5_hypothetical_profile import normalized_hypothetical_default
  before=subprocess.check_output(['git','show','ad445ae1:gpu_engine/transition.py'],text=True)
  self.assertEqual(dump_default_program(ast.parse(before)),dump_default_program(normalized_hypothetical_default(Path('gpu_engine/transition.py').read_text(),as_tree=True)))
  for name in ('entry.py','mechanics.py','engine_pressure_callbacks.py','engine_trace_callbacks.py'):
   self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show','ad445ae1:gpu_engine/'+name]))
