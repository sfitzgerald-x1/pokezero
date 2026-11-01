"""Source-bound Earthquake/Levitate literal transactions; no engine turns."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached,normalize_id
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_no_damage_callbacks as nd,engine_direct_hit_controls as hit,engine_kernel_frame_risk as risk
from gpu_engine import engine_execution_completeness as ex,engine_lifecycle_callbacks as life,engine_public_events as events
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_leftovers_callbacks as leftovers,engine_full_v4_event_device as full,engine_source_frame_witness as witness
from gpu_engine import public_belief,v4_encoder as v4,public_wire,public_inputs as inputs,transition,absorption
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_hidden_power_substitute_input as fixture
BASE='5e038e81a779ada0d442b4f723e40c2afbeeb8a5'

class EarthquakeLevitateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.HiddenPowerSubstituteInputTests.setUpClass();f=fixture.HiddenPowerSubstituteInputTests;cls.f=f
  cls.metadata=deepcopy(f.metadata);dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);cls.names=('earthquake','substitute','surf','sleeptalk')
  for name in cls.names:
   info=dex.move_info(name);moves[name]=info
   flags=json.loads(subprocess.check_output(['node','-e',"const m=require(process.argv[1]).Dex.mod('gen3').moves.get(process.argv[2]);console.log(JSON.stringify({flags:m.flags,accuracy:m.accuracy}))",str(showdown_root()/'dist/sim/dex.js'),name],text=True))
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   idx=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][idx]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=flags['accuracy'],priority=info.priority,flags=flags['flags'])
  if 'levitate'not in cls.metadata['vocabulary']['abilities']:cls.metadata['vocabulary']['abilities'].append('levitate')
  cls.levitate=cls.metadata['vocabulary']['abilities'].index('levitate');cls.move=cls.metadata['vocabulary']['moves'].index('earthquake');cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('levitate-'+name,'unit',name,'unit',80,cls.names,'Levitate','Leftovers'),))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-earthquake-levitate'),variants);labels=set(f.catalog.vocabulary.tokens)|{'ability:levitate','belief:possible_ability:levitate'}
  for name in cls.names:labels|={'move:'+name,'belief:possible_move:'+name}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  cls.table=transition.tables_from_metadata(cls.metadata);cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=events.HealthExposure(events.SHOWDOWN,'gen3randombattle',True,False));cls.rules=request.load_source_rules(showdown_root(),cls.metadata)
  req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=hit.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules);cls.legacy=nd.compile_data(cls.producer)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared)

 def component(self,b=8,mask=None):
  rows=j.arange(b);side=rows%2;z=j.zeros(b,j.int32);yes=j.ones(b,bool);no=~yes;active=yes if mask is None else mask;physical=z+3;target=z+4;move=z+self.move
  cursor=life.register(rows+211201,catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));phys=j.zeros((b,2),j.int32).at[rows,side].set(physical).at[rows,1-side].set(target)
  mapping=j.full((b,2,6),-1,j.int32).at[:,:,0].set(0).at[:,:,1].set(1).at[rows,side,physical].set(2).at[rows,1-side,target].set(2)
  cursor=cursor._replace(generation=j.full((b,2),3,j.int32),turn=z+1,initialized=yes,physical_active=phys,public_slot=j.full((b,2),2,j.int32),encounter=j.full((b,2),3,j.int32),established=j.ones((b,2),bool),physical_to_public=mapping,next_slot=j.full((b,2),3,j.int32))
  tape=life._tape(cursor,self.producer,64);acc=self.runtime['execution_compiled'];rc=self.runtime['kernel_risk_compiled'];tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=acc),acc.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rc.device),tape,rc.device,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rc.device)))
  tape=events.move(tape,side,physical,target,move,eligible=active);tape,ticket=ex.open_attempt(tape,ex.Site.DIRECT_HIT,1-side,active=active)
  types=j.stack((z+self.metadata['vocabulary']['types'].index('normal'),z),axis=1);item=z+self.metadata['vocabulary']['items'].index('leftovers')
  before=nd.Before(yes,z+self.metadata['vocabulary']['types'].index('ground'),z+self.levitate,z,z+100,z+100,j.zeros(b,j.uint32),z,z+100,no,z,z+70,types,item)
  pending=nd.prepare(tape,ticket,side,physical,target,move,before,self.runtime['direct_hit_context'],requested=active,caller_move=move)
  pending=nd.observed_absorption(pending,absorption.try_hit(self.table.absorption,before.ability,move,before.move_type,before.status,types,before.hp,before.maximum,before.sub_hp,before.volatile,eligible=active))
  write=hit.HitWrite(z+100,no,yes,yes,z+1,z+1,no,no,no,before.hp,before.hp,before.maximum);outcome=risk.Outcome(write.power,write.critical,write.hit,write.immune,write.numerator,write.denominator,pending.absorbed.blocked,no,no,no,z,no);tail=(z+100,before.volatile,z,before.actor_hp_type,before.actor_hp_power)
  return tape,pending,write,outcome,tail

 def test_source_grammar_default_ast_and_actual_observer_operands(self):
  script=r'''const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),P=require(root+'/dist/sim/pokemon.js').Pokemon,out=[];for(const [types,item,vol]of[[['Normal'],'leftovers',{}],[['Flying'],'leftovers',{}],[['Normal'],'ironball',{}],[['Normal'],'leftovers',{ingrain:{}}]]){const logs=[],battle={gen:3,field:{pseudoWeather:{}},runEvent:()=>true,suppressingAbility:()=>false,dex:d,add:(...a)=>logs.push(a.map(x=>typeof x==='object'?'p2a: Target':x).join('|'))},target={battle,volatiles:vol,item,ignoringItem:()=>false,hasType:t=>types.includes(t),hasAbility:a=>a==='levitate',isGrounded:P.prototype.isGrounded};out.push({result:P.prototype.runImmunity.call(target,d.moves.get('earthquake'),true),logs});}console.log(JSON.stringify(out));'''
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(got,[dict(result=False,logs=['-immune|p2a: Target|[from] ability: Levitate']),dict(result=False,logs=['-immune|p2a: Target']),dict(result=True,logs=[]),dict(result=False,logs=['-immune|p2a: Target|[from] ability: Levitate'])])
  self.assertTrue(bool(self.helper.device.no_damage.levitate_source));self.assertFalse(bool(self.legacy.levitate_source))
  new=ast.parse(Path('gpu_engine/transition.py').read_text());old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));baseline=dump_default_program(old);self.assertEqual(dump_default_program(new),baseline);assert_hp_rng_mutations_detected(self,new,baseline)
  call=next(n for n in ast.walk(new)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='no_damage_callbacks'and n.func.attr=='Before')
  self.assertEqual([ast.unparse(x)for x in call.args[-2:]],['_active(work.types, work, foe)','_active(work.item, work, foe)'])
  finish=next(n for n in ast.walk(new)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='no_damage_callbacks'and n.func.attr=='finish')
  self.assertEqual({k.arg:ast.unparse(k.value)for k in finish.keywords},dict(target_types='_active(work.types, work, foe)',item='_active(work.item, work, foe)',ability='_active(work.ability, work, foe)'))
  for name in ('mechanics.py','state.py','entry.py','secondary.py','engine_retained_encoder_driver.py'):self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+name]))
  witness._source()

 def test_mixed_source_capture_grounding_namespaces_and_outcomes(self):
  mask=j.array([True]*6+[False]*2);t,p,w,o,tail=self.component(mask=mask);self.assertEqual(nd.scheduled(p,w).tolist(),mask.tolist());out=nd.finish(t,p,w,o,*tail,target_types=p.before.target_types,item=p.before.item,ability=p.before.ability);cap=nd.Captured(p,w,*tail,p.before.target_types,p.before.item,p.before.ability)
  self.assertEqual(nd.captured_matches(out,cap,o).tolist(),mask.tolist());self.assertEqual(nd.owned_pair(out,1,self.runtime['public'].device.causal.no_damage)[0].tolist(),mask.tolist())
  self.assertTrue(bool(witness._raw_complete(out,self.runtime['frame_witness_compiled'].device).all()));self.assertFalse(bool(nd.scheduled(p,w._replace(numerator=j.zeros(8,j.int32))).any()))
  for field,value in [('volatile',j.ones(8,j.uint32)),('target_types',None),('item',None),('target_types',p.before.target_types.at[:,0].set(self.metadata['vocabulary']['types'].index('flying'))),('item',j.full(8,len(p.context.data.no_damage.levitate_items),j.int32))]:self.assertFalse(bool(nd.scheduled(p._replace(before=p.before._replace(**{field:value})),w).any()),field)
  self.assertFalse(bool(nd.scheduled(p,w._replace(power=w.power+1)).any()));self.assertFalse(bool(nd.scheduled(p._replace(caller_move=p.move+1),w).any()));self.assertFalse(bool(nd.scheduled(p._replace(caller_move=None),w).any()))
  for field,value in [('power',w.power+1),('critical',~w.critical),('hit',~w.hit),('immune',~w.immune),('numerator',w.numerator+1),('denominator',w.denominator+1),('absorbed',~o.absorbed),('substitute',~w.substitute),('protected',~w.protected),('enduring',~w.enduring),('recoil',o.recoil+1),('explodes',~o.explodes)]:self.assertFalse(bool(nd.captured_matches(out,cap,o._replace(**{field:value})).any()),field)
  for field,value in [('target_types',cap.target_types.at[0,0].set(self.metadata['vocabulary']['types'].index('water'))),('item',cap.item.at[0].set(0)),('ability',cap.ability.at[0].set(0))]:self.assertEqual(nd.captured_matches(out,cap._replace(**{field:value}),o).tolist(),[False]+mask.tolist()[1:])
  bad=out._replace(event_turn=out.event_turn.at[0,1].add(1));self.assertEqual(nd.owned_pair(bad,1,self.runtime['public'].device.causal.no_damage)[0].tolist(),[False]+mask.tolist()[1:])
  self.assertFalse(bool(nd.valid_raw_pair(*nd.words(p,w),self.legacy._replace(levitate_move=j.array(0,j.int32))).any()))
  current=risk.outcome_on_tape(out,o,eligible=mask,active=mask,captured_no_damage=cap);self.assertEqual(current.execution.risks.carry.risk.tolist(),[0]*8)
  prior=out._replace(refusal=out.refusal|256);current=risk.outcome_on_tape(prior,o,eligible=mask,active=mask,captured_no_damage=cap);self.assertTrue(bool(j.all(current.execution.risks.carry.risk[:6]&risk.UPSTREAM!=0)))

 def test_both_full_original_tagged_words_and_no_private_item_type_revelation(self):
  t,p,w,o,tail=self.component();t=nd.finish(t,p,w,o,*tail,target_types=p.before.target_types,item=p.before.item,ability=p.before.ability);kw=v4.wire_kwargs(self.catalog);ns={k:np.asarray(getattr(self.runtime['public'].device.namespaces,k))for k in full._NAMES};fold=jax.jit(lambda ledger,stream,frame,pub:ex.consume_frame_device(ledger,frame,stream,pub,active=j.ones(8,bool),tables=self.runtime['execution_compiled'].device,adapter_tables=self.runtime['public'].device,catalog=self.catalog));engine_ids=self.metadata['vocabulary']['species'];species_names=list(self.catalog.species.names[1:4]);initial=[];expected=[]
  for row in range(8):
   side=row%2;names=[species_names[2],species_names[2]];ident=[f'p{i+1}a: '+names[i]for i in (0,1)];lines=[]
   for earlier in species_names[:2]:
    for owner in (0,1):lines.append(f'|switch|p{owner+1}a: {earlier}|{earlier}, L80, M|100/100')
   for owner in (0,1):lines.append(f'|switch|{ident[owner]}|{names[owner]}, L80, M|100/100')
   lines.append('|turn|1');wire,packets=public_wire.project_lines(public_wire.WireState(),lines,**kw);initial.append([x for q in packets for x in q.events]);_,packets=public_wire.project_lines(wire,[f'|move|{ident[side]}|Earthquake|{ident[1-side]}',f'|-immune|{ident[1-side]}|[from] ability: Levitate'],**kw);expected.append(np.asarray([x.words()for q in packets for x in q.events],np.int32))
  compact=[]
  for owner in (0,1):
   pub=v4.initialize(self.catalog,stream_ids=t.stream_id,owner_sides=j.full(8,owner,j.int32))
   for chunk in inputs.pack_chunks(np.asarray(t.stream_id),initial,**ns):pub,error=v4.fold_chunk(pub,chunk,self.catalog);self.assertFalse(bool(error.any()))
   species=j.array([[engine_ids.index(normalize_id(self.catalog.species.names[int(pub.mons.species[r,s,2])]))for s in (0,1)]for r in range(8)],j.int32);cursor=t.cursor._replace(revealed_species_id=species);t=t._replace(cursor=cursor)
   stream=full.register(t.stream_id,j.full(8,owner,j.int32),compiled=self.runtime['public']);stream=stream._replace(causal=stream.causal._replace(cursor=cursor,sequence=j.full(8,7,j.int32)),sequence=pub.cursor.sequence,live=j.ones((8,2),bool),species=pub.mons.species,status=pub.mons.status,hp=pub.mons.hp_numerator,maximum=pub.mons.hp_denominator)
   frame=life.LifecycleResult(None,cursor,t,life.snapshot(cursor),life.snapshot(cursor),j.zeros(8,bool),j.zeros(8,j.int32));closed,got=fold(t.execution.ledger,stream,frame,pub);self.assertEqual(closed.stage[:,0].tolist(),[3]*8);self.assertEqual(closed.refusal.tolist(),[0]*8);self.assertEqual(got.projection.error.tolist(),[0]*8);self.assertEqual(got.projection.component_admitted.tolist(),[True]*8)
   for row in range(8):
    actual=np.asarray(got.projection.frames.words).transpose(1,0,2,3)[row][np.asarray(got.projection.frames.valid).transpose(1,0,2)[row]];np.testing.assert_array_equal(actual,expected[row]);self.assertEqual(int(got.public.context.abilities.original[row,1-row%2,2]),self.catalog.abilities.ids['levitate']);compact.append(dict(owner=owner,row=row,actual=actual.tolist(),expected=expected[row].tolist()))
   for x,y in zip(jax.tree.leaves(got.public.context.types),jax.tree.leaves(pub.context.types),strict=True):np.testing.assert_array_equal(x,y)
   for x,y in zip(jax.tree.leaves(got.public.context.items),jax.tree.leaves(pub.context.items),strict=True):np.testing.assert_array_equal(x,y)
  Path('/tmp/pokezero-earthquake-levitate-original-both-full-compact.json').write_text(json.dumps(compact)+'\n')
if __name__=='__main__':unittest.main()
