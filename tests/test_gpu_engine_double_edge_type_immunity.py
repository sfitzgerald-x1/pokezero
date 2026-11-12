"""Prospective literal Normal/Ghost zero-recoil receipts, never an engine turn."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_no_damage_callbacks as nd,engine_recoil_callbacks as recoil
from gpu_engine import engine_direct_hit_controls as direct,engine_execution_completeness as ex
from gpu_engine import engine_kernel_frame_risk as risk,engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import engine_secondary_status_callbacks as secondary,engine_source_frame_witness as witness
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,state
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_hidden_power_substitute_input as fixture
BASE='0d51ac06e34af37ad868850456c5209b2b045e29'

class DoubleEdgeTypeImmunityTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.HiddenPowerSubstituteInputTests.setUpClass();f=fixture.HiddenPowerSubstituteInputTests;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);species=dict(f.dex.species);v=cls.metadata['vocabulary']
  cls.names=('doubleedge','surf','rest','wish')
  for name in cls.names:
   info=dex.move_info(name);moves[name]=info
   if name not in v['moves']:v['moves'].append(name);cls.metadata['moves'].append(None)
   flags=json.loads(subprocess.check_output(['node','-e',"console.log(JSON.stringify(require(process.argv[1]).Dex.mod('gen3').moves.get(process.argv[2]).flags))",str(showdown_root()/'dist/sim/dex.js'),name],text=True))
   row=dict(id=name,base_power=info.base_power,type=v['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=100 if name=='doubleedge'else True,priority=info.priority,flags=flags)
   if name=='doubleedge':row['recoil']=[1,3]
   cls.metadata['moves'][v['moves'].index(name)]=row
  for name in ('sableye','gengar'):
   if name not in v['species']:v['species'].append(name)
   species[name]=dex.species_info(name)
  for name in ('keeneye','levitate','rockhead','wonderguard'):
   if name not in v['abilities']:v['abilities'].append(name)
  cls.dex=ShowdownDex(moves,species,dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant(name+str(i),'authored',name,'authored',80,cls.names,abi,'Leftovers')for i,abi in enumerate(('Keen Eye','Levitate'))))for name in v['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-double-edge-immunity'),variants)
  labels=set(f.catalog.vocabulary.tokens)
  for name in cls.names:labels|={'move:'+name,'belief:possible_move:'+name}
  for name in ('sableye','gengar'):labels.add('species:'+name)
  for name in ('keeneye','levitate','rockhead','wonderguard'):labels|={'ability:'+name,'belief:possible_ability:'+name}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules);cls.recoil=recoil.compile_recoil(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',recoil_compiled=cls.recoil,recoil_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))
  cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))

 def component(self,consume=False):
  b=8;rows=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);actor=rows%2;active=j.stack((z+3,z+4),axis=1);phys=active[rows,actor];target=active[rows,1-actor];v=self.metadata['vocabulary']
  species=j.full((b,2,6),v['species'].index('pidgeot'),j.int32);species=species.at[rows,1-actor,target].set(j.where(rows<4,v['species'].index('sableye'),v['species'].index('gengar')))
  type_table=j.array([[0,0]]+[[v['types'].index(x.lower())for x in self.dex.species[n].types]+[0]*(2-len(self.dex.species[n].types))for n in v['species'][1:]],j.int32);types=type_table[species]
  ability=j.full((b,2,6),v['abilities'].index('keeneye'),j.int32).at[rows,1-actor,target].set(j.where(rows<4,v['abilities'].index('keeneye'),v['abilities'].index('levitate')))
  s=state.empty_state(b)._replace(done=~yes,active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=species,types=types,base_types=types,ability=ability,base_ability=ability,level=j.full((b,2,6),80,j.int32))
  c=life.register(rows+310001,**self.kw);t=life._tape(c,self.producer,64)
  for side in(0,1):t=life._reveal(t,s,z+side,z-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  t=t._replace(cursor=t.cursor._replace(initialized=yes));t=life._turn(t,yes);initial=life.LifecycleResult(s,t.cursor,t,life.snapshot(c),life.snapshot(t.cursor),~yes,z);consumers=[]
  for owner in((0,1)if consume else()):
   stream=full.register(t.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=t.stream_id,owner_sides=z+owner);got=self.fold(stream,initial,pub);self.assertEqual(got.projection.error.tolist(),[0]*b);consumers.append((got.projection.state,got.public))
  t=life._tape(t.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;t=ex.begin_on_tape(t,ex.initialize(t.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,rt),t,rt,active=yes);t=t._replace(execution=t.execution._replace(risks=risk.Context(carry,rt)))
  move=z+v['moves'].index('doubleedge');t=public.move(t,actor,phys,target,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.DIRECT_HIT,1-actor,active=yes)
  before=nd.Before(yes,z+v['types'].index('normal'),ability[rows,1-actor,target],z,z+100,z+100,j.zeros(b,j.uint32),z,z+100,~yes,z,z+70,types[rows,1-actor,target],z)
  p=nd.prepare(t,ticket,actor,phys,target,move,before,self.runtime['direct_hit_context'],requested=yes,caller_move=move)
  w=direct.HitWrite(z+120,~yes,yes,yes,z,z+1,~yes,~yes,~yes,z+100,z+100,z+100);o=risk.Outcome(w.power,w.critical,w.hit,w.immune,w.numerator,w.denominator,~yes,~yes,~yes,~yes,z,~yes)
  return s,t,consumers,p,w,o
 def captured(self,p,w):
  a=p.before;return nd.Captured(p,w,a.actor_hp,a.volatile,a.sub_hp,a.actor_hp_type,a.actor_hp_power,a.target_types,a.item,a.ability)
 def finish_direct(self,t,p,w,o):
  a=p.before;return nd.finish(t,p,w,o,a.actor_hp,a.volatile,a.sub_hp,a.actor_hp_type,a.actor_hp_power,target_types=a.target_types,item=a.item,ability=a.ability)
 def prepare_recoil(self,t,p,w,o):
  b=len(p.actor);z=j.zeros(b,j.int32);cap=risk.CapturedDirectHit(p.context,p.ticket,p.actor,p.physical,p.target,p.move,w)
  return recoil.prepare_device(t,cap,z,z+self.metadata['vocabulary']['abilities'].index('keeneye'),z+100,z+100,z,self.runtime['recoil_context'],eligible=j.ones(b,bool),caller_move=p.move,no_damage=self.captured(p,w),no_damage_outcome=o)
 def record(self,t,p):
  # Original secondary.apply calls begin even when changed=False. Retain that
  # genuine SAME Direct context at the same later cutpoint, without a new field.
  b=len(p.requested);z=j.zeros(b,j.int32);no=j.zeros(b,bool);w=secondary.Write(no,no,no,no,~no,no,no,z,z,z,z,z+p.ability,p.no_damage.pending.before.ability,z)
  t,_=secondary.begin(t,p.context.direct,self.table.secondary,p.captured.actor,p.captured.actor_physical,p.captured.target_physical,p.captured.move,w)
  before=recoil.Before(p.captured.actor_physical,p.captured.target_physical,p.ability,p.old_hp,p.maximum,p.status);t=risk.recoil_on_tape(t,p.recoil,active=~no,pending=p,before=before)
  return recoil.record_device(t,p,recoil.Write(*before[:4],p.old_hp,p.maximum,p.status))

 def test_original_order_source_default_and_capability(self):
  script=r"""const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),lines=[];const actor={hp:100,boosts:{accuracy:0,evasion:0},volatiles:{}},target={hp:100,boosts:{accuracy:0,evasion:0},volatiles:{},runImmunity:()=>false,toString:()=> 'p2a: Gengar'};const battle={gen:3,activeMove:null,add:(...x)=>lines.push(x.map(String).join('|')),setActiveMove(){},singleEvent:()=>true,runEvent:(id,t,s,m,v)=>['ModifyBoost','ModifyAccuracy','Accuracy'].includes(id)?v:true,clampIntRange:(x,a,b)=>Math.min(b===undefined?Infinity:b,Math.max(a,x)),randomChance:()=>true,debug(){},damage(){throw Error('recoil reached')}};let move=d.moves.get('doubleedge');move={...move};const result=d.data.Scripts.actions.tryMoveHit.call({battle,dex:d,moveHit(){throw Error('damage reached')}},target,actor,move);console.log(JSON.stringify({result,lines}));"""
  self.assertEqual(json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True)),dict(result=False,lines=['-immune|p2a: Gengar']))
  # Run the original immunity selector too: Normal/Ghost bypasses Ground's
  # ability-dependent isGrounded branch, even when that member has Levitate.
  script2=r"""const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),P=require(root+'/dist/sim/pokemon.js').Pokemon,out=[];for(const ability of ['keeneye','levitate']){const events=[],mon={battle:{dex:d,runEvent:id=>{events.push(id);return true}},getTypes:()=>['Ghost','Poison'],isGrounded(){throw Error('Ground-only branch reached')}};out.push({ability,result:P.prototype.runImmunity.call(mon,d.moves.get('doubleedge')),events})}console.log(JSON.stringify(out));"""
  self.assertEqual(json.loads(subprocess.check_output(['node','-e',script2,str(showdown_root())],text=True)),[dict(ability=n,result=False,events=['NegateImmunity'])for n in ('keeneye','levitate')])
  original=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True));current=ast.parse(Path('gpu_engine/transition.py').read_text());expected=dump_default_program(original);self.assertEqual(expected,dump_default_program(current));assert_hp_rng_mutations_detected(self,current,expected)
  legacy=recoil.compile_recoil(engine_catalog=self.producer,metadata=self.metadata);self.assertFalse(bool(legacy.device.immune_source));self.assertTrue(bool(self.runtime['kernel_risk_compiled'].device.recoil_immune_source))
  legacy_direct=direct.compile_direct_hit_controls(engine_catalog=self.producer);r=risk.compile_risks(self.producer,self.runtime['execution_compiled'],direct_hit_compiled=legacy_direct,recoil_compiled=self.recoil);self.assertFalse(bool(r.device.recoil_immune_source))
  for modules in(('benchmark_retained_games','engine_lifecycle_event_device','engine_full_v4_event_device'),('engine_full_v4_event_device','engine_lifecycle_event_device','benchmark_retained_games')):subprocess.run([__import__('sys').executable,'-c',';'.join('import gpu_engine.'+x for x in modules)],check=True,capture_output=True)

 def test_same_call_two_claims_and_refusal_controls(self):
  s,t,_,p,w,o=self.component();out=self.finish_direct(t,p,w,o);self.assertEqual(out.count.tolist(),[3]*8);self.assertTrue(bool(nd.captured_matches(out,self.captured(p,w),o).all()))
  out,r=self.prepare_recoil(out,p,w,o);self.assertTrue(bool(recoil.matches_pending(out,r).all()));done=self.record(out,r);self.assertEqual(done.count.tolist(),[4]*8);self.assertEqual(done.words[0,:4,0].tolist(),[1,33,34,28]);self.assertEqual(done.execution.ledger.claimed[0,:4].tolist(),[0,1,1,2]);self.assertEqual(done.execution.ledger.stage[:,:2].tolist(),[[2,2]]*8)
  self.assertTrue(bool(recoil.owned_silent(done,3).all()));self.assertTrue(bool(nd.owned_pair(done,j.full(8,1,j.int32),self.helper.device.no_damage)[0].all()));self.assertEqual(done.execution.risks.carry.risk.tolist(),[0]*8)
  wt=self.runtime['frame_witness_compiled'].device;self.assertTrue(bool(witness._raw_complete(done,wt).all()))
  self.assertFalse(bool(recoil.owned_silent(done._replace(execution=done.execution._replace(secondary_status=None)),3).any()))
  for q,ww,oo in((p._replace(caller_move=p.move+1),w,o),(p._replace(before=p.before._replace(target_types=None)),w,o),(p._replace(before=p.before._replace(target_types=j.ones((8,2),j.int32))),w,o),(p._replace(before=p.before._replace(ability=j.full(8,self.metadata['vocabulary']['abilities'].index('wonderguard'),j.int32))),w,o),(p,w._replace(hit=j.zeros(8,bool)),o),(p,w._replace(substitute=j.ones(8,bool)),o),(p,w._replace(new_hp=j.zeros(8,j.int32)),o),(p,w,o._replace(recoil=j.ones(8,j.int32))),(p,w,o._replace(absorbed=j.ones(8,bool)))):
   bad=self.finish_direct(t,q,ww,oo);self.assertEqual(bad.count.tolist(),[1]*8)
  for key in('target_types','item','ability'):
   a=p.before;kw=dict(target_types=a.target_types,item=a.item,ability=a.ability);kw[key]=kw[key]+1
   bad=nd.finish(t,p,w,o,a.actor_hp,a.volatile,a.sub_hp,a.actor_hp_type,a.actor_hp_power,**kw);self.assertEqual(bad.count.tolist(),[1]*8)
  # Positive recoil: moving prepare past an inapplicable no-damage finish
  # preserves the complete old tape/ledger and the original positive RAW28.
  normal=j.array([self.metadata['vocabulary']['types'].index('normal'),0],j.int32)
  pp=p._replace(before=p.before._replace(target_types=j.broadcast_to(normal,(8,2))))
  positive=w._replace(immune=j.zeros(8,bool),numerator=j.ones(8,j.int32),new_hp=j.full(8,80,j.int32));po=o._replace(immune=positive.immune,numerator=positive.numerator,recoil=j.full(8,6,j.int32))
  pos=direct.record_registered_direct_hit_controls_device(t,p.actor,p.physical,p.target,p.move,positive,p.context,eligible=j.ones(8,bool));pos=public.condition(pos,1-p.actor,p.target,positive.new_hp,positive.maximum,p.before.status,eligible=j.ones(8,bool));pos=ex.capture_direct_hit_write(pos,p.ticket,p.actor,p.physical,p.target,p.move,positive,requested=j.ones(8,bool),data=p.context.data)
  cap=risk.CapturedDirectHit(p.context,p.ticket,p.actor,p.physical,p.target,p.move,positive);ability=j.full(8,self.metadata['vocabulary']['abilities'].index('keeneye'),j.int32)
  old,r_old=recoil.prepare_device(pos,cap,po.recoil,ability,p.before.actor_hp,p.before.maximum,p.before.status,self.runtime['recoil_context'],eligible=j.ones(8,bool),caller_move=p.move)
  moved=self.finish_direct(pos,pp,positive,po);new,r_new=recoil.prepare_device(moved,cap,po.recoil,ability,p.before.actor_hp,p.before.maximum,p.before.status,self.runtime['recoil_context'],eligible=j.ones(8,bool),caller_move=p.move,no_damage=self.captured(pp,positive),no_damage_outcome=po)
  for a,b in zip(jax.tree.leaves(old),jax.tree.leaves(new),strict=True):np.testing.assert_array_equal(a,b)
  wr=recoil.Write(p.physical,p.target,ability,p.before.actor_hp,p.before.actor_hp-po.recoil,p.before.maximum,p.before.status)
  for a,b in zip(jax.tree.leaves(recoil.record_device(old,r_old,wr)),jax.tree.leaves(recoil.record_device(new,r_new,wr)),strict=True):np.testing.assert_array_equal(a,b)
  led=done.execution.ledger
  for field,idx in(('no_output_facts',(0,0,0)),('no_output_facts',(0,1,0)),('no_output_binding',(0,1,0)),('claimed',(0,3)),('event_turn',(0,3)),('side',(0,0))):
   val=getattr(led,field);bad=done._replace(execution=done.execution._replace(ledger=led._replace(**{field:val.at[idx].add(1)})));self.assertFalse(bool(recoil.owned_silent(bad,3)[0]),(field,idx))
  for key in('engine_binding','source_binding','move_binding','producer_binding'):
   with self.assertRaises(ValueError):recoil.owned_silent(done._replace(**{key:j.tile(getattr(done,key),(8,1))}),3)
  bad=done._replace(cursor=done.cursor._replace(generation=done.cursor.generation.at[0,0].add(1)));self.assertFalse(bool(recoil.owned_silent(bad,3,bad.cursor)[0]))
  foreign=public.move(out,p.actor,p.physical,p.target,p.move,eligible=j.ones(8,bool));self.assertFalse(bool(recoil.matches_pending(foreign,r).any()))
  sticky=self.record(out._replace(refusal=out.refusal|64),r);self.assertTrue(bool((sticky.refusal&64).all()));self.assertTrue(bool((sticky.execution.ledger.refusal!=0).all()))

 def test_both_original_full_owners_exact_bare_immune_and_private_recoil(self):
  s,t,consumers,p,w,o=self.component(True);out=self.finish_direct(t,p,w,o);out,r=self.prepare_recoil(out,p,w,o);out=self.record(out,r);b=8;yes=j.ones(b,bool);z=j.zeros(b,j.int32);cursor=t.cursor
  out=life.action_end(out,eligible=yes);out=life.finish(out,s,live=yes,replacement=~yes,pending_before=~yes,residual_ready=yes,defer_residual=~yes,turn_context=yes);frame=life.LifecycleResult(s,out.cursor,out,life.snapshot(cursor),life.snapshot(out.cursor),~yes,z);closed=[];saved=[]
  self.assertEqual(full.Consumed._fields,('projection','public'));self.assertIn('component_admitted',full.Projection._fields);self.assertIn('callbacks',life.LifecycleResult._fields);self.assertIn('needs_more',type(consumers[0][1].cursor)._fields)
  consume=jax.jit(lambda stream,pub:ex.consume_frame_device(frame.callbacks.execution.ledger,frame,stream,pub,active=yes,tables=self.runtime['execution_compiled'].device,adapter_tables=self.runtime['public'].device,catalog=self.catalog))
  for owner,(stream,pub)in enumerate(consumers):
   end,got=consume(stream,pub);actual=[np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])].tolist()for row in range(b)];saved.append(dict(owner=owner,error=got.projection.error.tolist(),admitted=got.projection.component_admitted.tolist(),words=actual));Path('/tmp/pokezero-double-edge-immunity-full-current.json').write_text(json.dumps(saved)+'\n')
   self.assertEqual(got.projection.error.tolist(),[0]*b);self.assertTrue(bool(got.projection.component_admitted.all()));self.assertFalse(bool(got.public.cursor.needs_more.any()));self.assertTrue(bool(ex.report(end,tables=self.runtime['execution_compiled'].device).scoped_drained.all()));closed.append(end)
   for row in range(b):
    ident={};lines=[]
    for side in(0,1):
     phys=int(s.active[row,side]);name=self.catalog.species.names[int(self.runtime['public'].device.causal.species[int(s.species[row,side,phys])])];ident[side]=f'p{side+1}a: '+name;lines.append(f'|switch|{ident[side]}|{name}, L80, M|100/100')
    wire,_=public_wire.project_lines(public_wire.WireState(),lines+['|turn|1'],**v4.wire_kwargs(self.catalog));who=int(p.actor[row]);foe=1-who;_,packets=public_wire.project_lines(wire,[f'|move|{ident[who]}|Double-Edge|{ident[foe]}',f'|-immune|{ident[foe]}','|','|upkeep','|turn|2'],**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for packet in packets for e in packet.events],np.int32);np.testing.assert_array_equal(actual[row],want)
  _,good=ex.join_consumed_owners(*closed,tables=self.runtime['execution_compiled'].device);self.assertTrue(bool(good.all()))

if __name__=='__main__':unittest.main()
