"""Source-only original self heal callback and BOTH FULL components."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_self_heal_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,state
from gpu_engine import engine_source_frame_witness as witness
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixtures
BASE='8cdd966d843e2d9603f4372f593065a7bae8d99d'

class RecoveryFamilyTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.NoDamageOwningTests.setUpClass();f=fixtures.NoDamageOwningTests;cls.f=f;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);cls.names=('recover','synthesis','softboiled','surf','moonlight','morningsun','milkdrink','slackoff','rest','sleeptalk','wish')
  for name in cls.names:
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=True if name!='surf'else 100,priority=info.priority,flags={}if name!='surf'else{'protect':1})
  # This shared catalog also contains Encore; keep that unused fixture row
  # equal to the original installed move rather than the generic always-hit row.
  if 'encore' in cls.metadata['vocabulary']['moves']:
   encore=dex.move_info('encore');moves['encore']=encore
   cls.metadata['moves'][cls.metadata['vocabulary']['moves'].index('encore')]=dict(id='encore',base_power=encore.base_power,type=cls.metadata['vocabulary']['types'].index(encore.type.lower()),source_type=encore.type,category=encore.category,pp=encore.pp,accuracy=100,priority=encore.priority,flags={'protect':1,'mirror':1,'bypasssub':1,'metronome':1,'failencore':1})
  cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant('heal-'+name+str(i),'authored',name,'authored',80,cls.names[i:i+4],'Pressure','Leftovers')for i in(0,4,8)))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-original-self-heal'),variants)
  labels=set(f.catalog.vocabulary.tokens)
  for n in cls.names:labels|={'move:'+n,'belief:possible_move:'+n}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  # Keep the authored installed flags equal to the pinned original source.
  original_flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]).Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>!!d.moves.get(n).flags.failencore)))",str(showdown_root()/'dist/sim/dex.js'),json.dumps(cls.metadata['vocabulary']['moves'])],text=True))
  for row,flag in zip(cls.metadata['moves'],original_flags):
   if row is not None:row.setdefault('flags',{})['failencore']=flag
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  rules=request.load_source_rules(showdown_root(),cls.metadata);req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=rules)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))

 def component(self,consume=False):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);actor=rows%2;active=j.stack((z+3,z+4),axis=1);phys=active[rows,actor]
  hp=j.full((b,2,6),40,j.int32).at[rows[4:],actor[4:],phys[4:]].set(100)
  s=state.empty_state(b)._replace(active=active,gender=j.ones((b,2,6),j.int32),hp=hp,max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32))
  kw=dict(catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));cursor=life.register(rows+56001,**kw);initial=life._tape(cursor,self.producer,64)
  for side in(0,1):initial=life._reveal(initial,s,z+side,z-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in((0,1)if consume else()):
   stream=full.register(cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=z+owner);r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);consumers.append((r.projection.state,r.public))
  t=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;t=ex.begin_on_tape(t,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,rt),t,rt,active=yes);t=t._replace(execution=t.execution._replace(risks=risk.Context(carry,rt)))
  names=['recover','morningsun','moonlight','synthesis','recover','morningsun','moonlight','synthesis'];ids=self.metadata['vocabulary']['moves'];move=j.array([ids.index(n)for n in names],j.int32)
  t=public.move(t,actor,phys,s.active[rows,1-actor],move,eligible=yes);t,p=cb.begin(t,s,actor,move,self.runtime['direct_hit_context'],self.table,eligible=yes)
  after=s._replace(hp=s.hp.at[rows,actor,phys].set(j.array([90]*4+[100]*4,j.int32)));t,c=cb.finish(t,p,after,z,yes)
  return s,after,t,c,actor,phys,names,consumers

 def test_generic_source_family_same_write_and_fullhp_negative_guards(self):
  witness._source();ids=self.metadata['vocabulary']['moves'];d=self.helper.device.self_heal
  self.assertEqual({ids[i]for i in np.flatnonzero(d.candidates)},set(self.names)-{'surf','rest','sleeptalk','wish'})
  s,after,t,c,actor,phys,names,_=self.component();self.assertTrue(bool(cb.captured_matches(t,c).all()));self.assertEqual(t.words[:,1,9].tolist(),[1]*4+[3,5,5,5]);self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  for field in('engine_binding','source_binding','move_binding','producer_binding'):
   with self.assertRaises(ValueError):cb.owned(t._replace(**{field:j.broadcast_to(getattr(t,field),(8,8))}),1)
  for bad in(t._replace(words=t.words.at[4,1,9].set(5)),t._replace(words=t.words.at[5,1,9].set(3)),t._replace(event_turn=t.event_turn.at[:,1].add(1)),t._replace(words=t.words.at[:,0,5].set(0))):
   self.assertFalse(bool(cb.owned(bad,1).all()))
  self.assertFalse(bool(cb.captured_matches(t._replace(cursor=t.cursor._replace(turn=t.cursor.turn+1)),c).any()))
  rt=t.execution.risks.tables
  with self.assertRaises(ValueError):cb.owned(t._replace(execution=t.execution._replace(risks=t.execution.risks._replace(tables=rt._replace(self_heal_family=rt.self_heal_family.astype(j.float32))))),1)
  rawbad=t._replace(words=t.words.at[:,1,2].set(5));self.assertFalse(bool(cb.owned(rawbad,1).any()))
  self.assertIsNone(cb.begin(t,s,actor,j.ones(8,j.int32),None,self.table,eligible=j.ones(8,bool))[1])
  with self.assertRaises(ValueError):cb.begin(t,s,actor,j.ones(8,j.int32),self.runtime['direct_hit_context'],self.table._replace(weather=None),eligible=j.ones(8,bool))
  with self.assertRaises(ValueError):cb.finish(t,c.pending,after,None,j.ones(8,bool))
  prior=t._replace(refusal=t.refusal|64);r=risk.self_heal_on_tape(prior,j.ones(8,bool),active=j.ones(8,bool),captured=c);self.assertFalse(bool((r.execution.risks.carry.risk&risk.SELF_EFFECT).any()));self.assertTrue(bool((r.refusal&64).all()))
  # A wrong afterstate cannot resolve a requested fullHP transaction.
  p=c.pending;before=t._replace(count=j.ones(8,j.int32),execution=t.execution._replace(ledger=t.execution.ledger._replace(stage=t.execution.ledger.stage.at[:,0].set(1),end=t.execution.ledger.end.at[:,0].set(1),claimed=j.zeros_like(t.execution.ledger.claimed))))
  for field,value in(('hp',99),('max_hp',101),('status',1)):
   at=(j.arange(8),actor,phys);bad=after._replace(**{field:getattr(after,field).at[at].set(value)});failed,cap=cb.finish(before,p,bad,j.zeros(8,j.int32),j.ones(8,bool));self.assertFalse(bool(cb.captured_matches(failed,cap)[4:].any()),field)

 def test_both_full_original_positive_and_fullhp_half_vs_weather(self):
  s,after,t,c,actor,phys,names,consumers=self.component(True);yes=j.ones(8,bool);z=j.zeros(8,j.int32);frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(t.cursor),life.snapshot(t.cursor),~yes,z);capture=[]
  for owner,(stream,pub)in enumerate(consumers):
   r=self.fold(stream,frame,pub);capture.append(dict(owner=owner,error=r.projection.error.tolist(),words=r.projection.frames.words.tolist(),valid=r.projection.frames.valid.tolist()));Path('/tmp/pokezero-recovery-family-full.json').write_text(json.dumps(capture)+'\n');self.assertEqual(r.projection.error.tolist(),[0]*8)
   for row in range(8):
    ids={side:f'p{side+1}a: '+self.catalog.species.names[side+1]for side in(0,1)};who=int(actor[row]);hp='100/100'if row>=4 else'40/100'
    initial=[f'|switch|{ids[side]}|{self.catalog.species.names[side+1]}, L80, M|'+(hp if side==who else'40/100')for side in(0,1)]
    wire,_=public_wire.project_lines(public_wire.WireState(),initial+['|turn|1'],**v4.wire_kwargs(self.catalog));half_fail=row==4
    lines=[f'|move|{ids[who]}|{self.dex.move_info(names[row]).name}|'+(''if half_fail else ids[who])+('|[still]'if half_fail else'')]
    if row<4:lines.append(f'|-heal|{ids[who]}|90/100')
    elif half_fail:lines.append(f'|-fail|{ids[who]}|heal')
    _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));want=np.asarray([event.words()for packet in packets for event in packet.events],np.int32);actual=np.asarray(r.projection.frames.words[:,row])[np.asarray(r.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,want)
   self.assertFalse(bool(r.public.cursor.needs_more.any()))

 def test_default_source_and_original_fullhp_handlers(self):
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(subprocess.check_output(['git','show','fc84785f:gpu_engine/transition.py'],text=True));baseline=dump_default_program(old);self.assertEqual(dump_default_program(new),baseline);assert_hp_rng_mutations_detected(self,new,baseline)
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  script=r"""const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),{BattleActions}=require(root+'/dist/sim/battle-actions.js'),{Battle}=require(root+'/dist/sim/battle.js'),out=[];for(const id of ['recover','synthesis','morningsun','moonlight']){const move=d.moves.get(id),lines=[],p={hp:100,maxhp:100,baseMaxhp:100,fainted:false,isActive:true,status:'',toString:()=> 'p1a: Actor',getHealth:()=> '100/100',heal:()=>false},b={gen:3,add:(...xs)=>lines.push(xs.map(x=>typeof x==='function'?x.call(p):String(x)).join('|')),attrLastMove:(...xs)=>lines.push('attr:'+xs.join('|')),debug:()=>{},runEvent:()=>undefined,singleEvent:(name,m,st,t)=>m.onHit?.call(b,t),field:{isWeather:()=>false},trunc:Math.trunc};b.heal=n=>Battle.prototype.heal.call({...b,runEvent:(name,t,s,e,n)=>n},n,p,p,move);BattleActions.prototype.runMoveEffects.call({battle:b,combineResults:BattleActions.prototype.combineResults},[undefined],[p],p,move,move,false,false);out.push({id,lines})}console.log(JSON.stringify(out));"""
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(got[0]['lines'],['-fail|p1a: Actor|heal','attr:[still]']);self.assertTrue(all(not r['lines']for r in got[1:]))
  for modules in [('benchmark_retained_games','engine_lifecycle_event_device','engine_full_v4_event_device'),('engine_full_v4_event_device','engine_lifecycle_event_device','benchmark_retained_games')]:subprocess.run([__import__('sys').executable,'-c',';'.join('import gpu_engine.'+n for n in modules)],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
