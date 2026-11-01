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
BASE='977d524d2b947a421e94b40105970e8a52d4a298'

class SelfHealOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.NoDamageOwningTests.setUpClass();f=fixtures.NoDamageOwningTests;cls.f=f;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);cls.names=('recover','synthesis','softboiled','surf')
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
  variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('heal-'+name,'authored',name,'authored',80,cls.names,'Pressure','Leftovers'),))for name in cls.metadata['vocabulary']['species'][1:]}
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

 def test_original_source_callback_closed_scope_and_default_ast(self):
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),74)
  ids=self.metadata['vocabulary']['moves'];mask=self.helper.device.self_heal.candidates
  self.assertEqual(set(ids[i]for i in np.flatnonzero(mask)),{'recover','synthesis','softboiled'})
  self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.self_heal)
  for n in self.names[:3]:self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[ids.index(n)]),0)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(old)
  # The composed source adds exactly this independently reviewed observer
  # import; its numerical statements and RNG expressions remain in the oracle.
  observers=[n for n in new.body if isinstance(n,ast.ImportFrom) and n.level==1 and n.module is None and len(n.names)==1 and n.names[0].name=='engine_pinch_item_callbacks' and n.names[0].asname=='pinch_callbacks']
  self.assertEqual(len(observers),1);new.body.remove(observers[0]);self.assertEqual(dump_default_program(new),baseline);assert_hp_rng_mutations_detected(self,new,baseline)
  self.assertIn('eligible=acts&caller_acts&(caller_move==move)',Path('gpu_engine/transition.py').read_text())
  script=r'''const root=process.argv[1],{Dex}=require(root+'/dist/sim/dex.js'),{Battle}=require(root+'/dist/sim/battle.js'),d=Dex.mod('gen3'),out=[];
for(const id of ['recover','softboiled','synthesis'])for(const weather of ['','sunnyday','raindance','sandstorm','hail']){
 const lines=[],m=d.moves.get(id),p={hp:40,maxhp:101,baseMaxhp:101,isActive:true,status:'',getHealth(){return this.hp+'/101'},toString:()=> 'p1a: Actor',heal(n){const old=this.hp;this.hp=Math.min(101,this.hp+n);return this.hp-old}};
 const b={effect:m,event:{target:p,source:p},trunc:Math.trunc,runEvent:(id,t,s,e,n)=>n,field:{isWeather:xs=>xs.includes(weather)},heal:Battle.prototype.heal,add:(...xs)=>lines.push(xs.map(x=>typeof x==='function'?x.call(p):String(x)).join('|'))};
 if(m.heal)b.heal(p.baseMaxhp*m.heal[0]/m.heal[1],p,p,m);else m.onHit.call(b,p);out.push({id,weather,hp:p.hp,lines});}process.stdout.write(JSON.stringify(out));'''
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True))
  for row in got:
   hp=107 if row['id']=='synthesis'and row['weather']=='sunnyday'else 65 if row['id']=='synthesis'and row['weather']in('raindance','sandstorm','hail')else 90;hp=min(101,hp)
   self.assertEqual(row['hp'],hp);self.assertEqual(row['lines'],[f'-heal|p1a: Actor|{hp}/101'])

 def component(self):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);actor=rows%2;physical=j.where(actor==0,3,4);foephysical=j.where(actor==0,4,3);active=j.stack((j.full(b,3),j.full(b,4)),axis=1)
  s=state.empty_state(b)._replace(active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),40,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32))
  for row,hp,maxhp in ((0,77,239),(1,80,264),(2,219,341)):
   s=s._replace(hp=s.hp.at[row,actor[row],physical[row]].set(hp),max_hp=s.max_hp.at[row,actor[row],physical[row]].set(maxhp))
  kw=dict(catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));cursor=life.register(rows+54001,**kw);initial=life._tape(cursor,self.producer,64)
  for side in (0,1):initial=life._reveal(initial,s,z+side,z-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in (0,1):
   stream=full.register(cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=z+owner);r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);consumers.append((r.projection.state,r.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rc=self.runtime['kernel_risk_compiled'];tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rc.device),tape,rc.device,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rc.device)))
  ids=self.metadata['vocabulary']['moves'];move=j.array([ids.index(n)for n in ['recover','softboiled','synthesis','recover','recover','recover','recover','synthesis']],j.int32)
  s=s._replace(hp=s.hp.at[3,actor[3],physical[3]].set(100).at[4,actor[4],physical[4]].set(0))
  eligible=yes.at[6].set(False);tape=public.move(tape,actor,physical,foephysical,move,eligible=eligible);tape,p=cb.begin(tape,s,actor,move,self.runtime['direct_hit_context'],self.table,eligible=eligible)
  new=j.array([196,212,341,100,0,39,90,90],j.int32);after=s._replace(hp=s.hp.at[rows,actor,physical].set(new));weather=j.array([0,0,0,0,0,0,0,0],j.int32);tape,capture=cb.finish(tape,p,after,weather,yes)
  return s,after,initial.cursor,consumers,tape,capture,actor,eligible

 def test_mixed_source_span_namespace_lifetime_and_unsupported(self):
  s,after,cursor,consumers,t,c,actor,eligible=self.component();good=[True,True,True,False,False,False,False,True]
  self.assertEqual(cb.owned(t,1).tolist(),good);self.assertEqual(cb.captured_matches(t,c).tolist(),good)
  observed=risk.self_heal_on_tape(t,j.ones(8,bool),active=eligible,captured=c);self.assertEqual((observed.execution.risks.carry.risk&risk.SELF_EFFECT!=0).tolist(),[False,False,False,True,True,True,False,False])
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.owned(bad,1).any()))
   with self.assertRaises(ValueError):cb.owned(t._replace(**{name:j.broadcast_to(getattr(t,name),(8,8))}),1)
  for name in ('turn','phase','generation'):
   bad=t._replace(cursor=t.cursor._replace(**{name:getattr(t.cursor,name)+1}));self.assertFalse(bool(cb.captured_matches(bad,c).any()))
  for bad in [t._replace(words=t.words.at[0,0,5].set(0)),t._replace(event_turn=t.event_turn.at[0,1].add(1)),t._replace(execution=t.execution._replace(ledger=t.execution.ledger._replace(claimed=t.execution.ledger.claimed.at[0,1].set(0))))]:self.assertFalse(bool(cb.owned(bad,1)[0]))
  prior=t._replace(refusal=t.refusal.at[0].set(65),execution=t.execution._replace(risks=t.execution.risks._replace(carry=t.execution.risks.carry._replace(risk=t.execution.risks.carry.risk.at[0].set(64)))))
  kept=risk.self_heal_on_tape(prior,j.ones(8,bool),active=eligible,captured=c);self.assertEqual(int(kept.refusal[0])&65,65);self.assertEqual(int(kept.execution.risks.carry.risk[0])&64,64)

 def test_both_full_consumers_exact_original_move_heal_words(self):
  s,after,cursor,consumers,t,c,actor,eligible=self.component();t=t._replace(count=j.where(j.arange(8)<3,t.count,0));frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(cursor),life.snapshot(t.cursor),j.zeros(8,bool),j.zeros(8,j.int32))
  for owner,(stream,pub)in enumerate(consumers):
   r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error[:3].tolist(),[0]*3)
   for row,name in enumerate(('recover','softboiled','synthesis')):
    names={side:self.catalog.species.names[side+1]for side in(0,1)};ident={side:f'p{side+1}a: '+names[side]for side in(0,1)};who=row%2
    def shown(hp,maximum,side):
     if owner==side:return f'{hp}/{maximum}'
     percent=(100*hp+maximum-1)//maximum
     if hp<maximum and percent==100:percent=99
     return f'{percent}/100'
    initial=[]
    for side in(0,1):
     physical=int(s.active[row,side]);initial.append(f'|switch|{ident[side]}|{names[side]}, L80, M|{shown(int(s.hp[row,side,physical]),int(s.max_hp[row,side,physical]),side)}')
    wire,_=public_wire.project_lines(public_wire.WireState(),initial+['|turn|1'],**v4.wire_kwargs(self.catalog));physical=int(after.active[row,who]);health=shown(int(after.hp[row,who,physical]),int(after.max_hp[row,who,physical]),who)
    _,packets=public_wire.project_lines(wire,[f'|move|{ident[who]}|{self.dex.move_info(name).name}|{ident[who]}',f'|-heal|{ident[who]}|{health}'],**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(r.projection.frames.words[:,row])[np.asarray(r.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,want)
   self.assertFalse(bool(r.public.cursor.needs_more[:3].any()))
