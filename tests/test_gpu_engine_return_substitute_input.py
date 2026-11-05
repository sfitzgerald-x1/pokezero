"""Canonical Return same-call power and original Substitute BOTH FULL components."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_substitute_damage_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,state
from gpu_engine import engine_source_frame_witness as witness
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixtures
from gpu_engine import engine_direct_ko_callbacks as ko
from types import SimpleNamespace
from gpu_engine import engine_substitute_callbacks as sub
BASE='28ef438fde306222416154f39c233c7798fb67e2'
class ReturnSubstituteInputTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.NoDamageOwningTests.setUpClass();f=fixtures.NoDamageOwningTests;cls.f=f;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);cls.names=('return','substitute','surf','sleeptalk')
  for name in cls.names:
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=True if name in ('substitute','sleeptalk')else 100,priority=info.priority,flags={}if name!='surf'else{'protect':1})
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
  for n in cls.names:
   flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]).Dex.mod('gen3');console.log(JSON.stringify(d.moves.get(process.argv[2]).flags))",str(showdown_root()/'dist/sim/dex.js'),n],text=True));cls.metadata['moves'][cls.metadata['vocabulary']['moves'].index(n)]['flags']=flags
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  rules=request.load_source_rules(showdown_root(),cls.metadata);cls.rules=rules;req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=rules)
  from gpu_engine import engine_substitute_callbacks as sub
  cls.sub=sub.compile_substitute(engine_catalog=cls.producer,metadata=cls.metadata)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',substitute_compiled=cls.sub,substitute_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))

 def component(self,consume_initial=True):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);actor=rows%2;active=j.stack((z+3,z+4),axis=1);phys=active[rows,actor]
  s=state.empty_state(b)._replace(active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32))
  kw=dict(catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));cursor=life.register(rows+97001,**kw);initial=life._tape(cursor,self.producer,64)
  for side in(0,1):initial=life._reveal(initial,s,z+side,z-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in((0,1)if consume_initial else()):
   stream=full.register(cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=z+owner);r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);consumers.append((r.projection.state,r.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rt),tape,rt,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rt)))
  return s,tape,consumers,actor,phys

 def test_original_return_sub_callback_and_no_missing_operands(self):
  script=r'''const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),m=d.moves.get('return'),c=d.moves.get('substitute').condition,out=[];
for(const damage of [20,40]){const lines=[],source={happiness:255,toString:()=> 'p1a: Actor'},target={hp:75,volatiles:{substitute:{hp:25}},toString:()=> 'p2a: Target',removeVolatile(id){c.onEnd.call(b,this);delete this.volatiles[id]},addVolatile(id){this.volatiles[id]={}}},b={actions:{getDamage:()=>damage},add:(...xs)=>lines.push(xs.map(String).join('|')),runEvent(){},HIT_SUBSTITUTE:'sub'};const result=c.onTryPrimaryHit.call(b,target,source,m);out.push({power:m.basePowerCallback(source),targetHP:target.hp,remaining:target.volatiles.substitute?.hp||0,lines,result});}console.log(JSON.stringify(out));'''
  result=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(result,[dict(power=102,targetHP=75,remaining=5,lines=['-activate|p2a: Target|Substitute|[damage]'],result='sub'),dict(power=102,targetHP=75,remaining=0,lines=['-end|p2a: Target|Substitute'],result='sub')])
  s,t,_,side,phys=self.component(consume_initial=False);b=8;yes=j.ones(b,bool);no=~yes;z=j.zeros(b,j.int32);move=z+self.metadata['vocabulary']['moves'].index('return');kwargs=dict(acts=yes,power=z+102,hit=yes,immune=no,protected=no,enduring=no,bypass=no,absorbed=no,sub_hp=z+25)
  self.assertEqual(int(self.helper.device.substitute_damage.return_move),int(move[0]));self.assertTrue(bool(cb.route(self.runtime['direct_hit_context'],move,**kwargs,actor_happiness=z+255,caller_move=move).all()))
  for extras in ({},{'actor_happiness':z+255},{'actor_happiness':z+256,'caller_move':move},{'actor_happiness':z+128,'caller_move':move},{'actor_happiness':z+255,'caller_move':z+self.metadata['vocabulary']['moves'].index('sleeptalk')}):self.assertFalse(bool(cb.route(self.runtime['direct_hit_context'],move,**kwargs,**extras).any()))
  surf=z+self.metadata['vocabulary']['moves'].index('surf');self.assertTrue(bool(cb.route(self.runtime['direct_hit_context'],surf,**kwargs).all()))
  base=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));current=ast.parse(Path('gpu_engine/transition.py').read_text());expected=dump_default_program(base);self.assertEqual(dump_default_program(current),expected);assert_hp_rng_mutations_detected(self,current,expected)

 def test_typed_creation_return_survive_break_both_full_and_claim_guards(self):
  s,t,consumers,actor,phys=self.component();b=8;rows=j.arange(b);yes=j.ones(b,bool);no=~yes;z=j.zeros(b,j.int32);target=s.active[rows,1-actor];ids=self.metadata['vocabulary']['moves'];submove=z+ids.index('substitute');move=z+ids.index('return');context=self.runtime['direct_hit_context']
  t=public.move(t,1-actor,target,phys,submove,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_SELF,1-actor,active=yes);write=sub.Write(z+100,z+75,z+100,z,z,z+25);t=sub.record_device(t,1-actor,target,submove,write,self.runtime['substitute_context'],eligible=yes);t=ex.capture_substitute_write(t,ticket,1-actor,target,submove,write,requested=yes)
  before=s._replace(hp=s.hp.at[rows,1-actor,target].set(75),substitute_hp=s.substitute_hp.at[rows,1-actor,target].set(25));happiness=before.happiness[rows,actor,phys];power=z+102;damage=j.where(rows%2==0,20,40);dealt=j.minimum(25,damage);newsub=25-dealt
  requested=cb.route(context,move,acts=yes,power=power,hit=yes,immune=no,protected=no,enduring=no,bypass=no,absorbed=no,sub_hp=before.substitute_hp[rows,1-actor,target],actor_happiness=happiness,caller_move=move);self.assertTrue(bool(requested.all()))
  t=public.move(t,actor,phys,target,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_DAMAGE,1-actor,active=requested);pending=cb.Pending(context,ticket,actor,phys,target,move,requested,happiness,move)
  write=cb.Write(power,yes,no,no,no,no,no,no,z+1,z+1,z+25,newsub,dealt,damage,z+75,z+75)
  unrecorded=t
  t=cb.record(t,pending,write);self.assertTrue(bool(cb.span_matches(t,pending,write).all()));self.assertEqual(t.execution.ledger.stage[:,1].tolist(),[2]*b)
  o=risk.Outcome(power,no,yes,no,z+1,z+1,no,yes,no,no,z,no);observed=risk.outcome_on_tape(t,o,eligible=yes,active=yes,captured_substitute_damage=cb.Captured(pending,write));self.assertEqual(observed.execution.risks.carry.risk.tolist(),[0]*b)
  self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  after=before._replace(substitute_hp=before.substitute_hp.at[rows,1-actor,target].set(newsub));frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(t.cursor),life.snapshot(t.cursor),no,z);records=[]
  for owner,(stream,pub)in enumerate(consumers):
   got=self.fold(stream,frame,pub);records.append(dict(owner=owner,error=got.projection.error.tolist(),words=got.projection.frames.words.tolist(),valid=got.projection.frames.valid.tolist()));Path('/tmp/pokezero-return-sub-full-current.json').write_text(json.dumps(records)+'\n');self.assertEqual(got.projection.error.tolist(),[0]*b)
   for row in range(b):
    names={side:self.catalog.species.names[side+1]for side in(0,1)};who=int(actor[row]);foe=1-who;ident={side:f'p{side+1}a: '+names[side]for side in(0,1)}
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ident[side]}|{names[side]}, L80, M|100/100'for side in(0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));lines=[f'|move|{ident[foe]}|Substitute|{ident[foe]}',f'|-start|{ident[foe]}|Substitute',f'|-damage|{ident[foe]}|75/100',f'|move|{ident[who]}|Return|{ident[foe]}',f'|-activate|{ident[foe]}|Substitute|[damage]'if row%2==0 else f'|-end|{ident[foe]}|Substitute'];_,original=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));expected=np.asarray([e.words()for p in original for e in p.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,expected)
  for bad in (pending._replace(actor_happiness=None),pending._replace(actor_happiness=z+128),pending._replace(caller_move=z+ids.index('sleeptalk'))):
   kept=cb.record(unrecorded,bad,write);self.assertEqual(kept.count.tolist(),unrecorded.count.tolist());self.assertTrue(bool((kept.execution.ledger.refusal!=0).all()))
  self.assertFalse(bool(cb.span_matches(t._replace(event_turn=t.event_turn.at[:,2].add(1)),pending,write).any()))
  for field in ('hp','happiness','status','pp','keys','boosts'):np.testing.assert_array_equal(getattr(before,field),getattr(after,field))

if __name__=='__main__':unittest.main()
