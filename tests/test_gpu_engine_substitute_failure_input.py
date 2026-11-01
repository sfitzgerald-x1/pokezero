"""Source-bound no-write Substitute failures with both original FULL views."""
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
BASE='bd580cb6950f1cc2a5650b5f96679ffb1d7e27d5'
class SubstituteFailureInputTests(unittest.TestCase):
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
  cls.rules=rules;cls.sub=sub.compile_substitute(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=rules)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',substitute_compiled=cls.sub,substitute_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))

 def component(self,consume_initial=True):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);actor=rows%2;active=j.stack((z+3,z+4),axis=1);phys=active[rows,actor]
  s=state.empty_state(b)._replace(active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32))
  at=(rows,actor,phys);s=s._replace(hp=s.hp.at[at].set(j.where(rows<4,100,j.where(rows==7,1,25))),max_hp=s.max_hp.at[at].set(j.where(rows==7,1,100)))
  kw=dict(catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));cursor=life.register(rows+97001,**kw);initial=life._tape(cursor,self.producer,64)
  for side in(0,1):initial=life._reveal(initial,s,z+side,z-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in((0,1)if consume_initial else()):
   stream=full.register(cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=z+owner);r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);consumers.append((r.projection.state,r.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rt),tape,rt,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rt)))
  return s,tape,consumers,actor,phys


 def test_original_failure_callback_source_and_defaults(self):
  script=r"""const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),out=[];for(const [hp,maxhp,sub]of [[75,100,25],[25,100,25],[25,100,0],[1,1,0],[26,100,0]]){const lines=[],p={hp,maxhp,volatiles:sub?{substitute:{hp:sub}}:{},toString:()=> 'p1a: Actor'},b={NOT_FAIL:'nf',add:(...x)=>lines.push(x.map(String).join('|'))};out.push({result:d.moves.get('substitute').onTryHit.call(b,p)||null,lines,hp:p.hp,sub:p.volatiles.substitute?.hp||0});}console.log(JSON.stringify(out));"""
  r=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True))
  self.assertEqual([v['lines']for v in r],[['-fail|p1a: Actor|move: Substitute'],['-fail|p1a: Actor|move: Substitute'],['-fail|p1a: Actor|move: Substitute|[weak]'],['-fail|p1a: Actor|move: Substitute|[weak]'],[]]);self.assertEqual([v['result']for v in r],['nf']*4+[None]);self.assertEqual([v['hp']for v in r],[75,25,25,1,26]);self.assertEqual([v['sub']for v in r],[25,25,0,0,0])
  self.assertFalse(bool(sub.compile_substitute(engine_catalog=self.producer,metadata=self.metadata).device.failure));self.assertTrue(bool(self.sub.device.failure))
  with self.assertRaises(ValueError):sub.compile_substitute(engine_catalog=self.producer,metadata=self.metadata,source_rules=deepcopy(self.rules))
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(old);self.assertEqual(dump_default_program(new),baseline);assert_hp_rng_mutations_detected(self,new,baseline)
  witness._source()
  for order in [('benchmark_retained_games','engine_lifecycle_event_device','engine_full_v4_event_device'),('engine_full_v4_event_device','engine_lifecycle_event_device','benchmark_retained_games')]:subprocess.run([__import__('sys').executable,'-c',';'.join('import gpu_engine.'+n for n in order)],check=True,capture_output=True)

 def test_same_write_failures_both_full_and_refusals(self):
  s,t,consumers,actor,phys=self.component();b=8;rows=j.arange(b);yes=j.ones(b,bool);no=~yes;z=j.zeros(b,j.int32);created=rows<4;move=z+self.metadata['vocabulary']['moves'].index('substitute');target=s.active[rows,1-actor];at=(rows,actor,phys)
  t=public.move(t,actor,phys,target,move,eligible=created);t,ticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_SELF,actor,active=created)
  w=sub.Write(s.hp[at],s.hp[at]-25,s.max_hp[at],s.status[at],z,z+25);t=sub.record_device(t,actor,phys,move,w,self.runtime['substitute_context'],eligible=created);t=ex.capture_substitute_write(t,ticket,actor,phys,move,w,requested=created)
  before=s._replace(hp=s.hp.at[at].set(j.where(created,75,s.hp[at])),substitute_hp=s.substitute_hp.at[at].set(j.where(created,25,0)))
  t=public.move(t,actor,phys,target,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_SELF,actor,active=yes);unrecorded=t
  hp=before.hp[at];maximum=before.max_hp[at];oldsub=before.substitute_hp[at];w=sub.Write(hp,hp,maximum,before.status[at],oldsub,oldsub)
  t=sub.record_device(t,actor,phys,move,w,self.runtime['substitute_context'],eligible=yes,caller_move=move);t=ex.capture_substitute_write(t,ticket,actor,phys,move,w,requested=yes)
  idx=t.execution.ledger.start[rows,j.where(created,1,0)];self.assertTrue(bool(sub.owned_failure(t,idx).all()));self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  capture=risk.CapturedSubstitute(self.runtime['substitute_context'],ticket,actor,phys,move,w);r=risk.substitute_on_tape(t,yes,active=yes,captured=capture);self.assertEqual(r.execution.risks.carry.risk.tolist(),[0]*8)
  records=[];frame=life.LifecycleResult(before,t.cursor,t,life.snapshot(t.cursor),life.snapshot(t.cursor),no,z)
  for owner,(stream,pub)in enumerate(consumers):
   got=self.fold(stream,frame,pub);records.append(dict(owner=owner,error=got.projection.error.tolist(),words=got.projection.frames.words.tolist(),valid=got.projection.frames.valid.tolist()));Path('/tmp/pokezero-substitute-failure-full-current.json').write_text(json.dumps(records)+'\n');self.assertEqual(got.projection.error.tolist(),[0]*8)
   for row in range(b):
    who=int(actor[row]);ids={side:f'p{side+1}a: '+self.catalog.species.names[side+1]for side in(0,1)};initial=[]
    for side in(0,1):
     hptext='100/100'if side!=who or row<4 else('1/1'if row==7 and owner==who else'25/100'if row!=7 else'100/100');initial.append(f'|switch|{ids[side]}|{self.catalog.species.names[side+1]}, L80, M|{hptext}')
    wire,_=public_wire.project_lines(public_wire.WireState(),initial+['|turn|1'],**v4.wire_kwargs(self.catalog));lines=[]
    if row<4:lines=[f'|move|{ids[who]}|Substitute|{ids[who]}',f'|-start|{ids[who]}|Substitute',f'|-damage|{ids[who]}|75/100']
    lines += [f'|move|{ids[who]}|Substitute|{ids[who]}',f'|-fail|{ids[who]}|move: Substitute'+(''if row<4 else'|[weak]')]
    _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));expected=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,expected)
  for badcaller in (None,z+self.metadata['vocabulary']['moves'].index('sleeptalk')):
   bad=sub.record_device(unrecorded,actor,phys,move,w,self.runtime['substitute_context'],eligible=yes,caller_move=badcaller);self.assertEqual(bad.count.tolist(),unrecorded.count.tolist());self.assertTrue(bool((bad.refusal!=0).all()))
  wrong=w._replace(new_hp=hp-1);bad=sub.record_device(unrecorded,actor,phys,move,wrong,self.runtime['substitute_context'],eligible=yes,caller_move=move);self.assertTrue(bool((bad.refusal!=0).all()))
  for field in('engine_binding','source_binding','move_binding','producer_binding'):
   with self.assertRaises(ValueError):sub.owned_failure(t._replace(**{field:j.broadcast_to(getattr(t,field),(8,8))}),idx)
  self.assertFalse(bool(sub.owned_failure(t._replace(event_phase=t.event_phase.at[rows,idx-1].add(1)),idx).any()))
  old=t._replace(refusal=t.refusal|128);rr=risk.substitute_on_tape(old,yes,active=yes,captured=capture);self.assertTrue(bool((rr.refusal&128).all()))

if __name__=='__main__':unittest.main()
