"""Canonical Hidden Power current-member Substitute input."""
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
BASE='1ea9e1b0f4ebf9b5079107bb2c95151e285ed862'
class HiddenPowerSubstituteInputTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.NoDamageOwningTests.setUpClass();f=fixtures.NoDamageOwningTests;cls.f=f;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);cls.names=('hiddenpowerbug','hiddenpowerfire','substitute','sleeptalk')
  for name in cls.names:
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=True if name in ('substitute','sleeptalk')else 100,priority=info.priority,flags={}if name!='surf'else{'protect':1})
  # Normalize every installed legacy Hidden Power fixture alias to the same
  # original source row; its old generic Ice/Normal placeholder is not valid.
  for name in cls.metadata['vocabulary']['moves']:
   if not name.startswith('hiddenpower'):continue
   info=dex.move_info(name);moves[name]=info;i=cls.metadata['vocabulary']['moves'].index(name)
   flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]).Dex.mod('gen3');console.log(JSON.stringify(d.moves.get(process.argv[2]).flags))",str(showdown_root()/'dist/sim/dex.js'),name],text=True))
   cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=int(info.accuracy),priority=info.priority,flags=flags)
  # This shared catalog also contains Encore; keep that unused fixture row
  # equal to the original installed move rather than the generic always-hit row.
  if 'encore' in cls.metadata['vocabulary']['moves']:
   encore=dex.move_info('encore');moves['encore']=encore
   cls.metadata['moves'][cls.metadata['vocabulary']['moves'].index('encore')]=dict(id='encore',base_power=encore.base_power,type=cls.metadata['vocabulary']['types'].index(encore.type.lower()),source_type=encore.type,category=encore.category,pp=encore.pp,accuracy=100,priority=encore.priority,flags={'protect':1,'mirror':1,'bypasssub':1,'metronome':1,'failencore':1})
  moves['hiddenpower']=dex.move_info('hiddenpower');cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('heal-'+name,'authored',name,'authored',80,cls.names[:4],'Pressure','Leftovers'),))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-original-self-heal'),variants)
  labels=set(f.catalog.vocabulary.tokens)|{'move:hiddenpower','belief:possible_move:hiddenpower'}
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


 def literal(self):
  s,t,consumers,actor,phys=self.component();b=8;rows=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);no=~yes;target=s.active[rows,1-actor];ids=self.metadata['vocabulary']['moves'];move=j.where(rows%2==0,ids.index('hiddenpowerbug'),ids.index('hiddenpowerfire')).astype(j.int32)
  typ=j.where(rows%2==0,self.metadata['vocabulary']['types'].index('bug'),self.metadata['vocabulary']['types'].index('fire')).astype(j.int32);power=j.array([30,40,50,55,60,65,69,70],j.int32)
  s=s._replace(hidden_power_type=s.hidden_power_type.at[rows,actor,phys].set(typ),hidden_power_power=s.hidden_power_power.at[rows,actor,phys].set(power),hp=s.hp.at[rows,1-actor,target].set(75),substitute_hp=s.substitute_hp.at[rows,1-actor,target].set(25))
  # Current installed resolver is called with SAME member operands; no values
  # are reconstructed from the old R9 packet or synthesized default0/70.
  from gpu_engine import variable_moves
  resolved,move_type,_=variable_moves.resolve(self.table.variable_moves,move,self.table.power[move],self.table.type[move],self.table.physical[move],s.hidden_power_type[rows,actor,phys],s.hidden_power_power[rows,actor,phys],s.happiness[rows,actor,phys],s.hp[rows,actor,phys],s.max_hp[rows,actor,phys])
  ctx=self.runtime['direct_hit_context'];requested=cb.route(ctx,move,acts=yes,power=resolved,hit=yes,immune=no,protected=no,enduring=no,bypass=no,absorbed=no,sub_hp=z+25,caller_move=move,actor_hp_type=s.hidden_power_type[rows,actor,phys],actor_hp_power=s.hidden_power_power[rows,actor,phys],resolved_type=move_type)
  t=public.move(t,actor,phys,target,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_DAMAGE,1-actor,active=requested)
  p=cb.Pending(ctx,ticket,actor,phys,target,move,requested,None,move,None,s.hidden_power_type[rows,actor,phys],s.hidden_power_power[rows,actor,phys],move_type)
  damage=j.where(rows%2==0,10,40);dealt=j.minimum(25,damage);w=cb.Write(resolved,yes,no,no,no,no,no,rows%2==0,z+2,z+1,z+25,25-dealt,dealt,damage,z+75,z+75)
  return s,t,consumers,p,w

 def test_original_member_callbacks_aliases_and_exact_operand_controls(self):
  script=r"""const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),m=d.moves.get('hiddenpower'),out=[];for(const [hpType,hpPower]of[['Bug',30],['Fire',70]]){let move={type:'Normal',category:'Physical'},mon={hpType,hpPower};m.onModifyType(move,mon);m.onModifyMove(move,mon);out.push({type:move.type,category:move.category,power:m.basePowerCallback.call({debug(){}},mon),aliasID:d.moves.get('hiddenpower'+hpType.toLowerCase()).id})}console.log(JSON.stringify(out));"""
  self.assertEqual(json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True)),[dict(type='Bug',category='Physical',power=30,aliasID='hiddenpower'),dict(type='Fire',category='Special',power=70,aliasID='hiddenpower')])
  s,t,_,p,w=self.literal();self.assertTrue(bool(p.requested.all()));out=cb.record(t,p,w);self.assertTrue(bool(cb.span_matches(out,p,w).all()))
  for bad,write in((p._replace(actor_hp_type=None),w),(p._replace(actor_hp_power=None),w),(p._replace(resolved_type=None),w),(p._replace(caller_move=p.move+1),w),(p._replace(actor_hp_type=j.zeros(8,j.int32)),w),(p._replace(actor_hp_power=j.zeros(8,j.int32)),w),(p._replace(actor_hp_power=j.full(8,71,j.int32)),w),(p._replace(resolved_type=(p.resolved_type%17)+1),w),(p,w._replace(power=w.power+1))):
   kept=cb.record(t,bad,write);self.assertEqual(kept.count.tolist(),t.count.tolist());self.assertTrue(bool((kept.execution.ledger.refusal!=0).all()))
  # Unused crit/type values still belong to this SAME final Outcome.
  c=cb.Captured(p,w);o=risk.Outcome(w.power,w.critical,w.hit,w.immune,w.numerator,w.denominator,j.zeros(8,bool),j.ones(8,bool),w.protected,w.enduring,j.zeros(8,j.int32),j.zeros(8,bool));good=risk.outcome_on_tape(out,o,eligible=p.requested,active=p.requested,captured_substitute_damage=c);self.assertEqual(good.execution.risks.carry.risk.tolist(),[0]*8)
  bad=risk.outcome_on_tape(out,o._replace(power=o.power+1),eligible=p.requested,active=p.requested,captured_substitute_damage=c);self.assertTrue(bool((bad.execution.risks.carry.risk!=0).all()))
  old=dump_default_program(ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True)));current=ast.parse(Path('gpu_engine/transition.py').read_text());self.assertEqual(dump_default_program(current),old);assert_hp_rng_mutations_detected(self,current,old);witness._source()
  for modules in(('benchmark_retained_games','engine_lifecycle_event_device','engine_full_v4_event_device'),('engine_full_v4_event_device','engine_lifecycle_event_device','benchmark_retained_games')):subprocess.run([__import__('sys').executable,'-c',';'.join('import gpu_engine.'+n for n in modules)],check=True,capture_output=True)

 def test_actual_alias_current_write_both_full_survive_and_break(self):
  # Actual Sub creation precedes this literal current-member input transaction.
  s,t,_,actor,phys=self.component(False);b=8;rows=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);no=~yes;target=s.active[rows,1-actor];ids=self.metadata['vocabulary']['moves'];submove=z+ids.index('substitute')
  t=public.move(t,1-actor,target,phys,submove,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_SELF,1-actor,active=yes);create=sub.Write(z+100,z+75,z+100,z,z,z+25);t=sub.record_device(t,1-actor,target,submove,create,self.runtime['substitute_context'],eligible=yes);t=ex.capture_substitute_write(t,ticket,1-actor,target,submove,create,requested=yes)
  # Reuse SAME initial stream and exact successful creation tape, then capture
  # current member values with mixed aliases, powers, actors, survive/break.
  current,_,consumers,p,w=self.literal();t=public.move(t,p.actor,p.physical,p.target,p.move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_DAMAGE,1-p.actor,active=yes);p=p._replace(ticket=ticket);t=cb.record(t,p,w);self.assertTrue(bool(cb.span_matches(t,p,w).all()));self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  after=current._replace(substitute_hp=current.substitute_hp.at[rows,1-p.actor,p.target].set(w.new_sub));frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(t.cursor),life.snapshot(t.cursor),no,z);records=[]
  for owner,(stream,pub)in enumerate(consumers):
   got=self.fold(stream,frame,pub);records.append(dict(owner=owner,error=got.projection.error.tolist(),words=[np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])].tolist()for row in range(b)]));Path('/tmp/pokezero-hidden-power-sub-full-current.json').write_text(json.dumps(records)+'\n');self.assertEqual(got.projection.error.tolist(),[0]*b)
   for row in range(b):
    names={side:self.catalog.species.names[side+1]for side in(0,1)};who=int(p.actor[row]);foe=1-who;ident={side:f'p{side+1}a: '+names[side]for side in(0,1)};wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ident[side]}|{names[side]}, L80, M|100/100'for side in(0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));lines=[f'|move|{ident[foe]}|Substitute|{ident[foe]}',f'|-start|{ident[foe]}|Substitute',f'|-damage|{ident[foe]}|75/100',f'|move|{ident[who]}|Hidden Power|{ident[foe]}']
    if bool(w.critical[row]):lines.append(f'|-crit|{ident[foe]}')
    lines.append(f'|-supereffective|{ident[foe]}')
    lines.append(f'|-end|{ident[foe]}|Substitute'if int(w.new_sub[row])==0 else f'|-activate|{ident[foe]}|Substitute|[damage]')
    _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for packet in packets for e in packet.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,want)
  for field in('hp','status','pp','keys','boosts'):np.testing.assert_array_equal(getattr(current,field),getattr(after,field))

if __name__=='__main__':unittest.main()
