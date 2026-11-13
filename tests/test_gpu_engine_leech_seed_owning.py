"""Original Leech Seed helper transactions; no scientific games/models."""
from _component_evidence import capture_path
import json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_leech_seed_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,source_provenance
from test_gpu_engine_gameplay_stream_comparison import state as literal_state
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixture

class LeechSeedOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.NoDamageOwningTests.setUpClass();cls.f=fixture.NoDamageOwningTests;cls.metadata=deepcopy(cls.f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(cls.f.catalog.dex.moves);cls.names=('leechseed','raindance','sunnyday','hail');info=dex.move_info('leechseed');moves['leechseed']=info
  if 'leechseed' not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append('leechseed');cls.metadata['moves'].append(None)
  cls.move=cls.metadata['vocabulary']['moves'].index('leechseed');cls.metadata['moves'][cls.move]=dict(id='leechseed',base_power=0,type=cls.metadata['vocabulary']['types'].index('grass'),source_type='Grass',category='Status',pp=10,accuracy=90,priority=0,flags={'protect':1,'reflectable':1,'mirror':1,'metronome':1})
  if 'encore' in cls.metadata['vocabulary']['moves']:
   enc=dex.move_info('encore');moves['encore']=enc;cls.metadata['moves'][cls.metadata['vocabulary']['moves'].index('encore')]=dict(id='encore',base_power=enc.base_power,type=cls.metadata['vocabulary']['types'].index(enc.type.lower()),source_type=enc.type,category=enc.category,pp=enc.pp,accuracy=100,priority=enc.priority,flags={'protect':1,'mirror':1,'bypasssub':1,'metronome':1,'failencore':1})
  cls.dex=ShowdownDex(moves,cls.f.catalog.dex.species,cls.f.catalog.dex.type_chart)
  variants={n:Gen3RandbatSpeciesUniverse(n,80,(Gen3RandbatVariant('drum-'+n,'authored',n,'authored',80,cls.names,'Pressure','Leftovers'),))for n in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-leech-source'),variants)
  labels=set(cls.f.catalog.vocabulary.tokens)|{'move:leechseed','belief:possible_move:leechseed'}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.f.catalog.masks.__dict__)
  original_flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]).Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>!!d.moves.get(n).flags.failencore)))",str(showdown_root()/'dist/sim/dex.js'),json.dumps(cls.metadata['vocabulary']['moves'])],text=True))
  for row,flag in zip(cls.metadata['moves'],original_flags):
   if row is not None:row.setdefault('flags',{})['failencore']=flag
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer)
  cls.facts=own.compile_own_facts(cls.req,cls.catalog);cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(cls.req,cls.facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))

 @classmethod
 def setUp(cls):
  if not hasattr(cls,'fold'):
   cls.fold=staticmethod(jax.jit(lambda st,fr,p:full.project_and_fold(st,fr,cls.runtime['public'].device,p,cls.catalog)))
   cls.consume=staticmethod(jax.jit(lambda st,fr,p:ex.consume_frame_device(fr.callbacks.execution.ledger,fr,st,p,active=j.ones(len(fr.callbacks.count),bool),tables=cls.runtime['execution_compiled'].device,adapter_tables=cls.runtime['public'].device,catalog=cls.catalog)))
 def test_original_metadata_default_program_and_d_call_join(self):
  from gpu_engine import engine_source_frame_witness as witness
  witness._source();d=self.helper.device.leech_seed;self.assertEqual(int(d.move_id),self.move);self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[self.move]),0)
  self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.leech_seed)
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  from gpu_engine import v5_public_branch as branch,v5_hypothetical_batch as batch
  for file in(branch.__file__,batch.__file__):self.assertIn('driver._leech_history',Path(file).read_text())
  self.assertTrue(np.all(np.asarray(self.runtime['public'].device.leech)>0))
 def component(self,*,mixed=False):
  from gpu_engine import state,sustain,mechanics
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);source=rows%2;target=1-source;active=j.stack((z+3,z+4),axis=1);tp=active[rows,target];sp=active[rows,source]
  s=state.empty_state(b)._replace(done=j.zeros(b,bool),active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),185,j.int32),max_hp=j.full((b,2,6),201,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32),ability=j.full((b,2,6),self.metadata['vocabulary']['abilities'].index('pressure'),j.int32))
  types=self.rules.device.species_types[s.species];s=s._replace(types=types,base_types=types,hp=s.hp.at[rows,source,sp].set(j.where(rows%2==0,151,201)),status=s.status.at[7,target[7],tp[7]].set(mechanics.PSN))
  eligible=yes
  if mixed:
   s=s._replace(substitute_hp=s.substitute_hp.at[2,target[2],tp[2]].set(1),types=s.types.at[3,target[3],tp[3],0].set(self.table.sustain.grass_type),volatile=s.volatile.at[4,target[4],tp[4]].set(state.LEECH_SEED),hp=s.hp.at[5,target[5],tp[5]].set(0));eligible=yes.at[6].set(False)
  start=life.register(rows+65001,**self.kw);initial=life._tape(start,self.producer,64)
  for seat in(0,1):initial=life._reveal(initial,s,z+seat,z-1,active[:,seat],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes)
  frame0=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(start),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in(0,1):
   stream=full.register(start.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=start.stream_id,owner_sides=z+owner);got=self.fold(stream,frame0,pub)
   # Dead synthetic negative row is not a supported reveal; isolate its error.
   self.assertEqual(got.projection.error[j.array([0,1,2,3,4,6,7])].tolist(),[0]*7);consumers.append((got.projection.state,got.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;tape=ex.begin_on_tape(tape,ex.initialize(start.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rt),tape,rt,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rt),leech_history=cb.empty_history(tape.cursor,self.helper.device.leech_seed)))
  tape=public._append(tape,public.MOVE,source,sp,target,tp,(z+self.move,z,z,z,z),eligible)
  tape,p=cb.begin_seed(tape,s,source,z+self.move,self.runtime['direct_hit_context'],self.table.sustain,eligible=eligible,secondary=yes)
  seeded=sustain.seed_target(self.table.sustain,z+self.move,s.types[rows,target,tp],s.hp[rows,target,tp],s.substitute_hp[rows,target,tp],s.volatile[rows,target,tp],eligible=eligible);after=s._replace(volatile=s.volatile.at[rows,target,tp].set(seeded));tape,c=cb.finish_seed(tape,p,after)
  return s,after,source,target,eligible,tape,c,consumers,initial.cursor
 def residual(self,s,source,target,tape):
  from gpu_engine import sustain
  b=len(source);rows=j.arange(b);sp=s.active[rows,source];tp=s.active[rows,target];yes=j.ones(b,bool)
  tape,p=cb.begin_drain(tape,s,target,self.runtime['direct_hit_context'],eligible=yes)
  result=sustain.leech(s.hp[rows,target,tp],s.max_hp[rows,target,tp],s.volatile[rows,target,tp],s.hp[rows,source,sp],s.max_hp[rows,source,sp],eligible=yes)
  after=s._replace(hp=s.hp.at[rows,target,tp].set(result.hp));tape,c=cb.finish_drain(tape,p,after,result);tape,h=cb.begin_heal(tape,after,c);after=after._replace(hp=after.hp.at[rows,source,sp].set(result.recipient_hp));tape,hc=cb.finish_heal(tape,h,after)
  return after,tape,c,hc
 def test_mixed_same_write_namespaces_history_and_inactive(self):
  s,after,source,target,eligible,t,c,_,_=self.component(mixed=True);expected=[True,True,False,False,False,False,False,True]
  self.assertEqual(cb.captured_matches(t,c).tolist(),expected);self.assertEqual(cb.owned(t,j.maximum(t.count-1,0)).tolist(),expected)
  for field in('hp','pp','moves','keys','status','ability','types'):np.testing.assert_array_equal(getattr(s,field),getattr(after,field))
  for name in('engine_binding','source_binding','move_binding','producer_binding'):
   self.assertFalse(bool(cb.owned(t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))}),j.maximum(t.count-1,0)).any()))
   with self.assertRaises(ValueError):cb.owned(t._replace(**{name:j.broadcast_to(getattr(t,name),(8,8))}),0)
  h=t.execution.leech_history
  for name in('turn','generation','physical','source_slot','stream_id'):
   bad=h._replace(**{name:getattr(h,name)+1});got=t._replace(execution=t.execution._replace(leech_history=bad));self.assertFalse(bool(cb._known(got,self.helper.device.leech_seed,target,after.active[j.arange(8),target]).any()),name)
  prior=t._replace(refusal=t.refusal|65);got=risk.branch_on_tape(prior,risk.Branch.SEED_START,eligible,active=j.ones(8,bool),captured_leech=c);self.assertEqual(int(got.refusal[0])&65,65)
  keep=cb.commit_history(j.array([False,True,False,True,False,True,False,True]),h,h._replace(seeded=~h.seeded));np.testing.assert_array_equal(keep.seeded[::2],h.seeded[::2]);np.testing.assert_array_equal(keep.binding,h.binding)
 def wire_prefix(self,row,owner):
  from gpu_engine import mechanics
  names={seat:self.catalog.species.names[seat+1]for seat in(0,1)};ids={seat:f'p{seat+1}a: '+names[seat]for seat in(0,1)};source=row%2;target=1-source;status={0:'',mechanics.PSN:' psn'};lines=[]
  for seat in(0,1):
   hp=(151 if row%2==0 else 201)if seat==source else 185;num=hp if seat==owner else(100*hp+200)//201;den=201 if seat==owner else 100;suffix=' psn'if row==7 and seat==target else'';lines.append(f'|switch|{ids[seat]}|{names[seat]}, L80, M|{num}/{den}{suffix}')
  return ids,lines+['|turn|1']
 def test_both_original_full_and_prior_public_cross_turn(self):
  s,seeded,source,target,eligible,t,c,consumers,cursor=self.component();after,t,drain,heal=self.residual(seeded,source,target,t);self.assertEqual(cb.captured_matches(t,heal).tolist(),[True]*8)
  frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(cursor),life.snapshot(t.cursor),j.zeros(8,bool),j.zeros(8,j.int32));pubs=[];ledgers=[]
  for owner,(stream,pub)in enumerate(consumers):
   ledger,got=self.consume(stream,frame,pub);pubs.append(got.public);ledgers.append(ledger)
   capture=capture_path(self,'leech-seed-component-owner'+str(owner)+'.json');capture.write_text(json.dumps(dict(error=got.projection.error.tolist(),refusal=ledger.refusal.tolist(),words=[np.asarray(got.projection.frames.words[:,r])[np.asarray(got.projection.frames.valid[:,r])].tolist()for r in(0,1,7)]))+'\n')
   self.assertEqual(got.projection.error.tolist(),[0]*8);self.assertEqual(ex.report(ledger,tables=self.runtime['execution_compiled'].device).scoped_drained.tolist(),[True]*8)
   for row in(0,1,7):
    ids,lines=self.wire_prefix(row,owner);src=row%2;tar=1-src;status=' psn'if row==7 else'';targethp=160 if owner==tar else 80;targetmax=201 if owner==tar else 100
    lines2=[f'|move|{ids[src]}|Leech Seed|{ids[tar]}',f'|-start|{ids[tar]}|move: Leech Seed',f'|-damage|{ids[tar]}|{targethp}/{targetmax}{status}|[from] Leech Seed|[of] {ids[src]}']
    if row%2==0:lines2.append(f'|-heal|{ids[src]}|'+('176/201'if owner==src else'88/100')+'|[silent]')
    wire,_=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog));_,packets=public_wire.project_lines(wire,lines2,**v4.wire_kwargs(self.catalog));expected=np.array([x.words()for packet in packets for x in packet.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,expected)
  history=cb.history_from_public(after,t.cursor,tuple(pubs),data=self.helper.device.leech_seed,active_tables=self.catalog.active.device,healthy=j.ones(8,bool));self.assertEqual(history.seeded[j.arange(8),target].tolist(),[True]*8)
  missing=cb.history_from_public(after,t.cursor,(consumers[0][1],pubs[1]),data=self.helper.device.leech_seed,active_tables=self.catalog.active.device,healthy=j.ones(8,bool));self.assertFalse(bool(missing.seeded.any()))
  fresh=life._tape(t.cursor,self.producer,64);fresh=ex.begin_on_tape(fresh,ledgers[0],self.runtime['execution_compiled'].device,active=j.ones(8,bool));rt=self.runtime['kernel_risk_compiled'].device;fresh=fresh._replace(execution=fresh.execution._replace(risks=risk.Context(risk.begin_frame_device(t.execution.risks.carry,fresh,rt,active=j.ones(8,bool)),rt),leech_history=history))
  later,ft,dc,hc=self.residual(after,source,target,fresh);self.assertEqual(cb.captured_matches(ft,hc).tolist(),[True]*8)
  nohistory=fresh._replace(execution=fresh.execution._replace(leech_history=None));_,bad,_,_=self.residual(after,source,target,nohistory);self.assertTrue(bool((bad.refusal&256!=0).all()))
if __name__=='__main__':unittest.main()
