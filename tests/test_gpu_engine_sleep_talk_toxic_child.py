"""Original sleep/selection helpers and typed owning packets; no engine turn."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_sleep_talk_callbacks as cb,engine_execution_completeness as ex
from gpu_engine import engine_kernel_frame_risk as risk,engine_lifecycle_callbacks as life
from gpu_engine import engine_full_v4_event_device as full,mechanics as m,sleep_talk as talking,state,public_wire,v4_encoder as v4
from gpu_engine import engine_source_frame_witness as witness
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_hidden_power_substitute_input as fixtures
BASE='3a1655ad73da6f61fa5cbe5961c40600587162dc'
from gpu_engine import engine_toxic_callbacks as toxic,engine_status_callbacks as status

class SleepTalkToxicChildTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from pokezero.dex import ShowdownDex,load_showdown_dex_cached
  from pokezero.category_vocab import CategoryVocabulary
  from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
  from gpu_engine import transition as tr,public_belief,engine_public_events as public,engine_request_source as request,engine_own_facts as own,engine_direct_hit_controls as direct,engine_retained_encoder_driver as driver
  fixtures.HiddenPowerSubstituteInputTests.setUpClass();f=fixtures.HiddenPowerSubstituteInputTests;cls.f=f;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);names=('sleeptalk','toxic','rest','wish')
  flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>d.moves.get(n).flags)))",str(showdown_root()),json.dumps(names)],text=True))
  for name,flag in zip(names,flags):
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=85 if name=='toxic'else True,priority=info.priority,flags=flag)
  cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('talk-'+name,'unit',name,'unit',80,names,'Pressure','Leftovers'),Gen3RandbatVariant('talk-ground-'+name,'unit',name,'unit',80,('sleeptalk','earthquake','rest','wish'),'Pressure','Leftovers'),))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-sleep-talk-component'),variants)
  labels=set(f.catalog.vocabulary.tokens)
  for n in names:labels|={'move:'+n,'belief:possible_move:'+n}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  from gpu_engine import engine_leftovers_callbacks as leftovers,engine_toxic_callbacks as toxic
  cls.toxic=toxic.compile_toxic(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.legacy_toxic=toxic.compile_toxic(engine_catalog=cls.producer,metadata=cls.metadata)
  cls.prepared=driver.prepare(req,facts,leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',toxic_compiled=cls.toxic,toxic_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared)
  cls.fold=staticmethod(jax.jit(lambda st,fr,p:full.project_and_fold(st,fr,cls.runtime['public'].device,p,cls.catalog)))

 def component(self,*,mixed=False,initial_public=False,chosen='toxic',poison=False,outcomes=False,caller_slot=0,actor_hp=None):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);side=rows%2;active=j.stack((z+3,z+4),axis=1);phys=active[rows,side]
  s=state.empty_state(b)._replace(done=j.zeros(b,bool),active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32))
  if actor_hp is not None:s=s._replace(hp=s.hp.at[rows,side,phys].set(actor_hp))
  ids=self.metadata['vocabulary']['moves'];caller=z+ids.index('sleeptalk');moves=j.broadcast_to(j.array([ids.index(n)for n in('sleeptalk','toxic','rest','wish')],j.int32),(b,4));counter=z+3;pp=j.full((b,4),8,j.int32);eligible=yes;slots=z+caller_slot
  if caller_slot==3:moves=j.roll(moves,-1,axis=1)
  if outcomes:moves=moves.at[4:6,1].set(ids.index('earthquake'))
  if chosen=='struggle':moves=moves.at[:,2].set(ids.index('struggle'))
  if mixed:counter=counter.at[2].set(1);pp=pp.at[4,0].set(0).at[5,1].set(0);eligible=eligible.at[6].set(False)
  s=s._replace(status=s.status.at[rows,side,phys].set(m.SLP),status_counter=s.status_counter.at[rows,side,phys].set(counter),moves=s.moves.at[rows,side,phys].set(moves),pp=s.pp.at[rows,side,phys].set(pp),rest_origin=s.rest_origin.at[1,side[1],phys[1]].set(True))
  if mixed:s=s._replace(confusion_counter=s.confusion_counter.at[3,side[3],phys[3]].set(1))
  kw=dict(catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));start=life.register(rows+92001,**kw);initial=life._tape(start,self.producer,64)
  for seat in(0,1):initial=life._reveal(initial,s,z+seat,z-1,active[:,seat],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes)
  consumers=[]
  if initial_public:
   frame0=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(start),life.snapshot(initial.cursor),~yes,z)
   for owner in(0,1):
    stream=full.register(start.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=start.stream_id,owner_sides=z+owner);got=self.fold(stream,frame0,pub);self.assertEqual(got.projection.error.tolist(),[0]*b);consumers.append((got.projection.state,got.public))
  t=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;t=ex.begin_on_tape(t,ex.initialize(start.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,rt),t,rt,active=yes);t=t._replace(execution=t.execution._replace(risks=risk.Context(carry,rt)))
  if poison:t=t._replace(refusal=t.refusal|64)
  roll=z+(1 if chosen=='rest'else 0)
  table=self.producer.engine_table.sleep_talk;t,p=cb.begin(t,s,side,caller,slots,self.runtime['direct_hit_context'],table,choice_roll=roll,eligible=eligible,sleep_usable=yes,early_bird=~yes,loafing=~yes)
  before=m.before_move(s.status[rows,side,phys],counter,s.confusion_counter[rows,side,phys],sleep_usable=yes)
  after=talking.before_move(s,side,before.status,eligible=eligible,sleep_usable=yes)
  for name,value in(('status',before.status),('status_counter',before.counter),('confusion_counter',before.confusion_counter)):
   old=getattr(after,name);after=after._replace(**{name:old.at[rows,side,phys].set(j.where(eligible,value,old[rows,side,phys]))})
  caller_acts=eligible&before.can_act&(pp[rows,slots]>0);after=after._replace(pp=after.pp.at[rows,side,phys,slots].set(j.where(caller_acts,j.maximum(0,pp[rows,slots]-1),pp[rows,slots])))
  called,usable=talking.choose(table,after.moves[rows,side,phys],after.pp[rows,side,phys],roll);acts=caller_acts&usable&(before.status==m.SLP)
  t,c=cb.finish(t,p,after,before,called,usable,roll,acts=acts,caller_acts=caller_acts)
  return s,after,before,side,t,c,consumers,initial.cursor

 def append_toxic(self,after,t,c,*,unsupported=False):
  b=8;rows=j.arange(b);side=c.pending.side;phys=c.pending.physical;target=c.pending.target;move=c.called;yes=j.ones(b,bool);z=j.zeros(b,j.int32)
  immune=rows<2;miss=(rows>=2)&(rows<4);applied=rows>=4
  if unsupported:immune=~yes;miss=~yes;applied=~yes
  attempt=toxic.Attempt(~miss,~miss,immune,yes if unsupported else ~yes,~yes,yes,applied,~yes,~yes,~yes,~miss,z,j.where(applied,m.TOX,0),z,z)
  t,ticket=ex.open_attempt(t,ex.Site.TOXIC_ATTEMPT,1-side,active=yes)
  t=status.changed(t,1-side,target,attempt.old_status,attempt.new_status,status.PRIMARY,eligible=applied)
  t=toxic.record_device(t,side,phys,target,move,attempt,self.runtime['toxic_context'],eligible=yes,caller_move=c.pending.caller)
  t=ex.capture_toxic_attempt(t,ticket,side,phys,target,move,attempt,requested=yes)
  captured=risk.CapturedToxic(self.runtime['toxic_context'],ticket,side,phys,target,move,attempt)
  t=risk.primary_status_on_tape(t,yes,active=yes,captured=captured)
  after=after._replace(status=after.status.at[rows,1-side,target].set(attempt.new_status))
  return after,t,attempt,ticket

 def test_original_selection_source_same_rules_and_default_numerics(self):
  self.assertTrue(bool(self.runtime['kernel_risk_compiled'].device.sleep_talk_toxic_child))
  legacy=risk.compile_risks(self.producer,self.runtime['execution_compiled'],direct_hit_compiled=self.helper,toxic_compiled=self.legacy_toxic)
  self.assertFalse(bool(legacy.device.sleep_talk_toxic_child))
  # Equal metadata with a different original rules record is not SAME rules.
  from gpu_engine import engine_request_source as request
  other=request.load_source_rules(showdown_root(),self.metadata)
  other_toxic=toxic.compile_toxic(engine_catalog=self.producer,metadata=self.metadata,source_rules=other)
  separate=risk.compile_risks(self.producer,self.runtime['execution_compiled'],direct_hit_compiled=self.helper,toxic_compiled=other_toxic)
  self.assertFalse(bool(separate.device.sleep_talk_toxic_child))
  for name in('sleep_talk.py','engine_toxic_callbacks.py','engine_execution_completeness.py','engine_lifecycle_event_device.py'):
   self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+name]))
  script=r'''const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),lines=[],p={statusState:{time:3},hasAbility:()=>false,toString:()=> 'p2a: Actor',moveSlots:[{id:'sleeptalk',pp:8},{id:'toxic',pp:8},{id:'rest',pp:8}]};const b={effectState:{skippedTime:0},add:(...a)=>lines.push('|'+a.map(String).join('|')),sample:xs=>xs[0],dex:d,actions:{useMove:id=>lines.push('|move|p2a: Actor|'+d.moves.get(id).name+'|p1a: Target|[from] '+d.conditions.get('sleeptalk').name)}};d.conditions.get('slp').onBeforeMove.call(b,p,null,d.moves.get('sleeptalk'));lines.push('|move|p2a: Actor|Sleep Talk|p2a: Actor');d.moves.get('sleeptalk').onHit.call(b,p);console.log(JSON.stringify({time:p.statusState.time,skipped:b.effectState.skippedTime,lines}));'''
  actual=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(actual,dict(time=2,skipped=1,lines=['|cant|p2a: Actor|slp','|move|p2a: Actor|Sleep Talk|p2a: Actor','|move|p2a: Actor|Toxic|p1a: Target|[from] sleeptalk']))
  witness._source()

 def test_current_pending_join_and_child_site_stays_mandatory(self):
  s,after,before,side,t,c,_,_=self.component();self.assertTrue(bool(cb.captured_matches(t,c).all()));self.assertEqual(t.count.tolist(),[3]*8)
  # A true declaration alone never invents the child outcome.
  self.assertFalse(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).any()))
  original_t=t
  after,t,attempt,ticket=self.append_toxic(after,t,c);self.assertEqual(t.execution.ledger.refusal.tolist(),[0]*8);self.assertEqual(t.execution.risks.carry.risk.tolist(),[0]*8)
  self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  _,bad,_,_=self.append_toxic(after,original_t,c,unsupported=True)
  self.assertTrue(bool((bad.execution.risks.carry.risk&risk.FIELD_VOLATILE).all()));self.assertTrue(bool((bad.execution.ledger.refusal!=0).all()));self.assertFalse(bool(witness._raw_complete(bad,self.runtime['frame_witness_compiled'].device).any()))
  rt=t.execution.risks.tables
  closed=original_t._replace(execution=original_t.execution._replace(risks=original_t.execution.risks._replace(tables=rt._replace(sleep_talk_toxic_child=j.array(False)))))
  self.assertFalse(bool(cb.owned(closed,0).any()))
  with self.assertRaises(ValueError):cb.owned(original_t._replace(execution=original_t.execution._replace(risks=original_t.execution.risks._replace(tables=rt._replace(sleep_talk_toxic_child=j.ones(8,bool))))),0)
  for name in('engine_binding','source_binding','move_binding','producer_binding'):
   with self.assertRaises(ValueError):cb.owned(original_t._replace(**{name:j.broadcast_to(getattr(original_t,name),(8,8))}),0)
  # Coherent transport cannot replace the actual selected slot or original draw.
  for variant in('slot','draw'):
   col=6 if variant=='slot'else 8;delta=2 if variant=='slot'else 12
   words=original_t.words.at[0,1,col].add(delta);ledger=original_t.execution.ledger._replace(words=original_t.execution.ledger.words.at[0,1,col].add(delta));changed=original_t._replace(words=words,execution=original_t.execution._replace(ledger=ledger));cc=c._replace(raw=c.raw.at[0,1,col].add(delta),**({'selected_slot':c.selected_slot.at[0].add(2)}if variant=='slot'else{'roll':c.roll.at[0].add(3)}))
   self.assertEqual(cb.captured_matches(changed,cc).tolist(),[False]+[True]*7)
  for chosen in('struggle',):
   _,_,_,_,closed,cc,_,_=self.component(chosen=chosen);self.assertFalse(bool(cb.captured_matches(closed,cc).any()));self.assertTrue(bool((closed.execution.ledger.refusal!=0).all()))
  _,_,_,_,poison,pc,_,_=self.component(poison=True);self.assertFalse(bool(cb.captured_matches(poison,pc).any()));self.assertTrue(bool((poison.refusal&64).all()))

 def test_nonzero_caller_slot_same_actual_pp_and_selection(self):
  s,after,before,side,t,c,_,_=self.component(caller_slot=3);self.assertEqual(c.pending.slot.tolist(),[3]*8);self.assertEqual(c.selected_slot.tolist(),[0]*8)
  self.assertTrue(bool(cb.captured_matches(t,c).all()));self.assertEqual(c.raw[:,1,8:10].tolist(),[[3,7]]*8)
  words=t.words.at[0,1,9].add(1);ledger=t.execution.ledger._replace(words=t.execution.ledger.words.at[0,1,9].add(1));bad=t._replace(words=words,execution=t.execution._replace(ledger=ledger));changed=c._replace(raw=c.raw.at[0,1,9].add(1))
  self.assertEqual(cb.captured_matches(bad,changed).tolist(),[False]+[True]*7)
  after,t,attempt,ticket=self.append_toxic(after,t,c);self.assertEqual(t.execution.ledger.refusal.tolist(),[0]*8);self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))

 def test_both_full_original_toxic_child_immune_miss_write(self):
  s,after,before,side,t,c,consumers,cursor=self.component(initial_public=True)
  after,t,attempt,ticket=self.append_toxic(after,t,c);self.assertEqual(t.execution.ledger.refusal.tolist(),[0]*8);self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  yes=j.ones(8,bool);t=life.action_end(t,yes);t=life.finish(t,after,live=yes,replacement=~yes,pending_before=~yes,residual_ready=yes,defer_residual=~yes,turn_context=yes)
  self.assertEqual(t.cursor.turn.tolist(),[2]*8);self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(cursor),life.snapshot(t.cursor),~yes,j.zeros(8,j.int32));saved=[];all_words=[]
  for owner,(st,pub)in enumerate(consumers):
   got=self.fold(st,frame,pub);rows=[]
   for row in range(8):rows.append(np.asarray(got.projection.frames.words)[:,row].reshape(-1,16)[np.asarray(got.projection.frames.valid)[:,row].reshape(-1)].tolist())
   saved.append(dict(owner=owner,error=got.projection.error.tolist(),admitted=got.projection.component_admitted.tolist(),words=rows,raw=t.words.tolist(),counts=t.count.tolist(),ledger_refusal=t.execution.ledger.refusal.tolist()))
   Path('/tmp/pokezero-sleep-talk-toxic-child-full-current.json').write_text(json.dumps(saved)+'\n')
   self.assertEqual(got.projection.error.tolist(),[0]*8);self.assertTrue(bool(got.projection.component_admitted.all()))
   for row in range(8):
    actor=int(side[row]);other=1-actor;name=self.catalog.species.names[actor+1];foename=self.catalog.species.names[other+1];ident=f'p{actor+1}a: {name}';foe=f'p{other+1}a: {foename}'
    switch=[f'|switch|p1a: {self.catalog.species.names[1]}|{self.catalog.species.names[1]}, L80, M|100/100'+(' slp'if actor==0 else ''),f'|switch|p2a: {self.catalog.species.names[2]}|{self.catalog.species.names[2]}, L80, M|100/100'+(' slp'if actor==1 else '')]
    wire,_=public_wire.project_lines(public_wire.WireState(),switch+['|turn|1'],**v4.wire_kwargs(self.catalog));suffix='|[miss]'if row in(2,3)else''
    lines=[f'|cant|{ident}|slp',f'|move|{ident}|Sleep Talk|{ident}',f'|move|{ident}|Toxic|{foe}|[from] sleeptalk'+suffix]
    lines.append(f'|-immune|{foe}'if row<2 else f'|-miss|{ident}|{foe}'if row<4 else f'|-status|{foe}|tox');lines+=['|','|upkeep','|turn|2']
    _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));want=np.asarray([event.words()for packet in packets for event in packet.events],np.int32);want[:,1]+=int(st.sequence[row])-int(want[0,1])+1
    np.testing.assert_array_equal(np.asarray(rows[row],np.int32),want);all_words.append(rows[row])
  Path('/tmp/pokezero-sleep-talk-toxic-child-both-full-compact.json').write_text(json.dumps(dict(words=all_words,scope='Synthetic source helper components, no actual games/turns'))+'\n')

if __name__=='__main__':unittest.main()
