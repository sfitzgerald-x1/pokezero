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
BASE='e7b736c6b2a13900dd17f68413e0f408b3b0f87e'

class SleepTalkOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from pokezero.dex import ShowdownDex,load_showdown_dex_cached
  from pokezero.category_vocab import CategoryVocabulary
  from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
  from gpu_engine import transition as tr,public_belief,engine_public_events as public,engine_request_source as request,engine_own_facts as own,engine_direct_hit_controls as direct,engine_retained_encoder_driver as driver
  fixtures.HiddenPowerSubstituteInputTests.setUpClass();f=fixtures.HiddenPowerSubstituteInputTests;cls.f=f;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);names=('sleeptalk','surf','rest','wish')
  flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>d.moves.get(n).flags)))",str(showdown_root()),json.dumps(names)],text=True))
  for name,flag in zip(names,flags):
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=True if name!='surf'else 100,priority=info.priority,flags=flag)
  cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('talk-'+name,'unit',name,'unit',80,names,'Pressure','Leftovers'),Gen3RandbatVariant('talk-ground-'+name,'unit',name,'unit',80,('sleeptalk','earthquake','rest','wish'),'Pressure','Leftovers'),))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-sleep-talk-component'),variants)
  labels=set(f.catalog.vocabulary.tokens)
  for n in names:labels|={'move:'+n,'belief:possible_move:'+n}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  from gpu_engine import engine_leftovers_callbacks as leftovers
  cls.prepared=driver.prepare(req,facts,leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared)
  cls.fold=staticmethod(jax.jit(lambda st,fr,p:full.project_and_fold(st,fr,cls.runtime['public'].device,p,cls.catalog)))

 def component(self,*,mixed=False,initial_public=False,chosen='surf',poison=False,outcomes=False):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);side=rows%2;active=j.stack((z+3,z+4),axis=1);phys=active[rows,side]
  s=state.empty_state(b)._replace(active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32))
  ids=self.metadata['vocabulary']['moves'];caller=z+ids.index('sleeptalk');moves=j.broadcast_to(j.array([ids.index(n)for n in('sleeptalk','surf','rest','wish')],j.int32),(b,4));counter=z+3;pp=j.full((b,4),8,j.int32);eligible=yes
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
  table=self.producer.engine_table.sleep_talk;t,p=cb.begin(t,s,side,caller,z,self.runtime['direct_hit_context'],table,eligible=eligible,sleep_usable=yes,early_bird=~yes,loafing=~yes)
  before=m.before_move(s.status[rows,side,phys],counter,s.confusion_counter[rows,side,phys],sleep_usable=yes)
  after=talking.before_move(s,side,before.status,eligible=eligible,sleep_usable=yes)
  for name,value in(('status',before.status),('status_counter',before.counter),('confusion_counter',before.confusion_counter)):
   old=getattr(after,name);after=after._replace(**{name:old.at[rows,side,phys].set(j.where(eligible,value,old[rows,side,phys]))})
  caller_acts=eligible&before.can_act&(pp[:,0]>0);after=after._replace(pp=after.pp.at[rows,side,phys,0].set(j.where(caller_acts,j.maximum(0,pp[:,0]-1),pp[:,0])))
  roll=z+(1 if chosen=='rest'else 0);called,usable=talking.choose(table,after.moves[rows,side,phys],after.pp[rows,side,phys],roll);acts=caller_acts&usable&(before.status==m.SLP)
  t,c=cb.finish(t,p,after,before,called,usable,roll,acts=acts,caller_acts=caller_acts)
  if outcomes:
   from gpu_engine import engine_no_damage_callbacks as nd,engine_direct_hit_controls as hit,engine_public_events as public
   target=after.active[rows,1-side];nd_rows=(rows>=2)&(rows<6);miss=(rows>=2)&(rows<4);immune=(rows>=4)&(rows<6)
   t,ticket=ex.open_attempt(t,ex.Site.DIRECT_HIT,1-side,active=yes)
   typ=j.where(immune,ids.index('earthquake')*0+self.metadata['vocabulary']['types'].index('ground'),self.metadata['vocabulary']['types'].index('water')).astype(j.int32)
   pre=nd.Before(~miss,typ,z,z,z+100,z+100,j.zeros(b,j.uint32),z,z+100,~yes,z,z+70)
   pending=nd.prepare(t,ticket,side,phys,target,called,pre,self.runtime['direct_hit_context'],requested=yes,caller_move=caller)
   new_hp=j.where(nd_rows,100,60).astype(j.int32);write=hit.HitWrite(z+95,~yes,~miss,immune,j.where(immune,0,1).astype(j.int32),z+1,~yes,~yes,~yes,z+100,new_hp,z+100)
   t=hit.record_registered_direct_hit_controls_device(t,side,phys,target,called,write,self.runtime['direct_hit_context'],eligible=~nd_rows)
   t=public.condition(t,1-side,target,new_hp,z+100,z,eligible=~nd_rows)
   t=ex.capture_direct_hit_write(t,ticket,side,phys,target,called,write,requested=~nd_rows,data=self.runtime['direct_hit_context'].data)
   outcome=risk.Outcome(write.power,write.critical,write.hit,write.immune,write.numerator,write.denominator,~yes,~yes,~yes,~yes,z,~yes)
   t=nd.finish(t,pending,write,outcome,z+100,j.zeros(b,j.uint32),z,z,z+70)
   capture=nd.Captured(pending,write,z+100,j.zeros(b,j.uint32),z,z,z+70)
   t=risk.outcome_on_tape(t,outcome,eligible=yes,active=yes,captured_direct=risk.CapturedDirectHit(self.runtime['direct_hit_context'],ticket,side,phys,target,called,write),captured_no_damage=capture)
   self.assertEqual(t.execution.risks.carry.risk.tolist(),[0]*b)
   after=after._replace(hp=after.hp.at[rows,1-side,target].set(new_hp))
  return s,after,before,side,t,c,consumers,initial.cursor

 def test_original_source_default_numerics_and_called_writer_chronology(self):
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),85)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(old);self.assertEqual(dump_default_program(new),baseline);assert_hp_rng_mutations_detected(self,new,baseline)
  script=r'''const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),lines=[],p={statusState:{time:3},hasAbility:()=>false,toString:()=> 'p2a: Actor',moveSlots:[{id:'sleeptalk',pp:8},{id:'surf',pp:7},{id:'rest',pp:8}]};const b={effectState:{skippedTime:0},add:(...a)=>lines.push('|'+a.map(String).join('|')),sample:xs=>xs[0],dex:d,actions:{useMove:id=>lines.push('|move|p2a: Actor|'+d.moves.get(id).name+'|p1a: Target|[from] '+d.conditions.get('sleeptalk').name)}};d.conditions.get('slp').onBeforeMove.call(b,p,null,d.moves.get('sleeptalk'));lines.push('|move|p2a: Actor|Sleep Talk|p2a: Actor');d.moves.get('sleeptalk').onHit.call(b,p);console.log(JSON.stringify({time:p.statusState.time,skipped:b.effectState.skippedTime,lines}));'''
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(got,dict(time=2,skipped=1,lines=['|cant|p2a: Actor|slp','|move|p2a: Actor|Sleep Talk|p2a: Actor','|move|p2a: Actor|Surf|p1a: Target|[from] sleeptalk']))
  # Source-only import-order smoke; no constructor, sampler or model call.
  for code in ('from gpu_engine import benchmark_retained_games,engine_full_v4_event_device','from gpu_engine import engine_full_v4_event_device,benchmark_retained_games'):
   subprocess.check_call([__import__('sys').executable,'-c',code])

 def test_mixed_original_clock_selection_span_and_refusal_guards(self):
  s,after,before,side,t,c,_,_=self.component(mixed=True);expected=[True,True,False,False,False,False,False,True]
  self.assertEqual(cb.captured_matches(t,c).tolist(),expected)
  for i in range(3):self.assertEqual(cb.owned(t,i).tolist(),expected)
  self.assertEqual(t.count.tolist(),[3,3,0,0,0,0,0,3])
  self.assertEqual((t.execution.ledger.refusal!=0).tolist(),[False,False,False,True,True,True,False,False])
  for name in('engine_binding','source_binding','move_binding','producer_binding'):
   self.assertFalse(bool(cb.captured_matches(t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))}),c).any()))
   with self.assertRaises(ValueError):cb.owned(t._replace(**{name:j.broadcast_to(getattr(t,name),(8,8))}),0)
  for name in('turn','phase','generation'):
   self.assertFalse(bool(cb.captured_matches(t._replace(cursor=t.cursor._replace(**{name:getattr(t.cursor,name)+1})),c).any()))
  for col in(5,6,7,8,9):
   words=t.words.at[0,0,col].add(1);ledger=t.execution.ledger._replace(words=t.execution.ledger.words.at[0,0,col].add(1));bad=t._replace(words=words,execution=t.execution._replace(ledger=ledger));self.assertFalse(bool(cb.captured_matches(bad,c._replace(raw=c.raw.at[0,0,col].add(1)))[0]))
  trailing=t._replace(count=t.count+1,words=t.words.at[:,3].set(j.array([1,0,3,1,4,1,0,0,0,0],j.int32)));self.assertFalse(bool(cb.captured_matches(trailing,c).any()))

 def test_same_original_selection_slot_and_draw_cannot_be_replaced_coherently(self):
  _,_,_,_,t,c,_,_=self.component();self.assertTrue(bool(cb.captured_matches(t,c).all()))
  for variant in('slot','draw'):
   with self.subTest(variant=variant):
    column=6 if variant=='slot'else 8;delta=2 if variant=='slot'else 12
    words=t.words.at[0,1,column].add(delta);ledger=t.execution.ledger._replace(words=t.execution.ledger.words.at[0,1,column].add(delta));bad=t._replace(words=words,execution=t.execution._replace(ledger=ledger))
    changed=c._replace(raw=c.raw.at[0,1,column].add(delta),**({'selected_slot':c.selected_slot.at[0].add(2)}if variant=='slot'else{'roll':c.roll.at[0].add(3)}))
    self.assertTrue(bool(cb.owned(bad,0)[0]));self.assertFalse(bool(cb.captured_matches(bad,changed)[0]))

 def test_distinct_called_rest_source_pool_and_prior_unknown_remain_refused(self):
  for variant in('rest','struggle','prior'):
   with self.subTest(variant=variant):
    s,after,before,side,t,c,_,_=self.component(chosen=variant if variant!='prior'else'surf',poison=variant=='prior')
    self.assertFalse(bool(cb.captured_matches(t,c).any()));self.assertEqual(t.count.tolist(),[0]*8)
    self.assertTrue(bool((t.execution.ledger.refusal!=0).all()));self.assertTrue(bool((t.refusal&256).all()))
    if variant=='prior':self.assertTrue(bool((t.refusal&64).all()))
    np.testing.assert_array_equal(s.hp,after.hp)

 def test_both_full_exact_all_words_and_called_child_privacy(self):
  s,after,before,side,t,c,consumers,cursor=self.component(initial_public=True,outcomes=True);self.assertTrue(bool(cb.owned(t,0).all()));self.assertFalse(bool(cb.captured_matches(t,c).any()))
  Path('/tmp/pokezero-sleep-talk-typed-current.json').write_text(json.dumps(dict(words=t.words.tolist(),counts=t.count.tolist(),ledger_refusal=t.execution.ledger.refusal.tolist(),risk=t.execution.risks.carry.risk.tolist()))+'\n')
  self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  fr=life.LifecycleResult(after,t.cursor,t,life.snapshot(cursor),life.snapshot(t.cursor),j.zeros(8,bool),j.zeros(8,j.int32))
  all_words=[];folded=[]
  for st,pub in consumers:folded.append(self.fold(st,fr,pub))
  Path('/tmp/pokezero-sleep-talk-full-literal-current.json').write_text(json.dumps([dict(projection_error=g.projection.error.tolist(),words=g.projection.frames.words.tolist(),valid=g.projection.frames.valid.tolist(),component_admitted=g.projection.component_admitted.tolist(),public_needs_more=g.public.cursor.needs_more.tolist())for g in folded])+'\n')
  for owner,(st,pub) in enumerate(consumers):
   got=folded[owner];self.assertEqual(got.projection.error.tolist(),[0]*8);self.assertTrue(bool(got.projection.component_admitted.all()))
   for row in range(8):
    actor=int(side[row]);other=1-actor;name=self.catalog.species.names[actor+1];foename=self.catalog.species.names[other+1];ident=f'p{actor+1}a: {name}';foe=f'p{other+1}a: {foename}'
    wire=public_wire.WireState(turn=1);switch=[f'|switch|p1a: {self.catalog.species.names[1]}|{self.catalog.species.names[1]}, L80, M|100/100'+(' slp'if actor==0 else ''),f'|switch|p2a: {self.catalog.species.names[2]}|{self.catalog.species.names[2]}, L80, M|100/100'+(' slp'if actor==1 else '')]
    wire,_=public_wire.project_lines(public_wire.WireState(),switch+['|turn|1'],**v4.wire_kwargs(self.catalog))
    child='Earthquake'if row in(4,5)else'Surf';suffix='|[miss]'if row in(2,3)else''
    lines=[f'|cant|{ident}|slp',f'|move|{ident}|Sleep Talk|{ident}',f'|move|{ident}|{child}|{foe}|[from] sleeptalk'+suffix]
    lines.append(f'|-miss|{ident}|{foe}'if row in(2,3)else f'|-immune|{foe}'if row in(4,5)else f'|-damage|{foe}|60/100')
    wire,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog))
    expected=np.asarray([event.words()for packet in packets for event in packet.events],np.int32);expected[:,1]+=int(st.sequence[row])-int(expected[0,1])+1
    # Actual owner-independent declarations have exact original FULL words.
    actual=np.asarray(got.projection.frames.words)[:,row].reshape(-1,16);valid=np.asarray(got.projection.frames.valid)[:,row].reshape(-1);np.testing.assert_array_equal(actual[valid],expected)
    all_words.append(actual[valid].tolist())
  out=Path('/tmp/pokezero-sleep-talk-both-full-compact.json');out.write_text(json.dumps({'words':all_words,'scope':'synthetic original helper transaction; no engine turn/game'},sort_keys=True)+'\n')

if __name__=='__main__':unittest.main()
