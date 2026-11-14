"""Literal Protect setter/interception and original both FULL consumers; no games."""
from copy import deepcopy
from dataclasses import replace
import ast,json,subprocess,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_protect_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers,engine_thunder_wave_callbacks as thunder
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,source_provenance
from gpu_engine.effects import stall_success
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from test_gpu_engine_gameplay_stream_comparison import state as literal_state
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixtures

class ProtectOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.NoDamageOwningTests.setUpClass();cls.f=fixtures.NoDamageOwningTests;cls.metadata=deepcopy(cls.f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(cls.f.dex.moves);cls.names=('protect','detect','surf','thunderwave')
  labels=set(cls.f.catalog.vocabulary.tokens)
  for name in cls.names+('endure',):
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   at=cls.metadata['vocabulary']['moves'].index(name)
   cls.metadata['moves'][at]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=True if name in ('protect','detect','endure')else int(info.accuracy),priority=info.priority,flags={'protect':1}if name in ('surf','thunderwave')else{})
   labels|={'move:'+name,'belief:possible_move:'+name}
  cls.dex=ShowdownDex(moves,cls.f.dex.species,cls.f.dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('protect-'+name,'authored',name,'authored',80,cls.names,'Pressure','Leftovers'),))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-protect-source'),variants)
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer)
  cls.facts=own.compile_own_facts(cls.req,cls.catalog);cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.thunder=thunder.compile_thunder_wave(engine_catalog=cls.producer,metadata=cls.metadata)
  cls.prepared=driver.prepare(cls.req,cls.facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',thunder_wave_compiled=cls.thunder,thunder_wave_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))
  cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
 def test_original_source_default_and_compiler_capability(self):
  from gpu_engine import engine_source_frame_witness as witness
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),89)
  for name in self.names[:2]:
   at=self.metadata['vocabulary']['moves'].index(name);self.assertTrue(bool(self.helper.device.protect.candidates[at]));self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[at]),0)
  self.assertFalse(bool(self.helper.device.protect.candidates[self.metadata['vocabulary']['moves'].index('endure')]))
  self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.protect)
  old=ast.parse(subprocess.check_output(['git','show','0e1dc270:gpu_engine/transition.py'],text=True));after=ast.parse(subprocess.check_output(['git','show','5e2a0376:gpu_engine/transition.py'],text=True))
  baseline=dump_default_program(old);self.assertEqual(baseline,dump_default_program(after));assert_hp_rng_mutations_detected(self,after,baseline)
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  script="""const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),out=[];
for(const id of ['protect','detect']){const m=d.moves.get(id),c=d.conditions.get(m.volatileStatus),lines=[],p={toString:()=> 'p2a: Actor'};
const b={add:(...v)=>lines.push(v.map(String).join('|')),checkMoveBypassesProtect:()=>false};c.onStart.call(b,p);c.onTryHit.call(b,p,{getVolatile:()=>null},d.moves.get('surf'));
out.push({id,lines,queue:m.onPrepareHit.call({queue:{willAct:()=>false},runEvent:()=>{throw Error('queue shortcircuit')}},p)});}process.stdout.write(JSON.stringify(out));"""
  for value in json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True)):
   self.assertEqual(value['lines'],['-singleturn|p2a: Actor|Protect','-activate|p2a: Actor|Protect']);self.assertFalse(value['queue'])
  for name in self.names[:2]:
   pid=self.catalog.moves.names.index(name);self.assertTrue(bool(self.runtime['public'].device.protect_still_moves[pid]))
 def test_original_flags_and_exact_self_target_cannot_be_forged(self):
  protector=self.metadata['vocabulary']['moves'].index('protect');targets=list(self.producer.targets);targets[protector]=1
  changed=cb.compile_data(replace(self.producer,targets=tuple(targets)),self.metadata,self.rules)
  self.assertFalse(bool(changed.candidates[protector]))
  # An installed protect flag cannot invent an original Protect interception
  # on an original non-protectable row, even with a spoofed normal target.
  weather=self.metadata['vocabulary']['moves'].index('raindance');targets=list(self.producer.targets);targets[weather]=1
  effects=self.table.effects._replace(protectable=self.table.effects.protectable.at[weather].set(True))
  changed=cb.compile_data(replace(self.producer,targets=tuple(targets),engine_table=self.table._replace(effects=effects)),self.metadata,self.rules)
  self.assertFalse(bool(changed.block_candidates[weather]))
  with self.assertRaises(ValueError):direct.compile_direct_hit_controls(engine_catalog=replace(self.producer),metadata=self.metadata,source_rules=self.rules)
 def test_original_attr_last_move_erases_failed_protect_target(self):
  # Invoke the pinned original logger, not a reimplementation of its strings.
  script="""const {Battle}=require(process.argv[1]+'/dist/sim/battle.js');const out=[];
for(const name of ['Protect','Detect'])for(const side of ['p1','p2']){
const b={lastMoveLine:0,log:['|move|'+side+'a: Actor|'+name+'|'+side+'a: Actor']};
Battle.prototype.attrLastMove.call(b,'[still]');out.push(b.log[0]);}
process.stdout.write(JSON.stringify(out));"""
  original=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True))
  self.assertEqual(original,[f'|move|{side}a: Actor|{name}||[still]'for name in ('Protect','Detect')for side in ('p1','p2')])

 def component(self):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);zero=j.zeros(b,j.int32);actor=rows%2
  state=jax.tree.map(j.asarray,literal_state(b));active=j.stack((zero+3,zero+4),axis=1)
  state=state._replace(active=active,species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),ability=j.zeros_like(state.ability),base_ability=j.zeros_like(state.base_ability),gender=j.ones_like(state.gender))
  cursor=life.register(rows+41001,**self.kw);initial=life._tape(cursor,self.producer,64)
  for side in (0,1):initial=life._reveal(initial,state,zero+side,zero-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes)
  frame=life.LifecycleResult(state,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,zero)
  consumers=[]
  for owner in (0,1):
   stream=full.register(cursor.stream_id,zero+owner,compiled=self.runtime['public']);p=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=zero+owner)
   result=self.fold(stream,frame,p);self.assertEqual(result.projection.error.tolist(),[0]*b);consumers.append((result.projection.state,result.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];r=self.runtime['kernel_risk_compiled']
  tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes)
  carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,r.device),tape,r.device,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,r.device)))
  move=zero+self.metadata['vocabulary']['moves'].index('protect');physical=active[rows,actor]
  tape=public.move(tape,actor,physical,active[rows,1-actor],move,eligible=yes)
  counts=j.array([0,0,2,8,0,0,0,0],j.int32);roll=j.array([0,0,7,7,0,0,0,0],j.int32);foe_move=j.array([True]*4+[False]+[True]*3)
  eligible=j.array([True]*7+[False]);first=yes.at[5].set(False);not_pending=yes.at[6].set(False)
  tape,pending=cb.begin(tape,state,actor,move,counts,roll,first,not_pending,foe_move,self.runtime['direct_hit_context'],self.table.effects,eligible=eligible)
  stall=eligible&first&not_pending&foe_move&stall_success(counts,roll)
  state=state._replace(protect_counter=state.protect_counter.at[rows,actor,physical].set(j.where(stall,j.minimum(8,j.maximum(2,counts*2)),0)))
  tape=cb.finish(tape,pending,state,stall,stall,eligible)
  block_actor=1-actor;attack=zero+self.metadata['vocabulary']['moves'].index('surf')
  tape=public.move(tape,block_actor,active[rows,block_actor],physical,attack,eligible=yes)
  tape,p=cb.begin_block(tape,state,block_actor,attack,yes,stall,yes,self.runtime['direct_hit_context'],eligible=yes)
  tape,captured=cb.finish_block(tape,p,~stall)
  return state,initial.cursor,consumers,tape,captured,actor
 def test_mixed_setter_queue_stall_failure_current_cursor_sticky(self):
  state,cursor,consumers,tape,captured,actor=self.component()
  self.assertEqual(cb.owned_attempt(tape,1).tolist(),[True]*7+[False]);self.assertEqual(cb.captured_block_matches(tape,captured).tolist(),[True,True]+[False]*6)
  self.assertEqual(tape.words[:,1,8].tolist(),[2,2,0,0,0,0,0,0]);self.assertEqual(tape.execution.ledger.count.tolist(),[2,2,1,1,1,1,1,0])
  for name in ('turn','phase','generation'):
   value=getattr(tape.cursor,name);bad=tape._replace(cursor=tape.cursor._replace(**{name:value+1}));self.assertFalse(bool(cb.captured_block_matches(bad,captured).any()),name)
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=tape._replace(**{name:getattr(tape,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.owned_attempt(bad,1).any()),name)
  # Exact OLD lexical actor and target public encounters for BOTH owner views.
  one=j.array([True]*7+[False]);tape=tape._replace(count=j.where(one,tape.count,0))
  frame=life.LifecycleResult(state,tape.cursor,tape,life.snapshot(cursor),life.snapshot(tape.cursor),j.zeros(8,bool),j.zeros(8,j.int32))
  for owner,(stream,pub)in enumerate(consumers):
   result=self.fold(stream,frame,pub)
   for row in range(7):
    self.assertEqual(int(result.projection.error[row]),0)
    names={side:self.catalog.species.names[side+1]for side in (0,1)}
    ids={side:f'p{side+1}a: '+names[side]for side in (0,1)}
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[side]}|{names[side]}, L80, M|100/100'for side in (0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog))
    protector=row%2;attacker=1-protector
    success=row<2
    lines=[f'|move|{ids[protector]}|Protect|'+(ids[protector]if success else '|[still]'),f'|-singleturn|{ids[protector]}|Protect'if success else f'|-fail|{ids[protector]}',f'|move|{ids[attacker]}|Surf|{ids[protector]}']
    if success:lines.append(f'|-activate|{ids[protector]}|Protect')
    _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog))
    wanted=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(result.projection.frames.words[:,row])[np.asarray(result.projection.frames.valid[:,row])]
    np.testing.assert_array_equal(actual,wanted)

 def test_actual_success_queue_and_stall_fail_block_both_full_witness_encoder_physics(self):
  arrays=self.f.f.base().arrays('pressure','pressure');b=3
  arrays['moves'][:]=np.array([self.metadata['vocabulary']['moves'].index(n)for n in self.names]);arrays['pp'][:]=[16,8,24,32];arrays['item'][:]=0
  arrays['stats'][:,:,0]=300;arrays['stats'][:,:,3]=50;arrays['stats'][:,:,4]=200
  arrays['stats'][0,:,5]=150;arrays['stats'][1,:,5]=220
  with TemporaryDirectory()as tmp:
   p=Path(tmp)/'literal.npz';np.savez_compressed(p,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)));pool=source_provenance.load_constructor_pool(p)
   session=driver.construct(pool,np.tile([[0,1]],(b,1)).astype(np.int64),prepared=self.prepared,stream_ids=j.arange(42001,42001+b,dtype=j.int32),seed=31)
  self.assertEqual(session.state.core.admitted.tolist(),[True]*b)
  frames=[];original=self.runtime['frame']
  def capture(*a,**kw):frame=original(*a,**kw);frames.append(frame);return frame
  self.runtime['frame']=capture
  plain=jax.jit(lambda state,actions,chance:tr.step_with_chance(state,actions,self.table,chance).state)
  try:
   for step,actions in enumerate((j.array([[0,2],[0,4],[1,3]],j.int32),j.array([[0,2],[0,2],[1,3]],j.int32))):
    before=session.state;draws=j.zeros((b,2,12),j.int32).at[:,:,4].set(1).at[:,:,8].set(0 if step==0 else 7)
    chance=tr.ChanceInputs(draws,*[j.zeros((b,2),j.int32)for _ in range(4)],j.zeros(b,j.int32))
    result=driver.advance(session,actions,chance_inputs=chance);frame=frames[-1]
    for a,z in zip(jax.tree.leaves(frame.state),jax.tree.leaves(plain(before.core.engine,actions,chance)),strict=True):np.testing.assert_array_equal(a,z)
    self.assertEqual(result.state.core.admitted.tolist(),[True]*b,msg=json.dumps(dict(step=step,raw=frame.callbacks.words.tolist(),count=frame.callbacks.count.tolist(),source=frame.callbacks.refusal.tolist(),risk=result.state.core.risks.risk.tolist(),ledger=result.state.core.execution.refusal.tolist(),witness=result.state.core.frame_input_report.refusal.tolist(),expected=result.state.core.frame_input_report.expected_cutpoints.tolist(),actual=result.state.core.frame_input_report.observed_cutpoints.tolist(),owners=[o.reason.tolist()for o in result.owners])))
    for owner in result.owners:
     self.assertEqual(owner.full_line_consumers_complete.tolist(),[True]*b);self.assertEqual(owner.encoded.observation.admitted.tolist(),[True]*b);self.assertFalse(bool(owner.B4_qualified.any()))
    raw=frame.callbacks.words
    for row in range(b):
     words=np.asarray(raw[row,:int(frame.callbacks.count[row])]);attempt=words[words[:,0]==cb.PROTECT_ATTEMPT];self.assertEqual(len(attempt),1)
     success=(step==0 and row!=1)or(step==1 and row==1)
     self.assertEqual(bool(attempt[0,6]&cb.SUCCESS),success);self.assertEqual(int(attempt[0,8]),2 if success else 0)
     self.assertEqual(int(np.sum(words[:,0]==cb.PROTECT_BLOCK)),int(success))
    ledger=result.state.core.execution;used=(ledger.site==int(ex.Site.PROTECT_ATTEMPT))|(ledger.site==int(ex.Site.PROTECT_BLOCK))
    self.assertTrue(bool(j.all(j.where(used,ledger.stage==3,True))))
  finally:self.runtime['frame']=original


if __name__=='__main__':unittest.main()
