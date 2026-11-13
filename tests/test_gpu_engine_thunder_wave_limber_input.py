"""Same-call Limber status interruption; typed helpers, no engine turns."""
from _component_evidence import capture_path
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_thunder_wave_callbacks as cb,engine_toxic_callbacks as primary
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_direct_hit_controls as direct,engine_leftovers_callbacks as leftovers
from gpu_engine import engine_substitute_callbacks as sub_callbacks
from gpu_engine import transition,public_belief,v4_encoder as v4,public_wire,engine_source_frame_witness as witness
from _showdown_root import showdown_root
import test_gpu_engine_thunder_wave_substitute_failure as fixture
import test_gpu_engine_thunder_wave_ground_immunity as ground_fixture
BASE='de4180168925e163cb87d3220d58975086f8e96f'
class ThunderLimberTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.ThunderWaveSubstituteTests.setUpClass.__func__(cls)
  cls.metadata=deepcopy(cls.metadata)
  if 'limber' not in cls.metadata['vocabulary']['abilities']:cls.metadata['vocabulary']['abilities'].append('limber')
  variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant('limber-'+name+str(i),'unit',name,'unit',80,cls.names,a,'Leftovers')for i,a in enumerate(('Pressure','Limber'))))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-original-thunder-limber'),variants)
  labels=set(cls.catalog.vocabulary.tokens)|{'ability:limber','belief:possible_ability:limber'}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.catalog.masks.__dict__)
  cls.table=transition.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules);cls.thunder=cb.compile_thunder_wave(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules);cls.legacy=cb.compile_thunder_wave(engine_catalog=cls.producer,metadata=cls.metadata)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',thunder_wave_compiled=cls.thunder,thunder_wave_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
 component=fixture.ThunderWaveSubstituteTests.component
 def literal(self,consume=False,**changed):
  s,t,pubs,actor,phys=self.component(consume);b=8;r=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);target=s.active[r,1-actor];data=self.thunder.device;g=data.ground_immunity
  ability=int(np.flatnonzero(np.asarray(g.abilities))[0]);self.assertGreater(ability,0);normal=self.metadata['vocabulary']['types'].index('normal');types=j.full((b,2,6,2),normal,j.int32)
  abi=j.full((b,2,6),ability,j.int32).at[r,1-actor,target].set(data.limber_ability)
  s=s._replace(done=~yes,turn=t.cursor.turn-1,types=types,base_types=types,ability=abi,base_ability=abi,moves=s.moves.at[r,actor,phys,0].set(self.move_id),pp=s.pp.at[r,actor,phys,0].set(5))
  t=life.begin(t,s,yes,~yes);move=z+self.move_id;t=public.move(t,actor,phys,target,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.PRIMARY_PARALYSIS,1-actor,active=yes)
  p=cb.begin_ground(t,s,ticket,actor,move,move,z,self.runtime['thunder_wave_context'],requested=yes,resolved_type=z+int(g.electric),power=z,accuracy=yes,hit=yes,immune=~yes,numerator=z+1,denominator=z+1,protected=~yes,absorbed=~yes)
  a=cb.Attempt(yes,yes,~yes,yes,~yes,yes,~yes,~yes,~yes,~yes,yes,z,z,z,z)
  if changed:
   kind=changed.pop('kind');values=changed
   if kind=='attempt':a=a._replace(**values)
   elif kind=='pending':p=p._replace(**values)
   else:raise AssertionError(kind)
  before=t;t=cb.record_device(t,ticket,actor,phys,target,move,a,self.runtime['thunder_wave_context'],requested=yes,ground_pending=p,state=s)
  return s,t,pubs,actor,phys,target,move,a,ticket,p,before
 def test_01_actual_source_default_and_current_joins(self):
  s,t,_,actor,phys,target,move,a,ticket,p,before=self.literal();yes=j.ones(8,bool);z=j.zeros(8,j.int32);index=t.count-1
  self.assertEqual(t.count.tolist(),[3]*8);self.assertEqual(cb.owned_limber_immunity(t,index).tolist(),[True]*8)
  self.assertEqual(t.words[:,2,6].tolist(),[1067]*8)
  cap=risk.CapturedThunderWave(self.runtime['thunder_wave_context'],ticket,actor,phys,target,move,a)
  self.assertEqual(risk._captured_thunder_wave_matches(t,self.runtime['kernel_risk_compiled'].device,cap).tolist(),[True]*8)
  for name in cb.Attempt._fields:
   value=getattr(a,name);bad=value.at[0].set(~value[0])if value.dtype==j.bool_ else value.at[0].add(1)
   out=self.literal(kind='attempt',**{name:bad})[1];self.assertEqual(out.count.tolist(),[2]+[3]*7,name)
  out=self.literal(kind='attempt',new_counter=a.new_counter.at[0].set(268435456))[1];self.assertEqual(out.count.tolist(),[2]+[3]*7)
  for name in ('caller','slot','physical','target','resolved_type','power','numerator','denominator'):
   out=self.literal(kind='pending',**{name:getattr(p,name).at[0].add(1)})[1];self.assertEqual(out.count.tolist(),[2]+[3]*7,name)
  capture=t.execution.thunder_wave_immunity[0]
  for name in cb.Private._fields:
   value=getattr(capture.after,name);bad=value.at[0].add(j.ones_like(value[0]));c=capture._replace(after=capture.after._replace(**{name:bad}));mutated=t._replace(execution=t.execution._replace(thunder_wave_immunity=(c,)));self.assertEqual(cb.owned_limber_immunity(mutated,index).tolist(),[False]+[True]*7,name)
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.owned_limber_immunity(bad,index).any()))
  closed=t._replace(execution=t.execution._replace(risks=t.execution.risks._replace(tables=t.execution.risks.tables._replace(thunder_wave_limber_ability=j.array(0,j.int32)))))
  self.assertFalse(bool(cb.owned_limber_immunity(closed,index).any()));self.assertEqual(int(self.legacy.device.limber_ability),0)
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  # entry.py, effects.py, secondary.py and state.py are byte-certified against the reviewed SOURCE_PINS by transition_observer_wiring.
  for f in ('mechanics.py','engine_retained_encoder_driver.py'):
   self.assertEqual(Path('gpu_engine',f).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+f]))
  witness._source()
  # Original current setStatus -> Limber callback -> runMoveEffects null
  # interruption emits immunity only, without fail/still or private writes.
  script=r'''const r=process.argv[1],d=require(r+'/dist/sim/dex.js').Dex.mod('gen3'),P=require(r+'/dist/sim/pokemon.js').Pokemon,A=require(r+'/dist/sim/battle-actions.js').BattleActions,move=d.moves.get('thunderwave'),lines=[],source={hp:100,hasAbility:()=>false,toString:()=> 'p1a: Actor'},battle={dex:d,effect:move,event:null,debug(){},add:(...xs)=>lines.push('|'+xs.map(String).join('|')),attrLastMove:x=>lines.push(x)},target={hp:100,status:'',pp:[5],item:'leftovers',battle,trySetStatus:P.prototype.trySetStatus,setStatus:P.prototype.setStatus,runStatusImmunity:()=>true,toString:()=> 'p2a: Target'};battle.runEvent=(name,t,src,effect,status)=>{if(name!=='SetStatus')throw Error(name);return d.abilities.get('limber').onSetStatus.call(battle,status,t,src,effect)};const actions={battle,combineResults:A.prototype.combineResults};const result=A.prototype.runMoveEffects.call(actions,[undefined],[target],source,move,move,false,false);console.log(JSON.stringify({result,lines,hp:target.hp,status:target.status,pp:target.pp,item:target.item}));'''
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(got,dict(result=[False],lines=['|-immune|p2a: Target|[from] ability: Limber'],hp=100,status='',pp=[5],item='leftovers'))
 def test_04_each_private_after_field_keeps_actual_geometry(self):
  _,t,_,actor,phys,target,move,a,ticket,p,_=self.literal()
  c=t.execution.thunder_wave_immunity[0];self.assertEqual(cb.owned_limber_immunity(t,2).tolist(),[True]*8)
  for name in cb.Private._fields:
   value=getattr(c.after,name)
   for malformed in (value[:,:1],value.astype(j.float32)):
    after=c.after._replace(**{name:malformed});bad=t._replace(execution=t.execution._replace(thunder_wave_immunity=(c._replace(after=after),)))
    with self.assertRaises(ValueError,msg=name):cb.owned_limber_immunity(bad,2)
  with self.assertRaises(ValueError):cb._limber_attempt(p,tuple(c.after),a)
 def test_03_old_routes_and_legacy_remain_current(self):
  ground=ground_fixture.ThunderGroundTests.literal(self,False)
  self.assertEqual(cb.owned_ground_immunity(ground[1],ground[1].count-1).tolist(),[True]*8)
  # This compatibility fixture actually invokes the separate Substitute
  # producer; prepare that existing capability rather than inventing a Context.
  runtime=self.runtime
  req=runtime['request_compiled'];facts=runtime['own_compiled']
  sub_cap=sub_callbacks.compile_substitute(engine_catalog=self.producer,metadata=self.metadata,source_rules=self.rules)
  prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(self.metadata,engine_catalog=self.producer),leftovers_mode='retained',direct_hit_compiled=self.helper,direct_hit_mode='kernel-event',thunder_wave_compiled=self.thunder,thunder_wave_mode='kernel-event',substitute_compiled=sub_cap,substitute_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  try:
   self.runtime=driver._checked(prepared);sub=fixture.ThunderWaveSubstituteTests.literal(self,False)
   self.assertEqual(cb.owned_substitute_failure(sub[2],3).tolist(),[True]*8)
  finally:self.runtime=runtime
  s,t,_,actor,phys,target,move,a,ticket,p,before=self.literal()
  legacy=cb.Context(self.legacy.device,self.runtime['execution_compiled'].device)
  out=cb.record_device(before,ticket,actor,phys,target,move,a,legacy,requested=j.ones(8,bool),ground_pending=None,state=s)
  self.assertEqual(out.count.tolist(),[2]*8);self.assertTrue(bool(j.all(out.execution.ledger.refusal!=0)))
  # Current raw/event/ticket integrity is independent of identical-looking
  # scalar values in an unclaimed or stale record.
  for field in ('event_turn','event_phase','event_generation'):
   x=getattr(t,field);mutated=t._replace(**{field:x.at[0,2].add(1)});self.assertEqual(cb.owned_limber_immunity(mutated,2).tolist(),[False]+[True]*7)
  c=t.execution.thunder_wave_immunity[0]
  bad=c._replace(pending=c.pending._replace(ticket=c.pending.ticket._replace(serial=c.pending.ticket.serial+1)))
  self.assertFalse(bool(cb.owned_limber_immunity(t._replace(execution=t.execution._replace(thunder_wave_immunity=(bad,))),2).any()))
 def test_02_original_both_full(self):
  s,t,consumers,actor,phys,target,move,a,ticket,p,_=self.literal(True);yes=j.ones(8,bool);z=j.zeros(8,j.int32);cap=risk.CapturedThunderWave(self.runtime['thunder_wave_context'],ticket,actor,phys,target,move,a)
  t=risk.primary_status_on_tape(t,yes,active=yes,captured_thunder_wave=cap);self.assertEqual(t.execution.risks.carry.risk.tolist(),[0]*8);self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  t=life.action_end(t,yes);t=life.finish(t,s,live=yes,replacement=~yes,pending_before=~yes,residual_ready=yes,defer_residual=~yes,turn_context=yes);self.assertTrue(bool(cb.owned_limber_immunity(t,j.full(8,2)).all()))
  frame=life.LifecycleResult(s,t.cursor,t,life.snapshot(t.cursor),life.snapshot(t.cursor),~yes,z);packets=[];ledgers=[]
  consume=jax.jit(lambda ledger,frame,stream,p:ex.consume_frame_device(ledger,frame,stream,p,active=yes,tables=self.runtime['execution_compiled'].device,adapter_tables=self.runtime['public'].device,catalog=self.catalog))
  for owner,(stream,pub)in enumerate(consumers):
   ledger,got=consume(t.execution.ledger,frame,stream,pub);ledgers.append(ledger)
   for row in range(8):
    names={side:self.catalog.species.names[int(self.runtime['public'].device.causal.species[side+1])]for side in(0,1)};ident={side:f'p{side+1}a: '+names[side]for side in(0,1)};who=int(actor[row]);foe=1-who
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ident[side]}|{names[side]}, L80, M|100/100'for side in(0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));_,original=public_wire.project_lines(wire,['|','|t:|1',f'|move|{ident[who]}|Thunder Wave|{ident[foe]}',f'|-immune|{ident[foe]}|[from] ability: Limber','|','|upkeep','|turn|2'],**v4.wire_kwargs(self.catalog));wanted=np.asarray([x.words()for packet in original for x in packet.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])]
    packets.append(dict(owner=owner,row=row,error=int(got.projection.error[row]),admitted=bool(got.projection.component_admitted[row]),stage=ledger.stage[row].tolist(),refusal=int(ledger.refusal[row]),actual=actual.tolist(),expected=wanted.tolist()))
  Path('/tmp/pokezero-thunder-limber-original-both-full.json').write_text(json.dumps(packets)+'\n')
  for packet in packets:self.assertEqual(packet['error'],0);self.assertTrue(packet['admitted']);np.testing.assert_array_equal(packet['actual'],packet['expected'])
  self.assertEqual(ex.join_consumed_owners(*ledgers,tables=self.runtime['execution_compiled'].device)[1].tolist(),[True]*8)
  capture_path(self,'thunder-wave-limber-original-both-full-compact.json').write_text(json.dumps(packets,sort_keys=True)+'\n')
from gpu_engine import engine_full_v4_event_device as full
