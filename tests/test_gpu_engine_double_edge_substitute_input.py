"""Original same-call Sub/recoil writers only; no engine turn/model/game."""
import ast,json,subprocess,unittest
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_substitute_damage_callbacks as sd,engine_substitute_callbacks as sub
from gpu_engine import engine_recoil_callbacks as recoil,engine_direct_hit_controls as direct
from gpu_engine import engine_kernel_frame_risk as risk,engine_execution_completeness as ex
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_full_v4_event_device as full
from gpu_engine import v4_encoder as v4,public_wire,state,engine_source_frame_witness as witness
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_double_edge_type_immunity as fixture
BASE='fd3256d2305a0770764e737640fc80157156cb29'

class DoubleEdgeSubstituteTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.DoubleEdgeTypeImmunityTests.setUpClass();f=fixture.DoubleEdgeTypeImmunityTests
  for name in ('metadata','catalog','dex','table','producer','rules','helper','recoil','kw'):setattr(cls,name,getattr(f,name))
  cls.sub=sub.compile_substitute(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  old=f.runtime
  cls.prepared=driver.prepare(old['request_compiled'],old['own_compiled'],public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=old['leftovers_compiled'],leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',recoil_compiled=cls.recoil,recoil_mode='kernel-event',substitute_compiled=cls.sub,substitute_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared)
  cls.fold=staticmethod(jax.jit(lambda a,b,c:full.project_and_fold(a,b,cls.runtime['public'].device,c,cls.catalog)))
 def pending(self,consume=False,mask=None):
  b=8;rows=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);actor=rows%2;active=j.stack((z+3,z+4),axis=1);ap=active[rows,actor];tp=active[rows,1-actor];v=self.metadata['vocabulary'];mask=yes if mask is None else mask
  s=state.empty_state(b)._replace(done=~yes,active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),264,j.int32),max_hp=j.full((b,2,6),264,j.int32),species=j.ones((b,2,6),j.int32),level=j.full((b,2,6),80,j.int32),ability=j.full((b,2,6),v['abilities'].index('keeneye'),j.int32))
  s=s._replace(hp=s.hp.at[rows,actor,ap].set(285),max_hp=s.max_hp.at[rows,actor,ap].set(285),base_ability=s.ability)
  cursor=life.register(rows+470001,**self.kw);initial=life._tape(cursor,self.producer,64)
  for side in(0,1):initial=life._reveal(initial,s,z+side,z-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);consumers=[]
  if consume:
   frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,z)
   for owner in(0,1):
    stream=full.register(cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=z+owner);got=self.fold(stream,frame,pub);self.assertEqual(got.projection.error.tolist(),[0]*b);consumers.append((got.projection.state,got.public))
  t=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;t=ex.begin_on_tape(t,ex.initialize(t.stream_id,compiled=account),account.device,active=yes);t=t._replace(execution=t.execution._replace(risks=risk.Context(risk.begin_frame_device(risk.initialize_device(t.stream_id,rt),t,rt,active=yes),rt)))
  move=z+v['moves'].index('doubleedge');sm=z+v['moves'].index('substitute')
  t=public.move(t,1-actor,tp,ap,sm,eligible=mask);t,st=ex.open_attempt(t,ex.Site.SUBSTITUTE_SELF,1-actor,active=mask)
  sw=sub.Write(z+264,z+198,z+264,z,z,z+66);t=sub.record_device(t,1-actor,tp,sm,sw,self.runtime['substitute_context'],eligible=mask,caller_move=sm)
  t=ex.capture_substitute_write(t,st,1-actor,tp,sm,sw,requested=mask)
  s=s._replace(hp=s.hp.at[rows,1-actor,tp].set(198),substitute_hp=s.substitute_hp.at[rows,1-actor,tp].set(66))
  t=public.move(t,actor,ap,tp,move,eligible=mask);ctx=self.runtime['direct_hit_context']
  route=sd.route(ctx,move,acts=mask,power=z+120,hit=yes,immune=~yes,protected=~yes,enduring=~yes,bypass=~yes,absorbed=~yes,sub_hp=z+66,caller_move=move)
  self.assertEqual(route.tolist(),mask.tolist());t,ticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_DAMAGE,1-actor,active=route)
  p=sd.Pending(ctx,ticket,actor,ap,tp,move,route,caller_move=move)
  w=sd.Write(z+120,yes,~yes,~yes,~yes,~yes,~yes,~yes,z+1,z+1,z+66,z,z+66,z+90,z+198,z+198)
  t=sd.record(t,p,w);t,dt=ex.open_attempt(t,ex.Site.DIRECT_HIT,1-actor,active=~yes)
  dw=direct.HitWrite(w.power,w.critical,w.hit,w.immune,w.numerator,w.denominator,yes,w.protected,w.enduring,w.old_hp,w.new_hp,z+264)
  captured=risk.CapturedDirectHit(ctx,dt,actor,ap,tp,move,dw);subcap=sd.Captured(p,w)
  amount=j.where(mask,22,0)
  t,rp=recoil.prepare_device(t,captured,amount,z+v['abilities'].index('keeneye'),z+285,z+285,z,self.runtime['recoil_context'],eligible=mask,caller_move=move,substitute_damage=subcap)
  out=risk.Outcome(w.power,w.critical,w.hit,w.immune,w.numerator,w.denominator,~yes,yes,w.protected,w.enduring,amount,~yes)
  return s,t,rp,out,subcap,consumers,initial.cursor
 def observe(self,t,p,o,s):
  return risk.observe_outcome_device(t.execution.risks.carry,t,o,self.runtime['kernel_risk_compiled'].device,eligible=p.requested,active=j.ones(8,bool),captured_direct=p.captured,pending_recoil=p,captured_substitute_damage=s)
 def test_typed_pair_current_outcome_and_final_write_negatives(self):
  mask=j.array([True]*6+[False]*2);s,t,p,o,subcap,_,_=self.pending(mask=mask)
  self.assertEqual(recoil.matches_pending(t,p).tolist(),mask.tolist());self.assertEqual(self.observe(t,p,o,subcap).risk.tolist(),[0]*8)
  for name in o._fields:
   value=getattr(o,name);bad=~value if value.dtype==j.bool_ else value+1
   self.assertTrue(bool((self.observe(t,p,o._replace(**{name:bad}),subcap).risk[:6]!=0).all()),name)
  for field in subcap.write._fields:
   value=getattr(subcap.write,field);bad=~value if value.dtype==j.bool_ else value+1
   mutated=subcap._replace(write=subcap.write._replace(**{field:bad}))
   self.assertFalse(bool(recoil.matches_substitute_pending(t,p,mutated).any()),field)
  self.assertFalse(bool(recoil.matches_pending(t,p._replace(caller_move=p.caller_move+1)).any()))
  self.assertFalse(bool(recoil.matches_pending(t,p._replace(substitute_damage=None)).any()))
  for n in('engine_binding','source_binding','move_binding','producer_binding'):
   self.assertFalse(bool(recoil.matches_pending(t._replace(**{n:getattr(t,n)+1}),p).any()))
   with self.assertRaises(ValueError):recoil.matches_pending(t._replace(**{n:j.broadcast_to(getattr(t,n),(8,8))}),p)
  before=recoil.Before(p.captured.actor_physical,p.captured.target_physical,p.ability,p.old_hp,p.maximum,p.status)
  outt=risk.recoil_on_tape(t,p.recoil,active=j.ones(8,bool),pending=p,before=before)
  got=recoil.record_device(outt,p,recoil.Write(*before[:4],p.old_hp-p.recoil,p.maximum,p.status));self.assertEqual(got.count.tolist(),[5]*6+[0,0]);self.assertEqual(got.execution.risks.carry.risk.tolist(),[0]*8)
  sticky=t._replace(refusal=t.refusal|64);self.assertFalse(bool(recoil.matches_pending(sticky,p).any()))
 def test_original_writer_and_both_full_current_owners(self):
  s,t,p,o,subcap,consumers,cursor=self.pending(True);r=self.observe(t,p,o,subcap);self.assertEqual(r.risk.tolist(),[0]*8)
  t=t._replace(execution=t.execution._replace(risks=risk.Context(r,self.runtime['kernel_risk_compiled'].device)))
  before=recoil.Before(p.captured.actor_physical,p.captured.target_physical,p.ability,p.old_hp,p.maximum,p.status);t=risk.recoil_on_tape(t,p.recoil,active=j.ones(8,bool),pending=p,before=before)
  t=recoil.record_device(t,p,recoil.Write(*before[:4],p.old_hp-p.recoil,p.maximum,p.status));rows=j.arange(8);actor=p.captured.actor
  s=s._replace(substitute_hp=s.substitute_hp.at[rows,1-actor,p.captured.target_physical].set(0),hp=s.hp.at[rows,actor,p.captured.actor_physical].set(263))
  t=life.action_end(t,j.ones(8,bool));t=life.finish(t,s,live=j.ones(8,bool),replacement=j.zeros(8,bool),pending_before=j.zeros(8,bool),residual_ready=j.ones(8,bool),defer_residual=j.zeros(8,bool),turn_context=j.ones(8,bool))
  frame=life.LifecycleResult(s,t.cursor,t,life.snapshot(cursor),life.snapshot(t.cursor),j.zeros(8,bool),j.zeros(8,j.int32));captures=[]
  consume=jax.jit(lambda stream,pub:ex.consume_frame_device(t.execution.ledger,frame,stream,pub,active=j.ones(8,bool),tables=self.runtime['execution_compiled'].device,adapter_tables=self.runtime['public'].device,catalog=self.catalog))
  for owner,(stream,pub)in enumerate(consumers):
   ledger,got=consume(stream,pub)
   self.assertEqual(got.projection.error.tolist(),[0]*8);self.assertTrue(bool(ex.report(ledger,tables=self.runtime['execution_compiled'].device).scoped_drained.all()))
   for row in range(8):
    name=self.dex.species[self.metadata['vocabulary']['species'][1]].name;ids={i:f'p{i+1}a: {name}'for i in(0,1)};a=row%2;tar=1-a
    prefix=[f'|switch|{ids[i]}|{name}, L80, M|'+(('285/285'if i==a else'264/264')if owner==i else'100/100')for i in(0,1)]+['|turn|1']
    script=r'''const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),B=require(root+'/dist/sim/battle.js').Battle,ids=JSON.parse(process.argv[2]),lines=[];const battle={gen:3,dex:d,clampIntRange:(x,m)=>Math.max(x,m),runEvent:(id,t,s,e,n)=>n,add:(...x)=>lines.push('|'+x.map(String).join('|'))};const a={hp:285,isActive:true,toString:()=>ids.a,get getHealth(){return this.hp+'/285'},damage(n){this.hp-=n;return n}},t={hp:198,volatiles:{substitute:{hp:66}},toString:()=>ids.t,removeVolatile(id){delete this.volatiles[id];d.moves.get('substitute').condition.onEnd.call(battle,this)},addVolatile(id){this.volatiles[id]={};return true}};battle.actions={getDamage:()=>90,calcRecoilDamage:(n,m)=>d.data.Scripts.actions.calcRecoilDamage.call({battle},n,m)};battle.damage=(n,target,source,effect)=>B.prototype.spreadDamage.call(battle,[n],[target],source,effect)[0];d.moves.get('substitute').condition.onTryPrimaryHit.call(battle,t,a,d.moves.get('doubleedge'));process.stdout.write(JSON.stringify({lines,hp:a.hp,sub:t.volatiles.substitute||null}));'''
    original=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root()),json.dumps({'a':ids[a],'t':ids[tar]})],text=True));self.assertEqual(original['hp'],263);self.assertIsNone(original['sub'])
    lines=[f'|move|{ids[tar]}|Substitute|{ids[tar]}',f'|-start|{ids[tar]}|Substitute',f'|-damage|{ids[tar]}|'+('198/264'if owner==tar else'75/100'),f'|move|{ids[a]}|Double-Edge|{ids[tar]}']+original['lines']+['|','|upkeep','|turn|2']
    if owner!=a:lines=[line.replace('263/285','93/100')for line in lines]
    wire,_=public_wire.project_lines(public_wire.WireState(),prefix,**v4.wire_kwargs(self.catalog));_,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));expected=np.array([v.words()for packet in packets for v in packet.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,expected);captures.append(dict(owner=owner,row=row,expected=expected.tolist(),actual=actual.tolist()))
  Path('/Users/scott/workspace/agents/pokezero-b4-driver-independent-review-evidence-20261030/double-edge-substitute-original-both-full-compact.json').write_text(json.dumps(captures,sort_keys=True)+'\n')
 def test_source_default_and_actual_transition_operand_binding(self):
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(Path('gpu_engine/transition.py').read_text());self.assertEqual(dump_default_program(old),dump_default_program(new));assert_hp_rng_mutations_detected(self,new,dump_default_program(old));witness._source()
  call=next(n for n in ast.walk(new)if isinstance(n,ast.Call)and ast.unparse(n.func)=='recoil_callbacks.prepare_device');extra=next(k.value for k in call.keywords if k.arg=='substitute_damage');self.assertEqual(ast.unparse(extra),'substitute_damage_callbacks.Captured(_sub_damage_pending, _sub_damage_write)')
  self.assertEqual(int(self.helper.device.substitute_damage.double_edge_move),self.metadata['vocabulary']['moves'].index('doubleedge'))
  legacy=direct.compile_direct_hit_controls(engine_catalog=self.producer);self.assertIsNone(legacy.device.substitute_damage)
  # Whole original function scope contains both actual observer operands;
  # no guessed/undefined name is introduced at the production call site.
  import symtable,builtins
  module=symtable.symtable(Path('gpu_engine/transition.py').read_text(),'transition.py','exec');step=next(c for c in module.get_children()if c.get_name()=='step_with_chance')
  known=set(module.get_identifiers())|set(dir(builtins));unknown=[s.get_name()for s in step.get_symbols()if s.is_referenced()and s.is_global()and s.get_name()not in known];self.assertEqual(unknown,[])

if __name__=='__main__':unittest.main()
