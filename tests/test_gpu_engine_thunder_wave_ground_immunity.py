"""Same-write natural immunity inputs; no engine turn or scientific roots."""
import ast,hashlib,json,subprocess,unittest
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_thunder_wave_callbacks as cb,engine_execution_completeness as ex
from gpu_engine import engine_toxic_callbacks as primary,engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_kernel_frame_risk as risk,engine_source_frame_witness as witness
from gpu_engine import public_wire,v4_encoder as v4
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_thunder_wave_substitute_failure as fixture
BASE='6a55ed80787129fdfab289f20f97193e263f60c2'

class ThunderGroundTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):fixture.ThunderWaveSubstituteTests.setUpClass.__func__(cls)
 component=fixture.ThunderWaveSubstituteTests.component
 def literal(self,full=False,**change):
  s,t,pubs,actor,phys=self.component(full);b=8;r=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);target=s.active[r,1-actor];data=self.thunder.device;g=data.ground_immunity
  ability=int(np.flatnonzero(np.asarray(g.abilities))[0]);self.assertGreater(ability,0)
  types=j.full((b,2,6,2),self.metadata['vocabulary']['types'].index('normal'),j.int32).at[r,1-actor,target].set(j.stack((z+int(g.ground),z),axis=1))
  s=s._replace(done=~yes,turn=t.cursor.turn-1,types=types,base_types=types,ability=j.full((b,2,6),ability,j.int32),base_ability=j.full((b,2,6),ability,j.int32),moves=s.moves.at[r,actor,phys,0].set(self.move_id),pp=s.pp.at[r,actor,phys,0].set(5))
  t=life.begin(t,s,yes,~yes);move=z+self.move_id;t=public.move(t,actor,phys,target,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.PRIMARY_PARALYSIS,1-actor,active=yes)
  kw=dict(requested=yes,resolved_type=z+int(g.electric),power=z,accuracy=yes,hit=yes,immune=yes,numerator=z,denominator=z+1,protected=~yes,absorbed=~yes);kw.update(change)
  p=cb.begin_ground(t,s,ticket,actor,move,move,z,self.runtime['thunder_wave_context'],**kw)
  a=cb.Attempt(yes,yes,~yes,~yes,~yes,yes,~yes,~yes,~yes,yes,~yes,z,z,z,z)
  after=s;t=cb.record_device(t,ticket,actor,phys,target,move,a,self.runtime['thunder_wave_context'],requested=yes,ground_pending=p,state=after)
  return s,t,pubs,actor,phys,target,move,a,ticket,p
 def test_01_current_operands_source_boundaries_and_default(self):
  s,t,_,actor,phys,target,move,a,ticket,p=self.literal();yes=j.ones(8,bool);z=j.zeros(8,j.int32);index=t.count-1
  c=t.execution.thunder_wave_immunity[0];ledger=t.execution.ledger
  Path('/tmp/pokezero-thunder-ground-guard-initial.json').write_text(json.dumps(dict(count=t.count.tolist(),words=t.words[:,:4].tolist(),refusal=t.refusal.tolist(),ledger_refusal=ledger.refusal.tolist(),stage=ledger.stage[:,:2].tolist(),source=cb._ground_source(t,p.context).tolist(),valid=cb._ground_valid(t,p).tolist(),raw=primary.words_device(actor,phys,target,move,a).tolist(),ground_raw=cb.ground_raw(primary.words_device(actor,phys,target,move,a)).tolist(),owned=cb.owned_ground_immunity(t,index).tolist(),ability=p.before.ability.tolist()))+'\n')
  self.assertEqual(cb.owned_ground_immunity(t,index).tolist(),[True]*8)
  cap=risk.CapturedThunderWave(self.runtime['thunder_wave_context'],ticket,actor,phys,target,move,a)
  self.assertEqual(risk._captured_thunder_wave_matches(t,self.runtime['kernel_risk_compiled'].device,cap).tolist(),[True]*8)
  for kw in (dict(resolved_type=p.resolved_type.at[0].add(1)),dict(denominator=p.denominator.at[0].add(1)),dict(hit=p.hit.at[0].set(False))):
   got=self.literal(**kw)[1];self.assertEqual(cb.owned_ground_immunity(got,got.count-1).tolist(),[False]+[True]*7)
  for field,value in [('pp',p.before.pp.at[0,0,0].add(1)),('types',p.before.types.at[0,0,0].set(0)),('status',p.before.status.at[0,0].set(1))]:
   c=t.execution.thunder_wave_immunity[0];badc=c._replace(after=c.after._replace(**{field:value}));bad=t._replace(execution=t.execution._replace(thunder_wave_immunity=(badc,)));self.assertEqual(cb.owned_ground_immunity(bad,index).tolist(),[False]+[True]*7,field)
  c=t.execution.thunder_wave_immunity[0]
  for field,value in [('caller',p.caller+1),('slot',p.slot+1),('physical',p.physical+1),('numerator',p.numerator+1)]:
   bad=t._replace(execution=t.execution._replace(thunder_wave_immunity=(c._replace(pending=p._replace(**{field:value})),)));self.assertFalse(bool(cb.owned_ground_immunity(bad,index).any()),field)
  for field in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=t._replace(**{field:getattr(t,field).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.owned_ground_immunity(bad,index).any()))
   with self.assertRaises(ValueError):cb.owned_ground_immunity(t._replace(**{field:j.broadcast_to(getattr(t,field),(8,8))}),index)
  self.assertFalse(bool(cb.owned_ground_immunity(t._replace(execution=t.execution._replace(thunder_wave_immunity=None)),index).any()))
  self.assertIsNone(self.legacy.device.ground_immunity)
  prior=t._replace(refusal=t.refusal|128);kept=risk.primary_status_on_tape(prior,yes,active=yes,captured_thunder_wave=cap);self.assertTrue(bool((kept.refusal&128).all()))
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(subprocess.check_output(['git','show','974a4ceb:gpu_engine/transition.py'],text=True));self.assertEqual(dump_default_program(old),dump_default_program(new));assert_hp_rng_mutations_detected(self,new,dump_default_program(old))
  new=ast.parse(Path('gpu_engine/transition.py').read_text());integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True))
  self.assertEqual(ast.dump(new),ast.dump(integrated));assert_hp_rng_mutations_detected(self,new,dump_default_program(integrated))
  calls=[n for n in ast.walk(new)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='begin_ground'];self.assertEqual(len(calls),1);expr=calls[0];self.assertEqual([ast.unparse(n)for n in expr.args],['_public_trace','work','_thunder_wave_ticket','side','move','caller_move','chosen_slot','_thunder_wave_context']);self.assertEqual(ast.unparse(next(k.value for k in expr.keywords if k.arg=='resolved_type')),'typ')
  import sys
  from types import SimpleNamespace as NS
  driver_tree=ast.parse(Path('gpu_engine/engine_retained_encoder_driver.py').read_text())
  frame=next(n for n in driver_tree.body if isinstance(n,ast.FunctionDef)and n.name=='_frame')
  retained=next(k.value for n in ast.walk(frame)if isinstance(n,ast.Call)for k in n.keywords if k.arg=='thunder_wave_immunity')
  mask=j.array([True,False]*4)
  carried=eval(compile(ast.Expression(retained),'<actual driver context retention>','eval'),dict(thunder_callbacks=cb,source_ready=mask,result=NS(callbacks=t)))
  self.assertEqual(carried[0].pending.requested.tolist(),mask.tolist())
  self.assertEqual(cb.owned_ground_immunity(t._replace(execution=t.execution._replace(thunder_wave_immunity=carried)),index).tolist(),mask.tolist())
  for order in ('gpu_engine.benchmark_retained_games,gpu_engine.engine_full_v4_event_device','gpu_engine.engine_full_v4_event_device,gpu_engine.benchmark_retained_games'):
   subprocess.run([sys.executable,'-B','-c','import importlib;'+''.join('importlib.import_module('+repr(n)+');'for n in order.split(','))],check=True,capture_output=True,text=True)
  # Actual pinned runImmunity callback, inert target; no Battle is constructed.
  script=r"""const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),f=require(root+'/dist/sim/pokemon.js').Pokemon.prototype.runImmunity,rows=[];for(const type of ['Ground','Normal']){const lines=[],p={hp:100,status:'',pp:[5],getTypes:()=>[type],toString:()=> 'p2a: Target',battle:{dex:d,runEvent:()=>true,add:(...xs)=>lines.push('|'+xs.map(String).join('|'))}};const ok=f.call(p,d.moves.get('thunderwave'),true);rows.push({type,ok,lines,hp:p.hp,status:p.status,pp:p.pp})}console.log(JSON.stringify(rows))"""
  actual=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(actual,[dict(type='Ground',ok=False,lines=['|-immune|p2a: Target'],hp=100,status='',pp=[5]),dict(type='Normal',ok=True,lines=[],hp=100,status='',pp=[5])])
 def test_02_original_both_full_plain_immunity(self):
  s,t,consumers,actor,phys,target,move,a,ticket,p=self.literal(True);yes=j.ones(8,bool);z=j.zeros(8,j.int32);cap=risk.CapturedThunderWave(self.runtime['thunder_wave_context'],ticket,actor,phys,target,move,a)
  t=risk.primary_status_on_tape(t,yes,active=yes,captured_thunder_wave=cap);self.assertEqual(t.execution.risks.carry.risk.tolist(),[0]*8)
  self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  t=life.action_end(t,yes);t=life.finish(t,s,live=yes,replacement=~yes,pending_before=~yes,residual_ready=yes,defer_residual=~yes,turn_context=yes)
  self.assertTrue(bool(cb.owned_ground_immunity(t,j.full(8,2)).all()))
  frame=life.LifecycleResult(s,t.cursor,t,life.snapshot(t.cursor),life.snapshot(t.cursor),~yes,z);packets=[]
  for owner,(stream,pub)in enumerate(consumers):
   got=self.fold(stream,frame,pub)
   for row in range(8):
    names={side:self.catalog.species.names[side+1]for side in (0,1)};ident={side:f'p{side+1}a: '+names[side]for side in(0,1)};who=int(actor[row]);foe=1-who
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ident[side]}|{names[side]}, L80, M|100/100'for side in(0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));_,original=public_wire.project_lines(wire,['|','|t:|1',f'|move|{ident[who]}|Thunder Wave|{ident[foe]}',f'|-immune|{ident[foe]}','|','|upkeep','|turn|2'],**v4.wire_kwargs(self.catalog));wanted=np.asarray([x.words()for packet in original for x in packet.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])]
    packets.append(dict(owner=owner,row=row,error=int(got.projection.error[row]),admitted=bool(got.projection.component_admitted[row]),actual=actual.tolist(),expected=wanted.tolist()))
  # Persist both owners/all literal words before any comparison assertion.
  Path('/tmp/pokezero-thunder-ground-both-full-r5.json').write_text(json.dumps(packets)+'\n')
  for packet in packets:
   self.assertEqual(packet['error'],0);self.assertTrue(packet['admitted']);np.testing.assert_array_equal(packet['actual'],packet['expected'])
