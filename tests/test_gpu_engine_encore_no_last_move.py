"""Literal original no-last Encore failure, no engine step/game/RNG/model."""
import ast,json,subprocess,unittest
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_encore_callbacks as cb,engine_execution_completeness as ex,engine_lifecycle_callbacks as life
from gpu_engine import engine_public_events as public,engine_kernel_frame_risk as risk,engine_full_v4_event_device as full
from gpu_engine import encore,public_wire,v4_encoder as v4
from _showdown_root import showdown_root
import test_gpu_engine_hidden_power_substitute_input as fixture
BASE='f2f3b9ad725c7f3064f0420e07425ec68e95ec37'
class EncoreNoLastMoveTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.HiddenPowerSubstituteInputTests.setUpClass();cls.f=fixture.HiddenPowerSubstituteInputTests();cls.runtime=cls.f.runtime;cls.helper=cls.f.helper;cls.metadata=cls.f.metadata;cls.move=cls.metadata['vocabulary']['moves'].index('encore');cls.table=cls.f.table;cls.catalog=cls.f.catalog
 def component(self,consume=False):
  s,t,owners,side,phys=self.f.component(consume_initial=consume);b=8;r=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);kind=r%2;index=j.where(kind==0,-1,phys)
  s=s._replace(done=~yes,last_action_kind=s.last_action_kind.at[r,side].set(kind),last_action_index=s.last_action_index.at[r,side].set(index))
  t=public.move(t,1-side,s.active[r,1-side],phys,z+self.move,eligible=yes)
  t,p=cb.begin(t,s,side,z+self.move,self.runtime['direct_hit_context'],self.table.encore,eligible=yes,secondary=yes,roll=z,queued=~yes)
  after=encore.apply(s,side,z+self.move,z,self.table.encore,eligible=yes,queued=~yes);out,c=cb.finish(t,p,after)
  return s,t,after,out,c,owners
 def test_original_writer_static_and_current_pending_transaction_guards(self):
  script=r"""const root=process.argv[1],{BattleActions}=require(root+'/dist/sim/battle-actions.js'),d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),c=d.moves.get('encore').condition,lines=[],attrs=[];const a={lastMove:null,volatiles:{},toString:()=> 'p1a: Actor'},t={lastMove:null,volatiles:{},addVolatile(){return c.onStart.call({},t)},toString:()=> 'p2a: Target'};const battle={dex:d,add:(...x)=>lines.push(x.map(String).join('|')),attrLastMove:x=>attrs.push(x),runEvent:()=>true,singleEvent:()=>true,debug(){}};const result=BattleActions.prototype.runMoveEffects.call({battle,dex:d,combineResults:BattleActions.prototype.combineResults},[undefined],[t],a,d.moves.get('encore'),d.moves.get('encore'),false,false);console.log(JSON.stringify({result,lines,attrs,sha:require('crypto').createHash('sha256').update(''+BattleActions.prototype.runMoveEffects).digest('hex')}));"""
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(got,dict(result=[False],lines=['-fail|p1a: Actor'],attrs=['[still]'],sha=cb.RUN_MOVE_EFFECTS_SHA256))
  s,t,after,out,c,_=self.component();self.assertEqual(c.raw[:,3].tolist(),[cb.FAIL_NO_LAST]*8);self.assertTrue(bool(cb.captured_matches(out,c).all()));self.assertEqual(out.execution.ledger.stage[:,0].tolist(),[2]*8)
  for field in ('hp','max_hp','moves','pp','status','status_counter','keys','last_move','last_action_kind','last_action_index','encore_move','encore_counter'):np.testing.assert_array_equal(getattr(s,field),getattr(after,field))
  for field in ('last_move','hp','max_hp','status','status_counter','moves','pp','last_action_kind','last_action_index'):
   value=getattr(after,field);idx=(0,int(c.pending.side[0]),int(c.pending.physical[0]))if value.ndim>=3 else(0,int(c.pending.side[0]));idx=idx+(0,)if value.ndim==4 else idx;bad=after._replace(**{field:value.at[idx].add(1)});x,_=cb.finish(t,c.pending,bad);self.assertFalse(bool(cb.owned(x,j.maximum(x.count-1,0))[0]),field)
  for label,before in [('move',s._replace(last_action_kind=s.last_action_kind.at[0,0].set(2))),('struggle',s._replace(last_action_kind=s.last_action_kind.at[0,0].set(3))),('nonzero-last',s._replace(last_move=s.last_move.at[0,0,3].set(1))),('switch-foreign',s._replace(last_action_kind=s.last_action_kind.at[0,0].set(1),last_action_index=s.last_action_index.at[0,0].set(2))),('old-encore',s._replace(encore_move=s.encore_move.at[0,0,3].set(1)))]:
   x,p=cb.begin(t,before,c.pending.side,j.full(8,self.move,j.int32),self.runtime['direct_hit_context'],self.table.encore,eligible=j.ones(8,bool),secondary=j.ones(8,bool),roll=j.zeros(8,j.int32),queued=j.zeros(8,bool));self.assertFalse(bool(p.valid[0]),label)
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=out._replace(**{name:getattr(out,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.captured_matches(bad,c).any()))
   with self.assertRaises(ValueError):cb.owned(out._replace(**{name:j.tile(getattr(out,name),(8,1))}),1)
  for field in ('turn','phase','generation'):
   bad=out._replace(cursor=out.cursor._replace(**{field:getattr(out.cursor,field)+1}));self.assertFalse(bool(cb.captured_matches(bad,c).any()))
  # Coherent raw/captured/ledger edits cannot replace original Pending facts.
  raw=out.words.at[0,1,9].set(1);ledger=out.execution.ledger._replace(words=out.execution.ledger.words.at[0,1,9].set(1));bad=out._replace(words=raw,execution=out.execution._replace(ledger=ledger));forged=c._replace(raw=c.raw.at[0,9].set(1));self.assertFalse(bool(cb.captured_matches(bad,forged)[0]))
  self.assertFalse(bool(cb.captured_matches(out,c._replace(pending=c.pending._replace(failure=j.zeros(8,bool)))).any()))
  missing=out._replace(execution=out.execution._replace(ledger=out.execution.ledger._replace(no_output_binding=j.zeros_like(out.execution.ledger.no_output_binding))));self.assertFalse(bool(cb.owned(missing,1).any()))
  sticky=t._replace(refusal=t.refusal|64);x,_=cb.finish(sticky,c.pending,after);self.assertTrue(bool((x.refusal&64).all()))
  # Existing positive START/TICK/END helper transactions retain their codes.
  pos=s._replace(last_action_kind=j.full((8,2),2,j.int32),last_action_index=j.ones((8,2),j.int32));r=j.arange(8);side=c.pending.side;phys=c.pending.physical;surf=self.metadata['vocabulary']['moves'].index('surf');pos=pos._replace(moves=pos.moves.at[r,side,phys,1].set(surf),pp=pos.pp.at[r,side,phys,1].set(8))
  x,p=cb.begin(t,pos,side,j.full(8,self.move,j.int32),self.runtime['direct_hit_context'],self.table.encore,eligible=j.ones(8,bool),secondary=j.ones(8,bool),roll=j.zeros(8,j.int32),queued=j.ones(8,bool));start=encore.apply(pos,side,j.full(8,self.move,j.int32),j.zeros(8,j.int32),self.table.encore,eligible=j.ones(8,bool),queued=j.ones(8,bool));x,cc=cb.finish(x,p,start);self.assertEqual(cc.raw[:,3].tolist(),[cb.START]*8);self.assertTrue(bool(cb.captured_matches(x,cc).all()))
  for count,code in ((3,cb.TICK),(1,cb.END)):
   before=start._replace(encore_counter=start.encore_counter.at[r,side,phys].set(count));y,pp=cb.begin(x,before,side,j.zeros(8,j.int32),self.runtime['direct_hit_context'],self.table.encore,eligible=j.ones(8,bool),residual=True);after_tick=encore.residual(before,side,eligible=j.ones(8,bool));y,cc=cb.finish(y,pp,after_tick);self.assertEqual(cc.raw[:,3].tolist(),[code]*8);self.assertTrue(bool(cb.captured_matches(y,cc).all()))
  observed=risk.branch_on_tape(out,risk.Branch.ENCORE,j.ones(8,bool),active=j.ones(8,bool),captured_encore=c);self.assertEqual(observed.execution.risks.carry.risk.tolist(),[0]*8)
  for file in ('transition.py','encore.py','state.py','action_history.py','engine_retained_encoder_driver.py'):
   self.assertEqual(Path('gpu_engine',file).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+file]))
 def test_both_original_full_callers_empty_target_still_and_no_lock(self):
  s,t,after,out,c,owners=self.component(True);b=8;yes=j.ones(b,bool);z=j.zeros(b,j.int32);out=life.action_end(out,yes);out=life.finish(out,after,live=yes,replacement=~yes,pending_before=~yes,residual_ready=yes,defer_residual=~yes,turn_context=yes)
  frame=life.LifecycleResult(after,out.cursor,out,life.snapshot(t.cursor),life.snapshot(out.cursor),~yes,z);closed=[];saved=[]
  consume=jax.jit(lambda st,p:ex.consume_frame_device(out.execution.ledger,frame,st,p,active=yes,tables=self.runtime['execution_compiled'].device,adapter_tables=self.runtime['public'].device,catalog=self.catalog))
  for owner,(stream,pub)in enumerate(owners):
   proof=cb.expected_request(after,s,pub,owner=owner,tape=out,data=self.helper.device.encore,active_tables=self.catalog.active.device,rules=self.f.rules.device,healthy=yes);self.assertEqual(proof.slot.tolist(),[-1]*b)
   ledger,got=consume(stream,pub);actual=[np.asarray(got.projection.frames.words)[:,r].reshape(-1,16)[np.asarray(got.projection.frames.valid)[:,r].reshape(-1)].tolist()for r in range(b)];saved.append(dict(owner=owner,error=got.projection.error.tolist(),admitted=got.projection.component_admitted.tolist(),words=actual,raw=out.words.tolist(),counts=out.count.tolist(),ledger_refusal=ledger.refusal.tolist()));Path('/tmp/pokezero-encore-no-last-full-current.json').write_text(json.dumps(saved)+'\n')
   self.assertEqual(got.projection.error.tolist(),[0]*b);self.assertTrue(bool(got.projection.component_admitted.all()));self.assertFalse(bool(got.public.cursor.needs_more.any()));self.assertTrue(bool(ex.report(ledger,tables=self.runtime['execution_compiled'].device).scoped_drained.all()));closed.append(ledger)
   for row in range(b):
    ids={a:f'p{a+1}a: '+self.catalog.species.names[a+1]for a in(0,1)};lines=[f'|switch|{ids[a]}|{self.catalog.species.names[a+1]}, L80, M|100/100'for a in(0,1)];wire,_=public_wire.project_lines(public_wire.WireState(),lines+['|turn|1'],**v4.wire_kwargs(self.catalog));who=1-int(c.pending.side[row]);_,packets=public_wire.project_lines(wire,[f'|move|{ids[who]}|Encore||[still]',f'|-fail|{ids[who]}','|','|upkeep','|turn|2'],**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for p in packets for e in p.events],np.int32);np.testing.assert_array_equal(actual[row],want)
  _,good=ex.join_consumed_owners(*closed,tables=self.runtime['execution_compiled'].device);self.assertTrue(bool(good.all()))
if __name__=='__main__':unittest.main()
