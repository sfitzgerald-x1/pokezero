"""Bounded source helper packets, never a game or historical reconstruction."""
import json,subprocess,unittest
from pathlib import Path
import jax.numpy as j
import numpy as np
from gpu_engine import engine_self_move_callbacks as sm,engine_sleep_talk_callbacks as talk
from gpu_engine import engine_lifecycle_callbacks as life,engine_full_v4_event_device as full,engine_source_frame_witness as witness,engine_kernel_frame_risk as risk
from gpu_engine import public_wire,v4_encoder as v4
import test_gpu_engine_sleep_talk_toxic_child as fixture
from _showdown_root import showdown_root

class SleepTalkRestFailureTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):fixture.SleepTalkToxicChildTests.setUpClass.__func__(cls)
 component=fixture.SleepTalkToxicChildTests.component
 def rest(self,*,initial_public=False,alter=None):
  s,after,before,side,t,c,consumers,cursor=self.component(chosen='rest',initial_public=initial_public,actor_hp=j.where(j.arange(8)%2==0,65,100))
  self.assertTrue(bool(talk.captured_matches(t,c).all()))
  yes=j.ones(8,bool);rows=j.arange(8);at=(rows,side,c.pending.physical)
  t,p=sm.begin(t,after,side,c.called,self.runtime['direct_hit_context'],code=sm.REST_FAIL,eligible=yes,secondary=yes,caller_move=c.pending.caller)
  self.assertEqual(p.code.tolist(),[sm.REST_ASLEEP_FAIL]*8)
  if alter is not None:after=alter(after,at)
  t,cap=sm.finish(t,p,after)
  t=risk.self_heal_on_tape(t,yes,active=yes,captured_self_move=cap)
  return s,after,t,cap,c,consumers,cursor
 def test_original_ontry_and_guard_join(self):
  script="const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3');let touched=0;console.log(JSON.stringify({result:d.moves.get('rest').onTry.call({add:()=>touched++},{status:'slp',hasAbility:()=>false,hp:65,maxhp:100}),touched}));"
  self.assertEqual(json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True)),dict(result=False,touched=0))
  s,after,t,c,parent,_,_=self.rest();self.assertTrue(bool(sm.captured_matches(t,c).all()));self.assertTrue(bool(sm.owned(t,3).all()));self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  self.assertEqual(t.count.tolist(),[4]*8);self.assertEqual(t.execution.ledger.refusal.tolist(),[0]*8);self.assertEqual(t.execution.risks.carry.risk.tolist(),[0]*8)
  np.testing.assert_array_equal(c.raw[:,8:10],np.full((8,2),2));np.testing.assert_array_equal(after.pp[j.arange(8),parent.pending.side,parent.pending.physical],parent.pending.pp.at[j.arange(8),parent.pending.slot].add(-1))
  rt=t.execution.risks.tables
  disabled=t._replace(execution=t.execution._replace(risks=t.execution.risks._replace(tables=rt._replace(sleep_talk_rest_failure=j.array(False)))))
  self.assertFalse(bool(sm.owned(disabled,3).any()));self.assertFalse(bool(talk.owned(disabled,2).any()))
  for field in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=t._replace(**{field:getattr(t,field).at[0].add(1)});self.assertFalse(bool(sm.owned(bad,3).any()))
  for col in (1,2,4,7,8,9):
   words=t.words.at[0,3,col].add(1);ledger=t.execution.ledger._replace(words=t.execution.ledger.words.at[0,3,col].add(1));bad=t._replace(words=words,execution=t.execution._replace(ledger=ledger));self.assertEqual(sm.owned(bad,3).tolist(),[False]+[True]*7)
  # A coherent transport edit cannot replace the parent's same clock/selection.
  words=t.words.at[0,1,5].set(self.metadata['vocabulary']['moves'].index('wish'));ledger=t.execution.ledger._replace(words=t.execution.ledger.words.at[0,1,5].set(words[0,1,5]));bad=t._replace(words=words,execution=t.execution._replace(ledger=ledger));self.assertEqual(sm.owned(bad,3).tolist(),[False]+[True]*7)
  for name in ('hp','status_counter','sleep_skipped','rest_origin','ability','item','volatile','wish_counter','moves','pp'):
   def change(st,at,name=name):
    x=getattr(st,name);ix=(j.arange(8),at[1])if name=='wish_counter'else at
    val=x[ix];x=x.at[ix].set(~val if val.dtype==j.bool_ else val+1);return st._replace(**{name:x})
   _,_,bad,cap,_,_,_=self.rest(alter=change);self.assertFalse(bool(sm.captured_matches(bad,cap).any()));self.assertTrue(bool((bad.execution.ledger.refusal!=0).all()))
 def test_both_full_original_bare_failure(self):
  s,after,t,cap,parent,consumers,cursor=self.rest(initial_public=True);yes=j.ones(8,bool)
  t=life.action_end(t,yes);t=life.finish(t,after,live=yes,replacement=~yes,pending_before=~yes,residual_ready=yes,defer_residual=~yes,turn_context=yes)
  frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(cursor),life.snapshot(t.cursor),~yes,j.zeros(8,j.int32));saved=[]
  for owner,(st,pub)in enumerate(consumers):
   got=self.fold(st,frame,pub);self.assertEqual(got.projection.error.tolist(),[0]*8);self.assertTrue(bool(got.projection.component_admitted.all()))
   for row in range(8):
    actor=int(parent.pending.side[row]);other=1-actor;name=self.catalog.species.names[actor+1];foename=self.catalog.species.names[other+1];ident=f'p{actor+1}a: {name}';foe=f'p{other+1}a: {foename}'
    switch=[f'|switch|p1a: {self.catalog.species.names[1]}|{self.catalog.species.names[1]}, L80, M|'+('65/100'if actor==0 else '100/100')+(' slp'if actor==0 else ''),f'|switch|p2a: {self.catalog.species.names[2]}|{self.catalog.species.names[2]}, L80, M|100/100'+(' slp'if actor==1 else '')]
    wire,_=public_wire.project_lines(public_wire.WireState(),switch+['|turn|1'],**v4.wire_kwargs(self.catalog))
    lines=[f'|cant|{ident}|slp',f'|move|{ident}|Sleep Talk|{ident}',f'|move|{ident}|Rest||[from] sleeptalk|[still]',f'|-fail|{ident}','|','|upkeep','|turn|2']
    _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));want=np.asarray([event.words()for packet in packets for event in packet.events],np.int32);want[:,1]+=int(st.sequence[row])-int(want[0,1])+1
    actual=np.asarray(got.projection.frames.words)[:,row].reshape(-1,16)[np.asarray(got.projection.frames.valid)[:,row].reshape(-1)]
    saved.append(dict(owner=owner,row=row,actual=actual.tolist(),expected=want.tolist(),raw=t.words[row,:int(t.count[row])].tolist()))
    Path('/tmp/pokezero-sleep-talk-rest-failure-both-full-current.json').write_text(json.dumps(saved)+'\n')
    np.testing.assert_array_equal(actual,want)
  Path('/tmp/pokezero-sleep-talk-rest-failure-both-full-compact.json').write_text(json.dumps(dict(scope='Typed source components only, no game/replay or historical cure',packets=saved))+'\n')
if __name__=='__main__':unittest.main()
