"""Prospective Substitute creation and two owning Baton boundaries; no turns/games."""
import ast,hashlib,json,os,subprocess,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
import test_gpu_engine_baton_pass_transaction as fixtures
from gpu_engine import engine_baton_pass_callbacks as cb,engine_substitute_callbacks as sub
from gpu_engine import engine_execution_completeness as ex,engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_request_source as request,engine_own_facts as own,engine_retained_encoder_driver as driver
from gpu_engine import engine_leftovers_callbacks as leftovers,engine_full_v4_event_device as full
from gpu_engine import engine_source_frame_witness as witness,public_wire,v4_encoder as v4,public_active,state,baton,entry,private_requests
from _showdown_root import showdown_root
BASE='5e038e81a779ada0d442b4f723e40c2afbeeb8a5'
COMPACT='/tmp/pokezero-baton-substitute-original-both-full-compact.json'

class BatonSubstituteGuards(unittest.TestCase):
 def test_exact_copy_cleanup_and_closed_other_volatiles(self):
  b=8;r=j.arange(b);side=r%2;z=j.zeros(b,j.int32)
  s=state.empty_state(b)._replace(hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),ability=j.ones((b,2,6),j.int32),base_ability=j.ones((b,2,6),j.int32))
  s=s._replace(substitute_hp=s.substitute_hp.at[r,side,z].set(j.array([0,1,25,26,-1,101,25,25],j.int32)),confusion_counter=s.confusion_counter.at[6,side[6],0].set(1),status=s.status.at[7,side[7],0].set(1))
  d=SimpleNamespace(quiet_abilities=j.array([False,True]),quiet_items=j.array([True]))
  self.assertEqual(cb._quiet(s,side,z,d,outgoing_toxic=True,outgoing_substitute=True).tolist(),[True]*4+[False]*4)
  self.assertEqual(cb._quiet(s,side,z,d,outgoing_toxic=True).tolist(),[True]+[False]*7)
  before=cb.snapshot(s);after=baton.copy_to_recipient(s,side,z+3,eligible=j.ones(b,bool));self.assertTrue(bool(cb.equal(cb.copied_private(before,side,z,z+3),cb.snapshot(after)).all()))
  departed=entry.depart(after,side,None,eligible=j.ones(b,bool));self.assertTrue(bool(cb.equal(cb.departed_private(cb.copied_private(before,side,z,z+3),side,z),cb.snapshot(departed)).all()))
  for n in ('hp','max_hp','status','pp','keys'):np.testing.assert_array_equal(getattr(s,n),getattr(departed,n))
  self.assertEqual(cb.PRIVATE_FIELDS[-1],'substitute_hp')
 def test_original_inert_copy_default_bytes_and_cold_imports(self):
  script="""const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),P=require(root+'/dist/sim/pokemon.js').Pokemon,c=d.conditions.get('substitute'),out=[];for(const hp of[1,13,25]){const s={volatiles:{substitute:{hp}},boosts:{spa:2},clearVolatile(){this.boosts={};this.volatiles={}}},t={volatiles:{},clearVolatile(){this.volatiles={}},battle:{dex:d,initEffectState:x=>x,singleEvent(){}} ,getVolatile:n=>d.conditions.get(n)};P.prototype.copyVolatileFrom.call(t,s,'copyvolatile');out.push({hp:t.volatiles.substitute.hp,old:s.volatiles,boosts:t.boosts});}console.log(JSON.stringify({noCopy:c.noCopy,onCopy:typeof c.onCopy,out}));"""
  value=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(value,dict(noCopy=False,onCopy='undefined',out=[dict(hp=n,old={},boosts=dict(spa=2))for n in(1,13,25)]))
  for n in ('transition.py','baton.py','entry.py','state.py','engine_retained_encoder_driver.py','public_active.py','engine_full_v4_event_device.py','engine_kernel_frame_risk.py'):
   self.assertEqual(Path('gpu_engine',n).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+n]),n)
  for names in [('gpu_engine.benchmark_retained_games','gpu_engine.engine_baton_pass_callbacks','gpu_engine.engine_full_v4_event_device'),('gpu_engine.engine_full_v4_event_device','gpu_engine.engine_baton_pass_callbacks','gpu_engine.benchmark_retained_games')]:
   out=subprocess.run([sys.executable,'-B','-c','import importlib;'+''.join('importlib.import_module('+repr(n)+');'for n in names)],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30);self.assertEqual(out.returncode,0,out.stderr)

class BatonSubstituteInputTests(unittest.TestCase):
 initial=fixtures.BatonTransactionTests.initial
 new_tape=fixtures.BatonTransactionTests.new_tape
 produced=fixtures.BatonTransactionTests.produced
 prepare_tape=fixtures.BatonTransactionTests.prepare_tape
 @classmethod
 def setUpClass(cls):
  fixtures.BatonTransactionTests.setUpClass.__func__(cls)
  cls.names=('batonpass','substitute','tailglow','agility')
  compiled=sub.compile_substitute(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  facts=own.compile_own_facts(cls.req,cls.catalog)
  cls.prepared=driver.prepare(cls.req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',substitute_compiled=compiled,substitute_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared)
  cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
  cls.consume=staticmethod(jax.jit(lambda l,f,s,p:ex.consume_frame_device(l,f,s,p,active=j.ones(len(l.stream_id),bool),tables=cls.runtime['execution_compiled'].device,adapter_tables=cls.runtime['public'].device,catalog=cls.catalog)))
 def drain(self,x,t,s,tag):
  b=len(s.hp);z=j.zeros(b,j.int32);yes=j.ones(b,bool);frame=life.LifecycleResult(s,t.cursor,t,life.snapshot(x.cursor),life.snapshot(t.cursor),~yes,z);new=[];ledgers=[]
  for owner,(stream,pub)in enumerate(x.owners):
   ledger,out=self.consume(t.execution.ledger,frame,stream,pub)
   record=dict(phase=tag,owner=owner,error=out.projection.error.tolist(),words=[np.asarray(out.projection.frames.words[:,row])[np.asarray(out.projection.frames.valid[:,row])].tolist()for row in range(b)],stage=ledger.stage.tolist(),refusal=ledger.refusal.tolist())
   self.actual.append(record);Path(COMPACT).write_text(json.dumps(self.actual)+'\n')
   self.assertEqual(out.projection.error.tolist(),[0]*b,record);new.append((out.projection.state,out.public));ledgers.append(ledger)
  return frame,new,ledgers
 def test_prospective_creation_both_full_two_boundaries_and_refusals(self):
  self.actual=[];x=self.initial();b=2;r=j.arange(b);side=r%2;z=j.zeros(b,j.int32);yes=j.ones(b,bool)
  # SAME original typed Substitute write, at its original scalar maxHP/4 cut.
  t=life.begin(self.new_tape(x),x.s,yes,~yes);move=z+self.metadata['vocabulary']['moves'].index('substitute');t=public.move(t,side,z,z,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_SELF,side,active=yes)
  maximum=x.s.max_hp[r,side,0];cost=j.maximum(1,maximum//4);write=sub.Write(x.s.hp[r,side,0],x.s.hp[r,side,0]-cost,maximum,x.s.status[r,side,0],x.s.substitute_hp[r,side,0],cost)
  t=sub.record_device(t,side,z,move,write,self.runtime['substitute_context'],eligible=yes);t=ex.capture_substitute_write(t,ticket,side,z,move,write,requested=yes)
  s=x.s._replace(hp=x.s.hp.at[r,side,0].set(write.new_hp),substitute_hp=x.s.substitute_hp.at[r,side,0].set(write.new_sub_hp))
  _,owners,ledgers=self.drain(x,t,s,'substitute');x.s=s;x.cursor=t.cursor;x.owners=owners;ledger=ex.join_consumed_owners(*ledgers,tables=self.runtime['execution_compiled'].device)[0]
  proof=cb.public_proof(s,x.cursor,tuple(p for _,p in owners),yes,self.helper.device.baton_pass);self.assertEqual(proof.known.tolist(),[[True,True]]*2);self.assertEqual(proof.substitute[r,side].tolist(),[True,True])
  # Public mismatches/other volatiles cannot be manufactured from private HP.
  p0=owners[0][1];sid=public_active.NAMES.index('substitute');other=public_active.NAMES.index('confusion')
  for column in (sid,other):
   changed=p0._replace(context=p0.context._replace(active=p0.context.active._replace(volatiles=p0.context.active.volatiles.at[r,side,column].set(column!=sid))))
   bad=cb.public_proof(s,x.cursor,(changed,owners[1][1]),yes,self.helper.device.baton_pass);self.assertEqual(bad.known[r,side].tolist(),[False,False])
  s=s._replace(pp=s.pp.at[r,side,0,0].add(-1));x.s=s;q=self.produced(s,x.orders,x.regs,x.cursor);h=cb.empty_history(s,x.cursor,j.stack(tuple(o.request_to_physical for o in x.orders),axis=1));t=self.prepare_tape(x,self.new_tape(x,ledger),h,q,j.zeros((b,2),j.int32));t=life.begin(t,s,yes,~yes);move=z+self.move;t=public.move(t,side,z,z,move,eligible=yes)
  for altered in (s._replace(substitute_hp=s.substitute_hp.at[0,side[0],0].set(0)),s._replace(confusion_counter=s.confusion_counter.at[0,side[0],0].set(1))):
   _,p=cb.begin_start(t,altered,side,move,move,self.runtime['direct_hit_context'],acts=yes,eligible=yes);self.assertEqual(p.valid.tolist(),[False,True])
  _,called=cb.begin_start(t,s,side,move,move+1,self.runtime['direct_hit_context'],acts=yes,eligible=yes);self.assertEqual(called.valid.tolist(),[False,False])
  t,p=cb.begin_start(t,s,side,move,move,self.runtime['direct_hit_context'],acts=yes,eligible=yes);self.assertEqual(p.valid.tolist(),[True,True])
  s=s._replace(baton_pass=s.baton_pass.at[r,side].set(True));t=cb.finish_start(t,p,s,yes,yes);s=s._replace(forced_switch=s.baton_pass,pending_residual=yes,turn=z+1);t=cb.finalize_request(t,s);self.assertEqual(cb.owned(t,t.count-1,cb.START).tolist(),[True,True])
  frame,owners,ledgers=self.drain(x,t,s,'start');q=self.produced(s,x.orders,x.regs,t.cursor);h=cb.mature(t.execution.baton_transaction.history,frame,tuple(ledgers),tuple(p for _,p in owners),q);self.assertEqual(h.ready.tolist(),[True,True]);x.s=s;x.cursor=t.cursor;x.owners=owners
  ledger=ex.join_consumed_owners(*ledgers,tables=self.runtime['execution_compiled'].device)[0];actions=j.zeros((b,2),j.int32).at[r,side].set(7)
  for bad_h in (h._replace(private=h.private._replace(substitute_hp=h.private.substitute_hp.at[0,side[0],0].add(1))),h._replace(frame=h.frame-1),h._replace(stream_id=h.stream_id+1)):
   bad=self.prepare_tape(x,self.new_tape(x,ledger),bad_h,q,actions);self.assertFalse(bool(bad.execution.baton_transaction.request_ok[0]))
  t=self.prepare_tape(x,self.new_tape(x,ledger),h,q,actions);self.assertEqual(t.execution.baton_transaction.request_ok.tolist(),[True,True]);t=life.begin(t,s,yes,~yes);target=z+3;t,p=cb.before_copy(t,s,side,target,self.runtime['direct_hit_context'],eligible=yes);self.assertEqual(p.valid.tolist(),[True,True]);copied=baton.copy_to_recipient(s,side,target,eligible=yes)
  bad=cb.after_copy(t,p,copied._replace(substitute_hp=copied.substitute_hp.at[0,side[0],3].add(1)));self.assertEqual(bad.execution.baton_transaction.record.known.tolist(),[False,True])
  t=cb.after_copy(t,p,copied);after=entry.depart(copied,side,self.table.entry,eligible=yes);after=after._replace(active=after.active.at[r,side].set(target),baton_pass=after.baton_pass.at[r,side].set(False),forced_switch=after.forced_switch.at[r,side].set(False));self.assertEqual(cb.reveal_matches(t,after,side,z,target).tolist(),[True,True]);t=life.installed(t,after,side,z,target,eligible=yes,replacement=yes,baton=yes);t=cb.finish_install(t,after,side,eligible=yes)
  self.assertEqual(cb.owned(t,t.count-1,cb.INSTALL).tolist(),[True,True]);rec=t.execution.baton_transaction.record;bad_rec=rec._replace(after=rec.after._replace(substitute_hp=rec.after.substitute_hp.at[0,side[0],3].add(1)));bad=t._replace(execution=t.execution._replace(baton_transaction=t.execution.baton_transaction._replace(record=bad_rec)));self.assertEqual(cb.owned(bad,t.count-1,cb.INSTALL).tolist(),[False,True]);self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  install,new,closed=self.drain(x,t,after,'install');self.assertEqual(after.substitute_hp[r,side,0].tolist(),[0,0]);self.assertEqual(after.substitute_hp[r,side,3].tolist(),[25,25])
  # The ORIGINAL parser/writer's complete words, including absolute sequence.
  for owner in (0,1):
   for row,seat in enumerate(side.tolist()):
    first=self.catalog.species.names[1];incoming=self.catalog.species.names[2];ids={n:f'p{n+1}a: '+first for n in (0,1)}
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[n]}|{first}, L80, M|100/100'for n in(0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog))
    phases=[('substitute',['|','|t:|1',f'|move|{ids[seat]}|Substitute|{ids[seat]}',f'|-start|{ids[seat]}|Substitute',f'|-damage|{ids[seat]}|75/100']),('start',['|','|t:|1',f'|move|{ids[seat]}|Baton Pass|{ids[seat]}']),('install',['|','|t:|1',f'|switch|p{seat+1}a: {incoming}|{incoming}, L80, M|100/100'])]
    for phase,lines in phases:
     wire,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));want=np.array([e.words()for p in packets for e in p.events],np.int32);record=next(v for v in self.actual if v['phase']==phase and v['owner']==owner);actual=np.array(record['words'][row],np.int32);record.setdefault('expected',{})[str(row)]=want.tolist();Path(COMPACT).write_text(json.dumps(self.actual)+'\n');np.testing.assert_array_equal(actual,want)
    pub=new[owner][1];self.assertTrue(bool(pub.context.active.volatiles[row,seat,sid]));self.assertTrue(bool(pub.mons.arrived_by_baton[row,seat]))
  orders=tuple(life.apply_installations(o,install)for o in x.orders);current=self.produced(after,orders,x.regs,t.cursor);self.assertEqual([q.actions.admitted.tolist()for q in current],[[True,True],[True,True]])

if __name__=='__main__':unittest.main()
