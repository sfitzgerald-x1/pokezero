"""RockHead source-owned nonempty private transaction, exact zero public output."""
from _showdown_root import showdown_root_str
import ast,json,subprocess,unittest
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import test_gpu_engine_recoil_ko_pair as pair_fixture
from gpu_engine import engine_recoil_callbacks as recoil,engine_direct_hit_controls as direct
from gpu_engine import engine_kernel_frame_risk as risk,engine_execution_completeness as ex
from gpu_engine import engine_public_events as events,engine_lifecycle_callbacks as life
from gpu_engine import engine_full_v4_event_device as full,public_wire,v4_encoder as v4,damage_moves
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
ROOT=showdown_root_str()
BASE='1376098de0a5fc55e78660bbf3f90c446a26cea5'
class RockHeadSilentInputTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  pair_fixture.RecoilKOPairTests.setUpClass();cls.a=pair_fixture.RecoilKOPairTests();cls.f=cls.a.f;cls.context=cls.a.context;cls.helper=cls.a.helper;cls.tables=cls.a.risk.device
 def initial(self,b,consume):
  s,_,_=self.a.initial(b);rows=j.arange(b);actor=rows%2;ap=s.active[rows,actor]
  s=s._replace(ability=s.ability.at[rows,actor,ap].set(1),base_ability=s.base_ability.at[rows,actor,ap].set(1))
  c=life.register(j.arange(71001,71001+b,dtype=j.int32),**self.f.kw);before=life.snapshot(c);t=life._tape(c,self.f.producer,64)
  for side in(0,1):t=life._reveal(t,s,j.full(b,side,j.int32),j.full(b,-1,j.int32),s.active[:,side],eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
  t=t._replace(cursor=t.cursor._replace(initialized=j.ones(b,bool)));t=life._turn(t,j.ones(b,bool));consumers=[]
  if consume:
   frame=life.LifecycleResult(s,t.cursor,t,before,life.snapshot(t.cursor),j.zeros(b,bool),j.zeros(b,j.int32))
   fold=jax.jit(lambda stream,pub:full.project_and_fold(stream,frame,self.f.adapter.device,pub,self.f.catalog))
   for owner in(0,1):
    stream=full.register(t.stream_id,j.full(b,owner,j.int32),compiled=self.f.adapter);pub=v4.initialize(self.f.catalog,stream_ids=t.stream_id,owner_sides=j.full(b,owner,j.int32));got=fold(stream,pub);self.assertEqual(got.projection.error.tolist(),[0]*b);consumers.append((got.projection.state,got.public))
  return s,t.cursor,consumers
 def pending(self,b=8,*,consume=False,change=None):
  s,c,consumers=self.initial(b,consume);rows=j.arange(b);actor=rows%2;ap=s.active[rows,actor];tp=s.active[rows,1-actor];move=j.full(b,self.f.move,j.int32);yes=j.ones(b,bool);no=~yes;z=j.zeros(b,j.int32)
  t=life._tape(c,self.f.producer,64);t=ex.begin_on_tape(t,ex.initialize(c.stream_id,compiled=self.f.account),self.f.account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,self.tables),t,self.tables,active=yes);t=t._replace(execution=t.execution._replace(risks=risk.Context(carry,self.tables)))
  t=events.move(t,actor,ap,tp,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.DIRECT_HIT,1-actor,active=yes)
  w=direct.HitWrite(z+120,yes,yes,no,z+1,z+2,no,no,no,z+60,z+40,z+100)
  if change:w=change(w)
  t=direct.record_registered_direct_hit_controls_device(t,actor,ap,tp,move,w,self.context.direct,eligible=yes)
  t=events.condition(t,1-actor,tp,w.new_hp,w.maximum,z,eligible=yes);t=ex.capture_direct_hit_write(t,ticket,actor,ap,tp,move,w,requested=yes,data=self.context.direct.data)
  cap=risk.CapturedDirectHit(self.context.direct,ticket,actor,ap,tp,move,w);ability=s.ability[rows,actor,ap];amount=damage_moves.recoil_damage(self.f.table.damage_moves,move,ability,w.old_hp-w.new_hp)
  raw_before=t;t,p=recoil.prepare_device(t,cap,amount,ability,s.hp[rows,actor,ap],s.max_hp[rows,actor,ap],s.status[rows,actor,ap],self.context,eligible=yes,caller_move=move)
  return s,c,consumers,raw_before,t,p
 def write(self,p):return recoil.Write(p.captured.actor_physical,p.captured.target_physical,p.ability,p.old_hp,j.maximum(0,p.old_hp-p.recoil),p.maximum,p.status)
 def test_original_source_literal_same_write_and_meaningful_refusals(self):
  script=r"""const root=process.argv[1],a=require(root+'/dist/sim/dex.js').Dex.mod('gen3').abilities.get('rockhead'),lines=[],b={activeMove:{id:'doubleedge'},add:(...x)=>lines.push(x)};let absent=false;try{a.onDamage.call({activeMove:null},20,{},{},{id:'recoil'})}catch(e){absent=true}console.log(JSON.stringify({suppressed:a.onDamage.call(b,20,{},{},{id:'recoil'})===null,struggle:a.onDamage.call({...b,activeMove:{id:'struggle'}},20,{},{},{id:'recoil'})===undefined,other:a.onDamage.call(b,20,{},{},{id:'roughskin'})===undefined,absent,lines}));"""
  self.assertEqual(json.loads(subprocess.check_output(['node','-e',script,ROOT],text=True)),dict(suppressed=True,struggle=True,other=True,absent=True,lines=[]))
  s,c,_,raw,t,p=self.pending();w=self.write(p);self.assertTrue(bool((p.recoil==0).all()));self.assertTrue(bool(recoil.matches_pending(t,p).all()));out=recoil.record_device(t,p,w);at=int(out.count[0])-1;self.assertTrue(bool(recoil.owned_silent(out,at).all()));self.assertEqual(out.refusal.tolist(),[1]*8)
  o=risk.Outcome(p.captured.write.power,p.captured.write.critical,p.captured.write.hit,p.captured.write.immune,p.captured.write.numerator,p.captured.write.denominator,j.zeros(8,bool),j.zeros(8,bool),j.zeros(8,bool),j.zeros(8,bool),p.recoil,j.zeros(8,bool))
  observed=risk.outcome_on_tape(t,o,eligible=p.requested,active=p.requested,captured_direct=p.captured,pending_recoil=p);self.assertEqual(observed.execution.risks.carry.risk.tolist(),[0]*8)
  for bad in (p._replace(caller_move=None),p._replace(caller_move=p.captured.move+1),p._replace(ability=j.zeros(8,j.int32)),p._replace(recoil=j.ones(8,j.int32)),p._replace(context=self.a.f.context)):
   self.assertFalse(bool(recoil.matches_pending(t,bad).any()))
  for change in (lambda w:w._replace(new_hp=w.old_hp),lambda w:w._replace(new_hp=j.zeros(8,j.int32))):
   _,_,_,_,bad,pbad=self.pending(change=change);self.assertFalse(bool(recoil.matches_pending(bad,pbad).any()))
  for key in('new_hp','old_hp','maximum','status','ability'):
   val=getattr(w,key);bad=recoil.record_device(t,p,w._replace(**{key:val+1}));self.assertTrue(bool((bad.execution.ledger.refusal!=0).all()),key)
  ledger=out.execution.ledger
  for fields in (dict(no_output_facts=ledger.no_output_facts.at[0,1,0].set(0)),dict(no_output_binding=ledger.no_output_binding.at[0,1,0].add(j.uint32(1))),dict(no_output_facts=ledger.no_output_facts.at[0,1,4].set(40)),dict(claimed=ledger.claimed.at[0,3].set(2))):
   bad=out._replace(execution=out.execution._replace(ledger=ledger._replace(**fields)));self.assertFalse(bool(recoil.owned_silent(bad,at)[0]))
  for key in('engine_binding','source_binding','move_binding','producer_binding'):
   with self.assertRaises(ValueError):recoil.owned_silent(out._replace(**{key:j.tile(getattr(out,key),(8,1))}),at)
  sticky=recoil.record_device(t._replace(refusal=t.refusal|64),p,w);self.assertTrue(bool((sticky.refusal&64).all()))
  # Existing positive and target-KO paths still emit only their original spans.
  pos,kp,rp=self.a.pending();done=self.a.after(pos,kp,rp);self.assertEqual(done.execution.ledger.stage[:,:2].tolist(),[[2,2]]*8)
  before=dump_default_program(ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True)))
  historical=ast.parse(subprocess.check_output(['git','show','c979e060:gpu_engine/transition.py'],text=True))
  self.assertEqual(before,dump_default_program(historical));assert_hp_rng_mutations_detected(self,historical,before)
  current=ast.parse(Path('gpu_engine/transition.py').read_text())
  integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True))
  self.assertEqual(ast.dump(current),ast.dump(integrated));assert_hp_rng_mutations_detected(self,current,dump_default_program(integrated))
  for modules in(('benchmark_retained_games','engine_lifecycle_event_device','engine_full_v4_event_device'),('engine_full_v4_event_device','engine_lifecycle_event_device','benchmark_retained_games')):subprocess.run([__import__('sys').executable,'-c',';'.join('import gpu_engine.'+n for n in modules)],check=True,capture_output=True)
 def test_both_full_exact_original_zero_recoil_words_and_stage3(self):
  b=2;s,c,consumers,_,t,p=self.pending(b,consume=True);w=self.write(p);out=recoil.record_device(t,p,w);rows=j.arange(b);cap=p.captured
  self.assertEqual(w.new_hp.tolist(),w.old_hp.tolist());self.assertTrue(bool(recoil.owned_silent(out,int(out.count[0])-1).all()))
  after=s._replace(hp=s.hp.at[rows,1-cap.actor,cap.target_physical].set(cap.write.new_hp));frame=life.LifecycleResult(after,out.cursor,out,life.snapshot(c),life.snapshot(out.cursor),j.zeros(b,bool),j.zeros(b,j.int32));ledger=out.execution.ledger;closed=[];saved=[]
  consume=jax.jit(lambda stream,pub:ex.consume_frame_device(ledger,frame,stream,pub,active=j.ones(b,bool),tables=self.f.account.device,adapter_tables=self.f.adapter.device,catalog=self.f.catalog))
  for owner,(stream,pub)in enumerate(consumers):
   end,got=consume(stream,pub);saved.append(dict(owner=owner,error=got.projection.error.tolist(),words=[self.f.f.words(got.projection,row).tolist()for row in range(b)]));Path('/tmp/pokezero-rock-head-full-current.json').write_text(json.dumps(saved)+'\n');self.assertEqual(got.projection.error.tolist(),[0]*b);self.assertTrue(bool(ex.report(end,tables=self.f.account.device).scoped_drained.all()));closed.append(end)
   for row in range(b):
    identities={};initial=[]
    for side in(0,1):
     species=int(self.f.adapter.device.causal.species[side+1]);name=self.f.catalog.species.names[species];identities[side]=f'p{side+1}a: '+name;phys=int(s.active[row,side]);hp=int(s.hp[row,side,phys]);maximum=int(s.max_hp[row,side,phys]);health=f'{hp}/{maximum}'if side==owner else f'{min(99,(100*hp+maximum-1)//maximum)if hp<maximum else 100}/100';initial.append(f'|switch|{identities[side]}|{name}, L80, M|{health}')
    who=int(cap.actor[row]);foe=1-who;hp=int(cap.write.new_hp[row]);maximum=int(cap.write.maximum[row]);health=f'{hp}/{maximum}'if foe==owner else f'{(100*hp+maximum-1)//maximum}/100'
    wire,_=public_wire.project_lines(public_wire.WireState(),initial+['|turn|1'],**v4.wire_kwargs(self.f.catalog));lines=[f'|move|{identities[who]}|Double-Edge|{identities[foe]}',f'|-crit|{identities[foe]}',f'|-resisted|{identities[foe]}',f'|-damage|{identities[foe]}|{health}'];_,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.f.catalog));want=np.asarray([e.words()for packet in packets for e in packet.events],np.int32);np.testing.assert_array_equal(self.f.f.words(got.projection,row),want)
  _,good=ex.join_consumed_owners(*closed,tables=self.f.account.device);self.assertTrue(bool(good.all()))
if __name__=='__main__':unittest.main()
