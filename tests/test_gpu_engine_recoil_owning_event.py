"""Prospective literal kernel component roots, never games/Native/models/replay."""
import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess,unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import transition as tr,engine_recoil_callbacks as recoil,engine_direct_hit_controls as direct
from gpu_engine import engine_kernel_frame_risk as risks,engine_execution_completeness as ex
from gpu_engine import engine_public_events as events,engine_lifecycle_callbacks as life,engine_full_v4_event_device as full
from gpu_engine import engine_public_event_device as base,v4_encoder as v4,public_wire
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
import test_gpu_engine_pressure_entry_owning as fixtures
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected

class RecoilOwningEventTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.PressureEntryOwningTests.setUpClass();cls.f=fixtures.PressureEntryOwningTests()
  cls.metadata=deepcopy(cls.f.metadata);v=cls.metadata['vocabulary'];v['items']=['_none'];v['abilities']=['_none','rockhead'];v['moves'].append('doubleedge')
  for row in cls.metadata['moves'][1:]:row.update(accuracy=100,priority=0,flags={},source_type='Normal')
  cls.move=len(v['moves'])-1
  cls.metadata['moves'].append(dict(id='doubleedge',base_power=120,type=1,category='Physical',source_type='Normal',accuracy=100,priority=0,flags={'contact':1,'protect':1},recoil=[1,3]))
  cls.metadata['type_chart']=np.ones((17,17)).tolist()
  old=cls.f.catalog;info=replace(old.dex.moves['bodyslam'],id='doubleedge',name='Double-Edge',type='Normal',base_power=120,recoil=True,effect_chance=0,effect_label='',status=None,target='normal')
  dex=ShowdownDex(dict(old.dex.moves,doubleedge=info),old.dex.species,old.dex.type_chart)
  vocab=CategoryVocabulary(tuple(sorted(set(old.vocabulary.tokens)|{'move:doubleedge','move_effect:recoil'})))
  cls.catalog=v4.compile_encoder(old.source,dex,vocab,checkpoint_sha256='a'*64,feature_masks=fixtures.encoder_fixture.V4EncoderTests.masks)
  cls.table=tr.tables_from_metadata(cls.metadata)
  cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=fixtures.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog);cls.account=ex.compile_accounting(cls.adapter)
  cls.direct=direct.compile_direct_hit_controls(engine_catalog=cls.producer);cls.helper=recoil.compile_recoil(engine_catalog=cls.producer,metadata=cls.metadata)
  cls.context=recoil.Context(cls.helper.device,direct.RegisteredContext(cls.direct.device,cls.account.device))
  cls.risk=risks.compile_risks(cls.producer,cls.account,direct_hit_compiled=cls.direct,recoil_compiled=cls.helper)
  cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
 def same(self,a,b):
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
 def state(self,b=2):
  s=self.f.state(b);return s._replace(done=j.zeros(b,bool),entry_pending=j.zeros((b,2),bool),
      hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),stats=j.full_like(s.stats,142).at[:,1,:,5].set(131),
      ability=j.zeros_like(s.ability),base_ability=j.zeros_like(s.base_ability),moves=j.full_like(s.moves,self.move),pp=j.full_like(s.pp,24))
 def initial(self,b=2):
  s=self.state(b);c=life.register(j.arange(801,801+b,dtype=j.int32),**self.kw);before=life.snapshot(c);t=life._tape(c,self.producer,64)
  for side in (0,1):t=life._reveal(t,s,j.full(b,side,j.int32),j.full(b,-1,j.int32),s.active[:,side],eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
  t=t._replace(cursor=t.cursor._replace(initialized=j.ones(b,bool)));t=life._turn(t,j.ones(b,bool))
  frame=life.LifecycleResult(s,t.cursor,t,before,life.snapshot(t.cursor),j.zeros(b,bool),j.zeros(b,j.int32))
  consumers=[]
  fast=jax.jit(lambda stream,public:full.project_and_fold(stream,frame,self.adapter.device,public,self.catalog))
  for owner in (0,1):
   stream=full.register(t.stream_id,j.full(b,owner,j.int32),compiled=self.adapter);public=v4.initialize(self.catalog,stream_ids=t.stream_id,owner_sides=j.full(b,owner,j.int32))
   result=fast(stream,public);self.assertEqual(result.projection.error.tolist(),[0]*b);consumers.append((result.projection.state,result.public))
  return s,t.cursor,tuple(consumers)
 def play(self,s,c,*,context=True,prior=1):
  b=s.hp.shape[0];t=life._tape(c,self.producer,64)._replace(refusal=j.full(b,prior,j.int32));ledger=ex.initialize(c.stream_id,compiled=self.account)
  t=ex.begin_on_tape(t,ledger,self.account.device,active=j.ones(b,bool));carry=risks.initialize_device(t.stream_id,self.risk.device);carry=risks.begin_frame_device(carry,t,self.risk.device,active=j.ones(b,bool))
  t=t._replace(execution=t.execution._replace(risks=risks.Context(carry,self.risk.device)))
  chance=fixtures.engine_fixture.EnginePublicEventsTests().chance(b)
  return tr.step_with_chance(s,j.zeros((b,2),j.int32),self.table,chance,_public_trace=t,_lifecycle_trace=True,
      _direct_hit_context=self.context.direct,_recoil_context=self.context if context else None)
 def test_actual_pair_is_literal_numerically_identical_and_both_original_full_owners_drain(self):
  s,c,consumers=self.initial();actual=self.play(s,c);chance=fixtures.engine_fixture.EnginePublicEventsTests().chance(2)
  plain=tr.step_with_chance(s,j.zeros((2,2),j.int32),self.table,chance);self.same(actual.step.state,plain.state)
  self.assertEqual((actual.callbacks.refusal&~1).tolist(),[0,0]);self.assertEqual(actual.callbacks.execution.risks.carry.risk.tolist(),[0,0])
  for row in (0,1):
   raw=np.asarray(actual.callbacks.words[row,:int(actual.callbacks.count[row])]);self.assertEqual((raw[:,0]==28).sum(),2)
   for r in raw[raw[:,0]==28]:self.assertGreater(int(r[6]),int(r[7]));self.assertGreater(int(r[7]),0)
  frame=life.LifecycleResult(actual.step.state,actual.callbacks.cursor,actual.callbacks,life.snapshot(c),life.snapshot(actual.callbacks.cursor),j.zeros(2,bool),j.zeros(2,j.int32))
  ledger=actual.callbacks.execution.ledger;closed=[]
  consume=jax.jit(lambda stream,public:ex.consume_frame_device(ledger,frame,stream,public,active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog))
  for owner,(stream,public) in enumerate(consumers):
   end,result=consume(stream,public);self.assertEqual(result.projection.error.tolist(),[0,0]);closed.append(end)
   self.assertTrue(bool(ex.report(end,tables=self.account.device).scoped_drained.all()));self.assertFalse(bool(result.projection.B4_qualified.any()))
   for row in (0,1):
    expected=self.reference_words(actual.callbacks,row,owner)
    np.testing.assert_array_equal(self.f.words(result.projection,row),expected)
  joined,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
  for row in (0,1):self.assertEqual(joined.stage[row,:int(joined.count[row])].tolist(),[3,3,3,3])
 def test_rockhead_targetko_actorko_prior_and_no_context_never_admit(self):
  s,c,_=self.initial();variants=[s._replace(ability=s.ability.at[:,0,3].set(1)),s._replace(hp=s.hp.at[:,1,5].set(1)),s._replace(hp=s.hp.at[:,0,3].set(1))]
  for state in variants:
   actual=self.play(state,c);self.assertTrue(bool(((actual.callbacks.refusal&~1)!=0).all()));self.assertTrue(bool((actual.callbacks.execution.risks.carry.risk!=0).all()))
  old=self.play(s,c,context=False);self.assertTrue(bool((old.callbacks.execution.risks.carry.risk&risks.SELF_HP).all()))
  sticky=self.play(s,c,prior=65);self.assertTrue(bool((sticky.callbacks.refusal&64).all()));self.assertTrue(bool((sticky.callbacks.execution.risks.carry.risk!=0).all()))
 def test_default_ast_rejects_hp_rng_keyword_mutations_and_recoil_requires_original_compiler(self):
  old_source=subprocess.check_output(['git','show','58ec3459:gpu_engine/transition.py'],text=True);current_source=Path('gpu_engine/transition.py').read_text()
  if '_hypothetical_profile' in current_source:
   from test_gpu_engine_v5_hypothetical_profile import normalized_hypothetical_default
   before=normalized_hypothetical_default(old_source,as_tree=True);after=normalized_hypothetical_default(current_source,as_tree=True)
  else:before=ast.parse(old_source);after=ast.parse(current_source)
  baseline=dump_default_program(before)
  self.assertEqual(baseline,dump_default_program(after));assert_hp_rng_mutations_detected(self,after,baseline)
  with self.assertRaises(ValueError):recoil.check_compiled(replace(self.helper))
  with self.assertRaises(ValueError):jax.jit(lambda dummy:recoil.check_compiled(self.helper))(j.array(0))
  self.assertEqual(self.risk.device.selected_risk[self.move].item()&risks.SELF_HP,0)
  closed=risks.compile_risks(self.producer,self.account,direct_hit_compiled=self.direct)
  self.assertTrue(bool(closed.device.selected_risk[self.move]&risks.SELF_HP))
 def pending(self,b=8,*,mutate=None,eligible=None):
  s=self.state(b);c=life.register(j.arange(901,901+b,dtype=j.int32),**self.kw)
  t=life._tape(c,self.producer,64)
  for side in (0,1):t=life._reveal(t,s,j.full(b,side,j.int32),j.full(b,-1,j.int32),s.active[:,side],eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
  t=t._replace(cursor=t.cursor._replace(initialized=j.ones(b,bool)));t=life._turn(t,j.ones(b,bool))
  ledger=ex.initialize(c.stream_id,compiled=self.account);t=ex.begin_on_tape(t,ledger,self.account.device,active=j.ones(b,bool))
  carry=risks.initialize_device(t.stream_id,self.risk.device);carry=risks.begin_frame_device(carry,t,self.risk.device,active=j.ones(b,bool))
  t=t._replace(execution=t.execution._replace(risks=risks.Context(carry,self.risk.device)))
  side=j.zeros(b,j.int32);actor=s.active[:,0];target=s.active[:,1];move=j.full(b,self.move,j.int32)
  t=events.move(t,side,actor,target,move,eligible=j.ones(b,bool));t,ticket=ex.open_attempt(t,ex.Site.DIRECT_HIT,1-side,active=j.ones(b,bool))
  w=direct.HitWrite(j.full(b,120,j.int32),j.zeros(b,bool),j.ones(b,bool),j.zeros(b,bool),j.ones(b,j.int32),j.ones(b,j.int32),j.zeros(b,bool),j.zeros(b,bool),j.zeros(b,bool),j.full(b,1000,j.int32),j.full(b,900,j.int32),j.full(b,1000,j.int32))
  if mutate:w=mutate(w)
  t=direct.record_registered_direct_hit_controls_device(t,side,actor,target,move,w,self.context.direct,eligible=j.ones(b,bool))
  t=events.condition(t,1-side,target,w.new_hp,w.maximum,j.zeros(b,j.int32),eligible=w.hit&~w.immune&~w.substitute)
  t=ex.capture_direct_hit_write(t,ticket,side,actor,target,move,w,requested=j.ones(b,bool))
  cap=risks.CapturedDirectHit(self.context.direct,ticket,side,actor,target,move,w)
  return recoil.prepare_device(t,cap,j.full(b,33,j.int32),j.zeros(b,j.int32),j.full(b,1000,j.int32),j.full(b,1000,j.int32),j.zeros(b,j.int32),self.context,eligible=j.ones(b,bool) if eligible is None else eligible)
 def write(self,p):
  b=len(p.recoil);return recoil.Write(p.captured.actor_physical,p.captured.target_physical,p.ability,p.old_hp,p.old_hp-p.recoil,p.maximum,p.status)
 def test_pending_is_not_credit_missing_final_write_and_zero_recoil_need_real_pair(self):
  t,p=self.pending();self.assertTrue(bool(recoil.matches_pending(t,p).all()))
  self.assertEqual(t.execution.ledger.stage[:,:2].tolist(),[[2,1]]*8)
  self.assertFalse(bool(ex.report(t.execution.ledger,tables=self.account.device).scoped_drained.any()))
  outcome=risks.Outcome(p.captured.write.power,p.captured.write.critical,p.captured.write.hit,p.captured.write.immune,
      p.captured.write.numerator,p.captured.write.denominator,j.zeros(8,bool),j.zeros(8,bool),j.zeros(8,bool),j.zeros(8,bool),j.zeros(8,j.int32),j.zeros(8,bool))
  # Explicit candidate with zero literal recoil and noPending is NOT exempt.
  result=risks.observe_outcome_device(t.execution.risks.carry,t,outcome,self.risk.device,eligible=j.ones(8,bool),active=j.ones(8,bool),captured_direct=p.captured)
  self.assertTrue(bool((result.risk&risks.SELF_HP).all()))
  out=recoil.record_device(t,p,self.write(p));self.assertEqual(out.execution.ledger.stage[:,:2].tolist(),[[2,2]]*8)
  self.assertEqual(out.words[0,int(out.count[0])-1].tolist(),[28,0,3,1,5,self.move,1000,967,1000,0])
 def test_miss_protect_immune_substitute_and_unrepresented_interleaving_never_validate(self):
  for mutate in (lambda w:w._replace(hit=j.zeros(8,bool)),lambda w:w._replace(protected=j.ones(8,bool)),lambda w:w._replace(immune=j.ones(8,bool)),lambda w:w._replace(substitute=j.ones(8,bool))):
   t,p=self.pending(mutate=mutate);self.assertFalse(bool(recoil.matches_pending(t,p).any()));self.assertTrue(bool((t.refusal&256).all()))
  t,p=self.pending();w=self.write(p)
  for bad in (t._replace(words=t.words.at[0,3,5].add(1)),t._replace(event_generation=t.event_generation.at[0,4,0].add(1)),
      t._replace(cursor=t.cursor._replace(generation=t.cursor.generation.at[0,0].add(1))),
      events.condition(t,p.captured.actor,p.captured.actor_physical,j.full(8,999,j.int32),p.maximum,p.status,eligible=j.ones(8,bool))):
   out=recoil.record_device(bad,p,w);self.assertTrue(bool(out.refusal[0]&256));self.assertNotEqual(int(out.execution.ledger.refusal[0]),0)
  changed=recoil.record_device(t,p,w._replace(old_hp=w.old_hp-1,new_hp=w.new_hp-1));self.assertTrue(bool((changed.refusal&256).all()))
 def test_b8_inactive_global_namespaces_and_current_write_mutants_refuse(self):
  mask=j.array([True,True,False,False,False,False,False,False]);t,p=self.pending(eligible=mask);out=recoil.record_device(t,p,self.write(p))
  self.assertEqual(out.count.tolist(),[6,6]+[5]*6)
  for a,b in zip(jax.tree.leaves(t),jax.tree.leaves(out)):
   if getattr(a,'ndim',0) and a.shape[0]==8 and a.shape!=(8,):np.testing.assert_array_equal(a[2:],b[2:])
  for field in ('binding','engine_binding','source_binding','move_binding','producer_binding'):
   data=p.context.data._replace(**{field:j.tile(getattr(p.context.data,field),(8,1))})
   with self.assertRaises(ValueError):recoil.record_device(t,p._replace(context=p.context._replace(data=data)),self.write(p))
  for field in ('ability','actor_physical','target_physical','maximum','status'):
   w=self.write(p);bad=w._replace(**{field:getattr(w,field)+1});refused=recoil.record_device(t,p,bad)
   self.assertTrue(bool((refused.refusal[:2]&256).all()))

 def reference_words(self,tape,row,owner):
  a=self.catalog.species.names[int(self.adapter.device.causal.species[1])];b=self.catalog.species.names[int(self.adapter.device.causal.species[2])]
  ident={0:'p1a: '+a,1:'p2a: '+b};names=(a,b)
  initial=[f'|switch|{ident[side]}|{names[side]}, L80, M|'+('1000/1000' if side==owner else '100/100') for side in (0,1)]+['|turn|1']
  wire,_=public_wire.project_lines(public_wire.WireState(),initial,**v4.wire_kwargs(self.catalog));lines=[]
  tokens=('', ' par',' brn',' psn',' tox',' slp',' frz')
  def health(side,hp,maximum,status):
   if side!=owner:
    hp=min(99,(100*hp+maximum-1)//maximum) if hp<maximum else 100;maximum=100
   return f'{hp}/{maximum}'+tokens[status]
  for raw in np.asarray(tape.words[row,:int(tape.count[row])]):
   k=int(raw[0])
   if k==events.MOVE:lines.append(f'|move|{ident[int(raw[1])]}|Double-Edge|{ident[int(raw[3])]}')
   elif 21<=k<=23:lines.append('|'+{21:'-crit',22:'-supereffective',23:'-resisted'}[k]+'|'+ident[int(raw[3])])
   elif k==events.MAIN_HP:lines.append(f'|-damage|{ident[int(raw[3])]}|'+health(int(raw[3]),int(raw[5]),int(raw[6]),int(raw[7])))
   elif k==28:lines.append(f'|-damage|{ident[int(raw[1])]}|'+health(int(raw[1]),int(raw[7]),int(raw[8]),int(raw[9]))+f'|[from] Recoil|[of] {ident[int(raw[3])]}')
   elif k==life.ACTION_END:lines.append('|')
   elif k==life.UPKEEP:lines.append('|upkeep')
   elif k==life.TURN:lines.append(f'|turn|{int(tape.event_turn[row,len(lines)])}')
   else:self.fail('unrepresented reference raw '+str(k))
  _,projections=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog))
  return np.array([event.words() for projection in projections for event in projection.events],np.int32)
 def test_pinned_original_gen3_recoil_floor_rockhead_and_capitalized_of_grammar(self):
  import shutil,json
  node=shutil.which('node')
  if node is None:self.skipTest('node unavailable')
  script=r"""
const path=require('node:path');const root=process.argv[1];
const {Dex}=require(path.join(root,'dist/sim/dex.js'));const {Scripts}=require(path.join(root,'dist/data/mods/gen3/scripts.js'));const {Battle}=require(path.join(root,'dist/sim/battle.js'));
const dex=Dex.mod('gen3'),move=dex.moves.get('doubleedge');const damage=[1,2,3,4,104];
const values=damage.map(x=>Scripts.actions.calcRecoilDamage.call({battle:{clampIntRange:(x,min)=>Math.max(x,min)}},x,move));
const lines=[];const actor={hp:267,isActive:true,toString:()=> 'p1a: Snorlax',getHealth:'233/267',damage(x){this.hp-=x;return x;}};
const foe={toString:()=> 'p2a: Donphan'},battle={gen:3,dex,clampIntRange:(x,min)=>Math.max(x,min),runEvent:(id,t,s,e,n)=>n,add:(...xs)=>lines.push(xs.map(String).join('|'))};
Battle.prototype.spreadDamage.call(battle,[34],[actor],foe,'recoil');
process.stdout.write(JSON.stringify({recoil:move.recoil,values,lines,rockhead:dex.abilities.get('rockhead').onDamage.call({activeMove:move},34,actor,foe,{id:'recoil'})}));
"""
  actual=json.loads(subprocess.check_output([node,'-e',script,'/Users/scott/workspace/agents/pokemon-showdown-timing-smoke-f76228a'],text=True))
  self.assertEqual(actual,dict(recoil=[1,3],values=[1,1,1,1,34],lines=['-damage|p1a: Snorlax|233/267|[from] Recoil|[of] p2a: Donphan'],rockhead=None))
