"""Prospective literal entry fixtures only; no games/Native/models or replay."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import entry,engine_trace_callbacks as trace,engine_public_events as events
from gpu_engine import engine_lifecycle_callbacks as life,engine_execution_completeness as ex
from gpu_engine import engine_weather_speed_profile as speed,engine_full_v4_event_device as full
from gpu_engine import engine_public_event_device as base,public_belief,v4_encoder as v4,public_wire
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from pokezero.category_vocab import CategoryVocabulary
import test_gpu_engine_pressure_entry_owning as fixture
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected

class TraceOwningEventTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.PressureEntryOwningTests.setUpClass();cls.f=fixture.PressureEntryOwningTests()
  cls.metadata=deepcopy(cls.f.metadata);cls.metadata['vocabulary']['abilities']=['_none','pressure','trace','sturdy']
  original=cls.f.catalog
  variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('literal-'+name,'fixture',name,'fixture',80,
      ('Surf',),'Sturdy' if name=='absol' else 'Trace','Leftovers'),)) for name in ('snorlax','absol','gardevoir')}
  source=public_belief.compile_source(Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-trace'),variants))
  vocabulary=CategoryVocabulary(tuple(sorted(set(original.vocabulary.tokens)|{'ability:sturdy','belief:possible_ability:sturdy'})))
  cls.catalog=v4.compile_encoder(source,original.dex,vocabulary,checkpoint_sha256='a'*64,feature_masks=fixture.encoder_fixture.V4EncoderTests.masks)
  cls.table=entry.from_metadata(cls.metadata);cls.engine=cls.f.engine._replace(entry=cls.table)
  cls.producer=events.compile_catalog(cls.metadata,cls.engine,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=fixture.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog);cls.account=ex.compile_accounting(cls.adapter)
  cls.fast=staticmethod(jax.jit(lambda s,t,a,z:full.project_device(s,t,a,z,cls.adapter.device)))
 def same(self,a,b):
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
 def fixture(self,b=2):
  s=self.f.state(b);s=s._replace(done=j.zeros(b,bool),ability=j.full_like(s.ability,3).at[:,0,3].set(2),
      base_ability=j.full_like(s.base_ability,3).at[:,0,3].set(2),stats=j.full_like(s.stats,142).at[:,1,:,5].set(131))
  cursor=life.register(j.arange(601,601+b,dtype=j.int32),catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.array(self.producer.engine_binding,np.uint32))
  before=life.snapshot(cursor);t=life._tape(cursor,self.producer,64)
  ledger=ex.initialize(cursor.stream_id,compiled=self.account,capacity=32,receipt_capacity=64)
  t=ex.begin_on_tape(t,ledger,self.account.device,active=j.ones(b,bool))
  for side in (0,1):t=life._reveal(t,s,j.full(b,side,j.int32),j.full(b,-1,j.int32),s.active[:,side],eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
  d=speed.ProfileData(j.ones(8,j.uint32),t.engine_binding,t.source_binding,j.ones(4,bool),j.ones(1,bool),j.ones(1,bool),j.ones(4,bool))
  born=speed.qualify_birth_device(s,t.stream_id,d)
  return s,t,before,d,born
 def copied(self,b=2,active=None):
  s,t,before,d,born=self.fixture(b);mask=j.ones(b,bool) if active is None else active
  after,ko,out=entry.enter(s,0,self.table,eligible=mask,hazards=False,_weather_trace=t)
  return s,after,out,before,d,born
 def test_original_root_cause_and_actual_literal_copy_numeric_identity(self):
  s,after,out,before,d,born=self.copied()
  plain,ko=entry.enter(s,0,self.table,eligible=j.ones(2,bool),hazards=False);self.same(plain,after)
  self.assertEqual(after.ability[:,0,3].tolist(),[3,3]);self.assertEqual(after.base_ability[:,0,3].tolist(),[2,2])
  self.assertEqual(out.count.tolist(),[3,3]);self.assertEqual((out.refusal&~1).tolist(),[0,0])
  self.assertEqual(out.words[0,2].tolist(),[27,2,3,0,3,1,5,3,0,0])
  self.assertEqual(out.execution.ledger.stage[:,0].tolist(),[2,2])
  self.assertEqual(speed.retain_device(born,after,out.stream_id,d).qualified_birth.tolist(),[False,False])
  kept=speed.retain_frame_device(born,after,out.stream_id,d,out)
  self.assertEqual(kept.qualified_birth.tolist(),[True,True]);self.assertEqual(kept.current_ability[:,0,3].tolist(),[3,3])
  self.assertEqual(speed.retain_device(kept,after,out.stream_id,d).qualified_birth.tolist(),[True,True])
 def test_full_both_owners_exact_original_writer_and_no_private_payload(self):
  s,after,t,before,d,born=self.copied()
  t=t._replace(cursor=t.cursor._replace(initialized=j.ones(2,bool)));t=life._turn(t,j.ones(2,bool));end=life.snapshot(t.cursor)
  stream=full.register(t.stream_id,j.array([0,1],j.int32),compiled=self.adapter)
  p=self.fast(stream,t,before,end)
  self.assertEqual(p.error.tolist(),[0,0])
  for row in (0,1):
   a=self.catalog.species.names[int(self.adapter.device.causal.species[1])];b=self.catalog.species.names[int(self.adapter.device.causal.species[2])]
   lines=[f'|switch|p1a: {a}|{a}, L80, M|100/100',f'|switch|p2a: {b}|{b}, L80, M|100/100',
     f'|-ability|p1a: {a}|Sturdy|Trace|[from] ability: Trace|[of] p2a: {b}','|turn|1']
   _,ps=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog))
   expected=np.array([e.words() for pp in ps for e in pp.events],np.int32)
   np.testing.assert_array_equal(self.f.words(p,row),expected)
  self.assertFalse(bool(p.B4_qualified.any()))
 def test_b8_inactive_unsupported_prior_stale_claim_and_generation(self):
  mask=j.array([True]*4+[False]*4);s,after,out,before,d,born=self.copied(8,mask)
  self.assertEqual(out.count.tolist(),[3]*4+[2]*4)
  kept=speed.retain_frame_device(born,after,out.stream_id,d,out);self.assertTrue(bool(kept.qualified_birth.all()))
  # Actual unsupported Trace->Pressure is numerically unchanged yet UNKNOWN.
  original,t,_,d,birth=self.fixture(8);original=original._replace(ability=original.ability.at[0,1,5].set(1),base_ability=original.base_ability.at[0,1,5].set(1))
  t=t._replace(refusal=t.refusal.at[1].set(65));result,_,bad=entry.enter(original,0,self.table,eligible=mask,hazards=False,_weather_trace=t)
  self.assertTrue(bool(bad.refusal[0]&256));self.assertTrue(bool(bad.refusal[1]&64))
  for changed in (out._replace(event_generation=out.event_generation.at[0,2,0].add(1)),
    out._replace(execution=out.execution._replace(ledger=out.execution.ledger._replace(claimed=out.execution.ledger.claimed.at[0,2].set(0)))),
    out._replace(words=out.words.at[0,2,6].set(4))):
   retained=speed.retain_frame_device(born,after,out.stream_id,d,changed)
   self.assertEqual(retained.qualified_birth.tolist(),[False]+[True]*7)
  failed=born._replace(qualified_birth=j.array([False]+[True]*7))
  self.assertFalse(bool(speed.retain_frame_device(failed,after,out.stream_id,d,out).qualified_birth[0]))
 def test_default_numerical_AST_and_pinned_trace_grammar(self):
  old=subprocess.check_output(['git','show','cc42bce4:gpu_engine/transition.py'],text=True)
  current_text=Path('gpu_engine/transition.py').read_text()
  # Private assembly may also include the separately tested hypothetical-only
  # observer seam. Normalize that explicit extension before the unchanged
  # default-observer proof; real HP/RNG/keywords remain in both checks.
  if '_hypothetical_profile' in current_text:
   from test_gpu_engine_v5_hypothetical_profile import normalized_hypothetical_default
   def normalized(text):
    return normalized_hypothetical_default(text,as_tree=True)
   current=normalized(current_text);old_tree=normalized(old)
  else:current=ast.parse(current_text);old_tree=ast.parse(old)
  baseline=dump_default_program(old_tree)
  self.assertEqual(baseline,dump_default_program(current))
  assert_hp_rng_mutations_detected(self,current,baseline)
  def enter_body(text):
   return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='enter')
  old_entry=subprocess.check_output(['git','show','cc42bce4:gpu_engine/entry.py'],text=True)
  self.assertEqual(dump_default_program(enter_body(old_entry)),dump_default_program(enter_body(Path('gpu_engine/entry.py').read_text())))
  root=Path('/Users/scott/workspace/agents/pokemon-showdown-timing-smoke-f76228a')
  callback=(root/'data/mods/gen3/abilities.ts').read_text().split('\ttrace: {',1)[1].split('\n\t},',1)[0]
  self.assertIn('pokemon.setAbility(ability, target)',callback);self.assertIn('onUpdate: undefined',callback)
  self.assertIn('this.battle.gen > 3',(root/'sim/pokemon.ts').read_text())
 def test_actual_constructor_seam_and_later_switch_restoration_no_stale_copy(self):
  s,t,before,d,born=self.fixture();s=s._replace(active=j.zeros((2,2),j.int32),entry_pending=j.ones((2,2),bool),
      ability=s.ability.at[:,0,3].set(3).at[:,0,0].set(2),base_ability=s.base_ability.at[:,0,3].set(3).at[:,0,0].set(2))
  fresh=life.register(t.stream_id,catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.array(self.producer.engine_binding,np.uint32))
  record=speed.qualify_birth_device(s,t.stream_id,d)
  actual=life.initialize_with_lifecycle(s,self.engine,cursor=fresh,catalog=self.producer,expected_binding=self.producer.binding,
      expected_engine_binding=np.array(self.producer.engine_binding,np.uint32),tie_coin=j.zeros(2,j.int32),execution_compiled=self.account)
  plain=fixture.engine_fixture.EnginePublicEventsTests.t.initialize(s,self.engine,tie_coin=j.zeros(2,j.int32))
  self.same(actual.state,plain)
  self.assertEqual((actual.callbacks.refusal&~1).tolist(),[0,0])
  current=speed.retain_frame_device(record,actual.state,t.stream_id,d,actual.callbacks)
  self.assertTrue(bool(current.qualified_birth.all()));self.assertEqual(current.current_ability[:,0,0].tolist(),[3,3])
  # Genuine original FULL folds (separate owner streams), then stage3 join.
  frame=actual;ledger=frame.callbacks.execution.ledger;owners=[]
  fast=jax.jit(lambda stream,public:ex.consume_frame_device(ledger,frame,stream,public,active=j.ones(2,bool),
      tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog))
  for owner in (0,1):
   stream=full.register(t.stream_id,j.full(2,owner,j.int32),compiled=self.adapter)
   public=v4.initialize(self.catalog,stream_ids=t.stream_id,owner_sides=j.full(2,owner,j.int32))
   account,consumed=fast(stream,public);owners.append(account)
   self.assertEqual(consumed.projection.error.tolist(),[0,0]);self.assertEqual(account.stage[:,0].tolist(),[3,3])
  closed,good=ex.join_consumed_owners(*owners,tables=self.account.device);self.assertTrue(bool(good.all()))
  # Actual departure restores only the departing member; a real installed
  # receipt advances the carry. No reuse of a previous frame's copy claim.
  state=entry.depart(actual.state,0,self.table,eligible=j.ones(2,bool));state=state._replace(active=state.active.at[:,0].set(1))
  tape=life._tape(actual.cursor,self.producer,64);tape=ex.begin_on_tape(tape,closed,self.account.device,active=j.ones(2,bool))
  tape=life.installed(tape,state,j.zeros(2,j.int32),j.zeros(2,j.int32),j.ones(2,j.int32),eligible=j.ones(2,bool),replacement=j.zeros(2,bool),baton=j.zeros(2,bool))
  kept=speed.retain_frame_device(current,state,t.stream_id,d,tape)
  self.assertTrue(bool(kept.qualified_birth.all()));self.assertEqual(kept.current_ability[:,0,0].tolist(),[2,2])
  wrong=speed.retain_device(current,state,t.stream_id,d);self.assertFalse(bool(wrong.qualified_birth.any()))
  self.assertFalse(bool(speed.retain_frame_device(current,state,t.stream_id,d,actual.callbacks).qualified_birth.any()))
 def test_hot_jit_full_capacity_and_wrong_current_identity_stay_unknown(self):
  s,t,_,d,born=self.fixture(8);mask=j.array([True,True,False,False,False,False,False,False])
  t=t._replace(count=t.count.at[0].set(64),switch_details=t.switch_details.at[0,63,0].set(9))
  run=jax.jit(lambda raw:entry.enter(s,0,self.table,eligible=mask,hazards=False,_weather_trace=raw))
  after,_,out=run(t);self.assertTrue(bool(out.refusal[0]&events.OVERFLOW));self.assertEqual(int(out.count[0]),64)
  bad=t._replace(cursor=t.cursor._replace(physical_active=t.cursor.physical_active.at[1,1].set(4)))
  _,_,refused=run(bad);self.assertTrue(bool(refused.refusal[1]&256));self.assertTrue(bool(refused.refusal[1]&512))
  for a,b in zip(jax.tree.leaves(t),jax.tree.leaves(out)):
   if getattr(a,'ndim',0) and a.shape[0]==8 and a.shape!=(8,):np.testing.assert_array_equal(a[2:],b[2:])
 def test_original_gen3_callback_and_setability_emit_exact_trace_line_without_copied_start(self):
  # Original Dex callback and Pokemon method; no Battle object/world constructed.
  script=r'''
const path=require('node:path');const root=process.argv[1];
const {Dex}=require(path.join(root,'dist/sim/dex.js'));
const {Pokemon}=require(path.join(root,'dist/sim/pokemon.js'));
const dex=Dex.mod('gen3');const trace=dex.abilities.get('trace');const lines=[],calls=[];
const donor={getAbility:()=>dex.abilities.get('sturdy'),toString:()=> 'p2a: Donphan'};
const battle={gen:3,dex,effect:trace,runEvent:()=>true,singleEvent:(id)=>calls.push(id),
 initEffectState:x=>x,add:(...xs)=>lines.push(xs.map(String).join('|'))};
const p={hp:267,ability:'trace',abilityState:{},battle,side:{randomFoe:()=>donor},
 getAbility(){return dex.abilities.get(this.ability)},setAbility:Pokemon.prototype.setAbility,
 toString:()=> 'p1a: Porygon2'};
trace.onStart.call(battle,p);
process.stdout.write(JSON.stringify({ability:p.ability,lines,calls,update:typeof trace.onUpdate}));
'''
  import json,shutil
  node=shutil.which('node')
  if node is None:self.skipTest('node unavailable')
  data=json.loads(subprocess.check_output([node,'-e',script,'/Users/scott/workspace/agents/pokemon-showdown-timing-smoke-f76228a'],text=True))
  self.assertEqual(data,dict(ability='sturdy',lines=['-ability|p1a: Porygon2|Sturdy|Trace|[from] ability: Trace|[of] p2a: Donphan'],calls=['End'],update='undefined'))

 def test_frozen_source_pins_and_legacy_copy_path_stay_closed(self):
  from gpu_engine import engine_source_frame_witness as witness
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),49)
  s,t,_,_,_=self.fixture()
  legacy=events.initialize(2,64,catalog=self.producer)
  self.assertIs(trace.entry_trace(legacy,0,s.active[:,0],s.active[:,1],j.full(2,2,j.int32),
      j.full(2,3,j.int32),j.full(2,3,j.int32),self.table,eligible=j.ones(2,bool)),legacy)
