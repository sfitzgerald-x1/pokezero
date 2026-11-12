"""Inert original entry/FULL components; no step kernel, world/game or model."""
from _showdown_root import showdown_root_str
import ast,json,subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import entry,engine_entry_spikes_callbacks as spikes,engine_intimidate_callbacks as intim
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risks
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as events
from gpu_engine import engine_public_event_device as base,engine_full_v4_event_device as full
from gpu_engine import engine_source_frame_witness as witness,v4_encoder as v4,public_wire
from gpu_engine_default_observer_ast import dump_default_program
import test_gpu_engine_ordinary_intimidate_owning as ordinary


class EntrySpikesTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  ordinary.OrdinaryIntimidateTests.setUpClass();f=ordinary.OrdinaryIntimidateTests
  cls.f=f.f;cls.catalog=f.catalog;cls.metadata=deepcopy(f.metadata)
  cls.metadata['vocabulary']['abilities'].append('levitate')
  cls.metadata['vocabulary']['items']=['_none','heavydutyboots'] # Negative-only off-source item.
  cls.table=f.table._replace(entry=entry.from_metadata(cls.metadata))
  cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=ordinary.initial_fixture.fixtures.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog)
  cls.account=ex.compile_accounting(cls.adapter);cls.risk=risks.compile_risks(cls.producer,cls.account)
  cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
 state=ordinary.OrdinaryIntimidateTests.state
 initial=ordinary.OrdinaryIntimidateTests.initial
 same=ordinary.OrdinaryIntimidateTests.same
 before=ordinary.OrdinaryIntimidateTests.before
 install=ordinary.OrdinaryIntimidateTests.install
 def literal_before(self,b=8):
  state=self.state(b);mask=j.ones(b,bool);cursor=life.register(j.arange(3101,3101+b,dtype=j.int32),**self.kw)
  tape=life._tape(cursor,self.producer,64)
  for side in(0,1):tape=life._reveal(tape,state,j.full(b,side,j.int32),j.full(b,-1,j.int32),state.active[:,side],eligible=mask,replacement=~mask,baton=~mask,initial=True)
  cursor=tape.cursor._replace(initialized=mask,turn=j.ones(b,j.int32));tape=life._tape(cursor,self.producer,64)
  tape=ex.begin_on_tape(tape,ex.initialize(tape.stream_id,compiled=self.account),self.account.device,active=mask)
  carry=risks.begin_frame_device(risks.initialize_device(tape.stream_id,self.risk.device),tape,self.risk.device,active=mask)
  tape=tape._replace(execution=tape.execution._replace(risks=risks.Context(carry,self.risk.device)))
  side=j.zeros(b,j.int32);state,tape=self.install(state,tape,side)
  return state._replace(spikes=state.spikes.at[:,0].set(1)),tape
 def test_cheap_same_write_guards_inactive_and_all_global8(self):
  state,tape=self.literal_before();mask=j.arange(8)<7
  state=state._replace(types=state.types.at[1,0,3,0].set(self.table.entry.flying_type),
    ability=state.ability.at[2,0,3].set(4),item=state.item.at[3,0,3].set(1),
    hp=state.hp.at[4,0,3].set(1),spikes=state.spikes.at[6,0].set(0))
  tape=tape._replace(refusal=tape.refusal.at[5].set(257))
  actual,ko,out=entry.enter(state,j.zeros(8,j.int32),self.table.entry,eligible=mask,hazards=True,_weather_trace=tape)
  plain,plain_ko=entry.enter(state,j.zeros(8,j.int32),self.table.entry,eligible=mask,hazards=True)
  self.same(actual,plain);self.same(ko,plain_ko)
  self.assertEqual(out.execution.ledger.stage[:6,0].tolist(),[2,1,1,1,1,1])
  self.assertEqual(out.execution.ledger.site[:6,0].tolist(),[8]*6)
  self.assertEqual(out.words[0,1].tolist(),[50,1,1,0,3,100,88,100,0,1])
  self.assertEqual(out.words[0,2,0].item(),29);self.assertEqual(out.execution.ledger.stage[0,:2].tolist(),[2,2])
  self.assertTrue(bool((out.refusal[1:6]&512).all()));self.assertEqual(int(out.refusal[5])&257,257)
  self.assertEqual(out.execution.ledger.count[7].item(),0);self.assertEqual(out.count[7].item(),1)
  for name in('engine_binding','source_binding','move_binding','producer_binding'):
   for bad in(j.zeros((1,8),j.uint32),j.zeros((8,8),j.uint32),j.zeros((8,),j.int32)):
    with self.subTest(name=name,shape=bad.shape,dtype=bad.dtype):
     with self.assertRaises(ValueError):spikes.begin(tape._replace(**{name:bad}),state,j.zeros(8,j.int32),j.ones(8,j.int32),self.table.entry,requested=mask)
  old_state,old_tape=self.literal_before(2)
  legacy=self.table.entry._replace(spikes_type_count=None,spikes_item_count=None,spikes_boots_item=None)
  _,_,unknown=entry.enter(old_state,j.zeros(2,j.int32),legacy,eligible=j.ones(2,bool),hazards=True,_weather_trace=old_tape)
  self.assertEqual(unknown.count.tolist(),[1,1]);self.assertEqual(unknown.execution.ledger.stage[:,0].tolist(),[1,1])
  self.assertTrue(bool((unknown.refusal&512).all()))
 def test_cheap_exact_claim_chain_and_mutations(self):
  state,tape=self.literal_before(2);side=j.zeros(2,j.int32)
  _,_,out=entry.enter(state,side,self.table.entry,eligible=j.ones(2,bool),hazards=True,_weather_trace=tape)
  self.assertEqual(intim.ordinary_owned(out,j.full(2,2,j.int32),out.cursor).tolist(),[True,True])
  changes=[out._replace(execution=out.execution._replace(ledger=out.execution.ledger._replace(claimed=out.execution.ledger.claimed.at[0,1].set(0)))),
   out._replace(execution=out.execution._replace(ledger=out.execution.ledger._replace(site=out.execution.ledger.site.at[0,0].set(9)))),
   out._replace(execution=out.execution._replace(ledger=out.execution.ledger._replace(stage=out.execution.ledger.stage.at[0,0].set(1)))),
   out._replace(words=out.words.at[0,1,6].add(1)),out._replace(event_generation=out.event_generation.at[0,1,0].add(1)),
   out._replace(words=out.words.at[0,1,0].set(20))]
  for changed in changes:self.assertEqual(intim.ordinary_owned(changed,j.full(2,2,j.int32),changed.cursor).tolist(),[False,True])
  for name in('engine_binding','source_binding','move_binding','producer_binding'):
   self.assertEqual(spikes.claimed(out._replace(**{name:getattr(out,name).at[0].add(1)}),j.ones(2,j.int32),out.cursor).tolist(),[False,False])
 def test_actual_mirrored_entry_same_write_both_full_and_source_claims(self):
  state,tape,observers=self.before();side=j.array([0,1],j.int32);state,tape=self.install(state,tape,side)
  state=state._replace(spikes=state.spikes.at[j.arange(2),side].set(j.array([1,2],j.int32)))
  actual,ko,out=entry.enter(state,side,self.table.entry,eligible=j.ones(2,bool),hazards=True,_weather_trace=tape)
  plain,plain_ko=entry.enter(state,side,self.table.entry,eligible=j.ones(2,bool),hazards=True)
  self.same(actual,plain);self.same(ko,plain_ko);self.assertEqual(out.refusal.tolist(),[1,1])
  self.assertEqual(out.execution.ledger.site[:,:2].tolist(),[[8,9],[8,9]])
  self.assertEqual(out.execution.ledger.stage[:,:2].tolist(),[[2,2],[2,2]])
  source=SimpleNamespace(risk=self.risk.device,secondary_weather=j.zeros(len(self.table.power),bool),direct_hit_binding=j.zeros(8,j.uint32),substitute_binding=j.zeros(8,j.uint32),toxic_binding=j.zeros(8,j.uint32))
  self.assertEqual(witness._raw_complete(out,source).tolist(),[True,True])
  frame=life.LifecycleResult(actual,out.cursor,out,life.snapshot(observers[0].projection.state.causal.cursor),life.snapshot(out.cursor),j.zeros(2,bool),j.zeros(2,j.int32))
  captures=[];closed=[]
  for owner,prior in enumerate(observers):
   ledger,result=ex.consume_frame_device(out.execution.ledger,frame,prior.projection.state,prior.public,active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog)
   self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertEqual(ledger.stage[:,:2].tolist(),[[3,3],[3,3]]);closed.append(ledger)
   for row,actor in enumerate((0,1)):
    initial=['|switch|p1a: Celebi|Celebi, L80, M|100/100','|switch|p2a: Celebi|Celebi, L80, M|100/100','|turn|1']
    wire,_=public_wire.project_lines(public_wire.WireState(),initial,**v4.wire_kwargs(self.catalog))
    hp=(88,84)[row];lines=[f'|switch|p{actor+1}a: Arbok|Arbok, L80, M|100/100',f'|-damage|p{actor+1}a: Arbok|{hp}/100|[from] Spikes',f'|-ability|p{actor+1}a: Arbok|Intimidate|boost',f'|-unboost|p{2-actor}a: Celebi|atk|1']
    _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));expected=np.array([event.words()for packet in packets for event in packet.events],np.int32);got=self.f.words(result.projection,row)
    captures.append(dict(owner=owner,row=row,actual=got.tolist(),expected=expected.tolist()));Path('/tmp/pokezero-entry-spikes-full-words.json').write_text(json.dumps(captures)+'\n')
    np.testing.assert_array_equal(got,expected)
  _,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
 def test_static_default_physics_prefix_and_original_callback(self):
  old=ast.parse(subprocess.check_output(['git','show','ad7ecc8987d849921aea7db9c10c46d363507674:gpu_engine/entry.py'],text=True));new=ast.parse(Path('gpu_engine/entry.py').read_text())
  for name in('enter','depart','speed_modifier','restore_sleep'):
   a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name==name)
   if name=='depart':b=ordinary.strip_m1_recharge(deepcopy(b),required=True)     # fidelity M1's approved clear only
   self.assertEqual(dump_default_program(a),dump_default_program(b))
  a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='from_metadata');b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='from_metadata')
  self.assertEqual([ast.dump(x)for x in a.body[-1].value.args],[ast.dump(x)for x in b.body[-1].value.args[:-3]])
  script="const{Dex}=require(process.argv[1]+'/dist/sim/dex.js');const c=Dex.mod('gen3').conditions.get('spikes');const rows=[];for(let layers=1;layers<=3;layers++)for(const grounded of[false,true])for(const boots of[false,true]){const calls=[];c.onEntryHazard.call({effectState:{layers},damage:x=>calls.push(x)},{maxhp:254,isGrounded:()=>grounded,hasItem:x=>{if(x!=='heavydutyboots')throw Error('wrong guard');return boots;}});rows.push({layers,grounded,boots,calls});}process.stdout.write(JSON.stringify(rows));"
  rows=json.loads(subprocess.check_output(['node','-e',script,showdown_root_str()],text=True))
  for row in rows:self.assertEqual(row['calls'],[[0,3,4,6][row['layers']]*254/24]if row['grounded']and not row['boots']else[])


if __name__=='__main__':unittest.main()
