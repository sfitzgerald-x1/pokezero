"""Actual optional JAX callsites; literal components, no policy/game/physics run."""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_kernel_frame_risk as r, engine_execution_completeness as a
from gpu_engine import engine_lifecycle_callbacks as life, engine_public_events as events
from gpu_engine import transition as t, secondary, entry
import test_gpu_engine_engine_weather_public as fixture


# Exercise the integrated lifecycle context, including its optional transaction
# fields, rather than the three-field stand-in used before those hooks existed.
ExecutionContext = a.Context


class KernelRiskCallsitesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.WeatherPublicTests.setUpClass();cls.f=fixture.WeatherPublicTests
        cls.metadata=deepcopy(cls.f.metadata)
        cls.ids={}
        for name in ('bellydrum','swordsdance','recover','mudshot'):
            cls.ids[name]=len(cls.metadata['vocabulary']['moves'])
            cls.metadata['vocabulary']['moves'].append(name)
            cls.metadata['moves'].append(dict(id=name,base_power=0 if name!='mudshot' else 55,
                type=1,source_type='Normal',category='Status' if name!='mudshot' else 'Physical',
                pp=10,accuracy=100,priority=0,flags={}))
        cls.table=t.tables_from_metadata(cls.metadata)
        cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.f.catalog.moves,
            dex=cls.f.catalog.dex,exposure=events.HealthExposure(events.SHOWDOWN,'gen3randombattle',True,False))
        # Reuse original retained static Dex surfaces, bound to THIS literal
        # engine table; missing public move labels remain unsupported.
        from gpu_engine import engine_public_event_device as base,engine_full_v4_event_device as full
        projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,
            expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
        cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.f.catalog)
        cls.account=a.compile_accounting(cls.adapter);cls.risk=r.compile_risks(cls.producer,cls.account)
        cls.initial=cls.f().initial('hail',arrays=cls.f().arrays('pressure','pressure'))

    def same(self,x,y):
        for left,right in zip(jax.tree.leaves(x),jax.tree.leaves(y)):np.testing.assert_array_equal(left,right)

    def boundary(self,b=2,risky=True):
        tile=lambda x:j.repeat(x[:1],b,axis=0)
        streams=j.arange(1701,1701+b,dtype=j.int32)
        state=jax.tree.map(tile,self.initial.state)._replace(item=j.zeros((b,2,6),j.int32))
        cursor=jax.tree.map(tile,self.initial.cursor)._replace(stream_id=streams,
            engine_binding=j.broadcast_to(j.array(self.producer.engine_binding,j.uint32),(b,8)))
        state=state._replace(moves=state.moves.at[:,0,0,0].set(self.f.ids['hail']).at[:,1,0,0].set(self.f.ids['raindance']))
        tape=life._tape(cursor,self.producer,64)
        ledger=a.initialize(streams,compiled=self.account)
        ledger=a.begin_frame(ledger,tape,frame=j.ones(b,j.int32),active=j.ones(b,bool),tables=self.account.device)
        tape=tape._replace(execution=ExecutionContext(ledger,self.account.device))
        if risky:
            carry=r.begin_frame_device(r.initialize_device(streams,self.risk.device),tape,self.risk.device,active=True)
            tape=tape._replace(execution=tape.execution._replace(risks=r.Context(carry,self.risk.device)))
        return state,tape

    def chance(self,b):
        return t.ChanceInputs(j.zeros((b,2,12),j.int32).at[:,:,4].set(1),
            *[j.zeros((b,2),j.int32) for _ in range(4)],j.zeros(b,j.int32))

    def step(self,state,tape,actions=None):
        b=len(state.active);actions=j.zeros((b,2),j.int32) if actions is None else actions
        return t.step_with_chance(state,actions,self.table,self.chance(b),_public_trace=tape,_lifecycle_trace=True)

    def test_actual_weather_step_and_none_context_preserve_state_rng_and_ledger(self):
        with patch.object(a,'Context',ExecutionContext):
            state,tape=self.boundary();plain=tape._replace(execution=tape.execution._replace(risks=None))
            old=self.step(state,plain);actual=self.step(state,tape)
            self.same(old.step,actual.step)
            legacy=t.step_with_chance(state,j.zeros((2,2),j.int32),self.table,self.chance(2))
            self.same(legacy,actual.step)
            self.same(old.callbacks.words,actual.callbacks.words)
            self.same(old.callbacks.execution.ledger,actual.callbacks.execution.ledger)
            carry=actual.callbacks.execution.risks.carry
            self.assertEqual(carry.risk.tolist(),[0,0]);self.assertEqual(carry.counts[:,r.Branch.PRE_MOVE-1].tolist(),[2,2])
            self.assertEqual(carry.counts[:,r.Branch.SELECTED_MOVE-1].tolist(),[2,2])
            self.assertEqual(carry.counts[:,r.Branch.HIT_OUTCOME-1].tolist(),[2,2])
            self.assertEqual(carry.counts[:,r.Branch.FRAME_END-1].tolist(),[1,1])
            self.assertEqual(carry.counts[:,r.Branch.CONSTRUCTOR-1].tolist(),[0,0])
            self.assertFalse(bool((actual.callbacks.refusal&r.UNKNOWN_KERNEL_RISK).any()))

    def test_actual_capped_boost_full_hp_heal_and_failed_drum_attempts_are_observed(self):
        with patch.object(a,'Context',ExecutionContext):
            state,tape=self.boundary(3)
            moves=j.array([self.ids['swordsdance'],self.ids['recover'],self.ids['bellydrum']],j.int32)
            state=state._replace(moves=state.moves.at[:,0,0,0].set(moves),
                boosts=state.boosts.at[:,0,0,0].set(6))
            before=state.hp[:,0,0]
            actual=self.step(state,tape)
            carry=actual.callbacks.execution.risks.carry
            self.assertTrue(bool((carry.risk!=0).all()))
            self.assertTrue(bool((actual.callbacks.refusal&r.UNKNOWN_KERNEL_RISK).all()))
            self.assertEqual(actual.step.state.hp[:,0,0].tolist(),before.tolist())
            for row,branch in enumerate((r.Branch.SELF_BOOST,r.Branch.SELF_HEAL,r.Branch.BELLY_DRUM)):
                self.assertTrue(int(carry.observed[row])&(1<<(int(branch)-1)))
            plain=self.step(state,tape._replace(execution=tape.execution._replace(risks=None)))
            self.same(plain.step,actual.step)

    def test_actual_secondary_capped_speed_drop_and_entry_zero_confusion_attempt(self):
        with patch.object(a,'Context',ExecutionContext):
            state,tape=self.boundary()
            state=state._replace(boosts=state.boosts.at[:,1,0,4].set(-6))
            old=state.boosts
            updated,trace=secondary.apply(state,j.zeros(2,j.int32),j.full(2,self.ids['mudshot'],j.int32),
                j.zeros(2,j.int32),self.table.secondary,eligible=j.ones(2,bool),_status_trace=tape)
            self.same(old,updated.boosts)
            self.assertTrue(bool((trace.execution.risks.carry.risk&r.BOOST).all()))
            self.assertEqual(trace.execution.risks.carry.counts[:,r.Branch.SECONDARY_STATS-1].tolist(),[3,3])
            self.assertEqual(trace.execution.risks.carry.counts[:,r.Branch.SECONDARY_FLINCH-1].tolist(),[1,1])
            # Zero is an absent confusion volatile: visiting Own Tempo's entry
            # hook records the cutpoint without manufacturing a cure risk.
            ability=state.ability[:,0,0]
            entry_table=self.table.entry._replace(own_tempo=self.table.entry.own_tempo.at[ability].set(True))
            updated,_,trace=entry.enter(state,j.zeros(2,j.int32),entry_table,eligible=j.ones(2,bool),
                hazards=False,_weather_trace=tape)
            self.assertEqual(updated.confusion_counter[:,0,0].tolist(),[0,0])
            self.assertFalse(bool((trace.execution.risks.carry.risk&r.FIELD_VOLATILE).any()))
            self.assertEqual(trace.execution.risks.carry.counts[:,r.Branch.ENTRY_CONFUSION_CURE-1].tolist(),[1,1])
            # Existing confusion, including a malformed negative counter,
            # still takes the conservative risk path and is actually cleared.
            confused=state._replace(confusion_counter=state.confusion_counter.at[:,0,0].set(j.array([3,-1],j.int32)))
            updated,_,trace=entry.enter(confused,j.zeros(2,j.int32),entry_table,eligible=j.ones(2,bool),
                hazards=False,_weather_trace=tape)
            self.assertEqual(updated.confusion_counter[:,0,0].tolist(),[0,0])
            self.assertTrue(bool((trace.execution.risks.carry.risk&r.FIELD_VOLATILE).all()))
            self.assertEqual(trace.execution.risks.carry.counts[:,r.Branch.ENTRY_CONFUSION_CURE-1].tolist(),[1,1])

    def test_actual_hot_jit_b8_inactive_rows_and_context_only_updates_risks(self):
        with patch.object(a,'Context',ExecutionContext):
            state,tape=self.boundary(8)
            state=state._replace(done=j.array([False,True]*4))
            fast=jax.jit(self.step);actual=fast(state,tape);again=fast(state,tape)
            self.same(actual,again);self.assertEqual(fast._cache_size(),1)
            carry=actual.callbacks.execution.risks.carry;old=tape.execution.risks.carry
            inactive=state.done
            for name in r.Carry._fields[2:]:self.same(getattr(carry,name)[inactive],getattr(old,name)[inactive])
            for name in ('binding','accounting_binding'):self.same(getattr(carry,name),getattr(old,name))
            self.assertIs(r.branch_on_tape(tape._replace(execution=tape.execution._replace(risks=None)),r.Branch.RECOIL,True,active=True).execution.risks,None)
            changed=r.branch_on_tape(tape,r.Branch.RECOIL,j.array([True,False]*4),active=True)
            self.assertIs(changed.execution.ledger,tape.execution.ledger);self.assertIs(changed.execution.tables,tape.execution.tables)
            self.assertEqual(changed.refusal.tolist(),[1025,1]*4)

    def test_all_nineteen_named_cutpoints_are_present_in_owned_actual_sources(self):
        names=set();calls=set()
        for filename in ('transition.py','secondary.py','entry.py'):
            tree=ast.parse(Path(t.__file__).with_name(filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='kernel_risks':
                    calls.add(node.func.attr)
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='branch_on_tape':
                    names.add(node.args[1].attr)
        # Five former generic visits now validate typed SAME-call captures.
        # Follow their actual wrappers to the observer's exact Branch literal;
        # do not remove those cutpoints from the required inventory.
        helpers=ast.parse(Path(r.__file__).read_text())
        functions={node.name:node for node in helpers.body if isinstance(node,ast.FunctionDef)}
        for stem,branch in (('self_heal','SELF_HEAL'),('substitute','SUBSTITUTE'),
                            ('primary_status','PRIMARY_STATUS'),('self_boost','SELF_BOOST'),('spikes','SPIKES')):
            wrapper=stem+'_on_tape';observer='observe_'+stem+'_device'
            self.assertIn(wrapper,calls)
            delegated=[node for node in ast.walk(functions[wrapper]) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_on_tape']
            self.assertEqual(len(delegated),1)
            self.assertEqual(delegated[0].args[1].id,observer)
            observed=[node.args[3] for node in ast.walk(functions[observer]) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_observe']
            self.assertEqual([ast.unparse(node) for node in observed],['Branch.'+branch])
            names.add(branch)
        self.assertEqual(names,{x.name for x in list(r.Branch)[5:]})


if __name__=='__main__':unittest.main()
