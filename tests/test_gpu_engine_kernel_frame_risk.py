"""Declared kernel risks only: no shared callsite wiring or coverage proof."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_kernel_frame_risk as r, engine_execution_completeness as a
from gpu_engine import engine_lifecycle_callbacks as life, mechanics
import test_gpu_engine_engine_weather_public as fixture


class KernelFrameRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.WeatherPublicTests.setUpClass();cls.f=fixture.WeatherPublicTests
        cls.account=a.compile_accounting(cls.f.compiled)
        cls.compiled=r.compile_risks(cls.f.producer,cls.account);cls.tables=r.check_compiled(cls.compiled)
        cls.initial=cls.f().initial('hail',arrays=cls.f().arrays('pressure','pressure'))

    def same(self,x,y):
        for left,right in zip(jax.tree.leaves(x),jax.tree.leaves(y)):np.testing.assert_array_equal(left,right)

    def boundary(self,b=2):
        tile=lambda x:j.repeat(x[:1],b,axis=0)
        streams=j.arange(1401,1401+b,dtype=j.int32)
        state=jax.tree.map(tile,self.initial.state)
        cursor=jax.tree.map(tile,self.initial.cursor)._replace(stream_id=streams)
        tape=life._tape(cursor,self.f.producer,64)
        ledger=a.initialize(streams,compiled=self.account)
        ledger=a.begin_frame(ledger,tape,frame=j.ones(b,j.int32),active=j.ones(b,bool),tables=self.account.device)
        tape=tape._replace(execution=a.Context(ledger,self.account.device))
        carry=r.initialize_device(streams,self.tables)
        return r.begin_frame_device(carry,tape,self.tables,active=True),tape,state

    def test_actual_selected_columns_weather_domain_without_names_roster(self):
        table=self.f.table;weather=np.asarray(table.weather.move_weather)>0
        flags=np.asarray(self.tables.selected_risk)
        self.assertTrue(np.all((flags[~weather]&r.NON_WEATHER)!=0))
        self.assertTrue(np.all(flags[weather]==0))
        carry,tape,state=self.boundary()
        weather_id=int(np.flatnonzero(weather)[0]);other=int(np.flatnonzero(~weather)[1])
        result=r.observe_selected_device(carry,tape,j.array([weather_id,other],j.int32),self.tables,eligible=True,active=True)
        self.assertEqual(int(result.risk[0]),0);self.assertTrue(int(result.risk[1])&r.NON_WEATHER)
        bad=r.observe_selected_device(carry,tape,j.array([-1,len(flags)],j.int32),self.tables,eligible=True,active=True)
        self.assertEqual(bad.risk.tolist(),[r.NAMESPACE]*2)
        inventory=json.loads(r.inventory_json(self.compiled))
        self.assertEqual(inventory['scope'],'DECLARED_CUTPOINT_IDENTITY_ONLY_UNWIRED_NO_COMPLETENESS')
        self.assertEqual(len(inventory['declared']),26)
        self.assertEqual(inventory['declared'],{x.name:int(x) for x in r.Branch})
        self.assertEqual((inventory['declared']['PRIMARY_STATUS'],inventory['declared']['HEAL_BELL']),(25,26))
        for label in ('complete','admitted','exhaustive'):self.assertFalse(hasattr(result,label))

    def test_constructor_baseline_is_literal_invariant_not_origin_proof(self):
        carry,tape,state=self.boundary()
        actual=r.observe_constructor_device(carry,tape,state,self.tables,active=True)
        self.assertEqual(actual.risk.tolist(),[0,0])
        for field,value in (('confusion_counter',state.confusion_counter.at[0,0,0].set(3)),
                ('boosts',state.boosts.at[0,0,0,0].set(1)),('spikes',state.spikes.at[0,0].set(1)),
                ('wish_counter',state.wish_counter.at[0,0].set(2)),('turn',state.turn.at[0].set(1))):
            observed=r.observe_constructor_device(carry,tape,state._replace(**{field:value}),self.tables,active=True)
            self.assertTrue(int(observed.risk[0])&r.CONSTRUCTOR_STATE);self.assertEqual(int(observed.risk[1]),0)
        # Its name and source binding do not attest that a constructor called it.
        self.assertNotIn('constructor_origin_proved',r.Carry._fields)

    def test_named_actual_write_observations_stick_after_reversion_and_noop(self):
        carry,tape,_=self.boundary()
        for branch in list(r.Branch)[5:]:
            with self.subTest(branch=branch.name):
                first=r.observe_branch_device(carry,tape,branch,j.array([True,False]),self.tables,active=True)
                self.assertNotEqual(int(first.risk[0]),0);self.assertEqual(int(first.risk[1]),0)
                later=r.observe_branch_device(first,tape,branch,False,self.tables,active=True)
                self.assertEqual(later.risk.tolist(),first.risk.tolist())
                self.assertEqual(later.counts[:,int(branch)-1].tolist(),[2,2])
                self.assertTrue(int(later.observed[1])&(1<<(int(branch)-1)))
        with self.assertRaises(ValueError):r.observe_branch_device(carry,tape,r.Branch.CONSTRUCTOR,False,self.tables,active=True)

    def test_literal_before_move_and_hit_outcome_controls(self):
        carry,tape,state=self.boundary()
        before=mechanics.BeforeMove(j.zeros(2,j.int32),j.zeros(2,j.int32),j.zeros(2,j.int32),
            j.array([False,True]),j.zeros(2,bool))
        out=r.observe_pre_move_device(carry,tape,state,0,before,self.tables,eligible=True,active=True)
        self.assertEqual(out.risk.tolist(),[r.BLOCKED_ACTION,0])
        base=r.Outcome(0,True,True,False,1,1,False,False,False,False,0,False)
        # A critical RNG flag unused by a zero-power weather action is no crit line.
        self.assertEqual(r.observe_outcome_device(carry,tape,base,self.tables,eligible=True,active=True).risk.tolist(),[0,0])
        for changes in (dict(power=40),dict(absorbed=True),dict(recoil=1),dict(explodes=True),
                dict(power=40,critical=False,hit=False),dict(power=40,critical=False,numerator=2),
                dict(power=40,critical=False,substitute=True),dict(denominator=0)):
            value=r.observe_outcome_device(carry,tape,base._replace(**changes),self.tables,eligible=True,active=True)
            self.assertTrue(bool((value.risk!=0).all()))
        self.assertTrue(bool((r.observe_end_device(carry,tape,state._replace(done=j.ones(2,bool)),self.tables,active=True).risk&r.TERMINAL).all()))

    def test_genuine_accounting_frame_namespace_replay_missing_context_and_overflow(self):
        carry,tape,_=self.boundary()
        replay=r.begin_frame_device(carry,tape,self.tables,active=True)
        self.assertTrue(bool((replay.risk&r.ORDER).all()))
        missing=r.observe_branch_device(carry,tape._replace(execution=None),r.Branch.RECOIL,False,self.tables,active=True)
        self.assertTrue(bool((missing.risk&r.NAMESPACE).all()))
        bad=tape._replace(move_binding=tape.move_binding.at[7].add(1))
        self.assertTrue(bool((r.observe_branch_device(carry,bad,r.Branch.RECOIL,False,self.tables,active=True).risk&r.NAMESPACE).all()))
        prior=tape._replace(refusal=j.full(2,1|512,j.int32))
        self.assertTrue(bool((r.observe_branch_device(carry,prior,r.Branch.RECOIL,False,self.tables,active=True).risk&r.UPSTREAM).all()))
        overflow=carry._replace(counts=carry.counts.at[:,int(r.Branch.RECOIL)-1].set(2**31-1))
        observed=r.observe_branch_device(overflow,tape,r.Branch.RECOIL,False,self.tables,active=True)
        self.assertTrue(bool((observed.risk&r.OVERFLOW).all()))
        self.assertEqual(observed.counts[:,int(r.Branch.RECOIL)-1].tolist(),[2**31-1]*2)

    def test_pre_move_physical_bounds_typed_storage_eligible_rows_and_hot_jit(self):
        carry,tape,state=self.boundary(8)
        before=mechanics.BeforeMove(j.zeros(8,j.int32),j.zeros(8,j.int32),j.zeros(8,j.int32),
            j.ones(8,bool),j.zeros(8,bool))
        active=state.active.at[0,0].set(-1).at[1,0].set(6).at[2,0].set(2**31-1)
        active=active.at[3,1].set(-1).at[4,1].set(6).at[5,1].set(2**31-1)
        actual=state._replace(active=active)
        sides=j.array([0,0,0,1,1,1,0,1],j.int32)
        call=lambda c,s,e:r.observe_pre_move_device(c,tape,s,sides,before,self.tables,eligible=e,active=True)
        eligible=j.ones(8,bool);expected=call(carry,actual,eligible);fast=jax.jit(call)
        self.assertEqual(expected.risk.tolist(),[r.NAMESPACE]*6+[0,0])
        for _ in range(2):self.same(fast(carry,actual,eligible),expected)
        self.assertEqual(fast._cache_size(),1)
        eligible=eligible.at[1].set(False).at[4].set(False)
        masked=call(carry,actual,eligible)
        self.assertEqual(masked.risk.tolist(),[r.NAMESPACE,0,r.NAMESPACE,r.NAMESPACE,0,r.NAMESPACE,0,0])
        for value in (state.active.astype(j.float32),state.active[:,0],state.active[:,:1]):
            with self.subTest(shape=value.shape),self.assertRaises(ValueError):
                call(carry,state._replace(active=value),j.ones(8,bool))

    def test_b8_inactive_global_commit_and_hot_jit_no_host_callbacks(self):
        carry,tape,_=self.boundary(8);active=j.array([True,False]*4)
        call=lambda c,t:r.observe_branch_device(c,t,r.Branch.SECONDARY_STATS,True,self.tables,active=active)
        fast=jax.jit(call);expected=call(carry,tape)
        for _ in range(2):self.same(fast(carry,tape),expected)
        self.assertEqual(fast._cache_size(),1)
        committed=r.commit_rows(active,carry,expected)
        for name in ('binding','accounting_binding'):self.assertIs(getattr(committed,name),getattr(carry,name))
        for name in r.Carry._fields[2:]:self.same(getattr(committed,name)[~active],getattr(carry,name)[~active])
        names=[];seen=set()
        def walk(x):
            if id(x) in seen:return
            seen.add(id(x))
            if hasattr(x,'eqns'):
                for eq in x.eqns:
                    names.append(eq.primitive.name)
                    for v in eq.params.values():walk(v)
            elif hasattr(x,'jaxpr'):walk(x.jaxpr)
            elif isinstance(x,(list,tuple)):
                for v in x:walk(v)
            elif isinstance(x,dict):
                for v in x.values():walk(v)
        walk(jax.make_jaxpr(call)(carry,tape))
        self.assertFalse(set(names)&{'pure_callback','io_callback','debug_callback'})

    def test_original_guarded_compiler_and_all_closed_host_apis(self):
        with self.assertRaises(ValueError):r.check_compiled(replace(self.compiled))
        old=self.compiled.device
        try:
            object.__setattr__(self.compiled,'device',old._replace(selected_risk=j.array(old.selected_risk,copy=True)))
            with self.assertRaises(ValueError):r.check_compiled(self.compiled)
        finally:object.__setattr__(self.compiled,'device',old)
        for call in (lambda:r.compile_risks(self.f.producer,self.account),lambda:r.check_compiled(self.compiled),lambda:r.inventory_json(self.compiled)):
            for dummy in (False,True):
                fn=(lambda _:call()) if dummy else (lambda:call())
                with patch.object(a,'_checked',side_effect=AssertionError('late original guard')):
                    with self.assertRaisesRegex(ValueError,'cannot be traced'):
                        jax.jit(fn)(j.array(1)) if dummy else jax.jit(fn)()


if __name__=='__main__':unittest.main()
