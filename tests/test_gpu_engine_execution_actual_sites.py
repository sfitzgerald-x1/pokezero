"""Actual guarded kernel attempts; partial inventory, never source totality.

Numerical table mutations below are synthetic dispatch controls. They do not
prove original Dex legality, hook coverage, or scientific qualification.
"""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import transition, entry, mechanics as m
from gpu_engine import engine_lifecycle_callbacks as life
from gpu_engine import engine_execution_completeness as accounting
from gpu_engine.state import LEECH_SEED
from test_gpu_engine_engine_weather_public import WeatherPublicTests as Fixture


class ActualExecutionSitesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass();cls.f=Fixture()
        arrays=cls.f.arrays('pressure','pressure')
        arrays['moves'][:,0,0]=cls.f.ids['raindance']
        cls.initial=cls.f.initial('raindance',arrays=arrays)
        cls.compiled=accounting.compile_accounting(cls.f.compiled)
        cls.fast=staticmethod(jax.jit(lambda state,table,tape:transition.step_with_chance(
            state,j.zeros((2,2),j.int32),table,cls.f.chance(),
            _public_trace=tape,_lifecycle_trace=True)))

    def tape(self):
        tape=life._tape(self.initial.cursor,self.f.producer,64)
        ledger=accounting.initialize(tape.stream_id,compiled=self.compiled,capacity=32)
        return accounting.begin_on_tape(tape,ledger,self.compiled.device,active=j.ones(2,bool))

    def state(self):
        # Empty held items isolate the selected actual attempt. Retain the
        # real owning cursor/active mapping and ordinary transition execution.
        state=self.initial.state
        return state._replace(item=j.zeros_like(state.item))

    def run_step(self,state,table=None):
        table=self.f.table if table is None else table
        traced=self.fast(state,table,self.tape())
        legacy=transition.step_with_chance(state,j.zeros((2,2),j.int32),table,self.f.chance())
        self.f.same(traced.step.state,legacy.state)
        return traced.callbacks

    def sites(self,tape,row):
        ledger=tape.execution.ledger
        return np.array(ledger.site[row,:int(ledger.count[row])]).tolist()

    def assert_unknown(self,tape,row,sites):
        self.assertCountEqual(self.sites(tape,row),[int(s) for s in sites])
        self.assertNotEqual(int(tape.refusal[row])&accounting.UNKNOWN_EXECUTION,0)
        self.assertNotEqual(int(tape.execution.ledger.refusal[row]),0)
        self.assertFalse(bool(accounting.report(tape.execution.ledger,
            tables=self.compiled.device).scoped_drained[row]))

    def test_wish_pending_no_output_is_an_attempt_and_other_row_is_empty(self):
        state=self.state()._replace(wish_counter=self.state().wish_counter.at[0,0].set(2))
        tape=self.run_step(state)
        self.assert_unknown(tape,0,[accounting.Site.WISH])
        self.assertEqual(self.sites(tape,1),[])
        self.assertEqual(int(tape.execution.ledger.refusal[1]),0)

    def test_speed_boost_and_pinch_at_cap_still_enter_exactly_once(self):
        state=self.state();ability=int(state.ability[0,0,0])
        table=self.f.table._replace(entry=self.f.table.entry._replace(
            speed_boost=self.f.table.entry.speed_boost.at[ability].set(True)))
        state=state._replace(boosts=state.boosts.at[:,:,:,4].set(6))
        tape=self.run_step(state,table)
        for row in range(2):self.assert_unknown(tape,row,[accounting.Site.SPEED_BOOST]*2)
        # Synthetic selected held-item id tests the actual consumption guard;
        # a capped statistic still consumes, so zero boost delta is not proof.
        table=self.f.table._replace(items=self.f.table.items._replace(
            pinch_stat=self.f.table.items.pinch_stat.at[0].set(4)))
        state=state._replace(hp=state.hp.at[:,:,:].set(1))
        tape=self.run_step(state,table)
        for row in range(2):self.assert_unknown(tape,row,[accounting.Site.PINCH_ITEM]*2)

    def test_status_residual_and_leech_drain_heal_are_distinct_actual_sites(self):
        state=self.state()._replace(status=self.state().status.at[0,0,0].set(m.BRN),
            volatile=self.state().volatile.at[1,0,0].set(LEECH_SEED))
        tape=self.run_step(state)
        self.assert_unknown(tape,0,[accounting.Site.STATUS_RESIDUAL])
        self.assert_unknown(tape,1,[accounting.Site.LEECH_DRAIN,accounting.Site.LEECH_HEAL])

    def test_sleep_paralysis_and_frozen_are_not_status_damage_attempts(self):
        for status in (m.SLP,m.PAR,m.FRZ):
            state=self.state()._replace(status=self.state().status.at[:,0,0].set(status))
            tape=self.run_step(state)
            for row in range(2):self.assertNotIn(int(accounting.Site.STATUS_RESIDUAL),self.sites(tape,row))

    def test_full_hp_leftovers_and_missing_emitter_remain_unhandled(self):
        state=self.state();table=self.f.table._replace(items=self.f.table.items._replace(
            leftovers=self.f.table.items.leftovers.at[0].set(True)))
        # No registered emitter is supplied to this raw lifecycle seam.
        # Neither a full-HP no-op nor a real numeric heal can earn a receipt.
        for hurt in (False,True):
            current=state._replace(hp=state.hp.at[:,0,0].add(-10)) if hurt else state
            tape=self.run_step(current,table)
            for row in range(2):self.assert_unknown(tape,row,[accounting.Site.LEFTOVERS]*2)

    def test_blocked_spikes_and_intimidate_attempts_survive_zero_numeric_change(self):
        state=self.state();ability=int(state.ability[0,0,0])
        state=state._replace(spikes=state.spikes.at[:,0].set(3),
            boosts=state.boosts.at[:,1,0,0].set(-6))
        table=self.f.table.entry._replace(levitate=self.f.table.entry.levitate.at[ability].set(True),
            intimidate=self.f.table.entry.intimidate.at[ability].set(True),
            attack_drop_immune=j.ones_like(self.f.table.entry.attack_drop_immune))
        fast=jax.jit(lambda s,tape,active:entry.enter(s,j.zeros(2,j.int32),table,
            eligible=active,hazards=True,_weather_trace=tape))
        after,_,tape=fast(state,self.tape(),j.array([True,False]))
        self.assert_unknown(tape,0,[accounting.Site.ENTRY_HAZARD,accounting.Site.INTIMIDATE_ATTEMPT])
        self.assertEqual(self.sites(tape,1),[])
        np.testing.assert_array_equal(after.hp,state.hp)
        np.testing.assert_array_equal(after.boosts,state.boosts)
        # Fully inactive calls do not allocate a slot or modify the ledger.
        initial=self.tape();_,_,inactive=fast(state,initial,j.zeros(2,bool))
        self.f.same(initial.execution.ledger,inactive.execution.ledger)


if __name__=='__main__':unittest.main()
