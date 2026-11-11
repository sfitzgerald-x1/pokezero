"""Measured dead-slot stage correction, tiny original default components only."""
from _showdown_root import showdown_root_str
import ast
from pathlib import Path
import subprocess
import sys
import types
import unittest

import jax
import jax.numpy as j
import numpy as np
from gpu_engine import transition as tr,state,mechanics
import test_gpu_engine_self_destruct as destruct
import test_gpu_engine_weather as weather
import test_gpu_engine_entry as entry

BASE='5461516a5e80b4c97ff9ea4f7c7ac8a536457a82'


class FaintBoostResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_source=subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True)
        cls.old=types.ModuleType('gpu_engine._original_faint_boost_fixture')
        cls.old.__package__='gpu_engine';sys.modules[cls.old.__name__]=cls.old
        exec(compile(cls.old_source,'<original546-transition>','exec'),vars(cls.old))

    def chance(self,b):
        z=j.zeros((b,2),j.int32)
        return tr.ChanceInputs(draws=j.zeros((b,2,12),j.int32),phazing=z,secondary=z,sleep_talk=z,entry_ties=z,initial_tie=j.zeros(b,j.int32))

    def compare_original(self,before,table,actions):
        chance=self.chance(len(actions))
        actual=tr.step_with_chance(before,actions,table,chance)
        # The original predates the appended effect stream (fidelity R): its own six-field ChanceInputs, and
        # effect_draws (the block a deferring step saves) is outside the comparison.
        old=self.old.step_with_chance(before,actions,table,self.old.ChanceInputs(*chance[:6]))
        for name in actual.state._fields:
            if name not in ('boosts','effect_draws'):
                np.testing.assert_array_equal(getattr(actual.state,name),getattr(old.state,name),err_msg=name)
        np.testing.assert_array_equal(actual.invalid_actions,old.invalid_actions)
        return actual.state,old.state

    def test_newly_dead_all_seven_only_preserves_live_bench_prior_dead_and_ineligible(self):
        s=state.empty_state(8);boosts=j.broadcast_to(j.array([-3,-2,-1,1,2,3,4]),s.boosts.shape)
        old_hp=j.full_like(s.hp,100).at[1,1,4].set(0)
        hp=old_hp.at[0,1,3].set(0).at[1,1,4].set(0).at[2,0,0].set(0).at[3,1,5].set(0)
        s=s._replace(boosts=boosts,hp=hp);eligible=j.ones(8,bool).at[2].set(False)
        result=jax.jit(tr._clear_fainted_boosts)(s,old_hp,eligible=eligible)
        expected=np.asarray(boosts).copy();expected[0,1,3]=0;expected[3,1,5]=0
        np.testing.assert_array_equal(result.boosts,expected)
        for name in s._fields:
            if name!='boosts':np.testing.assert_array_equal(getattr(result,name),getattr(s,name),err_msg=name)

    def test_original_default_target_ko_and_explosion_self_ko(self):
        f=destruct.SelfDestructTests();f.setUp();s=f.live(2)
        s=s._replace(moves=s.moves.at[0,0,0,0].set(6),hp=s.hp.at[0,1,0].set(1),
            boosts=s.boosts.at[0,1,0,2:4].set(1).at[1,0,0,0].set(2).at[:,:,2,:].set(3))
        result,old=self.compare_original(s,f.table,j.zeros((2,2),j.int32))
        self.assertEqual(int(result.hp[0,1,0]),0);self.assertEqual(int(result.hp[1,0,0]),0)
        self.assertEqual(result.boosts[0,1,0].tolist(),[0]*7)
        self.assertEqual(result.boosts[1,0,0].tolist(),[0]*7)
        self.assertEqual(old.boosts[0,1,0,2:4].tolist(),[1,1]);self.assertEqual(int(old.boosts[1,0,0,0]),2)
        np.testing.assert_array_equal(result.boosts[:,:,2],s.boosts[:,:,2])

    def test_status_and_weather_handler_faints(self):
        f=destruct.SelfDestructTests();f.setUp();s=f.live()._replace(moves=j.full((1,2,6,4),3,j.int32))
        s=s._replace(hp=s.hp.at[0,1,0].set(1),status=s.status.at[0,1,0].set(mechanics.PSN),
            boosts=s.boosts.at[0,1,0].set(j.array([1,2,3,4,0,-1,-2])))
        result,old=self.compare_original(s,f.table,j.zeros((1,2),j.int32))
        self.assertEqual(int(result.hp[0,1,0]),0);self.assertEqual(result.boosts[0,1,0].tolist(),[0]*7)
        self.assertNotEqual(old.boosts[0,1,0].tolist(),[0]*7)
        w=weather.WeatherTests();w.setUp();s=w.live()._replace(weather=j.array([w.w.SAND]),weather_counter=j.array([5]))
        s=s._replace(hp=s.hp.at[:,:,0].set(1),boosts=s.boosts.at[:,:,0,:].set(2))
        result,_=self.compare_original(s,w.table,j.zeros((1,2),j.int32))
        self.assertEqual(result.hp[0,:,0].tolist(),[0,0]);self.assertEqual(result.boosts[0,:,0].tolist(),[[0]*7]*2)

    def test_original_replacement_entry_hazard_ko_and_invalid_identity(self):
        f=entry.EntryTests();f.setUp();s=f.live()
        s=s._replace(hp=s.hp.at[0,0,0].set(0).at[0,0,1].set(1),spikes=s.spikes.at[0,0].set(3),
            boosts=s.boosts.at[0,0,1,:].set(2),entry_pending=j.zeros((1,2),bool))
        result,old=self.compare_original(s,f.table,j.array([[4,0]],j.int32))
        self.assertEqual(int(result.active[0,0]),1);self.assertEqual(int(result.hp[0,0,1]),0)
        self.assertEqual(result.boosts[0,0,1].tolist(),[0]*7);self.assertEqual(old.boosts[0,0,1].tolist(),[2]*7)
        invalid=s._replace(hp=s.hp.at[0,0,0].set(100),boosts=j.ones_like(s.boosts)*2)
        result,_=self.compare_original(invalid,f.table,j.array([[99,0]],j.int32))
        np.testing.assert_array_equal(result.boosts,invalid.boosts)

    def test_default_delta_is_exact_boundary_calls_only_and_mutants_remain_visible(self):
        current=ast.parse(Path(tr.__file__).read_text());original=ast.parse(self.old_source)
        expected_eligibility=['used','do_switch & ~replacement','eligible','live & move_context','ordinary','eligible']
        calls=[x for x in ast.walk(current) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='_clear_fainted_boosts']
        calls.sort(key=lambda x:x.lineno)
        self.assertEqual([ast.unparse(x.keywords[0].value) for x in calls],expected_eligibility)
        class Normalize(ast.NodeTransformer):
            def visit_FunctionDef(self,node):
                if node.name=='_clear_fainted_boosts':return None
                return self.generic_visit(node)
            def visit_Assign(self,node):
                if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='_faint_before_hp':
                    if ast.unparse(node.value)!='work.hp':raise AssertionError('changed original HP boundary')
                    return None
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_clear_fainted_boosts':return None
                return self.generic_visit(node)
        normalized=Normalize().visit(current)
        self.assertEqual(ast.dump(normalized),ast.dump(original))
        from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
        baseline=dump_default_program(original)
        self.assertEqual(dump_default_program(normalized),baseline)
        assert_hp_rng_mutations_detected(self,normalized,baseline)

    def test_pinned_original_faint_clear_source_and_minimal_call_order(self):
        source=Path(showdown_root_str())
        battle=(source/'sim/battle.ts').read_text();pokemon=(source/'sim/pokemon.ts').read_text()
        faint=battle[battle.index('\tfaintMessages('):battle.index('\tcheckWin(',battle.index('\tfaintMessages('))]
        self.assertIn('pokemon.clearVolatile(false);',faint)
        clear=pokemon[pokemon.index('\tclearVolatile('):pokemon.index('\thasType(',pokemon.index('\tclearVolatile('))]
        self.assertIn('this.boosts = {',clear)
        for name in ('atk','def','spa','spd','spe','accuracy','evasion'):self.assertIn(name+': 0,',clear)
        # End-of-move faint processing follows all contact/Lum/recoil writes.
        code=Path(tr.__file__).read_text()
        begin=code.index('for order in range(2):\n        _faint_before_hp=work.hp')
        finish=code.index('work=_clear_fainted_boosts(work,_faint_before_hp,eligible=live & move_context)',begin)
        for marker in ('secondaries.apply_contact','_lum_update','-recoil','dragging.drag'):
            self.assertIn(marker,code[begin:finish])


if __name__=='__main__':unittest.main()
