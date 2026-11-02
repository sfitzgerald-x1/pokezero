"""Literal original-writer fixtures and pure device updates, never turns/games."""
import ast
from pathlib import Path
import subprocess
import unittest
from types import SimpleNamespace as NS
import jax
import jax.numpy as j
import numpy as np
from pokezero import showdown as original
from gpu_engine import engine_device_public_port as port,engine_device_public_hooks as h,engine_device_trackers as tr
from gpu_engine import mechanics as m,engine_device_features as features
from tests import test_gpu_engine_device_public_hooks as fixture

BASE='01674ab483eafbc2feedaa8efcf1f32140d956ae'


def tables():
    names=['','slaking',*['other'+str(i)for i in range(2,7)],'snorlax',*['other'+str(i)for i in range(8,13)]]
    return port.compile_tables({'vocabulary':{'species':names,'abilities':['','truant','trace','pressure']}})


def parser():
    p=original._ReplayParser();p.feed(['|gen|3','|gametype|singles',
        '|switch|p1a: Slaking|Slaking, L80|200/200','|switch|p2a: Snorlax|Snorlax, L80|100/100','|turn|1'])
    return p


class PublicPortTests(unittest.TestCase):
    def test_native_truant_turn_anchor_trace_unknown_and_post_upkeep_switch(self):
        state=fixture.state(1);c=h.initialize(state,port_tables=tables());yes=j.ones(1,bool)
        p=parser();view=lambda:tr.readout(c.trackers,j.array([0],j.int32)).own_mon_features['truant_loaf'][0,0]
        self.assertEqual(bool(view()),p.snapshot().truant_phase['p1']is True)
        h.public_turn(c,state,yes,j.array([2],j.int32));p.feed(['|turn|2'])
        self.assertEqual(bool(view()),p.snapshot().truant_phase['p1']is True)
        h.move(c,state,0,j.array([1],j.int32),yes);p.feed(['|move|p1a: Slaking|Tackle|p2a: Snorlax'])
        self.assertFalse(bool(view()));self.assertIs(p.snapshot().truant_phase['p1'],False)
        # Public copy announces holder only, not a guessed phase.
        h.reveal_ability(c,state,0,yes,ability=j.array([1],j.int32),original=False)
        self.assertFalse(bool(c.trackers.public_port.known[0,0]))
        result=m.BeforeMove(j.array([0]),j.array([0]),j.array([0]),j.array([False]),j.array([False]))
        h.before_move(c,state,0,result,yes,loafing=yes)
        self.assertTrue(bool(view()))
        # Actual sleep/freeze block precedes Truant; do not anchor a hidden phase.
        c.trackers=port.trace(c.trackers,tables(),0,j.array([1]),yes)
        h.before_move(c,state,0,result._replace(status=j.array([6])),yes,loafing=yes)
        self.assertFalse(bool(c.trackers.public_port.known[0,0]))
        # Native post-upkeep arrival skips the following public TURN flip.
        c.trackers=port.upkeep(c.trackers,yes);h.install(c,state,0,yes)
        h.public_turn(c,state,yes,j.array([3],j.int32));self.assertTrue(bool(view()))

    def test_distinct_sleep_holder_and_parser_victim_ledgers_rest_ability_cure_faint(self):
        state=fixture.state(2);c=h.initialize(state,port_tables=tables());yes=j.array([True,False])
        sleep=state._replace(status=state.status.at[:,1,0].set(5))
        h.observe(c,sleep,yes)
        view=tr.readout(c.trackers,j.zeros(2,j.int32));self.assertEqual(view.field_features['self_sleep_clause'].tolist(),[True,False])
        self.assertEqual(view.field_features['sleep_clause_blocks_self'].tolist(),[True,False])
        # Switching preserves the still-sleeping known victim.
        incoming=sleep._replace(active=sleep.active.at[:,1].set(1));h.install(c,incoming,1,yes)
        self.assertTrue(bool(tr.readout(c.trackers,j.zeros(2,j.int32)).field_features['self_sleep_clause'][0]))
        h.party_cure(c,sleep,state,1,yes)
        self.assertFalse(bool(tr.readout(c.trackers,j.zeros(2,j.int32)).field_features['self_sleep_clause'][0]))
        c=h.initialize(state,port_tables=tables())
        h.observe(c,sleep,yes,rest_side=1,rest_mask=yes)
        self.assertFalse(bool(c.trackers.public_port.victims.any()))
        # Original belief excludes ability source, parser victim set includes it.
        t=port.condition(c.trackers,1,j.zeros(2,j.int32),j.zeros(2,j.int32),j.full(2,5),yes,ability_source=True)
        self.assertEqual(t.public_port.holder[:,0].tolist(),[-1,-1]);self.assertEqual(t.public_port.victims[:,1,0].tolist(),[True,False])
        t=port.condition(t,1,j.zeros(2,j.int32),j.full(2,5),j.full(2,5),yes,faint=True)
        self.assertFalse(bool(t.public_port.victims.any()))
        p=parser();p.feed(['|-status|p2a: Snorlax|slp|[from] ability: Effect Spore'])
        self.assertTrue(p.snapshot().induced_sleep_victims['p1'])
        p.feed(['|faint|p2a: Snorlax']);self.assertFalse(p.snapshot().induced_sleep_victims['p1'])

    def test_spikes_fraction_both_observers_and_d_private_detach(self):
        state=fixture.state(2)._replace(max_hp=fixture.state(2).max_hp.at[:,1,0].set(233),hp=fixture.state(2).hp.at[:,1,0].set(233))
        c=h.initialize(state,port_tables=tables());after=state._replace(hp=state.hp.at[:,1,0].set(204));yes=j.array([True,False])
        c.trackers=port.spikes(c.trackers,state,after,1,yes,tr._percent)
        own=tr.readout(c.trackers,j.ones(2,j.int32)).field_features['self_hazard_credit']
        foe=tr.readout(c.trackers,j.zeros(2,j.int32)).field_features['opp_hazard_credit']
        np.testing.assert_allclose(own,[29/233,0],atol=1e-7,rtol=0);np.testing.assert_allclose(foe,[.12,0],atol=1e-7,rtol=0)
        p=parser();p.feed(['|-damage|p2a: Snorlax|88/100|[from] Spikes'])
        self.assertAlmostEqual(float(foe[0]),p.snapshot().hazard_damage_suffered['p2'],places=7)
        before=tr.readout(c.trackers,j.zeros(2,j.int32))
        redacted=c.trackers._replace(public_port=port.redact_for_owner(c.trackers.public_port,0))
        after=tr.readout(redacted,j.zeros(2,j.int32))
        for x,y in zip(jax.tree.leaves(before),jax.tree.leaves(after)):np.testing.assert_array_equal(x,y)
        self.assertFalse(bool(redacted.public_port.hazards[:,1,1].any()))

    def test_stats_protect_pursuit_and_weather_original_projection_values(self):
        state=fixture.state(2);c=h.initialize(state,port_tables=tables());yes=j.array([True,False])
        c.trackers=port.protect(c.trackers,0,j.ones(2,bool),yes,j.ones(2,bool),j.ones(2,bool))
        c.trackers=port.count(c.trackers,1,yes,'pursuit')
        for kind,ability in ((1,False),(2,True),(3,True)):
            c.trackers=port.weather(c.trackers,1,j.full(2,kind),yes,ability=ability)
        facts=tr.readout(c.trackers,j.zeros(2,j.int32)).stats_features
        self.assertEqual(facts['blocked_on_our_attack'].tolist(),[1,0]);self.assertEqual(facts['pursuit_intercept_predict'].tolist(),[1,0])
        self.assertEqual([int(facts['weather_reveal_'+str(i)][0])for i in range(6)],[1,0,1,1,1,1])
        compiled=features.Tables(j.zeros(1,j.int32),j.zeros(2,j.int32),j.zeros(2,j.int32),j.zeros(4,j.int32),j.array(3),j.array(True))
        numeric,_,on,_=features.project_stats(facts,compiled,2)
        self.assertEqual(float(numeric[0,original.v4_numeric_index(original.NUMERIC_STAT_BLOCKED_ON_OUR_ATTACK)]),1/64)
        from pokezero.transitions_fold import FoldState
        lines=['|switch|p1a: Slaking|Slaking, L80|200/200','|switch|p2a: Snorlax|Snorlax, L80|100/100','|turn|1',
            '|move|p1a: Slaking|Tackle|p2a: Snorlax','|-activate|p2a: Snorlax|move: Protect',
            '|-activate|p1a: Slaking|move: Pursuit','|move|p2a: Snorlax|Pursuit|p1a: Slaking','|turn|2',
            '|move|p2a: Snorlax|Rain Dance|p2a: Snorlax','|-weather|RainDance',
            '|-weather|SunnyDay|[from] ability: Drought|[of] p2a: Snorlax',
            '|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Snorlax','|turn|3']
        _,products=FoldState.initial(perspective_slot='p1').advance(lines)
        self.assertEqual(int(facts['blocked_on_our_attack'][0]),products.tendency_stats.blocked_on_our_attack_count)
        self.assertEqual(int(facts['pursuit_intercept_predict'][0]),products.tendency_stats.pursuit_intercept_predict_count)
        expected={x.weather:x.from_ability for x in products.tendency_stats.opponent_weather_reveals}
        self.assertEqual(expected,{'raindance':False,'sunnyday':True,'sandstorm':True})
        # weather history persists after public clear, not current-weather polling.
        c.trackers=port.weather(c.trackers,1,j.zeros(2,j.int32),yes)
        self.assertTrue(bool(c.trackers.public_port.weather[0,1].any()))
        rain=state._replace(weather=j.ones(2,j.int32))
        h.weather(c,state,rain,yes,permanent=True,side=1)
        self.assertTrue(bool(c.trackers.public_port.weather[0,1,0,1]))
        self.assertFalse(bool(c.trackers.public_port.weather[1].any()))

    def test_mixed_rows_jit_and_original_numeric_step_ast_unchanged(self):
        state=fixture.state(2)
        @jax.jit
        def update(state):
            c=h.initialize(state,port_tables=tables());use=j.array([True,False])
            c.trackers=port.count(c.trackers,j.array([0,1]),use,'blocked')
            h.public_turn(c,state,use,j.array([2,2],j.int32))
            return c.trackers
        t=update(state);self.assertEqual(t.public_port.blocked.tolist(),[[1,0],[0,0]])
        self.assertEqual(t.public_port.loaf[:,0].tolist(),[True,False])
        # Remove optional observer statements only, and compare original numeric
        # kernel AST. No game/turn executes in this source check.
        old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(Path('gpu_engine/transition.py').read_text())
        class Strip(ast.NodeTransformer):
            def visit_If(self,node):
                if '_device_public'in ast.unparse(node.test):return None
                return self.generic_visit(node)
        get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='step_with_chance')
        self.assertEqual(ast.dump(Strip().visit(get(old)),include_attributes=False),ast.dump(Strip().visit(get(new)),include_attributes=False))

if __name__=='__main__':unittest.main()
