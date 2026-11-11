"""Observed clock/stats/last-move meanings against original pure writers.

Synthetic source transactions only: no engine turn, actor, Native game or replay.
"""
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_device_observation as e, engine_device_public_hooks as h
from gpu_engine import engine_device_trackers as tr, engine_device_features as f, public_tendency as ref
from gpu_engine.state import empty_state
from pokezero import showdown as s

BASE='03be68de16f4265f5f9c226ce6ada80b8dd8619d'


class DeviceBoundaryParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_device_observation as fixture
        fixture.DeviceObservationTests.setUpClass()
        cls.fixture=fixture.DeviceObservationTests();cls.catalog=fixture.DeviceObservationTests.catalog
        cls.tables=fixture.DeviceObservationTests.tables;cls.metadata=fixture.DeviceObservationTests.metadata
        cls.original=ref.compile_tendency(cls.catalog.species,cls.catalog.moves,cls.catalog.vocabulary)

    def state(self):
        b=2;z=empty_state(b);old,_,_,_=self.fixture.facts(b)
        values={name:getattr(old,name)for name in old._fields if name in z._fields}
        values.update(active=j.zeros((b,2),j.int32),turn=j.zeros(b,j.int32),
            species=j.broadcast_to(j.array([[2,1,0,0,0,0],[3,1,0,0,0,0]],j.int32),(b,2,6)))
        return z._replace(**values)

    def reference(self):
        return ref.initialize(2,self.original.device,species_binding=self.original.device.species_binding,complete_prefix=True)

    def line(self,state,line):
        fact=ref.decode_line(line,catalog=self.original)
        out,valid=ref.apply(state,self.original.device,fact,producer_binding=self.original.device.binding)
        self.assertTrue(bool(valid.all()),line);return out

    def compare_tendency(self,c,state,reference):
        for owner in (0,1):
            owners=j.full(2,owner,j.int32);view=tr.readout(c.trackers,owners)
            species=j.asarray([[self.catalog.species.species_id(self.metadata['vocabulary']['species'][int(x)])if int(x)else 0
                for x in row]for row in np.asarray(view.species)],j.int32)
            own=[self.catalog.species.species_id(self.metadata['vocabulary']['species'][int(state.species[r,owner,state.active[r,owner]])])for r in range(2)]
            expected=ref.project(reference,self.original.device,observer_side=owners,opponent_species=species,
                own_active_species=j.asarray(own,j.int32),expected_vocabulary=self.original.device.vocabulary_binding)
            n,_,_=f.project_mon(j.zeros((2,6,132),j.float32),j.zeros((2,6,41),j.int32),view.mon_features,
                active=view.active,present=view.present,tables=self.tables.features,status=view.status)
            stat,cat,on,bad=f.project_stats(view.stats_features,self.tables.features,2)
            np.testing.assert_allclose(n[...,j.asarray(ref.MON_COLUMNS)],expected.numeric[...,j.asarray(ref.MON_COLUMNS)],atol=1e-7)
            for key in ('opp_switch_count','opp_decision_opportunities','my_switch_turns'):
                col=s.v4_numeric_index(getattr(s,'NUMERIC_'+f.STATS_FIELDS[key][0]))
                np.testing.assert_allclose(stat[:,col],expected.stats_numeric[:,col],atol=1e-7)
            np.testing.assert_array_equal(cat[:,s.CATEGORY_ROLE],expected.stats_role)
            self.assertTrue(bool(on.all()));self.assertFalse(bool(bad.any()))
            self.assertFalse(any(key in view.own_mon_features for key in f.TENDENCY_FIELDS))

    def test_initial_and_next_boundary_literal_columns_original_writers(self):
        state=self.state();c=h.initialize(state,self.tables.belief)
        for turn in (1,2):
            if turn==2:
                h.move(c,state,0,j.ones(2,j.int32),j.ones(2,bool),pp_use=True)
                h.move(c,state,1,j.ones(2,j.int32),j.ones(2,bool),pp_use=True)
                after=state._replace(turn=j.ones(2,j.int32));h.turn_end(c,state,after,j.ones(2,bool));state=after
            for owner in (0,1):
                owners=j.full(2,owner,j.int32);view=tr.readout(c.trackers,owners)
                out=e.encode(state,view,self.tables,observer_sides=owners,legal_action_mask=j.ones((2,9),bool))
                self.assertAlmostEqual(float(out.numeric_features[0,0,e._index('TURN_COUNT')]),turn/1000,places=7)
                self.assertTrue(bool(out.attention_mask[:,22].all()))
                self.assertTrue(bool((out.numeric_features[:,22,e._index('PRESENT')]==1).all()))
                np.testing.assert_array_equal(out.categorical_ids[:,22,s.CATEGORY_ROLE],self.tables.features.stats_role)
                for token in (1,7):
                    # Use the original exact-state writer for active-age meaning.
                    nums=[0.]*200
                    exact=NS(turns_active=turn,revealed_ability=None,possible_abilities=(),ruled_out_abilities=())
                    s._encode_mon_exact_state(nums,NS(active=True,ability=None),exact,
                        role='self'if token==1 else'opponent',status='',fainted=False)
                    self.assertAlmostEqual(float(out.numeric_features[0,token,e._index('TURNS_ACTIVE')]),nums[s.NUMERIC_TURNS_ACTIVE],places=7)
                self.assertEqual(float(out.numeric_features[0,1,e._index('MON_TURNS_ACTIVE_TOTAL')]),0)
                self.assertEqual(float(out.numeric_features[0,1,e._index('MON_STAYED_AND_ATTACKED')]),0)
                if turn==1:
                    for token in (1,7,23):self.assertEqual(int(out.categorical_ids[0,token,s.CATEGORY_LAST_USED_MOVE]),int(self.tables.features.switch_categories[0]))
                else:self.assertAlmostEqual(float(out.numeric_features[0,7,e._index('MON_TURNS_ACTIVE_TOTAL')]),2/64)

    def test_original_first_stay_switch_matchup_and_opportunity_algebra(self):
        state=self.state();c=h.initialize(state,self.tables.belief);r=self.reference()
        for line in ('|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100','|turn|1'):
            r=self.line(r,line)
        self.compare_tendency(c,state,r)
        yes=j.ones(2,bool);old=state.active[:,1]
        h.switch_decision(c,state,1,yes,baton=j.zeros(2,bool))
        state=state._replace(active=state.active.at[:,1].set(1));h.install(c,state,1,yes);h.withdraw(c,state,1,old,yes)
        r=self.line(r,'|switch|p2a: Gardevoir|Gardevoir, L80|100/100');self.compare_tendency(c,state,r)
        for turn in (1,2):
            h.move(c,state,0,j.ones(2,j.int32),yes,pp_use=True)
            r=self.line(r,'|move|p1a: Snorlax|Surf|p2a: Gardevoir');self.compare_tendency(c,state,r)
            if turn==1:
                after=state._replace(turn=j.ones(2,j.int32));h.turn_end(c,state,after,yes);state=after
                r=self.line(r,'|turn|2');self.compare_tendency(c,state,r)
        # A second declaration of the same stay is NOT another stayed datum.
        self.assertEqual(int(c.trackers.mon_features['stayed_and_attacked'][0,0,0]),1)

    def test_mixed_rows_pending_replacement_terminal_and_stats_mask(self):
        state=self.state();c=h.initialize(state,self.tables.belief);before=c.trackers
        def move_tracker(st,trackers):
            current=h.Context(trackers,self.tables.belief)
            h.move(current,st,j.array([0,1],j.int32),j.ones(2,j.int32),j.array([True,False]),pp_use=True)
            return current.trackers
        c.trackers=jax.jit(move_tracker)(state,c.trackers)
        self.assertEqual(c.trackers.tendency.opportunities.tolist(),[[1,0],[0,0]])
        after=state._replace(turn=j.ones(2,j.int32),pending_residual=j.array([True,False]),forced_switch=j.array([[False,False],[True,False]]))
        h.turn_end(c,state,after,j.ones(2,bool));self.assertEqual(c.trackers.turn.tolist(),[1,1])
        ready=after._replace(pending_residual=j.zeros(2,bool),forced_switch=j.zeros((2,2),bool))
        h.turn_end(c,after,ready,j.zeros(2,bool));self.assertEqual(c.trackers.turn.tolist(),[2,2])
        h.turn_end(c,after,ready,j.zeros(2,bool));self.assertEqual(c.trackers.turn.tolist(),[2,2])
        terminal=ready._replace(done=j.ones(2,bool),turn=j.full(2,2,j.int32))
        h.turn_end(c,ready,terminal,j.ones(2,bool));self.assertEqual(c.trackers.turn.tolist(),[2,2])
        n,cat,on,bad=f.project_stats({'opp_switch_count':j.ones(2)},self.tables.features._replace(stats_enabled=j.array(False)),2)
        self.assertFalse(bool(n.any()));self.assertFalse(bool(cat.any()));self.assertFalse(bool(on.any()))
        # An explicit own tendency input cannot populate opponent-only columns.
        n,_,_=f.project_mon(j.zeros((2,6,132)),j.zeros((2,6,41),j.int32),{'stayed_and_attacked':j.ones((2,6))},
            active=j.ones((2,6),bool),present=j.ones((2,6),bool),tables=self.tables.features,opponent_tendency=False)
        self.assertFalse(bool(n.any()))
        n,_,_=f.project_mon(j.zeros((2,6,132)),j.zeros((2,6,41),j.int32),{'stayed_and_attacked':j.ones((2,6))},
            active=j.ones((2,6),bool),present=j.ones((2,6),bool),tables=self.tables.features._replace(stats_enabled=j.array(False)))
        self.assertFalse(bool(n.any()))

    def test_transition_only_changes_actual_observer_call_and_original_physics_bytes(self):
        root=Path(__file__).resolve().parents[1]
        original=subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],cwd=root,text=True)
        # This exact two-boundary-call proof concerns the original change.
        # Retain it at that immutable feature commit; current tracker behavior
        # is exercised by the other tests and the full integrated source pins.
        introduced='693e7ffb2acd420fc692619229d276e129f25bf5'
        current=subprocess.check_output(['git','show',introduced+':gpu_engine/transition.py'],cwd=root,text=True)
        new="device_public.switch_decision(_device_public,work,side,do_switch & ~replacement,baton=_device_baton)"
        old="device_public.feature(_device_public,side,old_active,'switched_before_attack',jnp.ones_like(old_active),do_switch & ~replacement,add=True)"
        self.assertEqual(current.count(new),1);self.assertEqual(current.replace(new,old),original)
        for name in ('state.py','mechanics.py','entry.py','damage_moves.py','secondary.py'):
            self.assertEqual(subprocess.check_output(['git','show',introduced+':gpu_engine/'+name],cwd=root),subprocess.check_output(['git','show',BASE+':gpu_engine/'+name],cwd=root))

    def test_original_called_child_lastmove_and_preparation_header_are_distinct(self):
        import test_gpu_engine_public_moves as fixture
        fixture.PublicMovesTests.setUpClass();original=fixture.PublicMovesTests()
        moves,beliefs,mons=original.initial(batch=2)
        for line in ('|move|p1a: Snorlax|Body Slam|p2a: Absol',
                     '|move|p1a: Snorlax|Surf|p2a: Absol|[from] move: Sleep Talk'):
            value=original.apply_line(moves,beliefs,mons,line)
            self.assertTrue(bool(value.admitted.all()));moves,beliefs=value.state,value.beliefs
        state=self.state();c=h.initialize(state,self.tables.belief);yes=j.ones(2,bool)
        h.move(c,state,0,j.full(2,2,j.int32),yes,pp_use=True)
        h.move(c,state,0,j.ones(2,j.int32),yes,pp_use=False,count_turn=False)
        self.assertEqual(original.catalog.names[int(moves.last_id[0,0])],
            self.metadata['vocabulary']['moves'][int(c.trackers.last_moves[0,0,0])])
        self.assertEqual(c.trackers.tendency.opportunities.tolist(),[[1,0],[1,0]])
        # Preparation-only -singleturn remains a reveal, not a MOVE header;
        # only the subsequent ordinary header supplies lastmove/tendency.
        fresh=h.initialize(state,self.tables.belief)
        h.move(fresh,state,0,j.ones(2,j.int32),yes,pp_use=False,count_turn=False,preparation=True)
        self.assertEqual(int(fresh.trackers.last_moves[0,0,0]),-1)
        self.assertEqual(int(fresh.trackers.mon_features['stayed_and_attacked'][0,0,0]),0)
        self.assertEqual(int(fresh.trackers.tendency.opportunities[0,0]),0)
        self.assertEqual(int(fresh.trackers.observed_moves[0,0,0,0]),1)
        h.move(fresh,state,0,j.ones(2,j.int32),yes,pp_use=True)
        self.assertEqual(int(fresh.trackers.mon_features['stayed_and_attacked'][0,0,0]),1)

    def test_effective_gen3_failed_focus_retains_move_before_cant_order(self):
        import json,os
        root=os.environ['POKEZERO_SHOWDOWN_ROOT']
        code="""const {Dex}=require(process.argv[1]+'/dist/sim/dex.js');const d=Dex.mod('gen3'),m=d.moves.get('focuspunch'),a=d.data.Scripts.actions;const events=[];const result=m.onTry.call({attrLastMove:x=>events.push(['attr',x]),add:(...x)=>events.push(['add',...x.map(String)])},{toString:()=> 'p1a: Snorlax',volatiles:{focuspunch:{lostFocus:true}}});const use=a.useMoveInner.toString(),hit=a.tryMoveHit.toString();console.log(JSON.stringify({beforeMove:typeof m.beforeMoveCallback,resultIsNull:result===null,events,move:use.indexOf('this.battle.addMove('),hit:use.indexOf('this.tryMoveHit('),onTry:hit.indexOf('singleEvent("Try",')}));"""
        actual=json.loads(subprocess.check_output(['node','-e',code,root],text=True))
        self.assertEqual(actual['beforeMove'],'undefined');self.assertTrue(actual['resultIsNull'])
        self.assertEqual(actual['events'],[['attr','[still]'],['add','cant','p1a: Snorlax','Focus Punch','Focus Punch']])
        self.assertGreaterEqual(actual['move'],0);self.assertGreater(actual['hit'],actual['move']);self.assertGreaterEqual(actual['onTry'],0)
        # The ordinary MOVE has already opened this original-source window;
        # the following CANT does not erase its header/first-stay effects.
        refstate=self.reference()
        for line in ('|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100','|turn|1',
                     '|move|p1a: Snorlax|Surf|p2a: Absol|[still]','|cant|p1a: Snorlax|Focus Punch|Focus Punch'):
            refstate=self.line(refstate,line)
        self.assertEqual(int(refstate.mon[0,0,0,1]),1);self.assertEqual(int(refstate.opportunities[0,0]),1)


if __name__=='__main__':unittest.main()
