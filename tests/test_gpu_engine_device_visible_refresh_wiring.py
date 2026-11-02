"""Actual hook/source-basis wiring only; no engine turn/model/game/replay."""
from types import SimpleNamespace as NS
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_device_public_hooks as h,engine_device_trackers as tr
from gpu_engine import engine_device_visible_belief as visible,public_belief as original
from pokezero.belief import PublicBattleBeliefEngine
import test_gpu_engine_device_visible_belief as source_fixture
import test_gpu_engine_device_public_hooks as fixture


def pending(c,state,side,bits=1,use=True,physical=None):
    p=state.active[j.arange(len(state.hp)),side] if physical is None else physical
    incoming=j.broadcast_to(j.asarray([bits,0],j.uint32),(len(state.hp),2))
    c.trackers,_=visible.register_variant_survivors(c.trackers,c.belief_tables,side,p,incoming,use)


class VisibleRefreshWiringTests(unittest.TestCase):
    def test_current_hard_writes_reach_cached_readout_and_original_source_fallback(self):
        state,_,tables,source,host,_=source_fixture.fixture(3)
        moves=j.stack((host.device.pool,host.device.predicates[host.predicate_id('move','Tackle')],
            host.device.predicates[host.predicate_id('move','move2')]),axis=1)
        tables=NS(**vars(tables),move_mask=moves)
        @jax.jit
        def run(state):
            c=h.initialize(state,tables)
            h.move(c,state,1,j.full(3,2,j.int32),j.asarray([True,False,True]),pp_use=True)
            first=tr.readout(c.trackers,j.zeros(3,j.int32))
            h.reveal_ability(c,state,1,j.asarray([True,False,False]),ability=j.ones(3,j.int32))
            h.reveal_item(c,state,1,j.ones(3,j.int32),j.asarray([True,False,False]))
            return first,c.trackers,tr.readout(c.trackers,j.zeros(3,j.int32))
        first,carry,view=run(state)
        self.assertIsNotNone(carry.visible_belief)
        self.assertEqual(first.belief_mask[:,0,0].tolist(),[4,15,4])
        self.assertEqual(view.belief_mask[:,0,0].tolist(),[0,15,4])
        np.testing.assert_array_equal(view.belief_mask,carry.belief_mask[:,1])
        py=PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=source)
        mon=py._upsert(showdown_slot='p2',species='Swalot')
        mon=py._replace_belief(mon,revealed_moves=('move2',))
        self.assertEqual(mon.candidate_set_count,1)
        mon=py._replace_belief(mon,revealed_ability='Early Bird',revealed_item='Leftovers')
        state_=original.BeliefState(view.species,j.ones_like(view.present),view.belief_mask,
            view.belief_pin,view.belief_pin_active,j.ones_like(view.present),j.zeros_like(view.species))
        _,summary=original.materialize(state_,NS(pool=tables.species_initial_mask))
        self.assertEqual(int(summary.reason[0,0]),2)
        self.assertEqual(int(summary.count[0,0]),mon.candidate_set_count)

    def test_same_percent_hp_line_refreshes_only_actual_subject_and_valid_rows(self):
        s=fixture.state(3)._replace(hp=j.full((3,2,6),509,j.int32),max_hp=j.full((3,2,6),1000,j.int32))
        @jax.jit
        def run(s):
            c=h.initialize(s,fixture.tables())
            pending(c,s,0);pending(c,s,1)
            after=s._replace(hp=s.hp.at[:,1,0].set(508))
            h.hp_line(c,after,1,j.asarray([True,False,False]),source_class=h.HP_ACTION)
            first=c.trackers
            # Polling an unchanged other endpoint is not a source upsert.
            h.observe(c,after,j.asarray([False,True,False]))
            bad=after._replace(max_hp=after.max_hp.at[2,1,0].set(0))
            h.hp_line(c,bad,1,j.asarray([False,False,True]),source_class=h.HP_POST_ITEM)
            return first,c.trackers
        first,after=run(s)
        self.assertEqual(first.visible_belief.visible_pin_active[:,1,0].tolist(),[True,False,False])
        self.assertFalse(bool(after.visible_belief.visible_pin_active[:,0].any()))
        self.assertEqual(after.visible_belief.visible_pin_active[:,1,0].tolist(),[True,False,False])
        self.assertTrue(bool(after.error[2]))
        self.assertEqual(first.hp_percent[:,1,0].tolist(),[51,51,51])

    def test_known_bench_cure_updates_exact_subject_without_revealing_unseen_bench(self):
        s=fixture.state(2);tables=fixture.tables();yes=j.ones(2,bool)
        @jax.jit
        def run(s):
            c=h.initialize(s,tables)
            before=s._replace(status=s.status.at[:,1,3].set(5).at[:,1,5].set(5))
            bench=before._replace(active=before.active.at[:,1].set(3))
            h.install(c,bench,1,yes);h.install(c,before,1,yes)
            pending(c,before,1,physical=j.full(2,3,j.int32))
            after=before._replace(status=before.status.at[:,1,3].set(0).at[:,1,5].set(0))
            h.party_cure(c,before,after,1,j.asarray([True,False]))
            return c.trackers
        t=run(s)
        self.assertEqual(t.status[:,1,1].tolist(),[0,5])
        self.assertEqual(t.visible_belief.visible_pin_active[:,1,1].tolist(),[True,False])
        self.assertEqual(t.physical_to_public[:,1,5].tolist(),[-1,-1])
        self.assertFalse(bool(t.present[:,1,2:].any()))
        self.assertFalse(bool(t.visible_belief.visible_pin_active[:,1,0].any()))

    def test_copied_move_turn_and_sleep_source_upserts_keep_rows_and_hard_evidence_separate(self):
        s=fixture.state(2);tables=fixture.tables()
        @jax.jit
        def run(s):
            c=h.initialize(s,tables);pending(c,s,0);pending(c,s,1)
            h.move(c,s,1,j.full(2,2,j.int32),j.asarray([True,False]),original=False)
            first=c.trackers
            h.public_turn(c,s,j.asarray([False,True]),j.full(2,2,j.int32))
            return first,c.trackers
        first,later=run(s)
        self.assertEqual(first.visible_belief.visible_pin_active[:,1,0].tolist(),[True,False])
        self.assertFalse(bool(first.visible_belief.visible_pin_active[:,0].any()))
        self.assertEqual(later.visible_belief.visible_pin_active[:,0,0].tolist(),[False,True])
        self.assertEqual(later.visible_belief.visible_pin_active[:,1,0].tolist(),[True,True])
        np.testing.assert_array_equal(first.belief_mask,later.belief_mask)
        # Actual sleeping cant updates the active belief; other cant outcomes
        # do not inherit a summary refresh merely from attempted-action use.
        sleepy=s._replace(status=s.status.at[:,1,0].set(5))
        c=h.initialize(sleepy,tables);pending(c,sleepy,1)
        result=NS(status=j.asarray([5,1]),can_act=j.zeros(2,bool),confusion_self_hit=j.zeros(2,bool))
        h.before_move(c,sleepy,1,result,j.ones(2,bool))
        self.assertEqual(c.trackers.visible_belief.visible_pin_active[:,1,0].tolist(),[True,False])


if __name__=='__main__':unittest.main()
