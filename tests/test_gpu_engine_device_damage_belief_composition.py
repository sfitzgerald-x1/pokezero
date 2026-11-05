"""Same device HP cutpoint updates distinct public histories in source order."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_device_public_hooks as h, engine_device_trackers as tr
from test_gpu_engine_device_negative_items import fixture


class DamageBeliefCompositionTests(unittest.TestCase):
    def test_shared_hp_seam_keeps_round_damage_and_pre_item_evidence_separate(self):
        state,tables,leftovers,_=fixture(batch=3)
        @jax.jit
        def run(state):
            c=h.initialize(state,tables);yes=j.ones(3,bool)
            hit=state._replace(hp=state.hp.at[:,1,0].set(160))
            h.hp_line(c,hit,1,yes,source_class=h.HP_ACTION)
            healed=hit._replace(hp=hit.hp.at[1,1,0].set(200))
            h.hp_line(c,healed,1,j.arange(3)==1,source_class=h.HP_PRE_ITEM)
            residual=healed._replace(hp=healed.hp.at[:,1,0].set(150))
            h.hp_line(c,residual,1,j.arange(3)!=2,source_class=h.HP_POST_ITEM)
            h.hp_line(c,residual,1,j.arange(3)==2,source_class=h.HP_UNKNOWN)
            h.turn_end(c,state,residual,j.zeros(3,bool),upkeep=yes,leftovers=leftovers)
            pending=c.trackers
            h.public_turn(c,residual,yes,j.full(3,2,j.int32))
            return pending,c.trackers,tr.readout(c.trackers,j.zeros(3,j.int32)),tr.readout(c.trackers,j.ones(3,j.int32))
        pending,settled,mine,peer=run(state)
        self.assertEqual(settled.belief_mask[:,1,0,0].tolist(),[3,255,255])
        self.assertFalse(bool(settled.negative_items.known.any()))
        np.testing.assert_array_equal(pending.damage_exact_taken[:,1,0],0.)
        for value in (mine.mon_features['last_damage_taken'][:,0],peer.own_mon_features['last_damage_taken'][:,0]):
            np.testing.assert_allclose(value,[.25,.45,.25],rtol=0,atol=1e-7)

    def test_install_resets_round_damage_and_seeds_the_new_encounter(self):
        state,tables,leftovers,_=fixture(batch=2)
        c=h.initialize(state,tables)
        low=state._replace(hp=state.hp.at[:,1,0].set(160))
        h.hp_line(c,low,1,True,source_class=h.HP_ACTION)
        incoming=low._replace(active=low.active.at[:,1].set(3),hp=low.hp.at[:,1,3].set(170))
        h.install(c,incoming,1,True)
        h.turn_end(c,low,incoming,j.zeros(2,bool),upkeep=j.ones(2,bool),leftovers=leftovers)
        self.assertEqual(c.trackers.belief_mask[:,1,1,0].tolist(),[3,3])
        h.public_turn(c,incoming,j.ones(2,bool),j.full(2,2,j.int32))
        self.assertEqual(c.trackers.damage_exact_taken[:,1,3].tolist(),[0.,0.])
        np.testing.assert_array_equal(c.trackers.damage_history.hp[:,1],170)


if __name__=='__main__':unittest.main()
