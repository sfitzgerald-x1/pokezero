"""Public MOVE writer comparisons only; no engine step, model or game."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_device_public_hooks as h,engine_device_trackers as tr
from gpu_engine import public_moves,public_mon_state,public_belief
from gpu_engine.state import empty_state
import test_gpu_engine_device_observation as fixtures


class PublicMoveParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeviceObservationTests.setUpClass()
        cls.fixture=fixtures.DeviceObservationTests()
        cls.catalog=cls.fixture.catalog;cls.tables=cls.fixture.tables

    def test_struggle_is_header_not_reveal_or_pp_against_original_writer(self):
        b=4;catalog=self.catalog;rows=j.arange(b);side=j.ones(b,j.int32);slot=j.zeros(b,j.int32)
        source_id=catalog.source.species_id('snorlax');public_id=catalog.species.species_id('snorlax')
        source_species=j.zeros((b,2,6),j.int32).at[:,0,0].set(source_id)
        belief=public_belief.initialize(source_species,catalog.source.device)
        mons=public_mon_state.initialize(b,species_binding=catalog.species.device.binding)
        mons=mons._replace(encounter=mons.encounter.at[:,1,0].set(1),active=mons.active.at[:,1].set(0),
            species=mons.species.at[:,1,0].set(public_id),source_species=mons.source_species.at[:,1,0].set(source_id))
        original=public_moves.initialize(b,catalog.moves.device)
        moves=j.asarray([catalog.moves.move_id(x)for x in ('struggle','surf','struggle','surf')],j.int32)
        valid=j.asarray([True,True,False,True]);caller=j.asarray([public_moves.NONE]*3+[public_moves.SLEEP_TALK],j.int32)
        result=public_moves.record_move(original,belief,mons,catalog.moves.device,catalog.source.device,
            side=side,slot=slot,encounter=j.ones(b,j.int32),move_id=moves,caller=caller,
            has_from=caller!=public_moves.NONE,valid=valid)
        state=empty_state(b)._replace(done=j.zeros(b,bool),species=j.full((b,2,6),2,j.int32),
            hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),level=j.full((b,2,6),80,j.int32))
        engine_moves=j.asarray([3,1,3,1],j.int32)
        def write(state):
            c=h.initialize(state,self.tables.belief)
            h.move(c,state,1,engine_moves,valid,pp_use=caller!=public_moves.SLEEP_TALK)
            return tr.readout(c.trackers,j.zeros(b,j.int32))
        view=jax.jit(write)(state)
        expected_moves=result.state.move_ids[rows,1,0,:4]
        np.testing.assert_array_equal(self.tables.belief.engine_move[view.revealed_moves[:,0]],expected_moves)
        np.testing.assert_array_equal(view.move_pp_used[:,0],result.state.uses[rows,1,0,:4])
        self.assertEqual(view.last_moves[:,0].tolist(),[3,1,-1,1])
        self.assertFalse(bool(np.asarray(view.revealed_moves)[[0,2]].any()))

    def test_unrevealed_struggle_cannot_fill_a_real_set_slot(self):
        state=empty_state(1)._replace(done=j.zeros(1,bool),species=j.full((1,2,6),2,j.int32),
            hp=j.full((1,2,6),100,j.int32),max_hp=j.full((1,2,6),100,j.int32),level=j.full((1,2,6),80,j.int32))
        c=h.initialize(state,self.tables.belief);yes=j.ones(1,bool)
        h.move(c,state,1,j.asarray([3],j.int32),yes,pp_use=True)
        h.move(c,state,1,j.asarray([1],j.int32),yes,pp_use=True)
        view=tr.readout(c.trackers,j.zeros(1,j.int32))
        self.assertEqual(view.revealed_moves[0,0].tolist(),[1,0,0,0])
        self.assertEqual(view.move_pp_used[0,0].tolist(),[1,0,0,0])


if __name__=='__main__':unittest.main()
