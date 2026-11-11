"""Actual public Wish clock algebra, no engine turn, model or replay run."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_device_public_hooks as h,engine_device_trackers as tr
from gpu_engine.state import empty_state
from pokezero import showdown as reference

def initial(batch=1):
    shape=(batch,2,6)
    s=empty_state(batch)._replace(species=j.ones(shape,j.int32),level=j.full(shape,80,j.int32),
        hp=j.full(shape,100,j.int32),max_hp=j.full(shape,100,j.int32),done=j.zeros(batch,bool))
    return s,h.initialize(s)


def clock(c,owner=0):
    v=tr.readout(c.trackers,j.full(len(c.trackers.error),owner,j.int32)).field_features
    return tuple(v[k] for k in ('self_wish_turns','opp_wish_turns','self_wish_pending','opp_wish_pending'))


def source(lines,slot='p1'):
    # Original public parser only: these literal facts are synthetic UNIT
    # observations, not a scientific simulator replay or new game.
    state=reference.parse_showdown_replay(lines)
    return reference._wish_turns_remaining(state,slot),reference._wish_pending(state,slot)


class DeviceWishClockTests(unittest.TestCase):
    def test_declaration_and_expired_clock_match_original_public_turns(self):
        s,c=initial();lines=['|turn|1','|move|p1a: Fixture|Wish|p1a: Fixture']
        h.wish(c,0,j.ones(1,bool),start=True)
        turns,pending=source(lines)
        self.assertEqual(int(clock(c)[0][0]),turns)
        self.assertEqual(bool(clock(c)[2][0]),pending)
        h.wish(c,0,j.ones(1,bool));h.public_turn(c,s,j.ones(1,bool),j.asarray([2]))
        lines+=['|turn|2']
        self.assertEqual(int(clock(c)[0][0]),source(lines)[0])
        # A full-HP silent landing has no public heal clear: it expires only
        # at TURN3, not at the private residual cut while TURN2 is still open.
        h.wish(c,0,j.ones(1,bool))
        self.assertEqual(int(clock(c)[0][0]),source(lines)[0])
        h.public_turn(c,s,j.ones(1,bool),j.asarray([3]));lines+=['|turn|3']
        self.assertEqual(int(clock(c)[0][0]),source(lines)[0])
        self.assertFalse(bool(clock(c)[2][0]))

    def test_rearm_guard_actual_heal_and_wish_pass_preserve_side_clock(self):
        s,c=initial();lines=['|turn|1','|move|p1a: Fixture|Wish|p1a: Fixture']
        h.wish(c,0,True,start=True);h.wish(c,0,True)
        h.public_turn(c,s,j.ones(1,bool),j.asarray([2]));lines+=['|turn|2']
        # Even an observed second declaration cannot extend a pending Wish.
        h.wish(c,0,True,start=True);lines+=['|move|p1a: Fixture|Wish|p1a: Fixture']
        self.assertEqual(int(clock(c)[0][0]),source(lines)[0])
        switched=s._replace(active=s.active.at[0,0].set(3));h.install(c,switched,0,j.ones(1,bool))
        lines+=['|switch|p1a: Other|Other, L80|100/100']
        self.assertEqual(int(clock(c)[0][0]),source(lines)[0])
        h.wish(c,0,True,resolved=j.ones(1,bool));lines+=['|-heal|p1a: Other|100/100|[from] move: Wish']
        self.assertEqual(int(clock(c)[0][0]),source(lines)[0]);self.assertFalse(bool(clock(c)[2][0]))
        h.public_turn(c,switched,j.ones(1,bool),j.asarray([3]));lines+=['|turn|3']
        h.wish(c,0,True,start=True);lines+=['|move|p1a: Other|Wish|p1a: Other']
        self.assertEqual(int(clock(c)[0][0]),source(lines)[0]);self.assertTrue(bool(clock(c)[2][0]))

    def test_jit_mixed_seats_masks_silent_vs_healed_and_both_owner_readouts(self):
        s,_=initial(4);seats=j.asarray([0,1,0,1],j.int32)
        @jax.jit
        def run(s):
            c=h.initialize(s)
            h.wish(c,seats,j.asarray([True,True,True,False]),start=True)
            h.wish(c,seats,j.ones(4,bool))
            h.public_turn(c,s,j.ones(4,bool),j.full(4,2,j.int32))
            h.wish(c,seats,j.ones(4,bool),resolved=j.asarray([True,False,False,False]))
            return tuple(tr.readout(c.trackers,j.full(4,owner,j.int32)) for owner in (0,1))
        own0,own1=run(s)
        np.testing.assert_array_equal(own0.field_features['self_wish_turns'],[0,0,1,0])
        np.testing.assert_array_equal(own0.field_features['opp_wish_turns'],[0,1,0,0])
        np.testing.assert_array_equal(own0.field_features['self_wish_turns'],own1.field_features['opp_wish_turns'])
        np.testing.assert_array_equal(own0.field_features['self_wish_pending'],own0.field_features['self_wish_turns']>0)

    def test_actual_column_names(self):
        self.assertEqual(reference.v4_numeric_index(reference.NUMERIC_SELF_WISH_TURNS),113)
        self.assertEqual(reference.v4_numeric_index(reference.NUMERIC_OPP_WISH_TURNS),114)
        self.assertEqual(reference.v4_numeric_index(reference.NUMERIC_SELF_HAZARD_EXPECTED),117)
        self.assertEqual(reference.v4_numeric_index(reference.NUMERIC_OPP_HAZARD_EXPECTED),118)


if __name__=='__main__':unittest.main()
