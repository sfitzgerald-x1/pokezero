"""Pinned Gen3 residual cure, exact law, and saved independent chance stream."""
from fractions import Fraction
import unittest
import numpy as np
import jax
import jax.numpy as j
from gpu_engine import transition as t
from gpu_engine.finite_law import Axis, COORDINATES, Observation, Plan, chance_fields
from gpu_engine.state import BattleState, empty_state
import test_gpu_engine_fidelity_m2 as fixtures


class ShedSkinTests(unittest.TestCase):
    build=fixtures.TransitionTests.build
    chance=fixtures.TransitionTests.chance
    play=fixtures.TransitionTests.play
    at=fixtures.TransitionTests.at

    @classmethod
    def setUpClass(cls):
        cls.t,cls.State,cls.empty=t,BattleState,staticmethod(empty_state)
        meta=fixtures.metadata();meta['vocabulary']['abilities']+=['shedskin','speedboost']
        cls.skin=meta['vocabulary']['abilities'].index('shedskin')
        cls.table=t.tables_from_metadata(meta)
        cls.jitted={'step':jax.jit(t.step_with_chance)}

    def statused(self,status=2):
        return self.build([{'ability':[((0,0),self.skin),((1,0),self.skin)],
            'status':[((0,0),status),((1,0),status)],
            'status_counter':[((0,0),3),((1,0),3)]}]*8)

    def test_all_hundred_rolls_and_independent_seats(self):
        hits=[]
        for start in range(0,100,8):
            rolls=np.minimum(np.arange(start,start+8),99)
            chance=self.chance()._replace(shed_skin=j.asarray(np.stack([rolls,99-rolls],axis=1)))
            after=self.play(self.statused(),chance)
            for side,values in ((0,rolls),(1,99-rolls)):
                np.testing.assert_array_equal(self.at(after,'status',side)==0,values<33)
                np.testing.assert_array_equal(self.at(after,'hp',side),np.where(values<33,320,280))
            hits.extend((self.at(after,'status',0)==0)[:min(8,100-start)])
        self.assertEqual(sum(hits),33)

    def test_toxic_sleep_and_rest_cure_reset(self):
        for status in (4,5):
            state=self.statused(status)
            if status==5:
                state=state._replace(rest_origin=state.rest_origin.at[:,:,0].set(True),
                    sleep_skipped=state.sleep_skipped.at[:,:,0].set(1))
            after=self.play(state,self.chance()._replace(shed_skin=j.full((8,2),32,j.int32)))
            for field in ('status','status_counter','sleep_skipped','rest_origin'):
                np.testing.assert_array_equal(np.asarray(getattr(after,field))[:,:,0],0)

    def test_saved_roll_replayed_after_baton_replacement(self):
        state=self.statused()._replace(moves=self.statused().moves.at[:,0,0,0].set(fixtures.BATON))
        saved=self.play(state,self.chance()._replace(shed_skin=j.zeros((8,2),j.int32)))
        self.assertTrue(np.asarray(saved.pending_residual).all())
        np.testing.assert_array_equal(saved.status[:,1,0],2)
        resumed=self.play(saved,self.chance()._replace(shed_skin=j.full((8,2),99,j.int32)),a0=4)
        np.testing.assert_array_equal(resumed.status[:,1,0],0)
        np.testing.assert_array_equal(resumed.shed_skin_draws,0)

    def test_eligibility_and_entry_turn(self):
        state=self.statused()
        state=state._replace(status=state.status.at[1,0,0].set(0),
            hp=state.hp.at[2,0,0].set(1),
            weather=state.weather.at[2].set(4),weather_counter=state.weather_counter.at[2].set(2),
            ability=state.ability.at[0,0,0].set(0).at[3,0,1].set(self.skin),
            status_counter=state.status_counter.at[3,0,1].set(2))
        state=state._replace(status=state.status.at[3,0,1].set(4))
        actions=np.zeros(8,np.int32);actions[3]=4
        after=self.play(state,self.chance()._replace(shed_skin=j.zeros((8,2),j.int32)),a0=actions)
        self.assertEqual(int(after.status[0,0,0]),2)  # another ability never cures
        self.assertEqual(int(after.status[1,0,0]),0)
        self.assertEqual(int(after.hp[2,0,0]),0)
        self.assertEqual(int(after.status[2,0,0]),2)  # weather KO cannot cure
        self.assertEqual(int(after.active[3,0]),1)
        self.assertEqual(int(after.status[3,0,1]),0)  # newly installed holder can cure

    def test_legacy_decode_and_full_roll_range(self):
        from gpu_engine.device_recorded_reencoding import decode_chance
        chance=t.sample_chance_inputs(empty_state(1))
        packet={k:np.asarray(v).tolist() for k,v in chance._asdict().items()}
        packet['shed_skin']=[[99,0]]
        np.testing.assert_array_equal(decode_chance(packet).shed_skin,[[99,0]])
        legacy={k:v for k,v in packet.items() if k not in ('effect','shed_skin')}
        self.assertIsNone(decode_chance(legacy).shed_skin)
        packet['shed_skin']=[[100,0]]
        with self.assertRaisesRegex(ValueError,'shape/range'):decode_chance(packet)

    def test_new_roll_preserves_original_sampled_fields(self):
        state=self.statused();chance=t.sample_chance_inputs(state)
        keys=jax.vmap(lambda key:jax.random.split(key,2))(state.keys)[:,1]
        expected=jax.vmap(lambda key:jax.random.randint(key,(2,12),0,
            j.asarray([2,100,100,100,16,16,100,4,8,2,2,2],j.int32)))(keys)
        np.testing.assert_array_equal(chance.draws,expected)
        np.testing.assert_array_equal(chance.effect,t.effect_chance(keys))
        self.assertEqual(chance.shed_skin.shape,(8,2))

    def test_fidelity_tape_matches_sampled_and_saved_cure_stream(self):
        from gpu_engine.fidelity import chance_tape
        state=self.statused();chance=t.sample_chance_inputs(state)
        np.testing.assert_array_equal(chance_tape(state,0,0)['shed_skin'],chance.shed_skin[0])
        state=state._replace(pending_residual=j.ones(8,bool),
            shed_skin_draws=j.tile(j.asarray([[0,99]],j.int32),(8,1)))
        np.testing.assert_array_equal(chance_tape(state,0,1)['shed_skin'],[0,99])


class RegressionTests(unittest.TestCase):
    def test_simultaneous_incoming_trace_holder_is_involved(self):
        from gpu_engine.v2e_regress import Pre,shed_skin_holders
        from test_gpu_engine_v2e_regress import pre_arrays,VOCAB,CALLABLE
        state=pre_arrays(2);vocab=dict(VOCAB,abilities=['_none','trace','shedskin'])
        state['ability'][0,0,1]=2;state['ability'][0,1,1]=1
        p=Pre({**{'state.'+k:v for k,v in state.items()},'actions':np.full((2,2),4)},
            vocab,CALLABLE,np.zeros((2,2),bool))
        np.testing.assert_array_equal(shed_skin_holders(p),[[True,True],[False,False]])


class ExactLawTests(unittest.TestCase):
    def test_independent_two_seat_mass(self):
        axes=tuple(Axis(name,(Observation('less_than',33),),('Shed Skin',)) if name.startswith('shed_skin.')
            else Axis(name,(),inactive_reason='authored inactive') for name,_ in COORDINATES)
        plan=Plan(axes);cases=list(plan.bounded_cases(4))
        self.assertEqual(len(cases),4)
        self.assertEqual(sorted(c.mass for c in cases),sorted([Fraction(33,100)**2,Fraction(67,100)**2,
            Fraction(33*67,10000),Fraction(33*67,10000)]))
        self.assertEqual(COORDINATES[-2:],(('shed_skin.0',100),('shed_skin.1',100)))
        self.assertEqual(len(chance_fields(dict(cases[0].values))['shed_skin']),2)

if __name__=='__main__':unittest.main()
