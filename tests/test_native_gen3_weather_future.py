"""Actual native clock/switch continuations, not replay or fidelity credit."""
from copy import deepcopy
import importlib.util
import math
import unittest

from gpu_engine import future_state as f
from tests.test_gpu_engine_future_state import fixture


@unittest.skipUnless(importlib.util.find_spec('poke_engine'),'requires actual native Gen3 binding')
class NativeWeatherFutureTests(unittest.TestCase):
    def live(self):
        data,meta=fixture()
        meta['vocabulary']['species'].append('pikachu')
        meta['vocabulary']['moves'].append('splash')
        meta['vocabulary']['abilities']+=['drizzle','drought','sandstream','airlock','cloudnine']
        for side in range(2):
            data['species'][side][1]=2
            data['moves'][side]=[[9]*4 for _ in range(6)]
        return data,meta

    def compare(self,data,meta,moves,expected):
        import poke_engine as engine
        native=f.native_input_from_jax(data,None,meta,module=engine)
        self.assertIsInstance(native,f.NativeInput,native)
        before=native.state.to_string()
        self.assertEqual(dict(native.certificate.input_fields)['weather_counter'],data['weather_counter'])
        target=f.canonical_jax_future(expected,None,meta,certificate=native.certificate)
        self.assertIsInstance(target,f.CanonicalFuture,target)
        branches=engine.generate_instructions(native.state,*moves)
        mass=0;count=0
        for branch in branches:
            probability=float(branch.percentage)/100
            self.assertTrue(math.isfinite(probability) and probability>=0)
            mass+=probability
            if not probability:continue
            count+=1
            after=native.state.apply_instructions(branch)
            result=f.canonical_native_future(after,native.certificate)
            self.assertIsInstance(result,f.CanonicalFuture,result)
            self.assertEqual(result.key,target.key)
            self.assertEqual(f._native_wire(result.raw_native)['weather_turns_remaining'],
                             -1 if expected['weather'] and expected['weather_counter']==0
                             else expected['weather_counter'])
            self.assertEqual(after.reverse_instructions(branch).to_string(),before)
        self.assertGreater(count,0);self.assertAlmostEqual(mass,1,places=6)
        self.assertEqual(native.state.to_string(),before)

    def test_idle_permanent_temporary_expiry_and_suppression_keep_underlying_clock(self):
        for weather in (1,2,3,4):
            for clock in (0,1,2,5):
                for suppressor in (None,0,1):
                    with self.subTest(weather=weather,clock=clock,suppressor=suppressor):
                        data,meta=self.live();data.update(weather=weather,weather_counter=clock)
                        if suppressor is not None:data['ability'][suppressor][0]=5+suppressor
                        expected=deepcopy(data)
                        expected['weather_counter']=max(0,clock-1)
                        if clock==1:expected['weather']=0
                        for side in range(2):
                            expected['pp'][side][0][0]-=1
                            expected['active_turns'][side]+=1
                            if expected['weather'] in (3,4) and suppressor is None:
                                expected['hp'][side][0]-=62
                        self.compare(data,meta,('splash','splash'),expected)

    def test_weather_ability_switches_replace_temporary_clocks_both_seats(self):
        for actor in (0,1):
            for ability,weather in ((2,1),(3,2),(4,3)):
                for initial,clock in ((0,0),(weather,0),(weather,3),((weather%3)+1,0)):
                    with self.subTest(actor=actor,ability=ability,initial=initial,clock=clock):
                        data,meta=self.live();data.update(weather=initial,weather_counter=clock)
                        data['ability'][actor][1]=ability
                        expected=deepcopy(data);expected['active'][actor]=1
                        expected.update(weather=weather,weather_counter=0)
                        expected['pp'][1-actor][0][0]-=1
                        expected['active_turns'][1-actor]+=1
                        if weather==3:
                            expected['hp'][actor][1]-=62
                            expected['hp'][1-actor][0]-=62
                        moves=['splash','splash'];moves[actor]='pikachu'
                        self.compare(data,meta,moves,expected)


if __name__=='__main__':unittest.main()
