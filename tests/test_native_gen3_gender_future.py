"""Actual native constructor/positive-support guards, not fidelity credit."""
from copy import deepcopy
import importlib.util
import math
import unittest

from gpu_engine import future_state as f
from tests.test_gpu_engine_future_state import action_fixture


@unittest.skipUnless(importlib.util.find_spec('poke_engine'),'requires actual native Gen3 binding')
class NativeGenderFutureTests(unittest.TestCase):
    def live(self):
        data,meta=action_fixture()
        data['gender']=[[1,2,0,1,2,0],[2,1,0,2,1,0]]
        return data,meta

    def test_all12_genders_roundtrip_and_survive_ordinary_switch_and_turn(self):
        import poke_engine as engine
        data,meta=self.live()
        native=f.native_input_from_jax(data,None,meta,module=engine)
        self.assertIsInstance(native,f.NativeInput,native)
        left=f.canonical_jax_future(data,None,meta,certificate=native.certificate)
        self.assertEqual(left.key,f.canonical_native_future(native.state,native.certificate).key)
        roundtrip=engine.State.from_string(native.state.to_string())
        self.assertEqual(left.key,f.canonical_native_future(roundtrip,native.certificate).key)
        for moves,expected_active in ((('splash','splash'),(0,0)),
                                      (('slaking','splash'),(1,0)),
                                      (('splash','slaking'),(0,1))):
            before=roundtrip.to_string();branches=engine.generate_instructions(roundtrip,*moves)
            self.assertAlmostEqual(math.fsum(float(b.percentage)/100 for b in branches),1,places=6)
            for branch in branches:
                if not branch.percentage:continue
                after=roundtrip.apply_instructions(branch)
                result=f.canonical_native_future(after,native.certificate)
                self.assertIsInstance(result,f.CanonicalFuture,result)
                for side,index in zip(f.SIDE_NAMES,expected_active):
                    self.assertEqual(int(getattr(after,side).active_index),index)
                for side in f.SIDE_NAMES:
                    for p in range(6):
                        key=f'{side}.pokemon[{p}].gender'
                        self.assertEqual(dict(result.key)[key],dict(left.key)[key])
                self.assertEqual(after.reverse_instructions(branch).to_string(),before)
            self.assertEqual(roundtrip.to_string(),before)

    def test_positive_attraction_is_whole_slot_unknown_not_filtered(self):
        import poke_engine as engine
        from gpu_engine.fidelity_panel import require_future,WholeSlotUnknown
        for actor in range(2):
            data,meta=self.live();meta=deepcopy(meta)
            meta['vocabulary']['moves'].append('attract')
            data['moves'][actor][0][0]=len(meta['vocabulary']['moves'])-1
            native=f.native_input_from_jax(data,None,meta,module=engine)
            self.assertIsInstance(native,f.NativeInput,native)
            moves=['splash','splash'];moves[actor]='attract'
            positive_unknown=0;mass=0
            for branch in engine.generate_instructions(native.state,*moves):
                probability=float(branch.percentage)/100;mass+=probability
                if probability<=0:continue
                result=f.canonical_native_future(native.state.apply_instructions(branch),native.certificate)
                if isinstance(result,f.Refused):
                    self.assertIn('NATIVE_UNREPRESENTED_VOLATILE',{g.code for g in result.gaps})
                    with self.assertRaises(WholeSlotUnknown):require_future(result)
                    positive_unknown+=1
            self.assertAlmostEqual(mass,1,places=6)
            self.assertGreater(positive_unknown,0)


if __name__=='__main__':unittest.main()
