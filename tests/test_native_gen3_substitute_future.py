"""Native Gen3 continuation controls for the inert Substitute-storage quotient."""
from collections import defaultdict
from copy import deepcopy
import importlib.util
import math
import unittest

from gpu_engine import future_state as f
from tests.test_gpu_engine_future_state import FAKE, fixture


@unittest.skipUnless(importlib.util.find_spec('poke_engine'),'requires actual native Gen3 binding')
class NativeSubstituteFutureTests(unittest.TestCase):
    def law(self,side,stale,*,live=False,create=False):
        import poke_engine as engine
        data,meta=fixture()
        meta=deepcopy(meta)
        meta['vocabulary']['moves']+=['substitute','splash','seismictoss']
        names=meta['vocabulary']['moves']
        actor=f.SIDE_NAMES.index(side);foe=1-actor
        data['moves'][actor][0]=[names.index('substitute'),names.index('splash'),1,1]
        data['moves'][foe][0]=[names.index('seismictoss'),names.index('splash'),1,1]
        data['level'][foe][0]=10
        native=f.native_input_from_jax(data,None,meta,module=FAKE)
        self.assertIsInstance(native,f.NativeInput,native)
        target=getattr(native.state,side);target.substitute_health=stale
        if live:target.volatile_statuses.add('substitute')
        parent=engine.State.from_string(native.state.to_string());before=parent.to_string()
        moves=['splash','splash'];moves[actor]='substitute' if create else 'splash'
        if not create:moves[foe]='seismictoss'
        law=defaultdict(float);health=[]
        for branch in engine.generate_instructions(parent,*moves):
            mass=float(branch.percentage)/100
            self.assertTrue(math.isfinite(mass) and mass>=0)
            if not mass:continue
            after=parent.apply_instructions(branch)
            self.assertEqual(after.reverse_instructions(branch).to_string(),before,
                             'each instruction sequence must restore its own raw parent')
            canonical=f.canonical_native_future(after,native.certificate)
            self.assertIsInstance(canonical,f.CanonicalFuture,canonical)
            law[canonical.key]+=mass
            health.append(f._native_wire(after.to_string())[side]['substitute_health'])
        self.assertEqual(parent.to_string(),before,'native generation mutated parent')
        self.assertAlmostEqual(math.fsum(law.values()),1,places=6)
        return dict(law),health

    def test_real_native_damage_ignores_absent_substitute_storage_both_seats(self):
        for side in f.SIDE_NAMES:
            clean,_=self.law(side,0)
            stale,health=self.law(side,73)
            self.assertTrue(clean==stale,'inactive scalar changed full canonical chance law')
            self.assertEqual(set(health),{73},'raw stale storage remains evidence')

    def test_real_native_new_substitute_overwrites_stale_storage_both_seats(self):
        for side in f.SIDE_NAMES:
            clean,health=self.law(side,0,create=True)
            stale,other=self.law(side,73,create=True)
            self.assertTrue(clean==stale,'new Substitute did not overwrite stale storage')
            self.assertEqual(set(health+other),{250})

    def test_real_native_live_substitutes_remain_distinct_both_seats(self):
        for side in f.SIDE_NAMES:
            first,h1=self.law(side,73,live=True)
            second,h2=self.law(side,74,live=True)
            self.assertNotEqual(first,second)
            self.assertEqual(set(h1),{63})
            self.assertEqual(set(h2),{64})

    def test_real_native_ordinary_and_completed_baton_switches_keep_the_lifecycle(self):
        import poke_engine as engine
        for actor,side in enumerate(f.SIDE_NAMES):
            for baton in (False,True):
                data,meta=fixture();meta['vocabulary']['species'].append('pikachu')
                data['species'][actor][1]=2
                native=f.native_input_from_jax(data,None,meta,module=FAKE)
                target=getattr(native.state,side)
                target.substitute_health=73;target.volatile_statuses={'substitute'}
                target.baton_passing=baton;target.force_switch=baton
                parent=engine.State.from_string(native.state.to_string());before=parent.to_string()
                moves=['none','none'];moves[actor]='pikachu'
                branches=engine.generate_instructions(parent,*moves)
                self.assertAlmostEqual(math.fsum(float(b.percentage)/100 for b in branches),1,places=6)
                for branch in branches:
                    if not branch.percentage:continue
                    after=parent.apply_instructions(branch)
                    self.assertEqual(after.reverse_instructions(branch).to_string(),before)
                    result=f.canonical_native_future(after,native.certificate)
                    self.assertIsInstance(result,f.CanonicalFuture,result)
                    key=dict(result.key)
                    self.assertEqual(key[side+'.active_index'],'1')
                    self.assertEqual(key[side+'.substitute_health'],73 if baton else 0)
                    self.assertEqual(f._native_wire(result.raw_native)[side]['substitute_health'],73)


if __name__=='__main__':unittest.main()
