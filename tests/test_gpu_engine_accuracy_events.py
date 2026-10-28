"""Pinned complete accuracy-domain controls, not game/fidelity qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class AccuracyEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_gpu_engine_hustle import HustleTests
        HustleTests.setUpClass();cls.f=HustleTests()
        cls.catalog=json.loads((Path(__file__).parent/'data/gpu_accuracy_events_catalog.json').read_text())
        cls.f.metadata=cls.catalog['metadata'];cls.f.table=cls.f.tr.tables_from_metadata(cls.f.metadata)

    def live(self,side,move,source=0,target=0,weather=0):
        f=self.f;j=f.j;s=f.live(side,move,batch=100)
        ability=s.ability.at[:,side,0].set(source).at[:,1-side,0].set(target)
        return s._replace(ability=ability,base_ability=ability,
            weather=j.full_like(s.weather,weather),weather_counter=j.full_like(s.weather_counter,5))

    def ability(self,name):return self.f.metadata['vocabulary']['abilities'].index(name)

    def test_status_type_and_composed_events_both_seats(self):
        f=self.f
        for side in (0,1):
            for source,target,weather,count in [('_none','_none',0,85),('hustle','_none',0,68),
                    ('compoundeyes','_none',0,100),('_none','sandveil',3,68),
                    ('hustle','sandveil',3,54),('compoundeyes','sandveil',3,88)]:
                with self.subTest(side=side,source=source,target=target):
                    s=self.live(side,'toxic',self.ability(source),self.ability(target),weather)
                    out=f.play(s,side).state
                    self.assertEqual((out.status[:,1-side,0]!=0).tolist(),[True]*count+[False]*(100-count))

    def test_compound_eyes_and_sand_veil_inherit_scalar_gen4_rounding(self):
        f=self.f
        for source,target,weather,count in [('compoundeyes','_none',0,97),
                ('compoundeyes','sandveil',3,78),('_none','sandveil',3,60)]:
            s=self.live(0,'sleeppowder',self.ability(source),self.ability(target),weather)
            out=f.play(s).state
            self.assertEqual((out.status[:,1,0]!=0).tolist(),[True]*count+[False]*(100-count))

    def test_always_hit_rain_thunder_and_suppressed_sand(self):
        f=self.f;j=f.j
        for side in (0,1):
            s=self.live(side,'aerialace',self.ability('hustle'),self.ability('sandveil'),3)
            s=s._replace(boosts=s.boosts.at[:,side,0,5].set(-6).at[:,1-side,0,6].set(6))
            self.assertTrue(bool(j.all(f.play(s,side).state.hp[:,1-side,0]<1000)))
            s=self.live(side,'thunder',0,0,1)._replace(boosts=s.boosts.at[:,side,0,5].set(-6).at[:,1-side,0,6].set(6))
            self.assertTrue(bool(j.all(f.play(s,side).state.hp[:,1-side,0]<1000)))
            s=self.live(side,'toxic',self.ability('airlock'),self.ability('sandveil'),3)
            self.assertEqual((f.play(s,side).state.status[:,1-side,0]!=0).tolist(),[True]*85+[False]*15)

    def test_plain_binary64_boundary_is_shared_with_jit(self):
        f=self.f;j=f.j
        # Pinned staged80*(5/3)/(8/3) is just above50, hence51 integer rolls.
        self.assertEqual(int(j.sum(f.m.accuracy_hit(80,2,5,j.arange(100)))),51)
        s=self.live(0,'toxic')._replace(boosts=self.live(0,'toxic').boosts.at[:,0,0,5].set(2).at[:,1,0,6].set(5))
        table=f.table._replace(accuracy=f.table.accuracy.at[f.id('toxic')].set(80))
        eager=f.play(s,table=table);fast=f.play(s,table=table,jit=True);f.jax.block_until_ready(fast)
        self.assertEqual((eager.state.status[:,1,0]!=0).tolist(),[True]*51+[False]*49)
        for a,b in zip(f.jax.tree.leaves(eager),f.jax.tree.leaves(fast)):
            self.assertTrue(bool(j.array_equal(a,b)))

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown')
    def test_all102414_pinned_numeric_cells_and_native_true_catalog(self):
        f=self.f;j=f.j
        r=subprocess.run([NODE,str(Path(__file__).with_name('gpu_accuracy_events_oracle.cjs')),SHOWDOWN],
            text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr);d=json.loads(r.stdout)
        self.assertEqual({k:v for k,v in d.items() if k!='counts'},
            {k:v for k,v in self.catalog.items() if k!='generator'})
        self.assertEqual(len(d['counts']),102414)
        base=j.tile(j.repeat(j.arange(101),169),6)
        a=j.tile(j.repeat(j.arange(-6,7),13),101*6);e=j.tile(j.arange(-6,7),101*13*6)
        modifiers=self.f.m._ACCURACY_MODIFIERS
        h=j.repeat(j.array([v[0] for v in modifiers]),17069)
        c=j.repeat(j.array([v[1] for v in modifiers]),17069)
        s=j.repeat(j.array([v[2] for v in modifiers]),17069)
        count=j.array(d['counts'])
        self.assertTrue(bool(j.all(f.m.accuracy_hit(base,a,e,count-1,hustle=h,compound_eyes=c,sand_veil=s)|(count==0))))
        self.assertTrue(bool(j.all(~f.m.accuracy_hit(base,a,e,count,hustle=h,compound_eyes=c,sand_veil=s))))


if __name__=='__main__':unittest.main()
