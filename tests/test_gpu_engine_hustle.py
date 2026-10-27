"""Authored Hustle controls, including pinned binary64 accuracy boundaries."""
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
class HustleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import mechanics,transition
        from gpu_engine.state import empty_state
        cls.jax,cls.j,cls.m,cls.tr,cls.empty=jax,j,mechanics,transition,staticmethod(empty_state)
        cls.metadata=json.loads((Path(__file__).parent/'data/gpu_hustle_catalog.json').read_text())['metadata']
        cls.table=transition.tables_from_metadata(cls.metadata)

    def id(self,name):return self.metadata['vocabulary']['moves'].index(name)

    def live(self,side=0,move='hiddenpowerground',batch=100):
        j=self.j;s=self.empty(batch,seed=139)
        ability=s.ability.at[:,side,0].set(1).at[:,1-side,0].set(3)
        stats=j.full_like(s.stats,101).at[...,0].set(1000).at[:,side,:,1].set(103).at[:,side,:,3].set(103).at[:,side,:,5].set(300)
        return s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.max_hp,1000),stats=stats,
            species=j.broadcast_to(j.arange(1,7),(batch,2,6)),level=j.full_like(s.level,73),
            types=s.types.at[...,0].set(7),ability=ability,base_ability=ability,
            moves=j.full_like(s.moves,self.id('splash')).at[:,side,0,0].set(self.id(move)),
            pp=j.full_like(s.pp,10),done=j.zeros_like(s.done),
            hidden_power_type=s.hidden_power_type.at[:,side,0].set(15 if move=='hiddenpowerice' else 5),
            hidden_power_power=j.full_like(s.hidden_power_power,70))

    def play(self,s,side=0,*,table=None,jit=False):
        j=self.j;c=self.tr.sample_chance_inputs(s)
        d=c.draws.at[:,side,6].set(j.arange(s.hp.shape[0])%100).at[:,:,4].set(15).at[:,:,5].set(15)
        c=c._replace(draws=d)
        fn=self.jax.jit(self.tr.step_with_chance) if jit else self.tr.step_with_chance
        return fn(s,j.zeros(s.active.shape,j.int32),self.table if table is None else table,c)

    def test_hustle_damage_rounds_before_choice_band_and_is_physical_only(self):
        m=self.m
        kwargs=dict(level=73,power=50,attack=103,defense=101,attack_num=3,attack_den=2)
        got=m.gen3_damage(**kwargs,hustle=True)
        expected=m.gen3_damage(**dict(kwargs,attack=154))
        self.assertEqual(int(got),int(expected))
        self.assertEqual(int(m.modify(m.modify(103,3,2),3,2)),231)
        self.assertEqual(int(m.modify(103,9,4)),232)
        self.assertEqual(int(m.gen3_damage(**kwargs,hustle=True,physical=False)),
            int(m.gen3_damage(**kwargs,physical=False)))

    def test_resolved_hidden_power_status_and_always_hit_in_both_seats(self):
        for side in (0,1):
            for move,hits in [('hiddenpowerground',80),('hiddenpowerice',100),('aerialace',100),('toxic',68)]:
                with self.subTest(side=side,move=move):
                    s=self.live(side,move);out=self.play(s,side).state
                    hit=out.status[:,1-side,0]!=0 if move=='toxic' else out.hp[:,1-side,0]<1000
                    self.assertEqual(hit.tolist(),[True]*hits+[False]*(100-hits))

    def test_current_ability_overrides_base_and_confusion_uses_direct_stat_event(self):
        s=self.live(batch=100)
        out=self.play(s._replace(ability=s.ability.at[:,0,0].set(2))).state
        self.assertTrue(bool(self.j.all(out.hp[:,1,0]<1000)))
        s=self.live(batch=100)._replace(confusion_counter=self.live(batch=100).confusion_counter.at[:,0,0].set(3),
            item=s.item.at[:,0,0].set(1))
        plain=self.play(s._replace(ability=s.ability.at[:,0,0].set(2))).state
        boosted=self.play(s).state
        self.assertTrue(bool(self.j.any(boosted.hp[:,0,0]<plain.hp[:,0,0])))
        self.assertTrue(bool(self.j.all(boosted.hp[:,0,0]<=plain.hp[:,0,0])))

    def test_integer_fractional_and_binary64_stage_boundaries(self):
        j=self.j;m=self.m
        for base,a,e,want in [(100,0,0,80),(100,-4,0,43),(50,1,1,50),(85,1,2,54),(100,1,2,80)]:
            with self.subTest(base=base,a=a,e=e):
                hits=m.accuracy_hit(base,a,e,j.arange(100),hustle=True)
                self.assertEqual(int(j.sum(hits)),want)
                self.assertEqual(m.hustle_accuracy_count(base,a,e),want)

    def test_metadata_and_host_helper_reject_outside_finite_accuracy_domain(self):
        from copy import deepcopy
        for value in (-1,101,85.5,100.0,False):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'ACCURACY_DOMAIN'):
                self.m.hustle_accuracy_count(value,0,0)
            metadata=deepcopy(self.metadata);metadata['moves'][1]['accuracy']=value
            with self.assertRaisesRegex(ValueError,'ACCURACY_DOMAIN'):
                self.tr.tables_from_metadata(metadata)

    def test_jit_prefix_invalid_and_terminal_identity(self):
        j=self.j;s=self.live(batch=4)._replace(done=j.array([False,False,True,False]))
        a=j.array([[0,0],[0,0],[0,0],[9,0]])
        c=self.tr.sample_chance_inputs(s)
        eager=self.tr.step_with_chance(s,a,self.table,c)
        fast=self.jax.jit(self.tr.step_with_chance)(s,a,self.table,c);self.jax.block_until_ready(fast)
        for x,y in zip(self.jax.tree.leaves(eager),self.jax.tree.leaves(fast)):
            self.assertTrue(bool(j.array_equal(x,y)))
        for old,new in zip(s,eager.state):self.assertTrue(bool(j.array_equal(old[2:],new[2:])))
        small=self.tr.step_with_chance(self.jax.tree.map(lambda x:x[:2],s),a[:2],self.table,
            self.jax.tree.map(lambda x:x[:2],c))
        for x,y in zip(self.jax.tree.leaves(eager),self.jax.tree.leaves(small)):
            self.assertTrue(bool(j.array_equal(x[:2],y)))

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown')
    def test_pinned_showdown_all_accuracy_cells_damage_and_confusion(self):
        r=subprocess.run([NODE,str(Path(__file__).with_name('gpu_hustle_oracle.cjs')),SHOWDOWN],
            text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr);d=json.loads(r.stdout)
        self.assertEqual(d['metadata'],self.metadata)
        j=self.j;m=self.m
        base=j.repeat(j.arange(101),169);a=j.tile(j.repeat(j.arange(-6,7),13),101)
        e=j.tile(j.arange(-6,7),101*13)
        # Check both sides of every projected integer-roll boundary, on device.
        count=j.array(d['counts'])
        self.assertTrue(bool(j.all(m.accuracy_hit(base,a,e,count-1,hustle=True)|(count==0))))
        self.assertTrue(bool(j.all(~m.accuracy_hit(base,a,e,count,hustle=True))))
        for row in d['damage']:
            got=m.gen3_damage(level=73,power=50,attack=103,defense=101,
                attack_stage=row['attack_stage'],critical=row['critical'],physical=row['physical'],
                hustle=row['ability']=='Hustle',attack_num=3 if row['item'] and row['physical'] else 1,
                attack_den=2 if row['item'] and row['physical'] else 1)
            self.assertEqual(int(got),row['value'],row)
        for row in d['confusion']:
            got=m.gen3_damage(level=73,power=40,attack=103,defense=101,typeless=True,
                hustle=True,burned=row['burned'],attack_num=3 if row['item'] else 1,
                attack_den=2 if row['item'] else 1)
            self.assertEqual(int(got),row['value'],row)
        self.assertEqual([row['threshold'] for row in d['accuracy']],
            [80,100,68,None,100/(7/3),54,79.99999999999999])


if __name__=='__main__':unittest.main()
