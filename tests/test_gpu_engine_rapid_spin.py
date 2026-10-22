"""Gen3 represented Rapid Spin cleanup, not full move/game qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated spike JAX')
class RapidSpinTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,sustain,transition
        from gpu_engine.state import empty_state,LEECH_SEED,ROOTED
        self.jax,self.j,self.e,self.su,self.t,self.empty=jax,j,effects,sustain,transition,empty_state
        self.seed,self.root=LEECH_SEED,ROOTED
        names=['_none','rapidspin','splash','protect','endure','struggle','spikes']
        types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
            'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        meta={'vocabulary':{'moves':names,'types':types},'moves':[None]+[
            {'flags':{'protect':name=='rapidspin'}} for name in names[1:]]}
        n=len(names);num=j.ones((18,18),j.int32).at[1,8].set(0)
        self.table=transition.MoveTable(j.array([0,20,0,0,0,50,0]),j.ones(n,j.int32),j.ones(n,bool),
            j.full(n,100),j.ones(n,bool),j.array([0,0,0,3,3,0,0]),num,j.ones((18,18),j.int32),
            j.array(5),effects=effects.from_metadata(meta),sustain=sustain.from_metadata(meta))

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=211)
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),
            level=j.full_like(s.level,100),hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            moves=j.full_like(s.moves,2).at[:,0,0,0].set(1),pp=j.full_like(s.pp,64),
            volatile=s.volatile.at[:,0,0].set(self.seed),spikes=j.full_like(s.spikes,2),
            done=j.zeros_like(s.done))

    def play(self,s):return self.t.step(s,self.j.zeros(s.active.shape,self.j.int32),self.table)

    def test_success_clears_before_seed_residual_without_speed_boost(self):
        a=self.play(self.live()).state
        self.assertEqual((int(a.volatile[0,0,0]),int(a.spikes[0,0]),int(a.spikes[0,1])),(0,0,2))
        self.assertEqual((int(a.hp[0,0,0]),int(a.pp[0,0,0,0]),int(a.boosts[0,0,0,4])),(1000,63,0))
        self.assertLess(int(a.hp[0,1,0]),1000)

    def test_miss_protect_and_ghost_do_not_clear(self):
        j=self.j;s=self.live()
        for label,a,table in (("protect",s._replace(moves=s.moves.at[0,1,0,0].set(3)),self.table),
                ("ghost",s._replace(types=s.types.at[0,1,0,0].set(8)),self.table),
                ("miss",s,self.table._replace(accuracy=self.table.accuracy.at[1].set(0),
                    always_hit=self.table.always_hit.at[1].set(False)))):
            with self.subTest(label=label):
                r=self.t.step(a,j.zeros((1,2),j.int32),table).state
                self.assertEqual((int(r.volatile[0,0,0])&self.seed,int(r.spikes[0,0])),(self.seed,2))

    def test_unbroken_and_broken_substitute_clear_without_spill(self):
        for capacity in (19,1):
            s=self.live();s=s._replace(substitute_hp=s.substitute_hp.at[0,1,0].set(capacity))
            a=self.play(s).state
            self.assertEqual((int(a.volatile[0,0,0]),int(a.spikes[0,0]),int(a.hp[0,1,0])),(0,0,1000))
            self.assertLess(int(a.substitute_hp[0,1,0]),capacity)

    def test_zero_actual_endure_damage_and_target_ko_still_clear(self):
        for foe in (2,4):
            s=self.live();s=s._replace(hp=s.hp.at[0,1,0].set(1),moves=s.moves.at[0,1,0,0].set(foe))
            a=self.play(s).state
            self.assertEqual((int(a.volatile[0,0,0]),int(a.spikes[0,0]),int(a.hp[0,1,0])),(0,0,int(foe==4)))

    def test_cleanup_preserves_linked_external_rooted_and_other_side_state(self):
        j=self.j;s=self.live(3);s=s._replace(volatile=s.volatile.at[:,0,0].set(self.seed|self.root),
            trap_source=s.trap_source.at[:,0,0].set(0),trapped=j.ones_like(s.trapped))
        a=self.su.spin_cleanup(s,j.array([0,1,0]),eligible=j.array([True,True,False]))
        self.assertEqual(a.volatile[:,0,0].tolist(),[self.root,self.seed|self.root,self.seed|self.root])
        self.assertTrue(bool(j.array_equal(a.trap_source,s.trap_source)))
        self.assertTrue(bool(j.array_equal(a.trapped,s.trapped)))
        self.assertEqual(a.spikes.tolist(),[[0,2],[2,0],[2,2]])

    def test_jit_batch_prefix_invalid_and_terminal_identity(self):
        j=self.j;s=self.live(2);actions=j.zeros((2,2),j.int32)
        a=self.play(s);b=self.jax.jit(self.t.step)(s,actions,self.table)
        one=self.t.step(self.jax.tree.map(lambda x:x[:1],s),actions[:1],self.table)
        for x,y,z in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b),self.jax.tree.leaves(one)):
            self.assertTrue(bool(j.array_equal(x,y)));self.assertTrue(bool(j.array_equal(x[:1],z)))
        s=s._replace(done=s.done.at[1].set(True));actions=actions.at[0,0].set(99)
        r=self.t.step(s,actions,self.table)
        for x,y in zip(self.jax.tree.leaves(s),self.jax.tree.leaves(r.state)):
            self.assertTrue(bool(j.array_equal(x,y)))

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown')
    def test_pinned_gen3_hit_sub_endure_and_nonremovable_trap(self):
        cases=json.loads(subprocess.check_output([NODE,str(Path(__file__).with_name('gpu_rapid_spin_oracle.cjs')),SHOWDOWN],text=True))
        for name,c in cases.items():
            clears=name in {'normal','sub','ko','endure'}
            self.assertEqual((c['seed'],c['spikes'],c['partial']),(not clears,0 if clears else 2,not clears),name)
            self.assertEqual((c['trapped'],c['speed'],c['pp']),(True,0,63),name)
        self.assertGreater(cases['sub']['sub'],0)
        self.assertEqual((cases['ko']['targetHP'],cases['endure']['targetHP']),(0,1))
