"""Authored original-ability/departure checks; no scientific collection."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class BaseAbilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        import numpy as np
        from gpu_engine import entry, mechanics, phazing, transition
        from gpu_engine.state import empty_state, ROOTED
        cls.jax,cls.j,cls.np=jax,j,np
        cls.entry,cls.m,cls.ph,cls.t=entry,mechanics,phazing,transition
        cls.empty=staticmethod(empty_state);cls.rooted=ROOTED
        cls.meta={'vocabulary':{'moves':['_none','splash','roar','struggle'],
            'abilities':['_none','trace','naturalcure','static','suctioncups','soundproof'],
            'types':['_none','normal','flying','ground']}}
        cls.et=entry.from_metadata(cls.meta);cls.pt=phazing.from_metadata(cls.meta)
        cls.table=transition.MoveTable(j.array([0,0,0,50]),j.array([0,1,1,0]),
            j.zeros(4,bool),j.full(4,100),j.ones(4,bool),j.zeros(4,j.int32),
            j.ones((4,4),j.int32),j.ones((4,4),j.int32),j.array(3),entry=cls.et)

    def live(self,batch=1):
        s=self.empty(batch,seed=47);j=self.j
        return s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            stats=j.full_like(s.stats,200).at[...,0].set(1000),level=j.full_like(s.level,100),
            moves=j.ones_like(s.moves),pp=j.full_like(s.pp,10),types=s.types.at[...,0].set(1),
            done=j.zeros_like(s.done))

    def test_pool_initializes_all_originals_before_any_entry(self):
        s=self.live();names=('species','level','types','stats','moves','pp','item','ability',
            'hidden_power_type','hidden_power_power')
        pool={n:self.np.asarray(getattr(s,n)[0]) for n in names}
        pool['ability']=self.np.array([[1,2,3,4,5,0],[5,4,3,2,1,0]],self.np.int32)
        state=self.t.state_from_pool(pool,self.np.array([[1,0]]))
        self.assertEqual(state.ability.tolist(),[pool['ability'][[1,0]].tolist()])
        self.assertTrue(bool(self.j.array_equal(state.base_ability,state.ability)))
        changed=state._replace(ability=state.ability.at[0,0,0].set(3))
        self.assertEqual(int(changed.base_ability[0,0,0]),5)
        self.assertEqual(state.entry_pending.tolist(),[[True,True]])

    def test_current_natural_cure_runs_before_original_restoration_in_mixed_batches(self):
        j=self.j
        for side in (0,1):
            s=self.live(4)._replace(active=j.full((4,2),3,j.int32))
            s=s._replace(ability=s.ability.at[:,side,3].set(j.array([2,3,2,3])),
                base_ability=s.base_ability.at[:,side,3].set(j.array([1,2,1,2])),
                status=s.status.at[:,side,3].set(self.m.PSN))
            gate=j.array([True,True,False,False])
            for run in (self.entry.depart,self.jax.jit(self.entry.depart)):
                result=run(s,side,self.et,eligible=gate)
                self.assertEqual(result.ability[:,side,3].tolist(),[1,2,2,3])
                self.assertEqual(result.status[:,side,3].tolist(),[self.m.NONE,self.m.PSN,self.m.PSN,self.m.PSN])
                self.assertTrue(bool(j.array_equal(result.base_ability,s.base_ability)))
                self.assertTrue(bool(j.array_equal(result.ability[:,1-side],s.ability[:,1-side])))
                self.assertTrue(bool(j.array_equal(result.ability[:,side,:3],s.ability[:,side,:3])))
            self.assertEqual(s.ability[:,side,3].tolist(),[2,3,2,3])
            result=self.entry.depart(s,side,None,eligible=gate)
            self.assertEqual(result.ability[:,side,3].tolist(),[1,2,2,3])
            self.assertEqual(result.status[:,side,3].tolist(),[self.m.PSN]*4)

    def test_ordinary_switch_restores_both_seats_but_invalid_and_terminal_rows_freeze(self):
        j=self.j
        for side in (0,1):
            s=self.live(3)._replace(done=j.array([False,False,True]))
            s=s._replace(ability=s.ability.at[:,side,0].set(2),
                base_ability=s.base_ability.at[:,side,0].set(1),status=s.status.at[:,side,0].set(self.m.PSN))
            actions=j.zeros((3,2),j.int32).at[:,side].set(4).at[1,side].set(99)
            r=self.jax.jit(self.t.step)(s,actions,self.table)
            self.assertEqual(r.state.ability[:,side,0].tolist(),[1,2,2])
            self.assertEqual(r.state.status[:,side,0].tolist(),[0,self.m.PSN,self.m.PSN])
            self.assertTrue(bool(j.array_equal(r.state.base_ability,s.base_ability)))
            for before,after in zip(s,r.state):
                self.assertTrue(bool(j.array_equal(before[1:],after[1:])))

    def test_phazing_restores_only_actual_departures_and_preserves_blocked_rows(self):
        j=self.j
        for side in (0,1):
            s=self.live(6)
            s=s._replace(ability=s.ability.at[:,side,0].set(j.array([2,4,5,2,2,2])),
                base_ability=s.base_ability.at[:,side,0].set(1),status=s.status.at[:,side,0].set(self.m.PSN),
                volatile=s.volatile.at[3,side,0].set(self.rooted),hp=s.hp.at[5,side,1:].set(0))
            args=(s,j.full(6,side),j.full(6,2),j.zeros(6,j.int32),self.pt,self.et)
            for run in (self.ph.drag,self.jax.jit(self.ph.drag)):
                r,changed,ko=run(*args,eligible=j.array([True,True,True,True,False,True]))
                self.assertEqual(changed.tolist(),[True,False,False,False,False,False])
                self.assertFalse(bool(j.any(ko)))
                self.assertEqual((int(r.ability[0,side,0]),int(r.status[0,side,0])),(1,0))
                self.assertTrue(bool(j.array_equal(r.base_ability,s.base_ability)))
                for before,after in zip(s,r):
                    self.assertTrue(bool(j.array_equal(before[1:],after[1:])))


if __name__=='__main__':unittest.main()
