"""Rust Wish-state rules adapted to authored JAX contexts; no coverage credit."""
import importlib.util
import unittest


def fixture_components():
    import jax
    import jax.numpy as j
    from gpu_engine import baton,effects,entry,phazing,sustain,transition
    names=['_none','splash','wish','struggle','batonpass','roar','bodyslam','protect']
    types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
           'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
    meta={'vocabulary':{'moves':names,'types':types,'abilities':['_none'],
        'items':['_none'],'species':['_none','ditto','pikachu','blissey','snorlax','skarmory','slaking']},
        'moves':[None]+[{'id':name,'source_type':'Normal',
            'category':'Physical' if name in {'bodyslam','struggle'} else 'Status',
            'flags':{'protect':int(name in {'bodyslam','roar'})}} for name in names[1:]]}
    n=len(names)
    table=transition.MoveTable(j.array([0,0,0,50,0,0,85,0]),j.ones(n,j.int32),
        j.array([False,False,False,True,False,False,True,False]),j.full(n,100),
        j.ones(n,bool),j.zeros(n,j.int32).at[5].set(-6).at[7].set(3),
        j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(3),
        effects=effects.from_metadata(meta),entry=entry.from_metadata(meta),
        baton=baton.from_metadata(meta),phazing=phazing.from_metadata(meta),
        sustain=sustain.from_metadata(meta))
    return jax,j,meta,table


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class WishHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpu_engine import transition as t
        from gpu_engine.state import empty_state
        cls.jax,cls.j,cls.meta,cls.table=fixture_components()
        cls.t=t;cls.empty=staticmethod(empty_state)
        cls.step=staticmethod(cls.jax.jit(t.step_with_chance))

    def live(self):
        j=self.j;s=self.empty(1,seed=818)
        return s._replace(hp=j.full_like(s.hp,10),max_hp=j.full_like(s.max_hp,101),
            stats=j.full_like(s.stats,200).at[...,0].set(101).at[:,0,:,5].set(301),
            level=j.full_like(s.level,100),types=s.types.at[...,0].set(1),
            species=j.broadcast_to(j.arange(1,7),(1,2,6)),
            moves=j.ones_like(s.moves),pp=j.full_like(s.pp,10),done=j.array([False]))

    def caster(self,side):
        s=self.live()
        return s._replace(max_hp=s.max_hp.at[0,side,0].set(301),
            stats=s.stats.at[0,side,0,0].set(301),moves=s.moves.at[0,side,:,0].set(2))

    def actions(self,side,action):
        return self.j.ones((1,2),self.j.int32).at[0,side].set(action)

    def chance(self):
        j=self.j
        draws=j.zeros((1,2,12),j.int32).at[:,:,1:4].set(99).at[:,:,4:7].set(15)
        return self.t.ChanceInputs(draws,j.zeros((1,2),j.int32),j.full((1,2),99),
            j.zeros((1,2),j.int32),j.zeros((1,2),j.int32),j.zeros(1,j.int32))

    def play(self,s,actions):
        return self.step(s,actions,self.table,self.chance()).state

    def assert_wish(self,s,side,counter,amount):
        self.assertEqual((int(s.wish_counter[0,side]),int(s.wish_amount[0,side])),(counter,amount))

    def test_creation_switch_expiry_and_new_caster_recast_both_seats(self):
        for side in range(2):
            s=self.caster(side)
            cast=self.play(s,self.actions(side,0));self.assert_wish(cast,side,1,150)
            expired=self.play(cast,self.actions(side,4))
            self.assert_wish(expired,side,0,150)
            self.assertEqual(int(expired.hp[0,side,1]),60)  # Recipient101, not caster301.
            recast=self.play(expired,self.actions(side,0));self.assert_wish(recast,side,1,50)
            self.assertEqual(int(recast.pp[0,side,1,0]),9)

    def test_pending_recast_spends_pp_without_replacing_stored_history(self):
        for side in range(2):
            s=self.caster(side);s=s._replace(wish_counter=s.wish_counter.at[0,side].set(2),
                wish_amount=s.wish_amount.at[0,side].set(77))
            r=self.play(s,self.actions(side,0));self.assert_wish(r,side,1,77)
            self.assertEqual(int(r.pp[0,side,0,0]),9)
            r=self.play(r,self.actions(side,0));self.assert_wish(r,side,0,77)
            self.assertEqual(int(r.hp[0,side,0]),160)  # Healing ignores stored77.

    def test_full_hp_and_zero_amount_expiry_preserve_exact_wire_history(self):
        for side in range(2):
            for amount in (0,777):
                s=self.caster(side);s=s._replace(wish_counter=s.wish_counter.at[0,side].set(1),
                    wish_amount=s.wish_amount.at[0,side].set(amount),
                    hp=s.hp.at[0,side,0].set(301))
                r=self.play(s,self.actions(side,1));self.assert_wish(r,side,0,amount)
                self.assertEqual(int(r.hp[0,side,0]),301)

    def test_blocked_move_does_not_replace_expired_history(self):
        from gpu_engine import mechanics as m
        for side in range(2):
            s=self.caster(side);s=s._replace(status=s.status.at[0,side,0].set(m.SLP),
                status_counter=s.status_counter.at[0,side,0].set(3),
                wish_amount=s.wish_amount.at[0,side].set(77))
            r=self.play(s,self.actions(side,0));self.assert_wish(r,side,0,77)
            self.assertEqual(int(r.pp[0,side,0,0]),10)

    def test_expired_history_survives_baton_and_phazing(self):
        for side in range(2):
            s=self.caster(side);s=s._replace(wish_amount=s.wish_amount.at[0,side].set(150),
                moves=s.moves.at[0,side,0,0].set(4),stats=s.stats.at[0,side,0,5].set(500))
            paused=self.play(s,self.actions(side,0));self.assert_wish(paused,side,0,150)
            resumed=self.play(paused,self.actions(side,4));self.assert_wish(resumed,side,0,150)
            s=self.caster(side);s=s._replace(wish_amount=s.wish_amount.at[0,side].set(150),
                moves=s.moves.at[0,1-side,0,0].set(5))
            r=self.play(s,self.actions(1-side,0));self.assert_wish(r,side,0,150)

    def test_move_ko_and_repeated_hazard_replacements_defer_wish(self):
        for side in range(2):
            s=self.caster(side);s=s._replace(wish_counter=s.wish_counter.at[0,side].set(1),
                wish_amount=s.wish_amount.at[0,side].set(150),
                hp=s.hp.at[0,side,0].set(1).at[0,side,1:3].set(25).at[0,side,3].set(80),
                moves=s.moves.at[0,1-side,0,0].set(6),
                stats=s.stats.at[0,1-side,0,5].set(500),spikes=s.spikes.at[0,side].set(3))
            r=self.play(s,self.actions(1-side,0));self.assert_wish(r,side,1,150)
            self.assertTrue(bool(r.pending_residual[0]))
            for action in (4,5):
                r=self.play(r,self.actions(side,action))
                self.assert_wish(r,side,1,150);self.assertTrue(bool(r.pending_residual[0]))
            r=self.play(r,self.actions(side,6));self.assert_wish(r,side,0,150)
            self.assertFalse(bool(r.pending_residual[0]));self.assertEqual(int(r.hp[0,side,3]),101)

    def test_terminal_truncation_and_invalid_requests_preserve_history(self):
        for side in range(2):
            s=self.caster(side);s=s._replace(wish_counter=s.wish_counter.at[0,side].set(1),
                wish_amount=s.wish_amount.at[0,side].set(150),
                hp=s.hp.at[0,side,:].set(0).at[0,side,0].set(1),
                moves=s.moves.at[0,1-side,0,0].set(6),stats=s.stats.at[0,1-side,0,5].set(500))
            r=self.play(s,self.actions(1-side,0));self.assertTrue(bool(r.done[0]))
            self.assert_wish(r,side,1,150)
            terminal=self.play(r,self.actions(side,0));self.assert_wish(terminal,side,1,150)
            invalid=self.play(s,self.actions(side,99));self.assert_wish(invalid,side,1,150)


if __name__=='__main__':unittest.main()
