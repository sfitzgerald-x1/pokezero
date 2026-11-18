"""Gen3 infatuation law and lifecycle; effect probes supply the Showdown oracle."""
import unittest

import numpy as np


class InfatuationTests(unittest.TestCase):
    def setUp(self):
        import jax.numpy as j
        from gpu_engine import infatuation, state, secondary
        from tests.test_gpu_engine_fidelity_tier_d import metadata
        self.j, self.i = j, infatuation
        meta=metadata()
        meta['vocabulary']['abilities']=['_none','cutecharm','oblivious']
        meta['vocabulary']['items']=['_none','mentalherb','lumberry']
        self.table=infatuation.from_metadata(meta)
        self.secondary=secondary.from_metadata(meta)
        s=state.empty_state(60)
        self.s=s._replace(hp=j.full_like(s.hp,100),max_hp=j.full_like(s.max_hp,100),
            gender=j.full_like(s.gender,1).at[:,1,:].set(2),
            ability=s.ability.at[:,1,0].set(1))

    def test_contact_law_and_admission(self):
        from gpu_engine import secondary
        from gpu_engine.state import ATTRACT
        j=self.j
        def hit(s,move=3,eligible=True):
            return secondary.apply_contact(s,j.zeros(60,j.int32),j.full((60,),move),j.arange(60),self.secondary,
                eligible=j.full((60,),eligible),sleep_roll=j.zeros(60,j.int32),infatuation_table=self.table,
                _reveal_masks=True)
        out,reveal,_=hit(self.s)
        self.assertEqual(np.flatnonzero(np.asarray(out.volatile[:,0,0]&ATTRACT)).tolist(),list(range(20)))
        np.testing.assert_array_equal(reveal,np.arange(60)<20)
        for s in (self.s._replace(gender=self.s.gender.at[:,0,0].set(2)),
                  self.s._replace(gender=self.s.gender.at[:,0,0].set(0)),
                  self.s._replace(gender=self.s.gender.at[:,0,0].set(-1)),
                  self.s._replace(ability=self.s.ability.at[:,0,0].set(2)),
                  self.s._replace(hp=self.s.hp.at[:,0,0].set(0))):
            self.assertFalse(np.any(np.asarray(hit(s)[1])))
        self.assertFalse(np.any(np.asarray(hit(self.s,move=4)[1])))
        self.assertFalse(np.any(np.asarray(hit(self.s,eligible=False)[1])))
        # Holder KO still reveals; an already infatuated recipient does not restart.
        np.testing.assert_array_equal(hit(self.s._replace(hp=self.s.hp.at[:,1,0].set(0)))[1],reveal)
        self.assertFalse(np.any(np.asarray(hit(out)[1])))

    def test_before_move_priority_and_independent_paralysis(self):
        from gpu_engine import mechanics as m
        for confuse,paralysis,attract,can_act,self_hit,cant in (
                (99,99,0,False,True,False),(0,0,0,False,False,True),
                (0,0,1,False,False,False),(0,99,1,True,False,False)):
            r=m.before_move(m.PAR,0,3,confusion_roll=confuse,paralysis_roll=paralysis,
                infatuated=True,attract_roll=attract)
            self.assertEqual((bool(r.can_act),bool(r.confusion_self_hit),bool(r.infatuation_cant)),
                (can_act,self_hit,cant))
        asleep=m.before_move(m.SLP,3,infatuated=True,attract_roll=0)
        self.assertFalse(bool(asleep.infatuation_cant))
        talk=m.before_move(m.SLP,3,sleep_usable=True,infatuated=True,attract_roll=0)
        self.assertTrue(bool(talk.infatuation_cant))

    def test_source_faint_request_then_replacement_and_cures(self):
        from gpu_engine.state import ATTRACT
        j=self.j
        s,applied=self.i.apply(self.s,0,self.table,eligible=True)
        self.assertTrue(np.all(applied))
        faint=s._replace(hp=s.hp.at[:,1,0].set(0))
        kept=self.i.clear_fainted(faint,eligible=j.ones(60,bool))
        self.assertTrue(np.all(np.asarray(kept.volatile[:,0,0]&ATTRACT)))
        left=self.i.source_departed(kept,1,eligible=j.ones(60,bool))
        self.assertFalse(np.any(np.asarray(left.volatile[:,0,0]&ATTRACT)))
        herb=s._replace(item=s.item.at[:,0,0].set(1))
        cured=self.i.update(herb,self.table,eligible=True)
        self.assertFalse(np.any(np.asarray(cured.item[:,0,0])))
        oblivious=herb._replace(ability=herb.ability.at[:,0,0].set(2))
        self.assertTrue(np.all(np.asarray(self.i.update(oblivious,self.table,eligible=True).item[:,0,0])==1))
        dead=herb._replace(hp=herb.hp.at[:,0,0].set(0))
        self.assertTrue(np.all(np.asarray(self.i.update(dead,self.table,eligible=True).item[:,0,0])==1))
        lum=s._replace(item=s.item.at[:,0,0].set(2))
        self.assertTrue(np.all(np.asarray(self.i.update(lum,self.table,eligible=True).volatile[:,0,0]&ATTRACT)))

    def test_new_column_preserves_old_stream(self):
        import jax
        from gpu_engine import transition
        j=self.j
        keys=jax.random.split(jax.random.PRNGKey(831),60)
        old=jax.vmap(lambda key:jax.random.randint(jax.random.fold_in(key,1732050),(2,4),0,
            j.array([16,16,4,2],j.int32)))(keys)
        new=transition.effect_chance(keys)
        self.assertEqual(new.shape,(60,2,5))
        np.testing.assert_array_equal(new[...,:4],old)
        self.assertEqual(set(np.asarray(new[...,4]).flat),{0,1})

    def test_legacy_four_column_inputs_use_the_new_independent_sample(self):
        import jax
        from gpu_engine import transition
        from gpu_engine.state import ATTRACT
        from tests.test_gpu_engine_transition import TransitionTests
        fixture=TransitionTests();fixture.setUp()
        s=fixture.live();s=s._replace(moves=self.j.full_like(s.moves,2),
            volatile=s.volatile.at[:,0,0].set(ATTRACT))
        full=transition.sample_chance_inputs(s)
        legacy=full._replace(effect=full.effect[...,:4])
        actions=self.j.zeros((1,2),self.j.int32)
        with jax.disable_jit():
            a=transition.step_with_chance(s,actions,fixture.table,full)
            b=transition.step_with_chance(s,actions,fixture.table,legacy)
        for name in s._fields:
            np.testing.assert_array_equal(getattr(a.state,name),getattr(b.state,name),err_msg=name)

    def test_saved_fifth_column_is_retained_and_legacy_blocks_refuse(self):
        from gpu_engine.conditioned_future import _saved_block
        a=((1,2,3,1,0),(4,5,2,0,1))
        b=((1,2,3,1,1),(4,5,2,0,0))
        self.assertEqual(_saved_block({'effect_draws':a}),a)
        self.assertNotEqual(_saved_block({'effect_draws':a}),_saved_block({'effect_draws':b}))
        with self.assertRaisesRegex(ValueError,'INVALID_SAVED_EFFECT_BLOCK'):
            _saved_block({'effect_draws':tuple(row[:4] for row in a)})


if __name__=='__main__':unittest.main()
