"""Placeholder shape/perspective checks, explicitly not real-encoder parity."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires isolated spike JAX")
class ObservationTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import combat,observation,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.obs,self.tr,self.empty=jax,j,observation,transition,empty_state
        types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        traits=combat.from_metadata({"vocabulary":{"abilities":["_none","shadowtag"],"types":types}})
        self.table=transition.MoveTable(j.zeros(3,j.int32),j.zeros(3,j.int32),j.zeros(3,bool),
            j.full(3,100),j.ones(3,bool),j.zeros(3,j.int32),j.ones((18,18),j.int32),
            j.ones((18,18),j.int32),j.array(2),combat=traits)

    def live(self,batch=2):
        j=self.j;s=self.empty(batch,seed=173)
        species=j.broadcast_to(j.arange(1,13).reshape(1,2,6),(batch,2,6))
        return s._replace(species=species,stats=j.full_like(s.stats,200),
            hp=j.full_like(s.hp,100),max_hp=j.full_like(s.hp,200),level=j.full_like(s.level,80),
            moves=j.ones_like(s.moves),pp=j.full_like(s.pp,16),done=j.zeros_like(s.done))

    def encode(self,s):return self.obs.encode(s,self.table)

    def test_exact_champion_shape_dtypes_and_token_sections(self):
        j=self.j;o=self.encode(self.live(3))
        self.assertTrue(self.obs.IS_PLACEHOLDER)
        self.assertEqual((o.numeric_features.shape,o.categorical_ids.shape),((3,2,23,132),(3,2,23,41)))
        self.assertEqual(o.numeric_features.dtype,j.float32);self.assertEqual(o.categorical_ids.dtype,j.int32)
        self.assertEqual(o.token_type_ids[0,0].tolist(),[0]+[1]*6+[2]*6+[3]*9+[5])
        self.assertEqual(o.attention_mask.dtype,j.bool_);self.assertEqual(o.legal_action_mask.shape,(3,2,9))

    def test_own_and_opposing_party_layout_is_seat_relative(self):
        o=self.encode(self.live())
        self.assertEqual(o.categorical_ids[0,0,1:13,0].tolist(),list(range(1,13)))
        self.assertEqual(o.categorical_ids[0,1,1:13,0].tolist(),list(range(7,13))+list(range(1,7)))
        self.assertEqual(float(o.numeric_features[0,0,1,0]),.5)
        self.assertAlmostEqual(float(o.numeric_features[0,0,1,1]),.8,places=6)

    def test_actions_follow_current_active_moves_pp_and_ordered_bench(self):
        j=self.j;s=self.live();s=s._replace(active=s.active.at[0,0].set(3),
            moves=s.moves.at[0,0,3].set(j.array([1,2,1,2])),pp=s.pp.at[0,0,3].set(j.array([0,8,16,32])))
        o=self.encode(s)
        self.assertEqual(o.categorical_ids[0,0,13:22,0].tolist(),[1,2,1,2,1,2,3,5,6])
        self.assertEqual(o.numeric_features[0,0,13:17,3].tolist(),[0,.125,.25,.5])
        self.assertEqual(o.legal_action_mask[0,0,:4].tolist(),[False,True,True,True])

    def test_ability_traps_forced_and_terminal_masks_are_not_recomputed_from_pp(self):
        j=self.j;s=self.live(3);s=s._replace(ability=s.ability.at[:,1,0].set(1),
            forced_switch=s.forced_switch.at[1,0].set(True),done=s.done.at[2].set(True))
        o=self.encode(s)
        self.assertFalse(bool(j.any(o.legal_action_mask[0,0,4:])))
        self.assertTrue(bool(j.all(o.legal_action_mask[1,0,4:])))
        self.assertFalse(bool(j.any(o.legal_action_mask[1,0,:4])))
        self.assertFalse(bool(j.any(o.legal_action_mask[2])))
        self.assertTrue(bool(j.array_equal(o.numeric_features[:,:,13:22,0].astype(bool),o.legal_action_mask)))

    def test_absent_party_padding_and_zero_max_hp_remain_finite(self):
        j=self.j;s=self.live();s=s._replace(species=s.species.at[0,1,5].set(0),
            max_hp=s.max_hp.at[0,1,5].set(0),hp=s.hp.at[0,1,5].set(0))
        o=self.encode(s)
        self.assertFalse(bool(o.attention_mask[0,0,12]));self.assertFalse(bool(o.attention_mask[0,1,6]))
        self.assertTrue(bool(j.all(j.isfinite(o.numeric_features))))

    def test_eager_jit_batch_prefix_and_state_identity(self):
        j=self.j;s=self.live(4);before=self.jax.tree.map(lambda x:x.copy(),s)
        a=self.jax.jit(self.obs.encode)(s,self.table);a.numeric_features.block_until_ready()
        b=self.encode(s)
        small=self.jax.jit(self.obs.encode)(self.jax.tree.map(lambda x:x[:2],s),self.table)
        for x,y,z in zip(a,b,small):
            self.assertTrue(bool(j.array_equal(x,y)))
            self.assertTrue(bool(j.array_equal(x[:2],z)))
        for x,y in zip(s,before):self.assertTrue(bool(j.array_equal(x,y)))


if __name__=="__main__":unittest.main()
