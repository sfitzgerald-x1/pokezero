"""Authored combat/trapping fixtures; not game or move coverage qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE = os.environ.get("NODE") or shutil.which("node")
SHOWDOWN = os.environ.get("POKEZERO_SHOWDOWN_ROOT")


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires isolated spike JAX")
class CombatTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import combat,items,mechanics,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.c,self.m,self.tr,self.empty = jax,j,combat,mechanics,transition,empty_state
        self.abilities = ["_none","guts","thickfat","arenatrap","magnetpull","shadowtag","levitate"]
        self.types = ["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        self.meta = {"vocabulary":{"abilities":self.abilities,"types":self.types,
            "items":["_none","choiceband"],"species":["_none","mew"]}}
        self.ct = combat.from_metadata(self.meta)
        self.table = transition.MoveTable(j.array([0,0,50,40,95,95]),j.array([0,1,0,1,10,15]),
            j.array([False,False,True,True,False,False]),j.full(6,100),j.ones(6,bool),j.zeros(6,j.int32),
            j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(2),
            items=items.from_metadata(self.meta),combat=self.ct)

    def live(self,batch=1):
        j = self.j
        s = self.empty(batch,seed=71)
        stats = j.full_like(s.stats,200).at[...,0].set(1000).at[:,0,:,5].set(300)
        return s._replace(stats=stats,level=j.full_like(s.level,100),hp=j.full_like(s.hp,1000),
            max_hp=j.full_like(s.hp,1000),pp=j.full_like(s.pp,10),moves=j.full_like(s.moves,1),
            types=s.types.at[...,0].set(1),done=j.zeros_like(s.done))

    def play(self,s,actions=None,table=None):
        actions = self.j.zeros(s.active.shape,self.j.int32) if actions is None else self.j.array(actions)
        return self.tr.step(s,actions,self.table if table is None else table)

    def test_guts_all_statuses_physical_only_and_exact_item_chain(self):
        j = self.j
        num,den = self.c.attack_modifier(self.ct,j.ones(7,j.int32),j.arange(7),True)
        self.assertEqual(num.tolist(),[1,3,3,3,3,3,3])
        self.assertEqual(den.tolist(),[1,2,2,2,2,2,2])
        num,den = self.c.attack_modifier(self.ct,j.array([1]),j.array([self.m.BRN]),False,item_num=3,item_den=2)
        self.assertEqual((num.tolist(),den.tolist()),([3],[2]))
        num,den = self.c.attack_modifier(self.ct,j.array([1]),j.array([self.m.BRN]),True,item_num=3,item_den=2)
        self.assertEqual(int(self.m.modify(103,num,den)[0]),232)

    def test_thick_fat_changes_base_power_with_gen3_rounding(self):
        power = self.c.base_power(self.ct,self.j.array([2,2,2,0]),self.j.array([10,15,11,10]),self.j.array([95]*4))
        self.assertEqual(power.tolist(),[47,47,95,95])
        self.assertEqual(int(self.c.base_power(self.ct,2,10,self.j.array(1))),0)

    def test_guts_overrides_burn_damage_penalty_and_composes_choice_band(self):
        j = self.j
        s = self.live(32)
        s = s._replace(moves=s.moves.at[:,0,0,0].set(3),status=s.status.at[:,0,0].set(self.m.BRN))
        baseline = self.play(s).state
        guts = s._replace(ability=s.ability.at[:,0,0].set(1))
        result = self.play(guts).state
        band = self.play(guts._replace(item=guts.item.at[:,0,0].set(1))).state
        self.assertTrue(bool(j.all(result.hp[:,1,0] < baseline.hp[:,1,0])))
        self.assertTrue(bool(j.all(band.hp[:,1,0] < result.hp[:,1,0])))
        # Guts does not remove burn residual damage or cure the status.
        self.assertEqual(result.hp[:,0,0].tolist(),[875]*32)
        self.assertEqual(result.status[:,0,0].tolist(),[self.m.BRN]*32)

    def test_thick_fat_integrates_only_fire_and_ice_not_normal_damage(self):
        j = self.j
        for move in (3,4,5):
            s = self.live(16)._replace(moves=self.live(16).moves.at[:,0,0,0].set(move))
            baseline = self.play(s).state
            result = self.play(s._replace(ability=s.ability.at[:,1,0].set(2))).state
            if move == 3: self.assertTrue(bool(j.array_equal(result.hp,baseline.hp)))
            else: self.assertTrue(bool(j.all(result.hp[:,1,0] > baseline.hp[:,1,0])))

    def test_guts_and_item_modifiers_also_apply_to_confusion_self_damage(self):
        j = self.j
        s = self.live(64)
        s = s._replace(confusion_counter=s.confusion_counter.at[:,0,0].set(3),
            status=s.status.at[:,0,0].set(self.m.BRN))
        base = self.play(s).state
        boosted = self.play(s._replace(ability=s.ability.at[:,0,0].set(1),item=s.item.at[:,0,0].set(1))).state
        # Both have the same confusion draw; only self-hit battles lose extra HP.
        self.assertTrue(bool(j.all(boosted.hp[:,0,0] <= base.hp[:,0,0])))
        self.assertTrue(bool(j.any(boosted.hp[:,0,0] < base.hp[:,0,0])))

    def test_ability_traps_resolve_type_and_levitate_without_later_gen_exemptions(self):
        j = self.j
        cases = [(0,3,[1,0],True),(0,3,[3,0],False),(6,3,[1,0],False),
            (0,4,[9,0],True),(0,4,[8,0],False),(5,5,[8,0],True),
            (0,5,[3,0],True),(0,4,[3,9],True)]
        for own,foe,types,expected in cases:
            trapped = self.c.switch_trapped(self.ct,j.array([[own,foe]]),j.array([[types,[1,0]]]),j.array([[100,100]]))
            self.assertEqual(bool(trapped[0,0]),expected)

    def test_fainted_or_bench_trapper_does_not_block_switching(self):
        s = self.live()
        s = s._replace(ability=s.ability.at[0,1,1].set(5))
        self.assertTrue(bool(self.j.all(self.tr.action_mask(s,self.table)[0,0,4:])))
        s = s._replace(ability=s.ability.at[0,1,0].set(5),hp=s.hp.at[0,1,0].set(0))
        self.assertTrue(bool(self.j.all(self.tr.action_mask(s,self.table)[0,0,4:])))

    def test_trapped_voluntary_switch_is_invalid_but_forced_switch_remains_legal(self):
        s = self.live()._replace(ability=self.live().ability.at[0,1,0].set(5))
        bad = self.play(s,[[4,0]])
        self.assertEqual(bad.invalid_actions.tolist(),[True])
        for a,b in zip(s,bad.state): self.assertTrue(bool(self.j.array_equal(a,b)))
        forced = s._replace(forced_switch=s.forced_switch.at[0,0].set(True))
        good = self.play(forced,[[4,0]])
        self.assertEqual(good.invalid_actions.tolist(),[False])
        self.assertEqual(int(good.state.active[0,0]),1)
        self.assertEqual(int(good.state.turn[0]),0)

    def test_trap_recomputes_after_switch_and_does_not_reject_already_queued_switch(self):
        s = self.live()._replace(ability=self.live().ability.at[0,1,0].set(3))
        switched = self.play(s,[[0,4]]).state
        self.assertTrue(bool(self.j.all(self.tr.action_mask(switched,self.table)[0,0,4:])))
        s = self.live()._replace(ability=self.live().ability.at[0,1,1].set(5))
        both = self.play(s,[[4,4]])
        self.assertEqual(both.invalid_actions.tolist(),[False])
        self.assertEqual(both.state.active.tolist(),[[1,1]])
        self.assertFalse(bool(self.j.any(self.tr.action_mask(both.state,self.table)[0,0,4:])))

    def test_ability_mask_and_step_jit_batch_prefix_and_terminal_identity(self):
        j = self.j
        s = self.live(4)
        s = s._replace(moves=s.moves.at[:,0,0,0].set(3),
            ability=s.ability.at[:,1,0].set(3).at[:,0,0].set(1))
        # Invalid row2 and terminal row3 must remain unchanged.
        s = s._replace(done=s.done.at[3].set(True))
        actions = j.array([[0,0],[0,0],[4,0],[0,0]])
        result = self.jax.jit(self.tr.step)(s,actions,self.table); result.state.hp.block_until_ready()
        eager = self.tr.step(s,actions,self.table)
        for a,b in zip(self.jax.tree.leaves(result),self.jax.tree.leaves(eager)):
            self.assertTrue(bool(j.array_equal(a,b)))
        small = self.jax.jit(self.tr.action_mask)(self.jax.tree.map(lambda x:x[:2],s),self.table)
        self.assertTrue(bool(j.array_equal(small,self.tr.action_mask(s,self.table)[:2])))
        for a,b in zip(s,result.state): self.assertTrue(bool(j.array_equal(a[2:],b[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_gen3_combat_and_trap_events(self):
        result = subprocess.run([NODE,str(Path(__file__).with_name("gpu_combat_oracle.cjs")),
            str(Path(SHOWDOWN).resolve())],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        doc = json.loads(result.stdout)
        self.assertEqual(doc["guts"],{"attack":154,"confusion":53})
        self.assertEqual(doc["guts_band"],{"attack":232,"confusion":79})
        self.assertEqual(doc["thick_fat_power"],[47,47,95])
        self.assertEqual(doc["trap"],[True,False,False,True,False,True,True,True])


if __name__ == "__main__": unittest.main()
