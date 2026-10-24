"""Held-item and selected residual fixtures; not whole-game parity."""
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
class ItemTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,items,mechanics,transition
        from gpu_engine.state import empty_state,legal_actions
        self.jax,self.j,self.i,self.m,self.t,self.empty,self.legal = jax,j,items,mechanics,transition,empty_state,legal_actions
        self.names = ["_none","tackle","splash","struggle","toxic","rest","recover","substitute"]
        self.item_names = ["_none","leftovers","lumberry","choiceband","thickclub","lightball",
            "petayaberry","salacberry","liechiberry"]
        self.species = ["_none","pikachu","marowak","cubone","ditto"]
        types = ["_none","normal","fighting","flying","poison","ground","rock",
            "bug","ghost","steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        meta = {"vocabulary":{"moves":self.names,"items":self.item_names,"species":self.species,"types":types}}
        self.it = items.from_metadata(meta)
        self.table = transition.MoveTable(j.array([0,40,0,50,0,0,0,0]),j.array([0,1,1,0,4,14,1,1]),
            j.array([False,True,False,True,False,False,False,False]),j.full((8,),100),j.ones((8,),bool),
            j.zeros((8,),j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(3),
            effects.from_metadata(meta),self.it)

    def live(self,batch=1):
        j = self.j
        s = self.empty(batch,seed=71)
        stats = j.full_like(s.stats,200).at[...,0].set(101).at[:,0,:,5].set(300)
        return s._replace(stats=stats,level=j.full_like(s.level,100),hp=j.full_like(s.hp,101),
            max_hp=j.full_like(s.hp,101),pp=j.full_like(s.pp,10),moves=j.full_like(s.moves,2),
            species=j.full_like(s.species,4),types=s.types.at[...,0].set(1),done=j.zeros_like(s.done))

    def play(self,s,own="splash",foe="splash",table=None):
        s = s._replace(moves=s.moves.at[:,0,0,0].set(self.names.index(own)).at[:,1,0,0].set(self.names.index(foe)))
        return self.t.step(s,self.j.zeros(s.active.shape,self.j.int32),table or self.table).state

    def test_item_modifiers_match_category_and_species(self):
        for item,species,physical,expected in ((3,4,True,(3,2)),(3,4,False,(1,1)),
            (4,2,True,(2,1)),(4,3,True,(2,1)),(4,1,True,(1,1)),(4,2,False,(1,1)),
            (5,1,False,(2,1)),(5,1,True,(1,1)),(5,4,False,(1,1)),(0,1,True,(1,1))):
            num,den = self.i.attack_modifier(self.it,self.j.array(item),self.j.array(species),self.j.array(physical))
            self.assertEqual((int(num),int(den)),expected)

    def test_item_stat_modifiers_change_eligible_damage_only(self):
        s = self.live(64)
        s = s._replace(hp=self.j.full_like(s.hp,1000),max_hp=self.j.full_like(s.hp,1000))
        base = self.play(s,"tackle")
        for item,species,physical in ((3,4,True),(4,2,True),(5,1,False)):
            table = self.table._replace(physical=self.table.physical.at[1].set(physical))
            reference = self.play(s,"tackle",table=table)
            modified = s._replace(item=s.item.at[:,0,0].set(item),species=s.species.at[:,0,0].set(species))
            result = self.play(modified,"tackle",table=table)
            self.assertTrue(bool(self.j.all(result.hp[:,1,0] < reference.hp[:,1,0])))
        # Category-ineligible Light Ball does not change the physical attack.
        s = s._replace(item=s.item.at[:,0,0].set(5),species=s.species.at[:,0,0].set(1))
        self.assertTrue(bool(self.j.array_equal(self.play(s,"tackle").hp[:,1,0],base.hp[:,1,0])))

    def test_choice_band_and_club_also_modify_confusion_self_damage(self):
        s = self.live(64)
        s = s._replace(hp=self.j.full_like(s.hp,1000),max_hp=self.j.full_like(s.hp,1000),
            confusion_counter=s.confusion_counter.at[:,0,0].set(2))
        base = self.play(s)
        hit = base.hp[:,0,0] < 1000
        self.assertTrue(bool(self.j.any(hit)))
        for item,species in ((3,4),(4,2)):
            r = self.play(s._replace(item=s.item.at[:,0,0].set(item),species=s.species.at[:,0,0].set(species)))
            self.assertTrue(bool(self.j.all(~hit | (r.hp[:,0,0] < base.hp[:,0,0]))))
            self.assertTrue(bool(self.j.all(r.pp[:,0,0,0] == self.j.where(hit,10,9))))

    def test_choice_band_locks_attempted_move_including_status_and_miss(self):
        s = self.live()._replace(item=self.live().item.at[0,0,0].set(3))
        for move in ("splash","tackle"):
            table = self.table._replace(accuracy=self.table.accuracy.at[1].set(0),
                always_hit=self.table.always_hit.at[1].set(False))
            r = self.play(s,move,table=table)
            self.assertEqual(int(r.choice_move[0,0,0]),self.names.index(move))
            self.assertEqual(int(r.pp[0,0,0,0]),9)

    def test_choice_mask_refuses_other_move_and_switch_clears_departing_lock(self):
        s = self.live()
        s = s._replace(item=s.item.at[0,0,0].set(3),moves=s.moves.at[0,0,0].set(self.j.array([1,2,4,5])),
            choice_move=s.choice_move.at[0,0,0].set(1))
        self.assertEqual(self.legal(s)[0,0,:4].tolist(),[True,False,False,False])
        rejected = self.t.step(s,self.j.array([[1,0]]),self.table)
        self.assertEqual(rejected.invalid_actions.tolist(),[True])
        for a,b in zip(self.jax.tree.leaves(s),self.jax.tree.leaves(rejected.state)):
            self.assertTrue(bool(self.j.array_equal(a,b)))
        switched = self.t.step(s,self.j.array([[4,0]]),self.table).state
        self.assertEqual(int(switched.choice_move[0,0,0]),0)

    def test_choice_locked_move_exhaustion_selects_struggle_not_another_slot(self):
        s = self.live()
        s = s._replace(item=s.item.at[0,0,0].set(3),moves=s.moves.at[0,0,0].set(self.j.array([1,2,4,5])),
            choice_move=s.choice_move.at[0,0,0].set(2),pp=s.pp.at[0,0,0,1].set(0))
        self.assertEqual(self.legal(s)[0,0,:4].tolist(),[True,False,False,False])
        r = self.t.step(s,self.j.zeros((1,2),self.j.int32),self.table).state
        self.assertEqual(int(r.pp[0,0,0,0]),10)
        self.assertEqual(int(r.choice_move[0,0,0]),2)
        self.assertLess(int(r.hp[0,1,0]),101)
        self.assertLess(int(r.hp[0,0,0]),101)

    def test_inability_to_move_does_not_create_choice_lock(self):
        s = self.live()
        s = s._replace(item=s.item.at[0,0,0].set(3),status=s.status.at[0,0,0].set(self.m.SLP),
            status_counter=s.status_counter.at[0,0,0].set(3))
        r = self.play(s,"tackle")
        self.assertEqual((int(r.choice_move[0,0,0]),int(r.pp[0,0,0,0])),(0,10))

    def test_lum_cures_status_and_confusion_together_and_consumes_once(self):
        s = self.live(6)
        s = s._replace(item=s.item.at[:,0,0].set(2),status=s.status.at[:,0,0].set(self.j.arange(1,7)),
            status_counter=s.status_counter.at[:,0,0].set(3),confusion_counter=s.confusion_counter.at[:,0,0].set(3))
        r = self.play(s)
        self.assertTrue(bool(self.j.all(r.item[:,0,0] == 0)))
        self.assertTrue(bool(self.j.all(r.status[:,0,0] == 0)))
        self.assertTrue(bool(self.j.all(r.status_counter[:,0,0] == 0)))
        self.assertTrue(bool(self.j.all(r.confusion_counter[:,0,0] == 0)))
        self.assertTrue(bool(self.j.all(r.pp[:,0,0,0] == 9)))

    def test_lum_cures_new_toxic_before_slower_action_and_residual(self):
        s = self.live()._replace(item=self.live().item.at[0,1,0].set(2))
        r = self.play(s,"toxic","tackle")
        self.assertEqual((int(r.item[0,1,0]),int(r.status[0,1,0]),int(r.hp[0,1,0])),(0,0,101))
        self.assertLess(int(r.hp[0,0,0]),101)
        self.assertEqual(int(r.pp[0,1,0,0]),9)

    def test_rest_lum_cures_its_new_sleep_after_healing(self):
        s = self.live()
        s = s._replace(item=s.item.at[0,0,0].set(2),hp=s.hp.at[0,0,0].set(17))
        r = self.play(s,"rest")
        self.assertEqual((int(r.hp[0,0,0]),int(r.status[0,0,0]),int(r.status_counter[0,0,0]),int(r.item[0,0,0])),
            (101,0,0,0))

    def test_lum_does_not_consume_when_healthy_or_on_bench_or_fainted(self):
        s = self.live()
        s = s._replace(item=s.item.at[0,0,:3].set(2),status=s.status.at[0,0,1:].set(self.m.BRN),
            hp=s.hp.at[0,0,2].set(0))
        r = self.play(s)
        self.assertEqual(r.item[0,0,:3].tolist(),[2,2,2])
        self.assertEqual(int(r.status[0,0,1]),self.m.BRN)
        switched = self.t.step(r,self.j.array([[4,0]]),self.table).state
        self.assertEqual((int(switched.item[0,0,1]),int(switched.status[0,0,1])),(0,0))

    def test_leftovers_floors_sixteenth_min_one_and_caps_without_resurrection(self):
        for hp,max_hp,expected in ((50,101,56),(100,101,101),(5,15,6),(0,101,0)):
            result = self.i.leftovers_heal(self.it,self.j.array(1),self.j.array(hp),self.j.array(max_hp))
            self.assertEqual(int(result),expected)
        s = self.live()._replace(item=self.live().item.at[0,0,0].set(1),hp=self.live().hp.at[0,0,0].set(50))
        self.assertEqual(int(self.play(s).hp[0,0,0]),56)

    def test_leftovers_precedes_status_damage_in_gen3(self):
        s = self.live()
        s = s._replace(item=s.item.at[0,0,0].set(1),hp=s.hp.at[0,0,0].set(7),
            status=s.status.at[0,0,0].set(self.m.PSN))
        self.assertEqual(int(self.play(s).hp[0,0,0]),1)

    def test_pinch_berries_consume_once_on_living_active_with_thresholds_and_caps(self):
        j=self.j;rows=j.arange(24);side=j.tile(j.array([0,1]),12)
        item=j.repeat(j.array([6,7,8]),8);stat=j.repeat(j.array([2,4,0]),8)
        high=j.array([6,14,22]);low=high+1
        s=self.live(24);s=s._replace(item=s.item.at[rows,side,0].set(item).at[:,:,1].set(item[:,None]),
            hp=s.hp.at[rows,side,0].set(j.tile(j.array([25,25,25,25,26,26,25,25]),3)).at[:,:,1].set(1),
            boosts=s.boosts.at[high,side[high],0,stat[high]].set(6).at[low,side[low],0,stat[low]].set(-6))
        r=self.play(s)
        self.assertEqual(r.item[rows,side,0].tolist(),sum(([0,0,0,0,i,i,0,0] for i in (6,7,8)),[]))
        self.assertEqual(r.boosts[rows,side,0,stat].tolist(),[1,1,1,1,0,0,6,-5]*3)
        self.assertEqual(r.item[:,:,1].tolist(),[[i,i] for i in item.tolist()])  # no bench holder eats
        for column in range(7):
            self.assertTrue(bool(j.all(j.where(stat==column,True,r.boosts[rows,side,0,column]==0))))
        for a,b in zip(r.boosts,self.play(r).boosts):self.assertTrue(bool(j.array_equal(a,b)))
        compiled=self.jax.jit(self.t.step)(s,j.zeros((24,2),j.int32),self.table).state
        for a,b in zip(r,compiled):self.assertTrue(bool(j.array_equal(a,b)))

    def test_pinch_berry_waits_until_item_residual_before_poison_and_not_replacement(self):
        j=self.j
        for item,stat in ((7,4),(8,0)):
            s=self.live();s=s._replace(item=s.item.at[0,0,0].set(item),hp=s.hp.at[0,0,0].set(26),
                status=s.status.at[0,0,0].set(self.m.PSN))
            r=self.play(s)
            self.assertEqual((int(r.hp[0,0,0]),int(r.item[0,0,0]),int(r.boosts[0,0,0,stat])),(14,item,0))
            r=self.play(r)
            self.assertEqual((int(r.hp[0,0,0]),int(r.item[0,0,0]),int(r.boosts[0,0,0,stat])),(2,0,1))
        for item,stat in ((6,2),(7,4),(8,0)):
            for seat in (0,1):
                for forced in (False,True):
                    s=self.live();s=s._replace(hp=s.hp.at[0,seat,0].set(0 if forced else 101).at[0,seat,1].set(25),
                        item=s.item.at[0,seat,1].set(item))
                    actions=j.zeros((1,2),j.int32).at[0,seat].set(4)
                    result=self.t.step(s,actions,self.table)
                    self.assertFalse(bool(result.invalid_actions[0]))
                    self.assertEqual((int(result.state.active[0,seat]),int(result.state.item[0,seat,1]),
                        int(result.state.boosts[0,seat,1,stat])),(1,item if forced else 0,0 if forced else 1))

    def test_liechi_does_not_boost_current_action_but_boosts_next_physical_hit(self):
        for seat in (0,1):
            s=self.live(8);s=s._replace(hp=s.hp.at[:,seat,0].set(25).at[:,1-seat,0].set(1000),
                max_hp=s.max_hp.at[:,1-seat,0].set(1000),item=s.item.at[:,seat,0].set(8))
            moves=("tackle","splash") if seat==0 else ("splash","tackle")
            control=self.play(s,*moves,table=self.table._replace(items=None))
            first=self.play(s,*moves)
            self.assertEqual(first.hp.tolist(),control.hp.tolist())
            self.assertEqual(first.item[:,seat,0].tolist(),[0]*8)
            self.assertEqual(first.boosts[:,seat,0,0].tolist(),[1]*8)
            second=self.play(first,*moves)
            control=self.play(first._replace(boosts=first.boosts.at[:,seat,0,0].set(0)),*moves)
            self.assertTrue(bool(self.j.all(second.hp[:,1-seat,0] < control.hp[:,1-seat,0])))

    def test_petaya_does_not_boost_the_current_action(self):
        s=self.live(8);s=s._replace(hp=s.hp.at[:,0,0].set(25).at[:,1,0].set(1000),
            max_hp=s.max_hp.at[:,1,0].set(1000),item=s.item.at[:,0,0].set(6))
        # A special-damage table control isolates timing, not catalog coverage.
        table=self.table._replace(physical=self.table.physical.at[1].set(False))
        before=self.play(s,"tackle",table=table._replace(items=None))
        first=self.play(s,"tackle",table=table)
        self.assertEqual(first.hp.tolist(),before.hp.tolist())
        self.assertEqual(first.boosts[:,0,0,2].tolist(),[1]*8)
        second=self.play(first,"tackle",table=table)
        control=self.play(first._replace(boosts=first.boosts.at[:,0,0,2].set(0)),"tackle",table=table)
        self.assertTrue(bool(self.j.all(second.hp[:,1,0] < control.hp[:,1,0])))

    def test_faster_poison_faint_ends_before_slower_leftovers(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[:,:,1:].set(0).at[0,0,0].set(1).at[0,1,0].set(50),
            status=s.status.at[0,0,0].set(self.m.PSN),item=s.item.at[0,1,0].set(1))
        r = self.play(s)
        self.assertEqual(r.hp[0,:,0].tolist(),[0,50])
        self.assertEqual(int(r.winner[0]),1)

    def test_at_equal_speed_both_leftovers_precede_either_status(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[:,:,1:].set(0).at[0,0,0].set(1).at[0,1,0].set(50),
            status=s.status.at[0,0,0].set(self.m.PSN),item=s.item.at[0,1,0].set(1),
            stats=s.stats.at[0,0,0,5].set(200))
        self.assertEqual(self.play(s).hp[0,:,0].tolist(),[0,56])

    def test_residual_speed_order_decides_first_last_mon_faint(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[:,:,1:].set(0).at[:,:,0].set(1),
            status=s.status.at[:,:,0].set(self.m.PSN),stats=s.stats.at[0,0,0,5].set(100).at[0,1,0,5].set(300))
        r = self.play(s)
        self.assertEqual(r.hp[0,:,0].tolist(),[1,0])
        self.assertEqual((bool(r.done[0]),int(r.winner[0])),(True,0))

    def test_final_move_ko_ends_before_leftovers_and_replacement_has_no_residual(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[0,1,1:].set(0).at[0,0,0].set(50),item=s.item.at[0,0,0].set(1))
        table = self.table._replace(power=self.table.power.at[1].set(1000))
        self.assertEqual(int(self.play(s,"tackle",table=table).hp[0,0,0]),50)
        s = self.live()
        s = s._replace(hp=s.hp.at[0,1,0].set(0).at[0,0,0].set(50),item=s.item.at[0,0,0].set(1))
        r = self.t.step(s,self.j.array([[0,4]]),self.table).state
        self.assertEqual(int(r.hp[0,0,0]),50)

    def test_item_paths_match_jit(self):
        s = self.live(2)
        s = s._replace(item=s.item.at[:,0,0].set(self.j.array([1,3])),hp=s.hp.at[:,0,0].set(50),
            moves=s.moves.at[:,0,0,0].set(1))
        actions = self.j.zeros((2,2),self.j.int32)
        a = self.t.step(s,actions,self.table)
        b = self.jax.jit(self.t.step)(s,actions,self.table)
        self.jax.block_until_ready(b)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b)):
            self.assertTrue(bool(self.j.array_equal(x,y)))
        s=self.live(2);s=s._replace(hp=s.hp.at[:,:,0].set(25),
            item=s.item.at[:,0,0].set(6).at[:,1,0].set(7),stats=s.stats.at[...,5].set(200))
        chance=self.t.sample_chance_inputs(s)
        a=self.t.step_with_chance(s,actions,self.table,
            chance._replace(draws=chance.draws.at[:,0,9].set(0)))
        b=self.t.step_with_chance(s,actions,self.table,
            chance._replace(draws=chance.draws.at[:,0,9].set(1)))
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b)):
            self.assertTrue(bool(self.j.array_equal(x,y)))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Gen 3 item source")
    def test_item_rules_match_pinned_source_handlers(self):
        helper = Path(__file__).with_name("gpu_item_oracle.cjs")
        rows = json.loads(subprocess.run([NODE,str(helper),SHOWDOWN],text=True,capture_output=True,check=True).stdout)
        self.assertEqual(rows["multipliers"],{"band":1.5,"club_cubone":2,"club_marowak":2,"club_other":1,
            "ball_pikachu_spa":2,"ball_other_spa":1,"ball_atk_handler":False})
        self.assertEqual(rows["residual_order"],{"leftovers":[10,4],"psn":[10,6]})
        self.assertEqual(rows["leftovers_requested_heal"],101/16)
        self.assertEqual(rows["lum"],{"eaten":1,"status":"","confused":False})
        self.assertEqual(rows["choice_lock"],"tackle")
        self.assertEqual(rows["unequal_speed_events"],["heal0","status0","heal1","status1"])
        self.assertEqual(rows["equal_speed_event_kinds"],["heal","heal","status","status"])
        self.assertEqual(rows['pinch_handlers'],dict(petayaberry=[10,4,False],salacberry=[10,4,False],
            liechiberry=[10,4,False]))
        cases=rows['pinch'];j=self.j
        self.assertEqual(len(cases),48)
        for name in ("petayaberry","salacberry","liechiberry"):
            self.assertEqual(sum(r["name"]==name for r in cases),16)
        item=j.array([self.item_names.index(r['name']) for r in cases])
        boosts=j.array([r['before'] for r in cases])
        actual_item,actual_boosts=self.i.consume_pinch(self.it,item,boosts,
            j.array([r['hp'] for r in cases]),j.array([r['maxhp'] for r in cases]))
        self.assertEqual(actual_item.tolist(),[self.item_names.index(r['item'] or '_none') for r in cases])
        self.assertEqual(actual_boosts.tolist(),[r['after'] for r in cases])
        self.assertEqual(len(rows['pinch_turns']),4)
        self.assertEqual({(r['seat'],r['move']) for r in rows['pinch_turns']},
            {(seat,move) for seat in (0,1) for move in ('recover','substitute')})
        for row in rows['pinch_turns']:
            seat=row['seat'];move=row['move'];s=self.live()
            s=s._replace(item=s.item.at[0,seat,0].set(8),hp=s.hp.at[0,seat,0].set(row['before_hp']))
            moves=(move,'splash') if seat==0 else ('splash',move)
            result=self.play(s,*moves)
            expected_boosts=[1,0,0,0,0,0,0] if move=='substitute' else [0]*7
            self.assertEqual(row['boosts'],expected_boosts)
            self.assertEqual((row['hp'],row['item'],row['substitute_hp']),
                (25,'',25) if move=='substitute' else (75,'liechiberry',0))
            self.assertEqual((int(result.hp[0,seat,0]),self.item_names[int(result.item[0,seat,0])],
                int(result.substitute_hp[0,seat,0]),result.boosts[0,seat,0].tolist()),
                (row['hp'],row['item'] or '_none',row['substitute_hp'],row['boosts']))
            self.assertEqual(int(result.pp[0,seat,0,0]),9)
            self.assertEqual(row['before_pp']-row['after_pp'],1)
            self.assertEqual(10-int(result.pp[0,seat,0,0]),row['before_pp']-row['after_pp'])
            self.assertTrue(any('|move|' in line and move.capitalize() in line for line in row['events']))
            self.assertEqual(any('|-enditem|' in line and 'Liechi Berry' in line for line in row['events']),
                move=='substitute')


if __name__ == "__main__":
    unittest.main()
