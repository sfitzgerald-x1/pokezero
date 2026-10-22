"""Selected self-KO and defense-halving boundaries, not game qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get("NODE") or shutil.which("node")
SHOWDOWN=os.environ.get("POKEZERO_SHOWDOWN_ROOT")


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires isolated spike JAX")
class SelfDestructTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import critical,effects,items,mechanics,pressure,self_destruct,sleep_talk,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.m,self.sd,self.tr,self.empty=jax,j,mechanics,self_destruct,transition,empty_state
        self.names=["_none","explosion","selfdestruct","splash","protect","endure","tackle",
            "struggle","sleeptalk","batonpass"]
        types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        meta={"vocabulary":{"moves":self.names,"types":types,
            "abilities":["_none","damp","battlearmor","pressure"],
            "species":["_none"],"items":["_none","choiceband"]},
            "moves":[None]+[{"flags":{"protect":1}} for _ in self.names[1:]]}
        self.profile=self_destruct.from_metadata(meta)
        num=j.ones((18,18),j.int32).at[1,8].set(0)
        self.table=transition.MoveTable(j.array([0,250,200,0,0,0,35,50,0,0]),
            j.array([0,1,1,1,1,1,1,0,1,1]),j.ones(10,bool),j.full(10,100),
            j.ones(10,bool),j.array([0,0,0,0,3,3,0,0,0,0]),num,j.ones_like(num),j.array(7),
            effects=effects.from_metadata(meta),items=items.from_metadata(meta),
            critical=critical.from_metadata(meta),pressure=pressure.from_metadata(meta),
            sleep_talk=sleep_talk.from_metadata(meta),self_destruct=self.profile)

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=77)
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(300).at[:,1,:,2].set(201),
            level=j.full_like(s.level,100),hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            moves=j.full_like(s.moves,3).at[:,0,0,0].set(1),pp=j.full_like(s.pp,10),
            types=s.types.at[:,0,:,0].set(14).at[:,1,:,0].set(11),
            ability=s.ability.at[:,1,:].set(2),done=j.zeros_like(s.done))

    def play(self,s,table=None,actions=None):
        return self.tr.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else actions,
            self.table if table is None else table)

    def test_catalog_and_damp_only_checks_live_current_holders(self):
        j=self.j
        self.assertEqual(self.profile.move.tolist(),[False,True,True]+[False]*7)
        self.assertEqual(self.sd.blocked(self.profile,j.array([1,1,1,3]),
            j.array([[0,1],[1,0],[0,1],[0,1]]),j.array([[1,1],[1,1],[1,0],[1,1]])).tolist(),
            [True,True,False,False])

    def test_defense_halving_floors_after_stat_modification_and_clamps_one(self):
        m=self.m
        common=dict(level=100,power=250,attack=200,defense=201,typeless=False,roll=100)
        self.assertEqual(int(m.gen3_damage(**common,halve_defense=True)),422)
        self.assertEqual(int(m.gen3_damage(**common)),210)
        # (201*3/2 floored=301) modified by 3/2 rounds to 451, then floor/2=225.
        self.assertEqual(int(m.gen3_damage(**common,defense_stage=1,defense_num=3,
            defense_den=2,halve_defense=True)),188)
        low={**common,"defense":1}
        self.assertEqual(int(m.gen3_damage(**low,halve_defense=True)),
            int(m.gen3_damage(**low)))

    def test_normal_hit_self_ko_cancels_slower_action_and_defers_residual(self):
        j=self.j;s=self.live(16)
        r=self.play(s).state
        splits=self.jax.vmap(lambda k:self.jax.random.split(k,2))(s.keys)
        draws=self.jax.vmap(lambda k:self.jax.random.randint(k,(2,12),0,
            j.array([2,100,100,100,16,16,100,4,8,2,2,2])))(splits[:,1])
        damage=self.m.gen3_damage(level=100,power=250,attack=200,defense=201,
            halve_defense=True,roll=85+draws[:,0,5])
        self.assertEqual(r.hp[:,0,0].tolist(),[0]*16)
        self.assertEqual(r.hp[:,1,0].tolist(),(1000-damage).tolist())
        self.assertEqual(r.pp[:,0,0,0].tolist(),[9]*16)
        self.assertEqual(r.pp[:,1,0,0].tolist(),[10]*16)
        self.assertTrue(bool(j.all(r.pending_residual)))
        self.assertTrue(bool(j.all(r.forced_switch[:,0])))
        self.assertFalse(bool(j.any(r.forced_switch[:,1])))
        self.assertTrue(bool(j.array_equal(r.keys,splits[:,0])))

    def test_self_ko_occurs_even_for_miss_protect_ghost_and_substitute(self):
        s=self.live();j=self.j
        miss=self.table._replace(accuracy=self.table.accuracy.at[1].set(0),
            always_hit=self.table.always_hit.at[1].set(False))
        protected=s._replace(moves=s.moves.at[0,1,0,0].set(4))
        ghost=s._replace(types=s.types.at[0,1,0,0].set(8))
        sub=s._replace(substitute_hp=s.substitute_hp.at[0,1,0].set(19))
        for state,table in ((s,miss),(protected,self.table),(ghost,self.table),(sub,self.table)):
            r=self.play(state,table).state
            self.assertEqual(r.hp[0,:,0].tolist(),[0,1000])
            self.assertEqual(int(r.pp[0,0,0,0]),9)
        self.assertEqual(int(self.play(sub).state.substitute_hp[0,1,0]),0)

    def test_damp_charges_caller_without_self_ko_and_foe_still_acts(self):
        for side in (0,1):
            s=self.live();s=s._replace(ability=s.ability.at[0,side,0].set(1))
            r=self.play(s).state
            self.assertEqual(r.hp[0,:,0].tolist(),[1000,1000])
            self.assertEqual(r.pp[0,:,0,0].tolist(),[9,9])
            self.assertFalse(bool(r.pending_residual[0]))
        s=self.live();s=s._replace(ability=s.ability.at[0,1,1].set(1))
        self.assertEqual(int(self.play(s).state.hp[0,0,0]),0)

    def test_before_move_failure_never_reaches_self_ko_or_pp(self):
        from gpu_engine.state import FLINCH
        s=self.live()
        for state in (s._replace(status=s.status.at[0,0,0].set(self.m.SLP),
                status_counter=s.status_counter.at[0,0,0].set(5)),
                s._replace(volatile=s.volatile.at[0,0,0].set(FLINCH))):
            r=self.play(state).state
            self.assertEqual(r.hp[0,:,0].tolist(),[1000,1000])
            self.assertEqual(r.pp[0,:,0,0].tolist(),[10,9])

    def test_damp_refusal_still_choice_locks_the_original_caller(self):
        for move in (1,2):
            s=self.live();s=s._replace(ability=s.ability.at[0,1,0].set(1),
                moves=s.moves.at[0,0,0,0].set(move),item=s.item.at[0,0,0].set(1))
            r=self.play(s).state
            self.assertEqual(int(r.choice_move[0,0,0]),move)
            self.assertEqual(r.hp[0,:,0].tolist(),[1000,1000])

    def test_endure_and_last_party_termination_keep_gen3_tie(self):
        j=self.j;s=self.live();s=s._replace(hp=s.hp.at[0,1,0].set(1),
            moves=s.moves.at[0,1,0,0].set(5))
        self.assertEqual(self.play(s).state.hp[0,:,0].tolist(),[0,1])
        last=self.live();last=last._replace(hp=j.zeros_like(last.hp).at[0,0,0].set(1000).at[0,1,0].set(1))
        r=self.play(last).state
        self.assertTrue(bool(r.done[0]));self.assertEqual(int(r.winner[0]),2)
        survive=last._replace(hp=last.hp.at[0,1,0].set(1000))
        self.assertEqual(int(self.play(survive).state.winner[0]),1)

    def test_called_explosion_only_spends_sleep_talk_pp_and_respects_damp(self):
        s=self.live();s=s._replace(moves=s.moves.at[0,0,0].set(self.j.array([8,1,8,8])),
            status=s.status.at[0,0,0].set(self.m.SLP),status_counter=s.status_counter.at[0,0,0].set(5))
        r=self.play(s).state
        self.assertEqual(int(r.hp[0,0,0]),0)
        self.assertEqual(r.pp[0,0,0].tolist(),[9,10,10,10])
        damp=s._replace(ability=s.ability.at[0,1,0].set(1))
        r=self.play(damp).state
        self.assertEqual(r.hp[0,:,0].tolist(),[1000,1000])
        self.assertEqual(r.pp[0,0,0].tolist(),[9,10,10,10])

    def test_direct_pressure_charge_precedes_hit_and_called_move_is_not_recharged(self):
        s=self.live();s=s._replace(ability=s.ability.at[0,1,0].set(3))
        self.assertEqual(int(self.play(s).state.pp[0,0,0,0]),8)
        talk=s._replace(moves=s.moves.at[0,0,0].set(self.j.array([8,1,8,8])),
            status=s.status.at[0,0,0].set(self.m.SLP),status_counter=s.status_counter.at[0,0,0].set(5))
        self.assertEqual(self.play(talk).state.pp[0,0,0].tolist(),[9,10,10,10])

    def test_jit_batch_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(4);s=s._replace(done=s.done.at[3].set(True))
        actions=j.array([[0,0],[0,0],[9,0],[0,0]])
        a=self.jax.jit(self.tr.step)(s,actions,self.table);a.state.hp.block_until_ready()
        b=self.play(s,actions=actions)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b)):
            self.assertTrue(bool(j.array_equal(x,y)))
        small=self.jax.jit(self.tr.step)(self.jax.tree.map(lambda x:x[:2],s),actions[:2],self.table)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(small)):
            self.assertTrue(bool(j.array_equal(x[:2],y)))
        for x,y in zip(s,a.state):self.assertTrue(bool(j.array_equal(x[2:],y[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_gen3_self_ko_gates_damage_and_endings(self):
        r=subprocess.run([NODE,str(Path(__file__).with_name("gpu_self_destruct_oracle.cjs")),
            str(Path(SHOWDOWN).resolve())],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        result=json.loads(r.stdout);cases=result["cases"]
        for row in result["damage"]:
            got=self.m.gen3_damage(level=100,power={"explosion":250,"selfdestruct":200,"tackle":35}[row["name"]],
                attack=200,defense=201,attack_stage=row["atk"],defense_stage=row["def"],
                critical=row["critical"],burned=row["burn"],roll=self.j.arange(85,101),
                halve_defense=row["name"]!="tackle")
            self.assertEqual(got.tolist(),row["values"])
        self.assertEqual(cases["normal"]["hp"],[0,578])
        self.assertEqual(cases["selfdestruct"]["hp"],[0,662])
        for name in ("miss","protect","ghost","sub"):
            self.assertEqual(cases[name]["hp"],[0,1000])
        for name in ("damp","self_damp"):
            self.assertEqual(cases[name]["hp"],[1000,1000]);self.assertEqual(cases[name]["request"],"move")
        self.assertEqual(cases["last_ko"]["winner"],"");self.assertTrue(cases["last_ko"]["ended"])
        self.assertEqual(cases["last_survive"]["winner"],"seat1")
        self.assertEqual(cases["endure"]["hp"],[0,1])
        self.assertEqual(cases["foe_attack"]["pp"],[7,56])
        self.assertEqual(cases["pressure"]["pp"][0],6)
        for name in ("talk","talk_pressure"):
            self.assertEqual(cases[name]["hp"],[0,578])
            self.assertEqual((cases[name]["pp"][0],cases[name]["calledPP"]),(15,8))
        self.assertEqual(cases["talk_damp"]["hp"],[1000,1000])
        self.assertEqual(cases["damp_choice"]["lock"],"explosion")


if __name__=="__main__":unittest.main()
