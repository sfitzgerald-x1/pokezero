"""Selected target-secondary boundaries; all moves remain placeholders."""
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
class SecondaryTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import absorption,baton,effects,items,mechanics,secondary,transition,weather
        from gpu_engine.state import empty_state
        self.jax,self.j,self.m,self.s,self.tr,self.empty=jax,j,mechanics,secondary,transition,empty_state
        self.names=["_none"]+list(secondary.EFFECTS)+["protect","substitute","splash","struggle",
            "batonpass","thunderwave","toxic","willowisp"]
        self.types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        self.abilities=["_none","serenegrace","shielddust","clearbody","whitesmoke","innerfocus",
            "limber","waterveil","magmaarmor","immunity","cloudnine","voltabsorb"]
        meta={"vocabulary":{"moves":self.names,"types":self.types,"abilities":self.abilities,
            "species":["_none"],"items":["_none","lumberry"]},
            "moves":[None]+[{"id":name,"source_type":"Fire" if name in {"firepunch","fireblast","flamethrower"} else "Normal","category":"Physical",
                "flags":{"protect":1}} for name in self.names[1:]]}
        self.profile=secondary.from_metadata(meta)
        n=len(self.names);power=j.array([50 if name in secondary.EFFECTS or name=="struggle" else 0 for name in self.names])
        typ=j.array([10 if name in {"firepunch","fireblast","flamethrower"} else 1 for name in self.names]).at[self.id("thunderbolt")].set(13)
        self.table=transition.MoveTable(power,typ,j.ones(n,bool),j.full(n,100),j.zeros(n,bool),
            j.zeros(n,j.int32).at[self.id("protect")].set(4),j.ones((18,18),j.int32),
            j.ones((18,18),j.int32),j.array(self.id("struggle")),effects=effects.from_metadata(meta),
            items=items.from_metadata(meta),weather=weather.from_metadata(meta),
            absorption=absorption.from_metadata(meta),baton=baton.from_metadata(meta),secondary=self.profile)

    def id(self,name):return self.names.index(name)
    def ability(self,name):return self.abilities.index(name)

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=117)
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),
            hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),level=j.full_like(s.level,100),
            types=s.types.at[...,0].set(11),moves=j.full_like(s.moves,self.id("splash")),
            pp=j.full_like(s.pp,10),done=j.zeros_like(s.done))

    def direct(self,state,name,roll=0,eligible=True,weather=0):
        return self.s.apply(state,0,self.j.full(state.active.shape[0],self.id(name)),roll,self.profile,
            eligible=eligible,effective_weather=weather)

    def play(self,state,name,*,target="splash",table=None,actions=None):
        state=state._replace(moves=state.moves.at[:,0,0,0].set(self.id(name)).at[:,1,0,0].set(self.id(target)))
        if table is None:table=self.table
        return self.tr.step(state,self.j.zeros(state.active.shape,self.j.int32) if actions is None else actions,table)

    def certain(self):return self.table._replace(secondary=self.profile._replace(chance=self.j.full_like(self.profile.chance,100)))

    def test_exact_hundred_roll_thresholds_grace_and_dust(self):
        j=self.j;s=self.live(100);roll=j.arange(100)
        for name,(chance,status,drop,flinch) in self.s.EFFECTS.items():
            for grace in (False,True):
                p=s._replace(ability=s.ability.at[:,0,0].set(self.ability("serenegrace") if grace else 0))
                r=self.direct(p,name,roll)
                changed=(r.status[:,1,0]!=0) | (r.boosts[:,1,0,3]!=0) | ((r.volatile[:,1,0]&1)!=0)
                self.assertEqual(int(changed.sum()),chance*(2 if grace else 1),name)
            p=s._replace(ability=s.ability.at[:,1,0].set(self.ability("shielddust")))
            r=self.direct(p,name,roll)
            self.assertEqual(r.status.tolist(),p.status.tolist())
            self.assertEqual(r.boosts.tolist(),p.boosts.tolist())
            self.assertEqual(r.volatile.tolist(),p.volatile.tolist())

    def test_selected_status_ability_type_existing_and_sun_guards(self):
        j=self.j;m=self.m
        for move,ability,typ,old,sun,expected in (
            ("bodyslam","limber",11,0,0,0),("firepunch","waterveil",11,0,0,0),
            ("icebeam","magmaarmor",11,0,0,0),("sludgebomb","immunity",11,0,0,0),
            ("firepunch","_none",10,0,0,0),("icebeam","_none",15,0,0,0),
            ("sludgebomb","_none",9,0,0,0),("bodyslam","_none",13,0,0,m.PAR),
            ("icebeam","_none",11,0,2,0),("icebeam","_none",11,0,0,m.FRZ),
            ("bodyslam","_none",11,m.PSN,0,m.PSN)):
            s=self.live();s=s._replace(ability=s.ability.at[0,1,0].set(self.ability(ability)),
                types=s.types.at[0,1,0,0].set(typ),status=s.status.at[0,1,0].set(old))
            self.assertEqual(int(self.direct(s,move,weather=sun).status[0,1,0]),expected,(move,ability,typ))
        s=self.live();s=s._replace(hp=s.hp.at[0,1,0].set(0))
        self.assertEqual(int(self.direct(s,"bodyslam").status[0,1,0]),0)
        self.assertEqual(int(self.direct(self.live(),"bodyslam",eligible=False).status[0,1,0]),0)

    def test_spd_only_drop_clamp_and_clear_body(self):
        for name in ("shadowball","crunch","psychic"):
            for ability,stage,expected in (("_none",0,-1),("_none",-6,-6),("clearbody",0,0),("whitesmoke",0,0)):
                s=self.live();s=s._replace(ability=s.ability.at[0,1,0].set(self.ability(ability)),
                    boosts=s.boosts.at[0,1,0,3].set(stage).at[0,1,0,1].set(2))
                r=self.direct(s,name)
                self.assertEqual(r.boosts[0,1,0].tolist(),[0,2,0,expected,0,0,0])

    def test_faster_flinch_blocks_original_foe_action_but_not_inner_focus(self):
        for ability,pp in (("_none",10),("innerfocus",9),("shielddust",9)):
            s=self.live();s=s._replace(ability=s.ability.at[0,1,0].set(self.ability(ability)))
            r=self.play(s,"rockslide",table=self.certain()).state
            self.assertEqual(int(r.pp[0,1,0,0]),pp)
            self.assertEqual(int(r.volatile[0,1,0]&1),0)

    def test_breaking_sub_miss_protect_immunity_absorption_and_ko_do_not_proc(self):
        s=self.live();s=s._replace(substitute_hp=s.substitute_hp.at[0,1,0].set(1))
        r=self.play(s,"thunderbolt",table=self.certain()).state
        self.assertEqual((int(r.substitute_hp[0,1,0]),int(r.hp[0,1,0]),int(r.status[0,1,0])),(0,1000,0))
        cases=[(self.live(),self.certain()._replace(accuracy=self.table.accuracy.at[self.id("thunderbolt")].set(0)),"splash"),
            (self.live(),self.certain(),"protect")]
        s=self.live();s=s._replace(types=s.types.at[0,1,0,0].set(5))
        cases.append((s,self.certain()._replace(type_num=self.table.type_num.at[13,5].set(0)),"splash"))
        s=self.live();s=s._replace(ability=s.ability.at[0,1,0].set(self.ability("voltabsorb")))
        cases.append((s,self.certain(),"splash"))
        s=self.live();s=s._replace(hp=s.hp.at[0,1,0].set(1));cases.append((s,self.certain(),"splash"))
        for s,table,target in cases:
            r=self.play(s,"thunderbolt",target=target,table=table).state
            self.assertEqual(int(r.status[0,1,0]),0)
            self.assertEqual(int(r.pp[0,0,0,0]),9)

    def test_lum_cures_before_slower_action_and_weather_suppression_allows_freeze(self):
        s=self.live();s=s._replace(item=s.item.at[0,1,0].set(1))
        r=self.play(s,"bodyslam",table=self.certain()).state
        self.assertEqual((int(r.item[0,1,0]),int(r.status[0,1,0]),int(r.pp[0,1,0,0])),(0,0,9))
        for ability,status in (("_none",0),("cloudnine",self.m.FRZ)):
            s=self.live();s=s._replace(weather=self.j.full_like(s.weather,2),
                ability=s.ability.at[0,1,0].set(self.ability(ability)),stats=s.stats.at[0,1,0,5].set(500))
            r=self.play(s,"icebeam",table=self.certain()).state
            self.assertEqual(int(r.status[0,1,0]),status)

    def test_primary_target_status_uses_selected_ability_immunities_too(self):
        for move,ability in (("thunderwave","limber"),("toxic","immunity"),("willowisp","waterveil")):
            s=self.live();s=s._replace(ability=s.ability.at[0,1,0].set(self.ability(ability)))
            r=self.play(s,move).state
            self.assertEqual(int(r.status[0,1,0]),0)

    def test_fire_thaw_follows_secondary_and_does_not_turn_freeze_into_burn(self):
        s=self.live();s=s._replace(status=s.status.at[0,1,0].set(self.m.FRZ))
        r=self.play(s,"firepunch",table=self.certain()).state
        self.assertEqual(int(r.status[0,1,0]),0)
        r=self.play(self.live(),"firepunch",table=self.certain()).state
        self.assertEqual(int(r.status[0,1,0]),self.m.BRN)

    def test_baton_preserves_secondary_roll_for_original_queued_move(self):
        j=self.j;s=self.live(8)
        s=s._replace(moves=s.moves.at[:,0,0,0].set(self.id("batonpass")).at[:,1,0,0].set(self.id("thunderbolt")))
        split=self.jax.vmap(lambda k:self.jax.random.split(k,2))(s.keys)
        expected=self.jax.vmap(lambda k:self.jax.random.randint(self.jax.random.fold_in(k,1618033),(2,),0,100))(split[:,1])
        paused=self.tr.step(s,j.zeros((8,2),j.int32),self.table).state
        self.assertTrue(bool(j.array_equal(paused.secondary_draws,expected)))
        self.assertEqual(paused.pp[:,1,0,0].tolist(),[10]*8)
        r=self.tr.step(paused,j.tile(j.array([[4,3]]),(8,1)),self.table).state
        self.assertEqual(r.status[:,0,1].tolist(),j.where(expected[:,1]<10,self.m.PAR,0).tolist())
        self.assertEqual(r.pp[:,1,0,0].tolist(),[9]*8)
        self.assertTrue(bool(j.array_equal(r.keys,paused.keys)))
        self.assertTrue(bool(j.all(r.secondary_draws==0)))

    def test_jit_batch_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(4);s=s._replace(done=s.done.at[3].set(True),
            moves=s.moves.at[:,0,0,0].set(self.id("bodyslam")))
        actions=j.array([[0,0],[0,0],[9,0],[0,0]])
        a=self.jax.jit(self.tr.step)(s,actions,self.table);a.state.hp.block_until_ready()
        b=self.tr.step(s,actions,self.table)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b)):self.assertTrue(bool(j.array_equal(x,y)))
        small=self.jax.jit(self.tr.step)(self.jax.tree.map(lambda x:x[:2],s),actions[:2],self.table)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(small)):self.assertTrue(bool(j.array_equal(x[:2],y)))
        for x,y in zip(s,a.state):self.assertTrue(bool(j.array_equal(x[2:],y[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_catalog_and_authored_secondary_events(self):
        p=subprocess.run([NODE,str(Path(__file__).with_name("gpu_secondary_oracle.cjs")),str(Path(SHOWDOWN).resolve())],
            capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);result=json.loads(p.stdout)
        status={0:"",1:"par",2:"brn",3:"psn",4:"tox",5:"slp",6:"frz"}
        expected=[dict(id=name,chance=row[0],status=status[row[1]],spd=row[2],flinch=row[3]) for name,row in self.s.EFFECTS.items()]
        self.assertEqual(result["catalog"],expected)
        cases=result["cases"]
        self.assertTrue(cases["flinch"]["flinch"]);self.assertFalse(cases["focus"]["flinch"])
        for name in ("dust_status","limber","veil","magma","immunity","broken_sub","sun","ordinary","lum","ice","thaw_burn"):
            self.assertEqual(cases[name]["status"],"",name)
        for name in ("dust_drop","clear","smoke"):self.assertEqual(cases[name]["spd"],0)
        self.assertFalse(cases["broken_sub"]["substitute"])
        for name,value in (("suppressed_sun","frz"),("grace","par"),("electric","par")):
            self.assertEqual(cases[name]["status"],value,name)
        self.assertEqual(cases["lum"]["item"],"")


if __name__=="__main__":unittest.main()
