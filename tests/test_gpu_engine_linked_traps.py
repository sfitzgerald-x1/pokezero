"""Direct Gen 3 move-trap links; not all trap/volatile or gameplay coverage."""
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
class LinkedTrapTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import baton,combat,damage_moves,effects,entry,linked_traps,phazing,pressure,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.tr,self.empty=jax,j,transition,empty_state
        self.names=["_none","meanlook","spiderweb","block","batonpass","splash","struggle","protect","substitute","roar","seismictoss"]
        types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost","steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        meta={"moves":[None]+[{"flags":{"protect":1}} if name in {"meanlook","spiderweb","block"}
            else {"flags":{"protect":1,"bypasssub":1}} if name=="roar" else {"flags":{}} for name in self.names[1:]],
            "vocabulary":{"moves":self.names,"types":types,"species":["_none"],
                "items":["_none"],"abilities":["_none","naturalcure","shadowtag","levitate","intimidate","pressure"]}}
        n=len(self.names)
        self.table=transition.MoveTable(j.array([0]*6+[50]+[0]*4),j.array([0,1,7,1,1,1,0,1,1,1,2]),
            j.zeros(n,bool),j.full(n,100),j.ones(n,bool),j.array([0]*7+[3,0,-6,0]),
            j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(6),
            effects=effects.from_metadata(meta),entry=entry.from_metadata(meta),combat=combat.from_metadata(meta),
            pressure=pressure.from_metadata(meta),damage_moves=damage_moves.from_metadata(meta),
            baton=baton.from_metadata(meta),phazing=phazing.from_metadata(meta),linked_traps=linked_traps.from_metadata(meta))

    def live(self,batch=1,own="meanlook",foe="splash"):
        j=self.j;s=self.empty(batch,seed=71)
        return s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),level=j.full_like(s.level,73),
            pp=j.full_like(s.pp,10),types=s.types.at[...,0].set(1),done=j.zeros_like(s.done),
            moves=j.full_like(s.moves,5).at[:,0,0,0].set(self.names.index(own)).at[:,1,0,0].set(self.names.index(foe)))

    def play(self,s,actions=None):
        return self.tr.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else self.j.array(actions),self.table)

    def replace_move(self,s,side,name):
        rows=self.j.arange(s.hp.shape[0])
        return s._replace(moves=s.moves.at[rows,side,s.active[:,side],0].set(self.names.index(name)))

    def can_switch(self,s,side):
        return bool(self.j.any(self.tr.action_mask(s,self.table)[0,side,4:]))

    def test_three_direct_traps_bind_the_actual_opposing_owner(self):
        for name in ("meanlook","spiderweb","block"):
            r=self.play(self.live(own=name)).state
            self.assertEqual((int(r.trap_source[0,1,0]),int(r.pp[0,0,0,0])),(0,9))
            self.assertFalse(self.can_switch(r,1));self.assertTrue(self.can_switch(r,0))
            self.assertTrue(bool(self.j.all(r.trap_source[0,0]==-1)))

    def test_substitute_protect_ghost_and_pressure_follow_gen3_move_surface(self):
        for name in ("meanlook","spiderweb","block"):
            s=self.live(own=name);s=s._replace(substitute_hp=s.substitute_hp.at[0,1,0].set(100))
            r=self.play(s).state
            self.assertEqual(int(r.trap_source[0,1,0]),-1);self.assertTrue(self.can_switch(r,1))
            self.assertTrue(self.can_switch(self.play(self.live(own=name,foe="protect")).state,1))
        s=self.live();s=s._replace(types=s.types.at[0,1,0,0].set(8),ability=s.ability.at[0,1,0].set(5))
        table=self.table._replace(type_num=self.table.type_num.at[1,8].set(0))
        r=self.tr.step(s,self.j.array([[0,0]]),table).state
        self.assertFalse(self.can_switch(r,1));self.assertEqual(int(r.pp[0,0,0,0]),8)

    def test_baton_transfers_owner_then_target_and_normal_owner_departure_releases(self):
        s=self.play(self.live()).state
        s=self.replace_move(s,0,"batonpass");paused=self.play(s).state
        self.assertTrue(bool(paused.forced_switch[0,0]));self.assertFalse(self.can_switch(paused,1))
        owner=self.play(paused,[[4,0]]).state
        self.assertEqual((int(owner.active[0,0]),int(owner.trap_source[0,1,0])),(1,1))
        self.assertFalse(self.can_switch(owner,1))
        owner=self.replace_move(owner,1,"batonpass");target=self.play(owner).state
        self.assertTrue(bool(target.forced_switch[0,1]))
        passed=self.play(target,[[0,4]]).state
        self.assertEqual((passed.active.tolist(),int(passed.trap_source[0,1,1]),int(passed.trap_source[0,1,0])),([[1,1]],1,-1))
        self.assertFalse(self.can_switch(passed,1))
        released=self.play(passed,[[4,0]]).state
        self.assertTrue(self.can_switch(released,1));self.assertTrue(bool(self.j.all(released.trap_source==-1)))

    def test_mutual_traps_both_pass_without_losing_or_crossing_links(self):
        s=self.play(self.live(foe="meanlook")).state
        self.assertEqual(s.trap_source[0,:,0].tolist(),[0,0])
        for side in range(2):s=self.replace_move(s,side,"batonpass")
        first=self.play(s).state;middle=self.play(first,[[4,0]]).state;last=self.play(middle,[[0,4]]).state
        self.assertEqual((last.active.tolist(),last.trap_source[0,:,1].tolist(),last.trap_source[0,:,0].tolist()),([[1,1]],[1,1],[-1,-1]))
        self.assertFalse(self.can_switch(last,0));self.assertFalse(self.can_switch(last,1))
        self.assertEqual(int(last.turn[0]),2);self.assertTrue(bool(self.j.array_equal(first.keys,last.keys)))

    def test_owner_and_target_faint_clear_links_before_replacement(self):
        for victim in (0,1):
            s=self.play(self.live()).state;s=s._replace(hp=s.hp.at[0,victim,0].set(1))
            s=self.replace_move(s,1-victim,"seismictoss");r=self.play(s).state
            self.assertEqual(int(r.hp[0,victim,0]),0)
            self.assertTrue(bool(r.pending_residual[0]));self.assertTrue(bool(self.j.all(r.trap_source==-1)))
            self.assertTrue(self.can_switch(r,1))  # Forced target or released living target.

    def test_phazing_owner_or_target_releases_without_passing_link(self):
        for dragged in (0,1):
            s=self.play(self.live()).state;s=s._replace(hp=s.hp.at[0,dragged,2:].set(0))
            s=self.replace_move(s,0,"splash");s=self.replace_move(s,1-dragged,"roar")
            r=self.play(s).state
            self.assertEqual(int(r.active[0,dragged]),1)
            self.assertTrue(bool(self.j.all(r.trap_source==-1)));self.assertTrue(self.can_switch(r,1))

    def test_baton_owner_hazard_ko_releases_and_old_owner_reentry_cannot_reactivate(self):
        s=self.play(self.live()).state;s=self.replace_move(s,0,"batonpass")
        s=s._replace(hp=s.hp.at[0,0,1].set(1),spikes=s.spikes.at[0,0].set(3))
        first=self.play(s).state;middle=self.play(first,[[4,0]]).state
        self.assertEqual(int(middle.hp[0,0,1]),0);self.assertTrue(bool(middle.pending_residual[0]))
        self.assertTrue(bool(self.j.all(middle.trap_source==-1)));self.assertTrue(self.can_switch(middle,1))
        last=self.play(middle,[[4,0]]).state
        self.assertEqual(int(last.active[0,0]),0);self.assertTrue(bool(self.j.all(last.trap_source==-1)))
        self.assertTrue(self.can_switch(last,1));self.assertEqual(int(last.pp[0,1,0,0]),9)

    def test_nonzero_party_indices_and_baton_not_hardcoded_to_starting_slot(self):
        s=self.live();s=s._replace(active=self.j.array([[3,2]]),
            trap_source=s.trap_source.at[0,1,2].set(3),moves=s.moves.at[0,0,3,0].set(4))
        paused=self.play(s).state;r=self.play(paused,[[8,0]]).state
        self.assertEqual((int(r.active[0,0]),int(r.trap_source[0,1,2])),(5,5))
        self.assertFalse(self.can_switch(r,1))
        r=self.play(r,[[4,0]]).state
        self.assertEqual(int(r.active[0,0]),0);self.assertTrue(self.can_switch(r,1))

    def test_jit_batch_prefix_and_invalid_trapped_switch_identity(self):
        j=self.j;s=self.live(3);actions=j.zeros((3,2),j.int32)
        eager=self.tr.step(s,actions,self.table);compiled=self.jax.jit(self.tr.step)(s,actions,self.table)
        one=self.tr.step(self.jax.tree.map(lambda x:x[:1],s),actions[:1],self.table)
        for a,b,c in zip(eager.state,compiled.state,one.state):
            self.assertTrue(bool(j.array_equal(a,b)));self.assertTrue(bool(j.array_equal(a[:1],c)))
        trapped=eager.state;frozen=self.play(trapped,[[0,4],[0,4],[0,4]])
        self.assertTrue(bool(j.all(frozen.invalid_actions)))
        for a,b in zip(trapped,frozen.state):self.assertTrue(bool(j.array_equal(a,b)))
        done=trapped._replace(done=j.ones(3,bool));result=self.play(done)
        for a,b in zip(done,result.state):self.assertTrue(bool(j.array_equal(a,b)))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned source oracle")
    def test_pinned_gen3_link_transfer_and_release(self):
        p=subprocess.run([NODE,str(Path(__file__).with_name("gpu_linked_traps_oracle.cjs")),str(Path(SHOWDOWN).resolve())],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);d=json.loads(p.stdout)
        def bound(active,owner="mew",request="move"):
            return dict(active=active,trapped=owner is not None,owner=owner,request=request,canSwitch=owner is None)
        self.assertEqual(d["chain"],dict(initial=bound(["mew","mew"]),ownerPass=bound(["ditto","mew"],"ditto"),
            targetPass=bound(["ditto","ditto"],"ditto"),ownerLeaves=bound(["mew","ditto"],None)))
        for name in ("meanSub","webSub","blockProtect"):self.assertEqual(d[name],bound(["mew","mew"],None))
        for name in ("web","block"):self.assertEqual(d[name],bound(["mew","mew"]))
        self.assertEqual(d["ghost"],bound(["mew","gengar"]))
        self.assertEqual(d["ownerFaint"],bound(["mew","mew"],None,"switch"))


if __name__=="__main__":unittest.main()
