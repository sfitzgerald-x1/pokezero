"""Team cure and global boost fields without hidden state or partial commits."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires isolated JAX")
class PublicGlobalFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_context as fixtures
        fixtures.PublicContextTests.setUpClass.__func__(cls)
        from pokezero import showdown as reference
        cls.context_fixtures,cls.reference=fixtures,reference

    def initial(self,batch=1):
        fixture=self.context_fixtures.PublicContextTests
        cursor,_,mons,_=fixture.initial(self,batch=batch)
        mons,_=self.m.status_update(mons,side=0,slot=0,encounter=1,status=self.m.POISON)
        for slot,encounter,species,status in [(1,2,"Gardevoir",self.m.SLEEP),(2,3,"Absol",self.m.BURN)]:
            mons,_=self.m.arrive(mons,side=0,slot=slot,encounter=encounter,species=10+slot,
                source_species=self.host.species_id(species),level=80,gender=0,hp_numerator=100,hp_denominator=100,status=status)
        mons,_=self.m.faint(mons,side=0,slot=2,encounter=3)
        mons,_=self.m.arrive(mons,side=0,slot=1,encounter=2,species=11,source_species=self.host.species_id("Gardevoir"),
            level=80,gender=0,hp_numerator=100,hp_denominator=100,status=self.m.SLEEP)
        mons,_=self.m.status_update(mons,side=1,slot=0,encounter=1,status=self.m.PARALYSIS)
        for side in (0,1):mons,_=self.m.boost_update(mons,side=side,stat=side,amount=2 if side==0 else -3)
        beliefs=self.b.initialize(mons.source_species[:,::-1,:],self.host.device)
        return cursor,beliefs,mons,self.c.initialize(beliefs,self.context_tables)

    def test_living_team_status_cure_preserves_fainted_unknown_and_other_side(self):
        initial=self.initial();mons=initial[2]
        updated,ok=self.jax.jit(lambda x:self.m.cure_team(x,side=0))(mons)
        self.assertTrue(bool(ok[0]))
        lines=["|switch|p1a: Snorlax|Snorlax, L80|100/100 psn",
            "|switch|p1a: Gardevoir|Gardevoir, L80|100/100 slp",
            "|switch|p1a: Absol|Absol, L80|100/100 brn","|faint|p1a: Absol",
            "|switch|p1a: Gardevoir|Gardevoir, L80|100/100 slp",
            "|switch|p2a: Absol|Absol, L80|100/100 par","|-cureteam|p1a: Gardevoir"]
        parsed=self.reference.parse_showdown_replay(lines,battle_id="fixture")
        conditions={p.species:p.condition for p in parsed.public_revealed["p1"]}
        self.assertEqual(conditions,{"Snorlax":"100/100","Gardevoir":"100/100","Absol":"0 fnt"})
        self.assertEqual(updated.status[0,0,:3].tolist(),[self.m.NONE]*3)
        self.assertTrue(bool(updated.fainted[0,0,2]))
        self.assertEqual(int(updated.status[0,1,0]),self.m.PARALYSIS)
        self.assertTrue(bool(self.j.array_equal(updated.encounter,mons.encounter)))
        self.assertTrue(bool(self.j.array_equal(updated.hp_numerator,mons.hp_numerator)))

    def test_global_clear_matches_haze_even_without_active_addresses(self):
        mons=self.initial()[2]._replace(active=self.j.full((1,2),-1,self.j.int32))
        updated,ok=self.m.clear_all_boosts(mons)
        boosts={"p1":{"atk":2},"p2":{"def":-3}}
        self.reference._update_boosts(["","-clearallboost"],boosts)
        self.assertEqual(boosts,{"p1":{},"p2":{}})
        self.assertFalse(bool(updated.boosts.any()));self.assertTrue(bool(ok[0]))
        self.assertTrue(bool(self.j.array_equal(updated.active,mons.active)))

    def test_ordered_global_events_and_malformed_row_rollback(self):
        initial=self.initial(batch=2);k=self.i.Kind
        events=[self.i.PublicEvent(k.CURE_TEAM,1,0,self.i.Phase.ACTION,target_side=0),
                self.i.PublicEvent(k.CLEAR_ALL_BOOSTS,2,0,self.i.Phase.ACTION)]
        chunk=self.context_fixtures.PublicContextTests.chunks(self,[events,events])[0]
        chunk=chunk._replace(words=chunk.words.at[0,1,15].set(1))
        result=self.jax.jit(lambda cur,b,m,c,ch:self.i.fold_chunk(cur,b,self.host.device,ch,
            mon_state=m,context=c,context_tables=self.context_tables))(*initial,chunk)
        self.assertEqual(result.error.tolist(),[self.i.Error.PAYLOAD,0])
        for before,after in zip(self.jax.tree.leaves(initial),self.jax.tree.leaves(
            (result.cursor,result.beliefs,result.mon_state,result.context))):
            self.assertTrue(bool(self.j.array_equal(before[0],after[0])))
        self.assertFalse(bool(result.mon_state.boosts[1].any()))
        self.assertEqual(result.mon_state.status[1,0,:3].tolist(),[0,0,0])
        self.assertEqual(int(result.mon_state.status[1,1,0]),self.m.PARALYSIS)
        self.assertEqual(result.cursor.sequence.tolist(),[0,2])

    def test_global_payloads_and_types_refuse_without_valid_ledger(self):
        k=self.i.Kind
        for event in [self.i.PublicEvent(k.CURE_TEAM,1,0,self.i.Phase.ACTION,target_side=2),
            self.i.PublicEvent(k.CURE_TEAM,1,0,self.i.Phase.ACTION,target_side=0,target_slot=0),
            self.i.PublicEvent(k.CLEAR_ALL_BOOSTS,1,0,self.i.Phase.ACTION,target_side=0)]:
            with self.assertRaises(ValueError):event.words()
        mons=self.initial()[2]
        updated,ok=self.m.cure_team(mons,side=-1)
        self.assertFalse(bool(ok[0]))
        for x,y in zip(updated,mons):self.assertTrue(bool(self.j.array_equal(x,y)))
        with self.assertRaises(ValueError):self.m.cure_team(mons,side=0,valid=1)
        with self.assertRaises(ValueError):self.m.clear_all_boosts(mons,valid=1)
        with self.assertRaises(ValueError):self.m.cure_team(mons,side=0.0)

    def test_unnamespaced_zero_reveal_refuses_but_typed_offsource_move_keeps_fallback(self):
        fixture=self.context_fixtures.PublicContextTests
        initial=fixture.initial(self)
        generic=self.i.PublicEvent(self.i.Kind.SET_REVEAL,1,0,self.i.Phase.ACTION,
            target_side=0,target_slot=0,payload=(0,)*8)
        chunk=fixture.chunks(self,[[generic]])[0]
        result=fixture.fold(self,initial,chunk)
        self.assertEqual(result.error.tolist(),[self.i.Error.PAYLOAD])
        for before,after in zip(self.jax.tree.leaves(initial),self.jax.tree.leaves(
            (result.cursor,result.beliefs,result.mon_state,result.context))):
            self.assertTrue(bool(self.j.array_equal(before,after)))
        typed=self.i.PublicEvent(self.i.Kind.MOVE,1,0,self.i.Phase.ACTION,
            actor_side=0,actor_slot=0,target_side=1,target_slot=0,
            payload=(1,self.catalog.move_id('move00'),0,0,0,0,0,0))
        result=fixture.fold(self,initial,fixture.chunks(self,[[typed]])[0])
        self.assertEqual(result.error.tolist(),[0])
        _,summary=self.b.materialize(result.beliefs,self.host.device)
        self.assertEqual(int(summary.reason[0,1,0]),self.b.INCONSISTENT_REVEALS)


if __name__=="__main__":unittest.main()
