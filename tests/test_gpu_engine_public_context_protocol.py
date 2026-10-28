"""Real wire lines through typed context facts and the ordered device fold."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires isolated JAX")
class PublicContextProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_context as fixtures
        fixtures.PublicContextTests.setUpClass.__func__(cls)
        from gpu_engine import public_context_protocol as wire
        cls.context_fixtures,cls.wire=fixtures,wire
        cls.abilities=cls.a.compile_abilities(cls.host,cls.vocab,
            ability_names=["Pressure","Trace","Immunity","Levitate","Water Absorb","Soundproof"])
        cls.context_tables=cls.c.compile_context(cls.host,cls.abilities,cls.catalog)

    def addresses(self):
        w=self.wire
        return w.PublicAddresses({0:w.PublicAddress(0,0,1),1:w.PublicAddress(1,0,1)}, {},
            {w.PublicAddress(0,0,1):"Snorlax",w.PublicAddress(1,0,1):"Gardevoir"})

    def run_lines(self,lines):
        events=[];projections=[]
        for line in lines:
            projection=self.wire.project_line(line,sequence=len(events)+1,turn=0,
                addresses=self.addresses(),moves=self.catalog,abilities=self.abilities)
            self.assertEqual(projection.raw_line,line)
            self.assertFalse(projection.full_line_consumers_complete)
            events.extend(projection.events);projections.append(projection)
        fixture=self.context_fixtures.PublicContextTests
        initial=fixture.initial(self,opponent="Gardevoir")
        chunk=fixture.chunks(self,[events])[0]
        result=fixture.fold(self,initial,chunk)
        self.assertEqual(result.error.tolist(),[0])
        return result,projections

    def test_raw_trace_and_called_moves_match_actual_belief_and_pp(self):
        lines=["|-ability|p1a: Snorlax|Pressure", "|-ability|p2a: Gardevoir|Trace",
            "|-ability|p2a: Gardevoir|Pressure|Trace|[from] ability: Trace|[of] p1a: Snorlax",
            "|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
            "|move|p1a: Snorlax|Surf|p2a: Gardevoir|[from] move: Sleep Talk",
            "|move|p1a: Snorlax|Body Slam|p2a: Gardevoir"]
        result,_=self.run_lines(lines)
        reference=self.fixtures.PublicMovesTests.reference(self,lines,opponent="Gardevoir")
        actor=reference.snapshot().side("p1")[0]
        state=result.context.moves
        ids=state.move_ids[0,0,0].tolist();uses=state.uses[0,0,0].tolist()
        self.assertEqual({self.catalog.names[i]:u for i,u in zip(ids,uses) if i and u},dict(actor.move_uses))
        self.assertEqual([self.catalog.names[i] for i in ids if i],[x.lower().replace(" ","") for x in actor.revealed_moves])
        self.assertEqual(uses[:3],[1,0,2])

    def test_trace_lexical_discriminators_can_emit_two_facts_for_one_line(self):
        from pokezero.showdown import parse_showdown_replay
        cases=["|-ability|p2a: Gardevoir|Pressure|Trace|[from]ability: Trace|[of] p1a: Snorlax",
            "|ability|p2a: Gardevoir|Pressure|Trace|[from] ability: Trace|[of] p1a: Snorlax",
            "|-ability|p2a: Gardevoir|Pressure|Trace|[from] ability: Trace"]
        for index,line in enumerate(cases):
            with self.subTest(line=line):
                result,projections=self.run_lines([line])
                engine=self.fixtures.PublicMovesTests.reference(self,[line],opponent="Gardevoir")
                belief=engine.snapshot().side("p2")[0]
                self.assertEqual(self.abilities.names[int(result.context.abilities.original[0,1,0])],(belief.revealed_ability or "").lower())
                self.assertEqual(self.abilities.names[int(result.context.abilities.running[0,1,0])],engine._running_ability.get(belief.key,"").lower())
                parser=parse_showdown_replay(["|switch|p2a: Gardevoir|Gardevoir, L80|100/100",line],battle_id="fixture")
                self.assertEqual(self.abilities.names[int(result.context.abilities.traced[0,1])],parser.traced_ability["p2"] or "")
                self.assertEqual(len(projections[0].events),2 if index==0 else 1)

    def test_proc_attribution_uses_actual_source_holder_and_transform_current_context(self):
        w=self.wire;k=self.i.Kind
        for line,side in [("|-heal|p1a: Snorlax|100/100|[from] ability: Water Absorb|[of] p2a: Gardevoir",0),
            ("|-activate|p2a: Gardevoir|ability: Soundproof|[of] p1a: Snorlax",1),
            ("|-damage|p1a: Snorlax|90/100|[from] ability: Pressure|[of] p2a: Gardevoir",1)]:
            projection=w.project_line(line,sequence=1,turn=0,addresses=self.addresses(),moves=self.catalog,abilities=self.abilities)
            self.assertEqual(projection.events[0].kind,k.ABILITY)
            self.assertEqual(projection.events[0].target_side,side)
            self.assertEqual(projection.events[0].payload[2],0)
        lines=["|-ability|p2a: Gardevoir|Pressure", "|-transform|p1a: Snorlax|p2a: Gardevoir",
               "|move|p1a: Snorlax|Surf|p2a: Gardevoir"]
        result,_=self.run_lines(lines)
        self.assertTrue(bool(result.context.abilities.transformed[0,0,0]))
        self.assertEqual(int(result.context.abilities.running[0,0,0]),self.abilities.ability_id("Pressure"))
        self.assertFalse(bool(result.context.moves.move_ids[0,0,0].any()))

    def test_source_active_preference_and_explicit_unhandled_identity_refusals(self):
        w=self.wire
        registry=w.PublicAddresses({0:w.PublicAddress(0,0,1)}, {"p1a: Old":w.PublicAddress(0,1,2)})
        self.assertEqual(registry.resolve("p1a: Old"),w.PublicAddress(0,0,1))
        for line in ["|upkeep","|request|{}","|move|p3a: Unknown|Surf|p2a: Foe",
                     "|move|p1a: Snorlax|Not A Frozen Move|p2a: Foe",
                     "|move|p1a: Snorlax|Surf|p3a: Unknown",
                     "|-transform|p1a: Snorlax|p2a: Absol"]:
            with self.subTest(line=line),self.assertRaises(ValueError):
                w.project_line(line,sequence=1,turn=0,addresses=self.addresses(),moves=self.catalog,abilities=self.abilities)
        with self.assertRaises(ValueError):w.PublicAddress(0,0,True)
        with self.assertRaises(ValueError):w.PublicAddresses({},{}).resolve("p1a: Unknown")
        bench=w.PublicAddress(1,0,1)
        registry=w.PublicAddresses({0:w.PublicAddress(0,0,1)}, {"p2a: Gardevoir":bench},{bench:"Gardevoir"})
        self.assertEqual(registry.resolve("p2a: Gardevoir"),bench)
        with self.assertRaises(ValueError):w.project_line("|-transform|p1a: Snorlax|p2a: Gardevoir",
            sequence=1,turn=0,addresses=registry,moves=self.catalog,abilities=self.abilities)
        cleared=w.PublicAddresses({0:w.PublicAddress(0,0,1),1:None},registry.known_idents,registry.public_species)
        with self.assertRaises(ValueError):cleared.transform_target("p2a: Gardevoir")
        with self.assertRaises(ValueError):w.project_line("|move|p1a: Snorlax|Surf|p2a: Foe",
            sequence=True,turn=0,addresses=self.addresses(),moves=self.catalog,abilities=self.abilities)

    def test_transform_species_guard_keeps_nidoran_distinct_and_allows_deoxys_forme(self):
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        w=self.wire;target=w.PublicAddress(1,0,1)
        for named,tracked,accepted in [("Deoxys","Deoxys-Defense",True),("Nidoran-M","Nidoran-F",False),
                                       ("Absol","Gardevoir",False),("Gardevoir","Gardevoir",True)]:
            registry=w.PublicAddresses({0:w.PublicAddress(0,0,1),1:target},{},{target:tracked})
            line=f"|-transform|p1a: Snorlax|p2a: {named}"
            reference=PublicBattleBeliefEngine.from_events([_public_event_from_line(x) for x in
                ["|switch|p1a: Snorlax|Snorlax, L80|100/100",f"|switch|p2a: {named}|{tracked}, L80|100/100",line]],
                format_id="gen3randombattle",set_source=self.source)
            actual=reference.snapshot().side("p1")[0].transform_species
            if accepted:
                projection=w.project_line(line,sequence=1,turn=0,addresses=registry,moves=self.catalog,abilities=self.abilities)
                self.assertEqual(projection.events[0].target_side,1)
                self.assertEqual(actual,tracked)
            else:
                self.assertEqual(actual,named)
                with self.assertRaises(ValueError):w.project_line(line,sequence=1,turn=0,addresses=registry,moves=self.catalog,abilities=self.abilities)


if __name__=="__main__":unittest.main()
