"""Public wire HP/identity/stage facts through the real device input fold."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires isolated JAX")
class PublicMonProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_context as fixtures
        fixtures.PublicContextTests.setUpClass.__func__(cls)
        from gpu_engine import public_mon_protocol as wire
        from gpu_engine.public_context_protocol import PublicAddress,PublicAddresses
        from pokezero import showdown as reference
        cls.context_fixtures,cls.wire,cls.Address,cls.Addresses,cls.reference=fixtures,wire,PublicAddress,PublicAddresses,reference

    def registry(self,first=0):
        a,b,bench=self.Address(0,0,1),self.Address(1,0,1),self.Address(0,1,2)
        return self.Addresses({0:a if first==0 else bench,1:b},
            {"p1a: Snorlax":a,"p1: Snorlax":a,"p2a: Gardevoir":b,"p1a: Pikachu":bench},
            {a:"Snorlax",b:"Gardevoir",bench:"Pikachu"})

    def project(self,line,sequence,*,registry=None,baton=False):
        registry=registry or self.registry()
        active_idents={0:"p1a: Snorlax" if registry.active[0].slot==0 else "p1a: Pikachu",1:"p2a: Gardevoir"}
        return self.wire.project_line(line,sequence=sequence,turn=0,addresses=registry,active_idents=active_idents,
            species_ids={"snorlax":10,"gardevoir":11,"pikachu":12},source=self.host,baton_pass=baton)

    def fold(self,projections):
        events=[event for p in projections for event in p.events]
        for p in projections:self.assertFalse(p.full_line_consumers_complete)
        beliefs=self.b.initialize(self.j.zeros((1,2,6),self.j.int32),self.host.device)
        ctx=self.c.initialize(beliefs,self.context_tables);mons=self.m.initialize(1)
        cursor=self.i.initialize(self.j.array([1],self.j.int32),source_binding=self.host.device.binding)
        chunk=self.context_fixtures.PublicContextTests.chunks(self,[events])[0]
        result=self.i.fold_chunk(cursor,beliefs,self.host.device,chunk,mon_state=mons,context=ctx,context_tables=self.context_tables)
        self.assertEqual(result.error.tolist(),[0]);return result

    def test_arrivals_conditions_boosts_copy_and_faint_match_actual_parser(self):
        lines=["|switch|p1a: Snorlax|Snorlax, L80, M|100/100", "|switch|p2a: Gardevoir|Gardevoir, L80, F|100/100",
            "|-boost|p1a: Snorlax|atk|2", "|-setboost|p1a: Snorlax|spe|-3",
            "|-damage|p1a: Snorlax|75/100 par", "|-clearpositiveboost|p1a: Snorlax",
            "|-copyboost|p2a: Gardevoir|p1a: Snorlax", "|-restoreboost|p1a: Snorlax", "|faint|p2a: Gardevoir"]
        result=self.fold([self.project(line,n+1) for n,line in enumerate(lines)])
        parser=self.reference.parse_showdown_replay(lines,battle_id="fixture")
        for side in (0,1):
            self.assertEqual(result.mon_state.boosts[0,side].tolist(),[parser.boosts[f"p{side+1}"].get(k,0) for k in self.m.STATS])
            condition=parser.public_active[f"p{side+1}"].condition
            hp,maximum,status,fainted=self.wire.condition_words(condition)
            self.assertEqual(int(result.mon_state.hp_numerator[0,side,0]),hp)
            self.assertEqual(int(result.mon_state.hp_denominator[0,side,0]),maximum)
            self.assertEqual(int(result.mon_state.status[0,side,0]),status)
            self.assertEqual(bool(result.mon_state.fainted[0,side,0]),bool(fainted))
        self.assertEqual(result.mon_state.gender[0,:,0].tolist(),[1,2])

    def test_benched_cure_targets_registered_bench_not_current_active(self):
        lines=["|switch|p1a: Snorlax|Snorlax, L80|100/100 par", "|switch|p1a: Pikachu|Pikachu, L80|100/100 tox",
            "|-curestatus|p1: Snorlax|par|[silent]"]
        projections=[self.project(lines[0],1),self.project(lines[1],2,registry=self.registry(1)),
                     self.project(lines[2],3,registry=self.registry(1))]
        result=self.fold(projections)
        parser=self.reference.parse_showdown_replay(lines,battle_id="fixture")
        by_species={x.species:x for x in parser.public_revealed["p1"]}
        self.assertEqual(by_species["Snorlax"].condition,"100/100")
        self.assertEqual(by_species["Pikachu"].condition,"100/100 tox")
        self.assertEqual(int(result.mon_state.status[0,0,0]),self.m.NONE)
        self.assertEqual(int(result.mon_state.status[0,0,1]),self.m.TOXIC)

    def test_baton_tag_or_public_pending_fact_carries_boosts_and_level100_is_known(self):
        first="|switch|p1a: Snorlax|Snorlax|100/100"
        boost="|-boost|p1a: Snorlax|atk|2"
        for tagged in (False,True):
            line="|switch|p1a: Pikachu|Pikachu, L80|100/100"+("|[from] Baton Pass" if tagged else "")
            projection=self.project(line,3,registry=self.registry(1),baton=not tagged)
            result=self.fold([self.project(first,1),self.project(boost,2),projection])
            self.assertEqual(int(result.mon_state.level[0,0,0]),100)
            self.assertEqual(int(result.mon_state.boosts[0,0,0]),2)
            self.assertEqual(projection.events[0].kind,self.i.Kind.BATON_PASS)

    def test_condition_requires_canonical_active_ident_even_when_aliases_are_known(self):
        registry=self.registry()
        aliases=dict(registry.known_idents);aliases['p1a: Old']=registry.active[0]
        registry=self.Addresses(registry.active,aliases,registry.public_species)
        prefix='|switch|p1a: Snorlax|Snorlax, L80|100/100'
        for line in ['|-damage|p1: Snorlax|50/100','|-damage|p1a: Old|50/100']:
            parser=self.reference.parse_showdown_replay([prefix,line],battle_id='fixture')
            self.assertEqual(parser.public_active['p1'].condition,'100/100')
            with self.assertRaises(ValueError):self.project(line,2,registry=registry)
        # The source's positionless-cure species lookup also accepts the
        # current active mon; it is an exception to active-ident matching.
        cure=self.project('|-curestatus|p1: Snorlax|par',1,registry=registry).events[0]
        self.assertEqual(cure.kind,self.i.Kind.STATUS)
        self.assertEqual((cure.target_side,cure.target_slot,cure.payload[:2]),(0,0,(1,0)))

    def test_arrival_suffix_validation_and_drag_ignore_stale_pending_baton(self):
        for kind in ('drag','replace'):
            line=f'|{kind}|p1a: Pikachu|Pikachu, L80|100/100'
            projection=self.project(line,3,registry=self.registry(1),baton=True)
            self.assertEqual(projection.events[0].kind,getattr(self.i.Kind,kind.upper()))
            result=self.fold([self.project('|switch|p1a: Snorlax|Snorlax, L80|100/100',1),
                self.project('|-boost|p1a: Snorlax|atk|2',2),projection])
            self.assertEqual(int(result.mon_state.boosts[0,0,0]),0)
            for suffix in ('|[from] Baton Pass','|unexpected'):
                with self.assertRaises(ValueError):self.project(line+suffix,3,registry=self.registry(1))
        with self.assertRaises(ValueError):self.project('|switch|p1a: Snorlax|Snorlax, L80|100/100|unexpected',1)
        for suffix in (' ','\n'):
            with self.assertRaises(ValueError):self.project('|switch|p1a: Snorlax|Snorlax, L80|100/100'+suffix,1)

    def test_canonical_turn_and_species_bound_bench_aliases(self):
        for line in ('|turn|01','|turn|0','|turn|+1','|turn|1|extra'):
            with self.assertRaises(ValueError):self.project(line,1)
        self.assertEqual(self.project('|turn|1',1).events[0].turn,1)
        registry=self.registry(1);bench=self.Address(0,0,1)
        aliases=dict(registry.known_idents);aliases['p1: OldNickname']=bench
        registry=self.Addresses(registry.active,aliases,registry.public_species)
        with self.assertRaises(ValueError):self.project('|-curestatus|p1: OldNickname|par|[silent]',1,registry=registry)
        aliases['p1: Unown']=bench;species=dict(registry.public_species);species[bench]='Unown-Z'
        registry=self.Addresses(registry.active,aliases,species)
        self.assertEqual(self.project('|-curestatus|p1: Unown|par|[silent]',1,registry=registry).events[0].target_slot,0)

    def test_team_cure_and_global_boost_clear_are_ordered_field_consumers(self):
        lines=['|switch|p1a: Snorlax|Snorlax, L80|100/100 par',
            '|switch|p2a: Gardevoir|Gardevoir, L80|100/100 tox',
            '|-boost|p1a: Snorlax|atk|2','|-boost|p2a: Gardevoir|spe|-3',
            '|-cureteam|p1a: Snorlax','|-clearallboost']
        projections=[self.project(line,n+1) for n,line in enumerate(lines)]
        result=self.fold(projections);parser=self.reference.parse_showdown_replay(lines,battle_id='fixture')
        self.assertEqual(parser.public_active['p1'].condition,'100/100')
        self.assertEqual(parser.public_active['p2'].condition,'100/100 tox')
        self.assertEqual(result.mon_state.status[0,:,0].tolist(),[self.m.NONE,self.m.TOXIC])
        self.assertFalse(bool(result.mon_state.boosts.any()))
        self.assertEqual(parser.boosts,{'p1':{},'p2':{}})

    def test_explicit_refusal_keeps_unported_and_malformed_lines_visible(self):
        for line in ["|-clearallboost|unexpected", "|-cureteam|p3a: Unknown",
            "|-sethp|p1a: Snorlax|50/100|p2a: Gardevoir|50/100",
            "|faint|p1a: Snorlax|unexpected",
            "|-damage|p1a: Unknown|50/100", "|-curestatus|p1: Unknown|par",
            "|-boost|p1a: Snorlax|unknown|2", "|-boost|p1a: Snorlax|atk|2147483648",
            "|switch|p1a: Snorlax|Unknown, L80|100/100", "|-status|p1a: Snorlax|unknown"]:
            with self.subTest(line=line),self.assertRaises(ValueError):self.project(line,1)
        for condition in [None, "", "1.5/100", "101/100", "1/100 fnt", "100/100 par extra"]:
            with self.assertRaises(ValueError):self.wire.condition_words(condition)
        with self.assertRaises(ValueError):self.project("|-damage|p1a: Snorlax|50/100",True)


if __name__=="__main__":unittest.main()
