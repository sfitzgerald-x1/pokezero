"""Public encounter producer composed with the actual typed device consumers."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class PublicWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_context as fixtures
        fixtures.PublicContextTests.setUpClass.__func__(cls)
        from gpu_engine import public_wire as wire
        from pokezero import showdown
        from pokezero.dex import MoveInfo, ShowdownDex
        cls.wire, cls.reference, cls.context_fixtures = wire, showdown, fixtures
        moves = dict(cls.dex.moves)
        moves['batonpass'] = MoveInfo('batonpass', 'Baton Pass', 'Normal', 'Status', 'Status',
            0, 100, 0, False, False, False, None, {}, 'self', False, pp=40)
        cls.catalog = cls.p.compile_moves(cls.host, ShowdownDex(moves, {}, {}), cls.vocab)
        cls.context_tables = cls.c.compile_context(cls.host, cls.abilities, cls.catalog)
        from gpu_engine.public_team import compile_species
        cls.species_catalog=compile_species(cls.host,cls.dex,cls.vocab,cls.abilities,
            extra_species=('Pikachu','Unown-Z',*(f'Species{i}' for i in range(7))))
        cls.kwargs = dict(species_catalog=cls.species_catalog,
            source=cls.host, moves=cls.catalog, abilities=cls.abilities)

    def project(self, lines, initial=None):
        return self.wire.project_lines(initial or self.wire.WireState(), lines, **self.kwargs)

    def fold(self, projections):
        beliefs = self.b.initialize(self.j.zeros((1,2,6), self.j.int32), self.host.device)
        import numpy as np
        binding=np.asarray(projections[0].species_binding,np.uint32)
        context = self.c.initialize(beliefs, self.context_tables,species_binding=binding)
        mons = self.m.initialize(1,species_binding=binding)
        cursor = self.i.initialize(self.j.array([1], self.j.int32), source_binding=self.host.device.binding)
        events = [event for p in projections for event in p.events]
        for chunk in self.i.pack_chunks(np.asarray([1],np.int32),[events],source_binding=self.host.device.binding,
            species_binding=binding,ability_binding=np.asarray(projections[0].ability_binding,np.uint32),
            move_binding=np.asarray(projections[0].move_binding,np.uint32)):
            result = self.i.fold_chunk(cursor, beliefs, self.host.device, chunk, mon_state=mons,
                context=context, context_tables=self.context_tables)
            self.assertEqual(result.error.tolist(), [0])
            cursor, beliefs, mons, context = result.cursor, result.beliefs, result.mon_state, result.context
        return result

    def test_real_prefix_assigns_public_order_reuses_returner_and_matches_parser(self):
        lines=['|switch|p2a: Absol|Absol, L80|100/100',
            '|switch|p1a: Snorlax|Snorlax, L80|100/100 par', '|turn|1',
            '|-ability|p2a: Absol|Pressure', '|move|p1a: Snorlax|Body Slam|p2a: Absol',
            '|-damage|p2a: Absol|70/100', '|', '|upkeep',
            '|switch|p1a: Pikachu|Pikachu, L80|100/100', '|turn|2',
            '|-curestatus|p1: Snorlax|par|[silent]',
            '|switch|p1a: Snorlax|Snorlax, L80|100/100', '|faint|p2a: Absol']
        state, projections = self.project(lines)
        result = self.fold(projections)
        replay = self.reference.parse_showdown_replay(lines, battle_id='fixture')
        self.assertEqual([[r.species for r in rows] for rows in state.encounters],
            [[r.species for r in replay.public_revealed[f'p{s+1}']] for s in (0,1)])
        self.assertEqual(state.active, (0,0))
        self.assertEqual(result.mon_state.encounter[0,0,:2].tolist(), [1,2])
        self.assertEqual(state.next_sequence, len(lines)+1)
        self.assertEqual(state.turn, 2)
        self.assertEqual(result.mon_state.status[0,0,:2].tolist(), [0,0])
        self.assertTrue(bool(result.mon_state.fainted[0,1,0]))
        self.assertEqual(int(result.context.moves.uses[0,0,0,0]),2)
        self.assertTrue(all(not p.full_line_consumers_complete for p in projections))
        self.assertEqual([p.raw_line for p in projections], lines)

    def test_same_species_distinct_public_idents_are_never_species_joined(self):
        lines=['|switch|p1a: Left|Snorlax, L80|100/100',
               '|switch|p1a: Right|Snorlax, L80|100/100',
               '|switch|p1a: Left|Snorlax, L80|90/100']
        state, projections = self.project(lines)
        result = self.fold(projections)
        self.assertEqual(len(state.encounters[0]),2)
        self.assertEqual(result.mon_state.encounter[0,0,:2].tolist(),[1,2])
        self.assertEqual(result.mon_state.hp_numerator[0,0,:2].tolist(),[90,100])
        self.assertEqual(state.active,(0,-1))
        with self.assertRaises(ValueError):self.project(['|-curestatus|p1: Snorlax|par'],state)

    def test_multi_consumer_lines_have_unique_ordered_sequences_across_chunk_boundary(self):
        lines=['|switch|p1a: Snorlax|Snorlax, L80|100/100',
               '|switch|p2a: Gardevoir|Gardevoir, L80|100/100']+['|']*61
        lines+=['|-heal|p1a: Snorlax|90/100|[from] ability: Immunity',
                '|-boost|p1a: Snorlax|atk|2', '|-transform|p2a: Gardevoir|p1a: Snorlax']
        state, projections = self.project(lines)
        events=[e for p in projections for e in p.events]
        self.assertEqual([int(e.sequence) for e in events], list(range(1,len(events)+1)))
        self.assertEqual([int(e.kind) for e in projections[63].events],[self.i.Kind.ABILITY,self.i.Kind.CONDITION])
        self.assertEqual([int(e.kind) for e in projections[-1].events],[self.i.Kind.COPY_BOOST,self.i.Kind.TRANSFORM])
        result=self.fold(projections)
        self.assertEqual(int(result.context.abilities.original[0,0,0]),self.abilities.ability_id('Immunity'))
        self.assertEqual(result.mon_state.boosts[0,1].tolist(),[2,0,0,0,0,0,0])
        self.assertEqual(int(result.cursor.sequence[0]),state.next_sequence-1)

    def test_baton_pending_is_public_side_specific_consumed_and_other_move_clears(self):
        prefix=['|switch|p1a: Snorlax|Snorlax, L80|100/100',
                '|switch|p2a: Absol|Absol, L80|100/100', '|-boost|p1a: Snorlax|atk|2',
                '|move|p1a: Snorlax|Baton Pass|p1a: Snorlax']
        state, projections=self.project(prefix)
        self.assertEqual(state.pending_baton,(True,False))
        for kind, expected in [('switch',2),('drag',0),('replace',0)]:
            next_state, extra=self.project([f'|{kind}|p1a: Pikachu|Pikachu, L80|100/100'],state)
            self.assertFalse(any(next_state.pending_baton))
            result=self.fold(projections+extra)
            self.assertEqual(int(result.mon_state.boosts[0,0,0]),expected)
        next_state, extra=self.project(['|move|p1a: Snorlax|Protect|p1a: Snorlax',
            '|switch|p1a: Pikachu|Pikachu, L80|100/100'],state)
        self.assertFalse(any(next_state.pending_baton))
        self.assertEqual(int(self.fold(projections+extra).mon_state.boosts[0,0,0]),0)

    def test_refusal_rolls_back_whole_host_transaction_and_cannot_hide_unknown_lines(self):
        initial=self.wire.WireState()
        arrival='|switch|p1a: Snorlax|Snorlax, L80|100/100'
        for bad in ['|-weather|RainDance','|-damage|p1a: Wrong|50/100',
            '|switch|p1a: Snorlax|Unknown, L80|100/100', '|turn|01',
            '|switch|p1a: Snorlax|Snorlax, L80|100/100|unexpected']:
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.project([arrival,bad],initial)
            self.assertEqual(initial,self.wire.WireState())
        state,_=self.project([arrival])
        with self.assertRaises(ValueError):self.project(['|replace|p1a: Snorlax|Pikachu, L80|100/100'],state)
        self.assertEqual(state.encounters[0][0].species,'Snorlax')
        with self.assertRaises(TypeError):state.addresses().known_idents['private']=state.encounters[0][0].address

    def test_public_capacity_is_bounded_without_truncation_or_private_refill(self):
        state,_=self.project([f'|switch|p1a: Name{i}|Species{i}, L80|100/100' for i in range(6)])
        with self.assertRaises(ValueError):self.project(['|switch|p1a: Name6|Species6, L80|100/100'],state)
        self.assertEqual(len(state.encounters[0]),6)
        next_state,_=self.project(['|switch|p1a: Name0|Species0, L80|100/100'],state)
        self.assertEqual(next_state.active,(0,-1))

    def test_failed_baton_clears_at_new_turn_and_fainted_donor_is_not_live(self):
        prefix=['|switch|p1a: Snorlax|Snorlax, L80|100/100',
                '|switch|p2a: Absol|Absol, L80|100/100', '|turn|1',
                '|-boost|p1a: Snorlax|atk|2', '|move|p1a: Snorlax|Baton Pass|p1a: Snorlax',
                '|', '|upkeep', '|turn|2', '|switch|p1a: Pikachu|Pikachu, L80|100/100']
        state, projections=self.project(prefix)
        replay=self.reference.parse_showdown_replay(prefix,battle_id='fixture')
        result=self.fold(projections)
        self.assertEqual(replay.boosts['p1'],{})
        self.assertFalse(replay.arrived_by_baton_pass['p1'])
        self.assertEqual(int(result.mon_state.boosts[0,0,0]),0)
        self.assertFalse(bool(result.mon_state.arrived_by_baton[0,0]))
        state,projections=self.project(['|switch|p1a: Snorlax|Snorlax, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100','|-ability|p2a: Absol|Pressure','|faint|p2a: Absol'])
        self.assertNotIn(1,state.addresses().active)
        self.assertEqual(state.active,(0,0))  # parser mon pointer survives, live donor does not
        self.assertEqual(len(state.encounters[1]),1)
        with self.assertRaises(ValueError):self.project(['|-transform|p1a: Snorlax|p2a: Absol'],state)
        next_state,_=self.project(['|switch|p2a: Gardevoir|Gardevoir, L80|100/100'],state)
        self.assertIn(1,next_state.addresses().active)

    def test_raw_proc_precedes_move_charge_and_transform_copy_without_duplicate_reveals(self):
        lines=['|switch|p1a: Snorlax|Snorlax, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol|[from] ability: Pressure|[of] p2a: Absol',
            '|-transform|p1a: Snorlax|p2a: Absol|[from] ability: Pressure|[of] p2a: Absol']
        _,projections=self.project(lines)
        self.assertEqual([e.kind for e in projections[2].events],[self.i.Kind.ABILITY,self.i.Kind.MOVE])
        self.assertEqual([e.kind for e in projections[3].events],
            [self.i.Kind.ABILITY,self.i.Kind.COPY_BOOST,self.i.Kind.TRANSFORM])
        result=self.fold(projections)
        self.assertEqual(int(result.context.moves.uses[0,0,0,0]),2)
        self.assertEqual(int(result.context.abilities.running[0,0,0]),self.abilities.ability_id('Pressure'))
        from pokezero.belief import PublicBattleBeliefEngine
        ref=PublicBattleBeliefEngine.from_events([self.reference._public_event_from_line(x) for x in lines],
            format_id='gen3randombattle',set_source=self.source)
        self.assertEqual(dict(ref.snapshot().side('p1')[0].move_uses),{'bodyslam':2})
        with self.assertRaises(ValueError):self.project([
            '|request|{"note":"[from] ability: Pressure|[of] p2a: Absol"}'])

    def test_host_continuation_cannot_change_species_namespace(self):
        from gpu_engine.public_team import compile_species
        state,_=self.project(['|switch|p1a: Snorlax|Snorlax, L80|100/100'])
        changed=compile_species(self.host,self.dex,self.vocab,self.abilities,extra_species=('Aaa',))
        kwargs=dict(self.kwargs,species_catalog=changed)
        with self.assertRaises(ValueError):self.wire.project_line(state,'|',**kwargs)
        changed_abilities=self.a.compile_abilities(self.host,self.vocab,ability_names=('Aaa','Pressure','Trace'))
        with self.assertRaises(ValueError):self.wire.project_line(state,'|',**dict(self.kwargs,abilities=changed_abilities))
        from pokezero.dex import ShowdownDex
        changed_moves=self.p.compile_moves(self.host,ShowdownDex({}, {}, {}),self.vocab)
        with self.assertRaises(ValueError):self.wire.project_line(state,'|',**dict(self.kwargs,moves=changed_moves))


if __name__=='__main__':unittest.main()
