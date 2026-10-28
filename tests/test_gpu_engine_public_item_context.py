"""Raw item attribution composed with ordered device evidence and rollback."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class PublicItemContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataclasses
        import test_gpu_engine_public_wire as fixtures
        fixtures.PublicWireTests.setUpClass.__func__(cls)
        from gpu_engine import public_items as items, public_team
        from pokezero.belief import PublicBattleBeliefEngine
        cls.items, cls.Reference = items, PublicBattleBeliefEngine
        # Item and move constraints discriminate different dimensions, so a
        # mutation must recover exactly the eligible source variants.
        universes = dict(cls.source.universes)
        original = universes['snorlax']
        variants = tuple(dataclasses.replace(v, variant_id=f'{v.variant_id}-{item}', item=item)
                         for v in original.variants for item in ('Leftovers', 'Choice Band'))
        universes['snorlax'] = dataclasses.replace(original, variants=variants)
        cls.source = dataclasses.replace(cls.source, universes=universes)
        cls.host = cls.b.compile_source(cls.source)
        cls.catalog = cls.p.compile_moves(cls.host, cls.dex, cls.vocab)
        cls.abilities = cls.a.compile_abilities(cls.host, cls.vocab,
            ability_names=('Pressure', 'Trace', 'Immunity', 'Levitate'))
        cls.item_catalog = items.compile_items(cls.host, cls.vocab,
            item_names=('Sitrus Berry', 'Mystery Orb'))
        cls.species_catalog = public_team.compile_species(cls.host, cls.dex, cls.vocab, cls.abilities)
        cls.contexts = {n: cls.c.compile_context(cls.host, cls.abilities, cls.catalog,
            items=cls.item_catalog, item_belief_narrowing=n) for n in (False, True)}
        cls.kwargs = dict(species_catalog=cls.species_catalog, source=cls.host,
            moves=cls.catalog, abilities=cls.abilities, items=cls.item_catalog)
        cls.prefix = ('|switch|p1a: Snorlax|Snorlax, L80|100/100',
                      '|switch|p2a: Absol|Absol, L80|100/100')

    def project(self, lines, narrow=False, state=None, **changes):
        kwargs = dict(self.kwargs, context_tables=self.contexts[narrow]); kwargs.update(changes)
        return self.wire.project_lines(state or self.wire.WireState(), lines, **kwargs)

    def initial(self, narrow=False, batch=1):
        binding = self.species_catalog.device.binding
        beliefs = self.b.initialize(self.j.zeros((batch, 2, 6), self.j.int32), self.host.device)
        return (self.i.initialize(self.j.arange(1, batch+1, dtype=self.j.int32), source_binding=self.host.device.binding),
                beliefs, self.m.initialize(batch, species_binding=binding),
                self.c.initialize(beliefs, self.contexts[narrow], species_binding=binding))

    def chunks(self, projections):
        import numpy as np
        p = projections[0]
        return list(self.i.pack_chunks(np.array([1], np.int32),
            [[e for row in projections for e in row.events]], source_binding=self.host.device.binding,
            **{name: np.asarray(getattr(p, name), np.uint32) for name in
               ('species_binding', 'ability_binding', 'move_binding', 'item_binding', 'context_binding')}))

    def fold(self, initial, chunk, narrow=False):
        cur, b, m, c = initial
        return self.i.fold_chunk(cur, b, self.host.device, chunk, mon_state=m,
                                 context=c, context_tables=self.contexts[narrow])

    @staticmethod
    def carry(result):
        return result.cursor, result.beliefs, result.mon_state, result.context

    def run_lines(self, lines, narrow=False):
        state, projections = self.project(lines, narrow)
        current = self.initial(narrow)
        for chunk in self.chunks(projections):
            result = self.fold(current, chunk, narrow)
            self.assertEqual(result.error.tolist(), [0]); current = self.carry(result)
        self.assertEqual(int(result.cursor.sequence[0]), state.next_sequence-1)
        return result, projections

    def assert_reference(self, result, lines, narrow=False):
        ref = self.Reference.from_events([self.reference._public_event_from_line(x) for x in lines],
            format_id='gen3randombattle', set_source=self.source, item_belief_narrowing=narrow)
        _, summary = self.b.materialize(result.beliefs, self.host.device)
        for side, species in ((0, 'Snorlax'), (1, 'Absol')):
            expected = ref.snapshot().side(f'p{side+1}')[0]
            mask = summary.effective[0, 1-side, 0].tolist()
            rows = self.host.variants[self.host.species_id(species)]
            actual = [v.variant_id for i, v in enumerate(rows) if mask[i//32] & (1 << (i % 32))]
            self.assertEqual(actual, [v['variant_id'] for v in expected.candidate_variants])
            for field, key in (('revealed', 'revealed_item'), ('original', 'original_public_item'),
                               ('current', 'current_public_item')):
                ident = int(getattr(result.context.items, field)[0, side, 0])
                self.assertEqual(self.item_catalog.names[ident], self.reference._normalize_identifier(getattr(expected, key) or ''))
            self.assertEqual(bool(result.context.items.mutated[0, side, 0]), expected.item_mutated)
            self.assertEqual(bool(result.context.items.removed[0, side, 0]), expected.item_removed)

    def same(self, before, after):
        for x, y in zip(self.jax.tree.leaves(before), self.jax.tree.leaves(after)):
            self.assertTrue(bool(self.j.array_equal(x, y)))

    def test_mutation_recovers_item_candidates_preserves_move_and_original_mode(self):
        prefix = [*self.prefix, '|move|p1a: Snorlax|Surf|p2a: Absol',
                  '|-item|p1a: Snorlax|Leftovers']
        for narrow in (False, True):
            first, _ = self.run_lines(prefix, narrow)
            self.assertEqual(int(self.b.materialize(first.beliefs, self.host.device)[1].count[0, 1, 0]), 1)
            lines = prefix + ['|-enditem|p1a: Snorlax|Leftovers|[from] move: Knock Off',
                              '|-item|p1a: Snorlax|Choice Band|[from] move: Thief']
            result, _ = self.run_lines(lines, narrow)
            self.assert_reference(result, lines, narrow)
            self.assertEqual(int(self.b.materialize(result.beliefs, self.host.device)[1].count[0, 1, 0]), 1 if narrow else 2)

    def test_contiguous_trick_across_transport_chunks_and_interruption_match_source(self):
        activate = '|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol'
        pair = ['|-item|p1a: Snorlax|Choice Band|[from] move: Trick',
                '|-item|p2a: Absol|Leftovers|[from] move: Trick']
        # 4 arrival events + 58 blanks + activation places a raw item boundary
        # on the final word of the first chunk and its item operation in next.
        for narrow in (False, True):
            for interrupt in ([], ['|']):
                lines = [*self.prefix, *(['|']*29), activate, *interrupt, *pair]
                result, projections = self.run_lines(lines, narrow)
                self.assertGreater(len(self.chunks(projections)), 1)
                self.assert_reference(result, lines, narrow)
                self.assertEqual(bool(result.context.items.original.any()), not bool(interrupt))
                events = [e for p in projections for e in p.events]
                self.assertEqual([e.sequence for e in events], list(range(1, len(events)+1)))

    def test_raw_proc_item_and_main_consumer_keep_source_order(self):
        lines = [*self.prefix, '|move|p1a: Snorlax|Body Slam|p2a: Absol|[from] ability: Pressure|[of] p2a: Absol',
                 '|-heal|p1a: Snorlax|90/100|[from] item: Leftovers|[of] p2a: Absol',
                 '|-item|p1a: Snorlax|Choice Band|[from] item: Leftovers',
                 '|turn|1', '|', '|upkeep', '|turn|2']
        result, projections = self.run_lines(lines)
        self.assert_reference(result, lines)
        k = self.i.Kind
        self.assertEqual([e.kind for e in projections[2].events], [k.RAW_BEGIN, k.ABILITY, k.MOVE])
        self.assertEqual([e.kind for e in projections[3].events], [k.RAW_BEGIN, k.ITEM, k.CONDITION])
        self.assertEqual([e.kind for e in projections[4].events], [k.RAW_BEGIN, k.ITEM, k.ITEM])
        self.assertEqual(int(result.context.moves.uses[0, 0, 0, 0]), 2)
        self.assertEqual(int(result.cursor.turn[0]), 2)

    def test_bindings_configuration_and_bad_suffix_roll_back_every_ledger(self):
        state, projections = self.project([*self.prefix, '|-item|p1a: Snorlax|Leftovers'])
        chunk = self.chunks(projections)[0]; initial = self.initial()
        for bad in (chunk._replace(item_binding=None), chunk._replace(context_binding=None),
                    chunk._replace(item_binding=chunk.item_binding.at[0].add(1))):
            result = self.fold(initial, bad)
            self.assertEqual(result.error.tolist(), [self.i.Error.SOURCE]); self.same(initial, self.carry(result))
        # A real alternative flag has a different valid context digest.
        result = self.fold(self.initial(True), chunk, True)
        self.assertEqual(result.error.tolist(), [self.i.Error.SOURCE])
        bad = chunk._replace(words=chunk.words.at[0, 5, 9].set(2**31-1))
        result = self.jax.jit(lambda cur,b,m,c,ch: self.fold((cur,b,m,c), ch))(*initial, bad)
        self.assertNotEqual(int(result.error[0]), 0); self.same(initial, self.carry(result))
        for changes in (dict(items=None, context_tables=None), dict(context_tables=self.contexts[True])):
            with self.assertRaises(ValueError): self.project(['|'], state=state, **changes)
        changed = self.items.compile_items(self.host, self.vocab, item_names=('Aaa',))
        with self.assertRaises(ValueError): self.project(['|'], state=state, items=changed)

    def test_generic_item_reveal_refuses_but_exclusions_survive_mutation(self):
        _, projections = self.project(self.prefix)
        current = self.initial()
        first = self.fold(current, self.chunks(projections)[0]); self.assertEqual(first.error.tolist(), [0])
        state, _ = self.project(self.prefix)
        k = self.i.Kind
        def event(kind):
            return self.i.PublicEvent(kind, state.next_sequence, 0, self.i.Phase.ACTION,
                target_side=1, target_slot=0, payload=(self.host.predicate_id('item', 'Leftovers'),0,0,0,0,0,0,0))
        chunk = self.chunks(projections)[0]
        words = self.j.zeros_like(chunk.words).at[0,0].set(self.j.array(event(k.SET_REVEAL).words()))
        one = chunk._replace(words=words, valid=self.j.zeros_like(chunk.valid).at[0,0].set(True))
        bad = self.fold(self.carry(first), one)
        self.assertEqual(bad.error.tolist(), [self.i.Error.PAYLOAD]); self.same(self.carry(first), self.carry(bad))
        # A known generic exclusion is monotone and is not a replaceable reveal.
        words = words.at[0,0].set(self.j.array(event(k.SET_EXCLUDE).words()))
        allowed = self.fold(self.carry(first), one._replace(words=words))
        self.assertEqual(allowed.error.tolist(), [0])
        self.assertFalse(bool(allowed.context.base_hard[0, 0, 0].any()))

    def test_unknown_raw_and_private_lines_cannot_hide_behind_a_raw_boundary(self):
        state, _ = self.project(self.prefix)
        for line in ('|-weather|RainDance', '|request|{}', '|-activate|p1a: Snorlax|move: Trick',
                     '|-activate|p1a: Snorlax|move: Trick|[of] p3a: Missing'):
            with self.subTest(line=line), self.assertRaises(ValueError): self.project([line], state=state)
        # Belief attribution is active-first even if the wire nickname differs.
        line='|-activate|p1a: Different|move: Trick|[of] p2a: Absol'
        result,_=self.run_lines([*self.prefix,line,'|-item|p1a: Different|Choice Band|[from] move: Trick'])
        self.assert_reference(result,[*self.prefix,line,'|-item|p1a: Different|Choice Band|[from] move: Trick'])
        with self.assertRaises(ValueError): self.i.PublicEvent(self.i.Kind.RAW_BEGIN, 1, 0,
            self.i.Phase.ACTION, payload=(2,1,1,0,0,0,0,0)).words()


if __name__ == '__main__': unittest.main()
