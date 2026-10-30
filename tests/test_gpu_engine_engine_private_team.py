"""Synthetic owner arrays only: no engine step, model, pool or game execution."""
import copy
from dataclasses import replace
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class EnginePrivateTeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from gpu_engine import engine_private_team as e, private_team as p
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.dex import ShowdownDex, SpeciesInfo
        cls.jax, cls.j, cls.np, cls.e, cls.p = jax, jnp, np, e, p
        cls.metadata = {'schema': 'authored-static-engine-metadata', 'vocabulary': {
            'species': ['_none', 'pikachu', 'gardevoir', 'hooh'],
            'abilities': ['_none', 'static', 'trace', 'levitate'],
            'items': ['_none', 'lightball', 'leftovers']}}
        cls.dex = ShowdownDex({}, {
            'pikachu': SpeciesInfo('pikachu', 'Pikachu', ('Electric',),
                dict(hp=35, atk=55, **{'def': 40}, spa=50, spd=50, spe=90)),
            'gardevoir': SpeciesInfo('gardevoir', 'Gardevoir', ('Psychic',),
                dict(hp=68, atk=65, **{'def': 65}, spa=125, spd=115, spe=80)),
            'hooh': SpeciesInfo('hooh', 'Ho-Oh', ('Fire', 'Flying'),
                dict(hp=106, atk=130, **{'def': 90}, spa=110, spd=154, spe=90))}, {})
        labels = {'pokemon:self', 'species:pikachu', 'species:gardevoir',
            'species:ho-oh', 'type:fire', 'type:flying',
            'type:electric', 'type:psychic', 'belief:possible_item:light_ball',
            'belief:possible_item:leftovers'}
        labels |= {'status:'+x for x in ('none', 'par', 'brn', 'psn', 'tox', 'slp', 'frz', 'fnt')}
        labels |= {'belief:possible_ability:'+x for x in ('static', 'trace', 'levitate')}
        cls.vocab = CategoryVocabulary(tuple(sorted(labels)), aliases={
            'belief:possible_item:lightball': 'belief:possible_item:light_ball'})

    def tables(self, exposure=None, metadata=None, vocabulary=None, checkpoint='a'*64, dex=None):
        return self.e.compile_engine_private_team(metadata or self.metadata,
            vocabulary=vocabulary or self.vocab, dex=dex or self.dex, checkpoint_sha256=checkpoint,
            exposure=exposure or self.e.RequestExposure('current', 'physical', True, True))

    def state(self):
        from gpu_engine.state import BattleState
        j = self.j
        # Every unused BattleState field is None: the adapter may not read it.
        data = {field: None for field in BattleState._fields}
        base = j.zeros((2, 2, 6), j.int32)
        for field in self.e.OwnedTeam._fields[2:]:
            data[field] = (j.zeros((2, 2, 6, 6), j.int32) if field == 'stats'
                else j.zeros((2, 2), j.int32) if field == 'active' else base)
        for row, owner in enumerate((0, 1)):
            for field, values in {
                'species': [1, 2, 1, 0, 0, 0], 'level': [80, 79, 81, 0, 0, 0],
                'gender': [1, -1, 2, 0, 0, 0], 'hp': [99, 0, 100, 0, 0, 0],
                'max_hp': [201, 237, 220, 0, 0, 0], 'status': [1, 5, 4, 0, 0, 0],
                'item': [1, 0, 2, 0, 0, 0], 'ability': [3, 1, 1, 0, 0, 0],
                'base_ability': [1, 2, 1, 0, 0, 0]}.items():
                data[field] = data[field].at[row, owner].set(j.array(values, j.int32))
            data['stats'] = data['stats'].at[row, owner, :3].set(j.array([
                [201, 144, 110, 170, 88, 215], [237, 100, 101, 220, 190, 130],
                [220, 145, 111, 171, 89, 216]], j.int32))
        return BattleState(**data)

    def owned(self, state=None, binding=None, owners=None):
        j = self.j
        return self.e.select_owned_team(state or self.state(),
            j.array([0, 1], j.int32) if owners is None else owners,
            engine_binding=j.broadcast_to(self.e.engine_metadata_binding(self.metadata), (2, 8))
                if binding is None else binding)

    def requests(self, exposure):
        packets = []
        for owner in (0, 1):
            rows = []
            for i, (name, level, condition, sex) in enumerate((
                    ('Pikachu', 80, '99/201 par', 'M'),
                    ('Gardevoir', 79, '0 fnt', ''),
                    ('Pikachu', 81, '100/220 tox', 'F'))):
                row = dict(ident=f'p{owner+1}: mon{i}', details=f'{name}, L{level}'+(
                    ', '+sex if sex and exposure.gender == 'physical' else ''),
                    condition=condition, active=i == 0)
                if exposure.ability != 'hidden':
                    row['ability' if exposure.ability == 'current' else 'baseAbility'] = (
                        ('Levitate', 'Static', 'Static') if exposure.ability == 'current'
                        else ('Static', 'Trace', 'Static'))[i]
                if exposure.item:
                    row['item'] = ('Light Ball', '', 'Leftovers')[i]
                if exposure.stats:
                    values = ([144, 110, 170, 88, 215], [100, 101, 220, 190, 130],
                        [145, 111, 171, 89, 216])[i]
                    row['stats'] = dict(zip(('atk', 'def', 'spa', 'spd', 'spe'), values))
                rows.append(row)
            packets.append(dict(side=dict(id=f'p{owner+1}', pokemon=rows)))
        return packets

    def assert_source(self, exposure):
        tables = self.tables(exposure)
        result = self.e.team_facts(self.owned(), tables, expected_binding=tables.binding)
        self.assertEqual(result.admitted.tolist(), [True, True])
        parsed = [self.p.parse_team_facts(raw, owner_side=side, vocabulary=self.vocab, dex=self.dex)
            for side, raw in enumerate(self.requests(exposure))]
        expected = self.jax.tree.map(lambda *x: self.j.stack(x), *parsed)
        for name in expected._fields:
            self.assertEqual(getattr(result.facts, name).dtype, getattr(expected, name).dtype, name)
            self.np.testing.assert_array_equal(getattr(result.facts, name), getattr(expected, name), err_msg=name)
        observed = self.p.project_team(result.facts, observer_side=self.j.array([0, 1], self.j.int32),
            expected_vocabulary_binding=tables.vocabulary_binding)
        reference = self.p.project_team(expected, observer_side=self.j.array([0, 1], self.j.int32),
            expected_vocabulary_binding=tables.vocabulary_binding)
        for left, right in zip(observed, reference):
            self.np.testing.assert_array_equal(left, right)
        self.assertEqual(int(observed.numeric_columns.sum()), 27)
        self.assertEqual(int(observed.categorical_columns.sum()), 7)
        return result

    def test_every_team_fact_and_owned_writer_column_matches_source_both_owners(self):
        self.assert_source(self.e.RequestExposure('current', 'physical', True, True))

    def test_explicit_request_exposure_and_faint_denominator_quirks(self):
        for mode in ('current', 'base', 'hidden'):
            for gender in ('physical', 'absent'):
                for stats in (False, True):
                    for item in (False, True):
                        with self.subTest(mode=mode, gender=gender, stats=stats, item=item):
                            got = self.assert_source(self.e.RequestExposure(mode, gender, stats, item)).facts
                            self.assertEqual(float(got.hp_denominator[0, 1]), 1)
                            self.assertEqual(float(got.actual_stats[0, 1, 0]), 0)
                            self.assertEqual(int(got.gender[0, 1]), 0)

    def test_each_opponent_field_can_be_poisoned_independently(self):
        tables = self.tables(); state = self.state()
        baseline = self.e.team_facts(self.owned(state), tables, expected_binding=tables.binding)
        for field in self.e.OwnedTeam._fields[2:]:
            value = getattr(state, field)
            poisoned = value.at[0, 1].set(-2147483648).at[1, 0].set(2147483647)
            result = self.e.team_facts(self.owned(state._replace(**{field: poisoned})), tables,
                expected_binding=tables.binding)
            for left, right in zip(self.jax.tree.leaves(result), self.jax.tree.leaves(baseline)):
                self.np.testing.assert_array_equal(left, right, err_msg=field)

    def test_invalid_owned_values_and_bindings_refuse_without_emitting_facts(self):
        tables = self.tables(); owned = self.owned()
        bad = [self.owned(owners=self.j.array([2, 1], self.j.int32)),
            owned._replace(owner_side=self.j.array([2, 1], self.j.int32)),
            owned._replace(species=owned.species.at[0, 0].set(999)),
            owned._replace(status=owned.status.at[0, 0].set(99)),
            owned._replace(gender=owned.gender.at[0, 0].set(3)),
            owned._replace(hp=owned.hp.at[0, 0].set(202)),
            owned._replace(ability=owned.ability.at[0, 0].set(999)),
            owned._replace(item=owned.item.at[0, 0].set(999)),
            owned._replace(active=owned.active.at[0].set(3)),
            owned._replace(engine_binding=owned.engine_binding.at[0, 0].add(self.j.uint32(1)))]
        for packet in bad:
            result = self.e.team_facts(packet, tables, expected_binding=tables.binding)
            self.assertEqual(result.admitted.tolist(), [False, True])
            patch = self.p.project_team(result.facts, observer_side=self.j.array([0, 1], self.j.int32),
                expected_vocabulary_binding=tables.vocabulary_binding)
            self.assertEqual(patch.admitted.tolist(), [False, True])
            self.assertFalse(bool(self.j.any(patch.numeric[0])))
            self.assertFalse(bool(self.j.any(patch.categorical[0])))
        result = self.e.team_facts(owned, tables, expected_binding=tables.binding.at[0].add(self.j.uint32(1)))
        self.assertEqual(result.admitted.tolist(), [False, False])

    def test_catalog_namespace_checkpoint_oov_and_typed_boundary_checks(self):
        from pokezero.category_vocab import CategoryVocabulary
        tables = self.tables()
        self.assertEqual(int(tables.species_categories[3]), self.vocab.encode('species:Ho-Oh'))
        changed = copy.deepcopy(self.metadata)
        changed['vocabulary']['species'][1:] = reversed(changed['vocabulary']['species'][1:])
        other = self.tables(metadata=changed)
        self.assertFalse(bool(self.j.array_equal(other.engine_binding, tables.engine_binding)))
        result = self.e.team_facts(self.owned(), other, expected_binding=other.binding)
        self.assertEqual(result.admitted.tolist(), [False, False])
        for changed_tables in (self.tables(checkpoint='b'*64),
                self.tables(self.e.RequestExposure('base', 'physical', True, True))):
            result = self.e.team_facts(self.owned(), changed_tables, expected_binding=tables.binding)
            self.assertEqual(result.admitted.tolist(), [False, False])
        reduced = CategoryVocabulary(tuple(x for x in self.vocab.tokens if x != 'type:psychic'))
        partial = self.tables(vocabulary=reduced)
        self.assertFalse(bool(partial.species_supported[2]))
        refused = self.e.team_facts(self.owned(), partial, expected_binding=partial.binding)
        self.assertEqual(refused.admitted.tolist(), [False, False])
        self.assertEqual(reduced.observed_oov_tokens, frozenset())
        duplicate_zero = copy.deepcopy(self.metadata)
        duplicate_zero['vocabulary']['abilities'].append('_none')
        with self.assertRaisesRegex(ValueError, 'ordered engine namespace'):
            self.tables(metadata=duplicate_zero)
        for owners in (self.j.array([True, False]), self.j.array([0., 1.]), self.j.array([0], self.j.int32)):
            with self.assertRaises(ValueError): self.owned(owners=owners)
        with self.assertRaises(ValueError): self.owned(binding=self.j.zeros((2, 8), self.j.int32))
        with self.assertRaises(ValueError): self.tables(checkpoint='not-a-hash')
        with self.assertRaises(ValueError): self.e.RequestExposure('guessed', 'physical', True, True)

    def test_jit_matches_eager_and_invalid_rows_remain_refused(self):
        tables = self.tables(); owned = self.owned()
        fast = self.e.prepare_team_facts(tables, expected_binding=tables.binding)
        for data in (owned, owned._replace(owner_side=self.j.array([-1, 1], self.j.int32))):
            eager = self.e.team_facts(data, tables, expected_binding=tables.binding)
            actual = fast(data)
            for left, right in zip(self.jax.tree.leaves(actual), self.jax.tree.leaves(eager)):
                self.np.testing.assert_array_equal(left, right)

    def test_copied_tags_and_replaced_array_edges_cannot_enter_checked_api(self):
        compiled = self.tables(); owned = self.owned()
        raw = compiled.device
        forged = raw._replace(species_categories=raw.species_categories.at[1].set(raw.species_categories[2]))
        # Exact independent-review reproduction: a copied digest cannot seal
        # changed label data, nor can a new handle wrap it into legitimacy.
        for candidate in (raw, forged, self.e.Compiled(forged), replace(compiled)):
            with self.assertRaisesRegex(ValueError, 'original compiled holder'):
                self.e.team_facts(owned, candidate, expected_binding=compiled.binding)
            with self.assertRaisesRegex(ValueError, 'original compiled holder'):
                self.e.prepare_team_facts(candidate, expected_binding=compiled.binding)
        fast = self.e.prepare_team_facts(compiled, expected_binding=compiled.binding)
        self.assertEqual(fast(owned).admitted.tolist(), [True, True])
        # Even deliberate frozen-dataclass escape must refuse after a JIT
        # cache exists. A guard only during tracing would miss this change.
        binding = compiled.binding
        object.__setattr__(compiled, 'device', forged)
        with self.assertRaisesRegex(ValueError, 'original compiled holder'):
            fast(owned)
        with self.assertRaisesRegex(ValueError, 'original compiled holder'):
            self.e.team_facts(owned, compiled, expected_binding=binding)
        valid = self.tables()
        with self.assertRaisesRegex(ValueError, 'host boundary cannot be traced'):
            self.jax.jit(lambda x: self.e.team_facts(x, valid, expected_binding=valid.binding))(owned)
        prepared = self.e.prepare_team_facts(valid, expected_binding=valid.binding)
        with self.assertRaisesRegex(ValueError, 'host boundary cannot be traced'):
            self.jax.jit(prepared)(owned)
        with self.assertRaisesRegex(ValueError, 'host boundary cannot be traced'):
            self.jax.jit(lambda: self.e.team_facts(owned, valid, expected_binding=valid.binding))()
        with self.assertRaisesRegex(ValueError, 'host boundary cannot be traced'):
            self.jax.jit(lambda: prepared(owned))()

    def test_wrong_dex_lookup_identity_or_display_name_refuses_before_mapping(self):
        from pokezero.dex import ShowdownDex
        pikachu = self.dex.species['pikachu']
        for counterfeit in (self.dex.species['gardevoir'],
                replace(pikachu, id='gardevoir'), replace(pikachu, name='Gardevoir')):
            bad = ShowdownDex({}, dict(self.dex.species, pikachu=counterfeit), {})
            with self.assertRaisesRegex(ValueError, 'dex identity differs'):
                self.e.compile_engine_private_team(self.metadata, vocabulary=self.vocab,
                    dex=bad, checkpoint_sha256='a'*64,
                    exposure=self.e.RequestExposure('current', 'physical', True, True))
        # Punctuation is an identity-preserving spelling, not a species alias.
        tables = self.tables()
        self.assertEqual(int(tables.species_categories[3]), self.vocab.encode('species:Ho-Oh'))

    def test_full_id_order_retains_unsupported_rows_and_selected_rows_refuse(self):
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.dex import ShowdownDex, SpeciesInfo
        metadata = copy.deepcopy(self.metadata)
        metadata['vocabulary']['species'] += ['abra', 'unownz', 'missingno']
        metadata['vocabulary']['abilities'] += ['damp']
        metadata['vocabulary']['items'] += ['aguavberry']
        dex = ShowdownDex({}, dict(self.dex.species,
            abra=SpeciesInfo('abra', 'Abra', ('Psychic',), dict(hp=25)),
            missingno=SpeciesInfo('missingno', 'MissingNo.', ('Bird', 'Normal'), dict(hp=33))), {})
        vocab = CategoryVocabulary(tuple(sorted((*self.vocab.tokens, 'species:missingno.', 'type:normal'))),
            aliases=self.vocab.aliases)
        compiled = self.tables(metadata=metadata, dex=dex, vocabulary=vocab)
        self.assertEqual(compiled.species_categories.shape, (7,))
        self.assertEqual(compiled.species_supported.tolist(), [False, True, True, True, False, False, False])
        self.assertEqual(compiled.ability_supported.tolist(), [True, True, True, True, False])
        self.assertEqual(compiled.item_supported.tolist(), [True, True, True, False])
        self.assertEqual(compiled.status_supported.tolist(), [True]*7)
        self.assertEqual(vocab.observed_oov_tokens, frozenset())
        # Original engine IDs1..3 are still exactly the same rows, not a
        # compacted supported-species namespace.
        self.np.testing.assert_array_equal(compiled.species_categories[:4], self.tables(vocabulary=vocab).species_categories)
        owned = self.owned(binding=self.j.broadcast_to(self.e.engine_metadata_binding(metadata), (2, 8)))
        fast = self.e.prepare_team_facts(compiled, expected_binding=compiled.binding)
        self.assertEqual(fast(owned).admitted.tolist(), [True, True])
        for field, indices in (('species', (4, 5, 6)), ('ability', (4,)), ('item', (3,))):
            for index in indices:
                bad = owned._replace(**{field: getattr(owned, field).at[0, 0].set(index)})
                result = fast(bad)
                self.assertEqual(result.admitted.tolist(), [False, True])
                patch = self.p.project_team(result.facts, observer_side=self.j.array([0, 1], self.j.int32),
                    expected_vocabulary_binding=compiled.vocabulary_binding)
                self.assertFalse(bool(self.j.any(patch.numeric[0])))
                self.assertFalse(bool(self.j.any(patch.categorical[0])))

    def test_hidden_and_unused_owner_fields_do_not_invent_exposure_or_refusal(self):
        owned = self.owned()
        for mode, field in (('base', 'ability'), ('current', 'base_ability'), ('hidden', 'ability')):
            compiled = self.tables(self.e.RequestExposure(mode, 'absent', False, False))
            baseline = self.e.team_facts(owned, compiled, expected_binding=compiled.binding)
            changed = owned._replace(**{field: getattr(owned, field).at[0, 0].set(999999)},
                item=owned.item.at[0, 0].set(999999), gender=owned.gender.at[0, 0].set(999999),
                stats=owned.stats.at[0, 0].set(-999))
            if mode == 'hidden':
                changed = changed._replace(base_ability=changed.base_ability.at[0, 0].set(999999))
            result = self.e.team_facts(changed, compiled, expected_binding=compiled.binding)
            for left, right in zip(self.jax.tree.leaves(result), self.jax.tree.leaves(baseline)):
                self.np.testing.assert_array_equal(left, right)
        compiled = self.tables()
        baseline = self.e.team_facts(owned, compiled, expected_binding=compiled.binding)
        # Physical maxHP/status of a canonical 0 fnt row are not source facts;
        # nor is the physical stats[HP], which requests derive from condition.
        changed = owned._replace(max_hp=owned.max_hp.at[:, 1].set(-1),
            status=owned.status.at[:, 1].set(999), stats=owned.stats.at[..., 0].set(-1))
        result = self.e.team_facts(changed, compiled, expected_binding=compiled.binding)
        for left, right in zip(self.jax.tree.leaves(result), self.jax.tree.leaves(baseline)):
            self.np.testing.assert_array_equal(left, right)
        removed = owned._replace(item=owned.item.at[:, 0].set(0))
        result = self.e.team_facts(removed, compiled, expected_binding=compiled.binding)
        self.assertEqual(result.admitted.tolist(), [True, True])
        self.assertEqual(result.facts.item_known[:, 0].tolist(), [False, False])


if __name__ == '__main__': unittest.main()
