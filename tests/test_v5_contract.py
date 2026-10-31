"""Authored v5 data contracts only: no weights, network, worlds or games."""
from dataclasses import replace
import json
import math
import unittest
from types import SimpleNamespace

from pokezero.observation import (
    OBSERVATION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION_V4,
    ObservationPerspective, PokeZeroObservationV0, SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
)
from pokezero import v5_contract as c
from pokezero.search_label_data import SEARCH_COLUMNS, validate_batch_columns
from pokezero.randbat import Gen3RandbatSource, Gen3RandbatVariant, Gen3RandbatSpeciesUniverse, RandbatSourceMetadata


def vocabulary():
    return c.LabelVocabulary('a' * 64, ('surf', 'recover', 'thunderwave'), (('blissey', 3), ('swampert', 2)))


def v4(*, owner=0, offset=0):
    return PokeZeroObservationV0(
        categorical_ids=tuple(tuple(offset + i * 4 + j % 4 for j in range(41)) for i in range(23)),
        numeric_features=tuple(tuple((offset + i + j) / 100 for j in range(132)) for i in range(23)),
        token_type_ids=(0,) + (1,) * 6 + (2,) * 6 + (3,) * 9 + (5,),
        attention_mask=(True,) * 23, legal_action_mask=(True, True) + (False,) * 7,
        perspective=ObservationPerspective.from_showdown_slot('owner-' + str(owner), ('p1', 'p2')[owner]),
        schema_version=OBSERVATION_SCHEMA_VERSION_V4)


def observation():
    catalog = vocabulary()
    posterior = c.PublicPosterior(catalog, 'blissey', 0b101, (0.2, 0.3, 0.5))
    return c.append_belief_tokens(v4(), (posterior,) * 6,
        address=c.DecisionAddress(8, 9, 0), category_vocabulary_sha256='b' * 64)


def improvement():
    return c.RootImprovement((0.25, 0.75) + (0.0,) * 7, 0.65, (0.2, 0.8) + (None,) * 7, worlds=2)


class V5ContractTests(unittest.TestCase):
    def static_source(self, count=44):
        rows = tuple(Gen3RandbatVariant('variant-' + str(i), 'set', 'Swalot', 'role', 80,
            ('Surf', 'Recover'), 'Sticky Hold', 'Leftovers') for i in range(count))
        metadata = RandbatSourceMetadata('gen3randombattle', 3, None, None, None, 'f5a5265143d423af')
        return Gen3RandbatSource(metadata, {'swalot': Gen3RandbatSpeciesUniverse('Swalot', 80, rows)})

    def test_actual_source_builder_binds_every_original_variant_and_its_order(self):
        source = self.static_source()
        census = c.source_variant_census(source)
        self.assertEqual(census['maximum'], 44)
        self.assertEqual(census['variant_rows'], 44)
        self.assertEqual(census['species_over_capacity'], ())
        vocabulary = c.LabelVocabulary.from_public_source(source, move_ids=('surf', 'recover'),
            expected_source_hash='f5a5265143d423af')
        self.assertEqual(vocabulary.species_variants, (('swalot', 44),))
        self.assertEqual(vocabulary.source_binding_sha256, census['source_rows_sha256'])
        last = c.PublicPosterior(vocabulary, 'swalot', 1 << 43, (1 / 44,) * 44)
        self.assertEqual(last.numeric_row()[43], 1)
        self.assertEqual(last.numeric_row()[44:64], (0.0,) * 20)
        target = c.HiddenSet('swalot', (43,)).probabilities(vocabulary)
        self.assertEqual(target[43], 1)
        self.assertEqual(sum(target), 1)
        reordered = replace(source, universes={'swalot': replace(source.universes['swalot'],
            variants=source.universes['swalot'].variants[::-1])})
        self.assertNotEqual(c.source_variant_census(reordered)['source_rows_sha256'], census['source_rows_sha256'])
        changed = list(source.universes['swalot'].variants)
        changed[-1] = replace(changed[-1], item='Lum Berry')
        modified = replace(source, universes={'swalot': replace(source.universes['swalot'], variants=tuple(changed))})
        self.assertNotEqual(c.source_variant_census(modified)['source_rows_sha256'], census['source_rows_sha256'])
        with self.assertRaises(ValueError):
            c.LabelVocabulary.from_public_source(source, move_ids=('surf',), expected_source_hash='b' * 16)

    def test_future_source_over_capacity_is_reported_in_full_never_pruned(self):
        source = self.static_source(65)
        census = c.source_variant_census(source)
        self.assertEqual(census['maximum'], 65)
        self.assertEqual(census['variant_rows'], 65)
        self.assertEqual(census['species_over_capacity'], (('swalot', 65),))
        with self.assertRaisesRegex(ValueError, 'full source axis exceeds64; no rows removed'):
            c.LabelVocabulary.from_public_source(source, move_ids=('surf',),
                expected_source_hash='f5a5265143d423af')
        self.assertEqual(len(source.universes['swalot'].variants), 65)

    def test_append_preserves_every_v4_value_and_uses_a_new_role(self):
        old = v4(); new = observation()
        for name in ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask'):
            self.assertEqual(getattr(new, name)[:23], getattr(old, name))
        self.assertEqual(new.legal_action_mask, old.legal_action_mask)
        self.assertEqual(len(new.numeric_features), 29)
        self.assertTrue(all(len(row) == 132 for row in new.numeric_features))
        self.assertTrue(all(len(row) == 41 for row in new.categorical_ids))
        self.assertEqual(new.token_type_ids[23:], (7,) * 6)
        self.assertEqual(new.categorical_ids[23:], old.categorical_ids[7:13])
        self.assertEqual(new.attention_mask[23:], (True,) * 6)
        self.assertEqual(OBSERVATION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION_V4)
        self.assertNotIn(c.OBSERVATION_SCHEMA_VERSION_V5, SUPPORTED_OBSERVATION_SCHEMA_VERSIONS)

    def test_posterior_mask_preserves_source_order_prior_and_fallback_status(self):
        p = observation().posteriors[0]; row = p.numeric_row()
        self.assertAlmostEqual(row[0], 2 / 7); self.assertEqual(row[1], 0)
        self.assertAlmostEqual(row[2], 5 / 7)
        self.assertEqual(row[64:67], (0.2, 0.3, 0.5))
        self.assertEqual(row[67:128], (0.0,) * 61)
        self.assertEqual(row[128:], (3 / 64, 2 / 64, 0, 0))
        fallback = replace(p, effective_mask=0b111, reason='INCONSISTENT_REVEALS', pin_conflict=True)
        self.assertEqual(fallback.numeric_row()[:3], (0.2, 0.3, 0.5))
        self.assertEqual(fallback.numeric_row()[130:], (2 / 3, 1))
        namespace = c.LabelVocabulary('a' * 64, ('surf',), (('blissey', 64),))
        tail = c.PublicPosterior(namespace, 'blissey', 1 << 63, (1 / 64,) * 64)
        self.assertEqual(tail.numeric_row()[:63], (0.0,) * 63)
        self.assertEqual(tail.numeric_row()[63], 1.0)

    def test_unknown_and_missing_source_have_no_private_species_mass(self):
        catalog = vocabulary()
        unknown = c.PublicPosterior(catalog, None, 0, (), reason='UNKNOWN_SPECIES')
        missing = c.PublicPosterior(catalog, None, 0, (), reason='NO_SOURCE')
        self.assertEqual(unknown.numeric_row()[:130], (0.0,) * 130)
        self.assertNotEqual(unknown.numeric_row(), missing.numeric_row())
        old = replace(v4(), attention_mask=(True,) * 7 + (False,) * 6 + (True,) * 10)
        new = c.append_belief_tokens(old, (unknown,) * 6,
            address=c.DecisionAddress(1, 2, 0), category_vocabulary_sha256='b' * 64)
        self.assertEqual(new.numeric_features[:23], old.numeric_features)
        with self.assertRaises(ValueError):
            c.append_belief_tokens(old, observation().posteriors,
                address=c.DecisionAddress(1, 2, 0), category_vocabulary_sha256='b' * 64)
        with self.assertRaises(ValueError):
            c.PublicPosterior(catalog, None, 1, (1.0,), reason='UNKNOWN_SPECIES')

    def test_source_namespaces_masks_weights_and_typed_inputs_refuse_drift(self):
        p = observation().posteriors[0]
        for change in (dict(effective_mask=8), dict(effective_mask=0), dict(effective_mask=True),
                dict(prior_weights=(0.0, 0.0, 0.0)), dict(prior_weights=(0.2, math.nan, 0.8)),
                dict(prior_weights=(0.0, 1.0, 0.0)), dict(prior_weights=(True, 0.3, 0.5)),
                dict(prior_weights=('0.2', 0.3, 0.5)), dict(species='notinsource')):
            with self.subTest(change=change), self.assertRaises(ValueError):replace(p, **change)
        new = observation()
        other = replace(p, vocabulary=replace(p.vocabulary, source_binding_sha256='c' * 64))
        with self.assertRaises(ValueError):replace(new, posteriors=(other,) + new.posteriors[1:])
        with self.assertRaises(ValueError):replace(new, token_type_ids=new.token_type_ids[:23] + (6,) * 6)
        with self.assertRaises(ValueError):replace(new, categorical_ids=new.categorical_ids[:23] + ((0,) * 41,) * 6)
        with self.assertRaises(ValueError):replace(new, numeric_features=new.numeric_features[:28])
        with self.assertRaises(ValueError):
            c.append_belief_tokens(v4(owner=1), new.posteriors,
                address=new.address, category_vocabulary_sha256='b' * 64)

    def test_mutable_caller_lists_cannot_rewrite_bound_tokens_or_label_namespace(self):
        moves = ['surf']; species = [['blissey', 2]]
        catalog = c.LabelVocabulary('a' * 64, moves, species)
        prior = [0.25, 0.75]; p = c.PublicPosterior(catalog, 'blissey', 3, prior)
        binding = catalog.sha256; row = p.numeric_row()
        moves[0] = 'recover'; species[0][0] = 'swampert'; prior[0] = 1.0
        self.assertEqual(catalog.sha256, binding)
        self.assertEqual(p.numeric_row(), row)

    def test_privileged_surface_swaps_only_explicit_opponent_rows(self):
        public = observation(); actor = public.model_inputs()
        peer = v4(owner=1, offset=300); slots = (5, 4, 3, 2, 1, 0)
        private = c.privileged_view(public, peer,
            peer_address=replace(public.address, owner=1), opponent_slot_map=slots,
            peer_category_vocabulary_sha256='b' * 64)
        for name in ('categorical_ids', 'numeric_features'):
            self.assertEqual(getattr(private, name)[7:13], tuple(getattr(peer, name)[i + 1] for i in slots))
            self.assertEqual(getattr(private, name)[:7], getattr(public, name)[:7])
            self.assertEqual(getattr(private, name)[13:], getattr(public, name)[13:])
        self.assertEqual(public.model_inputs(), actor)
        self.assertEqual(private.legal_action_mask, public.legal_action_mask)
        with self.assertRaises(ValueError):private.model_inputs()

    def test_privileged_swap_refuses_cross_frame_owner_or_implicit_slot_order(self):
        public = observation(); peer = v4(owner=1)
        for address, slots in ((public.address, tuple(range(6))),
                (replace(public.address, owner=1, frame=10), tuple(range(6))),
                (replace(public.address, owner=1), (0, 0, 1, 2, 3, 4)),
                (replace(public.address, owner=1), (False, 1, 2, 3, 4, 5))):
            with self.subTest(address=address, slots=slots), self.assertRaises(ValueError):
                c.privileged_view(public, peer, peer_address=address, opponent_slot_map=slots,
                    peer_category_vocabulary_sha256='b' * 64)
        with self.assertRaises(ValueError):
            c.privileged_view(public, peer, peer_address=replace(public.address, owner=1),
                opponent_slot_map=tuple(range(6)), peer_category_vocabulary_sha256='c' * 64)

    def test_opponent_labels_name_moves_and_switch_species_not_slots(self):
        catalog = vocabulary()
        self.assertEqual(c.OpponentAction('MOVE', 'surf').index(catalog), 0)
        self.assertEqual(c.OpponentAction('SWITCH', 'blissey').index(catalog), 3)
        for kind, identity in (('MOVE', 0), ('MOVE', '0'), ('SLOT', 'surf'), ('SWITCH', 'unlisted')):
            with self.subTest(kind=kind, identity=identity), self.assertRaises(ValueError):
                c.OpponentAction(kind, identity).index(catalog)
        obs = observation(); before = obs.model_inputs()
        labels = c.V5Labels(obs.address, opponent_next_action=c.OpponentAction('SWITCH', 'blissey'))
        self.assertEqual(labels.columns(obs)['opponent_action_target'], 3)
        self.assertEqual(obs.model_inputs(), before)

    def test_auxiliary_masks_distinguish_unknown_from_actual_faint_and_end_hp(self):
        obs = observation(); base = c.V5Labels(obs.address)
        empty = base.columns(obs)
        self.assertFalse(any(empty['pokemon_end_target_mask']))
        self.assertFalse(any(empty['hidden_set_target_mask']))
        labels = replace(base, terminal_status='COMPLETE', outcome=1,
            pokemon_end=(c.PokemonEnd(False, 0), c.PokemonEnd(True, 0.4)) + (None,) * 10,
            hidden_sets=(c.HiddenSet('blissey', (0, 2)),) + (None,) * 5)
        cols = labels.columns(obs)
        self.assertEqual(cols['survival_targets'][:3], (0, 1, 0))
        self.assertEqual(cols['final_hp_targets'][:3], (0, 0.4, 0))
        self.assertEqual(cols['pokemon_end_target_mask'][:3], (True, True, False))
        self.assertEqual(cols['hidden_set_targets'][0][:3], (0.5, 0, 0.5))
        self.assertEqual(cols['label_vocabulary_sha256'], obs.label_vocabulary.sha256)
        for bad in (c.HiddenSet('blissey', (3,)), c.HiddenSet('blissey', (0, 0)), c.HiddenSet('swampert', (0,))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, hidden_sets=(bad,) + (None,) * 5).columns(obs)

    def test_capped_refused_and_incomplete_games_do_not_invent_terminal_labels(self):
        obs = observation()
        for status in ('IN_PROGRESS', 'CAPPED', 'REFUSED'):
            with self.subTest(status=status):
                labels = c.V5Labels(obs.address, terminal_status=status)
                self.assertFalse(any(labels.columns(obs)['pokemon_end_target_mask']))
                with self.assertRaises(ValueError):replace(labels, outcome=0).columns(obs)
                with self.assertRaises(ValueError):
                    replace(labels, pokemon_end=(c.PokemonEnd(True, 0.5),) + (None,) * 11).columns(obs)
        for values in ((True, 0), (False, 0.1), (1, 0.2), (True, math.nan)):
            with self.subTest(values=values), self.assertRaises(ValueError):c.PokemonEnd(*values)

    def test_D_labels_use_completed_Q_public_leaf_and_independent_value_mask(self):
        obs = observation(); actor = obs.model_inputs()
        labels = c.V5Labels(obs.address, root_improvement=improvement(), search_status='LABELED')
        cols = labels.columns(obs)
        self.assertTrue(cols['search_policy_target_mask'])
        self.assertFalse(cols['search_value_target_mask'])
        cols = replace(labels, terminal_status='COMPLETE', outcome=-1).columns(obs)
        self.assertAlmostEqual(cols['search_value_target'], -0.175)
        self.assertTrue(cols['search_value_target_mask'])
        self.assertEqual(obs.model_inputs(), actor)
        refused = c.V5Labels(obs.address, search_status='REFUSED', search_refusal='UNKNOWN_POSITIVE_WORLD')
        self.assertNotIn('search_policy_target', refused.columns(obs))
        self.assertEqual(refused.columns(obs)['search_refusal'], 'UNKNOWN_POSITIVE_WORLD')

    def test_D_cannot_substitute_a_privileged_critic_depth2_or_partial_Q(self):
        obs = observation(); root = improvement()
        for change in (dict(leaf='PRIVILEGED_VALUE'), dict(depth=2), dict(worlds=3), dict(worlds=True),
                dict(replies='OUR_POLICY'), dict(label_only=False), dict(root_value=True),
                dict(root_value=0.6),
                dict(completed_q=(None, 0.8) + (None,) * 7),
                dict(completed_q=(0.2, 0.8, 0.0) + (None,) * 6),
                dict(policy_target=(0.2, 0.7, 0.1) + (0.0,) * 6)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(root, **change).validate(obs.legal_action_mask)
        with self.assertRaises(ValueError):c.V5Labels(obs.address, root_improvement=root).columns(obs)
        with self.assertRaises(ValueError):
            c.V5Labels(obs.address, search_status='REFUSED', search_refusal='').columns(obs)

    def test_warm_start_is_a_shape_plan_and_preserves_input_namespace(self):
        old = {'category_embedding.weight': (916, 512), 'token_type_embedding.weight': (16, 512),
            'numeric_projection.weight': (512, 132), 'policy_head.weight': (1, 512),
            'value_head.weight': (1, 512), 'encoder.layer.norm.bias': (512,)}
        new = {k: v for k, v in old.items() if k not in ('value_head.weight', 'encoder.layer.norm.bias')}
        new.update({'value_head.mlp.weight': (512, 512), 'encoder.layer.rms.weight': (512,)})
        options = dict(source_category_vocabulary_sha256='b' * 64, target_category_vocabulary_sha256='b' * 64)
        plan = c.warm_start_shape_plan(old, new, **options)
        self.assertIn('numeric_projection.weight', plan['copy'])
        self.assertIn('policy_head.weight', plan['copy'])
        self.assertIn('value_head.mlp.weight', plan['initialize'])
        self.assertIn('encoder.layer.norm.bias', plan['unused_source'])
        self.assertEqual(plan['production_initialization'], 'scratch')
        with self.assertRaises(ValueError):
            c.warm_start_shape_plan(old, new, **dict(options, target_category_vocabulary_sha256='c' * 64))
        with self.assertRaises(ValueError):
            c.warm_start_shape_plan(old, dict(new, **{'numeric_projection.weight': (512, 173)}), **options)
        with self.assertRaises(ValueError):c.warm_start_shape_plan(old, new, **options, source_schema='v3')

    def test_actual_batch_columns_keep_independent_masks_and_model_window_shapes(self):
        first = observation(); second = replace(first, address=replace(first.address, frame=10))
        a = c.V5Labels(first.address, root_improvement=improvement(), search_status='LABELED',
            terminal_status='COMPLETE', outcome=-1,
            opponent_next_action=c.OpponentAction('MOVE', 'recover'))
        b = c.V5Labels(second.address, search_status='REFUSED', search_refusal='SOURCE_REFUSAL')
        batch = c.batch_columns((first, second), (a, b))
        self.assertEqual(batch['schema'], c.TRAINING_CONTRACT_SCHEMA)
        self.assertEqual(len(batch['numeric_features']), 2)
        self.assertEqual(len(batch['numeric_features'][0]), 1)
        self.assertEqual(len(batch['numeric_features'][0][0]), 29)
        self.assertEqual(len(batch['numeric_features'][0][0][0]), 132)
        self.assertEqual(batch['history_mask'], ((True,), (True,)))
        self.assertEqual(batch['opponent_semantic_action_targets'], (1, -1))
        self.assertEqual(batch['opponent_semantic_action_target_mask'], (True, False))
        self.assertEqual(batch['search_policy_target_mask'], (True, False))
        self.assertEqual(batch['search_value_target_mask'], (True, False))
        self.assertEqual(batch['search_policy_targets'][1], (0.0,) * 9)
        self.assertEqual(batch['search_status'], ('LABELED', 'REFUSED'))
        self.assertNotIn('privileged_view', batch)
        self.assertNotIn('action_indices', batch)
        self.assertNotIn('opponent_action_indices', batch)
        self.assertTrue(set(SEARCH_COLUMNS).issubset(batch))
        validate_batch_columns(SimpleNamespace(**batch))
        self.assertEqual(first.model_inputs()['numeric_features'][:23], v4().numeric_features)
        for observations, labels in (((), ()), ((first,), ()), ((first,), (b,)),
                ((first, replace(second, category_vocabulary_sha256='d' * 64)), (a, b))):
            with self.subTest(observations=observations), self.assertRaises(ValueError):
                c.batch_columns(observations, labels)

    def test_initial_four_arms_and_endpoint_follow_the_decided_contrasts(self):
        arms = c.phase_c_arms()
        self.assertEqual([(x['arm'], x['replicate']) for x in arms], [('A', 0), ('B', 0), ('D', 0), ('B', 1)])
        self.assertTrue(all(x['initialization'] == 'scratch' and not x['privileged_critic'] for x in arms))
        self.assertEqual(arms[2]['label_fraction'], 0.25)
        self.assertEqual(arms[2]['leaf'], 'PUBLIC_VALUE')
        self.assertEqual(arms[2]['actor'], 'RAW_POLICY')
        b, seed2 = dict(arms[1]), dict(arms[3]); b.pop('replicate'); seed2.pop('replicate')
        self.assertEqual(b, seed2)
        for b_lower, d_lower, result in ((0, 0, 'RUN_C_ON_B_FREED_GPUS'),
                (0.1, -0.1, 'ADVANCE_B'), (-0.1, 0.1, 'ACT_WITH_SEARCH_NEXT_FREE_GPU')):
            self.assertEqual(c.phase_c_next_step(endpoint_complete=True,
                b_minus_a_lower90=b_lower, d_minus_b_lower90=d_lower), result)
        with self.assertRaises(ValueError):
            c.phase_c_next_step(endpoint_complete=False, b_minus_a_lower90=0, d_minus_b_lower90=0)
        descriptor = c.contract_descriptor()
        self.assertIn('EXPLICIT_V5_CONSUMER_REQUIRED', descriptor['consumer_integration'])
        self.assertEqual(json.loads(json.dumps(descriptor, allow_nan=False))['tokens'], 29)
        self.assertEqual(c._sha(descriptor), c._sha(c.contract_descriptor()))


if __name__ == '__main__':
    unittest.main()
