"""Synthetic public-source/own-cell fixtures; no battle/model/world execution."""
from dataclasses import replace
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pokezero import v5_contract as contract
from pokezero.belief import PlayerBeliefView, RevealedPokemonBelief
from pokezero.determinization import (
    _fixture_from_variant_payload, _sample_gender, _sample_hidden_backline,
    _opponent_team_from_belief_result,
)
from pokezero.randbat import (
    Gen3RandbatSource, Gen3RandbatSpeciesUniverse, Gen3RandbatVariant,
    RandbatSourceMetadata,
)
from pokezero.v5_gumbel import Recipe
from gpu_engine import device_public_worlds as worlds


def source_fixture():
    # Mixed case deliberately exercises original normalized used-species rules.
    universes = {}
    for index, name in enumerate(('Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta')):
        variants = tuple(Gen3RandbatVariant(f'{name}-{v}', 'source', name, 'synthetic',
            80, ('tackle', 'surf', 'rest', 'protect'), 'pressure' if v else 'insomnia',
            'leftovers' if v else 'oranberry') for v in range(2))
        universes[name.lower()] = Gen3RandbatSpeciesUniverse(name, 80, variants)
    source = Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle', 3,
        None, None, None, 'synthetic-source'), universes,
        move_metadata={'tackle': {'type': 'Normal', 'category': 'Physical'}},
        species_metadata={name: {'gender': 'N'} for name in universes})
    metadata = {'vocabulary': {
        'species': ['', *[u.species for u in universes.values()]],
        'moves': ['', 'tackle', 'surf', 'rest', 'protect'],
        'abilities': ['', 'insomnia', 'pressure'],
        'items': ['', 'oranberry', 'leftovers']}}
    return source, metadata


def compile_fixture():
    source, metadata = source_fixture()
    census = contract.source_variant_census(source)
    return worlds.compile_source(source, metadata,
        expected_source_hash=source.metadata.source_hash,
        expected_source_rows_sha256=census['source_rows_sha256'])


def root_fixture(owner=0):
    recipe = Recipe(17)
    frame = next(i for i in range(100) if recipe.selected(contract.DecisionAddress(81, i, owner)))
    view = dict(species=np.asarray([[1, 0, 0, 0, 0, 0]], np.int32),
        present=np.asarray([[True, False, False, False, False, False]]),
        belief_mask=np.zeros((1, 6, 2), np.uint32),
        revealed_moves=np.zeros((1, 6, 4), np.int32),
        ability=np.zeros((1, 6), np.int32), item=np.zeros((1, 6), np.int32),
        gender=np.full((1, 6), -1, np.int32), level=np.full((1, 6), 80, np.int32),
        error=np.zeros(1, np.int32), mon_features={'item_swapped': np.zeros((1, 6), np.int32)})
    view['belief_mask'][0, 0, 0] = 3
    view['revealed_moves'][0, 0, 0] = 1
    return worlds.Root(contract.DecisionAddress(81, frame, owner),
        worlds._freeze({'hp': np.arange(6)}), worlds._freeze(view), 'root-public-clone',
        compile_fixture().binding_sha256), recipe


class DevicePublicWorldsTests(unittest.TestCase):
    def test_original_source_draw_order_two_worlds_and_no_pool_prior(self):
        compiled = compile_fixture()
        for owner in (0, 1):
            root, recipe = root_fixture(owner)
            result = worlds.plan_starts(root, compiled, recipe)
            self.assertTrue(result.selected)
            self.assertEqual(len(result.starts), 2)
            self.assertEqual(result.source_fixture_unknown_mass, 0)
            self.assertEqual(result.current_world_unknown_mass, 1)
            self.assertFalse(result.device_branch_ready)
            self.assertFalse(result.teacher_ready)
            for index, start in enumerate(result.starts):
                rng = random.Random(recipe.world_seed(root.address, index))
                universe = compiled.original.universe_for('Alpha')
                gender = _sample_gender('Alpha', set_source=compiled.original, rng=rng)
                vi = rng.randrange(len(universe.variants))
                first = _fixture_from_variant_payload(universe.variants[vi].to_summary(),
                    fallback_species='Alpha', set_source=compiled.original, gender=gender)
                rest = _sample_hidden_backline(compiled.original,
                    used_species={'alpha'}, count=5, rng=rng)
                self.assertEqual(start.opponent_fixtures, (first, *rest))
                original_view = PlayerBeliefView(f'p{owner+1}', f'p{2-owner}', (),
                    (RevealedPokemonBelief(f'p{2-owner}a', 'Alpha',
                        revealed_moves=('tackle',)),))
                expected, reason = _opponent_team_from_belief_result(original_view,
                    set_source=compiled.original, format_id='gen3randombattle',
                    rng=random.Random(start.seed), team_size=6,
                    canonicalize_revealed_variants=True)
                self.assertIsNone(reason)
                self.assertEqual(start.opponent_fixtures, expected)
                self.assertEqual(start.revealed_source_indices, ((0, vi),))
                self.assertEqual(len({mon.species for mon in start.opponent_fixtures}), 6)
                self.assertEqual(start.seed, recipe.world_seed(root.address, index))
            not_selected = next(i for i in range(100) if not recipe.selected(
                contract.DecisionAddress(81, i, owner)))
            skipped = worlds.plan_starts(replace(root, address=contract.DecisionAddress(
                81, not_selected, owner)), compiled, recipe)
            self.assertFalse(skipped.selected)
            self.assertEqual(skipped.starts, ())

    def test_current_public_source_refusals_keep_both_slots_without_retry(self):
        compiled = compile_fixture()
        root, recipe = root_fixture()
        # Every bad root retains BOTH original declared seeds and weight1/2.
        mutations = []
        def changed(name, index, value):
            view = {k: (dict(v) if isinstance(v, dict) else v) for k, v in root.view.items()}
            view[name] = np.array(view[name], copy=True)
            view[name][index] = value
            return replace(root, view=worlds._freeze(view))
        mutations.extend((changed('belief_mask', (0, 0, 0), 0),
            changed('belief_mask', (0, 0, 1), 1),  # off original 2-row axis
            changed('species', (0, 0), 999),
            changed('species', (0, 1), 2),  # unrevealed identity leak
            changed('error', 0, 1), changed('level', (0, 0), 50),
            changed('revealed_moves', (0, 0, 0), 999)))
        # Mask only original insomnia row, contradict original publicly shown pressure.
        view = dict(root.view)
        view['belief_mask'] = np.zeros((1, 6, 2), np.uint32);view['belief_mask'][0, 0, 0] = 1
        view['ability'] = np.zeros((1, 6), np.int32);view['ability'][0, 0] = 2
        mutations.append(replace(root, view=worlds._freeze(view)))
        for bad in mutations:
            with self.subTest(view=bad.view):
                result = worlds.plan_starts(bad, compiled, recipe)
                self.assertEqual(result.source_fixture_unknown_mass, 1)
                self.assertEqual(len(result.starts), 2)
                self.assertTrue(all(v.opponent_fixtures is None and v.refusals for v in result.starts))
                self.assertEqual(tuple(v.seed for v in result.starts),
                    tuple(recipe.world_seed(root.address, k) for k in (0, 1)))
        # Public mutated item must not become a false initial-set constraint.
        view = dict(root.view)
        view['belief_mask'] = np.zeros((1, 6, 2), np.uint32);view['belief_mask'][0, 0, 0] = 1
        view['item'] = np.zeros((1, 6), np.int32);view['item'][0, 0] = 2
        view['mon_features'] = {'item_swapped': np.asarray([[1, 0, 0, 0, 0, 0]], np.int32)}
        self.assertEqual(worlds.plan_starts(replace(root, view=worlds._freeze(view)),
            compiled, recipe).source_fixture_unknown_mass, 0)

    def test_source_digest_order_and_caller_mutation_cannot_change_compiled_prior(self):
        source, metadata = source_fixture()
        census = contract.source_variant_census(source)
        kwargs = dict(expected_source_hash=source.metadata.source_hash,
            expected_source_rows_sha256=census['source_rows_sha256'])
        compiled = worlds.compile_source(source, metadata, **kwargs)
        root, recipe = root_fixture()
        baseline = worlds.plan_starts(root, compiled, recipe)
        source.universes.clear();source.species_metadata.clear()
        metadata['vocabulary']['species'][1] = 'foreign'
        self.assertEqual(worlds.plan_starts(root, compiled, recipe), baseline)
        with self.assertRaises(TypeError):compiled.original.species_metadata['alpha'] = {}
        fresh, meta = source_fixture()
        for overrides in ({'expected_source_hash': 'foreign'},
                {'expected_source_rows_sha256': '0'*64}):
            with self.assertRaises(ValueError):worlds.compile_source(fresh, meta, **{**kwargs, **overrides})
        swapped = replace(fresh, universes={**fresh.universes,
            'alpha': replace(fresh.universes['alpha'], variants=fresh.universes['alpha'].variants[::-1])})
        with self.assertRaises(ValueError):worlds.compile_source(swapped, meta, **kwargs)
        meta['vocabulary']['moves'][1:3] = meta['vocabulary']['moves'][1:3][::-1]
        foreign_namespace = worlds.compile_source(fresh, meta, **kwargs)
        with self.assertRaises(ValueError):worlds.plan_starts(root, foreign_namespace, recipe)

    def test_root_extract_indexes_own_first_and_retains_only_root_public(self):
        from gpu_engine.v5_device_rollout import RootContext
        for owner in (0, 1):
            accesses = []
            class OwnOnly:
                def __init__(self, name):self.name = name
                def __getitem__(self, key):
                    self_key = (1, owner)
                    if key != self_key:raise AssertionError('opposing or unindexed private read')
                    accesses.append((self.name, key))
                    return np.asarray([13, 14], np.int32)
                def __array__(self, *args):raise AssertionError('full private host conversion')
            state = SimpleNamespace(**{name: OwnOnly(name) for name in
                worlds.OWN_MON_FIELDS + worlds.OWN_SIDE_FIELDS})
            view = SimpleNamespace(_asdict=lambda: {'error': np.zeros(1, np.int32)})
            cloned = object();original_public = object()
            context = RootContext(state, original_public, np.asarray([80, 81]), np.asarray([2, 3]))
            compiled = compile_fixture()
            with patch.object(worlds.public, 'clone_for_worlds', return_value=cloned) as gather, \
                    patch.object(worlds.public, 'readout', return_value=view) as readout:
                got = worlds.detach_root(context, compiled, row=1, owner=owner, enable_device_worlds=True)
                self.assertIs(got.public, cloned)
                self.assertEqual(got.address, contract.DecisionAddress(81, 3, owner))
                self.assertEqual(len(accesses), len(worlds.OWN_MON_FIELDS + worlds.OWN_SIDE_FIELDS))
                self.assertIs(gather.call_args.args[0], original_public)
                self.assertIs(readout.call_args.args[0], cloned)
                self.assertNotIn('state', got.__dict__)
                self.assertNotIn('queued_actions', got.own)
                with self.assertRaises(ValueError):got.own['hp'][0] = 99
            with self.assertRaises(ValueError):worlds.detach_root(context, compiled, row=1, owner=owner)

    def test_opaque_public_hidden_power_and_struggle_never_invent_source_facts(self):
        source, metadata = source_fixture()
        universe = source.universes['alpha']
        source.universes['alpha'] = replace(universe, variants=tuple(
            replace(v, moves=('hiddenpowerice' if i == 0 else 'hiddenpowerfire', 'surf', 'rest', 'protect'))
            for i, v in enumerate(universe.variants)))
        metadata['vocabulary']['moves'].extend(('hiddenpowerice', 'hiddenpowerfire', 'struggle'))
        compiled = worlds.compile_source(source, metadata, expected_source_hash=source.metadata.source_hash,
            expected_source_rows_sha256=contract.source_variant_census(source)['source_rows_sha256'])
        root, recipe = root_fixture()
        view = dict(root.view)
        view['revealed_moves'] = np.asarray([[[5, 7, 0, 0], *[[0]*4 for _ in range(5)]]], np.int32)
        root = replace(root, view=worlds._freeze(view), source_binding_sha256=compiled.binding_sha256)
        self.assertEqual(len(worlds._revealed(root, compiled)[0][2]), 2)
        self.assertEqual(worlds.plan_starts(root, compiled, recipe).source_fixture_unknown_mass, 0)


if __name__ == '__main__':
    unittest.main()
