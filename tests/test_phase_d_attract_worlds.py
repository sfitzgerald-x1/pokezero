"""Attract carry and prospective gender isolation at the public-world boundary."""
from dataclasses import replace
import unittest

import jax
import numpy as np

from gpu_engine import phase_d_live_worlds as live, phase_d_protocol_world as protocol
from gpu_engine import device_public_world_state as states, v5_hypothetical_state as mapping, phase_d_tail as tail
from gpu_engine.state import ATTRACT
from pokezero.v5_contract import DecisionAddress, source_variant_census


class TypedAttractTests(unittest.TestCase):
    def test_active_only_host_and_device_and_source_faint_boundary(self):
        from test_gpu_engine_v5_public_world import TypedMappingTests
        spec, metadata = TypedMappingTests().fixture()
        # Different cursors catch accidental slot-zero or source-side writes.
        mon = spec.side_one.pokemon[0]
        side = replace(spec.side_one, pokemon=(mon, mon), active_index=1,
                       volatile_statuses=('attract',))
        spec = replace(spec, side_one=side)
        args = dict(metadata=metadata, seed=7, turn=2, active_turns=(1, 1), happiness=((255, 255), (255,)))
        self.assertIn('volatile:0', mapping.map_spec(spec, **args).refusals)
        for host in (False, True):
            mapped = mapping.map_spec(spec, held_volatiles=True,
                arrays=(lambda: protocol._host_arrays(7, None)) if host else None, **args)
            self.assertFalse(mapped.refusals)
            self.assertEqual(np.asarray(mapped.state.volatile).tolist(), [[[0, ATTRACT, 0, 0, 0, 0], [0]*6]])
            original = replace(spec, side_two=replace(spec.side_two, pokemon=(replace(mon, hp=0), mon)))
            facts = dict(replacement=dict(pending_residual=False, queued_actor=-1, turn=1, ages=(1, 1)),
                         fresh_substitutes={}, substitute_origins={}, wish_set_turns={}, turn=2)
            projected, cursors = tail.mapping_spec(original, facts['replacement'])
            pending = mapping.map_spec(projected, held_volatiles=True,
                arrays=(lambda: protocol._host_arrays(7, None)) if host else None,
                **{**args, 'happiness': ((255, 255), (255, 255))})
            self.assertFalse(pending.refusals)
            restored = tail.restore(pending.state, original, cursors, facts, metadata=metadata, checks=[])
            self.assertEqual(int(restored.hp[0, 1, 0]), 0)
            self.assertEqual(int(restored.volatile[0, 0, 1]) & ATTRACT, ATTRACT)


class PublicAttractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_phase_d_device_worlds import WorldsTests
        WorldsTests.setUpClass()
        cls.helper = WorldsTests()

    def test_b2_bags_on_both_sides_and_departure(self):
        h = self.helper
        for owner in (0, 1):
            for holder in (owner, 1-owner):
                adapter, request = h.adapter(owner), h.request(owner)
                name = 'Alpha' if holder == owner else 'Beta'
                source = 'Beta' if holder == owner else 'Alpha'
                wire = f'|-start|p{holder+1}a: {name}|Attract|[of] p{2-holder}a: {source}'
                adapter.feed((wire,))
                address = DecisionAddress(1, 0, owner)
                root = live.from_b2(adapter, request, h.source, h.metadata, address=address)
                made = live.sample_worlds(root, h.source, h.tables, count=16, seed=17)
                self.assertTrue(made.fully_constructed, [w.refusals for w in made.worlds])
                for world in made.worlds:
                    expected = np.zeros((1, 2, 6), np.uint32)
                    expected[0, holder, int(world.state.active[0, holder])] = ATTRACT
                    np.testing.assert_array_equal(np.asarray(world.state.volatile) & ATTRACT, expected)
                adapter.feed((f'|-end|p{holder+1}a: {name}|Attract|[silent]',))
                ended = live.from_b2(adapter, request, h.source, h.metadata, address=address)
                cleared = live.sample_worlds(ended, h.source, h.tables, count=16, seed=17)
                self.assertTrue(cleared.fully_constructed)
                self.assertTrue(all(not np.any(np.asarray(w.state.volatile) & ATTRACT) for w in cleared.worlds))

    def test_sampled_hidden_genders_do_not_enter_public_carry_or_reveal_order(self):
        h = self.helper
        # Every synthetic species is mixed despite a strongly biased dex ratio.
        raw = replace(h.source.original, species_metadata={name: {**info, 'gender': '', 'genderRatio': {'M': .875, 'F': .125}}
            for name, info in h.source.original.species_metadata.items()})
        source = live.original.compile_source(raw, h.metadata, expected_source_hash=raw.metadata.source_hash,
            expected_source_rows_sha256=source_variant_census(raw)['source_rows_sha256'])
        tables = states.compile_tables(source, h.metadata)
        for owner in (0, 1):
            adapter, request = h.adapter(owner), h.request(owner)
            # Public arrival F overrides the sampler; own request M overrides
            # the species recipe even on still-unrevealed bench members.
            adapter.feed((f'|switch|p{2-owner}a: Beta|Beta, L80, F|100/100',))
            for row in request['side']['pokemon']:row['details'] += ', M'
            root = live.from_b2(adapter, request, source, h.metadata, address=DecisionAddress(1, 0, owner))
            before = [np.asarray(x).tobytes() for x in jax.tree.leaves(root.public)]
            samples = live.sample_worlds(root, source, tables, count=32, seed=29)
            self.assertTrue(samples.fully_constructed, [w.refusals for w in samples.worlds])
            hidden = []
            for world in samples.worlds:
                np.testing.assert_array_equal(world.state.gender[0, owner], [1]*6)
                self.assertEqual(int(world.state.gender[0, 1-owner, 0]), 2)
                hidden.extend(np.asarray(world.state.gender)[0, 1-owner, 1:].tolist())
                np.testing.assert_array_equal(world.public.gender, root.public.gender)
                np.testing.assert_array_equal(world.public.species, root.public.species)
                np.testing.assert_array_equal(world.public.physical_to_public[0, 1-owner], [0, -1, -1, -1, -1, -1])
                np.testing.assert_array_equal(world.public.present, root.public.present)
            self.assertEqual(set(hidden), {1, 2})
            self.assertEqual(before, [np.asarray(x).tobytes() for x in jax.tree.leaves(root.public)])


if __name__ == '__main__':
    unittest.main()
