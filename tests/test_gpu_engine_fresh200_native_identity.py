"""Native identity fixtures only; no Native battle or new corpus root."""
import copy
import unittest

from gpu_engine.fresh200_native_identity import PartyMap, reference_identifier


class NativeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.species = [['ditto', 'pikachu', 'ditto', 'snorlax', 'clefable', 'wobbuffet'], ['mrmime']*6]
        self.battle = dict(sides=[])
        for side, team in enumerate(('316254', '123456')):
            permutation = [int(value)-1 for value in team];pokemon = [None]*6
            for original, current in enumerate(permutation):
                pokemon[current] = dict(set=dict(species=self.species[side][original]), hp=100+original,
                    species='[Species:castform]')
            self.battle['sides'].append(dict(id='p'+str(side+1), n=side, team=team,
                pokemon=pokemon, active=['[Pokemon:p%da]' % (side+1)]))

    def test_original_permutation_distinguishes_duplicate_species(self):
        mapping = PartyMap(self.battle, self.species)
        self.assertEqual(mapping.resolve_pokemon('[Pokemon:p1c]'), (0, 0))
        self.assertEqual(mapping.resolve_pokemon('[Pokemon:p1f]'), (0, 2))
        self.assertEqual(mapping.mon(0, 0)['hp'], 100);self.assertEqual(mapping.mon(0, 2)['hp'], 102)
        self.assertEqual(mapping.active(0), 1)
        self.assertEqual(mapping.resolve_pokemon('[Pokemon:p2c]'), (1, 2))

    def test_current_effective_species_never_replaces_original_constructor_identity(self):
        mapping = PartyMap(self.battle, self.species)
        self.assertEqual(mapping.mon(0, 0)['species'], '[Species:castform]')
        self.assertEqual(mapping.mon(0, 0)['set']['species'], 'ditto')
        changed = copy.deepcopy(self.species);changed[0][0] = 'castform'
        with self.assertRaisesRegex(ValueError, 'constructor species'):PartyMap(self.battle, changed)

    def test_bad_permutation_topology_and_cross_side_active_fail(self):
        for field, value in (('team', '112345'), ('team', [1, 2, 3, 4, 5, 6]),
                ('pokemon', self.battle['sides'][0]['pokemon'][:5]), ('active', ['[Pokemon:p2a]']), ('n', True)):
            changed = copy.deepcopy(self.battle);changed['sides'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):PartyMap(changed, self.species)

    def test_noncanonical_or_unknown_pokemon_references_fail(self):
        mapping = PartyMap(self.battle, self.species)
        for value in ('[Pokemon:p1g]', '[Species:ditto]', 'p1a: Ditto', None, '[Pokemon:p3a]'):
            with self.subTest(value=value), self.assertRaises(ValueError):mapping.resolve_pokemon(value)
        for side, slot in ((True, 0), (0, -1), (2, 0), (0, 6)):
            with self.subTest(side=side, slot=slot), self.assertRaises(ValueError):mapping.mon(side, slot)

    def test_active_move_and_bare_ids_decode_without_display_name_guessing(self):
        self.assertEqual(reference_identifier({'move':'[Move:solarbeam]', 'hit':1}, 'Move'), 'solarbeam')
        self.assertEqual(reference_identifier({'move':'[Move:solarbeam]', 'id':'solarbeam'}, 'Move'), 'solarbeam')
        self.assertEqual(reference_identifier('hiddenpowerfire', 'Move'), 'hiddenpowerfire')
        self.assertEqual(reference_identifier('[Species:castformsnowy]', 'Species'), 'castformsnowy')
        self.assertIsNone(reference_identifier(None, 'Move', allow_none=True))
        for value in ({'move':'[Move:solarbeam]', 'id':'tackle'}, '[Species:solarbeam]', 'Hidden Power Fire', None):
            with self.subTest(value=value), self.assertRaises(ValueError):reference_identifier(value, 'Move')


if __name__ == '__main__':unittest.main()
