import unittest

from gpu_engine.fresh200_native_material import MATERIAL_FIELDS, project_material
from fresh200_native_projector_test import boundary_with_fainted_sleeper
from fresh200_native_initial_test import fixture


def material_input():
    boundary, _ = boundary_with_fainted_sleeper()
    _, _, vocabulary, _ = fixture()
    teams = [[dict(moves=['Tackle', 'Hidden Power Ground'], species='duplicate', ability='Stench')
              for _ in range(6)] for _ in range(2)]
    boundary.history.update(events=[], transform_records=[])
    return boundary, teams, vocabulary


class NativeMaterialTest(unittest.TestCase):
    def test_hidden_power_uses_original_slot_identity(self):
        boundary, teams, vocabulary = material_input()
        mon = boundary.snapshot['battle']['sides'][0]['pokemon'][1]
        mon['lastMove'] = '[Move:hiddenpower]'
        values = project_material(boundary, teams=teams, vocabulary=vocabulary)
        self.assertEqual(set(values), MATERIAL_FIELDS)
        self.assertEqual(values['last_move'][0, 0, 1], 2)
        self.assertEqual(values['moves'][0, 0, 1].tolist(), [1, 2, 0, 0])
        mon['moveSlots'].reverse()
        with self.assertRaisesRegex(ValueError, 'positional constructor/Native move identity differs'):
            project_material(boundary, teams=teams, vocabulary=vocabulary)

    def test_transform_saved_base_and_donor_slots_are_independently_bound(self):
        boundary, teams, vocabulary = material_input()
        vocabulary['moves'].append('Recover')
        teams[1][0]['moves'] = ['Recover']
        donor = boundary.snapshot['battle']['sides'][1]['pokemon'][0]
        donor['moveSlots'] = [dict(id='recover', pp=10)]
        mon = boundary.snapshot['battle']['sides'][0]['pokemon'][0]
        mon.update(transformed=True, baseStoredStats=mon['storedStats'], baseHpType='Ground', baseHpPower=70,
                   baseMoveSlots=mon['moveSlots'], moveSlots=[dict(id='recover', pp=5)])
        cached = boundary.snapshot['witness']['nativeObservation']['state']['sides'][0]['pokemon'][0]['fields']
        species = dict(kind='Effect', object=61, id='duplicate', effectType='Pokemon')
        slots = dict(object=63, items=[dict(object=64 + i, fields=[['pp', slot['pp']]])
                                     for i, slot in enumerate(mon['baseMoveSlots'])])
        stats = dict(object=67, fields=[[key, value] for key, value in mon['baseStoredStats'].items()])
        cached.update(species=species, baseMoveSlots=slots, baseStoredStats=stats)
        saved = dict(baseMoveSlots=slots, baseStoredStats=stats, baseHpType='Ground', baseHpPower=70)
        boundary.history['transform_records'] = [
            dict(actor=[0, 0], donor=[1, 0], donor_effective_species=species, sequence=6, saved=saved),
        ]
        values = project_material(boundary, teams=teams, vocabulary=vocabulary)
        self.assertTrue(values['transformed'][0, 0])
        self.assertEqual(values['moves'][0, 0, 0].tolist(), [3, 0, 0, 0])
        self.assertEqual(values['transform_moves'][0, 0].tolist(), [1, 2, 0, 0])
        self.assertEqual(values['transform_pp'][0, 0].tolist(), [10, 15, 0, 0])
        mon['baseMoveSlots'][0]['pp'] += 1
        with self.assertRaisesRegex(ValueError, 'base restoration state differs'):
            project_material(boundary, teams=teams, vocabulary=vocabulary)
