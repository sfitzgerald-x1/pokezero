from dataclasses import replace
import unittest

from gpu_engine.fresh200_native_archive import RecordedBoundary
from gpu_engine.fresh200_native_phase import PhaseProof, retained_inputs
from gpu_engine.fresh200_native_projector import CORE_FIELDS, _major_status, project_core
from fresh200_native_initial_test import fixture


def boundary_with_fainted_sleeper():
    snapshot, _, _, constructor = fixture()
    phase = PhaseProof('replacement-before-residual', 7, (3, 5), (True, False),
                       True, 1, (1, 2), (0, 4), 7)
    actors = []
    for side, team in enumerate(snapshot['battle']['sides']):
        actors.append(dict(pokemon=[
            dict(identity=dict(side=side, original=original),
                 fields=dict(hp=mon['hp'], status=mon['status'], fainted=mon['fainted'], statusState=None))
            for original, mon in enumerate(team['pokemon'])
        ]))
    snapshot['battle']['sides'][0]['pokemon'][0].update(hp=0, fainted=True, status='fnt')
    actors[0]['pokemon'][0]['fields'].update(hp=0, fainted=True, status='fnt')
    witness = snapshot['witness']
    witness['nativeObservation'] = dict(state=dict(sides=actors))
    witness['receipt']['boundary'] = {name: witness[name] for name in ('nativeObservation', 'elidedProperties')}
    sleep = dict(object=43, fields=[['id', 'slp'], ['time', 2], ['skippedTime', 1], ['startTime', 3]])
    history = dict(source_registry_hash='a'*64, last_action_kind=[2, 2], last_action_index=[1, 2],
                   retained_status=[dict(actor=[0, 0], status='slp', state=sleep, sequence=6)],
                   rest_state_objects=[43], wish_amount=[0, 0])
    cleared = replace(phase, pending_residual=False)
    retained = retained_inputs(cleared, None)
    retained['draws'][1][0] = 5
    return RecordedBoundary(7, snapshot, phase, history, (), (), 0, retained), constructor.species


class NativeBoundaryCoreTest(unittest.TestCase):
    def test_retained_major_status_and_native_phase(self):
        boundary, species = boundary_with_fainted_sleeper()
        values = project_core(boundary, original_species=species)
        self.assertEqual(set(values), CORE_FIELDS)
        self.assertEqual(values['hp'][0, 0, 0], 0)
        self.assertEqual(values['status'][0, 0, 0], 5)
        self.assertEqual(values['status_counter'][0, 0, 0], 2)
        self.assertEqual(values['sleep_skipped'][0, 0, 0], 1)
        self.assertTrue(values['rest_origin'][0, 0, 0])
        self.assertEqual(values['queued_actor'].tolist(), [1])
        self.assertEqual(values['queued_stall'].tolist(), [[0, 4]])
        self.assertEqual(values['residual_draws'][0, 1, 0], 5)

    def test_physical_status_codes_follow_engine_state(self):
        for status, code in (('', 0), ('par', 1), ('brn', 2), ('psn', 3), ('frz', 6)):
            with self.subTest(status=status):
                self.assertEqual(_major_status(status, None, set())['status'], code)
        toxic = {'object': 17, 'fields': [['id', 'tox'], ['stage', 3]]}
        self.assertEqual(_major_status('tox', toxic, set()),
                         dict(status=4, status_counter=3, sleep_skipped=0, rest_origin=False))
        # Frozen root 3/p1 POST1: the incoming Relicanth has just used Rest.
        rest = {'object': 18, 'fields': [['id', 'slp'], ['time', 3], ['skippedTime', 0], ['startTime', 3]]}
        self.assertEqual(_major_status('slp', rest, {18}),
                         dict(status=5, status_counter=3, sleep_skipped=0, rest_origin=True))

    def test_missing_retained_status_and_mismatched_boundary_are_fatal(self):
        boundary, species = boundary_with_fainted_sleeper()
        with self.assertRaisesRegex(ValueError, 'pre-faint major-status'):
            project_core(replace(boundary, history=dict(boundary.history, retained_status=[])), original_species=species)
        with self.assertRaisesRegex(ValueError, 'same boundary'):
            project_core(replace(boundary, phase=replace(boundary.phase, snapshot_sequence=8)), original_species=species)


class NativeFaintMarkerTest(unittest.TestCase):
    def test_fainted_replacement_clears_status_only_with_closed_clear_witness(self):
        boundary, species = boundary_with_fainted_sleeper()
        survivor = project_core(boundary, original_species=species)
        snapshot = boundary.snapshot
        mon = snapshot['battle']['sides'][0]['pokemon'][0]
        cached = snapshot['witness']['nativeObservation']['state']['sides'][0]['pokemon'][0]['fields']
        mon['status'] = cached['status'] = ''
        prior, = boundary.history['retained_status']
        for clear in (None, 6, 8, True):
            with self.subTest(clear=clear):
                history = dict(boundary.history, retained_status=[dict(prior, replacement_clear_sequence=clear)])
                with self.assertRaisesRegex(ValueError, 'differs from the retained pre-faint status'):
                    project_core(replace(boundary, history=history), original_species=species)
        history = dict(boundary.history, retained_status=[dict(prior, replacement_clear_sequence=7)])
        projected = project_core(replace(boundary, history=history), original_species=species)
        self.assertNotEqual(survivor['status'][0, 0, 0], 0)
        self.assertEqual(survivor['status_counter'][0, 0, 0], 2)
        self.assertTrue(survivor['rest_origin'][0, 0, 0])
        self.assertEqual(projected['status'][0, 0, 0], 0)
        self.assertEqual(projected['status_counter'][0, 0, 0], 0)
        self.assertEqual(projected['sleep_skipped'][0, 0, 0], 0)
        self.assertFalse(projected['rest_origin'][0, 0, 0])

    def test_fainted_member_can_retain_the_native_major_status_before_fnt_marker(self):
        boundary, species = boundary_with_fainted_sleeper()
        snapshot = boundary.snapshot
        mon = snapshot['battle']['sides'][0]['pokemon'][0]
        cached = snapshot['witness']['nativeObservation']['state']['sides'][0]['pokemon'][0]['fields']
        mon['status'] = cached['status'] = 'slp'
        result = project_core(boundary, original_species=species)
        self.assertEqual(result['hp'][0, 0, 0], 0)
        self.assertEqual(result['status'][0, 0, 0], 5)
        self.assertEqual(result['status_counter'][0, 0, 0], 2)

        mon['status'] = cached['status'] = 'par'
        with self.assertRaisesRegex(ValueError, 'differs from the retained pre-faint status'):
            project_core(boundary, original_species=species)
