import unittest

from gpu_engine.fresh200_native_field import FIELD_LEAVES, project_field
from fresh200_native_projector_test import boundary_with_fainted_sleeper


def replacement():
    boundary, species = boundary_with_fainted_sleeper()
    snapshot = boundary.snapshot
    snapshot['battle']['formatid'] = 'gen3customgame'
    snapshot['boundaryRequests'] = {'p1': {'forceSwitch': [True]}, 'p2': {'wait': True}}
    observed = snapshot['witness']['nativeObservation']['state']
    undefined = {'$native': 'undefined'}
    rule = dict(object=4, fields=[
        ['id', 'sleepclausemod'], ['source', None], ['sourceSlot', undefined],
        ['duration', undefined], ['effectOrder', 0],
    ])
    observed['fields'] = dict(ended=False, winner={'$native': 'missing'}, field=dict(
        weather='', terrain='', pseudoWeather=dict(object=1, fields=[['sleepclausemod', rule]]),
    ))
    for side in (0, 1):
        active = observed['sides'][side]['pokemon'][0]['fields']
        active.update(isStarted=side == 1, switchFlag=side == 0)
    return boundary, species


class NativeFieldTest(unittest.TestCase):
    def test_forced_request_is_bound_to_faint_and_live_bench(self):
        boundary, species = replacement()
        values = project_field(boundary, original_species=species)
        self.assertEqual(set(values), FIELD_LEAVES)
        self.assertEqual(values['entry_pending'].tolist(), [[False, False]])
        self.assertEqual(values['forced_switch'].tolist(), [[True, False]])
        self.assertEqual(values['baton_pass'].tolist(), [[False, False]])
        self.assertEqual(values['done'].tolist(), [False])
        self.assertEqual(values['winner'].tolist(), [-1])
        boundary.snapshot['boundaryRequests']['p1']['forceSwitch'][0] = False
        with self.assertRaisesRegex(ValueError, 'forced request and switch flag disagree'):
            project_field(boundary, original_species=species)

    def test_permanent_weather_requires_the_observed_ability_creation(self):
        boundary, species = replacement()
        snapshot = boundary.snapshot
        source = dict(kind='Pokemon', object=2, side=1, original=0)
        live = snapshot['witness']['nativeObservation']['state']['fields']['field']
        live.update(weather='sandstorm', weatherState=dict(object=3, fields=[
            ['id', 'sandstorm'], ['source', source], ['duration', 0],
        ]))
        snapshot['battle']['field'].update(weather='sandstorm', weatherState=dict(duration=0))
        creation = dict(weather='sandstorm', state_object=3, source=[1, 0], sequence=4,
                        effect=dict(effect_type='Ability', id='sandstream'))
        boundary.history['weather_creation'] = creation
        values = project_field(boundary, original_species=species)
        self.assertEqual(values['weather'].tolist(), [3])
        self.assertEqual(values['weather_counter'].tolist(), [0])
        creation['effect']['id'] = 'drought'
        with self.assertRaisesRegex(ValueError, 'matching ability creation'):
            project_field(boundary, original_species=species)


class NativeFormatConditionTest(unittest.TestCase):
    def test_permanent_format_rule_is_required_and_cannot_hide_material_effects(self):
        boundary, species = replacement()
        live = boundary.snapshot['witness']['nativeObservation']['state']['fields']['field']
        self.assertEqual(project_field(boundary, original_species=species)['weather'].tolist(), [0])
        rules = live['pseudoWeather']['fields']
        rules.append(['gravity', dict(object=9, fields=[])])
        with self.assertRaisesRegex(ValueError, 'unrepresented Native field rules'):
            project_field(boundary, original_species=species)
        rules.pop()
        rules[0][1]['fields'][3][1] = 3
        with self.assertRaisesRegex(ValueError, 'unexpected material state'):
            project_field(boundary, original_species=species)
