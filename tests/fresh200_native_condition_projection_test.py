from copy import deepcopy
from dataclasses import replace
import unittest

from gpu_engine.fresh200_native_conditions import CONDITION_FIELDS, project_conditions, project_trapped
from gpu_engine.fresh200_native_ledger import Frame, NativeHistory
from gpu_engine.fresh200_native_material import project_material
from fresh200_native_material_test import material_input


def conditions_input():
    boundary, teams, vocabulary = material_input()
    snapshot = boundary.snapshot
    members = snapshot['witness']['nativeObservation']['state']['sides']
    for side, team in enumerate(members):
        for original, member in enumerate(team['pokemon']):
            token = 100 + side * 100 + original * 10
            native = snapshot['battle']['sides'][side]['pokemon'][original]
            member['fields'].update(volatiles=dict(object=token, fields=[]), moveSlots=dict(object=token+1, items=[
                dict(object=token+2+index, fields=list(map(list, slot.items())))
                for index, slot in enumerate(native['moveSlots'])
            ]))
    boundary.history.update(disable_causes=[], encore_slots=[], charge_records=[])
    material = project_material(boundary, teams=teams, vocabulary=vocabulary)
    species = tuple(tuple(mon['species'] for mon in team) for team in teams)
    return boundary, members, dict(original_species=species, vocabulary=vocabulary, material=material)


class NativeConditionTest(unittest.TestCase):
    def test_encore_is_bound_to_its_actual_slot_without_double_counting_disable(self):
        boundary, members, inputs = conditions_input()
        actor = 1, 0
        mon = boundary.snapshot['battle']['sides'][1]['pokemon'][0]
        cache = members[1]['pokemon'][0]['fields']
        state = dict(move='hiddenpower', duration=3)
        mon['volatiles']['encore'] = state
        cache['volatiles']['fields'].append(['encore', dict(object=500, fields=list(map(list, state.items())))])
        first, second = cache['moveSlots']['items']
        first['fields'][-1] = ['disabled', True]
        boundary.history['disable_causes'] = [dict(actor=list(actor), slot_object=first['object'], causes=[
            dict(effect=dict(id='encore'), physical_slot=0, sequence=5),
        ])]
        boundary.history['encore_slots'] = [
            dict(actor=list(actor), state_object=500, slot_object=second['object'], physical_slot=1, sequence=6),
        ]
        values = project_conditions(boundary, **inputs)
        self.assertEqual(set(values), CONDITION_FIELDS)
        self.assertEqual(values['encore_move'][0, 1, 0], 2)
        self.assertEqual(values['encore_counter'][0, 1, 0], 3)
        self.assertEqual(values['disabled'][0, 1, 0].tolist(), [False] * 4)
        boundary.history['encore_slots'][0]['slot_object'] = first['object']
        with self.assertRaisesRegex(ValueError, 'different physical slot'):
            project_conditions(boundary, **inputs)

    def test_root13_literal_seismic_toss_break_marker_has_no_physical_leaf(self):
        boundary, members, inputs = conditions_input()
        battle = boundary.snapshot['battle']
        battle['formatid'] = 'gen3customgame'
        target = battle['sides'][1]['pokemon'][0]
        cache = members[1]['pokemon'][0]['fields']
        wire = dict(id='substitutebroken', name='substitutebroken', target='[Pokemon:p2a]', effectOrder=12,
                    source='[Pokemon:p1a]', sourceSlot='p1a', sourceEffect='[Condition:substitute]', move='seismictoss')
        native = dict(wire, target=dict(kind='Pokemon', object=230, side=1, original=0),
                      source=dict(kind='Pokemon', object=90, side=0, original=0),
                      sourceEffect=dict(kind='Effect', effectType='Condition', id='substitute', object=21354))
        marker = {'object': 22894, 'fields': list(map(list, native.items()))}
        target['volatiles']['substitutebroken'] = wire
        cache['volatiles']['fields'].append(['substitutebroken', marker])
        values = project_conditions(boundary, **inputs)
        self.assertEqual(values['volatile'][0, 1, 0], 0)
        self.assertEqual(values['substitute_hp'][0, 1, 0], 0)
        wire['move'] = 'uturn'
        marker['fields'][-1][1] = 'uturn'
        with self.assertRaisesRegex(ValueError, 'admitted Seismic Toss source'):
            project_conditions(boundary, **inputs)

    def test_partial_trap_uses_literal_source_original_slot(self):
        boundary, members, inputs = conditions_input()
        target = boundary.snapshot['battle']['sides'][0]['pokemon'][2]
        target['volatiles']['partiallytrapped'] = dict(source='[Pokemon:p1b]', duration=2)
        cache = members[0]['pokemon'][2]['fields']
        cache['volatiles']['fields'].append(['partiallytrapped', dict(object=510, fields=[])])
        with self.assertRaisesRegex(ValueError, 'same-side source'):
            project_conditions(boundary, **inputs)
        target['volatiles']['partiallytrapped']['source'] = '[Pokemon:p2b]'
        result = project_conditions(boundary, **inputs)
        self.assertEqual(result['partial_trap_source'][0, 0, 2], 1)
        self.assertEqual(result['partial_trap_counter'][0, 0, 2], 2)


class NativeChoiceAndTrapTest(unittest.TestCase):
    def test_choice_band_item_witness_matches_the_live_lock_source(self):
        boundary, members, inputs = conditions_input()
        mon = boundary.snapshot['battle']['sides'][1]['pokemon'][0]
        mon['item'] = 'choiceband'
        mon['volatiles']['choicelock'] = dict(move='tackle')
        cache = members[1]['pokemon'][0]['fields']
        effect = dict(id='choiceband', effect_type='Item', object=600)
        source = dict(kind='Effect', id='choiceband', effectType='Item', object=600)
        cache['volatiles']['fields'].append(['choicelock', dict(object=601, fields=[['sourceEffect', source]])])
        slot = cache['moveSlots']['items'][1]
        slot['fields'][-1] = ['disabled', True]
        boundary.history['disable_causes'] = [dict(actor=[1, 0], slot_object=slot['object'], causes=[
            dict(effect=effect, physical_slot=1, sequence=5),
        ])]
        result = project_conditions(boundary, **inputs)
        self.assertEqual(result['choice_move'][0, 1, 0], 1)
        self.assertEqual(result['disabled'][0, 1, 0].tolist(), [False] * 4)
        source['object'] += 1
        with self.assertRaisesRegex(ValueError, 'differs from its live lock'):
            project_conditions(boundary, **inputs)

    def test_direct_ability_trap_is_represented_but_unexplained_overlap_is_fatal(self):
        boundary, members, inputs = conditions_input()
        sides = boundary.snapshot['battle']['sides']
        sides[0]['pokemon'][0]['hp'] = 80
        sides[1]['pokemon'][0]['ability'] = 'shadowtag'
        members[0]['pokemon'][0]['fields']['trapped'] = True
        members[1]['pokemon'][0]['fields']['trapped'] = False
        boundary.history.update(trap_causes=[], lock_results=[])
        boundary = replace(boundary, relations=(
            dict(kind='ordinary', physical_mask=[False]*9),
            dict(kind='ordinary', physical_mask=[False]*4+[True]*5),
        ))
        conditions = project_conditions(boundary, **inputs)
        args = dict(original_species=inputs['original_species'], conditions=conditions)
        self.assertEqual(project_trapped(boundary, **args)['trapped'].tolist(), [[False, False]])
        boundary.history['trap_causes'] = [dict(actor=[0, 0], causes=[
            dict(effect=dict(id='other'), sequence=6),
        ])]
        with self.assertRaisesRegex(ValueError, 'unexplained contribution'):
            project_trapped(boundary, **args)


class NativeRetiredLockTest(unittest.TestCase):
    def test_choice_disable_proof_survives_only_the_unchanged_base_slot(self):
        boundary, members, inputs = conditions_input()
        member = members[1]['pokemon'][1]
        subject = dict(kind='Pokemon', object=800, side=1, original=1)
        effect = dict(kind='Effect', id='choiceband', effectType='Item', object=801)
        member['fields']['volatiles']['fields'].append(
            ['choicelock', dict(object=802, fields=[['sourceEffect', effect]])]
        )
        before = deepcopy(dict(sides=members))
        slot = member['fields']['moveSlots']['items'][1]
        slot['fields'][-1] = ['disabled', True]
        slot['fields'].append(['disabledSource', 'Choice Band'])
        after = deepcopy(dict(sides=members))
        history = NativeHistory(battle_id='b', source_registry_hash='a'*64)
        history.initial = True
        enter = dict(subject=subject, before=before, arguments=dict(object=803, items=['hiddenpower', False, effect]))
        disable = Frame(10, None, 'Pokemon.disableMove', 2, enter, dict(after=after), 3)
        history._disable(disable)

        member['fields']['volatiles']['fields'].clear()
        cleared = deepcopy(dict(sides=members))
        frame = Frame(11, None, 'Pokemon.clearVolatile', 4, dict(before=after), dict(after=cleared), 5)
        history._clear_move_restrictions(frame, (1, 1))
        boundary.history['disable_causes'] = history.snapshot()['disable_causes']
        self.assertEqual(project_conditions(boundary, **inputs)['disabled'][0, 1, 1].tolist(), [False]*4)

        slot['object'] += 1000
        with self.assertRaisesRegex(ValueError, 'lacks its actual slot causes'):
            project_conditions(boundary, **inputs)
