"""The per-move probe's pure parts, the committed scenarios, and its Showdown side."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

from _showdown_root import requires_showdown, showdown_root
from gpu_engine import effect_probe as probe

ROOT = Path(__file__).resolve().parents[1]
NODE = os.environ.get('NODE') or shutil.which('node')
SCENARIOS = sorted((ROOT / 'gpu_engine/effect_probes').glob('*.json'))
VOLATILES = dict(confusion=None, choicelock=None, encore=None, perishsong=None, yawn=None, partiallytrapped=None,
                 mustrecharge=None, destinybond=False, substitute=None, names=[])
VOCAB = dict(items=['_none', 'leftovers'], moves=['_none', 'tackle', 'encore'])


def scenario(**extra):
    team = [dict(species='Snorlax', moves=['tackle'], ability='Thick Fat')] * 6
    data = dict(schema=probe.SCENARIO_SCHEMA, move='tackle', teams=[team, team], script=[['move 1', 'move 1']] * 2)
    data.update(extra);return data


def mon(**extra):
    base = dict(hp=100, maxhp=100, status='', fainted=False, status_time=None, tox_stage=None, boosts=[0] * 7,
                item='leftovers', item_knocked=False, moves=['tackle'], pp=[56], volatiles=copy.deepcopy(VOLATILES))
    base.update(extra);return base


def snapshot():
    sides = [dict(active=0, mons=[mon() for _ in range(6)], mask=[True] + [False] * 3 + [True] * 5) for _ in range(2)]
    return dict(turn=1, ended=False, winner=None, sides=sides)


def view():
    z = lambda *shape: np.zeros(shape, np.int32)
    v = dict(active=z(2), hp=np.full((2, 6), 100), status=z(2, 6), status_counter=z(2, 6), item=np.ones((2, 6), np.int32),
             pp=np.full((2, 6, 4), 56), boosts=z(2, 6, 7), confusion_counter=z(2, 6), choice_move=z(2, 6),
             encore_counter=z(2, 6), encore_move=z(2, 6), substitute_hp=z(2, 6), volatile=np.zeros((2, 6), np.uint32),
             done=np.bool_(False), winner=np.int32(-1))
    v['pp'][..., 1:] = 0
    return v


MASK = np.array([[True] + [False] * 3 + [True] * 5] * 2)


class PureTests(unittest.TestCase):
    def write(self, data):
        path = Path(self.tmp) / 's.json';path.write_text(json.dumps(data));return path

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix='probe-test-')

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_scenario_validation(self):
        loaded = probe.load_scenario(self.write(scenario()))
        self.assertEqual(loaded['format'], probe.SLEEP_CLAUSE_FORMAT);self.assertEqual(loaded['expect'], dict(exact=True))
        bad = [scenario(schema='x'), dict(scenario(), teams=[[]] * 2), scenario(axes=[dict(name='a', field='draws.0.6', values=[100])]),
               scenario(axes=[dict(name='a', field='effect.0.3', values=[2])]), scenario(axes=[dict(name='a', field='nope.0', values=[0])]),
               scenario(axes=[dict(name='a', field='draws.0', values=[0])]), scenario(tape_steps={'0': {'secondary.2': 0}}),
               scenario(tape={'draws.1.4': 16}), scenario(tape={'phazing.0': 60}), scenario(tape={'initial_tie': 2})]
        for i, data in enumerate(bad):
            with self.subTest(i=i), self.assertRaises((ValueError, KeyError)):probe.load_scenario(self.write(data))

    def test_cases_are_the_axis_product_and_axes_apply_on_their_steps(self):
        s = scenario(axes=[dict(name='hit', field='draws.0.6', steps=[0], values=[0, 99]),
                           dict(name='roll', field='effect.1.2', values=[0, 1, 3])],
                     tape={'secondary.1': 7, 'phazing.0': 5}, tape_steps={'1': {'draws.1.4': 0}})
        cases = probe.enumerate_cases(s)
        self.assertEqual(len(cases), 6);self.assertEqual(cases[0]['case_id'], 'hit=0,roll=0')
        case = cases[-1]                                   # hit=99, roll=3
        tape, aux = probe.step_inputs(s, case, 0)
        self.assertEqual((tape['draws'][0][6], tape['effect'][1][2], tape['secondary'][1], aux['phazing'][0]), (99, 3, 7, 5))
        tape, _ = probe.step_inputs(s, case, 1)
        self.assertEqual((tape['draws'][0][6], tape['draws'][1][4], tape['effect'][1][2]), (probe.DEFAULT_ROW[6], 0, 3))
        self.assertEqual(probe.enumerate_cases(scenario()), [dict(case_id='base', assignment={})])

    def test_choices_map_to_engine_actions_by_original_party_position(self):
        self.assertEqual(probe.engine_action('move 3', 0), 2)
        self.assertEqual([probe.engine_action(f'switch {k}', 0) for k in range(2, 7)], [4, 5, 6, 7, 8])
        self.assertEqual([probe.engine_action(f'switch {k}', 2) for k in (1, 2, 4, 6)], [4, 5, 6, 8])
        self.assertEqual(probe.engine_action('switch 3', 2), -1)
        self.assertEqual(probe.engine_action(None, 0), 0)

    def test_compare_lane_fields(self):
        self.assertEqual(probe.compare_lane(view(), snapshot(), VOCAB, mask=MASK), [])
        sd = snapshot();sd['sides'][1]['mons'][2]['hp'] = 90
        sd['sides'][0]['mons'][0]['volatiles']['encore'] = dict(duration=3, move='tackle')
        sd['sides'][0]['mons'][0].update(status='slp', status_time=2)
        sd['sides'][0]['mons'][3]['volatiles']['confusion'] = 4   # a benched mon's volatiles are not compared
        fields = {d['field'] for d in probe.compare_lane(view(), sd, VOCAB, mask=MASK)}
        self.assertEqual(fields, {'1:2:hp', '0:0:encore_counter', '0:0:encore_move', '0:0:status', '0:0:status_counter'})
        mask = MASK.copy();mask[1, 4] = False
        self.assertEqual([d['field'] for d in probe.compare_lane(view(), snapshot(), VOCAB, mask=mask)], ['1:mask'])

    def test_absent_engine_fields_compare_as_neutral_values(self):
        sd = snapshot();sd['sides'][0]['mons'][0]['volatiles']['perishsong'] = 3
        sd['sides'][0]['mons'][0]['volatiles']['destinybond'] = True
        sd['sides'][1]['mons'][2]['item_knocked'] = True
        sd['sides'][1]['mons'][0]['volatiles']['partiallytrapped'] = dict(duration=4, source=0)
        fields = [d['field'] for d in probe.compare_lane(view(), sd, VOCAB, mask=MASK)]
        self.assertEqual(sorted(fields), ['0:0:destiny_bond', '0:0:perish_counter', '1:0:partial_trap_counter',
                                          '1:0:partial_trap_source', '1:2:item_knocked'])
        v = view();v['perish_counter'] = np.zeros((2, 6), np.int32);v['perish_counter'][0, 0] = 3;v['volatile'][0, 0] = 64
        v['item_knocked'] = np.zeros((2, 6), bool);v['item_knocked'][1, 2] = True
        v['partial_trap_counter'] = np.zeros((2, 6), np.int32);v['partial_trap_counter'][1, 0] = 4
        v['partial_trap_source'] = np.full((2, 6), -1, np.int32);v['partial_trap_source'][1, 0] = 0
        self.assertEqual(probe.compare_lane(v, sd, VOCAB, mask=MASK, destiny_bond=64), [])

    def test_require_fields_are_validated(self):
        loaded = probe.load_scenario(self.write(scenario(require_fields=['perish_counter', 'destiny_bond'])))
        self.assertEqual(loaded['require_fields'], ['perish_counter', 'destiny_bond'])
        with self.assertRaises(ValueError):probe.load_scenario(self.write(scenario(require_fields=['perish'])))
        engine = type('E', (), dict(fields=set(), lanes=16))()
        result = probe.run_scenario(dict(loaded, expect=dict(exact=True)), engine, None, None)
        self.assertEqual(result['status_counts'], {'MISSING_ENGINE_FIELDS': 1});self.assertFalse(result['expectation_met'])

    def test_expectation_judge(self):
        exact = dict(case_id='a', assignment=dict(hit=99), status='EXACT', steps=[dict(diffs=[])])
        wrong = dict(case_id='b', assignment=dict(hit=0), status='MISMATCH', steps=[dict(diffs=[dict(field='1:0:status')])])
        self.assertTrue(probe.judge(dict(expect=dict(exact=True)), [exact])[0])
        self.assertFalse(probe.judge(dict(expect=dict(exact=True)), [exact, wrong])[0])
        expect = dict(expect=dict(mismatch_fields=['status'], exact_cases=[dict(hit=99)]))
        self.assertTrue(probe.judge(expect, [exact, wrong])[0])
        stray = dict(wrong, steps=[dict(diffs=[dict(field='1:0:hp')])])
        self.assertFalse(probe.judge(expect, [exact, stray])[0])
        self.assertFalse(probe.judge(expect, [dict(exact, status='UNRESOLVED'), wrong])[0])

    def test_committed_scenarios_load(self):
        self.assertTrue(SCENARIOS)
        for path in SCENARIOS:
            with self.subTest(path=path.name):
                data = probe.load_scenario(path);self.assertEqual(path.stem, data['move'])
                self.assertLessEqual(len(probe.enumerate_cases(data)), 16)


@requires_showdown()
@unittest.skipUnless(NODE, 'requires node')
class ShowdownSideTests(unittest.TestCase):
    def test_prepare_start_and_step_protocol(self):
        sd = probe.Showdown(NODE, showdown_root())
        try:
            data = probe.load_scenario(ROOT / 'gpu_engine/effect_probes/thunderbolt.json')
            prepared = sd.ask(cmd='prepare', teams=data['teams'])
            self.assertEqual(prepared['sets'][0][0]['level'], 83)            # randbats level for Raichu
            self.assertEqual(prepared['sets'][0][0]['evs']['spe'], 85)
            self.assertEqual(len(prepared['rows'][1]), 6)
            self.assertEqual(prepared['vocabulary']['moves'][prepared['rows'][0][0]['moves'][0]], 'thunderbolt')
            started = sd.ask(cmd='start', format=data['format'], seed=data['seed'], cases=2, sets=prepared['sets'])
            self.assertEqual([s['sides'][0]['active'] for s in started['snapshots']], [0, 0])
            self.assertEqual({e['kind'] for e in started['audit']} - {'INERT_GENDER', 'INERT_QUICK_CLAW_ROLL'}, set())
            mirror = [prepared['sets'][1][0]] + prepared['sets'][0][1:]                      # equal-Speed leads
            tie = sd.ask(cmd='start', format=data['format'], seed=data['seed'], cases=1, sets=[mirror, prepared['sets'][1]])
            self.assertIn('INITIAL_RANDOM', {e['kind'] for e in tie['audit']})
            started = sd.ask(cmd='start', format=data['format'], seed=data['seed'], cases=2, sets=prepared['sets'])
            tape, _ = probe.step_inputs(data, dict(assignment=dict(par=0, full_para=99)), 0)
            tapes = [dict(schema=probe.TAPE_SCHEMA, nonce=f'test:{i}', **tape) for i in range(2)]
            tapes[1]['secondary'] = [99, 99]
            reply = sd.ask(cmd='step', tapes=tapes, choices=[['move 1', 'move 1']] * 2, live=[True, True])
            first, second = reply['results']
            self.assertIsNone(first['error']);self.assertEqual(first['snapshot']['sides'][1]['mons'][0]['status'], 'par')
            self.assertEqual(second['snapshot']['sides'][1]['mons'][0]['status'], '')
            self.assertIn('secondary', [e['role'] for e in first['audit'] if e['kind'] == 'COUPLED'])
            bad = sd.ask(cmd='step', tapes=tapes, choices=[['move 1', None]] * 2, live=[True, False])
            self.assertIn('SCRIPT_MISSING_CHOICE', bad['results'][0]['error']);self.assertIsNone(bad['results'][1])
        finally:
            sd.close()


if __name__ == '__main__':
    unittest.main()
