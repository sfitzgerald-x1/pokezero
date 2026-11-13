"""The per-move probe's pure parts, the committed scenarios, its Showdown side, and its device mode.

DeviceModeTests runs a representative subset of the committed scenarios with --device's comparison: each case through
step_with_public as well, its engine state equal to step_with_chance's, and the device public carry equal to B2's fold
of the case's Showdown lines (gpu_engine/effect_probe_device.py). The full set runs as a pod script. Light by design:
step_with_public's CPU compile peaks far above the light tier at any optimization level, so the subset runs it eagerly
(Engine.enable_device(eager=True): the same program op by op, no step compile) on one lane; step_with_chance and B2's
fold stay jitted.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

from _showdown_root import requires_showdown, showdown_root
from gpu_engine import effect_probe as probe
from gpu_engine import effect_probe_device as device

ROOT = Path(__file__).resolve().parents[1]
NODE = os.environ.get('NODE') or shutil.which('node')
HAS_JAX = bool(importlib.util.find_spec('jax'))
POOL = os.environ.get('POKEZERO_GPU_TEAM_POOL')
# Device mode's light subset (eager steps are slow): a White Herb received by Trick with a negative stage, the WH10
# catcher's site (Trick's -item lines after the herb's -enditem / -clearnegativeboost, B2's broken pairing).
DEVICE_SUBSET = ('wh_trick_receive_negative',)
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

    def test_showdown_lines_are_validated_and_matched_as_one_ordered_run(self):
        steps = len(scenario()['script'])
        loaded = probe.load_scenario(self.write(scenario(showdown_lines={'0': ['|-item|p2a: X|Choice Band|[from] move: Trick']})))
        self.assertIn('0', loaded['showdown_lines'])
        for bad in ({str(steps): ['|x']}, {'0': []}, {'0': ['-item|p2a: X']}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):probe.load_scenario(self.write(scenario(showdown_lines=bad)))
        log = ['|move|a', '|-item|b', '|-enditem|c', '|-clearnegativeboost|c|[silent]', '|-item|c', '|upkeep']
        self.assertTrue(probe.lines_in_order(log, log[1:5]))
        self.assertFalse(probe.lines_in_order(log, [log[2], log[1], log[3], log[4]]))      # out of order
        self.assertFalse(probe.lines_in_order(log, [log[1], log[3]]))                       # not adjacent

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


def side_view(**mon):
    base = dict(present=True, species='raichu', level=83, fainted=False, hp_percent=100, status='', moves=[('thunderbolt', 1)])
    base.update(mon)
    return dict(mons=[base] + [None] * 5, active=dict(boosts=[0] * 7, volatiles=[]),
                side=dict(active_slot=0, moved=False, matchup={(0, 0): [0, 0]}))


class DevicePureTests(unittest.TestCase):
    def views(self):
        return {0: side_view(), 1: side_view(species='snorlax'), 'field': dict(turn=1, weather='')}

    def test_compare_views_keys_and_presence(self):
        self.assertEqual(device.compare_views(self.views(), self.views()), [])
        dev, b2 = self.views(), self.views()
        dev[1]['mons'][0]['shown_item'] = b2[1]['mons'][0]['shown_item'] = 'whiteherb'
        dev[1]['mons'][0]['shown_item'] = 'leftovers';dev[0]['active']['volatiles'] = ['confusion']
        dev[0]['side']['matchup'][(0, 0)] = [1, 0];dev['field']['turn'] = 2
        dev[0]['mons'][1] = dict(dev[0]['mons'][0])                           # a slot only the device shows
        fields = [d['field'] for d in device.compare_views(dev, b2)]
        self.assertEqual(fields, ['0:1:present', '0:active:volatiles', '0:matchup:0:0', '1:0:shown_item', 'field:turn'])
        self.assertEqual([device.field_kind(f) for f in fields], ['present', 'volatiles', 'matchup', 'shown_item', 'turn'])

    def test_float_fields_within_one_ulp_are_numeric_notes(self):
        dev, b2 = self.views(), self.views()
        for v in (dev, b2):v[1]['active']['last_damage_taken'] = 0.30000001192092896
        dev[1]['active']['last_damage_taken'] = 0.29999998211860657             # float32 30 * 0.01: one ulp low
        numeric = []
        self.assertEqual(device.compare_views(dev, b2, numeric), [])
        self.assertEqual([n['field'] for n in numeric], ['1:active:last_damage_taken'])
        dev[1]['active']['last_damage_taken'] = 0.29                             # a real difference stays a diff
        self.assertEqual([d['field'] for d in device.compare_views(dev, b2, [])], ['1:active:last_damage_taken'])
        dev[1]['active']['last_damage_taken'] = b2[1]['active']['last_damage_taken']
        dev[1]['active']['boosts'] = [1] + [0] * 6                               # integers never take the ulp class
        self.assertEqual([d['field'] for d in device.compare_views(dev, b2, [])], ['1:active:boosts'])

    def test_allowlist_is_keyed_by_scenario_case_step_field(self):
        diffs = [dict(field='1:0:hp_percent', b2=50, device=51), dict(field='1:0:status', b2='', device='par')]
        allow = {('wrap', 'base', 3, '1:0:hp_percent'): 'reason'}
        used = set()
        open_, allowed = device.apply_allowlist('wrap', 'base', 3, diffs, used, allow)
        self.assertEqual([d['field'] for d in open_], ['1:0:status']);self.assertEqual(allowed[0]['reason'], 'reason')
        self.assertEqual(used, set(allow))
        for key in (('wrap', 'base', 2), ('wrap', 'other', 3), ('yawn', 'base', 3)):
            self.assertEqual(len(device.apply_allowlist(*key, diffs, set(), allow)[0]), 2)
        for key, reason in device.ALLOWLIST.items():
            with self.subTest(key=key):
                self.assertEqual(len(key), 4);self.assertTrue(reason.strip())
                self.assertTrue((ROOT / 'gpu_engine/effect_probes' / f'{key[0]}.json').exists())

    def test_wire_lines_percent_and_field_tables(self):
        lines = ['|tier|[Gen 3] Custom Game', '|debug|x', '|raw|<div>1 custom rule</div>', '|move|p1a: A|Splash|p1a: A', '|-nothing',
                 '|turn|1']
        self.assertEqual(device.wire_lines(lines), [device.TIER_LINE, '|move|p1a: A|Splash|p1a: A', '|turn|1'])
        self.assertEqual([device._percent(n, d) for n, d in ((0, 0), (253, 253), (252, 253), (1, 253), (37, 100))], [0, 100, 99, 1, 37])
        names = [f[0] for f in device.FIELDS]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue({'shown_item', 'item_removed', 'item_mutated', 'item_original', 'boosts'} <= set(names))
        left = {n.strip() for key in device.LEFT_OUT for n in key.split(',')}
        self.assertFalse(left & set(names))


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

    def test_device_channels_are_each_players_stream_with_shared_percent(self):
        sd = probe.Showdown(NODE, showdown_root())
        try:
            data = probe.load_scenario(ROOT / 'gpu_engine/effect_probes/thunderbolt.json')
            prepared = sd.ask(cmd='prepare', teams=data['teams'])
            started = sd.ask(cmd='start', format=data['format'], seed=data['seed'], cases=1, sets=prepared['sets'], device=True)
            p1, p2 = started['channels'][0]
            self.assertIn('|turn|1', p1);self.assertIn('|turn|1', p2)
            self.assertFalse(any(line.startswith('|split|') for line in p1 + p2))
            mine = [x for x in p1 if x.startswith('|switch|p1a:')];theirs = [x for x in p2 if x.startswith('|switch|p1a:')]
            self.assertNotEqual(mine[0].rsplit('|', 1)[1], '100/100')            # p1's own exact HP (the split's secret)
            self.assertTrue(theirs[0].endswith('|100/100'))                     # HP Percentage Mod's shared percent
            tape, _ = probe.step_inputs(data, dict(assignment=dict(par=0, full_para=99)), 0)
            reply = sd.ask(cmd='step', tapes=[dict(schema=probe.TAPE_SCHEMA, nonce='t:0', **tape)], choices=[['move 1', 'move 1']],
                           live=[True])
            p1, p2 = reply['results'][0]['channels']
            self.assertIn('|turn|2', p1);self.assertNotIn('|turn|1', p1)          # only the step's new lines
            self.assertTrue(any(x.startswith('|-damage|p2a:') and x.endswith('/100') for x in p1))
        finally:
            sd.close()


@requires_showdown()
@unittest.skipUnless(NODE and HAS_JAX and POOL, 'requires node, JAX and POKEZERO_GPU_TEAM_POOL')
class DeviceModeTests(unittest.TestCase):
    """A representative subset in device mode: state results unchanged and every case EXACT on the public path."""
    @classmethod
    def setUpClass(cls):
        cls.metadata, _ = probe.load_pool(POOL, None)
        cls.scenarios = [probe.load_scenario(ROOT / f'gpu_engine/effect_probes/{name}.json') for name in DEVICE_SUBSET]
        lanes = max(len(probe.enumerate_cases(s)) for s in cls.scenarios)
        cls.sd = probe.Showdown(NODE, showdown_root())
        cls.device = probe.Device.for_scenarios(cls.scenarios, cls.sd, showdown_root(), cls.metadata, lanes)
        cls.engine = probe.Engine(cls.metadata, lanes);cls.engine.enable_device(cls.device.belief_tables, eager=True)

    @classmethod
    def tearDownClass(cls):
        cls.sd.close()

    def test_subset_is_exact_on_the_device_public_path(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario['move']):
                result = probe.run_scenario(scenario, self.engine, self.sd, self.metadata, device=self.device)
                self.assertTrue(result['expectation_met'], result['expectation_detail'])
                self.assertTrue(result['device']['state_equal'])
                self.assertTrue(result['device']['met'], result['device']['detail'] + ' ' + json.dumps(
                    [s for r in result['results'] for s in r['device']['steps'] if s['diffs']][:2], default=str)[:2000])
                self.assertGreater(result['device']['steps_compared'], result['cases'])


if __name__ == '__main__':
    unittest.main()
