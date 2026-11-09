"""Screen-r3's arm fairness rule (depthn_live_fairness.RULE_R3): exact alive bands, the sparse merge at 120 and minimum support;
the default RULE's outputs pinned byte for byte; the r3 final gate and its CLI."""
import collections
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from pokezero import depthn_live_fairness as fairness
from scripts import depthn_fairness_final as F

STRATA = ('4x500ms', '2x1000ms')


def think_rows(rng, n):
    """n Foul Play think rows, a few malformed (no rate, a bool or NaN rate, no stratum, no or a negative round, no overlap, a round
    beyond the capture)."""
    rows = []
    for i in range(n):
        row = dict(round=i, stratum=rng.choice(STRATA), iterations_per_budget_second=rng.uniform(5e4, 9e5), wait_seconds=rng.uniform(0, 2),
                   overlap_seconds=rng.uniform(0, 1))
        kind = rng.random()
        if kind < .03: del row['iterations_per_budget_second']
        elif kind < .05: row['iterations_per_budget_second'] = True
        elif kind < .06: row['iterations_per_budget_second'] = float('nan')
        elif kind < .08: del row['stratum']
        elif kind < .10: row['round'] = None
        elif kind < .11: row['round'] = -2
        elif kind < .13: del row['overlap_seconds']
        elif kind < .15: row['round'] = i + 500
        rows.append(row)
    return rows


def write_capture(path, rng, rounds, owner='p1'):
    """An owner capture: turn lines (some missing or malformed), faints of either side, one request per round (the first names the
    owner), an event that is not WIRE and an interrupted last line."""
    lines, turn = [], 0
    for r in range(rounds):
        event = []
        if rng.random() < .8: turn += 1; event.append(f'|turn|{turn}')
        if rng.random() < .05: event.append('|turn|x')
        if rng.random() < .15: event.append(f"|faint|{rng.choice(('p1', 'p2'))}a: Mon")
        event.append('|request|' + json.dumps(dict(side=dict(id=owner)) if r == 0 else {}))
        lines.append(json.dumps(dict(event='WIRE', lines=event)))
        if rng.random() < .05: lines.append(json.dumps(dict(event='STATE', lines=['|request|{}', '|faint|p1a: X'])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n{"event": "WIR')


def rule_outputs():
    """The default RULE's outputs on seeded fixtures: per cell moves (panel.cell_moves, with and without a capture; fairness.moves
    without turns), round_turns, pool, reading (pooled, sparse, empty, a zero mean), flag and descriptive_waits. Key order counts."""
    from pokezero import depthn_live_panel as panel
    rng = random.Random(20261109);out = {}
    with TemporaryDirectory() as tmp:
        root = Path(tmp);arms = {'RAW': [], 'search': []}
        for ordinal in range(24):
            result = dict(source_game=dict(opponent_think=think_rows(rng, rng.randrange(10, 70)), opponent_think_failures=rng.choice((0, 0, 2, None))))
            if ordinal % 5:write_capture(root/'owning-wire'/f'{ordinal}.jsonl', rng, rng.randrange(5, 60), owner=('p1', 'p2')[ordinal % 2])
            moves = panel.cell_moves(root, ordinal, result);arms[('RAW', 'search')[ordinal % 2]].append(moves)
            out[f'cell {ordinal}'] = dict(cell_moves=moves, plain=fairness.moves(result),
                                          turns=fairness.round_turns(root/'owning-wire'/f'{ordinal}.jsonl'))
    raw, search = fairness.pool(arms['RAW']), fairness.pool(arms['search'])
    out['pool'] = dict(RAW=raw, search=search);value = fairness.reading(raw, search);out['reading'] = value
    out['flags'] = [fairness.flag(value), fairness.flag(value, acknowledged=dict(difference=.06, coverage_difference=.01))]
    out['waits'] = fairness.descriptive_waits(raw, search)
    def pooled(rows): return dict(rated=list(rows), moves=len(rows) + 3, band_from_round=1, waits=[])
    sparse_raw = pooled([('2x1000ms', b, 100. + b) for b in (0,)*10 + (1,)*40 + (2,)*40 + (3,)*5] + [('4x500ms', 0, 50.)]*12 + [('x', 1, 0.)]*31)
    sparse_search = pooled([('2x1000ms', b, 103. - b) for b in (0,)*40 + (1,)*40 + (2,)*40 + (3,)*40] + [('4x500ms', 0, 51.)]*12 + [('x', 1, 4.)]*31)
    out['sparse'] = fairness.reading(sparse_raw, sparse_search)
    out['empty'] = fairness.reading(pooled([]), pooled([]))
    out['one_arm'] = fairness.reading(sparse_raw, pooled([]))
    return out


def digest(value): return hashlib.sha256(json.dumps(value).encode()).hexdigest()


# rule_outputs() under the pre-r3 depthn_live_fairness and depthn_live_panel.cell_moves (5cc49523): the default RULE is byte-identical
RULE_GOLDEN = 'ea348c22a3b4835b85b637680ef7a7d81aebc029e39c1ca15c85f6512646b1a6'


def old_rounds(capture):
    """scripts/depthn_fairness_final.py's rounds before round_states replaced it (5cc49523), verbatim: the reference."""
    if capture is None or not Path(capture).exists():return None
    out, turn, faints, owner = [], 0, collections.Counter(), None
    for raw in Path(capture).read_bytes().split(b'\n'):
        try:row = json.loads(raw)
        except ValueError:continue
        if type(row) is not dict or row.get('event') != 'WIRE':continue
        for line in row.get('lines') or ():
            if line.startswith('|turn|'):
                value = line.split('|')[2];turn = int(value) if value.isascii() and value.isdigit() else turn
            elif line.startswith('|faint|'):faints[line.split('|')[2][:2]] += 1
            elif line.startswith('|request|'):
                if owner is None:
                    try:owner = (json.loads(line[len('|request|'):]).get('side') or {}).get('id')
                    except ValueError:owner = None
                other = 'p2' if owner == 'p1' else 'p1'
                out.append((turn, 6 - faints[owner], 6 - faints[other]))
    return out


def lopsided_capture(path, rounds, lopsided_from=None):
    """The owner's (p1) capture: one request per round at turn round + 1; from round lopsided_from on, the owner has 1 Pokemon alive
    and Foul Play (p2) 2 (ours 1, theirs 2)."""
    lines = []
    for r in range(rounds):
        event = [f'|turn|{r + 1}']
        if r == lopsided_from:event += ['|faint|p1a: X'] * 5 + ['|faint|p2a: Y'] * 4
        event.append('|request|' + json.dumps(dict(side=dict(id='p1'))))
        lines.append(json.dumps(dict(event='WIRE', lines=event)))
    path.parent.mkdir(parents=True, exist_ok=True);path.write_text('\n'.join(lines) + '\n')


def lopsided_run(root, *, pairs=40, rounds=30, lopsided_pairs=11, lopsided_from=20, search_scale=1., lopsided_scale=2., arm='DN2W16'):
    """COMPLETE cells (RAW ordinal 3p, search 3p + 1) with owner captures: Foul Play rates 100 (the search arm's times search_scale), and in
    the first lopsided_pairs pairs a sparse lopsided endgame (ours 1 vs theirs 2, from round lopsided_from: 110 moves per arm by default)
    where the search arm's rate is lopsided_scale times higher."""
    (root/'results').mkdir(parents=True, exist_ok=True)
    for pair in range(pairs):
        for ordinal, name in ((3*pair, 'RAW'), (3*pair + 1, arm)):
            lopsided = lopsided_from if pair < lopsided_pairs else None
            lopsided_capture(root/'owning-wire'/f'{ordinal}.jsonl', rounds, lopsided)
            think = [dict(round=r, stratum='2x1000ms', wait_seconds=.4, overlap_seconds=.6, iterations_per_budget_second=100. * (
                1. if name == 'RAW' else search_scale * (lopsided_scale if lopsided is not None and r >= lopsided else 1.))) for r in range(rounds)]
            (root/'results'/f'{ordinal}.json').write_text(json.dumps(dict(status='COMPLETE', cell=dict(arm=name, pair_ordinal=pair, ordinal=ordinal),
                source_game=dict(opponent_think=think))))


def pooled(rows, extra=0):
    return dict(rated=list(rows), moves=len(rows) + extra, band_from_round=0, waits=[])


class RuleTests(unittest.TestCase):
    def test_the_default_rule_is_byte_identical(self):
        self.assertEqual(digest(rule_outputs()), RULE_GOLDEN)

    def test_the_rules(self):
        self.assertEqual(fairness.RULE, dict(metric='ITERATIONS_PER_BUDGET_SECOND', strata='FOULPLAY_BUDGET_STRATUM_X_TURN_BAND',
            combine='POOLED_COUNT_WEIGHTED_RATIO_OF_MEANS', reference='RAW', ratio_tolerance=.05, coverage_difference_max=.05,
            min_moves_per_arm=1000, min_pairs=50, checkpoint_pairs=50, rearm_margin=.025, min_cell_moves=30))
        self.assertEqual(fairness.RULE_R3, dict(fairness.RULE, strata='FOULPLAY_BUDGET_STRATUM_X_TURN_BAND_X_ALIVE_BANDS',
            alive_bands=['1', '2', '3-4', '5-6'], min_cell_moves=120, min_support_moves=120))
        self.assertIs(type(fairness.RULE_R3['alive_bands']), list)
        self.assertEqual(json.loads(json.dumps(fairness.RULE_R3)), fairness.RULE_R3)   # a JSON-loaded declaration compares equal
        self.assertFalse(fairness.alive_labelled(fairness.RULE));self.assertTrue(fairness.alive_labelled(json.loads(json.dumps(fairness.RULE_R3))))
        with self.assertRaisesRegex(ValueError, 'alive bands'):fairness.alive_labelled(dict(fairness.RULE_R3, alive_bands=['1-2', '3-4', '5-6']))


class RoundStatesTests(unittest.TestCase):
    def test_round_states_equal_the_final_gates_old_rounds(self):
        self.assertIs(F.rounds, fairness.round_states)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in range(40):
                rng = random.Random(seed);path = root/f'{seed}.jsonl'
                write_capture(path, rng, rng.randrange(1, 80), owner=('p1', 'p2')[seed % 2])
                if seed % 7 == 3:   # an owner never named (a first request without side.id), or a first request that is not JSON
                    path.write_text(path.read_text().replace('{"side": {"id": "p2"}}', '{}', 1).replace('{"side": {"id": "p1"}}', 'x{', 1))
                with self.subTest(seed=seed):
                    self.assertEqual(fairness.round_states(path), old_rounds(path))
                    self.assertEqual([t for t, _, _ in fairness.round_states(path)], fairness.round_turns(path))
            self.assertIsNone(fairness.round_states(root/'missing.jsonl'));self.assertIsNone(fairness.round_states(None))
            lopsided_capture(root/'l.jsonl', 4, 2)
            self.assertEqual(fairness.round_states(root/'l.jsonl'), [(1, 6, 6), (2, 6, 6), (3, 1, 2), (4, 1, 2)])

    def test_the_alive_bands(self):
        self.assertEqual(fairness.ALIVE_BANDS, ('1', '2', '3-4', '5-6'))
        self.assertEqual([fairness.alive_band(n) for n in range(7)], ['1', '1', '2', '3-4', '3-4', '5-6', '5-6'])
        self.assertEqual([fairness.alive_band(n) for n in range(7)], [F.alive_bucket(n, 'exact') for n in range(7)])
        self.assertEqual(F.ALIVE_BANDS['exact'], fairness.ALIVE_BANDS)


class AliveMovesTests(unittest.TestCase):
    def test_alive_labels_and_their_fallback(self):
        rows = [dict(round=r, stratum='2x1000ms', iterations_per_budget_second=10. + r) for r in range(4)] + [
            dict(round=None, stratum='4x500ms', iterations_per_budget_second=1.), dict(round=1, iterations_per_budget_second=1.)]
        result = dict(source_game=dict(opponent_think=rows))
        states = [(1, 6, 5), (12, 2, 1), (25, 4, 0)]
        value = fairness.moves(result, states, alive=True)
        self.assertEqual(value['rated'], [('2x1000ms|ours 5-6|theirs 5-6', 0, 10.), ('2x1000ms|ours 2|theirs 1', 2, 11.),
            ('2x1000ms|ours 3-4|theirs 1', 3, 12.), ('2x1000ms|ours 5-6|theirs 5-6', 1, 13.),   # round 3 beyond the capture: turn 4, 6 alive
            ('4x500ms|ours 5-6|theirs 5-6', 0, 1.)])
        self.assertEqual((value['band_from_round'], value['moves']), (2, 6))
        none = fairness.moves(result, None, alive=True)   # no capture: every move by its round, both sides 6 alive
        self.assertEqual((none['band_from_round'], {s for s, _, _ in none['rated']}), (5, {'2x1000ms|ours 5-6|theirs 5-6', '4x500ms|ours 5-6|theirs 5-6'}))
        self.assertEqual(fairness.moves(result, [1, 12, 25]), fairness.moves(result, [1, 12, 25], alive=False))
        with self.assertRaisesRegex(ValueError, 'round_states'):fairness.moves(result, [1, 12, 25], alive=True)


class MinimumSupportTests(unittest.TestCase):
    def test_sparse_cells_are_left_out_and_the_support_is_reported(self):
        common, lopsided = '2x1000ms|ours 5-6|theirs 5-6', '2x1000ms|ours 1|theirs 2'
        raw = pooled([(common, 0, 100.)] * 130 + [(common, 1, 100.)] * 50 + [(common, 3, 100.)] * 500 + [(lopsided, 3, 100.)] * 110
                     + [('4x500ms|ours 5-6|theirs 5-6', 0, 0.)] * 150 + [('4x500ms|ours 1|theirs 1', 0, 50.)] * 5, extra=20)
        search = pooled([(common, 0, 100.)] * 130 + [(common, 1, 100.)] * 50 + [(common, 3, 100.)] * 500 + [(lopsided, 3, 200.)] * 110
                        + [('4x500ms|ours 5-6|theirs 5-6', 0, 7.)] * 150 + [('4x500ms|ours 1|theirs 1', 0, 50.)] * 5)
        old = fairness.reading(raw, search)   # RULE's merge and no support: the lopsided cell's ratio 2 enters
        self.assertEqual(fairness.flag(old), 'ratio');self.assertNotIn('supported_cells', old)
        self.assertTrue(all('supported' not in c for c in old['cells']))
        value = fairness.reading(raw, search, fairness.RULE_R3)
        self.assertEqual(list(value)[:len(old)], list(old))   # RULE's keys first, in RULE's order
        self.assertEqual(list(value)[len(old):], ['supported_cells', 'unsupported_cells', 'supported_moves', 'supported_share'])
        self.assertEqual(value['merge_map'][common], [['1-3'], ['4-9', '20+']])   # merged at 120 within the (budget, ours, theirs) group
        self.assertEqual(old['merge_map']['2x1000ms|ours 5-6|theirs 5-6'], [['1-3'], ['4-9'], ['20+']])   # RULE merges at 30
        cells = {(c['stratum'], tuple(c['bands'])): c for c in value['cells']}
        self.assertEqual({k: c['supported'] for k, c in cells.items()}, {
            (common, ('1-3',)): True, (common, ('4-9', '20+')): True, (lopsided, ('20+',)): False,
            ('4x500ms|ours 1|theirs 1', ('1-3',)): False, ('4x500ms|ours 5-6|theirs 5-6', ('1-3',)): False})   # the last: no ratio (RAW mean 0)
        self.assertEqual((value['supported_cells'], value['unsupported_cells'], value['excluded_cells']), (2, 3, 1))   # excluded: no ratio only
        self.assertEqual(value['excluded_cells'], fairness.reading(raw, search, dict(fairness.RULE_R3, min_support_moves=0))['excluded_cells'])
        self.assertEqual(value['weighted_ratio'], 1.);self.assertIsNone(fairness.flag(value, fairness.RULE_R3))
        self.assertEqual(value['supported_moves'], dict(RAW=680, search=680))
        self.assertEqual(value['supported_share'], dict(RAW=680/945, search=680/945))
        self.assertEqual(value['moves'], dict(RAW=945, search=945));self.assertEqual(value['all_moves'], dict(RAW=965, search=945))

    def test_the_boundary_and_an_empty_reading(self):
        def at(n):
            return fairness.reading(pooled([('s|ours 1|theirs 1', 0, 100.)] * n + [('s|ours 5-6|theirs 5-6', 0, 100.)] * 500),
                                    pooled([('s|ours 1|theirs 1', 0, 300.)] * n + [('s|ours 5-6|theirs 5-6', 0, 100.)] * 500), fairness.RULE_R3)
        self.assertEqual(at(119)['weighted_ratio'], 1.);self.assertEqual(at(119)['unsupported_cells'], 1)
        self.assertAlmostEqual(at(120)['weighted_ratio'], (240*3 + 1000)/1240);self.assertEqual(at(120)['unsupported_cells'], 0)
        empty = fairness.reading(pooled([]), pooled([]), fairness.RULE_R3)
        self.assertEqual((empty['weighted_ratio'], empty['supported_cells'], empty['supported_share']), (None, 0, dict(RAW=None, search=None)))
        sparse = fairness.reading(pooled([('s', 0, 1.)] * 10), pooled([('s', 0, 1.)] * 10), fairness.RULE_R3)   # nothing supported
        self.assertEqual((sparse['weighted_ratio'], sparse['difference'], sparse['excluded_cells']), (None, None, 0))
        self.assertIsNone(fairness.flag(sparse, fairness.RULE_R3))


class FinalGateR3Tests(unittest.TestCase):
    def setUp(self): self.directory = TemporaryDirectory(); self.root = Path(self.directory.name)

    def tearDown(self): self.directory.cleanup()

    def test_a_sparse_lopsided_cell_voids_the_exact_gate_but_not_r3(self):
        lopsided_run(self.root)
        exact, value = F.final_gate(self.root, 'DN2W16', 500, bands='exact'), F.final_gate_r3(self.root, 'DN2W16', 500)
        self.assertFalse(exact['within']);self.assertAlmostEqual(exact['finer']['difference'], 220/2400)   # ratio 2 on 220 of 2,400 moves
        self.assertTrue(value['within']);self.assertEqual(value['difference'], 0.)
        self.assertEqual(value['arm_fairness'], fairness.RULE_R3);self.assertEqual(value['limit'], 500);self.assertEqual(value['tolerance'], .05)
        self.assertEqual(value['supported_share'], dict(RAW=1090/1200, search=1090/1200))
        reading = value['reading']
        self.assertEqual((reading['supported_cells'], reading['unsupported_cells'], reading['moves']), (4, 1, dict(RAW=1200, search=1200)))
        unsupported, = [c for c in reading['cells'] if not c['supported']]
        self.assertEqual((unsupported['stratum'], unsupported['raw_moves'], unsupported['ratio']), ('2x1000ms|ours 1|theirs 2', 110, 2.))
        self.assertIn('RULE_R3', value['rule']);self.assertEqual(reading['band_from_round'], dict(RAW=0, search=0))
        (self.root/'owning-wire'/'0.jsonl').rename(self.root/'owning-wire'/'moved.jsonl')   # a RAW cell without its capture: 6 alive, by round
        reading = F.final_gate_r3(self.root, 'DN2W16', 500)['reading']
        self.assertEqual((reading['band_from_round'], reading['moves']), (dict(RAW=30, search=0), dict(RAW=1200, search=1200)))

    def test_a_uniform_deficit_still_voids_and_cells_outside_the_gate_are_left_out(self):
        lopsided_run(self.root, search_scale=.9)
        value = F.final_gate_r3(self.root, 'DN2W16', 500)
        self.assertFalse(value['within']);self.assertAlmostEqual(value['difference'], -.1)
        for pair in (600, 601):   # outside a 500-pair screen: a 50% deficit counts only at 2,000
            (self.root/'results'/f'{3*pair + 1}.json').write_text(json.dumps(dict(status='COMPLETE', cell=dict(arm='DN2W16', pair_ordinal=pair, ordinal=3*pair + 1),
                source_game=dict(opponent_think=[dict(round=r, stratum='2x1000ms', iterations_per_budget_second=5.) for r in range(200)]))))
        (self.root/'results'/'9999.json').write_text(json.dumps(dict(status='UNKNOWN', cell=dict(arm='DN2W16', pair_ordinal=3, ordinal=9999),
            source_game=dict(opponent_think=[dict(round=0, stratum='2x1000ms', iterations_per_budget_second=1.)] * 500))))
        self.assertEqual(F.final_gate_r3(self.root, 'DN2W16', 500), value)
        self.assertLess(F.final_gate_r3(self.root, 'DN2W16', 2000)['difference'], value['difference'])
        self.assertEqual(F.final_gate_r3(self.root, 'DN2W16', 2000)['missing_captures'], dict(per_arm={'RAW': 0, 'DN2W16': 2}, total=2, ordinals=[1801, 1804]))
        self.assertIsNone(F.final_gate_r3(self.root, 'DN2W8', 500)['difference'])   # no rated search moves: not within
        self.assertFalse(F.final_gate_r3(self.root, 'DN2W8', 500)['within'])

    def test_missing_captures_are_counted_at_the_path_the_panel_reads(self):
        lopsided_run(self.root)
        none = dict(per_arm={'RAW': 0, 'DN2W16': 0}, total=0, ordinals=[])
        self.assertEqual(F.final_gate_r3(self.root, 'DN2W16', 500)['missing_captures'], none)
        lopsided_capture(self.root/'owning-wire'/'4.jsonl', 5)   # a capture shorter than its 30 rounds: not missing (rounds beyond it, by round)
        value = F.final_gate_r3(self.root, 'DN2W16', 500)
        self.assertEqual((value['missing_captures'], value['reading']['band_from_round']), (none, dict(RAW=0, search=25)))
        (self.root/'owning-wire'/'0.jsonl').rename(self.root/'elsewhere.jsonl')   # one COMPLETE RAW cell without its capture
        value = F.final_gate_r3(self.root, 'DN2W16', 500)
        self.assertEqual(value['missing_captures'], dict(per_arm={'RAW': 1, 'DN2W16': 0}, total=1, ordinals=[0]))
        self.assertEqual(value['reading']['band_from_round'], dict(RAW=30, search=25))   # unchanged: report only; within is the rule's
        result = json.loads((self.root/'results'/'0.json').read_text())   # the result's own owning_wire is the path read
        (self.root/'results'/'0.json').write_text(json.dumps(dict(result, owning_wire=str(self.root/'elsewhere.jsonl'))))
        self.assertEqual(F.final_gate_r3(self.root, 'DN2W16', 500)['missing_captures'], none)
        result = json.loads((self.root/'results'/'3.json').read_text())
        (self.root/'results'/'3.json').write_text(json.dumps(dict(result, owning_wire=str(self.root/'nowhere.jsonl'))))   # its default exists
        self.assertEqual(F.final_gate_r3(self.root, 'DN2W16', 500)['missing_captures'], dict(per_arm={'RAW': 1, 'DN2W16': 0}, total=1, ordinals=[3]))
        (self.root/'owning-wire').rename(self.root/'gone')   # every capture missing: all counted, 20 ordinals listed
        value = F.final_gate_r3(self.root, 'DN2W16', 500)['missing_captures']
        self.assertEqual((value['per_arm'], value['total']), ({'RAW': 39, 'DN2W16': 40}, 79))   # cell 0 still reads elsewhere.jsonl
        self.assertEqual(value['ordinals'], sorted(o for p in range(40) for o in (3*p, 3*p + 1) if o != 0)[:20])

    def test_the_cli(self):
        lopsided_run(self.root)
        with redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(F.main(['final', '--run-directory', str(self.root), '--arm', 'DN2W16', '--rule', 'r3']), 0)
        self.assertEqual(json.loads(stdout.getvalue()), json.loads(json.dumps(F.final_gate_r3(self.root, 'DN2W16', 500))))
        with redirect_stdout(io.StringIO()) as stdout:
            F.main(['final', '--run-directory', str(self.root), '--arm', 'DN2W16', '--rule', 'r3', '--alive-bands', 'exact', '--limit', '20'])
        self.assertEqual(json.loads(stdout.getvalue())['limit'], 20)
        with redirect_stdout(io.StringIO()) as stdout:   # the defaults are the existing gate
            F.main(['final', '--run-directory', str(self.root), '--arm', 'DN2W16'])
        self.assertEqual(json.loads(stdout.getvalue()), json.loads(json.dumps(F.final_gate(self.root, 'DN2W16', 500))))
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            F.main(['final', '--run-directory', str(self.root), '--arm', 'DN2W16', '--rule', 'r3', '--alive-bands', 'coarse'])

    def test_a_windowed_diagnosis_under_r3_reproduces_its_flag(self):
        from pokezero import depthn_live_panel as panel
        lopsided_run(self.root, pairs=60, search_scale=1.07, lopsided_pairs=15)   # the window holds 50 lopsided moves per arm: unsupported
        entries = [(3*p + k, ('RAW', 'DN2W16')[k], p) for p in range(60) for k in (0, 1)]
        (self.root/'state').mkdir();(self.root/'state'/'completions.jsonl').write_text(
            ''.join(json.dumps(dict(ordinal=o, arm=a, pair_ordinal=p)) + '\n' for o, a, p in entries))
        window = entries[20:120]   # a flag acknowledged at log length 20, its checkpoint at 50 pairs
        moves = {arm: fairness.pool([panel.cell_moves(self.root, o, rule=fairness.RULE_R3) for o, a, _ in window if a == arm]) for arm in ('RAW', 'DN2W16')}
        recorded = fairness.reading(moves['RAW'], moves['DN2W16'], fairness.RULE_R3)
        self.assertEqual(fairness.flag(recorded, fairness.RULE_R3), 'ratio')
        flags = self.root/'state'/'fairness'/'screen';flags.mkdir(parents=True)
        (flags/'flag-1.json').write_text(json.dumps(dict(phase='screen', reason='ratio', window=20, pairs=50, reading=recorded)))
        out = self.root/'diag.json'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(F.main(['window', '--run-directory', str(self.root), '--arm', 'DN2W16', '--flag', '1', '--rule', 'r3', '--output', str(out)]), 0)
        value = json.loads(out.read_text())
        self.assertEqual((value['flag'], value['window']['end']), (dict(number=1, recorded_weighted_ratio=recorded['weighted_ratio'], reproduced=True), 120))
        self.assertEqual(value['arm_fairness'], fairness.RULE_R3)
        self.assertEqual(value['declared']['supported_share'], recorded['supported_share'])
        self.assertEqual(value['declared']['unsupported_cells'], recorded['unsupported_cells'])
        self.assertTrue(all('supported' in c for c in value['finer']['cell_ratios']))
        default = F.window_diagnosis(self.root, 'DN2W16', 20, pairs=50)   # the default rule: no rule named, no support fields
        self.assertNotIn('arm_fairness', default);self.assertNotIn('supported_share', default['declared'])
        self.assertAlmostEqual(value['declared']['difference'], .07);self.assertGreater(default['declared']['difference'], .08)


if __name__ == '__main__':
    unittest.main()
