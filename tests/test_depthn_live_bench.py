import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import depthn_live_bench as B


def response(status='COMPLETE', cause=None, prepare=.5, kinds=(), agrees=True, classes=None, refusal='no'):
    return dict(status=status, cause=cause, refusal=None if status == 'COMPLETE' else refusal, action_index=0 if status == 'COMPLETE' else None,
        mean_q_agrees=agrees if status == 'COMPLETE' else None, constructed_worlds=8 if status == 'COMPLETE' else 3, decision_seconds=prepare + .4,
        timings=dict(queue_seconds=0., prepare_seconds=prepare, classify_seconds=.02, build_seconds=.05, price_seconds=.3, reduce_seconds=.01),
        world_refusals=[dict(world=1, kinds=list(kinds))] if kinds else [], root_classes=classes,
        root_class=None if classes is None else next((c for c in ('forced_switch', 'fresh_substitute', 'wish', 'sleep') if c in classes), 'other'),
        root_class_source=None if classes is None else 'protocol_facts')


class FakeSearch:
    W, rung, tree_commit = 16, 'DN2W16', 'abc'

    def __init__(self, responses):
        self.responses, self.seen, self.warmed, self.resets = list(responses), [], None, []

    def reset_sessions(self):
        self.resets.append(len(self.seen))

    def warmup(self, requests):
        self.warmed = list(requests);return dict(buckets={1: 1.})

    def process_many(self, requests):
        assert len(requests) == 1;self.seen.append(requests[0]);return [dict(self.responses.pop(0), request_id=requests[0]['request_id'])]


def mon(i, active=False, hp='100/100'):
    return dict(ident=f'p1: M{i}', details=f'M{i}, L80', condition=hp, active=active)


def own_request(single):
    """A literal own request with one legal action (a forced switch to the one living bench member) or nine (four moves,
    five switches)."""
    if single:raw = dict(forceSwitch=[True], side=dict(id='p1', name='a', pokemon=[mon(0, True, '0 fnt'), mon(1), *[mon(i, hp='0 fnt') for i in range(2, 6)]]))
    else:raw = dict(active=[dict(moves=[dict(move=f'm{i}', id=f'm{i}', pp=5, maxpp=5, disabled=False) for i in range(4)])],
        side=dict(id='p1', name='a', pokemon=[mon(i, i == 0) for i in range(6)]))
    return '|request|' + json.dumps(raw)


def live_request(i, single=False):
    line = own_request(single)
    return dict(battle_id=f'b{i}', owner='p1', stream_id=1, decision_index=i, generation=1, prefix_sha256=f'{i:064x}', lines=[line], own_request_line=line)


def address_request(i):
    return dict(battle_id=f'b{i}', owner='p1', stream_id=1, decision_index=i, generation=1, prefix_sha256=f'{i:064x}', lines=[f'|request|{i}'],
        own_request_line=f'|request|{i}')


class BenchTests(unittest.TestCase):
    def test_percentiles_nearest_rank(self):
        p = B.percentiles([float(x) for x in range(1, 101)] + [None, float('nan')])
        self.assertEqual((p['n'], p['p50'], p['p95'], p['p99'], p['max']), (100, 50., 95., 99., 100.))
        self.assertIsNone(B.percentiles([None]))

    def test_summary_counts_and_w16_flag(self):
        rows = [B.record('r', 0, ('a',), response(prepare=.5), 1.0), B.record('r', 1, ('b',), response(prepare=2.5), 3.0),
            B.record('r', 2, ('c',), response('REFUSED', 'constructor_refused', kinds=('X',)), .2), B.record('r', 3, ('d',), None, 'gone')]
        s = B.summarize(rows, 16)
        self.assertEqual((s['roots'], s['unavailable'], s['priced'], s['complete']), (4, 1, 3, 2))
        self.assertAlmostEqual(s['searched_share'], 2 / 3)
        self.assertEqual(s['causes'], {'constructor_refused': 1});self.assertEqual(s['world_refusal_kinds'], {'X': 1})
        self.assertEqual(s['stages']['wall_seconds']['p95'], 3.0)   # COMPLETE roots only
        self.assertEqual(s['w16_condition'], dict(p95_wall_seconds=3.0, bound=2.6, met=False))
        self.assertIsNone(B.summarize(rows, 8)['w16_condition'])

    def test_per_class_breakdown_counts_structural_failures_apart(self):
        rows = [B.record('r', 0, ('a',), response(prepare=.5, classes=['forced_switch', 'sleep']), 1.),
            B.record('r', 1, ('b',), response('REFUSED', 'server_error', classes=['forced_switch'], refusal='ValueError: one world source per frame'), .1),
            B.record('r', 2, ('c',), response('REFUSED', 'depthn_refused', classes=['wish'], refusal='WORLD_STRUCTURE_MISMATCH:world 0: missing x'), .1),
            B.record('r', 3, ('d',), response('REFUSED', 'constructor_refused', classes=[], refusal='constructor-root:X'), .1),
            B.record('r', 4, ('e',), response(prepare=1.5, classes=[]), 2.),
            B.record('r', 5, ('f',), response('REFUSED', 'class_refused', classes=['sleep'], refusal='ROOT_CLASS_REFUSED:sleep'), .1),
            B.record('r', 6, ('g',), None, 'gone'), B.record('r', 7, ('h',), response(prepare=.7), 1.2)]   # unavailable; never classified
        self.assertEqual([r['structural'] for r in rows], [False, True, True, False, False, False, False, False])
        s = B.summarize(rows, 8);c = s['classes']
        self.assertEqual(list(c), ['forced_switch', 'fresh_substitute', 'wish', 'sleep', 'other', 'unclassified'])
        got = {g: (x['roots'], x['priced'], x['complete'], x['structural'], x['clean_refusals']) for g, x in c.items()}
        self.assertEqual(got, dict(forced_switch=(2, 2, 1, 1, 0), fresh_substitute=(0, 0, 0, 0, 0), wish=(1, 1, 0, 1, 0), sleep=(2, 2, 1, 0, 1),
            other=(2, 2, 1, 0, 1), unclassified=(1, 1, 1, 0, 0)))   # a root in two classes counts in each; unavailable roots in none
        self.assertEqual((c['forced_switch']['searched_share'], c['fresh_substitute']['searched_share'], c['wish']['causes']), (.5, None, {'depthn_refused': 1}))
        self.assertEqual((c['forced_switch']['stages']['prepare_seconds']['p50'], c['other']['stages']['prepare_seconds']['p50']), (.5, 1.5))
        self.assertEqual(c['sleep']['stages']['classify_seconds']['max'], .02)
        self.assertEqual((s['structural'], s['structural_classes']), (2, ['forced_switch', 'wish']))
        md = B.markdown(dict(rung='DN2W8', W=8, tree_commit='t', warmup_seconds=1., warmup_count=1, saved_root_admission=True, rosters={'r': s}, pooled=s,
            structural_classes=s['structural_classes']))
        self.assertIn('| forced_switch | 2 | 2 | 0 | 1 | 0.500 | 0.500 | 1 | 0 |', md);self.assertIn('| prepare_seconds | 1 | 0.500 |', md)
        self.assertIn('other: stage (COMPLETE roots)', md);self.assertNotIn('fresh_substitute: stage', md)
        self.assertIn("structural_classes (pooled): ['forced_switch', 'wish']", md)

    def test_single_legal_roots_leave_the_declared_denominator(self):
        q = lambda legal: [0. if x else None for x in legal]
        rows = [B.record('r', 0, ('a',), dict(response(prepare=.5), q_values=q([True] * 9)), 1., request=live_request(0), turn=2),
            B.record('r', 1, ('b',), response('REFUSED', 'depthn_refused'), .3, request=live_request(1), turn=2),
            B.record('r', 2, ('c',), dict(response(prepare=.5), q_values=q([False] * 4 + [True] + [False] * 4)), 4., request=live_request(2, True), turn=25),
            B.record('r', 3, ('d',), response('REFUSED', 'constructor_refused'), 5., request=live_request(3, True), turn=25),
            B.record('r', 4, ('e',), dict(response(prepare=.5), q_values=q([True] * 8 + [False])), 2., request=live_request(4), turn=30),   # lanes disagree
            B.record('r', 5, ('f',), None, 'gone', turn=7)]
        self.assertEqual([(r['legal_count'], r['single_legal'], r['band']) for r in rows], [(9, False, '1-3'), (9, False, '1-3'), (1, True, '20+'),
            (1, True, '20+'), (9, False, '20+'), (None, False, '4-9')])
        self.assertEqual([r['legal_mask_agrees'] for r in rows], [True, None, True, None, False, None])
        s = B.summarize(rows, 8, (3., 1., 1., 1.))
        self.assertEqual((s['priced'], s['single_legal'], s['eligible'], s['searched'], s['complete']), (5, 2, 3, 2, 3))
        self.assertAlmostEqual(s['searched_share'], 2 / 3);self.assertAlmostEqual(s['searched_share_all'], 3 / 5)   # declared; all roots secondary
        self.assertEqual({b: (x['eligible'], x['searched']) for b, x in s['bands'].items()}, {'1-3': (2, 1), '4-9': (0, 0), '10-19': (0, 0), '20+': (1, 1)})
        self.assertAlmostEqual(s['searched_share_reweighted'], (3 * .5 + 1 * 1.) / 4)   # bands without eligible roots drop out of the weights
        self.assertEqual((s['wall_eligible']['n'], s['wall_eligible']['max'], s['stages']['wall_seconds']['n'], s['stages']['wall_seconds']['max']), (3, 2., 3, 4.))
        self.assertEqual(s['legal_mask_disagreements'], 1);self.assertEqual(s['classes']['unclassified']['single_legal'], 2)
        md = B.markdown(dict(rung='DN2W8', W=8, tree_commit='t', warmup_seconds=1., warmup_count=1, saved_root_admission=True, rosters={}, pooled=s,
            structural_classes=[]))
        self.assertIn('single-legal 2, eligible 3, COMPLETE 3 (searched share 0.667, declared: searched / eligible; all-roots share 0.600', md)
        self.assertIn('1-3 1/2 = 0.500', md);self.assertIn('reweighted by bands [3.0, 1.0, 1.0, 1.0]: 0.625', md)
        with self.assertRaises(Exception):B.band_weights('1,2,3')
        self.assertEqual(B.band_weights('4,3,2,1'), (4., 3., 2., 1.))

    def test_pooled_stages_cover_complete_roots_only(self):
        rows = [B.record('r', 0, ('a',), response(prepare=.5), 1.), B.record('r', 1, ('b',), response('REFUSED', 'depthn_refused', prepare=9.), 9.)]
        s = B.summarize(rows, 8)
        self.assertEqual((s['stages']['prepare_seconds']['n'], s['stages']['prepare_seconds']['max'], s['stages']['wall_seconds']['max']), (1, .5, 1.))
        self.assertEqual(s['wall_eligible']['max'], 9.)   # every eligible root, refused or not

    def test_options_refuse_other_buckets(self):
        base = ['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'x', '--vocabulary', 'v',
            '--expected-vocabulary-sha256', 'y', '--roster', 'r', 'z', '--out', 'o']
        with patch.object(B.server.evaluators, 'check_arguments'), self.assertRaises(SystemExit):
            B.options(base + ['--buckets', '1,2'])
        with patch.object(B.server.evaluators, 'check_arguments'), self.assertRaises(SystemExit):
            B.options(base)   # no champion archive
        args = B.parser().parse_args(base + ['--refuse-classes', 'wish,forced_switch'])   # the server's flag, passed through to build
        self.assertEqual((args.refuse_classes, B.parser().parse_args(base).refuse_classes), (('wish', 'forced_switch'), ()))

    def test_main_prices_each_root_alone_and_writes_reports(self):
        items = [(i, ('addr', i), live_request(i, single=i == 1), None, 4 + i) for i in range(3)] + [(3, ('addr', 3), None, 'OSError: gone', 20)]
        fake = FakeSearch([response(prepare=.5, classes=['wish']), response('REFUSED', 'server_error', classes=['forced_switch']),
            response(prepare=1.5, agrees=False, classes=[])])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            args = B.parser().parse_args(['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'x',
                '--vocabulary', 'v', '--expected-vocabulary-sha256', 'y', '--roster', 'r.json', 'z', '--out', str(out), '--warmup-count', '2',
                '--refuse-classes', 'sleep', '--band-weights', '1,2,3,4'])
            args.port, args.max_requests = 0, 1
            with patch.object(B, 'options', return_value=args), patch.object(B, 'roster_requests', return_value=items), \
                    patch.object(B.server, 'build', return_value=fake) as build, patch.object(B.server, 'header', return_value=dict(h=1)):
                self.assertEqual(B.main([]), 0)
            self.assertEqual(build.call_args.args[0].refuse_classes, ('sleep',))
            self.assertEqual(len(fake.warmed), 2)
            self.assertEqual([r['battle_id'] for r in fake.seen], ['b0', 'b1', 'b2'])
            self.assertTrue(all(r['rung'] == 'DN2W16' and r['budget_seconds'] == B.BUDGET for r in fake.seen))
            self.assertEqual(len({r['request_id'] for r in fake.seen}), 3)
            rows = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in rows], ['COMPLETE', 'REFUSED', 'COMPLETE', 'UNAVAILABLE'])
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual((summary['pooled']['complete'], summary['pooled']['priced'], summary['pooled']['mean_q_disagreements']), (2, 3, 1))
            self.assertIn('searched share 1.000, declared: searched / eligible; all-roots share 0.667', (out / 'summary.md').read_text())   # root 1 is single-legal
            self.assertEqual((summary['structural_classes'], summary['refuse_classes'], summary['pooled']['classes']['wish']['complete']), (['forced_switch'], ['sleep'], 1))
            self.assertEqual([r['root_classes'] for r in rows], [['wish'], ['forced_switch'], [], None])
            self.assertEqual([(r['band'], r['single_legal']) for r in rows], [('4-9', False), ('4-9', True), ('4-9', False), ('20+', False)])
            self.assertEqual((summary['pooled']['eligible'], summary['pooled']['searched_share'], summary['band_weights']), (2, 1., [1., 2., 3., 4.]))
            self.assertEqual(list(summary['rosters']), ['r.json@z']);self.assertEqual(summary['rosters_declared'], [dict(path='r.json', sha256='z')])
            with patch.object(B, 'options', return_value=args), self.assertRaises(FileExistsError):
                B.main([])   # never into an existing directory

    def test_sessions_reset_per_roster_so_a_shared_root_prices_in_each(self):
        shared = (0, ('addr', 0), live_request(0), None, 2)
        fake = FakeSearch([response(), response()])
        with tempfile.TemporaryDirectory() as tmp:
            args = B.parser().parse_args(['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'x',
                '--vocabulary', 'v', '--expected-vocabulary-sha256', 'y', '--roster', 'a/roster.json', 'z1', '--roster', 'b/roster.json', 'z2',
                '--out', str(Path(tmp) / 'o')])
            with patch.object(B, 'options', return_value=args), patch.object(B, 'roster_requests', return_value=[shared]), \
                    patch.object(B.server, 'build', return_value=fake), patch.object(B.server, 'header', return_value={}):
                self.assertEqual(B.main([]), 0)
            summary = json.loads((Path(tmp) / 'o' / 'summary.json').read_text())
        self.assertEqual(fake.resets, [0, 1])   # before each roster's first root
        self.assertEqual(list(summary['rosters']), ['a/roster.json@z1', 'b/roster.json@z2'])   # same file name, two rosters

    def test_turn_of_reads_the_last_turn_line(self):
        self.assertEqual(B.turn_of(['|start', '|turn|1', '|move|x', '|turn|12', '|request|{}']), 12)
        self.assertEqual(B.turn_of(['|start', '|request|{}']), 0)   # before the first turn
        self.assertIsNone(B.turn_of(['|turn|x']))
        self.assertEqual((B.band(12), B.band(0), B.band(None)), ('10-19', None, None))

    def test_latency_coverage_and_bands_over_eligible_decisions(self):
        rows = [dict(wall_seconds=w, prepare_seconds=w - .4, band=b) for w, b in ((1., '1-3'), (2.9, '1-3'), (4.85, '4-9'), (4.9, '4-9'), (9.9, '20+'), (20., '20+'))]
        lat = B.latency(rows)
        self.assertEqual(lat['n'], 6);self.assertAlmostEqual(lat['within_ladder'], 1 / 6)   # 2.9 + 0.15 overruns the 3.0 s cap
        self.assertEqual(lat['deadline_coverage'], {'5.0': 3 / 6, '10.0': 4 / 6, '15.0': 5 / 6})   # wall + 0.15 must fit: 4.85 fits 5, 4.9 does not
        self.assertEqual((lat['bands']['1-3']['wall']['n'], lat['bands']['10-19']['wall'], lat['wall']['max']), (2, None, 20.))
        self.assertEqual(B.latency([])['within_ladder'], None)

    def test_battle_requests_take_distinct_battles_in_roster_order_with_every_decision(self):
        selected = [dict(capture_path='w/2.jsonl'), dict(capture_path='w/1.jsonl'), dict(capture_path='w/2.jsonl'), dict(capture_path='w/3.jsonl')]
        def capture(path):
            return [dict(address_request(int(path[2]) * 10 + i), lines=[f'|turn|{i}', f'|request|{i}'], capture_row=i) for i in range(3)]
        with patch('pokezero.phase_d_ladder_panel.read_bound', return_value=dict(selected=selected)), \
                patch('pokezero.phase_d_owning_capture.read_capture', side_effect=capture), \
                patch('pokezero.phase_d_ladder_admission.address', side_effect=lambda row: (row['battle_id'], row['decision_index'])):
            got = B.battle_requests([('r.json', 'z')], 2)
            three = B.battle_requests([('r.json', 'z')], 3)
        self.assertEqual(list(got), ['w/2.jsonl', 'w/1.jsonl'])   # distinct, roster order, first two
        self.assertEqual(list(three), ['w/2.jsonl', 'w/1.jsonl', 'w/3.jsonl'])   # the repeated w/2 is one battle
        self.assertEqual([(o, t, r['battle_id']) for o, _, r, _, t in got['w/2.jsonl']], [(0, 0, 'b20'), (1, 1, 'b21'), (2, 2, 'b22')])
        self.assertTrue(all(set(r) == set(B.KEYS) for _, _, r, _, _ in got['w/1.jsonl']))

    def test_session_mode_turns_saved_root_admission_off_and_pools_only(self):
        base = ['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'x', '--vocabulary', 'v',
            '--expected-vocabulary-sha256', 'y', '--roster', 'r.json', 'z', '--champion-archive', 'a', '--champion-archive-sha256', 'c' * 64,
            '--champion-checkpoint-sha256', 'd' * 64]
        with patch.object(B.server.evaluators, 'check_arguments'):
            self.assertTrue(B.options(base + ['--out', 'o']).saved_root_admission)
            self.assertFalse(B.options(base + ['--out', 'o', '--sessions', '3']).saved_root_admission)
            with self.assertRaises(SystemExit):B.options(base + ['--out', 'o', '--sessions', '0'])
            with self.assertRaises(SystemExit):B.options(base + ['--out', 'o', '--sessions', '2', '--limit', '5'])
        battles = {'w/1.jsonl': [(i, ('a', i), live_request(i, single=i == 1), None, i) for i in range(3)],
            'w/2.jsonl': [(i, ('b', i), live_request(10 + i), None, 12) for i in range(2)]}
        fake = FakeSearch([response(prepare=p) for p in (.5, .7, 2.5, 6.)])
        with tempfile.TemporaryDirectory() as tmp:
            args = B.parser().parse_args(base[:-6] + ['--out', str(Path(tmp) / 'o'), '--sessions', '2'])
            args.saved_root_admission = False
            with patch.object(B, 'options', return_value=args), patch.object(B, 'battle_requests', return_value=battles) as source, \
                    patch.object(B.server, 'build', return_value=fake), patch.object(B.server, 'header', return_value={}):
                self.assertEqual(B.main([]), 0)
            summary = json.loads((Path(tmp) / 'o' / 'summary.json').read_text());md = (Path(tmp) / 'o' / 'summary.md').read_text()
        self.assertEqual(source.call_args.args[1], 2)
        self.assertEqual(fake.resets, [])   # sessions persist across battles, as on the live server
        self.assertEqual([r['battle_id'] for r in fake.seen], ['b0', 'b2', 'b10', 'b11'])   # in order; the single-legal b1 is never sent
        self.assertEqual((summary['rosters'], summary['battles']['count'], summary['battles']['decisions'], summary['sessions']), ({}, 2, 5, 2))
        self.assertEqual((summary['pooled']['latency']['n'], summary['pooled']['skipped_single_legal'], summary['pooled']['priced']), (4, 1, 4))
        self.assertIn('session mode: 2 battles, 5 decisions', md)
        self.assertIn('deadline coverage', md)

    def test_main_exit_2_when_nothing_priced(self):
        items = [(0, ('a',), address_request(0), None, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            args = B.parser().parse_args(['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'x',
                '--vocabulary', 'v', '--expected-vocabulary-sha256', 'y', '--roster', 'r.json', 'z', '--out', str(Path(tmp) / 'o')])
            with patch.object(B, 'options', return_value=args), patch.object(B, 'roster_requests', return_value=items), \
                    patch.object(B.server, 'build', return_value=FakeSearch([response('REFUSED', 'constructor_refused')])), \
                    patch.object(B.server, 'header', return_value={}):
                self.assertEqual(B.main([]), 2)


if __name__ == '__main__':
    unittest.main()
