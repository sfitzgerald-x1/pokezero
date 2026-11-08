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
        self.responses, self.seen, self.warmed = list(responses), [], None

    def warmup(self, requests):
        self.warmed = list(requests);return dict(buckets={1: 1.})

    def process_many(self, requests):
        assert len(requests) == 1;self.seen.append(requests[0]);return [dict(self.responses.pop(0), request_id=requests[0]['request_id'])]


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
        self.assertIn('| forced_switch | 2 | 2 | 1 | 0.500 | 1 | 0 |', md);self.assertIn('| prepare_seconds | 1 | 0.500 |', md)
        self.assertIn('other: stage (COMPLETE roots)', md);self.assertNotIn('fresh_substitute: stage', md)
        self.assertIn("structural_classes (pooled): ['forced_switch', 'wish']", md)

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
        items = [(i, ('addr', i), address_request(i), None) for i in range(3)] + [(3, ('addr', 3), None, 'OSError: gone')]
        fake = FakeSearch([response(prepare=.5, classes=['wish']), response('REFUSED', 'server_error', classes=['forced_switch']),
            response(prepare=1.5, agrees=False, classes=[])])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            args = B.parser().parse_args(['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'x',
                '--vocabulary', 'v', '--expected-vocabulary-sha256', 'y', '--roster', 'r.json', 'z', '--out', str(out), '--warmup-count', '2',
                '--refuse-classes', 'sleep'])
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
            self.assertIn('searched share 0.667', (out / 'summary.md').read_text())
            self.assertEqual((summary['structural_classes'], summary['refuse_classes'], summary['pooled']['classes']['wish']['complete']), (['forced_switch'], ['sleep'], 1))
            self.assertEqual([r['root_classes'] for r in rows], [['wish'], ['forced_switch'], [], None])
            with patch.object(B, 'options', return_value=args), self.assertRaises(FileExistsError):
                B.main([])   # never into an existing directory

    def test_main_exit_2_when_nothing_priced(self):
        items = [(0, ('a',), address_request(0), None)]
        with tempfile.TemporaryDirectory() as tmp:
            args = B.parser().parse_args(['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'x',
                '--vocabulary', 'v', '--expected-vocabulary-sha256', 'y', '--roster', 'r.json', 'z', '--out', str(Path(tmp) / 'o')])
            with patch.object(B, 'options', return_value=args), patch.object(B, 'roster_requests', return_value=items), \
                    patch.object(B.server, 'build', return_value=FakeSearch([response('REFUSED', 'constructor_refused')])), \
                    patch.object(B.server, 'header', return_value={}):
                self.assertEqual(B.main([]), 2)


if __name__ == '__main__':
    unittest.main()
