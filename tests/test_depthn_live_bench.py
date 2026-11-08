import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import depthn_live_bench as B


def response(status='COMPLETE', cause=None, prepare=.5, kinds=(), agrees=True):
    return dict(status=status, cause=cause, refusal=None if status == 'COMPLETE' else 'no', action_index=0 if status == 'COMPLETE' else None,
        mean_q_agrees=agrees if status == 'COMPLETE' else None, constructed_worlds=8 if status == 'COMPLETE' else 3, decision_seconds=prepare + .4,
        timings=dict(queue_seconds=0., prepare_seconds=prepare, build_seconds=.05, price_seconds=.3, reduce_seconds=.01),
        world_refusals=[dict(world=1, kinds=list(kinds))] if kinds else [])


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

    def test_options_refuse_other_buckets(self):
        base = ['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'x', '--vocabulary', 'v',
            '--expected-vocabulary-sha256', 'y', '--roster', 'r', 'z', '--out', 'o']
        with patch.object(B.server.evaluators, 'check_arguments'), self.assertRaises(SystemExit):
            B.options(base + ['--buckets', '1,2'])
        with patch.object(B.server.evaluators, 'check_arguments'), self.assertRaises(SystemExit):
            B.options(base)   # no champion archive

    def test_main_prices_each_root_alone_and_writes_reports(self):
        items = [(i, ('addr', i), address_request(i), None) for i in range(3)] + [(3, ('addr', 3), None, 'OSError: gone')]
        fake = FakeSearch([response(prepare=.5), response('REFUSED', 'depthn_refused'), response(prepare=1.5, agrees=False)])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            args = B.parser().parse_args(['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'x',
                '--vocabulary', 'v', '--expected-vocabulary-sha256', 'y', '--roster', 'r.json', 'z', '--out', str(out), '--warmup-count', '2'])
            args.port, args.max_requests = 0, 1
            with patch.object(B, 'options', return_value=args), patch.object(B, 'roster_requests', return_value=items), \
                    patch.object(B.server, 'build', return_value=fake), patch.object(B.server, 'header', return_value=dict(h=1)):
                self.assertEqual(B.main([]), 0)
            self.assertEqual(len(fake.warmed), 2)
            self.assertEqual([r['battle_id'] for r in fake.seen], ['b0', 'b1', 'b2'])
            self.assertTrue(all(r['rung'] == 'DN2W16' and r['budget_seconds'] == B.BUDGET for r in fake.seen))
            self.assertEqual(len({r['request_id'] for r in fake.seen}), 3)
            rows = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in rows], ['COMPLETE', 'REFUSED', 'COMPLETE', 'UNAVAILABLE'])
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual((summary['pooled']['complete'], summary['pooled']['priced'], summary['pooled']['mean_q_disagreements']), (2, 3, 1))
            self.assertIn('searched share 0.667', (out / 'summary.md').read_text())
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
