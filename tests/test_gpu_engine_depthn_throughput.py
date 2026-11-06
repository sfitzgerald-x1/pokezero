"""d3_depthn_throughput on stand-in pricers (no bank, no device): configs, the searched pipeline at the maps' own widths,
per-W derivation against the budget, the labelled per-ply split estimate, isolation argv and argument checks."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from gpu_engine import d3_depthn_throughput as tool


class FakeDepthN:
    """The pricer's contract: under `searched` the ply-2 k's derive the slot widths (here k + 3 / k + 2); compiles are per call.
    Timing: P1 = 0.01 s, plies = 1e-6 s per evaluated lane, host packing 2 ms inside the ply loop; device = P1 + plies."""
    built = []

    def __init__(self, executor, model, *, capacity, depth, k_own, k_opp, selection='prior', **kw):
        widths = lambda k: tuple(k) if isinstance(k, tuple) else (k,) * (depth - 1)
        self.given = (widths(k_own), widths(k_opp));self.k_own, self.k_opp = self.given;self.searched_k = None
        if selection == 'searched':self.searched_k = (self.k_own[0], self.k_opp[0]);self.k_own, self.k_opp = (self.k_own[0] + 3,) + self.k_own[1:], (self.k_opp[0] + 2,) + self.k_opp[1:]
        self.depth, self.selection, self.capacity, self.calls = depth, selection, capacity, 0;FakeDepthN.built.append(self)

    def price(self, requests, materializer, recipe, params, *, selection_maps=None):
        if (self.selection == 'searched') != (selection_maps is not None):raise ValueError('maps belong to searched')
        if selection_maps is not None:assert selection_maps[2][0].shape[-1] <= self.k_own[0] and selection_maps[2][1].shape[-1] <= self.k_opp[0]
        self.compiles = (('p', .5, 1.5),) if not self.calls else ();self.calls += 1
        nodes, plies = 81 * 2 * len(requests), []
        for p in range(2, self.depth + 1):
            K = self.k_own[p - 2] * self.k_opp[p - 2];plies.append(dict(ply=p, lanes=nodes * K, evaluated_lanes=nodes * K + 16, enumerated=nodes * K));nodes *= K
        loop = 1e-6 * sum(x['evaluated_lanes'] for x in plies) + .002
        return SimpleNamespace(timing=dict(device_seconds=.01 + loop - .002, ply_loop_seconds=loop, host_pack_seconds=.002, frame_wall_seconds=.01 + loop + .001,
            enumerated_leaves=plies[-1]['enumerated'], evaluated_leaf_lanes=plies[-1]['lanes'], plies=tuple(plies), searched_k=self.searched_k,
            searched_widths=None if self.searched_k is None else (self.k_own[0], self.k_opp[0])), pre=np.zeros((len(requests), 2)))


class FakeFramePricer:
    def __init__(self, *a, **kw):pass

    def price(self, requests, materializer, recipe, params):
        self.last_timing = dict(device_seconds=.01, frame_wall_seconds=.012, lower_seconds=.1, compile_seconds=.2, evaluated_joint_slots=81 * 2 * len(requests))
        return [SimpleNamespace(world_reports=({'required_pairs': 9}, {'required_pairs': 9})) for _ in requests]


def searched_maps(frame, k_own, k_opp):
    R = frame.pre.shape[0];return np.zeros((R, 2, 9, 9, k_own + 3), np.int32), np.zeros((R, 2, 9, 9, k_opp + 1), np.int32), np.zeros(R, bool)


MODULE = SimpleNamespace(DepthNPricer=FakeDepthN, as_depth2_frame=lambda f: f, searched_maps=searched_maps)


class ThroughputTests(unittest.TestCase):
    def args(self, *extra):
        return tool.parse(['--d3-root', 'r', '--chunk', '0', '--out', 'o', '--capacity', '4', '--repeats', '2', *extra])

    def ctx(self, args):
        planned = [(b, None, [object()] * args.capacity, contextlib.nullcontext) for b in range(max(args.w_values) // 2)]
        return SimpleNamespace(provider=SimpleNamespace(executor=None, materializer=None), plan=SimpleNamespace(model=None), params=(), planned=planned, actor_capacity=1)

    def run_configs(self, args):
        ctx = self.ctx(args);sections = {}
        with patch.object(tool.t2, 'memory_stats', lambda: dict(process_peak_bytes_so_far=10, bytes_in_use=4)):
            for config in args.configs:
                caller = tool.Caller(config, args, ctx, module=MODULE, frame_pricer_class=FakeFramePricer)
                blocks = tool.measure(caller, ctx.planned, args.repeats, memory=tool.t2.memory_stats, progress=None)
                sections[config] = dict(config=tool.parse_config(config), blocks=blocks, derived={str(w): tool.derive(blocks, args.capacity, w) for w in args.w_values})
        return sections

    def test_configs_and_names(self):
        args = self.args()
        self.assertEqual(args.configs, ['depth1', 'depth2-k2', 'depth2-k3', 'depth2-k4', 'depth2-fw', 'depth3-prior-k2', 'depth3-prior-k3', 'depth3-prior-k4',
            'depth3-searched-k2', 'depth3-searched-k3', 'depth3-searched-k4'])
        self.assertEqual(tool.parse_config('depth3-searched-k4'), dict(kind='searched', depth=3, mode='searched', k=4))
        self.assertEqual(tool.parse_config('depth2-k3'), dict(kind='depthn', depth=2, mode='prior', k=3))
        self.assertEqual(tool.parse_config('depth2-fw'), dict(kind='depthn', depth=2, mode='prior', k=9))

    def test_searched_runs_the_pass_the_public_maps_and_depth_n_at_the_maps_widths(self):
        FakeDepthN.built = [];args = self.args('--ks', '2', '--worlds', '2', '--modes', 'searched')
        sections = self.run_configs(args);searched = sections['depth3-searched-k2']
        depthn = [p for p in FakeDepthN.built if p.selection == 'searched']
        self.assertEqual([(p.given, p.k_own, p.k_opp) for p in depthn], [(((2, 2), (2, 2)), (5, 2), (4, 2))])   # built with k's; widths derived
        call = searched['blocks'][0]['calls'][1];self.assertEqual((call['maps_widths'], call['slot_widths']), ([5, 3], [5, 4]))
        self.assertEqual((call['searched_k'], call['searched_widths']), ((2, 2), (5, 4)))
        self.assertAlmostEqual(call['device_seconds'], call['parts']['depth2_full_width_pass']['device_seconds'] + call['parts']['depthn']['device_seconds'])
        derived = searched['derived']['2'];self.assertIn('depth2_full_width_pass', derived['parts_ms_per_root'])
        self.assertEqual(searched['blocks'][0]['cold_compile_seconds'], 4.)   # both pricers' first-call compiles
        self.assertEqual(searched['blocks'][0]['calls'][1]['compile_seconds'], 0.)
        self.assertAlmostEqual(call['p1_seconds'], .01);self.assertAlmostEqual(call['p1_seconds'] + call['plies_seconds'], call['parts']['depthn']['device_seconds'])

    def test_derivation_budget_and_split(self):
        args = self.args('--ks', '2', '--worlds', '2,8');sections = self.run_configs(args)
        for name, section in sections.items():
            self.assertEqual(set(section['derived']), {'2', '8'})
            two, eight = section['derived']['2'], section['derived']['8']
            self.assertEqual((two['blocks'], eight['blocks']), (1, 4));self.assertAlmostEqual(eight['device_ms_per_root'], 4 * two['device_ms_per_root'])
            self.assertEqual(two['budget_ms_per_root'], 1000.);self.assertTrue(two['within_budget_device'])
            self.assertEqual(two['peak_over_baseline_in_use'], 6)
        context = dict(evaluator=dict(override=False), root_values=[], roots_available=2, source={}, devices=['cpu'])
        report = tool.report_of(args, sections, context, isolated_run=False)
        split = report['by_worlds']['2']['ply_split_estimate']['depth3-prior-k2']
        self.assertEqual(split['p1_and_plies'], 'MEASURED');self.assertAlmostEqual(split['p1_ms_per_root'], 1e3 * .01 / 4)   # the pricer's own P1
        total = report['by_worlds']['2']['configs']['depth3-prior-k2']['device_ms_per_root']
        self.assertAlmostEqual(split['p1_ms_per_root'] + split['leaf_ply_ms_per_root'] + split['expand_plies_ms_per_root'], total)
        self.assertEqual(set(split['expand_by_ply']), {'2'});self.assertAlmostEqual(split['host_pack_ms_per_root'], 1e3 * .002 / 4)
        leaf_lanes = 81 * 2 * 4 * 4 * 4 + 16   # 4 roots x 81 x 2 cells x 4 ply-2 nodes x 4 leaf lanes, plus call padding
        self.assertAlmostEqual(split['leaf_ply_ms_per_root'], 1e3 * 1e-6 * leaf_lanes / 4)   # depth2-k2's 1 us per evaluated lane
        searched = report['by_worlds']['2']['ply_split_estimate']['depth3-searched-k2'];self.assertIsNotNone(searched['depth2_full_width_pass_ms_per_root'])
        self.assertAlmostEqual(searched['p1_ms_per_root'] + searched['plies_ms_per_root'], report['by_worlds']['2']['configs']['depth3-searched-k2']['parts_ms_per_root']['depthn'])

    def test_isolate_argv_and_argument_checks(self):
        args = self.args('--isolate');calls = []
        def call(argv):
            calls.append(argv);out = Path(argv[argv.index('--out') + 1]);out.mkdir(parents=True)
            (out / 'throughput.json').write_text('{"sections": {"%s": {}}, "context": {"evaluator": {}}}' % argv[argv.index('--only') + 1]);return 0
        with patch('subprocess.call', side_effect=call):
            sections, _ = tool.isolated(args, ['--d3-root', 'r', '--chunk', '0', '--out', 'o', '--capacity', '4', '--isolate'], Path(tempfile.mkdtemp()))
        self.assertEqual(len(calls), len(args.configs));self.assertNotIn('--isolate', calls[0]);self.assertEqual(calls[0].count('--out'), 1)
        out = Path(tempfile.gettempdir()) / f'depthn-throughput-never-{uuid.uuid4().hex}'
        with contextlib.redirect_stderr(io.StringIO()):
            for extra in (['--depth', '2'], ['--worlds', '3'], ['--ks', '10'], ['--modes', 'other'], ['--only', 'depth9-prior-k2'], ['--isolate', '--only', 'depth1']):
                with self.assertRaises(SystemExit) as raised:tool.parse(['--d3-root', 'r', '--chunk', '0', '--out', str(out), '--capacity', '4', *extra])
                self.assertEqual(raised.exception.code, 2, extra)
        with patch('importlib.util.find_spec', return_value=None), self.assertRaises(SystemExit) as raised:
            tool.main(['--d3-root', 'r', '--chunk', '0', '--out', str(out), '--capacity', '4'])
        self.assertIn('device_depthn_pricer', str(raised.exception.code));self.assertFalse(out.exists())


if __name__ == '__main__':unittest.main()
