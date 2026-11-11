"""Depth-N gate with the real depth-N pricer and fixed program shapes on the inert fake engine (CPU only).

Split from test_gpu_engine_depthn_oracle (its fixtures and helpers stay there) so each module fits the light tier.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch
import jax
import jax.numpy as j
from gpu_engine import d3_depthn_gate as gate, depthn_oracle as reference, depth2_backups
from tests.test_gpu_engine_depth2_oracle import PARAMS

from tests import test_gpu_engine_depthn_oracle as N
from tests.test_gpu_engine_depthn_oracle import (
    FAKE_BACKUPS_SOURCE,
    FAKE_MODULE,
    FakeBackupsN,
    FixtureN,
    HAVE_PRICER,
    executor_n,
    failing,
    permuted_n,
    recording,
    saved_arrays,
    world)


@unittest.skipUnless(HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
class RealPricerTests(FixtureN, unittest.TestCase):
    def module(self):
        return importlib.import_module('gpu_engine.device_depthn_pricer')

    def real(self, **kw):
        return self.pricers(module=self.module(), **kw)

    def test_depth3_prior_passes_every_check(self):
        evaluation, summary = self.call(self.real(lanes=(8, 16)), backups=None)
        result, (status, reasons) = self.judged(evaluation)
        self.assertEqual(status, 'PASS', (failing(result), reasons, result['checks'].get('a', {}).get('first_mismatches')))
        self.assertTrue(result['checks']['block_invariance']['discrete_equal'])
        self.assertTrue(summary['refusals']['depthn_sources_agree'], summary['refusals'])
        self.assertEqual(summary['refusals']['depthn']['source'], 'depthn_backups.refusal_summary')

    def test_depth3_searched_passes_every_check(self):
        evaluation, summary = self.call(self.real(selection='searched'), backups=None)
        result, (status, reasons) = self.judged(evaluation)
        self.assertEqual(status, 'PASS', (failing(result), reasons, result['checks'].get('a', {}).get('first_mismatches')))
        searched_widths = self.module().searched_widths(2, 2)
        self.assertEqual((summary['plan']['k_own'][0], summary['plan']['k_opp'][0]), searched_widths)
        self.assertEqual(summary['searched'], dict(searched_k=[2, 2], searched_widths=list(searched_widths)))
        self.assertTrue(summary['searched_maps']['widths']['own'] <= searched_widths[0] and summary['searched_maps']['widths']['opp'] <= searched_widths[1])

    def test_searched_on_permuted_lanes_with_errored_rows(self):
        """The permuted-lanes fixture: errored turn-2 pairs sit in rows that also have valid leaves, so the rule's ordering of
        refutations (over the kept moves, before the errored rows) is exercised against the real searched_maps."""
        class Permuted(FixtureN):
            make = staticmethod(permuted_n)
        Permuted.setUpClass();fixture = Permuted()
        for k in (1, 2, 3):
            with self.subTest(k=k):
                evaluation, summary = fixture.call(fixture.pricers(module=self.module(), selection='searched', k_own=k, k_opp=k), backups=None)
                result, (status, reasons) = self.judged(evaluation)
                self.assertEqual(status, 'PASS', (failing(result), reasons, [m for m in result['checks']['a']['first_mismatches'] if 'searched' in m['field']][:3]))
                self.assertGreater(result['checks']['a']['compared']['searched_maps_opp_rule'], 0)

    def test_depth4_and_mixed_widths_pass(self):
        for depth, k_own, k_opp, k1 in ((4, 2, 2, (9, 9)), (3, (3, 2), (1, 3), (9, 9)), (3, 2, 2, (3, 2))):
            with self.subTest(depth=depth, k_own=k_own, k_opp=k_opp, k1=k1):
                evaluation, summary = self.call(self.real(depth=depth, k_own=k_own, k_opp=k_opp, k1_own=k1[0], k1_opp=k1[1]), backups=None)
                self.assertTrue(summary['refusals']['depthn_sources_agree'], summary['refusals'])
                result, (status, reasons) = self.judged(evaluation)
                self.assertEqual(status, 'PASS', (failing(result), reasons, result['checks'].get('a', {}).get('first_mismatches')))

    def test_n2_equivalence_with_the_depth2_kernel(self):
        from gpu_engine.device_depth2_pricer import Depth2Pricer
        module = self.module();configs = []
        for name, k in (('full_width', 9), ('top2', 2)):
            configs.append((name, module.DepthNPricer(self.ex, object(), capacity=32, depth=2, k_own=k, k_opp=k, digest=True, actor_capacity=3),
                Depth2Pricer(self.ex, object(), capacity=32, k_own=k, k_opp=k, digest=True, actor_capacity=3)))
        with self.engine():
            results = gate.n2_equivalence_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, configs=configs,
                frame_pricer=self.frame_pricer(), oracle=self.oracle, context=dict(block=0, call=0), module=module)
        report = gate.summarize_n2({name: [result] for name, result in results.items()})
        for name, r in report.items():
            self.assertTrue(r['passed'], (name, r['discrete']['mismatched'], {k: v['max_abs_difference'] for k, v in r['floats'].items()}, r['e'].get('reasons')))
            self.assertGreater(r['e']['roots_compared'], 0)

    def test_live_diagnostics_from_real_frames(self):
        module = self.module();backups = importlib.import_module('gpu_engine.depthn_backups')
        with self.engine():
            depth2 = module.as_depth2_frame(gate.depth2_pass_pricer(module, self.ex, object(), capacity=32, actor_capacity=3).price(
                self.requests, self.tables, self.recipe, PARAMS))
            frames = {}
            for mode in gate.MODES:
                pricers = self.real(selection=mode);maps, first, _ = pricers.maps(depth2);_, (pricer,) = pricers.get()
                frames[(mode, 2)] = pricer.price(self.requests, self.tables, self.recipe, PARAMS, **({} if maps is None else dict(selection_maps=maps)))
        report = gate.merge_diagnostics([gate.diagnostics_call(depth2, frames, self.requests, backups=backups, backups2=depth2_backups)])
        self.assertEqual(set(report['change_rate']), {'prior-k2', 'searched-k2'});self.assertGreater(report['prior_vs_searched']['k2']['cells'], 0)
        for mode in gate.MODES:   # depthn_backups' own learner_max_root agrees with the gate's turn-2 choices' building blocks
            choices = gate.turn2_choices(frames[(mode, 2)], backups);self.assertTrue(choices)


    def test_choices_and_the_mutation_test_on_the_real_pricer(self):
        """The real pricer at depth 3 (k = 3) with --mutation-test on depthn_backups' own SOURCE: (f) passes clean, all ten mutants are killed
        (the backup path by (f) alone, read identity included), neither sweep has a material survivor; read identity passes on the clean real
        frame; a grid permuted only where depthn_backups hands it to the rule fails (f)'s per-child grid gate and read identity; the gate's
        OracleTree equals depthn_backups.node_value on every interior node of the real frame (both
        read the oracle's own values here: the real frame matches them exactly on this engine)."""
        backups = importlib.import_module('gpu_engine.depthn_backups');pricers = self.real(k_own=3, k_opp=3);module = self.module()
        depth2_pricer = gate.depth2_pass_pricer(module, self.ex, object(), capacity=32, actor_capacity=3, digest=True)
        with self.engine():
            ev, _ = gate.run_call_n(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, pricers=pricers, depth2_pricer=depth2_pricer,
                frame_pricer=self.frame_pricer(), oracle=self.oracle, sample=None, block=0, context=dict(block=0, call=0), mutation={})
        inputs = dict(ev[gate.MUTATION_INPUTS]);result, (status, reasons) = self.judged(ev)
        (m,) = gate.mutation_tests([ev], value_bound=result['tolerance']['value_bound'])
        self.assertEqual(status, 'PASS', (failing(result), reasons));self.assertGreater(result['checks']['f']['comparisons'], 0)
        self.assertTrue(m['passed'], {n: (r.get('caught_by'), r.get('reason')) for n, r in m['mutants'].items()})
        self.assertEqual({n: r['caught_by'] for n, r in m['mutants'].items() if r['path'] == 'backup'}, {f'{p}-{k}-{v}': ['f'] for p, k, v in gate.MUTANTS if p == 'backup'})
        self.assertTrue(all(not r['caught']['f']['read_identity']['passed'] for r in m['mutants'].values() if r['path'] == 'backup' and r['kind'] != 'priors'))
        self.assertFalse(m['mutants']['backup-priors-systematic']['caught']['f']['priors']['passed'])
        self.assertEqual(result['checks']['f']['priors']['passed'], True);self.assertGreater(result['checks']['f']['priors']['nodes'], 200)
        self.assertEqual(m['material_survivors'], dict(scatter=0, backup=0));identity = result['checks']['f']['read_identity']
        self.assertEqual((identity['passed'], identity['injective'], identity['interned_collisions']), (True, True, 0));self.assertGreater(identity['nodes'], 200)
        frame, data = inputs['frame'], inputs['data'];key = next(k for k in gate._parents(frame) if gate.pricer_record(frame, backups, *k) is not None)
        permuted = gate.mutant_backups([]);permuted._AT_RULE = lambda grid, root, world, ply, node: [row[::-1] for row in grid] if (root, world, ply, node) == key else grid
        f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=permuted), result['tolerance']['value_bound'])
        self.assertFalse(f['grid_within_bound']);self.assertIn(key, [(r['root'], r['world'], r['ply'], r['node']) for r in f['first_grid_mismatches']])
        self.assertFalse(f['read_identity']['passed']);self.assertIn(key, [(r['root'], r['world'], r['ply'], r['node']) for r in f['read_identity']['first_mismatches']])
        tree = gate.OracleTree(frame, data);nodes = gate.interior_nodes(frame, data)
        for q, w, ply, node in nodes:self.assertEqual(backups.node_value(frame, q, w, ply, node), tree.solve(q, w, ply, node)[0], (q, w, ply, node))
        self.assertGreater(sum(tree.solve(*k)[0] is not None for k in nodes), 200)

    def test_throughput_callers_run_the_real_pricer(self):
        from gpu_engine import d3_depthn_throughput as tool
        args = tool.parse(['--d3-root', 'r', '--chunk', '0', '--out', 'o', '--capacity', '32', '--ks', '2', '--worlds', '2', '--repeats', '1', '--block-lanes', '16'])
        ctx = SimpleNamespace(provider=SimpleNamespace(executor=self.ex, materializer=self.tables), plan=SimpleNamespace(model=object()), params=PARAMS,
            planned=[(0, self.recipe, list(self.requests), contextlib.nullcontext)], actor_capacity=3)
        with self.engine(), patch.object(tool.t2, 'memory_stats', lambda: {}):
            sections = {c: tool.measure_config(c, args, ctx, None) for c in args.configs}
        searched = sections['depth3-searched-k2']['blocks'][0]['calls'][1]
        self.assertEqual(searched['maps_widths'], [int(x) for x in searched['maps_widths']]);self.assertGreater(searched['leaves_evaluated'], 0)
        for name, section in sections.items():self.assertIsNotNone(section['derived']['2']['device_ms_per_root'], name)

    def test_the_gate_catches_a_real_pricer_on_a_broken_ply3(self):
        """The real pricer driven by mutated engines (the gate's oracle and D3 keep the true one): the ply-3 step's chance
        drawn from a constant, and (on permuted lanes) the ply-2 views carrying the engine mask."""
        bad = executor_n(sample=lambda state: j.where(state.turn == 2, 0, state.seed % 17).astype(j.int32))
        evaluation, _ = self.call(self.real(executor=bad), backups=None, depth2=False)
        result, (status, _) = self.judged(evaluation);mismatched = result['checks']['a']['mismatched']
        self.assertEqual(status, 'FAIL');self.assertGreater(mismatched['leaf_digest'], 0);self.assertEqual(mismatched['ply2_digest_own'], 0)
        class Permuted(FixtureN):
            make = staticmethod(permuted_n)
        Permuted.setUpClass();fixture = Permuted()
        clean, _ = fixture.call(fixture.pricers(module=self.module()), backups=None, depth2=False)
        self.assertEqual(self.judged(clean)[1][0], 'PASS', failing(self.judged(clean)[0]))
        evaluation, _ = fixture.call(fixture.pricers(module=self.module(), executor=permuted_n(engine_mask_at_turn=2)), backups=None, depth2=False)
        result, (status, _) = self.judged(evaluation);mismatched = result['checks']['a']['mismatched']
        self.assertEqual(status, 'FAIL');self.assertGreater(mismatched['ply2_own_legal'], 0);self.assertEqual(mismatched['t1_own_legal'], 0)


class FixedShapesTests(FixtureN, unittest.TestCase):
    """--fixed-shapes on the fixture split into two calls: --capacity 16 over the first 19 roots is 16 requests (31 world rows) then 3 (6).
    With the flag every DepthNPricer is built with fixed_capacity=True and both oracle stacks have 2 x 16 = 32 rows; without it the stacks
    bucket to 32 and 8 rows and nothing is built, logged or written differently. The verdict and every exact field are the same."""
    USED = 19
    main = N.CommandLineRunTests.main

    def roots(self):
        roots = N.CommandLineRunTests.roots(self);roots.groups = [list(self.requests[:self.USED])];return roots

    @contextlib.contextmanager
    def stacks(self):
        """Every oracle stack the gate builds (run_call_n, n2_equivalence_call; not the stand-in pricer's own): its size argument, world
        rows and the leading dimension of every stacked array."""
        original, seen = reference.stack, []
        def stack(requests, worlds, **kw):
            s = original(requests, worlds, **kw)
            if sys._getframe(1).f_code.co_name in ('run_call_n', 'n2_equivalence_call'):
                seen.append(dict(size=kw.get('size'), rows=len(s.rows), shape={int(x.shape[0]) for x in jax.tree.leaves((s.states, s.publics, s.owners))}))
            return s
        with patch.object(reference, 'stack', stack):yield seen

    def test_the_flag_reaches_every_pricer_and_every_oracle_stack(self):
        for flag in (True, False):
            with self.subTest(fixed_shapes=flag):
                module, built = recording(FAKE_MODULE);lines = []
                with self.stacks() as seen:code, out = self.main('--capacity', '16', *(('--fixed-shapes',) if flag else ()), lines=lines, module=module)
                report = json.loads((out / 'gate.json').read_text());start = next(x for x in lines if x['message'] == 'GATE_START')
                self.assertEqual((code, report['status'], report['split_capacity'], len(report['calls'])), (0, 'PASS', 16, 2), report.get('status_reasons'))
                self.assertEqual([(s['rows'], s['shape']) for s in seen], [(31, {32}), (6, {32 if flag else 8})])
                self.assertEqual(len(built), 2)   # the depth-N pricer and the full-width depth-2 pass
                if flag:
                    self.assertTrue(all(kw.get('fixed_capacity') is True for kw in built), built);self.assertEqual([s['size'] for s in seen], [32, 32])
                    want = dict(fixed_shapes=True, pricer_fixed_capacity=True, oracle_stack_rows=32, split_capacity=16, rule=gate.FIXED_SHAPES)
                    self.assertEqual((report['harness'], start['harness']), (want, want));self.assertIs(report['pricer_options']['fixed_capacity'], True)
                else:
                    self.assertFalse(any('fixed_capacity' in kw for kw in built), built);self.assertEqual([s['size'] for s in seen], [None, None])
                    self.assertNotIn('harness', report);self.assertNotIn('harness', start);self.assertNotIn('fixed_capacity', report['pricer_options'])

    def test_n2_equivalence_refuses_and_diagnostics_take_the_flag(self):
        """--n2-equivalence refuses the flag (exit 2, nothing created: its Depth2Pricer keeps per-call buckets, so the compared pricers would
        price turn 1 at different batch sizes); --depth3-diagnostics: every DepthNPricer fixed, no oracle (oracle_stack_rows None)."""
        out = str(Path(tempfile.gettempdir()) / f'depthn-gate-never-created-{uuid.uuid4().hex}')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            gate.main(['--out', out, '--d3-root', 'b', '--chunk', '0', '--n2-equivalence', '--fixed-shapes'])
        self.assertEqual(raised.exception.code, 2);self.assertFalse(Path(out).exists())
        module, built = recording(FAKE_MODULE);lines = []
        code, out = self.main('--depth3-diagnostics', '--diagnostic-ks', '2', '--fixed-shapes', lines=lines, module=module)
        report = json.loads((out / 'diagnostics.json').read_text());start = next(x for x in lines if x['message'] == 'DIAGNOSTICS_START')
        self.assertEqual((code, len(built)), (0, 3));self.assertTrue(all(kw.get('fixed_capacity') is True for kw in built), built)
        self.assertEqual((report['harness']['oracle_stack_rows'], start['harness']), (None, report['harness']))
        self.assertTrue(all(o['fixed_capacity'] for o in report['pricer_options'].values()))

    def test_offline_modes_refuse_the_flag(self):
        out = str(Path(tempfile.gettempdir()) / f'depthn-gate-never-created-{uuid.uuid4().hex}')
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (['--reverdict', 'gate.json', '--fixed-shapes'], ['--out', out, '--mutation-from', 'x', '--fixed-shapes'],
                    ['--out', out, '--diagnostics-from', 'x', '--fixed-shapes']):
                with self.assertRaises(SystemExit) as raised:gate.main(argv)
                self.assertEqual(raised.exception.code, 2, argv)
        self.assertFalse(Path(out).exists());self.assertFalse(gate.parse(['--out', 'o', '--d3-root', 'b', '--chunk', '0'])[1].fixed_shapes)

    def test_the_verdict_and_every_exact_field_are_unchanged(self):
        """The same two-call run with and without the flag (the stand-in, and the real pricer when it is on the tree): the same status and
        checks ((a)-(f), (f)'s read identity and priors included), tolerance, header, refusals and per-call results; every saved frame and
        oracle array (digests, statuses, maps, values) bit-equal; --reverdict and --mutation-from reproduce on both, a report without
        `harness` (the no-flag run's) read as before."""
        for real in (False, True) if HAVE_PRICER else (False,):
            with self.subTest(real=real):
                runs = {flag: self.main('--capacity', '16', '--save-oracle', *(('--fixed-shapes',) if flag else ()), real=real) for flag in (False, True)}
                self.assertEqual([code for code, _ in runs.values()], [0, 0])
                plain, fixed = (json.loads((runs[flag][1] / 'gate.json').read_text()) for flag in (False, True))
                self.assertEqual((plain['status'], set(fixed) - set(plain), set(plain) - set(fixed)), ('PASS', {'harness'}, set()))
                for key in ('status', 'passed', 'status_reasons', 'checks', 'decision', 'tolerance', 'header', 'leaves', 'refusals', 'topk_boundary',
                        'depth2_pass', 'split_capacity', 'searched_maps'):
                    self.assertEqual(fixed[key], plain[key], key)
                self.assertEqual({k: v for k, v in fixed['pricer_options'].items() if k != 'fixed_capacity'}, plain['pricer_options'])
                for a, b in zip(fixed['calls'], plain['calls']):
                    for key in ('roots', 'worlds', 'plan', 'refusals', 'nodes_checked', 'leaves_enumerated', 'leaves_checked', 'forced_leaves', 'passed',
                            'choices', 'topk', 'depth2_pass'):
                        self.assertEqual(a[key], b[key], key)
                f = fixed['checks']['f'];self.assertGreater(f['comparisons'], 0);self.assertTrue(f['read_identity']['passed'] and f['priors']['passed'])
                names = sorted(p.name for p in runs[False][1].glob('*.npz'));self.assertEqual(names, sorted(p.name for p in runs[True][1].glob('*.npz')))
                self.assertEqual(len(names), 6)   # per call: the depth-N frame, the depth-2 pass and the oracle data
                for name in names:
                    a, b = saved_arrays(runs[True][1] / name), saved_arrays(runs[False][1] / name);self.assertEqual(set(a), set(b), name)
                    for key in a:self.assertEqual((a[key].dtype, a[key].shape, a[key].tobytes()), (b[key].dtype, b[key].shape, b[key].tobytes()), (name, key))
                again = {}
                for flag, (_, out) in runs.items():
                    with patch.object(gate, 'log', lambda *a, **k: None):self.assertEqual(gate.main(['--reverdict', str(out / 'gate.json')]), 0)
                    again[flag] = json.loads((out / gate.REVERDICT_FILE).read_text())
                self.assertEqual((again[True]['harness'], 'harness' in again[False]), (fixed['harness'], False))
                self.assertEqual({k: v for k, v in again[True]['reverdict'].items() if k != 'source'}, {k: v for k, v in again[False]['reverdict'].items() if k != 'source'})
                if real:continue   # the offline mutation test is the same code on either frame; once on the stand-in
                mutation = {}
                for flag, (_, out) in runs.items():
                    offline = Path(tempfile.mkdtemp(prefix='depthn-fixed-')) / 'offline'
                    with patch.object(gate, 'log', lambda *a, **k: None), patch.object(gate, 'backups_source', lambda: FAKE_BACKUPS_SOURCE), \
                            patch.dict(sys.modules, {'gpu_engine.depthn_backups': FakeBackupsN}):
                        self.assertEqual(gate.main(['--out', str(offline), '--mutation-from', str(out)]), 0)
                    mutation[flag] = json.loads((offline / gate.MUTATION_FILE).read_text())
                verdicts = lambda m: [(c['passed'], {n: (r['killed'], r['caught_by']) for n, r in c['mutants'].items()}, c['material_survivors']) for c in m['calls']]
                self.assertEqual(verdicts(mutation[True]), verdicts(mutation[False]));self.assertEqual(len(mutation[True]['calls']), 2)

    @unittest.skipUnless(HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
    def test_two_calls_compile_each_program_once(self):
        """run_call_n on 16 requests then 3 with the real pricer. Fixed shapes: both oracle stacks are 32 rows, the second call compiles
        no pricer program (the depth-N pricer's and the depth-2 pass's program keys are the same after each call; turn 1 at capacity 16)
        and every oracle program holds one compiled shape. The control (no flag): stacks of 32 and 8 rows, turn 1 recompiled at
        capacity 4 and the oracle's programs compiled for both shapes."""
        module = importlib.import_module('gpu_engine.device_depthn_pricer')
        for fixed in (True, False):
            with self.subTest(fixed_shapes=fixed):
                pricers = gate.DepthNPricers(module, self.ex, object(), depth=3, k_own=2, k_opp=2, selection='prior', capacity=16, actor_capacity=3,
                    fixed_capacity=fixed)
                depth2_pricer = gate.depth2_pass_pricer(module, self.ex, object(), capacity=16, actor_capacity=3, digest=True, fixed_capacity=fixed)
                oracle = reference.DepthNOracle(self.ex, object(), chunk=64);keys, compiled = [], []
                with self.stacks() as seen, self.engine():
                    for call, requests in enumerate((self.requests[:16], self.requests[16:19])):
                        evaluation, _ = gate.run_call_n(requests=requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, pricers=pricers,
                            depth2_pricer=depth2_pricer, frame_pricer=self.frame_pricer(), oracle=oracle, sample=None, block=0, context=dict(block=0, call=call),
                            stack_rows=gate.fixed_shapes_harness(16)['oracle_stack_rows'] if fixed else None)
                        self.assertEqual(self.judged(evaluation)[1][0], 'PASS', failing(self.judged(evaluation)[0]))
                        engines = (pricers.instances[-1], depth2_pricer)
                        keys.append([sorted(map(repr, e.programs)) for e in engines]);compiled.append([e.last_timing['compiled'] for e in engines])
                sizes = {name: f._cache_size() for name, f in [('roots', oracle._root), ('children', oracle._child)] + [(str(k), f) for k, f in oracle._walks.items()]}
                p1 = [[k for k in c[0] if k[0] == 'P1'] for c in compiled]
                if fixed:
                    self.assertEqual([(s['size'], s['shape']) for s in seen], [(32, {32}), (32, {32})])
                    self.assertEqual(keys[0], keys[1]);self.assertEqual(compiled[1], [(), ()]);self.assertEqual([k[1] for k in p1[0]], [16])
                    self.assertEqual(sizes, dict.fromkeys(sizes, 1));self.assertGreaterEqual(len(sizes), 4)
                else:
                    self.assertEqual([(s['size'], s['shape']) for s in seen], [(None, {32}), (None, {8})])
                    self.assertNotEqual(keys[0], keys[1]);self.assertEqual(([k[1] for k in p1[0]], [k[1] for k in p1[1]]), ([16], [4]))
                    self.assertEqual(sizes, dict.fromkeys(sizes, 2))


if __name__ == '__main__':unittest.main()
