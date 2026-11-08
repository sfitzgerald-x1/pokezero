"""Inert DepthN panel controls: declarations, cells, claims, probe, dispatch and readout."""
import asyncio
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import socket
from statistics import fmean, variance
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pokezero import depthn_live_actor as live
from pokezero import depthn_live_fairness as fairness
from pokezero import depthn_live_panel as panel
from pokezero import phase_d_ladder_panel as ladder
from pokezero import phase_d_owning_capture as capture
from pokezero.live_protocol_actor import ActorContract
from tests import test_depthn_live_actor as actor_tests
from tests import test_phase_d_ladder_panel as ladder_fixture
from tests import test_phase_d_prerequisites as fixture

ROOT = Path(__file__).resolve().parents[1]
DISPATCH_TIMEOUT = 30


EXPECTED = dict(value=.944, source=dict(run_id='constructor-world-gate-fixture', report_sha256='f'*64))


CALIBRATION_BAND = dict(battle_seed_start=50000, foulplay_seed_start=60000, count=200)
RATIOS = {'4x500ms': .95, '2x1000ms': .97}


def calibration_fixture(ladder_sha256, *, observable=True, ratios=RATIOS, lanes=4, moves=300):
    """A valid calibration.json: per-stratum rates whose concurrent/solo quotient is exactly the ratio (solo rate 1.0)."""
    strata = {s: dict(solo=dict(moves=moves, rate=1.), concurrent=dict(moves=moves, rate=float(r)), ratio=float(r)) for s, r in ratios.items()}
    return dict(schema=panel.CALIBRATION_SCHEMA, status='MEASURED' if observable else 'UNMEASURABLE', iterations_observable=observable,
                ratios=dict(ratios) if observable else None, strata=strata, floors=dict(panel.CALIBRATION_FLOORS), seed_band=dict(CALIBRATION_BAND), lanes=lanes,
                ladder_declaration_sha256=ladder_sha256, base_config_sha256=panel.sha({}), checkpoint_sha256=ActorContract.champion().checkpoint_sha256)


PROVENANCE = dict(
    foulplay=dict(commit='9'*40, applied_diff_sha256='d'*64, patch_blobs={'third_party/foulplay-rest-cant.patch': 'b3e95eab'},
                  poke_engine=dict(version='0.0.47', features='poke-engine/gen3, no default features', sdist_sha256='8'*64,
                                   setup_script_sha256='c'*64, gen3_patch_list_sha256='f'*64, native_so_sha256='f'*64),
                  runtime_metadata_sha256='1'*64, python='3.12.13 aarch64'),
    showdown_commit='f'*40, engine_tables=dict(digest='e'*64, constructor_metadata_sha256='f'*64, bank_metadata_sha256='1'*64),
    base_config_sha256=panel.sha({}))


def declaration(ladder_sha256, arms=('RAW', 'DN2W8'), expected=EXPECTED, deadline=3., observable=True, ratios=RATIOS, calibration_sha='c'*64,
                provenance=PROVENANCE):
    return dict(schema=panel.SCHEMA, provenance=copy.deepcopy(provenance), ladder_declaration_sha256=ladder_sha256, expected_searched_share=copy.deepcopy(expected),
        worker_lanes=4, fairness_observable=dict(value=observable, evidence=dict(path='calibration/calibration.json', sha256=calibration_sha)),
        calibration=dict(status='MEASURED' if observable else 'UNMEASURABLE', sha256=calibration_sha, ratios=dict(ratios) if observable else None,
                         seed_band=dict(CALIBRATION_BAND), floors=dict(panel.CALIBRATION_FLOORS)),
        expected_searched_share_roster=.955, checkpoint_sha256=ActorContract.champion().checkpoint_sha256,
        archive_sha256=live.CHAMPION_ARCHIVE_SHA256, arms=list(arms),
        search={a: dict(depth=2, worlds=panel.WORLDS[a]) for a in arms[1:]}, root_operator='ARGMAX_MEAN_COMPLETED_Q',
        deadline_seconds=deadline, reserve_seconds=.15, screen_paired_cells=500, max_paired_cells=2000, confidence=.9,
        primary_comparison='SEARCH_MINUS_OUR_RAW_PAIRED_BY_CELL', bar='LOWER_BOUND_GT_0_AT_2000_FULL_ROSTER_NO_BREAKAGE',
        screen_rule=panel.SCREEN_RULE, breakage=dict(panel.BREAKAGE, latency_p95_max_seconds=deadline), matmul_precision='highest',
        screen_latency_rule=copy.deepcopy(panel.LATENCY_RULE), bench_p95_wall_seconds=1.39,
        bench_source=dict(path='bench/w8-sessions/summary.json', sha256='e'*64), server_processes_per_gpu=1,
        interleave=panel.INTERLEAVE, arm_fairness=dict(panel.ARM_FAIRNESS))


def make_run(directory, arms=('RAW', 'DN2W8'), expected=EXPECTED, deadline=3., observable=True, ratios=RATIOS):
    root = Path(directory).resolve(); inputs = root/'inputs'
    lad = ladder_fixture.declaration(); calibration = calibration_fixture(panel.sha(lad), observable=observable, ratios=ratios)
    d = declaration(panel.sha(lad), arms, expected, deadline, observable, ratios, panel.sha(calibration)); base = {}
    for name, value in (('ladder', lad), ('declaration', d), ('base', base), ('calibration', calibration)):
        panel.write_new(inputs/f'{name}.json', value)
    run = panel.init(root/'run', declaration=inputs/'declaration.json', declaration_sha256=panel.sha(d),
        ladder_declaration=inputs/'ladder.json', ladder_declaration_sha256=panel.sha(lad), checkpoint='/inert/champion.pt',
        base_config=inputs/'base.json', base_config_sha256=panel.sha(base), calibration=inputs/'calibration.json')
    return root/'run', run, d, lad


def decision_row(cell, i, *, source='search', cause=None, kind=None, differs=False, select=.5,
                 operator='ARGMAX_MEAN_COMPLETED_Q', worlds=8, archive=live.CHAMPION_ARCHIVE_SHA256, precision='highest'):
    kind = kind or ('forced' if source == 'single_legal' else 'move')
    search = None if source in ('raw_arm', 'single_legal') else dict(status='COMPLETE' if source == 'search' else 'REFUSED',
        cause=cause, client_wall_seconds=select-.05, server_wall_seconds=select-.1, tree_commit='t'*40, archive_sha256=archive,
        request_id=f"r{cell['ordinal']}-{i}",
        root_operator=operator if source == 'search' else None, requested_worlds=worlds if source == 'search' else None,
        pricer=dict(k=1, depth=2, W=8, K=4, matmul_precision=precision) if source == 'search' else None,
        timings=dict(queue_seconds=.01, prepare_seconds=.2, build_seconds=.05, price_seconds=.2, reduce_seconds=.01) if source == 'search' else None)
    return dict(schema=live.ROW_SCHEMA, arm=cell['arm'], cell=cell, battle_id=f"b{cell['ordinal']}", owner='p1', decision_index=i,
        prefix_sha256=f'{i:064x}', request_kind=kind, source=source, cause=cause, search_differs=differs if source == 'search' else None,
        encode_seconds=.01, select_seconds=select, greedy=dict(seconds=.05), search=search)


def write_cell(root, cell, d_sha, *, score, status='COMPLETE', error=None, rows=(), decisions=None, think=None, observable=True):
    for name in ('results', 'telemetry'): (root/name).mkdir(exist_ok=True)
    game = dict(pokezero_decisions=len(rows) if decisions is None else decisions, **({} if think is None else dict(opponent_think=list(think))))
    value = dict(schema=panel.CELL_SCHEMA, cell=cell, declaration_sha256=d_sha, status=status, score=score if status == 'COMPLETE' else None,
                 error=error, source_game=game, benchmark=dict(foulplay_think=dict(iterations_observable=observable)))
    (root/'results'/f"{cell['ordinal']}.json").write_bytes(panel.encoded(value))
    if rows: (root/'telemetry'/f"{cell['ordinal']}.jsonl").write_bytes(b''.join(panel.encoded(r)+b'\n' for r in rows))


def populate(root, run, lad, *, limit=500, scores=None, telemetry=True, select=.5, skip=(), arms=('RAW', 'DN2W8'), rows_for=None):
    scores = scores or (lambda p: (float(p//2 % 3 == 0), float(p//2 % 3 != 1)))
    for arm in arms:
        for cell in panel.cells(lad, arm, limit=limit):
            if cell['pair_ordinal'] in skip: continue
            if not telemetry: rows = ()
            elif rows_for is not None: rows = rows_for(cell)
            else: rows = [decision_row(cell, i, source='raw_arm' if arm == 'RAW' else 'search', differs=i == 0, select=select,
                                       worlds=panel.WORLDS.get(arm)) for i in range(2)]
            write_cell(root, cell, run['declaration_sha256'], score=scores(cell['pair_ordinal'])[arm != 'RAW'], rows=rows,
                       decisions=None if telemetry else 2)


def seed_cluster_interval(deltas):
    from scipy.stats import t
    clusters = [(deltas[i]+deltas[i+1])/2 for i in range(0, len(deltas), 2)]
    radius = float(t.ppf(.95, len(clusters)-1))*math.sqrt(variance(clusters)/len(clusters))
    return [max(-1., fmean(clusters)-radius), min(1., fmean(clusters)+radius)]


class DeclarationTests(unittest.TestCase):
    def test_accepts_the_declared_panel(self):
        for arms in (('RAW', 'DN2W8'), ('RAW', 'DN2W16')):
            self.assertEqual(panel.validate(declaration('a'*64, arms))['arms'], list(arms))
        with self.assertRaises(ValueError): panel.validate(declaration('a'*64, ('RAW', 'DN2W8', 'DN2W16')))   # one search arm per run
        self.assertEqual(panel.BREAKAGE['searched_share_min'], .5)
        for value in (None, dict(EXPECTED, value=0.), dict(EXPECTED, value=1.)):
            self.assertEqual(panel.validate(declaration('a'*64, expected=value))['expected_searched_share'], value)

    def test_calibration_strata_and_floors_are_foulplays_own(self):
        self.assertEqual(fairness.CALIBRATION_STRATA, ('4x500ms', '2x1000ms'))   # parallelism 1, no timer
        self.assertEqual(panel.CALIBRATION_FLOORS, {'4x500ms': 60, '2x1000ms': 300})

    def test_ratio_ci90_is_descriptive_and_brackets_the_ratio(self):
        lo, hi = panel.ratio_ci90([9., 10., 11.], [9., 10., 11.])
        self.assertLess(lo, 1.);self.assertGreater(hi, 1.);self.assertAlmostEqual((lo + hi) / 2, 1.)
        self.assertIsNone(panel.ratio_ci90([1.], [1., 2.]))

    def test_refuses_a_malformed_provenance(self):
        fp, pe, tables = PROVENANCE['foulplay'], PROVENANCE['foulplay']['poke_engine'], PROVENANCE['engine_tables']
        for value in (None, {}, dict(PROVENANCE, extra=1), {k: v for k, v in PROVENANCE.items() if k != 'engine_tables'},
                      dict(PROVENANCE, showdown_commit='f'*39), dict(PROVENANCE, base_config_sha256='F'*64),
                      dict(PROVENANCE, foulplay=dict(fp, commit='g'*40)), dict(PROVENANCE, foulplay=dict(fp, patch_blobs={})),
                      dict(PROVENANCE, foulplay=dict(fp, patch_blobs={'p': 'b3e9'})), dict(PROVENANCE, foulplay=dict(fp, python=' ')),
                      dict(PROVENANCE, foulplay=dict(fp, poke_engine=dict(pe, native_so_sha256='x'))),
                      dict(PROVENANCE, foulplay=dict(fp, poke_engine={k: v for k, v in pe.items() if k != 'features'})),
                      dict(PROVENANCE, engine_tables=dict(tables, digest='e'*63))):
            with self.subTest(value), self.assertRaises(ValueError):
                panel.validate(declaration('a'*64, provenance=value))

    def test_init_refuses_a_provenance_for_another_base_config(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(); inputs = root/'inputs'
            lad = ladder_fixture.declaration(); calibration = calibration_fixture(panel.sha(lad))
            d = declaration(panel.sha(lad), calibration_sha=panel.sha(calibration), provenance=dict(PROVENANCE, base_config_sha256='0'*64))
            for name, value in (('ladder', lad), ('declaration', d), ('base', {}), ('calibration', calibration)):
                panel.write_new(inputs/f'{name}.json', value)
            with self.assertRaisesRegex(ValueError, 'provenance must pin this base config'):
                panel.init(root/'run', declaration=inputs/'declaration.json', declaration_sha256=panel.sha(d),
                    ladder_declaration=inputs/'ladder.json', ladder_declaration_sha256=panel.sha(lad), checkpoint='/inert/champion.pt',
                    base_config=inputs/'base.json', base_config_sha256=panel.sha({}), calibration=inputs/'calibration.json')

    def test_refuses_a_malformed_expected_searched_share(self):
        source = EXPECTED['source']
        for value in (.9, dict(EXPECTED, value=1.2), dict(EXPECTED, value=-.1), dict(EXPECTED, value=1), dict(EXPECTED, value=True),
                      dict(EXPECTED, extra=1), dict(value=.9), dict(EXPECTED, source=dict(source, report_sha256='F'*64)),
                      dict(EXPECTED, source=dict(source, run_id=' ')), dict(EXPECTED, source=dict(run_id='x')),
                      dict(EXPECTED, source=dict(source, path='/x'))):
            with self.subTest(value), self.assertRaisesRegex(ValueError, 'expected searched share'):
                panel.validate(declaration('a'*64, expected=value))

    def test_refuses_any_other_declaration(self):
        changes = [('extra', 'host:1'), ('schema', 'other'), ('ladder_declaration_sha256', 'A'*64), ('checkpoint_sha256', 'b'*64),
                   ('archive_sha256', 'b'*64), ('archive_sha256', None),
                   ('arms', ['DN2W8', 'RAW']), ('arms', ['RAW']), ('arms', ['RAW', 'R1']), ('arms', ['RAW', 'DN2W8', 'DN2W8']),
                   ('arms', ['RAW', 'DN2W8', 'DN2W16']), ('arms', ['DN2W8', 'DN2W16']),
                   ('search', {'DN2W8': dict(depth=2, worlds=16)}), ('root_operator', 'ARGMAX_MEAN_COMPLETED_Q_COMMON_ALL_RUNGS'),
                   ('deadline_seconds', 2.5), ('reserve_seconds', 0.), ('screen_paired_cells', 400), ('max_paired_cells', 1000),
                   ('confidence', .95), ('primary_comparison', 'SEARCH_MINUS_CODEX_RAW'), ('bar', 'LOWER_BOUND_GT_0_AT_2000'),
                   ('screen_rule', 'STOP_HARM_IF_LOWER_BOUND_LT_0'), ('breakage', dict(panel.BREAKAGE, latency_p95_max_seconds=4.)),
                   ('breakage', dict(panel.BREAKAGE, unknown_cells_max=5)), ('breakage', dict(panel.BREAKAGE, transport_failure_share_max=.1)),
                   ('breakage', dict(panel.BREAKAGE, searched_share_min=.8))]
        for key, value in changes:
            d = declaration('a'*64); d[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): panel.validate(d)
        d = declaration('a'*64); d.pop('bar')
        with self.assertRaises(ValueError): panel.validate(d)


class CellTests(unittest.TestCase):
    def test_cells_equal_codex_raw_cells_on_every_pairing_field(self):
        lad = ladder_fixture.declaration(); seen = set()
        for limit in (500, 2000):
            codex = ladder.cells(lad, 'RAW', limit=limit)
            for arm in panel.ARMS:
                ours = panel.cells(lad, arm, limit=limit)
                self.assertEqual([panel._pairing(c) for c in ours], [panel._pairing(c) for c in codex])
                self.assertTrue(all(c['arm'] == arm and c['ordinal'] % 3 == panel.ARMS.index(arm) for c in ours))
                if limit == 2000: seen |= {c['ordinal'] for c in ours}
        self.assertEqual(len(seen), 6000)
        cell = panel.cells(lad, 'DN2W8')[7]
        self.assertEqual((cell['seed'], cell['seat'], cell['fp_seed'], cell['offset']), (10003, 1, 20003, 3))
        self.assertIs(panel.checked_cell(lad, cell), cell)
        for key in ('seed', 'fp_seed', 'seat', 'ordinal'):
            bad = dict(cell); bad[key] += 1
            with self.assertRaises(ValueError): panel.checked_cell(lad, bad)
        with self.assertRaises(ValueError): panel.cells(lad, 'R1')
        with self.assertRaises(ValueError): panel.cells(lad, 'RAW', limit=501)

    def test_any_divergence_from_codex_raw_cells_refuses(self):
        lad = ladder_fixture.declaration(); original = ladder.cells
        def shifted(d, arm, *, limit=2000): return tuple(dict(c, fp_seed=c['fp_seed']+1) for c in original(d, arm, limit=limit))
        with patch.object(panel.ladder, 'cells', shifted), self.assertRaisesRegex(ValueError, 'differ'): panel.cells(lad, 'DN2W8')
        bad = copy.deepcopy(lad); bad['foulplay_ms'] = 500
        with self.assertRaises(ValueError): panel.cells(bad, 'RAW')


class TreeTests(unittest.TestCase):
    def test_identity_in_git_and_declared_commit_outside_git(self):
        tree = panel.tree_identity()
        self.assertEqual(tree['commit_source'], 'git'); self.assertEqual(len(tree['commit']), 40); self.assertIs(type(tree['dirty']), bool)
        self.assertEqual(panel.tree_identity(tree['commit']), tree)
        with self.assertRaisesRegex(ValueError, 'HEAD'): panel.tree_identity('0'*40)
        with patch.object(panel, '_git', return_value=None):
            with self.assertRaisesRegex(ValueError, 'outside git'): panel.tree_identity()
            with self.assertRaisesRegex(ValueError, 'outside git'): panel.tree_identity('xyz')
            declared = panel.tree_identity('b'*40)
        self.assertEqual((declared['commit'], declared['dirty'], declared['commit_source']), ('b'*40, None, 'declared'))
        self.assertEqual(declared['source_sha256'], tree['source_sha256'])
        for status, dirty in ((' M src/pokezero/x.py\n', True), ('', False)):
            git = lambda *args, s=status: 'c'*40+'\n' if args[0] == 'rev-parse' else s
            with patch.object(panel, '_git', git): self.assertEqual(panel.tree_identity()['dirty'], dirty)

    def test_code_digest_covers_the_package_gpu_engine_and_bridge_scripts(self):
        with TemporaryDirectory() as directory:
            repo = Path(directory)
            files = dict(package=repo/'src/pokezero/a.py', engine=repo/'gpu_engine/sub/b.py', bridge=repo/'scripts/c.mjs',
                         docs=repo/'docs/x.md', cache=repo/'gpu_engine/__pycache__/b.py', script=repo/'scripts/d.py')
            for path in files.values(): path.parent.mkdir(parents=True, exist_ok=True); path.write_text('0')
            base = panel.code_digest(repo)
            for name, path in files.items():
                path.write_text('1'); changed = panel.code_digest(repo) != base; path.write_text('0')
                with self.subTest(name): self.assertEqual(changed, name in ('package', 'engine', 'bridge'))

    def test_same_tree_compares_commit_digest_and_dirty_fail_closed(self):
        git = dict(commit='c'*40, dirty=False, source_sha256='d'*64, commit_source='git')
        declared = dict(git, dirty=None, commit_source='declared')
        self.assertTrue(panel.same_tree(git, declared)); self.assertTrue(panel.same_tree(declared, git))
        self.assertTrue(panel.same_tree(declared, declared)); self.assertTrue(panel.same_tree(dict(git, dirty=True), dict(git, dirty=True)))
        for a, b in ((dict(git, dirty=True), declared), (declared, dict(git, dirty=True)), (git, dict(git, dirty=True)),
                     (git, dict(git, commit='e'*40)), (git, dict(git, source_sha256='e'*64)), (declared, dict(declared, commit='e'*40))):
            with self.subTest(a=a, b=b): self.assertFalse(panel.same_tree(a, b))
        run = dict(tree=declared)
        with patch.object(panel, 'tree_identity', return_value=git): self.assertEqual(panel.require_tree(run), git)
        with patch.object(panel, 'tree_identity', return_value=dict(git, dirty=True)), self.assertRaisesRegex(ValueError, 'code identity'):
            panel.require_tree(run)

    def test_every_mode_refuses_another_code_identity(self):
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory); populate(root, run, lad)
            self.assertEqual(run['tree'], panel.tree_identity()); self.assertEqual(panel.readout(root, arm='DN2W8', limit=500)['tree'], run['tree'])
            for change in (dict(commit='c'*40), dict(dirty=not run['tree']['dirty']), dict(source_sha256='d'*64)):
                with self.subTest(change), patch.object(panel, 'tree_identity', return_value=dict(run['tree'], **change)):
                    for call in (lambda: panel.readout(root, arm='DN2W8', limit=500), lambda: panel.pending(root, arm='RAW', limit=500),
                                 lambda: panel.probe(root, arm='DN2W8', server=dict(host='127.0.0.1', port=1), capture='/x')):
                        with self.assertRaisesRegex(ValueError, 'code identity'): call()


class RunTests(unittest.TestCase):
    def test_init_binds_pinned_inputs_once(self):
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory)
            self.assertEqual(panel.load(root)[1], run); self.assertEqual(run['checkpoint_sha256'], lad['checkpoint_sha256'])
            inputs = Path(directory).resolve()/'inputs'
            kwargs = dict(declaration=inputs/'declaration.json', declaration_sha256=panel.sha(d), ladder_declaration=inputs/'ladder.json',
                ladder_declaration_sha256=panel.sha(lad), checkpoint='/x', base_config=inputs/'base.json', base_config_sha256=panel.sha({}),
                calibration=inputs/'calibration.json')
            with self.assertRaises(FileExistsError): panel.init(root, **kwargs)
            with self.assertRaisesRegex(ValueError, 'differs'): panel.init(Path(directory)/'b', **dict(kwargs, ladder_declaration_sha256='0'*64))
            other = declaration('c'*64); panel.write_new(inputs/'other.json', other)
            with self.assertRaisesRegex(ValueError, 'pin this ladder'):
                panel.init(Path(directory)/'c', **dict(kwargs, declaration=inputs/'other.json', declaration_sha256=panel.sha(other)))
            with patch.object(panel, '_git', return_value=None), self.assertRaisesRegex(ValueError, 'outside git'):
                panel.init(Path(directory)/'d', **kwargs)

    def test_pending_claim_once_and_explicit_resume(self):
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory)
            packets = panel.pending(root, arm='RAW', limit=500)
            self.assertEqual(len(packets), 500); self.assertEqual(panel.pending(root, arm='RAW', limit=500), packets)
            first = panel.cells(lad, 'RAW', limit=500)[0]
            self.assertEqual(ladder.read_bound(packets[0]['path'], packets[0]['sha256']), panel.request(run, first))
            for cell in panel.cells(lad, 'RAW', limit=500)[:3]: panel.write_new(root/'claims'/f"{cell['ordinal']}.json", {})
            with self.assertRaisesRegex(ValueError, 'resume'): panel.pending(root, arm='RAW', limit=500)
            resumed = panel.pending(root, arm='RAW', limit=500, resume=True)
            self.assertEqual(resumed, packets[3:])
            Path(packets[5]['path']).write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'differs'): panel.pending(root, arm='RAW', limit=500, resume=True)
            with self.assertRaisesRegex(ValueError, 'undeclared'): panel.pending(root, arm='DN2W16', limit=500)

    def test_no_dispatch_before_the_expected_searched_share_is_set(self):
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory, expected=None)
            for arm in ('RAW', 'DN2W8'):
                with self.subTest(arm), self.assertRaisesRegex(ValueError, 'must be set before dispatch'): panel.pending(root, arm=arm, limit=500)
            self.assertFalse((root/'requests').exists())


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root, self.run, self.d, self.lad = make_run(self.directory.name)
        populate(self.root, self.run, self.lad)
        for arm in ('RAW', 'DN2W8'):  # Played screen cells are claimed; pre-claim most of the extension too.
            for cell in panel.cells(self.lad, arm)[:1995]:
                (self.root/'claims').mkdir(exist_ok=True); (self.root/'claims'/f"{cell['ordinal']}.json").write_bytes(b'{}')

    def tearDown(self): self.directory.cleanup()

    def pin(self, name, value):
        path = Path(self.directory.name)/f'{name}.json'; panel.write_new(path, value); return path, panel.sha(value)

    def test_extension_needs_a_genuine_continuing_screen_readout(self):
        with self.assertRaisesRegex(ValueError, 'pinned 500-cell'): panel.pending(self.root, arm='DN2W8', limit=2000, resume=True)
        genuine = panel.readout(self.root, arm='DN2W8', limit=500); self.assertEqual(genuine['decision'], 'CONTINUE_TO_2000')
        screen = self.pin('go', genuine)
        self.assertEqual(len(panel.pending(self.root, arm='DN2W8', limit=2000, resume=True, screen=screen)), 5)
        self.assertEqual(len(panel.pending(self.root, arm='RAW', limit=2000, resume=True, screen=screen)), 5)
        with self.assertRaisesRegex(ValueError, 'differs'): panel.pending(self.root, arm='DN2W8', limit=2000, resume=True, screen=(screen[0], '0'*64))

    def test_stopped_foreign_or_mismatched_readouts_are_refused(self):
        genuine = panel.readout(self.root, arm='DN2W8', limit=500)
        cases = dict(stop=dict(genuine, decision='STOP_BREAKAGE'), incomplete=dict(genuine, decision='INCOMPLETE'),
                     declaration=dict(genuine, declaration_sha256='e'*64), cells=dict(genuine, paired_cells=2000))
        for name, value in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, 'pinned 500-cell'):
                panel.pending(self.root, arm='DN2W8', limit=2000, resume=True, screen=self.pin(name, value))
        with self.assertRaisesRegex(ValueError, 'pinned 500-cell'):   # another arm's screen: refused before any recomputation
            panel.pending(self.root, arm='DN2W8', limit=2000, resume=True, screen=self.pin('other-arm', dict(genuine, arm='DN2W16')))

    def test_a_relabelled_or_edited_readout_fails_recomputation(self):
        genuine = panel.readout(self.root, arm='DN2W8', limit=500)
        edited = copy.deepcopy(genuine); edited['paired']['interval90'][0] = .5
        with self.assertRaisesRegex(ValueError, 'recomputation'):
            panel.pending(self.root, arm='DN2W8', limit=2000, resume=True, screen=self.pin('edited', edited))
        cell = panel.cells(self.lad, 'DN2W8', limit=500)[3]
        write_cell(self.root, cell, self.run['declaration_sha256'], score=None, status='UNKNOWN', error='RuntimeError: late', rows=[])
        broken = panel.readout(self.root, arm='DN2W8', limit=500); self.assertEqual(broken['decision'], 'STOP_BREAKAGE')
        for name, value in (('relabelled', dict(broken, decision='CONTINUE_TO_2000')), ('stale', genuine)):
            with self.subTest(name), self.assertRaisesRegex(ValueError, 'recomputation'):
                panel.pending(self.root, arm='DN2W8', limit=2000, resume=True, screen=self.pin(name, value))


class FakeRuntime:
    made = []
    runs = []
    def __init__(self, *, client, telemetry_path, arm, cell, original, archive_sha256):
        self.client, self.telemetry_path, self.arm, self.cell, self.original = client, telemetry_path, arm, cell, original
        self.archive_sha256 = archive_sha256
        FakeRuntime.made.append(self)
    async def run(self, config):
        FakeRuntime.runs.append(self.cell['ordinal'])
        self.claimed = (Path(self.telemetry_path).parents[1]/'claims'/f"{self.cell['ordinal']}.json").exists()
        return FakeRuntime.result(config, self.cell)

    @staticmethod
    def result(config, cell):
        sha = ActorContract.champion().checkpoint_sha256; game = dict(seed=cell['seed'], pokezero_decisions=3)
        document = dict(game_results=[game], checkpoint_sha256=sha, policy_mode='raw', seed_start=cell['seed'],
            pokezero_player=config.pokezero_player, encoder_kind=config.encoder_kind, capture_driver=config.capture_driver,
            format_id=config.format_id, foulplay_random_seed_schedule=dict(seeds=[cell['fp_seed']]))
        return NS(config=config, checkpoint_sha256=sha, foulplay_random_seed_schedule=(cell['fp_seed'],),
                  games=(NS(to_dict=lambda: game),), to_dict=lambda: document)


@contextmanager
def inert_capture(*a, **kw): yield


class WorkerTests(unittest.TestCase):
    def setUp(self):
        FakeRuntime.made.clear(); FakeRuntime.runs.clear()
        self.directory = TemporaryDirectory(); self.root, self.run, self.d, self.lad = make_run(self.directory.name)
        self.config = NS(checkpoint=Path('/inert/champion.pt'), pokezero_player='p1', encoder_kind='jax-owning',
                         capture_driver='checkpoint', format_id='gen3randombattle')
        sha = ActorContract.champion().checkpoint_sha256
        self.patches = [patch.object(panel.ladder, 'configuration', return_value=self.config),
            patch.object(panel.ladder, 'owning_capture', inert_capture),
            patch('pokezero.live_protocol_runtime._checkpoint_sha', return_value=sha),
            patch('pokezero.v5_foulplay._score', return_value=1.)]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in self.patches: p.stop()
        self.directory.cleanup()

    def packet(self, arm, pair=0, server=None, **cell_change):
        cell = panel.cells(self.lad, arm)[pair]; path = self.root/'requests'/f"{cell['ordinal']}.json"
        value = panel.request(self.run, dict(cell, **cell_change))
        if not path.exists(): panel.write_new(path, value)
        return dict(path=str(path), sha256=panel.sha(value), server=server), cell

    def work(self, packet):
        return asyncio.run(panel.worker(packet, expected_pair=(1, 2), original='shared', runtime_class=FakeRuntime))

    def result(self, cell): return json.loads((self.root/'results'/f"{cell['ordinal']}.json").read_bytes())

    def test_search_cell_completes_with_its_client_and_is_never_rerun(self):
        packet, cell = self.packet('DN2W8', 3, server=dict(host='127.0.0.1', port=9))
        self.assertEqual(self.work(packet), 0)
        runtime, = FakeRuntime.made
        self.assertTrue(runtime.claimed); self.assertEqual(runtime.original, 'shared')
        self.assertIs(type(runtime.client), live.Client); self.assertEqual((runtime.client.rung, runtime.client.deadline, runtime.client.reserve), ('DN2W8', 3., .15))
        value = self.result(cell)
        self.assertEqual((value['status'], value['score'], value['schema'], value['cell']), ('COMPLETE', 1., panel.CELL_SCHEMA, cell))
        self.assertEqual(value['server'], dict(host='127.0.0.1', port=9)); self.assertTrue(value['telemetry'].endswith(f"telemetry/{cell['ordinal']}.jsonl"))
        self.assertEqual(value['owning_wire'], str(self.root/'owning-wire'/f"{cell['ordinal']}.jsonl"))
        self.assertEqual(runtime.archive_sha256, live.CHAMPION_ARCHIVE_SHA256)
        claim = (self.root/'claims'/f"{cell['ordinal']}.json").read_bytes()
        with self.assertRaises(FileExistsError): self.work(packet)
        self.assertEqual(FakeRuntime.runs, [cell['ordinal']]); self.assertEqual((self.root/'claims'/f"{cell['ordinal']}.json").read_bytes(), claim)

    def test_the_declared_deadline_reaches_the_client_and_the_cell_records_its_host(self):
        root, run, d, lad = make_run(Path(self.directory.name)/'ten', deadline=10.)
        cell = panel.cells(lad, 'DN2W8')[0]; path = root/'requests'/f"{cell['ordinal']}.json"; value = panel.request(run, cell)
        panel.write_new(path, value)
        with patch.dict(os.environ, {'POD_NAME': 'pod-x'}):
            asyncio.run(panel.worker(dict(path=str(path), sha256=panel.sha(value), server=dict(host='127.0.0.1', port=9)),
                                     expected_pair=(1, 2), original='shared', runtime_class=FakeRuntime))
        runtime = FakeRuntime.made[-1]; self.assertEqual((runtime.client.deadline, runtime.client.reserve), (10., .15))
        result = json.loads((root/'results'/f"{cell['ordinal']}.json").read_bytes())
        self.assertEqual((result['worker']['hostname'], result['worker']['pod_name'], result['cpu_pair']), (socket.gethostname(), 'pod-x', [1, 2]))
        self.assertEqual((len(result['load_average']['start']), len(result['load_average']['end'])), (3, 3))

    def test_interrupted_game_is_unknown_and_claimed(self):
        packet, cell = self.packet('RAW', 0)
        with patch.object(FakeRuntime, 'run', side_effect=RuntimeError('inert interrupted game')):
            self.assertEqual(self.work(packet), 0)
        value = self.result(cell); self.assertEqual((value['status'], value['score']), ('UNKNOWN', None))
        self.assertIn('inert interrupted', value['error']); self.assertIsNone(FakeRuntime.made[0].client)
        self.assertTrue((self.root/'claims'/f"{cell['ordinal']}.json").exists())
        with self.assertRaises(FileExistsError): self.work(packet)

    def test_refusals_precede_the_claim(self):
        cases = [self.packet('RAW', 1, server=dict(host='h', port=1)), self.packet('DN2W8', 1),
                 self.packet('DN2W8', 2, server=dict(host='h', port=1, extra=1)), self.packet('DN2W8', 4, server=dict(host='h', port=0))]
        bad, cell = self.packet('RAW', 5); cases.append((dict(bad, sha256='0'*64), cell))
        tampered, cell = self.packet('RAW', 6, seed=1); cases.append((tampered, cell))
        d16 = panel.cells(self.lad, 'DN2W16')[0]; path = self.root/'requests'/f"{d16['ordinal']}.json"
        value = panel.request(self.run, d16); panel.write_new(path, value)
        cases.append((dict(path=str(path), sha256=panel.sha(value), server=dict(host='h', port=1)), d16))
        moved, cell = self.packet('RAW', 7); other = self.root/'elsewhere.json'; other.write_bytes(Path(moved['path']).read_bytes())
        cases.append((dict(moved, path=str(other)), cell))
        for packet, cell in cases:
            with self.subTest(packet=packet), self.assertRaises(ValueError): self.work(packet)
            self.assertFalse((self.root/'claims'/f"{cell['ordinal']}.json").exists())
        packet, cell = self.packet('RAW', 8)
        with patch.object(panel.ladder, 'configuration', side_effect=ValueError('B2 protocol-fed encoder required')), self.assertRaises(ValueError):
            self.work(packet)
        self.assertFalse((self.root/'claims'/f"{cell['ordinal']}.json").exists()); self.assertEqual(FakeRuntime.made, [])
        self.assertEqual(FakeRuntime.runs, [])


def make_capture(directory, ordinal=0, turns=3):
    with capture.capture_to(directory, cell={'ordinal': ordinal}, declaration_sha256='a'*64): a = capture.capture_actor(fixture.actor())
    a.begin_game(battle_id='authored', owner='p1', seed=1, stream_id=1)
    for turn in range(1, turns+1):
        fixture.feed(a, (f'|turn|{turn}', fixture.request())); s = a.snapshot_boundary(); a.decide(s, rng=random.Random(7))
    a.close_game()
    return Path(directory)/f'{ordinal}.jsonl'


def champion_complete(request, **change):
    return {**actor_tests.complete(request), 'archive_sha256': live.CHAMPION_ARCHIVE_SHA256,
            'pricer': dict(k=[9, 9], depth=2, W=8, K=4, matmul_precision='highest'), **change}


def champion_refused(request, cause):
    return dict(actor_tests.refused(request, cause), archive_sha256=live.CHAMPION_ARCHIVE_SHA256)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(); self.root, self.run, self.d, self.lad = make_run(self.directory.name)
        self.capture = make_capture(Path(self.directory.name)/'wire')

    def tearDown(self): self.directory.cleanup()

    def probe(self, reply):
        with actor_tests.FakeServer(reply) as server:
            try: return panel.probe(self.root, arm='DN2W8', server=dict(host='127.0.0.1', port=server.port), capture=self.capture), server
            finally: self.requests = server.requests

    def records(self): return [json.loads(p.read_bytes()) for p in sorted((self.root/'probes').glob('*.json'))]

    def test_pass_on_the_declared_rung_worlds_and_operator(self):
        value, server = self.probe(actor_tests.send(champion_complete))
        self.assertEqual((value['status'], len(value['tried'])), ('PASS', 1)); self.assertEqual(self.records(), [value])
        request, = self.requests; root = next(capture.read_capture(self.capture))
        self.assertTrue(request['battle_id'].startswith('probe-')); self.assertEqual(request['rung'], 'DN2W8')
        self.assertEqual((request['lines'], request['own_request_line'], request['prefix_sha256']),
                         (list(root['lines']), root['own_request_line'], root['prefix_sha256']))

    def test_the_probe_budgets_by_the_declared_deadline(self):
        root, run, d, lad = make_run(Path(self.directory.name)/'ten', deadline=10.)
        with actor_tests.FakeServer(actor_tests.send(champion_complete)) as server:
            value = panel.probe(root, arm='DN2W8', server=dict(host='127.0.0.1', port=server.port), capture=self.capture)
            request, = server.requests
        self.assertEqual(value['status'], 'PASS'); self.assertTrue(3. < request['budget_seconds'] <= 9.85)

    def test_constructor_refusals_move_to_the_next_root(self):
        replies = iter(['constructor_refused', 'class_refused', 'depthn_refused', None])
        def reply(request, conn):
            cause = next(replies)
            actor_tests.send(lambda r: champion_complete(r) if cause is None else champion_refused(r, cause))(request, conn)
        self.capture = make_capture(Path(self.directory.name)/'wire4', turns=4)
        value, _ = self.probe(reply)
        self.assertEqual([t['status'] for t in value['tried']], ['REFUSED']*3+['COMPLETE']); self.assertEqual(value['status'], 'PASS')

    def test_wrong_or_dead_servers_fail_and_are_recorded(self):
        cases = dict(worlds=actor_tests.send(lambda r: champion_complete(r, requested_worlds=16)),
                     operator=actor_tests.send(lambda r: champion_complete(r, root_operator='ARGMAX_POLICY_TARGET')),
                     rung=actor_tests.send(lambda r: champion_complete(r, rung='DN2W16')),
                     archive=actor_tests.send(lambda r: champion_complete(r, archive_sha256='a'*64)),
                     precision=actor_tests.send(lambda r: champion_complete(r, pricer=dict(W=8, matmul_precision='default'))),
                     no_precision=actor_tests.send(lambda r: champion_complete(r, pricer=None)),
                     refusal_archive=actor_tests.send(lambda r: dict(champion_refused(r, 'constructor_refused'), archive_sha256='a'*64)),
                     server_error=actor_tests.send(lambda r: champion_refused(r, 'server_error')),
                     all_refused=actor_tests.send(lambda r: champion_refused(r, 'constructor_refused')))
        for name, reply in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, 'probe failed'): self.probe(reply)
            self.assertEqual(self.records()[-1]['status'], 'FAIL')
            if 'archive' in name: self.assertIn('archive', self.records()[-1]['reason'])
        self.assertEqual(len(self.records()[-1]['tried']), 3)
        dead = socket.create_server(('127.0.0.1', 0)); port = dead.getsockname()[1]; dead.close()
        with self.assertRaisesRegex(ValueError, 'connection_error'):
            panel.probe(self.root, arm='DN2W8', server=dict(host='127.0.0.1', port=port), capture=self.capture)
        with self.assertRaisesRegex(ValueError, 'search arm'):
            panel.probe(self.root, arm='RAW', server=dict(host='127.0.0.1', port=port), capture=self.capture)

    def test_default_capture_is_the_first_closed_raw_cell(self):
        with self.assertRaisesRegex(ValueError, 'no closed RAW capture'): panel.probe_capture(self.root)
        cell = panel.cells(self.lad, 'RAW', limit=500)[2]; make_capture(self.root/'owning-wire', cell['ordinal'])
        with self.assertRaisesRegex(ValueError, 'no closed RAW capture'): panel.probe_capture(self.root)
        write_cell(self.root, cell, self.run['declaration_sha256'], score=1.)
        self.assertEqual(panel.probe_capture(self.root), self.root/'owning-wire'/f"{cell['ordinal']}.jsonl")


FAKE_WORKER = r'''
import json, os, sys, time
for line in sys.stdin:
    packet = json.loads(line)
    if os.path.basename(packet['path']).startswith('fail'): sys.exit(3)
    if os.path.basename(packet['path']).startswith('slow'): time.sleep(.8)
    with open(packet['path'] + '.done', 'x') as stream: json.dump(dict(pid=os.getpid(), server=packet['server']), stream)
    print(json.dumps({'exit': 0}), flush=True)
'''


SERVERS = {'RAW': None, 'DN2W8': dict(host='h', port=1)}


class DispatchTests(unittest.TestCase):
    def run_dispatch(self, names, arms=None, servers=SERVERS, **kwargs):
        directory = Path(self.directory.name)
        arms = arms or ['RAW' if i % 2 == 0 else 'DN2W8' for i in range(len(names))]
        packets = [dict(path=str(directory/n), sha256='0'*64, arm=a) for n, a in zip(names, arms)]
        work = panel.dispatch(packets, servers=servers, command=lambda pair, cache: [sys.executable, '-c', FAKE_WORKER], **kwargs)
        result = asyncio.run(asyncio.wait_for(work, DISPATCH_TIMEOUT))
        done = {n: json.loads((directory/(n+'.done')).read_text()) for n in names if (directory/(n+'.done')).exists()}
        return result, done

    def setUp(self): self.directory = TemporaryDirectory()
    def tearDown(self): self.directory.cleanup()

    def test_one_cell_per_fresh_worker_process_by_default(self):
        result, done = self.run_dispatch(['a', 'b', 'c'], pairs=[(0, 1), (2, 3)])
        self.assertEqual(result, dict(dispatched=3, failed=[], not_dispatched=0, stopped=None))
        self.assertEqual(len({v['pid'] for v in done.values()}), 3)
        self.assertEqual((done['a']['server'], done['b']['server']), (None, dict(host='h', port=1)))

    def test_persistent_worker_when_asked(self):
        result, done = self.run_dispatch(['a', 'b', 'c'], pairs=[(0, 1)], cells_per_worker=5)
        self.assertEqual(result['dispatched'], 3); self.assertEqual(len({v['pid'] for v in done.values()}), 1)

    def test_a_failed_worker_stops_all_dispatch(self):
        for kwargs in (dict(pairs=[(0, 1)]), dict(pairs=[(0, 1)], cells_per_worker=4), dict(pairs=[(0, 1), (2, 3)])):
            with self.subTest(kwargs):
                self.directory.cleanup(); self.directory = TemporaryDirectory()
                result, done = self.run_dispatch(['a', 'fail', 'c', 'd', 'e', 'f'], **kwargs)
                self.assertGreaterEqual(len(result['failed']), 1); self.assertTrue(any(f.endswith('fail') for f in result['failed']))
                self.assertLessEqual(result['dispatched'], 2); self.assertGreaterEqual(result['not_dispatched'], 3)
                self.assertNotIn('f', done)

    def test_a_lane_failure_stops_a_persistent_worker_in_another_lane(self):
        names = ['slow-a', 'fail']+[f'slow-{c}' for c in 'cdefgh']
        result, done = self.run_dispatch(names, pairs=[(0, 1), (2, 3)], cells_per_worker=10)
        self.assertTrue(any(f.endswith('fail') for f in result['failed'])); self.assertLessEqual(result['dispatched'], 2)
        self.assertGreaterEqual(result['not_dispatched'], 5); self.assertNotIn('slow-h', done)

    def test_no_arm_runs_more_than_the_concurrency_ahead(self):
        for pairs, dispatched in (([(0, 1)], 1), ([(0, 1), (2, 3)], 2)):
            with self.subTest(lanes=len(pairs)):
                self.directory.cleanup(); self.directory = TemporaryDirectory()
                result, done = self.run_dispatch(['a', 'b', 'c', 'd', 'e'], arms=['RAW']*5, pairs=pairs, cells_per_worker=5)
                self.assertEqual(result['dispatched'], dispatched); self.assertTrue(result['stopped'].startswith('interleave'))
                self.assertEqual(result['not_dispatched'], 5-dispatched)

    def test_a_resume_catches_the_lagging_arm_up_first(self):
        order = []
        result, _ = self.run_dispatch(['r0', 'd0', 'r1', 'd1', 'r2', 'd2'], pairs=[(0, 1)], cells_per_worker=10,
                                      attempted={'RAW': 3, 'DN2W8': 0}, after=lambda packet: order.append(Path(packet['path']).name))
        self.assertEqual(order, ['d0', 'd1', 'd2', 'r0']); self.assertTrue(result['stopped'].startswith('interleave'))

    def test_the_after_hook_stops_dispatch(self):
        calls = []
        def after(packet):
            calls.append(packet['arm']); return 'VOID_LATENCY' if len(calls) == 2 else None
        result, done = self.run_dispatch(['a', 'b', 'c', 'd', 'e', 'f'], pairs=[(0, 1)], cells_per_worker=10, after=after)
        self.assertEqual((result['stopped'], result['dispatched'], result['not_dispatched'], calls), ('VOID_LATENCY', 2, 4, ['RAW', 'DN2W8']))

    def test_core_pairs_must_be_disjoint_and_every_arm_needs_a_server(self):
        for kwargs in (dict(pairs=[(0, 1), (1, 2)]), dict(pairs=[(0, 0)]), dict(pairs=[]), dict(pairs=[(0, 1)], cells_per_worker=0)):
            with self.subTest(kwargs), self.assertRaises(ValueError): asyncio.run(panel.dispatch([], servers={'RAW': None}, **kwargs))
        with self.assertRaisesRegex(ValueError, 'every arm'):
            asyncio.run(panel.dispatch([dict(path='x', sha256='0'*64, arm='DN2W8')], servers={'RAW': None}, pairs=[(0, 1)]))


class EstimatorTests(unittest.TestCase):
    def roster(self, rng, limit):
        rows = {'RAW': [], 'search': []}; p_missing = rng.choice((0., .02, .1)); tilt = rng.uniform(-.3, .3)
        lad = ladder_fixture.declaration()
        for arm, slot, bias in (('RAW', 'RAW', 0.), ('search', 'R1', tilt)):
            ours, codex = panel.cells(lad, 'RAW' if arm == 'RAW' else 'DN2W8', limit=limit), ladder.cells(lad, slot, limit=limit)
            for cell, codex_cell in zip(ours, codex):
                draw = rng.random()
                if draw < p_missing/2: continue
                if draw < p_missing: row = dict(status='UNKNOWN', score=None)
                else: row = dict(status='COMPLETE', score=rng.choices((0., .5, 1.), weights=(1-.5*bias, .1, 1+.5*bias))[0])
                rows[arm].append((dict(row, cell=cell), dict(row, cell=codex_cell, declaration_sha256=panel.sha(lad))))
        return lad, rows

    def test_equals_codex_paired_on_randomized_rosters_with_missing_cells(self):
        from pokezero import phase_d_ladder_results as results
        rng = random.Random(20261107); lad = ladder_fixture.declaration()
        valid = {arm: {panel.encoded(c) for c in ladder.cells(lad, arm)} for arm in ('RAW', 'R1')}
        def checked(d, cell):  # Codex's per-row check rebuilds 2000 cells; same check by lookup.
            if panel.encoded(cell) not in valid[cell['arm']]: raise ValueError('fixed ladder cell address/seed/seat differs')
            return cell
        for trial in range(24):
            limit = 2000 if trial % 6 == 0 else 500
            lad, rows = self.roster(rng, limit)
            ours = panel.paired([a for a, _ in rows['RAW']], [a for a, _ in rows['search']], limit=limit)
            with patch.object(ladder, 'checked_cell', checked):
                codex = results.paired(lad, [b for _, b in rows['RAW']], [b for _, b in rows['search']], rung='R1', limit=limit,
                                       expected_declaration_sha256=panel.sha(lad))
            with self.subTest(trial=trial):
                for key, value in ours.items(): self.assertEqual(value, codex[key], key)
        self.assertEqual(set(ours), {'paired_cells','known_pairs','unknown_pairs','point_difference','full_roster_identification_bounds',
            'complete_seed_clusters','complete_seed_difference','interval90','interval_status','discordant_known','discordant_known_rate',
            'discordant_full_bounds'})

    def test_degenerate_rosters_and_literal_scores(self):
        lad = ladder_fixture.declaration(); same = [dict(cell=c, status='COMPLETE', score=1.) for c in panel.cells(lad, 'RAW', limit=500)]
        value = panel.paired(same, [dict(r) for r in same], limit=500)
        self.assertIsNone(value['interval90']); self.assertEqual(value['point_difference'], 0.)
        value = panel.paired(same[:3], same[:3], limit=500)
        self.assertEqual((value['complete_seed_clusters'], value['interval90'], value['point_difference']), (1, None, None))
        for bad in (dict(same[0], score=.7), dict(same[0], status='UNKNOWN')):
            with self.assertRaisesRegex(ValueError, 'literal'): panel.paired([bad], [], limit=500)


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(); self.root, self.run, self.d, self.lad = make_run(self.directory.name)

    def tearDown(self): self.directory.cleanup()

    def readout(self, limit=500, **kwargs): return panel.readout(self.root, arm='DN2W8', limit=limit, **kwargs)

    def test_clean_screen_continues_with_the_seed_cluster_interval(self):
        populate(self.root, self.run, self.lad)
        value = self.readout()
        deltas = [float(p//2 % 3 != 1)-float(p//2 % 3 == 0) for p in range(500)]
        self.assertEqual(value['decision'], 'CONTINUE_TO_2000'); self.assertEqual(value['breakage'], [])
        self.assertAlmostEqual(value['paired']['point_difference'], fmean(deltas), places=12)
        for got, want in zip(value['paired']['interval90'], seed_cluster_interval(deltas)): self.assertAlmostEqual(got, want, places=12)
        self.assertEqual((value['paired']['known_pairs'], value['paired']['complete_seed_clusters']), (500, 250))
        self.assertIsNone(value['bar']); self.assertEqual(value['arms']['DN2W8']['decisions']['decisions'], 1000)
        self.assertEqual(value['dilution'], dict(searched_share=1., fallback_share=0., search_differs_share_of_all=.5))
        self.assertEqual(value['arms']['RAW']['decisions']['by_source'], {'raw_arm': 1000})
        self.assertEqual(value['tree'], self.run['tree']); json.dumps(value, allow_nan=False)

    def test_an_interval_straddling_zero_continues(self):
        populate(self.root, self.run, self.lad, scores=lambda p: (float(p//2 % 2 == 0), float(p//2 % 2 == 1)))
        value = self.readout(); low, high = value['paired']['interval90']
        self.assertLess(low, 0); self.assertGreater(high, 0); self.assertEqual(value['decision'], 'CONTINUE_TO_2000')

    def test_harm_stops_a_complete_screen_only(self):
        harmful = lambda p: (float(p % 7 != 0), float(p % 5 == 0))
        populate(self.root, self.run, self.lad, scores=harmful, skip={499})
        value = self.readout(); self.assertTrue(value['harm']); self.assertEqual(value['decision'], 'INCOMPLETE')
        populate(self.root, self.run, self.lad, scores=harmful)
        value = self.readout()
        self.assertLess(value['paired']['interval90'][1], 0); self.assertEqual(value['decision'], 'STOP_HARM')

    def test_breakage_stops_the_screen(self):
        populate(self.root, self.run, self.lad, select=3.5)
        value = self.readout()
        self.assertEqual((value['decision'], value['breakage']), ('STOP_BREAKAGE', ['latency_p95']))
        self.assertEqual(value['arms']['DN2W8']['decisions']['late_decisions'], 1000)
        cell = panel.cells(self.lad, 'DN2W8', limit=500)[9]
        write_cell(self.root, cell, self.run['declaration_sha256'], score=None, status='UNKNOWN',
                   error='TimeoutError: owning candidate decision deadline expired', rows=[decision_row(cell, 0)])
        raw = panel.cells(self.lad, 'RAW', limit=500)[11]
        write_cell(self.root, raw, self.run['declaration_sha256'], score=1., rows=[decision_row(raw, 0, source='raw_arm')], decisions=5)
        value = self.readout()
        self.assertEqual(value['breakage'], ['unknown_cells', 'timeout_forfeits', 'latency_p95', 'telemetry'])
        self.assertEqual(value['arms']['RAW']['telemetry_mismatch_pairs'], [11]); self.assertEqual(value['arms']['DN2W8']['timeout_forfeits'], 1)
        self.assertEqual(value['paired']['unknown_pairs'], 1); self.assertIsNone(value['paired']['point_difference'])

    def test_search_provenance_and_worlds_must_match_the_declaration(self):
        populate(self.root, self.run, self.lad)
        cell = panel.cells(self.lad, 'DN2W8', limit=500)[0]
        write_cell(self.root, cell, self.run['declaration_sha256'], score=1., rows=[decision_row(cell, 0, operator='ARGMAX_POLICY_TARGET')])
        self.assertEqual(self.readout()['breakage'], ['search_provenance'])
        write_cell(self.root, cell, self.run['declaration_sha256'], score=1., rows=[decision_row(cell, 0, worlds=16)])
        value = self.readout()
        self.assertEqual(value['breakage'], ['requested_worlds']); self.assertEqual(value['arms']['DN2W8']['decisions']['wrong_worlds'], 1)

    def test_a_dead_or_failing_server_is_breakage(self):
        def rows(cause):
            def build(cell):
                if cell['arm'] == 'RAW': return [decision_row(cell, i, source='raw_arm') for i in range(2)]
                return [decision_row(cell, 0, source='greedy_fallback', cause=cause), decision_row(cell, 1, source='single_legal')]
            return build
        for cause, expected in (('connection_error', ['transport_failures', 'searched_share']), ('timeout', ['transport_failures', 'searched_share']),
                                ('server_error', ['transport_failures', 'searched_share']), ('constructor_refused', ['searched_share']),
                                ('class_refused', ['searched_share'])):
            with self.subTest(cause):
                populate(self.root, self.run, self.lad, rows_for=rows(cause))
                value = self.readout(); search = value['arms']['DN2W8']['decisions']
                self.assertEqual(value['breakage'], expected); self.assertEqual(value['decision'], 'STOP_BREAKAGE')
                self.assertEqual((search['eligible_decisions'], search['searched_share'], search['fallback_share']), (500, 0., 1.))
        def failing(count, cause='timeout'):
            def build(cell):
                if cell['arm'] == 'RAW': return [decision_row(cell, 0, source='raw_arm')]
                if cell['pair_ordinal'] < count: return [decision_row(cell, 0, source='greedy_fallback', cause=cause)]
                return [decision_row(cell, 0)]
            return build
        for count, cause, share, breakage in ((10, 'server_error', .02, []), (11, 'server_error', .022, ['transport_failures']),
                                              (15, 'timeout', .03, ['transport_failures']), (60, 'class_refused', 0., [])):
            with self.subTest(count=count, cause=cause):
                populate(self.root, self.run, self.lad, rows_for=failing(count, cause)); value = self.readout()
                search = value['arms']['DN2W8']['decisions']
                self.assertEqual((search['transport_failure_share'], value['breakage']), (share, breakage))
                self.assertEqual(search['fallback_by_cause'][cause]['count'], count)

    def test_unattempted_raw_cells_keep_the_screen_incomplete(self):
        populate(self.root, self.run, self.lad, arms=('RAW',), skip={499}); populate(self.root, self.run, self.lad, arms=('DN2W8',))
        value = self.readout()
        self.assertEqual((value['arms']['RAW']['not_attempted'], value['arms']['DN2W8']['not_attempted']), (1, 0))
        self.assertEqual(value['decision'], 'INCOMPLETE')

    def test_a_server_on_another_archive_is_breakage(self):
        populate(self.root, self.run, self.lad, rows_for=lambda cell: [decision_row(cell, 0, source='raw_arm')] if cell['arm'] == 'RAW'
                 else [decision_row(cell, 0, archive='a'*64)])
        value = self.readout()
        self.assertEqual(value['breakage'], ['archive']); self.assertEqual(value['arms']['DN2W8']['decisions']['wrong_archive'], 500)
        self.assertEqual(value['decision'], 'STOP_BREAKAGE')

    def test_searched_share_floor_is_fifty_percent(self):
        def share(refused):
            def build(cell):
                if cell['arm'] == 'RAW': return [decision_row(cell, 0, source='raw_arm')]
                if cell['pair_ordinal'] < refused: return [decision_row(cell, 0, source='greedy_fallback', cause='constructor_refused')]
                return [decision_row(cell, 0)]
            return build
        for refused, searched, breakage in ((200, .6, []), (250, .5, []), (300, .4, ['searched_share'])):
            with self.subTest(searched):
                populate(self.root, self.run, self.lad, rows_for=share(refused)); value = self.readout()
                self.assertEqual((value['arms']['DN2W8']['decisions']['searched_share'], value['breakage']), (searched, breakage))

    def test_expected_searched_share_is_descriptive_and_never_changes_a_decision(self):
        def refuse_most(cell):
            if cell['arm'] == 'RAW': return [decision_row(cell, 0, source='raw_arm')]
            return [decision_row(cell, 0, source='greedy_fallback', cause='depthn_refused')] if cell['pair_ordinal'] < 350 else [decision_row(cell, 0)]
        scenarios = dict(clean=dict(limit=500), floor=dict(limit=500, rows_for=refuse_most), bar=dict(limit=2000))
        keys = ('decision', 'harm', 'breakage', 'bar', 'paired', 'dilution', 'arms')
        for name, kwargs in scenarios.items():
            outcomes = []
            for i, expected in enumerate((None, dict(EXPECTED, value=.05), dict(EXPECTED, value=.99)) if name != 'bar' else (EXPECTED, None)):
                root, run, d, lad = make_run(Path(self.directory.name)/f'{name}-{i}', expected=expected)
                populate(root, run, lad, **kwargs); value = panel.readout(root, arm='DN2W8', limit=kwargs['limit'])
                outcomes.append(value); comparison = value['expected_searched_share']; observed = value['dilution']['searched_share']
                self.assertEqual((comparison['scope'], comparison['observed']), ('DESCRIPTIVE_NOT_A_GATE', observed))
                if expected is None: self.assertEqual((comparison['expected'], comparison['difference'], comparison['source']), (None, None, None))
                else:
                    self.assertEqual((comparison['expected'], comparison['source']), (expected['value'], expected['source']))
                    self.assertAlmostEqual(comparison['difference'], observed-expected['value'], places=12)
            with self.subTest(name):
                self.assertEqual({tuple(json.dumps(v[k], sort_keys=True) for k in keys) for v in outcomes}.__len__(), 1)
                self.assertEqual(outcomes[0]['decision'], dict(clean='CONTINUE_TO_2000', floor='STOP_BREAKAGE', bar='BAR_MET')[name])

    def test_unattempted_cells_are_incomplete_but_lost_claims_are_unknown(self):
        populate(self.root, self.run, self.lad, skip={498, 499})
        value = self.readout()
        self.assertEqual(value['decision'], 'INCOMPLETE'); self.assertEqual(value['arms']['DN2W8']['not_attempted'], 2)
        self.assertEqual(value['paired']['unknown_pairs'], 2)
        cell = panel.cells(self.lad, 'DN2W8', limit=500)[498]; panel.write_new(self.root/'claims'/f"{cell['ordinal']}.json", {})
        for c in panel.cells(self.lad, 'RAW', limit=500)[498:]+panel.cells(self.lad, 'DN2W8', limit=500)[499:]:
            panel.write_new(self.root/'claims'/f"{c['ordinal']}.json", {})
        value = self.readout()
        self.assertEqual(value['arms']['DN2W8']['claimed_without_result'], 2); self.assertEqual(value['decision'], 'STOP_BREAKAGE')

    def test_bar_met_at_two_thousand(self):
        populate(self.root, self.run, self.lad, limit=2000)
        value = self.readout(2000)
        self.assertGreater(value['paired']['interval90'][0], 0); self.assertEqual(value['breakage'], [])
        self.assertEqual(value['bar'], dict(rule=self.d['bar'], met=True, met_full_roster=True, breakage_free=True))
        self.assertEqual(value['decision'], 'BAR_MET')

    def test_positive_point_with_lower_bound_below_zero_is_not_met(self):
        populate(self.root, self.run, self.lad, limit=2000,
                 scores=lambda p: (0., 1.) if (p//2) % 2 == 0 or p//2 == 1 else (1., 0.))
        value = self.readout(2000); low, high = value['paired']['interval90']
        self.assertGreater(value['paired']['point_difference'], 0); self.assertLess(low, 0); self.assertGreater(high, 0)
        self.assertEqual(value['decision'], 'BAR_NOT_MET'); self.assertFalse(value['bar']['met'])

    def test_bar_needs_no_breakage_and_the_full_roster(self):
        populate(self.root, self.run, self.lad, limit=2000, select=3.5)
        value = self.readout(2000)
        self.assertEqual((value['decision'], value['bar']['met'], value['bar']['breakage_free']), ('BAR_NOT_MET', True, False))
        skip = set(random.Random(5).sample(range(2000), 700)); root, run, d, lad = make_run(Path(self.directory.name)/'lost')
        populate(root, run, lad, limit=2000, skip=skip)
        for arm in ('RAW', 'DN2W8'):
            for cell in panel.cells(lad, arm, limit=2000):
                if cell['pair_ordinal'] in skip: panel.write_new(root/'claims'/f"{cell['ordinal']}.json", {})
        value = panel.readout(root, arm='DN2W8', limit=2000)
        self.assertTrue(value['bar']['met']); self.assertFalse(value['bar']['met_full_roster'])
        self.assertLess(value['paired']['full_roster_identification_bounds'][0], 0)
        self.assertIn('unknown_cells', value['breakage']); self.assertEqual(value['decision'], 'BAR_NOT_MET')

    def test_readout_refuses_foreign_results_and_undeclared_arms(self):
        populate(self.root, self.run, self.lad)
        with self.assertRaises(ValueError): panel.readout(self.root, arm='DN2W16', limit=500)
        with self.assertRaises(ValueError): self.readout(1000)
        cell = panel.cells(self.lad, 'RAW', limit=500)[0]
        write_cell(self.root, cell, 'f'*64, score=1., rows=[decision_row(cell, 0, source='raw_arm')])
        with self.assertRaisesRegex(ValueError, 'foreign'): self.readout()

    def test_codex_raw_agreement_is_a_plumbing_check(self):
        populate(self.root, self.run, self.lad)
        codex = Path(self.directory.name)/'codex'; (codex/'results').mkdir(parents=True); lad_sha = self.run['ladder_declaration_sha256']
        for cell in ladder.cells(self.lad, 'RAW', limit=500)[:10]:
            ours = float(cell['pair_ordinal']//2 % 3 == 0); score = ours if cell['pair_ordinal'] != 4 else 1.-ours
            (codex/'results'/f"{cell['ordinal']}.json").write_bytes(panel.encoded(dict(cell=cell, status='COMPLETE', score=score, declaration_sha256=lad_sha)))
        value = self.readout(codex_raw_directory=codex)
        agreement = value['codex_raw_agreement']; self.assertEqual(value['codex_raw_directory'], str(codex.resolve()))
        self.assertEqual((agreement['codex_results'], agreement['shared_complete'], agreement['agree'], agreement['disagreeing_pairs']), (10, 10, 9, [4]))
        self.assertEqual(agreement['scope'], 'PLUMBING_CHECK_ONLY_NOT_THE_COMPARISON')
        first = ladder.cells(self.lad, 'RAW', limit=500)[0]
        (codex/'results'/f"{first['ordinal']}.json").write_bytes(panel.encoded(dict(cell=first, status='COMPLETE', score=1., declaration_sha256='0'*64)))
        with self.assertRaisesRegex(ValueError, 'foreign'): self.readout(codex_raw_directory=codex)


class DecisionSummaryTests(unittest.TestCase):
    def test_shares_over_non_single_legal_decisions_by_cause_and_kind(self):
        cell = dict(arm='DN2W8', ordinal=1)
        specs = [('search', None, 'move', True), ('search', None, 'move', False), ('search', None, 'switch', True),
                 ('single_legal', None, 'forced', False), ('greedy_fallback', 'timeout', 'move', None), ('greedy_fallback', 'timeout', 'move', None),
                 ('greedy_fallback', 'constructor_refused', 'switch', None), ('greedy_fallback', 'expired_in_queue', 'move', None),
                 ('search', None, 'move', False), ('greedy_fallback', 'bad_q', 'switch', None), ('single_legal', None, 'forced', False)]
        rows = [decision_row(cell, i, source=s, cause=c, kind=k, differs=bool(x), select=(i+1)/10, worlds=8 if i != 8 else 16)
                for i, (s, c, k, x) in enumerate(specs)]
        value = panel.decisions(rows, deadline=.75, worlds=8)
        self.assertEqual((value['decisions'], value['eligible_decisions']), (11, 9))
        self.assertEqual(value['by_source'], dict(search=4, greedy_fallback=5, single_legal=2))
        self.assertEqual((value['searched_share'], value['fallback_share'], value['transport_failure_share']), (4/9, 5/9, 2/9))
        self.assertEqual(value['fallback_by_cause'], dict(bad_q=dict(count=1, share=1/9), constructor_refused=dict(count=1, share=1/9),
            expired_in_queue=dict(count=1, share=1/9), timeout=dict(count=2, share=2/9)))
        move = value['by_request_kind']['move']
        self.assertEqual((move['decisions'], move['searched'], move['fallback'], move['search_differs']), (6, 3, 3, 1))
        self.assertEqual(move['fallback_by_cause'], dict(expired_in_queue=1, timeout=2)); self.assertAlmostEqual(move['search_differs_share'], 1/3)
        self.assertEqual(value['by_request_kind']['forced'], dict(decisions=2, searched=0, fallback=0, searched_share=0., fallback_share=0.,
            fallback_by_cause={}, search_differs=0, search_differs_share=None))
        self.assertEqual((value['search_differs'], value['search_differs_share_of_searched'], value['search_differs_share_of_all']), (2, .5, 2/11))
        self.assertEqual(value['latency']['select'], dict(n=11, p50=.6, p95=1.1, p99=1.1, max=1.1))
        self.assertEqual(value['latency']['prepare']['n'], 4); self.assertEqual(value['latency']['client']['n'], 9)
        self.assertEqual(value['late_decisions'], 4); self.assertEqual(value['wrong_worlds'], 1)
        self.assertEqual(value['provenance']['root_operator'], ['ARGMAX_MEAN_COMPLETED_Q'])
        with self.assertRaisesRegex(ValueError, 'unique'): panel.decisions(rows+rows[:1], deadline=3.)
        raw = dict(arm='RAW', ordinal=0)
        self.assertEqual(panel.decisions([decision_row(raw, 0, source='raw_arm', kind='forced')], deadline=3.)['eligible_decisions'], 1)
        for bad in (dict(rows[0], cause='timeout'), dict(rows[4], cause=None), dict(rows[4], cause='other'), dict(rows[0], request_kind='team'),
                    dict(rows[0], source='raw_arm'), dict(rows[0], schema='other'), dict(rows[3], request_kind='move'),
                    dict(rows[0], request_kind='forced'), dict(rows[4], request_kind='forced'), dict(rows[3], search=rows[0]['search'])):
            with self.assertRaises(ValueError): panel.decisions([bad], deadline=3.)


def completed(root, cell):
    return dict(path=str(root/'requests'/f"{cell['ordinal']}.json"), arm=cell['arm'])


def think(n, rate, *, stratum='2x1000ms'):
    """n Foul Play moves at one rate (rounds 0..n-1: with no capture, bands by round + 1)."""
    return [dict(round=i, stratum=stratum, iterations_per_budget_second=float(rate), iterations=int(rate), wait_seconds=.4, overlap_seconds=.6)
            for i in range(n)]


class EvalReadinessTests(unittest.TestCase):
    """The lead's rulings: precision pin, declared deadline, screen latency rule, arm fairness."""
    def setUp(self):
        self.directory = TemporaryDirectory(); self.root, self.run, self.d, self.lad = make_run(self.directory.name)
        self.raws, self.searches = panel.cells(self.lad, 'RAW'), panel.cells(self.lad, 'DN2W8')

    def tearDown(self): self.directory.cleanup()

    def play(self, monitor, ks, *, timeouts=(), raw_think=None, search_think=None, observable=True):
        """Complete RAW then search cell by cell for pairs ks (in that order): 30 searched decisions and one single-legal
        decision per search cell; `timeouts` are (pair, row) positions."""
        reasons = []
        for k in ks:
            raw, search = self.raws[k], self.searches[k]
            for cell in (raw, search): panel.write_new(self.root/'claims'/f"{cell['ordinal']}.json", {})
            write_cell(self.root, raw, self.run['declaration_sha256'], score=1., rows=[decision_row(raw, 0, source='raw_arm')], think=raw_think,
                       observable=observable)
            reasons.append(monitor(completed(self.root, raw)))
            rows = [decision_row(search, i, source='greedy_fallback', cause='timeout') if (k, i) in timeouts else decision_row(search, i)
                    for i in range(30)]+[decision_row(search, 30, source='single_legal')]
            write_cell(self.root, search, self.run['declaration_sha256'], score=0., rows=rows, think=search_think, observable=observable)
            reasons.append(monitor(completed(self.root, search)))
        return reasons

    def rule(self): return json.loads(panel.state_path(self.root, 'latency_rule.json').read_bytes())

    def test_new_declaration_fields_are_validated(self):
        self.assertEqual(panel.validate(declaration('a'*64))['expected_searched_share_roster'], .955)
        for deadline in panel.DEADLINES: self.assertEqual(panel.validate(declaration('a'*64, deadline=deadline))['deadline_seconds'], deadline)
        with self.assertRaises(ValueError): panel.validate(declaration('a'*64, deadline=4.))   # membership alone: its breakage bound agrees
        self.assertEqual(panel.validate(declaration('a'*64, observable=False))['calibration']['ratios'], None)
        band = dict(CALIBRATION_BAND)
        changes = [('deadline_seconds', 5), ('breakage', dict(panel.BREAKAGE, latency_p95_max_seconds=5.)),
                   ('matmul_precision', 'default'), ('screen_latency_rule', dict(panel.LATENCY_RULE, timeout_rate_max=.02)),
                   ('screen_latency_rule', dict(panel.LATENCY_RULE, searched_decisions=500)), ('bench_p95_wall_seconds', 0.),
                   ('bench_p95_wall_seconds', 1), ('bench_p95_wall_seconds', float('nan')), ('bench_source', dict(path='', sha256='e'*64)),
                   ('bench_source', dict(path='x', sha256='E'*64)), ('bench_source', dict(path='x')), ('server_processes_per_gpu', 0),
                   ('server_processes_per_gpu', True), ('server_processes_per_gpu', 1.), ('expected_searched_share_roster', 1.2),
                   ('expected_searched_share_roster', 1), ('interleave', 'SEARCH_THEN_RAW'), ('arm_fairness', dict(panel.ARM_FAIRNESS, ratio_tolerance=.1)),
                   ('arm_fairness', dict(panel.ARM_FAIRNESS, min_cell_moves=10)), ('worker_lanes', 0), ('worker_lanes', True),
                   ('fairness_observable', dict(value='yes', evidence=dict(path='x', sha256='e'*64))),
                   ('fairness_observable', dict(value=True, evidence=dict(path='x'))),
                   ('calibration', dict(status='UNMEASURABLE', sha256='c'*64, ratios=None, seed_band=band)),
                   ('calibration', dict(status='MEASURED', sha256='c'*64, ratios={'4x500ms': .95}, seed_band=band)),
                   ('calibration', dict(status='MEASURED', sha256='c'*64, ratios={'4x500ms': .95, '2x1000ms': 1}, seed_band=band)),
                   ('calibration', dict(status='MEASURED', sha256='c'*64, ratios=dict(RATIOS), seed_band=dict(band, count=0))),
                   ('calibration', dict(status='MEASURED', sha256='C'*64, ratios=dict(RATIOS), seed_band=band)),
                   ('calibration', dict(status='MEASURED', sha256='c'*64, ratios=dict(RATIOS), seed_band=band,
                                        floors={'4x500ms': 59, '2x1000ms': 300})),
                   ('calibration', dict(status='MEASURED', sha256='c'*64, ratios=dict(RATIOS), seed_band=band, floors={'2x1000ms': 300}))]
        for key, value in changes:
            d = declaration('a'*64); d[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): panel.validate(d)
        d = declaration('a'*64, observable=False); d['calibration']['ratios'] = dict(RATIOS)
        with self.assertRaises(ValueError): panel.validate(d)
        d = declaration('a'*64); d['expected_searched_share_roster'] = None; self.assertIsNone(panel.validate(d)['expected_searched_share_roster'])
        d = declaration('a'*64, deadline=10.); self.assertEqual(panel.validate(d)['breakage']['latency_p95_max_seconds'], 10.)

    def test_a_response_priced_under_another_precision_is_breakage(self):
        populate(self.root, self.run, self.lad, rows_for=lambda cell: [decision_row(cell, 0, source='raw_arm')] if cell['arm'] == 'RAW'
                 else [decision_row(cell, 0, precision='default')])
        value = panel.readout(self.root, arm='DN2W8', limit=500)
        self.assertEqual((value['breakage'], value['decision']), (['matmul_precision'], 'STOP_BREAKAGE'))
        self.assertEqual(value['arms']['DN2W8']['decisions']['wrong_precision'], 500)

    def test_ten_timeouts_in_the_first_thousand_continue_and_eleven_void(self):
        for count, status in ((9, 'CONTINUE'), (10, 'CONTINUE'), (11, 'VOID_LATENCY')):
            with self.subTest(count=count):
                self.tearDown(); self.setUp(); monitor = panel.Monitor(self.root, search_arm='DN2W8')
                reasons = self.play(monitor, range(34), timeouts={(0, i) for i in range(count)}|{(33, i) for i in range(25, 30)})
                self.assertEqual(reasons[-1], None if status == 'CONTINUE' else 'VOID_LATENCY'); self.assertEqual(set(reasons[:-1]), {None})
                self.assertEqual((self.rule()['status'], self.rule()['timeouts'], self.rule()['searched_decisions']), (status, count, 1000))

    def test_the_window_follows_completion_order_and_is_evaluated_once(self):
        monitor = panel.Monitor(self.root, search_arm='DN2W8')
        order = list(range(33, -1, -1))   # completion order reversed from ordinal order
        self.assertEqual(set(self.play(monitor, order, timeouts={(0, i) for i in range(20, 30)}|{(1, i) for i in range(5)})), {None})
        rule = self.rule(); recorded = panel.state_path(self.root, 'latency_rule.json').read_bytes()
        self.assertEqual((rule['status'], rule['timeouts']), ('CONTINUE', 5)); self.assertEqual(rule['cells'], [self.searches[k]['ordinal'] for k in order])
        self.assertEqual(set(self.play(monitor, range(34, 37), timeouts={(k, i) for k in range(34, 37) for i in range(30)})), {None})
        self.assertEqual(panel.state_path(self.root, 'latency_rule.json').read_bytes(), recorded)
        self.assertEqual(panel.readout(self.root, arm='DN2W8', limit=500)['latency_rule'], rule)

    def test_a_readout_between_a_crash_and_a_resume_uses_the_monitors_window(self):
        monitor = panel.Monitor(self.root, search_arm='DN2W8'); self.play(monitor, [5, 4])
        for k in range(33, 5, -1):   # completed, but never logged (dispatch stopped): placed RAW then search, by ordinal
            for cell in (self.raws[k], self.searches[k]):
                timeouts = {(k, i) for i in range(30)} if k == 31 else set()
                rows = [decision_row(cell, i, source='greedy_fallback', cause='timeout') if (k, i) in timeouts else decision_row(cell, i)
                        for i in range(30)] if cell['arm'] == 'DN2W8' else [decision_row(cell, 0, source='raw_arm')]
                write_cell(self.root, cell, self.run['declaration_sha256'], score=1., rows=rows)
        for k in (2, 3, 1, 0): write_cell(self.root, self.searches[k], self.run['declaration_sha256'], score=1., rows=[decision_row(self.searches[k], i) for i in range(30)])
        seen = panel.readout(self.root, arm='DN2W8', limit=500)['latency_rule']
        recorded = panel.Monitor(self.root, search_arm='DN2W8').latency()
        self.assertEqual(seen, recorded); self.assertEqual(recorded['status'], 'VOID_LATENCY')
        self.assertEqual(recorded['cells'][:4], [self.searches[5]['ordinal'], self.searches[4]['ordinal'], self.searches[0]['ordinal'], self.searches[1]['ordinal']])

    def test_missing_cells_join_the_log_raw_then_search_by_ordinal(self):
        for cell in (self.searches[1], self.raws[2], self.searches[0], self.raws[0]):
            write_cell(self.root, cell, self.run['declaration_sha256'], score=1., rows=[decision_row(cell, 0, source='raw_arm' if cell['arm'] == 'RAW' else 'search')])
        expected = [self.raws[0]['ordinal'], self.raws[2]['ordinal'], self.searches[0]['ordinal'], self.searches[1]['ordinal']]
        self.assertEqual([e['ordinal'] for e in panel.effective_log(self.root, self.lad, 'DN2W8')], expected)
        panel.Monitor(self.root, search_arm='DN2W8')
        self.assertEqual([e['ordinal'] for e in panel.completions(self.root)], expected)

    def test_eleven_timeouts_void_the_screen_and_dispatch_never_resumes(self):
        monitor = panel.Monitor(self.root, search_arm='DN2W8')
        reasons = self.play(monitor, range(34), timeouts={(0, i) for i in range(11)})
        self.assertEqual(reasons[-1], 'VOID_LATENCY')
        with self.assertRaisesRegex(ValueError, 'never resumed'): panel.pending(self.root, arm='RAW', limit=500, resume=True)
        with self.assertRaisesRegex(ValueError, 'never resumed'): panel.interleaved(self.root, search_arm='DN2W8', limit=500, resume=True)
        value = panel.readout(self.root, arm='DN2W8', limit=500)
        self.assertEqual((value['decision'], value['paired'], value['bar']), ('VOID_LATENCY', None, None))
        coverage = value['deadline_coverage']
        self.assertEqual((coverage['decisions'], coverage['censored_timeouts'], coverage['smallest_covering_deadline']), (1020, 11, None))
        self.assertEqual(coverage['coverage']['5.0'], dict(lower=1009/1020, upper=1.)); self.assertEqual(coverage['wall']['max'], .5)
        late = Path(self.directory.name)/'server.jsonl'
        def reply(i, status): return dict(kind='request', response=dict(request_id=f"r{self.searches[0]['ordinal']}-{i}", status=status), server_seconds=6.)
        late.write_text(''.join(json.dumps(reply(i, 'COMPLETE'))+'\n' for i in range(11)))
        coverage = panel.readout(self.root, arm='DN2W8', limit=500, server_telemetry=[late])['deadline_coverage']
        self.assertEqual((coverage['censored_timeouts'], coverage['smallest_covering_deadline']), (0, 10.))
        self.assertEqual((coverage['coverage']['5.0']['lower'], coverage['coverage']['10.0']['lower']), (1009/1020, 1.))
        late.write_text(''.join(json.dumps(reply(i, 'COMPLETE' if i < 5 else 'REFUSED'))+'\n' for i in range(11)))   # e.g. expired_in_queue
        coverage = panel.readout(self.root, arm='DN2W8', limit=500, server_telemetry=[late])['deadline_coverage']
        self.assertEqual((coverage['censored_timeouts'], coverage['coverage']['10.0']['lower']), (6, 1014/1020))

    def test_fewer_than_a_thousand_searched_decisions_never_void(self):
        monitor = panel.Monitor(self.root, search_arm='DN2W8')
        self.assertEqual(set(self.play(monitor, range(20), timeouts={(k, i) for k in range(10) for i in range(30)})), {None})
        self.assertFalse(panel.state_path(self.root, 'latency_rule.json').exists())
        value = panel.readout(self.root, arm='DN2W8', limit=500)
        self.assertIsNone(value['latency_rule']); self.assertEqual(value['decision'], 'INCOMPLETE'); self.assertIsNotNone(value['paired'])

    def test_arms_interleave_pair_by_pair_and_resume_keeps_the_count(self):
        packets, attempted = panel.interleaved(self.root, search_arm='DN2W8', limit=500)
        self.assertEqual(attempted, dict(RAW=0, DN2W8=0)); self.assertEqual(len(packets), 1000)
        self.assertEqual([(p['pair_ordinal'], p['arm']) for p in packets[:4]], [(0, 'RAW'), (0, 'DN2W8'), (1, 'RAW'), (1, 'DN2W8')])
        for cell in self.raws[:3]: panel.write_new(self.root/'claims'/f"{cell['ordinal']}.json", {})
        packets, attempted = panel.interleaved(self.root, search_arm='DN2W8', limit=500, resume=True)
        self.assertEqual(attempted, dict(RAW=3, DN2W8=0)); self.assertEqual([(p['pair_ordinal'], p['arm']) for p in packets[:4]],
                                                                          [(0, 'DN2W8'), (1, 'DN2W8'), (2, 'DN2W8'), (3, 'RAW')])
        with self.assertRaisesRegex(ValueError, 'search arm'): panel.interleaved(self.root, search_arm='RAW', limit=500)


def pooled_moves(rows):
    return dict(rated=list(rows), moves=len(rows), band_from_round=0, waits=[])


class FairnessReadingTests(unittest.TestCase):
    def test_strata_correct_a_schedule_mix_the_unstratified_ratio_does_not(self):
        raw = pooled_moves([('4x500ms', 0, 100.)]*500+[('2x1000ms', 2, 200.)]*500)
        search = pooled_moves([('4x500ms', 0, 100.)]*800+[('2x1000ms', 2, 200.)]*200)   # shorter games: more early moves
        value = fairness.reading(raw, search)
        self.assertAlmostEqual(value['weighted_ratio'], 1., places=12); self.assertAlmostEqual(value['unstratified_ratio'], 120/150, places=12)
        self.assertIsNone(fairness.flag(value)); self.assertEqual(value['moves'], dict(RAW=1000, search=1000))

    def test_weights_are_the_pooled_counts(self):
        raw = pooled_moves([('4x500ms', 0, 100.)]*100+[('2x1000ms', 2, 100.)]*300)
        search = pooled_moves([('4x500ms', 0, 110.)]*300+[('2x1000ms', 2, 100.)]*100)
        self.assertAlmostEqual(fairness.reading(raw, search)['weighted_ratio'], (400*1.1+400*1.)/800, places=12)

    def test_sparse_cells_merge_into_the_adjacent_band_of_their_stratum(self):
        raw = pooled_moves([('2x1000ms', 0, 100.)]*10+[('2x1000ms', 1, 100.)]*40+[('2x1000ms', 2, 100.)]*40+[('2x1000ms', 3, 100.)]*5
                           +[('4x500ms', 0, 50.)]*12)
        search = pooled_moves([('2x1000ms', 0, 100.)]*40+[('2x1000ms', 1, 100.)]*40+[('2x1000ms', 2, 100.)]*40+[('2x1000ms', 3, 100.)]*40
                              +[('4x500ms', 0, 50.)]*12)
        value = fairness.reading(raw, search)
        self.assertEqual(value['merge_map'], {'2x1000ms': [['1-3', '4-9'], ['10-19', '20+']], '4x500ms': [['1-3']]})
        self.assertTrue(all(c['raw_moves'] >= 30 and c['search_moves'] >= 30 for c in value['cells'] if c['stratum'] == '2x1000ms'))
        self.assertEqual([c['raw_moves'] for c in value['cells'] if c['stratum'] == '4x500ms'], [12])   # a single cell stays sparse

    def test_coverage_counts_unrated_moves_and_flags_a_gap(self):
        cell = lambda rated, unrated: dict(source_game=dict(opponent_think=think(rated, 100)+[dict(round=0, wait_seconds=.5)]*unrated, opponent_think_failures=0))
        raw, search = fairness.moves(cell(950, 50)), fairness.moves(cell(890, 110))
        value = fairness.reading(raw, search)
        self.assertEqual(value['coverage'], dict(RAW=.95, search=.89)); self.assertAlmostEqual(value['coverage_difference'], .06)
        self.assertEqual(fairness.flag(value), 'coverage')
        for rated, reason in ((50, None), (49, 'coverage')):   # coverages .10 and .05 (.049): a gap of exactly five points does not flag
            gap = fairness.reading(fairness.moves(cell(100, 900)), fairness.moves(cell(rated, 1000-rated)))
            with self.subTest(rated=rated): self.assertEqual(fairness.flag(gap), reason)
        self.assertEqual(fairness.reading(fairness.moves(cell(100, 900)), fairness.moves(cell(50, 950)))['coverage_difference'], .05)
        failures = fairness.moves(dict(source_game=dict(opponent_think=think(9, 100), opponent_think_failures=1)))
        self.assertEqual((failures['moves'], len(failures['rated'])), (10, 9))

    def test_rows_without_a_rate_or_stratum_never_rate_and_turns_come_from_the_capture(self):
        rows = [dict(round=0, stratum='4x500ms', wait_seconds=1., overlap_seconds=1.), dict(round=1, iterations_per_budget_second=5.),
                dict(round=2, stratum='4x500ms', iterations_per_budget_second=True), dict(round=0, stratum='4x500ms', iterations_per_budget_second=7.)]
        value = fairness.moves(dict(source_game=dict(opponent_think=rows)), [12, 25])
        self.assertEqual((value['rated'], value['band_from_round'], value['waits']), ([('4x500ms', 2, 7.)], 0, [2.]))
        with TemporaryDirectory() as directory:
            capture_path = make_capture(Path(directory), turns=3)
            self.assertEqual(fairness.round_turns(capture_path), [1, 2, 3]); self.assertIsNone(fairness.round_turns(Path(directory)/'none.jsonl'))

    def test_a_cells_bands_come_from_its_capture_when_it_exists(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); (root/'owning-wire').mkdir()
            (root/'owning-wire'/'7.jsonl').write_text(json.dumps(dict(event='WIRE', lines=['|turn|12', '|request|{}', '|request|{}']))+'\n')
            result = dict(source_game=dict(opponent_think=think(2, 100)))
            captured = panel.cell_moves(root, 7, result); fallback = panel.cell_moves(root, 8, result)
        self.assertEqual(([b for _, b, _ in captured['rated']], captured['band_from_round']), ([2, 2], 0))   # turn 12: band 10-19
        self.assertEqual(([b for _, b, _ in fallback['rated']], fallback['band_from_round']), ([0, 0], 2))   # rounds 0, 1: turns 1, 2

    def test_the_baseline_flag_and_the_benign_re_arm_rule(self):
        def at(ratio, gap=0.): return dict(weighted_ratio=ratio, difference=ratio-1, coverage_difference=gap)
        self.assertIsNone(fairness.flag(at(1.049))); self.assertEqual(fairness.flag(at(1.051)), 'ratio'); self.assertEqual(fairness.flag(at(.94)), 'ratio')
        acknowledged = dict(difference=.06, coverage_difference=.01)
        for ratio, reason in ((1.084, None), (1.086, 'drift'), (.951, None), (.949, 'sign'), (.96, None), (1.0, None)):
            with self.subTest(ratio=ratio): self.assertEqual(fairness.flag(at(ratio), acknowledged=acknowledged), reason)
        small = dict(difference=.01, coverage_difference=0.)   # a small acknowledged difference keeps the baseline bar
        self.assertIsNone(fairness.flag(at(1.036), acknowledged=small)); self.assertEqual(fairness.flag(at(1.051), acknowledged=small), 'drift')
        self.assertIsNone(fairness.flag(at(1.07, .06), acknowledged=dict(difference=.06, coverage_difference=.04)))
        self.assertEqual(fairness.flag(at(1.07, .066), acknowledged=dict(difference=.06, coverage_difference=.04)), 'coverage')
        self.assertIsNone(fairness.flag(dict(weighted_ratio=None, difference=None, coverage_difference=None)))


class FairnessMonitorTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(); self.root, self.run, self.d, self.lad = make_run(self.directory.name)
        self.raws, self.searches = panel.cells(self.lad, 'RAW'), panel.cells(self.lad, 'DN2W8')

    def tearDown(self): self.directory.cleanup()

    def play(self, monitor, ks, raw_rate, search_rate, moves=20, observable=True):
        reasons = []
        for k in ks:
            for cell, rate in ((self.raws[k], raw_rate), (self.searches[k], search_rate)):
                panel.write_new(self.root/'claims'/f"{cell['ordinal']}.json", {})
                write_cell(self.root, cell, self.run['declaration_sha256'], score=1., rows=[decision_row(cell, 0, source='raw_arm' if cell['arm'] == 'RAW' else 'search')],
                           think=think(moves, rate), observable=observable)
                reasons.append(monitor(completed(self.root, cell)))
        return reasons

    def state(self, phase='screen'): return panel.fairness_state(self.root, phase)

    def test_fairness_is_evaluated_only_at_checkpoints_once_the_floors_hold(self):
        monitor = panel.Monitor(self.root, search_arm='DN2W8')
        self.assertEqual(set(self.play(monitor, range(49), 100, 90, moves=25)), {None})   # 1,225 moves per arm, but no checkpoint yet
        self.assertEqual(self.state()['checks'], [])
        self.assertEqual(self.play(monitor, [49], 100, 90, moves=25), [None, 'FLAG_ARM_FAIRNESS'])
        check, = self.state()['checks']; flag, = self.state()['flags']
        self.assertEqual((check['pairs'], check['floors'], check['reason'], flag['reason']), (50, True, 'ratio', 'ratio'))
        self.assertAlmostEqual(flag['reading']['difference'], -.1)
        with self.assertRaisesRegex(ValueError, 'FLAG_ARM_FAIRNESS'): panel.pending(self.root, arm='RAW', limit=500, resume=True)

    def test_below_the_move_floor_a_checkpoint_records_and_waits(self):
        monitor = panel.Monitor(self.root, search_arm='DN2W8')
        self.assertEqual(set(self.play(monitor, range(50), 100, 90, moves=10)), {None})
        check, = self.state()['checks']; self.assertEqual((check['pairs'], check['floors'], check['reason']), (50, False, None))
        self.assertEqual(self.play(monitor, range(50, 100), 100, 90, moves=10)[-1], 'FLAG_ARM_FAIRNESS')
        self.assertEqual([c['pairs'] for c in self.state()['checks']], [50, 100])

    def flagged_then_acknowledged(self, monitor, cause='benign'):
        self.assertEqual(self.play(monitor, range(50), 100, 106)[-1], 'FLAG_ARM_FAIRNESS')
        return panel.acknowledge_arm_fairness(self.root, cause=cause, note='residual position mix after a host restart')

    def test_a_benign_acknowledgement_re_arms_over_later_moves(self):
        for later, reason in ((107, None), (109, 'drift'), (94, 'sign'), (96, None)):
            with self.subTest(later=later):
                self.tearDown(); self.setUp(); monitor = panel.Monitor(self.root, search_arm='DN2W8')
                ack = self.flagged_then_acknowledged(monitor)
                self.assertEqual((ack['cause'], ack['flag'], ack['log_length']), ('benign', 1, 100)); self.assertAlmostEqual(ack['difference'], .06)
                self.assertEqual(len(panel.pending(self.root, arm='RAW', limit=500, resume=True)), 450)
                reasons = self.play(monitor, range(50, 100), 100, later)
                self.assertEqual(reasons[-1], None if reason is None else 'FLAG_ARM_FAIRNESS'); self.assertEqual(set(reasons[:-1]), {None})
                checks = self.state()['checks']; self.assertEqual([(c['window'], c['pairs']) for c in checks], [(0, 50), (100, 50)])
                if reason: self.assertEqual(self.state()['flags'][1]['reason'], reason)

    def test_a_fault_voids_the_run_for_good(self):
        monitor = panel.Monitor(self.root, search_arm='DN2W8'); self.flagged_then_acknowledged(monitor, cause='fault')
        self.assertEqual(self.state()['void']['status'], 'VOID_FAIRNESS')
        with self.assertRaisesRegex(ValueError, 'VOID_FAIRNESS'): panel.pending(self.root, arm='RAW', limit=500, resume=True)
        self.assertEqual(panel.readout(self.root, arm='DN2W8', limit=500)['decision'], 'VOID_FAIRNESS')
        for cause, note in (('neither', 'x'), ('benign', ' ')):
            with self.assertRaisesRegex(ValueError, 'cause'): panel.acknowledge_arm_fairness(self.root, cause=cause, note=note)
        with self.assertRaisesRegex(ValueError, 'no arm fairness flag'): panel.acknowledge_arm_fairness(self.root, cause='benign', note='x')

    def test_the_extension_starts_unacknowledged(self):
        monitor = panel.Monitor(self.root, search_arm='DN2W8'); self.flagged_then_acknowledged(monitor)
        self.assertEqual(self.play(monitor, range(500, 550), 100, 106)[-1], 'FLAG_ARM_FAIRNESS')   # 6% re-flags: no screen acknowledgement applies
        flag, = self.state('extension')['flags']; self.assertEqual((flag['reason'], flag['acknowledged'], flag['window']), ('ratio', None, 0))

    def test_unobservable_fairness_never_flags_and_says_why(self):
        self.tearDown(); self.directory = TemporaryDirectory()
        self.root, self.run, self.d, self.lad = make_run(self.directory.name, observable=False)
        monitor = panel.Monitor(self.root, search_arm='DN2W8')
        self.assertEqual(set(self.play(monitor, range(60), 100, 70, observable=False)), {None})
        self.assertEqual(self.state()['checks'], [])
        value = panel.readout(self.root, arm='DN2W8', limit=500)
        reading = value['arm_fairness']['phases']['screen']['reading']
        self.assertEqual(reading['status'], 'UNMEASURABLE'); self.assertIn('isolation', reading['basis'])
        self.assertEqual(reading['descriptive_request_to_answer_seconds']['RAW'], dict(moves=1200, mean_seconds=1.))
        self.assertNotIn('fairness_observability', value['breakage'])
        cell = self.raws[60]; write_cell(self.root, cell, self.run['declaration_sha256'], score=1., rows=[decision_row(cell, 0, source='raw_arm')])
        self.assertIn('fairness_observability', panel.readout(self.root, arm='DN2W8', limit=500)['breakage'])   # a cell contradicts it

    def test_an_observable_declaration_contradicted_mid_run_is_breakage(self):
        populate(self.root, self.run, self.lad)
        self.assertNotIn('fairness_observability', panel.readout(self.root, arm='DN2W8', limit=500)['breakage'])
        cell = self.searches[3]
        write_cell(self.root, cell, self.run['declaration_sha256'], score=1., rows=[decision_row(cell, i) for i in range(2)], observable=None)
        value = panel.readout(self.root, arm='DN2W8', limit=500)   # the bridge's probe failed: not a reading
        self.assertNotIn('fairness_observability', value['breakage']); self.assertEqual(value['arm_fairness']['observability'], dict(contradictions=0, unknown=1))
        write_cell(self.root, cell, self.run['declaration_sha256'], score=1., rows=[decision_row(cell, i) for i in range(2)], observable=False)
        value = panel.readout(self.root, arm='DN2W8', limit=500)
        self.assertIn('fairness_observability', value['breakage']); self.assertEqual(value['arm_fairness']['observability'], dict(contradictions=1, unknown=0))

    def test_an_extension_flag_never_changes_the_pinned_screen_readout(self):
        populate(self.root, self.run, self.lad)
        for arm in ('RAW', 'DN2W8'):
            for cell in panel.cells(self.lad, arm)[:1990]:
                (self.root/'claims').mkdir(exist_ok=True); (self.root/'claims'/f"{cell['ordinal']}.json").write_bytes(b'{}')
        genuine = panel.readout(self.root, arm='DN2W8', limit=500); self.assertEqual(genuine['decision'], 'CONTINUE_TO_2000')
        path = Path(self.directory.name)/'screen.json'; panel.write_new(path, genuine); screen = (path, panel.sha(genuine))
        self.assertEqual(len(panel.pending(self.root, arm='DN2W8', limit=2000, resume=True, screen=screen)), 10)
        directory = panel.fairness_directory(self.root, 'extension'); directory.mkdir(parents=True)
        panel.write_new(directory/'flag-1.json', dict(phase='extension', reason='ratio', reading=dict(difference=.07, coverage_difference=0.)))
        with self.assertRaisesRegex(ValueError, 'FLAG_ARM_FAIRNESS'): panel.pending(self.root, arm='DN2W8', limit=2000, resume=True, screen=screen)
        self.assertEqual(panel.acknowledge_arm_fairness(self.root, cause='benign', note='transient host event')['phase'], 'extension')
        self.assertEqual(len(panel.pending(self.root, arm='DN2W8', limit=2000, resume=True, screen=screen)), 10)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(); root = Path(self.directory.name).resolve(); inputs = root/'inputs'
        self.lad = ladder_fixture.declaration()
        for name, value in (('ladder', self.lad), ('base', {})): panel.write_new(inputs/f'{name}.json', value)
        self.kwargs = dict(ladder_declaration=inputs/'ladder.json', ladder_declaration_sha256=panel.sha(self.lad), checkpoint='/inert/champion.pt',
                           base_config=inputs/'base.json', base_config_sha256=panel.sha({}))
        self.root = root/'calibration'

    def tearDown(self): self.directory.cleanup()

    def test_calibration_seeds_are_disjoint_from_every_eval_cell(self):
        band = panel.calibration_cells(self.lad, CALIBRATION_BAND)
        self.assertEqual(len(band), 400); self.assertTrue(all(c['calibration'] and c['arm'] == 'RAW' for c in band))
        eval_seeds = {(c['seed'], c['fp_seed']) for c in panel.cells(self.lad, 'RAW')}
        self.assertFalse({c['seed'] for c in band} & {s for s, _ in eval_seeds}); self.assertFalse({c['fp_seed'] for c in band} & {f for _, f in eval_seeds})
        for overlap in (dict(CALIBRATION_BAND, battle_seed_start=10999), dict(CALIBRATION_BAND, foulplay_seed_start=19900, count=101)):
            with self.subTest(overlap), self.assertRaisesRegex(ValueError, 'overlap'): panel.calibration_cells(self.lad, overlap)
        self.assertEqual(len(panel.calibration_cells(self.lad, dict(CALIBRATION_BAND, foulplay_seed_start=19900, count=100))), 200)

    def fake_dispatch(self, rates, observable=True, moves=40):
        calls = []
        async def run(packets, *, pairs, servers, after, **kwargs):
            calls.append((packets[0]['setting'], len(pairs), len(packets))); done = 0
            for packet in packets:
                cell = ladder.read_bound(packet['path'], packet['sha256'])['cell']
                panel.write_new(self.root/'claims'/f"{cell['ordinal']}.json", dict(setting=packet['setting']))
                rows = think(moves, rates[packet['setting']]['4x500ms'], stratum='4x500ms')+think(moves, rates[packet['setting']]['2x1000ms'])
                panel.write_new(self.root/'results'/f"{cell['ordinal']}.json", dict(status='COMPLETE', cell=cell, setting=packet['setting'],
                    cpu_pair=list(pairs[done % len(pairs)]), source_game=dict(opponent_think=rows), worker=dict(hostname='h'),
                    benchmark=dict(foulplay_think=dict(iterations_observable=observable))))
                done += 1; reason = after(packet)
                if reason: return dict(dispatched=done, failed=[], not_dispatched=len(packets)-done, stopped=reason)
            return dict(dispatched=done, failed=[], not_dispatched=0, stopped=None)
        return run, calls

    def test_solo_then_concurrent_until_every_stratum_has_its_moves(self):
        panel.calibration_init(self.root, band=dict(CALIBRATION_BAND), lanes=4, **self.kwargs)
        rates = dict(solo={'4x500ms': 100., '2x1000ms': 200.}, concurrent={'4x500ms': 95., '2x1000ms': 180.})
        run, calls = self.fake_dispatch(rates)
        value = asyncio.run(panel.calibrate(self.root, pairs=[(0, 1), (2, 3), (4, 5), (6, 7)], dispatch_function=run))
        self.assertEqual([(s, lanes) for s, lanes, _ in calls], [('solo', 1), ('concurrent', 4)])
        self.assertEqual((value['status'], value['ratios'], value['games']), ('MEASURED', {'4x500ms': .95, '2x1000ms': .9}, dict(solo=8, concurrent=8)))
        self.assertEqual(value['strata']['4x500ms']['solo']['moves'], 320); self.assertEqual(value['hosts'], ['{"hostname":"h"}'])
        claimed = sorted(int(p.stem) for p in (self.root/'claims').glob('*.json')); self.assertEqual(claimed, list(range(16)))
        stored = ladder.read_bound(self.root/'calibration.json', panel.sha(value)); self.assertEqual(stored, value)
        with self.assertRaises(FileExistsError): asyncio.run(panel.calibrate(self.root, pairs=[(0, 1), (2, 3), (4, 5), (6, 7)], dispatch_function=run))
        with self.assertRaisesRegex(ValueError, 'planned worker lanes'): asyncio.run(panel.calibrate(self.root, pairs=[(0, 1)], dispatch_function=run))

    def test_unobservable_work_makes_calibration_unmeasurable(self):
        panel.calibration_init(self.root, band=dict(CALIBRATION_BAND), lanes=2, **self.kwargs)
        run, calls = self.fake_dispatch(dict(solo={'4x500ms': 1., '2x1000ms': 1.}, concurrent={'4x500ms': 1., '2x1000ms': 1.}), observable=False)
        value = asyncio.run(panel.calibrate(self.root, pairs=[(0, 1), (2, 3)], dispatch_function=run))
        self.assertEqual((value['status'], value['iterations_observable'], value['ratios'], value['games']), ('UNMEASURABLE', False, None, dict(solo=1, concurrent=0)))

    def test_too_few_moves_is_insufficient_never_measured(self):
        panel.calibration_init(self.root, band=dict(CALIBRATION_BAND), lanes=2, **self.kwargs)
        rates = dict(solo={'4x500ms': 100., '2x1000ms': 200.}, concurrent={'4x500ms': 95., '2x1000ms': 190.})
        run, _ = self.fake_dispatch(rates)
        for setting in ('solo', 'concurrent'):
            asyncio.run(run(panel.calibration_packets(self.root, setting=setting)[:7], pairs=[(0, 1)], servers={'RAW': None}, after=lambda p: None))
        value = panel.calibration_summary(self.root)   # 7 games: 280 rated moves per stratum and setting, under the 300 floor
        self.assertEqual((value['status'], value['ratios'], value['strata']['4x500ms']['solo']['moves']), ('INSUFFICIENT', None, 280))

    def test_calibration_is_unmeasurable_only_when_no_game_reports_true(self):
        panel.calibration_init(self.root, band=dict(CALIBRATION_BAND), lanes=1, **self.kwargs)
        rates = dict(solo={'4x500ms': 100., '2x1000ms': 200.}, concurrent={'4x500ms': 100., '2x1000ms': 200.})
        run, _ = self.fake_dispatch(rates, observable=None)
        asyncio.run(run(panel.calibration_packets(self.root, setting='solo')[:1], pairs=[(0, 1)], servers={'RAW': None}, after=lambda p: 'stop'))
        self.assertEqual(panel.calibration_summary(self.root)['iterations_observable'], None)
        self.assertEqual(panel.calibration_summary(self.root)['status'], 'UNMEASURABLE')
        run, _ = self.fake_dispatch(rates, observable=True)
        asyncio.run(run(panel.calibration_packets(self.root, setting='solo')[:1], pairs=[(0, 1)], servers={'RAW': None}, after=lambda p: 'stop'))
        self.assertEqual(panel.calibration_summary(self.root)['iterations_observable'], True)
        self.assertFalse(panel.calibration_summary(self.root)['observability_mixed'])
        run, _ = self.fake_dispatch(rates, observable=False)   # games contradicting each other: fail closed
        asyncio.run(run(panel.calibration_packets(self.root, setting='solo')[:1], pairs=[(0, 1)], servers={'RAW': None}, after=lambda p: 'stop'))
        value = panel.calibration_summary(self.root)
        self.assertEqual((value['status'], value['observability_mixed'], value['ratios']), ('INSUFFICIENT', True, None))

    def test_a_calibration_game_stays_in_its_own_directory(self):
        panel.calibration_init(self.root, band=dict(CALIBRATION_BAND), lanes=1, **self.kwargs)
        packet = panel.calibration_packets(self.root, setting='solo')[0]; config = NS(checkpoint=Path('/inert/champion.pt'))
        sha = ActorContract.champion().checkpoint_sha256
        class Runtime:
            def __init__(self, **kw): self.kw = kw
            async def run(self, config): raise RuntimeError('inert game')
        with patch.object(panel.ladder, 'configuration', return_value=config), patch.object(panel.ladder, 'owning_capture', inert_capture), \
                patch('pokezero.live_protocol_runtime._checkpoint_sha', return_value=sha):
            wire = {k: packet[k] for k in ('path', 'sha256', 'setting')}
            self.assertEqual(asyncio.run(panel.calibration_worker(dict(wire, server=None), expected_pair=(1, 2), original='o', runtime_class=Runtime)), 0)
            result = json.loads((self.root/'results'/'0.json').read_bytes())
            self.assertEqual((result['status'], result['setting'], result['cell']['calibration']), ('UNKNOWN', 'solo', True))
            with self.assertRaises(FileExistsError): asyncio.run(panel.calibration_worker(dict(wire, server=None), expected_pair=(1, 2), original='o', runtime_class=Runtime))
            run_root, run, d, lad = make_run(Path(self.directory.name)/'eval')
            eval_packet = panel.pending(run_root, arm='RAW', limit=500)[0]
            with self.assertRaises(ValueError):
                asyncio.run(panel.calibration_worker(dict(path=eval_packet['path'], sha256=eval_packet['sha256'], setting='solo', server=None),
                                                     expected_pair=(1, 2), original='o', runtime_class=Runtime))
        self.assertFalse((run_root/'claims').exists())

    def test_init_verifies_the_calibration_and_dispatch_gates_on_it(self):
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory)
            self.assertIsNone(panel.require_calibration(root, d, pairs=[(0, 1)]*4))
            with self.assertRaisesRegex(ValueError, 'more worker lanes'): panel.require_calibration(root, d, pairs=[(0, 1)]*5)
            for ratios, ok in (({'4x500ms': .9, '2x1000ms': .95}, True), ({'4x500ms': .89, '2x1000ms': .95}, False)):
                d2 = declaration('a'*64, ratios=ratios)
                if ok: self.assertIsNone(panel.require_calibration(root, d2, pairs=[(0, 1)]))
                else:
                    with self.assertRaisesRegex(ValueError, 'below 0.9'): panel.require_calibration(root, d2, pairs=[(0, 1)])
            inputs = Path(directory).resolve()/'inputs'; wrong = calibration_fixture(panel.sha(lad), ratios={'4x500ms': .95, '2x1000ms': .5})
            panel.write_new(inputs/'wrong.json', wrong); d3 = dict(d, calibration=dict(d['calibration'], sha256=panel.sha(wrong)))
            panel.write_new(inputs/'d3.json', d3)
            with self.assertRaisesRegex(ValueError, 'calibration.json must match'):
                panel.init(Path(directory)/'other', declaration=inputs/'d3.json', declaration_sha256=panel.sha(d3), ladder_declaration=inputs/'ladder.json',
                    ladder_declaration_sha256=panel.sha(lad), checkpoint='/x', base_config=inputs/'base.json', base_config_sha256=panel.sha({}),
                    calibration=inputs/'wrong.json')

    def test_init_refuses_a_calibration_that_is_not_valid_evidence(self):
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory); inputs = Path(directory).resolve()/'inputs'
            good = calibration_fixture(panel.sha(lad))
            def bad(name, change, message):
                name = 'evidence-'+name; value = copy.deepcopy(good); change(value); panel.write_new(inputs/f'{name}.json', value)
                declared = dict(d, calibration=dict(d['calibration'], sha256=panel.sha(value)))
                panel.write_new(inputs/f'{name}-declaration.json', declared)
                with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                    panel.init(Path(directory)/name, declaration=inputs/f'{name}-declaration.json', declaration_sha256=panel.sha(declared),
                        ladder_declaration=inputs/'ladder.json', ladder_declaration_sha256=panel.sha(lad), checkpoint='/x', base_config=inputs/'base.json',
                        base_config_sha256=panel.sha({}), calibration=inputs/f'{name}.json')
            bad('floor', lambda v: v.update(floors=dict(panel.CALIBRATION_FLOORS, **{'2x1000ms': 299})), 'not valid evidence')
            bad('floor-missing', lambda v: v.pop('floors'), 'not valid evidence')
            bad('solo-moves', lambda v: v['strata']['4x500ms']['solo'].update(moves=59), 'not valid evidence')
            bad('later-moves', lambda v: v['strata']['2x1000ms']['concurrent'].update(moves=299), 'not valid evidence')
            bad('concurrent-moves', lambda v: v['strata']['2x1000ms']['concurrent'].update(moves=10), 'not valid evidence')
            bad('ratio', lambda v: v['strata']['4x500ms']['concurrent'].update(rate=.96), 'not valid evidence')
            bad('stratum', lambda v: v['strata'].pop('2x1000ms'), 'not valid evidence')
            bad('base', lambda v: v.update(base_config_sha256='0'*64), 'base config')
            bad('checkpoint', lambda v: v.update(checkpoint_sha256='0'*64), 'base config')
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory, observable=False)
            with self.assertRaisesRegex(ValueError, 'accept-uncalibrated'): panel.require_calibration(root, d, pairs=[(0, 1)])
            recorded = panel.require_calibration(root, d, pairs=[(0, 1)], accept_uncalibrated=True)
            self.assertEqual(recorded['status'], 'UNMEASURABLE'); self.assertEqual(panel.require_calibration(root, d, pairs=[(0, 1)], accept_uncalibrated=True), recorded)
            self.assertTrue(panel.readout(root, arm='DN2W8', limit=500)['arm_fairness']['accepted_uncalibrated'])


class ScriptTests(unittest.TestCase):
    def script(self):
        spec = importlib.util.spec_from_file_location('depthn_foulplay_panel_script', ROOT/'scripts'/'depthn_foulplay_panel.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

    def test_cli_readout_writes_once_and_dispatch_binds_server_to_arm(self):
        script = self.script()
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory); populate(root, run, lad); output = Path(directory)/'screen.json'
            argv = ['readout', '--run-directory', str(root), '--arm', 'DN2W8', '--limit', '500', '--output', str(output)]
            printed = io.StringIO()
            with redirect_stdout(printed): self.assertEqual(script.main(argv), 0)
            self.assertEqual(json.loads(output.read_bytes())['decision'], 'CONTINUE_TO_2000')
            self.assertEqual(json.loads(printed.getvalue())['sha256'], ladder.sha(json.loads(output.read_bytes())))
            with self.assertRaises(FileExistsError): script.main(argv)
            base = ['dispatch', '--run-directory', str(root), '--limit', '500', '--cpu-pair', '0', '1', '--cache-directory', directory]
            for extra in (['--arm', 'RAW', '--server', 'h:1'], ['--search-arm', 'DN2W8'], ['--search-arm', 'RAW', '--server', 'h:1']):
                with self.subTest(extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): script.main(base+extra)
        self.assertEqual(script.server('10.0.0.1:7000'), dict(host='10.0.0.1', port=7000))

    def test_a_failed_probe_dispatches_nothing(self):
        script = self.script()
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory); wire = make_capture(Path(directory)/'wire')
            dead = socket.create_server(('127.0.0.1', 0)); port = dead.getsockname()[1]; dead.close()
            argv = ['dispatch', '--run-directory', str(root), '--limit', '500', '--cpu-pair', '0', '1', '--cache-directory', directory,
                    '--search-arm', 'DN2W8', '--server', f'127.0.0.1:{port}', '--probe-capture', str(wire)]
            with patch.object(panel, 'dispatch', side_effect=AssertionError('must not dispatch')), self.assertRaisesRegex(ValueError, 'probe failed'):
                script.main(argv)
            self.assertFalse((root/'claims').exists())
            with patch.object(panel, 'dispatch', side_effect=AssertionError('must not dispatch')), self.assertRaisesRegex(ValueError, 'no closed RAW'):
                script.main(argv[:-2])

    def test_cli_dispatch_never_skips_the_calibration_gate(self):
        script = self.script()
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory, ratios={'4x500ms': .89, '2x1000ms': .95})
            argv = ['dispatch', '--run-directory', str(root), '--limit', '500', '--cpu-pair', '0', '1', '--cache-directory', directory,
                    '--search-arm', 'DN2W8', '--server', '127.0.0.1:9', '--probe-capture', '/never/read']
            with patch.object(panel, 'probe', side_effect=AssertionError('the gate comes first')), self.assertRaisesRegex(ValueError, 'below 0.9'):
                script.main(argv)
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory)
            lanes = [x for k in range(5) for x in ('--cpu-pair', str(2*k), str(2*k+1))]
            argv = ['dispatch', '--run-directory', str(root), '--limit', '500', '--cache-directory', directory, '--search-arm', 'DN2W8',
                    '--server', '127.0.0.1:9', '--probe-capture', '/never/read']+lanes
            with patch.object(panel, 'probe', side_effect=AssertionError('the gate comes first')), self.assertRaisesRegex(ValueError, 'more worker lanes'):
                script.main(argv)


if __name__ == '__main__': unittest.main()
