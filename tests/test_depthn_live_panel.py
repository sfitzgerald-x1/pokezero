"""Inert DepthN panel controls: declarations, cells, claims, probe, dispatch and readout."""
import asyncio
from contextlib import contextmanager, redirect_stdout
import copy
import importlib.util
import io
import json
import math
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
from pokezero import depthn_live_panel as panel
from pokezero import phase_d_ladder_panel as ladder
from pokezero import phase_d_owning_capture as capture
from pokezero.live_protocol_actor import ActorContract
from tests import test_depthn_live_actor as actor_tests
from tests import test_phase_d_ladder_panel as ladder_fixture
from tests import test_phase_d_prerequisites as fixture

ROOT = Path(__file__).resolve().parents[1]
DISPATCH_TIMEOUT = 30


EXPECTED = dict(value=.9, source=dict(run_id='constructor-world-gate-fixture', report_sha256='f'*64))


def declaration(ladder_sha256, arms=('RAW', 'DN2W8'), expected=EXPECTED):
    return dict(schema=panel.SCHEMA, ladder_declaration_sha256=ladder_sha256, expected_searched_share=copy.deepcopy(expected),
        checkpoint_sha256=ActorContract.champion().checkpoint_sha256, archive_sha256=live.CHAMPION_ARCHIVE_SHA256, arms=list(arms),
        search={a: dict(depth=2, worlds=panel.WORLDS[a]) for a in arms[1:]}, root_operator='ARGMAX_MEAN_COMPLETED_Q',
        deadline_seconds=3., reserve_seconds=.15, screen_paired_cells=500, max_paired_cells=2000, confidence=.9,
        primary_comparison='SEARCH_MINUS_OUR_RAW_PAIRED_BY_CELL', bar='LOWER_BOUND_GT_0_AT_2000_FULL_ROSTER_NO_BREAKAGE',
        screen_rule=panel.SCREEN_RULE, breakage=dict(panel.BREAKAGE))


def make_run(directory, arms=('RAW', 'DN2W8'), expected=EXPECTED):
    root = Path(directory).resolve(); inputs = root/'inputs'
    lad = ladder_fixture.declaration(); d = declaration(panel.sha(lad), arms, expected); base = {}
    for name, value in (('ladder', lad), ('declaration', d), ('base', base)): panel.write_new(inputs/f'{name}.json', value)
    run = panel.init(root/'run', declaration=inputs/'declaration.json', declaration_sha256=panel.sha(d),
        ladder_declaration=inputs/'ladder.json', ladder_declaration_sha256=panel.sha(lad), checkpoint='/inert/champion.pt',
        base_config=inputs/'base.json', base_config_sha256=panel.sha(base))
    return root/'run', run, d, lad


def decision_row(cell, i, *, source='search', cause=None, kind=None, differs=False, select=.5,
                 operator='ARGMAX_MEAN_COMPLETED_Q', worlds=8, archive=live.CHAMPION_ARCHIVE_SHA256):
    kind = kind or ('forced' if source == 'single_legal' else 'move')
    search = None if source in ('raw_arm', 'single_legal') else dict(status='COMPLETE' if source == 'search' else 'REFUSED',
        cause=cause, client_wall_seconds=select-.05, server_wall_seconds=select-.1, tree_commit='t'*40, archive_sha256=archive,
        root_operator=operator if source == 'search' else None, requested_worlds=worlds if source == 'search' else None,
        pricer=dict(k=1, depth=2, W=8, K=4) if source == 'search' else None,
        timings=dict(queue_seconds=.01, prepare_seconds=.2, build_seconds=.05, price_seconds=.2, reduce_seconds=.01) if source == 'search' else None)
    return dict(schema=live.ROW_SCHEMA, arm=cell['arm'], cell=cell, battle_id=f"b{cell['ordinal']}", owner='p1', decision_index=i,
        prefix_sha256=f'{i:064x}', request_kind=kind, source=source, cause=cause, search_differs=differs if source == 'search' else None,
        encode_seconds=.01, select_seconds=select, greedy=dict(seconds=.05), search=search)


def write_cell(root, cell, d_sha, *, score, status='COMPLETE', error=None, rows=(), decisions=None):
    for name in ('results', 'telemetry'): (root/name).mkdir(exist_ok=True)
    value = dict(schema=panel.CELL_SCHEMA, cell=cell, declaration_sha256=d_sha, status=status, score=score if status == 'COMPLETE' else None,
                 error=error, source_game=dict(pokezero_decisions=len(rows) if decisions is None else decisions))
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
        for arms in (('RAW', 'DN2W8'), ('RAW', 'DN2W16'), ('RAW', 'DN2W8', 'DN2W16')):
            self.assertEqual(panel.validate(declaration('a'*64, arms))['arms'], list(arms))
        self.assertEqual(panel.BREAKAGE['searched_share_min'], .5)
        for value in (None, dict(EXPECTED, value=0.), dict(EXPECTED, value=1.)):
            self.assertEqual(panel.validate(declaration('a'*64, expected=value))['expected_searched_share'], value)

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
                ladder_declaration_sha256=panel.sha(lad), checkpoint='/x', base_config=inputs/'base.json', base_config_sha256=panel.sha({}))
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
        self.root, self.run, self.d, self.lad = make_run(self.directory.name, arms=('RAW', 'DN2W8', 'DN2W16'))
        populate(self.root, self.run, self.lad, arms=('RAW', 'DN2W8', 'DN2W16'))
        for arm in panel.ARMS:  # Played screen cells are claimed; pre-claim most of the extension too.
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
        other = panel.readout(self.root, arm='DN2W16', limit=500); self.assertEqual(other['decision'], 'CONTINUE_TO_2000')
        with self.assertRaisesRegex(ValueError, 'pinned 500-cell'):
            panel.pending(self.root, arm='DN2W8', limit=2000, resume=True, screen=self.pin('other-arm', other))

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
    return {**actor_tests.complete(request), 'archive_sha256': live.CHAMPION_ARCHIVE_SHA256, **change}


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


class DispatchTests(unittest.TestCase):
    def run_dispatch(self, names, **kwargs):
        directory = Path(self.directory.name); packets = [dict(path=str(directory/n), sha256='0'*64) for n in names]
        work = panel.dispatch(packets, command=lambda pair, cache: [sys.executable, '-c', FAKE_WORKER], **kwargs)
        result = asyncio.run(asyncio.wait_for(work, DISPATCH_TIMEOUT))
        done = {n: json.loads((directory/(n+'.done')).read_text()) for n in names if (directory/(n+'.done')).exists()}
        return result, done

    def setUp(self): self.directory = TemporaryDirectory()
    def tearDown(self): self.directory.cleanup()

    def test_one_cell_per_fresh_worker_process_by_default(self):
        result, done = self.run_dispatch(['a', 'b', 'c'], pairs=[(0, 1), (2, 3)], server=dict(host='h', port=1))
        self.assertEqual(result, dict(dispatched=3, failed=[], not_dispatched=0))
        self.assertEqual(len({v['pid'] for v in done.values()}), 3); self.assertEqual(done['a']['server'], dict(host='h', port=1))

    def test_persistent_worker_when_asked(self):
        result, done = self.run_dispatch(['a', 'b', 'c'], pairs=[(0, 1)], server=None, cells_per_worker=5)
        self.assertEqual(result['dispatched'], 3); self.assertEqual(len({v['pid'] for v in done.values()}), 1)

    def test_a_failed_worker_stops_all_dispatch(self):
        for kwargs in (dict(pairs=[(0, 1)]), dict(pairs=[(0, 1)], cells_per_worker=4), dict(pairs=[(0, 1), (2, 3)])):
            with self.subTest(kwargs):
                self.directory.cleanup(); self.directory = TemporaryDirectory()
                result, done = self.run_dispatch(['a', 'fail', 'c', 'd', 'e', 'f'], server=None, **kwargs)
                self.assertGreaterEqual(len(result['failed']), 1); self.assertTrue(any(f.endswith('fail') for f in result['failed']))
                self.assertLessEqual(result['dispatched'], 2); self.assertGreaterEqual(result['not_dispatched'], 3)
                self.assertNotIn('f', done)

    def test_a_lane_failure_stops_a_persistent_worker_in_another_lane(self):
        names = ['slow-a', 'fail']+[f'slow-{c}' for c in 'cdefgh']
        result, done = self.run_dispatch(names, pairs=[(0, 1), (2, 3)], server=None, cells_per_worker=10)
        self.assertTrue(any(f.endswith('fail') for f in result['failed'])); self.assertLessEqual(result['dispatched'], 2)
        self.assertGreaterEqual(result['not_dispatched'], 5); self.assertNotIn('slow-h', done)

    def test_core_pairs_must_be_disjoint(self):
        for kwargs in (dict(pairs=[(0, 1), (1, 2)]), dict(pairs=[(0, 0)]), dict(pairs=[]), dict(pairs=[(0, 1)], cells_per_worker=0)):
            with self.subTest(kwargs), self.assertRaises(ValueError): asyncio.run(panel.dispatch([], server=None, **kwargs))


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
            for extra in (['--arm', 'RAW', '--server', 'h:1'], ['--arm', 'DN2W8']):
                with self.assertRaises(SystemExit): script.main(base+extra)
        self.assertEqual(script.server('10.0.0.1:7000'), dict(host='10.0.0.1', port=7000))

    def test_a_failed_probe_dispatches_nothing(self):
        script = self.script()
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory); wire = make_capture(Path(directory)/'wire')
            dead = socket.create_server(('127.0.0.1', 0)); port = dead.getsockname()[1]; dead.close()
            argv = ['dispatch', '--run-directory', str(root), '--limit', '500', '--cpu-pair', '0', '1', '--cache-directory', directory,
                    '--arm', 'DN2W8', '--server', f'127.0.0.1:{port}', '--probe-capture', str(wire)]
            with patch.object(panel, 'dispatch', side_effect=AssertionError('must not dispatch')), self.assertRaisesRegex(ValueError, 'probe failed'):
                script.main(argv)
            self.assertFalse((root/'claims').exists())
            with patch.object(panel, 'dispatch', side_effect=AssertionError('must not dispatch')), self.assertRaisesRegex(ValueError, 'no closed RAW'):
                script.main(argv[:-2])


if __name__ == '__main__': unittest.main()
