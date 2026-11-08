"""Inert DepthN panel controls: declarations, cells, claims, dispatch and readout."""
import asyncio
from contextlib import contextmanager, redirect_stdout
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
from statistics import fmean, variance
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pokezero import depthn_live_actor as live
from pokezero import depthn_live_panel as panel
from pokezero import phase_d_ladder_panel as ladder
from pokezero.live_protocol_actor import ActorContract
from tests import test_phase_d_ladder_panel as ladder_fixture

ROOT = Path(__file__).resolve().parents[1]


def declaration(ladder_sha256, arms=('RAW', 'DN2W8')):
    return dict(schema=panel.SCHEMA, ladder_declaration_sha256=ladder_sha256,
        checkpoint_sha256=ActorContract.champion().checkpoint_sha256, arms=list(arms),
        search={a: dict(depth=2, worlds=panel.WORLDS[a]) for a in arms[1:]}, root_operator='ARGMAX_MEAN_COMPLETED_Q',
        deadline_seconds=3., reserve_seconds=.15, screen_paired_cells=500, max_paired_cells=2000, confidence=.9,
        primary_comparison='SEARCH_MINUS_OUR_RAW_PAIRED_BY_CELL', bar='LOWER_BOUND_GT_0_AT_2000', screen=dict(panel.SCREEN))


def make_run(directory, arms=('RAW', 'DN2W8')):
    root = Path(directory).resolve(); inputs = root/'inputs'
    lad = ladder_fixture.declaration(); d = declaration(panel.sha(lad), arms); base = {}
    for name, value in (('ladder', lad), ('declaration', d), ('base', base)): panel.write_new(inputs/f'{name}.json', value)
    run = panel.init(root/'run', declaration=inputs/'declaration.json', declaration_sha256=panel.sha(d),
        ladder_declaration=inputs/'ladder.json', ladder_declaration_sha256=panel.sha(lad), checkpoint='/inert/champion.pt',
        base_config=inputs/'base.json', base_config_sha256=panel.sha(base))
    return root/'run', run, d, lad


class DeclarationTests(unittest.TestCase):
    def test_accepts_the_declared_panel(self):
        for arms in (('RAW', 'DN2W8'), ('RAW', 'DN2W16'), ('RAW', 'DN2W8', 'DN2W16')):
            self.assertEqual(panel.validate(declaration('a'*64, arms))['arms'], list(arms))

    def test_refuses_any_other_declaration(self):
        changes = [('extra', 'host:1'), ('schema', 'other'), ('ladder_declaration_sha256', 'A'*64), ('checkpoint_sha256', 'b'*64),
                   ('arms', ['DN2W8', 'RAW']), ('arms', ['RAW']), ('arms', ['RAW', 'R1']), ('arms', ['RAW', 'DN2W8', 'DN2W8']),
                   ('search', {'DN2W8': dict(depth=2, worlds=16)}), ('root_operator', 'ARGMAX_MEAN_COMPLETED_Q_COMMON_ALL_RUNGS'),
                   ('deadline_seconds', 2.5), ('reserve_seconds', 0.), ('screen_paired_cells', 400), ('max_paired_cells', 1000),
                   ('confidence', .95), ('primary_comparison', 'SEARCH_MINUS_CODEX_RAW'), ('bar', 'POINT_GT_0'),
                   ('screen', dict(panel.SCREEN, latency_p95_max_seconds=4.)), ('screen', dict(panel.SCREEN, unknown_cells_max=5))]
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

    def test_extension_past_the_screen_needs_a_continuing_screen_readout(self):
        with TemporaryDirectory() as directory:
            root, run, d, lad = make_run(directory)
            with self.assertRaisesRegex(ValueError, 'screen'): panel.pending(root, arm='DN2W8', limit=2000)
            readout = dict(schema=panel.READOUT_SCHEMA, declaration_sha256=run['declaration_sha256'], paired_cells=500,
                           decision='STOP_HARM', arm='DN2W8')
            for decision in ('STOP_HARM', 'STOP_BREAKAGE', 'INCOMPLETE'):
                path = Path(directory)/f'{decision}.json'; value = dict(readout, decision=decision); panel.write_new(path, value)
                with self.assertRaisesRegex(ValueError, 'screen'): panel.pending(root, arm='DN2W8', limit=2000, screen=(path, panel.sha(value)))
            path = Path(directory)/'go.json'; value = dict(readout, decision='CONTINUE_TO_2000'); panel.write_new(path, value)
            with self.assertRaisesRegex(ValueError, 'differs'): panel.pending(root, arm='DN2W8', limit=2000, screen=(path, '0'*64))
            self.assertEqual(len(panel.pending(root, arm='RAW', limit=2000, screen=(path, panel.sha(value)))), 2000)


class FakeRuntime:
    made = []
    def __init__(self, *, client, telemetry_path, arm, cell, original):
        self.client, self.telemetry_path, self.arm, self.cell, self.original = client, telemetry_path, arm, cell, original
        FakeRuntime.made.append(self)
    async def run(self, config):
        self.claimed = (Path(self.telemetry_path).parents[1]/'claims'/f"{self.cell['ordinal']}.json").exists()
        if self.cell.get('fail'): raise RuntimeError('inert interrupted game')
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
        FakeRuntime.made.clear()
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
        with self.assertRaises(FileExistsError): self.work(packet)
        self.assertEqual(len(FakeRuntime.made), 2)

    def test_interrupted_game_is_unknown_and_claimed(self):
        packet, cell = self.packet('RAW', 0)
        with patch.object(FakeRuntime, 'run', side_effect=RuntimeError('inert interrupted game')):
            self.assertEqual(self.work(packet), 0)
        value = self.result(cell); self.assertEqual((value['status'], value['score']), ('UNKNOWN', None))
        self.assertIn('inert interrupted', value['error']); self.assertIsNone(FakeRuntime.made[0].client)
        self.assertTrue((self.root/'claims'/f"{cell['ordinal']}.json").exists())

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


FAKE_WORKER = r'''
import json, os, sys
for line in sys.stdin:
    packet = json.loads(line)
    if os.path.basename(packet['path']).startswith('fail'): sys.exit(3)
    with open(packet['path'] + '.done', 'x') as stream: json.dump(dict(pid=os.getpid(), server=packet['server']), stream)
    print(json.dumps({'exit': 0}), flush=True)
'''


class DispatchTests(unittest.TestCase):
    def run_dispatch(self, names, **kwargs):
        directory = Path(self.directory.name); packets = [dict(path=str(directory/n), sha256='0'*64) for n in names]
        result = asyncio.run(panel.dispatch(packets, command=lambda pair, cache: [sys.executable, '-c', FAKE_WORKER], **kwargs))
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
        result, done = self.run_dispatch(['a', 'fail', 'c', 'd'], pairs=[(0, 1)], server=None)
        self.assertEqual(result['dispatched'], 1); self.assertEqual(result['not_dispatched'], 2)
        self.assertEqual(len(result['failed']), 2); self.assertTrue(result['failed'][0].endswith('fail')); self.assertEqual(set(done), {'a'})

    def test_core_pairs_must_be_disjoint(self):
        for kwargs in (dict(pairs=[(0, 1), (1, 2)]), dict(pairs=[(0, 0)]), dict(pairs=[]), dict(pairs=[(0, 1)], cells_per_worker=0)):
            with self.subTest(kwargs), self.assertRaises(ValueError): asyncio.run(panel.dispatch([], server=None, **kwargs))


def decision_row(cell, i, *, source='search', cause=None, kind='move', differs=False, select=.5, operator='ARGMAX_MEAN_COMPLETED_Q'):
    search = None if source == 'raw_arm' else dict(status='COMPLETE' if source == 'search' else 'REFUSED', cause=cause,
        client_wall_seconds=select-.05, server_wall_seconds=select-.1, tree_commit='t'*40, archive_sha256='a'*64,
        root_operator=operator if source == 'search' else None, pricer=dict(k=1, depth=2, W=8, K=4) if source == 'search' else None,
        timings=dict(queue_seconds=.01, prepare_seconds=.2, build_seconds=.05, price_seconds=.2, reduce_seconds=.01) if source == 'search' else None)
    return dict(schema=live.ROW_SCHEMA, arm=cell['arm'], cell=cell, battle_id=f"b{cell['ordinal']}", owner='p1', decision_index=i,
        prefix_sha256=f'{i:064x}', request_kind=kind, source=source, cause=cause, search_differs=differs if source == 'search' else None,
        select_seconds=select, greedy=dict(seconds=.05), search=search)


def write_cell(root, cell, d_sha, *, score, status='COMPLETE', error=None, rows=(), decisions=None):
    for name in ('results', 'telemetry'): (root/name).mkdir(exist_ok=True)
    value = dict(schema=panel.CELL_SCHEMA, cell=cell, declaration_sha256=d_sha, status=status, score=score if status == 'COMPLETE' else None,
                 error=error, source_game=dict(pokezero_decisions=len(rows) if decisions is None else decisions))
    (root/'results'/f"{cell['ordinal']}.json").write_bytes(panel.encoded(value))
    if rows: (root/'telemetry'/f"{cell['ordinal']}.jsonl").write_bytes(b''.join(panel.encoded(r)+b'\n' for r in rows))


def populate(root, run, lad, *, limit=500, scores=None, telemetry=True, select=.5, skip=()):
    scores = scores or (lambda p: (float(p//2 % 3 == 0), float(p//2 % 3 != 1)))
    for arm, side in (('RAW', 0), ('DN2W8', 1)):
        for cell in panel.cells(lad, arm, limit=limit):
            if cell['pair_ordinal'] in skip: continue
            source = 'raw_arm' if arm == 'RAW' else 'search'
            rows = [decision_row(cell, i, source=source, differs=i == 0, select=select) for i in range(2)] if telemetry else ()
            write_cell(root, cell, run['declaration_sha256'], score=scores(cell['pair_ordinal'])[side], rows=rows, decisions=None if telemetry else 2)


def seed_cluster_interval(deltas):
    from scipy.stats import t
    clusters = [(deltas[i]+deltas[i+1])/2 for i in range(0, len(deltas), 2)]
    radius = float(t.ppf(.95, len(clusters)-1))*math.sqrt(variance(clusters)/len(clusters))
    return [max(-1., fmean(clusters)-radius), min(1., fmean(clusters)+radius)]


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(); self.root, self.run, self.d, self.lad = make_run(self.directory.name)

    def tearDown(self): self.directory.cleanup()

    def test_clean_screen_continues_with_the_seed_cluster_interval(self):
        populate(self.root, self.run, self.lad)
        value = panel.readout(self.root, arm='DN2W8', limit=500)
        deltas = [float(p//2 % 3 != 1)-float(p//2 % 3 == 0) for p in range(500)]
        self.assertEqual(value['decision'], 'CONTINUE_TO_2000'); self.assertEqual(value['breakage'], [])
        self.assertAlmostEqual(value['paired']['point_difference'], fmean(deltas), places=12)
        for got, want in zip(value['paired']['interval90'], seed_cluster_interval(deltas)): self.assertAlmostEqual(got, want, places=12)
        self.assertEqual((value['paired']['known_pairs'], value['paired']['complete_seed_clusters']), (500, 250))
        self.assertIsNone(value['bar']); self.assertEqual(value['arms']['DN2W8']['decisions']['decisions'], 1000)
        self.assertEqual(value['dilution'], dict(searched_share=1., fallback_share=0., search_differs_share_of_all=.5))
        self.assertEqual(value['arms']['RAW']['decisions']['by_source'], {'raw_arm': 1000})
        json.dumps(value, allow_nan=False)

    def test_harm_stops_the_screen(self):
        populate(self.root, self.run, self.lad, scores=lambda p: (float(p % 7 != 0), float(p % 5 == 0)))
        value = panel.readout(self.root, arm='DN2W8', limit=500)
        self.assertLess(value['paired']['interval90'][1], 0); self.assertEqual(value['decision'], 'STOP_HARM')

    def test_breakage_stops_the_screen(self):
        populate(self.root, self.run, self.lad, select=3.5)
        value = panel.readout(self.root, arm='DN2W8', limit=500)
        self.assertEqual((value['decision'], value['breakage']), ('STOP_BREAKAGE', ['latency_p95']))
        self.assertEqual(value['arms']['DN2W8']['decisions']['late_decisions'], 1000)
        cell = panel.cells(self.lad, 'DN2W8', limit=500)[9]
        write_cell(self.root, cell, self.run['declaration_sha256'], score=None, status='UNKNOWN',
                   error='TimeoutError: owning candidate decision deadline expired', rows=[decision_row(cell, 0)])
        raw = panel.cells(self.lad, 'RAW', limit=500)[11]
        write_cell(self.root, raw, self.run['declaration_sha256'], score=1., rows=[decision_row(raw, 0, source='raw_arm')], decisions=5)
        value = panel.readout(self.root, arm='DN2W8', limit=500)
        self.assertEqual(value['breakage'], ['unknown_cells', 'timeout_forfeits', 'latency_p95', 'telemetry'])
        self.assertEqual(value['arms']['RAW']['telemetry_mismatch_pairs'], [11]); self.assertEqual(value['arms']['DN2W8']['timeout_forfeits'], 1)
        self.assertEqual(value['paired']['unknown_pairs'], 1); self.assertIsNone(value['paired']['point_difference'])

    def test_search_provenance_must_be_single_and_declared(self):
        populate(self.root, self.run, self.lad)
        cell = panel.cells(self.lad, 'DN2W8', limit=500)[0]
        write_cell(self.root, cell, self.run['declaration_sha256'], score=1., rows=[decision_row(cell, 0, operator='ARGMAX_POLICY_TARGET')])
        self.assertEqual(panel.readout(self.root, arm='DN2W8', limit=500)['breakage'], ['search_provenance'])

    def test_unattempted_cells_are_incomplete_but_lost_claims_are_unknown(self):
        populate(self.root, self.run, self.lad, skip={498, 499})
        value = panel.readout(self.root, arm='DN2W8', limit=500)
        self.assertEqual(value['decision'], 'INCOMPLETE'); self.assertEqual(value['arms']['DN2W8']['not_attempted'], 2)
        self.assertEqual(value['paired']['unknown_pairs'], 2)
        cell = panel.cells(self.lad, 'DN2W8', limit=500)[498]; panel.write_new(self.root/'claims'/f"{cell['ordinal']}.json", {})
        value = panel.readout(self.root, arm='DN2W8', limit=500)
        self.assertEqual(value['arms']['DN2W8']['claimed_without_result'], 1); self.assertEqual(value['decision'], 'STOP_BREAKAGE')

    def test_bar_at_two_thousand(self):
        populate(self.root, self.run, self.lad, limit=2000, telemetry=False)
        value = panel.readout(self.root, arm='DN2W8', limit=2000)
        self.assertGreater(value['paired']['interval90'][0], 0); self.assertEqual(value['decision'], 'BAR_MET')
        self.assertEqual(value['bar'], dict(rule='LOWER_BOUND_GT_0_AT_2000', met=True, met_full_roster=True))
        self.assertIn('telemetry', value['breakage'])

    def test_bar_not_met_at_two_thousand(self):
        populate(self.root, self.run, self.lad, limit=2000, telemetry=False, scores=lambda p: (float(p % 3 == 0), float(p % 4 == 0)))
        value = panel.readout(self.root, arm='DN2W8', limit=2000)
        self.assertLessEqual(value['paired']['interval90'][0], 0); self.assertEqual(value['decision'], 'BAR_NOT_MET')
        self.assertFalse(value['bar']['met'])

    def test_readout_refuses_foreign_results_and_undeclared_arms(self):
        populate(self.root, self.run, self.lad)
        with self.assertRaises(ValueError): panel.readout(self.root, arm='DN2W16', limit=500)
        with self.assertRaises(ValueError): panel.readout(self.root, arm='DN2W8', limit=1000)
        cell = panel.cells(self.lad, 'RAW', limit=500)[0]
        write_cell(self.root, cell, 'f'*64, score=1., rows=[decision_row(cell, 0, source='raw_arm')])
        with self.assertRaisesRegex(ValueError, 'foreign'): panel.readout(self.root, arm='DN2W8', limit=500)

    def test_codex_raw_agreement_is_a_plumbing_check(self):
        populate(self.root, self.run, self.lad)
        codex = Path(self.directory.name)/'codex'; (codex/'results').mkdir(parents=True); lad_sha = self.run['ladder_declaration_sha256']
        for cell in ladder.cells(self.lad, 'RAW', limit=500)[:10]:
            ours = float(cell['pair_ordinal']//2 % 3 == 0); score = ours if cell['pair_ordinal'] != 4 else 1.-ours
            (codex/'results'/f"{cell['ordinal']}.json").write_bytes(panel.encoded(dict(cell=cell, status='COMPLETE', score=score, declaration_sha256=lad_sha)))
        value = panel.readout(self.root, arm='DN2W8', limit=500, codex_raw_directory=codex)['codex_raw_agreement']
        self.assertEqual((value['codex_results'], value['shared_complete'], value['agree'], value['disagreeing_pairs']), (10, 10, 9, [4]))
        self.assertEqual(value['scope'], 'PLUMBING_CHECK_ONLY_NOT_THE_COMPARISON')
        first = ladder.cells(self.lad, 'RAW', limit=500)[0]
        (codex/'results'/f"{first['ordinal']}.json").write_bytes(panel.encoded(dict(cell=first, status='COMPLETE', score=1., declaration_sha256='0'*64)))
        with self.assertRaisesRegex(ValueError, 'foreign'): panel.readout(self.root, arm='DN2W8', limit=500, codex_raw_directory=codex)


class DecisionSummaryTests(unittest.TestCase):
    def test_fallback_by_cause_and_kind_latency_and_differs(self):
        cell = dict(arm='DN2W8', ordinal=1)
        specs = [('search', None, 'move', True), ('search', None, 'move', False), ('search', None, 'switch', True),
                 ('search', None, 'forced', False), ('greedy_fallback', 'timeout', 'move', None), ('greedy_fallback', 'timeout', 'move', None),
                 ('greedy_fallback', 'constructor_refused', 'switch', None), ('greedy_fallback', 'expired_in_queue', 'move', None),
                 ('search', None, 'move', False), ('greedy_fallback', 'bad_q', 'forced', None)]
        rows = [decision_row(cell, i, source=s, cause=c, kind=k, differs=bool(x), select=(i+1)/10) for i, (s, c, k, x) in enumerate(specs)]
        value = panel.decisions(rows, deadline=.75)
        self.assertEqual((value['decisions'], value['by_source']), (10, dict(search=5, greedy_fallback=5)))
        self.assertEqual(value['fallback_by_cause'], dict(bad_q=dict(count=1, share=.1), constructor_refused=dict(count=1, share=.1),
            expired_in_queue=dict(count=1, share=.1), timeout=dict(count=2, share=.2)))
        move = value['by_request_kind']['move']
        self.assertEqual((move['decisions'], move['searched'], move['fallback'], move['search_differs']), (6, 3, 3, 1))
        self.assertEqual(move['fallback_by_cause'], dict(expired_in_queue=1, timeout=2)); self.assertAlmostEqual(move['search_differs_share'], 1/3)
        self.assertEqual(value['by_request_kind']['switch']['fallback_by_cause'], dict(constructor_refused=1))
        self.assertEqual((value['search_differs'], value['search_differs_share_of_searched'], value['search_differs_share_of_all']), (2, .4, .2))
        self.assertEqual(value['latency']['select'], dict(n=10, p50=.5, p95=1., p99=1., max=1.))
        self.assertEqual(value['latency']['prepare']['n'], 5); self.assertEqual(value['latency']['client']['n'], 10)
        self.assertEqual(value['late_decisions'], 3); self.assertEqual(value['provenance']['root_operator'], ['ARGMAX_MEAN_COMPLETED_Q'])
        with self.assertRaisesRegex(ValueError, 'unique'): panel.decisions(rows+rows[:1], deadline=3.)
        for bad in (dict(rows[0], cause='timeout'), dict(rows[4], cause=None), dict(rows[4], cause='other'), dict(rows[0], request_kind='team'),
                    dict(rows[0], source='raw_arm'), dict(rows[0], schema='other')):
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


if __name__ == '__main__': unittest.main()
