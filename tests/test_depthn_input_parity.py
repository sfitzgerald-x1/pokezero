"""CPU tests of scripts/depthn_input_parity.py: no GPU, no champion, no torch checkpoint. The capture runs the real
DepthNLiveSearch (process_many, _batch, _search) and the real re-pricer main over the re-pricer tests' fakes (constructor, pricing,
backups), with a richer fake build, a fake root forward and fake root views; the torch path runs observation_window_to_torch under a
fake torch module (and the real one when installed)."""
from contextlib import ExitStack, redirect_stderr
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import depthn_input_parity as P
from scripts import depthn_rr_reprice as R
from tests import test_depthn_rr_reprice as RT
from gpu_engine import constructor_worlds as cw, depthn_live_search as live
from pokezero import neural_policy
from pokezero.observation import PokeZeroObservationV0

ORIGINAL_BUILD, ORIGINAL_CAPTURE, ORIGINAL_ROW = cw.build, R.Capture, R.decision_row
ROLES = (0,) + (1,) * 6 + (2,) * 6 + (3,) * 9 + (5,) + (7,) * 6
try:
    import torch as TORCH   # noqa: F401
except ImportError:
    TORCH = None


def root_inputs(seed=0):
    """World 0's V2 encoder inputs of one root: [1,1,29,132] / [1,1,29,41] / [1,1,29] / [1,1,29] / [1,1]."""
    g = np.random.default_rng(seed)
    return dict(numeric_features=g.standard_normal((1, 1, 29, 132)).astype(np.float32), categorical_ids=g.integers(0, 900, (1, 1, 29, 41)).astype(np.int32),
        token_type_ids=np.asarray(ROLES, np.int32).reshape(1, 1, 29), attention_mask=np.ones((1, 1, 29), bool), history_mask=np.ones((1, 1), bool))


def b2_of(inputs, *, shift=.5):
    """B2's encoding of the same root: the world-0 prefix, except opponent token 8's CANDIDATE_SET_COUNT (numeric column 43)."""
    out = {k: np.array(inputs[k][0, 0, :23]) for k in P.B2_KEYS}
    out['numeric_features'][8, 43] += np.float32(shift);out['legal_action_mask'] = np.asarray(RT.LEGAL, bool)
    return out


ROOT = root_inputs()
B2 = b2_of(ROOT)


def fake_forward(p, inputs):
    """A deterministic root forward: LOGITS moved by opponent token 8's column 43 (so B2 and world 0 give different logits)."""
    x = np.asarray(inputs['numeric_features'], np.float32)[:, 0, 8, 43]
    logits = (np.asarray(RT.LOGITS, np.float32)[None] + np.float32(.001) * x[:, None]).astype(np.float32)
    return SimpleNamespace(policy_logits=logits, value=np.tanh(x).astype(np.float32), valid=np.ones(x.shape, bool))


def fake_views(state, public, owner):
    """Our own view of a world: world 0's inputs, except world 5 (numeric token 3, column 7) and world 6 (belief token 25)."""
    w = int(np.asarray(public['w']));own = {k: np.array(v) for k, v in ROOT.items()}
    if w == 5:own['numeric_features'][0, 0, 3, 7] += 1
    if w == 6:own['categorical_ids'][0, 0, 25, 0] += 1
    return None, None, own, None


def priced_logits():
    return tuple(map(float, np.asarray(fake_forward(None, ROOT).policy_logits).reshape(-1)))


def fake_build(constructor, prepared, pricer, params, *, binding, recipe, only_complete):
    """build's output shape: per root its index, rows (8 blocks), root prior from fake_forward on ROOT, b2_view; Requests holding the
    root inputs."""
    roots, requests = [], []
    for index, item in enumerate(prepared):
        if type(item) is cw.RootRefusal:
            roots.append(dict(index=index, request_id=item.request_id, status='REFUSED', reason='constructor-root:' + item.kind, rows=()));continue
        rows = tuple(range(len(requests), len(requests) + 8));requests += [SimpleNamespace(root_inputs=ROOT) for _ in rows]
        out = fake_forward(None, ROOT)
        roots.append(dict(index=index, request_id=item.request_id, status='PRICED', reason=None, rows=rows, root_logits=priced_logits(),
            root_legal=item.legal, root_value=float(out.value[0]), root_view_world=0, b2_view=dict(equal=False)))
    return SimpleNamespace(requests=tuple(requests), roots=tuple(roots))


class B2Constructor(RT.FakeConstructor):
    """The re-pricer tests' constructor, its roots carrying B2's encoding."""
    def prepare(self, request, index):
        detached, made, legal = super().prepare(request, index)
        return SimpleNamespace(address=detached.address, view=dict(_b2_root_inputs=B2)), made, legal


def selected(*keys, name='disagree'):
    return {k: dict(set=name, live_greedy=0, root_logits=list(RT.LOGITS), root_legal=list(RT.LEGAL), server_top=3, battle_id=None) for k in keys}


def body(i):
    return dict(RT.body(i), request_id=f'rr-1001-{i}-7')


def capture_fakes(stack):
    """The re-pricer tests' pricing, then this module's build, forward and views (still clock)."""
    RT.pricing(stack, still_clock=True)
    stack.enter_context(patch.object(cw, 'build', fake_build))
    stack.enter_context(patch.object(P, 'root_forward_of', lambda pricer: fake_forward))
    stack.enter_context(patch.object(P, 'views_of', lambda pricer: fake_views))
    stack.enter_context(patch.object(P, 'installed_manifest', lambda public, owner: [dict(section='own', kind='numeric', column=0, status='DIRECT', reason='x')]))


# Selection ----------------------------------------------------------------------------------------------------------------------
def rr_row(ordinal, index, logits, legal, greedy, status='COMPLETE'):
    return dict(ordinal=ordinal, decision_index=index, battle_id=f'b{ordinal}', replay=dict(status=status), root=dict(root_logits=logits, root_legal=legal),
        live=None if greedy is None else dict(greedy_action_index=greedy, source='search'))


class SelectionTests(unittest.TestCase):
    def test_disagree_control_tie_rule_and_unclassified(self):
        legal = [True, True, False, True] + [False] * 5
        tie = [1., 2., 9., 2.] + [0.] * 5   # actions 1 and 3 tie: the lowest index wins
        self.assertEqual(P.argmax_legal(tie, legal), 1);self.assertIsNone(P.argmax_legal(tie, [False] * 9))
        rows = [dict(ordinal=1, decision_index=i, battle_id='b', replay_status='COMPLETE', root_logits=tie, root_legal=legal, live_greedy=g, live_source='search', source='x')
            for i, g in enumerate([1, 3, 0, 1, 1])]
        rows += [dict(rows[0], decision_index=5, replay_status='REFUSED'), dict(rows[0], decision_index=6, live_greedy=None)]
        c = P.classify(rows)
        self.assertEqual(c['disagree'], [(1, 1), (1, 2)]);self.assertEqual(c['agree'], [(1, 0), (1, 3), (1, 4)])
        self.assertEqual([x[1] for x in c['unclassified']], ['replay REFUSED', 'no root prior or live greedy'])
        chosen, counts = P.select(rows, 7)
        self.assertEqual(sorted(k for k, v in chosen.items() if v['set'] == 'disagree'), [(1, 1), (1, 2)])
        control = sorted(k for k, v in chosen.items() if v['set'] == 'control')
        self.assertEqual(len(control), 2);self.assertTrue(set(control) <= {(1, 0), (1, 3), (1, 4)})
        self.assertEqual((counts['disagree'], counts['agree'], counts['control'], counts['unclassified']), (2, 3, 2, 2))
        self.assertEqual(chosen[1, 1]['server_top'], 1)

    def test_seed_determinism_and_equal_size(self):
        legal = [True] * 9;logits = [float(a) for a in range(9)]   # server top 8
        rows = [dict(ordinal=o, decision_index=i, battle_id='b', replay_status='COMPLETE', root_logits=logits, root_legal=legal,
            live_greedy=8 if (o + i) % 5 else 0, live_source='search', source='x') for o in range(10) for i in range(10)]
        a, _ = P.select(rows, 1);b, _ = P.select(list(reversed(rows)), 1);c, _ = P.select(rows, 2)
        control = lambda s: sorted(k for k, v in s.items() if v['set'] == 'control')
        self.assertEqual(a, b);self.assertNotEqual(control(a), control(c))
        self.assertEqual(len(control(a)), sum(v['set'] == 'disagree' for v in a.values()))
        self.assertEqual(control(a), sorted(random.Random(1).sample(sorted((r['ordinal'], r['decision_index']) for r in rows if r['live_greedy'] == 8), 20)))

    def test_rr_rows_read_sorted_and_refuse_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp) / 'rows-1.jsonl', Path(tmp) / 'rows-0.jsonl'
            one.write_text(json.dumps(rr_row(5, 2, [0.] * 9, [True] * 9, 1)) + '\n\n' + json.dumps(rr_row(3, 1, [0.] * 9, [True] * 9, None)) + '\n')
            two.write_text(json.dumps(rr_row(5, 0, [0.] * 9, [True] * 9, 4, status='REFUSED')) + '\n')
            rows = P.rr_rows([one, two])
            self.assertEqual([(r['ordinal'], r['decision_index']) for r in rows], [(3, 1), (5, 0), (5, 2)])
            self.assertEqual((rows[0]['live_greedy'], rows[1]['replay_status'], rows[2]['live_greedy']), (None, 'REFUSED', 1))
            two.write_text(json.dumps(rr_row(5, 2, [0.] * 9, [True] * 9, 4)) + '\n')
            with self.assertRaisesRegex(ValueError, 'twice'):P.rr_rows([one, two])

    def test_battle_sharding(self):
        chosen = {(o, i): None for o in (11, 3, 7, 5, 9) for i in range(2)}
        shards = [P.shard_battles(chosen, s, 2) for s in range(2)]
        self.assertEqual(shards, [[3, 7, 11], [5, 9]]);self.assertEqual(P.shard_battles(chosen, 0, 2, battles=[7, 11]), [7, 11])
        self.assertEqual(P.shard_battles(chosen, 0, 2, limit=1), [3]);self.assertEqual(P.shard_battles(chosen, 1, 2, battles=[9], limit=5), [9])
        with self.assertRaisesRegex(ValueError, 'battles'):P.shard_battles(chosen, 0, 2, battles=[5])
        self.assertEqual((P.request_key('rr-1001-17-99'), P.request_key('warmup-b1-0-0'), P.request_key(None)), ((1001, 17), None, None))


# Splice and diffs ----------------------------------------------------------------------------------------------------------------
class SpliceTests(unittest.TestCase):
    def test_splice_replaces_exactly_the_first_23_tokens(self):
        before = {k: v.copy() for k, v in ROOT.items()}
        b2 = {k: v.copy() for k, v in B2.items()};b2['categorical_ids'] = (b2['categorical_ids'] + 1).astype(np.int64);b2['token_type_ids'] = b2['token_type_ids'] * 0
        b2['attention_mask'] = np.zeros(23, bool)
        X, lossy = P.splice(ROOT, b2)
        self.assertEqual(lossy, [])
        for k in ROOT:self.assertTrue(P.bit_equal(ROOT[k], before[k]), k)   # the root inputs are never written
        for k, v in X.items():self.assertEqual((v.shape, v.dtype), (ROOT[k].shape, ROOT[k].dtype), k)
        for k in P.B2_KEYS:
            self.assertTrue(np.array_equal(X[k][0, 0, :23], b2[k].astype(ROOT[k].dtype)), k)
            self.assertTrue(P.bit_equal(X[k][0, 0, 23:], ROOT[k][0, 0, 23:]), k)   # the belief tokens stay world 0's
        self.assertTrue(P.bit_equal(X['history_mask'], ROOT['history_mask']));self.assertIsNot(X['history_mask'], ROOT['history_mask'])
        self.assertEqual(float(X['numeric_features'][0, 0, 8, 43]), float(ROOT['numeric_features'][0, 0, 8, 43] + np.float32(.5)))
        wide = dict(b2, categorical_ids=b2['categorical_ids'].copy());wide['categorical_ids'][0, 0] = 2 ** 40
        self.assertEqual(P.splice(ROOT, wide)[1], ['categorical_ids'])   # a value the root dtype cannot hold is named
        with self.assertRaisesRegex(ValueError, 'does not fit'):P.splice(ROOT, dict(b2, numeric_features=np.zeros((22, 132), np.float32)))

    def test_encoding_diff(self):
        b2 = {k: v.copy() for k, v in B2.items()};b2['categorical_ids'][13, 0] += 1
        d = P.encoding_diff(b2, ROOT)
        self.assertEqual(d['keys'], dict(numeric_features=False, categorical_ids=False, token_type_ids=True, attention_mask=True))
        self.assertEqual([tuple(map(int, x)) for x in zip(*np.nonzero(d['numeric_features']))], [(8, 43)])
        self.assertEqual([tuple(map(int, x)) for x in zip(*np.nonzero(d['categorical_ids']))], [(13, 0)])
        self.assertAlmostEqual(d['max_numeric_diff'], .5, places=6);self.assertFalse(d['bit_only'])
        same = {k: np.array(ROOT[k][0, 0, :23]) for k in P.B2_KEYS};zero = dict(same, numeric_features=same['numeric_features'].copy())
        zero['numeric_features'][0, 0] = 0.;flipped = {k: np.array(v) for k, v in ROOT.items()};flipped['numeric_features'][0, 0, 0, 0] = -0.
        d = P.encoding_diff(zero, flipped)
        self.assertTrue(all(d['keys'].values()));self.assertTrue(d['bit_only'])   # +0.0 against -0.0
        self.assertEqual(P.diff_cells(P.encoding_diff(b2, ROOT)), {('opponent', 'numeric', 43), ('actions', 'categorical', 0)})

    def test_own_view_diff_across_worlds(self):
        views = {w: fake_views(None, dict(w=w), 0)[2] for w in range(8)}
        c = P.own_view_diff(views, 0)
        self.assertFalse(c['prefix_equal']);self.assertFalse(c['full_equal']);self.assertFalse(c['value_equal'])
        self.assertEqual((c['differing_worlds'], c['differing_keys'], c['beyond_prefix_keys']), ([5, 6], ['categorical_ids', 'numeric_features'], ['categorical_ids']))
        self.assertEqual(c['cells'], [['numeric_features', 3, 7]]);self.assertAlmostEqual(c['max_abs_diff'], 1., places=5)
        self.assertEqual(P.crossworld_cells(c), {('own', 'numeric', 7)})
        same = P.own_view_diff({0: ROOT, 1: {k: np.array(v) for k, v in ROOT.items()}}, 0)
        self.assertTrue(same['prefix_equal'] and same['full_equal'] and same['value_equal']);self.assertEqual(same['differing_worlds'], [])
        signed = {k: np.array(v) for k, v in ROOT.items()};signed['numeric_features'][0, 0, 0, 0] = 0.;other = {k: np.array(v) for k, v in signed.items()}
        other['numeric_features'][0, 0, 0, 0] = -0.
        c = P.own_view_diff({0: signed, 1: other}, 0)   # the pricer's == check passes; bits differ
        self.assertTrue(c['value_equal']);self.assertFalse(c['prefix_equal']);self.assertEqual(c['cells'], [['numeric_features', 0, 0]])


# Capture -------------------------------------------------------------------------------------------------------------------------
class CaptureTests(unittest.TestCase):
    def test_call_sites(self):
        self.assertTrue(P.check_build_call_site());self.assertTrue(P.check_reprice_hooks())

        def bound(self, todo):   # a search holding its own reference to build would escape the wrapper
            build = cw.build
            return build(None, todo)
        with self.assertRaises(RuntimeError):P.check_build_call_site(bound)

        def local(argv):
            Capture, decision_row = object, print
            return Capture(), decision_row()
        with self.assertRaises(RuntimeError):P.check_reprice_hooks(local)

    def test_build_wrapper_leaves_return_and_decisions_unchanged(self):
        requests = [body(0), body(1), body(2), body(3)]
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            capture_fakes(stack)
            priced = []
            stack.enter_context(patch.object(cw, 'price', side_effect=lambda pricer, made, *a, **k: priced.append(made) or (SimpleNamespace(), None)))
            plain = RT.fake_search(B2Constructor(refuse={1}))
            without = [plain.process_many([r])[0] for r in requests]
            built = []
            real = cw.build
            stack.enter_context(patch.object(cw, 'build', lambda *a, **k: built.append(real(*a, **k)) or built[-1]))
            recorder = P.Recorder(selected((1001, 0), (1001, 1), (1001, 2)), tmp);wrapped = RT.fake_search(B2Constructor(refuse={1}))
            inner = cw.build;within = []
            for r in requests:
                capture = P.capture_class(recorder)()
                with capture.installed():
                    self.assertIsNot(cw.build, inner);self.assertIsNot(live.decide, RT.ORIGINAL_DECIDE)
                    within.append(wrapped.process_many([r])[0])
                self.assertIs(cw.build, inner);self.assertIs(live.decide, RT.ORIGINAL_DECIDE);self.assertIs(cw.prepare_roots, RT.ORIGINAL_PREPARE)
                self.assertEqual(len(capture.prepared), 1)   # rr's own capture still works underneath
            recorder.close()
        self.assertEqual([r['status'] for r in without], ['COMPLETE', 'REFUSED', 'COMPLETE', 'COMPLETE'])
        self.assertEqual(json.dumps(within, sort_keys=True), json.dumps(without, sort_keys=True))   # byte for byte, timings included
        wrapped_priced = priced[len(priced) - 3:]
        self.assertEqual(len(built), 4);priced_built = [b for b in built if b.requests]
        self.assertEqual(len(priced_built), 3)
        for made, b in zip(wrapped_priced, priced_built):self.assertIs(made, b)   # price got build's own objects
        self.assertEqual(sorted(recorder.pending), ['rr-1001-0-7', 'rr-1001-1-7', 'rr-1001-2-7'])   # only the selected; refused too
        refused, record = recorder.pending['rr-1001-1-7'], recorder.pending['rr-1001-0-7']
        self.assertEqual((refused['build_status'], refused['arrays']), ('REFUSED', {}))
        self.assertEqual(record['errors'], []);self.assertEqual(record['path2'], dict(logits_equal=True, value_equal=True, valid=True))
        self.assertEqual(record['path3']['logits'], P.floats(fake_forward(None, P.splice(ROOT, B2)[0]).policy_logits))
        self.assertNotEqual(record['path3']['logits'], list(priced_logits()))
        self.assertEqual((record['path3']['valid'], record['path3']['lossy_keys'], record['recorded_float32_exact']), (True, [], True))
        self.assertTrue(P.bit_equal(record['arrays']['root__numeric_features'], ROOT['numeric_features']))
        self.assertTrue(P.bit_equal(record['arrays']['b2__numeric_features'], B2['numeric_features']))
        cross = record['crossworld']
        self.assertEqual((cross['differing_worlds'], cross['prefix_equal'], cross['reference_reproduces_root_inputs']), ([5, 6], False, True))
        self.assertEqual(sorted({n.split('__')[1] for n in record['arrays'] if n.startswith('xw__')}, key=int), ['0', '1', '5', '6'])
        self.assertEqual(recorder.manifest[0]['status'], 'DIRECT')

    def test_wrappers_restore_when_the_call_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = P.Recorder({}, tmp);capture = P.capture_class(recorder)()
            with self.assertRaises(KeyError):
                with capture.installed():raise KeyError('boom')
            recorder.close()
            self.assertIs(cw.build, ORIGINAL_BUILD);self.assertIs(live.decide, RT.ORIGINAL_DECIDE);self.assertIs(cw.prepare_roots, RT.ORIGINAL_PREPARE)
            with self.assertRaises(ValueError):
                with P.reprice_hooks(recorder):
                    self.assertIsNot(R.Capture, ORIGINAL_CAPTURE);self.assertIsNot(R.decision_row, ORIGINAL_ROW);raise ValueError('boom')
            self.assertIs(R.Capture, ORIGINAL_CAPTURE);self.assertIs(R.decision_row, ORIGINAL_ROW)

    def test_a_failing_capture_never_reaches_the_search(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            capture_fakes(stack)
            stack.enter_context(patch.object(P, 'root_forward_of', lambda pricer: (_ for _ in ()).throw(RuntimeError('no device'))))
            recorder = P.Recorder(selected((1001, 0)), tmp);search = RT.fake_search(B2Constructor())
            with P.capture_class(recorder)().installed():response = search.process_many([body(0)])[0]
            recorder.close()
        self.assertEqual(response['status'], 'COMPLETE');self.assertEqual(recorder.pending['rr-1001-0-7']['errors'], ['RuntimeError: no device'])


class MainTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'JAX_DEFAULT_MATMUL_PRECISION': 'highest'});environment.start();self.addCleanup(environment.stop)

    def run_directory(self, tmp):
        """Cell 1001: decisions 0-4 (1 single-legal, 4 refused by the constructor); live greedy 0, -, 3, 3, 3 (the prior's top is 3)."""
        root = Path(tmp) / 'run';RT.write_run(root, [(1001, 'DN2W16', 7, 1, 'COMPLETE')])
        captured = RT.write_capture(root / 'owning-wire' / '1001.jsonl', [False, True, False, False, False])
        greedy = {0: 0, 1: 4, 2: 3, 3: 3, 4: 3}
        with (root / 'telemetry' / '1001.jsonl').open('w') as f:
            for i in range(5):f.write(json.dumps(dict(decision_index=i, source='search' if i != 1 else 'single_legal', search=dict(status='COMPLETE'),
                request_kind='move', prefix_sha256=captured[i]['prefix_sha256'], greedy=dict(action_index=greedy[i]), turn=1, played_action_index=0)) + '\n')
        rows = Path(tmp) / 'rows-0.jsonl'
        rows.write_text(''.join(json.dumps(rr_row(1001, i, list(priced_logits()), list(RT.LEGAL), greedy[i], 'COMPLETE' if i != 4 else 'REFUSED')) + '\n'
            for i in (0, 2, 3, 4)))
        return root, rows

    def patches(self, stack, search):
        capture_fakes(stack)
        header = dict(R.LIVE_IDENTITY, rung='DN2W16', pricer=dict(buckets=[1], W=16), tree_commit='t')
        for target, name, value in ((R, 'tree_guard', dict(head='h')), (R.server, 'read_requests', [dict(saved=1)]), (R.server, 'build', search),
                (R.server, 'header', header), (R, 'PeerLanes', RT.fake_peer), (R.server.evaluators, 'check_arguments', None)):
            stack.enter_context(patch.object(target, name, return_value=value))
        stack.enter_context(redirect_stderr(io.StringIO()));stack.enter_context(patch('sys.stdout', new_callable=io.StringIO))

    def test_capture_replays_records_and_matches_the_plain_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, rows = self.run_directory(tmp);out, plain = Path(tmp) / 'cap', Path(tmp) / 'plain'
            server = RT.BASE + RT.CHAMPION + ['--run-directory', str(root)]
            with ExitStack() as stack:
                self.patches(stack, RT.fake_search(B2Constructor(refuse={4})))
                code = P.main(['capture', '--rr-rows', str(rows), '--select-seed', '3', '--out', str(out), *server])
                self.assertIs(R.Capture, ORIGINAL_CAPTURE);self.assertIs(R.decision_row, ORIGINAL_ROW)
            with ExitStack() as stack:   # the re-pricer alone, on the same inputs
                self.patches(stack, RT.fake_search(B2Constructor(refuse={4})))
                R.main(server + ['--cells', '1001', '--out', str(plain)])
            replay = [json.loads(x) for x in (out / 'replay' / 'rows-0.jsonl').read_text().splitlines()]
            alone = [json.loads(x) for x in (plain / 'rows-0.jsonl').read_text().splitlines()]
            index = [json.loads(x) for x in (out / 'index.jsonl').read_text().splitlines()]
            summary = json.loads((out / 'capture-summary.json').read_text());selection = json.loads((out / 'selection.json').read_text())
            with np.load(out / 'arrays' / '1001.npz') as z:arrays = {k: z[k] for k in z.files}
            entries = P.capture_entries([out]);loaded = dict(next(x for x in P.decision_arrays(entries) if x[1] is not None)[1])
        self.assertEqual(code, 0)
        strip = lambda r: {k: v for k, v in r.items() if k not in ('wall_seconds', 'timings', 'request_id', 'decision_seconds')}
        self.assertEqual([strip(r['replay']) for r in replay], [strip(r['replay']) for r in alone])   # the same decisions
        self.assertEqual([(r['root'], r['parity']) for r in replay], [(r['root'], r['parity']) for r in alone])
        self.assertEqual([r['decision_index'] for r in replay], [0, 2, 3, 4])
        self.assertEqual(selection['counts']['disagree'], 1);self.assertEqual(selection['counts']['control'], 1)
        sets = {r['decision_index']: r['set'] for r in selection['decisions']}
        self.assertEqual(sets[0], 'disagree');self.assertEqual(len(sets), 2)
        self.assertEqual([(e['decision_index'], e['set']) for e in index], sorted(sets.items()))
        first = index[0]
        self.assertEqual((first['captured'], first['build_status'], first['replay_status'], first['errors']), (True, 'PRICED', 'COMPLETE', []))
        self.assertEqual(first['path2'], dict(logits_equal=True, value_equal=True, valid=True))
        self.assertEqual((first['selection']['live_greedy'], first['live']['greedy_action_index'], first['live_greedy_matches_selection']), (0, 0, True))
        self.assertTrue(first['root_logits_match_selection'] and first['decide_root_logits_match'])
        self.assertEqual(first['parity']['replay_complete'], True);self.assertEqual(first['crossworld']['differing_worlds'], [5, 6])
        self.assertTrue(P.bit_equal(arrays['d0__root__numeric_features'], ROOT['numeric_features']))
        self.assertTrue(P.bit_equal(arrays['d0__b2__numeric_features'], B2['numeric_features']))
        self.assertEqual(P.floats(arrays['d0__path3__logits']), first['path3']['logits'])
        self.assertTrue(P.bit_equal(loaded['root__logits'], np.asarray(priced_logits(), np.float32)))
        self.assertEqual((summary['selected_in_shard'], summary['recorded'], summary['captured'], summary['path2_exact']), (2, 2, 2, 2))
        self.assertEqual((summary['missing'], summary['errors'], summary['interrupted'], summary['reprice_exit']), ([], [], False, 0))

    def test_capture_refuses_reprice_selection_flags(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            P.main(['capture', '--rr-rows', 'x', '--select-seed', '1', '--out', 'o', '--cells', '1'])


# Torch window tensors --------------------------------------------------------------------------------------------------------------
class FakeTorch:
    """What observation_window_to_torch uses of torch, over numpy."""
    long, float32, bool = np.int64, np.float32, np.bool_

    @staticmethod
    def tensor(data, dtype=None, device=None):
        return np.array(data, dtype=dtype)


class TorchWindowTests(unittest.TestCase):
    def check(self, torch_module, to_numpy):
        legal = [True, False] * 4 + [True]
        observation = P.actor_observation(B2, legal)
        self.assertIsInstance(observation, PokeZeroObservationV0);self.assertEqual(observation.schema_version, 'pokezero.observation.v4')
        got = {k: to_numpy(v) for k, v in P.actor_tensors(observation).items()}
        # The same fields hand-built as LiveProtocolActor._observation does, on a plain object.
        fake = SimpleNamespace(numeric_features=B2['numeric_features'].astype(np.float32).tolist(), categorical_ids=B2['categorical_ids'].tolist(),
            token_type_ids=B2['token_type_ids'].tolist(), attention_mask=B2['attention_mask'].tolist(), legal_action_mask=legal)
        want = {k: to_numpy(v) for k, v in neural_policy.observation_window_to_torch([fake], window_size=1).items()}
        self.assertEqual(sorted(got), sorted(want))
        for k in want:self.assertTrue(P.bit_equal(got[k], want[k]), k)
        self.assertEqual(got['numeric_features'].shape, (1, 1, 23, 132));self.assertEqual(got['categorical_ids'].shape, (1, 1, 23, 41))
        self.assertEqual((got['token_type_ids'].shape, got['attention_mask'].shape), ((1, 1, 23), (1, 1, 23)))
        self.assertEqual(got['history_mask'].tolist(), [[True]]);self.assertEqual(got['legal_action_mask'].tolist(), [legal])
        self.assertTrue(P.bit_equal(got['numeric_features'][0, 0], B2['numeric_features']))
        self.assertTrue(np.array_equal(got['categorical_ids'][0, 0], B2['categorical_ids']))
        prefix = P.prefix_arrays({'root__' + k: v for k, v in ROOT.items()})
        self.assertTrue(P.bit_equal(to_numpy(P.actor_tensors(P.actor_observation(prefix, legal))['numeric_features'])[0, 0], ROOT['numeric_features'][0, 0, :23]))
        with self.assertRaisesRegex(ValueError, 'owning shape'):P.actor_observation(dict(B2, numeric_features=np.zeros((23, 131))), legal)

    def test_window_tensors_match_observation_window_to_torch(self):
        with patch.object(neural_policy, 'require_torch', return_value=FakeTorch):self.check(FakeTorch, np.asarray)

    @unittest.skipIf(TORCH is None, 'torch is not installed')
    def test_window_tensors_with_real_torch(self):
        self.check(TORCH, lambda t: t.numpy())

    @unittest.skipIf(TORCH is None, 'torch is not installed')
    def test_run_uses_the_actor_probabilities_and_greedy(self):
        runtime = object.__new__(P.ActorRuntime);logits = TORCH.tensor([[.3, .9, .9, -1., 2., 0., 0., 0., 5.]])
        policy = SimpleNamespace(device='cpu', sampling_temperature=1., family_gated_selection=False, reset=lambda: None,
            forward_fn=lambda tensors: SimpleNamespace(policy_logits=logits, value=TORCH.tensor([.25])))
        runtime.torch, runtime.policy = TORCH, policy;runtime.raw = SimpleNamespace(select_action=lambda o, rng: SimpleNamespace(action_index=1))
        legal = [True, True, True, True, False, False, False, False, False]
        out = runtime.run(B2, legal)
        self.assertEqual((out['greedy'], out['select_action'], out['select_agrees']), (1, 1, True))   # tie 1/2: the lower; 4 and 8 illegal
        self.assertEqual(out['probabilities'][4], 0.);self.assertAlmostEqual(sum(out['probabilities']), 1., places=6)


class TorchCommandTests(unittest.TestCase):
    def test_checkpoint_bytes_are_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt';path.write_bytes(b'not the champion')
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                P.main(['torch', '--capture', tmp, '--checkpoint', str(path), '--checkpoint-sha256', 'f' * 64, '--out', str(Path(tmp) / 'o.jsonl')])
            with self.assertRaisesRegex(SystemExit, 'not 0fd09592'):
                P.main(['torch', '--capture', tmp, '--checkpoint', str(path), '--checkpoint-sha256', P.CHAMPION_CHECKPOINT_SHA256, '--out', str(Path(tmp) / 'o.jsonl')])
            self.assertFalse((Path(tmp) / 'o.jsonl').exists())


# Report ---------------------------------------------------------------------------------------------------------------------------
def toy_capture(directory, decisions):
    """A capture directory: selection.json, index.jsonl and arrays/<ordinal>.npz for (decision_index, set, b2 shift, path3 logits,
    live greedy, path2 exact, cross-world cells) tuples of battle 1."""
    d = Path(directory);(d / 'arrays').mkdir(parents=True);arrays, entries = {}, []
    for i, name, shift, p3, live_greedy, exact, cells in decisions:
        b2 = b2_of(ROOT, shift=shift)
        for k, v in ROOT.items():arrays[f'd{i}__root__{k}'] = v
        for k, v in b2.items():arrays[f'd{i}__b2__{k}'] = v
        arrays[f'd{i}__root__legal'] = np.asarray(RT.LEGAL, bool)
        entries.append(dict(schema=P.INDEX_SCHEMA, ordinal=1, decision_index=i, set=name, selection=dict(live_greedy=live_greedy, server_top=3), captured=True,
            root_logits=list(RT.LOGITS), root_legal=list(RT.LEGAL), path2=dict(logits_equal=exact, value_equal=True, valid=True),
            path3=dict(logits=p3, value=0., valid=True, lossy_keys=[]), crossworld=dict(prefix_equal=not cells, full_equal=not cells, value_equal=not cells,
            cells=cells, reference_reproduces_root_inputs=True), parity=dict(action_equal=True, q_equal=True), errors=[], arrays='arrays/1.npz',
            root_logits_match_selection=True, live_greedy_matches_selection=True, b2_view=dict(equal=not shift)))
    np.savez_compressed(d / 'arrays' / '1.npz', **arrays)
    (d / 'index.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in entries))
    (d / 'selection.json').write_text(json.dumps(dict(selection_sha256='s', counts=dict(disagree=2), select_seed=1,
        decisions=[dict(ordinal=1, decision_index=x[0], set=x[1]) for x in decisions])))
    (d / 'capture-summary.json').write_text(json.dumps(dict(selected_in_shard=len(decisions), recorded=len(decisions), missing=[], errors=[])))
    return entries


def torch_row(i, name, p1, p4):
    path = lambda logits, greedy: dict(logits=logits, probabilities=[0.] * 9, greedy=greedy, value=0., select_action=greedy, select_agrees=True)
    return dict(ordinal=1, decision_index=i, set=name, path1=path(*p1), path4=path(*p4))


class ReportTests(unittest.TestCase):
    def test_quantiles_and_gaps(self):
        q = P.quantiles([float(x) for x in range(1, 101)] + [None, float('nan')])
        self.assertEqual((q['n'], q['nonfinite'], q['p50'], q['p95'], q['max']), (100, 2, 50., 95., 100.))
        self.assertEqual(P.quantiles([]), dict(n=0, nonfinite=0, p50=None, p95=None, max=None))
        self.assertEqual(P.max_gap([0., 1., 5.] + [0.] * 6, [0., 3., -5.] + [0.] * 6, [True, True, False] + [False] * 6), 2.)
        self.assertEqual(P.max_gap([float('nan')] * 9, [0.] * 9, [True] * 9), float('inf'));self.assertIsNone(P.max_gap(None, [0.] * 9, [True] * 9))

    def test_gate(self):
        entries = [dict(ordinal=1, decision_index=i, set='disagree', captured=True, path2=dict(logits_equal=True, value_equal=True)) for i in range(200)]
        decisions = [dict(live=0, logits={1: [0.] * 9}, action={1: 0 if i else 1}) for i in range(200)]
        g = P.gate(entries, decisions)
        self.assertEqual((g['verdict'], g['path1']['reproduced'], g['path1']['share']), ('HARNESS_EXACT', 199, .995))
        decisions[1]['action'][1] = 2;g = P.gate(entries, decisions)
        self.assertEqual((g['verdict'], [m['decision_index'] for m in g['path1']['misses']]), ('HARNESS_NOT_EXACT', [0, 1]))
        decisions[1]['action'][1] = 0;entries[5]['path2'] = dict(logits_equal=False, value_equal=True);g = P.gate(entries, decisions)
        self.assertEqual((g['verdict'], g['path1']['passed'], [m['decision_index'] for m in g['path2']['misses']]), ('HARNESS_NOT_EXACT', True, [5]))
        self.assertEqual(P.gate([], [])['verdict'], 'HARNESS_NOT_EXACT')   # nothing checked is not exact

    def test_tally_ranking(self):
        t = P.Tally()
        for i in range(4):t.add('disagree', ['a', 'b'] if i < 3 else ['b'])
        for i in range(4):t.add('control', ['b'] if i < 2 else ['c'])
        rows = t.ranked()
        self.assertEqual([r['feature'] for r in rows], ['a', 'b', 'c'])
        a, b, c = rows
        self.assertEqual((a['disagree'], a['control'], a['ratio'], a['control_zero']), (3, 0, None, True))
        self.assertEqual((b['disagree_share'], b['control_share'], b['ratio']), (1., .5, 2.));self.assertEqual(c['disagree'], 0)
        self.assertEqual(t.table(['b']), [dict(feature='b', disagree=4, control=2, disagree_share=1., control_share=.5)])

    def test_report_on_toy_data(self):
        move = lambda a: [5. if x == a else 0. for x in range(9)]
        decisions = [(0, 'disagree', .5, move(0), 0, True, [['numeric_features', 3, 7]]), (1, 'disagree', .5, move(3), 0, True, []),
            (2, 'control', 0., move(3), 3, True, []), (3, 'control', 5e-7, move(3), 3, True, [])]
        with tempfile.TemporaryDirectory() as tmp:
            toy_capture(Path(tmp) / 'cap', decisions);entries = P.capture_entries([Path(tmp) / 'cap'])
            rows = {(1, 0): torch_row(0, 'disagree', (move(0), 0), (move(3), 3)), (1, 1): torch_row(1, 'disagree', (move(0), 0), (move(0), 0)),
                (1, 2): torch_row(2, 'control', (move(3), 3), (move(3), 3)), (1, 3): torch_row(3, 'control', (move(3), 3), (move(3), 3))}
            s = P.report(entries, rows)
            torch_file = Path(tmp) / 't.jsonl';torch_file.write_text(''.join(json.dumps(r) + '\n' for r in rows.values()))
            self.assertEqual(P.main(['report', '--capture', str(Path(tmp) / 'cap'), '--torch', str(torch_file), '--out', str(Path(tmp) / 'rep')]), 0)
            written = json.loads((Path(tmp) / 'rep' / 'summary.json').read_text());md = (Path(tmp) / 'rep' / 'summary.md').read_text()
        self.assertEqual(s['gate']['verdict'], 'HARNESS_EXACT');self.assertEqual(written['gate']['verdict'], 'HARNESS_EXACT')
        self.assertIn('Gate: HARNESS_EXACT', md);self.assertIn('Cross-world', md)
        p = s['paths']
        self.assertEqual((p['1']['disagree']['agree_live'], p['2']['disagree']['agree_live'], p['3']['disagree']['agree_live'], p['4']['disagree']['agree_live']),
            (2, 0, 1, 1))
        self.assertEqual((p['2']['control']['agree_live'], p['2']['control']['agree_server'], p['1']['disagree']['agree_server']), (2, 2, 0))
        self.assertEqual(p['2']['disagree']['dlogit_vs_recorded']['max'], 0.)
        self.assertEqual(p['4']['disagree']['dlogit_vs_path1']['max'], 5.);self.assertEqual(p['4']['disagree']['dlogit_vs_path1']['p50'], 0.)
        a = s['attribution']
        self.assertEqual((a['disagree']['path3_agrees_live'], a['disagree']['path3_n'], a['disagree']['path4_disagrees_live']), (1, 2, 1))
        e = s['encoding_diff']
        self.assertEqual(e['compared'], 4);top = e['ranked_cells'][0]
        self.assertEqual((top['feature'], top['names'], top['side'], top['info'], top['disagree'], top['control']),
            (['opponent', 'numeric', 43], ['CANDIDATE_SET_COUNT'], 'opponent', 'opponent-unrevealed', 2, 1))
        self.assertEqual(next(r for r in e['tokens'] if r['feature'] == 8)['label'], 'opponent[1]')
        self.assertEqual(next(r for r in e['numeric_columns'] if r['feature'] == 43)['disagree_share'], 1.)
        self.assertEqual(e['float_noise'], {'control:differing': 1, 'control:max_numeric_diff_below_1e-6': 1, 'control:numeric_only_below_1e-6': 1,
            'disagree:differing': 2})
        tags = {tuple(r['feature']): (r['disagree'], r['control']) for r in e['by_tag']}
        self.assertEqual(tags['info', 'opponent-unrevealed'], (2, 1))
        c = s['crossworld']
        self.assertEqual(c['any_difference'][0], dict(feature='prefix_differs', disagree=1, control=0, disagree_share=.5, control_share=0.))
        self.assertEqual(c['ranked_cells'][0]['feature'], ['own', 'numeric', 7]);self.assertIn('pre = 3 +', c['turn1_input_check'])
        self.assertEqual(s['replay']['captured'], 4)

    def test_layout_names_every_column_and_checks_the_token_roles(self):
        tokens = P.token_layout()
        self.assertEqual([t['section'] for t in tokens], ['field'] + ['own'] * 6 + ['opponent'] * 6 + ['actions'] * 9 + ['stats'])
        self.assertEqual([t['expert_group'] for t in tokens], [2] + [0] * 12 + [1] * 9 + [2])
        self.assertNotIn(['unknown'], [r['names'] for r in P.numeric_layout()]);self.assertNotIn(['unknown'], [r['names'] for r in P.categorical_layout()])
        cells = P.feature_layout()
        self.assertEqual(len(cells), 5 * (132 + 41 + 2) + 1)
        self.assertEqual((cells['field', 'numeric', 0]['info'], cells['opponent', 'numeric', 0]['info']), ('none', 'opponent-revealed'))   # HP on the field token
        self.assertEqual(cells['own', 'numeric', 18]['names'], ['ACTUAL_HP']);self.assertEqual(cells['own', 'numeric', 18]['info'], 'own-private')
        self.assertEqual(cells['stats', 'numeric', 121]['side'], 'opponent');self.assertEqual(cells['stats', 'numeric', 121]['info'], 'public')


if __name__ == '__main__':
    unittest.main()
