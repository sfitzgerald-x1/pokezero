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

    def test_explicit_decisions_are_read_selected_and_labelled_by_the_same_rule(self):
        legal = [True, True, False, True] + [False] * 5;tie = [1., 2., 9., 2.] + [0.] * 5
        rows = [dict(ordinal=1, decision_index=i, battle_id='b', replay_status='COMPLETE', root_logits=tie, root_legal=legal, live_greedy=g, live_source='search',
            source='x') for i, g in enumerate([1, 3, 0, 1, 1])] + [dict(ordinal=1, decision_index=5, battle_id='b', replay_status='REFUSED', root_logits=tie,
            root_legal=legal, live_greedy=1, live_source='search', source='x')]
        with tempfile.TemporaryDirectory() as tmp:
            listed = Path(tmp) / 'pairs.json';listed.write_text(json.dumps([[1, 2], [1, 0], [1, 2]]))
            jsonl = Path(tmp) / 'rows.jsonl'   # re-pricer rows filtered by jq: extra keys ignored
            jsonl.write_text(''.join(json.dumps(dict(ordinal=1, decision_index=i, replay=dict(status='COMPLETE'))) + '\n' for i in (3, 1)))
            bad = Path(tmp) / 'bad.json';bad.write_text(json.dumps([[1, '2']]))
            self.assertEqual(P.read_decisions(listed), [(1, 2), (1, 0)]);self.assertEqual(P.read_decisions(jsonl), [(1, 3), (1, 1)])
            with self.assertRaisesRegex(ValueError, 'integers'):P.read_decisions(bad)
        chosen, counts = P.select_decisions(rows, [(1, 2), (1, 0), (1, 1)])
        self.assertEqual({k: v['set'] for k, v in chosen.items()}, {(1, 0): 'control', (1, 1): 'disagree', (1, 2): 'disagree'})
        self.assertEqual((counts['explicit'], counts['disagree'], counts['control']), (3, 2, 1));self.assertEqual(chosen[1, 1]['server_top'], 1)
        with self.assertRaisesRegex(ValueError, 'no re-pricer row'):P.select_decisions(rows, [(2, 0)])
        with self.assertRaisesRegex(ValueError, 'cannot be classified'):P.select_decisions(rows, [(1, 5)])

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
        self.assertEqual(record['errors'], []);self.assertEqual(record['path2'], dict(logits_equal=True, value_equal=True, valid=True, inputs='world0'))
        self.assertEqual((record['root_view_source'], record['world0']), (None, None))   # fake_build names no source: world 0's view
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

    def test_a_b2_sourced_prior_is_reproduced_on_the_spliced_inputs(self):
        """build's root prior read B2's encoding (root_view_source 'b2'): path 2 re-runs it on splice(root_inputs, B2), and world 0's own
        encoding through the same callable is recorded beside it."""
        spliced = fake_forward(None, P.splice(ROOT, B2)[0])

        def b2_build(*a, **k):
            out = fake_build(*a, **k)
            roots = tuple(dict(r, root_logits=tuple(P.floats(spliced.policy_logits)), root_value=float(spliced.value[0]), root_view_source='b2')
                if r['status'] == 'PRICED' else r for r in out.roots)
            return SimpleNamespace(requests=out.requests, roots=roots)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            capture_fakes(stack);stack.enter_context(patch.object(cw, 'build', b2_build))
            recorder = P.Recorder(selected((1001, 0)), tmp);search = RT.fake_search(B2Constructor())
            with P.capture_class(recorder)().installed():response = search.process_many([body(0)])[0]
            recorder.close()
        record = recorder.pending['rr-1001-0-7']
        self.assertEqual((response['status'], response['root_view_source'], record['errors'], record['root_view_source']), ('COMPLETE', 'b2', [], 'b2'))
        self.assertEqual(record['path2'], dict(logits_equal=True, value_equal=True, valid=True, inputs='b2_spliced'))
        self.assertEqual(record['world0']['logits'], list(priced_logits()));self.assertEqual(record['path3']['logits'], P.floats(spliced.policy_logits))

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
        self.assertEqual(first['path2'], dict(logits_equal=True, value_equal=True, valid=True, inputs='world0'))
        self.assertEqual((first['selection']['live_greedy'], first['live']['greedy_action_index'], first['live_greedy_matches_selection']), (0, 0, True))
        self.assertTrue(first['root_logits_match_selection'] and first['decide_root_logits_match'])
        self.assertEqual(first['parity']['replay_complete'], True);self.assertEqual(first['crossworld']['differing_worlds'], [5, 6])
        self.assertTrue(P.bit_equal(arrays['d0__root__numeric_features'], ROOT['numeric_features']))
        self.assertTrue(P.bit_equal(arrays['d0__b2__numeric_features'], B2['numeric_features']))
        self.assertEqual(P.floats(arrays['d0__path3__logits']), first['path3']['logits'])
        self.assertTrue(P.bit_equal(loaded['root__logits'], np.asarray(priced_logits(), np.float32)))
        self.assertEqual((summary['selected_in_shard'], summary['recorded'], summary['captured'], summary['path2_exact']), (2, 2, 2, 2))
        self.assertEqual((summary['missing'], summary['errors'], summary['interrupted'], summary['reprice_exit']), ([], [], False, 0))

    def test_capture_selects_explicit_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, rows = self.run_directory(tmp);out = Path(tmp) / 'cap';server = RT.BASE + RT.CHAMPION + ['--run-directory', str(root)]
            pairs = Path(tmp) / 'pairs.jsonl';pairs.write_text(json.dumps(dict(ordinal=1001, decision_index=3)) + '\n')
            with ExitStack() as stack:
                self.patches(stack, RT.fake_search(B2Constructor(refuse={4})))
                code = P.main(['capture', '--rr-rows', str(rows), '--select-decisions', str(pairs), '--out', str(out), *server])
            selection = json.loads((out / 'selection.json').read_text())
            index = [json.loads(x) for x in (out / 'index.jsonl').read_text().splitlines()]
        self.assertEqual(code, 0);self.assertEqual([(d['decision_index'], d['set']) for d in selection['decisions']], [(3, 'control')])
        self.assertEqual((selection['select_seed'], selection['select_decisions']['path']), (None, str(pairs)))
        self.assertEqual([(e['decision_index'], e['captured']) for e in index], [(3, True)])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):   # one selection only
            P.main(['capture', '--rr-rows', 'x', '--select-seed', '1', '--select-decisions', str(pairs), '--out', 'o'])

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


class FakeRuntime:
    """torch_main's runtime without a checkpoint: each path's outputs read its encoding's opponent token 8, column 43."""
    calls = []

    def __init__(self, checkpoint, digest):
        self.state, self.environment = dict(threads=1), dict(torch_config='fake config')

    def run(self, arrays, legal):
        FakeRuntime.calls.append(list(legal));x = float(np.asarray(arrays['numeric_features'])[8, 43])
        return dict(logits=[x] * 9, probabilities=[0.] * 9, greedy=0, value=x, select_action=0, select_agrees=True)


class TorchCommandTests(unittest.TestCase):
    def test_checkpoint_bytes_are_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt';path.write_bytes(b'not the champion')
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                P.main(['torch', '--capture', tmp, '--checkpoint', str(path), '--checkpoint-sha256', 'f' * 64, '--out', str(Path(tmp) / 'o.jsonl')])
            with self.assertRaisesRegex(SystemExit, 'not 0fd09592'):
                P.main(['torch', '--capture', tmp, '--checkpoint', str(path), '--checkpoint-sha256', P.CHAMPION_CHECKPOINT_SHA256, '--out', str(Path(tmp) / 'o.jsonl')])
            self.assertFalse((Path(tmp) / 'o.jsonl').exists())

    def test_paths_take_their_own_encodings_and_the_root_legal_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            toy_capture(Path(tmp) / 'cap', [(0, 'disagree', .5, [0.] * 9, 0, True, [])]);out = Path(tmp) / 't.jsonl';FakeRuntime.calls = []
            with patch.object(P, 'ActorRuntime', FakeRuntime), patch.object(P, 'file_sha256', return_value=P.CHAMPION_CHECKPOINT_SHA256), \
                    patch('sys.stdout', new_callable=io.StringIO):
                code = P.main(['torch', '--capture', str(Path(tmp) / 'cap'), '--checkpoint', 'ck.pt', '--checkpoint-sha256', P.CHAMPION_CHECKPOINT_SHA256,
                    '--out', str(out)])
            row = json.loads(out.read_text());header = json.loads(Path(str(out) + '.header.json').read_text())
        self.assertEqual(code, 0)
        self.assertEqual(row['path1']['value'], float(b2_of(ROOT)['numeric_features'][8, 43]))   # path 1: B2's encoding
        self.assertEqual(row['path4']['value'], float(ROOT['numeric_features'][0, 0, 8, 43]))   # path 4: the world-0 prefix
        self.assertNotEqual(row['path1']['value'], row['path4']['value'])
        self.assertEqual(row['legal'], list(RT.LEGAL));self.assertEqual(FakeRuntime.calls, [list(RT.LEGAL)] * 2)   # root_legal, not all nine
        self.assertTrue(row['b2_legal_equals_root_legal'])
        env = header['environment']
        self.assertEqual((env['torch_config'], env['src_tree']), ('fake config', P.git_tree('src')['src']))
        self.assertTrue(env['machine']);self.assertIn('numpy_blas', env);self.assertIn('processors', env['cpu'])


# Capture survivors ---------------------------------------------------------------------------------------------------------------
class CaptureEdgeTests(unittest.TestCase):
    def capture_one(self, *, build=fake_build, forward=fake_forward, views=fake_views):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            capture_fakes(stack)
            stack.enter_context(patch.object(cw, 'build', build))
            stack.enter_context(patch.object(P, 'root_forward_of', lambda pricer: forward));stack.enter_context(patch.object(P, 'views_of', lambda pricer: views))
            recorder = P.Recorder(selected((1001, 0)), tmp);search = RT.fake_search(B2Constructor())
            with P.capture_class(recorder)().installed():response = search.process_many([body(0)])[0]
            recorder.close()
        return response, recorder.pending['rr-1001-0-7']

    def test_path2_value_mismatch(self):
        def build(*a, **k):
            out = fake_build(*a, **k);out.roots = tuple(dict(r, root_value=float(np.float32(r['root_value']) + np.float32(1e-3))) for r in out.roots);return out
        response, record = self.capture_one(build=build)
        self.assertEqual(response['status'], 'COMPLETE');self.assertEqual(record['path2'], dict(logits_equal=True, value_equal=False, valid=True, inputs='world0'))
        self.assertTrue(record['recorded_float32_exact'])

    def test_recorded_prior_not_float32(self):
        def build(*a, **k):
            out = fake_build(*a, **k);out.roots = tuple(dict(r, root_logits=RT.LOGITS) for r in out.roots);return out   # -.2 is no float32
        _, record = self.capture_one(build=build)
        self.assertFalse(record['recorded_float32_exact']);self.assertFalse(record['path2']['logits_equal'])

    def test_reference_world_not_reproducing_root_inputs(self):
        def views(state, public, owner):
            _, _, own, _ = fake_views(state, public, owner);own['categorical_ids'][0, 0, 26, 1] += 1   # every world, beyond the prefix
            return None, None, own, None
        _, record = self.capture_one(views=views)
        cross = record['crossworld']
        self.assertFalse(cross['reference_reproduces_root_inputs']);self.assertEqual(cross['differing_worlds'], [5, 6])

    def test_invalid_path3(self):
        def picky(p, inputs):   # valid only on world 0's own value at opponent token 8, column 43
            out = fake_forward(p, inputs);x = np.asarray(inputs['numeric_features'])[:, 0, 8, 43]
            return SimpleNamespace(policy_logits=out.policy_logits, value=out.value, valid=x == ROOT['numeric_features'][0, 0, 8, 43])
        response, record = self.capture_one(forward=picky)
        self.assertEqual(response['status'], 'COMPLETE');self.assertFalse(record['path3']['valid']);self.assertEqual(record['errors'], ['path 3 forward not valid'])
        self.assertTrue(record['path2']['logits_equal'])

    def test_per_battle_flush(self):
        record = dict(live=dict(greedy_action_index=0), replay=dict(status='COMPLETE'), parity=dict(replay_complete=True), root=dict(root_logits=list(RT.LOGITS)))
        captured = lambda v: dict(errors=[], arrays={'root__x': np.full(3, v)}, build_status='PRICED', path2=dict(logits_equal=True, value_equal=True),
            root_logits=list(RT.LOGITS))
        with tempfile.TemporaryDirectory() as tmp:
            rec = P.Recorder(selected((1, 0), (2, 0), (2, 1)), tmp, log=io.StringIO())
            for o, i in ((1, 0), (2, 0), (2, 1)):rec.pending[f'rr-{o}-{i}-7'] = captured(10 * o + i)
            row = lambda o, i: (dict(ordinal=o), dict(decision_index=i, battle_id=f'b{o}'), dict(request_id=f'rr-{o}-{i}-7'), record)
            rec.after_row(*row(1, 0));rec.after_row(*row(1, 1))   # decision 1 is not selected
            index = Path(tmp) / 'index.jsonl'
            self.assertEqual(index.read_text(), '');self.assertFalse((Path(tmp) / 'arrays' / '1.npz').exists())   # battle 1 still open
            rec.after_row(*row(2, 0))
            self.assertEqual([(e['ordinal'], e['decision_index']) for e in map(json.loads, index.read_text().splitlines())], [(1, 0)])
            with np.load(Path(tmp) / 'arrays' / '1.npz') as z:self.assertEqual({k: z[k].tolist() for k in z.files}, {'d0__root__x': [10] * 3})
            rec.after_row(*row(2, 1));rec.close()
            entries = [json.loads(x) for x in index.read_text().splitlines()]
            with np.load(Path(tmp) / 'arrays' / '2.npz') as z:second = {k: z[k].tolist() for k in z.files}
        self.assertEqual([(e['ordinal'], e['decision_index'], e['arrays']) for e in entries], [(1, 0, 'arrays/1.npz'), (2, 0, 'arrays/2.npz'), (2, 1, 'arrays/2.npz')])
        self.assertEqual(second, {'d0__root__x': [20] * 3, 'd1__root__x': [21] * 3})
        self.assertEqual((rec.errors, rec.counts['recorded'], rec.counts['captured'], rec.counts['path2_exact']), ([], 3, 3, 3))

    def test_captures_of_different_selections_refuse_to_combine(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = toy_capture(Path(tmp) / 'a', [(0, 'disagree', .5, [0.] * 9, 0, True, [])], ordinal=1)
            toy_capture(Path(tmp) / 'b', [(0, 'control', 0., [0.] * 9, 3, True, [])], ordinal=2)
            toy_capture(Path(tmp) / 'c', [(1, 'control', 0., [0.] * 9, 3, True, [])], ordinal=3, selection_sha='t')
            self.assertEqual([(e['ordinal'], e['directory']) for e in P.capture_entries([Path(tmp) / 'b', Path(tmp) / 'a'])], [(1, str(Path(tmp) / 'a')), (2, str(Path(tmp) / 'b'))])
            with self.assertRaisesRegex(ValueError, 'different rows or seeds'):P.capture_entries([Path(tmp) / 'a', Path(tmp) / 'c'])
            with self.assertRaisesRegex(ValueError, 'two captures'):P.capture_entries([Path(tmp) / 'a', Path(tmp) / 'a'])
        self.assertEqual(len(one), 1)


# Report ---------------------------------------------------------------------------------------------------------------------------
def toy_capture(directory, decisions, *, ordinal=1, invalid_path3=(), edits=None, uncaptured=(), problems=None, selection_sha='s'):
    """A capture directory: selection.json, index.jsonl, arrays/<ordinal>.npz and replay/rows-0.jsonl for (decision_index, set, b2
    shift, path3 logits, live greedy, path2 exact, cross-world cells) tuples of one battle; `edits` {decision: [(token, column, delta)]}
    on B2's numerics, `uncaptured` (decision_index, set) selected but never recorded, `problems` {decision: live_problem}."""
    d = Path(directory);(d / 'arrays').mkdir(parents=True);(d / 'replay').mkdir();arrays, entries = {}, []
    for i, name, shift, p3, live_greedy, exact, cells in decisions:
        b2 = b2_of(ROOT, shift=shift)
        for t, c, delta in (edits or {}).get(i, ()):b2['numeric_features'][t, c] += np.float32(delta)
        for k, v in ROOT.items():arrays[f'd{i}__root__{k}'] = v
        for k, v in b2.items():arrays[f'd{i}__b2__{k}'] = v
        arrays[f'd{i}__root__legal'] = np.asarray(RT.LEGAL, bool)
        entries.append(dict(schema=P.INDEX_SCHEMA, ordinal=ordinal, decision_index=i, set=name, selection=dict(live_greedy=live_greedy, server_top=3), captured=True,
            root_logits=list(RT.LOGITS), root_legal=list(RT.LEGAL), path2=dict(logits_equal=exact, value_equal=True, valid=True),
            path3=dict(logits=p3, value=0., valid=i not in invalid_path3, lossy_keys=[]), crossworld=dict(prefix_equal=not cells, full_equal=not cells,
            value_equal=not cells, cells=cells, reference_reproduces_root_inputs=True), parity=dict(action_equal=True, q_equal=True), errors=[],
            arrays=f'arrays/{ordinal}.npz', root_logits_match_selection=True, live_greedy_matches_selection=True, b2_view=dict(equal=not shift)))
    np.savez_compressed(d / 'arrays' / f'{ordinal}.npz', **arrays)
    (d / 'index.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in entries))
    listing = [dict(ordinal=ordinal, decision_index=x[0], set=x[1], live_greedy=x[4], root_legal=list(RT.LEGAL)) for x in decisions]
    listing += [dict(ordinal=ordinal, decision_index=i, set=name, live_greedy=3, root_legal=list(RT.LEGAL)) for i, name in uncaptured]
    (d / 'selection.json').write_text(json.dumps(dict(selection_sha256=selection_sha, counts=dict(disagree=2), select_seed=1, decisions=listing)))
    (d / 'capture-summary.json').write_text(json.dumps(dict(selected_in_shard=len(listing), recorded=len(decisions), missing=[], errors=[])))
    (d / 'replay' / 'rows-0.jsonl').write_text(''.join(json.dumps(dict(ordinal=ordinal, decision_index=x['decision_index'],
        live_problem=(problems or {}).get(x['decision_index']))) + '\n' for x in listing))
    return entries


def torch_row(i, name, p1, p4, *, ordinal=1, agrees=True):
    path = lambda logits, greedy: dict(logits=logits, probabilities=[0.] * 9, greedy=greedy, value=0., select_action=greedy, select_agrees=agrees)
    return dict(ordinal=ordinal, decision_index=i, set=name, path1=path(*p1), path4=path(*p4), b2_legal_equals_root_legal=True)


def gate_entry(i, **changes):
    return dict(dict(ordinal=1, decision_index=i, set='disagree', captured=True, path2=dict(logits_equal=True, value_equal=True),
        root_logits_match_selection=True, live_greedy_matches_selection=True, selection=dict(live_greedy=0),
        parity=dict(live_searched=True, replay_complete=True, action_equal=True, q_equal=True)), **changes)


def gate_torch(greedy=0, agrees=True):
    return dict(path1=dict(greedy=greedy, select_agrees=agrees), path4=dict(greedy=greedy, select_agrees=agrees))


class ReportTests(unittest.TestCase):
    def test_quantiles_and_gaps(self):
        q = P.quantiles([float(x) for x in range(1, 101)] + [None, float('nan')])
        self.assertEqual((q['n'], q['nonfinite'], q['p50'], q['p95'], q['max']), (100, 2, 50., 95., 100.))
        self.assertEqual(P.quantiles([]), dict(n=0, nonfinite=0, p50=None, p95=None, max=None))
        self.assertEqual(P.max_gap([0., 1., 5.] + [0.] * 6, [0., 3., -5.] + [0.] * 6, [True, True, False] + [False] * 6), 2.)
        self.assertEqual(P.max_gap([float('nan')] * 9, [0.] * 9, [True] * 9), float('inf'));self.assertIsNone(P.max_gap(None, [0.] * 9, [True] * 9))

    def test_gate(self):
        entries = [gate_entry(i) for i in range(200)];rows = {(1, i): gate_torch(0 if i else 1) for i in range(200)}
        check = lambda **k: P.gate(entries, rows, **k)
        g = check()
        self.assertEqual((g['verdict'], g['failed'], g['path1']['reproduced'], g['path1']['share'], g['exact_on']), ('HARNESS_EXACT', [], 199, .995, dict(k=200, n=200)))
        rows[1, 1] = gate_torch(2);g = check()
        self.assertEqual((g['verdict'], g['failed'], [m['decision_index'] for m in g['path1']['misses']]), ('HARNESS_NOT_EXACT', ['path1'], [0, 1]))
        rows[1, 0] = rows[1, 1] = gate_torch()   # no path-1 miss from here on: each case fails its own check alone
        entries[5] = gate_entry(5, path2=dict(logits_equal=True, value_equal=False));g = check()   # the value alone
        self.assertEqual((g['failed'], [m['decision_index'] for m in g['path2']['misses']], g['exact_on']['k']), (['path2'], [5], 199))
        entries[5] = gate_entry(5, path2=dict(logits_equal=False, value_equal=True))
        self.assertEqual(check()['failed'], ['path2']);entries[5] = gate_entry(5)
        entries[6], entries[7] = gate_entry(6, root_logits_match_selection=False), gate_entry(7, live_greedy_matches_selection=False);g = check()
        self.assertEqual((g['failed'], [m['decision_index'] for m in g['selection']['misses']]), (['selection'], [6, 7]))
        entries[6], entries[7] = gate_entry(6), gate_entry(7)
        entries[8] = gate_entry(8, parity=dict(live_searched=True, replay_complete=True, action_equal=True, q_equal=False))
        self.assertEqual(check()['failed'], ['replay_parity'])
        entries[8] = gate_entry(8, parity=dict(live_searched=False, replay_complete=True, action_equal=None, q_equal=None))   # not searched live: no parity
        self.assertEqual((check()['verdict'], check()['replay_parity']['n']), ('HARNESS_EXACT', 199))
        rows[1, 9] = gate_torch(agrees=False);del rows[1, 10];g = check()
        self.assertEqual((g['failed'], [m['decision_index'] for m in g['torch']['misses']], g['path1']['n']), (['torch'], [9, 10], 199))
        rows[1, 9] = rows[1, 10] = gate_torch()
        entries[11] = dict(ordinal=1, decision_index=11, set='control', captured=False, path2=None);g = check()   # selected, never captured
        self.assertEqual((g['failed'], g['coverage']['missing'], g['path2']['misses'], g['coverage']['captured'], g['exact_on']), (['coverage'],
            [dict(ordinal=1, decision_index=11, set='control')], [], 199, dict(k=199, n=200)))
        entries[11] = gate_entry(11)
        g = check(selected={(1, i) for i in range(200)} | {(2, 0)})
        self.assertEqual((g['failed'], g['coverage']['missing'], g['coverage']['selected']), (['coverage'], [dict(ordinal=2, decision_index=0, set=None)], 201))
        self.assertEqual(check(selected={(1, i) for i in range(199)})['coverage']['unexpected'], [dict(ordinal=1, decision_index=199, set='disagree')])
        g = check(excluded={(1, 0): ['live_greedy_illegal'], (1, 1): ['live_problem: duplicated live telemetry rows']})
        self.assertEqual((g['verdict'], g['path1']['misses'], g['coverage']['excluded'], g['coverage']['analysed'], g['exact_on']['k']), ('HARNESS_EXACT', [], 2, 198, 198))
        self.assertEqual(g['excluded_reasons'], {'live_greedy_illegal': 1, 'live_problem': 1})
        self.assertEqual(P.gate([], {})['verdict'], 'HARNESS_NOT_EXACT')   # nothing checked is not exact

    def test_tally_ranking(self):
        t = P.Tally()
        for i in range(4):t.add('disagree', ['a', 'b'] if i < 3 else ['b'])
        for i in range(4):t.add('control', ['b'] if i < 2 else ['c'])
        rows = t.ranked()
        self.assertEqual([r['feature'] for r in rows], ['a', 'b', 'c'])
        a, b, c = rows
        self.assertEqual((a['disagree'], a['control'], a['ratio'], a['control_zero'], a['excess_share']), (3, 0, None, True, .75))
        self.assertEqual((b['disagree_share'], b['control_share'], b['ratio'], b['excess_share']), (1., .5, 2., .5))
        self.assertEqual((c['disagree'], c['excess_share']), (0, -.5))
        self.assertEqual(t.table(['b']), [dict(feature='b', disagree=4, control=2, disagree_share=1., control_share=.5)])
        u = P.Tally()   # x: rare but control-free (the larger smoothed ratio); y: common, twice the control share (the larger excess)
        for i in range(100):u.add('disagree', ['x'] * (i < 1) + ['y'] * (i < 60));u.add('control', ['y'] * (i < 30))
        ranked = u.ranked()
        self.assertEqual([r['feature'] for r in ranked], ['y', 'x']);self.assertGreater(ranked[1]['smoothed_ratio'], ranked[0]['smoothed_ratio'])

    def test_report_on_toy_data(self):
        move = lambda a: [5. if x == a else 0. for x in range(9)]
        decisions = [(0, 'disagree', .5, move(0), 0, True, [['numeric_features', 3, 7]]), (1, 'disagree', .5, move(3), 0, True, []),
            (2, 'control', 0., move(3), 3, True, []), (3, 'control', 5e-7, move(3), 3, True, [])]
        with tempfile.TemporaryDirectory() as tmp:
            toy_capture(Path(tmp) / 'cap', decisions, invalid_path3=(1,), edits={2: [(0, 0, 1.)]})   # field HP_FRACTION: a structural zero
            entries = P.capture_entries([Path(tmp) / 'cap'])
            rows = {(1, 0): torch_row(0, 'disagree', (move(0), 0), (move(3), 3)), (1, 1): torch_row(1, 'disagree', (move(0), 0), (move(0), 0)),
                (1, 2): torch_row(2, 'control', (move(3), 3), (move(3), 3)), (1, 3): torch_row(3, 'control', (move(3), 3), (move(3), 3))}
            s = P.report(entries, rows)
            torch_file = Path(tmp) / 't.jsonl';torch_file.write_text(''.join(json.dumps(r) + '\n' for r in rows.values()))
            Path(str(torch_file) + '.header.json').write_text(json.dumps(dict(precision=dict(threads=1), environment=dict(machine='aarch64',
                cpu={'CPU part': '0xd4f'}, src_tree='abc', torch_config='blas'))))
            with patch('sys.stdout', new_callable=io.StringIO):
                self.assertEqual(P.main(['report', '--capture', str(Path(tmp) / 'cap'), '--torch', str(torch_file), '--out', str(Path(tmp) / 'rep')]), 0)
            written = json.loads((Path(tmp) / 'rep' / 'summary.json').read_text());md = (Path(tmp) / 'rep' / 'summary.md').read_text()
        self.assertEqual(s['gate']['verdict'], 'HARNESS_EXACT');self.assertEqual(written['gate']['verdict'], 'HARNESS_EXACT')
        for text in ('Gate: HARNESS_EXACT', 'exact on 4/4', '## Coverage', 'selected 4, captured 4, missing 0', 'Cross-world', 'Anomalies',
                'not matched on request kind', 'request boundary', 'aarch64 0xd4f', 'src tree abc'):self.assertIn(text, md)
        self.assertEqual(written['torch_runtime'][0]['torch_config'], 'blas')
        p = s['paths']
        self.assertEqual((p['1']['disagree']['agree_live'], p['2']['disagree']['agree_live'], p['3']['disagree']['agree_live'], p['4']['disagree']['agree_live']),
            (2, 0, 1, 1))
        self.assertEqual((p['2']['control']['agree_live'], p['2']['control']['agree_server'], p['1']['disagree']['agree_server']), (2, 2, 0))
        self.assertEqual(p['2']['disagree']['dlogit_vs_recorded']['max'], 0.)
        self.assertEqual(p['4']['disagree']['dlogit_vs_path1']['max'], 5.);self.assertEqual(p['4']['disagree']['dlogit_vs_path1']['p50'], 0.)
        a = s['attribution']
        self.assertEqual((a['disagree']['path3_agrees_live'], a['disagree']['path3_n'], a['disagree']['path4_disagrees_live']), (1, 1, 1))   # an invalid path 3 has no action
        self.assertEqual(s['replay']['path3_invalid'], 1)
        e = s['encoding_diff']
        self.assertEqual(e['compared'], 4);top = e['ranked_cells'][0]
        self.assertEqual((top['feature'], top['names'], top['side'], top['info'], top['disagree'], top['control'], top['excess_share'], top['anomaly']),
            (['opponent', 'numeric', 43], ['CANDIDATE_SET_COUNT'], 'opponent', 'opponent-unrevealed', 2, 1, .5, False))
        self.assertEqual([(r['feature'], r['info'], r['control'], r['anomaly']) for r in e['anomalies']], [(['field', 'numeric', 0], 'none', 1, True)])
        self.assertEqual(next(r for r in e['tokens'] if r['feature'] == 8)['label'], 'opponent[1]')
        self.assertEqual(next(r for r in e['numeric_columns'] if r['feature'] == 43)['disagree_share'], 1.)
        self.assertEqual(e['float_noise'], {'control:differing': 2, 'control:max_numeric_diff_below_1e-6': 1, 'control:numeric_only_below_1e-6': 1,
            'disagree:differing': 2})
        tags = {tuple(r['feature']): (r['disagree'], r['control']) for r in e['by_tag']}
        self.assertEqual((tags['info', 'opponent-unrevealed'], tags['info', 'none']), ((2, 1), (0, 1)))
        c = s['crossworld']
        self.assertEqual(c['any_difference'][0], dict(feature='prefix_differs', disagree=1, control=0, disagree_share=.5, control_share=0.))
        self.assertEqual(c['ranked_cells'][0]['feature'], ['own', 'numeric', 7]);self.assertIn('pre = 3 +', c['turn1_input_check'])
        self.assertEqual((s['replay']['captured'], s['torch']['b2_legal_equals_root_legal'], s['torch']['select_agrees']['path1']), (4, {'True': 4}, {'True': 4}))

    def test_a_selection_with_an_empty_set_reports(self):
        # --select-decisions labels each root by the classification rule, so one set can be empty (every root agrees, or none does)
        move = lambda a: [5. if x == a else 0. for x in range(9)]
        for name, greedy in (('control', 3), ('disagree', 0)):
            decisions = [(0, name, .5, move(3), greedy, True, []), (1, name, .5, move(3), greedy, True, [['numeric_features', 3, 7]])]
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                toy_capture(Path(tmp) / 'cap', decisions);s = P.report(P.capture_entries([Path(tmp) / 'cap']), {})
                with patch('sys.stdout', new_callable=io.StringIO):
                    self.assertEqual(P.main(['report', '--capture', str(Path(tmp) / 'cap'), '--out', str(Path(tmp) / 'rep')]), 0)
                top = s['encoding_diff']['ranked_cells'][0];other = 'disagree' if name == 'control' else 'control'
                self.assertEqual((top['feature'], top[name], top[other], top[f'{other}_share'], top['excess_share'], top['ratio']),
                    (['opponent', 'numeric', 43], 2, 0, None, None, None))
                self.assertEqual(top[f'{name}_share'], 1.)

    def test_report_excludes_and_gates_coverage(self):
        move = lambda a: [5. if x == a else 0. for x in range(9)]
        decisions = [(0, 'disagree', .5, move(0), 0, True, []), (1, 'disagree', .5, move(0), 2, True, []),   # live greedy 2 is illegal
            (2, 'control', 0., move(3), 3, True, []), (3, 'control', 0., move(3), 3, True, [])]
        with tempfile.TemporaryDirectory() as tmp:
            toy_capture(Path(tmp) / 'cap', decisions, uncaptured=[(4, 'control')], problems={2: 'duplicated live telemetry rows'})
            rows = [torch_row(i, name, (move(0 if i < 2 else 3), 0 if i < 2 else 3), (move(3), 3)) for i, name, *_ in decisions]
            torch_file = Path(tmp) / 't.jsonl';torch_file.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with patch('sys.stdout', new_callable=io.StringIO):
                P.main(['report', '--capture', str(Path(tmp) / 'cap'), '--torch', str(torch_file), '--out', str(Path(tmp) / 'rep')])
            s = json.loads((Path(tmp) / 'rep' / 'summary.json').read_text());md = (Path(tmp) / 'rep' / 'summary.md').read_text()
        g = s['gate']
        self.assertEqual((g['verdict'], g['failed'], g['exact_on']), ('HARNESS_NOT_EXACT', ['coverage'], dict(k=2, n=5)))
        self.assertEqual({k: g['coverage'][k] for k in ('selected', 'captured', 'excluded', 'analysed')}, dict(selected=5, captured=4, excluded=2, analysed=2))
        self.assertEqual(g['coverage']['missing'], [dict(ordinal=1, decision_index=4, set=None)])
        self.assertEqual(g['excluded_reasons'], {'live_greedy_illegal': 1, 'live_problem': 1});self.assertEqual(g['path1']['n'], 2)
        self.assertEqual(s['excluded'], [dict(ordinal=1, decision_index=1, reasons=['live_greedy_illegal']),
            dict(ordinal=1, decision_index=2, reasons=['live_problem: duplicated live telemetry rows'])])
        self.assertEqual(s['paths']['1']['disagree']['n'] + s['paths']['1']['control']['n'], 2)   # the analysis leaves the excluded out
        self.assertEqual(s['coverage']['replay_rows_with_live_problem'], 1)
        for text in ('Gate: HARNESS_NOT_EXACT', 'exact on 2/5', 'failed: coverage', 'selected 5, captured 4, missing 1', 'excluded 2'):self.assertIn(text, md)

    def test_layout_names_every_column_and_checks_the_token_roles(self):
        tokens = P.token_layout()
        self.assertEqual([t['section'] for t in tokens], ['field'] + ['own'] * 6 + ['opponent'] * 6 + ['actions'] * 9 + ['stats'])
        self.assertEqual([t['expert_group'] for t in tokens], [2] + [0] * 12 + [1] * 9 + [2])
        self.assertNotIn(['unknown'], [r['names'] for r in P.numeric_layout()]);self.assertNotIn(['unknown'], [r['names'] for r in P.categorical_layout()])
        cells = P.feature_layout()
        self.assertEqual(len(cells), 5 * (132 + 41 + 2) + 1)
        self.assertEqual((cells['field', 'numeric', 0]['info'], cells['opponent', 'numeric', 0]['info']), ('none', 'opponent-revealed'))   # HP on the field token
        self.assertEqual(cells['own', 'numeric', 18]['names'], ['ACTUAL_HP']);self.assertEqual(cells['own', 'numeric', 18]['info'], 'own-private')
        for c in range(18, 24):self.assertEqual(cells['opponent', 'numeric', c]['info'], 'opponent-unrevealed', c)   # unavailable hidden actual stats
        self.assertEqual(cells['stats', 'numeric', 121]['side'], 'opponent');self.assertEqual(cells['stats', 'numeric', 121]['info'], 'public')


if __name__ == '__main__':
    unittest.main()
