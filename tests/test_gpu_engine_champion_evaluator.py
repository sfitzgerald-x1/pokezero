"""Champion 9375 through the search pricers' forward= seam (gpu_engine.champion_evaluator), on CPU with tiny champion weights.

- The adapter: exactly champion.forward on the independently sliced 23-token layout, value sign included; V2's validity rule on
  the champion's slice (belief tokens never read); the view, digests and seam identity; the loader's sha checks; the bank binding;
  the arm check bind_roots runs on the first root.
- End to end on the depth-N gate's fake engine with V2-layout encoder inputs: the real DepthNPricer, the full-width depth-2 pass,
  D3's FramePricer and the N-step oracle all price with ONE tiny champion and the gate passes, with the V2 forward patched to
  raise; N=2 equivalence against the merged Depth2Pricer; program keys per evaluator in one shared table, V2's unchanged.
- The depth-N gate's main() under the champion with --declared-bounds: the full-width depth-2 gate and --n2-equivalence PASS under a
  declaration for this champion, which every report records; a declaration for another evaluator identity is refused before loading.
- The tools' champion flags (parse; exclusive with --head-archive) and the forward microbenchmark at tiny shapes.
"""
import contextlib
import copy
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from pokezero.v5_model import TOKEN_ROLES
from gpu_engine import champion, champion_evaluator as ce, depth2_digest, device_public_teacher as serial, device_public_teacher_batch as batch
from gpu_engine import device_public_world_state as mapping
from tests import test_gpu_engine_depthn_oracle as N

V2_TOKENS = 29
F = 8   # the fake engine's numeric width (its five features, zero-padded)
CATEGORICAL = 3
SCALES = np.ones(V2_TOKENS, np.float32);SCALES[13:22] = 1 + np.arange(9, dtype=np.float32) / 8;SCALES[23:] = 1000.   # belief tokens: never read
CHECKPOINT = 'c' * 64


def tiny_config(**kw):
    return champion.Config(**dict(dict(numeric=F, categorical=CATEGORICAL, vocabulary=12, width=16, layers=1, heads=2, feedforward=32), **kw))


def tiny_params(c, seed=11, scale=.3):
    rng = np.random.default_rng(seed)
    return {k: j.asarray(rng.normal(0, scale, s).astype(np.float32)) for k, s in champion.shapes(c).items()}


def tiny_champion(precision='bfloat16', *, config=None, seed=11, archive='a' * 64):
    c = config or tiny_config();p = tiny_params(c, seed)
    meta = dict(schema='pokezero.gpu_champion_forward.v1', weights_archive_sha256=archive, checkpoint_sha256=CHECKPOINT,
        parameter_count=int(sum(np.asarray(x).size for x in p.values())))
    return ce.ChampionEvaluator(p, c, meta, precision)


def v2_inputs(c, n, seed=3):
    """V2 public29 encoder inputs [n,1,29,...], all valid for the champion; belief tokens 23..28 random (never read)."""
    rng = np.random.default_rng(seed)
    attention = rng.random((n, 1, V2_TOKENS)) > .25;attention[..., 0] = True
    return dict(numeric_features=rng.normal(0, 1, (n, 1, V2_TOKENS, c.numeric)).astype(np.float32),
        categorical_ids=rng.integers(0, c.vocabulary, (n, 1, V2_TOKENS, c.categorical)).astype(np.int32),
        token_type_ids=np.broadcast_to(np.asarray(TOKEN_ROLES, np.int32), (n, 1, V2_TOKENS)).copy(), attention_mask=attention,
        history_mask=np.ones((n, 1), bool))


def sliced(inputs):
    """The champion layout sliced by hand ([:, 0, :23]), independently of ChampionEvaluator.view."""
    return [np.asarray(inputs[k])[:, 0, :23] for k in ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask')]


def write_archive(path, c, params, checkpoint=CHECKPOINT):
    """champion.export's archive format, written from params (no Torch); returns its sha256."""
    meta = {'schema': 'pokezero.gpu_champion_forward.v1', 'config': asdict(c), 'checkpoint_sha256': checkpoint,
        'parameter_count': int(sum(np.asarray(a).size for a in params.values()))}
    encoded = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), np.uint8)
    with Path(path).open('xb') as stream:np.savez_compressed(stream, metadata_json=encoded, **{k: np.asarray(v) for k, v in params.items()})
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- the adapter

class ChampionEvaluatorTests(unittest.TestCase):
    def test_adapter_equals_champion_forward_on_the_sliced_inputs_sign_included(self):
        for precision in ('bfloat16', 'float32'):
            with self.subTest(precision=precision):
                ev = tiny_champion(precision);inputs = v2_inputs(ev.config, 11)
                # Centre the value head's bias on these rows so the values take both signs (tanh is odd: the sign is the raw score's).
                raw = np.arctanh(np.asarray(jax.jit(ev)(ev.params, inputs).value, np.float64))
                ev = replace(ev, params=dict(ev.params, **{'value_head.bias': ev.params['value_head.bias'] - np.float32(np.median(raw))}))
                out = jax.jit(ev)(ev.params, inputs)
                ref = jax.jit(lambda p, *x: champion.forward(p, *x, ev.config, compute_dtype=precision))(ev.params, *sliced(inputs))
                self.assertIsNone(out.value_logits);self.assertTrue(np.asarray(out.valid).all())
                self.assertEqual((out.policy_logits.shape, out.value.shape, out.policy_logits.dtype, out.value.dtype), ((11, 9), (11,), j.float32, j.float32))
                np.testing.assert_array_equal(np.asarray(out.policy_logits), np.asarray(ref.policy_logits))
                np.testing.assert_array_equal(np.asarray(out.value), np.asarray(ref.value))   # no sign flip, no rescale
                value = np.asarray(out.value);self.assertTrue((value > 0).any() and (value < 0).any() and (np.abs(value) < 1).all())
                np.testing.assert_array_equal(np.sign(value), np.sign(np.asarray(ref.value)))
                # Negating the value head negates the value: the observer-side tanh passes through linearly in sign.
                flipped = dict(ev.params, **{'value_head.weight': -ev.params['value_head.weight'], 'value_head.bias': -ev.params['value_head.bias']})
                np.testing.assert_allclose(np.asarray(jax.jit(ev)(flipped, inputs).value), -value, rtol=0, atol=1e-6)

    def test_belief_tokens_are_never_read(self):
        ev = tiny_champion('float32');inputs = v2_inputs(ev.config, 5);other = {k: np.array(v) for k, v in inputs.items()}
        other['numeric_features'][..., 23:, :] = np.nan;other['categorical_ids'][..., 23:, :] = 10 ** 6
        other['token_type_ids'][..., 23:] = 15;other['attention_mask'][..., 23:] = False
        a, b = jax.jit(ev)(ev.params, inputs), jax.jit(ev)(ev.params, other)
        for name in ('policy_logits', 'value', 'valid'):np.testing.assert_array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)), err_msg=name)

    def test_valid_is_the_v2_rule_on_the_champion_slice(self):
        ev = tiny_champion('float32');c = ev.config;inputs = {k: np.array(v) for k, v in v2_inputs(c, 9).items()}
        x = inputs
        x['numeric_features'][1, 0, 5, 2] = np.nan            # 1 non-finite numeric in a read token
        x['categorical_ids'][2, 0, 3, 1] = c.vocabulary       # 2 category out of range
        x['categorical_ids'][3, 0, 22, 0] = -1                # 3 negative category in the last read token
        x['token_type_ids'][4, 0, 10] = 6                     # 4 wrong token role
        x['history_mask'][5, 0] = False                       # 5 no real frame
        x['attention_mask'][6, 0, :23] = False                # 6 no attended token among the 23 (beliefs still attended)
        x['attention_mask'][7, 0, 23:] = False                # 7 beliefs unattended: V2 would refuse, the champion reads none of them
        x['numeric_features'][8, 0, 25, :] = np.inf           # 8 a non-finite belief token: never read
        out = jax.jit(ev)(ev.params, x);valid = np.asarray(out.valid)
        self.assertEqual(valid.tolist(), [True, False, False, False, False, False, False, True, True])
        self.assertTrue(np.isnan(np.asarray(out.value)[~valid]).all() and np.isnan(np.asarray(out.policy_logits)[~valid]).all())
        ref = champion.forward(ev.params, *[a[valid] for a in sliced(x)], c, compute_dtype='float32')   # a different batch: equal to rounding
        np.testing.assert_allclose(np.asarray(out.value)[valid], np.asarray(ref.value), rtol=0, atol=1e-5)

    def test_view_layout_and_digests(self):
        ev = tiny_champion();inputs = v2_inputs(ev.config, 2);view = ev.view(inputs)
        self.assertEqual(set(view), set(ce.INPUT_KEYS));self.assertEqual(ev.input_keys, depth2_digest.INPUT_KEYS)
        self.assertEqual({k: v.shape for k, v in view.items()}, dict(numeric_features=(2, 1, 23, F), categorical_ids=(2, 1, 23, CATEGORICAL),
            token_type_ids=(2, 1, 23), attention_mask=(2, 1, 23), history_mask=(2, 1)))
        for a, b in zip([view[k][:, 0] for k in ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask')], sliced(inputs)):
            np.testing.assert_array_equal(a, b)
        with self.assertRaisesRegex(ValueError, '29-token'):ev.view({k: (v[..., :23, :] if v.ndim == 4 else v[..., :23] if v.ndim == 3 else v) for k, v in inputs.items()})
        with self.assertRaisesRegex(ValueError, 'five V2'):ev.view({k: v for k, v in inputs.items() if k != 'history_mask'})
        leaf = {k: v[:1] for k, v in inputs.items()};belief = {k: np.array(v) for k, v in leaf.items()};belief['numeric_features'][..., 27, :] += 1
        read = {k: np.array(v) for k, v in leaf.items()};read['numeric_features'][..., 22, :] += 1
        champion_digest = lambda x: np.asarray(depth2_digest.digest(ce.digest_view(ev)(x)))
        np.testing.assert_array_equal(champion_digest(leaf), champion_digest(belief))   # what the champion does not read is not certified
        self.assertFalse(np.array_equal(champion_digest(leaf), champion_digest(read)))
        self.assertFalse(np.array_equal(np.asarray(depth2_digest.digest(leaf)), np.asarray(depth2_digest.digest(belief))))   # V2 reads beliefs
        self.assertFalse(np.array_equal(champion_digest(leaf), np.asarray(depth2_digest.digest(leaf))))

    def test_seam_identity_suffix_and_view(self):
        bf16, f32 = tiny_champion(), tiny_champion('float32');fn = lambda p, x: None
        self.assertEqual((ce.identity(None), ce.program_suffix(None), ce.digest_view(None)), (('v2',), (), None))
        self.assertEqual(ce.identity(bf16), ('champion', 'a' * 64, 'bfloat16'));self.assertEqual(ce.identity(f32)[2], 'float32')
        self.assertEqual(ce.program_suffix(bf16), (('evaluator', 'champion', 'a' * 64, 'bfloat16'),))
        self.assertNotEqual(ce.program_suffix(bf16), ce.program_suffix(f32))
        self.assertEqual(ce.program_suffix(tiny_champion(seed=12)), ce.program_suffix(bf16))   # the archive sha, not the object, is the identity
        self.assertEqual(ce.program_suffix(fn), (('evaluator', 'forward', fn),));self.assertIsNone(ce.digest_view(fn))
        self.assertEqual(ce.digest_view(bf16), bf16.view)
        self.assertEqual(ce.describe(bf16), dict(kind='champion', identity=['champion', 'a' * 64, 'bfloat16']))
        self.assertEqual(ce.describe(None), dict(kind='v2', identity=['v2']));self.assertEqual(ce.describe(fn)['kind'], 'custom')
        self.assertNotIn('params', repr(bf16))
        header = bf16.header()
        self.assertEqual((header['kind'], header['identity'], header['precision'], header['tokens']), ('champion', ['champion', 'a' * 64, 'bfloat16'], 'bfloat16', 23))
        with self.assertRaises(ValueError):tiny_champion('float16')
        with self.assertRaisesRegex(ValueError, '23-token'):ce.ChampionEvaluator({}, tiny_config(tokens=29), bf16.meta)
        with self.assertRaisesRegex(ValueError, 'sha256'):ce.ChampionEvaluator({}, tiny_config(), dict(bf16.meta, checkpoint_sha256='x'))

    def test_load_checks_the_archive_and_checkpoint_sha256(self):
        c = tiny_config();params = tiny_params(c)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'champion.npz';digest = write_archive(path, c, params)
            ev = ce.load(path, expected_archive_sha256=digest, expected_checkpoint_sha256=CHECKPOINT, precision='float32')
            self.assertEqual((ev.identity, ev.checkpoint_sha256, ev.config), (('champion', digest, 'float32'), CHECKPOINT, c))
            for name in params:np.testing.assert_array_equal(np.asarray(ev.params[name]), np.asarray(params[name]))
            with self.assertRaisesRegex(ValueError, 'archive sha256'):ce.load(path, expected_archive_sha256='0' * 64, expected_checkpoint_sha256=CHECKPOINT)
            with self.assertRaisesRegex(ValueError, 'checkpoint sha256'):ce.load(path, expected_archive_sha256=digest, expected_checkpoint_sha256='d' * 64)
            with self.assertRaisesRegex(ValueError, '64-hex'):ce.load(path, expected_archive_sha256=digest.upper(), expected_checkpoint_sha256=CHECKPOINT)
            with self.assertRaisesRegex(ValueError, 'precision'):ce.load(path, expected_archive_sha256=digest, expected_checkpoint_sha256=CHECKPOINT, precision='bf16')
            wide = tiny_config(tokens=29);other = Path(directory) / 'wide.npz';wide_digest = write_archive(other, wide, tiny_params(wide))
            with self.assertRaisesRegex(ValueError, '23-token'):ce.load(other, expected_archive_sha256=wide_digest, expected_checkpoint_sha256=CHECKPOINT)
            args = SimpleNamespace(champion_archive=str(path), champion_archive_sha256=digest, champion_checkpoint_sha256=CHECKPOINT, champion_precision='bf16')
            self.assertEqual(ce.load_from_arguments(args).identity, ('champion', digest, 'bfloat16'))
            self.assertIsNone(ce.load_from_arguments(SimpleNamespace(champion_archive=None)))

    def test_check_binding_against_the_bank_catalog(self):
        from pokezero import live_protocol_actor as actor
        contract = actor.ActorContract.champion()
        ev = ce.ChampionEvaluator({}, tiny_config(vocabulary=916), dict(weights_archive_sha256='a' * 64, checkpoint_sha256=contract.checkpoint_sha256))
        masks = SimpleNamespace(**dict(contract.feature_masks))
        catalog = lambda **kw: SimpleNamespace(**dict(dict(checkpoint_sha256=contract.checkpoint_sha256, masks=masks,
            vocabulary=SimpleNamespace(sha=contract.category_vocab_sha256, size=916)), **kw))
        with patch.object(actor, '_vocab_sha', lambda vocabulary: vocabulary.sha):
            self.assertTrue(ce.check_binding(ev, catalog()))
            with self.assertRaisesRegex(ValueError, 'real original 9375'):ce.check_binding(replace(ev, meta=dict(ev.meta, checkpoint_sha256='d' * 64)), catalog())
            with self.assertRaisesRegex(ValueError, 'compiled for checkpoint'):ce.check_binding(ev, catalog(checkpoint_sha256='d' * 64))
            with self.assertRaisesRegex(ValueError, 'category vocabulary'):ce.check_binding(ev, catalog(vocabulary=SimpleNamespace(sha='e' * 64, size=916)))
            with self.assertRaisesRegex(ValueError, 'category vocabulary'):ce.check_binding(ev, catalog(vocabulary=SimpleNamespace(sha=contract.category_vocab_sha256, size=915)))
            bad = SimpleNamespace(**dict(contract.feature_masks, stats_block=not dict(contract.feature_masks)['stats_block']))
            with self.assertRaisesRegex(ValueError, 'feature masks'):ce.check_binding(ev, catalog(masks=bad))

    def test_arm_check_on_the_first_root(self):
        ev = tiny_champion();root = {k: v[:1] for k, v in v2_inputs(ev.config, 1).items()}
        self.assertTrue(ev.check(ev.params, root))
        with self.assertRaisesRegex(ValueError, 'differs from champion.forward'):Shifted(ev.params, ev.config, ev.meta, ev.precision).check(ev.params, root)
        nan = dict(root, numeric_features=np.full_like(root['numeric_features'], np.nan))
        with self.assertRaisesRegex(ValueError, 'invalid or non-finite'):ev.check(ev.params, nan)

    def test_bind_roots_refuses_a_mis_slicing_or_sign_flipped_evaluator(self):
        """gate.bind_roots runs the evaluator's check on the first root: a wrong slice or a negated value fails the run before any
        root is bound (a bind that skipped the check would hand every pricer the faulty evaluator)."""
        from gpu_engine import d3_depth2_gate as g2
        ev = tiny_champion();requests = small_requests(ev.config)
        groups, values, dropped, checked = g2.bind_roots([requests], (ev.params,), None, evaluator=ev)
        self.assertTrue(checked);self.assertEqual((len(groups[0]), dropped), (len(requests), []))
        for faulty in (Shifted, Negated):
            with self.subTest(evaluator=faulty.__name__), self.assertRaisesRegex(ValueError, 'differs from champion.forward'):
                g2.bind_roots([requests], (ev.params,), None, evaluator=faulty(ev.params, ev.config, ev.meta, ev.precision))


class Shifted(ce.ChampionEvaluator):
    """A wrong slice: numeric and categorical tokens 1..23 instead of 0..22."""
    def view(self, inputs):
        v = super().view(inputs)
        return dict(v, numeric_features=inputs['numeric_features'][..., 1:24, :], categorical_ids=inputs['categorical_ids'][..., 1:24, :])


class Negated(ce.ChampionEvaluator):
    """A sign error: the champion's value from the other side."""
    def __call__(self, params, inputs):
        out = super().__call__(params, inputs);return out._replace(value=-out.value)


# ---------------------------------------------------------------- end to end on the depth-N gate's fake engine

def champion_layout(make):
    """The fake engine with V2-layout encoder inputs: its feature vector x (zero-padded to F) is token 0's numeric row, the action
    tokens 13..21 carry x * (1 + lane / 8), the six belief tokens 1000 x (read by V2, never by the champion); categories 0, the V2
    roles, every token attended, history True. Everything else is the fake engine's."""
    def build(**kw):
        base = make(**kw)
        def observe(state, public, owner, mask):
            leaf = base.observe(state, public, owner, mask);x = leaf.inputs['numeric_features'][:, :, 0, :]
            n = x.shape[0];row = j.pad(x, ((0, 0), (0, 0), (0, F - x.shape[-1])))
            return leaf._replace(inputs=dict(numeric_features=row[:, :, None, :] * j.asarray(SCALES)[None, None, :, None],
                categorical_ids=j.zeros((n, 1, V2_TOKENS, CATEGORICAL), j.int32), token_type_ids=j.broadcast_to(j.asarray(TOKEN_ROLES, j.int32), (n, 1, V2_TOKENS)),
                attention_mask=j.ones((n, 1, V2_TOKENS), bool), history_mask=j.ones((n, 1), bool)))
        ex = replace(base, observe=observe, joint=None)
        return replace(ex, joint=serial._make_joint(ex))
    return build


def no_v2(*args, **kwargs):
    raise AssertionError('the V2 forward ran under the champion evaluator')


class ExactChampion(ce.ChampionEvaluator):
    """The real adapter (view, validity rule, NaN poisoning, digest view, identity, arm check) around an EXACT stand-in network that
    reads only the champion's 23-token view: value = token 0's first feature (the fake engine's learner-relative value) + offset;
    logits = the actor's logits + 0.5 on lane (own_act + 2 peer_act + turn) mod 9 (node-dependent priors, as forward_n) + each action
    token's third feature / 64. Every quantity is dyadic, so every batch geometry computes the same bits and the gate's exact checks
    and value bound (2 x a zero depth-1 floor) apply unchanged."""
    def core(self, params, numeric, categorical, token_types, attention):
        x = numeric[:, 0];lane = j.mod(x[:, 2] + 2. * x[:, 3] + x[:, 4], 9.)
        logits = params['logits'] + .5 * (j.arange(9, dtype=j.float32)[None, :] == lane[:, None]) + numeric[:, 13:22, 2] / 64
        return champion.Output(logits.astype(j.float32), (x[:, 0] + params['offset']).astype(j.float32), j.zeros_like(logits))


def exact_champion(precision='bfloat16'):
    params = {'offset': j.asarray(.0625, j.float32), 'logits': j.asarray([.25, 0, .5, .125, .375, 0, .25, .5, .125], j.float32)}
    return ExactChampion(params, tiny_config(), dict(weights_archive_sha256='f' * 64, checkpoint_sha256=CHECKPOINT, parameter_count=10), precision)


class ChampionFixture(N.FixtureN):
    """The fixture's requests on V2-layout inputs (request 7's root inputs shifted: pre-refused), bound to one champion evaluator by
    gate.bind_roots (its prior, actors 0/0); every pricer, D3's FramePricer and the oracle take the same evaluator, and the V2
    forward is patched to raise."""
    make = staticmethod(champion_layout(N.executor_n))
    PRE_REFUSED = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        requests = list(cls.requests);r = requests[7];shifted = r.root_inputs['numeric_features'].at[..., 5, :].add(1.)   # token 5: the pre check sees it (code 3), the stand-in's value does not
        requests[7] = replace(r, root_inputs=dict(r.root_inputs, numeric_features=shifted))
        cls.v2_requests = tuple(requests);cls.champion, cls.bound = cls.bind(exact_champion())
        cls.params, cls.requests = (cls.champion.params,), cls.bound['requests']
        cls.oracle = cls.oracle_for(cls.champion)

    @classmethod
    def bind(cls, evaluator):
        from gpu_engine import d3_depth2_gate as g2
        with patch.object(batch.v5_model, 'forward', side_effect=no_v2):
            groups, values, dropped, checked = g2.bind_roots([list(cls.v2_requests)], (evaluator.params,), None, evaluator=evaluator)
        return evaluator, dict(requests=tuple(groups[0]), values=values, dropped=dropped, checked=checked)

    @classmethod
    def oracle_for(cls, evaluator):
        from gpu_engine import depthn_oracle
        return depthn_oracle.DepthNOracle(cls.ex, object(), chunk=64, forward=evaluator)

    @contextlib.contextmanager
    def engine(self, v2=no_v2):
        with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=v2):
            yield

    def module(self):
        from gpu_engine import device_depthn_pricer
        return device_depthn_pricer

    def frame_pricer(self, evaluator=None):
        from gpu_engine import d3_depth2_gate as g2
        return g2.with_forward(super().frame_pricer(), evaluator or self.champion)

    def champion_pricers(self, depth=3, k=2, selection='prior', lanes=(2187,), evaluator=None):
        from gpu_engine import d3_depthn_gate as gate
        return gate.DepthNPricers(self.module(), self.ex, object(), depth=depth, k_own=k, k_opp=k, selection=selection, capacity=32, actor_capacity=3,
            lanes=lanes, forward=evaluator or self.champion)

    def call(self, pricers, *, sample=None):
        from gpu_engine import d3_depthn_gate as gate
        depth2_pricer = gate.depth2_pass_pricer(pricers.module, self.ex, object(), capacity=32, actor_capacity=3, digest=True, forward=self.champion)
        with self.engine():
            return gate.run_call_n(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=self.params, pricers=pricers,
                depth2_pricer=depth2_pricer, frame_pricer=self.frame_pricer(), oracle=self.oracle, sample=sample, block=0, context=dict(block=0, call=0), backups=None)


@unittest.skipUnless(N.HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
class ChampionPricerTests(ChampionFixture, unittest.TestCase):
    """ChampionFixture's champion everywhere: the gate runs on the exact stand-in network; the real champion.forward (tiny, f32) is checked
    against the oracle on the learner's own leaf views."""
    def test_bind_roots_takes_the_champions_prior_and_actors(self):
        self.assertTrue(self.bound['checked']);self.assertEqual(self.bound['dropped'], [])
        self.assertEqual({(r.learner_actor, r.opponent_actor) for r in self.requests}, {(0, 0)})
        for r, before in zip(self.requests, self.v2_requests):
            out = jax.jit(self.champion)(self.champion.params, {k: np.asarray(v) for k, v in before.root_inputs.items()})
            self.assertEqual(r.root_logits, tuple(map(float, np.asarray(out.policy_logits, np.float32).reshape(-1))))
            self.assertEqual(r.root_inputs, before.root_inputs);self.assertNotEqual(r.root_logits, before.root_logits)
        self.assertEqual(len(self.bound['values']), len(self.requests));self.assertTrue(all(0 <= v < 1 for v in self.bound['values']))

    def test_depth3_gate_passes_with_the_champion_everywhere(self):
        for selection in ('prior', 'searched'):
            with self.subTest(selection=selection):
                pricers = self.champion_pricers(selection=selection, lanes=(8, 2187) if selection == 'prior' else (2187,))
                evaluation, summary = self.call(pricers)
                result, (status, reasons) = self.judged(evaluation)
                self.assertEqual(status, 'PASS', (N.failing(result), reasons, result['checks'].get('a', {}).get('first_mismatches')))
                a, c = result['checks']['a']['compared'], result['checks']['c']['quantities']
                for name in ('leaf_digest', 'ply2_digest_own', 'ply2_digest_opp', 't1_digest_own', 'pre'):self.assertGreater(a.get(name, 0), 0, name)
                for name in ('leaf_value', 'ply2_value_head', 't1_value_head'):self.assertGreater(c[name]['compared'], 0, name)
                self.assertEqual(evaluation['actors'], 1);self.assertEqual(pricers.options()['evaluator'], ce.describe(self.champion))
                self.assertGreater(summary['refusals']['depth1']['refused_count'], 0)
                if selection == 'prior':self.assertTrue(result['checks']['block_invariance']['discrete_equal'])

    def test_a_v2_digest_oracle_fails_the_champion_frame(self):
        """Negative control: the same oracle with V2 digests (the inputs unchanged, beliefs included) disagrees with every digest the
        champion pricer wrote, so (a) really compares the champion's own view."""
        from gpu_engine import d3_depthn_gate as gate
        oracle = self.oracle_for(self.champion);oracle.view = None   # read when its walks trace (first use): V2's digests, the champion's heads
        pricers = self.champion_pricers(depth=2, k=2)
        with self.engine():
            evaluation, _ = gate.run_call_n(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=self.params, pricers=pricers,
                depth2_pricer=None, frame_pricer=self.frame_pricer(), oracle=oracle, sample=None, block=0, context=dict(block=0, call=0), backups=None)
        mismatched = evaluation['a']['mismatched']
        self.assertGreater(mismatched.get('leaf_digest', 0), 0, mismatched);self.assertGreater(mismatched.get('t1_digest_own', 0), 0, mismatched)

    def test_real_champion_leaf_values_are_its_own_on_the_learners_view(self):
        """The real (tiny, f32) champion.forward under DepthNPricer: every head-valued leaf is the champion's value of the LEARNER's leaf
        view, unchanged (the oracle walks the same path and evaluates its learner view with the champion directly); the values take
        both signs, so a sign flip would show."""
        c = tiny_config();params = tiny_params(c, seed=13)
        params['numeric_projection.weight'] = params['numeric_projection.weight'] * .02   # the fake engine's clock feature is up to 78
        meta = dict(weights_archive_sha256='9' * 64, checkpoint_sha256=CHECKPOINT, parameter_count=0)
        real = ce.ChampionEvaluator(params, c, meta, 'float32')
        raw = []
        for r in self.v2_requests:raw.append(float(np.arctanh(np.asarray(jax.jit(real)(params, {k: np.asarray(v) for k, v in r.root_inputs.items()}).value)[0])))
        params['value_head.bias'] = params['value_head.bias'] - np.float32(np.median(raw))   # centre: values of both signs
        real, bound = self.bind(ce.ChampionEvaluator(params, c, meta, 'float32'));requests = bound['requests']
        pricers = self.champion_pricers(depth=2, k=2, evaluator=real);_, (pricer,) = pricers.get();oracle = self.oracle_for(real)
        with self.engine():
            frame = pricer.price(requests, self.tables, self.recipe, (params,))
            stack = N.reference.stack(requests, batch._numpy_worlds(requests, self.tables, self.recipe))
        self.assertTrue((frame.pre[7] == 3).all())   # request 7's shifted root inputs differ from the device view: pre code 3 (numeric_features)
        q, w, cell, i, x = np.nonzero(frame.leaf_status == 4);a1, r1 = cell // 9, cell % 9
        a2, r2 = frame.t1_own_actions[q, w, a1, r1, i], frame.t1_opp_actions[q, w, a1, r1, x]
        pick = np.random.default_rng(0).choice(len(q), size=min(64, len(q)), replace=False)
        specs = np.stack([q, w, a1, r1, a2, r2], axis=1)[pick];self.assertGreater(len(specs), 20)
        with self.engine():leaves = oracle.leaves_n(stack, specs, 2, (params,))
        self.assertTrue((leaves['status'] == 4).all())
        got = frame.leaf_value[q, w, cell, i, x][pick];np.testing.assert_allclose(got, leaves['head_value'], rtol=0, atol=1e-5)
        self.assertTrue((got > 0).any() and (got < 0).any() and (np.abs(got) < .999).all())
        np.testing.assert_array_equal(np.sign(got), np.sign(leaves['head_value']))
        np.testing.assert_array_equal(frame.leaf_digest[q, w, cell, i, x][pick].astype(np.uint32), leaves['digest'])

    def test_n2_equivalence_with_the_merged_depth2_pricer(self):
        from gpu_engine import d3_depthn_gate as gate
        from gpu_engine.device_depth2_pricer import Depth2Pricer
        module = self.module();configs = []
        for name, k in (('full_width', 9), ('top2', 2)):
            configs.append((name, module.DepthNPricer(self.ex, object(), capacity=32, depth=2, k_own=k, k_opp=k, digest=True, actor_capacity=3, forward=self.champion),
                Depth2Pricer(self.ex, object(), capacity=32, k_own=k, k_opp=k, digest=True, actor_capacity=3, forward=self.champion)))
        with self.engine():
            results = gate.n2_equivalence_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=self.params, configs=configs,
                frame_pricer=self.frame_pricer(), oracle=self.oracle, context=dict(block=0, call=0), module=module)
        report = gate.summarize_n2({name: [result] for name, result in results.items()})
        for name, r in report.items():
            self.assertTrue(r['passed'], (name, r['discrete']['mismatched'], {k: v['max_abs_difference'] for k, v in r['floats'].items()}, r['e'].get('reasons')))
            self.assertGreater(r['e']['roots_compared'], 0)

    def test_v2_defaults_run_bf16_on_all_29_tokens_and_digest_them(self):
        """The V2 fixtures' blind spot pinned: their fake forward ignores compute_dtype and their engine encodes one token. Here the
        defaults run on V2-layout inputs under a spy: FramePricer, Depth2Pricer, DepthNPricer and the oracle each call v5_model.forward
        with compute_dtype='bfloat16' on all 29 tokens, and every digest (both pricers and the oracle) hashes the five V2 inputs whole."""
        from gpu_engine import depth2_digest as dg
        from gpu_engine.device_depth2_pricer import Depth2Pricer
        module = self.module();requests = self.v2_requests[:8];forwards, hashed = [], [];real = dg.digest
        def spy(params, inputs, model=None, **kw):
            forwards.append((kw.get('compute_dtype'), {k: tuple(v.shape[2:]) for k, v in inputs.items()}));return N.forward_n(params, inputs)
        def digest(inputs):
            hashed.append({k: tuple(v.shape) for k, v in inputs.items()});return real(inputs)
        def run(price):
            forwards.clear();hashed.clear()
            with self.engine(spy), patch.object(dg, 'digest', digest):price()
            return list(forwards), list(hashed)
        oracle = N.reference.DepthNOracle(self.ex, object(), chunk=64)
        def oracle_walks():
            stack = N.reference.stack(requests, batch._numpy_worlds(requests, self.tables, self.recipe))
            roots = oracle.roots(stack, requests, N.PARAMS);cells = N.reference.turn1_cells(requests, self.recipe, roots)[:16]
            oracle.nodes(stack, cells, 1, N.PARAMS);oracle.leaves_n(stack, [tuple(c) + (0, 0) for c in cells], 2, N.PARAMS)
        runs = dict(
            frame_pricer=run(lambda: batch.FramePricer(self.ex, object(), capacity=8, block_rows=8, actor_capacity=3, vectorized=True,
                host_prep=True).price(requests, self.tables, self.recipe, N.PARAMS)),
            depth2_pricer=run(lambda: Depth2Pricer(self.ex, object(), capacity=8, actor_capacity=3, k_own=2, k_opp=2, digest=True).price(
                requests, self.tables, self.recipe, N.PARAMS)),
            depthn_pricer=run(lambda: module.DepthNPricer(self.ex, object(), capacity=8, depth=3, k_own=2, k_opp=2, digest=True, actor_capacity=3).price(
                requests, self.tables, self.recipe, N.PARAMS)),
            oracle=run(oracle_walks))
        whole = dict(numeric_features=(V2_TOKENS, F), categorical_ids=(V2_TOKENS, CATEGORICAL), token_type_ids=(V2_TOKENS,), attention_mask=(V2_TOKENS,),
            history_mask=())
        for name, (calls, digests) in runs.items():
            with self.subTest(path=name):
                self.assertTrue(calls);self.assertEqual({dtype for dtype, _ in calls}, {'bfloat16'})
                for _, shapes in calls:self.assertEqual({k: v[-len(whole[k]):] if whole[k] else () for k, v in shapes.items()}, whole)
                if name == 'frame_pricer':self.assertEqual(digests, []);continue   # D3's frame has no digest
                self.assertTrue(digests)
                for shapes in digests:
                    self.assertEqual(set(shapes), set(whole))
                    for key, tail in whole.items():self.assertEqual(shapes[key][len(shapes[key]) - len(tail):], tail, (key, shapes[key]))

    def test_program_keys_carry_the_evaluator_and_are_reused_within_one(self):
        """One shared program table (the latency tool's): the V2 default's keys carry no evaluator; a champion's carry its identity; a
        second pricer of the same evaluator compiles nothing; a second evaluator (f32) never reuses the bf16 programs. Depth2Pricer's
        and FramePricer's kernel keys follow the same rule."""
        from gpu_engine.device_depth2_pricer import Depth2Pricer
        module = self.module();programs = {};suffix = ce.program_suffix(self.champion)
        def priced(forward, params, requests, v2=no_v2):
            pricer = module.DepthNPricer(self.ex, object(), capacity=32, depth=3, k_own=2, k_opp=2, digest=False, actor_capacity=3, fixed_capacity=True,
                **({} if forward is None else dict(forward=forward)))
            pricer.bind(params);pricer.programs = programs
            with self.engine(v2):pricer.price(requests, self.tables, self.recipe, params)
            return pricer
        priced(None, N.PARAMS, self.v2_requests, v2=N.forward_n);v2_keys = set(programs)
        self.assertTrue(v2_keys and all(not any(isinstance(x, tuple) and x[:1] == ('evaluator',) for x in key) for key in v2_keys))
        self.assertIn(('P1', 32, (64, 64), (64, 64)), v2_keys)   # the pre-evaluator key (two actors), unchanged
        first = priced(self.champion, self.params, self.requests);champion_keys = set(programs) - v2_keys
        self.assertTrue(champion_keys and all(key[-1] == suffix[0] for key in champion_keys))
        self.assertIn(('P1', 32, (64,), (64,)) + suffix, champion_keys)
        self.assertEqual({key[:-1] for key in champion_keys if key[0] in ('expand', 'leaf')}, {key for key in v2_keys if key[0] in ('expand', 'leaf')})
        self.assertTrue(first.compiles and all(c[0][-1] == suffix[0] for c in first.compiles))
        again = priced(self.champion, self.params, self.requests);self.assertEqual(again.compiles, ())   # the same evaluator reuses every program
        f32 = exact_champion('float32');other = priced(f32, (f32.params,), self.bind(f32)[1]['requests'])
        self.assertTrue(other.compiles and all(c[0][-1] == ce.program_suffix(f32)[0] for c in other.compiles))
        self.assertEqual(len(programs), len(v2_keys) + 2 * len(champion_keys))
        kernels = {}
        for forward, params, requests, patched in ((None, N.PARAMS, self.v2_requests, N.forward_n), (self.champion, self.params, self.requests, no_v2)):
            for pricer in (Depth2Pricer(self.ex, object(), capacity=32, actor_capacity=3, **({} if forward is None else dict(forward=forward))),
                    batch.FramePricer(self.ex, object(), capacity=32, block_rows=32, actor_capacity=3, vectorized=True, host_prep=True,
                        **({} if forward is None else dict(forward=forward)))):
                with self.engine(patched):pricer.price(requests, self.tables, self.recipe, params)
                kernels[(type(pricer).__name__, forward is None)] = set(pricer.kernels)
        self.assertEqual(kernels[('FramePricer', True)], {(32, (64, 32), (64, 32))})   # D3's own key, unchanged
        for name in ('Depth2Pricer', 'FramePricer'):
            (plain,), (keyed,) = kernels[(name, True)], kernels[(name, False)]
            self.assertFalse(any(isinstance(x, tuple) and x[:1] == ('evaluator',) for x in plain))
            self.assertEqual(keyed[-1], suffix[0])


@unittest.skipUnless(N.HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
class ChampionCommandLineTests(ChampionFixture, unittest.TestCase):
    """The depth-N gate's main() under the champion (the roots loader replaced by the fixture's bound requests; the real pricers, D3's
    FramePricer and the oracle on the exact stand-in champion; the V2 forward patched to raise) with --declared-bounds for this champion:
    the full-width depth-2 gate and --n2-equivalence (DepthNPricer N=2 against the merged Depth2Pricer, both under the champion) PASS
    under the declared numbers, every report carrying the declaration; a declaration for another evaluator identity is refused before
    anything is created or loaded."""
    FLAGS = ('--champion-archive', '/shared/x/champion.npz', '--champion-archive-sha256', 'f' * 64, '--champion-checkpoint-sha256', CHECKPOINT)

    def roots(self):
        """What load_roots resolves for FLAGS: the bound requests, the champion as the one evaluator and its header."""
        provider = SimpleNamespace(executor=self.ex, materializer=self.tables, recipe=self.recipe, source=self.tables.source)
        loaded = SimpleNamespace(provider=provider, plan=SimpleNamespace(model=object(), historical_window=2), frame_pricer=N.FixtureN.frame_pricer(self),
            source=dict(fixture=True))
        return SimpleNamespace(loaded=loaded, head=None, params=self.params, groups=[list(self.requests)], root_values=list(self.bound['values']),
            evaluator=dict(override=True, **self.champion.header(), arm_checked=True, dropped_roots=[]), actor_capacity=1, forward=self.champion)

    def main(self, *argv, out=None, lines=None):
        out = out or Path(tempfile.mkdtemp(prefix='champion-cli-')) / 'out';log = (lambda m, **k: lines.append(dict(k, message=m))) if lines is not None else (lambda *a, **k: None)
        with self.engine(), patch.object(gate_module(), 'load_roots', lambda args: self.roots()), patch.object(gate_module(), 'log', log), \
                patch.object(gate_module(), 'REPO_ROOT', getattr(self, 'repo', gate_module().REPO_ROOT)):
            code = gate_module().main(['--out', str(out), '--d3-root', 'x', '--chunk', '0', '--capacity', '32', *self.FLAGS, *argv])
        return code, out

    def declared(self, *, bounds=None, **evaluator):
        """(content, flags, sha256) of a declaration for this champion (value bound 1e-6, turn-1 logits .05 unless `bounds`; its source measurement a
        readable file), committed under DECLARED_SUBDIR of a scratch repository that main() and --reverdict take as the checkout (self.repo, self.record)."""
        measurement = Path(tempfile.mkdtemp(prefix='champion-measure-')) / 'spread.json';measurement.write_bytes(b'{"bf16_vs_f32_value_spread": 0.0005}\n')
        content = N.declaration(evaluator=dict(dict(kind='champion', archive_sha256='f' * 64, precision='bfloat16', checkpoint_sha256=CHECKPOINT), **evaluator),
            bounds=bounds or N.bounds(value_bound=1e-6, turn1_logit_bound=.05),
            source_measurement=dict(path=str(measurement), sha256=hashlib.sha256(measurement.read_bytes()).hexdigest()))
        self.record = record = N.committed(content);self.repo = record.root
        return content, ('--declared-bounds', str(record.path), '--declared-bounds-sha256', record.sha256, '--declared-bounds-commit', record.commit), record.sha256

    def reverdicted(self, report, *extra):
        """(exit code or the SystemExit text, gate-reverdict.json or None) of --reverdict on `report` written to a new directory, the scratch
        repository as the checkout."""
        gate = gate_module();directory = Path(tempfile.mkdtemp(prefix='champion-declared-'));(directory / 'gate.json').write_text(json.dumps(report))
        with patch.object(gate, 'log', lambda *a, **k: None), patch.object(gate, 'REPO_ROOT', self.repo):
            try:code = gate.main(['--reverdict', str(directory / 'gate.json'), *extra])
            except SystemExit as raised:return str(raised.code), None
        return code, json.loads((directory / gate.REVERDICT_FILE).read_text())

    def test_depth2_full_width_and_n2_equivalence_under_a_declaration(self):
        content, flags, sha = self.declared()
        code, out = self.main('--depth', '2', '--k-own', '9', '--k-opp', '9', *flags);report = json.loads((out / 'gate.json').read_text())
        self.assertEqual((code, report['status']), (0, 'PASS'), report.get('status_reasons'));self.assertEqual(report['evaluator']['kind'], 'champion')
        self.assertEqual((report['header']['declared_bounds']['sha256'], report['header']['declared_bounds']['content']), (sha, content))
        self.assertEqual((report['tolerance']['value_bound'], report['tolerance']['logit_bound']), (1e-6, .05))
        self.assertEqual(set(report['checks']), {'a', 'b', 'c', 'd', 'oracle_consistency', 'e', 'f'});self.assertGreater(report['checks']['f']['comparisons'], 0)
        c = report['checks']['c']['quantities']
        for name in ('leaf_value', 't1_value_head', 't1_own_logits', 'depth2_pass_t2_value'):self.assertGreater(c[name]['compared'], 0, name)
        self.assertTrue(all(m['bound_rule'].startswith('declared_') for n, m in c.items() if n not in gate_module().g2.FLOORS))
        code, out = self.main('--n2-equivalence', '--n2-k', '2', *flags);report = json.loads((out / 'n2-equivalence.json').read_text())
        self.assertEqual((code, report['status']), (0, 'PASS'), {n: c.get('status_reasons') for n, c in report['configs'].items()})
        self.assertEqual((report['header']['declared_bounds']['sha256'], set(report['configs']), report['status_reasons']), (sha, {'full_width', 'top2'}, []))
        for config in report['configs'].values():
            self.assertEqual((config['tolerance']['value_bound'], config['tolerance']['logit_bound'], config['tolerance']['q_noise_bound']), (1e-6, .05, 1e-6))
            self.assertGreater(config['e']['roots_compared'], 0);self.assertGreater(config['discrete']['compared']['t2_status'], 0)

    def d2_call(self):
        """One depth-2 full-width call on the champion path's oracle consistency (run_call_n d2_consistency=True)."""
        gate = gate_module();pricers = self.champion_pricers(depth=2, k=9)
        depth2_pricer = gate.depth2_pass_pricer(pricers.module, self.ex, object(), capacity=32, actor_capacity=3, digest=True, forward=self.champion)
        with self.engine():
            evaluation, _ = gate.run_call_n(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=self.params, pricers=pricers,
                depth2_pricer=depth2_pricer, frame_pricer=self.frame_pricer(), oracle=self.oracle, sample=None, block=0, context=dict(block=0, call=0),
                d2_consistency=True)
        return evaluation

    def test_depth2_oracle_consistency_passes_clean_and_catches_corrupted_digests(self):
        """D2_ORACLE_CONSISTENCY: clean, every leaf_child_* and leaf_walk_* field compared and equal; the depth-2 leaf walk returning one leaf's
        parent own-view digest and another leaf's own digest one bit off FAILs exactly those two fields, and only the oracle-consistency
        check (the pricer still equals the depth-N oracle)."""
        evaluation = self.d2_call();result, (status, _) = self.judged(evaluation);compared = evaluation['oracle']['compared']
        self.assertEqual(status, 'PASS', N.failing(result));self.assertTrue(evaluation['oracle']['passed'])
        for name in ('leaf_path', 'leaf_child_known', 'leaf_child_expanded', 'leaf_child_invalid', 'leaf_child_done', 'leaf_child_own_digest', 'leaf_child_peer_digest',
                'leaf_walk_status', 'leaf_walk_error', 'leaf_walk_digest'):
            self.assertGreater(compared.get(name, 0), 0, name)
        self.assertEqual(compared['leaf_child_known'], compared['leaf_path'])   # every checked leaf, re-walked
        real = self.oracle.leaves
        def corrupted(stack, specs, params):
            out = dict(real(stack, specs, params));i = int(np.flatnonzero(np.asarray(out['status']) == 4)[0])
            out['child_own_digest'] = np.array(out['child_own_digest'], copy=True);out['child_own_digest'][0, 0] ^= np.uint32(1)
            out['digest'] = np.array(out['digest'], copy=True);out['digest'][i, 1] ^= np.uint32(1);return out
        with patch.object(self.oracle, 'leaves', corrupted):evaluation = self.d2_call()
        result, (status, _) = self.judged(evaluation);mismatched = {k: v for k, v in evaluation['oracle']['mismatched'].items() if v}
        self.assertEqual((status, set(N.failing(result))), ('FAIL', {'oracle_consistency'}));self.assertTrue(evaluation['a']['passed'])
        self.assertEqual(mismatched, dict(leaf_child_own_digest=1, leaf_walk_digest=1))

    def test_depth2_gate_and_n2_headers_record_the_merged_gate_parity(self):
        """main(): the champion depth-2 gate gates the oracle consistency and says so in merged_depth2_parity with every remaining
        difference; --n2-equivalence without --block-invariance says child-block invariance did not run."""
        code, out = self.main('--depth', '2', '--k-own', '9', '--k-opp', '9');report = json.loads((out / 'gate.json').read_text())
        self.assertEqual((code, report['status']), (0, 'PASS'), report.get('status_reasons'))
        parity = report['header']['merged_depth2_parity'];oracle = report['checks']['oracle_consistency']['compared']
        self.assertEqual((parity['mode'], parity['champion_path_additions']['d2_oracle_consistency'][:3]), ('gate', 'ran'))
        self.assertEqual([d['name'] for d in parity['remaining_differences']], [d['name'] for d in gate_module().MERGED_DEPTH2_DIFFERENCES])
        self.assertGreater(oracle['leaf_child_own_digest'], 0);self.assertGreater(oracle['leaf_walk_digest'], 0)
        code, out = self.main('--n2-equivalence', '--n2-k', '2');report = json.loads((out / 'n2-equivalence.json').read_text())
        self.assertEqual((code, report['status']), (0, 'PASS'));parity = report['header']['merged_depth2_parity']
        self.assertEqual((parity['mode'], parity['champion_path_additions']['child_block_invariance'][:7]), ('n2_equivalence', 'not run'))
        self.assertEqual(parity['champion_path_additions']['merged_depth2_decision'][:3], 'ran')
        self.assertFalse(any('block_invariance' in c for c in report['configs'].values()))
        for name, config in report['configs'].items():   # MERGED_DEPTH2_DECISION: depth2_backups on the merged Depth2Pricer against the oracle heads
            merged = config['e_merged_depth2'];self.assertEqual((merged['passed'], merged['rule']), (True, gate_module().MERGED_DEPTH2_DECISION), name)
            self.assertGreater(merged['roots_compared'], 0);self.assertGreater(merged['replaced']['t2_value'], 0);self.assertGreater(merged['cells_compared'], 0)
            self.assertEqual(merged['replaced']['unmatched_head_leaves'], 0);self.assertEqual(merged['q_noise_bound'], config['tolerance']['q_noise_bound'])
        code, out = self.main('--depth', '3', '--k-own', '2', '--k-opp', '2');report = json.loads((out / 'gate.json').read_text())
        self.assertEqual(code, 0);self.assertNotIn('merged_depth2_parity', report['header']);self.assertNotIn('leaf_child_known', report['checks']['oracle_consistency']['compared'])

    def test_the_merged_depth2_decision_catches_oracle_heads_the_merged_pricer_does_not_have(self):
        """MERGED_DEPTH2_DECISION fails closed on the merged path alone: the depth-2 oracle's leaf heads .25 higher than the merged Depth2Pricer's
        (the leaves only this check walks) FAIL each config's e_merged_depth2 on max |dQ|, while the pricer-against-pricer checks pass."""
        from gpu_engine import depth2_oracle
        real = depth2_oracle.Oracle.leaves
        def shifted(oracle, stack, specs, params):
            out = dict(real(oracle, stack, specs, params));out['head_value'] = np.where(np.asarray(out['status']) == 4, np.asarray(out['head_value']) + np.float32(.25), out['head_value'])
            return out
        with patch.object(depth2_oracle.Oracle, 'leaves', shifted):code, out = self.main('--n2-equivalence', '--n2-k', '2')
        report = json.loads((out / 'n2-equivalence.json').read_text());self.assertEqual((code, report['status']), (1, 'FAIL'))
        for name, config in report['configs'].items():
            merged = config['e_merged_depth2'];self.assertFalse(merged['passed'], name);self.assertFalse(merged['q_noise_within_bound'], name)
            self.assertTrue(config['discrete']['passed'] and config['e']['passed'] and all(m['passed'] for m in config['floats'].values()), name)

    def test_n2_child_block_invariance_passes_clean_and_catches_a_child_block_dependent_pricer(self):
        """--n2-equivalence --block-invariance: each config's Depth2Pricer at child_block 27 and 81, gated per config; a Depth2Pricer stand-in
        whose child_block-27 frames carry every head-valued t2_value .25 higher FAILs each config on block invariance alone."""
        g2 = gate_module().g2;code, out = self.main('--n2-equivalence', '--n2-k', '2', '--block-invariance')
        report = json.loads((out / 'n2-equivalence.json').read_text())
        self.assertEqual((code, report['status']), (0, 'PASS'), {n: c.get('block_invariance', {}).get('differing') for n, c in report['configs'].items()})
        self.assertEqual(report['header']['merged_depth2_parity']['champion_path_additions']['child_block_invariance'][:3], 'ran')
        for name, config in report['configs'].items():
            inv = config['block_invariance'];self.assertEqual((inv['passed'], inv['discrete_equal'], inv['child_blocks'], inv['calls']), (True, True, [27, 81], 1), name)
            self.assertGreater(inv['arrays_compared'], 10);self.assertGreater(inv['floats']['t2_value']['compared'], 0)
            self.assertEqual(report['options'][name]['child_blocks'], [27, 81]);self.assertEqual(report['options'][name]['depth2_options']['child_block'], 81)
        real = g2.make_pricer
        def make(*args, **kw):
            pricer, options = real(*args, **kw);return ChildBlockDependent(pricer), options
        with patch.object(g2, 'make_pricer', make):code, out = self.main('--n2-equivalence', '--n2-k', '2', '--block-invariance')
        report = json.loads((out / 'n2-equivalence.json').read_text());self.assertEqual((code, report['status']), (1, 'FAIL'))
        for name, config in report['configs'].items():
            inv = config['block_invariance'];self.assertEqual((inv['passed'], inv['discrete_equal'], inv['floats']['t2_value']['within_bound']), (False, True, False), name)
            self.assertTrue(config['discrete']['passed'] and config['e']['passed'] and all(m['passed'] for m in config['floats'].values()), name)   # the compared frame is clean

    def test_header_mutation_test_and_reverdict_round_trip(self):
        """Depth 3 under a declaration (value bound 1e-6: the exact champion's own floor is 0): gate.json PASS with the header's declared_bounds
        section beside the unchanged rules, the declared numbers in the tolerance and the measured floor beside them; mutation-test.json at the
        declared bound with the same header, every mutant killed; --mutation-from refuses the saved calls; --reverdict reproduces the verdict
        and the header, and refuses a section that fails its checks."""
        gate = gate_module();content, flags, sha = self.declared();lines = []
        code, out = self.main('--depth', '3', '--k-own', '2', '--k-opp', '2', '--mutation-test', '--save-oracle', *flags, lines=lines)
        report, mutation = (json.loads((out / name).read_text()) for name in ('gate.json', gate.MUTATION_FILE))
        self.assertEqual((code, report['status'], mutation['status']), (0, 'PASS', 'PASS'), (report.get('status_reasons'), mutation.get('survivors')))
        header = report['header'];section = header['declared_bounds'];default = gate.header_block(report['checks'])
        self.assertEqual(list(header), ['near_ties', 'rules_changed_after_failure', 'frozen_rule_set', 'declared_bounds'])
        self.assertEqual({k: header[k] for k in default}, default)   # the rules changed after a failure, untouched
        self.assertEqual((section['sha256'], section['content'], section['content_text'], section['note']), (sha, content, Path(flags[1]).read_text(), gate.DECLARED_NOTE))
        self.assertEqual(section['evaluator'], dict(kind='champion', archive_sha256='f' * 64, precision='bfloat16', checkpoint_sha256=CHECKPOINT))
        self.assertEqual((section['repo_path'], section['commit'], section['git_blob_sha1'], section['git_verification']['status']),
            (gate.DECLARED_SUBDIR + '/champion.json', self.record.commit, N.git(self.repo, 'hash-object', flags[1]), 'verified'))
        self.assertEqual((section['source_measurement_verification']['status'], report['status_reasons']), ('verified', []))   # nothing provisional
        tolerance = report['tolerance'];self.assertEqual((tolerance['value_bound'], tolerance['logit_bound'], tolerance['depth1_noise_floor']), (1e-6, .05, 0.))
        self.assertEqual(tolerance['declared_bounds']['frozen_rule_would_give']['value_bound'], 0.);self.assertEqual(tolerance['declared_bounds']['sha256'], sha)
        self.assertEqual((mutation['header'], mutation['value_bound'], mutation['calls'][0]['value_bound']), (header, 1e-6, 1e-6))
        self.assertEqual(mutation['calls'][0]['value_bound_source'], "the run's declared value_bound (--declared-bounds)")
        self.assertTrue(all(m['killed'] for m in mutation['calls'][0]['mutants'].values()))
        start = next(x for x in lines if x['message'] == 'GATE_START');self.assertEqual(start['declared_bounds_sha256'], sha)
        self.assertEqual(next(x for x in lines if x['message'] == 'DECLARED_BOUNDS')['sha256'], sha)
        with patch.object(gate, 'log', lambda *a, **k: None), self.assertRaises(SystemExit) as raised:
            gate.main(['--out', str(Path(tempfile.mkdtemp()) / 'o'), '--mutation-from', str(out)])
        self.assertIn('--declared-bounds run', str(raised.exception.code))
        code, again = self.reverdicted(report, '--declared-bounds-commit', self.record.commit, '--declared-bounds-sha256', sha)
        self.assertEqual((code, again['status'], again['header'], again['reverdict']['declared_bounds_sha256'], again['reverdict']['changed_quantities']), (0, 'PASS', header, sha, []))
        self.assertEqual((again['reverdict']['declared_bounds_commit'], again['reverdict']['declared_bounds_reverified']['git']['status']), (self.record.commit, 'verified'))
        verdicts = lambda r: {k: (m.get('passed'), m.get('bound'), m.get('bound_rule')) for k, m in r['checks']['c']['quantities'].items()}
        self.assertEqual(verdicts(again), verdicts(report));self.assertEqual(again['tolerance']['declared_bounds'], tolerance['declared_bounds'])
        for case, (change, extra, message) in self.tampered(report).items():
            with self.subTest(case):
                r = copy.deepcopy(report);change(r);text, out = self.reverdicted(r, *extra)
                self.assertIsNone(out);self.assertIn(message, text);self.assertIn('fail closed', text)

    def tampered(self, report):
        """{case: (edit of the declared gate.json, extra --reverdict arguments, the refusal it must meet)}: the recorded text, sha256, content,
        blob, repo path and bounds edited; a section or every section removed; another evaluator; a later run start; a consistent forgery (new
        text, its sha256 and blob, matching tolerance numbers) that the committed record contradicts; a commit or sha256 given that is not the
        recorded one."""
        gate = gate_module();h = lambda r: r['header']['declared_bounds']
        def forged(r):
            content = copy.deepcopy(h(r)['content']);content['bounds'].update(value_bound=.5, turn1_logit_bound=.9);text = json.dumps(content, indent=1)
            h(r).update(content=content, content_text=text, sha256=hashlib.sha256(text.encode()).hexdigest(), git_blob_sha1=gate.git_blob_sha1(text.encode()))
            r['tolerance'].update(value_bound=.5, logit_bound=.9);r['tolerance']['declared_bounds']['sha256'] = h(r)['sha256']
        def stripped(r):
            r['header'].pop('declared_bounds');r['tolerance'].pop('declared_bounds')
        return dict(
            text=(lambda r: h(r).__setitem__('content_text', h(r)['content_text'].replace('1e-06', '0.5')), (), 'does not hash'),
            sha=(lambda r: h(r).__setitem__('sha256', '0' * 64), (), 'does not hash'),
            blob=(lambda r: h(r).__setitem__('git_blob_sha1', '0' * 40), (), 'git_blob_sha1'),
            content=(lambda r: h(r)['content']['bounds'].__setitem__('value_bound', .5), (), 'content differs'),
            bound=(lambda r: r['tolerance'].__setitem__('value_bound', .5), (), 'not the declared'),
            tolerance_sha=(lambda r: r['tolerance']['declared_bounds'].__setitem__('sha256', '0' * 64), (), 'same sha256'),
            header=(lambda r: r['header'].pop('declared_bounds'), (), 'without header.declared_bounds'),
            stripped=(stripped, (), 'left traces'),
            repo_path=(lambda r: h(r).__setitem__('repo_path', gate.DECLARED_SUBDIR + '/other.json'), (), 'has no'),
            outside=(lambda r: h(r).__setitem__('repo_path', 'gpu_engine/champion.json'), (), 'not under'),
            evaluator=(lambda r: r['evaluator'].__setitem__('archive_sha256', '6' * 64), (), 'evaluator.archive_sha256'),
            started=(lambda r: h(r).__setitem__('run_started_at', '2026-10-01T00:00:00+00:00'), (), 'before the recorded run start'),
            forged=(forged, (), 'holds'),
            other_commit=(lambda r: None, ('--declared-bounds-commit', 'b' * 40), 'is not the recorded'),
            other_sha=(lambda r: None, ('--declared-bounds-sha256', '0' * 64), 'is not the recorded'))

    def test_far_from_frozen_bounds_reach_every_site(self):
        """Depth 3, k 2, --block-invariance and --mutation-test under a declaration far from the frozen numbers (the exact champion's floor is 0,
        so the frozen value bound is 0; here value 1e-3, turn-1 logits .3, near-tie Q 5e-4, deeper .4): (b)'s values and root prior, every (c)
        quantity, block invariance's values and logits, (e)'s q_noise_bound (distinct from the value bound), (f), topk_boundary's gap count and
        every mutation verdict take the declared numbers, and the run passes with every mutant killed."""
        gate = gate_module();numbers = N.bounds(value_bound=1e-3, turn1_logit_bound=.3, near_tie_q_bound=5e-4, deeper_logit_rule=None, deeper_logit_bound=.4)
        _, flags, sha = self.declared(bounds=numbers)
        code, out = self.main('--depth', '3', '--k-own', '2', '--k-opp', '2', '--block-invariance', '--mutation-test', *flags)
        report, mutation = (json.loads((out / name).read_text()) for name in ('gate.json', gate.MUTATION_FILE));checks = report['checks']
        self.assertEqual((code, report['status'], mutation['status']), (0, 'PASS', 'PASS'), (report.get('status_reasons'), mutation.get('survivors')))
        self.assertEqual(report['tolerance']['declared_bounds']['frozen_rule_would_give']['value_bound'], 0.)
        self.assertEqual((checks['b']['values']['t1_value']['bound'], checks['b']['values']['root_peer_logits']['bound']), (1e-3, .3))   # (b)
        for name, m in checks['c']['quantities'].items():   # (c)
            if name not in gate.g2.FLOORS:self.assertEqual(m['bound'], .4 if gate.deeper_logit(name, 3) else .3 if gate.g2.is_logit(name) else 1e-3, name)
        floats = checks['block_invariance']['floats'];self.assertEqual((floats['leaf_value']['bound'], floats['t1_own_logits']['bound']), (1e-3, .3))   # block invariance
        self.assertEqual((checks['e']['q_noise_bound'], report['tolerance']['q_noise_bound'], checks['f']['value_bound']), (5e-4, 5e-4, 1e-3))   # (e), (f)
        topk = lambda bound: gate.merge_topk_n([c.get('topk') for c in report['calls']], logit_bound=bound)
        self.assertEqual(report['topk_boundary'], topk(.3));self.assertNotEqual(topk(.3), topk(gate.g2.LOGIT_BOUND))   # the gap count under .3, not 0.1
        call = mutation['calls'][0];recorded = dict(declared_bounds_sha256=sha, logit_bound=.3, q_noise_bound=5e-4)
        self.assertEqual({k: call['clean'][k] for k in recorded}, recorded);self.assertEqual(call['value_bound'], 1e-3)
        for name, verdict in call['mutants'].items():
            self.assertTrue(verdict['killed'], name);self.assertEqual({k: verdict[k] for k in recorded}, recorded, name);self.assertEqual(verdict['value_bound'], 1e-3, name)

    def test_a_declared_root_logit_bound_reaches_every_root_site(self):
        """Depth 3, k 2, k1_opp 1 (a root selection that prunes) and --block-invariance under turn1_logit_bound .3 with root_logit_bound .05 (and --mutation-test), then
        swapped: the root opponent's prior takes root_logit_bound at (b), (c), depth2_pass and block invariance (bound_rule
        declared_root_logit_bound) and in topk_boundary's root gap count; every turn-1 prior and every other quantity keeps its declared
        number; the tolerance, its applied section and table, the TOLERANCE and DECLARED_BOUNDS log lines and every mutation verdict record it.
        --reverdict re-judges the root under it: a saved root difference .1, consistent with a saved turn-1 bound .3, FAILs at .05 while the
        same on the turn-1 priors passes; a tolerance.root_logit_bound off the declared one (or none) is refused."""
        gate = gate_module()
        for turn1, root, mutation in ((.3, .05, True), (.05, .3, False)):
            with self.subTest(turn1_logit_bound=turn1, root_logit_bound=root):
                _, flags, sha = self.declared(bounds=N.bounds(value_bound=1e-3, turn1_logit_bound=turn1, root_logit_bound=root, deeper_logit_rule=None, deeper_logit_bound=.4))
                lines = [];extra = ('--mutation-test',) if mutation else ()
                code, out = self.main('--depth', '3', '--k-own', '2', '--k-opp', '2', '--k1-opp', '1', '--block-invariance', *extra, *flags, lines=lines)
                report = json.loads((out / 'gate.json').read_text());tolerance = report['tolerance']
                self.assertEqual((code, report['status']), (0, 'PASS'), report.get('status_reasons'))
                for check, name in N.ROOT_SITES:
                    m = N.site(report, check, name);self.assertGreater(m['compared'], 0, (check, name))
                    self.assertEqual((m['bound'], m['bound_rule']), (root, 'declared_root_logit_bound'), (check, name))
                for check, name in N.TURN1_SITES:self.assertEqual(N.site(report, check, name)['bound'], turn1, (check, name))
                for name, m in report['checks']['c']['quantities'].items():
                    if name not in gate.g2.FLOORS:
                        self.assertEqual(m['bound'], root if name in gate.ROOT_LOGITS else .4 if gate.deeper_logit(name, 3) else turn1 if gate.g2.is_logit(name) else 1e-3, name)
                rows = {(r['check'], r['quantity']): r['bound'] for r in tolerance['table']};self.assertEqual({k: rows[k] for k in N.ROOT_SITES}, dict.fromkeys(N.ROOT_SITES, root))
                self.assertEqual((tolerance['logit_bound'], tolerance['root_logit_bound'], tolerance['declared_bounds']['applied']['root_logit_bound']), (turn1, root, root))
                gaps = [g for c in report['calls'] for g in c['topk']['root']['opp']['gaps']];topk = report['topk_boundary']['root']['opp']
                self.assertEqual((topk['selections'], topk['gap_below_logit_bound'], report['topk_boundary']['root_logit_bound']), (len(gaps), sum(g < root for g in gaps), root))
                self.assertEqual(report['topk_boundary'], gate.merge_topk_n([c.get('topk') for c in report['calls']], logit_bound=turn1, root_logit_bound=root))
                self.assertNotEqual(sum(g < root for g in gaps), sum(g < turn1 for g in gaps))   # the root's count under root_logit_bound, not turn1_logit_bound
                self.assertEqual(next(x for x in lines if x['message'] == 'TOLERANCE')['root_logit_bound'], root)
                self.assertEqual(next(x for x in lines if x['message'] == 'DECLARED_BOUNDS')['bounds']['root_logit_bound'], root)
                if not mutation:continue
                call = json.loads((out / gate.MUTATION_FILE).read_text())['calls'][0];recorded = dict(declared_bounds_sha256=sha, logit_bound=turn1, root_logit_bound=root, q_noise_bound=1e-3)
                self.assertEqual({k: call['clean'][k] for k in recorded}, recorded)
                for name, verdict in call['mutants'].items():self.assertTrue(verdict['killed'], name);self.assertEqual({k: verdict[k] for k in recorded}, recorded, name)
                code, again = self.reverdicted(report, '--declared-bounds-commit', self.record.commit, '--declared-bounds-sha256', sha)
                self.assertEqual((code, again['status'], again['reverdict']['changed_quantities']), (0, 'PASS', []))
                def saved(names):
                    r = copy.deepcopy(report)
                    for name in names:r['checks']['c']['quantities'][name].update(bound=turn1, max_abs_difference=.1, ratio_to_bound=.1 / turn1, within_bound=True, passed=True)
                    return r
                code, again = self.reverdicted(saved(gate.ROOT_LOGITS));quantities = again['checks']['c']['quantities']
                self.assertEqual((code, again['status'], {c['quantity'] for c in again['reverdict']['changed_quantities']}), (1, 'FAIL', set(gate.ROOT_LOGITS)))
                for name in gate.ROOT_LOGITS:self.assertEqual({k: quantities[name][k] for k in ('bound', 'bound_rule', 'within_bound', 'passed')}, dict(bound=root,
                    bound_rule='declared_root_logit_bound', within_bound=False, passed=False), name)
                code, again = self.reverdicted(saved(('t1_own_logits', 'depth2_pass_t1_own_logits')))
                self.assertEqual((code, again['status'], again['checks']['c']['quantities']['t1_own_logits']['bound']), (0, 'PASS', turn1))
                for change in (lambda r: r['tolerance'].__setitem__('root_logit_bound', turn1), lambda r: r['tolerance'].pop('root_logit_bound')):
                    r = copy.deepcopy(report);change(r);text, again = self.reverdicted(r)
                    self.assertIsNone(again);self.assertIn('tolerance.root_logit_bound', text);self.assertIn('fail closed', text)

    def test_the_declared_guard_round_trips_through_reverdict(self):
        """The champion's declared depth-2 gate with --block-invariance (one parameter set, every digest matching, the guard applied): every
        guarded quantity has its GUARD_DECLARED_RULE mode, digest_certified except c.depth2_pass_t1_value (gated: no certifying digest), and the
        header's declared_bounds carries the rule. --reverdict re-derives the same modes from the saved digest findings and evidence and refuses
        a saved mode flipped either way, or a saved digest finding or per-position count that no longer supports the saved mode."""
        gate = gate_module();_, flags, sha = self.declared()
        code, out = self.main('--depth', '2', '--k-own', '9', '--k-opp', '9', '--block-invariance', *flags);report = json.loads((out / 'gate.json').read_text())
        self.assertEqual((code, report['status']), (0, 'PASS'), report.get('status_reasons'));self.assertIsNotNone(report['tolerance']['guard_bound'])
        record = report['tolerance']['guard_declared']
        certified = ('b.t1_value', 'c.t1_value_head', 'c.leaf_value', 'c.depth2_pass_t2_value', 'block_invariance.t1_value', 'block_invariance.leaf_value')
        self.assertEqual(record['modes'], dict(dict.fromkeys(certified, 'digest_certified'), **{'c.depth2_pass_t1_value': 'gated'}))
        self.assertEqual((record['gated_no_certifying_digest'], record['rule']), (['c.depth2_pass_t1_value'], gate.GUARD_DECLARED_RULE))
        self.assertEqual(report['header']['declared_bounds']['guard_rule'], gate.GUARD_DECLARED_RULE)
        code, again = self.reverdicted(report, '--declared-bounds-commit', self.record.commit, '--declared-bounds-sha256', sha)
        self.assertEqual((code, again['status'], again['tolerance']['guard_declared'], again['reverdict']['changed_quantities'], again['reverdict']['guard_declared_rule']),
            (0, 'PASS', record, [], gate.GUARD_DECLARED_RULE))
        for check in ('b', 'c', 'block_invariance'):self.assertEqual(again['checks'][check], report['checks'][check], check)
        mode = lambda check, key, name, value: lambda r: r['checks'][check][key][name].__setitem__('guard_mode', value)
        for case, change in dict(b=mode('b', 'values', 't1_value', 'gated'), c=mode('c', 'quantities', 't1_value_head', 'gated'),
                c_no_digest=mode('c', 'quantities', 'depth2_pass_t1_value', 'digest_certified'), invariance=mode('block_invariance', 'floats', 'leaf_value', 'gated'),
                digest_finding=lambda r: r['checks']['a']['mismatched'].__setitem__('t1_digest_own', 1),
                invariance_evidence=lambda r: r['checks']['block_invariance']['floats']['t1_value']['guard_evidence'].__setitem__('digest_mismatched', 1),
                inexact_evidence=lambda r: r['checks']['b']['values']['t1_value']['guard_evidence'].__setitem__('value_positions_outside_digest_inexact', 1)).items():
            with self.subTest(case):
                r = copy.deepcopy(report);change(r);text, again = self.reverdicted(r)
                self.assertIsNone(again);self.assertIn('saved guard modes differ', text);self.assertIn('fail closed', text)

    def test_a_run_without_git_is_provisional_and_the_audit_requires_verified(self):
        """The committed declaration's bytes in a checkout that is not a git work tree (an exported image): the run keeps PASS and exit 0 with
        'provisional: git_verification ...' in status_reasons; --reverdict alone keeps the reason; the audit, --reverdict --declared-bounds-commit,
        refuses there (it requires verified) and passes on the work tree that holds the commit, where the reason clears."""
        gate = gate_module();_, flags, sha = self.declared();repo, commit = self.repo, self.record.commit
        exported = Path(tempfile.mkdtemp(prefix='champion-exported-'));target = exported / gate.DECLARED_SUBDIR / 'champion.json';target.parent.mkdir(parents=True)
        target.write_bytes(self.record.path.read_bytes());self.repo = exported
        code, out = self.main('--depth', '2', '--k-own', '2', '--k-opp', '2', '--declared-bounds', str(target), *flags[2:]);report = json.loads((out / 'gate.json').read_text())
        self.assertEqual((code, report['status'], report['passed'], report['header']['declared_bounds']['git_verification']['status']), (0, 'PASS', True, 'not verifiable here'))
        self.assertEqual(len(report['status_reasons']), 1);self.assertTrue(report['status_reasons'][0].startswith('provisional: git_verification not verifiable here'))
        text, again = self.reverdicted(report, '--declared-bounds-commit', commit);self.assertIsNone(again);self.assertIn('requires git_verification verified', text)
        code, again = self.reverdicted(report);self.assertEqual((code, again['status']), (0, 'PASS'));self.assertTrue(again['status_reasons'][0].startswith('provisional: git_verification'))
        self.repo = repo;code, again = self.reverdicted(report, '--declared-bounds-commit', commit, '--declared-bounds-sha256', sha)
        self.assertEqual((code, again['status'], again['status_reasons'], again['reverdict']['declared_bounds_reverified']['git']['status']), (0, 'PASS', [], 'verified'))

    def test_the_resolved_evaluator_is_matched_again(self):
        """The flags name the declared champion but load_roots resolves another identity (a precision the header reports differently): refused
        after loading, before any pricing."""
        _, flags, _ = self.declared();roots = self.roots;self.roots = lambda: dict.__setitem__((r := roots()).evaluator, 'precision', 'float32') or r
        try:
            with self.assertRaises(SystemExit) as raised:self.main(*flags)
        finally:self.roots = roots
        self.assertIn('not for the resolved evaluator', str(raised.exception.code));self.assertIn('evaluator.precision', str(raised.exception.code))

    def test_a_declaration_for_another_evaluator_is_refused(self):
        for key, value in (('precision', 'float32'), ('archive_sha256', 'e' * 64), ('kind', 'v2'), ('checkpoint_sha256', 'b' * 64)):
            with self.subTest(key):
                _, flags, _ = self.declared(**{key: value});out = Path(tempfile.mkdtemp(prefix='champion-cli-')) / 'never'
                with patch.object(gate_module(), 'load_roots', side_effect=AssertionError('loaded')), patch.object(gate_module(), 'REPO_ROOT', self.repo), \
                        self.assertRaises(SystemExit) as raised:
                    gate_module().main(['--out', str(out), '--d3-root', 'x', '--chunk', '0', *self.FLAGS, *flags])
                self.assertIn(f'evaluator.{key}', str(raised.exception.code));self.assertFalse(out.exists())


def gate_module():
    from gpu_engine import d3_depthn_gate
    return d3_depthn_gate


class ChildBlockDependent:
    """A Depth2Pricer stand-in that depends on its child block: the real pricer, except that child_block-27 frames carry every head-valued
    t2_value .25 higher."""
    def __init__(self, pricer):self.pricer, self.child_block, self.capacity = pricer, pricer.child_block, pricer.capacity

    def __getattr__(self, name):return getattr(self.pricer, name)   # widths and options, as the real pricer's

    def price(self, *args, **kwargs):
        frame = self.pricer.price(*args, **kwargs)
        if self.child_block != 27:return frame
        value = np.array(frame.t2_value, copy=True);value[np.asarray(frame.t2_status) == 4] += np.float32(.25)
        return replace(frame, t2_value=value)


# ---------------------------------------------------------------- the tools' flags

CHAMPION = ['--champion-archive', '/shared/x/champion.npz', '--champion-archive-sha256', 'e' * 64, '--champion-checkpoint-sha256', CHECKPOINT]
HEAD = ['--head-archive', '/shared/x/actor-99.npz', '--head-archive-sha256', '1' * 64, '--head-checkpoint-sha256', '2' * 64]


class ChampionFlagTests(unittest.TestCase):
    def tools(self, out):
        from gpu_engine import d3_depthn_gate, d3_depthn_throughput, d3_depth2_throughput, d3_search_latency, champion_forward_bench
        base = ['--d3-root', '/d3', '--chunk', '0']
        return dict(
            d3_depthn_gate=lambda argv: d3_depthn_gate.parse(['--out', str(out)] + base + argv)[1],
            d3_depthn_throughput=lambda argv: d3_depthn_throughput.parse(['--out', str(out), '--capacity', '1'] + base + argv),
            d3_depth2_throughput=lambda argv: d3_depth2_throughput.parse(['--out', str(out), '--capacity', '256', '--worlds-blocks', '1', '--profile-split'] + base + argv),
            d3_search_latency=lambda argv: d3_search_latency.parse(['--out', str(out), '--configs', 'depth2-fw@8/16/32,depth2-merged@8/16/32'] + base + argv),
            champion_forward_bench=lambda argv: champion_forward_bench.parse(['--out', str(out)] + argv))

    def refused(self, parse, argv, message):
        stderr = io.StringIO()
        with patch.object(sys, 'stderr', stderr), self.assertRaises(SystemExit):parse(argv)
        self.assertIn(message, stderr.getvalue())

    def test_flags_parse_and_are_exclusive_with_the_head_override(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, parse in self.tools(Path(directory) / 'out').items():
                with self.subTest(tool=name):
                    args = parse(CHAMPION)
                    self.assertEqual((args.champion_archive, args.champion_archive_sha256, args.champion_checkpoint_sha256, args.champion_precision),
                        ('/shared/x/champion.npz', 'e' * 64, CHECKPOINT, 'bf16'))
                    self.assertEqual(parse(CHAMPION + ['--champion-precision', 'f32']).champion_precision, 'f32')
                    self.assertEqual(ce.log_fields(args), dict(champion_archive='/shared/x/champion.npz', champion_archive_sha256='e' * 64,
                        champion_checkpoint_sha256=CHECKPOINT, champion_precision='bf16'))
                    self.refused(parse, CHAMPION[:4], 'go together')
                    self.refused(parse, CHAMPION[:2] + CHAMPION[4:], 'go together')
                    self.refused(parse, ['--champion-precision', 'f32'] + ([] if name != 'champion_forward_bench' else []), '--champion-precision needs')
                    self.refused(parse, CHAMPION[:3] + ['E' * 64] + CHAMPION[4:], '64-hex')
                    self.refused(parse, CHAMPION + ['--champion-precision', 'fp16'], 'invalid choice')
                    if name != 'champion_forward_bench':
                        self.refused(parse, CHAMPION + HEAD, 'mutually exclusive')
                        self.refused(parse, CHAMPION + ['--head-category-sha256', '3' * 64], 'need --head-archive')
                        none = parse([]);self.assertIsNone(none.champion_archive);self.assertIsNone(none.champion_precision);self.assertEqual(ce.log_fields(none), {})
            from gpu_engine import d3_depthn_gate
            self.refused(lambda argv: d3_depthn_gate.parse(argv), ['--out', str(Path(directory) / 'o'), '--diagnostics-from', directory] + CHAMPION, 'offline')
            self.refused(lambda argv: d3_depthn_gate.parse(argv), ['--out', str(Path(directory) / 'o'), '--d3-root', '/d3', '--chunk', '0', '--raw-head-check'] + CHAMPION,
                '--raw-head-check')

    def test_load_champion_binds_the_bank_catalog(self):
        from gpu_engine import d3_depth2_gate as g2
        ev = tiny_champion();loaded = SimpleNamespace(provider=SimpleNamespace(prepared=SimpleNamespace(catalog='CATALOG')))
        with patch.object(ce, 'load_from_arguments', return_value=ev) as load, patch.object(ce, 'check_binding', return_value=True) as bind:
            self.assertIs(g2.load_champion(SimpleNamespace(), loaded), ev)
        self.assertEqual(bind.call_args.args, (ev, 'CATALOG'));self.assertEqual(load.call_count, 1)
        frame = batch.FramePricer('ex', 'model', capacity=16, block_rows=16, actor_capacity=33, vectorized=True, host_prep=True)
        self.assertIs(g2.with_forward(frame, None), frame)
        bound = g2.with_forward(frame, ev)
        self.assertEqual((bound.capacity, bound.block_rows, bound.actor_capacity, bound.vectorized, bound.host_prep, bound.forward), (16, 16, 33, True, True, ev))
        self.assertEqual(bound.evaluator_suffix, ce.program_suffix(ev));self.assertEqual(frame.evaluator_suffix, ())

    def test_evaluator_header(self):
        from gpu_engine import d3_depth2_gate as g2
        ev = tiny_champion();args = SimpleNamespace(champion_archive='/shared/x/champion.npz', head_archive=None)
        loaded = SimpleNamespace(plan=SimpleNamespace(model=SimpleNamespace(arm='D')), labels=SimpleNamespace(sha256='l' * 64), category_sha256='k' * 64,
            snapshot=dict(iteration=-1, archive='run/actor--1.npz', archive_sha256='4' * 64))
        header = g2.evaluator_header(args, loaded, None, champion=ev)
        self.assertEqual((header['override'], header['kind'], header['identity'], header['archive'], header['archive_sha256'], header['checkpoint_sha256'],
            header['precision'], header['pricer_arm']), (True, 'champion', ['champion', 'a' * 64, 'bfloat16'], '/shared/x/champion.npz', 'a' * 64, CHECKPOINT,
            'bfloat16', None))
        self.assertEqual(header['bank_actor']['iteration'], -1);self.assertNotIn('warning', header)
        own = g2.evaluator_header(args, replace_snapshot(loaded), None)
        self.assertEqual((own['kind'], own['identity'], own['precision'], own['override']), ('v2', ['v2'], 'bfloat16', False))
        with self.assertRaisesRegex(ValueError, 'exclusive'):g2.bind_roots([], (ev.params,), None, head_config='HEAD', evaluator=ev)
        with self.assertRaisesRegex(ValueError, 'alone'):g2.bind_roots([[]], ('a', 'b'), None, evaluator=ev)


def replace_snapshot(loaded):
    return SimpleNamespace(**dict(vars(loaded), snapshot=dict(loaded.snapshot, checkpoint='c.pt', checkpoint_sha256='3' * 64, model_sha256='5' * 64)))


# ---------------------------------------------------------------- the tools thread the evaluator

class Capture:
    """A pricer stand-in recording its constructor keywords (every option the tools pass)."""
    built = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs;Capture.built.append(self);self.child_block = kwargs.get('child_block', 27)


def small_requests(c, count=3):
    """Requests whose V2 root inputs give the exact stand-in a usable output (actors 1/0: the bank's, before binding)."""
    rng = np.random.default_rng(9);out = []
    for i in range(count):
        inputs = {k: v[:1] for k, v in v2_inputs(c, 1, seed=20 + i).items()};inputs['numeric_features'] = inputs['numeric_features'] * np.float32(.05)
        out.append(batch.Request(SimpleNamespace(root=SimpleNamespace(address=SimpleNamespace(stream=100 + i, frame=4, owner=i % 2))), (0.,) * 9,
            (True,) * 9, inputs, 1, 0))
    return out


def fake_loaded(requests):
    snapshot = dict(iteration=-1, archive='run/actor--1.npz', archive_sha256='4' * 64, checkpoint='c.pt', checkpoint_sha256='3' * 64, model_sha256='5' * 64)
    return SimpleNamespace(provider=SimpleNamespace(executor='ex', recipe=SimpleNamespace(seed=1779000000), materializer='materializer', source='source'),
        plan=SimpleNamespace(model=SimpleNamespace(arm='D'), historical_window=32), params=('bank actor',), groups=[list(requests)], source=dict(fixture=True),
        labels=SimpleNamespace(sha256='l' * 64), category_sha256='k' * 64, snapshot=snapshot, frame_pricer='D3 frame pricer')


class ToolThreadingTests(unittest.TestCase):
    """Each tool binds the champion once (load_champion, bind_roots) and hands the same evaluator to every pricer, FramePricer and
    oracle it builds; without the flags every call is the unchanged one (no forward= keyword)."""
    def setUp(self):
        Capture.built = [];self.champion = exact_champion();self.requests = small_requests(self.champion.config)

    def args(self, **kw):
        return SimpleNamespace(**dict(dict(head_archive=None, champion_archive='/shared/x/champion.npz', champion_archive_sha256='e' * 64,
            champion_checkpoint_sha256=CHECKPOINT, champion_precision='bf16', chunk=0, saved_roots=None, max_roots=None, capacity=4, w_values=(2,),
            worlds_blocks=1, k_own=9, k_opp=9, k1_own=9, k1_opp=9), **kw))

    def loaded(self):
        """The gate's bank loader and champion loader replaced (a fake chunk of self.requests; the exact stand-in), logs silenced."""
        from gpu_engine import d3_depth2_gate as g2
        stack = contextlib.ExitStack();loaded = fake_loaded(self.requests)
        for p in (patch.object(g2, 'bank_chunk_requests', return_value=loaded), patch.object(g2, 'load_champion', return_value=self.champion),
                patch.object(g2, 'log', lambda *a, **k: None)):stack.enter_context(p)
        return stack

    def test_depthn_gate_roots_bind_the_champion(self):
        from gpu_engine import d3_depthn_gate as gate
        with self.loaded(), patch.object(gate, 'log', lambda *a, **k: None):roots = gate.load_roots(self.args(d3_root='/d3'))
        self.assertIs(roots.forward, self.champion);self.assertEqual(roots.params, (self.champion.params,));self.assertEqual(roots.actor_capacity, 1)
        self.assertEqual({(r.learner_actor, r.opponent_actor) for g in roots.groups for r in g}, {(0, 0)})
        self.assertEqual((roots.evaluator['kind'], roots.evaluator['arm_checked'], roots.evaluator['dropped_roots']), ('champion', True, []))
        pricers = gate.DepthNPricers(SimpleNamespace(DepthNPricer=Capture), 'ex', 'model', depth=3, k_own=2, k_opp=2, selection='prior', capacity=4,
            actor_capacity=1, forward=roots.forward)
        pricers.get();self.assertIs(Capture.built[-1].kwargs['forward'], self.champion)
        self.assertEqual(pricers.options()['evaluator'], ce.describe(self.champion))
        gate.depth2_pass_pricer(SimpleNamespace(DepthNPricer=Capture), 'ex', 'model', capacity=4, actor_capacity=1, forward=roots.forward)
        self.assertIs(Capture.built[-1].kwargs['forward'], self.champion)
        plain = gate.DepthNPricers(SimpleNamespace(DepthNPricer=Capture), 'ex', 'model', depth=3, k_own=2, k_opp=2, selection='prior', capacity=4, actor_capacity=3)
        plain.get();self.assertNotIn('forward', Capture.built[-1].kwargs);self.assertNotIn('evaluator', plain.options())
        gate.depth2_pass_pricer(SimpleNamespace(DepthNPricer=Capture), 'ex', 'model', capacity=4, actor_capacity=3);self.assertNotIn('forward', Capture.built[-1].kwargs)
        self.assertEqual(gate.forward_kwargs(None), {});self.assertEqual(gate.forward_kwargs(self.champion), dict(forward=self.champion))

    def test_throughput_tools_hand_the_evaluator_to_every_pricer(self):
        from gpu_engine import d3_depthn_throughput as tn, d3_depth2_throughput as t2
        with self.loaded():ctx = tn.prepare(self.args())
        self.assertIs(ctx.forward, self.champion);self.assertEqual((ctx.params, ctx.actor_capacity), ((self.champion.params,), 1))
        self.assertEqual(ctx.evaluator['weights_affect_speed'], t2.CHAMPION_SPEED_NOTE);self.assertEqual(ctx.evaluator['kind'], 'champion')
        args = SimpleNamespace(capacity=4, k1_own=9, k1_opp=9, block_lanes=2187)
        tn.Caller('depth2-fw', args, ctx, module=SimpleNamespace(DepthNPricer=Capture));self.assertIs(Capture.built[-1].kwargs['forward'], self.champion)
        tn.Caller('depth1', args, ctx, module=SimpleNamespace(DepthNPricer=Capture), frame_pricer_class=Capture)
        self.assertIs(Capture.built[-1].kwargs['forward'], self.champion)
        tn.Caller('depth1', args, SimpleNamespace(**dict(vars(ctx), forward=None)), module=SimpleNamespace(DepthNPricer=Capture), frame_pricer_class=Capture)
        self.assertNotIn('forward', Capture.built[-1].kwargs)
        with self.loaded():ctx2 = t2.prepare(self.args(capacity=4))
        self.assertIs(ctx2.forward, self.champion);self.assertEqual(ctx2.evaluator['identity'], list(self.champion.identity))
        calls, made = [], []
        def make(executor, model, *, capacity, widths, child_block, actor_capacity, forward=None):
            made.append(forward);return Capture(forward=forward), dict(child_block=27)
        def seen(params, inputs):
            calls.append(int(jax.tree.leaves(inputs)[0].shape[0]));return self.champion(params, inputs)
        ctx2.forward = seen
        with patch.object(t2, 'make_depth2', side_effect=make), patch.object(t2, 'measure', return_value=[]), patch.object(t2, 'derive', return_value={}), \
                patch('gpu_engine.device_public_teacher_batch.FramePricer', Capture):
            t2.measure_config('depth2', SimpleNamespace(capacity=256, child_block=None, k_own=9, k_opp=9, repeats=1), ctx2, None)
            self.assertIs(made[-1], seen)
            t2.measure_config('depth1', SimpleNamespace(capacity=256, child_block=None, k_own=9, k_opp=9, repeats=1, d1=True), ctx2, None)
            self.assertIs(Capture.built[-1].kwargs['forward'], seen)
            t2.measure_config('constant', SimpleNamespace(capacity=256, child_block=None, k_own=9, k_opp=9, repeats=1), ctx2, None)
        variant = made[-1];self.assertIsNot(variant, seen)
        inputs = {k: j.asarray(v) for k, v in v2_inputs(self.champion.config, 3).items()}
        out = variant(self.champion.params, inputs);self.assertEqual(calls[-1], 3)   # a turn-1 batch: the champion itself
        np.testing.assert_array_equal(np.asarray(out.value), np.asarray(self.champion(self.champion.params, inputs).value))

    def test_latency_tool_roots_and_merged_pricer_take_the_evaluator(self):
        from gpu_engine import d3_search_latency as tool
        pricer, options = tool.merged_pricer(SimpleNamespace(Depth2Pricer=FakeMerged), 'ex', 'model', capacity=1, actor_capacity=1, k1_own=9, k1_opp=9,
            forward=self.champion)
        self.assertIs(pricer.kwargs['forward'], self.champion);self.assertEqual(options['evaluator'], ce.describe(self.champion));self.assertNotIn('forward', options)
        plain, plain_options = tool.merged_pricer(SimpleNamespace(Depth2Pricer=FakeMerged), 'ex', 'model', capacity=1, actor_capacity=1, k1_own=9, k1_opp=9)
        self.assertNotIn('forward', plain.kwargs);self.assertEqual(plain_options, dict(digest=True, actor_capacity=1, k_own=9, k_opp=9, k1_own=9, k1_opp=9,
            capacity=1, fixed_capacity=True))
        program = tool.root_program_for('model', self.champion);root = {k: j.asarray(v) for k, v in self.requests[0].root_inputs.items()}
        np.testing.assert_array_equal(np.asarray(program(self.champion.params, root).policy_logits), np.asarray(self.champion(self.champion.params, root).policy_logits))
        fake = lambda p, x, model, compute_dtype: dict(model=j.asarray(model == 'MODEL'), bf16=j.asarray(compute_dtype == 'bfloat16'))
        with patch.object(batch.v5_model, 'forward', side_effect=fake) as v2:out = tool.root_program_for('MODEL', None)({'w': j.zeros(1)}, root)
        self.assertEqual((bool(out['model']), bool(out['bf16']), v2.call_count), (True, True, 1))   # the V2 default: the D-arm head in bf16, unchanged
        search = tool.Search('depth2-fw', 1, SimpleNamespace(k1_own=9, k1_opp=9, block_lanes=2187), SimpleNamespace(provider=SimpleNamespace(executor='ex'),
            plan=SimpleNamespace(model='model'), actor_capacity=1, params=(self.champion.params,), forward=self.champion), module=SimpleNamespace(DepthNPricer=Capture))
        self.assertIs(search.pricer.kwargs['forward'], self.champion)


class N2Stop(Exception):
    pass


class N2Oracle:
    """The oracle run_n2 builds last: records its keywords, then stops the run (nothing past the pricers' construction is needed)."""
    built = []

    def __init__(self, *args, **kwargs):
        N2Oracle.built.append(kwargs);raise N2Stop


class LatencyThreadingTests(unittest.TestCase):
    """--n2-equivalence builds both compared pricers and its oracle with the evaluator; the latency tool's prepare marks a champion
    run as bound (head_override: every Request keeps bind_roots' champion prior and actors 0/0) and roots it with the champion."""
    def test_n2_equivalence_builds_both_pricers_and_the_oracle_with_the_evaluator(self):
        from gpu_engine import d3_depthn_gate as gate, d3_depth2_gate as g2
        champion = exact_champion();made = []
        def make_pricer(executor, model, **kw):
            made.append(kw);return SimpleNamespace(capacity=4), {}
        args = SimpleNamespace(capacity=4, k1_own=9, k1_opp=9, block_lanes=2187, n2_k=3, oracle_chunk=8)
        for forward in (champion, None):
            with self.subTest(evaluator='champion' if forward else 'v2'):
                Capture.built, N2Oracle.built, made[:] = [], [], []
                roots = SimpleNamespace(forward=forward, actor_capacity=1, groups=[])
                with patch.object(g2, 'make_pricer', side_effect=make_pricer), self.assertRaises(N2Stop):
                    gate.run_n2(args, roots=roots, provider=SimpleNamespace(executor='ex'), plan=SimpleNamespace(model='model'), module=SimpleNamespace(DepthNPricer=Capture),
                        frame_pricer=SimpleNamespace(capacity=4), seeds=(), out=None, reference=SimpleNamespace(DepthNOracle=N2Oracle))
                self.assertEqual((len(Capture.built), len(made), len(N2Oracle.built)), (2, 2, 1))
                given = [c.kwargs for c in Capture.built] + made + N2Oracle.built
                if forward is None:self.assertTrue(all('forward' not in kw for kw in given), given)
                else:self.assertTrue(all(kw.get('forward') is champion for kw in given), given)

    def test_latency_prepare_binds_and_roots_a_champion_run(self):
        from gpu_engine import d3_search_latency as tool, v5_device_teacher as teacher
        from pokezero import v5_contract as contract
        champion = exact_champion();at = lambda stream, frame, owner: SimpleNamespace(plans=SimpleNamespace(root=SimpleNamespace(
            address=contract.DecisionAddress(stream, frame, owner))))
        frames = SimpleNamespace(streams=np.array([[10, 11]]), frames=np.array([[5, 2]]))
        @contextlib.contextmanager
        def capture(captured, **kw):
            captured.update(roots='device roots', frames=frames, bank='bank');yield captured
        root = {k: j.asarray(v) for k, v in small_requests(champion.config, 1)[0].root_inputs.items()}
        with tempfile.TemporaryDirectory() as directory:
            for name, argv, forward, override in (('champion', CHAMPION, champion, True), ('v2', [], None, False), ('head', HEAD, None, True)):
                with self.subTest(run=name):
                    args = tool.parse(['--d3-root', 'r', '--chunk', '0', '--out', str(Path(directory) / name), '--configs', 'depth2-fw@2'] + argv)
                    base = SimpleNamespace(provider=SimpleNamespace(recipe=SimpleNamespace(seed=1779000000), source='source'), available=[at(10, 5, 0), at(11, 2, 1)],
                        plan=SimpleNamespace(model=None), params=(champion.params,), seeds=(1779000000,), forward=forward)
                    with patch.object(tool, 'capture_chunk', capture), patch.object(tool.tn, 'prepare', lambda view, base=base: base), \
                            patch.object(teacher, '_requests', lambda f, b: np.array([[0, 0, 0], [0, 1, 1]])):
                        ctx = tool.prepare(args)
                    self.assertEqual((ctx.head_override, ctx.forward), (override, forward))
                    if forward is not None:
                        np.testing.assert_array_equal(np.asarray(ctx.root_program(champion.params, root).policy_logits),
                            np.asarray(champion(champion.params, root).policy_logits))


class FakeMerged:
    def __init__(self, executor, model, *, capacity=2048, actor_capacity=33, k_own=9, k_opp=9, k1_own=9, k1_opp=9, child_block=27, digest=False, forward=None,
            fixed_capacity=False):
        self.kwargs = dict(capacity=capacity, digest=digest, fixed_capacity=fixed_capacity, **({} if forward is None else dict(forward=forward)))


class ReportTextTests(unittest.TestCase):
    def test_dropped_root_reason_names_the_evaluator_and_keeps_the_v2_wording(self):
        from gpu_engine import d3_depth2_gate as g2
        v2 = 'head override: the head\'s D-arm output on the root inputs is invalid or non-finite; never priced'
        reason = lambda *evaluator: g2.with_dropped(dict(depth2=dict(refused_count=0)), [[1, 2, 0]], *evaluator)['dropped_roots']['reason']
        self.assertEqual((reason(), reason(None), reason(dict(kind='v2', override=True))), (v2, v2, v2))
        self.assertIn('champion evaluator', reason(dict(kind='champion')));self.assertNotIn('head', reason(dict(kind='champion')))
        self.assertEqual(g2.with_dropped({}, [[1, 2, 0]], dict(kind='champion'))['dropped_roots']['count'], 1)

    def test_latency_stage_scope_keeps_the_v2_wording(self):
        from gpu_engine import d3_search_latency as tool
        self.assertIs(tool.stage_scope(dict(kind='v2')), tool.STAGE_SCOPE);self.assertIs(tool.stage_scope(None), tool.STAGE_SCOPE)
        self.assertIn('D-arm v5 head in bf16', tool.STAGE_SCOPE['root_forward'])
        scope = tool.stage_scope(dict(kind='champion'))
        self.assertEqual(scope['root_forward'], tool.CHAMPION_ROOT_FORWARD);self.assertEqual({k: v for k, v in scope.items() if k != 'root_forward'},
            {k: v for k, v in tool.STAGE_SCOPE.items() if k != 'root_forward'})
        context = dict(world_construction_path={}, solvers=[], evaluator=dict(kind='champion'))
        self.assertEqual(tool.header(context, 'c')['stages']['root_forward'], tool.CHAMPION_ROOT_FORWARD)
        self.assertIs(tool.header(dict(context, evaluator=dict(kind='v2')), 'c')['stages'], tool.STAGE_SCOPE)


# ---------------------------------------------------------------- the forward microbenchmark

class ForwardBenchTests(unittest.TestCase):
    def test_runs_tiny_on_cpu_with_compile_apart(self):
        from gpu_engine import champion_forward_bench as bench
        from gpu_engine.device_depth2_pricer import default_forward
        from pokezero.v5_model import V5ModelConfig
        from pokezero.v5_contract import LabelVocabulary
        from gpu_engine import v5_model
        ev = tiny_champion(config=tiny_config(numeric=132, categorical=41, vocabulary=16))
        config = V5ModelConfig(16, 'b' * 64, LabelVocabulary('a' * 64, ('surf', 'rest'), (('blissey', 3), ('swalot', 2))), arm='B', width=8, layers=1,
            attention_heads=2, expert_width=8, value_hidden=(8, 8), value_bins=5)
        rng = np.random.default_rng(5);v2 = {k: j.asarray(rng.normal(0, .2, s).astype(np.float32)) for k, s in v5_model.shapes(config).items()}
        clock = iter(np.arange(0., 1000., .5));records = []
        report = bench.run([('champion', ev, ev.params, ev.header(), ev.config.vocabulary), ('v2', default_forward(config), v2, dict(kind='v2'), 16)], (3, 5),
            warmup=1, repeats=3, progress=records.append, timer=lambda: float(next(clock)))
        self.assertEqual(set(report['results']), {'champion', 'v2'});self.assertEqual(set(report['results']['champion']), {'3', '5'})
        for name in ('champion', 'v2'):
            for rows in (3, 5):
                r = report['results'][name][str(rows)]
                self.assertEqual((r['rows'], r['repeats'], r['valid_rows'], r['finite'], len(r['samples_ms'])), (rows, 3, rows, True, 3))
                self.assertEqual((r['median_ms'], r['lower_seconds'], r['compile_seconds']), (500., .5, .5))   # every timer read is .5 s apart
                self.assertEqual(r['rows_per_second'], rows / .5);self.assertEqual(r['output_shapes'], dict(policy_logits=[rows, 9], value=[rows]))
        self.assertEqual(report['ratio_champion_over_v2_median'], {'3': 1., '5': 1.});self.assertEqual(len(records), 4)
        self.assertEqual(report['inputs']['shapes']['5']['numeric_features'], [5, 1, 29, 132])
        real = bench.run([('champion', ev, ev.params, {}, 16)], (4,), warmup=0, repeats=2)['results']['champion']['4']
        self.assertGreater(real['median_ms'], 0.);self.assertGreater(real['compile_seconds'], 0.)

    def test_main_writes_bench_json_from_a_bound_archive(self):
        from gpu_engine import champion_forward_bench as bench
        from pokezero.live_protocol_actor import ActorContract
        c = tiny_config(numeric=132, categorical=41, vocabulary=16);real = ActorContract.champion().checkpoint_sha256
        self.assertEqual(real, '0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be')
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, 'stdout', io.StringIO()):
            other = Path(directory) / 'other.npz';other_digest = write_archive(other, c, tiny_params(c))   # a checkpoint that is not 9375's
            with self.assertRaisesRegex(ValueError, 'real original 9375'):
                bench.main(['--out', str(Path(directory) / 'refused'), '--rows', '3', '--champion-archive', str(other), '--champion-archive-sha256', other_digest,
                    '--champion-checkpoint-sha256', CHECKPOINT])
            path = Path(directory) / 'champion.npz';digest = write_archive(path, c, tiny_params(c), checkpoint=real);out = Path(directory) / 'bench'
            self.assertEqual(bench.main(['--out', str(out), '--rows', '3,4', '--warmup', '0', '--repeats', '2', '--champion-archive', str(path),
                '--champion-archive-sha256', digest, '--champion-checkpoint-sha256', real, '--champion-precision', 'f32']), 0)
            report = json.loads((out / 'bench.json').read_text())
            self.assertEqual((report['schema'], report['rows'], report['evaluators']['champion']['identity']), (bench.SCHEMA, [3, 4], ['champion', digest, 'float32']))
            self.assertEqual(set(report['results']['champion']), {'3', '4'});self.assertIsNone(report['ratio_champion_over_v2_median'])
            stderr = io.StringIO()
            with patch.object(sys, 'stderr', stderr), self.assertRaises(SystemExit):bench.parse(['--out', str(Path(directory) / 'x')])
            self.assertIn('at least one of', stderr.getvalue())


if __name__ == '__main__':unittest.main()
