"""Generated-kernel conformance: v2e_j2c output against JAX/XLA on GPU, bitwise.

Each case is a small JAX function exercising primitive families the engine
uses (integer edge arithmetic, conversions, NaN min/max, division, per-lane
gathers/scatters, table lookups, stable sorts, argmax, cumsum, reductions,
shape ops). Inputs mix random values with edge values. Prints one JSON line
per case and exits nonzero if any output bit differs.
"""
import json
import sys

import numpy as np


def cases(rng, B):
    import jax
    import jax.numpy as j
    i32 = lambda *s: j.asarray(rng.integers(-2**31, 2**31, s, dtype=np.int64).astype(np.int32))
    small = lambda lo, hi, *s: j.asarray(rng.integers(lo, hi, s).astype(np.int32))
    edge_i = np.array([0, 1, -1, 2, -2, 7, -7, 2**31 - 1, -2**31, 31, 32, 33, 100, 714], np.int32)
    def edgy(*s):
        x = np.array(i32(*s)).reshape(-1);x[:min(len(x), 14 * 14)] = np.resize(edge_i, min(len(x), 14 * 14))
        return j.asarray(x.reshape(s))
    def edgy2(*s):
        x = np.array(i32(*s)).reshape(-1);x[:min(len(x), 14 * 14)] = np.repeat(edge_i, 14)[:min(len(x), 14 * 14)]
        return j.asarray(x.reshape(s))
    f32 = lambda *s: j.asarray((rng.standard_normal(s) * 10.0 ** rng.integers(-4, 5, s)).astype(np.float32))
    def edgy_f(*s):
        x = np.array(f32(*s)).reshape(-1)
        e = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e-40, 3e9, -3e9, 0.5, 2.5, -2.5, 714.0], np.float32)
        nan = np.array([0x7fc00000, 0xffc00000, 0x7f800001, 0xff812345, 0x7fffffff], np.uint32).view(np.float32)
        x[:len(e)] = e;x[len(e):len(e) + len(nan)] = nan;return j.asarray(x.reshape(s))
    rows = lambda x: j.arange(x.shape[0])
    yield 'int_arith', (lambda a, b: (a + b, a - b, a * b, -a, j.abs(a), j.sign(a), a & b, a | b, a ^ b, ~a, j.maximum(a, b), j.minimum(a, b))), (edgy(B, 14), edgy2(B, 14))
    yield 'int_div', (lambda a, b: (jax.lax.div(a, b), jax.lax.rem(a, b), a // b, a % b, j.floor_divide(a, j.where(b == 0, 1, b)))), (edgy(B, 14), edgy2(B, 14))
    yield 'shifts', (lambda a, s: (jax.lax.shift_left(a, s), jax.lax.shift_right_arithmetic(a, s), jax.lax.shift_right_logical(a, s),
        jax.lax.shift_left(a.astype(j.uint32), s.astype(j.uint32)), jax.lax.population_count(a))), (edgy(B, 14), small(-3, 40, B, 14))
    yield 'float_arith', (lambda a, b: (a + b, a - b, a * b, a / b, j.maximum(a, b), j.minimum(a, b), j.clip(a, 0, 1), j.floor(a),
        j.round(a), j.sign(a), -a, j.abs(a), j.isfinite(a), a * b + a, j.where(a > b, a, b), a - a, a / a, a * 0.0)), (edgy_f(B, 14), edgy_f(B, 14)[:, ::-1])
    yield 'convert', (lambda a, x: (a.astype(j.int32), a.astype(j.uint32), a.astype(j.int8), x.astype(j.float32), x.astype(j.uint32).astype(j.float32),
        (x > 0).astype(j.float32), a.astype(bool), x.astype(j.int8), x.astype(j.uint8).astype(j.int32), x / 7, x / 200)), (edgy_f(B, 14), edgy(B, 14))
    yield 'compare', (lambda a, b, x, y: (a < b, a <= b, a == b, a != b, a > b, a >= b, x < y, x.astype(j.uint32) < y.astype(j.uint32))), (edgy_f(B, 14), edgy_f(B, 14)[:, ::-1], edgy(B, 14), edgy2(B, 14))
    yield 'point_gather_scatter', (lambda x, s, a, v: (x[rows(x), s, a], x.at[rows(x), s, a].set(v), x.at[rows(x), s, a].set(j.where(v > 0, v, x[rows(x), s, a])),
        x[rows(x), s], x.at[rows(x), s].set(x[rows(x), 1 - j.clip(s, 0, 1)]))), (small(-50, 50, B, 2, 6), small(-2, 4, B), small(-8, 9, B), small(-9, 9, B))
    table = j.asarray(rng.integers(-1000, 1000, (421, 6)).astype(np.int32));ftable = j.asarray(rng.standard_normal((374, 3)).astype(np.float32))
    yield 'table_lookup', (lambda ids, k: (table[j.clip(ids, 0, 420)], table[ids], ftable[j.clip(k, 0, 373), 1], table[j.clip(ids, 0, 420)][..., 3])), (small(-30, 450, B, 2, 6), small(-5, 400, B, 4))
    yield 'take_along', (lambda x, ix: (j.take_along_axis(x, ix, axis=-1), j.take_along_axis(x, j.clip(ix, 0, 5)[..., :1], axis=-1))), (small(0, 99, B, 2, 6), small(-1, 7, B, 2, 6))
    yield 'sort', (lambda k, v: (j.sort(k, axis=-1), j.argsort(k, axis=-1), jax.lax.sort((k, v), dimension=1, num_keys=1)[1], j.sort(k[:, :5], axis=-1))), (small(0, 6, B, 40), small(-9, 9, B, 40))
    yield 'argmax_cumsum', (lambda m, x: (j.argmax(m, axis=-1), j.argmax(m[..., :1], axis=-1), j.cumsum(x, axis=-1), j.cumsum(x[:, ::-1], axis=-1),
        j.any(m, axis=-1), j.all(m, axis=-1), j.sum(x, axis=-1), j.max(x, axis=-1), j.min(x, axis=-1))), (j.asarray(rng.random((B, 12)) < .2), small(-9, 9, B, 12))
    yield 'float_pair_sum', (lambda p: (j.sum(p, axis=-1), p.sum(axis=-1)[..., None] + p[..., 0:1])), (edgy_f(B, 2, 2, 2),)
    yield 'shape_ops', (lambda x, y, z: (j.concatenate([x, y], axis=-1), j.stack([x, y], axis=1), j.pad(x, ((0, 0), (2, 1))), x[:, ::-1], x[:, 1:5:2],
        j.broadcast_to(z[:, None, :], (z.shape[0], 3, 4)), j.reshape(x, (x.shape[0], 2, 3)), jax.lax.dynamic_slice_in_dim(x, 3, 2, axis=1),
        jax.lax.dynamic_update_slice_in_dim(x, y[:, :2], 9, axis=1), j.where(z[:, :1] > 0, x, -x), j.transpose(x.reshape(x.shape[0], 2, 3), (0, 2, 1)))), (small(-9, 9, B, 6), small(-9, 9, B, 6), small(-9, 9, B, 4))
    yield 'bool_ops', (lambda a, b: (a & b, a | b, a ^ b, ~a, j.where(a, b, ~b), jax.lax.select_n(a.astype(j.int32), b, a), a.astype(j.int32) + b)), (j.asarray(rng.random((B, 9)) < .5), j.asarray(rng.random((B, 9)) < .5))
    yield 'iota_lane', (lambda x: (j.arange(x.shape[0])[:, None] + x, jax.lax.broadcasted_iota(j.int32, x.shape, 1) * x)), (small(-9, 9, B, 6),)


def bits(x):
    import jax
    import jax.numpy as j
    x = j.asarray(x)
    if x.dtype == j.bool_:return np.asarray(x)
    width = {1: np.uint8, 2: np.uint16, 4: np.uint32}[x.dtype.itemsize]
    return np.asarray(x).view(width)


def main(argv=None):
    import jax
    from gpu_engine import v2e_gen
    rng = np.random.default_rng(0);ok = True;B = int((argv or sys.argv[1:] or ['4099'])[0])
    for name, fn, args in cases(rng, B):
        try:
            gen = v2e_gen.compile_function(f'conf_{name}', fn, args)
            got, err = gen(*args)
            want = jax.jit(fn)(*args)
            diffs = [int(np.sum(np.any((bits(g) != bits(w)).reshape(B, -1), axis=1))) for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want))]
            res = dict(case=name, outputs=len(diffs), lanes_differing=diffs, err=int(np.asarray(err)[0]), arena_words=gen.module.arena_words,
                equal=all(d == 0 for d in diffs) and int(np.asarray(err)[0]) == 0)
        except Exception as error:
            res = dict(case=name, equal=False, exception=f'{type(error).__name__}: {str(error)[:1500]}')
        ok &= res['equal'];print(json.dumps(res), flush=True)
    print(json.dumps(dict(all_equal=bool(ok))))
    return 0 if ok else 1


if __name__ == '__main__':raise SystemExit(main())
