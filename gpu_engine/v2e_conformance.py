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
    edge_f = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e-40, -1e-40, 2.5, -2.5, 0.5, 3e9, 714.0], np.float32)
    nan_b = np.array([0x7fc00000, 0xffc00000, 0x7f800001, 0xff812345], np.uint32).view(np.float32)
    edge_f = np.concatenate([edge_f, nan_b])
    def cross_f(B, first):
        # Every ordered pair of edge floats (so -0 meets +0 and NaN payloads meet each other) in the first lanes.
        n = len(edge_f);x = np.array(f32(B, n * n)).reshape(-1)
        x[:n * n] = np.repeat(edge_f, n) if first else np.tile(edge_f, n)
        return j.asarray(x.reshape(B, n * n))
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
    yield ('float_cross', (lambda a, b: (a + b, a - b, a * b, a / b, j.maximum(a, b), j.minimum(a, b), jax.lax.rem(a, b), j.fmod(a, b),
        a < b, a == b, a >= b, jax.lax.clamp(j.float32(-1.0), a, j.float32(1.0)), jax.lax.clamp(b, a, b),
        jax.lax.round(a, jax.lax.RoundingMethod.AWAY_FROM_ZERO), jax.lax.round(a, jax.lax.RoundingMethod.TO_NEAREST_EVEN))),
        (cross_f(B, True), cross_f(B, False)))
    yield ('narrow_ints', (lambda x, y, k: (x.astype(j.int16) * y.astype(j.int16), x.astype(j.uint16) * y.astype(j.uint16),
        x.astype(j.int8) + y.astype(j.int8), jax.lax.clz(x.astype(j.uint8)), jax.lax.clz(x.astype(j.int16)), jax.lax.clz(x),
        jax.lax.select_n(k, x, y, x ^ y), jax.lax.population_count(x.astype(j.int8)))),
        (edgy(B, 14), edgy2(B, 14), small(0, 3, B, 14)))
    c1 = rng.standard_normal(324).astype(np.float32);c2 = (rng.standard_normal(324) * 7).astype(np.float32);ones = np.ones(324, np.float32)
    e1 = rng.standard_normal(325).astype(np.float32);e2 = (rng.standard_normal(325) * 3).astype(np.float32)
    nanc = np.resize(np.array([0x7fc00001, 0xffc00005, 0x7f800003, 0x3f800000], np.uint32).view(np.float32), 324)
    yield ('constant_folds', (lambda a, b: (a + j.where(False, 1.0, 0.0), a * ones, a + (j.float32(0.5) - j.float32(0.5)),
        a * (j.float32(1.0) / j.float32(3.0) * j.float32(3.0)), j.broadcast_to(j.asarray(c1) / j.asarray(c2), a.shape),
        a + j.asarray(c1) / j.asarray(c2), -(-j.where(a > b, a, b)), -(-(a * 2.0)),
        a * (jax.lax.optimization_barrier(j.float32(1.0)) / j.float32(3.0)), a + (j.asarray(e1)[1:] / j.asarray(e2)[1:]),
        a + j.concatenate([j.asarray(c1)[:4], j.asarray(c2)[4:]]) / j.float32(3.0), j.broadcast_to(-j.asarray(nanc), a.shape),
        a + (j.float32(np.inf) - j.float32(np.inf)), j.broadcast_to(j.asarray(nanc) * j.float32(2.0), a.shape))),
        (cross_f(B, True), cross_f(B, False)))
    qn, sn, nn = (np.array([w], np.uint32).view(np.float32)[0] for w in (0x7fc00001, 0x7f800003, 0xffc00005))
    yield ('nan_constants', (lambda a: (a + qn, a * qn, qn - a, a / qn, qn / a, a + sn, sn * a, a - nn, j.maximum(a, qn), j.minimum(qn, a),
        j.maximum(a, sn), a + j.float32(np.nan))), (cross_f(B, True),))
    sn_lit = np.array([0x7f800003], np.uint32).view(np.float32)[0]
    yield ('iota_and_lane_folds', (lambda a, b: (a + j.arange(324, dtype=j.float32) / 7.0, a + 1.0 / (j.arange(324, dtype=j.float32) + 1.0),
        j.broadcast_to(j.asarray(nanc), a.shape) * 2.0, j.floor(j.broadcast_to(j.asarray(nanc), a.shape)), a + j.broadcast_to(j.asarray(c1), a.shape) / 3.0,
        j.where(a > 0, a, sn_lit), jax.jit(lambda t: -t)(-j.where(a > b, a, b)),
        a.reshape(a.shape[0], 2, 162) + j.stack([j.asarray(c1[:162]), j.asarray(c1[162:])]) / 3.0,
        a + j.arange(a.shape[0], dtype=j.float32)[:, None] / 3.0)), (cross_f(B, True), cross_f(B, False)))
    keys = j.asarray(rng.integers(0, 2**32, (B, 2), dtype=np.uint64).astype(np.uint32))
    yield ('prng', (lambda k: (jax.vmap(lambda q: jax.random.bits(q, (5,)))(k), jax.vmap(lambda q: jax.random.split(q, 3))(k),
        jax.vmap(lambda q: jax.random.fold_in(q, 7))(k), jax.vmap(lambda q: jax.random.randint(q, (4,), -5, 40))(k),
        jax.random.bits(j.array([1, 2], j.uint32), (k.shape[0], 3)))), (keys,))
    # An in-place scatter must not clobber a value CSE would hand to a later identical expression.
    yield ('cse_inplace', (lambda x, s: (lambda c: (c.at[rows(x), s].set(0), x * 2, (x * 2)[:, :5]))(x * 2)), (small(-50, 50, B, 64), small(0, 64, B)))
    yield ('sort_layout', (lambda x, y, z: (j.sort(x.reshape(x.shape[0], 4, 10), axis=1), j.argsort(x.reshape(x.shape[0], 4, 10), axis=1),
        j.sort(y, axis=-1), jax.lax.sort((y, z), dimension=1, num_keys=1)[1])), (small(0, 5, B, 40), small(-3000, 3000, B, 1300), small(-9, 9, B, 1300)))
    yield ('small_combiners', (lambda x, f, s, a, v: (x.at[rows(x), s, a].add(v), x.at[rows(x), s, a].max(v), f.at[rows(f), s, a].min(v.astype(j.float32)),
        f.at[rows(f), s, a].add(v.astype(j.float32)), x.at[rows(x), s].add(v[:, None]))),
        (small(-50, 50, B, 2, 6), edgy_f(B, 2, 6), small(-1, 3, B), small(-7, 8, B), small(-9, 9, B)))
    yield ('float_identities', (lambda a, b: (j.stack([a, b], -1).sum(-1), a + 0.0, a * 1.0, a - 0.0, 0.0 + a, a * -1.0, a / 1.0,
        j.cumsum(a[:, :1], axis=-1), j.cumsum(b[:, :1].astype(j.int32), axis=-1), a + (-0.0), a - (-0.0), a / 4.0, a * 0.25, -(-a),
        j.maximum(a, a), a + a, a * 2.0, j.minimum(a, j.float32(np.inf)), j.maximum(a, j.float32(-np.inf)), j.sum(j.stack([a, b], -1) * 1.0, -1),
        j.maximum(a, j.float32(np.inf)), j.minimum(a, j.float32(-np.inf)), j.abs(j.abs(a)), j.abs(-a), (a + 0.0) * 1.0, -(-(-a)), j.minimum(a, a) - 0.0)),
        (cross_f(B, True), cross_f(B, False)))
    # Large per-lane values: warp reductions/scans, chunked scatters with duplicates, lazy broadcasts.
    yield ('large_reduce_scan', (lambda x, m, f: (j.sum(x, axis=-1), j.max(x, axis=-1), j.min(x.reshape(x.shape[0], 3, 100), axis=-1), j.any(m, axis=-1),
        j.all(m.reshape(m.shape[0], 4, 75), axis=-1), j.argmax(x, axis=-1), j.argmin(x.reshape(x.shape[0], 3, 100), axis=-1), j.cumsum(x, axis=-1),
        j.cumsum(x[:, ::-1], axis=-1), jax.lax.cumsum(x, axis=1, reverse=True), j.max(f[:, 3:4], axis=-1), j.argmax(f[:, 3:4], axis=-1), j.sum(f[:, 5:6], axis=-1),
        j.sum(f[:, 4:6], axis=-1))), (small(-4, 4, B, 300), j.asarray(rng.random((B, 300)) < .995), edgy_f(B, 300)))
    perm = j.asarray(np.stack([rng.permutation(230)[:150] for _ in range(B)]).astype(np.int32))
    yield ('large_scatter', (lambda x, ix, v, f, u: (x.at[rows(x)[:, None], u].set(v[:, :150]), x.at[rows(x)[:, None], ix].add(v), x.at[rows(x)[:, None], ix].max(v),
        j.zeros_like(f).at[rows(f)[:, None], u].add(v[:, :150].astype(j.float32), unique_indices=True), f.at[rows(f)[:, None], ix].min(v.astype(j.float32) * 0.5),
        x.at[rows(x)[:, None], u].set(v[:, :150], mode='drop'), x[:, :20].at[rows(x)[:, None], u[:, :30]].set(v[:, :30], mode='drop'))),
        (small(-50, 50, B, 200), small(-5, 210, B, 300), small(-9, 9, B, 300), edgy_f(B, 200), perm))
    yield ('float_sort', (lambda f, k: (j.sort(f, axis=-1), j.argsort(f, axis=-1), jax.lax.sort((k, f, k), dimension=1, num_keys=2)[2], j.sort(f[:, :20], axis=-1))),
        (edgy_f(B, 64), small(0, 3, B, 64)))
    yield ('lazy_broadcast', (lambda s, x, y: (s[:, None] * x, j.where(s[:, None] > 0, x, y[:, :1]), (j.arange(100) * s[:, None]) + x, s[:, None, None] + x.reshape(x.shape[0], 4, 25))),
        (small(-9, 9, B), small(-9, 9, B, 100), small(-9, 9, B, 100)))


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
    argv = list(argv or sys.argv[1:]);mode = 'warp'
    if '--mode' in argv:mode = argv.pop(argv.index('--mode') + 1);argv.remove('--mode')
    rng = np.random.default_rng(0);ok = True;B = int((argv or ['4099'])[0])
    for name, fn, args in cases(rng, B):
        try:
            gen = v2e_gen.compile_function(f'conf_{mode}_{name}', fn, args, mode=mode)
            got, err = gen(*args)
            want = jax.jit(fn)(*args)
            diffs = [int(np.sum(np.any((bits(g) != bits(w)).reshape(B, -1), axis=1))) for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want))]
            res = dict(case=name, mode=mode, outputs=len(diffs), lanes_differing=diffs, err=int(np.asarray(err)[0]), arena_words=gen.module.arena_words,
                equal=all(d == 0 for d in diffs) and int(np.asarray(err)[0]) == 0)
            for k, (g, w) in enumerate(zip(jax.tree.leaves(got), jax.tree.leaves(want))):
                gb, wb = bits(g).reshape(B, -1), bits(w).reshape(B, -1)
                bad = np.argwhere(gb != wb)
                if len(bad):
                    lane, col = map(int, bad[0])
                    res.setdefault('first', {})[k] = dict(lane=lane, element=col, got=hex(int(gb[lane, col])), want=hex(int(wb[lane, col])))
        except Exception as error:
            res = dict(case=name, equal=False, exception=f'{type(error).__name__}: {str(error)[:1500]}')
        ok &= res['equal'];print(json.dumps(res), flush=True)
    print(json.dumps(dict(all_equal=bool(ok))))
    return 0 if ok else 1


if __name__ == '__main__':raise SystemExit(main())
