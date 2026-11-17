"""Compare ordered public-state leaves against a separately checked-out baseline.

Run ``snapshot --out FILE`` in each tree, then ``compare --base FILE --head FILE``.
Only the two named provenance leaves may be added. Snapshots retain byte hashes,
shape, dtype and ordering for every old leaf, and check new values against the host.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def snapshot(*, device_only=False):
    import jax
    import numpy as np
    from pokezero import tier2
    import test_gpu_engine_type_provenance as corpus
    from test_gpu_engine_device_public_hooks import state, tables
    from gpu_engine import engine_device_public_hooks as hooks
    import jax.numpy as j

    cls = corpus.TypeProvenanceTests
    cls.setUpClass()
    fixture = cls()
    syntax = ast.parse(Path(corpus.__file__).read_text())
    cases = next(ast.literal_eval(n.value) for n in ast.walk(syntax)
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cases' for t in n.targets))
    rows = []

    def record(name, value, path, expected):
        # Hook Context is a mutable trace-local builder, not a registered pytree.
        if name.startswith('device/'):
            value = hooks.Initialized(s, value.trackers)
        leaves = []
        for key, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
            a = np.asarray(leaf)
            assert not a.dtype.hasobject, (name, key, type(leaf))
            label = jax.tree_util.keystr(key)
            row = dict(path=label, shape=list(a.shape), dtype=str(a.dtype),
                       sha256=hashlib.sha256(a.tobytes()).hexdigest())
            if label == path:
                assert a.dtype == np.bool_ and a.tolist() == expected, (name, a, expected)
                row['value'] = a.tolist()
            leaves.append(row)
        rows.append(dict(name=name, allowed_new_path=path, expected=expected, leaves=leaves))

    for name, events in (() if device_only else cases.items()):
        host = tier2._IncrementalContextFold()
        lines = list(fixture.prefix)
        for i, event in enumerate([*events, '|switch|p2a: Spare|Absol, L80|100/100']):
            lines.append(event)
            host.process(lines)
            result, _ = fixture.run_lines(lines)
            record(f'b2/{name}/{i}', result, '.context.types.type_changed',
                   [[host.type_changed['p1'], host.type_changed['p2']]])

    s = state()._replace(types=j.broadcast_to(j.array([1, 0], j.int32), (2, 2, 6, 2)))
    c = hooks.initialize(s, tables())
    record('device/initialize', c, '.trackers.type_changed', [[False, False], [False, False]])
    fire = s._replace(types=s.types.at[:, 1, 0].set(j.array([2, 0], j.int32)))
    c = hooks.color_change(c, s, fire, 1, j.array([True, False]))
    record('device/fire', c, '.trackers.type_changed', [[False, True], [False, False]])
    c = hooks.color_change(c, fire, s, 1, j.array([True, True]))
    record('device/native_restore', c, '.trackers.type_changed', [[False, True], [False, True]])
    dead = s._replace(hp=s.hp.at[0, 1, 0].set(0))
    c = hooks.faint_clear(c, s, dead, j.ones(2, bool))
    record('device/faint', c, '.trackers.type_changed', [[False, False], [False, True]])
    c = hooks.install(c, s, 1, j.array([False, True]))
    record('device/switch', c, '.trackers.type_changed', [[False, False], [False, False]])
    return rows


def compare(base, head):
    assert len(base) == len(head) and len(base) > 5
    compared = 0
    for old, new in zip(base, head):
        assert old['name'] == new['name']
        path = old['allowed_new_path']
        assert path == new['allowed_new_path']
        assert path in ('.context.types.type_changed', '.trackers.type_changed')
        assert not any(x['path'] == path for x in old['leaves'])
        added = [x for x in new['leaves'] if x['path'] == path]
        assert len(added) == 1 and added[0]['value'] == new['expected'] == old['expected']
        normalized = [x for x in new['leaves'] if x['path'] != path]
        assert old['leaves'] == normalized, 'old leaf bytes, metadata or order changed: ' + old['name']
        compared += len(normalized)
    return dict(passed=True, snapshots=len(base), old_leaves_compared=compared,
                normalized_paths=['.context.types.type_changed', '.trackers.type_changed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    snap = sub.add_parser('snapshot'); snap.add_argument('--out', type=Path, required=True)
    snap.add_argument('--device-only', action='store_true')
    check = sub.add_parser('compare')
    check.add_argument('--base', type=Path, required=True); check.add_argument('--head', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'snapshot':
        rows = snapshot(device_only=args.device_only)
        args.out.write_text(json.dumps(rows) + '\n')
        print(json.dumps(dict(snapshots=len(rows), out=str(args.out))))
    else:
        print(json.dumps(compare(json.loads(args.base.read_text()), json.loads(args.head.read_text()))))
