"""Owning B2-to-root transport snapshots for both owners, without engine games."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import NamedTuple


def snapshots():
    import jax
    import numpy as np
    from gpu_engine import phase_d_live_worlds as live
    from pokezero import tier2, v5_contract
    from test_phase_d_device_worlds import WorldsTests

    class RootLeaves(NamedTuple):
        trackers: object
        inputs: object

    WorldsTests.setUpClass()
    fixture = WorldsTests()
    rows = []
    for owner in (0, 1):
        adapter = fixture.adapter(owner)
        own, peer = f'p{owner+1}', f'p{2-owner}'
        lines = [f'|switch|{own}a: Alpha|Alpha, L80|291/291',
                 f'|switch|{peer}a: Beta|Beta, L80|100/100', '|turn|1']
        host = tier2._IncrementalContextFold()
        stages = [('initial', []), ('native_restore', [
            f'|-start|{peer}a: Beta|typechange|Normal', f'|-end|{peer}a: Beta|typechange']),
            ('switch', [f'|switch|{peer}a: Beta|Beta, L80|100/100'])]
        for index, (name, events) in enumerate(stages):
            if events:
                adapter.feed(tuple(events))
                lines.extend(events)
            host.process(lines)
            expected = [[host.type_changed['p1'], host.type_changed['p2']]]
            root = live.from_b2(adapter, fixture.request(owner), fixture.source, fixture.metadata,
                               address=v5_contract.DecisionAddress(1, index, owner))
            value = RootLeaves(root.public, dict(root.view['_b2_root_inputs']))
            leaves = []
            for key, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
                a = np.asarray(leaf)
                assert not a.dtype.hasobject, (name, key)
                path = jax.tree_util.keystr(key)
                row = dict(path=path, shape=list(a.shape), dtype=str(a.dtype),
                           sha256=hashlib.sha256(a.tobytes()).hexdigest())
                if path == '.trackers.type_changed':
                    assert a.dtype == np.bool_ and a.tolist() == expected, (owner, name, a, expected)
                    row['value'] = a.tolist()
                leaves.append(row)
            rows.append(dict(name=f'root/owner{owner}/{name}', allowed_new_path='.trackers.type_changed',
                             expected=expected, leaves=leaves))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    rows = snapshots()
    args.out.write_text(json.dumps(rows) + '\n')
    print(json.dumps(dict(snapshots=len(rows), out=str(args.out))))
