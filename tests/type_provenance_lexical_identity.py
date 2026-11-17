"""Empty type payloads admitted by an incidental tendency outcome retain history."""
import argparse
import hashlib
import json
from pathlib import Path


def snapshots():
    import jax
    import numpy as np
    from pokezero import tier2
    from gpu_engine import public_tendency
    from test_gpu_engine_type_provenance import TypeProvenanceTests
    from test_gpu_engine_public_tendency_context import PublicTendencyContextTests

    class Fixture(TypeProvenanceTests):
        initial = PublicTendencyContextTests.initial
        def chunks(self, projections):
            p = projections[0]
            return list(self.i.pack_chunks(np.array([1], np.int32),
                [[event for row in projections for event in row.events]],
                source_binding=self.host.device.binding, **{name: np.asarray(getattr(p, name), np.uint32)
                for name in ('species_binding', 'ability_binding', 'move_binding', 'item_binding',
                             'context_binding', 'field_binding', 'type_binding', 'tendency_binding')}))
    Fixture.setUpClass()
    cls = Fixture
    cls.tendency_catalog = public_tendency.compile_tendency(cls.species_catalog, cls.catalog, cls.vocab)
    cls.contexts = {n: cls.c.compile_context(cls.host, cls.abilities, cls.catalog,
        items=cls.item_catalog, item_belief_narrowing=n, field_catalog=cls.field_catalog,
        type_catalog=cls.type_catalog, tendency_tables=cls.tendency_catalog.device) for n in (False, True)}
    cls.kwargs = dict(cls.kwargs, tendency_catalog=cls.tendency_catalog)
    fixture = cls()
    rows = []
    for prior in (False, True):
        lines = [*fixture.prefix, '|-ability|p2a: Lizard|Water Absorb']
        if prior: lines.append('|-start|p2a: Lizard|typechange|Fire')
        for index, event in enumerate(('|-start|p2a: Lizard|typechange||[from] ability: Water Absorb',
                                      '|-end|p2a: Lizard|typechange',
                                      '|switch|p2a: Spare|Absol, L80|100/100')):
            lines.append(event)
            host = tier2._IncrementalContextFold(); host.process(lines)
            expected = [[host.type_changed['p1'], host.type_changed['p2']]]
            result, _ = fixture.run_lines(lines)
            leaves = []
            for key, leaf in jax.tree_util.tree_flatten_with_path(result)[0]:
                a = np.asarray(leaf); path = jax.tree_util.keystr(key)
                assert not a.dtype.hasobject
                row = dict(path=path, shape=list(a.shape), dtype=str(a.dtype),
                           sha256=hashlib.sha256(a.tobytes()).hexdigest())
                if path == '.context.types.type_changed':
                    assert a.tolist() == expected, (prior, index, a, expected)
                    row['value'] = a.tolist()
                leaves.append(row)
            rows.append(dict(name=f'empty/prior{prior}/{index}', allowed_new_path='.context.types.type_changed',
                             expected=expected, leaves=leaves))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); rows = snapshots()
    args.out.write_text(json.dumps(rows) + '\n'); print(json.dumps(dict(snapshots=len(rows))))
