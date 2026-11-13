"""Replay retained constructor fixtures with explicitly supplied static inputs.

Run on remote compute, once on main with --baseline and once on the PR. No
model forward, new games, hidden opponent state, or source-limit overrides.
"""
import argparse
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor, load_inputs
from gpu_engine.phase_d_constructor_rules import own_pp_carry
from pokezero.live_protocol_actor import _packet


class ConstructorWallReplay(unittest.TestCase):
    constructor = None
    baseline = False
    fixtures = Path(__file__).parent / 'fixtures' / 'phase_d_constructor_walls'
    outcomes = []

    def replay(self, name):
        data = json.loads((self.fixtures / (name + '.json')).read_text())
        request = data['request']
        self.assertEqual(hashlib.sha256(''.join(x + '\n' for x in request['lines']).encode()).hexdigest(), data['prefix_sha256'])
        made = None
        error = None
        try:
            detached, made, legal = self.constructor.prepare(request, data['decision_index'])
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            error = type(exc).__name__ + ':' + str(exc)
        built = 0 if made is None else sum(w.state is not None for w in made.worlds)
        refusals = [] if made is None else [reason for w in made.worlds for reason in w.refusals]
        outcome = dict(fixture=name, baseline=self.baseline, built=built, error=error, refusals=refusals)
        self.outcomes.append(outcome)
        if self.baseline:
            self.assertEqual(built, 0, outcome)
            expected = data['code']
            self.assertTrue(expected in (error or '') or expected in refusals, outcome)
            return
        self.assertIsNone(error, outcome)
        self.assertEqual(built, 16, outcome)
        self.assertEqual(refusals, [], outcome)
        if 'own_pp_carry' in data:
            carry = data['own_pp_carry']
            rows = own_pp_carry(request['lines'], _packet(request['own_request_line'], request['owner']), carry['maximum_pp'])
            self.assertEqual(rows, carry['rows'])
            outcome['pp_rows'] = rows
        owner = int(request['owner'][1]) - 1
        if name == 'unown_forme':
            for world in made.worlds:
                np.testing.assert_array_equal(np.asarray(world.state.species)[0, owner], detached.own['species'])
                diagnostic = next(row for row in made.mapper_diagnostics if row['world'] == world.index)
                checks = [r for r in diagnostic['checks'] if r['kind'] == 'own-request-mapper-disagreement:species']
                self.assertTrue(checks)
                self.assertTrue(all(r['status'] == 'CLEAR' for r in checks))
            outcome['own_species_and_diagnostic'] = 'CLEAR'
        if name == 'passed_substitute':
            values = []
            for world in made.worlds:
                names = [self.constructor.metadata['vocabulary']['species'][int(x)] for x in world.state.species[0, owner]]
                self.assertEqual(names.count('jolteon'), 1)
                passer = names.index('jolteon')
                active = int(world.state.active[0, owner])
                self.assertEqual(names[active], 'pidgeot')
                actual = int(world.state.substitute_hp[0, owner, active])
                expected = int(world.state.max_hp[0, owner, passer]) // 4
                self.assertEqual(actual, expected)
                values.append(actual)
            outcome['substitute_hp'] = values

    def test_pp_arrival(self):
        self.replay('pp_arrival')

    def test_fainted_effect(self):
        self.replay('fainted_effect')

    def test_unown_forme(self):
        self.replay('unown_forme')

    def test_ditto_transform(self):
        self.replay('ditto_transform')

    def test_passed_substitute(self):
        self.replay('passed_substitute')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('metadata', 'vocabulary', 'showdown-root', 'node-binary', 'expected-metadata-sha256', 'expected-vocabulary-sha256'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--fixtures', type=Path, default=ConstructorWallReplay.fixtures)
    args = parser.parse_args()
    catalog, source, metadata, config = load_inputs(args)
    ConstructorWallReplay.constructor = ProtocolConstructor(catalog=catalog, source=source, metadata=metadata,
        owning_config=config, count=16, seed=20261103)
    ConstructorWallReplay.baseline = args.baseline
    ConstructorWallReplay.fixtures = args.fixtures
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ConstructorWallReplay))
    print(json.dumps(dict(outcomes=ConstructorWallReplay.outcomes, passed=result.wasSuccessful()), sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
