"""Prospective held-D endpoint controls; fake artifacts, no model or games."""
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero import v5_endpoint as endpoint, v5_evaluation as evaluation, v5_foulplay as fp
from test_v5_endpoint import artifact, fixture, source_result, complete_rows


def held_fixture(root):
    _, original, original_configs = fixture(root)
    protocol = dict(original, schema=endpoint.HELD_SCHEMA, held_arms=dict(endpoint.HELD_D),
                    arms=[arm for arm in original['arms'] if arm['name'] != 'D0'])
    reference = evaluation.Artifact(**artifact(root / 'held-recipe.json', protocol))
    protocol, configs = endpoint._protocol(reference, parse_configs=True)
    return reference, protocol, configs, original, original_configs


class HeldEndpointTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(endpoint.TRAINING_GAMES, 2_000_000)
        budget = patch.object(endpoint, 'TRAINING_GAMES', 1)
        budget.start()
        self.addCleanup(budget.stop)

    def test_all_original_ordinals_seeds_and_core_rotation_survive_held_D(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref, protocol, configs, original, original_configs = held_fixture(root)
            document = evaluation.Artifact.document
            def no_D_artifact(reference):
                if '/D0/' in reference.path:
                    raise AssertionError('held D must not require its final artifacts')
                return document(reference)
            with patch.object(evaluation.Artifact, 'document', no_D_artifact):
                endpoint._protocol(ref, parse_configs=True)
            old = endpoint._rows(original, original_configs)
            rows = endpoint._rows(protocol, configs)
            identity = ('ordinal', 'arm', 'offset', 'seed', 'seat', 'fp_seed', 'fp_core', 'slot')
            self.assertEqual(len(rows), 8000)
            self.assertEqual([{k: r[k] for k in identity} for r in rows],
                             [{k: r[k] for k in identity} for r in old])
            self.assertEqual(sum(r['status'] == 'HELD' for r in rows), 2000)
            self.assertTrue(all(r['result'] is None and r['invocation_id'] is None for r in rows))
            self.assertEqual(tuple(configs.models), ('A0', 'B0', 'B1'))

    def test_only_the_explicit_prospective_D_hold_contract_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, protocol, _, _, _ = held_fixture(root)
            mutations = (
                dict(protocol, schema=endpoint.SCHEMA),
                dict(protocol, held_arms={}),
                dict(protocol, held_arms={'A0': 'D_T_NOT_ADMITTED'}),
                dict(protocol, held_arms={'D0': 'bad-after-outcome-selection'}),
                dict(protocol, arms=protocol['arms'][:-1]),
                dict(protocol, arms=list(reversed(protocol['arms']))),
            )
            for index, changed in enumerate(mutations):
                reference = evaluation.Artifact(**artifact(root / f'bad-{index}.json', changed))
                with self.subTest(index=index), self.assertRaises(ValueError):
                    endpoint._protocol(reference)

    def test_resume_skips_held_ordinals_without_replaying_completed_or_lost_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref, protocol, configs, _, _ = held_fixture(root)
            store = endpoint.Store(root / 'endpoint', ref, protocol, configs)
            for ordinal in (0, 1):
                store.emit('START', ordinal, 'first')
                store.emit('END', ordinal, 'first', result=source_result(store.rows[ordinal], protocol, configs))
            with self.assertRaises(ValueError):
                store.emit('START', 2, 'first')
            store.emit('START', 3, 'first')
            resumed = endpoint.Store(root / 'endpoint', ref, protocol, configs, resume=True)
            resumed.interrupt_older('second')
            self.assertEqual([r['status'] for r in resumed.rows[:5]],
                             ['COMPLETE', 'COMPLETE', 'HELD', 'INTERRUPTED', 'NOT_ATTEMPTED'])
            resumed.emit('START', 4, 'second')
            with self.assertRaises(ValueError):
                resumed.emit('START', 3, 'second')
            self.assertEqual([e['ordinal'] for e in resumed.events if e['kind'] == 'START'], [0, 1, 3, 4])
            report = resumed.report('second')
            self.assertEqual(report['schema'], endpoint.HELD_SCHEMA)
            self.assertEqual(report['counts']['HELD'], 2000)

    def test_fake_runner_never_requires_D_factory_or_claims_D_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref, protocol, configs, _, _ = held_fixture(root)
            store = endpoint.Store(root / 'endpoint', ref, protocol, configs)
            factories = {a: object() for a in ('A0', 'B0', 'B1')}
            calls = []
            async def panel(config, factory, **kwargs):
                arm = next(a for a, f in factories.items() if f is factory)
                row = next(r for r in store.rows if r['arm'] == arm and
                           r['offset'] == kwargs['game_offsets'][0] and r['seat'] == config.pokezero_player)
                calls.append(row['ordinal'])
                await asyncio.sleep(0)
                result = source_result(row, protocol, configs, status='FAILED', reason='TimeoutError: unit')
                report = dict(rows=[result['row']], recipe=result['recipe'], namespace=result['namespace'],
                              invocation_id=result['invocation_id'])
                kwargs['progress_callback'](report)
                return report
            with patch.object(fp, '_recipe', return_value={}), patch.object(fp, 'run_panel', side_effect=panel):
                report = asyncio.run(endpoint.execute_cells(store, factories, invocation='one'))
            self.assertEqual(calls, [0, 1, 3])
            self.assertEqual(report['counts']['FAILED'], 3)
            self.assertEqual(report['counts']['NOT_ATTEMPTED'], 5997)
            self.assertEqual(report['counts']['HELD'], 2000)
            self.assertTrue(all(r['invocation_id'] is None for r in store.rows if r['arm'] == 'D0'))

    def test_active_contrasts_identical_and_D_remains_full_denominator_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, protocol, configs, original, original_configs = held_fixture(root)
            score = lambda row: 1. if row['arm'] in ('B0', 'D0') else .5
            baseline = endpoint.reduce_endpoint(original, complete_rows(original, original_configs, score), original_configs)
            rows = endpoint._rows(protocol, configs)
            for row in rows:
                if row['arm'] != 'D0':
                    row.update(status='COMPLETE', result=source_result(row, protocol, configs, score(row)), invocation_id='unit')
            result = endpoint.reduce_endpoint(protocol, rows, configs)
            for contrast in ('B0-A0', 'B1-B0'):
                self.assertEqual(result['contrasts'][contrast], baseline['contrasts'][contrast])
            self.assertEqual((result['declared_cells'], result['attemptable_declared_cells'], result['held_declared_cells']),
                             (8000, 6000, 2000))
            self.assertEqual(result['arms']['D0']['known_completed'], 0)
            self.assertEqual(result['arms']['D0']['unknown_cells'], 2000)
            self.assertEqual(result['arms']['D0']['full_roster_identification_bounds'], [0., 1.])
            d = result['contrasts']['D0-B0']
            self.assertEqual(d['status'], 'HELD_UNMEASURED')
            self.assertEqual(d['unknown_cells_across_arms'], 2000)
            self.assertEqual(d['paired_seed_bootstrap_90'], [-1., 0.])
            self.assertFalse(d['supported_positive_primary'])

    def test_active_and_held_cells_cannot_be_reclassified_or_given_fake_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, protocol, configs, _, _ = held_fixture(root)
            for mutation in ('active-held', 'held-failed', 'held-source', 'held-claim'):
                rows = endpoint._rows(protocol, configs)
                if mutation == 'active-held': rows[0]['status'] = 'HELD'
                if mutation == 'held-failed': rows[2]['status'] = 'FAILED'
                if mutation == 'held-source': rows[2]['result'] = {'fake': True}
                if mutation == 'held-claim': rows[2]['invocation_id'] = 'claimed'
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    endpoint.reduce_endpoint(protocol, rows, configs)


if __name__ == '__main__':
    unittest.main()
