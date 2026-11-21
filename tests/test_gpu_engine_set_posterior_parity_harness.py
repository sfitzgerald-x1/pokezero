"""The parity receipt must reject incomplete evidence rather than print a zero-mismatch pass."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import set_posterior_device_parity as parity


class EvidenceTests(unittest.TestCase):
    def test_no_selected_files_fails_before_loading_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'result'
            with self.assertRaisesRegex(ValueError, 'no corpus files'):
                parity.main(['--corpus', directory, '--metadata', 'unused', '--out', str(out)])
            self.assertFalse(out.exists())

    def test_missing_requested_shard_and_duplicate_inputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / 'strikes-00.jsonl'
            shard.write_text('{}\n')
            self.assertEqual(parity.shard_files([directory], {0}), [shard])
            with self.assertRaisesRegex(ValueError, 'missing requested shards'):
                parity.shard_files([directory], {0, 1})
            with self.assertRaisesRegex(ValueError, 'duplicate corpus files'):
                parity.shard_files([directory, str(shard)], None)

    def test_constants_must_match_and_be_available(self):
        names = ('FIXED_DAMAGE_MOVES', 'OHKO_MOVES', 'CONDITIONAL_POWER_MOVES', 'MULTI_HIT_MOVES')
        host = SimpleNamespace(**{name: frozenset([name]) for name in names})
        device = SimpleNamespace(**vars(host))
        self.assertTrue(all(parity.constant_parity(host, device).values()))
        device.MULTI_HIT_MOVES = frozenset()
        with self.assertRaisesRegex(ValueError, 'move classes differ'):
            parity.constant_parity(host, device)
        with self.assertRaises(AttributeError):parity.constant_parity(SimpleNamespace(), device)

    def test_empty_or_parse_only_corpus_does_not_certify_kernel(self):
        check = parity.Check(10)
        parity.check_coverage(check)
        self.assertEqual(check.stats['mismatch:missing-coverage'], 4)
        check = parity.Check(10)
        check.stats['rows'] = 1
        parity.check_coverage(check)
        self.assertEqual(check.stats['mismatch:missing-coverage'], 3)
        complete = parity.Check(10)
        complete.stats.update({'rows': 4, 'core-rows': 2, 'variant-entries': 2, 'supports-root': 1})
        parity.check_coverage(complete)
        self.assertFalse(any(k.startswith('mismatch:') for k in complete.stats))

    def test_unmapped_core_input_fails_and_cannot_claim_support(self):
        row = dict(ordinal=1, line=2, attacker_species='missing', decision_index=3,
                   admitted=True, reason='', move='tackle', variants=[])
        mapper = SimpleNamespace(inputs=lambda row: ({}, ['unmapped-species']))
        kernel = SimpleNamespace(dsp=SimpleNamespace(), strike=lambda fields: {})
        check = parity.Check(10)
        parity.check_shard([row], kernel, mapper, {}, check, {})
        self.assertEqual(check.stats['mismatch:core-inputs'], 1)
        self.assertEqual(check.stats['mismatch:support-inputs-missing'], 1)
        self.assertEqual(check.stats['supports-root'], 0)

    def test_empty_shard_main_returns_nonzero_receipt(self):
        import builtins
        import pokezero
        from gpu_engine import device_set_posterior as dsp, public_belief
        from pokezero import dex, randbat
        import numpy as np
        table = SimpleNamespace(**{n: np.zeros(1, np.int32) for n in
            ('level', 'stats', 'ability', 'item', 'move_class', 'physical', 'hp_type', 'binding')},
            variable=SimpleNamespace(special_type=np.zeros(1, bool)))
        constants = SimpleNamespace(**{n: getattr(dsp, n) for n in
            ('FIXED_DAMAGE_MOVES', 'OHKO_MOVES', 'CONDITIONAL_POWER_MOVES', 'MULTI_HIT_MOVES')})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);(root/'strikes-00.jsonl').write_text('');(root/'metadata.json').write_text('{}')
            with mock.patch.object(dex, 'load_showdown_dex_cached', return_value=None), \
                 mock.patch.object(randbat, 'load_gen3_randbat_source_cached', return_value=None), \
                 mock.patch.object(public_belief, 'compile_source', return_value=None), \
                 mock.patch.object(dsp, 'compile_tables', return_value=(table, {})), \
                 mock.patch.object(pokezero, 'set_posterior', constants, create=True), \
                 mock.patch.object(parity, 'Mapper'), mock.patch.object(parity, 'Kernel'), \
                 mock.patch('builtins.print'):
                code = parity.main(['--corpus', directory, '--metadata', str(root/'metadata.json'),
                                    '--out', str(root/'result')])
                original_import = builtins.__import__
                def unavailable(name, globals=None, locals=None, fromlist=(), level=0):
                    if name == 'pokezero' and 'set_posterior' in fromlist:raise ImportError('host unavailable')
                    return original_import(name, globals, locals, fromlist, level)
                with mock.patch('builtins.__import__', side_effect=unavailable):
                    with self.assertRaisesRegex(ImportError, 'host unavailable'):
                        parity.main(['--corpus', directory, '--metadata', str(root/'metadata.json'),
                                     '--out', str(root/'missing-host')])
            import json
            receipt = json.loads((root/'result/summary.json').read_text())
            self.assertEqual(code, 1)
            self.assertEqual(receipt['counts']['mismatch:empty-shard'], 1)
            self.assertEqual(receipt['counts']['mismatch:missing-coverage'], 4)


if __name__ == '__main__':unittest.main()
