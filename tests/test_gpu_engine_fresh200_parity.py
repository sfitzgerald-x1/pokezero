"""Strict array rules only; these synthetic arrays never qualify a corpus."""
import copy
import hashlib
from types import SimpleNamespace
import unittest

import numpy as np

from gpu_engine import encoder_parity_policy as policy, fresh200_parity as parity

NAMESPACE = dict(categorical_vocab_size=17, token_type_vocab_size=3, device_token_type_vocab_size=5)


class StrictArrayParityTests(unittest.TestCase):
    def setUp(self):
        self.python = dict(numeric_features=np.full((200, 23, 132), .5, np.float64),
            categorical_ids=np.zeros((200, 23, 41), np.int64), token_type_ids=np.zeros((200, 23), np.int64),
            attention_mask=np.ones((200, 23), bool), legal_action_mask=np.ones((200, 9), bool))
        self.b2 = {name:value.astype(np.float32 if name == 'numeric_features' else np.int32 if value.dtype.kind in 'iu' else bool)
            for name, value in self.python.items()}
        self.device = dict(numeric_features=np.zeros((200, 29, 132), np.float32),
            categorical_ids=np.zeros((200, 29, 41), np.int32), token_type_ids=np.zeros((200, 29), np.int32),
            attention_mask=np.zeros((200, 29), bool), legal_action_mask=self.b2['legal_action_mask'].copy(), error=np.zeros(200, np.int32))
        for name, value in self.b2.items():
            if name != 'legal_action_mask':self.device[name][:, :23] = value
        self.device['token_type_ids'][:, 23:] = 4

    def compare(self):
        return parity.compare(self.device, self.b2, self.python, namespace=NAMESPACE, policy=policy)

    def test_exact_arrays_preserve_full_device_and_original_float64(self):
        report, converted = self.compare()
        self.assertTrue(report['array_rules_passed']);self.assertFalse(report['qualification'])
        self.assertEqual(report['full_device']['numeric_features']['shape'], [200, 29, 132])
        self.assertEqual(report['extra_device_tokens']['tokens'], 6)
        self.assertEqual(report['original_python']['numeric_features']['dtype'], self.python['numeric_features'].dtype.str)
        self.assertEqual(converted.dtype, np.dtype('float32'));self.assertFalse(converted.flags.writeable)

    def test_exact_float64_bytes_bind_the_separate_float32_conversion(self):
        self.python['numeric_features'][0, 0, 0] = 1/3
        self.b2['numeric_features'][0, 0, 0] = np.float32(1/3)
        self.device['numeric_features'][0, 0, 0] = np.float32(1/3)
        original = self.python['numeric_features'].tobytes()
        report, converted = self.compare()
        self.assertTrue(report['array_rules_passed'])
        self.assertEqual(report['python_conversion']['input']['sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(report['python_conversion']['output']['sha256'], hashlib.sha256(converted.tobytes()).hexdigest())
        self.assertEqual(self.python['numeric_features'].tobytes(), original)

    def test_only_the_existing_four_columns_admit_one_ulp(self):
        for column in (38, 39, 115, 116):self.device['numeric_features'].view(np.uint32)[0, 0, column] += 1
        report, _ = self.compare();self.assertTrue(report['array_rules_passed'])
        self.assertEqual(report['pairs']['device_python']['cumulative_one_ulp_cells'], 4)
        self.device['numeric_features'].view(np.uint32)[0, 0, 37] += 1
        report, _ = self.compare();self.assertFalse(report['array_rules_passed'])
        self.assertEqual(report['pairs']['device_python']['first_numeric_difference']['column'], 37)

    def test_pairwise_check_rejects_references_on_opposite_sides(self):
        self.device['numeric_features'].view(np.uint32)[0, 0, 38] += 1
        self.b2['numeric_features'].view(np.uint32)[0, 0, 38] -= 1
        report, _ = self.compare()
        self.assertTrue(report['pairs']['device_python']['array_rules_passed'])
        self.assertTrue(report['pairs']['b2_python']['array_rules_passed'])
        self.assertFalse(report['pairs']['device_b2']['array_rules_passed']);self.assertFalse(report['array_rules_passed'])

    def test_signed_zero_changes_are_fatal_even_in_cumulative_columns(self):
        for record in (self.python, self.b2, self.device):record['numeric_features'][0, 0, 38] = 0
        self.device['numeric_features'].view(np.uint32)[0, 0, 38] = 0x80000000
        report, _ = self.compare();self.assertFalse(report['array_rules_passed'])
        self.assertEqual(report['pairs']['device_python']['cumulative_one_ulp_cells'], 0)

    def test_nonfinite_original_or_device_and_overflow_conversion_are_fatal(self):
        for target, value in ((self.python, np.nan), (self.device, np.inf)):
            target['numeric_features'][0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, 'nonfinite'):self.compare()
            target['numeric_features'][0, 0, 0] = .5
        self.python['numeric_features'][0, 0, 0] = 1e300
        with self.assertRaisesRegex(ValueError, 'nonfinite Python float32 conversion'):self.compare()

    def test_full_masks_padding_and_categories_compare_exactly(self):
        self.device['categorical_ids'][199, 22, 40] = 1
        self.device['attention_mask'][199, 22] = False
        self.device['legal_action_mask'][199, 8] = False
        report, _ = self.compare();bad = report['pairs']['device_python']['other_bad_cells']
        self.assertEqual(bad['categorical_ids'], 1);self.assertEqual(bad['attention_mask'], 1);self.assertEqual(bad['legal_action_mask'], 1)
        self.assertFalse(report['array_rules_passed'])

    def test_no_denominator_dtype_namespace_error_or_policy_relaxation(self):
        changed = copy.deepcopy(self.device);changed['numeric_features'] = changed['numeric_features'][:199]
        with self.assertRaises(ValueError):parity.compare(changed, self.b2, self.python, namespace=NAMESPACE, policy=policy)
        changed = copy.deepcopy(self.device);changed['attention_mask'] = changed['attention_mask'].astype(np.int32)
        with self.assertRaises(ValueError):parity.compare(changed, self.b2, self.python, namespace=NAMESPACE, policy=policy)
        changed = copy.deepcopy(self.device);changed['categorical_ids'][0, 0, 0] = 17
        with self.assertRaisesRegex(ValueError, 'namespace'):parity.compare(changed, self.b2, self.python, namespace=NAMESPACE, policy=policy)
        changed = copy.deepcopy(self.device);changed['error'][0] = 1
        with self.assertRaisesRegex(ValueError, 'errors are fatal'):parity.compare(changed, self.b2, self.python, namespace=NAMESPACE, policy=policy)
        changed_policy = SimpleNamespace(cumulative_columns=lambda: {37:'unapproved'})
        with self.assertRaisesRegex(ValueError, 'policy changed'):
            parity.compare(self.device, self.b2, self.python, namespace=NAMESPACE, policy=changed_policy)


if __name__ == '__main__':unittest.main()
