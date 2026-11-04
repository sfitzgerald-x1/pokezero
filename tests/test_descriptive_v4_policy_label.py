import unittest
from pokezero.b4_device_canary_runtime import _checked_model_hash
from pokezero.live_protocol_actor import _sha,_canonical


class DescriptivePolicyLabelTests(unittest.TestCase):
    def test_only_declared_label_differs_and_native_hash_survives(self):
        original=dict(policy_id='reference',shape=[23,132,41],vocab=['a','b'])
        native=dict(original,policy_id='earlier')
        expected=_sha(_canonical(original))
        self.assertEqual(_checked_model_hash(native,expected,('earlier','reference')),_sha(_canonical(native)))
        self.assertEqual(native['policy_id'],'earlier')
        for changed in (dict(native,shape=[29,132,41]),dict(native,vocab=['b','a'])):
            with self.assertRaises(ValueError):_checked_model_hash(changed,expected,('earlier','reference'))

    def test_original_default_is_strict(self):
        config=dict(policy_id='reference',shape=[23,132,41]);expected=_sha(_canonical(config))
        self.assertEqual(_checked_model_hash(config,expected),expected)
        with self.assertRaises(ValueError):_checked_model_hash(dict(config,policy_id='earlier'),expected)

    def test_alias_must_match_actual_and_explicit_reference(self):
        native=dict(policy_id='earlier');expected=_sha(_canonical(dict(policy_id='reference')))
        for aliases in (('other','reference'),('earlier','wrong'),('earlier','earlier'),['earlier','reference'],('', 'reference')):
            with self.subTest(aliases=aliases),self.assertRaises(ValueError):_checked_model_hash(native,expected,aliases)
