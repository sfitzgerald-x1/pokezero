"""Per-column audit retains late columns, identity and exact gate scopes."""
import copy
import unittest

import numpy as np

from gpu_engine.v4_parity import ColumnParity


def pair(identity='d1'):
    record=dict(decision_id=identity,prefix_sha256='e'*64,
        numeric_features=np.zeros((23,132)),categorical_ids=np.zeros((23,41),np.int32),
        token_type_ids=np.array([0]+[1]*6+[2]*6+[3]*9+[5],np.int32),
        attention_mask=np.ones(23,bool),legal_action_mask=np.ones(9,bool),switch_targets=np.arange(5,dtype=np.int32))
    return dict(decision_id=identity,battle_id='b1',seat='p1',turn=1,prefix_sha256='e'*64,
        reference=record,candidate=copy.deepcopy(record))


class ColumnParityTests(unittest.TestCase):
    def audit(self):return ColumnParity(reference_source_sha256='a'*64,encoder_source_sha256='b'*64,vocabulary_sha256='c'*64)

    def test_every_column_token_and_structural_difference_is_preserved(self):
        audit=self.audit();record=pair()
        record['candidate']['numeric_features'][22,131]=1.1e-6
        record['candidate']['numeric_features'][1,7]=1e-6
        record['candidate']['categorical_ids'][12,40]=3
        record['candidate']['attention_mask'][17]=False
        audit.add(**record);report=audit.report()
        self.assertEqual(len(report['columns']),173)
        numbers=report['columns'][:132];categories=report['columns'][132:]
        self.assertEqual(numbers[131]['token_mismatches'][22],1)
        self.assertEqual(numbers[7]['mismatched_cells'],0)
        self.assertEqual(categories[40]['token_mismatches'][12],1)
        self.assertEqual(report['structure_mismatched_decisions']['attention_mask'],1)
        self.assertFalse(report['supplied_tensor_parity_pass']);self.assertFalse(report['recorded_provenance_verified'])

    def test_duplicate_or_cross_prefix_or_partial_forward_refuses_before_mutation(self):
        audit=self.audit();audit.add(**pair());before=audit.report()
        with self.assertRaisesRegex(ValueError,'unique'):audit.add(**pair())
        with self.assertRaisesRegex(ValueError,'canonical'):audit.add(**pair('alias'))
        bad=pair('d2');bad['candidate']['prefix_sha256']='f'*64
        bad['turn']=2
        with self.assertRaisesRegex(ValueError,'identical'):audit.add(**bad)
        bad=pair('d3');bad['reference']['policy_logits']=np.arange(9,dtype=float)
        bad['turn']=3
        bad['reference']['value']=0.
        with self.assertRaises((KeyError,ValueError)):audit.add(**bad)
        self.assertEqual(audit.report(),before)

    def test_forward_argmax_and_value_are_separate_from_tensor_matches(self):
        audit=self.audit();record=pair()
        record['reference'].update(policy_logits=np.arange(9,dtype=float),value=.1)
        record['candidate'].update(policy_logits=np.arange(9,dtype=float)[::-1],value=.1002)
        audit.add(**record);report=audit.report()
        self.assertEqual(report['forward_decisions'],1);self.assertEqual(report['policy_argmax_agreement'],0.)
        self.assertAlmostEqual(report['max_value_difference'],.0002)
        self.assertFalse(report['minimum_record_count_met']);self.assertFalse(report['supplied_forward_gate_pass'])


if __name__=='__main__':unittest.main()
