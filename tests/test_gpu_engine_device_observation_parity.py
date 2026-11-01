import unittest
import numpy as np
from gpu_engine.device_observation_parity import AgreementParity


def view():
    return dict(numeric_features=np.zeros((29,132),np.float32),
        categorical_ids=np.zeros((29,41),np.int32),token_type_ids=np.zeros(29,np.int32),
        attention_mask=np.ones(29,bool),legal_action_mask=np.ones(9,bool),switch_targets=np.arange(5,dtype=np.int32))


class DeviceAgreementParityTests(unittest.TestCase):
    def test_integer_numeric_inputs_cannot_collapse_into_false_matches(self):
        r=AgreementParity(source_bindings={'test':'synthetic'});a=view();b=view()
        a['numeric_features']=np.zeros((29,132),np.uint64)
        b['numeric_features']=a['numeric_features'].copy()
        a['numeric_features'][0,0]=2**53;b['numeric_features'][0,0]=2**53+1
        r.add(decision_id='bad',turn=10,line_kinds=['move'],agreement=True,reference=a,candidate=b)
        p=r.report();self.assertEqual(p['counts']['comparison_refused'],1)
        self.assertEqual(p['joint_match_fraction_on_ALL_agreement_views'],0)

    def test_scalar_admission_types_and_unknowns_remain_nonfatal_refusals(self):
        for status in (False, np.bool_(False), np.asarray(False), None, 0, [False]):
            for side in ('reference', 'candidate'):
                with self.subTest(status=repr(status), side=side):
                    r=AgreementParity(source_bindings={'test':'synthetic'});a=view();b=view()
                    (a if side=='reference' else b)['admitted']=status
                    r.add(decision_id='bad',turn=10,line_kinds=['move'],agreement=True,reference=a,candidate=b)
                    r.add(decision_id='good',turn=11,line_kinds=['move'],agreement=True,reference=view(),candidate=view())
                    p=r.report();self.assertEqual(p['counts']['compared'],1)
                    self.assertEqual(p['counts']['comparison_refused'],1)
                    self.assertEqual(p['joint_match_fraction_on_ALL_agreement_views'],.5)

    def test_joint_all_columns_includes_zero_columns_and_29th_token(self):
        r=AgreementParity(source_bindings={'test':'synthetic'})
        a=view();b=view();b['numeric_features'][28,131]=2e-6
        r.add(decision_id='a',turn=11,line_kinds=['-damage','move'],agreement=True,reference=a,candidate=b)
        b=view();b['categorical_ids'][0,40]=1
        r.add(decision_id='b',turn=2,line_kinds=['switch'],agreement=True,reference=a,candidate=b)
        r.add(decision_id='c',turn=12,line_kinds=['move'],agreement=True,reference=a,candidate=a)
        p=r.report();self.assertEqual(p['joint_match_fraction_on_ALL_agreement_views'],1/3)
        self.assertEqual(p['compared_views_turn10_or_later'],2)
        self.assertEqual(p['columns'][131]['mismatched_cells'],1)
        self.assertEqual(p['columns'][132+40]['mismatched_cells'],1)
        self.assertFalse(p['numerical_gate_only'])

    def test_refusals_are_nonfatal_and_keep_agreement_denominator(self):
        r=AgreementParity(source_bindings={'test':'synthetic'});a=view()
        r.add(decision_id='a',turn=1,line_kinds=['UNSUPPORTED_KIND'],agreement=True,
            refusal=dict(source='B2',column='numeric:63',reason='unsupported actual line'))
        r.add(decision_id='b',turn=3,line_kinds=[{'malformed':'not hashable'}],agreement=True,reference={},candidate=a)
        r.add(decision_id='c',turn=4,line_kinds=['move'],agreement=False,reference=a,candidate=a)
        r.add(decision_id='d',turn=5,line_kinds=['move'],agreement=None,reference=a,candidate=a)
        r.add(decision_id='e',turn=15,line_kinds=['move'],agreement=True,reference=a,candidate=a)
        p=r.report();self.assertEqual(p['counts']['attempted'],5)
        self.assertEqual(p['counts']['agreement'],3);self.assertEqual(p['counts']['comparison_refused'],2)
        self.assertEqual(p['joint_match_fraction_on_ALL_agreement_views'],1/3)
        self.assertTrue(any(x['kind']=='UNSUPPORTED_KIND' and x['column']=='numeric:63' for x in p['refusals_by_kind_column']))

    def test_nonfinite_and_duplicate_rows_do_not_erase_other_decisions(self):
        r=AgreementParity(source_bindings={'test':'synthetic'});a=view();b=view()
        b['numeric_features'][10,2]=np.nan
        r.add(decision_id='a',turn=10,line_kinds=['-status'],agreement=True,reference=a,candidate=b)
        r.add(decision_id='a',turn=10,line_kinds=['-status'],agreement=True,reference=a,candidate=a)
        r.add(decision_id='b',turn=11,line_kinds=['move'],agreement=True,reference=a,candidate=a)
        p=r.report();self.assertEqual(p['counts']['attempted'],3);self.assertEqual(p['counts']['compared'],2)
        self.assertEqual(p['counts']['metadata_refused'],1);self.assertEqual(p['columns'][2]['nonfinite_cells'],1)
        self.assertEqual(p['joint_match_fraction_on_ALL_agreement_views'],.5)


if __name__=='__main__':unittest.main()
