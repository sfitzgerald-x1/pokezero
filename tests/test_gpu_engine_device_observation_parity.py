import unittest
import json
import subprocess
import numpy as np
from gpu_engine.device_observation_parity import AgreementParity


def view():
    return dict(numeric_features=np.zeros((29,132),np.float32),
        categorical_ids=np.zeros((29,41),np.int32),token_type_ids=np.zeros(29,np.int32),
        attention_mask=np.ones(29,bool),legal_action_mask=np.ones(9,bool),switch_targets=np.arange(5,dtype=np.int32))


class DeviceAgreementParityTests(unittest.TestCase):
    def test_source_specific_switch_coordinates_are_separate_from_joint_columns(self):
        a=view();b=view()
        a['switch_targets']=np.array([1,2,3,4,5],np.int32)
        b['switch_targets']=np.array([1,2,3,4,0],np.int32)
        own_request_physical=np.array([5,1,2,3,4,0],np.int32)
        np.testing.assert_array_equal(own_request_physical[a['switch_targets']],b['switch_targets'])
        names=np.array(['a','b','c','d','e','f'])
        np.testing.assert_array_equal(names[own_request_physical[a['switch_targets']]],names[b['switch_targets']])
        for mode,match in (('legacy_strict',0),('source_routing_coordinates',1)):
            r=AgreementParity(source_bindings={'test':'synthetic'},comparison_mode=mode)
            r.add(decision_id='same-named-own-pokemon',turn=11,line_kinds=['switch'],agreement=True,reference=a,candidate=b)
            report=r.report();self.assertEqual(report['counts']['joint_match'],match)
            self.assertEqual(report['structure_mismatched_views']['switch_targets'],1)
            if mode=='source_routing_coordinates':
                self.assertEqual(report['schema'],'pokezero.engine-device-public29-agreement-parity.source-routing.v2')
                self.assertEqual(report['switch_target_routing_coordinates']['raw_differing_slots'],[0,0,0,0,1])
                self.assertFalse(report['switch_target_routing_coordinates']['semantic_routing_equivalence_verified'])

    def test_new_mode_ignores_only_routing_differences_and_keeps_refusals(self):
        for name,index,value in (('numeric_features',(28,131),2e-6),('categorical_ids',(0,40),1),
                ('token_type_ids',0,1),('attention_mask',0,False),('legal_action_mask',0,False)):
            a=view();b=view();b['switch_targets'][-1]=99;b[name][index]=value
            r=AgreementParity(source_bindings={'test':'synthetic'},comparison_mode='source_routing_coordinates')
            r.add(decision_id=name,turn=11,line_kinds=['move'],agreement=True,reference=a,candidate=b)
            self.assertEqual(r.report()['counts']['joint_match'],0)
        reports=[]
        for mode in ('legacy_strict','source_routing_coordinates'):
            r=AgreementParity(source_bindings={'test':'synthetic'},comparison_mode=mode)
            a=view();b=view();b['switch_targets']=b['switch_targets'][:4]
            r.add(decision_id='malformed',turn=11,line_kinds=['switch'],agreement=True,reference=a,candidate=b)
            r.add(decision_id='refused',turn=11,line_kinds=['move'],agreement=True,refusal={'source':'B2','reason':'unknown'})
            r.add(decision_id='state-unknown',turn=11,line_kinds=['move'],agreement=None,reference=a,candidate=a)
            r.add(decision_id='good',turn=12,line_kinds=['move'],agreement=True,reference=a,candidate=a)
            reports.append(r.report())
        for key in ('counts','views_by_turn','columns','refusals_by_kind_column','joint_match_fraction_on_ALL_agreement_views'):
            self.assertEqual(reports[0][key],reports[1][key])
        self.assertEqual(reports[1]['counts']['comparison_refused'],2)
        self.assertEqual(reports[1]['joint_match_fraction_on_ALL_agreement_views'],1/3)
        with self.assertRaises(ValueError):AgreementParity(source_bindings={'test':'synthetic'},comparison_mode='guess')

    def test_default_and_explicit_strict_reports_are_byte_exact_frozen_baseline(self):
        baseline={}
        source=subprocess.check_output(['git','show','f7b1ebc919738775d4bb259d8d339952c33fbd71:gpu_engine/device_observation_parity.py'],text=True)
        exec(compile(source,'<frozen-original-comparator>','exec'),baseline)
        reports=[]
        for cls,kwargs in ((baseline['AgreementParity'],{}),(AgreementParity,{}),(AgreementParity,{'comparison_mode':'legacy_strict'})):
            r=cls(source_bindings={'test':'synthetic'},**kwargs);a=view();b=view();b['switch_targets'][-1]=99
            r.add(decision_id='routing-diff',turn=11,line_kinds=['switch','move'],agreement=True,reference=a,candidate=b)
            b=view();b['numeric_features'][28,131]=np.nan
            r.add(decision_id='bad-number',turn=11,line_kinds=['-damage'],agreement=True,reference=a,candidate=b)
            r.add(decision_id='unknown',turn=12,line_kinds=['move'],agreement=None)
            r.add(decision_id='refusal',turn=12,line_kinds=['move'],agreement=True,reference={},candidate=a)
            r.add(decision_id='good',turn=13,line_kinds=['move'],agreement=True,reference=a,candidate=a)
            r.add(decision_id='good',turn=13,line_kinds=['move'],agreement=True,reference=a,candidate=a)
            reports.append(json.dumps(r.report(),sort_keys=True,indent=2,allow_nan=False)+'\n')
        self.assertEqual(reports[0],reports[1]);self.assertEqual(reports[0],reports[2])

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
