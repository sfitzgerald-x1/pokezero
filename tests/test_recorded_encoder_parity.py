from dataclasses import replace
import copy
import json
import unittest
import numpy as np
from pokezero.mirror_value_probe import digest
from pokezero.recorded_encoder_corpus import EncoderView
from pokezero.recorded_encoder_parity import ParityTally,SHAPES


class RecordedParityTests(unittest.TestCase):
    def example(self,index=0):
        actual={name:np.zeros(shape,np.float32 if name=='numeric_features' else np.bool_ if name.endswith('mask') else np.int32)
            for name,shape in SHAPES.items()}
        actual['attention_mask'][0]=True;actual['legal_action_mask'][0]=True
        reference={k:v.tolist() for k,v in actual.items()};reference['turn_number']=0
        view=EncoderView(digest(['authored',index]),'authored','p1',index,0,'a'*64,json.dumps({'side':{'id':'p1'}}).encode(),json.dumps(reference).encode())
        return view,actual

    def test_all173_columns_include_padding_and_all_views(self):
        tally=ParityTally(expected_views=2);view,actual=self.example();actual['numeric_features'][22,131]=np.float32(2e-6)
        actual['categorical_ids'][22,40]=3;actual['token_type_ids'][22]=4
        tally.add(view,actual);view,actual=self.example(1);tally.add(view,actual)
        r=tally.report();self.assertEqual((len(r['numeric_columns']),len(r['categorical_columns'])),(132,41))
        self.assertEqual(r['numeric_columns'][131]['mismatched_cells'],1);self.assertEqual(r['numeric_columns'][0]['cells'],46)
        self.assertEqual(r['categorical_columns'][40]['mismatched_cells'],1);self.assertEqual(r['other_tensor_mismatched_cells']['token_type_ids'],1)
        self.assertFalse(r['recorded_input_numeric_criteria_met']);self.assertFalse(r['minimum10000_views_met'])
        self.assertFalse(r['model_forward_complete']);self.assertFalse(r['full_encoder_admitted'])

    def test_nan_dtype_duplicate_partial_output_fail_without_partial_metrics(self):
        tally=ParityTally(expected_views=2);view,actual=self.example();before=tally.report()
        for mutation in ('nan','dtype','output'):
            value=copy.deepcopy(actual);kw={}
            if mutation=='nan':value['numeric_features'][0,0]=np.nan
            if mutation=='dtype':value['numeric_features']=value['numeric_features'].astype(np.float64)
            if mutation=='output':kw={'actual_forward':{'policy_logits':np.zeros(9,np.float32),'value':np.float32(0)}}
            with self.assertRaises(ValueError):tally.add(view,value,**kw)
            self.assertEqual(tally.report(),before)
        tally.add(view,actual);before=tally.report()
        with self.assertRaises(ValueError):tally.add(view,actual)
        self.assertEqual(tally.report(),before)

    def test_paired_forward_uses_each_legal_mask_and_strict_value_bound(self):
        tally=ParityTally(expected_views=2);view,actual=self.example();actual['legal_action_mask'][1]=True
        a={'policy_logits':np.arange(9,dtype=np.float32),'value':np.float32(.5001)}
        r={'policy_logits':np.arange(9,dtype=np.float32),'value':np.float32(.5)}
        tally.add(view,actual,actual_forward=a,reference_forward=r)
        view,actual=self.example(1);tally.add(view,actual,actual_forward=r,reference_forward=r)
        report=tally.report();self.assertTrue(report['model_forward_complete']);self.assertEqual(report['policy_argmax_agreement'],1.);self.assertEqual(report['legal_policy_argmax_agreement'],.5)
        self.assertGreater(report['maximum_value_difference'],1e-4);self.assertFalse(report['measured_value_criterion_met']);self.assertFalse(report['recorded_forward_numeric_criteria_met'])

    def test_complete_input_count_never_substitutes_for_missing_forward(self):
        tally=ParityTally(expected_views=2)
        for i in range(2):view,actual=self.example(i);tally.add(view,actual)
        r=tally.report();self.assertTrue(r['complete_view_count']);self.assertFalse(r['model_forward_complete'])
        self.assertFalse(r['recorded_forward_numeric_criteria_met']);self.assertIsNone(r['policy_argmax_agreement'])
        with self.assertRaises(ValueError):view,actual=self.example(2);tally.add(view,actual)

    def test_canonical_alias_domains_pair_identity_and_chronology_refuse_atomically(self):
        tally=ParityTally(expected_views=4);view,actual=self.example();tally.add(view,actual);before=tally.report()
        for bad in (replace(view,decision_id='b'*64),replace(view,seat='p3'),replace(view,battle_id=''),
                replace(view,decision_round_index=True),replace(view,turn_number=-1),replace(view,source_prefix_sha256='bad'),
                replace(view,decision_id='c'*64,seat='p2',own_request_json=b'{"side":{"id":"p2"}}',source_prefix_sha256='c'*64)):
            with self.assertRaises(ValueError):tally.add(bad,actual)
            self.assertEqual(tally.report(),before)
        for name,index in (('categorical_ids',(0,0)),('token_type_ids',(0,))):
            fresh,got=self.example(1);got[name][index]=-1
            ref=json.loads(fresh.own_reference_json)
            if len(index)==2:ref[name][index[0]][index[1]]=-1
            else:ref[name][index[0]]=-1
            with self.assertRaises(ValueError):tally.add(replace(fresh,own_reference_json=json.dumps(ref).encode()),got)
            self.assertEqual(tally.report(),before)
        fresh,got=self.example(2);tally.add(fresh,got);before=tally.report()
        old,got=self.example(1)
        with self.assertRaises(ValueError):tally.add(old,got)
        self.assertEqual(tally.report(),before)

    def test_synthetic10000_boundary_and_exact_argmax_threshold_are_metrics_only(self):
        tally=ParityTally(expected_views=10000)
        ref={'policy_logits':np.array([2,1,0,0,0,0,0,0,0],np.float32),'value':np.float32(.5)}
        different={'policy_logits':np.array([1,2,0,0,0,0,0,0,0],np.float32),'value':np.float32(.5)}
        for i in range(9999):
            view,actual=self.example(i);actual['legal_action_mask'][1]=True
            reference=json.loads(view.own_reference_json);reference['legal_action_mask'][1]=True
            view=replace(view,own_reference_json=json.dumps(reference).encode())
            tally.add(view,actual,actual_forward=different if i<10 else ref,reference_forward=ref)
        self.assertFalse(tally.report()['recorded_input_numeric_criteria_met'])
        eleven=copy.deepcopy(tally);view,actual=self.example(9999);actual['legal_action_mask'][1]=True
        reference=json.loads(view.own_reference_json);reference['legal_action_mask'][1]=True
        view=replace(view,own_reference_json=json.dumps(reference).encode())
        tally.add(view,actual,actual_forward=ref,reference_forward=ref)
        eleven.add(view,actual,actual_forward=different,reference_forward=ref)
        r=tally.report();self.assertEqual(r['policy_argmax_agreement'],.999)
        self.assertTrue(r['recorded_input_numeric_criteria_met']);self.assertTrue(r['recorded_forward_numeric_criteria_met'])
        self.assertFalse(eleven.report()['recorded_forward_numeric_criteria_met'])
        self.assertEqual(r['status'],'METRICS_ONLY_REQUIRES_CALLER_ATTESTATION')
        self.assertFalse(r['public_wire_qualified']);self.assertFalse(r['full_encoder_admitted'])

    def test_mask_cannot_hide_raw_argmax_difference(self):
        tally=ParityTally(expected_views=1);view,actual=self.example()
        a={'policy_logits':np.arange(9,dtype=np.float32),'value':np.float32(0)}
        r={'policy_logits':np.arange(9,0,-1,dtype=np.float32),'value':np.float32(0)}
        tally.add(view,actual,actual_forward=a,reference_forward=r)
        report=tally.report();self.assertEqual(report['policy_argmax_agreement'],0.)
        self.assertEqual(report['legal_policy_argmax_agreement'],1.)
