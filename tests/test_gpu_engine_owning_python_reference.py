"""Literal authored owning packets and saved artifacts; no process/model/game."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace as NS
from collections import namedtuple
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import owning_python_reference as reference
from gpu_engine import finite_stream_comparison_saved as saved
from gpu_engine.public_inputs import PublicEvent, Kind, Phase
from pokezero.live_protocol_actor import _vocab_sha
from tests import test_saved_player_projection as fixture_source


class OwningReferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_source.ProductionHistoryTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.fixture.source.metadata=NS(source_hash='b'*16)
        contract=replace(reference.ActorContract.champion(),belief_set_source_hash='b'*16,
            category_vocab_sha256=_vocab_sha(self.fixture.config.category_vocab),
            categorical_vocab_size=self.fixture.config.category_vocab.size)
        self.patch=patch.object(reference.ActorContract,'champion',return_value=contract)
        self.patch.start();self.addCleanup(self.patch.stop)

    def lines(self,owner):
        peer='p2' if owner=='p1' else 'p1'
        lines=list(fixture_source.leads())
        for i,line in enumerate(lines):
            if line.startswith('|switch|'+peer):lines[i]=line.rsplit('|',1)[0]+'|75/100'
        return lines+[fixture_source.request(owner,move='thunderbolt')]

    def observe(self,owner,lines=None):
        return reference.observe_owner(self.fixture.config,self.lines(owner) if lines is None else lines,
            owner=owner,battle_id='authored-actual-owner',stream_id=123,seed=987)

    def test_two_real_production_backends_have_separate_owner_only_histories_and_hp_provenance(self):
        original=reference.PythonBackend;backends=[]
        def construct(*args,**kwargs):
            backend=original(*args,**kwargs);backends.append(backend);return backend
        with patch.object(reference,'PythonBackend',side_effect=construct):
            first,one=self.observe('p1');second,two=self.observe('p2')
        for owner,backend,receipt in zip(('p1','p2'),backends,(one,two)):
            peer='p2' if owner=='p1' else 'p1'
            self.assertIsNone(backend.env._process)
            self.assertEqual(backend.env.protocol_lines,tuple(self.lines(owner)))
            self.assertEqual(set(backend.env._parser.requests),{owner})
            self.assertEqual(receipt['hp_visibility'],{owner:'exact',peer:'percentage'})
            self.assertEqual(receipt['own_request_count'],1)
        for field in ('_parser','_belief_engine','_tier2_trackers','_investment_trackers'):
            self.assertIsNot(getattr(backends[0].env,field),getattr(backends[1].env,field))
        self.assertEqual(np.asarray(first['numeric_features']).shape,(23,132))
        self.assertEqual(np.asarray(second['categorical_ids']).shape,(23,41))
        # A later peer reference cannot mutate the first owner's tensor truth.
        before=deepcopy(first);self.observe('p2')
        self.assertEqual(first,before)

    def test_foreign_request_and_opponent_exact_HP_poison_are_refused_never_filtered(self):
        foreign=self.lines('p1')+[fixture_source.request('p2')]
        with self.assertRaisesRegex(ValueError,'owning request identity'):self.observe('p1',foreign)
        poisoned=self.lines('p1')
        poisoned=[line.rsplit('|',1)[0]+'|200/301' if line.startswith('|switch|p2') else line for line in poisoned]
        with self.assertRaisesRegex(ValueError,'opponent exact HP'):self.observe('p1',poisoned)
        with self.assertRaisesRegex(ValueError,'actual owning request'):self.observe('p1',self.lines('p1')[:-1])

    def artifacts(self):
        event=PublicEvent(Kind.TURN,1,1,Phase.ACTION)
        words=np.zeros((8,2,1,64,16),np.int32);words[:,:,0,0]=event.words()
        valid=np.zeros((8,2,1,64),bool);valid[:,:,0,0]=True
        cohort=[dict(stream_id=100+i,team_indices=[2*i,2*i+1]) for i in range(8)]
        observations=[]
        for side in range(2):
            values={name:np.zeros((8,*shape),np.float32 if name=='numeric_features' else
                bool if name.endswith('mask') else np.int32).tolist() for name,shape in saved.comparison.SHAPES.items()}
            values['attention_mask']=np.ones((8,23),bool).tolist()
            values['legal_action_mask']=np.tile([True]+[False]*8,(8,1)).tolist()
            values['admitted']=[i<(7 if side==0 else 6) for i in range(8)]
            observations.append(values)
        measurement=dict(original_denominator=8,cohort=cohort,runtime_binding='authored-runtime',
            initial_owner_observations=observations,initial_frame_input_witness=dict(owner_full_words=words.tolist(),
                owner_full_valid=valid.tolist(),owner_full_more=np.zeros((8,2,1),bool).tolist()))
        raw=json.dumps(measurement).encode();sha=hashlib.sha256(raw).hexdigest()
        rows=[dict(root_id=row['stream_id'],team_indices=row['team_indices'],native_seed=700+i,
            state_status='AGREE',state_differences=[],native_owner_lines=[self.lines('p1'),self.lines('p2')],
            owners=[dict(status='MISMATCH'),dict(status='UNKNOWN')]) for i,row in enumerate(cohort)]
        native=dict(schema=saved.comparison.SCHEMA,measurement_file_sha256=sha,
            source_measurement_binding='authored-runtime',rows=rows,summary=dict(original_roots=8,old_numeric_bad=76))
        return raw,native,event

    def reprocess(self,raw,native,event):
        payload=json.dumps(native).encode()
        # Real typed PublicEvent transport; the wire projector's separate
        # lexical contracts are already covered by the comparator tests.
        def projected(lines,catalog,**kwargs):
            return dict(words=[event.words().tolist()],refused_lines=[],raw_kinds={'turn':1})
        with patch.object(saved.comparison,'reference_words',side_effect=projected), \
                patch.object(saved.comparison,'collect_constructors',side_effect=AssertionError('no Native')), \
                patch.object(saved.comparison,'OwningShowdownEnv',side_effect=AssertionError('no simulator')), \
                patch.object(saved.comparison.fidelity,'compare',side_effect=AssertionError('reuse original state judgment')):
            return saved.reprocess_saved(raw,payload,None,env_config=self.fixture.config,
                expected_measurement_sha256=hashlib.sha256(raw).hexdigest(),
                expected_native_result_sha256=hashlib.sha256(payload).hexdigest())

    def test_pure_saved_reprocessing_keeps_original_roster_states_denominators_and_unknown_views(self):
        raw,native,event=self.artifacts();before=deepcopy(native)
        corrected=self.reprocess(raw,native,event)
        self.assertEqual(native,before)
        self.assertEqual(corrected['new_native_calls'],0)
        self.assertEqual([row['root_id'] for row in corrected['rows']],list(range(100,108)))
        self.assertEqual([row['state_status'] for row in corrected['rows']],['AGREE']*8)
        self.assertEqual(corrected['summary']['original_owner_views'],16)
        self.assertEqual(corrected['summary']['numeric_columns'][0]['compared_cells'],299)
        self.assertEqual(corrected['summary']['numeric_columns'][0]['unknown_cells'],69)
        self.assertEqual(corrected['summary']['categorical_columns'][0]['unknown_cells'],69)
        self.assertEqual(len(corrected['summary']['numeric_columns']),132)
        self.assertEqual(len(corrected['summary']['categorical_columns']),41)
        json.dumps(corrected,allow_nan=False)

    def test_saved_binding_missing_lines_and_roster_drift_refuse_without_fallback(self):
        raw,native,event=self.artifacts()
        for alter in ('root','team','lines','measurement'):
            changed=deepcopy(native)
            if alter=='root':changed['rows'][0]['root_id']=12345
            elif alter=='team':changed['rows'][0]['team_indices']=[9,10]
            elif alter=='lines':changed['rows'][0].pop('native_owner_lines')
            else:changed['measurement_file_sha256']='e'*64
            with self.subTest(alter=alter),self.assertRaises(ValueError):self.reprocess(raw,changed,event)
        payload=json.dumps(native).encode()
        with self.assertRaisesRegex(ValueError,'SHA256'):
            saved.reprocess_saved(raw,payload,None,env_config=self.fixture.config,
                expected_measurement_sha256='e'*64,expected_native_result_sha256=hashlib.sha256(payload).hexdigest())

    def test_foreign_private_request_makes_only_that_column_view_unknown_other_owner_continues(self):
        raw,native,event=self.artifacts()
        native['rows'][0]['native_owner_lines'][0].append(fixture_source.request('p2'))
        corrected=self.reprocess(raw,native,event)
        owners=corrected['rows'][0]['owners']
        self.assertEqual(owners[0]['columns']['reason'],'OWNING_PYTHON_REFERENCE_REFUSED')
        self.assertEqual(owners[1]['columns']['status'],'COMPARED')
        self.assertEqual(corrected['summary']['numeric_columns'][0]['compared_cells'],276)
        self.assertEqual(corrected['summary']['numeric_columns'][0]['unknown_cells'],92)
        self.assertEqual(corrected['summary']['conditional_on_measured_state_agreement']['matched'],16)

    def test_new_native_collector_uses_only_real_owning_backend_never_simulator_observe(self):
        comparison=saved.comparison
        raw,_,event=self.artifacts();measured=json.loads(raw)
        actual=saved.captured_observations(measured,0,8)
        capture=tuple(np.asarray(measured['initial_frame_input_witness'][name])[:1]
            for name in ('owner_full_words','owner_full_valid','owner_full_more'))
        State=namedtuple('State','turn gender happiness')
        state=State(np.zeros(1,np.int32),np.ones((1,2,6),np.int32),np.full((1,2,6),111,np.int32))
        pool=NS(arrays={'species':np.ones((2,6),np.int32)},metadata=lambda:dict(vocabulary={'species':['_none','blissey']}))
        owner_lines=[self.lines('p1'),self.lines('p2')]
        class FakeNative:
            def __init__(self,config):self.owner_transcripts=deepcopy(owner_lines)
            def reset_pool_sets(self,*args,**kwargs):pass
            def snapshot(self):return object()
            def observe(self,*args):raise AssertionError('omniscient simulator observe is forbidden')
            def close(self):pass
        with patch.object(comparison,'OwningShowdownEnv',FakeNative), \
                patch.object(comparison,'pool_sets',return_value=[[],[]]), \
                patch.object(comparison,'pack_sets',return_value=['packed1','packed2']), \
                patch.object(comparison.fidelity,'compare',return_value=[]), \
                patch.object(comparison,'reference_words',return_value=dict(words=[event.words().tolist()],
                    refused_lines=[],raw_kinds={'turn':1})):
            result=comparison.collect_constructors(pool,[[0,1]],state,capture,None,root_ids=[100],
                showdown_root='/not-opened',node='not-run',native_seed_start=700,env_config=self.fixture.config,
                engine_observations=actual)
        owners=result['rows'][0]['owners']
        self.assertEqual([owner['columns']['status'] for owner in owners],['COMPARED','COMPARED'])
        self.assertEqual([owner['python_reference']['owner'] for owner in owners],['p1','p2'])

    def test_cli_new_output_is_exclusive_before_any_source_or_runtime_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'original.json';output.write_text('immutable original')
            with self.assertRaises(FileExistsError):
                saved.main(['--measurement','not-read','--vocabulary','not-read','--showdown-root','not-read',
                    '--native-result','not-read','--measurement-sha256','a'*64,'--native-result-sha256','b'*64,
                    '--out',str(output)])
            self.assertEqual(output.read_text(),'immutable original')
