"""Inert collection/report plumbing and actual hook manifest; no game/model."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from gpu_engine import device_observation_collector as c
from gpu_engine.device_observation_parity import AgreementParity
import test_gpu_engine_device_observation_collector as f


class RecollectionComparisonTests(unittest.TestCase):
    def test_same_adds_keep_legacy_report_and_only_exclude_raw_coordinates(self):
        bindings={'source_commit':'a'*40};pair=c._Comparisons(bindings,'source_routing_coordinates')
        legacy=AgreementParity(source_bindings=bindings)
        cases=[]
        a=f.view();b=f.view();b['switch_targets'][-1]=99
        cases.append(dict(decision_id='coordinates',reference=a,candidate=b))
        for kind,index,value in (('numeric_features',(28,131),2e-6),('categorical_ids',(0,40),1)):
            bad=f.view();bad[kind][index]=value;bad['switch_targets'][-1]=99
            cases.append(dict(decision_id=kind,reference=a,candidate=bad))
        cases.append(dict(decision_id='refused',refusal={'source':'B2','reason':'unsupported'}))
        bad=f.view();bad['switch_targets']=bad['switch_targets'][:4]
        cases.append(dict(decision_id='malformed',reference=a,candidate=bad))
        cases.append(dict(decision_id='unknown',reference=a,candidate=a,agreement=None))
        cases.append(dict(decision_id='coordinates',reference=a,candidate=a))
        for case in cases:
            row=dict(turn=11,line_kinds=['move'],agreement=True);row.update(case)
            pair.add(**row);legacy.add(**row)
        self.assertEqual(pair.strict.report(),legacy.report())
        selected=pair.report();strict=pair.strict.report()
        self.assertEqual(selected['counts']['joint_match'],1);self.assertEqual(strict['counts']['joint_match'],0)
        for name in ('attempted','agreement','compared','comparison_refused','metadata_refused','state_agreement_unknown'):
            self.assertEqual(selected['counts'][name],strict['counts'][name])
        self.assertEqual(selected['columns'],strict['columns'])
        self.assertFalse(selected['switch_target_routing_coordinates']['semantic_routing_equivalence_verified'])
        default=c._Comparisons(bindings,'legacy_strict');self.assertIs(default.selected,default.strict)

    def test_actual_initialized_readouts_produce_manifest_without_encoder_or_host_tensor_copy(self):
        from gpu_engine import engine_device_public_hooks as hooks,engine_device_trackers as public
        from gpu_engine import engine_device_observation as encoder
        import test_gpu_engine_device_public_hooks as actual
        trackers=hooks.initialize(actual.state(1),actual.tables()).trackers
        runtime=object.__new__(c.Runtime);runtime.bindings={'source_commit':'b'*40}
        with tempfile.TemporaryDirectory() as d:
            capture=c._ColumnManifest(runtime,Path(d),f.CELL)
            with patch.object(c.Runtime,'construct',side_effect=AssertionError('extra constructor')), \
                    patch.object(encoder,'encode',side_effect=AssertionError('extra encoding')), \
                    patch.object(type(trackers.error),'__array__',side_effect=AssertionError('extra host tensor copy')), \
                    patch.object(public,'readout',wraps=public.readout) as reads, \
                    patch.object(encoder,'column_manifest',wraps=encoder.column_manifest) as manifests:
                capture.capture(trackers,f.CELL)
            reference=capture.finish([dict(status='TERMINAL')]);record=json.loads(Path(reference['path']).read_text())
            self.assertEqual(reads.call_count,2);self.assertEqual(manifests.call_count,2)
            self.assertEqual(record['status'],'RECORDED');self.assertEqual(record['cell'],f.CELL)
            self.assertEqual([r['owner_side'] for r in record['owners']],[0,1])
            for call in manifests.call_args_list:self.assertIsInstance(call.kwargs['public'],public.PublicView)
            for owner in record['owners']:
                self.assertEqual(len(owner['columns']),1038)
                field=next(x for x in owner['columns'] if x['section']=='belief' and x['kind']=='numeric' and x['column']==131)
                self.assertIn('constant0',field['reason'])
            self.assertEqual(reference['sha256'],c.digest(reference['path']))

    def test_unsupported_manifest_and_reference_do_not_change_cells_or_refusal_denominators(self):
        runtime=f.Runtime(poison=True);runtime.bindings={'source_commit':'c'*40}
        runtime.column_manifests=lambda trackers:(_ for _ in ()).throw(ValueError('unsupported actual readout'))
        with tempfile.TemporaryDirectory() as d:
            manifest=c._ColumnManifest(runtime,Path(d),f.CELL)
            writer=f.Writer();pair=c._Comparisons(runtime.bindings,'source_routing_coordinates')
            with patch('gpu_engine.finite_stream_comparison.pack_sets',return_value=['set1','set2']), \
                    patch('gpu_engine.finite_gameplay_comparison.native_evidence',side_effect=lambda snapshot,env:dict(turn=snapshot.turn)), \
                    patch('gpu_engine.finite_gameplay_comparison.native_actions',return_value={'p1':0,'p2':0}):
                root=c.collect_game(runtime,f.CELL,writer,pair,env_factory=f.Env,coupling_factory=f.Coupling,
                    folder='/inert-not-created',manifest_sink=manifest.capture)
            self.assertEqual(root['status'],'TERMINAL');self.assertEqual(root['executed_steps'],3)
            self.assertEqual(pair.report()['counts'],pair.strict.report()['counts'])
            self.assertEqual(pair.report()['counts']['agreement'],6)
            self.assertEqual(pair.report()['counts']['comparison_refused'],6)
            record=json.loads(Path(manifest.finish([root])['path']).read_text())
            self.assertEqual(record['status'],'UNKNOWN');self.assertIn('unsupported',record['reason']['message'])
            self.assertEqual(len(runtime.samples),3)
            # A failed first constructor is not replaced by a later root's
            # manifest, and unsupported metadata never becomes produced data.
        with tempfile.TemporaryDirectory() as d:
            other=dict(f.CELL,stream_id=99)
            manifest=c._ColumnManifest(runtime,Path(d),f.CELL);manifest.capture(object(),other)
            record=json.loads(Path(manifest.finish([dict(status='COLLECTION_CELL_FAILED')])['path']).read_text())
            self.assertEqual(record['status'],'UNKNOWN');self.assertIsNone(record['owners'])
            self.assertEqual(record['cell'],f.CELL)

    def test_collect_exports_both_reports_and_explicit_mode_without_changing_rows(self):
        ref=dict(path='/inert-not-read',sha256='a'*64)
        plan=dict(schema=c.SCHEMA,source_commit='b'*40,pool=ref,vocabulary=ref,labels=ref,gender_policy=ref,
            showdown_root='/inert',node='/inert',node_sha256='c'*64,constructor_seed=1,sampling_seed=2,
            gender_seed=3,gender_domain=4,max_requests=3,max_turns=100,cells=[f.CELL],
            policy='PCG64_SEEDSEQUENCE_UNIFORM_PHYSICAL_LEGAL_V1')
        original=c.collect_game
        for mode in ('legacy_strict','source_routing_coordinates'):
            runtime=f.Runtime();runtime.bindings={'source_commit':plan['source_commit']}
            runtime.column_manifests=lambda trackers:[dict(owner_side=s,status='UNKNOWN',columns=None,error={'reason':'inert transport has no real trackers'}) for s in (0,1)]
            observe=runtime.observe
            def changed(s,t):
                mask,requested,usable,views=observe(s,t)
                return mask,requested,usable,tuple(NS(**dict(vars(v),switch_targets=np.asarray([[1,2,3,4,0]],np.int32))) for v in views)
            runtime.observe=changed
            def inert(*args,**kwargs):
                return original(*args,**kwargs,env_factory=f.Env,coupling_factory=f.Coupling)
            with tempfile.TemporaryDirectory() as d,patch.object(c,'prepare',return_value=runtime), \
                    patch.object(Path,'is_relative_to',return_value=True),patch.object(c,'collect_game',side_effect=inert), \
                    patch('gpu_engine.finite_stream_comparison.pack_sets',return_value=['set1','set2']), \
                    patch('gpu_engine.finite_gameplay_comparison.native_evidence',side_effect=lambda snapshot,env:dict(turn=snapshot.turn)), \
                    patch('gpu_engine.finite_gameplay_comparison.native_actions',return_value={'p1':0,'p2':0}):
                out=Path(d)/'out';result=c.collect(plan,out=out,enable_collection=True,comparison_mode=mode)
                report=json.loads(Path(result['summary_path']).read_text())
                strict=json.loads(Path(result['legacy_strict_report']['path']).read_text())
                self.assertEqual(report['counts']['attempted'],6);self.assertEqual(strict['counts']['attempted'],6)
                self.assertEqual(report['counts']['joint_match'],0 if mode=='legacy_strict' else 6)
                self.assertEqual(strict['counts']['joint_match'],0)
                self.assertEqual(report['cells'][0]['executed_steps'],3)
                self.assertEqual(report['installed_column_manifest']['status'],'UNKNOWN')
                self.assertEqual(json.loads((out/'comparison-config.json').read_text())['comparison_mode'],mode)
                with np.load(out/'rows/12-0-0.npz',allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive['candidate__switch_targets'],[1,2,3,4,0])
                    np.testing.assert_array_equal(archive['reference__switch_targets'],[0]*5)
        with patch.object(c,'prepare',side_effect=AssertionError('must not prepare')):
            with self.assertRaises(ValueError):c.collect(plan,out='/shared/inert',enable_collection=True,comparison_mode='unknown')


if __name__=='__main__':unittest.main()
