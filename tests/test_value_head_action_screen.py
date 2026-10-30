"""Synthetic numerical-producer controls. No model, saved corpus, or API access."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest import mock
from types import SimpleNamespace
import random
from pokezero import failure_capture as fc

P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('action_screen',P/'scripts/screen_value_head_actions.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class ScreenTests(unittest.TestCase):
    def capture_context(self):
        return SimpleNamespace(battle_id='synthetic',seed=7,player_id='p1',decision_round_index=0,
            observation=SimpleNamespace(categorical_ids=((1,),),numeric_features=((.5,),),token_type_ids=(0,),
                attention_mask=(True,),legal_action_mask=(True,False),perspective='p1',metadata={},schema_version=1))

    def test_native_batch_links_all_witnesses_once_with_exact_counts_rng_and_lazy_evidence(self):
        class Fold:
            def to_payload(self):return {'fold':'immutable'}
        events=[dict(reason_codes=['crate.prior.unmapped_action'],engine_state='state-'+str(i),instruction_list=[i],action_map=[None]) for i in range(100)]
        reply=json.dumps(dict(prior_fallbacks=100,capture_refusals=events));calls=[]
        class Native:
            def search_batched_multi_encoded(self,*args,**kwargs):calls.append((args,kwargs));return reply
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);q=fc.Quarantine(root/'quarantine',build={'fingerprint':'synthetic'});directory=root/'query';directory.mkdir()
            original=s.install_native_refusal_batches(fc,root);rng=random.Random(2);before=rng.getstate();lazy=[]
            try:
                with fc.decision_scope(q,self.capture_context(),rng) as capture:
                    capture.update(engine_state='current-world',world_spec={'sampled':'world'})
                    capture.defer('active_batch',lambda:lazy.append(1) or {'row':9})
                    witness=s.NativeWitness(Native(),directory,fc.canonical,capture_module=fc,evidence_root=root)
                    args=('state',2048,64,'tables',{},'ctx',Fold(),8,1.4,3)
                    self.assertEqual(witness.search_batched_multi_encoded(*args,flag=True),reply)
                    fc.record_native_refusals(json.loads(reply))
                    expected_fields=fc.snapshot(capture.fields);expected_fields.pop('input_receipt')
                self.assertEqual(calls,[(args,{'flag':True})]);self.assertIs(calls[0][0][6],args[6]);self.assertEqual(rng.getstate(),before)
                self.assertEqual(lazy,[1]);self.assertEqual((q.events,q.successes,q.failures),(100,100,0))
                self.assertEqual(q.reason_counts,{'crate.prior.unmapped_action':100})
                self.assertEqual(len(list(q.directory.iterdir())),3)  # input + one batch + terminal
                batches=list(q.directory.glob('native-refusal-batch-*.json'));self.assertEqual(len(batches),1)
                batch=json.loads(batches[0].read_bytes());self.assertEqual(batch['event_count'],100)
                actual_events=s.read_link(batch['native_reply'],root)['capture_refusals'];self.assertEqual(actual_events,events)
                self.assertEqual(batch['event_index'],[dict(index=i,reason_codes=e['reason_codes']) for i,e in enumerate(events)])
                reconstructed=s.read_link(batch['root_input'],root);reconstructed.pop('build');reconstructed.pop('schema');reconstructed.update(batch['field_updates'])
                self.assertEqual(reconstructed,expected_fields);self.assertEqual(batch['lazy_evidence'],{'active_batch':{'row':9}})
                self.assertNotIn('observation',batch);self.assertNotIn('inline_events',batch)
            finally:fc._capture_native_events=original

    def test_native_exception_batch_keeps_full_inline_witnesses_and_single_failure_compatibility(self):
        with tempfile.TemporaryDirectory() as d:
            q=fc.Quarantine(Path(d)/'q',build={'fingerprint':'toy'});original=s.install_native_refusal_batches(fc,d)
            events=[dict(reason_codes=['crate.prior.other'],engine_state='origin',instruction_list=['exact'])]
            try:
                with fc.decision_scope(q,self.capture_context(),random.Random(1)) as capture:
                    fc._capture_native_events(capture,events)
                    capture.capture('ordinary',codes=['ordinary.contract_failure'],witness={'engine_state':'single'})
                self.assertEqual((q.events,q.successes,q.failures),(2,2,0))
                batch=json.loads(next(q.directory.glob('native-refusal-batch-*')).read_bytes());self.assertEqual(batch['inline_events'],events)
                single=json.loads(next(q.directory.glob('failure-*')).read_bytes());self.assertEqual(single['schema'],'pokezero.quarantined-failure.v1');self.assertIn('observation',single)
            finally:fc._capture_native_events=original

    def test_batch_rejects_bad_witness_links_arrays_lazy_or_writer_without_false_success(self):
        for failure in ('malformed','missing-input','reply-mismatch','reply-tamper','lazy-error','write-error'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                root=Path(d);q=fc.Quarantine(root/'q',build={'fingerprint':'toy'});original=s.install_native_refusal_batches(fc,root)
                events=[dict(reason_codes=['crate.prior.unmapped_action'],engine_state='origin')]
                try:
                    with self.assertRaises(fc.CaptureFailed):
                        with fc.decision_scope(q,self.capture_context(),random.Random(1)) as capture:
                            if failure=='malformed':events.append({'reason_codes':[]})
                            if failure=='missing-input':capture.fields.pop('input_receipt')
                            if failure.startswith('reply-'):
                                p=root/'reply.json';s.write(p,{'capture_refusals':[] if failure=='reply-mismatch' else events})
                                capture._screen_native_reply=s.evidence_link(p,root)
                                if failure=='reply-tamper':p.write_bytes(b'{}\n')
                            if failure=='lazy-error':capture.defer('bad',lambda:(_ for _ in ()).throw(ValueError('lazy')))
                            if failure=='write-error':
                                write=q.write
                                def fail(kind,payload):
                                    if kind=='native-refusal-batch':raise OSError('disk full')
                                    return write(kind,payload)
                                q.write=fail
                            fc._capture_native_events(capture,events)
                    self.assertEqual(q.successes,0);self.assertEqual(q.failures,1)
                finally:fc._capture_native_events=original

    def test_original_prior_count_validation_is_still_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            q=fc.Quarantine(Path(d)/'q',build={});original=s.install_native_refusal_batches(fc,d)
            try:
                with self.assertRaises(fc.CaptureFailed):
                    with fc.decision_scope(q,self.capture_context(),random.Random(1)):
                        fc.record_native_refusals({'prior_fallbacks':2,'capture_refusals':[{'reason_codes':['x']}]})
                self.assertEqual(q.successes,0);self.assertEqual(q.failures,1)
            finally:fc._capture_native_events=original

    def test_failed_next_native_call_cannot_reuse_previous_reply_link(self):
        class Fold:
            def to_payload(self):return {}
        calls=[];events=[{'reason_codes':['crate.prior.unmapped_action'],'engine_state':'first'}]
        class Native:
            def search_batched_multi_encoded(self,*args):
                calls.append(args)
                if len(calls)==1:return json.dumps({'prior_fallbacks':1,'capture_refusals':events})
                error=ValueError('native failure');error.pokezero_refusal_events=[dict(events[0],engine_state='second')];raise error
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);q=fc.Quarantine(root/'q',build={});query=root/'query';query.mkdir();original=s.install_native_refusal_batches(fc,root)
            try:
                with fc.decision_scope(q,self.capture_context(),random.Random(1)):
                    witness=s.NativeWitness(Native(),query,fc.canonical,capture_module=fc,evidence_root=root)
                    args=('state',1,1,'tables',{},'ctx',Fold(),8,1.4,1)
                    fc.record_native_refusals(json.loads(witness.search_batched_multi_encoded(*args)))
                    try:witness.search_batched_multi_encoded(*args)
                    except ValueError as error:fc.record_refusal('crate.model_search',error)
                    else:self.fail('missing native failure')
                batches=[json.loads(p.read_bytes()) for p in q.directory.glob('native-refusal-batch-*')]
                self.assertEqual(len(calls),2);self.assertEqual(len(batches),2)
                inline=next(b for b in batches if 'inline_events' in b)
                self.assertEqual(inline['inline_events'][0]['engine_state'],'second');self.assertNotIn('native_reply',inline)
                self.assertEqual((q.events,q.successes,q.failures),(3,3,0))
            finally:fc._capture_native_events=original

    def checkpoint_fixture(self,root):
        prior=root/'prior';prior.mkdir();native_input={'input':'fixed'};reply={'witness':'fixed'}
        semantic=s.digest({k:reply.get(k) for k in ('side_one','side_two','root_value','iterations','model_evals','prior_branches','prior_fallbacks','depth_reached','capture_refusals')})
        result=dict(status='SEARCHED_WITH_NATIVE_PRIOR_REFUSALS',action=0,native_calls=[dict(started=True,completed=True,input_sha256=s.digest(native_input),reply_sha256=s.digest(reply),semantic_sha256=semantic,prior_fallbacks=1)])
        for ordinal,head,label in [(0,h,l) for h in ('champion','refit') for l in ('primary','determinism')]+[(1,'champion','primary'),(1,'refit','primary'),(1,'champion','determinism')]:
            p=prior/f'root-{ordinal:03d}-{head}-{label}';p.mkdir();value=copy.deepcopy(result)
            if ordinal==1 and label=='determinism':value.update(status='CAPTURE_FAILED',action=None)
            s.write(p/'decision.json',value);s.write(p/'native-00-input.json',native_input);s.write(p/'native-00-reply.json',reply)
        bank={'roots':[dict(root_id=str(i),cost_seat='p1',search_seed=i) for i in range(2)]};s.write(prior/'root-bank.json',bank)
        for i in range(2):s.write(prior/f'row-{i:03d}.json',dict(ordinal=i,root_id=str(i),seat='p1',search_seed=i,paired_clean_search=False,chosen_actions_differ=None,heads={h:copy.deepcopy(result) for h in ('champion','refit')}))
        runtime=dict(build={'fingerprint':'toy'},search_config={'worlds':1},numeric_mode='FP32_MHAOFF_TF32OFF_HIGHEST',models={'champion.pt':'model-sha'})
        s.write(prior/'runtime-and-config.json',runtime)
        files={p.relative_to(prior).as_posix():{'bytes':p.stat().st_size,'sha256':s.file_sha(p)} for p in prior.rglob('*') if p.is_file()}
        checkpoint=dict(prior_output=str(prior),fixed_bank_sha256=s.file_sha(prior/'root-bank.json'),completed_root_ordinals=[0],consumed_partial_root_ordinals=[1],continue_only_root_ordinals=list(range(2,256)),
            slots=[{'directory':p.name} for p in prior.glob('root-*') if p.is_dir()],files=files,prior_terminal_summary={'completed_root_rows':2,'failure_capture_telemetry':{'capture_failures':1}})
        p=root/'checkpoint.json';s.write(p,checkpoint)
        plan=dict(root_bank_sha256=checkpoint['fixed_bank_sha256'],runtime_capture_build=runtime['build'],search_config=runtime['search_config'],immutable_inputs={'champion.pt':{'sha256':'model-sha'}},continuation=dict(checkpoint=str(p),checkpoint_sha256=s.file_sha(p),prior_output=str(prior)))
        return plan,bank,prior,p

    def test_checkpoint_retains_both_primaries_and_exact_interrupted_repeat_without_reexecution(self):
        with tempfile.TemporaryDirectory() as d:
            plan,bank,prior,p=self.checkpoint_fixture(Path(d));before={q:s.file_sha(q) for q in prior.rglob('*') if q.is_file()}
            rows,repeats,checkpoint=s.restore_checkpoint(plan,bank)
            self.assertEqual([r['ordinal'] for r in rows],[0,1]);self.assertEqual([r['heads']['champion']['action'] for r in rows],[0,0])
            self.assertEqual(len(repeats),4);self.assertEqual(sum(r['native_calls'] for r in repeats),3)
            self.assertEqual([r['status'] for r in repeats[2:]],['CAPTURE_FAILED_CONSUMED_ROOT_UNKNOWN','NOT_ATTEMPTED_CONSUMED_ROOT_UNKNOWN'])
            self.assertEqual(before,{q:s.file_sha(q) for q in before})
            self.assertEqual(checkpoint['continue_only_root_ordinals'],list(range(2,256)))
            full_bank={'roots':bank['roots']+[{'root_id':str(i)} for i in range(2,256)]}
            self.assertEqual([r['root_id'] for r in s.remaining_fixed_roots(full_bank,rows)],[str(i) for i in range(2,256)])
            with self.assertRaises(ValueError):s.remaining_fixed_roots(full_bank,rows[:1])
            changed=copy.deepcopy(rows);changed[1]['ordinal']=2
            with self.assertRaises(ValueError):s.remaining_fixed_roots(full_bank,changed)

    def test_checkpoint_refuses_mutation_foreign_attempt_missing_slot_or_runtime_drift(self):
        for failure in ('bytes','new-file','new-slot','missing-slot','bank','runtime'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                plan,bank,prior,p=self.checkpoint_fixture(Path(d))
                if failure=='bytes':(prior/'root-000-champion-primary/decision.json').write_bytes(b'{}')
                if failure=='new-file':s.write(prior/'root-000-champion-primary/native-01-input.json',{})
                if failure=='new-slot':(prior/'root-002-champion-primary').mkdir()
                if failure=='missing-slot':(prior/'root-001-refit-primary').rename(prior/'moved')
                if failure=='bank':plan['root_bank_sha256']='other'
                if failure=='runtime':plan['search_config']={'worlds':2}
                with self.assertRaises((ValueError,FileNotFoundError)):s.restore_checkpoint(plan,bank)

    def test_later_history_is_hash_disclosed_without_bypassing_library_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);legacy=base/'legacy';fresh=base/'fresh';output=base/'new-screen'
            legacy.mkdir();fresh.mkdir();output.mkdir();history=legacy/'quarantine/engine-mcts-history';history.mkdir(parents=True)
            known=['decision.contract_failure','renderer.sleeptalk_marker_missing_or_conflicting']
            extra=['fallback.crate_search_failed','renderer.confusion_selfhit_ambiguous_executed_self_damage']
            for i,code in enumerate(known+extra):s.write(history/f'failure-{i}.json',{'reason_codes':[code]})
            inventory=dict(reason_codes=known,source_runs=[{k:{'file':k+'.json','sha256':'original-'+k} for k in ('bundle','diagnostic_receipt','terminal')}])
            s.write(legacy/'gate0-inventory-r2.json',inventory);build={'fingerprint':'synthetic-build'};fixtures=[]
            for i,code in enumerate(known):
                row=dict(reason_codes=[code],original_bundle={'file':f'bundle-{i}.json','sha256':'original-bundle'},original_diagnostic={'sha256':'original-diagnostic'})
                for field in ('fixture','verification'):
                    p=fresh/f'{field}-{i}.json';s.write(p,{'synthetic':field});row[field]={'file':p.name,'sha256':s.file_sha(p)}
                fixtures.append(row)
            s.write(fresh/'current-capture-source-verification.json',dict(build=build,fixtures=fixtures))
            plan=dict(admission_parent=str(base),legacy_capture_relative_root='legacy',legacy_inventory_sha256=s.file_sha(legacy/'gate0-inventory-r2.json'),
                fresh_capture_fixtures=str(fresh),fresh_capture_verification_sha256=s.file_sha(fresh/'current-capture-source-verification.json'))
            calls=[]
            def library(path,**kwargs):
                calls.append((path,kwargs));v=json.loads(path.read_bytes())
                self.assertEqual(v['observed_reason_codes'],sorted(known));self.assertEqual(len(v['fixtures']),2)
                self.assertEqual(kwargs,dict(build=build,scientific_path='engine-mcts',capture_directory=output/'quarantine'))
                return {'sha256':s.file_sha(path)}
            module=SimpleNamespace(require_capture_admission=library)
            with mock.patch.dict('sys.modules',{'pokezero.failure_capture':module}),mock.patch.dict('os.environ'):
                proof=s.prepare_capture_admission(plan,build,output)
            self.assertEqual(len(calls),1);self.assertFalse(proof['all_historical_classes_qualified'])
            self.assertEqual(proof['additional_historical_reason_codes_not_requalified'],sorted(extra))
            disclosure=json.loads((output/'capture-history-disclosure.json').read_bytes())
            self.assertEqual(len(disclosure['history_files']),4)
            for row in disclosure['history_files']:self.assertEqual(row['sha256'],s.file_sha(legacy/row['file']))
            # A real library refusal is still fatal; no substituted success proof.
            another=base/'another-screen';another.mkdir()
            with mock.patch.dict('sys.modules',{'pokezero.failure_capture':SimpleNamespace(require_capture_admission=mock.Mock(side_effect=ValueError('library refuses')))}):
                with self.assertRaisesRegex(ValueError,'library refuses'):s.prepare_capture_admission(plan,build,another)

    def candidates(self):
        return [dict(root_id=str(i),battle_id='battle-'+str(i//3),cost_seat=seat,split='train') for i in range(12) for seat in ('p1','p2')]

    def test_hash_rank_is_order_invariant_unique_balanced_and_excludes_old_roots(self):
        candidates=self.candidates();a=s.select_roots(candidates,{'1','2'},count=8)
        b=s.select_roots(list(reversed(candidates)),{'1','2'},count=8)
        self.assertEqual(a,b);self.assertEqual(len({r['root_id'] for r in a}),8)
        self.assertEqual([r['cost_seat'] for r in a],['p1','p2']*4)
        self.assertFalse({'1','2'} & {r['root_id'] for r in a})

    def test_outcome_action_value_fields_cannot_select_roots(self):
        candidates=self.candidates();a=s.select_roots(candidates,set(),count=8)
        for i,row in enumerate(candidates):row.update(outcome=i%2,value=100000-i,action=i%9,refusal=True)
        b=s.select_roots(candidates,set(),count=8)
        self.assertEqual([(r['root_id'],r['cost_seat'],r['search_seed']) for r in a],[(r['root_id'],r['cost_seat'],r['search_seed']) for r in b])

    def test_missing_or_duplicate_roots_refuse_without_refill(self):
        for candidates in (self.candidates()+[self.candidates()[0]],self.candidates()[:2]):
            with self.assertRaises(ValueError):s.select_roots(candidates,set(),count=8)
        with self.assertRaises(ValueError):s.select_roots(self.candidates(),{str(i) for i in range(12)},count=8)

    def test_full_denominators_and_unresolved_bounds(self):
        rows=[]
        for i in range(4):
            rows.append(dict(ordinal=i,paired_clean_search=i<2,chosen_actions_differ=i==0 if i<2 else None,
                heads={'champion':{'status':'SEARCHED','action':0},'refit':{'status':'SEARCHED' if i<2 else 'REFUSED','action':1 if i==0 else 0 if i<2 else None}}))
        report=s.summarize(rows,count=4)
        self.assertEqual((report['chosen_actions_differ'],report['paired_clean_searches'],report['unresolved_roots']),(1,2,2))
        self.assertEqual((report['full_roster_difference_fraction_lower'],report['full_roster_difference_fraction_upper']),(.25,.75))
        self.assertEqual(report['head_refusals'],{'champion':0,'refit':2})
        with self.assertRaises(ValueError):s.summarize(rows[:3],count=4)

    def test_native_witness_delegates_exact_arguments_once_and_preserves_reply(self):
        class Fold:
            def to_payload(self):return {'exact':'fold'}
        calls=[]
        class Native:
            def search_batched_multi_encoded(self,*args,**kwargs):
                calls.append((args,kwargs));return '{"side_one":[],"side_two":[],"prior_fallbacks":0,"capture_refusals":[]}'
        with tempfile.TemporaryDirectory() as d:
            witness=s.NativeWitness(Native(),d,s.canonical);fold=Fold();args=('state',2048,64,'tables',{},'ctx',fold,8,1.4,123,True,True)
            reply=witness.search_batched_multi_encoded(*args,test_kw=True)
            self.assertEqual(len(calls),1);self.assertIs(calls[0][0][6],fold);self.assertEqual(calls[0][0],args)
            self.assertEqual(calls[0][1],{'test_kw':True});self.assertTrue(witness.rows[0]['completed'])
            self.assertEqual(witness.rows[0]['native_seed'],123);self.assertEqual(json.loads(reply)['prior_fallbacks'],0)
            self.assertTrue((Path(d)/'native-00-input.json').is_file())

    def test_native_failure_retains_started_input_and_does_not_repeat(self):
        class Fold:
            def to_payload(self):return {}
        calls=[]
        class Native:
            def search_batched_multi_encoded(self,*args,**kwargs):calls.append(1);raise ValueError('authored refusal')
        with tempfile.TemporaryDirectory() as d:
            witness=s.NativeWitness(Native(),d,s.canonical)
            with self.assertRaisesRegex(ValueError,'authored refusal'):witness.search_batched_multi_encoded('x',1,1,'t',{},'c',Fold(),8,1.4,4)
            self.assertEqual(calls,[1]);self.assertFalse(witness.rows[0]['completed']);self.assertTrue((Path(d)/'native-00-input.json').exists())

    def test_roles_shapes_and_unselected_malformed_observations_fail(self):
        obs=dict(categorical_ids=[[0]*41 for _ in range(23)],numeric_features=[[0.]*132 for _ in range(23)],token_type_ids=[0]*23,
                 attention_mask=[True]*23,legal_action_mask=[True]+[False]*8)
        row=dict(requests={p:{'side':{'id':p},'active':[{}]} for p in ('p1','p2')},requested_players=['p1','p2'],
                 observations={p:copy.deepcopy(obs) for p in ('p1','p2')},actions={'p1':0,'p2':0})
        self.assertEqual(s.roles(row),['p1','p2'])
        for mutate in (lambda r:r['observations']['p2']['numeric_features'][0].pop(),lambda r:r['requested_players'].append('p1'),lambda r:r['actions'].update(p2=8)):
            bad=copy.deepcopy(row);mutate(bad)
            with self.assertRaises(ValueError):s.roles(bad)

    def test_freeze_reads_only192_train_battles_and_persists_before_any_search(self):
        obs=dict(categorical_ids=[[0]*41 for _ in range(23)],numeric_features=[[0.]*132 for _ in range(23)],token_type_ids=[0]*23,
                 attention_mask=[True]*23,legal_action_mask=[True]+[False]*8)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);roster=[];receipts={};contract={'authored':'no weights'}
            for i in range(256):
                battle=dict(index=i,battle_id='synthetic-'+str(i),seed=i,split='train' if i<192 else 'test',split_key=str(i));roster.append(battle)
                if i>=192:continue  # Any holdout read would fail: no directory exists.
                rows=[];prefix=[]
                for turn in range(2):
                    start=len(prefix);prefix.append('|turn|'+str(turn+1));requests={p:{'side':{'id':p},'active':[{}]} for p in ('p1','p2')}
                    row=dict(battle_id=battle['battle_id'],seed=i,decision_round_index=turn,protocol_start=start,protocol_end=len(prefix),protocol_delta=prefix[-1:],
                        public_prefix_sha256=s.digest(prefix),requests=requests,requested_players=['p1','p2'],observations={'p1':obs,'p2':obs},actions={'p1':0,'p2':0})
                    row['root_id']=s.digest([battle['battle_id'],turn,row['public_prefix_sha256'],requests]);rows.append(row)
                rows.append(dict(event='end',protocol_start=len(prefix),protocol_end=len(prefix),protocol_delta=[]))
                path=root/f'battle-{i:04d}';path.mkdir();blob=gzip.compress(b'\n'.join(s.canonical(r) for r in rows))
                (path/'boundaries.jsonl.gz').write_bytes(blob)
                receipt=dict(battle,contract=contract,files={'boundaries.jsonl.gz':s.file_sha(path/'boundaries.jsonl.gz')})
                s.write(path/'battle.json',receipt);receipts[f'battle-{i:04d}/battle.json']=s.file_sha(path/'battle.json')
            cohort=dict(battles=roster,config={'champion_sha256':'fake-sha'})
            s.write(root/'cohort.json',cohort)
            manifest=dict(status='COMPLETE_FIXED_COHORT',config=cohort['config'],cohort_sha256=s.file_sha(root/'cohort.json'),battle_receipt_hashes=receipts)
            s.write(root/'collection.json',manifest)
            plan=dict(collection=str(root),collection_sha256=s.file_sha(root/'collection.json'),cohort_sha256=s.file_sha(root/'cohort.json'),checkpoint_sha256='fake-sha',checkpoint_contract=contract,excluded_root_ids=[],source_bundle=str(root/'bank.tar.gz'))
            bank=s.freeze(plan,root/'bank.json')
            self.assertEqual(bank['count'],256);self.assertEqual(len({r['root_id'] for r in bank['roots']}),256)
            self.assertEqual(bank['search_calls'],0);self.assertTrue((root/'bank.json').is_file())
            self.assertFalse(any(int(name.split('/')[0].split('-')[-1])>=192 for name in bank['source_members'] if name.startswith('battle-')))
            with tarfile.open(root/'bank.tar.gz') as archive:
                self.assertEqual({m.name for m in archive.getmembers()}, {'root-bank.json'}|{'collection/'+n for n in bank['source_members']})
                self.assertTrue(all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in archive.getmembers()))
            with self.assertRaises(FileExistsError):s.freeze(plan,root/'bank.json')

if __name__=='__main__':unittest.main(verbosity=2)
