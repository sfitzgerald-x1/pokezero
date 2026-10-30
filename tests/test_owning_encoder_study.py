"""Authored metadata/outcome/async fakes only; no model, encoder or game run."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from math import sqrt
import os
from pathlib import Path
import stat
from statistics import fmean, stdev
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from pokezero.live_protocol_actor import ActorContract
from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
from pokezero import owning_encoder_study as study


def game(spec, ordinal, offset, score=1):
    kind, seat = study.ORDER[ordinal]
    peer = 'p2' if seat == 'p1' else 'p1'
    seed = study.SEED_START+offset
    battle = f'battle-gen3randombattle-controlled-{seed}'
    winner = seat if score == 1 else peer if score == 0 else None
    players = {seat:'PokeZeroBot',peer:'FoulPlayBot'}
    terminal_line='|tie' if winner is None else '|win|'+players[winner]
    return dict(seed=seed, battle_id=battle, encoder_kind=kind, capped=False, decision_rounds=1,
        pokezero_decisions=1, pokezero_decision_players=[seat], pokezero_submitted_choice_players=[seat],
        winner=players.get(winner), tied=score == .5, pokezero_won=score == 1, pokezero_score=score,
        terminal_witness=dict(schema='pokezero.controlled-terminal-witness.v1',players=players,
            capped=False, decision_rounds=1, max_decision_rounds=spec['max_decision_rounds'],
            winner_player=winner, status='PUBLIC_TIE' if winner is None else 'PUBLIC_WIN',
            public_terminal_line=terminal_line),
        encoder_provenance=dict(encoder_kind=kind, status='CLOSED_UNQUALIFIED',
            input_source='ACTUAL_GETPLAYERSTREAMS_OWNER_LINES_UNQUALIFIED',
            owner_terminal_timeout_seconds=spec['owning_terminal_timeout_seconds'],
            owning_decision_timeout_seconds=spec['owning_decision_timeout_seconds'],
            owning_public_single_line=True,
            actor_namespace=ActorContract.champion().namespace, actual_owner_terminal_seen=True,
            completed_terminal_streams=['omniscient','p1','p2'], bridge_terminal_marker=True,
            bridge_started_witness=True, full_encoder_admitted=False, B3_qualified=False,
            actor_receipt=dict(status='TERMINAL_UNQUALIFIED', battle_id=battle, owner=seat,
                terminal_seen=True, error=None, close_error=None, prefix_sha256=study.digest([battle,seat]),
                terminal_lifecycle=dict(consumed=True,kind=terminal_line.split('|')[1],
                    line_sha256=hashlib.sha256(terminal_line.encode()).hexdigest(),
                    scope='OWNING_HOST_LIFECYCLE_NOT_A_DEVICE_FEATURE'))))


def result(spec, ordinal, rows):
    document = dict(encoder_kind=study.ORDER[ordinal][0], pokezero_player=study.ORDER[ordinal][1],
        checkpoint_sha256=ActorContract.champion().checkpoint_sha256,
        seed_start=study.SEED_START, games=study.COUNT, completed_games=len(rows),
        policy_mode='raw', format_id='gen3randombattle', game_results=rows,
        foulplay_random_seed_schedule=dict(seeds=list(range(study.FP_SEED_START,study.FP_SEED_START+len(rows)))),
        execution_integrity={k:0 for k in ('retries','errors','refusal_records',
            'refusal_recorder_instrument_errors','refusal_records_unrowed')})
    return SimpleNamespace(to_dict=lambda:deepcopy(document))


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch('subprocess.Popen', side_effect=AssertionError('no process')).start()
        patch('subprocess.run', side_effect=AssertionError('no runtime')).start()
        patch('pokezero.owning_encoder_study._configure_precision', side_effect=AssertionError('no real dependency setup')).start()
        self.spec = self.make_spec()

    def put(self, name, document):
        path=self.root/name; path.write_bytes(study.canonical(document)+b'\n')
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def gate_document(self):
        from pokezero.actual_saved_wire import (PRODUCER_SOURCE, PRODUCER_FILES,
            PRODUCER_COHORT_FILE_SHA, PRODUCER_ROSTER_DIGEST)
        n=10000
        columns=lambda width:[dict(index=i,cells=n*23,expected_cells=n*23,compared_cells=n*23,
            mismatched_cells=0,refused_cells=0,unprocessed_cells=0,maximum_absolute_error=0.) for i in range(width)]
        return dict(schema='pokezero.B2-actual-saved-wire-comparison.v1', visibility=study.VISIBILITY,
            actual_captured_wire=True, original_reference_values_compared=True,
            producer_source=PRODUCER_SOURCE,producer_scientific_file_sha256=deepcopy(PRODUCER_FILES),
            cohort_file_sha256=PRODUCER_COHORT_FILE_SHA,canonical_roster_digest=PRODUCER_ROSTER_DIGEST,
            full_encoder_admitted=False, raw_bridge_chunk_bytes_captured=False,
            selection_sha256='a'*64, selection_metadata_sha256='b'*64, collection_sha256='c'*64,
            runtime=dict(contract=deepcopy(CHAMPION_CONTRACT)), metrics=dict(
                schema='pokezero.B2-actual-saved-wire-metrics.v1',visibility=study.VISIBILITY,
                actual_captured_wire=True,captured_player_wire=True,
                original_reference_values_compared=True, expected_views=n, views=n, processed_views=n,
                model_forward_views=n,complete_view_count=True,complete_roster_processed=True,model_forward_complete=True,
                refused_input_views=0,refused_forward_views=0,unprocessed_views=0,
                numeric_columns=columns(132),categorical_columns=columns(41),
                other_tensor_mismatched_cells=dict(token_type_ids=0,attention_mask=0,legal_action_mask=0),
                other_tensor_cells=dict(token_type_ids=n*23,attention_mask=n*23,legal_action_mask=n*9),
                other_tensor_expected_cells=dict(token_type_ids=n*23,attention_mask=n*23,legal_action_mask=n*9),
                other_tensor_refused_cells=dict(token_type_ids=0,attention_mask=0,legal_action_mask=0),
                policy_argmax_agreement=.999,legal_policy_argmax_agreement=.999,maximum_value_difference=.000099))

    def make_spec(self, document=None, review_change=None):
        document=self.gate_document() if document is None else document
        comparison=self.put('comparison.json',document)
        review=dict(verdict='CLEAR',comparison_sha256=comparison['sha256'],reviewed_commit='d'*40,
            checkpoint_sha256=ActorContract.champion().checkpoint_sha256,
            selection_sha256='a'*64,selection_metadata_sha256='b'*64,collection_sha256='c'*64)
        if review_change:review.update(review_change)
        admission=dict(comparison=comparison,scientific_review=self.put('review.json',review),source_commit='d'*40,
            selection_sha256='a'*64,selection_metadata_sha256='b'*64,collection_sha256='c'*64)
        return study.make_prespec(study_id='authored-source-fixture',
            runtime={key:'/authored/'+key for key in study.RUNTIME_KEYS},admission=admission,
            owning_decision_timeout_seconds=.25)

    def empty_ledger(self):
        return dict(schema=study.LEDGER_SCHEMA,prespec_sha256=study.prespec_file_sha256(self.spec),status='RUNNING',
            cells=study.planned_cells(),invocations=[dict(i,status='NOT_ATTEMPTED') for i in self.spec['invocations']],
            actual_exit=None,failure=None)

    def add(self, ledger, ordinal, offset, score):
        ledger['cells'][ordinal*study.COUNT+offset].update(
            study.checked_game(game(self.spec,ordinal,offset,score),self.spec,ordinal,offset))

    def test_fixed_full_roster_nonoverlap_and_counterbalanced_order(self):
        self.assertEqual(study.seed_audit()['consumed_unique_seeds'],4352)
        self.assertEqual(study.seed_audit()['overlap'],[])
        cells=study.planned_cells()
        self.assertEqual(len(cells),2000)
        for kind in ('python-owning','jax-owning'):
            keys={(c['seed'],c['seat'],c['foulplay_seed']) for c in cells if c['encoder_kind']==kind}
            self.assertEqual(len(keys),1000)
            self.assertEqual(keys,{(study.SEED_START+i,s,study.FP_SEED_START+i) for i in range(500) for s in ('p1','p2')})
        self.assertEqual([s[0] for s in study.ORDER],['python-owning','jax-owning','jax-owning','python-owning'])
        altered=deepcopy(self.spec);altered['invocations'].reverse()
        with self.assertRaisesRegex(ValueError,'prespec drift'):study.validate_prespec(altered)

    def test_actual_173_fp32_admission_accepts_honest_broad_false_flags(self):
        checked=study.validate_admission(self.spec)
        self.assertIs(checked['full_encoder_admitted'],False)
        self.assertIs(checked['raw_bridge_chunk_bytes_captured'],False)

    def test_gate_rejects_projected_stale_reference_unknown_denominators_and_numeric_failures(self):
        def projected(d):d['schema']='pokezero.B2-projected-player-collection-comparison.v1'
        mutations=[projected,lambda d:d.update(actual_captured_wire=False),
            lambda d:d.update(original_reference_values_compared=False),
            lambda d:d.update(producer_source='e'*40),
            lambda d:d.update(canonical_roster_digest='e'*64),
            lambda d:d['metrics'].update(captured_player_wire=False),
            lambda d:d.update(selection_sha256='e'*64),
            lambda d:d['runtime']['contract'].update(checkpoint_sha256='e'*64),
            lambda d:d['metrics'].update(refused_input_views=1),
            lambda d:d['metrics'].update(refused_forward_views=1),
            lambda d:d['metrics'].update(unprocessed_views=1),
            lambda d:d['metrics'].update(model_forward_views=9999),
            lambda d:d['metrics'].update(processed_views=9999),
            lambda d:d['metrics'].update(model_forward_views=10000.),
            lambda d:d['metrics']['numeric_columns'].pop(),
            lambda d:d['metrics']['categorical_columns'][0].update(index=1),
            lambda d:d['metrics']['numeric_columns'][0].update(maximum_absolute_error=1.01e-6),
            lambda d:d['metrics']['categorical_columns'][0].update(mismatched_cells=1),
            lambda d:d['metrics']['numeric_columns'][0].update(refused_cells=23),
            lambda d:d['metrics']['other_tensor_mismatched_cells'].update(legal_action_mask=1),
            lambda d:d['metrics']['other_tensor_cells'].update(legal_action_mask=1),
            lambda d:d['metrics'].update(policy_argmax_agreement=.9989),
            lambda d:d['metrics'].update(maximum_value_difference=.0001)]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                document=self.gate_document();mutation(document)
                with self.assertRaises(ValueError):study.validate_admission(self.make_spec(document))
        for change in ({'verdict':'PENDING'},{'reviewed_commit':'e'*40},{'comparison_sha256':'e'*64},
                {'checkpoint_sha256':'e'*64},{'selection_metadata_sha256':'e'*64}):
            with self.subTest(review=change),self.assertRaises(ValueError):
                study.validate_admission(self.make_spec(review_change=change))

    def test_missing_and_capped_cells_use_fixed_denominator_and_no_partial_interval(self):
        ledger=self.empty_ledger()
        self.add(ledger,1,0,1)
        ledger['cells'][0]['status']='CAPPED'
        readout=study.summarize(self.spec,ledger)
        self.assertEqual(readout['arms']['jax-owning']['known'],1)
        self.assertEqual(readout['arms']['python-owning']['unknown'],1000)
        self.assertEqual(readout['paired_jax_minus_python']['score_rate']['full_roster_bounds'],[-.999,1])
        self.assertIsNone(readout['paired_jax_minus_python']['score_rate']['interval90'])
        ledger['cells'][0]['score']=.5
        with self.assertRaisesRegex(ValueError,'unknown cell'):study.summarize(self.spec,ledger)

    def test_genuine_draw_terminal_owner_and_provenance_guards(self):
        row=game(self.spec,2,0,.5)
        self.assertEqual(study.checked_game(row,self.spec,2,0)['score'],.5)
        for mutate in (lambda r:r['encoder_provenance'].update(actual_owner_terminal_seen=False),
                lambda r:r['terminal_witness'].update(public_terminal_line='|tie|'),
                lambda r:r['encoder_provenance']['actor_receipt'].update(owner='p1'),
                lambda r:r['encoder_provenance']['actor_receipt']['terminal_lifecycle'].update(line_sha256='e'*64),
                lambda r:r.update(pokezero_submitted_choice_players=['p1']),
                lambda r:r.update(pokezero_score=True)):
            bad=deepcopy(row);mutate(bad)
            with self.assertRaises(ValueError):study.checked_game(bad,self.spec,2,0)
        row['capped']=True
        with self.assertRaises(study.CappedGame):study.checked_game(row,self.spec,2,0)
        row=game(self.spec,0,0,1);row['pokezero_score']=True
        with self.assertRaises(ValueError):study.checked_game(row,self.spec,0,0)

    def test_complete_four_fake_invocations_fresh_factories_and_seed_cluster_90_interval(self):
        calls=[];factories=[];published=[]
        patterns=((0,.5,1,0),(1,.5,1,0),(0,1,0,0),(1,0,.5,0))
        def factory(config):
            value=object();factories.append((config,value));return value
        async def benchmark(config,*,owning_actor_factory,progress_callback):
            ordinal=len(calls);calls.append((config.encoder_kind,config.pokezero_player))
            self.assertIs(owning_actor_factory,factories[-1][1])
            self.assertEqual(config.games,500);self.assertEqual(config.search_time_ms,1000)
            self.assertEqual(config.opponent_crash_retries,0);self.assertFalse(config.allow_search_fallback)
            rows=[game(self.spec,ordinal,i,patterns[ordinal][i%4]) for i in range(500)]
            answer=result(self.spec,ordinal,rows);progress_callback(answer);return answer
        def write(path,document):
            if Path(path).name=='ledger.json':published.append((document['status'],document['recorded_games']))
        with patch.object(study,'atomic_json',side_effect=write):
            ledger=asyncio.run(study.run_study(self.spec,self.root/'run',benchmark=benchmark,
                factory_builder=factory,configure=lambda:None))
        self.assertEqual(calls,list(study.ORDER));self.assertEqual(len({id(v) for _,v in factories}),4)
        self.assertEqual(ledger['actual_exit'],0);self.assertEqual(ledger['recorded_games'],2000)
        self.assertEqual(published[0],('PREPARING',0));self.assertEqual(published[-1],('COMPLETE',2000))
        delta=[fmean([patterns[1][i%4]==1,patterns[2][i%4]==1])-
               fmean([patterns[0][i%4]==1,patterns[3][i%4]==1]) for i in range(500)]
        from scipy.stats import t
        radius=float(t.ppf(.95,499))*stdev(delta)/sqrt(500)
        got=ledger['readout']['paired_jax_minus_python']['win_rate']
        self.assertAlmostEqual(got['point_estimate'],fmean(delta))
        self.assertAlmostEqual(got['interval90'][0],fmean(delta)-radius)
        self.assertAlmostEqual(got['interval90'][1],fmean(delta)+radius)
        self.assertFalse(ledger['readout']['B3_qualified']);self.assertFalse(ledger['readout']['equivalence_claim'])

    def test_failure_durable_prefix_failed_and_unattempted_no_later_invocations(self):
        calls=[]
        async def benchmark(config,**kwargs):
            calls.append(config)
            answer=result(self.spec,0,[game(self.spec,0,i) for i in range(2)])
            kwargs['progress_callback'](answer)
            error=RuntimeError('owning game round cap without actual owner terminal')
            error.owning_encoder_receipt={'status':'FAILED_CLOSED_UNQUALIFIED'}
            raise error
        ledger=asyncio.run(study.run_study(self.spec,self.root/'run',benchmark=benchmark,
            factory_builder=lambda _:object(),configure=lambda:None))
        saved=json.loads((self.root/'run'/'ledger.json').read_text())
        self.assertEqual(saved['prespec_sha256'],hashlib.sha256((self.root/'run'/'prespec.json').read_bytes()).hexdigest())
        self.assertEqual(saved,ledger);self.assertEqual(len(calls),1);self.assertEqual(ledger['actual_exit'],1)
        self.assertEqual([r['status'] for r in ledger['cells'][:4]],['COMPLETE','COMPLETE','CAPPED','NOT_ATTEMPTED'])
        self.assertEqual(sum(r['status']=='NOT_ATTEMPTED' for r in ledger['cells']),1997)
        self.assertEqual(ledger['failure']['seed'],study.SEED_START+2)
        self.assertIsNone(ledger['readout']['paired_jax_minus_python']['win_rate']['interval90'])
        with self.assertRaises(FileExistsError):
            asyncio.run(study.run_study(self.spec,self.root/'run',benchmark=benchmark,factory_builder=lambda _:object(),configure=lambda:None))

    def test_permission_and_admission_fail_before_game_calls_with_full_unknown_ledger(self):
        async def forbidden(*args,**kwargs):raise AssertionError('no game calls')
        def denied(_):raise PermissionError(13,'denied','/authored/showdown')
        ledger=asyncio.run(study.run_study(self.spec,self.root/'denied',benchmark=forbidden,
            factory_builder=denied,configure=lambda:None))
        self.assertEqual(ledger['failure']['missing_permission'],dict(errno=13,filename='/authored/showdown'))
        self.assertTrue(all(r['status']=='NOT_ATTEMPTED' for r in ledger['cells']))
        Path(self.spec['admission']['comparison']['path']).write_text('{}')
        ledger=asyncio.run(study.run_study(self.spec,self.root/'bad-gate',benchmark=forbidden,
            factory_builder=lambda _:self.fail('no factory'),configure=lambda:self.fail('no precision setup')))
        self.assertEqual(ledger['failure']['stage'],'admission')
        self.assertEqual(ledger['actual_exit'],1)
        self.assertEqual(ledger['readout']['paired_jax_minus_python']['score_rate']['full_roster_bounds'],[-1,1])

    def test_first_failure_preserves_bounded_message_and_throw_site_without_retry(self):
        calls=[]
        def authored_failure():
            private_local = 'must never enter failure evidence'
            raise ValueError('authored first-boundary failure ' + 'x' * 5000)
        async def benchmark(config,**kwargs):
            calls.append(config)
            authored_failure()
        ledger=asyncio.run(study.run_study(self.spec,self.root/'diagnostics',benchmark=benchmark,
            factory_builder=lambda _:object(),configure=lambda:None))
        saved=json.loads((self.root/'diagnostics'/'ledger.json').read_text())
        self.assertEqual(saved,ledger)
        failure=ledger['failure']
        self.assertEqual(failure['error_type'],'ValueError')
        self.assertTrue(failure['error_message'].startswith('authored first-boundary failure '))
        self.assertEqual(len(failure['error_message']),4096)
        self.assertTrue(failure['error_message_truncated'])
        self.assertEqual(failure['error_traceback'][-1]['function'],'authored_failure')
        self.assertTrue(failure['error_traceback'][-1]['file'].endswith(__file__))
        self.assertFalse(failure['error_traceback_truncated'])
        self.assertNotIn('private_local',study.canonical(failure).decode())
        self.assertNotIn('must never enter',study.canonical(failure).decode())
        self.assertEqual(len(calls),1)
        self.assertEqual(ledger['actual_exit'],1)
        self.assertEqual(ledger['recorded_games'],0)
        self.assertEqual([row['status'] for row in ledger['cells'][:2]],['FAILED','NOT_ATTEMPTED'])
        self.assertEqual(sum(row['status']=='NOT_ATTEMPTED' for row in ledger['cells']),1999)
        self.assertEqual(ledger['readout']['paired_jax_minus_python']['score_rate']['full_roster_bounds'],[-1,1])

    def test_failure_diagnostics_bounds_frames_and_survives_broken_exception_string(self):
        class BrokenMessage(ValueError):
            def __str__(self):raise RuntimeError('authored formatting failure')
        def nested(depth):
            if depth:nested(depth-1)
            else:raise BrokenMessage()
        try:nested(24)
        except BrokenMessage as error:diagnostics=study._failure_diagnostics(error)
        self.assertEqual(diagnostics['error_message'],'<exception message unavailable>')
        self.assertFalse(diagnostics['error_message_truncated'])
        self.assertEqual(len(diagnostics['error_traceback']),16)
        self.assertTrue(diagnostics['error_traceback_truncated'])
        self.assertEqual(diagnostics['error_traceback'][-1]['function'],'nested')
        self.assertTrue(all(set(frame)=={'file','line','function'}
                            for frame in diagnostics['error_traceback']))

    def test_broken_runtime_error_string_still_publishes_terminal_failed_ledger(self):
        class BrokenRuntimeError(RuntimeError):
            def __str__(self):raise ValueError('authored formatting failure')
        calls=[]
        async def benchmark(config,**kwargs):
            calls.append(config)
            raise BrokenRuntimeError()
        ledger=asyncio.run(study.run_study(self.spec,self.root/'broken-runtime-message',
            benchmark=benchmark,factory_builder=lambda _:object(),configure=lambda:None))
        saved=json.loads((self.root/'broken-runtime-message'/'ledger.json').read_text())
        self.assertEqual(saved,ledger)
        self.assertEqual(ledger['status'],'FAILED')
        self.assertEqual(ledger['actual_exit'],1)
        self.assertEqual(ledger['failure']['error_type'],'BrokenRuntimeError')
        self.assertEqual(ledger['failure']['error_message'],'<exception message unavailable>')
        self.assertEqual(ledger['failure']['stage'],'benchmark')
        self.assertEqual(ledger['failure']['seed'],study.SEED_START)
        self.assertEqual(ledger['failure']['error_traceback'][-1]['function'],'benchmark')
        self.assertEqual(len(calls),1)
        self.assertEqual(ledger['recorded_games'],0)
        self.assertEqual(len(ledger['cells']),2000)
        self.assertEqual(ledger['cells'][0]['status'],'FAILED')
        self.assertEqual(sum(row['status']=='NOT_ATTEMPTED' for row in ledger['cells']),1999)
        self.assertTrue(all(row['win'] is None and row['score'] is None for row in ledger['cells']))
        self.assertEqual([row['status'] for row in ledger['invocations']],
                         ['FAILED','NOT_ATTEMPTED','NOT_ATTEMPTED','NOT_ATTEMPTED'])
        self.assertEqual(ledger['readout']['paired_jax_minus_python']['score_rate']['full_roster_bounds'],[-1,1])
        self.assertFalse(ledger['B3_qualified'])

    def test_progress_write_failure_aborts_and_publishes_failure_without_game_retry(self):
        real_write=study.atomic_json
        writes=[];calls=[]
        def write(path, document):
            if Path(path).name=='ledger.json' and document['status']=='RUNNING' \
                    and document['recorded_games']==1 and not writes:
                writes.append('failed');raise OSError('authored durable write failure')
            return real_write(path,document)
        async def benchmark(config,**kwargs):
            calls.append(True)
            kwargs['progress_callback'](result(self.spec,0,[game(self.spec,0,0)]))
            self.fail('failed progress must abort the invocation')
        with patch.object(study,'atomic_json',side_effect=write):
            ledger=asyncio.run(study.run_study(self.spec,self.root/'write-fault',benchmark=benchmark,
                factory_builder=lambda _:object(),configure=lambda:None))
        saved=json.loads((self.root/'write-fault'/'ledger.json').read_text())
        self.assertEqual(saved,ledger);self.assertEqual(calls,[True])
        self.assertEqual(ledger['actual_exit'],1);self.assertEqual(ledger['failure']['error_type'],'OSError')
        self.assertEqual(ledger['cells'][1]['status'],'FAILED')

    def test_atomic_publication_orders_file_sync_replace_and_directory_sync(self):
        events=[]
        original_open,original_sync,original_replace,original_close=os.open,os.fsync,os.replace,os.close
        def opened(path, flags, *args, **kwargs):
            fd=original_open(path,flags,*args,**kwargs)
            if Path(path)==self.root:events.append('directory-open')
            return fd
        def synced(fd):
            events.append('directory-sync' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file-sync')
            return original_sync(fd)
        def replaced(*args,**kwargs):
            events.append('replace');return original_replace(*args,**kwargs)
        def closed(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):events.append('directory-close')
            return original_close(fd)
        with patch.object(study.os,'open',side_effect=opened),patch.object(study.os,'fsync',side_effect=synced), \
             patch.object(study.os,'replace',side_effect=replaced),patch.object(study.os,'close',side_effect=closed):
            study.atomic_json(self.root/'ordered.json',{'authored':1})
        self.assertEqual(events,['file-sync','replace','directory-open','directory-sync','directory-close'])
        self.assertEqual(json.loads((self.root/'ordered.json').read_text()),{'authored':1})

    def test_initial_directory_sync_failure_blocks_all_setup_and_game_calls(self):
        original_sync=os.fsync;failed=[];closed=[];original_close=os.close
        def synced(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                failed.append(fd);raise OSError('authored directory sync failure')
            return original_sync(fd)
        def close(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):closed.append(fd)
            return original_close(fd)
        async def forbidden(*args,**kwargs):self.fail('no benchmark before durable initial publications')
        with patch.object(study.os,'fsync',side_effect=synced),patch.object(study.os,'close',side_effect=close):
            with self.assertRaisesRegex(OSError,'directory sync failure'):
                asyncio.run(study.run_study(self.spec,self.root/'sync-fault',benchmark=forbidden,
                    factory_builder=lambda _:self.fail('no factory'),configure=lambda:self.fail('no precision setup')))
        self.assertEqual(len(failed),1);self.assertEqual(closed,failed)

    def test_new_output_parent_and_initial_publications_sync_before_setup(self):
        events=[];original_sync=study._sync_directory;original_write=study.atomic_json
        output=self.root/'durable-output'
        def sync(path):
            events.append(('sync',Path(path)));return original_sync(path)
        def write(path,document):
            events.append(('publish',Path(path).name));return original_write(path,document)
        def configure():events.append(('configure',None))
        def factory(_):events.append(('factory',None));return object()
        async def benchmark(*args,**kwargs):
            events.append(('benchmark',None));raise RuntimeError('authored stop')
        with patch.object(study,'_sync_directory',side_effect=sync),patch.object(study,'atomic_json',side_effect=write):
            ledger=asyncio.run(study.run_study(self.spec,output,benchmark=benchmark,
                factory_builder=factory,configure=configure))
        self.assertEqual(events[:6],[('sync',self.root),('publish','prespec.json'),('sync',output),
            ('publish','ledger.json'),('sync',output),('configure',None)])
        self.assertEqual(ledger['actual_exit'],1)
        with self.assertRaises(FileNotFoundError):
            asyncio.run(study.run_study(self.spec,self.root/'missing-parent'/'output',
                benchmark=benchmark,factory_builder=factory,configure=configure))

    def test_changed_prefix_wrong_fp_seed_and_wrong_seat_cannot_publish_scored_cells(self):
        for mode in ('changed','fp-seed','foreign-seat','duplicate','cap'):
            with self.subTest(mode=mode):
                async def benchmark(config,**kwargs):
                    rows=[game(self.spec,0,i) for i in range(2)]
                    kwargs['progress_callback'](result(self.spec,0,rows))
                    following=deepcopy(rows)+[game(self.spec,0,2)]
                    if mode=='changed':following[0]=game(self.spec,0,0,0)
                    if mode=='foreign-seat':following[2]['pokezero_submitted_choice_players']=['p2']
                    if mode=='duplicate':following[2]['seed']=study.SEED_START
                    if mode=='cap':following[2]['capped']=True
                    answer=result(self.spec,0,following)
                    if mode=='fp-seed':
                        original=answer.to_dict();original['foulplay_random_seed_schedule']['seeds'][-1]+=1
                        answer=SimpleNamespace(to_dict=lambda:original)
                    kwargs['progress_callback'](answer)
                    return answer
                ledger=asyncio.run(study.run_study(self.spec,self.root/mode,benchmark=benchmark,
                    factory_builder=lambda _:object(),configure=lambda:None))
                self.assertEqual(ledger['actual_exit'],1);self.assertEqual(ledger['recorded_games'],2)
                self.assertEqual(ledger['cells'][0]['score'],1)
                self.assertEqual(ledger['cells'][2]['status'],'CAPPED' if mode=='cap' else 'FAILED')

    def test_complete_scores_with_abnormal_exit_and_degenerate_variance_have_no_interval(self):
        ledger=self.empty_ledger()
        for ordinal in range(4):
            for offset in range(500):self.add(ledger,ordinal,offset,1)
        ledger['status']='COMPLETE';ledger['actual_exit']=0
        for invocation in ledger['invocations']:invocation['status']='COMPLETE'
        got=study.summarize(self.spec,ledger)['paired_jax_minus_python']['win_rate']
        self.assertEqual(got['point_estimate'],0);self.assertIsNone(got['interval90'])
        self.assertEqual(got['interval_status'],'DEGENERATE_VARIANCE')
        ledger['status']='FAILED';ledger['actual_exit']=1
        got=study.summarize(self.spec,ledger)['paired_jax_minus_python']['win_rate']
        self.assertEqual(got['interval_status'],'UNAVAILABLE_INCOMPLETE_OR_ABNORMAL')
        ledger['cells'][0]['encoder_provenance']['actual_owner_terminal_seen']=False
        with self.assertRaises(ValueError):study.summarize(self.spec,ledger)

    def test_prepare_cli_writes_only_prespec_no_admission_or_runtime_calls(self):
        path=Path(__file__).resolve().parents[1]/'scripts'/'owning_encoder_study.py'
        module_spec=importlib.util.spec_from_file_location('owning_study_cli',path)
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        inputs={key:self.spec[key] for key in ('study_id','runtime','admission','max_decision_rounds','owning_terminal_timeout_seconds','owning_decision_timeout_seconds')}
        source=self.put('input.json',inputs)['path'];out=self.root/'prepared.json'
        with patch.object(module,'run_study',side_effect=AssertionError('no run')):
            self.assertEqual(module.main(['prepare','--input',source,'--out',str(out)]),0)
        self.assertEqual(json.loads(out.read_text()),self.spec)
        with self.assertRaises(FileExistsError):module.main(['prepare','--input',source,'--out',str(out)])
        spec_ref=dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest())
        ledger_ref=self.put('ledger.json',self.empty_ledger())
        reduced=self.root/'reduced.json'
        self.assertEqual(module.main(['reduce','--prespec',spec_ref['path'],'--prespec-sha256',spec_ref['sha256'],
            '--ledger',ledger_ref['path'],'--ledger-sha256',ledger_ref['sha256'],'--out',str(reduced)]),0)
        self.assertEqual(json.loads(reduced.read_text())['paired_jax_minus_python']['win_rate']['full_roster_bounds'],[-1,1])
        with patch.object(module,'run_study',new=AsyncMock(return_value=dict(status='FAILED',actual_exit=1,recorded_games=0))):
            self.assertEqual(module.main(['run','--prespec',spec_ref['path'],'--prespec-sha256',spec_ref['sha256'],
                '--out',str(self.root/'fake-run')]),1)


if __name__=='__main__':unittest.main()
