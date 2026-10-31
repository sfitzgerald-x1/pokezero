"""Synthetic ledger/fake-game controls only; no game, model or cluster access."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from pokezero import owning_encoder_continuation as cont
from pokezero import owning_encoder_study as study
from tests import test_owning_encoder_study as fixture_source


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_source.StudyTests('test_fixed_full_roster_nonoverlap_and_counterbalanced_order')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.spec, self.root = self.fixture.spec, self.fixture.root
        self.ledger = self.fixture.empty_ledger()
        self.ledger['invocations'][0]['status'] = 'RUNNING'
        for offset in range(325):self.fixture.add(self.ledger, 0, offset, 1 if offset % 2 else 0)

    def plan(self, *, ledger=None, possible=(325,), **changes):
        ledger = self.ledger if ledger is None else ledger
        refs = dict(prespec=self.fixture.put('original-prespec.json', self.spec),
            declaration=self.fixture.put('original-declaration.json', dict(original_study=self.spec['study_id'])),
            ledger=self.fixture.put('original-ledger.json', ledger))
        closure = dict(schema=cont.CLOSURE_SCHEMA, source_run_closed=True,
            **{key+'_sha256':value['sha256'] for key,value in refs.items()},
            potentially_attempted_indices=list(possible), completed_game_duration_upper_bound_seconds=100,
            timing_evidence=self.fixture.put('original-timing.json', dict(synthetic_elapsed_seconds=100)))
        closure.update(changes)
        return cont.make_plan(**refs, closure=self.fixture.put('closure.json', closure), cpu_pairs=((0,1),(2,3)))

    def receipt(self, index, pair, *, duration=3):
        ordinal, offset = divmod(index, study.COUNT)
        document = fixture_source.result(self.spec, ordinal, [fixture_source.game(self.spec, ordinal, offset)]).to_dict()
        document.update(seed_start=self.spec['seed_start']+offset, games=1,
            foulplay_random_seed_schedule=dict(seeds=[self.spec['foulplay_seed_start']+offset]))
        return dict(index=index, cpu_pair=list(pair), document=document, precision={'synthetic':True},
            timing=dict(started_epoch=1000, completed_epoch=1000+duration, duration_seconds=duration))

    def test_bindings_original_order_and_uncertain_first_cell_protection(self):
        plan = self.plan()
        self.assertEqual(plan['ordered_indices'], list(range(326, 2000)))
        self.assertEqual(plan['potentially_attempted_indices'], [325])
        self.assertEqual(plan['max_workers'], 2)
        cont.validate_plan(plan)
        altered = deepcopy(plan); altered['ordered_indices'].reverse()
        with self.assertRaisesRegex(ValueError, 'plan drift'):cont.validate_plan(altered)
        for changes in (dict(source_run_closed=False), dict(potentially_attempted_indices=[]),
                dict(ledger_sha256='e'*64)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):self.plan(**changes)
        plan = self.plan()
        # Hash drift refuses before runtime/model/process setup.
        Path(plan['references']['ledger']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'artifact hash'):cont.validate_plan(plan)

    def test_worker_slots_require_disjoint_dedicated_cores_and_bound(self):
        plan = self.plan()
        for pairs in (((0,1),(2,1)), ((0,0),), (), tuple((i*2,i*2+1) for i in range(17)), ((True,1),)):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                cont.make_plan(**plan['references'], cpu_pairs=pairs)
        for value in (1, 0, float('nan'), True):
            with self.subTest(cap=value), self.assertRaises(ValueError):
                cont.make_plan(**plan['references'], cpu_pairs=((0,1),), total_timeout_seconds=value)

    def test_only_same_original_cell_config_and_result_can_complete(self):
        for index in (326, 500, 1499, 1999):
            config = cont.single_config(self.spec, index, 7)
            ordinal, offset = divmod(index, 500)
            self.assertEqual((config.games,config.seed_start,config.foulplay_random_seed,config.foulplay_cpu_affinity),
                (1,self.spec['seed_start']+offset,self.spec['foulplay_seed_start']+offset,7))
            self.assertEqual((config.encoder_kind,config.pokezero_player), study.ORDER[ordinal])
            cont.checked_result(self.receipt(index,[6,7])['document'], self.spec, index)
        for field,value in (('seed_start', 123), ('games',2), ('checkpoint_sha256','a'*64)):
            document=self.receipt(326,[0,1])['document'];document[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                cont.checked_result(document,self.spec,326)

    def test_fake_concurrency_fixed_dispatch_and_failure_never_retries_or_changes_completed_cells(self):
        plan=self.plan(); output=self.root/'continuation'
        started=[]; running=set(); maximum=0
        async def worker(index,pair):
            nonlocal maximum
            # Attempt journal exists before fake invocation too.
            durable=json.loads((output/'ledger.json').read_bytes())
            self.assertEqual(durable['cells'][index]['status'],'ATTEMPTED')
            started.append(index);running.add(index);maximum=max(maximum,len(running))
            try:
                if index==326:await asyncio.Event().wait()
                if index==328:raise RuntimeError('authored fake game failure')
                await asyncio.sleep(0)
                return self.receipt(index,pair)
            finally:running.remove(index)
        ledger=asyncio.run(cont.run_continuation(plan,output,worker=worker))
        self.assertEqual(started,[326,327,328])
        self.assertEqual(maximum,2)
        self.assertEqual(ledger['actual_exit'],1)
        self.assertEqual(ledger['cells'][:325],self.ledger['cells'][:325])
        self.assertEqual([ledger['cells'][i]['status'] for i in (325,326,327,328,329)],
            ['UNKNOWN_ATTEMPTED','UNKNOWN_ATTEMPTED','COMPLETE','UNKNOWN_ATTEMPTED','NOT_ATTEMPTED'])
        self.assertEqual(ledger['cells'][327]['timing']['duration_seconds'],3)
        self.assertFalse(ledger['B3_qualified'])
        self.assertEqual(running,set())
        self.assertEqual(study.read_bound(plan['references']['ledger']),self.ledger)
        with self.assertRaises(FileExistsError):asyncio.run(cont.run_continuation(plan,output,worker=worker))

    def test_unknown_attempted_prior_cell_is_immutable_and_never_retried(self):
        original=deepcopy(self.ledger)
        original['continuation']=dict(schema=cont.SCHEMA)
        original['cells'][325]['status']='ATTEMPTED'
        original['cells'][326]['status']='CAPPED'
        original['cells'][327]['status']='FAILED'
        plan=self.plan(ledger=original,possible=())
        self.assertEqual(plan['ordered_indices'][0],328)
        async def fail(index,pair):raise RuntimeError('authored stop')
        ledger=asyncio.run(cont.run_continuation(plan,self.root/'prior-unknown',worker=fail))
        self.assertEqual(ledger['cells'][:328],original['cells'][:328])
        self.assertEqual(ledger['cells'][325]['status'],'ATTEMPTED')

    def test_stall_floor_and_distant_cap_use_completion_not_worker_activity(self):
        for elapsed, message in ((28801,'stall window'),(cont.TOTAL_SECONDS+1,'stall window')):
            plan=self.plan();calls=iter((0,0,elapsed))
            def clock():return next(calls,elapsed)
            async def waiting(index,pair):await asyncio.Event().wait()
            ledger=asyncio.run(cont.run_continuation(plan,self.root/f'stall-{elapsed}',worker=waiting,clock=clock))
            self.assertIn(message,ledger['failure']['error_message'])
            self.assertEqual(ledger['continuation']['stall_window_seconds'],28800)
            self.assertEqual(ledger['continuation']['dispatch_order'],[326,327])
            self.assertEqual(ledger['actual_exit'],1)

    def test_completion_updates_longest_duration_stall_window_without_changing_deadline(self):
        plan=self.plan()
        async def worker(index,pair):
            if index==326:return self.receipt(index,pair,duration=10000)
            raise RuntimeError('authored bounded stop')
        ledger=asyncio.run(cont.run_continuation(plan,self.root/'long-duration',worker=worker))
        self.assertEqual(ledger['continuation']['stall_window_seconds'],40000)
        self.assertEqual(plan['total_timeout_seconds'],cont.TOTAL_SECONDS)

    def test_distant_total_cap_remains_separate_from_healthy_recent_completion(self):
        plan=self.plan();calls=iter((0,cont.TOTAL_SECONDS+1,cont.TOTAL_SECONDS+1))
        async def worker(index,pair):
            if index==326:return self.receipt(index,pair)
            await asyncio.Event().wait()
        ledger=asyncio.run(cont.run_continuation(plan,self.root/'distant-cap',worker=worker,
            clock=lambda:next(calls,cont.TOTAL_SECONDS+1)))
        self.assertIn('distant total-time',ledger['failure']['error_message'])
        self.assertEqual(ledger['cells'][326]['status'],'COMPLETE')

    def test_default_foulplay_command_unchanged_and_opt_in_child_affinity(self):
        from pokezero.foulplay_bridge import _foulplay_command
        config=study._config(self.spec,0)
        command=_foulplay_command(config,'ws://synthetic')
        self.assertNotIn('sched_setaffinity',command[2])
        pinned=_foulplay_command(replace(config,foulplay_cpu_affinity=7),'ws://synthetic')
        self.assertEqual(pinned[:2]+pinned[3:],command[:2]+command[3:])
        self.assertTrue(pinned[2].endswith(command[2]))
        self.assertIn('os.sched_setaffinity(0, {7})',pinned[2])
        for cpu in (True,-1,'7'):
            with self.assertRaises(ValueError):replace(config,foulplay_cpu_affinity=cpu)

    def test_distinct_physical_FP_cores_cannot_share_actor_SMT_siblings(self):
        pairs=((0,1),(2,3))
        mapping={0:(0,0),1:(0,1),2:(0,2),3:(0,3)}
        self.assertEqual(cont.cpu_allocation(pairs,available=range(4),topology=mapping.__getitem__)['1'],[0,1])
        for changed in ({**mapping,1:(0,0)}, {**mapping,3:(0,1)}):
            with self.assertRaisesRegex(ValueError,'physical cores'):
                cont.cpu_allocation(pairs,available=range(4),topology=changed.__getitem__)
        with self.assertRaisesRegex(ValueError,'absent'):
            cont.cpu_allocation(pairs,available=(0,1,2),topology=mapping.__getitem__)

    def test_child_launch_uses_module_and_cleanup_only_its_owned_process_group(self):
        from types import SimpleNamespace
        receipt=self.root/'cell.json';receipt.write_text(json.dumps(self.receipt(326,[0,1])))
        process=SimpleNamespace(pid=12345,returncode=0,wait=AsyncMock(return_value=0))
        with patch.object(asyncio,'create_subprocess_exec',AsyncMock(return_value=process)) as spawn, \
                patch.object(cont.os,'killpg') as stop:
            actual=asyncio.run(cont._child('plan.json',326,[0,1],receipt))
        self.assertEqual(actual['index'],326)
        self.assertEqual(spawn.call_args.args[1:3],('-m','pokezero.owning_encoder_continuation'))
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        stop.assert_called_once_with(12345,cont.signal.SIGTERM)

    def test_child_claim_requires_durable_attempt_and_cannot_reexecute(self):
        plan=self.plan();output=self.root/'claim';output.mkdir()
        original=deepcopy(self.ledger)
        original['continuation']=dict(plan_sha256=study.prespec_file_sha256(plan),dispatch_order=[326])
        original['cells'][326].update(status='ATTEMPTED',attempt=dict(cpu_pair=[0,1]))
        study.atomic_json(output/'ledger.json',original)
        receipt=output/'cell-326.json'
        cont.claim_cell(plan,326,[0,1],receipt)
        self.assertTrue(receipt.with_suffix('.attempted').exists())
        with self.assertRaises(FileExistsError):cont.claim_cell(plan,326,[0,1],receipt)
        original['cells'][327].update(status='COMPLETE')
        study.atomic_json(output/'ledger.json',original)
        with self.assertRaises(ValueError):cont.claim_cell(plan,327,[2,3],output/'cell-327.json')

    def test_known_worker_cap_is_retained_and_never_retried(self):
        plan=self.plan()
        async def capped(index,pair):
            raise cont.WorkerFailure(dict(index=index,status='CAPPED',failure=dict(error_type='CappedGame')))
        ledger=asyncio.run(cont.run_continuation(plan,self.root/'capped',worker=capped))
        self.assertEqual(ledger['cells'][326]['status'],'CAPPED')
        self.assertEqual(ledger['failure']['worker_receipt']['index'],326)
        self.assertIsNone(ledger['cells'][326]['score'])


if __name__=='__main__':unittest.main()
