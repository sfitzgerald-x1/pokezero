"""Fake owned workers/affinity only. No FP, models, battles or processes."""
import asyncio
from dataclasses import replace
import json
import contextvars
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock,patch

from pokezero import b4_fidelity_canary as c,b4_canary_endpoint as e,foulplay_bridge as fp
import test_b4_fidelity_canary as fixture


def literal(request,plan,*,error=None,capped=False):
    cell=request['cell'];config=e.configuration(plan,cell,request['checkpoint']);win=cell['arm']=='CANARY'
    name=config.pokezero_username if win else config.foulplay_username
    side=config.pokezero_player if win else ('p2' if config.pokezero_player=='p1' else 'p1')
    game=dict(seed=config.seed_start,capped=capped,tied=False,winner=name,pokezero_won=win,
        terminal_witness=dict(status='PUBLIC_WIN',winner_player=side,public_terminal_line='|win|'+name))
    benchmark=dict(checkpoint_sha256=request['checkpoint_sha256'],policy_mode='raw',seed_start=config.seed_start,
        pokezero_player=config.pokezero_player,foulplay_random_seed_schedule={'seeds':[config.foulplay_random_seed]},game_results=[game])
    return dict(schema='pokezero.b4.canary.fp-cell.v1',request_sha256=c.sha(request),cell=cell,cpu_pair=request['cpu_pair'],
        checkpoint_sha256=request['checkpoint_sha256'],duration_seconds=2.,status='UNKNOWN' if error or capped else 'COMPLETE',
        score=None if error or capped else float(win),error=error,source_game=game,benchmark=benchmark)


class ConcurrentEndpointTests(unittest.TestCase):
    def setUp(self):
        for target,name,value in [(c,'DECLARED_GAMES',4),(c,'PANEL_SEEDS',2)]:
            mock=patch.object(target,name,value);mock.start();self.addCleanup(mock.stop)

    def prepare(self,root,*,pairs=((0,1),(2,3),(4,5),(6,7))):
        recipe=replace(fixture.plan(),fp_cpu_pairs=pairs)
        result,_,_=fixture.CanaryTests()._run(root,recipe)
        return recipe,result['checkpoint']

    def run_panel(self,root,recipe,checkpoint,launcher):
        real=c.file_sha
        with patch.object(c,'file_sha',side_effect=lambda p:c.CHAMPION_SHA if str(p)=='/untouched' else real(p)),\
             patch.object(e,'check_placement',return_value={}),patch.object(e,'launch',side_effect=launcher),\
             patch.object(e.os,'sched_setaffinity',side_effect=AssertionError('controller affinity must not mutate'),create=True):
            return asyncio.run(c.evaluate(recipe,root,champion_checkpoint='/untouched',final_checkpoint=checkpoint,enable_canary=True))

    def test_four_slots_preserve_start_order_when_finishes_are_out_of_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';recipe,checkpoint=self.prepare(root);starts=[];finishes=[];active=0;maximum=0
            async def fake(path):
                nonlocal active,maximum
                request=json.loads(Path(path).read_bytes());cell=request['cell'];index=cell['ordinal'];starts.append(index)
                book=c.Journal(root,recipe,resume=True)
                self.assertEqual(book.matching('FP_START',ordinal=index)[0]['cpu_pair'],list(recipe.fp_cpu_pairs[(index+index//4)%4]))
                self.assertTrue((root/'fp-starts'/f'{index}.json').is_file())
                config=e.configuration(recipe,cell,request['checkpoint']);self.assertEqual(config.foulplay_cpu_affinity,recipe.fp_cpu_pairs[(index+index//4)%4][1])
                command=fp._foulplay_command(config,'ws://unit-not-opened',run_count=1)
                self.assertIn('os.sched_setaffinity(0, {'+str(config.foulplay_cpu_affinity)+'})',command[2])
                active+=1;maximum=max(maximum,active)
                await asyncio.sleep(.025 if index==0 else .001)
                value=literal(request,recipe);e._write(root/'fp-results'/Path(path).name,value)
                active-=1;finishes.append(index);return value
            report=self.run_panel(root,recipe,checkpoint,fake)
            self.assertEqual(starts,list(range(8)));self.assertEqual(maximum,4);self.assertNotEqual(finishes,list(range(8)))
            self.assertEqual(report['known_pairs'],4);self.assertEqual(report['max_workers'],4)

    def test_runtime_failure_leaves_true_na_then_resume_never_retries_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';recipe,checkpoint=self.prepare(root);calls=[]
            async def fake(path):
                request=json.loads(Path(path).read_bytes());index=request['cell']['ordinal'];calls.append(index)
                await asyncio.sleep(0 if index==0 else .005)
                value=literal(request,recipe,error='unit-transport-failure' if index==0 else None)
                e._write(root/'fp-results'/Path(path).name,value);return value
            report=self.run_panel(root,recipe,checkpoint,fake)
            self.assertEqual(calls,[0,1,2,3]);self.assertEqual(sum(r['status']=='NOT_ATTEMPTED' for r in report['rows']),4)
            again=self.run_panel(root,recipe,checkpoint,fake)
            self.assertEqual(calls,list(range(8)));self.assertEqual(again['unknown_pairs'],1);self.assertIsNone(again['rows'][0]['score'])

    def test_durable_completed_child_survives_missing_parent_end_without_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';recipe,checkpoint=self.prepare(root,pairs=((0,1),));calls=[]
            async def interrupted(path):
                request=json.loads(Path(path).read_bytes());calls.append(request['cell']['ordinal'])
                e._write(root/'fp-results'/Path(path).name,literal(request,recipe));raise asyncio.CancelledError()
            with self.assertRaises(asyncio.CancelledError):self.run_panel(root,recipe,checkpoint,interrupted)
            self.assertFalse(c.Journal(root,recipe,resume=True).matching('FP_END',ordinal=0))
            async def finish(path):
                request=json.loads(Path(path).read_bytes());calls.append(request['cell']['ordinal'])
                value=literal(request,recipe);e._write(root/'fp-results'/Path(path).name,value);return value
            result=self.run_panel(root,recipe,checkpoint,finish)
            self.assertEqual(calls,list(range(8)));self.assertEqual(result['known_pairs'],4)

    def test_literal_caps_and_wrong_seed_or_score_never_become_wins(self):
        recipe=fixture.plan();cell=e.cells(recipe)[0]
        request=dict(schema='pokezero.b4.canary.fp-request.v1',run_directory='/unit',declaration_sha256='a'*64,
            cell=cell,checkpoint='/untouched',checkpoint_sha256=c.CHAMPION_SHA,cpu_pair=list(recipe.fp_cpu_pairs[cell['slot']]))
        value=literal(request,recipe,capped=True);self.assertIsNone(e.checked_result(value,request,recipe)['score'])
        for mutation in ('score','seed','checkpoint'):
            value=literal(request,recipe)
            if mutation=='score':value['score']=1.
            if mutation=='seed':value['source_game']['seed']+=1
            if mutation=='checkpoint':value['checkpoint_sha256']='f'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):e.checked_result(value,request,recipe)

    def test_child_pins_only_own_threads_and_probes_fp_core_without_real_affinity(self):
        masks={11:{0},12:{0}};calls=[]
        def get(tid):return set(masks[11 if tid==0 else tid])
        def put(tid,mask):calls.append((tid,set(mask)));masks[11 if tid==0 else tid]=set(mask)
        with patch.object(e.os,'sched_getaffinity',side_effect=get,create=True),patch.object(e.os,'sched_setaffinity',side_effect=put,create=True),\
             patch.object(e.Path,'iterdir',side_effect=lambda:iter((Path('/proc/self/task/11'),Path('/proc/self/task/12')))):
            e.pin_owned_actor((0,1))
        self.assertEqual(calls,[(0,{1}),(11,{0}),(12,{0})]);self.assertEqual(masks,{11:{0},12:{0}})

    def test_owned_launcher_checks_group_cleanup_even_after_exited_leader(self):
        from pokezero import owning_encoder_continuation as owned
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);path=root/'fp-requests/0.json';e._write(path,{'unit':'not-science'})
                e._write(root/'fp-results/0.json',{'unit':'not-science'})
                process=NS(pid=987654321,returncode=0,wait=AsyncMock(return_value=0));cleanup=AsyncMock(side_effect=RuntimeError('owned group persists') if fail else None)
                with patch.object(e.asyncio,'create_subprocess_exec',new=AsyncMock(return_value=process)) as spawn,\
                     patch.object(owned,'_stop_owned_group',new=cleanup):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'group persists'):asyncio.run(e.launch(path))
                    else:self.assertEqual(asyncio.run(e.launch(path)),{'unit':'not-science'})
                self.assertTrue(spawn.call_args.kwargs['start_new_session']);cleanup.assert_awaited_once_with(process)

    def test_fixed_worker_checks_start_and_claim_before_original_fp_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';recipe,checkpoint=self.prepare(root,pairs=((0,1),(2,3)));calls=[];pins=[]
            async def benchmark(config):
                offset=config.seed_start-recipe.fp_configs[0]['seed_start'];seat=int(config.pokezero_player=='p2')
                arm=int(str(config.checkpoint)!='/untouched');ordinal=4*offset+2*seat+arm
                request=json.loads((root/'fp-requests'/f'{ordinal}.json').read_bytes())
                self.assertTrue((root/'fp-claims'/f'{ordinal}.json').is_file())
                self.assertEqual(pins[-1],tuple(request['cpu_pair']));calls.append(ordinal)
                value=literal(request,recipe);await asyncio.sleep(0)
                return NS(config=config,checkpoint_sha256=request['checkpoint_sha256'],
                    foulplay_random_seed_schedule=(config.foulplay_random_seed,),games=(NS(to_dict=lambda:value['source_game']),),
                    to_dict=lambda:value['benchmark'])
            async def invoke_worker(path):
                await e.worker(path,c.file_sha(path))
                return e._read_result(root/'fp-results'/Path(path).name)
            real=c.file_sha
            with patch.object(c,'file_sha',side_effect=lambda p:c.CHAMPION_SHA if str(p)=='/untouched' else real(p)),\
                 patch.object(e,'check_placement',return_value={}),patch.object(e,'pin_owned_actor',side_effect=lambda pair:pins.append(pair)),\
                 patch.object(e,'launch',side_effect=invoke_worker),patch.object(fp,'run_controlled_foulplay_benchmark',side_effect=benchmark):
                report=asyncio.run(c.evaluate(recipe,root,champion_checkpoint='/untouched',final_checkpoint=checkpoint,enable_canary=True))
                with self.assertRaises(FileExistsError):asyncio.run(e.worker(root/'fp-requests/0.json',c.file_sha(root/'fp-requests/0.json')))
            self.assertEqual(calls,list(range(8)));self.assertEqual(report['known_pairs'],4)

    def test_controller_placement_is_read_only_and_rejects_shared_fp_core(self):
        from pokezero import owning_encoder_continuation as placement
        recipe=replace(fixture.plan(),fp_cpu_pairs=((0,1),(2,3),(4,5),(6,7)))
        for unsafe in (False,True):
            with self.subTest(unsafe=unsafe),patch.object(e.Path,'is_file',autospec=True,side_effect=lambda p:str(p)=='/sys/fs/cgroup/cpuset.cpus.effective'),\
                 patch.object(e.Path,'read_text',return_value='0-7'),\
                 patch.object(e.Path,'iterdir',side_effect=lambda:iter((Path('/proc/self/task/11'),Path('/proc/self/task/12')))),\
                 patch.object(e.os,'sched_getaffinity',return_value={0,1} if unsafe else {0,2,4,6},create=True),\
                 patch.object(e.os,'sched_setaffinity',side_effect=AssertionError('parent must not mutate'),create=True),\
                 patch.object(placement,'cpu_allocation',return_value={}) as checked:
                if unsafe:
                    with self.assertRaisesRegex(ValueError,'controller threads'):e.check_placement(recipe)
                else:self.assertEqual(e.check_placement(recipe),{})
                checked.assert_called_once_with(recipe.fp_cpu_pairs,available=set(range(8)))

    def test_second_controller_cannot_dispatch_while_endpoint_lock_is_owned(self):
        import fcntl
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';recipe,checkpoint=self.prepare(root)
            with (root/'fp-invocation.lock').open('a+b') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):self.run_panel(root,recipe,checkpoint,AsyncMock())
                fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

    def test_scoped_term_is_idempotent_and_defers_prior_handler_until_cleanup(self):
        for failure in (False,True):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)/'run';recipe,checkpoint=self.prepare(root);closed=[];prior_calls=[];handlers={}
                real=c.file_sha
                async def scenario():
                    ready=asyncio.Event();release=asyncio.Event();never=asyncio.Event();started=[];loop=asyncio.get_running_loop()
                    prior=lambda sig,frame:prior_calls.append((sig,frame,len(closed)))
                    state=[prior];registry={}
                    def install(sig,callback,*args):
                        handlers[sig]=lambda:callback(*args);registry[sig]=asyncio.Handle(callback,args,loop);state[0]=lambda *_:None
                    def remove(sig):handlers.pop(sig,None);registry.pop(sig,None);state[0]=signal.SIG_DFL;return True
                    async def fake(path):
                        ordinal=int(Path(path).stem);started.append(ordinal)
                        if len(started)==4:ready.set()
                        try:await never.wait()
                        finally:
                            await release.wait();closed.append(ordinal)
                            if failure and ordinal==0:raise RuntimeError('owned cleanup failed')
                    with patch.object(loop,'_signal_handlers',registry),patch.object(loop,'add_signal_handler',side_effect=install),\
                         patch.object(loop,'remove_signal_handler',side_effect=remove),patch.object(e.signal,'getsignal',side_effect=lambda _:state[0]),\
                         patch.object(e.signal,'signal',side_effect=lambda _,value:state.__setitem__(0,value)),\
                         patch.object(c,'file_sha',side_effect=lambda p:c.CHAMPION_SHA if str(p)=='/untouched' else real(p)),\
                         patch.object(e,'check_placement',return_value={}),patch.object(e,'launch',side_effect=fake):
                        task=asyncio.create_task(c.evaluate(recipe,root,champion_checkpoint='/untouched',final_checkpoint=checkpoint,enable_canary=True))
                        await ready.wait();handlers[signal.SIGTERM]();await asyncio.sleep(0);handlers[signal.SIGTERM]()
                        task.cancel();await asyncio.sleep(0);task.cancel();await asyncio.sleep(0)
                        self.assertFalse(task.done());self.assertEqual(prior_calls,[]);release.set()
                        with self.assertRaisesRegex(RuntimeError,'cleanup failed') if failure else self.assertRaises(asyncio.CancelledError):await task
                        self.assertEqual(sorted(closed),list(range(4)));self.assertIs(state[0],prior)
                asyncio.run(scenario())
                self.assertEqual(prior_calls,[] if failure else [(signal.SIGTERM,None,4)])

    def test_prior_async_handle_and_context_are_restored_exactly(self):
        async def scenario():
            loop=asyncio.get_running_loop();variable=contextvars.ContextVar('unit-context');calls=[];handlers={}
            variable.set('original');prior=asyncio.Handle(lambda arg:calls.append((arg,variable.get())),('saved',),loop)
            variable.set('invocation');registry={signal.SIGTERM:prior};state=[lambda *_:None]
            def install(sig,callback,*args):handlers[sig]=lambda:callback(*args);registry[sig]=asyncio.Handle(callback,args,loop)
            def remove(sig):handlers.pop(sig,None);registry.pop(sig,None);return True
            async def body(*args,**kwargs):handlers[signal.SIGTERM]();await asyncio.sleep(0)
            with patch.object(loop,'_signal_handlers',registry),patch.object(loop,'add_signal_handler',side_effect=install),\
                 patch.object(loop,'remove_signal_handler',side_effect=remove),patch.object(e.signal,'getsignal',side_effect=lambda _:state[0]),\
                 patch.object(e.signal,'signal',side_effect=lambda _,value:state.__setitem__(0,value)),patch.object(e,'evaluate_declared',side_effect=body):
                with self.assertRaises(asyncio.CancelledError):await e.evaluate_with_sigterm(object(),object(),champion_checkpoint='/unit',final_checkpoint='/unit')
                self.assertIs(registry[signal.SIGTERM],prior);self.assertEqual(calls,[('saved','original')])
                registry[signal.SIGTERM]._run();self.assertEqual(calls,[('saved','original')]*2)
        asyncio.run(scenario())

    def test_cancellation_during_normal_reap_waits_and_preserves_cleanup_failure(self):
        from pokezero import owning_encoder_continuation as owned
        for failure in (False,True):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);path=root/'fp-requests/0.json';e._write(path,{'unit':'not-science'})
                async def scenario():
                    entered=asyncio.Event();release=asyncio.Event();ended=[]
                    async def cleanup(process):
                        entered.set();await release.wait();ended.append('reaped')
                        if failure:raise RuntimeError('owned group persists')
                    process=NS(pid=987654321,returncode=0,wait=AsyncMock(return_value=0))
                    with patch.object(e.asyncio,'create_subprocess_exec',new=AsyncMock(return_value=process)),\
                         patch.object(owned,'_stop_owned_group',side_effect=cleanup) as reap:
                        task=asyncio.create_task(e.launch(path));await entered.wait();task.cancel();await asyncio.sleep(0)
                        task.cancel();await asyncio.sleep(0);self.assertFalse(task.done());release.set()
                        with self.assertRaisesRegex(RuntimeError,'group persists') if failure else self.assertRaises(asyncio.CancelledError):await task
                        self.assertEqual(ended,['reaped']);self.assertEqual(reap.call_count,1)
                asyncio.run(scenario())

    def test_scope_does_not_replace_new_unrelated_parent_handler(self):
        async def scenario():
            loop=asyncio.get_running_loop();registry={};calls=[]
            parent=lambda:calls.append('parent')
            def install(sig,callback,*args):registry[sig]=asyncio.Handle(callback,args,loop)
            def remove(sig):registry.pop(sig,None);return True
            async def body(*args,**kwargs):install(signal.SIGTERM,parent);return 'unit-result'
            with patch.object(loop,'_signal_handlers',registry),patch.object(loop,'add_signal_handler',side_effect=install),\
                 patch.object(loop,'remove_signal_handler',side_effect=remove) as removed,\
                 patch.object(e.signal,'getsignal',return_value=signal.SIG_DFL),patch.object(e.signal,'signal') as setter,\
                 patch.object(e,'evaluate_declared',side_effect=body):
                result=await e.evaluate_with_sigterm(object(),object(),champion_checkpoint='/unit',final_checkpoint='/unit')
                self.assertEqual(result,'unit-result');self.assertIs(registry[signal.SIGTERM]._callback,parent)
                removed.assert_not_called();setter.assert_not_called();registry[signal.SIGTERM]._run()
            self.assertEqual(calls,['parent'])
        asyncio.run(scenario())

    def test_cancellation_during_spawn_keeps_ownership_until_process_is_reaped(self):
        from pokezero import owning_encoder_continuation as owned
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'fp-requests/0.json';e._write(path,{'unit':'not-science'})
            async def scenario():
                entered=asyncio.Event();release=asyncio.Event();reaped=[]
                process=NS(pid=987654321,returncode=None,wait=AsyncMock(return_value=0))
                async def spawn(*args,**kwargs):entered.set();await release.wait();return process
                async def reap(actual):self.assertIs(actual,process);reaped.append(actual.pid)
                with patch.object(e.asyncio,'create_subprocess_exec',side_effect=spawn),patch.object(owned,'_stop_owned_group',side_effect=reap):
                    task=asyncio.create_task(e.launch(path));await entered.wait();task.cancel();await asyncio.sleep(0)
                    task.cancel();await asyncio.sleep(0);self.assertFalse(task.done());self.assertEqual(reaped,[])
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):await task
                    self.assertEqual(reaped,[process.pid]);process.wait.assert_not_awaited()
            asyncio.run(scenario())


if __name__=='__main__':unittest.main()
