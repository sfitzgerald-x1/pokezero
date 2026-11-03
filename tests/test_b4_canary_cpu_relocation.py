"""Inert request/worker transport only; no model, game or real CPU mutation."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock,patch

from pokezero import b4_fidelity_canary as c,b4_canary_endpoint as e
import test_b4_fidelity_canary as fixture
import test_b4_canary_endpoint as endpoint_fixture


class CanaryCpuRelocationTests(unittest.TestCase):
    def setUp(self):
        for name,value in (('DECLARED_GAMES',4),('PANEL_SEEDS',2)):
            p=patch.object(c,name,value);p.start();self.addCleanup(p.stop)

    def prepare(self,root):
        p=fixture.plan()
        configs=tuple(dict(x,encoder_kind='jax-owning',device='cpu',belief_set_source=True,
            owning_decision_timeout_seconds=60.,allow_search_fallback=False,opponent_crash_retries=0)for x in p.fp_configs)
        p=replace(p,fp_configs=configs)
        result,_,_=fixture.CanaryTests()._run(root,p,device_path=True)
        return p,result['checkpoint']

    def panel(self,root,plan,checkpoint,pairs,launcher):
        real=c.file_sha
        options={}if pairs is None else dict(endpoint_cpu_pairs=pairs)
        with patch.object(c,'file_sha',side_effect=lambda p:c.CHAMPION_SHA if str(p)=='/untouched'else real(p)),\
             patch.object(e,'check_placement',return_value={'actual':'inert-topology'}) as placement,\
             patch.object(e.WorkerPool,'launch',side_effect=launcher),patch.object(e.WorkerPool,'close',new=AsyncMock()):
            value=asyncio.run(c.evaluate_device(plan,root,champion_checkpoint='/untouched',
                final_checkpoint=checkpoint,enable_canary=True,**options))
        return value,placement

    def test_runtime_pairs_are_finite_distinct_and_do_not_mutate_declared_cells(self):
        plan=fixture.plan();before=c.encoded(plan.manifest())
        self.assertIs(e._cpu_pairs(plan),plan.fp_cpu_pairs)
        for pairs in ((),((10,11),(11,12)),((10,True),(12,13)),((10,-1),(12,13)),tuple((2*i,2*i+1)for i in range(9))):
            with self.subTest(pairs=pairs),self.assertRaises(ValueError):e._cpu_pairs(plan,pairs)
        self.assertEqual(e._cpu_pairs(plan,[[10,11],[12,13]]),((10,11),(12,13)))
        self.assertEqual(e._cpu_pairs(plan,((10,11),)),((10,11),))
        self.assertEqual(e._cpu_pairs(plan,((10,11),(12,13),(14,15))),((10,11),(12,13),(14,15)))
        self.assertEqual(c.encoded(plan.manifest()),before)

    def test_all2000_scientific_cells_keep_old_slots_under_wider_physical_mapping(self):
        plan=fixture.plan()
        with patch.object(c,'PANEL_SEEDS',500):
            roster=e.cells(plan);pairs=e._cpu_pairs(plan,tuple((2*i,2*i+1)for i in range(64)))
            self.assertEqual(len(roster),2000)
            self.assertEqual([x['ordinal']for x in roster],list(range(2000)))
            for cell in roster:
                self.assertEqual(cell['slot'],(cell['ordinal']+cell['offset'])%2)
                self.assertEqual(e._execution_slot(cell,plan,pairs),cell['ordinal']%64)
            self.assertEqual(e.cells(plan),roster)

    def test_wider_runtime_uses_four_pairs_with_original_order_and_scores(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)/'run';plan,checkpoint=self.prepare(root)
            declaration=(root/'declaration.json').read_bytes();calls=[];active=0;maximum=0
            pairs=((10,11),(12,13),(14,15),(16,17))
            async def launch(path):
                nonlocal active,maximum
                request=json.loads(Path(path).read_bytes());cell=request['cell'];actual=e._request_pairs(request,plan)
                self.assertEqual(cell,e.cells(plan)[cell['ordinal']])
                self.assertEqual(request['cpu_pair'],list(pairs[cell['ordinal']%4]))
                config=e.configuration(plan,cell,request['checkpoint'],cpu_pairs=actual)
                self.assertEqual(config.foulplay_cpu_affinity,request['cpu_pair'][1])
                calls.append(cell['ordinal']);active+=1;maximum=max(maximum,active)
                await asyncio.sleep(.01 if cell['ordinal']==0 else .001)
                value=endpoint_fixture.literal(request,plan);value['benchmark']['encoder_kind']='jax-owning'
                e._write(root/'fp-results'/Path(path).name,value);active-=1;return value
            result,_=self.panel(root,plan,checkpoint,pairs,launch)
            self.assertEqual(calls,list(range(8)));self.assertEqual(maximum,4)
            self.assertEqual(result['max_workers'],4);self.assertEqual(result['known_pairs'],4)
            self.assertEqual((root/'declaration.json').read_bytes(),declaration)

    def test_cpu_only_transport_moves_actual_workers_but_preserves_order_seeds_and_training_prefix(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)/'run';plan,checkpoint=self.prepare(root)
            declaration=(root/'declaration.json').read_bytes();journal=(root/'journal.jsonl').read_bytes();calls=[]
            pairs=((10,11),(12,13))
            async def launch(path):
                request=json.loads(Path(path).read_bytes());cell=request['cell'];actual=e._request_pairs(request,plan)
                config=e.configuration(plan,cell,request['checkpoint'],cpu_pairs=actual)
                calls.append((cell['ordinal'],cell['slot'],config.seed_start,config.pokezero_player,config.foulplay_cpu_affinity))
                async def run(current,*,expected_checkpoint_sha256):
                    self.assertEqual(current,config);self.assertEqual(current.device,'cpu')
                    self.assertEqual(expected_checkpoint_sha256,request['checkpoint_sha256'])
                    value=endpoint_fixture.literal(request,plan);value['benchmark']['encoder_kind']='jax-owning'
                    return NS(config=current,checkpoint_sha256=expected_checkpoint_sha256,
                        foulplay_random_seed_schedule=(current.foulplay_random_seed,),
                        games=(NS(to_dict=lambda:value['source_game']),),to_dict=lambda:value['benchmark'])
                await e.worker(path,c.file_sha(path),runtime=NS(run=run),expected_pair=tuple(request['cpu_pair']))
                return e._read_result(root/'fp-results'/Path(path).name)
            result,placement=self.panel(root,plan,checkpoint,pairs,launch)
            self.assertEqual(calls,[(i,(i+i//4)%2,6000+i//4,'p1'if i%4<2 else'p2',pairs[(i+i//4)%2][1])for i in range(8)])
            self.assertEqual(result['known_pairs'],4);self.assertEqual(result['max_workers'],2)
            self.assertEqual((root/'declaration.json').read_bytes(),declaration)
            self.assertTrue((root/'journal.jsonl').read_bytes().startswith(journal))
            placement.assert_called_once_with(plan,cpu_pairs=pairs)
            record=e._read_result(root/'fp-cpu-runtime'/(result['endpoint_cpu_runtime_sha256']+'.json'))
            self.assertEqual(record['cpu_pairs'],list(map(list,pairs)))
            self.assertEqual(plan.fp_cpu_pairs,((0,1),(2,3)))

    def test_relocated_resume_keeps_old_requests_and_unknown_attempts_without_replay(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)/'run';plan,checkpoint=self.prepare(root);calls=[]
            async def failed(path):
                request=json.loads(Path(path).read_bytes());i=request['cell']['ordinal'];calls.append(i)
                value=endpoint_fixture.literal(request,plan,error='inert-failure'if i==0 else None)
                value['benchmark']['encoder_kind']='jax-owning';e._write(root/'fp-results'/Path(path).name,value);return value
            first,_=self.panel(root,plan,checkpoint,((10,11),(12,13)),failed)
            self.assertEqual(calls,[0,1]);self.assertEqual(first['unknown_pairs'],4)
            old={str(p):p.read_bytes()for p in (root/'fp-requests').glob('*.json')}
            snapshot=(root/'fp-snapshot.json').read_bytes()
            async def complete(path):
                request=json.loads(Path(path).read_bytes());calls.append(request['cell']['ordinal'])
                self.assertIn(tuple(request['cpu_pair']),((20,21),(22,23),(24,25),(26,27)))
                value=endpoint_fixture.literal(request,plan);value['benchmark']['encoder_kind']='jax-owning'
                e._write(root/'fp-results'/Path(path).name,value);return value
            result,_=self.panel(root,plan,checkpoint,((20,21),(22,23),(24,25),(26,27)),complete)
            self.assertEqual(calls,list(range(8)));self.assertEqual(result['unknown_pairs'],1)
            self.assertEqual((root/'fp-snapshot.json').read_bytes(),snapshot)
            for path,blob in old.items():self.assertEqual(Path(path).read_bytes(),blob)
            self.assertEqual(len(list((root/'fp-cpu-runtime').glob('*.json'))),2)

    def test_legacy_default_keeps_old_requests_and_pool_uses_runtime_pairs(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)/'run';plan,checkpoint=self.prepare(root)
            async def launch(path):
                request=json.loads(Path(path).read_bytes());self.assertNotIn('endpoint_cpu_runtime_sha256',request)
                value=endpoint_fixture.literal(request,plan);value['benchmark']['encoder_kind']='jax-owning'
                e._write(root/'fp-results'/Path(path).name,value);return value
            result,placement=self.panel(root,plan,checkpoint,None,launch)
            self.assertNotIn('endpoint_cpu_runtime_sha256',result);self.assertFalse((root/'fp-cpu-runtime').exists())
            placement.assert_called_once_with(plan)
            class Input:
                def write(self,line):pass
                async def drain(self):pass
            process=NS(returncode=None,stdin=Input(),stdout=NS(readline=AsyncMock(return_value=b'{"exit":0}\n')))
            pool=e.WorkerPool(root,plan,cpu_pairs=((30,31),(32,33)))
            async def run():
                await pool.launch(root/'fp-requests/0.json');await pool.close()
            with patch.object(e,'_spawn_owned',new=AsyncMock(return_value=process))as spawn,\
                 patch.object(e,'_finish_owned_cleanup',new=AsyncMock()):asyncio.run(run())
            self.assertEqual(spawn.call_args.args[4:],('--cpu-pair','30','31'))


if __name__=='__main__':unittest.main()
