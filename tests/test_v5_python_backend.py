"""Inert constructor, durable-result and original-runner resume controls.

No simulator process, real model, forward, trajectory or allocation runs here.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch,Mock

import numpy as np
from pokezero import v5_python_backend as b, v5_run as r
from pokezero import v5_python_collection as c
import test_v5_run as run_fixture
import test_v5_python_collection as capture_fixture


def prepared():
    names=capture_fixture.NAMES
    arrays=dict(species=np.tile(np.arange(1,7),(8,1)),moves=np.tile([1,2,0,0],(8,6,1)),
        item=np.ones((8,6),int),ability=np.ones((8,6),int),nature=np.ones((8,6),int),
        level=np.full((8,6),80),ivs=np.tile([31,30,29,28,27,26],(8,6,1)),
        evs=np.tile([85]*6,(8,6,1)),happiness=np.full((8,6),117))
    metadata={'vocabulary':dict(species=['_none',*names],moves=['_none','surf','recover'],
        items=['_none','leftovers'],abilities=['_none','pressure'],natures=['_none','serious'])}
    return b.Prepared(arrays,metadata,'a'*64,None,None,None,None,None,None,None)


def capture(game,side=0):
    batch=next(batch for iteration in run_fixture.plan().iterations for batch in iteration if game in batch.games)
    with patch('gpu_engine.v5_live_protocol.check_factory'):
        return c.collect_game(capture_fixture.Env(),c.Game(game.stream_id,game.battle_id,game.seed,201),
            factories=capture_fixture.factories(),max_requests=2,collect_owners=(side,),
            chance_seed=b.chance_seed(batch,game),enable_v5=True)


def publish(store,game,result,*,iteration=0,index=0,part=None):
    facts=dict(stream_id=game.stream_id,status=result.status,duration_seconds=.1,
        iteration=iteration,batch=index,requests=result.frames,reason=result.reason)
    document=dict(schema='pokezero.python-game-publication.v5',game=game.__dict__,facts=facts,
        cache=part,source=result.source,auxiliary_refusals=result.auxiliary_refusals)
    return store.artifact('python-games/result-'+str(game.stream_id)+'.json',r._json(document).encode(),
        kind='PYTHON_GAME_PUBLISHED',stream_id=game.stream_id,iteration=iteration,batch=index)


class PythonBackendTests(unittest.TestCase):
    def test_packed_teams_preserve_literal_constructor_fields(self):
        p=prepared();game=run_fixture.plan().games[0];gender=np.array([[0,1,2,0,1,2]]*2)
        start=b.packed_start(p,game,gender)
        self.assertEqual(start.observation_format_id,'gen3randombattle')
        for seat in('p1','p2'):
            rows=[x.split('|')for x in start.player_teams[seat].split(']')]
            self.assertEqual([x[0]for x in rows],list(capture_fixture.NAMES))
            for i,row in enumerate(rows):
                self.assertEqual(row[2:7],['leftovers','pressure','surf,recover','serious','85,85,85,85,85,85'])
                self.assertEqual(row[7],('N','M','F')[i%3])
                self.assertEqual(row[8],',30,29,28,27,26')
                self.assertEqual(row[10:],['80','117'])
        for invalid in(np.full((2,6),-1),np.full((2,6),3),np.zeros((1,6))):
            with self.assertRaises(ValueError):b.packed_start(p,game,invalid)

    def test_happiness_absence_uses_original_constructor_default_not_text_frustration_zero(self):
        p=prepared();p.arrays.pop('happiness')
        self.assertTrue(all(mon.split('|')[11]=='255'for team in b.packed_start(
            p,run_fixture.plan().games[0],np.zeros((2,6),int)).player_teams.values()for mon in team.split(']')))

    def test_gender_addresses_survive_pending_suffix_filter(self):
        p=prepared();policy=dict(schema='pokezero.gen3-gender-policy.v1',species=p.metadata['vocabulary']['species'],
            codes=[-1]+[3]*6,provenance={'pool_sha256':p.pool_sha256})
        p=replace(p,gender_policy=policy)
        batch=replace(run_fixture.plan().iterations[0][0],gender_seed=42,gender_domain=9,gender_root_ids=(101,102))
        all_values=b._genders(p,batch)
        suffix=replace(batch,games=batch.games[1:],gender_root_ids=batch.gender_root_ids[1:])
        np.testing.assert_array_equal(b._genders(p,suffix)[0],all_values[1])
        with self.assertRaises(ValueError):b._genders(replace(p,gender_policy=None),batch)
        self.assertEqual(b.chance_seed(batch,batch.games[1]),b.chance_seed(suffix,suffix.games[0]))
        self.assertNotEqual(b.chance_seed(batch,batch.games[0]),b.chance_seed(batch,batch.games[1]))
        self.assertNotEqual(b.chance_seed(batch,batch.games[0]),b.chance_seed(replace(batch,constructor_seed=43),batch.games[0]))

    def test_chance_seed_is_distinct_from_immutable_cell_identity(self):
        env=capture_fixture.Env()
        with patch.object(env,'reset',wraps=env.reset)as reset,patch('gpu_engine.v5_live_protocol.check_factory'):
            result=c.collect_game(env,c.Game(1,'fixed',42,7),factories=capture_fixture.factories(),
                max_requests=2,chance_seed=19,enable_v5=True)
        self.assertEqual(result.status,'COMPLETE');reset.assert_called_once_with(seed=19,format_id='gen3randombattle')
        self.assertEqual(result.source['chance_seed'],19);self.assertTrue(all(x.row.seed==42 for x in result.decisions))

    def test_original_cache_merge_matches_direct_combined_trajectory_writer(self):
        from gpu_engine import v5_collection_data as data
        from gpu_engine.v5_device_spool import merge_game_parts
        from pokezero import v5_data
        recipe=run_fixture.plan();games=recipe.iterations[0][0].games
        captures=[capture(g)for g in games]
        with tempfile.TemporaryDirectory()as folder:
            folder=Path(folder);parts=[]
            for i,item in enumerate(captures):
                path=folder/('part'+str(i));b._write_game_cache(path,item.decisions,recipe)
                parts.append((i,path,r._file_sha(path/'metadata.json')))
            merge_game_parts(folder/'merged',parts,tuple(g.stream_id for g in games))
            ordered=sorted((x for item in captures for x in item.decisions),
                key=lambda x:(x.row.observation.address.frame,x.row.observation.address.stream,x.row.observation.address.owner))
            data.write_cache(folder/'direct',ordered,loss_config=recipe.loss,batch_size=2)
            left=json.loads((folder/'merged/metadata.json').read_text());right=json.loads((folder/'direct/metadata.json').read_text())
            self.assertEqual(left,right)
            for name in left['arrays']:
                np.testing.assert_array_equal(np.load(folder/'merged'/(name+'.npy'),allow_pickle=False),
                    np.load(folder/'direct'/(name+'.npy'),allow_pickle=False))
            batches=tuple(v5_data.iter_cache_batches(folder/'merged',batch_size=2))
            self.assertEqual(sum(x.batch_size for x in batches),4)

    def test_published_result_recovers_without_replay_and_missing_merge_recovers_once(self):
        from gpu_engine import v5_collection_data as data
        recipe=run_fixture.plan();game=recipe.games[0];result=capture(game)
        backend=b.RuntimeBackend(prepared())
        with tempfile.TemporaryDirectory()as folder:
            store=r.Store(Path(folder)/'run',recipe);store.emit('BATCH_STARTED',iteration=0,batch=0,
                streams=[1,2],assignments=[],deferred_starts=True)
            store.emit('GAME_STARTED',stream_id=1,iteration=0,batch=0)
            path=store.path/'part';data.write_cache(path,result.decisions,loss_config=recipe.loss,batch_size=2)
            part={'path':'part','sha256':r._file_sha(path/'metadata.json')};publish(store,game,result,part=part)
            backend.recover_game_ends(store);backend.recover_game_ends(store)
            self.assertEqual(len(store.matching('GAME_END')),1)
            statuses={1:'COMPLETE',2:'NOT_ATTEMPTED',3:'NOT_ATTEMPTED',4:'NOT_ATTEMPTED'}
            backend.recover_caches(store,statuses);self.assertEqual(store.matching('CACHE'),[])
            statuses[2]='FAILED';backend.recover_caches(store,statuses);backend.recover_caches(store,statuses)
            self.assertEqual(len(store.matching('CACHE')),1)
            cache=store.matching('CACHE')[0];self.assertTrue(store.resolve(cache['path'],cache['sha256']).is_dir())

    def test_actual_dispatch_failure_does_not_start_queued_suffix(self):
        recipe=run_fixture.plan();backend=b.RuntimeBackend(prepared(),workers=1)
        value=dict(plan=recipe,batch=recipe.iterations[0][0]);started=[]
        with patch.object(backend,'check_plan'),patch.object(backend,'_one',side_effect=PermissionError('inert denied')):
            with self.assertRaises(PermissionError):backend.collect(value,None,lambda *a,**k:None,lambda:None,on_start=started.append)
        self.assertEqual(started,[1])

    def test_python_route_is_explicit_exclusive_and_never_enters_engine_preparation(self):
        from pokezero import v5_worker as worker
        recipe=replace(run_fixture.plan(),evaluation=r.EvaluationProtocol('c'*64))
        inputs=NS(plan=recipe,runtime={'memory_policy':{}},vocabulary={},gender_policy=None)
        for options in(dict(python_rollout=True,device_rollout=True),dict(python_workers=2),
                dict(python_rollout=True,python_workers=0),dict(python_rollout=True,cache_workers=1),
                dict(python_rollout=True,raw_retention_every=50),dict(python_rollout=True,device_teacher=True)):
            with patch.object(worker,'_source')as source:
                with self.assertRaises(ValueError):worker.execute(inputs,'/inert-not-created',evaluation_hook=lambda:None,**options)
                source.assert_not_called()
        runtime=Mock();prepared_token=object()
        with patch.object(worker,'_source'),patch.object(worker,'_memory'),patch.object(worker,'_probe'),\
                patch.object(worker,'prepare_runtime',side_effect=AssertionError('legacy engine called')),\
                patch.object(worker,'prepare_device_runtime',side_effect=AssertionError('device engine called')),\
                patch.object(b,'prepare',return_value=prepared_token),patch.object(b,'RuntimeBackend',return_value=runtime)as create,\
                patch.object(r,'run',return_value={'status':'INERT_NOT_EXECUTED'})as run:
            value=worker.execute(inputs,'/inert-not-created',evaluation_hook=lambda:None,python_rollout=True,
                python_workers=3,iteration_policy_lag=1)
        self.assertEqual(value['status'],'INERT_NOT_EXECUTED');create.assert_called_once_with(prepared_token,workers=3)
        self.assertIs(run.call_args.kwargs['backend'],runtime);self.assertEqual(run.call_args.kwargs['iteration_policy_lag'],1)

    def test_actor_factories_reuse_immutable_snapshot_and_reject_another_checkpoint(self):
        recipe=run_fixture.plan();backend=b.RuntimeBackend(prepared());snapshot=dict(archive='inert',archive_sha256='c'*64,
            checkpoint_sha256='d'*64,model_sha256=r._sha(recipe.model.metadata()))
        factory=NS(config=recipe.model,metadata=json.dumps({'checkpoint_sha256':'d'*64}))
        with patch.object(b,'_factory',return_value=factory)as load:
            self.assertIs(backend._actor(snapshot,NS(plan=recipe)),factory)
            self.assertIs(backend._actor(snapshot,NS(plan=recipe)),factory)
            load.assert_called_once()
            with self.assertRaises(ValueError):backend._actor(dict(snapshot,checkpoint_sha256='e'*64),NS(plan=recipe))

    def test_original_runner_resumes_only_actual_not_attempted_suffix(self):
        class Deferred(run_fixture.FakeBackend):
            deferred_game_starts=True
            def recover_game_ends(self,store):pass
            def recover_caches(self,store,statuses):pass
            def collect(self,prepared,store,on_end,check_progress,*,on_start):
                for game in prepared.batch.games:
                    on_start(game.stream_id)
                    self.collect_calls.append(game.stream_id);on_end(game.stream_id,'COMPLETE',duration_seconds=.1)
                    if self.fail_after_start:raise ValueError('inert interrupted before next dispatch')
                path='cache-'+str(prepared.batch.games[0].stream_id);directory=store.path/path;directory.mkdir()
                (directory/'metadata.json').write_text('{}');return path,r._file_sha(directory/'metadata.json')
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'run';recipe=run_fixture.plan();helper=run_fixture.V5RunTests();first=Deferred(fail_after_start=True)
            stopped=helper.execute(path,recipe,first,run_fixture.FakeTraining())
            self.assertEqual(stopped['statuses'],{1:'COMPLETE',2:'NOT_ATTEMPTED',3:'NOT_ATTEMPTED',4:'NOT_ATTEMPTED'})
            next_backend=Deferred();done=helper.execute(path,recipe,next_backend,run_fixture.FakeTraining(),resume=True)
            self.assertEqual(done['status'],'COMPLETE_DECLARATION');self.assertEqual(next_backend.collect_calls,[2,3,4])
            events=r.Store(path,recipe,resume=True)
            self.assertEqual([x['stream_id']for x in events.matching('GAME_STARTED')],[1,2,3,4])


if __name__=='__main__':unittest.main()
