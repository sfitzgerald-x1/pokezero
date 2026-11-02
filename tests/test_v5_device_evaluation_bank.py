"""Inert controllers/tensors only: no engine turn, actor or model execution."""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from pokezero import v5_contract as c, v5_run as run, v5_evaluation as evaluation
from pokezero import v5_device_evaluation_bank as schema
from gpu_engine import state, engine_device_trackers as trackers, engine_device_observation as encoder
from gpu_engine import device_root_snapshot as snapshots, v5_device_evaluation_bank as producer
from gpu_engine import engine_device_damage_history as damage, engine_device_negative_items as negative
from gpu_engine.v5_device_rollout import RootContext
from gpu_engine.engine_device_public_hooks import Initialized, PublicStepResult
from gpu_engine.transition import StepResult


def artifact(path,value):
    path.write_text(run._json(value));return dict(path=str(path),sha256=run._file_sha(path))


def observation(b, offset=0):
    return encoder.Observation(j.full((b,29,132),offset,j.float32),j.zeros((b,29,41),j.int32),
        j.broadcast_to(j.asarray((0,)+(1,)*6+(2,)*6+(3,)*9+(5,)+(7,)*6,j.int32),(b,29)),
        j.ones((b,29),bool),j.broadcast_to(j.asarray([True,True]+[False]*7),(b,9)),
        j.broadcast_to(j.arange(1,6,dtype=j.int32),(b,5)),j.zeros(b,j.int32),
        j.arange(9,dtype=j.int32),j.broadcast_to(j.arange(4,dtype=j.int32),(b,4)))


def initial(b):
    s=state.empty_state(b)._replace(done=j.zeros(b,bool),species=j.ones((b,2,6),j.int32),
        hp=j.full((b,2,6),10,j.int32),max_hp=j.full((b,2,6),10,j.int32),level=j.full((b,2,6),50,j.int32))
    t=trackers.initialize(s)._replace(damage_history=damage.initialize(s),negative_items=negative.empty(b),
        mon_features={'unit_current':j.zeros((b,2,6),j.float32)})
    return Initialized(s,t)


def declaration(root):
    batches=[]
    for k in range(16):
        games=tuple(run.Game(1000+16*k+i,'unit-'+str(16*k+i),100+16*k+i,(32*k+2*i,32*k+2*i+1)) for i in range(16))
        batches.append(run.Batch(games,200+k,300+k,400+k,500,tuple(16*k+i for i in range(16))))
    games=tuple(g for batch in batches for g in batch.games)
    labels=c.LabelVocabulary('a'*64,('surf','recover'),(('swampert',2),))
    doc=dict(schema=schema.DECLARATION,role='NONTRAINING_EVALUATION',stop_rule=schema.STOP_RULE,
        source_commit='b'*40,source_sha256='c'*64,pool_sha256='d'*64,category_vocabulary_sha256='e'*64,
        category_count=100,label_vocabulary=asdict(labels),actors=[dict(archive=dict(path='/unit/archive.npz',sha256='f'*64),checkpoint_sha256='1'*64)]*2,
        batches=[asdict(b) for b in batches],max_requests=1024,output_directory=str(root/'output'),
        training_roster_index=dict(path='/unit/original-index.json',sha256='2'*64),cost_recipe_seed=17,
        cost_selectors=schema.cost_selectors(games,17))
    return artifact(root/'declaration.json',doc),doc,games,labels


class DeviceHeldoutBankTests(unittest.TestCase):
    def test_geometry_and_hash_selection_are_pre_outcome_and_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reference,doc,games,_=declaration(root)
            parsed,batches,actual,_=schema.declaration(reference)
            self.assertEqual(actual,games);self.assertEqual(len(batches),16)
            self.assertEqual(len(schema.selectors(games)),1024)
            self.assertEqual(len({s['stream'] for s in parsed['cost_selectors']}),40)
            self.assertEqual([sum(s['frame']==f for s in parsed['cost_selectors']) for f in schema.FRAMES],[10]*4)
            self.assertTrue(all(s['owner']==(s['stream']-1000)%2 for s in parsed['cost_selectors']))
            for change in ('selector','stream','team','cap'):
                bad=copy.deepcopy(doc)
                if change=='selector':bad['cost_selectors'].reverse()
                if change=='stream':bad['batches'][1]['games'][0]['stream_id']=1000
                if change=='team':bad['batches'][1]['games'][0]['team_indices']=[0,33]
                if change=='cap':bad['max_requests']=100
                with self.subTest(change=change),self.assertRaises(ValueError):schema.declaration(artifact(root/'bad.json',bad))

    def test_typed_root_roundtrip_both_source_views_and_missing_history_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);i=initial(3);row=2
            root=RootContext(producer._one(i.state,row),producer._one(i.trackers,row),j.asarray([77],j.int32),j.asarray([12],j.int32))
            views=tuple(producer._one_view(observation(3,n),row) for n in (1,2))
            fields=dict(observations=views,requested=j.ones((1,2),bool),policy_logits=j.zeros((1,2,9),j.float32),
                policy_finite=j.ones((1,2),bool),actions=j.asarray([[0,1]],j.int32))
            namespace={'unit':'source'};selector=dict(stream=77,frame=12,owner=0)
            ref=snapshots.save(directory/'root.npz',root,fields,namespace=namespace,selector=selector)
            entry=dict(selector=selector,status='SAVED',artifact=ref,reason=None)
            index=dict(schema=schema.ROOT_INDEX,namespace=namespace,cost_roots=[entry])
            restored,source=producer.load_root(entry,index)
            for a,b in zip(jax.tree.leaves(root),jax.tree.leaves(restored),strict=True):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(source['observations'][0].error_counts,np.arange(9))
            self.assertIsNone(restored.public.tendency)
            self.assertEqual(source['namespace'],namespace)
            with self.assertRaises(ValueError):producer.load_root(entry,dict(index,namespace={'unit':'other'}))
            # Genuine typed transport never silently fills a newly missing tail.
            with np.load(ref['path'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            body=json.loads(arrays['metadata_json'].tobytes());body['root']['values'][1]['fields'].pop();body['root']['values'][1]['values'].pop()
            arrays['metadata_json']=np.frombuffer(json.dumps(body).encode(),np.uint8)
            bad=directory/'missing.npz';np.savez(bad,**arrays)
            forged=dict(entry,artifact=dict(path=str(bad),sha256=run._file_sha(bad)))
            with self.assertRaisesRegex(ValueError,'schema changed'):snapshots.load(forged,index)

    def test_controller_finite_dispositions_all_legal_siblings_and_no_outcome_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('roots','positions','games'):(root/name).mkdir()
            batch=run.Batch(tuple(run.Game(100+i,'u'+str(i),200+i,(2*i,2*i+1)) for i in range(3)),1,2)
            init=initial(3);seen=[]
            def query(s,p,streams,frames,params):
                return dict(observations=(observation(len(streams)),observation(len(streams),1)),
                    policy_logits=j.zeros((len(streams),2,9),j.float32),policy_finite=j.ones((len(streams),2),bool),
                    requested=j.ones((len(streams),2),bool),actions=j.zeros((len(streams),2),j.int32),
                    physical=j.zeros((len(streams),2),j.int32),input_valid=j.asarray([True,True,False]),chance=j.arange(len(streams),dtype=j.int32)[:,None])
            def advance(s,p,actions,chance):
                seen.append((np.asarray(actions),np.asarray(chance)))
                if len(s.done)==1:
                    # Child action0 terminates for the opposite seat; action1
                    # remains live. The ordinary game below terminates owner0.
                    terminal=actions[:,0]==0
                    after=s._replace(done=terminal,winner=j.where(terminal,1,-1))
                else:
                    after=s._replace(done=s.done|j.asarray([True,False,False]),winner=j.asarray([0,-1,-1],j.int32))
                return PublicStepResult(StepResult(after,j.zeros(len(s.done),bool),j.zeros(len(s.done),bool)),p)
            executor=producer.Executor(query,advance,lambda s,p:(observation(len(s.done)),observation(len(s.done))))
            cost={(100,0,0):dict(selector=dict(stream=100,frame=0,owner=0),status='UNREACHED',artifact=None,reason='notyet')}
            with patch.object(producer.encoder,'map_actions',side_effect=lambda obs,s,side,choice:choice):
                statuses=producer._collect_batch(executor,init,batch,[0,0,0],dict(max_requests=2),(),root,{},cost,0)
            self.assertEqual(statuses,['COMPLETE','CAPPED','REFUSED'])
            record=json.loads((root/'games/000.json').read_text())
            self.assertEqual(len(record['positions']),4);self.assertEqual(record['terminal']['winner'],0)
            children=record['siblings'][0]['branches'];self.assertEqual([b['action'] for b in children],[0,1])
            self.assertEqual(children[0]['terminal']['winner'],1);self.assertIsNone(children[1]['terminal'])
            self.assertEqual(record['positions'][1]['status'],'UNREACHED')
            self.assertEqual(cost[(100,0,0)]['status'],'SAVED')
            self.assertEqual(children[0]['chance_id'],children[1]['chance_id'])
            np.testing.assert_array_equal(seen[0][1],seen[1][1])
            self.assertEqual(len(list((root/'games').glob('*.json'))),3)

    def test_resolver_preserves_full_denominator_unknowns_and_actual_child_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reference,doc,games,labels=declaration(root)
            selections=schema.selectors(games);positions=[];groups=[]
            for selector in selections:
                p,g=producer._unknown(selector,'UNREACHED','fixed-selector-unreached');positions.append(p);groups.append(g)
            literal=producer._observation(observation(1));selector=selections[0]
            positions[0]=dict(selector=selector,status='OBSERVED',root_id='root',world_id='world',observation=literal,
                terminal=dict(status='COMPLETE',done=True,winner=0),reason=None)
            group=dict(selector=selector,status='OBSERVED',root_id='root',world_id='world',chance={'draws':[1,2]},reply=1,branches=[])
            group['chance_id']=schema.digest(group['chance'])
            for action in (0,1):
                group['branches'].append(dict(action=action,root_id='root',world_id='world',chance_id=group['chance_id'],reply=1,
                    status='COMPLETE' if action==0 else 'OBSERVED',terminal=dict(done=True,winner=1) if action==0 else None,
                    observation=None if action==0 else literal,reason=None))
            groups[0]=group
            header=dict(split='PHASE_C_FIXED_HELDOUT',rows=1024,device_bank=reference,device_source=doc['source_commit'],
                category_vocabulary_sha256=doc['category_vocabulary_sha256'],label_vocabulary_sha256=labels.sha256,category_count=100)
            def write(name,kind,rows):
                path=root/name;path.write_text(''.join(run._json(x)+'\n' for x in [dict(header,schema=kind),*rows]));return dict(path=str(path),sha256=run._file_sha(path))
            protocol=dict(positions=write('positions.jsonl',schema.POSITIONS,positions),siblings=write('siblings.jsonl',schema.SIBLINGS,groups),label_vocabulary_sha256=labels.sha256,batch_size=4,bins=4)
            data=schema.resolve_device_bank(protocol,None)
            self.assertEqual(len(data.positions),1024);self.assertEqual(data.positions[0][2],1)
            self.assertEqual(sum(o is not None for _,o,_ in data.positions),1)
            self.assertEqual(data.groups[0]['branches'][0]['terminal']['winner'],1)
            self.assertIs(evaluation._capture_resolver(protocol).resolve_original_capture,schema.resolve_device_bank)
            self.assertEqual(evaluation._capture_resolver().__name__,'pokezero.v5_evaluation_capture')
            # Arithmetic/namespace plumbing through the actual resolver with
            # an inert value function; never run a neural model.
            config=NS(label_vocabulary=labels,category_vocabulary_sha256=doc['category_vocabulary_sha256'],category_vocab_size=100)
            factory=NS(config=config,forward=lambda inputs:NS(valid=np.ones(len(inputs['categorical_ids']),bool),value=np.zeros(len(inputs['categorical_ids']))))
            with patch('gpu_engine.v5_live_protocol.check_factory'),patch.object(producer.v5_model,'prepare_inputs',side_effect=lambda inputs,config:inputs):
                measured=schema.evaluate_device_bank(protocol,factory,data)
            self.assertEqual(measured['known_outcomes'],1);self.assertEqual(measured['unknown_outcomes'],1023)
            self.assertEqual(measured['outcome_rmse'],1.);self.assertEqual(measured['scored_sibling_groups'],1)
            self.assertEqual(measured['top_two_rows'][0]['values'],[-1.,0.])
            self.assertEqual(measured['actual_ranking_accuracy'],'UNMEASURED_NO_MATCHED_TERMINAL_CONTINUATIONS')
            for change in ('omitted','chance','copiedwinner','unknownterminal'):
                bad=copy.deepcopy(groups)
                if change=='omitted':bad[0]['branches'].pop()
                if change=='chance':bad[0]['branches'][1]['chance_id']='different'
                if change=='copiedwinner':bad[0]['branches'][0]['terminal']['winner']=True
                if change=='unknownterminal':bad[0]['branches'][1]['terminal']=dict(done=True,winner=0)
                altered=dict(protocol,siblings=write('bad.jsonl',schema.SIBLINGS,bad))
                with self.subTest(change=change),self.assertRaises(ValueError):schema.resolve_device_bank(altered,None)

    def test_actual_executor_call_contract_uses_inert_original_api_seams(self):
        # Traces this producer's actual query only, replacing every physical,
        # encoder and neural implementation with explicit UNIT functions.
        init=initial(2);prepared=NS(engine_tables=object(),observation_tables=NS(belief=None))
        calls=[]
        def fake_forward(params,inputs,config,*,compute_dtype):
            calls.append(compute_dtype);b=len(inputs['categorical_ids'])
            return NS(policy_logits=j.zeros((b,9),j.float32),value=j.zeros(b),value_logits=None,valid=j.ones(b,bool))
        with patch.object(producer.transition,'action_mask',return_value=j.ones((2,2,9),bool)), \
                patch.object(producer.sampling,'request_seats',return_value=j.ones((2,2),bool)), \
                patch.object(producer.trackers,'readout',return_value=None), \
                patch.object(producer.encoder,'encode',side_effect=lambda *a,**kw:observation(2)), \
                patch.object(producer.v5_model,'forward',side_effect=fake_forward), \
                patch.object(producer.encoder,'map_actions',side_effect=lambda obs,s,side,choice:choice), \
                patch.object(producer.transition,'sample_chance_inputs',return_value=j.ones((2,2),j.int32)):
            executor=producer.make_executor(prepared,object(),sampling_seed=9)
            result=jax.block_until_ready(executor.query(init.state,init.trackers,j.asarray([3,4]),j.asarray([0,12]),({},{})))
        self.assertEqual(calls,['bfloat16','bfloat16']);self.assertEqual(result['actions'].shape,(2,2))
        np.testing.assert_array_equal(result['input_valid'],True)


if __name__=='__main__':unittest.main()
