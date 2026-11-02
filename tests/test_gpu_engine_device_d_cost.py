"""Declared cost algebra/fake callbacks only; no model, world or game calls."""
from dataclasses import replace
import json
import sys
from types import SimpleNamespace as NS
import unittest
from typing import NamedTuple
from unittest.mock import patch
import jax.numpy as j
import numpy as np
from pokezero import v5_run as runs,v5_contract as c,v5_gumbel as g
from gpu_engine import device_d_cost as cost,v5_device_teacher as teacher,v5_device_rollout as rollout
from tests import test_v5_run as fixtures


class InertState(NamedTuple):
    turn:object


def plan():
    old=fixtures.plan('D');games=tuple(runs.Game(9000+i,'declared-'+str(i),20000+i,(2*i,2*i+1))for i in range(256))
    return replace(old,iterations=((runs.Batch(games,100,200),),))


def inputs():
    p=plan();provider=teacher.Provider.__new__(teacher.Provider);provider.plan=p
    provider.prepared=NS(source_sha256='d'*64,source_binding_json=json.dumps(dict(source_commit='1'*40,pool_metadata_sha256='e'*64)),
        source=NS(metadata=NS(source_hash='literal-original-source')),engine_tables=object(),observation_tables=object())
    snapshot=runs.Snapshot(-1,'original.pt','2'*64,'current-source-d.npz','3'*64,runs._sha(p.model.metadata()))
    namespace=dict(source_commit='1'*40,source_sha256='d'*64,public_source_hash='literal-original-source',pool_sha256=p.pool_sha256,
        pool_metadata_sha256='e'*64,category_vocabulary_sha256=p.model.category_vocabulary_sha256,label_vocabulary_sha256=p.model.label_vocabulary.sha256)
    entries=[dict(selector=dict(stream=a.stream,frame=a.frame,owner=a.owner),status='SAVED',artifact={'path':'inert','sha256':'4'*64},reason=None)for a in cost.select(p)]
    return provider,snapshot,dict(schema='pokezero.device_heldout_saved_roots.v1',namespace=namespace,cost_roots=entries)


class DeviceDCostTests(unittest.TestCase):
    def test_fixed40_selection_is_outcome_free_unique_streams_and_original_hash(self):
        p=plan();selected=cost.select(p)
        self.assertEqual(len(selected),40);self.assertEqual(len(set(a.stream for a in selected)),40)
        self.assertEqual([a.frame for a in selected],[frame for frame in cost.FRAMES for _ in range(10)])
        recipe=g.Recipe(p.selection_seed)
        self.assertTrue(all(recipe.selected(a)for a in selected))
        original={game.stream_id:i for i,game in enumerate(p.games)}
        self.assertTrue(all(a.owner==original[a.stream]%2 for a in selected))
        self.assertEqual(selected,cost.select(p))
        with self.assertRaises(ValueError):cost.select(replace(p,iterations=((replace(p.iterations[0][0],games=p.games[:20]),),)))

    def test_cost_plan_preserves_actual_saved_bank_and_strict_archive_loader(self):
        from tests import test_v5_phasec as phase
        p=plan();actual=cost.plan_from_bank(profile=phase.profile(),labels=phase.vocabulary(),category=phase.category(),
            batches=p.iterations[0],pool_sha256=p.pool_sha256,selection_seed=p.selection_seed,run_id='saved-bank-cost',max_requests=p.max_requests)
        self.assertIs(actual.iterations[0][0],p.iterations[0][0]);self.assertEqual(actual.pool_sha256,p.pool_sha256)
        self.assertEqual(actual.model.arm,'D');self.assertIsNone(actual.evaluation)
        snapshot=runs.Snapshot(-1,'genuine.pt','2'*64,'current-d.npz','3'*64,runs._sha(actual.model.metadata()))
        params=object()
        with patch.object(cost.v5_model,'load',return_value=(params,actual.model,{'current':'metadata'}))as load:
            result=cost.actor('literal-current-archive',snapshot=snapshot,plan=actual)
        self.assertIs(result[0],params)
        self.assertEqual(load.call_args.kwargs,dict(expected_archive_sha256=snapshot.archive_sha256,
            expected_checkpoint_sha256=snapshot.checkpoint_sha256,expected_category_vocabulary_sha256=actual.model.category_vocabulary_sha256,
            expected_label_vocabulary_sha256=actual.model.label_vocabulary.sha256,enable_v5=True))
        with patch.object(cost.v5_model,'load')as load:
            with self.assertRaises(ValueError):cost.actor('unused',snapshot=replace(snapshot,model_sha256='f'*64),plan=actual)
            load.assert_not_called()

    def test_cold_first_p95_unknown_denominators_no_refill(self):
        values=[100.,*([2.]*19)]
        records=[dict(status='LABELED'if i%2 else'PROVIDER_REFUSED',label_seconds=x)for i,x in enumerate(values)]
        records.extend(dict(status='UNREACHED',label_seconds=None)for _ in range(20))
        report=cost.summarize(records)
        self.assertEqual(report['timed_provider_attempts'],20);self.assertEqual(report['usable_labeled'],10)
        self.assertEqual(report['cold_first_label_seconds'],100.);self.assertEqual(report['p95_label_seconds'],2.)
        self.assertEqual(report['full_roster_p95_upper_seconds'],'unbounded');self.assertEqual(report['cost_admission'],'PASS')
        self.assertEqual(report['warm_descriptive_p95_label_seconds'],2.)
        bad=[dict(status='LABELED',label_seconds=31.)for _ in range(40)]
        self.assertEqual(cost.summarize(bad)['cost_admission'],'FAIL')
        with self.assertRaises(ValueError):cost.summarize(records[:-1])
        with self.assertRaises(ValueError):cost.summarize([dict(status='UNKNOWN-MALFORMED',label_seconds=None)]*40)
        with self.assertRaises(ValueError):cost.summarize([dict(status='LABELED',label_seconds=float('nan'))]*40)

    def test_actual_provider_frames_both_seats_archive_and_saved_slot_accounting(self):
        provider,snapshot,index=inputs();calls=[];seen=[];param=object()
        index['cost_roots'][-1].update(status='UNREACHED',artifact=None,reason='original game ended before frame24')
        def load(entry,index):
            a=entry['selector'];context=rollout.RootContext(InertState(j.asarray([a['frame']],j.int32)),j.zeros(1),j.asarray([a['stream']],j.int32),j.asarray([a['frame']],j.int32))
            return context,{'actual_bank_logits_are_NOT_D_logits':'UNUSED'}
        def observe(state,public):
            return tuple(NS(numeric_features=j.full((1,29,132),owner+.25),categorical_ids=j.full((1,29,41),owner,j.int32),
                token_type_ids=j.zeros((1,29),j.int32),attention_mask=j.ones((1,29),bool),legal_action_mask=j.ones((1,9),bool),error=j.zeros(1,j.int32))for owner in (0,1))
        def forward(p,columns,config,*,compute_dtype):
            self.assertIs(p,param);self.assertEqual(compute_dtype,'bfloat16');calls.append(float(columns['numeric_features'][0,0,0,0]))
            return NS(policy_logits=j.full((1,9),len(calls),j.float32),value=j.asarray([.2],j.float32),valid=j.asarray([True]),value_logits=j.zeros((1,101)))
        def label(roots,frames,bank,params):
            self.assertEqual(params,(param,));self.assertEqual(frames.requested.shape,(1,1,2))
            self.assertEqual(frames.raw_policy_logits.shape,(1,1,2,9));self.assertIsNone(frames.actions)
            self.assertTrue(frames.policy_valid.all());self.assertEqual(frames.numeric_features[0,0,1,0,0],1.25)
            owner=bank.learner_mask[0].index(True);address=dict(stream=bank.streams[0],frame=int(roots.frames[0,0]),owner=owner)
            seen.append(address);return [dict(address=address,status='REFUSED',refusal='actual explicit unknown world',improvement=None)]
        module=NS(load_root=load)
        with patch.dict(sys.modules,{'gpu_engine.v5_device_evaluation_bank':module}),\
                patch.object(cost,'actor',return_value=(param,dict(parameter_count=99)))as actor,patch.object(cost,'_observer',return_value=observe),\
                patch.object(cost.v5_model,'forward',side_effect=forward),patch.object(cost.jax,'jit',side_effect=lambda fn:fn),\
                patch.object(cost.jax,'block_until_ready',side_effect=lambda x:x),patch.object(cost,'request_seats',return_value=j.ones((1,2),bool)),\
                patch.object(provider,'label_chunk',side_effect=label):
            result=cost.run(provider,snapshot,'strict-current-source-archive',index,enable_cost=True)
        actor.assert_called_once();self.assertEqual(len(calls),78);self.assertEqual(len(seen),39)
        self.assertEqual(result['summary']['timed_provider_attempts'],39);self.assertEqual(result['summary']['usable_labeled'],0)
        self.assertEqual(result['summary']['dispositions'],{'PROVIDER_REFUSED':39,'UNREACHED':1})
        self.assertTrue(result['records'][0]['cold_first_label_call']);self.assertFalse(result['records'][1]['cold_first_label_call'])
        self.assertEqual(seen,[entry['selector']for entry in index['cost_roots'][:-1]])

    def test_foreign_namespace_or_changed_selector_refuses_before_model(self):
        provider,snapshot,index=inputs();index['namespace']['pool_sha256']='f'*64
        with patch.dict(sys.modules,{'gpu_engine.v5_device_evaluation_bank':NS(load_root=lambda *a:None)}),patch.object(cost,'actor')as actor:
            with self.assertRaisesRegex(ValueError,'namespaces differ'):cost.run(provider,snapshot,'unused',index,enable_cost=True)
            actor.assert_not_called()
        provider,snapshot,index=inputs();index['cost_roots'].reverse()
        with patch.dict(sys.modules,{'gpu_engine.v5_device_evaluation_bank':NS(load_root=lambda *a:None)}),patch.object(cost,'actor')as actor:
            with self.assertRaisesRegex(ValueError,'predeclared'):cost.run(provider,snapshot,'unused',index,enable_cost=True)
            actor.assert_not_called()

if __name__=='__main__':unittest.main()
