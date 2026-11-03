"""Source/inert actors and teacher seams only; no worlds/engine/model execution."""
from dataclasses import asdict,replace
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import jax
import numpy as np
from gpu_engine import v5_device_teacher as t,v5_device_rollout as d,v5_device_backend as backend
from gpu_engine.v5_device_spool import Spool
from pokezero import v5_contract as c,v5_gumbel as g,v5_data,v5_phasec as phase
from pokezero.v5_loss import V5LossConfig
from tests import test_gpu_engine_v5_device_actor_bank as bank_fixture
from tests import test_gpu_engine_v5_device_backend as backend_fixture
from tests import test_v5_contract as contract_fixture


def inert_d():
    """Actual collector scan with existing inert encode/step/actor fixtures."""
    fixture=bank_fixture.DeviceActorBankTests();bank=fixture.bank();old=d.make_collector
    def factory(*a,**kw):
        a=list(a);a[2]=replace(a[2],arm='D');return old(*a,**kw,retain_teacher_roots=True)
    with patch.object(d,'make_collector',side_effect=factory):raw,_=fixture.run_inert(bank)
    return raw,bank


def records(raw,bank):
    f=raw.frames;requests=t._requests(f,bank)
    recipe=next(g.Recipe(seed)for seed in range(10000)if all(g.Recipe(seed).selected(c.DecisionAddress(int(f.streams[s,r]),int(f.frames[s,r]),int(o)))for s,r,o in requests))
    result=[]
    for index,(s,r,o)in enumerate(requests):
        s,r,o=map(int,(s,r,o));legal=tuple(map(bool,f.legal_action_mask[s,r,o]));n=sum(legal)
        pi=tuple(1/n if use else 0. for use in legal);q=tuple(.4 if use else None for use in legal)
        improvement=c.RootImprovement(pi,.4,q,2)
        result.append(dict(t.binding(f,bank,s,r,o),recipe_seed=recipe.seed,status='LABELED'if index%2==0 else 'REFUSED',
            refusal=None if index%2==0 else 'unknown-required-world-or-reply',improvement=asdict(improvement)if index%2==0 else None,diagnostics={'declared_worlds':2}))
    return recipe,result


class DeviceTeacherTests(unittest.TestCase):
    def test_original_search_target_factor_exact_known_and_unknown_outcomes(self):
        observation=contract_fixture.observation();improvement=contract_fixture.improvement()
        for outcome in (None,-1,0,1):
            labels=c.V5Labels(observation.address,terminal_status='CAPPED'if outcome is None else 'COMPLETE',outcome=outcome,
                search_status='LABELED',root_improvement=improvement)
            columns=labels.columns(observation);direct=c.search_target_columns(improvement,outcome,observation.legal_action_mask)
            self.assertEqual(direct,{k:columns[k]for k in direct})
            self.assertEqual(direct['search_value_target'],0. if outcome is None else (outcome+improvement.root_value)/2)
        with self.assertRaises(ValueError):c.search_target_columns(improvement,.5,observation.legal_action_mask)

    def test_actual_scan_pre_roots_raw_logits_and_actor_assignments_no_action_change(self):
        raw,bank=inert_d();old,_=bank_fixture.DeviceActorBankTests().run_inert(bank)
        self.assertEqual(raw.frames.raw_policy_logits.shape,(3,2,2,9));self.assertEqual(raw.roots.frames.tolist(),[[0,0],[1,1],[2,2]])
        self.assertEqual(raw.roots.state.turn.tolist(),[[0,0],[1,1],[2,2]])
        for name in d.Frames._fields:
            if name!='raw_policy_logits':np.testing.assert_array_equal(getattr(raw.frames,name),getattr(old.frames,name))
        self.assertTrue(all(np.array_equal(x,y)for x,y in zip(jax.tree.leaves(raw.carry),jax.tree.leaves(old.carry))))
        self.assertIsNone(old.roots);self.assertIsNone(old.frames.raw_policy_logits)

    def test_all_selected_slots_current_bindings_and_no_survivor_repair(self):
        raw,bank=inert_d();recipe,items=records(raw,bank);f=raw.frames
        self.assertEqual(len(t.validate_records(f,bank,recipe,items)),4)
        mutations=[]
        mutations.append(items[:-1]);mutations.append(items+[items[0]])
        for key,value in [('input_sha256','f'*64),('actor_archive_sha256','f'*64),('recipe_seed',recipe.seed+1),('status','UNSELECTED')]:
            bad=copy.deepcopy(items);bad[0][key]=value;mutations.append(bad)
        bad=copy.deepcopy(items);bad[0]['improvement']['worlds']=1;mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(ValueError):t.validate_records(f,bank,recipe,bad)
        changed=f.raw_policy_logits.copy();changed[0,0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'binding differs'):t.validate_records(f._replace(raw_policy_logits=changed),bank,recipe,items)
        # A selected invalid learner input remains a refusal in all raw slots.
        trainable=f.trainable.copy();trainable[0,0,0]=False
        policy_valid=f.policy_valid.copy();policy_valid[0,0,0]=False
        with self.assertRaises(ValueError):t.validate_records(f._replace(trainable=trainable,policy_valid=policy_valid),bank,recipe,items)
        items[0].update(t.binding(f._replace(trainable=trainable,policy_valid=policy_valid),bank,0,0,0),
            status='REFUSED',refusal='invalid-actual-root-input',improvement=None)
        t.validate_records(f._replace(trainable=trainable,policy_valid=policy_valid),bank,recipe,items)

    def test_actual_provider_address_and_archive_callbacks_no_full_root_host_copy(self):
        raw,bank=inert_d();recipe,items=records(raw,bank);params=tuple({'inert':i}for i in range(4))
        provider=t.Provider.__new__(t.Provider);provider.prepared=NS(source_sha256=bank.bindings.source_sha256)
        provider.plan=NS(model=replace(bank_fixture.fixture()[3],arm='D'));provider.recipe=recipe;provider.declaration_sha256=bank.bindings.declaration_sha256;provider.source=object();provider.materializer=object();provider.executor=object();provider.forwards={}
        seen=[]
        def detach(context,source,*,row,owner,enable_device_worlds):
            self.assertIs(source,provider.source);self.assertTrue(enable_device_worlds)
            return NS(address=c.DecisionAddress(int(context.streams[row]),int(context.frames[row]),owner))
        def price(plans,materializer,executor,recipe,**kw):
            key=plans.root.address
            match=next(x for x in items if x['address']==asdict(key))
            seen.append(asdict(key))
            self.assertIs(kw['opponent_forward'],provider.forwards[match['opponent_archive_sha256']][1])
            self.assertIs(kw['learner_forward'],provider.forwards[match['actor_archive_sha256']][1])
            self.assertEqual(kw['root_inputs']['numeric_features'].shape,(1,1,29,132));self.assertEqual(len(kw['root_logits']),9)
            return NS(status=match['status'],reason=match['refusal'],reduction=NS(improvement=None if match['improvement']is None else c.RootImprovement(**match['improvement'])),
                world_reports=(),missing_worlds=0,actual_teacher_calls={'reply_policy':0},reply_input_approximation=())
        with patch.object(t.worlds,'detach_root',side_effect=detach),patch.object(t.worlds,'plan_starts',side_effect=lambda root,*a:NS(root=root)),patch.object(t.teacher,'price',side_effect=price):
            output=provider.label_chunk(raw.roots,raw.frames,bank,params)
        self.assertEqual(len(seen),4);self.assertEqual([x['status']for x in output],[x['status']for x in items])
        self.assertEqual(len(provider.forwards),4)
        foreign=replace(bank,bindings=replace(bank.bindings,declaration_sha256='f'*64))
        with self.assertRaisesRegex(ValueError,'verified prepared ActorBank'):provider.bind(foreign,params)
        with patch.object(t.teacher,'price',side_effect=AssertionError('must not price an invalid root')):
            trainable=raw.frames.trainable.copy();trainable[raw.frames.requested&np.asarray(bank.learner_mask)]=False
            valid=raw.frames.policy_valid.copy();valid[raw.frames.requested&np.asarray(bank.learner_mask)]=False
            output=provider.label_chunk(raw.roots,raw.frames._replace(trainable=trainable,policy_valid=valid),bank,params)
        self.assertTrue(all(x['status']=='REFUSED'for x in output));self.assertTrue(all(x['diagnostics']is None for x in output))

    def test_frame_batch_groups_roots_without_changing_selection_bindings_or_records(self):
        raw,bank=inert_d();recipe,items=records(raw,bank);params=tuple({'inert':i}for i in range(4))
        provider=t.Provider.__new__(t.Provider);provider.prepared=NS(source_sha256=bank.bindings.source_sha256)
        provider.plan=NS(model=replace(bank_fixture.fixture()[3],arm='D'));provider.recipe=recipe;provider.declaration_sha256=bank.bindings.declaration_sha256
        provider.source=object();provider.materializer=object();provider.forwards={};provider.frame_batch=True
        calls=[]
        def price(requests,materializer,r,p):
            calls.append(tuple(x.plans.root.address for x in requests));self.assertIs(p,params)
            result=[]
            for request in requests:
                item=next(x for x in items if x['address']==asdict(request.plans.root.address))
                self.assertEqual(request.root_inputs['numeric_features'].shape,(1,1,29,132))
                result.append(NS(status=item['status'],reason=item['refusal'],reduction=NS(improvement=None if item['improvement']is None else c.RootImprovement(**item['improvement'])),world_reports=(),missing_worlds=0,actual_teacher_calls={},reply_input_approximation=()))
            return result
        provider.frame_pricer=NS(capacity=2048,price=price,last_timing=dict(roots=2,world_prepare_seconds=1.,batch_pack_and_device_seconds=2.,device_invocations=1,lower_seconds=0.,compile_seconds=0.,device_seconds=1.))
        provider.reset_metrics()
        with patch.object(t.worlds,'detach_root',side_effect=lambda ctx,src,*,row,owner,**kw:NS(address=c.DecisionAddress(int(ctx.streams[row]),int(ctx.frames[row]),owner))),patch.object(t.worlds,'plan_starts',side_effect=lambda root,*a:NS(root=root)):
            output=provider.label_chunk(raw.roots,raw.frames,bank,params)
        self.assertEqual([len(x)for x in calls],[2,2]);self.assertEqual([r['status']for r in output],[r['status']for r in items])
        metrics=provider.metrics();self.assertEqual(metrics['by_frame']['0']['refused_share_selected'],.5)
        self.assertEqual(metrics['batch_timing_totals']['device_invocations'],2)
        provider.reset_metrics();self.assertEqual(provider.metrics()['by_frame'],{})

    def test_chunk_batch_preserves_exact_records_and_per_frame_refusals(self):
        raw,bank=inert_d();recipe,items=records(raw,bank);params=tuple({'inert':i}for i in range(4))
        outputs=[]
        for chunk,capacity in ((False,2048),(True,2048),(True,3),(True,1)):
            provider=t.Provider.__new__(t.Provider);provider.recipe=recipe;provider.source=object();provider.materializer=object()
            provider.frame_batch=True;provider.chunk_batch=chunk;provider.bind=lambda *a:None;provider.reset_metrics();calls=[]
            def price(requests,*args):
                calls.append([asdict(x.plans.root.address)for x in requests]);result=[]
                for request in requests:
                    item=next(x for x in items if x['address']==asdict(request.plans.root.address))
                    result.append(NS(status=item['status'],reason=item['refusal'],reduction=NS(improvement=None if item['improvement']is None else c.RootImprovement(**item['improvement'])),world_reports=(),missing_worlds=0,actual_teacher_calls={},reply_input_approximation=()))
                return result
            provider.frame_pricer=NS(capacity=capacity,price=price,last_timing={})
            with patch.object(t.worlds,'detach_root',side_effect=lambda ctx,src,*,row,owner,**kw:NS(address=c.DecisionAddress(int(ctx.streams[row]),int(ctx.frames[row]),owner))),patch.object(t.worlds,'plan_starts',side_effect=lambda root,*a:NS(root=root)):
                output=provider.label_chunk(raw.roots,raw.frames,bank,params)
            outputs.append((output,provider.metrics()['by_frame']))
            self.assertEqual([x for call in calls for x in call],[x['address']for x in items])
            self.assertTrue(all(len(call)<=capacity for call in calls))
            self.assertEqual(len(calls),2 if not chunk else (4+capacity-1)//capacity)
        for output in outputs[1:]:self.assertEqual(output,outputs[0])

    def test_queued_backend_keeps_original_pre_root_cache_and_action_boundary(self):
        old=backend.RuntimeBackend.__init__
        def queued(worker,*args,**kw):
            kw['teacher'].chunk_batch=True;kw['teacher_queue_chunks']=2
            return old(worker,*args,**kw)
        with patch.object(backend.RuntimeBackend,'__init__',new=queued):
            self.test_backend_collect_uses_actual_pre_context_and_original_cache_path()

    def test_spool_original_cache_complete_capped_targets_once_and_tamper_refusal(self):
        raw,bank=inert_d();recipe,items=records(raw,bank)
        items[-1].update(status='LABELED',refusal=None,improvement=copy.deepcopy(items[0]['improvement']))
        raw=raw._replace(carry=raw.carry._replace(status=np.array([d.COMPLETE,d.CAPPED],np.int32)))
        with tempfile.TemporaryDirectory()as folder:
            plan,_,store,_,_,_=backend_fixture.DeviceBackendTests().setup_backend(folder)
            spool=Spool(store,bank,bank_sha256='a'*64,teacher_recipe=recipe)
            for start,stop in ((0,1),(1,3)):
                piece=raw._replace(frames=d.Frames(*(None if x is None else x[start:stop]for x in raw.frames)),roots=None)
                labels=[x for x in items if start<=x['address']['frame']<stop]
                spool.append(piece,teacher_records=labels)
            loss=V5LossConfig(arm='D');games=plan.iterations[0][0].games
            with patch.object(d,'_complete_gae',side_effect=lambda v,o,l,discount:(v*0+o,v*0+o))as gae,patch.object(c,'search_target_columns',wraps=c.search_target_columns)as target:
                for row in (1,0):spool.finalize_game(row,games[row],bindings=bank.bindings,loss_config=loss,discount=1.)
                self.assertEqual(gae.call_count,1);self.assertEqual(target.call_count,3)
            spool.merge(Path(folder)/'merged')
            actual=next(v5_data.iter_cache_batches(Path(folder)/'merged',batch_size=100))
            self.assertEqual(tuple(actual.v5_columns['search_status']),('LABELED','REFUSED','LABELED','LABELED'))
            np.testing.assert_array_equal(actual.search_policy_target_mask,[True,False,True,True])
            np.testing.assert_array_equal(actual.search_value_target_mask,[True,False,True,False])
            np.testing.assert_allclose(actual.search_value_targets[[0,2]],[.7,.7],rtol=0,atol=1e-7)
            self.assertEqual(float(actual.search_value_targets[3]),0.)
            self.assertEqual(len(store.matching('DEVICE_TEACHER_CHUNK')),2)
            # Coherently changing a saved record is still an immutable-file refusal.
            path=store.path/spool.records[0]['teacher']['path'];path.chmod(0o644);path.write_text('[]\n')
            with self.assertRaisesRegex(ValueError,'teacher records changed'):
                spool.finalize_game(0,games[0],bindings=bank.bindings,loss_config=loss,discount=1.)

    def test_configured_provider_compiles_actual_source_once_without_constructor(self):
        from tests.test_gpu_engine_device_public_worlds import source_fixture
        source,metadata=source_fixture();metadata['moves']=[None,*[{'pp':10}for _ in range(4)]]
        plan,prepared,_=backend_fixture.inputs()
        vocabulary=c.LabelVocabulary(c.source_variant_census(source)['source_rows_sha256'],('tackle','surf','rest','protect'),
            tuple((name.lower(),2)for name in metadata['vocabulary']['species'][1:]))
        plan=replace(plan,model=replace(plan.model,arm='D',label_vocabulary=vocabulary),loss=V5LossConfig(arm='D'))
        prepared=replace(prepared,source=source,metadata_json=json.dumps(metadata))
        with patch.object(t.teacher,'make_executor',wraps=t.teacher.make_executor)as factory:
            provider=t.Provider(prepared,plan)
            self.assertEqual(provider.source.vocabulary['types'],('','Normal'))
            self.assertEqual(provider.recipe,g.Recipe(plan.selection_seed));factory.assert_called_once()
        bad=replace(plan,model=replace(plan.model,label_vocabulary=replace(vocabulary,source_binding_sha256='f'*64)))
        with self.assertRaisesRegex(ValueError,'source identity'):t.Provider(prepared,bad)
        with self.assertRaisesRegex(ValueError,'source/pool/category'):t.Provider(replace(prepared,pool_sha256='f'*64),plan)

    def test_backend_collect_uses_actual_pre_context_and_original_cache_path(self):
        from pokezero import v5_run as runs
        raw,_=inert_d();recipe,_=records(raw,bank_fixture.DeviceActorBankTests().bank())
        plan,prepared,carry=backend_fixture.inputs()
        plan=replace(plan,model=replace(plan.model,arm='D'),loss=V5LossConfig(arm='D'),selection_seed=recipe.seed)
        provider=t.Provider.__new__(t.Provider);provider.prepared=prepared;provider.plan=plan;provider.recipe=recipe;provider.declaration_sha256=runs._sha(plan.manifest())
        provider.source=object();provider.materializer=object();provider.executor=object();provider.forwards={}
        worker=backend.RuntimeBackend(prepared,'inert-pool',teacher=provider,chunk_steps=3)
        with tempfile.TemporaryDirectory()as folder:
            store=runs.Store(folder,plan,resume=False);snapshots=[]
            for index in range(3):
                artifact=store.artifact(f'actors/{index}.npz',f'inert-{index}'.encode(),kind='INERT_TEST_ACTOR')
                snapshots.append(dict(archive=artifact['path'],archive_sha256=artifact['sha256'],checkpoint_sha256=str(index+2)*64,model_sha256=runs._sha(plan.model.metadata())))
            assignments=({'stream_id':101,'learner_side':0,'opponent':snapshots[1]},{'stream_id':202,'learner_side':1,'opponent':snapshots[2]})
            with patch('pokezero.v5_contract.source_variant_census',return_value={'source_rows_sha256':'0'*64}),patch.object(backend.v5_model,'load',
                    side_effect=lambda path,**kw:({'inert':np.array(0)},plan.model,{'checkpoint_sha256':kw['expected_checkpoint_sha256']})),patch.object(d,'make_collector',return_value=lambda c,p:raw)as collector:
                batch=worker.prepare_batch(plan,plan.iterations[0][0],snapshots[0],assignments,store)
            self.assertTrue(collector.call_args.kwargs['retain_teacher_roots'])
            seen=[]
            def detach(context,source,*,row,owner,enable_device_worlds):
                seen.append((int(context.streams[row]),int(context.frames[row]),owner))
                return NS(address=c.DecisionAddress(*seen[-1]))
            def price(*a,**kw):
                legal=kw['root_legal'];n=sum(legal);target=c.RootImprovement(tuple(1/n if x else 0. for x in legal),.4,tuple(.4 if x else None for x in legal),2)
                return NS(status='LABELED',reason=None,reduction=NS(improvement=target),world_reports=(),missing_worlds=0,actual_teacher_calls={},reply_input_approximation=())
            actual_boundary=d.boundary
            def no_roots(value):
                self.assertIsNone(value.roots);self.assertIsNone(value.carry.state);self.assertIsNone(value.carry.public)
                return actual_boundary(value)
            with patch.object(worker,'_construct',return_value=carry),patch.object(t.worlds,'detach_root',side_effect=detach),                    patch.object(t.worlds,'plan_starts',side_effect=lambda root,*a:NS(root=root)),patch.object(t.teacher,'price',side_effect=price),                    patch.object(d,'boundary',side_effect=no_roots),patch.object(d,'_complete_gae',side_effect=lambda v,o,l,discount:(v*0+o,v*0+o)):
                cache,_=worker.collect(batch,store,lambda *a,**kw:None,lambda:None)
            self.assertEqual(seen,[(101,0,0),(202,0,1),(101,1,0),(202,1,1)])
            result=next(v5_data.iter_cache_batches(store.path/cache,batch_size=100))
            self.assertTrue(result.search_policy_target_mask.all());self.assertTrue(result.search_value_target_mask.all())
            self.assertEqual(len(store.matching('DEVICE_TEACHER_CHUNK')),1)
            self.assertEqual(tuple(result.player_ids),('p1','p2','p1','p2'))

    def test_default_closed_factory_and_real_provider_startup(self):
        plan,prepared,_=backend_fixture.inputs();plan=replace(plan,model=replace(plan.model,arm='D'),loss=V5LossConfig(arm='D'))
        with self.assertRaisesRegex(Exception,'no FULL fallback'):phase.build_backend(plan,prepared,'pool',device_rollout=True)
        sentinel=object()
        with patch.object(t,'Provider',return_value=sentinel)as provider,patch.object(backend,'RuntimeBackend',return_value='configured')as factory:
            self.assertEqual(phase.build_backend(plan,prepared,'pool',device_rollout=True,device_teacher=True),'configured')
            provider.assert_called_once_with(prepared,plan);self.assertIs(factory.call_args.kwargs['teacher'],sentinel)
        for kw in ({'device_teacher':True},{'device_rollout':True,'device_teacher':1}):
            with self.assertRaises(ValueError):phase.build_backend(plan,prepared,'pool',**kw)

if __name__=='__main__':unittest.main()
