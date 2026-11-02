"""Declared synthetic source/state and fake callbacks, never engine/model runs."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_contract, v5_gumbel
from pokezero.gen3_damage import gen3_hp_stat
from gpu_engine import device_public_worlds as plans
from gpu_engine import device_public_world_state as mapping
from gpu_engine import device_public_teacher as teacher
from gpu_engine.engine_device_public_hooks import initialize
from gpu_engine.state import empty_state
from gpu_engine.v5_device_rollout import RootContext
import test_gpu_engine_device_public_worlds as fixtures


def facts(owner=0, frame=2):
    raw_source, metadata = fixtures.source_fixture()
    metadata['vocabulary']['types'] = ['', 'Normal']
    metadata['moves'] = [None, *[{'pp': 10} for _ in range(4)]]
    for info in raw_source.species_metadata.values():
        info.update(baseStats={name: 100 for name in ('hp','atk','def','spa','spd','spe')},types=['Normal'])
    source = plans.compile_source(raw_source, metadata, expected_source_hash='synthetic-source',
        expected_source_rows_sha256=v5_contract.source_variant_census(raw_source)['source_rows_sha256'])
    state = empty_state(1)._replace(
        species=j.asarray([[[1,2,3,4,5,6],[1,2,3,4,5,6]]],j.int32),
        hp=j.full((1,2,6),100,j.int32),max_hp=j.full((1,2,6),100,j.int32),
        stats=j.full((1,2,6,6),100,j.int32),level=j.full((1,2,6),80,j.int32),
        types=j.asarray([[[[1,0]]*6]*2],j.int32),base_types=j.asarray([[[[1,0]]*6]*2],j.int32),
        moves=j.broadcast_to(j.arange(1,5,dtype=j.int32),(1,2,6,4)),
        pp=j.full((1,2,6,4),10,j.int32),gender=j.zeros((1,2,6),j.int32),
        active=j.zeros((1,2),j.int32),entry_pending=j.zeros((1,2),bool),done=j.asarray([False]))
    belief=SimpleNamespace(species_initial_mask=j.tile(j.asarray([[3,0]],j.uint32),(8,1)),
        species_types=j.tile(j.asarray([[1,0]],j.int32),(8,1)))
    tracker=initialize(state,belief).trackers
    context=RootContext(state,tracker,j.asarray([81],j.int32),j.asarray([frame],j.int32))
    root=plans.detach_root(context,source,row=0,owner=owner,enable_device_worlds=True)
    recipe=next(v5_gumbel.Recipe(seed) for seed in range(100)
        if v5_gumbel.Recipe(seed).selected(root.address))
    starts=plans.plan_starts(root,source,recipe)
    return context,root,starts,mapping.compile_tables(source,metadata),recipe


class DevicePublicTeacherTests(unittest.TestCase):
    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            np.testing.assert_array_equal(x,y)

    def test_tensor_world_owner_exact_source_foe_public_scaling_and_rebase(self):
        for owner in (0,1):
            context,root,starts,tables,recipe=facts(owner)
            first=starts.starts[0]
            world=mapping.materialize(root,first,tables,recipe)
            self.assertFalse(world.refusals)
            self.assertIsNotNone(world.state)
            for name in plans.OWN_MON_FIELDS+plans.OWN_SIDE_FIELDS:
                np.testing.assert_array_equal(getattr(world.state,name)[0,owner],getattr(context.state,name)[0,owner])
            foe=1-owner
            mon=first.opponent_fixtures[0]
            maximum=gen3_hp_stat(100,mon.ivs['hp'],mon.evs['hp'],mon.level)
            self.assertEqual(int(world.state.hp[0,foe,0]),maximum)
            self.assertEqual(int(world.state.stats[0,foe,0,0]),maximum)
            np.testing.assert_array_equal(world.public.physical_to_public[0,foe],[0,-1,-1,-1,-1,-1])
            self.assertEqual(int(world.public.active_physical[0,foe]),0)
            np.testing.assert_array_equal(world.public.own_order[0,foe],np.arange(6))
            self.same(plans.public.readout(root.public,j.asarray([owner],j.int32)),
                plans.public.readout(world.public,j.asarray([owner],j.int32)))
            # Private actual-foe changes cannot enter detached materialization.
            changed=context.state
            for name in ('species','hp','max_hp','stats','moves','pp','ability','types','status_counter'):
                value=getattr(changed,name);changed=changed._replace(**{name:value.at[:,foe].set(999)})
            detached=plans.detach_root(context._replace(state=changed),tables.source,
                row=0,owner=owner,enable_device_worlds=True)
            altered=mapping.materialize(detached,first,tables,recipe)
            self.same(world.state,altered.state);self.same(world.public,altered.public)
            # Float32 encoder fraction cannot perturb the original integer
            # percentage -> round(sampled maxHP) materializer convention.
            view=dict(root.view)
            view['hp_percent']=np.asarray([[69,0,0,0,0,0]],np.int32)
            view['hp_fraction']=np.asarray([[.69,0,0,0,0,0]],np.float32)
            scaled=mapping.materialize(replace(root,view=plans._freeze(view)),first,tables,recipe)
            self.assertFalse(scaled.refusals)
            self.assertEqual(int(scaled.state.hp[0,foe,0]),max(1,round(69*maximum/100)))

    def test_physical_unknowns_keep_world_slots_and_do_not_use_donor_clocks(self):
        _,root,starts,tables,recipe=facts()
        first=starts.starts[0]
        for name,value in (('forced_switch',np.asarray(True)),('entry_pending',np.asarray(True))):
            own={**root.own,name:value}
            bad=mapping.materialize(replace(root,own=own),first,tables,recipe)
            self.assertIsNone(bad.state);self.assertTrue(bad.refusals)
        for field,value in (('status',5),('hp_percent',0)):
            view=dict(root.view);view[field]=np.array(view[field],copy=True);view[field][0,0]=value
            if field=='hp_percent':
                view['hp_fraction']=np.array(view['hp_fraction'],copy=True);view['hp_fraction'][0,0]=0
            bad=mapping.materialize(replace(root,view=plans._freeze(view)),first,tables,recipe)
            self.assertIsNone(bad.state);self.assertTrue(bad.refusals)
        with self.assertRaises(ValueError):mapping.materialize(root,replace(first,seed=first.seed+1),tables,recipe)
        changed=replace(first.opponent_fixtures[0],evs={**first.opponent_fixtures[0].evs,'atk':252})
        bad=mapping.materialize(root,replace(first,opponent_fixtures=(changed,*first.opponent_fixtures[1:])),tables,recipe)
        self.assertIsNone(bad.state);self.assertIn('source spread',bad.refusals[0])

    def executor(self, owner, *, failed_pair=None):
        calls={'sample':[],'advance':[],'observe':[],'opponent':[],'learner':[]}
        mask=j.asarray([[[True,True,*([False]*7)]]*2])
        def observe(state,public,seat,legal):
            who=int(seat[0]);calls['observe'].append((who,public))
            inputs={'numeric_features':j.asarray([[state.hp[0,who,0]/100,public.turn[0]]],j.float32)}
            return SimpleNamespace(public=public,observer_sides=seat,inputs=inputs,
                observation=SimpleNamespace(legal_action_mask=legal,error=j.asarray([0],j.int32)))
        def sample(state):
            chance=object();calls['sample'].append(chance);return chance
        def advance(state,public,actions,chance):
            a,r=int(actions[0,owner]),int(actions[0,1-owner])
            calls['advance'].append((a,r,chance,public))
            if failed_pair==(a,r):raise ValueError('declared synthetic physical refusal')
            after=state._replace(hp=state.hp.at[0,owner,0].set(50+a+r))
            return SimpleNamespace(step=SimpleNamespace(state=after,invalid_actions=j.asarray([False])),
                trackers=public._replace(turn=public.turn+1))
        executor=teacher.Executor(lambda state:mask,observe,lambda obs,state,seat,action:action,sample,advance)
        def opponent(inputs):
            calls['opponent'].append(inputs)
            return SimpleNamespace(policy_logits=j.asarray([[0.,1.,0.,0.,0.,0.,0.,0.,0.]]),
                value=j.asarray([0.]),valid=j.asarray([True]))
        def learner(inputs):
            calls['learner'].append(inputs)
            return SimpleNamespace(policy_logits=j.zeros((1,9)),value=inputs['numeric_features'][:,0],valid=j.asarray([True]))
        return executor,opponent,learner,calls

    def test_complete_original_two_world_all_reply_math_same_public_leaf(self):
        for owner in (0,1):
            _,root,starts,tables,recipe=facts(owner)
            executor,opponent,learner,calls=self.executor(owner)
            world=mapping.materialize(root,starts.starts[0],tables,recipe)
            root_inputs=executor.observe(world.state,world.public,j.asarray([owner],j.int32),
                executor.legal(world.state)[:,owner]).inputs
            result=teacher.price(starts,tables,executor,recipe,root_logits=(0.,)*9,
                root_legal=(True,True,*([False]*7)),root_inputs=root_inputs,
                opponent_forward=opponent,learner_forward=learner,enable_device_teacher=True)
            self.assertEqual(result.status,'LABELED')
            self.assertEqual(result.actual_teacher_calls,dict(reply_policy=2,public_leaf=8,chance=2,transitions=8))
            self.assertEqual(result.reduction.required_pairs,8)
            self.assertEqual(result.reduction.unknown_pairs,0)
            self.assertEqual(len(result.reply_input_approximation),2)
            self.assertTrue(all(not row['exact_known_zero'] and row['treatment']=='DECLARED_UNKNOWN_ZERO'
                for row in result.reply_input_approximation))
            for k in range(2):
                self.assertTrue(all(row[2] is calls['sample'][k] for row in calls['advance'][4*k:4*k+4]))
            expected=v5_gumbel.reduce_completed_q(root_logits=(0.,)*9,root_legal=(True,True,*([False]*7)),
                worlds=[v5_gumbel.WorldPrices(True,(True,True,*([False]*7)),
                    (0.,1.,0.,0.,0.,0.,0.,0.,0.),report['joint_values']) for report in result.world_reports])
            self.assertEqual(result.reduction,expected)
            self.assertTrue(all(float(inp['numeric_features'][0,1])==2 for inp in calls['learner']))

    def test_refused_positive_branch_not_renormalized_and_world_unknown_not_wait(self):
        _,root,starts,tables,recipe=facts()
        executor,opponent,learner,calls=self.executor(0,failed_pair=(0,1))
        world=mapping.materialize(root,starts.starts[0],tables,recipe)
        inputs=executor.observe(world.state,world.public,j.asarray([0],j.int32),executor.legal(world.state)[:,0]).inputs
        args=dict(root_logits=(0.,)*9,root_legal=(True,True,*([False]*7)),root_inputs=inputs,
            opponent_forward=opponent,learner_forward=learner,enable_device_teacher=True)
        result=teacher.price(starts,tables,executor,recipe,**args)
        self.assertEqual(result.status,'REFUSED');self.assertEqual(result.reduction.unknown_pairs,2)
        self.assertGreater(result.reduction.unknown_reply_mass,.5)
        self.assertEqual(result.actual_teacher_calls['transitions'],8)
        refused=replace(starts,starts=(replace(starts.starts[0],opponent_fixtures=None,refusals=('source',)),starts.starts[1]))
        unknown=teacher.price(refused,tables,executor,recipe,**args)
        self.assertEqual(unknown.missing_worlds,1);self.assertIsNone(unknown.reduction)
        self.assertIsNone(unknown.world_reports[0]['required_pairs'])

    def test_cached_factory_wires_same_encoder_explicit_chance_and_no_models(self):
        # Inspect actual retained closures by replacing jit with identity and
        # original pure callees with sentinels. No engine/model body is run.
        engine,observation=object(),SimpleNamespace(belief=object())
        marker=object();chance=object();state=object();public=object()
        with patch.object(teacher.jax,'jit',side_effect=lambda fn:fn) as jit, \
                patch.object(teacher.transition,'action_mask',return_value=j.ones((1,2,9),bool)) as legal, \
                patch.object(teacher,'request_seats',return_value=j.asarray([[True,False]])), \
                patch.object(teacher.leaf,'from_public',return_value=marker) as observe, \
                patch.object(teacher.transition,'sample_chance_inputs',return_value=chance) as sample, \
                patch.object(teacher.transition,'step_with_public',return_value=marker) as advance:
            executor=teacher.make_executor(engine,observation,enable_device_teacher=True)
            self.assertEqual(jit.call_count,5)
            self.assertFalse(np.asarray(executor.legal(state))[0,1].any())
            self.assertIs(executor.observe(state,public,j.asarray([0]),j.ones((1,9),bool)),marker)
            self.assertIs(observe.call_args.args[1],public)
            self.assertIs(observe.call_args.args[2],observation)
            self.assertIs(executor.sample(state),chance)
            self.assertIs(executor.advance(state,public,j.zeros((1,2),j.int32),chance),marker)
            self.assertIs(advance.call_args.kwargs['chance_inputs'],chance)
            self.assertIs(advance.call_args.kwargs['belief_tables'],observation.belief)
            self.assertEqual(sample.call_count,1)


if __name__=='__main__':unittest.main()
