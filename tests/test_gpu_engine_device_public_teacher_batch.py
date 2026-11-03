"""Traceable inert worlds/actors: original serial math versus frame mapping."""
from dataclasses import replace
from typing import NamedTuple
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from pokezero import v5_contract, v5_gumbel
from gpu_engine import device_public_teacher as serial,device_public_teacher_batch as batch
from gpu_engine import device_public_worlds as plans,device_public_world_state as mapping
from tests.test_gpu_engine_device_public_teacher import facts

class State(NamedTuple):
    value:object
    done:object
    winner:object
    mode:object
    seed:object
    owner:object
class Public(NamedTuple):
    clock:object
    turn:object
class Observation(NamedTuple):
    legal_action_mask:object
    error:object
class Leaf(NamedTuple):
    observation:object
    inputs:object
    observer_sides:object
class Step(NamedTuple):
    state:object
    invalid_actions:object
class After(NamedTuple):
    step:object
    trackers:object
class Prediction(NamedTuple):
    policy_logits:object
    value:object
    valid:object

def forward(params,inputs,model=None,**kw):
    x=inputs['numeric_features'][:,0,0]
    return Prediction(j.broadcast_to(params['logits'],(len(x),9)),
        x[:,0]+params['offset'],j.ones(len(x),bool))

def executor():
    def legal(state):
        mask=j.asarray([[[True,True,*([False]*7)]]*2])
        return mask.at[:,1-state.owner[0]].set(j.where(state.mode[:,None]==6,False,mask[:,0]))
    def observe(state,public,owner,mask):
        # The namespace is global per one-row world, deliberately not batched.
        if public.clock.ndim!=0:raise ValueError('one-world namespace was flattened')
        error=j.where((state.mode==4)&(public.turn>0),1,0)
        feature=j.stack((state.value+(owner-state.owner)*.01,public.clock+j.zeros(1)),axis=-1)
        return Leaf(Observation(mask,error),{'numeric_features':feature[:,None,None,:]},owner)
    def advance(state,public,actions,chance):
        a=actions[:,state.owner[0]];r=actions[:,1-state.owner[0]]
        value=state.value+a*.03+j.maximum(r,0)*.02+chance*.001
        value=j.where(state.mode==5,j.nan,value)
        terminal=(state.mode==1)|(state.mode==2)|(state.mode==7)
        winner=j.where(state.mode==2,2,j.where(state.mode==7,9,state.owner))
        after=state._replace(value=value,done=terminal,winner=winner)
        return After(Step(after,(state.mode==3)&(a==1)&(r==1)),Public(public.clock,public.turn+1))
    ex=serial.Executor(legal,observe,lambda obs,state,seat,action:action,
        lambda state:state.seed%17,advance)
    return ex,replace(ex,joint=serial._make_joint(ex))

class FrameBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.root,_,cls.tables,cls.recipe=facts()
    def requests(self,modes=(0,0),owners=(0,1)):
        ex,_=executor();requests=[]
        for i,(mode,owner)in enumerate(zip(modes,owners)):
            stream=100+i
            while not self.recipe.selected(v5_contract.DecisionAddress(stream,4+i,owner)):stream+=100
            root=replace(self.root,address=v5_contract.DecisionAddress(stream,4+i,owner))
            start=plans.plan_starts(root,self.tables.source,self.recipe)
            state,public=self.world_data(root,start.starts[0],mode)
            inputs=ex.observe(state,public,j.asarray([owner]),ex.legal(state)[:,owner]).inputs
            requests.append(batch.Request(start,(0.,)*9,(True,True,*([False]*7)),inputs,i%2,1-i%2))
        return tuple(requests)
    @staticmethod
    def world_data(root,start,mode):
        state=State(j.asarray([.2],j.float32),j.asarray([False]),j.asarray([0]),j.asarray([mode]),
            j.asarray([start.seed%(2**32)],j.uint32),j.asarray([root.address.owner]))
        return state,Public(j.asarray(float(root.address.stream%79)),j.asarray([0]))
    def materialize(self,modes,missing=()):
        def materialize(root,start,tables,recipe):
            index=root.address.frame-4
            state,public=self.world_data(root,start,modes[index])
            absent=(index,start.world)in missing
            return mapping.World(start.world,None if absent else state,None if absent else public,
                ('inert missing',)if absent else(),start.seed,False)
        return materialize
    def compare(self,modes=(0,0),owners=(0,1),missing=(),alter=None):
        requests=self.requests(modes,owners);ex,mapped=executor()
        if alter:requests=alter(requests)
        params=({'offset':j.asarray(0.,j.float32),'logits':j.arange(9,dtype=j.float32)/9},
            {'offset':j.asarray(.07,j.float32),'logits':-j.arange(9,dtype=j.float32)/9})
        engine=batch.FramePricer(mapped,object(),capacity=4,block_rows=2,actor_capacity=3)
        with patch.object(mapping,'materialize',side_effect=self.materialize(modes,missing)),patch.object(batch.v5_model,'forward',side_effect=forward):
            actual=engine.price(requests,self.tables,self.recipe,params)
            expected=[]
            for r in requests:
                expected.append(serial.price(r.plans,self.tables,ex,self.recipe,root_logits=r.root_logits,
                    root_legal=r.root_legal,root_inputs=r.root_inputs,
                    opponent_forward=lambda x,p=params[r.opponent_actor]:forward(p,x),
                    learner_forward=lambda x,p=params[r.learner_actor]:forward(p,x),enable_device_teacher=True))
            cached=engine.params
            repeated=engine.price(requests,self.tables,self.recipe,params)
            self.assertIs(engine.params,cached)
        self.assertEqual(engine.last_timing['device_invocations'],int(len(missing)<4))
        for a,e,r in zip(actual,expected,repeated):
            self.assertEqual((a.status,a.reason,a.missing_worlds),(e.status,e.reason,e.missing_worlds))
            self.assertEqual(a.reduction,e.reduction);self.assertEqual(a.reduction,r.reduction)
            self.assertEqual(a.reply_input_approximation,e.reply_input_approximation)
            for aw,ew in zip(a.world_reports,e.world_reports):
                for key in ('world','posterior_seed','kernel_seed','status','required_pairs','completed_pairs','unknown_pairs','joint_values'):
                    self.assertEqual(aw[key],ew[key],key)
        return actual
    def test_same_keys_public_namespace_both_owners_mixed_actors_padding_and_reuse(self):self.compare()
    def test_terminal_win_tie_and_invalid_winner(self):
        self.compare((1,2));self.compare((7,0))
    def test_positive_reply_failure_encoder_error_and_nonfinite_are_not_renormalized(self):
        for mode in (3,4,5):self.compare((mode,0))
    def test_missing_one_or_all_worlds_retains_slots(self):
        self.compare(missing=((0,0),));self.compare(missing=((0,0),(0,1),(1,0),(1,1)))
    def test_no_opponent_request_keeps_literal_dummy_reply(self):self.compare((6,0))
    def test_changed_root_inputs_refuse_before_pricing(self):
        self.compare(alter=lambda rs:(replace(rs[0],root_inputs={'numeric_features':rs[0].root_inputs['numeric_features']+1}),rs[1]))
    def test_uniform_current_learner_path_has_the_same_targets(self):
        self.compare(alter=lambda rs:tuple(replace(r,learner_actor=0)for r in rs))
    def test_first_mismatching_field_is_serial_numeric_before_categorical(self):
        plain,_=executor()
        def observe(*args):
            view=plain.observe(*args)
            return view._replace(inputs=dict(view.inputs,categorical_ids=j.ones((1,1,1,1),j.int32)))
        ex=replace(plain,observe=observe);mapped=replace(ex,joint=serial._make_joint(ex))
        r=self.requests()[0]
        r=replace(r,root_inputs=dict(r.root_inputs,numeric_features=r.root_inputs['numeric_features']+1,categorical_ids=j.zeros((1,1,1,1),j.int32)))
        params=({'offset':j.asarray(0.,j.float32),'logits':j.zeros(9)},{'offset':j.asarray(.07,j.float32),'logits':j.zeros(9)})
        pricer=batch.FramePricer(mapped,object(),capacity=2,block_rows=1,actor_capacity=2)
        with patch.object(mapping,'materialize',side_effect=self.materialize((0,0))),patch.object(batch.v5_model,'forward',side_effect=forward):
            actual=pricer.price((r,),self.tables,self.recipe,params)[0]
            expected=serial.price(r.plans,self.tables,ex,self.recipe,root_logits=r.root_logits,root_legal=r.root_legal,root_inputs=r.root_inputs,
                opponent_forward=lambda x:forward(params[1],x),learner_forward=lambda x:forward(params[0],x),enable_device_teacher=True)
        self.assertEqual(actual.world_reports[0]['reason'],expected.world_reports[0]['reason'])
        self.assertTrue(actual.world_reports[0]['reason'].endswith('numeric_features'))
        self.assertEqual(pricer.last_timing['pre_refused_available_worlds'],2)

    def test_evaluated_padding_substitution_and_pre_refusal_work_count_once_per_frame(self):
        _,ex=executor();r=self.requests()[0];params=({'offset':j.asarray(0.,j.float32),'logits':j.zeros(9)},{'offset':j.asarray(.07,j.float32),'logits':j.zeros(9)})
        pricer=batch.FramePricer(ex,object(),capacity=4,block_rows=2,actor_capacity=2);shapes=[]
        def traced(p,inputs,*a,**kw):
            shapes.append(inputs['numeric_features'].shape[0]);return forward(p,inputs,*a,**kw)
        with patch.object(mapping,'materialize',side_effect=self.materialize((0,0),missing=((0,0),))),patch.object(batch.v5_model,'forward',side_effect=traced):
            result=pricer.price((r,),self.tables,self.recipe,params)[0]
        work=pricer.last_timing
        self.assertIn(324,shapes);self.assertEqual(work['evaluated_leaf_forward_rows'],324)
        self.assertEqual(work['evaluated_root_slots'],2);self.assertEqual(work['evaluated_world_slots'],4)
        self.assertEqual(work['padding_root_slots'],1);self.assertEqual(work['padding_world_slots'],2)
        self.assertEqual(work['substituted_world_slots'],1)
        self.assertEqual(result.actual_teacher_calls['public_leaf'],81)
        self.assertIn('LOGICAL',result.actual_teacher_calls['scope'])

    def test_capacities_and_actor_bank_fail_closed(self):
        _,ex=executor()
        for cap,block,actors in ((0,2,1),(3,2,1),(4,0,1),(4,2,0)):
            with self.assertRaises(ValueError):batch.FramePricer(ex,None,capacity=cap,block_rows=block,actor_capacity=actors)
        pricer=batch.FramePricer(ex,None,capacity=2,block_rows=1,actor_capacity=1)
        with self.assertRaises(ValueError):pricer.bind(({},{}))

if __name__=='__main__':unittest.main()
