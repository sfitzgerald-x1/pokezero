"""Inert R7 batch geometry and chance-domain checks; no saved roots/models/games."""
import math
from types import SimpleNamespace
from typing import NamedTuple
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine.phase_d_depthn import AddressedState,folded_state,PreparedPricer,Prediction,addressed_executor
from gpu_engine.phase_d_ladder_search import Node,domain_seed


class State(NamedTuple):
    keys:object
    turn:object
    score:object
    done:object
    winner:object


class Obs(NamedTuple):
    legal_action_mask:object
    error:object


class View(NamedTuple):
    observation:object
    inputs:object


class Step(NamedTuple):
    invalid_actions:object
    state:object


class After(NamedTuple):
    step:object
    trackers:object


def legal(s):return j.tile(j.array([True,True]+[False]*7),(1,2,1))


def observe(s,p,owner,mask):
    # Different seat policies and a nonzero value; request lanes are reversed
    # against physical engine actions, so borrowing the other mapping is visible.
    x=j.stack((s.score,owner.astype(j.float32)),axis=-1)[:,None,:]
    return View(Obs(mask,j.zeros(1,j.int32)),dict(numeric=x,categorical=j.zeros((1,1,1),j.int32),
        token_types=j.zeros((1,1),j.int32),attention=j.ones((1,1),bool)))


def dispatch(obs,s,owner,a):return 1-a


def sample(s):return jax.random.randint(s.keys[0],(),0,97)


def advance(s,p,actions,chance):
    splits=jax.vmap(lambda k:jax.random.split(k,2))(s.keys)
    score=s.score+.13*actions[:,0]-.21*actions[:,1]+chance*.0001
    return After(Step(j.zeros(1,bool),s._replace(keys=splits[:,0],turn=s.turn+1,score=score)),p+1)


def forward(inputs):
    x=inputs['numeric'][:,0];n=x.shape[0]
    logits=j.zeros((n,9)).at[:,0].set(x[:,1]*.7).at[:,1].set(.4-x[:,1]*.2)
    return Prediction(logits,j.tanh(x[:,0]),j.ones(n,bool))


def state(i):return State(jax.random.PRNGKey(31+i)[None],j.zeros(1,j.int32),j.array([i*.01]),j.zeros(1,bool),j.full(1,-1,j.int32))


class ProtocolDepthNTests(unittest.TestCase):
    def test_original_domain_fold_and_propagated_keys_at_both_plies(self):
        for world in (0,1,31):
            s=state(world);prefix='a'*64
            seeds=j.array([[domain_seed(prefix,world,p,0,domain='sampled-chance')for p in (0,1)]],j.uint32)
            wrapped=AddressedState(s,seeds,j.zeros(1,j.int32))
            for ply in (0,1):
                original=s._replace(keys=jax.vmap(jax.random.fold_in)(s.keys,seeds[:,ply]))
                np.testing.assert_array_equal(folded_state(wrapped).keys,original.keys)
                executor=addressed_executor(legal,observe,dispatch,sample,advance)
                actual=executor.advance(wrapped,j.array([0]),j.array([[1,0]]),executor.sample(wrapped))
                expected=advance(original,j.array([0]),j.array([[1,0]]),sample(original))
                for a,b in zip(jax.tree.leaves(actual.step.state.state),jax.tree.leaves(expected.step.state)):
                    np.testing.assert_array_equal(a,b)
                s=expected.step.state;wrapped=actual.step.state

    def test_full_width_prepared_worlds_match_independent_serial_two_ply(self):
        prefix='b'*64;nodes=[Node(state(i),j.array([0]),0,prefix,i,0)for i in range(2)]
        executor=addressed_executor(legal,observe,dispatch,sample,advance)
        pricer=PreparedPricer(executor,forward,worlds=2,block_lanes=81,blocks_per_call=1)
        _,reports=pricer.price(nodes,[True,True]+[False]*7)
        def policy(owner):
            logits=np.array(forward(observe(state(0),None,j.array([owner]),legal(state(0))[:,owner]).inputs).policy_logits[0,:2])
            e=np.exp(logits-logits.max());return e/e.sum()
        weights=policy(1)
        def step(s,world,ply,a,r):
            seed=domain_seed(prefix,world,ply,0,domain='sampled-chance')
            s=s._replace(keys=jax.vmap(jax.random.fold_in)(s.keys,j.array([seed],j.uint32)))
            return advance(s,j.array([0]),j.array([[1-a,1-r]]),sample(s)).step.state
        for wi,node in enumerate(nodes):
            for a in (0,1):
                q=[]
                for r in (0,1):
                    child=step(node.state,wi,0,a,r);scores=[]
                    for aa in (0,1):
                        values=[float(forward(observe(step(child,wi,1,aa,rr),None,j.array([0]),legal(child)[:,0]).inputs).value[0])for rr in (0,1)]
                        scores.append(math.fsum(float(w)*v for w,v in zip(weights,values)))
                    q.append(max(scores))
                self.assertAlmostEqual(reports[wi][a],math.fsum(float(w)*v for w,v in zip(weights,q)),places=6)
        self.assertEqual(set(reports[0]),{0,1})
        self.assertEqual(pricer.last_timing['substituted_world_slots'],0)
        self.assertEqual(pricer.last_timing['enumerated_leaves'],32)

    def test_terminals_are_owner_relative_and_do_not_expand(self):
        def terminal(s,p,a,c):
            result=advance(s,p,a,c)
            return result._replace(step=result.step._replace(state=result.step.state._replace(
                done=j.ones(1,bool),winner=j.ones(1,j.int32))))
        for owner in (0,1):
            nodes=[Node(state(i),j.array([0]),owner,'c'*64,i,0)for i in range(2)]
            ex=addressed_executor(legal,observe,dispatch,sample,terminal)
            p=PreparedPricer(ex,forward,worlds=2,block_lanes=81,blocks_per_call=1)
            _,reports=p.price(nodes,[True,True]+[False]*7)
            expected=1. if owner==1 else -1.
            self.assertEqual(reports,[{0:expected,1:expected}]*2)
            self.assertEqual(p.last_timing['enumerated_leaves'],0)

    def test_one_required_leaf_refuses_full_point_without_survivors(self):
        def invalid(s,p,a,c):
            result=advance(s,p,a,c)
            return result._replace(step=result.step._replace(invalid_actions=
                (s.turn==1)&(a[:,0]==1)&(a[:,1]==0)))
        nodes=[Node(state(i),j.array([0]),0,'d'*64,i,0)for i in range(2)]
        ex=addressed_executor(legal,observe,dispatch,sample,invalid)
        p=PreparedPricer(ex,forward,worlds=2,block_lanes=81,blocks_per_call=1)
        with self.assertRaisesRegex(ValueError,'REQUIRED_BRANCH_UNKNOWN'):
            p.price(nodes,[True,True]+[False]*7)

    def test_original_world_order_rejected_not_resampled(self):
        ex=addressed_executor(legal,observe,dispatch,sample,advance)
        p=PreparedPricer(ex,forward,worlds=2)
        nodes=[Node(state(i),j.array([0]),0,'a'*64,1-i,0)for i in range(2)]
        with self.assertRaisesRegex(ValueError,'WORLD_ORDER'):p.price(nodes,[True,True]+[False]*7)


if __name__=='__main__':unittest.main()
