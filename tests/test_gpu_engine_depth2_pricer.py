"""Depth-2 pricer on inert two-ply worlds: a pure-Python two-step oracle, D3 turn-1 equality, pure backups."""
from dataclasses import replace
import math
from typing import NamedTuple
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_contract, v5_gumbel
from gpu_engine import device_public_teacher as serial, device_public_teacher_batch as batch
from gpu_engine import device_depth2_pricer as pricer, depth2_backups as backups, depth2_digest
from gpu_engine import device_public_worlds as plans, device_public_world_state as mapping
from tests.test_gpu_engine_device_public_teacher import facts
from tests.test_gpu_engine_device_public_teacher_batch import Observation, Leaf, Step, After, Prediction


class State(NamedTuple):
    value:object
    done:object
    winner:object
    mode:object
    seed:object
    owner:object
    turn:object
    last:object
class Public(NamedTuple):
    clock:object
    turn:object

ROOT_OWN=np.array([1,1,1,0,0,0,0,0,0],bool);ROOT_FOE=np.array([1,1,0,0,0,0,0,0,0],bool)
CHILD_OWN=np.array([1,1,0,1,1,0,0,0,0],bool);CHILD_FOE=np.array([1,0,1,1,1,0,0,0,0],bool)
ROOT_LOGITS=(0.,)*9;ROOT_LEGAL=tuple(map(bool,ROOT_OWN))
# Modes: turn-1 (root step) 1 win, 2 draw, 7 bad winner, 3 invalid (1,1), 4 encoder error, 5 nonfinite, 6 waiting
# opponent at the root; turn-2 (child step) 20 waiting learner, 21 waiting opponent, 22 terminal for a2==1 or r2==3
# (draw when r2==3), 23 leaf encoder error for a2==1, 24 invalid (1,2), 25 nonfinite leaf for a2==3, 28 bad winner
# for a2==0, 27 the learner's child mask depends on its turn-1 action.
# Every fake quantity is a small dyadic rational, so jit fusion (FMA contraction) and eager evaluation agree bit for bit.
PARAMS=({'offset':j.asarray(0.,j.float32),'logits':j.asarray([.25,.125,.375,.125,.5,.875,.25,.625,.5],j.float32),'tilt':j.asarray(.5,j.float32)},
    {'offset':j.asarray(.0625,j.float32),'logits':j.asarray([.25,.75,.125,.625,.375,0.,.5,.875,.125],j.float32),'tilt':j.asarray(-.5,j.float32)},
    {'offset':j.asarray(-.0625,j.float32),'logits':j.zeros(9,j.float32),'tilt':j.asarray(0.,j.float32)})


def forward(params,inputs,model=None,**kw):
    x=inputs['numeric_features'][:,0,0]
    return Prediction(params['logits']+x[:,:1]*params['tilt']*j.arange(9),x[:,0]+x[:,2]+params['offset'],j.ones(len(x),bool))


def executor():
    def legal(state):
        root=(state.turn==0)[:,None];mode=state.mode[:,None];owner=state.owner[0]
        own=j.where(root,ROOT_OWN,j.where(mode==20,False,j.where((mode==27)&(state.last==1)[:,None],CHILD_OWN&(j.arange(9)!=1),CHILD_OWN)))
        foe=j.where(root,j.where(mode==6,False,ROOT_FOE),j.where(mode==21,False,CHILD_FOE))
        mask=j.zeros((1,2,9),bool).at[:,owner].set(own).at[:,1-owner].set(foe)
        return mask&~state.done[:,None,None]
    def observe(state,public,owner,mask):
        if public.clock.ndim!=0:raise ValueError('one-world namespace was flattened')
        error=j.where(((state.mode==4)&(public.turn>0))|((state.mode==23)&(state.turn==2)&(state.last==1)),1,0)
        poison=j.where((state.mode==25)&(state.turn==2)&(state.last==3),j.nan,0.)
        feature=j.stack((state.value+(owner-state.owner)*(1/128),public.clock+j.zeros(1),poison),axis=-1)
        return Leaf(Observation(mask,error),{'numeric_features':feature[:,None,None,:]},owner)
    def advance(state,public,actions,chance):
        a=actions[:,state.owner[0]];r=actions[:,1-state.owner[0]];root=state.turn==0;mode=state.mode
        value=state.value+a*(1/32)+j.maximum(r,0)*(1/64)+chance*(1/1024)
        value=j.where(mode==5,j.nan,value)
        terminal=j.where(root,(mode==1)|(mode==2)|(mode==7),((mode==22)&((a==1)|(r==3)))|((mode==28)&(a==0)))
        winner=j.where(root,j.where(mode==2,2,j.where(mode==7,9,state.owner)),j.where(mode==28,9,j.where(r==3,2,state.owner)))
        invalid=j.where(root,(mode==3)&(a==1)&(r==1),(mode==24)&(a==1)&(r==2))
        after=state._replace(value=value,done=terminal,winner=winner,seed=state.seed*3+1,turn=state.turn+1,last=a)
        return After(Step(after,invalid),Public(public.clock,public.turn+1))
    ex=serial.Executor(jax.jit(legal),jax.jit(observe),lambda obs,state,seat,action:action,jax.jit(lambda state:state.seed%17),jax.jit(advance))
    return ex,replace(ex,joint=serial._make_joint(ex))


def world_data(root,start,mode):
    state=State(j.asarray([.25],j.float32),j.asarray([False]),j.asarray([0],j.int32),j.asarray([mode],j.int32),
        j.asarray([start.seed%(2**32)],j.uint32),j.asarray([root.address.owner],j.int32),j.asarray([0],j.int32),j.asarray([-1],j.int32))
    return state,Public(j.asarray(float(root.address.stream%79),j.float32),j.asarray([0],j.int32))


def usable(p):
    return bool(p.valid[0])and bool(np.isfinite(np.asarray(p.policy_logits)).all())and bool(np.isfinite(np.asarray(p.value)).all())and abs(float(p.value[0]))<=1


def oracle(ex,params,state,public,owner,order,learner,opponent,k_own,k_opp,k1_opp):
    """Two plies by hand on one row at a time: the contract's semantics, independent of the kernel's layout."""
    Z=lambda *s,d=np.float32:np.zeros(s,d)
    o=dict(t1_status=Z(9,9,d=np.int8),t1_value=Z(9,9),t1_error=Z(9,9,d=np.int8),t1_own_requested=Z(9,9,d=bool),t1_own_legal=Z(9,9,9,d=bool),
        t1_own_logits=Z(9,9,9),t1_peer_requested=Z(9,9,d=bool),t1_peer_legal=Z(9,9,9,d=bool),t1_peer_logits=Z(9,9,9),
        t2_status=Z(9,9,9,9,d=np.int8),t2_value=Z(9,9,9,9),t2_error=Z(9,9,9,9,d=np.int8),inputs1={},inputs2={})
    seat=j.asarray([owner],j.int32);masks=ex.legal(state)
    peer=ex.observe(state,public,1-seat,masks[:,1-owner])
    legal=np.asarray(peer.observation.legal_action_mask[0]);requested=bool(legal.any())
    pick=lambda legal,lg,k,req:sorted([i for i in range(9)if legal[i]],key=lambda i:(-float(lg[i]),i))[:k]if req else[0]
    if requested:lg=np.asarray(forward(params[opponent],peer.inputs).policy_logits[0]);replies=sorted(pick(legal,lg,k1_opp,True))
    else:lg=np.zeros(9,np.float32);replies=[0]
    o.update(root_peer_requested=requested,root_peer_legal=legal,root_peer_logits=lg)
    outcome=lambda winner:0. if winner==2 else(1. if winner==owner else-1.)
    chance=ex.sample(state)
    for a1 in order:
        for r1 in replies:
            actions=j.full((1,2),-1,j.int32).at[:,owner].set(a1)
            if requested:actions=actions.at[:,1-owner].set(r1)
            after=ex.advance(state,public,actions,chance);child=after.step.state;trackers=after.trackers
            invalid=bool(after.step.invalid_actions[0]);done=bool(child.done[0]);winner=int(child.winner[0])
            if invalid:o['t1_status'][a1,r1]=3;o['t1_error'][a1,r1]=1;continue
            if done:
                if winner in(0,1,2):o['t1_status'][a1,r1]=1;o['t1_value'][a1,r1]=outcome(winner)
                else:o['t1_status'][a1,r1]=3;o['t1_error'][a1,r1]=2
                continue
            cm=ex.legal(child);vo=ex.observe(child,trackers,seat,cm[:,owner]);vp=ex.observe(child,trackers,1-seat,cm[:,1-owner])
            po=forward(params[learner],vo.inputs);pp=forward(params[opponent],vp.inputs)
            own_legal=np.asarray(cm[0,owner]);peer_legal=np.asarray(cm[0,1-owner]);own_req=bool(own_legal.any());peer_req=bool(peer_legal.any())
            if int(vo.observation.error[0])or int(vp.observation.error[0]):o['t1_status'][a1,r1]=3;o['t1_error'][a1,r1]=3;continue
            if not usable(po)or(peer_req and not usable(pp)):o['t1_status'][a1,r1]=3;o['t1_error'][a1,r1]=4;continue
            o['t1_status'][a1,r1]=2;o['t1_value'][a1,r1]=float(po.value[0]);o['inputs1'][a1,r1]=(vo.inputs,vp.inputs)
            own_lg=np.asarray(po.policy_logits[0])if own_req else np.zeros(9,np.float32)
            peer_lg=np.asarray(pp.policy_logits[0])if peer_req else np.zeros(9,np.float32)
            o['t1_own_requested'][a1,r1]=own_req;o['t1_own_legal'][a1,r1]=own_legal;o['t1_own_logits'][a1,r1]=own_lg
            o['t1_peer_requested'][a1,r1]=peer_req;o['t1_peer_legal'][a1,r1]=peer_legal;o['t1_peer_logits'][a1,r1]=peer_lg
            chance2=ex.sample(child)
            for a2 in pick(own_legal,own_lg,k_own,own_req):
                for r2 in pick(peer_legal,peer_lg,k_opp,peer_req):
                    acts=j.full((1,2),-1,j.int32)
                    if own_req:acts=acts.at[:,owner].set(a2)
                    if peer_req:acts=acts.at[:,1-owner].set(r2)
                    leaf=ex.advance(child,trackers,acts,chance2);ls=leaf.step.state
                    invalid=bool(leaf.step.invalid_actions[0]);done=bool(ls.done[0]);winner=int(ls.winner[0])
                    if invalid:status,value,err=3,0.,1
                    elif done:status,value,err=(1,outcome(winner),0)if winner in(0,1,2)else(3,0.,2)
                    else:
                        lm=ex.legal(ls);lv=ex.observe(ls,leaf.trackers,seat,lm[:,owner]);pl=forward(params[learner],lv.inputs)
                        o['inputs2'][a1,r1,a2,r2]=lv.inputs
                        if int(lv.observation.error[0]):status,value,err=3,0.,3
                        elif not usable(pl):status,value,err=3,0.,4
                        else:status,value,err=4,float(pl.value[0]),0
                    o['t2_status'][a1,r1,a2,r2]=status;o['t2_value'][a1,r1,a2,r2]=value;o['t2_error'][a1,r1,a2,r2]=err
    return o


def synthetic(roots=1):
    Z=lambda shape,d:np.zeros((roots,2)+shape,d)
    return pricer.Depth2Frame(pre=Z((),np.int32),root_peer_requested=Z((),bool),root_peer_legal=Z((9,),bool),root_peer_logits=Z((9,),np.float32),
        t1_status=Z((9,9),np.int8),t1_value=Z((9,9),np.float32),t1_error=Z((9,9),np.int8),
        t1_own_requested=Z((9,9),bool),t1_own_legal=Z((9,9,9),bool),t1_own_logits=Z((9,9,9),np.float32),
        t1_peer_requested=Z((9,9),bool),t1_peer_legal=Z((9,9,9),bool),t1_peer_logits=Z((9,9,9),np.float32),
        t2_status=Z((9,9,9,9),np.int8),t2_value=Z((9,9,9,9),np.float32),t2_error=Z((9,9,9,9),np.int8),world_reports=(),timing={})


def reference_backup(frame,root,world,maximize):
    """Brute-force WorldPrices surface from the frame arrays, written without the backups module."""
    f=lambda name:getattr(frame,name)[root,world]
    grid=[[None]*9 for _ in range(9)]
    for a in range(9):
        for r in range(9):
            s=int(f('t1_status')[a,r])
            if s==1:grid[a][r]=float(f('t1_value')[a,r])
            if s!=2:continue
            st=f('t2_status')[a,r]
            if (st==3).any():continue
            A=[x for x in range(9)if st[x].any()];R=[y for y in range(9)if st[:,y].any()]
            def soft(lg,S):
                e={x:math.exp(float(lg[x]))for x in S};z=sum(e.values());return{x:e[x]/z for x in S}
            po=soft(f('t1_own_logits')[a,r],A);pp=soft(f('t1_peer_logits')[a,r],R)
            rows={x:sum(pp[y]*float(f('t2_value')[a,r,x,y])for y in R)for x in A}
            grid[a][r]=max(rows.values())if maximize else sum(po[x]*rows[x]for x in A)
    kept=tuple(bool(f('t1_status')[:,r].any())for r in range(9))if f('root_peer_requested')else(False,)*9
    return bool(f('root_peer_requested')),kept,tuple(map(float,f('root_peer_logits'))),grid


class Depth2PricerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.root,_,cls.tables,cls.recipe=facts()

    def requests(self,specs,ex):
        requests=[]
        for i,(modes,owner,learner,opponent)in enumerate(specs):
            stream=100+i
            while not self.recipe.selected(v5_contract.DecisionAddress(stream,4+i,owner)):stream+=100
            root=replace(self.root,address=v5_contract.DecisionAddress(stream,4+i,owner))
            start=plans.plan_starts(root,self.tables.source,self.recipe)
            state,public=world_data(root,start.starts[0],modes[0])
            inputs=ex.observe(state,public,j.asarray([owner],j.int32),ex.legal(state)[:,owner]).inputs
            requests.append(batch.Request(start,ROOT_LOGITS,ROOT_LEGAL,inputs,learner,opponent))
        return tuple(requests)

    @staticmethod
    def materialize(specs,missing=()):
        def materialize(root,start,tables,recipe,keys=None):
            index=root.address.frame-4;state,public=world_data(root,start,specs[index][0][start.world])
            absent=(index,start.world)in missing
            return mapping.World(start.world,None if absent else state,None if absent else public,('inert missing',)if absent else(),start.seed,False)
        return materialize

    def price(self,specs,missing=(),params=PARAMS,**kw):
        ex,_=executor();requests=self.requests(specs,ex)
        kw.setdefault('capacity',8);engine=pricer.Depth2Pricer(ex,object(),actor_capacity=3,**kw)
        with patch.object(mapping,'materialize_host',side_effect=self.materialize(specs,missing)),patch.object(batch.v5_model,'forward',side_effect=forward):
            frame=engine.price(requests,self.tables,self.recipe,params)
        return frame,requests,engine,ex

    def check_oracle(self,frame,requests,ex,specs,params=PARAMS,k_own=9,k_opp=9,k1_own=9,k1_opp=9,missing=()):
        oracles={}
        for i,(r,(modes,owner,learner,opponent))in enumerate(zip(requests,specs)):
            order=self.recipe.pricing_order(r.plans.root.address,r.root_logits,r.root_legal)[:k1_own]
            for wi,start in enumerate(r.plans.starts):
                if (i,wi)in missing:
                    self.assertEqual(frame.pre[i,wi],pricer.UNAVAILABLE)
                    for name in('t1_status','t2_status','root_peer_legal','root_peer_logits'):self.assertFalse(getattr(frame,name)[i,wi].any(),name)
                    continue
                self.assertEqual(frame.pre[i,wi],0)
                state,public=world_data(r.plans.root,start,modes[wi])
                o=oracles[i,wi]=oracle(ex,params,state,public,owner,order,learner,opponent,k_own,k_opp,k1_opp)
                for name,value in o.items():
                    if name.startswith('inputs'):continue
                    np.testing.assert_array_equal(getattr(frame,name)[i,wi],value,err_msg=f'{name} root{i} world{wi}')
                self.assertEqual(frame.t2_status[i,wi].shape,(9,9,9,9))
        return oracles

    def test_layout_and_indexing_match_the_two_step_oracle(self):
        specs=(((0,27),0,0,1),((27,0),1,1,0),((0,0),0,2,2))
        frame,requests,engine,ex=self.price(specs,child_block=5)
        self.check_oracle(frame,requests,ex,specs)
        self.assertTrue(all(w['status']=='COMPLETE'for pair in frame.world_reports for w in pair))
        self.assertEqual(engine.last_timing['root_capacity'],4);self.assertEqual(engine.last_timing['device_invocations'],1)
        # Every expanded cell enumerates the full legal product; everything else is zero.
        for i in range(3):
            for wi in range(2):
                for a,r in zip(*np.nonzero(frame.t1_status[i,wi]==2)):
                    own=frame.t1_own_legal[i,wi,a,r];peer=frame.t1_peer_legal[i,wi,a,r]
                    np.testing.assert_array_equal(frame.t2_status[i,wi,a,r]!=0,own[:,None]&peer[None,:])
                self.assertFalse(frame.t2_status[i,wi][frame.t1_status[i,wi]!=2].any())

    def test_top_k_with_ties_takes_the_lower_index(self):
        specs=(((0,0),0,2,2),((0,0),1,0,2),((0,0),0,2,1))
        frame,requests,engine,ex=self.price(specs,k_own=2,k_opp=3,child_block=4)
        self.check_oracle(frame,requests,ex,specs,k_own=2,k_opp=3)
        tied=frame.t2_status[0,0][frame.t1_status[0,0]==2]
        self.assertTrue(len(tied))
        for cell in tied:  # all-tied logits: the two lowest legal own actions {0,1}, the three lowest replies {0,2,3}
            self.assertEqual(set(zip(*np.nonzero(cell))),{(a,r)for a in(0,1)for r in(0,2,3)})
        for i in range(3):
            for wi in range(2):
                for a,r in zip(*np.nonzero(frame.t1_status[i,wi]==2)):
                    self.assertEqual(np.count_nonzero(frame.t2_status[i,wi,a,r]),6)

    def test_waiting_learner_and_waiting_opponent_at_a_child(self):
        specs=(((20,21),0,0,1),((21,20),1,1,0))
        frame,requests,engine,ex=self.price(specs,child_block=7)
        self.check_oracle(frame,requests,ex,specs)
        for i,wi,waiting in((0,0,'own'),(1,1,'own'),(0,1,'peer'),(1,0,'peer')):
            cells=list(zip(*np.nonzero(frame.t1_status[i,wi]==2)));self.assertTrue(cells)
            for a,r in cells:
                requested=frame.t1_own_requested[i,wi,a,r]if waiting=='own'else frame.t1_peer_requested[i,wi,a,r]
                self.assertFalse(requested)
                grid=frame.t2_status[i,wi,a,r]!=0
                if waiting=='own':self.assertFalse(grid[1:].any());self.assertEqual(np.count_nonzero(grid[0]),4)
                else:self.assertFalse(grid[:,1:].any());self.assertEqual(np.count_nonzero(grid[:,0]),4)
        # The waiting seat's engine action is -1: the fake adds a/32 for the learner and max(r,0)/64 for the opponent.
        learner=0
        for i,wi,waiting in((0,0,'own'),(0,1,'peer')):
            r=requests[i];state,public=world_data(r.plans.root,r.plans.starts[wi],specs[i][0][wi])
            seed=r.plans.starts[wi].seed%2**32;chance1=seed%17;chance2=((seed*3+1)%2**32)%17
            for a,rr in zip(*np.nonzero(frame.t1_status[i,wi]==2)):
                child=.25+a/32+rr/64+chance1/1024
                for a2,r2 in zip(*np.nonzero(frame.t2_status[i,wi,a,rr])):
                    own_step=-1/32 if waiting=='own'else a2/32;foe_step=0. if waiting=='peer'else r2/64
                    self.assertEqual(float(frame.t2_value[i,wi,a,rr,a2,r2]),child+own_step+foe_step+chance2/1024+float(PARAMS[learner]['offset']))

    def test_terminal_and_error_propagation(self):
        specs=(((1,2),0,0,1),((7,3),1,1,0),((4,5),0,0,1),((22,23),1,0,1),((24,25),0,1,0),((28,6),1,0,1))
        frame,requests,engine,ex=self.price(specs,child_block=9)
        self.check_oracle(frame,requests,ex,specs)
        s,e,v=frame.t1_status,frame.t1_error,frame.t1_value
        self.assertTrue((s[0,0][s[0,0]!=0]==1).all());self.assertTrue((v[0,0][s[0,0]==1]==1).all())  # win
        self.assertTrue((v[0,1][s[0,1]==1]==0).all())  # draw
        self.assertTrue((e[1,0][s[1,0]!=0]==2).all())  # bad winner
        self.assertEqual(e[1,1,1,1],1);self.assertEqual(s[1,1,1,1],3);self.assertEqual(np.count_nonzero(e[1,1]),1)  # invalid (1,1)
        self.assertTrue((e[2,0][s[2,0]!=0]==3).all());self.assertTrue((e[2,1][s[2,1]!=0]==4).all())
        s2,e2,v2=frame.t2_status,frame.t2_error,frame.t2_value
        for a,r in zip(*np.nonzero(s[3,0]==2)):  # terminal leaves: a2==1 learner (owner) win, r2==3 draw; replies are {0,2,3,4}
            cell=s2[3,0,a,r];self.assertTrue((cell[1,[0,2,3,4]]==1).all());self.assertTrue((cell[[0,1,3,4],3]==1).all())
            self.assertTrue((v2[3,0,a,r,1,[0,2,4]]==1).all());self.assertTrue((v2[3,0,a,r,[0,1,3,4],3]==0).all())
            self.assertTrue((cell[0,[0,2,4]]==4).all());self.assertEqual(cell[0,3],1)
        for a,r in zip(*np.nonzero(s[3,1]==2)):self.assertTrue((e2[3,1,a,r,1][s2[3,1,a,r,1]!=0]==3).all());self.assertFalse(e2[3,1,a,r,0].any())
        for a,r in zip(*np.nonzero(s[4,0]==2)):self.assertEqual(e2[4,0,a,r,1,2],1);self.assertEqual(np.count_nonzero(e2[4,0,a,r]),1)
        for a,r in zip(*np.nonzero(s[4,1]==2)):self.assertTrue(s2[4,1,a,r,3].any());self.assertTrue((e2[4,1,a,r,3][s2[4,1,a,r,3]!=0]==4).all());self.assertFalse(e2[4,1,a,r,0].any())
        for a,r in zip(*np.nonzero(s[5,0]==2)):self.assertTrue((e2[5,0,a,r,0][s2[5,0,a,r,0]!=0]==2).all())
        self.assertFalse(frame.root_peer_requested[5,1]);self.assertFalse(frame.t1_status[5,1][:,1:].any())
        # Errors refuse through both backups and show up in the refusal accounting.
        for i in range(6):
            for fn in(backups.backup_learner_max,backups.backup_average):
                prices=fn(frame,i)
                for wi in range(2):
                    refused=bool((frame.t1_status[i,wi]==3).any()or(frame.t2_status[i,wi]==3).any())
                    self.assertEqual(any(v is None for a,row in enumerate(prices[wi].joint_values)for r,v in enumerate(row)if frame.t1_status[i,wi,a,r]),refused)
            reduced=backups.learner_max_root(frame,i,ROOT_LOGITS,ROOT_LEGAL)
            self.assertEqual(reduced.status,'LABELED'if i==0 else'REFUSED')
        summary=backups.refusal_summary(frame)
        self.assertEqual([x['cause']for x in summary['roots']],[None,'turn1','turn1','turn2','turn2','turn2'])
        self.assertEqual(summary['refused_count'],5);self.assertEqual(summary['turn1_errors_by_code'][1],1)
        self.assertEqual(set(summary['refused_turn2_errors']),{1,2,3,4,5});self.assertGreater(summary['refused_turn2_errors'][3],0)
        self.assertEqual(sum(summary['turn2_errors_by_code'].values()),int(np.count_nonzero(frame.t2_status==3)))

    def test_padding_is_never_enumerated(self):
        specs=(((0,0),0,0,1),((0,27),1,1,0),((22,0),0,0,2));missing=((1,1),)
        frame,requests,engine,ex=self.price(specs,missing=missing,k_own=2,k_opp=3,child_block=5)
        self.check_oracle(frame,requests,ex,specs,k_own=2,k_opp=3,missing=missing)
        t=engine.last_timing
        self.assertEqual((t['root_capacity'],t['padding_root_slots'],t['substituted_world_slots']),(4,1,1))
        self.assertEqual(t['turn2_blocks'],-(-81*8//5)+2*-(-81*2//5))  # archives share history_size, present or not;self.assertEqual(t['turn2_lanes_per_block'],30)
        enumerated=int(np.count_nonzero(frame.t2_status));self.assertEqual(t['turn2_enumerated_leaves'],enumerated)
        self.assertEqual(t['turn2_evaluated_leaf_lanes'],t['turn2_active_blocks']*30);self.assertLess(enumerated,t['turn2_evaluated_leaf_lanes'])
        # Exactly the blocks holding an expanded child ran; the rest were skipped.
        slots,_=engine._blocks(requests,4,PARAMS);expanded=set()
        for i,r in enumerate(requests):
            order=self.recipe.pricing_order(r.plans.root.address,r.root_logits,r.root_legal)
            for wi in range(2):
                replies=[x for x in range(9)if frame.root_peer_legal[i,wi,x]]or[0]
                for a,rr in zip(*np.nonzero(frame.t1_status[i,wi]==2)):expanded.add((2*i+wi)*81+order.index(a)*len(replies)+replies.index(rr))
        self.assertEqual(t['turn2_active_blocks'],sum(any(int(x)in expanded for x in block)for block in slots))
        for i,r in enumerate(requests):
            order=self.recipe.pricing_order(r.plans.root.address,r.root_logits,r.root_legal)
            for wi in range(2):
                if (i,wi)in missing:continue
                replies=2 if frame.root_peer_requested[i,wi]else 1
                self.assertEqual(np.count_nonzero(frame.t1_status[i,wi]),len(order)*replies)
                for a,rr in zip(*np.nonzero(frame.t1_status[i,wi]==2)):
                    own=frame.t1_own_legal[i,wi,a,rr];peer=frame.t1_peer_legal[i,wi,a,rr]
                    cell=frame.t2_status[i,wi,a,rr]!=0
                    self.assertEqual(np.count_nonzero(cell),min(own.sum(),2)*min(peer.sum(),3));self.assertFalse((cell&~(own[:,None]&peer[None,:])).any())
        self.assertEqual(frame.world_reports[1][1]['reason'],'physical-materialization:inert missing')

    def test_child_block_invariance(self):
        specs=(((0,22),0,0,1),((27,21),1,1,0),((20,0),0,2,1))
        frames=[self.price(specs,child_block=cb)[0]for cb in(1,5,81,500)]
        for other in frames[1:]:
            for name in pricer.Depth2Frame.__dataclass_fields__:
                if name in('world_reports','timing')or getattr(frames[0],name)is None:continue
                np.testing.assert_array_equal(getattr(frames[0],name),getattr(other,name),err_msg=name)
        self.assertEqual([w['status']for pair in frames[0].world_reports for w in pair],[w['status']for pair in frames[-1].world_reports for w in pair])

    def test_turn1_equals_frame_pricer_vectorized(self):
        specs=(((0,0),0,0,1),((1,2),1,1,0),((3,4),0,0,2),((5,6),1,2,1),((7,27),0,1,1))
        frame,requests,engine,ex=self.price(specs)
        _,mapped=executor();d3=batch.FramePricer(mapped,object(),capacity=8,block_rows=1,actor_capacity=3,vectorized=True,host_prep=True)
        with patch.object(mapping,'materialize_host',side_effect=self.materialize(specs)),patch.object(batch.v5_model,'forward',side_effect=forward):
            results=d3.price(requests,self.tables,self.recipe,PARAMS)
        for i,result in enumerate(results):
            for wi,report in enumerate(result.world_reports):
                mine=frame.world_reports[i][wi]
                self.assertEqual((mine['kernel_seed'],mine['posterior_seed']),(report['kernel_seed'],report['posterior_seed']))
                if report['joint_values']is None:
                    self.assertNotEqual(frame.pre[i,wi],0);self.assertEqual(mine['reason'],report['reason']);continue
                self.assertEqual(frame.pre[i,wi],0)
                self.assertEqual((bool(frame.root_peer_requested[i,wi]),tuple(map(bool,frame.root_peer_legal[i,wi])),tuple(map(float,frame.root_peer_logits[i,wi]))),
                    (report['opponent_requested'],report['reply_legal'],report['reply_logits']))
                self.assertEqual(int(np.count_nonzero(frame.t1_status[i,wi])),report['required_pairs'])
                for a in range(9):
                    for r in range(9):
                        value=report['joint_values'][a][r]
                        if value is None:self.assertIn(frame.t1_status[i,wi,a,r],(0,3))
                        else:
                            self.assertIn(frame.t1_status[i,wi,a,r],(1,2))
                            self.assertEqual(frame.t1_value[i,wi,a,r],np.float32(value))
                errors={(a,r):reason for a,r,reason in report.get('errors',())}
                self.assertEqual({(a,r):reason for a,r,reason in mine.get('errors',())},errors)

    def test_backups_match_brute_force(self):
        specs=(((0,0),0,0,1),((27,21),1,1,0),((20,0),0,2,1),((0,0),1,0,2))
        frame,requests,engine,ex=self.price(specs,k_own=2,k_opp=3)
        for i in range(len(specs)):
            for fn,maximize in((backups.backup_learner_max,True),(backups.backup_average,False)):
                prices=fn(frame,i)
                for wi in range(2):
                    requested,kept,logits,grid=reference_backup(frame,i,wi,maximize)
                    self.assertEqual((prices[wi].opponent_requested,prices[wi].opponent_legal,prices[wi].opponent_logits),(requested,kept,logits))
                    for a in range(9):
                        for r in range(9):
                            if grid[a][r]is None:self.assertIsNone(prices[wi].joint_values[a][r])
                            else:self.assertAlmostEqual(prices[wi].joint_values[a][r],grid[a][r],places=12)
            for fn in(backups.learner_max_root,backups.average_root):
                reduced=fn(frame,i,ROOT_LOGITS,ROOT_LEGAL)
                self.assertEqual(reduced.status,'LABELED');self.assertEqual(reduced.improvement.worlds,2)
            # Learner-max dominates the prior average cell by cell.
            top=backups.backup_learner_max(frame,i);avg=backups.backup_average(frame,i)
            for wi in range(2):
                for a in range(9):
                    for r in range(9):
                        if top[wi].joint_values[a][r]is not None:self.assertGreaterEqual(top[wi].joint_values[a][r],avg[wi].joint_values[a][r]-1e-12)

    def test_learner_max_ties_waiting_seat_and_top_k_renormalization_synthetic(self):
        frame=synthetic();f=lambda name:getattr(frame,name)[0,0]
        frame.root_peer_requested[0,0]=True;f('root_peer_legal')[[0,1]]=True;f('root_peer_logits')[1]=.5
        f('t1_status')[0,0]=2;f('t1_status')[0,1]=1;f('t1_value')[0,1]=-1;f('t1_status')[1,0]=2;f('t1_status')[1,1]=2;f('t1_status')[2,:2]=2
        # Cell (0,0): learner legal {0,1,3,4} but only {1,3} enumerated (top-2); opponent {0,2,3} enumerated of {0,2,3,4}.
        f('t1_own_legal')[0,0][[0,1,3,4]]=True;f('t1_own_logits')[0,0][[0,1,3,4]]=[5.,1.,2.,4.]
        f('t1_peer_legal')[0,0][[0,2,3,4]]=True;f('t1_peer_logits')[0,0][[0,2,3,4]]=[0.,0.,0.,9.]
        for a2 in(1,3):f('t2_status')[0,0,a2,[0,2,3]]=4
        f('t2_value')[0,0,1,0]=.75;f('t2_value')[0,0,3,3]=.75
        # Rows 1 and 3 are both .75/3 (uniform over three enumerated replies): a tie, so the lower index 1 wins.
        # Cell (1,0): waiting learner, row 0 only; opponent enumerated {2,3} with equal logits.
        f('t1_peer_requested')[1,0]=True;f('t2_status')[1,0,0,[2,3]]=4;f('t2_value')[1,0,0,[2,3]]=[.5,-.25]
        # Cell (1,1): waiting opponent, column 0 only; learner rows {0,4}, row 4 strictly better.
        f('t1_own_requested')[1,1]=True;f('t2_status')[1,1,[0,4],0]=4;f('t2_value')[1,1,[0,4],0]=[.125,.375];f('t1_own_logits')[1,1][[0,4]]=[3.,-3.]
        # Cell (2,0): one turn-2 error refuses the cell; (2,1): a terminal leaf counts as a value.
        f('t2_status')[2,0,0,0]=4;f('t2_status')[2,0,0,2]=3;f('t2_error')[2,0,0,2]=1
        f('t2_status')[2,1,0,0]=1;f('t2_value')[2,1,0,0]=1.
        top,=backups.backup_learner_max(frame,0)[:1];avg,=backups.backup_average(frame,0)[:1]
        self.assertAlmostEqual(top.joint_values[0][0],.25,places=12)
        choice=backups.learner_max_choice(frame,0,0);self.assertEqual(choice[0,0],1);self.assertEqual(choice[1,0],0);self.assertEqual(choice[1,1],4);self.assertEqual(choice[2,0],-1)
        pi1=math.exp(1.)/(math.exp(1.)+math.exp(2.))
        self.assertAlmostEqual(avg.joint_values[0][0],pi1*.25+(1-pi1)*.25,places=12)
        self.assertAlmostEqual(top.joint_values[1][0],.125,places=12);self.assertAlmostEqual(avg.joint_values[1][0],.125,places=12)
        self.assertAlmostEqual(top.joint_values[1][1],.375,places=12)
        p0=math.exp(3.)/(math.exp(3.)+math.exp(-3.));self.assertAlmostEqual(avg.joint_values[1][1],p0*.125+(1-p0)*.375,places=12)
        self.assertEqual(top.joint_values[0][1],-1.);self.assertIsNone(top.joint_values[2][0]);self.assertEqual(top.joint_values[2][1],1.)
        self.assertEqual(top.opponent_legal,(True,True)+(False,)*7)
        self.assertEqual(backups.learner_max_root(frame,0,ROOT_LOGITS,(True,True,True)+(False,)*6).status,'REFUSED')  # world 1 priced nothing
        frame.pre[0,1]=9;self.assertIsNone(backups.backup_learner_max(frame,0)[1])
        self.assertEqual(backups.refusal_summary(frame)['roots'][0]['cause'],'pre')

    def test_refusal_summary_synthetic(self):
        frame=synthetic(5)
        frame.pre[1,1]=9;frame.pre[2,0]=pricer.UNAVAILABLE
        frame.t1_status[3,0,0,0]=3;frame.t1_error[3,0,0,0]=3;frame.t1_status[3,0,1,0]=2
        frame.t1_status[4,1,0,0]=2;frame.t2_status[4,1,0,0,0,0]=3;frame.t2_error[4,1,0,0,0,0]=1;frame.t2_status[4,1,0,0,1,0]=3;frame.t2_error[4,1,0,0,1,0]=4
        frame.t2_status[4,1,0,0,2,0]=4;frame.t2_status[0,0,0,0]=4
        summary=backups.refusal_summary(frame)
        self.assertEqual([x['cause']for x in summary['roots']],[None,'pre','materialization','turn1','turn2'])
        self.assertEqual(summary['roots'][1]['pre'],(0,9));self.assertEqual(summary['refused_count'],4)
        self.assertEqual(summary['refused_by_cause'],{'materialization':1,'pre':1,'turn1':1,'turn2':1})
        self.assertEqual(summary['turn1_errors_by_code'],{1:0,2:0,3:1,4:0});self.assertEqual(summary['turn2_errors_by_code'],{1:1,2:0,3:0,4:1})
        self.assertEqual(summary['refused_turn2_errors'],{1:0,2:0,3:0,4:2})

    def test_digests_match_the_oracle_leaf_inputs(self):
        specs=(((0,27),0,0,1),((20,22),1,1,0))
        frame,requests,engine,ex=self.price(specs,digest=True,child_block=6)
        oracles=self.check_oracle(frame,requests,ex,specs)
        self.assertEqual(frame.t2_digest.shape,(2,2,9,9,9,9,2));self.assertEqual(frame.t1_digest_own.shape,(2,2,9,9,2))
        for (i,wi),o in oracles.items():
            np.testing.assert_array_equal(frame.t1_digest_own[i,wi].any(-1),frame.t1_status[i,wi]!=0)
            np.testing.assert_array_equal(frame.t2_digest[i,wi].any(-1),frame.t2_status[i,wi]!=0)
            for (a,r),(own,peer)in o['inputs1'].items():
                np.testing.assert_array_equal(frame.t1_digest_own[i,wi,a,r],np.asarray(depth2_digest.digest(own)))
                np.testing.assert_array_equal(frame.t1_digest_peer[i,wi,a,r],np.asarray(depth2_digest.digest(peer)))
            for (a,r,a2,r2),inputs in o['inputs2'].items():
                np.testing.assert_array_equal(frame.t2_digest[i,wi,a,r,a2,r2],np.asarray(depth2_digest.digest(inputs)))
        plain=self.price(specs,child_block=6)[0];self.assertIsNone(plain.t2_digest)
        np.testing.assert_array_equal(plain.t2_value,frame.t2_value)

    def test_turn1_widths_prefix_and_top_replies(self):
        specs=(((0,0),0,0,1),((0,6),1,1,0))
        frame,requests,engine,ex=self.price(specs,k1_own=2,k1_opp=1,k_own=3,k_opp=2)
        self.check_oracle(frame,requests,ex,specs,k_own=3,k_opp=2,k1_own=2,k1_opp=1)
        self.assertEqual(np.count_nonzero(frame.t1_status[0,0]),2);self.assertEqual(np.count_nonzero(frame.t1_status[1,1]),2)
        prices=backups.backup_learner_max(frame,0)
        self.assertEqual(sum(prices[0].opponent_legal),1);self.assertEqual(backups.learner_max_root(frame,0,ROOT_LOGITS,ROOT_LEGAL).status,'REFUSED')

    def test_single_actor_bank_runs_without_stacking(self):
        specs=(((0,22),0,0,0),((27,0),1,0,0))
        frame,requests,engine,ex=self.price(specs,params=PARAMS[:1],capacity=2,child_block=3)
        self.check_oracle(frame,requests,ex,specs,params=PARAMS[:1])
        self.assertEqual(engine.last_timing['turn2_blocks'],-(-81*4//3))

    def test_empty_and_fail_closed(self):
        ex,_=executor()
        for kw in(dict(k_own=0),dict(k_opp=10),dict(k1_own=0),dict(k1_opp=10),dict(child_block=0),dict(capacity=0),dict(digest=1)):
            with self.assertRaises(ValueError):pricer.Depth2Pricer(ex,None,actor_capacity=2,**kw)
        with self.assertRaises(ValueError):pricer.make_depth2_kernel(ex,None,params=PARAMS,capacity=1,k_own=11)
        engine=pricer.Depth2Pricer(ex,None,capacity=1,actor_capacity=1)
        with self.assertRaises(ValueError):engine.bind(({},{}))
        empty=engine.price((),self.tables,self.recipe,PARAMS[:1])
        self.assertEqual(empty.pre.shape,(0,2));self.assertEqual(empty.t2_status.shape,(0,2,9,9,9,9))
        with self.assertRaises(ValueError):engine.price(self.requests((((0,0),0,0,0),((0,0),1,0,0)),ex),self.tables,self.recipe,PARAMS[:1])


if __name__=='__main__':unittest.main()
