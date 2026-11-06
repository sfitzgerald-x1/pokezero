"""Depth-N pricer on inert worlds: N=2 bit-equality with Depth2Pricer, an N-step pure-Python oracle, both selection modes, backups."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import device_public_teacher as serial,device_public_teacher_batch as batch,device_public_world_state as mapping
from gpu_engine import device_depth2_pricer as d2,device_depthn_pricer as dn,depth2_backups,depthn_backups as backups,depth2_digest
from tests.test_gpu_engine_device_public_teacher import facts
from tests import test_gpu_engine_depth2_pricer as T

PARAMS=T.PARAMS;forward=T.forward;usable=T.usable;world_data=T.world_data;ROOT_LOGITS=T.ROOT_LOGITS;ROOT_LEGAL=T.ROOT_LEGAL
T1_NAMES=('t1_status','t1_value','t1_error','t1_own_requested','t1_own_legal','t1_own_logits','t1_peer_requested','t1_peer_legal','t1_peer_logits')


def deep_executor():
    """The depth-2 fake with modes for plies past turn 2 (a state's `turn` is its ply: a ply-2 node has turn 2).

    40 the learner waits at ply-2 nodes only; 41 the opponent waits there only; 43 the opponent's view errs at ply-2
    nodes (code 5); 44 its prior there is nonfinite (code 6); 46 the (1,0) pair is invalid at the ply-3 step only;
    47 the ply-3 step is terminal (a learner win) when the learner plays 1.
    """
    base,_=T.executor()
    def legal(state):
        m=base.legal(state);t=state.turn[:,None];mode=state.mode[:,None];owner=state.owner[0]
        return m.at[:,owner].set(m[:,owner]&~((mode==40)&(t==2))).at[:,1-owner].set(m[:,1-owner]&~((mode==41)&(t==2)))
    def observe(state,public,owner,mask):
        leaf=base.observe(state,public,owner,mask);foe=(owner!=state.owner)&(state.turn==2)
        error=leaf.observation.error|j.where((state.mode==43)&foe,1,0)
        features=leaf.inputs['numeric_features'].at[:,0,0,2].add(j.where((state.mode==44)&foe,j.nan,0.))
        return leaf._replace(observation=leaf.observation._replace(error=error),inputs={'numeric_features':features})
    def advance(state,public,actions,chance):
        after=base.advance(state,public,actions,chance);a=actions[:,state.owner[0]];r=actions[:,1-state.owner[0]];third=state.turn==2
        invalid=after.step.invalid_actions|((state.mode==46)&third&(a==1)&(r==0))
        terminal=(state.mode==47)&third&(a==1);s=after.step.state
        s=s._replace(done=s.done|terminal,winner=j.where(terminal,state.owner,s.winner))
        return after._replace(step=after.step._replace(state=s,invalid_actions=invalid))
    return serial.Executor(jax.jit(legal),jax.jit(observe),base.dispatch,base.sample,jax.jit(advance))


def oracle(ex,params,state,public,owner,order,learner,opponent,ks,k1_opp,maps=None):
    """N plies by hand, one row at a time, in the dense id layout: the contract's semantics, independent of the engine.

    `ks` holds (k_own, k_opp) per ply 2..N; `maps`, when given, holds {ply: (own [M,k], opp [M,k])} slot -> action maps
    by parent id (the prior rule applies elsewhere). Nodes are dicts by id per ply; leaves (status, value, error) by
    (parent id, own slot, opponent slot); encoder inputs are kept for digests.
    """
    N=len(ks)+1;Z=lambda *s,d=np.float32:np.zeros(s,d)
    o=dict(t1_status=Z(9,9,d=np.int8),t1_value=Z(9,9),t1_error=Z(9,9,d=np.int8),t1_own_requested=Z(9,9,d=bool),t1_own_legal=Z(9,9,9,d=bool),
        t1_own_logits=Z(9,9,9),t1_peer_requested=Z(9,9,d=bool),t1_peer_legal=Z(9,9,9,d=bool),t1_peer_logits=Z(9,9,9),
        t1_own_actions=np.full((9,9,ks[0][0]),-1,np.int32),t1_opp_actions=np.full((9,9,ks[0][1]),-1,np.int32),
        nodes={p:{}for p in range(2,N)},leaves={},inputs1={},node_inputs={},leaf_inputs={})
    seat=j.asarray([owner],j.int32);masks=ex.legal(state)
    own=ex.observe(state,public,seat,masks[:,owner]);peer=ex.observe(state,public,1-seat,masks[:,1-owner])
    act=lambda view,st,s,action:ex.dispatch(view.observation,st,s,j.asarray([action],j.int32))
    legal=np.asarray(peer.observation.legal_action_mask[0]);requested=bool(legal.any())
    pick=lambda legal,lg,k,req:sorted([i for i in range(9)if legal[i]],key=lambda i:(-float(lg[i]),i))[:k]if req else[0]
    if requested:lg=np.asarray(forward(params[opponent],peer.inputs).policy_logits[0]);replies=sorted(pick(legal,lg,k1_opp,True))
    else:lg=np.zeros(9,np.float32);replies=[0]
    o.update(root_peer_requested=requested,root_peer_legal=legal,root_peer_logits=lg)
    outcome=lambda winner:0. if winner==2 else(1. if winner==owner else-1.)
    def expand(p,pid,st,tr,own_req,opp_req,own_legal,opp_legal,own_lg,opp_lg,vo,vp):
        k_own,k_opp=ks[p-2];K=k_own*k_opp
        if maps is not None and p in maps:own_map=[int(x)for x in maps[p][0][pid]];opp_map=[int(x)for x in maps[p][1][pid]]
        else:
            own_map=pick(own_legal,own_lg,k_own,own_req);own_map+=[-1]*(k_own-len(own_map))
            opp_map=pick(opp_legal,opp_lg,k_opp,opp_req);opp_map+=[-1]*(k_opp-len(opp_map))
        chance=ex.sample(st)
        for i,a in enumerate(own_map):
            if a<0:continue
            for x,r in enumerate(opp_map):
                if r<0:continue
                acts=j.full((1,2),-1,j.int32)
                if own_req:acts=acts.at[:,owner].set(act(vo,st,seat,a))
                if opp_req:acts=acts.at[:,1-owner].set(act(vp,st,1-seat,r))
                after=ex.advance(st,tr,acts,chance);cs=after.step.state;ct=after.trackers;cid=pid*K+i*k_opp+x
                invalid=bool(after.step.invalid_actions[0]);done=bool(cs.done[0]);winner=int(cs.winner[0])
                if p==N:
                    if invalid:rec=(3,0.,1)
                    elif done:rec=(1,outcome(winner),0)if winner in(0,1,2)else(3,0.,2)
                    else:
                        lm=ex.legal(cs);lv=ex.observe(cs,ct,seat,lm[:,owner]);pl=forward(params[learner],lv.inputs);o['leaf_inputs'][pid,i,x]=lv.inputs
                        rec=(3,0.,3)if int(lv.observation.error[0])else(3,0.,4)if not usable(pl)else(4,float(pl.value[0]),0)
                    o['leaves'][pid,i,x]=rec;continue
                node=dict(status=0,value=0.,error=0,own_requested=False,opp_requested=False,own_legal=np.zeros(9,bool),opp_legal=np.zeros(9,bool),
                    own_logits=np.zeros(9,np.float32),opp_logits=np.zeros(9,np.float32),own_actions=np.full(ks[p-1][0],-1,np.int32),opp_actions=np.full(ks[p-1][1],-1,np.int32))
                o['nodes'][p][cid]=node
                if invalid:node.update(status=3,error=1);continue
                if done:
                    node.update(status=1,value=outcome(winner))if winner in(0,1,2)else node.update(status=3,error=2);continue
                cm=ex.legal(cs);cvo=ex.observe(cs,ct,seat,cm[:,owner]);cvp=ex.observe(cs,ct,1-seat,cm[:,1-owner])
                po=forward(params[learner],cvo.inputs);pp=forward(params[opponent],cvp.inputs)
                c_own_req=bool(np.asarray(cm[0,owner]).any());c_opp_req=bool(np.asarray(cm[0,1-owner]).any())
                c_own_legal=np.asarray(cvo.observation.legal_action_mask[0]);c_opp_legal=np.asarray(cvp.observation.legal_action_mask[0])
                if int(cvo.observation.error[0]):node.update(status=3,error=3);continue
                if not usable(po):node.update(status=3,error=4);continue
                c_own_lg=np.asarray(po.policy_logits[0])if c_own_req else np.zeros(9,np.float32)
                node.update(value=float(po.value[0]),own_requested=c_own_req,opp_requested=c_opp_req,own_legal=c_own_legal,opp_legal=c_opp_legal,own_logits=c_own_lg)
                if c_opp_req and int(cvp.observation.error[0]):node.update(status=3,error=5);continue
                if c_opp_req and not usable(pp):node.update(status=3,error=6);continue
                c_opp_lg=np.asarray(pp.policy_logits[0])if c_opp_req else np.zeros(9,np.float32)
                node.update(status=2,opp_logits=c_opp_lg);o['node_inputs'][p,cid]=(cvo.inputs,cvp.inputs)
                own_m,opp_m=expand(p+1,cid,cs,ct,c_own_req,c_opp_req,c_own_legal,c_opp_legal,c_own_lg,c_opp_lg,cvo,cvp)
                node['own_actions'][:]=own_m;node['opp_actions'][:]=opp_m
        return own_map,opp_map
    chance=ex.sample(state)
    for a1 in order:
        for r1 in replies:
            actions=j.full((1,2),-1,j.int32).at[:,owner].set(act(own,state,seat,a1))
            if requested:actions=actions.at[:,1-owner].set(act(peer,state,1-seat,r1))
            after=ex.advance(state,public,actions,chance);child=after.step.state;trackers=after.trackers
            invalid=bool(after.step.invalid_actions[0]);done=bool(child.done[0]);winner=int(child.winner[0])
            if invalid:o['t1_status'][a1,r1]=3;o['t1_error'][a1,r1]=1;continue
            if done:
                if winner in(0,1,2):o['t1_status'][a1,r1]=1;o['t1_value'][a1,r1]=outcome(winner)
                else:o['t1_status'][a1,r1]=3;o['t1_error'][a1,r1]=2
                continue
            cm=ex.legal(child);vo=ex.observe(child,trackers,seat,cm[:,owner]);vp=ex.observe(child,trackers,1-seat,cm[:,1-owner])
            po=forward(params[learner],vo.inputs);pp=forward(params[opponent],vp.inputs)
            own_req=bool(np.asarray(cm[0,owner]).any());peer_req=bool(np.asarray(cm[0,1-owner]).any())
            own_legal=np.asarray(vo.observation.legal_action_mask[0]);peer_legal=np.asarray(vp.observation.legal_action_mask[0])
            if int(vo.observation.error[0]):o['t1_status'][a1,r1]=3;o['t1_error'][a1,r1]=3;continue
            if not usable(po):o['t1_status'][a1,r1]=3;o['t1_error'][a1,r1]=4;continue
            o['t1_value'][a1,r1]=float(po.value[0]);own_lg=np.asarray(po.policy_logits[0])if own_req else np.zeros(9,np.float32)
            o['t1_own_requested'][a1,r1]=own_req;o['t1_own_legal'][a1,r1]=own_legal;o['t1_own_logits'][a1,r1]=own_lg
            o['t1_peer_requested'][a1,r1]=peer_req;o['t1_peer_legal'][a1,r1]=peer_legal
            if peer_req and int(vp.observation.error[0]):o['t1_status'][a1,r1]=3;o['t1_error'][a1,r1]=5;continue
            if peer_req and not usable(pp):o['t1_status'][a1,r1]=3;o['t1_error'][a1,r1]=6;continue
            o['t1_status'][a1,r1]=2;o['inputs1'][a1,r1]=(vo.inputs,vp.inputs)
            peer_lg=np.asarray(pp.policy_logits[0])if peer_req else np.zeros(9,np.float32);o['t1_peer_logits'][a1,r1]=peer_lg
            own_m,opp_m=expand(2,a1*9+r1,child,trackers,own_req,peer_req,own_legal,peer_legal,own_lg,peer_lg,vo,vp)
            o['t1_own_actions'][a1,r1]=own_m;o['t1_opp_actions'][a1,r1]=opp_m
    return o


def brute_value(frame,r,w,ply,node,rule):
    """Recursive backup from the frame arrays, written without the backups module; rule(rows: {own slot: q}, pi_own) -> value."""
    N=frame.depth
    if ply==1:
        a,b=divmod(node,9);f=lambda x:x[r,w,a,b]
        status,value,own_map,opp_map,own_lg,opp_lg=f(frame.t1_status),f(frame.t1_value),f(frame.t1_own_actions),f(frame.t1_opp_actions),f(frame.t1_own_logits),f(frame.t1_peer_logits)
    else:
        nd=frame.nodes[ply-2];f=lambda x:x[r,w,node]
        status,value,own_map,opp_map,own_lg,opp_lg=f(nd.status),f(nd.value),f(nd.own_actions),f(nd.opp_actions),f(nd.own_logits),f(nd.opp_logits)
    if status==1:return float(value)
    if status!=2:return None
    k_own,k_opp=frame.k_own[ply-1],frame.k_opp[ply-1];K=k_own*k_opp
    I=[i for i in range(k_own)if own_map[i]>=0];X=[x for x in range(k_opp)if opp_map[x]>=0]
    if not I or not X:return None
    def soft(lg,S):
        e={s:math.exp(float(lg[s]))for s in S};z=sum(e.values());return{s:e[s]/z for s in S}
    po=soft(own_lg,[int(own_map[i])for i in I]);pp=soft(opp_lg,[int(opp_map[x])for x in X])
    vals={}
    for i in I:
        for x in X:
            if ply+1==N:
                s=int(frame.leaf_status[r,w,node,i,x]);v=float(frame.leaf_value[r,w,node,i,x])if s in(1,4)else None
            else:v=brute_value(frame,r,w,ply+1,node*K+i*k_opp+x,rule)
            if v is None:return None
            vals[i,x]=v
    rows={i:sum(pp[int(opp_map[x])]*vals[i,x]for x in X)for i in I}
    return rule(rows,{i:po[int(own_map[i])]for i in I})


MAXIMIZE=lambda rows,po:max(rows.values())
AVERAGE=lambda rows,po:sum(po[i]*q for i,q in rows.items())
MINIMIZE=lambda rows,po:min(rows.values())


def pessimistic(grid,pi_own,pi_opp):
    """A solver hook for the test: the worst own slot's opponent-averaged value."""
    return min(math.fsum(p*v for p,v in zip(pi_opp,row))for row in grid)


class DepthNPricerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.root,_,cls.tables,cls.recipe=facts()

    def requests(self,specs,ex):return T.Depth2PricerTests.requests(self,specs,ex)

    def price(self,specs,*,depth,k_own,k_opp,missing=(),params=PARAMS,ex=None,selection_maps=None,engine=None,**kw):
        if ex is None:ex=deep_executor()
        requests=self.requests(specs,ex)
        kw.setdefault('capacity',8);kw.setdefault('block_lanes',40)
        if engine is None:engine=dn.DepthNPricer(ex,object(),depth=depth,k_own=k_own,k_opp=k_opp,actor_capacity=3,**kw)
        with patch.object(mapping,'materialize_host',side_effect=T.Depth2PricerTests.materialize(specs,missing)),patch.object(batch.v5_model,'forward',side_effect=forward):
            frame=engine.price(requests,self.tables,self.recipe,params,selection_maps=selection_maps)
        return frame,requests,engine,ex

    def depth2(self,specs,requests,ex,**kw):
        kw.setdefault('capacity',8);engine=d2.Depth2Pricer(ex,object(),actor_capacity=3,**kw)
        with patch.object(mapping,'materialize_host',side_effect=T.Depth2PricerTests.materialize(specs)),patch.object(batch.v5_model,'forward',side_effect=forward):
            return engine.price(requests,self.tables,self.recipe,PARAMS)

    def check_oracle(self,frame,requests,ex,specs,ks,k1_own=9,k1_opp=9,missing=(),maps=None,params=PARAMS):
        N=len(ks)+1;oracles={}
        for p in range(2,N):  # the dense parent pointers are the id formula
            nd=frame.nodes[p-2];K=ks[p-2][0]*ks[p-2][1];M=nd.status.shape[2];ids=np.arange(M)
            np.testing.assert_array_equal(nd.parent,ids//K);np.testing.assert_array_equal(nd.parent_own_slot,(ids%K)//ks[p-2][1]);np.testing.assert_array_equal(nd.parent_opp_slot,ids%ks[p-2][1])
            self.assertEqual(nd.ply,p)
        for i,(r,(modes,owner,learner,opponent))in enumerate(zip(requests,specs)):
            order=self.recipe.pricing_order(r.plans.root.address,r.root_logits,r.root_legal)[:k1_own]
            for wi,start in enumerate(r.plans.starts):
                if (i,wi)in missing:
                    self.assertEqual(frame.pre[i,wi],dn.UNAVAILABLE)
                    for name in('t1_status','leaf_status','root_peer_legal','root_peer_logits'):self.assertFalse(getattr(frame,name)[i,wi].any(),name)
                    for nd in frame.nodes:self.assertFalse(nd.status[i,wi].any())
                    continue
                self.assertEqual(frame.pre[i,wi],0)
                state,public=world_data(r.plans.root,start,modes[wi])
                m=None if maps is None else{p:(np.asarray(own[i,wi]).reshape(-1,ks[p-2][0]),np.asarray(opp[i,wi]).reshape(-1,ks[p-2][1]))for p,(own,opp)in maps.items()}
                o=oracles[i,wi]=oracle(ex,params,state,public,owner,order,learner,opponent,ks,k1_opp,m)
                for name in T1_NAMES+('root_peer_requested','root_peer_legal','root_peer_logits','t1_own_actions','t1_opp_actions'):
                    np.testing.assert_array_equal(getattr(frame,name)[i,wi],o[name],err_msg=f'{name} root{i} world{wi}')
                for p in range(2,N):
                    nd=frame.nodes[p-2];M=nd.status.shape[2]
                    dense={name:np.zeros((M,)+np.shape(getattr(nd,name)[i,wi][0]),getattr(nd,name).dtype)for name in dn.NODE_FIELDS+('own_actions','opp_actions')}
                    dense['own_actions'][:]=-1;dense['opp_actions'][:]=-1
                    for cid,node in o['nodes'][p].items():
                        for name in dense:dense[name][cid]=node[name]
                    for name,value in dense.items():np.testing.assert_array_equal(getattr(nd,name)[i,wi],value,err_msg=f'ply{p}.{name} root{i} world{wi}')
                k_own,k_opp=ks[-1];M=frame.leaf_status.shape[2]
                leaves={name:np.zeros((M,k_own,k_opp),dtype)for name,dtype in(('status',np.int8),('value',np.float32),('error',np.int8))}
                for (pid,a,b),(s,v,e)in o['leaves'].items():leaves['status'][pid,a,b]=s;leaves['value'][pid,a,b]=v;leaves['error'][pid,a,b]=e
                for name,value in leaves.items():np.testing.assert_array_equal(getattr(frame,'leaf_'+name)[i,wi],value,err_msg=f'leaf_{name} root{i} world{wi}')
        return oracles

    def assert_frames_equal(self,a,b,msg=''):
        A=dict(dn.arrays(a));B=dict(dn.arrays(b));self.assertEqual(set(A),set(B))
        for name in A:
            self.assertEqual((A[name].shape,A[name].dtype),(B[name].shape,B[name].dtype),name)
            np.testing.assert_array_equal(A[name],B[name],err_msg=f'{name} {msg}')

    # (a) N=2 equivalence -------------------------------------------------------------------------------------------
    def test_n2_as_depth2_frame_is_bit_equal_to_depth2_pricer(self):
        specs=(((0,27),0,0,1),((27,21),1,1,0),((22,23),0,2,2),((24,25),1,0,2),((29,30),0,1,1),((20,6),1,2,0))
        for ex,widths,extra in((T.executor()[0],(9,9),dict()),(T.executor()[0],(2,3),dict()),(T.permuting_executor(),(3,2),dict()),(T.executor()[0],(9,9),dict(digest=True))):
            k_own,k_opp=widths
            frame,requests,engine,_=self.price(specs,depth=2,k_own=(k_own,),k_opp=(k_opp,),ex=ex,block_lanes=5*k_own*k_opp,**extra)
            reference=self.depth2(specs,requests,ex,k_own=k_own,k_opp=k_opp,child_block=5,**extra)
            mine=dn.as_depth2_frame(frame)
            for name in d2.Depth2Frame.__dataclass_fields__:
                if name in('world_reports','timing')or getattr(reference,name)is None:continue
                a,b=getattr(reference,name),getattr(mine,name)
                self.assertEqual((a.shape,a.dtype),(b.shape,b.dtype),name);np.testing.assert_array_equal(a,b,err_msg=f'{name} k={widths}')
            self.assertEqual([(w['status'],w['reason'],w['turn2_required'],w['turn2_unknown'])for pair in reference.world_reports for w in pair],
                [(w['status'],w['reason'],w['turn2_required'],w['turn2_unknown'])for pair in mine.world_reports for w in pair])
            for i in range(len(specs)):  # the recursive backup reproduces the depth-2 backups number for number
                for fn2,fnN in((depth2_backups.backup_learner_max,backups.backup_learner_max),(depth2_backups.backup_average,backups.backup_average)):
                    for a,b in zip(fn2(reference,i),fnN(frame,i)):
                        if a is None:self.assertIsNone(b)
                        else:self.assertEqual((a.opponent_requested,a.opponent_legal,a.opponent_logits,a.joint_values),(b.opponent_requested,b.opponent_legal,b.opponent_logits,b.joint_values))
                self.assertEqual(depth2_backups.learner_max_root(reference,i,ROOT_LOGITS,ROOT_LEGAL),backups.learner_max_root(frame,i,ROOT_LOGITS,ROOT_LEGAL))
            self.assertEqual(frame.nodes,());self.assertEqual(frame.leaf_status.shape,(6,2,81,k_own,k_opp))
        with self.assertRaises(ValueError):dn.as_depth2_frame(self.price(specs[:1],depth=3,k_own=(2,2),k_opp=(2,2))[0])

    # (b) N=3 against the three-step oracle -------------------------------------------------------------------------
    def test_n3_matches_the_three_step_oracle(self):
        specs=(((0,27),0,0,1),((27,0),1,1,0),((0,0),0,2,2))
        frame,requests,engine,ex=self.price(specs,depth=3,k_own=(2,3),k_opp=(3,2),block_lanes=40)
        self.check_oracle(frame,requests,ex,specs,((2,3),(3,2)))
        self.assertTrue(all(w['status']=='COMPLETE'for pair in frame.world_reports for w in pair))
        t=engine.last_timing;self.assertEqual((t['root_capacity'],t['depth'],len(t['plies'])),(4,3,2))
        self.assertEqual(t['plies'][0]['nodes_per_world'],81*6);self.assertEqual(t['plies'][1]['nodes_per_world'],81*36)
        self.assertEqual([k for k in t['programs']if k[0]in('expand','leaf')],[('expand',6,2,3),('leaf',6,3,2)])
        for i in range(3):self.assertEqual(backups.learner_max_root(frame,i,ROOT_LOGITS,ROOT_LEGAL).status,'LABELED')

    def test_n3_full_width_and_n4_match_the_oracle(self):
        specs=(((0,27),1,1,0),)
        frame,requests,engine,ex=self.price(specs,depth=3,k_own=(9,9),k_opp=(9,9),block_lanes=2187)
        self.check_oracle(frame,requests,ex,specs,((9,9),(9,9)))
        parents=int(np.count_nonzero(frame.nodes[0].status==2));self.assertEqual(engine.last_timing['plies'][1]['blocks'],-(-parents//27))  # B = 2187 // 81 = 27 nodes per block
        self.assertEqual(frame.leaf_status.shape,(1,2,6561,9,9));self.assertEqual(int(np.count_nonzero(frame.leaf_status[0,0])),6*16*16)
        self.assertLess(int(np.count_nonzero(frame.leaf_status[0,1])),6*16*16)  # 27: the own mask loses action 1 after action 1
        frame,requests,engine,ex=self.price(specs,depth=4,k_own=(2,1,2),k_opp=(1,2,2),block_lanes=12)
        self.check_oracle(frame,requests,ex,specs,((2,1),(1,2),(2,2)))
        self.assertEqual([k for k in engine.last_timing['programs']if k[0]in('expand','leaf')],[('expand',6,2,1),('expand',6,1,2),('leaf',3,2,2)])

    # (c) selection modes -------------------------------------------------------------------------------------------
    def test_searched_maps_rank_by_value_and_threat_and_keep_the_prior_top1(self):
        specs=(((0,27),0,0,1),((22,0),1,1,0),((20,21),0,2,1))
        ex,_=T.executor();requests=self.requests(specs,ex);reference=self.depth2(specs,requests,ex)
        own,opp=dn.searched_maps(reference,2,2)
        self.assertEqual((own.shape,opp.shape,own.dtype),((3,2,9,9,3),(3,2,9,9,3),np.int32))
        appended=0
        for i in range(3):
            for w in range(2):
                for a1 in range(9):
                    for r1 in range(9):
                        if reference.t1_status[i,w,a1,r1]!=2:self.assertTrue((own[i,w,a1,r1]==-1).all());self.assertTrue((opp[i,w,a1,r1]==-1).all());continue
                        st=reference.t2_status[i,w,a1,r1];v=reference.t2_value[i,w,a1,r1].astype(np.float64)
                        rows=[a for a in range(9)if st[a].any()];cols=[x for x in range(9)if st[:,x].any()]
                        lg=reference.t1_peer_logits[i,w,a1,r1];e={x:math.exp(float(lg[x]))for x in cols};pi={x:e[x]/sum(e.values())for x in cols}
                        q={a:sum(pi[x]*v[a,x]for x in cols)for a in rows};threat={x:max(v[a,x]for a in rows)for x in cols}
                        def top1(logits,legal):return min((a for a in range(9)if legal[a]),key=lambda a:(-float(logits[a]),a))
                        for maps,requested,ranking,logits,legal in((own,reference.t1_own_requested,sorted(q,key=lambda a:(-q[a],a)),reference.t1_own_logits,reference.t1_own_legal),
                                (opp,reference.t1_peer_requested,sorted(threat,key=lambda x:(threat[x],x)),reference.t1_peer_logits,reference.t1_peer_legal)):
                            kept=[int(x)for x in maps[i,w,a1,r1]if x>=0]
                            self.assertTrue((maps[i,w,a1,r1][len(kept):]==-1).all())
                            if not requested[i,w,a1,r1]:self.assertEqual(kept,[0]);continue
                            best=top1(logits[i,w,a1,r1],legal[i,w,a1,r1]);expected=ranking[:2]
                            if best not in expected:expected=expected+[best];appended+=1
                            self.assertEqual(kept,expected);self.assertLessEqual(len(kept),3);self.assertEqual(len(set(kept)),len(kept))
        self.assertGreater(appended,0)
        # Terminal leaves (mode 22: r2==3 draws, a2==1 wins) rank by their exact values: a draw column is the most threatening reply.
        for a1,r1 in zip(*np.nonzero(reference.t1_status[1,0]==2)):self.assertEqual(int(opp[1,0,a1,r1,0]),3);self.assertEqual(int(own[1,0,a1,r1,0]),1)
        for k in((0,2),(10,1)):
            with self.assertRaises(ValueError):dn.searched_maps(reference,*k)

    def test_searched_mode_expands_the_given_maps_and_prior_mode_its_own(self):
        specs=(((0,27),0,0,1),((22,0),1,1,0),((20,21),0,2,1))
        ex,_=T.executor();requests=self.requests(specs,ex);reference=self.depth2(specs,requests,ex)
        maps={2:dn.searched_maps(reference,2,2)}
        frame,_,engine,_=self.price(specs,depth=3,k_own=(3,2),k_opp=(3,2),ex=ex,selection='searched',selection_maps=maps,block_lanes=30)
        self.check_oracle(frame,requests,ex,specs,((3,3),(2,2)),maps=maps)
        expanded=frame.t1_status==2
        np.testing.assert_array_equal(frame.t1_own_actions[expanded],maps[2][0][expanded]);np.testing.assert_array_equal(frame.t1_opp_actions[expanded],maps[2][1][expanded])
        self.assertTrue((frame.t1_own_actions[~expanded]==-1).all())
        prior,_,_,_=self.price(specs,depth=3,k_own=(3,2),k_opp=(3,2),ex=ex,block_lanes=30)
        self.check_oracle(prior,requests,ex,specs,((3,3),(2,2)))
        # Both modes share the block programs; searched kept a different ply-2 set in at least one cell, and ply 3 follows the prior in both.
        self.assertEqual([k for k in engine.last_timing['programs']if k[0]in('expand','leaf')],[('expand',3,3,3),('leaf',7,2,2)])
        self.assertFalse(np.array_equal(prior.t1_own_actions,frame.t1_own_actions))
        self.assertTrue((frame.nodes[0].own_actions[frame.nodes[0].status==2]>=0).any())
        for i in range(3):self.assertEqual(backups.learner_max_root(frame,i,ROOT_LOGITS,ROOT_LEGAL).status,'LABELED')
        # Fail closed: maps in prior mode, no maps in searched mode, a wrong shape, an illegal action, a hole, a waiting seat with a move.
        with self.assertRaises(ValueError):self.price(specs,depth=3,k_own=(3,2),k_opp=(3,2),ex=ex,selection_maps=maps)
        with self.assertRaises(ValueError):self.price(specs,depth=3,k_own=(3,2),k_opp=(3,2),ex=ex,selection='searched')
        with self.assertRaises(ValueError):self.price(specs,depth=3,k_own=(2,2),k_opp=(3,2),ex=ex,selection='searched',selection_maps=maps)
        cell=next(zip(*np.nonzero(reference.t1_status[0,0]==2)))
        for bad in(lambda m:m.__setitem__((0,0)+cell+(0,),8),lambda m:m.__setitem__((0,0)+cell,[m[(0,0)+cell][0],-1,m[(0,0)+cell][0]])):
            own=maps[2][0].copy();bad(own)
            with self.assertRaises(ValueError):self.price(specs,depth=3,k_own=(3,2),k_opp=(3,2),ex=ex,selection='searched',selection_maps={2:(own,maps[2][1])})
        waiting=next(zip(*np.nonzero(reference.t1_status[2,0]==2)));own=maps[2][0].copy();own[(2,0)+waiting]=[1,-1,-1]
        with self.assertRaises(ValueError):self.price(specs,depth=3,k_own=(3,2),k_opp=(3,2),ex=ex,selection='searched',selection_maps={2:(own,maps[2][1])})
        with self.assertRaises(ValueError):dn.DepthNPricer(ex,None,capacity=1,depth=3,k_own=(10,2),k_opp=(2,2))
        dn.DepthNPricer(ex,None,capacity=1,depth=3,k_own=(10,2),k_opp=(2,2),selection='searched')

    # (d) edge cases ------------------------------------------------------------------------------------------------
    def test_waiting_terminal_and_error_nodes_at_every_ply(self):
        specs=(((40,41),0,0,1),((20,21),1,1,0),((22,23),0,0,1),((24,25),1,1,0),((43,44),0,2,1),((28,27),1,1,0),((46,47),0,1,2))
        frame,requests,engine,ex=self.price(specs,depth=3,k_own=(3,2),k_opp=(2,3),block_lanes=40)
        self.check_oracle(frame,requests,ex,specs,((3,2),(2,3)))
        nd=frame.nodes[0];live=lambda m:[int(x)for x in m if x>=0]
        def grids(i,w):
            for node in np.flatnonzero(nd.status[i,w]==2):yield node,frame.leaf_status[i,w,node]!=0
        # Waiting seats at ply 2 only (40/41) and everywhere (20/21): the map is [0,-1,...] and only slot 0 is enumerated.
        for (i,w),seat in(((0,0),'own'),((0,1),'opp'),((1,0),'own'),((1,1),'opp')):
            req,maps=(nd.own_requested,nd.own_actions)if seat=='own'else(nd.opp_requested,nd.opp_actions);cells=nd.status[i,w]==2;self.assertTrue(cells.any())
            self.assertFalse(req[i,w][cells].any());self.assertTrue(all(live(m)==[0]for m in maps[i,w][cells]))
            for node,grid in grids(i,w):  # ply-3 widths are (k_own, k_opp) = (2, 3): one row of three replies, or two rows of one
                self.assertEqual(int(grid.sum()),3 if seat=='own'else 2);self.assertTrue(grid[0].all()if seat=='own'else grid[:,0].all())
            if i==1:self.assertFalse((frame.t1_own_requested if seat=='own'else frame.t1_peer_requested)[i,w][frame.t1_status[i,w]==2].any())
            else:self.assertTrue((frame.t1_own_requested if seat=='own'else frame.t1_peer_requested)[i,w][frame.t1_status[i,w]==2].all())
        # Terminal ply-2 nodes (22) and errored ones (23-25, 28, 43, 44) are never expanded: no leaves, all -1 maps.
        for i,w,code in((2,0,0),(2,1,3),(3,0,1),(3,1,4),(4,0,5),(4,1,6),(5,0,2)):
            status=nd.status[i,w];stopped=status==(1 if code==0 else 3);self.assertTrue(stopped.any(),(i,w))
            self.assertFalse(frame.leaf_status[i,w][stopped].any());self.assertTrue((nd.own_actions[i,w][stopped]==-1).all());self.assertTrue((nd.opp_actions[i,w][stopped]==-1).all())
            if code:self.assertTrue((nd.error[i,w][stopped]==code).all())
            else:self.assertTrue(np.isin(nd.value[i,w][stopped],(0.,1.)).all())
        self.assertTrue((nd.value[4,0][nd.status[4,0]==3]!=0).all());self.assertTrue(nd.own_legal[4,0][nd.status[4,0]==3].any(axis=-1).all())  # codes 5/6 keep the learner's value and masks
        self.assertFalse(nd.opp_logits[4,0].any());self.assertFalse(nd.opp_logits[4,1].any())
        self.assertTrue((frame.leaf_error[6,0][frame.leaf_status[6,0]==3]==1).all());self.assertTrue((frame.leaf_status[6,0]==3).any())  # 46: a leaf-only invalid pair
        self.assertFalse((nd.status[6,0]==3).any())
        wins=frame.leaf_status[6,1]==1;self.assertTrue(wins.any());self.assertTrue((frame.leaf_value[6,1][wins]==1).all());self.assertFalse((nd.status[6,1]==1).any())  # 47: leaf-only terminals
        summary=backups.refusal_summary(frame)
        self.assertEqual([x['cause']for x in summary['roots']],[None,None,'turn2','turn2','turn2','turn2','turn3'])
        self.assertEqual(summary['depth'],3);self.assertEqual(summary['deeper_refused_count'],5);self.assertEqual(summary['depth1_refused_count'],0)
        self.assertEqual(summary['errors_by_code'][2][5],int(np.count_nonzero(nd.error==5)));self.assertGreater(summary['errors_by_code'][2][6],0)
        self.assertEqual(summary['errors_by_code'][3][1],int(np.count_nonzero(frame.leaf_error==1)));self.assertEqual(set(summary['errors_by_code'][3]),{1,2,3,4})
        self.assertEqual(summary['refused_errors_by_ply'][6],{2:0,3:int(np.count_nonzero(frame.leaf_status[6]==3))})
        self.assertEqual(sum(summary['refused_by_cause'].values()),5);self.assertEqual(backups.causes(3),('materialization','pre','turn1','turn1_depth2_only','turn2','turn3','k1_pruned'))
        for i in range(7):
            self.assertEqual(backups.learner_max_root(frame,i,ROOT_LOGITS,ROOT_LEGAL).status,'LABELED'if i<2 else'REFUSED')
            self.assertEqual(frame.world_reports[i][0]['status']=='COMPLETE'and frame.world_reports[i][1]['status']=='COMPLETE',i<2)
        self.assertEqual(frame.world_reports[6][0]['ply_unknown'][3],int(np.count_nonzero(frame.leaf_status[6,0]==3)))

    def test_padding_is_never_enumerated(self):
        specs=(((0,0),0,0,1),((0,27),1,1,0),((32,32),0,0,2));missing=((1,1),)
        frame,requests,engine,ex=self.price(specs,depth=3,k_own=(2,2),k_opp=(2,2),missing=missing,block_lanes=20)
        self.check_oracle(frame,requests,ex,specs,((2,2),(2,2)),missing=missing)
        t=engine.last_timing;self.assertEqual((t['root_capacity'],t['padding_root_slots'],t['substituted_world_slots']),(4,1,1))
        parents2=int(np.count_nonzero(frame.t1_status==2));parents3=int(np.count_nonzero(frame.nodes[0].status==2))
        self.assertEqual((t['plies'][0]['parents'],t['plies'][1]['parents']),(parents2,parents3))
        # Blocks hold 5 nodes (20 lanes / 4); every group (actor pair at ply 2, learner at ply 3) pads its last block alone.
        groups2={};groups3={}
        for i,r in enumerate(requests):
            for w in range(2):
                if (i,w)in missing:continue
                groups2[r.learner_actor,r.opponent_actor]=groups2.get((r.learner_actor,r.opponent_actor),0)+int(np.count_nonzero(frame.t1_status[i,w]==2))
                groups3[r.learner_actor]=groups3.get(r.learner_actor,0)+int(np.count_nonzero(frame.nodes[0].status[i,w]==2))
        self.assertEqual(t['plies'][0]['blocks'],sum(-(-c//5)for c in groups2.values()));self.assertEqual(t['plies'][1]['blocks'],sum(-(-c//5)for c in groups3.values()))
        self.assertEqual(t['plies'][0]['lanes'],t['plies'][0]['blocks']*20);self.assertGreater(t['plies'][1]['lanes'],t['plies'][1]['enumerated'])
        self.assertEqual(t['enumerated_leaves'],int(np.count_nonzero(frame.leaf_status)))
        # Enumeration is exactly the live slot product under expanded nodes; dead k slots, padding worlds and block padding stay zero.
        nd=frame.nodes[0]
        for i in range(3):
            for w in range(2):
                for node in range(nd.status.shape[2]):
                    grid=frame.leaf_status[i,w,node]!=0
                    if nd.status[i,w,node]!=2:self.assertFalse(grid.any());continue
                    own=nd.own_actions[i,w,node]>=0;opp=nd.opp_actions[i,w,node]>=0
                    np.testing.assert_array_equal(grid,own[:,None]&opp[None,:])
                    self.assertEqual(int(own.sum()),min(2,int(nd.own_legal[i,w,node].sum()))if nd.own_requested[i,w,node]else 1)
                self.assertEqual(int(np.count_nonzero(nd.status[i,w])),sum(int((frame.t1_own_actions[i,w,a,r]>=0).sum()*(frame.t1_opp_actions[i,w,a,r]>=0).sum())for a,r in zip(*np.nonzero(frame.t1_status[i,w]==2))))
        self.assertEqual(int(np.count_nonzero(frame.t1_status[2,0]==2)),81)  # mode 32: every lane of the last real world is placed and expanded
        self.assertEqual(frame.world_reports[1][1]['reason'],'physical-materialization:inert missing')
        self.assertEqual(backups.refusal_summary(frame)['roots'][1]['cause'],'materialization')

    def test_child_masks_are_encoder_lanes_at_every_ply(self):
        ex=T.permuting_executor();specs=(((0,27),0,0,1),((0,0),1,2,0),((20,21),0,1,1))
        frame,requests,engine,_=self.price(specs,ex=ex,depth=3,k_own=(9,2),k_opp=(2,9),block_lanes=54)
        self.check_oracle(frame,requests,ex,specs,((9,2),(2,9)))
        self.assertFalse((frame.leaf_status==3).any());self.assertFalse((frame.nodes[0].status==3).any())
        nd=frame.nodes[0]
        for i,(modes,owner,learner,opponent)in enumerate(specs):
            for w in range(2):
                for node in np.flatnonzero(nd.status[i,w]==2):
                    engine_own=T.CHILD_OWN.copy();engine_foe=T.CHILD_FOE.copy()
                    if modes[w]==20:engine_own[:]=False
                    if modes[w]==21:engine_foe[:]=False
                    if modes[w]!=27:np.testing.assert_array_equal(nd.own_legal[i,w,node],engine_own[T.PERM])  # 27: the own mask follows the last action
                    np.testing.assert_array_equal(nd.opp_legal[i,w,node],engine_foe[T.PERM])
                    self.assertTrue(nd.own_legal[i,w,node][nd.own_actions[i,w,node][nd.own_actions[i,w,node]>=0]].all()or not nd.own_requested[i,w,node])
            self.assertEqual(backups.learner_max_root(frame,i,ROOT_LOGITS,ROOT_LEGAL).status,'LABELED')

    # (e) invariance ------------------------------------------------------------------------------------------------
    def test_block_lanes_invariance(self):
        specs=(((0,22),0,0,1),((27,21),1,1,0),((20,0),0,2,1))
        frames=[self.price(specs,depth=3,k_own=(2,3),k_opp=(3,2),block_lanes=lanes)[0]for lanes in(1,7,40,2187,10000)]
        for other in frames[1:]:self.assert_frames_equal(frames[0],other)
        self.assertEqual([w['status']for pair in frames[0].world_reports for w in pair],[w['status']for pair in frames[-1].world_reports for w in pair])
        self.assertEqual(dn.block_nodes(1,2,3),1);self.assertEqual(dn.block_nodes(2187,9,9),27);self.assertEqual(dn.block_nodes(40,2,3),6)

    def test_one_block_program_compile_across_capacities_plies_and_depths(self):
        specs=(((0,22),0,0,1),((27,21),1,1,0),((20,0),0,2,1))
        ex=deep_executor();engine=dn.DepthNPricer(ex,object(),capacity=8,depth=4,k_own=(2,2,2),k_opp=(2,2,2),block_lanes=16,actor_capacity=3)
        frames=[]
        for k in(1,3,2):
            frame,requests,_,_=self.price(specs[:k],ex=ex,engine=engine,depth=4,k_own=None,k_opp=None);frames.append(frame)
            kinds=[key[0]for key in engine.last_timing['compiled']]
            if k==1:self.assertEqual(kinds.count('expand'),1);self.assertEqual(kinds.count('leaf'),1)
            else:self.assertNotIn('expand',kinds);self.assertNotIn('leaf',kinds)  # plies 2 and 3 and a new capacity reuse the (B=4, K=4) programs
            self.assertEqual(kinds.count('P1'),1)  # capacities 1, 4 and 2: P1 alone follows the frame size
        keys=engine.last_timing['programs']
        self.assertEqual([k for k in keys if k[0]in('expand','leaf')],[('expand',4,2,2),('leaf',4,2,2)]);self.assertEqual(sum(k[0]=='P1'for k in keys),3)
        self.check_oracle(frames[1],self.requests(specs,ex),ex,specs,((2,2),(2,2),(2,2)))
        # The same engine object prices N=2 frames bit-equal to Depth2Pricer whatever it compiled before, through a sibling at depth 2.
        sibling=dn.DepthNPricer(ex,object(),capacity=8,depth=2,k_own=(2,),k_opp=(2,),block_lanes=16,actor_capacity=3)
        frame,requests,_,_=self.price(specs,ex=ex,engine=sibling,depth=2,k_own=None,k_opp=None)
        reference=self.depth2(specs,requests,ex,k_own=2,k_opp=2,child_block=4);mine=dn.as_depth2_frame(frame)
        for name in('t1_value','t2_status','t2_value','t2_error'):np.testing.assert_array_equal(getattr(reference,name),getattr(mine,name),err_msg=name)

    def test_fixed_capacity_compiles_p1_once_and_matches_the_bucketed_frames(self):
        specs=(((0,22),0,0,1),((27,21),1,1,0),((20,0),0,2,1));ex=deep_executor()
        loose=[self.price(specs[:k],ex=ex,depth=3,k_own=(2,2),k_opp=(2,2),block_lanes=16)[0]for k in(1,2,3)]
        engine=dn.DepthNPricer(ex,object(),capacity=8,depth=3,k_own=(2,2),k_opp=(2,2),block_lanes=16,fixed_capacity=True,actor_capacity=3)
        for k,a in zip((1,2,3),loose):
            b,_,_,_=self.price(specs[:k],ex=ex,engine=engine,depth=3,k_own=None,k_opp=None)
            self.assert_frames_equal(a,b,f'k={k}');self.assertEqual((b.timing['root_capacity'],b.timing['padding_root_slots']),(8,8-k))
        self.assertEqual(sum(k[0]=='P1'for k in engine.programs),1)

    def test_digests_match_the_oracle_inputs_at_every_ply(self):
        specs=(((0,27),0,0,1),((40,22),1,1,0))
        frame,requests,engine,ex=self.price(specs,depth=3,k_own=(2,2),k_opp=(2,2),digest=True,block_lanes=12)
        oracles=self.check_oracle(frame,requests,ex,specs,((2,2),(2,2)))
        nd=frame.nodes[0];self.assertEqual((frame.t1_digest_own.shape,nd.digest_own.shape,frame.leaf_digest.shape),((2,2,9,9,2),(2,2,324,2),(2,2,324,2,2,2)))
        for (i,w),o in oracles.items():
            np.testing.assert_array_equal(frame.t1_digest_own[i,w].any(-1),frame.t1_status[i,w]!=0)
            np.testing.assert_array_equal(nd.digest_own[i,w].any(-1),nd.status[i,w]!=0);np.testing.assert_array_equal(frame.leaf_digest[i,w].any(-1),frame.leaf_status[i,w]!=0)
            for (a,r),(own,peer)in o['inputs1'].items():
                np.testing.assert_array_equal(frame.t1_digest_own[i,w,a,r],np.asarray(depth2_digest.digest(own)));np.testing.assert_array_equal(frame.t1_digest_peer[i,w,a,r],np.asarray(depth2_digest.digest(peer)))
            for (p,cid),(own,opp)in o['node_inputs'].items():
                np.testing.assert_array_equal(nd.digest_own[i,w,cid],np.asarray(depth2_digest.digest(own)));np.testing.assert_array_equal(nd.digest_opp[i,w,cid],np.asarray(depth2_digest.digest(opp)))
            for (pid,a,b),inputs in o['leaf_inputs'].items():np.testing.assert_array_equal(frame.leaf_digest[i,w,pid,a,b],np.asarray(depth2_digest.digest(inputs)))
        plain=self.price(specs,depth=3,k_own=(2,2),k_opp=(2,2),block_lanes=12)[0];self.assertIsNone(plain.leaf_digest);self.assertIsNone(plain.nodes[0].digest_own)
        np.testing.assert_array_equal(plain.leaf_value,frame.leaf_value)

    # (f) backups ---------------------------------------------------------------------------------------------------
    def test_backups_match_brute_force(self):
        specs=(((0,0),0,0,1),((27,21),1,1,0),((20,0),0,2,1),((22,40),1,1,2),((46,0),0,1,1))
        frame,requests,engine,ex=self.price(specs,depth=3,k_own=(2,3),k_opp=(3,2),block_lanes=40)
        for i in range(len(specs)):
            for fn,rule in((backups.backup_learner_max,MAXIMIZE),(backups.backup_average,AVERAGE),(lambda f,r:backups.backup(f,r,pessimistic),MINIMIZE)):
                prices=fn(frame,i)
                for w in range(2):
                    f=lambda name:getattr(frame,name)[i,w]
                    kept=tuple(bool(f('t1_status')[:,r].any())for r in range(9))if f('root_peer_requested')else(False,)*9
                    self.assertEqual((prices[w].opponent_requested,prices[w].opponent_legal,prices[w].opponent_logits),(bool(f('root_peer_requested')),kept,tuple(map(float,f('root_peer_logits')))))
                    for a in range(9):
                        for r in range(9):
                            expected=brute_value(frame,i,w,1,a*9+r,rule)if f('t1_status')[a,r]else None
                            if expected is None:self.assertIsNone(prices[w].joint_values[a][r])
                            else:self.assertAlmostEqual(prices[w].joint_values[a][r],expected,places=12)
            top=backups.backup_learner_max(frame,i);avg=backups.backup_average(frame,i);low=backups.backup(frame,i,pessimistic)
            for w in range(2):
                for a in range(9):
                    for r in range(9):
                        if top[w].joint_values[a][r]is not None:
                            self.assertGreaterEqual(top[w].joint_values[a][r],avg[w].joint_values[a][r]-1e-12);self.assertGreaterEqual(avg[w].joint_values[a][r],low[w].joint_values[a][r]-1e-12)
            for fn in(backups.learner_max_root,backups.average_root):
                reduced=fn(frame,i,ROOT_LOGITS,ROOT_LEGAL);self.assertEqual(reduced.status,'LABELED'if i<4 else'REFUSED')
        # Terminal ply-2 nodes (22) back up their exact value (a win is exactly 1); a waiting learner (40) leaves one row and still prices.
        nd=frame.nodes[0];wins=np.flatnonzero((nd.status[3,0]==1)&(nd.value[3,0]==1.));self.assertTrue(len(wins))
        self.assertEqual(backups.node_value(frame,3,0,2,int(wins[0])),1.)
        self.assertTrue(all(backups.node_value(frame,3,w,1,a*9+r)is not None for w in range(2)for a,r in zip(*np.nonzero(frame.t1_status[3,w]==2))))
        self.assertIsNone(backups.node_value(frame,4,0,1,int(np.flatnonzero(frame.t1_status[4,0].reshape(-1)==2)[0])))  # an error under the enumerated path
        frame2=replace(frame,pre=frame.pre.copy());frame2.pre[0,1]=9;self.assertIsNone(backups.backup_learner_max(frame2,0)[1])
        self.assertEqual(backups.refusal_summary(frame2)['roots'][0]['cause'],'pre')

    def test_learner_max_ties_and_renormalization_synthetic(self):
        # A hand-built N=3 frame: one root, world 0, one expanded turn-1 cell (0,0) with widths (2,2) at ply 2 and (2,2) at ply 3.
        Z=lambda shape,d=np.float32:np.zeros((1,2)+shape,d)
        nodes=dn.PlyNodes(ply=2,parent=np.arange(324)//4,parent_own_slot=((np.arange(324)%4)//2).astype(np.int8),parent_opp_slot=(np.arange(324)%2).astype(np.int8),
            status=Z((324,),np.int8),value=Z((324,)),error=Z((324,),np.int8),own_requested=Z((324,),bool),opp_requested=Z((324,),bool),
            own_legal=Z((324,9),bool),opp_legal=Z((324,9),bool),own_logits=Z((324,9)),opp_logits=Z((324,9)),own_actions=np.full((1,2,324,2),-1,np.int32),opp_actions=np.full((1,2,324,2),-1,np.int32))
        frame=dn.DepthNFrame(depth=3,k_own=(2,2),k_opp=(2,2),pre=Z((),np.int32),root_peer_requested=Z((),bool),root_peer_legal=Z((9,),bool),root_peer_logits=Z((9,)),
            t1_status=Z((9,9),np.int8),t1_value=Z((9,9)),t1_error=Z((9,9),np.int8),t1_own_requested=Z((9,9),bool),t1_own_legal=Z((9,9,9),bool),t1_own_logits=Z((9,9,9)),
            t1_peer_requested=Z((9,9),bool),t1_peer_legal=Z((9,9,9),bool),t1_peer_logits=Z((9,9,9)),t1_own_actions=np.full((1,2,9,9,2),-1,np.int32),t1_opp_actions=np.full((1,2,9,9,2),-1,np.int32),
            nodes=(nodes,),leaf_status=Z((324,2,2),np.int8),leaf_value=Z((324,2,2)),leaf_error=Z((324,2,2),np.int8),world_reports=(),timing={})
        f=lambda name:getattr(frame,name)[0,0];n=lambda name:getattr(nodes,name)[0,0]
        frame.root_peer_requested[0,0]=True;f('root_peer_legal')[0]=True
        f('t1_status')[0,0]=2;f('t1_own_requested')[0,0]=True;f('t1_peer_requested')[0,0]=True
        f('t1_own_actions')[0,0]=[3,1];f('t1_own_logits')[0,0][[1,3]]=[0.,2.];f('t1_opp_actions')[0,0]=[4,2];f('t1_peer_logits')[0,0][[2,4]]=[1.,0.]
        # Ply-2 children of cell 0 (ids 0..3 = slots (0,0),(0,1),(1,0),(1,1)): a terminal, two expanded nodes, and one with a waiting opponent.
        n('status')[[0,1,2,3]]=[1,2,2,2];n('value')[0]=.5
        for cid,own,opp in((1,[0,1],[2,3]),(2,[1,0],[0,-1]),(3,[4,-1],[3,2])):
            n('own_actions')[cid]=own;n('opp_actions')[cid]=opp;n('own_requested')[cid]=True;n('opp_requested')[cid]=cid!=2
        n('own_logits')[1][[0,1]]=[1.,1.];n('opp_logits')[1][[2,3]]=[0.,0.];n('opp_logits')[3][[2,3]]=[1.,0.]
        leaves=f('leaf_status');vals=f('leaf_value');f32=lambda x:float(np.float32(x));e=math.exp(1.);lo,hi=1/(1+e),e/(1+e)
        leaves[1]=4;vals[1]=[[.4,.2],[.2,.4]]  # rows tie at (.4+.2)/2: the lower slot wins; both rows are equal
        leaves[2,:,0]=4;vals[2,:,0]=[.1,-.2]   # waiting opponent: column 0 only; learner max is slot 0 (.1)
        leaves[3,0]=[1,4];vals[3,0]=[1.,-.5]   # one own slot: pi_opp = (lo on slot 0 (action 3, logit 0), hi on slot 1 (action 2, logit 1))
        v1,v2,v3=(f32(.4)+f32(.2))/2,f32(.1),1.*lo+f32(-.5)*hi
        self.assertAlmostEqual(backups.node_value(frame,0,0,2,1),v1,places=12);self.assertAlmostEqual(backups.node_value(frame,0,0,2,2),v2,places=12)
        self.assertAlmostEqual(backups.node_value(frame,0,0,2,3),v3,places=12);self.assertEqual(backups.node_value(frame,0,0,2,0),.5)
        # Cell (0,0): opponent slot 0 is action 4 (logit 0, weight lo), slot 1 action 2 (logit 1, weight hi); own slot 0 (action 3)
        # averages children 0 and 1, own slot 1 (action 1) children 2 and 3. Learner-max takes the better row.
        row0=.5*lo+v1*hi;row1=v2*lo+v3*hi
        self.assertAlmostEqual(backups.node_value(frame,0,0,1,0),max(row0,row1),places=12)
        # The average rule applies at every node: child 2's two rows (own logits equal) average instead of maximizing.
        p3=math.exp(2.)/(math.exp(2.)+1.);row1_avg=((f32(.1)+f32(-.2))/2)*lo+v3*hi
        self.assertAlmostEqual(backups.node_value(frame,0,0,1,0,backups.average),p3*row0+(1-p3)*row1_avg,places=12)
        prices=backups.backup_learner_max(frame,0)[0];self.assertAlmostEqual(prices.joint_values[0][0],max(row0,row1),places=12);self.assertEqual(prices.opponent_legal,(True,)+(False,)*8)
        self.assertIsNone(prices.joint_values[0][1])
        leaves[1,1,1]=3;self.assertIsNone(backups.node_value(frame,0,0,2,1));self.assertIsNone(backups.node_value(frame,0,0,1,0))
        self.assertEqual(backups.refusal_summary(frame)['roots'][0]['cause'],'turn3')

    def test_empty_and_fail_closed(self):
        ex=deep_executor()
        for kw in(dict(depth=1,k_own=(),k_opp=()),dict(depth=3,k_own=(2,),k_opp=(2,2)),dict(depth=2,k_own=(0,),k_opp=(2,)),dict(depth=2,k_own=(2,),k_opp=(10,)),
                dict(depth=2,k_own=2,k_opp=2,k1_own=0),dict(depth=2,k_own=2,k_opp=2,block_lanes=0),dict(depth=2,k_own=2,k_opp=2,selection='value'),
                dict(depth=2,k_own=2,k_opp=2,digest=1),dict(depth=2,k_own=2,k_opp=2,fixed_capacity=1),dict(depth=2,k_own=2,k_opp=2,capacity=0)):
            kw.setdefault('capacity',1)
            with self.assertRaises(ValueError):dn.DepthNPricer(ex,None,actor_capacity=2,**kw)
        engine=dn.DepthNPricer(ex,None,capacity=1,depth=3,k_own=3,k_opp=(2,2),actor_capacity=1)
        self.assertEqual((engine.k_own,engine.k_opp),((3,3),(2,2)))
        with self.assertRaises(ValueError):engine.bind(({},{}))
        empty=engine.price((),self.tables,self.recipe,PARAMS[:1])
        self.assertEqual((empty.pre.shape,empty.t1_own_actions.shape,empty.nodes[0].status.shape,empty.leaf_status.shape),((0,2),(0,2,9,9,3),(0,2,486),(0,2,486,3,2)))
        with self.assertRaises(ValueError):engine.price(self.requests((((0,0),0,0,0),((0,0),1,0,0)),ex),self.tables,self.recipe,PARAMS[:1])
        with self.assertRaises(ValueError):engine.price((),self.tables,self.recipe,PARAMS[:1],selection_maps={2:None})
        single,requests,_,_=self.price((((0,22),0,0,0),((27,0),1,0,0)),depth=3,k_own=(2,2),k_opp=(2,2),params=PARAMS[:1],capacity=2,block_lanes=8)
        self.check_oracle(single,requests,ex,(((0,22),0,0,0),((27,0),1,0,0)),((2,2),(2,2)),params=PARAMS[:1])


if __name__=='__main__':unittest.main()
