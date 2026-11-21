"""Frozen host pack/place oracle from 5d4a41d5; keep independent of production optimizations."""
import numpy as np
from gpu_engine.device_depthn_pricer import WidePack


def pack_wide_leaf(group,own_live,opp_live,*,nodes,lanes,chunk):
    """Pack only the enumerated (parent, own slot, opponent slot) lanes into calls of `lanes` lanes and `nodes` node slots.

    `group` [P] (the forward's actor per parent; equal groups contiguous), `own_live` [P,k_own] and `opp_live` [P,k_opp]
    (the live map slots). Parents are taken in order, greedily: a parent's lanes are contiguous, in (own slot, opponent
    slot) order, and never span calls; a call closes when the next parent's lanes or node slot do not fit; a group change
    starts a new forward chunk, so every chunk holds one group. A parent without an enumerated lane is not packed. Pure
    numpy; raises ValueError on bad capacities or non-contiguous groups."""
    group=np.asarray(group,np.int64);own_live=np.asarray(own_live,bool);opp_live=np.asarray(opp_live,bool)
    P=len(group);k_own,k_opp=own_live.shape[1],opp_live.shape[1];K=k_own*k_opp
    if any(type(x)is not int or x<=0 for x in(nodes,lanes,chunk))or lanes%chunk or chunk<K:
        raise ValueError('positive capacities, lanes a whole number of chunks and a chunk of at least K lanes required')
    if own_live.shape[0]!=P or opp_live.shape[0]!=P:raise ValueError('one map row per parent required')
    starts=np.flatnonzero(np.r_[True,group[1:]!=group[:-1]])if P else np.zeros(0,np.int64)
    if len(starts)!=len(np.unique(group)):raise ValueError('each group\'s parents must be contiguous')
    live=(own_live[:,:,None]&opp_live[:,None,:]).reshape(P,K);count=live.sum(1)
    node_call=np.full(P,-1,np.int64);node_slot=np.full(P,-1,np.int64);node_lane=np.full(P,-1,np.int64)
    packable=np.flatnonzero(count>0);cum=np.concatenate(([0],np.cumsum(count[packable])))
    call=-1;used_lanes=lanes;used_nodes=nodes;i=0;prev=None
    while i<len(packable):
        g=group[packable[i]]
        if g!=prev:
            used_lanes=-(-used_lanes//chunk)*chunk;prev=g   # a new group starts a new forward chunk
        run=i+int(np.searchsorted(group[packable[i:]]!=g,True))   # packable parents of this group from i (contiguous)
        e=min(run,i+nodes-used_nodes,int(np.searchsorted(cum,cum[i]+lanes-used_lanes,'right'))-1)
        if e<=i:call+=1;used_lanes=0;used_nodes=0;continue
        idx=packable[i:e];node_call[idx]=call;node_slot[idx]=used_nodes+np.arange(e-i);node_lane[idx]=used_lanes+cum[i:e]-cum[i]
        used_nodes+=e-i;used_lanes+=int(cum[e]-cum[i]);i=e
    calls=call+1;C=lanes//chunk
    pi,q=np.nonzero(live&(node_call>=0)[:,None])   # row-major: one parent's lanes contiguous, (own, opp) slot order
    rank=np.arange(len(pi))-np.searchsorted(pi,pi,'left')
    pos=node_call[pi]*lanes+node_lane[pi]+rank
    flat=lambda dtype:np.zeros(calls*lanes,dtype)
    lane_node=flat(np.int32);lane_own=flat(np.int32);lane_opp=flat(np.int32);lane_live=flat(bool)
    lane_node[pos]=node_slot[pi];lane_own[pos]=q//k_opp;lane_opp[pos]=q%k_opp;lane_live[pos]=True
    call_nodes=np.full((calls,nodes),-1,np.int64);packed=np.flatnonzero(node_call>=0);call_nodes[node_call[packed],node_slot[packed]]=packed
    chunk_group=np.full(calls*C,-1,np.int64);chunk_group[pos//chunk]=group[pi]
    if len(np.unique(pos))!=len(pos)or not np.array_equal(chunk_group[pos//chunk],group[pi]):raise RuntimeError('wide leaf packing overlaps lanes or mixes groups in a chunk')
    shape=lambda x:x.reshape(calls,lanes)
    return WidePack(calls=calls,nodes=nodes,lanes=lanes,chunk=chunk,node_call=node_call,node_slot=node_slot,node_lane=node_lane,call_nodes=call_nodes,
        lane_node=shape(lane_node),lane_own=shape(lane_own),lane_opp=shape(lane_opp),lane_live=shape(lane_live),chunk_group=chunk_group.reshape(calls,C),
        lane_parent=pi.astype(np.int64),lane_slot_own=(q//k_opp).astype(np.int64),lane_slot_opp=(q%k_opp).astype(np.int64),lane_pos=pos)


def place_wide_leaf(pack,fields,pw,pid,n,M,k_own,k_opp,specs):
    """The dense leaf arrays [n,M,k_own,k_opp(,...)] from a wide ply's flat call outputs (`fields[name]` [calls*lanes(,...)]):
    each packed lane at (world, node, own slot, opponent slot) of its parent; every other position stays zero, as the
    blocks layout leaves non-enumerated lanes. `specs`: (name, trailing shape, dtype)."""
    out={name:np.zeros((n,M,k_own,k_opp)+shape,dtype)for name,shape,dtype in specs}
    if len(pack.lane_pos):
        w,node=pw[pack.lane_parent],pid[pack.lane_parent]
        for name,x in out.items():x[w,node,pack.lane_slot_own,pack.lane_slot_opp]=np.asarray(fields[name])[pack.lane_pos]
    return out
