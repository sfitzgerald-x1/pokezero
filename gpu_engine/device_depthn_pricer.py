"""Depth-N selective search prices: D3's turn 1 at full width, then top-k plies under every expanded node.

Programs (compile is flat in depth and, past turn 1, independent of capacity):
- P1 is the depth-2 kernel's turn-1 stage (`device_depth2_pricer.make_turn1_stage`):
  D3's grid and chance, keeping the 81 children's states and trackers, both
  seats' request flags, encoder masks and priors. `t1_value` and the learner's
  child prior are D3's bits. Its shape key is D3's: (capacity, actor rows).
- P_expand(B, K) steps one block of B nodes x K = k_own*k_opp pairs under each
  node's own draw (`executor.sample(node)`), observes both seats at the new
  nodes and runs both prior forwards (learner params on the learner's view,
  opponent params on the opponent's) plus the learner value from the same
  [B*K] geometry. It writes status, value, error, request flags, encoder
  masks, logits and the new states and trackers. Terminal, error and codes 1-6
  follow depth 2 at every ply; a terminal or errored node is never expanded.
- P_leaf(B, K) steps, observes the learner and runs the value forward.
  Both block programs are jitted once per (B, K) and reused across plies,
  capacities and depths; B = block_lanes // K sizes blocks in lanes.
Selection is an input to the block programs: per node, each seat's slot ->
action map ([k], -1 padded; a waiting seat [0, -1, ...] with engine action
-1). `prior` ranks each seat's encoder-legal actions by its own logits (the
depth-2 rule, `ranked`); `searched` takes the maps from `selection_maps`
(`searched_maps` builds them from a full-width depth-2 frame).
The host loop over plies 2..N packs the previous ply's expanded nodes into
blocks grouped by actor pair and into calls of `blocks_per_call` blocks; a
call is one jitted gather from the old pool, one block program and one
in-place (donated) slice write into the next pool, preallocated at the call
granularity, so only two plies' pools are ever alive and node fields reach
the host once per ply. The gather and write programs are keyed by the pool's
row count, but they are a bare XLA gather and dynamic-update-slice, so a new
capacity or ply recompiles nothing expensive. Padding (roots, worlds, lanes,
dead slots, block and call slots) is computed but never enumerated.
"""
from dataclasses import dataclass,replace
import math
import time

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_gumbel as original
from . import device_public_teacher as serial,device_public_teacher_batch as batch,device_public_world_state as mapping
from . import device_depth2_pricer as depth2,depth2_backups,depth2_digest as digests

LANES=depth2.LANES;UNAVAILABLE=depth2.UNAVAILABLE;INPUT_KEYS=depth2.INPUT_KEYS
PEER_VIEW_ERROR=depth2.PEER_VIEW_ERROR;PEER_PRIOR_REFUSED=depth2.PEER_PRIOR_REFUSED
PRE_REASONS=depth2.PRE_REASONS;ERROR_REASONS=depth2.ERROR_REASONS
SELECTIONS=('prior','searched')
T1_FIELDS=('pre','root_peer_requested','root_peer_legal','root_peer_logits','t1_status','t1_value','t1_error',
    't1_own_requested','t1_own_legal','t1_own_logits','t1_peer_requested','t1_peer_legal','t1_peer_logits')
NODE_FIELDS=('status','value','error','own_requested','opp_requested','own_legal','opp_legal','own_logits','opp_logits')


@dataclass(frozen=True)
class PlyNodes:
    """One intermediate ply p (2 <= p < N): dense node arrays [R,2,M_p(,...)], node id = parent*K_p + own_slot*k_opp,p + opp_slot.

    `parent`, `parent_own_slot` and `parent_opp_slot` are [M_p] (the dense
    layout makes them the same in every world; the parent of a ply-2 node is
    its turn-1 cell a1*9+r1). `status`: 0 not enumerated, 1 terminal, 2
    expanded, 3 error; `value` exact if terminal else the learner's head
    value; `*_actions` are the selection maps for ply p+1.
    """
    ply:int
    parent:object
    parent_own_slot:object
    parent_opp_slot:object
    status:object
    value:object
    error:object
    own_requested:object
    opp_requested:object
    own_legal:object
    opp_legal:object
    own_logits:object
    opp_logits:object
    own_actions:object
    opp_actions:object
    digest_own:object=None
    digest_opp:object=None


@dataclass(frozen=True)
class DepthNFrame:
    """The dense surface, leading axes [root, world]. Ply 1 keeps the depth-2 `t1_*` fields (action-indexed [R,2,9,9]) plus its
    ply-2 maps; `nodes[p-2]` is ply p for 2 <= p < N; leaves are [R,2,M_{N-1},k_own_N,k_opp_N] (status 1 terminal, 4 head, 3 error)."""
    depth:int
    k_own:tuple
    k_opp:tuple
    pre:object
    root_peer_requested:object
    root_peer_legal:object
    root_peer_logits:object
    t1_status:object
    t1_value:object
    t1_error:object
    t1_own_requested:object
    t1_own_legal:object
    t1_own_logits:object
    t1_peer_requested:object
    t1_peer_legal:object
    t1_peer_logits:object
    t1_own_actions:object
    t1_opp_actions:object
    nodes:tuple
    leaf_status:object
    leaf_value:object
    leaf_error:object
    world_reports:tuple
    timing:dict
    t1_digest_own:object=None
    t1_digest_peer:object=None
    leaf_digest:object=None


def arrays(frame):
    """Every array of a frame as (name, array) pairs, plies included; for equality checks."""
    for name in DepthNFrame.__dataclass_fields__:
        value=getattr(frame,name)
        if name in('nodes','world_reports','timing','depth','k_own','k_opp')or value is None:continue
        yield name,value
    for p in frame.nodes:
        for name in PlyNodes.__dataclass_fields__:
            value=getattr(p,name)
            if name=='ply'or value is None:continue
            yield f'ply{p.ply}.{name}',value


def block_nodes(block_lanes,k_own,k_opp):
    """B for a lane budget: blocks are sized in lanes (B*K <= block_lanes), never below one node."""
    return max(1,block_lanes//(k_own*k_opp))


def _make_expand_joint(executor):
    """The depth-2 child joint with both seats' request flags: at plies past turn 1 the learner may be waiting too."""
    def single(state,public,own,peer,owner,requested,pair,chance):
        actions=j.full((1,2),-1,j.int32)
        chosen=executor.dispatch(own,state,owner,pair[:1])
        actions=actions.at[:,owner[0]].set(j.where(requested[owner[0]],chosen,-1))
        reply=executor.dispatch(peer,state,1-owner,pair[1:])
        actions=actions.at[:,1-owner[0]].set(j.where(requested[1-owner[0]],reply,-1))
        after=executor.advance(state,public,actions,chance)
        masks=executor.legal(after.step.state)
        view=executor.observe(after.step.state,after.trackers,owner,masks[:,owner[0]])
        foe=executor.observe(after.step.state,after.trackers,1-owner,masks[:,1-owner[0]])
        return (after.step.invalid_actions,after.step.state.done,after.step.state.winner,view.observation.error,view.inputs,
            after.step.state,after.trackers,masks,view.observation.legal_action_mask,foe.observation.legal_action_mask,foe.observation.error,foe.inputs)
    return jax.vmap(single,in_axes=(None,None,None,None,None,None,0,None))


def make_p1(executor,model,*,params,capacity,k1_own=9,k1_opp=9,digest=False,forward=None):
    """P1: the turn-1 stage's cell-indexed frame, each lane's cell id (81 when not enumerated) and the flat child pools."""
    turn1,forward=depth2.make_turn1_stage(executor,model,params=params,capacity=capacity,k1_own=k1_own,k1_opp=k1_opp,digest=digest,forward=forward)
    n=2*capacity
    def run(a):
        t=turn1(a);flat=lambda x:x.reshape((n*LANES,)+x.shape[2:])
        lane_cell=j.where(t['enumerated'],t['pairs'][...,0]*9+t['pairs'][...,1],LANES).astype(j.int32)
        return dict(t['frame'],lane_cell=lane_cell,states=jax.tree.map(flat,t['child_states']),publics=jax.tree.map(flat,t['child_publics']))
    return jax.jit(run)


def make_block_program(executor,*,params,forward,nodes,blocks,k_own,k_opp,leaf,digest=False):
    """P_expand(B, G, K) (leaf=False) or P_leaf(B, G, K) (leaf=True): G blocks of B nodes in one device call.

    Inputs are one call's gathered batch, [G*B,...]: the nodes' states and
    trackers, `owner`, `requested` [G*B,2] (seat-indexed), both maps [G*B,k],
    `live` [G*B] (a real node, not block or call padding) and the actor
    indices per block [G,2] into the bound bank. Blocks run under jax.lax.map,
    so one block's states, views and leaves are live at a time, and a block
    without a live node is skipped. A dead map slot steps slot 0's action and
    is never enumerated. Outputs are flattened [G*B*K(,...)] in (block, node,
    own slot, opponent slot) order; at the leaf ply status/value/error (and
    the digest) only. The shape key is (B, G, K): nothing here depends on the
    pool, the capacity or the ply.
    """
    b=nodes;g=blocks;K=k_own*k_opp
    joint=(depth2._make_leaf_joint if leaf else _make_expand_joint)(executor);views=depth2._make_views(executor)
    stacked=params[0]if len(params)==1 else jax.tree.map(lambda *xs:j.stack(xs),*params)
    actor_params=lambda actor:params[0]if len(params)==1 else jax.tree.map(lambda x:x[actor],stacked)
    def head(actor,inputs):
        prediction=forward(actor_params(actor),jax.tree.map(lambda x:x.reshape((b*K,)+x.shape[3:]),inputs))
        lg=prediction.policy_logits.reshape(b,K,9).astype(j.float32);v=prediction.value.reshape(b,K).astype(j.float32);ok=prediction.valid.reshape(b,K)
        return lg,v,ok&j.all(j.isfinite(lg),axis=-1)&j.isfinite(v)&(j.abs(v)<=1)
    keep=lambda mask,x:j.where(mask.reshape(mask.shape+(1,)*(x.ndim-2)),x,j.zeros((),x.dtype))
    def block(args):
        states,publics,owner,requested,own_map,opp_map,live,learner_actor,opp_actor=args
        def compute():
            own_act=j.where(own_map>=0,own_map,own_map[:,:1]);opp_act=j.where(opp_map>=0,opp_map,opp_map[:,:1])
            pair=j.stack((j.broadcast_to(own_act[:,:,None],(b,k_own,k_opp)),j.broadcast_to(opp_act[:,None,:],(b,k_own,k_opp))),axis=-1).reshape(b,K,2)
            # Re-observing the node gives the dispatch views its own expansion saw, without holding them in the pool.
            bo,bpe,_,_=jax.vmap(views)(states,publics,owner)
            chance=jax.vmap(executor.sample)(states)
            out=jax.vmap(joint)(states,publics,bo,bpe,owner[:,None],requested,pair,chance)
            invalid,done,winner,error=(x[:,:,0]for x in out[:4]);inputs=out[4]
            lg,v,usable=head(learner_actor,inputs)
            terminal_ok=(winner>=0)&(winner<=2)
            err=j.where(invalid,1,j.where(done&~terminal_ok,2,j.where(~done&(error!=0),3,j.where(~done&~usable,4,0))))
            terminal=j.where(winner==2,0.,j.where(winner==owner[:,None],1.,-1.))
            enumerated=(live[:,None,None]&(own_map>=0)[:,:,None]&(opp_map>=0)[:,None,:]).reshape(b,K)
            if leaf:
                result=dict(status=j.where(~enumerated,0,j.where(err!=0,3,j.where(done,1,4))).astype(j.int8),
                    value=j.where(enumerated&(err==0),j.where(done,terminal,v),0.).astype(j.float32),error=j.where(enumerated,err,0).astype(j.int8))
                if digest:result['digest']=keep(enumerated,digests.batched(inputs,axes=2))
                return result
            new_states,new_publics,masks,own_legal,opp_legal,opp_error,opp_inputs=out[5:]
            values=j.where(done,terminal,v)
            priced=err==0;viewed=priced&~done
            # As at turn 1: request flags from the engine mask, legal masks from each seat's encoder view, the opponent's
            # view held to the root's rule with codes 5/6 so the learner's value and prior survive under them.
            masks=masks[:,:,0];seat=owner[:,None,None,None]
            own_req=j.any(j.take_along_axis(masks,seat,axis=2)[:,:,0],axis=-1);opp_req=j.any(j.take_along_axis(masks,1-seat,axis=2)[:,:,0],axis=-1)
            own_legal=own_legal[:,:,0];opp_legal=opp_legal[:,:,0];opp_error=opp_error[:,:,0]
            plg,_,pusable=head(opp_actor,opp_inputs)
            err=j.where(viewed&opp_req&(opp_error!=0),PEER_VIEW_ERROR,err)
            err=j.where(viewed&(err==0)&opp_req&~pusable,PEER_PRIOR_REFUSED,err)
            priced&=enumerated;viewed&=enumerated
            status=j.where(~enumerated,0,j.where(err!=0,3,j.where(done,1,2))).astype(j.int8);expanded=status==2
            result=dict(status=status,value=j.where(priced,values,0.).astype(j.float32),error=j.where(enumerated,err,0).astype(j.int8),
                own_requested=keep(viewed,own_req),opp_requested=keep(viewed,opp_req),own_legal=keep(viewed,own_legal),opp_legal=keep(viewed,opp_legal),
                own_logits=keep(viewed,j.where(own_req[...,None],lg,0.)),opp_logits=keep(expanded,j.where(opp_req[...,None],plg,0.)),
                states=new_states,publics=new_publics)
            if digest:result.update(digest_own=keep(enumerated,digests.batched(inputs,axes=2)),digest_opp=keep(enumerated,digests.batched(opp_inputs,axes=2)))
            return result
        def empty():
            z=lambda shape,dtype:j.zeros((b,K)+shape,dtype)
            result=dict(status=z((),j.int8),value=z((),j.float32),error=z((),j.int8))
            if leaf:
                if digest:result['digest']=z((2,),j.uint32)
                return result
            result.update(own_requested=z((),bool),opp_requested=z((),bool),own_legal=z((9,),bool),opp_legal=z((9,),bool),own_logits=z((9,),j.float32),opp_logits=z((9,),j.float32),
                states=jax.tree.map(lambda x:z(x.shape[1:],x.dtype),states),publics=jax.tree.map(lambda x:z(x.shape[1:],x.dtype),publics))
            if digest:result.update(digest_own=z((2,),j.uint32),digest_opp=z((2,),j.uint32))
            return result
        return jax.lax.cond(j.any(live),compute,empty)
    def run(states,publics,owner,requested,own_map,opp_map,live,actors):
        per=lambda x:x.reshape((g,b)+x.shape[1:])
        out=jax.lax.map(block,(jax.tree.map(per,states),jax.tree.map(per,publics),per(owner),per(requested),per(own_map),per(opp_map),per(live),actors[:,0],actors[:,1]))
        return jax.tree.map(lambda x:x.reshape((g*b*K,)+x.shape[3:]),out)
    return jax.jit(run)


def make_gather():
    """One call's node batch from a pool by row: a plain XLA gather, keyed by the pool's row count (negligible compile)."""
    return jax.jit(lambda pool,rows:jax.tree.map(lambda x:x[rows],pool))


def make_write():
    """Write one call's flattened new nodes into the next pool at `offset`, in place: the pool buffer is donated, so the
    update is a dynamic-update-slice on the same memory (keyed by the pool's row count; negligible compile)."""
    def write(pool,states,publics,offset):
        update=lambda p,x:jax.lax.dynamic_update_slice_in_dim(p,x,offset,0)
        return jax.tree.map(update,pool[0],states),jax.tree.map(update,pool[1],publics)
    return jax.jit(write,donate_argnums=0)


def prior_maps(legal,logits,requested,k):
    """The `prior` selection: slot -> action maps [...,k] by `ranked` (logit descending, lower action on ties), -1 past the live slots."""
    chosen,live=depth2.ranked(legal,logits,k,requested)
    return j.where(live,chosen,-1).astype(j.int32)


def searched_widths(k_own,k_opp):
    """The fixed map widths `searched_maps` returns for (k_own, k_opp): the learner's k_own plus its prior top-1; the opponent's prior
    top-1 plus one refutation per kept learner move, or the k_opp+1 fill, whichever is wider. Configure the depth-N pricer's ply-2
    widths with these (widths up to 10 are allowed in `searched` mode; nine legal actions bound them at 9)."""
    if any(type(k)is not int or not 1<=k<=9 for k in(k_own,k_opp)):raise ValueError('searched widths are integers in1..9')
    return min(k_own+1,9),min(max(k_opp+1,k_own+2),9)


def _dedupe(items):
    seen=[]
    for x in items:
        if int(x)not in seen:seen.append(int(x))
    return seen


def searched_maps(frame,k_own,k_opp):
    """Iterative-deepening selection from a FULL-WIDTH depth-2 frame: the ply-2 maps (per turn-1 cell) for a `searched` run.

    Per expanded turn-1 cell (`t1_status == 2`), with v the turn-2 leaf
    values, valid leaves those of status 1 or 4, and pi_opp the opponent's
    child prior (a masked softmax over the enumerated replies, as
    `depth2_backups`), every ranking reads VALID leaves only:
    - the learner keeps the k_own a2 with the highest depth-2 learner-max row
      value q(a2) = sum over the row's valid r2 of pi_opp(r2) v(a2, r2), with
      pi_opp renormalized over those r2 (q descending, lower action on ties;
      a row with no valid leaf has no q), plus its prior top-1 (highest child
      logit over its encoder-legal actions, lower action on ties);
    - the opponent keeps, in this order: its prior top-1; the REFUTATION of
      each kept learner move, argmin over the row's valid r2 of v(a2, r2)
      (lower action on ties); then the fill up to k_opp+1 entries by the lowest
      threat t(r2) = min over the kept a2 with a valid leaf in r2 of v(a2, r2)
      (t ascending, lower action on ties). If the top-1 and refutations alone
      exceed k_opp+1 they are all kept. Ranking replies by harm against the
      kept moves (and never dropping a kept move's refutation) is what stops a
      maximin over the kept grid from being optimistic.
    - Errored pairs stay reachable: every (a2, r2) of status 3 puts a2 and r2
      at the FRONT of its seat's set (in pair order), so depth 3 re-steps the
      same state, pair and chance, re-hits the error and the root refuses under
      D's rule, as depth 2 did. A root that depth 2 refuses is flagged in the
      returned `depth2_refused` [R] (by `depth2_backups.refusal_summary`); its
      price is moot, so truncating its ranked entries to the width is harmless.
    A waiting seat keeps [0, -1, ...]; a cell that is not expanded keeps all
    -1. Slot order is the order above. Returns int32 maps [R,2,9,9,w_own],
    [R,2,9,9,w_opp] with (w_own, w_opp) = `searched_widths(k_own, k_opp)`, and
    `depth2_refused`; `{2: (own, opp)}` is the `selection_maps` input of
    `DepthNPricer(selection='searched', k_own=(k_own, ...), k_opp=(k_opp, ...))`,
    whose ply-2 slot widths are these same `searched_widths` (the pricer
    derives them from the k's it is given, so the two calls must share k).
    Pure: numpy and float64 host arithmetic only. Raises ValueError unless
    every expanded cell enumerates its full legal product (a top-k depth-2
    pass has no complete grid to rank).
    """
    w_own,w_opp=searched_widths(k_own,k_opp)
    R=frame.pre.shape[0];own=np.full((R,2,9,9,w_own),-1,np.int32);opp=np.full((R,2,9,9,w_opp),-1,np.int32)
    refused=np.asarray([x['refused']for x in depth2_backups.refusal_summary(frame)['roots']],bool)
    def top1(logits,legal,where):
        legal=[a for a in range(9)if legal[a]]
        if not legal:raise ValueError(f'requested seat with an empty encoder mask at {where}')
        return min(legal,key=lambda a:(-float(logits[a]),a))
    for r in range(R):
        for w in range(2):
            if int(frame.pre[r,w])!=0:continue
            for a1,r1 in zip(*np.nonzero(frame.t1_status[r,w]==2)):
                where=f'root{r} world{w} cell({a1},{r1})';st=frame.t2_status[r,w,a1,r1];v=frame.t2_value[r,w,a1,r1];valid=(st==1)|(st==4)
                own_req=bool(frame.t1_own_requested[r,w,a1,r1]);opp_req=bool(frame.t1_peer_requested[r,w,a1,r1])
                rows=[a for a in range(9)if st[a].any()];cols=[x for x in range(9)if st[:,x].any()]
                full_rows=np.flatnonzero(frame.t1_own_legal[r,w,a1,r1])if own_req else[0];full_cols=np.flatnonzero(frame.t1_peer_legal[r,w,a1,r1])if opp_req else[0]
                if not len(full_rows)or not len(full_cols):raise ValueError(f'requested seat with an empty encoder mask at {where}')
                if rows!=list(full_rows)or cols!=list(full_cols)or not(st[rows][:,cols]!=0).all():raise ValueError(f'searched maps need a full-width depth-2 frame: {where}')
                if own_req:
                    pi=depth2_backups._softmax(frame.t1_peer_logits[r,w,a1,r1],[x in cols for x in range(9)]);q={}
                    for a in rows:
                        vc=[x for x in cols if valid[a,x]]
                        if vc:q[a]=math.fsum(pi[x]*float(v[a,x])for x in vc)/math.fsum(pi[x]for x in vc)
                    kept_own=_dedupe(sorted(q,key=lambda a:(-q[a],a))[:k_own]+[top1(frame.t1_own_logits[r,w,a1,r1],frame.t1_own_legal[r,w,a1,r1],where)])
                else:kept_own=[0]
                if opp_req:
                    best=top1(frame.t1_peer_logits[r,w,a1,r1],frame.t1_peer_legal[r,w,a1,r1],where)
                    refutations=[min((x for x in cols if valid[a,x]),key=lambda x:(float(v[a,x]),x))for a in kept_own if any(valid[a,x]for x in cols)]
                    t={x:min(float(v[a,x])for a in kept_own if valid[a,x])for x in cols if any(valid[a,x]for a in kept_own)}
                    base=_dedupe([best]+refutations);fill=[x for x in sorted(t,key=lambda x:(t[x],x))if x not in base]
                    kept_opp=base+fill[:max(0,k_opp+1-len(base))]
                else:kept_opp=[0]
                pairs=sorted(zip(*np.nonzero(st==3)))
                kept_own=_dedupe([a for a,_ in pairs]+kept_own)[:w_own];kept_opp=_dedupe([x for _,x in pairs]+kept_opp)[:w_opp]
                own[r,w,a1,r1,:len(kept_own)]=kept_own;opp[r,w,a1,r1,:len(kept_opp)]=kept_opp
    return own,opp,refused


def as_depth2_frame(frame):
    """An N=2 frame on the depth-2 surface: leaves land at [a1][r1][a2][r2] through the turn-1 cells' maps; reports gain D2's turn-2 keys."""
    if frame.depth!=2:raise ValueError('only an N=2 frame has a depth-2 surface')
    R=frame.pre.shape[0];k_own,k_opp=frame.k_own[0],frame.k_opp[0]
    own=frame.t1_own_actions;opp=frame.t1_opp_actions
    valid=(own[...,:,None]>=0)&(opp[...,None,:]>=0)
    r,w,a1,r1,i,x=np.nonzero(valid);a2=own[r,w,a1,r1,i];r2=opp[r,w,a1,r1,x];cell=a1*9+r1
    def place(leaf):
        out=np.zeros((R,2,9,9,9,9)+leaf.shape[5:],leaf.dtype);out[r,w,a1,r1,a2,r2]=leaf[r,w,cell,i,x];return out
    reports=tuple(tuple(dict({k:v for k,v in rep.items()if k not in('ply_required','ply_unknown')},
        turn2_required=None if rep.get('ply_required')is None else rep['ply_required'][2],
        turn2_unknown=None if rep.get('ply_unknown')is None else rep['ply_unknown'][2])for rep in pair)for pair in frame.world_reports)
    digest=dict(t2_digest=None if frame.leaf_digest is None else place(frame.leaf_digest),t1_digest_own=frame.t1_digest_own,t1_digest_peer=frame.t1_digest_peer)
    return depth2.Depth2Frame(**{name:getattr(frame,name)for name in T1_FIELDS},t2_status=place(frame.leaf_status),t2_value=place(frame.leaf_value),
        t2_error=place(frame.leaf_error),world_reports=reports,timing=frame.timing,**digest)


def _ply_widths(k,depth,name,limit):
    if type(k)is int:k=(k,)*(depth-1)
    k=tuple(k)
    if len(k)!=depth-1 or any(type(x)is not int or not 1<=x<=limit for x in k):
        raise ValueError(f'{name}: one integer width in1..{limit} per ply2..{depth} required')
    return k


class DepthNPricer:
    def __init__(self,executor,model,*,capacity,depth,k_own,k_opp,k1_own=9,k1_opp=9,block_lanes=2187,blocks_per_call=16,selection='prior',
            fixed_capacity=False,digest=False,forward=None,actor_capacity=33):
        """`k_own`/`k_opp`: one integer per ply 2..N (an int repeats). In `prior` mode they are the slot widths (top-k by prior).
        In `searched` mode the ply-2 entries are the search widths k of `searched_maps(frame, k_own, k_opp)`, and the pricer
        derives the ply-2 SLOT widths from them with `searched_widths` (k_own+1 for the learner, up to k_own+2 for the
        opponent; `self.k_own`/`self.k_opp` hold the slot widths, `self.searched_k` the k's, both logged in `timing`); a
        narrower map pads, a wider one is refused. Plies past 2 are prior widths in both modes. No width exceeds 9.

        `block_lanes` sizes every block in lanes (B = block_lanes // K per ply);
        `blocks_per_call` maps G blocks per device call (a ply's last call pads
        with skipped blocks), so a ply costs about 3 dispatches per call and one
        host sync. `fixed_capacity` compiles P1 at exactly `capacity` so a
        shrinking request count never recompiles; the block programs never
        depend on capacity. `digest` adds bit-exact view digests at every ply;
        `forward(params, inputs)` is the only actor seam (default: the v5 head
        in bf16).
        """
        if type(capacity)is not int or capacity<=0 or type(actor_capacity)is not int or actor_capacity<=0:
            raise ValueError('finite explicit root/actor capacities required')
        if type(depth)is not int or depth<2:raise ValueError('search depth is an integer >=2')
        if selection not in SELECTIONS:raise ValueError(f'selection in {SELECTIONS} required')
        self.k_own=_ply_widths(k_own,depth,'k_own',9);self.k_opp=_ply_widths(k_opp,depth,'k_opp',9);self.searched_k=None
        if selection=='searched':
            self.searched_k=(self.k_own[0],self.k_opp[0]);w_own,w_opp=searched_widths(*self.searched_k)
            self.k_own=(w_own,)+self.k_own[1:];self.k_opp=(w_opp,)+self.k_opp[1:]
        depth2._widths(9,9,k1_own,k1_opp,1)
        if any(type(x)is not int or x<=0 for x in(block_lanes,blocks_per_call)):raise ValueError('positive block lane budget and blocks per call required')
        if any(type(x)is not bool for x in(fixed_capacity,digest)):raise ValueError('explicit digest/fixed-capacity bools required')
        self.executor=executor;self.model=model;self.capacity=capacity;self.actor_capacity=actor_capacity;self.depth=depth
        self.k1_own=k1_own;self.k1_opp=k1_opp;self.block_lanes=block_lanes;self.blocks_per_call=blocks_per_call;self.selection=selection;self.fixed_capacity=fixed_capacity
        self.digest=digest;self.forward=forward if forward is not None else depth2.default_forward(model)
        self.programs={};self.compiles=();self.param_key=None;self.params=None;self.source_params=None;self.last_timing=None

    def bind(self,params):
        if not 0<len(params)<=self.actor_capacity:raise ValueError('declared actor bank exceeds fixed capacity')
        key=tuple(map(id,params))
        if key!=self.param_key:
            self.params=tuple(params);self.param_key=key;self.programs={};self.source_params=tuple(params)

    def _program(self,key,build,*args):
        """Lower and compile once per shape key; later calls reuse the executable (args must keep their shapes and dtypes)."""
        if key not in self.programs:
            fn=build();clock=time.perf_counter();lowered=fn.lower(*args);lower=time.perf_counter()-clock
            clock=time.perf_counter();self.programs[key]=lowered.compile();self.compiles+=((key,lower,time.perf_counter()-clock),)
        return self.programs[key]

    def _block_program(self,p,b):
        k_own,k_opp=self.k_own[p-2],self.k_opp[p-2];leaf=p==self.depth;g=self.blocks_per_call
        return ('leaf'if leaf else'expand',b,g,k_own,k_opp),lambda:make_block_program(self.executor,params=self.params,forward=self.forward,
            nodes=b,blocks=g,k_own=k_own,k_opp=k_opp,leaf=leaf,digest=self.digest)

    def _gather(self,pool,rows):
        rows=j.asarray(rows,j.int32);key=('gather',jax.tree.leaves(pool)[0].shape[0],rows.shape[0])
        return self._program(key,make_gather,pool,rows)(pool,rows)

    def _write(self,pool,states,publics,offset):
        offset=j.asarray(offset,j.int32);key=('write',jax.tree.leaves(pool)[0].shape[0],jax.tree.leaves(states)[0].shape[0])
        return self._program(key,make_write,pool,states,publics,offset)(pool,states,publics,offset)

    def _select(self,legal,logits,requested,k):
        key=('select',legal.shape,k);args=(j.asarray(legal),j.asarray(logits),j.asarray(requested))
        return np.array(self._program(key,lambda:jax.jit(lambda l,g,r:prior_maps(l,g,r,k)),*args)(*args))

    def _maps(self,level,p,given,count,n):
        """Both seats' slot -> action maps [n,M,k] for the nodes of `level` (ply p-1) expanding into ply p: given or prior; -1 unless expanded."""
        k_own,k_opp=self.k_own[p-2],self.k_opp[p-2];M=level['status'].shape[1];expanded=level['status']==2
        if given is not None and p in given:
            maps=[]
            for name,k,legal,requested in(('own',k_own,level['own_legal'],level['own_requested']),('opp',k_opp,level['opp_legal'],level['opp_requested'])):
                source=np.asarray(given[p][0 if name=='own'else 1])
                shape=(count,2,9,9)if p==2 else(count,2,M)
                # A narrower map (searched_widths below the configured width) pads with -1; a wider one cannot be stepped.
                if source.ndim!=len(shape)+1 or source.shape[:-1]!=shape or source.shape[-1]>k:
                    raise ValueError(f'selection map for ply{p} {name} must have shape {shape+(k,)} (last axis at most {k}), got {source.shape}')
                if source.shape[-1]<k:source=np.concatenate((source,np.full(shape+(k-source.shape[-1],),-1,np.int32)),axis=-1)
                m=np.full((n,M,k),-1,np.int32);m[:2*count]=source.reshape(2*count,M,k)
                # Fail closed on a map the engine could not step as enumerated: an entry outside -1..8, an illegal or
                # repeated action, a hole before the padding, an empty requested seat, or anything but [0,-1,...] for a
                # waiting seat. Non-expanded nodes are not read (their maps are cleared below).
                for w,node in zip(*np.nonzero(expanded)):
                    row=m[w,node];where=f'ply{p} {name} map at world{w} node{node}'
                    if row.min()<-1 or row.max()>8:raise ValueError(f'{where}: entries are actions0..8 or -1 padding')
                    live=row>=0;c=int(live.sum())
                    if live[c:].any():raise ValueError(f'{where}: padding must follow the live slots')
                    if not requested[w,node]:
                        if c!=1 or row[0]!=0:raise ValueError(f'{where}: a waiting seat keeps [0,-1,...]')
                        continue
                    if c==0:raise ValueError(f'{where}: a requested seat keeps at least one action')
                    acts=row[:c]
                    if len(set(acts.tolist()))!=c or not legal[w,node][acts].all():raise ValueError(f'{where}: distinct encoder-legal actions required')
                maps.append(m)
            own,opp=maps
        else:
            own=self._select(level['own_legal'],level['own_logits'],level['own_requested'],k_own)
            opp=self._select(level['opp_legal'],level['opp_logits'],level['opp_requested'],k_opp)
        own[~expanded]=-1;opp[~expanded]=-1
        return own,opp

    def price(self,requests,materializer,recipe,params,*,selection_maps=None):
        requests=tuple(requests);count=len(requests)
        if self.selection=='prior'and selection_maps is not None:raise ValueError('selection maps belong to the searched mode')
        if self.selection=='searched'and(not isinstance(selection_maps,dict)or 2 not in selection_maps):
            raise ValueError('searched selection needs selection_maps={2: (own, opp), ...}')
        if selection_maps is not None and any(type(p)is not int or not 2<=p<=self.depth for p in selection_maps):raise ValueError('selection map plies are2..N')
        if not count:return self._frame(None,None,0,(),dict(roots=0,device_invocations=0))
        if type(recipe)is not original.Recipe or type(materializer)is not mapping.Tables:
            raise ValueError('original materializer and random recipe required')
        if count>self.capacity:raise ValueError('one frame exceeds declared root capacity')
        capacity=self.capacity if self.fixed_capacity else min(self.capacity,1<<(count-1).bit_length())
        self.bind(params);started=time.perf_counter();self.compiles=()
        depth2.validate_requests(requests,materializer,recipe,params)
        worlds=batch._numpy_worlds(requests,materializer,recipe);prepared=time.perf_counter()
        available=[w for pair in worlds for w in pair if w.state is not None]
        n=2*capacity;N=self.depth;device=0.;invocations=0;host={};pool=None
        data=depth2.pack_turn1(requests,worlds,recipe,capacity,params,k1_own=self.k1_own,fixed_capacity=self.fixed_capacity)
        if data is not None:
            key=('P1',capacity,tuple(len(x)for x in data['peer_rows']),tuple(len(x)for x in data['leaf_rows']))
            device_data=jax.device_put(data)
            program=self._program(key,lambda:make_p1(self.executor,self.model,params=self.params,capacity=capacity,k1_own=self.k1_own,k1_opp=self.k1_opp,
                digest=self.digest,forward=self.forward),device_data)
            clock=time.perf_counter();out=jax.block_until_ready(program(device_data));device+=time.perf_counter()-clock;invocations=1
            pool=(out.pop('states'),out.pop('publics'));host=jax.device_get(out);device_data=out=None
        node_bytes=self._node_bytes(pool);ply_loop=host_pack=0.
        # Ply 1 as a node level: cell-indexed [n,81] fields, pool rows through each cell's lane.
        shapes=self._shapes()
        t1={name:(np.ascontiguousarray(host[src]).reshape((n,)+shape).astype(dtype)if host else np.full((n,)+shape,UNAVAILABLE if name=='pre'else 0,dtype))
            for name,(src,shape,dtype)in shapes.items()}
        row_of=np.full((n,LANES),-1,np.int32)
        if host:
            w,lane=np.nonzero(host['lane_cell']<LANES);row_of[w,host['lane_cell'][w,lane]]=w*LANES+lane
        level=dict(ply=1,status=t1['t1_status'].reshape(n,LANES),own_requested=t1['t1_own_requested'].reshape(n,LANES),opp_requested=t1['t1_peer_requested'].reshape(n,LANES),
            own_legal=t1['t1_own_legal'].reshape(n,LANES,9),opp_legal=t1['t1_peer_legal'].reshape(n,LANES,9),own_logits=t1['t1_own_logits'].reshape(n,LANES,9),
            opp_logits=t1['t1_peer_logits'].reshape(n,LANES,9),pool=pool,row_of=row_of)
        pool=None  # the level owns the turn-1 children now; the loop drops them once ply 2 has read them
        owners=np.repeat(data['owner']if data is not None else np.zeros(capacity,np.int32),2)
        actors=np.zeros((n,2),np.int32);actors[:2*count]=np.repeat([(r.learner_actor,r.opponent_actor)for r in requests],2,axis=0)
        levels=[level];work=[]
        for p in range(2,N+1):
            k_own,k_opp=self.k_own[p-2],self.k_opp[p-2];K=k_own*k_opp;B=block_nodes(self.block_lanes,k_own,k_opp);leaf=p==N
            own_maps,opp_maps=self._maps(level,p,selection_maps,count,n)
            level['own_actions']=own_maps;level['opp_actions']=opp_maps
            pw,pid=np.nonzero(level['status']==2)
            group=actors[pw,0]*(1 if leaf else len(params))+(0 if leaf else actors[pw,1])
            order=np.lexsort((pid,pw,group));pw,pid,group=pw[order],pid[order],group[order]
            blocks=[]
            for g in np.unique(group):
                members=np.flatnonzero(group==g)
                for start in range(0,len(members),B):
                    blocks.append(members[start:start+B])
            # Calls of G blocks: one gather from the old pool, one block program, one in-place write into the new pool,
            # no host sync until the ply is done. Block g of call c is global block c*G+g; its slot s holds pool rows
            # ((c*G+g)*B+s)*K + lane, which is also each small field's row. The pool is preallocated for the call count
            # rounded up to a power of two, so the gather/write programs (keyed by pool rows) form a bounded, fixed set.
            key,build=self._block_program(p,B);G=self.blocks_per_call;nblocks=len(blocks);calls=-(-nblocks//G)
            pool_rows=(1<<(calls-1).bit_length())*G*B*K if calls else 0
            new_pool=None if leaf or not calls else jax.tree.map(lambda x:j.zeros((pool_rows,)+x.shape[1:],x.dtype),level['pool'])
            small=[];started_ply=time.perf_counter()
            for c in range(calls):
                clock=time.perf_counter()
                fill=np.zeros((G,B),np.int64);live=np.zeros((G,B),bool);call_actors=np.zeros((G,2),np.int32)
                for gi,members in enumerate(blocks[c*G:(c+1)*G]):
                    m=len(members);fill[gi,:m]=members;fill[gi,m:]=members[0];live[gi,:m]=True;call_actors[gi]=actors[pw[members[0]]]
                fill=fill.reshape(-1);bw,bid=pw[fill],pid[fill]
                requested=np.zeros((G*B,2),bool);requested[np.arange(G*B),owners[bw]]=level['own_requested'][bw,bid];requested[np.arange(G*B),1-owners[bw]]=level['opp_requested'][bw,bid]
                rows,own_block,opp_block=level['row_of'][bw,bid],own_maps[bw,bid],opp_maps[bw,bid];host_pack+=time.perf_counter()-clock
                states,publics=self._gather(level['pool'],rows)
                args=(states,publics,j.asarray(owners[bw],j.int32),j.asarray(requested),j.asarray(own_block,j.int32),j.asarray(opp_block,j.int32),
                    j.asarray(live.reshape(-1)),j.asarray(call_actors))
                out=self._program(key,build,*args)(*args);invocations+=2
                if not leaf:new_pool=self._write(new_pool,out.pop('states'),out.pop('publics'),c*G*B*K);invocations+=1
                small.append(out);del states,publics,args,out
            level['pool']=None
            small=jax.device_get(small);ply_loop+=time.perf_counter()-started_ply
            fields={name:np.concatenate([s[name]for s in small])for name in small[0]}if calls else{}
            M=level['status'].shape[1]*K
            # Dense placement: child id = parent*K + own_slot*k_opp + opp_slot; row = global block slot*K + lane.
            slot=np.concatenate([np.arange(len(b))+g*B for g,b in enumerate(blocks)])if nblocks else np.zeros(0,np.int64)
            members=np.concatenate(blocks)if nblocks else np.zeros(0,np.int64);lw,lid=pw[members],pid[members]
            stats=dict(ply=p,parents=len(pw),blocks=nblocks,calls=calls,blocks_per_call=G,lanes=nblocks*B*K,evaluated_lanes=calls*G*B*K,nodes_per_world=M)
            if leaf:
                out={name:np.zeros((n,level['status'].shape[1],k_own,k_opp)+shape,dtype)for name,shape,dtype in(('status',(),np.int8),('value',(),np.float32),('error',(),np.int8))}
                if self.digest:out['digest']=np.zeros((n,level['status'].shape[1],k_own,k_opp,2),np.uint32)
                for name,x in out.items():
                    if nblocks:x[lw,lid]=fields[name].reshape((calls*G*B,k_own,k_opp)+x.shape[4:])[slot]
                leaves=out;work.append(dict(stats,enumerated=int(np.count_nonzero(out['status']))))
                break
            ids=(lid[:,None]*K+np.arange(K)).reshape(-1);rw=np.repeat(lw,K);src=(slot[:,None]*K+np.arange(K)).reshape(-1)
            nxt=dict(ply=p,pool=new_pool,row_of=np.full((n,M),-1,np.int32))
            nxt['row_of'][rw,ids]=src
            specs=self._node_specs()
            if self.digest:specs.update(digest_own=((2,),np.uint32),digest_opp=((2,),np.uint32))
            for name,(shape,dtype)in specs.items():
                x=np.zeros((n,M)+shape,dtype)
                if nblocks:x[rw,ids]=fields[name].reshape((calls*G*B*K,)+shape)[src]
                nxt[name]=x
            work.append(dict(stats,enumerated=int(np.count_nonzero(nxt['status'])),pool_rows=pool_rows))
            levels.append(nxt);level=nxt
        finished=time.perf_counter()
        # Device seconds: P1's synced call plus each ply's loop net of its measured host packing (the loop dispatches
        # every call and syncs once, so device work overlapping the packing of later calls is counted once, not twice).
        timing=dict(roots=count,world_prepare_seconds=prepared-started,batch_pack_and_device_seconds=finished-prepared,device_invocations=invocations,
            device_seconds=device+ply_loop-host_pack,ply_loop_seconds=ply_loop,host_pack_seconds=host_pack,
            timing_notes='device_seconds = synced P1 + sum over plies of (ply_loop_seconds - host_pack_seconds); ply_loop_seconds = host packing, dispatch and the one '
                'sync per ply; host_pack_seconds = numpy call packing (index, request and map arrays) inside the loops',
            lower_seconds=sum(c[1]for c in self.compiles),compile_seconds=sum(c[2]for c in self.compiles),compiled=tuple(c[0]for c in self.compiles),
            programs=tuple(self.programs),actor_strategy='DEPTHN_D3_TURN1_ARCHIVE_BATCHES_BLOCKED_PLIES',root_capacity=capacity,frame_wall_seconds=finished-started,
            depth=N,k_own=self.k_own,k_opp=self.k_opp,searched_k=self.searched_k,searched_widths=None if self.searched_k is None else(self.k_own[0],self.k_opp[0]),
            k1_own=self.k1_own,k1_opp=self.k1_opp,block_lanes=self.block_lanes,blocks_per_call=self.blocks_per_call,selection=self.selection,
            fixed_capacity=self.fixed_capacity,evaluated_root_slots=0 if data is None else capacity,padding_root_slots=0 if data is None else capacity-count,
            substituted_world_slots=0 if data is None else 2*count-len(available),pre_refused_available_worlds=int(np.count_nonzero(t1['pre'][:2*count]>0)),
            plies=tuple(work),enumerated_leaves=work[-1]['enumerated']if work else 0,evaluated_leaf_lanes=work[-1]['lanes']if work else 0,
            bytes_per_node_state=node_bytes)
        self.last_timing=timing
        frame=self._frame(t1,levels,count,leaves,timing)
        reports=tuple(self._reports(i,r,pair,frame)for i,(r,pair)in enumerate(zip(requests,worlds)))
        return replace(frame,world_reports=reports)

    @staticmethod
    def _node_bytes(pool):
        if pool is None:return None
        leaves=jax.tree.leaves(pool)
        return int(sum(x.dtype.itemsize*int(np.prod(x.shape[1:]))for x in leaves))if leaves else 0

    def _shapes(self):
        shapes=dict(pre=('pre',(),np.int32),root_peer_requested=('peer_requested',(),bool),root_peer_legal=('peer_legal',(9,),bool),root_peer_logits=('peer_logits',(9,),np.float32),
            t1_status=('t1_status',(9,9),np.int8),t1_value=('t1_value',(9,9),np.float32),t1_error=('t1_error',(9,9),np.int8),
            t1_own_requested=('t1_own_requested',(9,9),bool),t1_own_legal=('t1_own_legal',(9,9,9),bool),t1_own_logits=('t1_own_logits',(9,9,9),np.float32),
            t1_peer_requested=('t1_peer_requested',(9,9),bool),t1_peer_legal=('t1_peer_legal',(9,9,9),bool),t1_peer_logits=('t1_peer_logits',(9,9,9),np.float32))
        if self.digest:shapes.update(t1_digest_own=('t1_digest_own',(9,9,2),np.uint32),t1_digest_peer=('t1_digest_peer',(9,9,2),np.uint32))
        return shapes

    def _frame(self,t1,levels,count,leaves,timing):
        """Slice the host levels to the real roots as the dense [R,2,...] surface (zero-size at count 0)."""
        N=self.depth;R=count;rooted=lambda x:np.ascontiguousarray(x[:2*R]).reshape((R,2)+x.shape[1:])
        if t1 is None:
            t1={name:np.full((0,)+shape,0,dtype)for name,(_,shape,dtype)in self._shapes().items()}
            M=[LANES]
            for p in range(2,N):M.append(M[-1]*self.k_own[p-2]*self.k_opp[p-2])
            levels=[dict(own_actions=np.zeros((0,LANES,self.k_own[0]),np.int32),opp_actions=np.zeros((0,LANES,self.k_opp[0]),np.int32))]
            for p in range(2,N):
                lv={name:np.zeros((0,M[p-1])+shape,dtype)for name,(shape,dtype)in self._node_specs().items()}
                lv.update(own_actions=np.zeros((0,M[p-1],self.k_own[p-1]),np.int32),opp_actions=np.zeros((0,M[p-1],self.k_opp[p-1]),np.int32));levels.append(lv)
            leaves={name:np.zeros((0,M[-1],self.k_own[-1],self.k_opp[-1])+shape,dtype)for name,shape,dtype in(('status',(),np.int8),('value',(),np.float32),('error',(),np.int8))}
            if self.digest:leaves['digest']=np.zeros((0,M[-1],self.k_own[-1],self.k_opp[-1],2),np.uint32)
        fields={name:rooted(x)for name,x in t1.items()}
        fields.update(t1_own_actions=rooted(levels[0]['own_actions']).reshape(R,2,9,9,self.k_own[0]),t1_opp_actions=rooted(levels[0]['opp_actions']).reshape(R,2,9,9,self.k_opp[0]))
        nodes=[]
        for p in range(2,N):
            lv=levels[p-1];K=self.k_own[p-2]*self.k_opp[p-2];M=lv['status'].shape[1];ids=np.arange(M,dtype=np.int32)
            nodes.append(PlyNodes(ply=p,parent=ids//K,parent_own_slot=((ids%K)//self.k_opp[p-2]).astype(np.int8),parent_opp_slot=(ids%self.k_opp[p-2]).astype(np.int8),
                own_actions=rooted(lv['own_actions']),opp_actions=rooted(lv['opp_actions']),**{name:rooted(lv[name])for name in NODE_FIELDS},
                **({'digest_own':rooted(lv['digest_own']),'digest_opp':rooted(lv['digest_opp'])}if self.digest else{})))
        return DepthNFrame(depth=N,k_own=self.k_own,k_opp=self.k_opp,nodes=tuple(nodes),leaf_status=rooted(leaves['status']),leaf_value=rooted(leaves['value']),
            leaf_error=rooted(leaves['error']),leaf_digest=rooted(leaves['digest'])if self.digest else None,world_reports=(),timing=timing,**fields)

    @staticmethod
    def _node_specs():
        return dict(status=((),np.int8),value=((),np.float32),error=((),np.int8),own_requested=((),bool),opp_requested=((),bool),
            own_legal=((9,),bool),opp_legal=((9,),bool),own_logits=((9,),np.float32),opp_logits=((9,),np.float32))

    def _reports(self,index,request,worlds,frame):
        N=self.depth;reports=[]
        for wi,(start,world)in enumerate(zip(request.plans.starts,worlds)):
            report=dict(world=start.world,posterior_seed=start.seed,kernel_seed=world.kernel_seed,status='REFUSED',reason=None,
                required_pairs=None,completed_pairs=0,unknown_pairs=None,ply_required=None,ply_unknown=None,
                reply_history_approximation=None if world.state is None or world.reply_private_history_available else serial.REPLY_HISTORY_APPROXIMATION)
            reports.append(report)
            if world.state is None:report['reason']='physical-materialization:'+','.join(world.refusals);continue
            code=int(frame.pre[index,wi])
            if code<0:report['reason']='no world state';continue
            if code:
                reason=PRE_REASONS.get(code)
                if reason is None:reason='actual learner root input differs: '+tuple(k for k in INPUT_KEYS if k in request.root_inputs)[code-3]
                report['reason']='ValueError: '+reason;continue
            status=frame.t1_status[index,wi];required=int(np.count_nonzero(status));unknown=int(np.count_nonzero(status==3))
            deeper={p:frame.nodes[p-2].status[index,wi]for p in range(2,N)};deeper[N]=frame.leaf_status[index,wi]
            ply_required={p:int(np.count_nonzero(s))for p,s in deeper.items()};ply_unknown={p:int(np.count_nonzero(s==3))for p,s in deeper.items()}
            errors=tuple((int(a),int(r),'ValueError: '+ERROR_REASONS[int(frame.t1_error[index,wi,a,r])])for a,r in zip(*np.nonzero(status==3)))
            complete=not unknown and not any(ply_unknown.values())
            report.update(status='COMPLETE'if complete else'REFUSED',required_pairs=required,completed_pairs=required-unknown,unknown_pairs=unknown,
                ply_required=ply_required,ply_unknown=ply_unknown,errors=errors,
                opponent_requested=bool(frame.root_peer_requested[index,wi]),reply_legal=tuple(map(bool,frame.root_peer_legal[index,wi])),
                reply_logits=tuple(map(float,frame.root_peer_logits[index,wi])),reason=None if complete else'unknown-required-public-leaf')
        return tuple(reports)
