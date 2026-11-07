"""Depth-2 D prices: D3's turn-1 frame, then one more joint ply under every child.

Turn 1 is FramePricer(vectorized=True) unchanged: the same views and pre codes,
the same pricing-order x ascending-reply pair layout padded with the first pair,
the same world chance capsule and the same [n,81] bf16 leaf forward, so
`t1_value` and the learner's child logits are D3's bits. The joint keeps what
D3 discards: each child's engine state and trackers, both seats' child masks
and the opponent's child view, whose prior is one more archive-grouped forward.
Turn 2 ranks each seat's legal actions by its child prior, steps the top
k_own x k_opp pairs under the engine's own child draw (executor.sample of the
child; a deferred turn resumes its saved draws inside the engine) and values
the learner's leaf view with its own params. Children are processed in
learner-archive-homogeneous blocks under jax.lax.map, so the second ply's
device memory is bounded by child_block*k_own*k_opp leaf lanes whatever the
frame size; a block without an expanded child is skipped. Padding (roots,
worlds, lanes, k slots, block slots) is computed but never enumerated.

The turn-1 stage (`make_turn1_stage`), the top-k rule (`ranked`) and the host
packing (`validate_requests`, `pack_turn1`) are module-level so the depth-N
engine (`device_depthn_pricer`) reuses them unchanged as its program P1.
"""
from dataclasses import dataclass
import time

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_gumbel as original
from . import device_public_teacher as serial,device_public_teacher_batch as batch,device_public_world_state as mapping
from . import depth2_digest as digests,v5_model

INPUT_KEYS=batch.SERIAL_INPUT_ORDER
LANES=serial.JOINT_LANES
# Codes beyond D's. `pre` -1: a world slot with no state (materialization refusal
# or frame padding), never usable. `t1_error` 5/6: the opponent's child view
# failed (encoder error / prior refused while requested), so the cell is not
# expanded; D3 still priced it, and `t1_value`/`t1_own_*` keep D3's bits.
UNAVAILABLE=-1
PEER_VIEW_ERROR=5;PEER_PRIOR_REFUSED=6
PRE_REASONS={1:'current direct-encoder input error',2:'same actual root legal-action denominator required',8:'current direct-encoder input error',
    9:'current public actor output refused/nonfinite',10:'actual learner root input fields differ'}
ERROR_REASONS={1:'hypothetical original action route invalid',2:'hypothetical terminal winner unavailable',
    3:'current direct-encoder input error',4:'current public actor output refused/nonfinite',
    PEER_VIEW_ERROR:'opponent child view direct-encoder input error',PEER_PRIOR_REFUSED:'opponent child prior refused/nonfinite'}


@dataclass(frozen=True)
class Depth2Frame:
    """The contract's numpy surface, leading axes [root, world]; digest fields are None unless requested."""
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
    t2_status:object
    t2_value:object
    t2_error:object
    world_reports:tuple
    timing:dict
    t2_digest:object=None
    t1_digest_own:object=None
    t1_digest_peer:object=None


def _make_child_joint(executor):
    """D3's _make_joint with the same dispatch/advance/observe sequence, keeping the child.

    The extra outputs (child state and trackers, the engine request masks, both
    seats' encoder masks, the opponent's child view) are independent
    computations after the same step, so the learner's leaf view is unchanged.
    The one-row program is still mapped, never flattened, so public tracker
    namespaces stay per world.
    """
    def single(state,public,own,peer,owner,requested,pair,chance):
        actions=j.full((1,2),-1,j.int32)
        chosen=executor.dispatch(own,state,owner,pair[:1])
        actions=actions.at[:,owner[0]].set(chosen)
        reply=executor.dispatch(peer,state,1-owner,pair[1:])
        actions=actions.at[:,1-owner[0]].set(j.where(requested,reply,-1))
        after=executor.advance(state,public,actions,chance)
        masks=executor.legal(after.step.state)
        view=executor.observe(after.step.state,after.trackers,owner,masks[:,owner[0]])
        foe=executor.observe(after.step.state,after.trackers,1-owner,masks[:,1-owner[0]])
        return (after.step.invalid_actions,after.step.state.done,after.step.state.winner,view.observation.error,view.inputs,
            after.step.state,after.trackers,masks,view.observation.legal_action_mask,foe.observation.legal_action_mask,foe.observation.error,foe.inputs)
    return jax.jit(jax.vmap(single,in_axes=(None,None,None,None,None,None,0,None)))


def _make_leaf_joint(executor):
    """Turn 2: both seats may be waiting, so each seat's engine action is -1 unless it is requested."""
    def single(state,public,own,peer,owner,requested,pair,chance):
        actions=j.full((1,2),-1,j.int32)
        chosen=executor.dispatch(own,state,owner,pair[:1])
        actions=actions.at[:,owner[0]].set(j.where(requested[owner[0]],chosen,-1))
        reply=executor.dispatch(peer,state,1-owner,pair[1:])
        actions=actions.at[:,1-owner[0]].set(j.where(requested[1-owner[0]],reply,-1))
        after=executor.advance(state,public,actions,chance)
        masks=executor.legal(after.step.state)
        view=executor.observe(after.step.state,after.trackers,owner,masks[:,owner[0]])
        return after.step.invalid_actions,after.step.state.done,after.step.state.winner,view.observation.error,view.inputs
    return jax.vmap(single,in_axes=(None,None,None,None,None,None,0,None))


def _make_views(executor):
    """Both seats' observations of one world (dispatch views and encoder inputs), from the engine masks."""
    def views(state,public,owner):
        masks=executor.legal(state);seat=j.asarray([owner],j.int32)
        own=executor.observe(state,public,seat,masks[:,owner])
        peer=executor.observe(state,public,1-seat,masks[:,1-owner])
        return own.observation,peer.observation,own.inputs,peer.inputs
    return views


def ranked(mask,logits,k,requested):
    """Top-k by prior over the last axis: logit descending, lower action index on ties (a stable sort of -logit).

    Returns the chosen actions [...,k] and the live slots. A waiting seat enumerates column 0 alone; k slots past the
    legal count repeat the first choice and are never live.
    """
    order=j.argsort(j.where(mask&j.isfinite(logits),-logits,j.inf),axis=-1,stable=True)
    count=j.where(requested,j.minimum(j.sum(mask,axis=-1),k),1)
    slot=j.arange(k);live=slot<count[...,None]
    chosen=j.take_along_axis(order,j.where(live,slot,0),axis=-1)
    return j.where(requested[...,None],chosen,0),live


def _widths(k_own,k_opp,k1_own,k1_opp,child_block):
    if any(type(k)is not int or not 1<=k<=9 for k in(k_own,k_opp,k1_own,k1_opp)):raise ValueError('turn widths are integers in1..9')
    if type(child_block)is not int or child_block<=0:raise ValueError('positive child block required')


def default_forward(model):
    def forward(p,inputs):return v5_model.forward(p,inputs,model,compute_dtype='bfloat16')
    return forward


def make_turn1_stage(executor,model,*,params,capacity,k1_own=9,k1_opp=9,digest=False,forward=None):
    """D3's turn-1 frame with the child carry: the depth-2 kernel's first stage, traced inside the caller's jit.

    `turn1(a)` returns the cell-indexed [capacity,2,9,9,...] turn-1 fields under
    'frame' (D3's bits for `t1_value` and the learner's child prior) and the
    lane-indexed [n,81] carry a second stage needs: the children's states and
    trackers, both seats' request flags, encoder masks and priors, the
    enumeration and the pairs. `forward(params, inputs)` is the only actor
    seam; it defaults to the v5 head in bf16 and is returned for the caller's
    later plies.
    """
    _widths(9,9,k1_own,k1_opp,1)
    if type(capacity)is not int or capacity<=0 or type(digest)is not bool:raise ValueError('explicit root capacity and digest bool required')
    if forward is None:forward=default_forward(model)
    n=capacity*2
    child_joint=_make_child_joint(executor);views=_make_views(executor)
    def grouped_forward(inputs,rows,*,joint=False):
        # D3's archive batches, verbatim: the current actor gets one dense
        # batch, historical archives share one padded geometry under vmap.
        lanes=LANES if joint else 1
        logits=j.zeros((n,lanes,9),j.float32);values=j.zeros((n,lanes),j.float32);valid=j.zeros((n,lanes),bool)
        def evaluate(p,indices):
            count=indices.shape[0]
            selected=jax.tree.map(lambda x:x[j.minimum(indices,n-1)],inputs)
            flat=jax.tree.map(lambda x:x.reshape((count*lanes,)+x.shape[3 if joint else 2:]),selected)
            prediction=forward(p,flat)
            return (prediction.policy_logits.reshape(count,lanes,9),prediction.value.reshape(count,lanes),prediction.valid.reshape(count,lanes))
        if rows[0].shape[0]:
            lp,vp,ok=evaluate(params[0],rows[0])
            logits=logits.at[rows[0]].set(lp,mode='drop');values=values.at[rows[0]].set(vp,mode='drop');valid=valid.at[rows[0]].set(ok,mode='drop')
        if len(params)>1 and rows[1].shape[0]:
            indices=j.stack(rows[1:]);weights=jax.tree.map(lambda *xs:j.stack(xs),*params[1:])
            lp,vp,ok=jax.vmap(evaluate)(weights,indices)
            logits=logits.at[indices].set(lp,mode='drop');values=values.at[indices].set(vp,mode='drop');valid=valid.at[indices].set(ok,mode='drop')
        return logits,values,valid
    def turn1(a):
        states,publics=a['states'],a['publics']
        flatten=lambda x:x.reshape((n,)+x.shape[2:])
        states=jax.tree.map(flatten,states);publics=jax.tree.map(flatten,publics)
        owners=j.repeat(a['owner'],2)
        own,peer,own_inputs,peer_inputs=jax.vmap(views)(states,publics,owners)
        root_legal=j.repeat(a['legal'],2,axis=0)
        legal=peer.legal_action_mask[:,0];requested=j.any(legal,axis=-1)
        pre=j.where(own.error[:,0]!=0,1,0)
        pre=j.where((pre==0)&~j.all(own.legal_action_mask[:,0]==root_legal,axis=-1),2,pre)
        if set(own_inputs)!=set(a['expected']):pre=j.where(pre==0,10,pre)
        for index,key in enumerate(k for k in INPUT_KEYS if k in a['expected']):
            if key not in own_inputs:continue
            expected=j.repeat(a['expected'][key],2,axis=0);value=own_inputs[key]
            same=j.all(value==expected,axis=tuple(range(1,value.ndim)))
            pre=j.where((pre==0)&~same,3+index,pre)
        pre=j.where((pre==0)&(peer.error[:,0]!=0),8,pre)
        logits,pv,valid=grouped_forward(peer_inputs,a['peer_rows'])
        logits=logits.reshape(n,9);pv=pv.reshape(n);valid=valid.reshape(n)
        usable=valid&j.all(j.isfinite(logits),axis=-1)&j.isfinite(pv)&(j.abs(pv)<=1)
        pre=j.where((pre==0)&requested&~usable,9,pre)
        logits=j.where(requested[:,None],logits,0.)
        kept=legal
        if k1_opp<9:
            # Optional turn-1 width: the top replies by root prior, in D's ascending order.
            rank=j.argsort(j.argsort(j.where(legal,-logits,j.inf),axis=-1,stable=True),axis=-1,stable=True)
            kept=legal&(rank<k1_opp)
        reply_count=j.where(requested,j.sum(kept,axis=-1),1)
        reply_order=j.argsort(j.where(kept,j.arange(9),j.arange(9)+9),axis=-1)
        reply_order=j.where(requested[:,None],reply_order,0)
        order=j.repeat(a['order'],2,axis=0);sizes=j.repeat(a['order_size'],2)
        ordinal=j.broadcast_to(j.arange(LANES),(n,LANES));required=sizes*reply_count
        action_row=j.where(ordinal<required[:,None],ordinal//reply_count[:,None],0)
        reply_row=j.where(ordinal<required[:,None],ordinal%reply_count[:,None],0)
        pairs=j.stack((j.take_along_axis(order,action_row,axis=1),
            j.take_along_axis(reply_order,reply_row,axis=1)),axis=-1)
        chance=jax.vmap(executor.sample)(states)
        (invalid,done,winner,error,inputs,child_states,child_publics,masks,own_legal2,peer_legal2,peer_error,peer_child_inputs
            )=jax.vmap(child_joint)(states,publics,own,peer,owners[:,None],requested[:,None],pairs,chance)
        invalid,done,winner,error,peer_error=(x[:,:,0]for x in(invalid,done,winner,error,peer_error))
        leaf_logits,values,valid=grouped_forward(inputs,a['leaf_rows'],joint=True)
        usable=valid&j.all(j.isfinite(leaf_logits),axis=-1)&j.isfinite(values)&(j.abs(values)<=1)
        terminal_ok=(winner>=0)&(winner<=2)
        errors=j.where(invalid,1,j.where(done&~terminal_ok,2,j.where(~done&(error!=0),3,j.where(~done&~usable,4,0))))
        terminal=j.where(winner==2,0.,j.where(winner==owners[:,None],1.,-1.))
        values=j.where(done,terminal,values)
        # D3 ends here; `values`, `errors` and `leaf_logits` are its bits.
        # The opponent's child view is held to the root's rule (an encoder
        # error, or a refused prior while requested) but with its own codes,
        # so D3's priced surface survives: such a cell keeps the learner's
        # value and prior, is not expanded, and refuses through the backups.
        priced=errors==0;viewed=priced&~done
        # Request flags come from the engine mask (robust when a view errors).
        # Legal masks are each seat's ENCODER mask: the request-order lane
        # space the policy logits and executor.dispatch use, as D at the root.
        # The engine mask orders switch lanes by physical bench slot, which
        # differs once a side has switched, so it is never enumerated.
        masks=masks[:,:,0];seat=owners[:,None,None,None]
        own_req2=j.any(j.take_along_axis(masks,seat,axis=2)[:,:,0],axis=-1);peer_req2=j.any(j.take_along_axis(masks,1-seat,axis=2)[:,:,0],axis=-1)
        own_legal2=own_legal2[:,:,0];peer_legal2=peer_legal2[:,:,0]
        peer_logits2,peer_pv2,peer_valid2=grouped_forward(peer_child_inputs,a['peer_rows'],joint=True)
        peer_usable2=peer_valid2&j.all(j.isfinite(peer_logits2),axis=-1)&j.isfinite(peer_pv2)&(j.abs(peer_pv2)<=1)
        errors=j.where(viewed&peer_req2&(peer_error!=0),PEER_VIEW_ERROR,errors)
        errors=j.where(viewed&(errors==0)&peer_req2&~peer_usable2,PEER_PRIOR_REFUSED,errors)
        pre=j.where(a['available'].reshape(n),pre,UNAVAILABLE)
        enumerated=(ordinal<required[:,None])&(pre==0)[:,None];priced&=enumerated;viewed&=enumerated
        status1=j.where(~enumerated,0,j.where(errors!=0,3,j.where(done,1,2))).astype(j.int8);expanded=status1==2
        errors=j.where(enumerated,errors,0).astype(j.int8)
        own_logits2=j.where(own_req2[...,None],leaf_logits,0.);peer_logits2=j.where(peer_req2[...,None],peer_logits2,0.)
        a1=j.where(enumerated,pairs[...,0],9);r1=j.where(enumerated,pairs[...,1],9)
        def scatter1(x):
            return j.zeros((n,9,9)+x.shape[2:],x.dtype).at[j.arange(n)[:,None],a1,r1].set(x,mode='drop')
        def rooted(x):return x.reshape((capacity,2)+x.shape[1:])
        usable_world=(pre==0)[:,None]
        # Child-level fields follow what is known at the child: the learner's
        # side and both seats' masks wherever its view is D3-valid (`viewed`,
        # so also under codes 5/6); the opponent's prior only when expanded.
        keep=lambda mask,x:j.where(mask.reshape(mask.shape+(1,)*(x.ndim-2)),x,j.zeros((),x.dtype))
        frame=dict(pre=rooted(pre),
            peer_requested=rooted(requested&usable_world[:,0]),peer_legal=rooted(legal&usable_world),peer_logits=rooted(j.where(usable_world,logits,0.)),
            t1_status=rooted(scatter1(status1)),t1_value=rooted(scatter1(j.where(priced,values,0.))),t1_error=rooted(scatter1(errors)),
            t1_own_requested=rooted(scatter1(keep(viewed,own_req2))),t1_own_legal=rooted(scatter1(keep(viewed,own_legal2))),t1_own_logits=rooted(scatter1(keep(viewed,own_logits2))),
            t1_peer_requested=rooted(scatter1(keep(viewed,peer_req2))),t1_peer_legal=rooted(scatter1(keep(viewed,peer_legal2))),t1_peer_logits=rooted(scatter1(keep(expanded,peer_logits2))))
        if digest:
            lane=lambda x:j.where(enumerated[...,None],x,j.uint32(0))
            frame.update(t1_digest_own=rooted(scatter1(lane(digests.batched(inputs,axes=2)))),
                t1_digest_peer=rooted(scatter1(lane(digests.batched(peer_child_inputs,axes=2)))))
        return dict(frame=frame,pre=pre,pairs=pairs,enumerated=enumerated,expanded=expanded,owners=owners,
            own_req2=own_req2,peer_req2=peer_req2,own_legal2=own_legal2,peer_legal2=peer_legal2,own_logits2=own_logits2,peer_logits2=peer_logits2,
            child_states=child_states,child_publics=child_publics)
    return turn1,forward


def make_depth2_kernel(executor,model,*,params,capacity,k_own=9,k_opp=9,k1_own=9,k1_opp=9,child_block=27,digest=False,forward=None):
    """One device invocation: D3's turn-1 frame, then blocked turn-2 leaves.

    `forward(params, inputs)` is the only actor seam (turn-1 leaves, both child
    priors, turn-2 leaves); it defaults to the v5 head in bf16. Block slots and
    actors come from the host (`Depth2Pricer._blocks`), grouped by the learner's
    archive so each block runs one parameter tree.
    """
    _widths(k_own,k_opp,k1_own,k1_opp,child_block)
    turn1,forward=make_turn1_stage(executor,model,params=params,capacity=capacity,k1_own=k1_own,k1_opp=k1_opp,digest=digest,forward=forward)
    n=capacity*2;children=n*LANES;pairs2=k_own*k_opp;cb=child_block
    leaf_joint=_make_leaf_joint(executor);views=_make_views(executor)
    def run(a):
        t=turn1(a);pre,pairs,enumerated,expanded,owners=t['pre'],t['pairs'],t['enumerated'],t['expanded'],t['owners']
        own_req2,peer_req2=t['own_req2'],t['peer_req2']
        sel_own,live_own=ranked(t['own_legal2'],t['own_logits2'],k_own,own_req2)
        sel_opp,live_opp=ranked(t['peer_legal2'],t['peer_logits2'],k_opp,peer_req2)
        # Turn 2 over flattened children. Each block gathers its children
        # inside the conditional, so a skipped block moves nothing and one
        # block of states, views and leaves is live at a time.
        flat=lambda x:x.reshape((children,)+x.shape[2:])
        child_states=jax.tree.map(flat,t['child_states']);child_publics=jax.tree.map(flat,t['child_publics'])
        owners_child=j.repeat(owners,LANES)
        req_seat=flat(j.where(owners[:,None,None]==j.arange(2),own_req2[...,None],peer_req2[...,None]))
        sel_own_c,sel_opp_c,expanded_c=flat(sel_own),flat(sel_opp),flat(expanded)
        stacked=params[0]if len(params)==1 else jax.tree.map(lambda *xs:j.stack(xs),*params)
        actor_params=lambda actor:params[0]if len(params)==1 else jax.tree.map(lambda x:x[actor],stacked)
        def block(args):
            slots,actor=args;idx=j.minimum(slots,children-1);live=expanded_c[idx]&(slots<children)
            def compute():
                pick=lambda x:x[idx]
                bs=jax.tree.map(pick,child_states);bp=jax.tree.map(pick,child_publics);owner=owners_child[idx];req=req_seat[idx]
                pair=j.stack((j.broadcast_to(sel_own_c[idx][:,:,None],(cb,k_own,k_opp)),
                    j.broadcast_to(sel_opp_c[idx][:,None,:],(cb,k_own,k_opp))),axis=-1).reshape(cb,pairs2,2)
                # Re-observing the child gives the same dispatch views the
                # turn-1 joint saw, without holding [children] encodings.
                bo,bpe,_,_=jax.vmap(views)(bs,bp,owner)
                chance=jax.vmap(executor.sample)(bs)
                invalid,done,winner,error,inputs=jax.vmap(leaf_joint)(bs,bp,bo,bpe,owner[:,None],req,pair,chance)
                invalid,done,winner,error=(x[:,:,0]for x in(invalid,done,winner,error))
                prediction=forward(actor_params(actor),jax.tree.map(lambda x:x.reshape((cb*pairs2,)+x.shape[3:]),inputs))
                lg=prediction.policy_logits.reshape(cb,pairs2,9);v=prediction.value.reshape(cb,pairs2);ok=prediction.valid.reshape(cb,pairs2)
                usable=ok&j.all(j.isfinite(lg),axis=-1)&j.isfinite(v)&(j.abs(v)<=1)
                terminal_ok=(winner>=0)&(winner<=2)
                err=j.where(invalid,1,j.where(done&~terminal_ok,2,j.where(~done&(error!=0),3,j.where(~done&~usable,4,0)))).astype(j.int8)
                terminal=j.where(winner==2,0.,j.where(winner==owner[:,None],1.,-1.))
                out=(j.where(err!=0,0.,j.where(done,terminal,v)).astype(j.float32),j.where(err!=0,3,j.where(done,1,4)).astype(j.int8),err)
                return out+((digests.batched(inputs,axes=2),)if digest else())
            def empty():
                out=(j.zeros((cb,pairs2),j.float32),j.zeros((cb,pairs2),j.int8),j.zeros((cb,pairs2),j.int8))
                return out+((j.zeros((cb,pairs2,2),j.uint32),)if digest else())
            return jax.lax.cond(j.any(live),compute,empty)+(j.any(live),)
        out=jax.lax.map(block,(a['child_slots'],a['block_actor']))
        active_blocks=out[-1]
        # One scatter from block slots straight into [world][a1][r1][a2][r2]:
        # sentinel slots, padding lanes, dead k slots and non-expanded
        # children all index out of bounds and are dropped, never read back.
        slots=a['child_slots'].reshape(-1);idx=j.minimum(slots,children-1)
        a1=j.where(enumerated,pairs[...,0],9);r1=j.where(enumerated,pairs[...,1],9)
        a2=j.where(live_own&expanded[...,None],sel_own,9);r2=j.where(live_opp,sel_opp,9)
        world=j.where(slots<children,idx//LANES,n)[:,None,None];A1=flat(a1)[idx][:,None,None];R1=flat(r1)[idx][:,None,None]
        A2=flat(a2)[idx][:,:,None];R2=flat(r2)[idx][:,None,:]
        def place(x):
            x=x.reshape((slots.shape[0],k_own,k_opp)+x.shape[3:])
            return j.zeros((n,9,9,9,9)+x.shape[3:],x.dtype).at[world,A1,R1,A2,R2].set(x,mode='drop')
        def rooted(x):return x.reshape((capacity,2)+x.shape[1:])
        result=dict(t['frame'],active_blocks=active_blocks,t2_value=rooted(place(out[0])),t2_status=rooted(place(out[1])),t2_error=rooted(place(out[2])))
        if digest:result.update(t2_digest=rooted(place(out[3])))
        return result
    return jax.jit(run)


def validate_requests(requests,materializer,recipe,params):
    """D3's request checks plus the verified actor-index range; raises ValueError."""
    if type(recipe)is not original.Recipe or type(materializer)is not mapping.Tables:
        raise ValueError('original materializer and random recipe required')
    for request in requests:
        p=request.plans
        if (not p.selected or p.selected!=recipe.selected(p.root.address)or len(p.starts)!=2
                or tuple(x.world for x in p.starts)!=(0,1)
                or p.source_hash!=materializer.source.source_hash
                or p.source_rows_sha256!=materializer.source.source_rows_sha256
                or p.root.source_binding_sha256!=materializer.source.binding_sha256):
            raise ValueError('same selected root/source/world/recipe required')
        original._probabilities(tuple(request.root_logits),original._mask(request.root_legal))
        if not request.root_inputs or set(request.root_inputs)-set(INPUT_KEYS):
            raise ValueError('original encoder input keys required')
        if any(type(x)is not int or not 0<=x<len(params)for x in(request.learner_actor,request.opponent_actor)):
            raise ValueError('verified original actor index required')


def pack_turn1(requests,worlds,recipe,capacity,params,*,k1_own=9,fixed_capacity=False):
    """D3's host inputs for one frame at `capacity` root slots, or None when no world has a state.

    States and publics are stacked [capacity,2,...] with a missing world taking
    the first available state and padding roots repeating it (`available`
    marks the real ones); `order` is the k1_own prefix of each root's pricing
    order padded with its first action; `peer_rows`/`leaf_rows` are D3's
    archive-grouped actor rows (every archive sized 2*capacity under
    fixed_capacity, so the compile key never moves).
    """
    available=[w for pair in worlds for w in pair if w.state is not None]
    if not available:return None
    count=len(requests);first=available[0];flat=[w if w.state is not None else first for pair in worlds for w in pair]
    def stacked(values,world_axis=False):
        values=jax.device_get(tuple(values))
        def pack(*leaves):
            value=np.stack(leaves);length=capacity*(2 if world_axis else 1)
            padded=np.empty((length,)+value.shape[1:],value.dtype);padded[:len(value)]=value;padded[len(value):]=value[0]
            return padded.reshape((capacity,2)+value.shape[1:])if world_axis else padded
        return jax.tree.map(pack,*values)
    def vector(values,dtype,default=0):
        result=np.full(capacity,default,dtype);result[:count]=values;return result
    data=dict(states=stacked([w.state for w in flat],True),publics=stacked([w.public for w in flat],True),
        expected=stacked([r.root_inputs for r in requests]),
        legal=np.zeros((capacity,9),bool),order=np.zeros((capacity,9),np.int32),
        order_size=np.ones(capacity,np.int32),available=np.zeros((capacity,2),bool),
        owner=vector([r.plans.root.address.owner for r in requests],np.int32))
    data['legal'][:count]=[r.root_legal for r in requests];data['legal'][count:,0]=True
    data['available'][:count]=[[w.state is not None for w in pair]for pair in worlds]
    for i,r in enumerate(requests):
        order=recipe.pricing_order(r.plans.root.address,r.root_logits,r.root_legal)[:k1_own]
        data['order'][i]=order+(order[0],)*(9-len(order));data['order_size'][i]=len(order)
    def actor_rows(field):
        actors=np.repeat([getattr(r,field)for r in requests],2)
        counts=[int(np.count_nonzero(actors==actor))for actor in range(len(params))]
        history=max(counts[1:],default=0)
        history_size=0 if history==0 else 1<<(history-1).bit_length()
        rows=[]
        for actor in range(len(params)):
            indices=np.flatnonzero(actors==actor).astype(np.int32)
            size=2*capacity if actor==0 or fixed_capacity else history_size
            padded=np.full(size,2*capacity,np.int32);padded[:len(indices)]=indices;rows.append(padded)
        return tuple(rows)
    data['peer_rows']=actor_rows('opponent_actor');data['leaf_rows']=actor_rows('learner_actor')
    return data


class Depth2Pricer:
    def __init__(self,executor,model,*,capacity=2048,actor_capacity=33,k_own=9,k_opp=9,k1_own=9,k1_opp=9,child_block=27,digest=False,forward=None,
            fixed_capacity=False):
        """`fixed_capacity=True` compiles every frame at exactly `capacity`, with every actor's rows and blocks sized
        for it, so a shrinking request count (the head-to-head loop) never recompiles; padding rows stay inactive
        and their turn-2 blocks are skipped. The default buckets to the next power of two, as D3 does."""
        if type(capacity)is not int or capacity<=0 or type(actor_capacity)is not int or actor_capacity<=0:
            raise ValueError('finite explicit root/actor capacities required')
        _widths(k_own,k_opp,k1_own,k1_opp,child_block)
        if type(digest)is not bool or type(fixed_capacity)is not bool:raise ValueError('explicit digest/fixed-capacity bools required')
        self.executor=executor;self.model=model;self.capacity=capacity;self.actor_capacity=actor_capacity;self.fixed_capacity=fixed_capacity
        self.k_own=k_own;self.k_opp=k_opp;self.k1_own=k1_own;self.k1_opp=k1_opp;self.child_block=child_block;self.digest=digest;self.forward=forward
        self.kernels={};self.param_key=None;self.params=None;self.source_params=None;self.last_timing=None

    def bind(self,params):
        if not 0<len(params)<=self.actor_capacity:raise ValueError('declared actor bank exceeds fixed capacity')
        key=tuple(map(id,params))
        if key!=self.param_key:
            self.params=tuple(params);self.param_key=key;self.kernels={};self.source_params=tuple(params)

    def _blocks(self,requests,capacity,params):
        """Child slots per block, grouped by learner archive and padded like D3's actor rows.

        Actor 0 spans 2*capacity worlds and each archive history_size worlds (or
        2*capacity under fixed_capacity), so the block count depends on the same
        frame geometry as D3's compile key, not on per-archive permutations.
        Sentinel slots (== child count) and absent archives' blocks are skipped
        on device.
        """
        actors=np.repeat([r.learner_actor for r in requests],2);cb=self.child_block;children=LANES*2*capacity
        counts=[int(np.count_nonzero(actors==actor))for actor in range(len(params))]
        history=max(counts[1:],default=0);history_size=0 if history==0 else 1<<(history-1).bit_length()
        slots=[];block_actor=[]
        for actor in range(len(params)):
            size=2*capacity if actor==0 or self.fixed_capacity else history_size
            if not size:continue
            worlds=np.flatnonzero(actors==actor);blocks=-(-LANES*size//cb)
            s=np.full(blocks*cb,children,np.int32);s[:LANES*len(worlds)]=(worlds[:,None]*LANES+np.arange(LANES)).reshape(-1)
            slots.append(s.reshape(blocks,cb));block_actor+=[actor]*blocks
        return np.concatenate(slots),np.asarray(block_actor,np.int32)

    def price(self,requests,materializer,recipe,params):
        requests=tuple(requests)
        if not requests:return Depth2Frame(world_reports=(),timing=dict(roots=0,device_invocations=0),**self._frame(None,0))
        if type(recipe)is not original.Recipe or type(materializer)is not mapping.Tables:
            raise ValueError('original materializer and random recipe required')
        if len(requests)>self.capacity:raise ValueError('one frame exceeds declared root capacity')
        capacity=self.capacity if self.fixed_capacity else min(self.capacity,1<<(len(requests)-1).bit_length())
        self.bind(params);started=time.perf_counter()
        validate_requests(requests,materializer,recipe,params)
        worlds=batch._numpy_worlds(requests,materializer,recipe);prepared=time.perf_counter()
        available=[w for pair in worlds for w in pair if w.state is not None]
        count=len(requests);output=None;data=None;lower_seconds=compile_seconds=device_seconds=0.
        if available:
            data=pack_turn1(requests,worlds,recipe,capacity,params,k1_own=self.k1_own,fixed_capacity=self.fixed_capacity)
            data['child_slots'],data['block_actor']=self._blocks(requests,capacity,params)
            key=(capacity,tuple(len(x)for x in data['peer_rows']),tuple(len(x)for x in data['leaf_rows']),data['child_slots'].shape)
            device_data=jax.device_put(data)
            if key not in self.kernels:
                fn=make_depth2_kernel(self.executor,self.model,params=self.params,capacity=capacity,k_own=self.k_own,k_opp=self.k_opp,
                    k1_own=self.k1_own,k1_opp=self.k1_opp,child_block=self.child_block,digest=self.digest,forward=self.forward)
                clock=time.perf_counter();lowered=fn.lower(device_data);lower_seconds=time.perf_counter()-clock
                clock=time.perf_counter();self.kernels[key]=lowered.compile();compile_seconds=time.perf_counter()-clock
            clock=time.perf_counter();output=jax.device_get(jax.block_until_ready(self.kernels[key](device_data)))
            device_seconds=time.perf_counter()-clock
        frame=self._frame(output,count)
        evaluated=0 if output is None else capacity;lanes=self.child_block*self.k_own*self.k_opp
        blocks=0 if output is None else int(data['child_slots'].shape[0]);live=0 if output is None else int(np.count_nonzero(output['active_blocks']))
        enumerated2=int(np.count_nonzero(frame['t2_status']))
        work=dict(evaluated_root_slots=evaluated,evaluated_world_slots=2*evaluated,evaluated_joint_slots=2*LANES*evaluated,
            evaluated_peer_forward_rows=0 if output is None else sum(map(len,data['peer_rows'])),
            evaluated_leaf_forward_rows=0 if output is None else LANES*sum(map(len,data['leaf_rows'])),
            evaluated_child_prior_forward_rows=0 if output is None else LANES*sum(map(len,data['peer_rows'])),
            padding_root_slots=0 if output is None else evaluated-count,padding_world_slots=0 if output is None else 2*(evaluated-count),
            substituted_world_slots=0 if output is None else 2*count-len(available),
            pre_refused_available_worlds=0 if output is None else int(np.count_nonzero(frame['pre']>0)),
            turn2_blocks=blocks,turn2_active_blocks=live,turn2_lanes_per_block=lanes,turn2_evaluated_leaf_lanes=live*lanes,
            turn2_enumerated_leaves=enumerated2,turn2_padded_leaf_lanes=live*lanes-enumerated2)
        finished=time.perf_counter()
        timing=dict(roots=count,world_prepare_seconds=prepared-started,batch_pack_and_device_seconds=finished-prepared,
            device_invocations=int(output is not None),lower_seconds=lower_seconds,compile_seconds=compile_seconds,device_seconds=device_seconds,
            actor_strategy='DEPTH2_D3_TURN1_ARCHIVE_BATCHES_BLOCKED_TURN2',root_capacity=capacity,frame_wall_seconds=finished-started,
            child_block=self.child_block,k_own=self.k_own,k_opp=self.k_opp,k1_own=self.k1_own,k1_opp=self.k1_opp,fixed_capacity=self.fixed_capacity,**work)
        self.last_timing=timing
        reports=tuple(self._reports(i,r,pair,frame)for i,(r,pair)in enumerate(zip(requests,worlds)))
        return Depth2Frame(world_reports=reports,timing=timing,**frame)

    def _frame(self,output,count):
        shapes=dict(pre=((2,),np.int32),root_peer_requested=((2,),bool),root_peer_legal=((2,9),bool),root_peer_logits=((2,9),np.float32),
            t1_status=((2,9,9),np.int8),t1_value=((2,9,9),np.float32),t1_error=((2,9,9),np.int8),
            t1_own_requested=((2,9,9),bool),t1_own_legal=((2,9,9,9),bool),t1_own_logits=((2,9,9,9),np.float32),
            t1_peer_requested=((2,9,9),bool),t1_peer_legal=((2,9,9,9),bool),t1_peer_logits=((2,9,9,9),np.float32),
            t2_status=((2,9,9,9,9),np.int8),t2_value=((2,9,9,9,9),np.float32),t2_error=((2,9,9,9,9),np.int8))
        if self.digest:shapes.update(t2_digest=((2,9,9,9,9,2),np.uint32),t1_digest_own=((2,9,9,2),np.uint32),t1_digest_peer=((2,9,9,2),np.uint32))
        source={'root_peer_requested':'peer_requested','root_peer_legal':'peer_legal','root_peer_logits':'peer_logits'}
        frame={}
        for name,(shape,dtype)in shapes.items():
            if output is None:
                frame[name]=np.full((count,)+shape,UNAVAILABLE if name=='pre'else 0,dtype)
            else:frame[name]=np.ascontiguousarray(output[source.get(name,name)][:count]).astype(dtype)
        return frame

    def _reports(self,index,request,worlds,frame):
        reports=[]
        for wi,(start,world)in enumerate(zip(request.plans.starts,worlds)):
            report=dict(world=start.world,posterior_seed=start.seed,kernel_seed=world.kernel_seed,status='REFUSED',reason=None,
                required_pairs=None,completed_pairs=0,unknown_pairs=None,turn2_required=None,turn2_unknown=None,
                reply_history_approximation=None if world.state is None or world.reply_private_history_available else serial.REPLY_HISTORY_APPROXIMATION)
            reports.append(report)
            if world.state is None:report['reason']='physical-materialization:'+','.join(world.refusals);continue
            code=int(frame['pre'][index,wi])
            if code<0:report['reason']='no world state';continue
            if code:
                reason=PRE_REASONS.get(code)
                if reason is None:reason='actual learner root input differs: '+tuple(k for k in INPUT_KEYS if k in request.root_inputs)[code-3]
                report['reason']='ValueError: '+reason;continue
            status=frame['t1_status'][index,wi];status2=frame['t2_status'][index,wi]
            required=int(np.count_nonzero(status));unknown=int(np.count_nonzero(status==3))
            required2=int(np.count_nonzero(status2));unknown2=int(np.count_nonzero(status2==3))
            errors=tuple((int(a),int(r),'ValueError: '+ERROR_REASONS[int(frame['t1_error'][index,wi,a,r])])for a,r in zip(*np.nonzero(status==3)))
            report.update(status='COMPLETE'if not unknown and not unknown2 else'REFUSED',required_pairs=required,completed_pairs=required-unknown,
                unknown_pairs=unknown,turn2_required=required2,turn2_unknown=unknown2,errors=errors,
                opponent_requested=bool(frame['root_peer_requested'][index,wi]),reply_legal=tuple(map(bool,frame['root_peer_legal'][index,wi])),
                reply_logits=tuple(map(float,frame['root_peer_logits'][index,wi])),
                reason=None if not unknown and not unknown2 else'unknown-required-public-leaf')
        return tuple(reports)
