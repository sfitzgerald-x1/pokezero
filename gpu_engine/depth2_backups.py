"""Pure backups over a Depth2Frame: the learner-max reference, the average diagnostic, refusal accounting.

Host float64 arithmetic only. A prior at a child is a masked softmax over the
actions the frame enumerated there (the top-k set under pruning), so pruned
mass is renormalized away; a waiting seat enumerates action 0 alone and so gets
unit mass. Terminal turn-1 cells keep their exact value. Any error under an
enumerated cell yields None, D's refusal. The returned WorldPrices keep the
root opponent's full prior but list only the replies turn 1 enumerated, so
under k1_opp the reply mass is renormalized the same way; a pruned learner root
action (k1_own) stays None and reduce_completed_q refuses, as D's denominator
requires. A world with `pre != 0` has no price surface and is returned as None.
"""
import math

import numpy as np

from pokezero import v5_gumbel as original

# Refusal causes in precedence order. The first three are D's own (depth 1 refuses
# the root too); 'turn1_depth2_only' (codes 5/6: the opponent's child view),
# 'turn2' and 'k1_pruned' (a legal root action cut by k1_own) refuse only at depth 2.
CAUSES=('materialization','pre','turn1','turn1_depth2_only','turn2','k1_pruned')
DEPTH1_CAUSES=CAUSES[:3]


def _softmax(logits,mask):
    top=max(float(logits[i])for i in range(9)if mask[i])
    weights=[math.exp(float(logits[i])-top)if mask[i]else 0. for i in range(9)]
    total=math.fsum(weights)
    return [w/total for w in weights]


def _rows(frame,root,world,a1,r1):
    """An expanded cell's opponent-averaged row values per enumerated a2 (None elsewhere) and the learner's prior."""
    t2=frame.t2_status[root,world,a1,r1]
    own=np.any(t2!=0,axis=1);opp=np.any(t2!=0,axis=0)
    if np.any(t2==3)or not own.any():return None
    pi_own=_softmax(frame.t1_own_logits[root,world,a1,r1],own)
    pi_opp=_softmax(frame.t1_peer_logits[root,world,a1,r1],opp)
    v=frame.t2_value[root,world,a1,r1]
    return [math.fsum(pi_opp[r2]*float(v[a2,r2])for r2 in range(9)if opp[r2])if own[a2]else None for a2 in range(9)],pi_own


def _cell(frame,root,world,a1,r1,rule):
    status=int(frame.t1_status[root,world,a1,r1])
    if status==1:return float(frame.t1_value[root,world,a1,r1])
    if status!=2:return None
    rows=_rows(frame,root,world,a1,r1)
    return None if rows is None else min(1.,max(-1.,rule(*rows)))


def _learner_max(rows,pi_own):
    best=None
    for q in rows:
        if q is not None and(best is None or q>best):best=q  # strict: ties keep the lower index
    return best


def _average(rows,pi_own):
    return math.fsum(pi_own[a2]*q for a2,q in enumerate(rows)if q is not None)


def learner_max_choice(frame,root,world):
    """Diagnostic: the a2 learner-max picks in each expanded cell (lowest index among ties); -1 elsewhere."""
    choice=np.full((9,9),-1,np.int32)
    for a1 in range(9):
        for r1 in range(9):
            if int(frame.t1_status[root,world,a1,r1])!=2:continue
            rows=_rows(frame,root,world,a1,r1)
            if rows is None:continue
            choice[a1,r1]=rows[0].index(_learner_max(*rows))
    return choice


def _world(frame,root,world,rule):
    if int(frame.pre[root,world])!=0:return None
    requested=bool(frame.root_peer_requested[root,world]);status=frame.t1_status[root,world]
    kept=tuple(bool(np.any(status[:,r]!=0))for r in range(9))if requested else(False,)*9
    grid=[[_cell(frame,root,world,a,r,rule)if status[a,r]else None for r in range(9)]for a in range(9)]
    return original.WorldPrices(requested,kept,tuple(map(float,frame.root_peer_logits[root,world])),grid)


def backup_learner_max(frame,root):
    """The reference: per expanded cell, max over enumerated a2 of the opponent-prior average of v(a2, r2)."""
    return tuple(_world(frame,root,world,_learner_max)for world in range(2))


def backup_average(frame,root):
    """Diagnostic: per expanded cell, the learner's own prior play at turn 2, sum pi_own pi_opp v."""
    return tuple(_world(frame,root,world,_average)for world in range(2))


def _root(prices,root_logits,root_legal):
    if any(p is None for p in prices):
        return original.Reduction('REFUSED',None,0,0,0.,'unknown-required-world-or-reply-denominator')
    return original.reduce_completed_q(root_logits=root_logits,root_legal=root_legal,worlds=prices)


def learner_max_root(frame,root,root_logits,root_legal):
    return _root(backup_learner_max(frame,root),root_logits,root_legal)


def average_root(frame,root,root_logits,root_legal):
    return _root(backup_average(frame,root),root_logits,root_legal)


def refusal_summary(frame,root_legal=None):
    """Refused roots and causes under the reference rule, and error counts by code; pure numpy.

    Cause precedence per root: a world without a state, a world pre code, a
    turn-1 error with a D code (1-4), a turn-1 error only the depth-2 pricer
    raises (5/6), a turn-2 error, a k1_own-pruned legal root action.
    `depth1_refused_count` is the share D3 would refuse on the same frame;
    `depth2_only_refused_count` the extra refusals. Pruning is exact when
    `root_legal` ([R,9] bools) is given (a legal action with no enumerated row
    in a usable world, which learner_max_root refuses); without it every usable
    root of a frame priced with k1_own<9 is flagged, so a refused root never
    reads as not refused. Reply pruning (k1_opp) renormalizes, never refuses.
    """
    pre,t1,t1e,t2=frame.pre,frame.t1_status,frame.t1_error,frame.t2_status
    narrow=isinstance(frame.timing,dict)and int(frame.timing.get('k1_own',9))<9
    roots=[]
    for r in range(pre.shape[0]):
        codes=tuple(int(c)for c in pre[r]);failed=t1[r]==3;usable=[w for w in range(2)if codes[w]==0]
        e1=int(np.count_nonzero(failed&(t1e[r]<=4)));only2=int(np.count_nonzero(failed&(t1e[r]>=5)));e2=int(np.count_nonzero(t2[r]==3))
        if root_legal is not None:pruned=any(root_legal[r][a]and not t1[r,w,a].any()for w in usable for a in range(9))
        else:pruned=narrow and bool(usable)
        cause=('materialization'if any(c<0 for c in codes)else'pre'if any(c>0 for c in codes)else'turn1'if e1
            else'turn1_depth2_only'if only2 else'turn2'if e2 else'k1_pruned'if pruned else None)
        roots.append(dict(root=r,refused=cause is not None,cause=cause,pre=codes,turn1_errors=e1,turn1_depth2_only_errors=only2,
            turn2_errors=e2,k1_pruned=bool(pruned)))
    refused=[x for x in roots if x['refused']]
    by_code=lambda errors,status,codes:{code:int(np.count_nonzero((status==3)&(errors==code)))for code in codes}
    return dict(roots=tuple(roots),root_count=len(roots),refused_count=len(refused),
        depth1_refused_count=sum(x['cause']in DEPTH1_CAUSES for x in refused),
        depth2_only_refused_count=sum(x['cause']not in DEPTH1_CAUSES for x in refused),
        refused_by_cause={c:sum(x['cause']==c for x in refused)for c in CAUSES},
        turn1_errors_by_code=by_code(t1e,t1,(1,2,3,4,5,6)),turn2_errors_by_code=by_code(frame.t2_error,t2,(1,2,3,4)),
        refused_turn2_errors={x['root']:x['turn2_errors']for x in refused})
