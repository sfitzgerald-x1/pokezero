"""Pure backups over a DepthNFrame: the recursive learner-max reference, a per-node solver hook, refusal accounting.

Host float64 arithmetic only, from the leaves up. At every node a seat's prior
is a masked softmax over the slots the frame enumerated there (its selection
map), so pruned mass is renormalized away; a waiting seat has slot 0 alone with
unit mass. Terminal nodes keep their exact value. An error anywhere under an
enumerated path makes the node None, and so the turn-1 cell and the root: D's
refusal. At N=2 every number equals `depth2_backups` on `as_depth2_frame` (the
same softmax, fsum rows and strict max). The returned WorldPrices keep the root
opponent's full prior but list only the replies turn 1 enumerated, so under
k1_opp the reply mass is renormalized the same way; a pruned learner root
action (k1_own) stays None and reduce_completed_q refuses. A world with
`pre != 0` has no price surface and is returned as None.
"""
import math

import numpy as np

from pokezero import v5_gumbel as original

DEPTH1_CAUSES=('materialization','pre','turn1')


def causes(depth):
    """Refusal causes in precedence order for an N-ply frame; 'turn{p}' holds every code at ply p >= 2."""
    return DEPTH1_CAUSES+('turn1_depth2_only',)+tuple(f'turn{p}'for p in range(2,depth+1))+('k1_pruned',)


def _softmax(logits,actions):
    """Masked softmax of action-indexed logits over `actions`, in that order (a renormalized prior over the enumerated slots)."""
    top=max(float(logits[a])for a in actions)
    weights=[math.exp(float(logits[a])-top)for a in actions]
    total=math.fsum(weights)
    return [w/total for w in weights]


def learner_max(grid,pi_own,pi_opp):
    """The reference rule: max over own slots of the opponent-prior average of the row; a strict max keeps the lower slot on ties."""
    best=None
    for row in grid:
        q=math.fsum(p*v for p,v in zip(pi_opp,row))
        if best is None or q>best:best=q
    return best


def average(grid,pi_own,pi_opp):
    """Diagnostic rule: both seats play their prior, sum pi_own pi_opp v (row sums first, as depth 2)."""
    return math.fsum(p*math.fsum(q*v for q,v in zip(pi_opp,row))for p,row in zip(pi_own,grid))


def _node(frame,root,world,ply,node):
    """(status, value, own_actions, opp_actions, own_logits, opp_logits) of a node; ply 1 is a turn-1 cell id a1*9+r1."""
    if ply==1:
        a,r=divmod(int(node),9);at=lambda x:x[root,world,a,r]
        return (int(at(frame.t1_status)),at(frame.t1_value),at(frame.t1_own_actions),at(frame.t1_opp_actions),at(frame.t1_own_logits),at(frame.t1_peer_logits))
    p=frame.nodes[ply-2];at=lambda x:x[root,world,node]
    return int(at(p.status)),at(p.value),at(p.own_actions),at(p.opp_actions),at(p.own_logits),at(p.opp_logits)


def node_value(frame,root,world,ply,node,rule=learner_max):
    """The backed-up value of one node (None if it is not priced or any enumerated path under it errs).

    `rule(grid, pi_own, pi_opp)` solves one node from its enumerated slots:
    `grid[i][j]` is the child value at own slot i, opponent slot j (all
    floats), `pi_own`/`pi_opp` the seats' renormalized priors in slot order.
    Other backups (the measurement agent's hedge) plug in here.
    """
    status,value,own_actions,opp_actions,own_logits,opp_logits=_node(frame,root,world,ply,node)
    if status==1:return float(value)
    if status!=2:return None
    k_own,k_opp=frame.k_own[ply-1],frame.k_opp[ply-1];K=k_own*k_opp;leaf=ply+1==frame.depth
    own_slots=[i for i in range(k_own)if own_actions[i]>=0];opp_slots=[i for i in range(k_opp)if opp_actions[i]>=0]
    if not own_slots or not opp_slots:return None
    grid=[]
    for i in own_slots:
        row=[]
        for x in opp_slots:
            if leaf:
                s=int(frame.leaf_status[root,world,node,i,x])
                v=float(frame.leaf_value[root,world,node,i,x])if s in(1,4)else None
            else:v=node_value(frame,root,world,ply+1,node*K+i*k_opp+x,rule)
            if v is None:return None
            row.append(v)
        grid.append(row)
    pi_own=_softmax(own_logits,[int(own_actions[i])for i in own_slots]);pi_opp=_softmax(opp_logits,[int(opp_actions[x])for x in opp_slots])
    return min(1.,max(-1.,rule(grid,pi_own,pi_opp)))


def _world(frame,root,world,rule):
    if int(frame.pre[root,world])!=0:return None
    requested=bool(frame.root_peer_requested[root,world]);status=frame.t1_status[root,world]
    kept=tuple(bool(np.any(status[:,r]!=0))for r in range(9))if requested else(False,)*9
    grid=[[node_value(frame,root,world,1,a*9+r,rule)if status[a,r]else None for r in range(9)]for a in range(9)]
    return original.WorldPrices(requested,kept,tuple(map(float,frame.root_peer_logits[root,world])),grid)


def depth1_world(frame,root,world):
    """The depth-1 WorldPrices of one block world: D3's turn-1 value (t1_value, the learner's value head on its own view of each
    child, or the exact terminal value) on every enumerated cell D3 priced (t1_status 1 terminal or 2 expanded), None on any
    other cell; the same kept replies and root prior as _world. None for a world without a price surface (pre != 0)."""
    if int(frame.pre[root,world])!=0:return None
    requested=bool(frame.root_peer_requested[root,world]);status=frame.t1_status[root,world]
    kept=tuple(bool(np.any(status[:,r]!=0))for r in range(9))if requested else(False,)*9
    grid=[[float(frame.t1_value[root,world,a,r])if int(status[a,r])in(1,2)else None for r in range(9)]for a in range(9)]
    return original.WorldPrices(requested,kept,tuple(map(float,frame.root_peer_logits[root,world])),grid)


def backup_depth1(frame,root):
    """Both block worlds' depth-1 WorldPrices (depth1_world): the turn-1 grid the depth-2 backup recurses below."""
    return tuple(depth1_world(frame,root,world)for world in range(2))


def backup(frame,root,rule=learner_max):
    """Both worlds' WorldPrices under `rule` at every node (None for a world without a surface)."""
    return tuple(_world(frame,root,world,rule)for world in range(2))


def backup_learner_max(frame,root):
    """The reference: recursive learner-max, terminals exact, any error under a cell refuses it."""
    return backup(frame,root,learner_max)


def backup_average(frame,root):
    """Diagnostic: both seats' prior play at every ply below turn 1."""
    return backup(frame,root,average)


def _root(prices,root_logits,root_legal):
    if any(p is None for p in prices):
        return original.Reduction('REFUSED',None,0,0,0.,'unknown-required-world-or-reply-denominator')
    return original.reduce_completed_q(root_logits=root_logits,root_legal=root_legal,worlds=prices)


def learner_max_root(frame,root,root_logits,root_legal):
    return _root(backup_learner_max(frame,root),root_logits,root_legal)


def average_root(frame,root,root_logits,root_legal):
    return _root(backup_average(frame,root),root_logits,root_legal)


def refusal_summary(frame,root_legal=None):
    """Refused roots and causes under the reference rule, and error counts by ply and code; pure numpy.

    Cause precedence per root: a world without a state, a world pre code, a
    turn-1 error with a D code (1-4), a turn-1 error only a deeper pricer
    raises (5/6), an error at ply 2, ..., at ply N (leaves), a k1_own-pruned
    legal root action. `depth1_refused_count` is the share D3 would refuse on
    the same frame; `deeper_refused_count` the extra refusals. Pruning is exact
    when `root_legal` ([R,9] bools) is given; without it every usable root of a
    frame priced with k1_own<9 is flagged, so a refused root never reads as not
    refused. `errors_by_ply[p]` and `errors_by_code[p]` count every world.
    """
    N=frame.depth;pre,t1,t1e=frame.pre,frame.t1_status,frame.t1_error
    narrow=isinstance(frame.timing,dict)and int(frame.timing.get('k1_own',9))<9
    ply_status={p:frame.nodes[p-2].status for p in range(2,N)};ply_status[N]=frame.leaf_status
    ply_error={p:frame.nodes[p-2].error for p in range(2,N)};ply_error[N]=frame.leaf_error
    roots=[]
    for r in range(pre.shape[0]):
        codes=tuple(int(c)for c in pre[r]);failed=t1[r]==3;usable=[w for w in range(2)if codes[w]==0]
        e1=int(np.count_nonzero(failed&(t1e[r]<=4)));only2=int(np.count_nonzero(failed&(t1e[r]>=5)))
        deeper={p:int(np.count_nonzero(ply_status[p][r]==3))for p in range(2,N+1)}
        if root_legal is not None:pruned=any(root_legal[r][a]and not t1[r,w,a].any()for w in usable for a in range(9))
        else:pruned=narrow and bool(usable)
        cause=('materialization'if any(c<0 for c in codes)else'pre'if any(c>0 for c in codes)else'turn1'if e1
            else'turn1_depth2_only'if only2 else next((f'turn{p}'for p in range(2,N+1)if deeper[p]),'k1_pruned'if pruned else None))
        roots.append(dict(root=r,refused=cause is not None,cause=cause,pre=codes,turn1_errors=e1,turn1_depth2_only_errors=only2,
            errors_by_ply=deeper,k1_pruned=bool(pruned)))
    refused=[x for x in roots if x['refused']]
    by_code=lambda errors,status,codes:{code:int(np.count_nonzero((status==3)&(errors==code)))for code in codes}
    errors_by_code={1:by_code(t1e,t1,(1,2,3,4,5,6))}
    errors_by_code.update({p:by_code(ply_error[p],ply_status[p],(1,2,3,4,5,6)if p<N else(1,2,3,4))for p in range(2,N+1)})
    return dict(depth=N,roots=tuple(roots),root_count=len(roots),refused_count=len(refused),
        depth1_refused_count=sum(x['cause']in DEPTH1_CAUSES for x in refused),
        deeper_refused_count=sum(x['cause']not in DEPTH1_CAUSES for x in refused),
        refused_by_cause={c:sum(x['cause']==c for x in refused)for c in causes(N)},
        errors_by_ply={p:int(np.count_nonzero(ply_status[p]==3))for p in range(2,N+1)},errors_by_code=errors_by_code,
        refused_errors_by_ply={x['root']:x['errors_by_ply']for x in refused})
