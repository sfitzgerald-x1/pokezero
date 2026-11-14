"""DepthN depth-2 live search: Codex's ladder search-server requests answered over ProtocolConstructor worlds (gap 5).

Design. A DepthNLiveSearch owns one ProtocolConstructor (W = 8 or 16 worlds: rung DN2W8 / DN2W16; DN2W32 / DN2W64 only for
offline replays, the MC ladder's world-count rungs), one DepthNPricer per
capacity bucket of roots (make_pricers: depth 2, full width k 9 and k1 9/9, `prior`, fixed_capacity, the champion as
forward=, one shared program table) and the champion bank. phase_d_search_server.Server's single queue consumer calls
process_many(requests) on its one executor thread with at most max_requests requests and checks the responses' order and
request_id. Per batch (process_many chunks a longer list into batches of max_requests):
- validate: exactly Codex's ten ladder keys (what ProtocolConstructor.prepare accepts) plus ours, `rung` (this server's)
  and `budget_seconds` (finite > 0), and the server queue's private ARRIVAL stamp; anything else is REFUSED bad_request.
- expired_in_queue: a request whose queue wait (ARRIVAL -> the batch's start, both time.monotonic; 0 without a stamp)
  leaves less than min_price_seconds of its budget is refused unpriced and unprepared (its address stays unattempted).
- the rest, stripped to Codex's keys: constructor_worlds.prepare_roots (CPU device) -> build (only_complete: Codex's all-W
  rule refuses a root with any refused world before any device work) -> ONE price call on the smallest bucket holding the
  batch's blocks -> per root, depthn_backups.backup(frame, row, learner_max) per block, once. DepthN's capacity counts
  requests, one per block of two worlds (count > capacity raises; n = 2 x capacity world slots), so the bucket of b roots
  is a pricer of capacity b x K, K = W/2.
- the decision (decide): the deployment cell's operator itself, search_hedge.averaging over the root's 2K backed-up
  worlds in block order (block k's world j is averaging's world 2k+j), as v2_search_head_to_head's 'average' cell calls it:
  its argmax over the legal actions (lowest index on ties) is the played action_index and its per-action values are
  q_values (None on illegal lanes). Its validity rules hold on the way in: WorldPrices refuses a value outside [-1, 1] or
  a non-finite value or logit, _mask a malformed legal mask, _probabilities an empty reply set. The label stays
  ARGMAX_MEAN_COMPLETED_Q (the panel's declaration pins it); root_operator_impl names averaging.
- reporting only: constructor_worlds' composition on the same backups (reduce_completed_q per block, the equal-weight mean
  of the blocks' completed-Q, constructor_worlds.root_actions) gives completed_q, policy_target, root_value,
  action_mean_q, action_policy_target (never played) and mean_q_agrees (action_mean_q == action_index; they can differ by
  one ulp at a near tie). If it refuses or raises (reduce_completed_q's range check raises when a block's completed-Q
  rounds outside [-1, 1]: an action winning, or losing, against every reply in both worlds of a block, its reply prior
  summing to 1 + ulp), or fails in any other way, those fields are None and reduction_note says why; the root still plays
  averaging's action.
- causes: a root prepare() refused, or one with a refused world, is constructor_refused; one refused by build's root
  checks (root view, legal mask, B2 inputs: ROOT_B2_INPUTS_UNUSABLE, root forward), by DepthN's pricing (a block cause, depthn_backups.refusal_summary) or by
  averaging (an unknown price, or its error) is depthn_refused. Any other exception in the shared batch (prepare_roots,
  build, price, the backups) refuses every request of the batch server_error with its text; MemoryError propagates.
  Every response must pass phase_d_search_server.encode (else server_error), so no value breaks the wire.
- root classes (batch 2 admits forced-switch, sleep, fresh-substitute and wish roots): `root_classes`, the sorted subset of
  CLASSES the root belongs to (empty: other), and `root_class`, the first by priority forced_switch > fresh_substitute >
  wish > sleep, else 'other', on every response of a request that reached prepare (None otherwise). Source: batch 2's own
  protocol facts, recomputed after prepare with the constructor's own helpers on the request's own lines, because prepare
  keeps neither its public root nor its facts and reaching them would mean patching Codex's module: capture_root (the
  constructor module's reference, the same config and seed) gives the public materialization payload, then
  phase_d_tail.inspect_facts(own request, payload, lines): replacement -> forced_switch, fresh_substitutes ->
  fresh_substitute, wish_set_turns -> wish, a fact check that TRIGGERED for a class counting in it too (the constructor
  refuses those roots, they are still of that class); the own request's forceSwitch -> forced_switch; sleep: a sleeper in
  the payload (phase_d_protocol_world._defer_sleep, what the constructor defers) or in the public history
  (_sleep_origins(lines)). Constructor-refused roots are classified too. When the capture itself fails, only the request
  and the lines classify (forceSwitch, _sleep_origins): root_class_source says which ('protocol_facts' or
  'request_and_lines'). Its cost is timings.classify_seconds (one more public-root capture per root).
  For a fully constructed root, root_class_crosscheck compares them with the CLEAR checks the constructor's own world
  builds wrote (class_crosscheck), reported only.
- refuse_classes: a fully constructed root in any of these classes is REFUSED before pricing, cause class_refused,
  refusal ROOT_CLASS_REFUSED:<its refused classes>; one that could not be classified, ROOT_CLASS_UNKNOWN, when any class
  is refused. A root prepare() refused, or one with a refused world, keeps its constructor cause.
- structure check, per fully constructed root before packing (timings.structure_seconds): every world's (state, public)
  pytree paths, shapes and dtypes (what DepthN's carry signature keys programs on) must equal the reference; a root that
  differs is REFUSED depthn_refused, WORLD_STRUCTURE_MISMATCH:<short diff>, and the batch's other roots still price
  (DepthN would refuse the whole frame). Warmup records the reference from its first fully constructed root, refuses to
  finish unless every warmup root shares it, and checks it is the only constructor-world carry signature in the warmed
  program keys; the warm report (and READY) names the reference root. Before any warmup the first such root seen is a
  provisional reference (carry['provisional']); warmup replaces it.
- precision: the champion gates R1/R2/R3 were certified under JAX_DEFAULT_MATMUL_PRECISION=highest (and
  JAX_ENABLE_X64=0), set by the devbox environment; Codex's ladder server also prices under 'highest' (full-f32 matmul
  algorithms where JAX's default may use faster reduced-precision ones on GPU). The live server hard-pins it
  (depthn_live_server --matmul-precision, default 'highest': it refuses to start on any other setting); this module
  records jax.config.jax_default_matmul_precision in the pricer identity (matmul_precision) of every response, and the
  panel's probe and readout refuse a response priced under another precision.
- the effect guard (`effect_guard`, every COMPLETE response; pokezero.depthn_effect_guard, its inputs
  depthn_guard_inputs): the pricers are ObservedDepthNPricers whose observer seam reads the P1 call's turn-1 children while
  this batch prices (observer only: the frame and every program are the base pricer's); the record of the root (our root
  prior, the peer's per world, both sides' turn-2 priors per expanded cell, the search's own action) with `tripped` (the
  live rule 'reach': turn-1 (b), the reach M_own / M_opp >= 5%, or (c); the any-cell arm kept for analysis), which the live
  actor turns into greedy (cause UNREPRESENTED_EFFECT). A root whose inputs cannot be read gets the fail-closed
  record (tripped, with its error), as does one whose expanded cells have no captured children.
- the root prior's inputs (reported): `root_view_source`, the build root dict's ('b2': the champion read B2's own encoding of
  the literal root, as Codex's ladder; 'world0': world 0's own view, for a root without B2 inputs), and `b2_view`, its
  diagnostic of world 0's view against B2's encoding (wire_b2_view: equal, per-key equal flags, the largest numeric
  difference, error). Both are None for a root that never reached the root view; a root refused ROOT_LEGAL_EMPTY has
  b2_view set and root_view_source None (build checks the legal mask between the two).
- private reveals (`private_reveals`, every response whose root prepare() built; None otherwise): constructor_worlds'
  private_reveal_record (enabled, found, applicable, redacted, unmatched): the owner's abilities its lines revealed only on
  Showdown's side-private Gen 3 Pressure line, and, with redact_private_reveals (the server's --redact-private-reveals; off
  by default), the members whose public ability it cleared from the peer's view of the owner side in every world and
  (belief) per such member its belief restored to the peer wire's fold (variant counts before and after, changed).
- peer view b (`peer_view_b`; ONLY in the responses of a search built with peer_view_b, the server's --peer-view-b, off by
  default: a search without it adds no key, so its responses are exactly as before the flag): constructor_worlds'
  peer_view_b_record of every response whose root prepare() built (enabled, cells: per projected cell its column, side,
  slot, species, from_b2's carry value, the exact rule's and view b's public value; projected, changed, exact_agrees), None
  otherwise. View b implies the private-reveal redaction (private_reveals then reads enabled).
Response, per the live contract: schema, request_id, rung, archive_sha256, status, action_index, q_values (9, None on
illegal lanes), refusal, cause, requested_worlds, constructed_worlds, decision_seconds (batch start -> result, COMPLETE
only); ours: root_operator, root_operator_impl, completed_q, policy_target, root_value, action_mean_q,
action_policy_target, mean_q_agrees, reduction_note, root_view_source, b2_view, effect_guard, private_reveals, block_causes, world_refusals, timings (queue_seconds,
prepare_seconds (the root's own), build/price/reduce_seconds (the shared batch's; reduce = backups, averaging and the
reporting reduction)), batch_size, bucket, tree_commit, pricer. Fields a request never reached are None. `rung` echoes the
request's when it is a string (so a RUNG_NOT_SERVED refusal still matches the client's rung), else this server's.
Sessions: Codex's constructor keeps `sessions` (the incremental protocol fold per (rung, battle, owner, stream)) and `seen`
(every attempted address) forever. Here the battles least recently sent to prepare beyond session_cap lose both. A battle
of the current batch is never evicted (session_cap >= max_requests), so a live battle keeps its fold and its no-retry
rule; an evicted battle that returns folds its full prefix again from a fresh adapter (the same state, more host time).
warmup(requests): per bucket, the full path on saved Showdown requests forced to that bucket (cold: compiles P1 and the
selection per capacity, the gather per pool, the shared leaf block program, the root forward), then once more (warm).
Session state is cleared around every pass, and full (gen-2) collections are deferred while the programs trace. At the end
jit's tracing caches are emptied, the heap is collected once and each bucket's warm batch runs again
(gpu_engine.trace_release: the executables stay, the traces go; the warm report's trace_release records it). The server is
READY only after it.
"""
from collections import OrderedDict
import gc
import math
import time

import jax
import numpy as np

from pokezero import v5_gumbel as original
from pokezero.phase_d_search_server import SCHEMA,encode
from pokezero import depthn_effect_guard as effect_guard
from . import champion_evaluator as evaluators,constructor_worlds as cw,depthn_backups as backups,device_depthn_pricer as dn,search_hedge as hedge
from . import depthn_guard_inputs as guard_inputs,trace_release

RUNGS={'DN2W8':8,'DN2W16':16,'DN2W32':32,'DN2W64':64}   # W32 and W64: the MC ladder's world-count rungs (offline replays)
ROOT_OPERATOR='ARGMAX_MEAN_COMPLETED_Q'
ROOT_OPERATOR_IMPL='search_hedge.averaging'
NOISE_GATED_K2='NOISE_GATED_K2'   # screen-r4a variant B: averaging's (br_fp's, under a reply model) choice only past a 2-SE per-world gap
ROOT_OPERATORS=(ROOT_OPERATOR,NOISE_GATED_K2)   # the declarable live root operators; the first is the default
NOISE_GATE_K=2
CAUSES=('constructor_refused','depthn_refused','expired_in_queue','server_error','bad_request','class_refused')
CLASSES=('forced_switch','fresh_substitute','wish','sleep')   # root_class priority order
PROTOCOL_KEYS=('schema','request_id','battle_id','owner','stream_id','decision_index','generation','prefix_sha256','lines','own_request_line')
OURS=('rung','budget_seconds')
ARRIVAL='_arrival_monotonic'   # stamped by the server's queue (time.monotonic at arrival); a client's value is overwritten
STAGES=('queue_seconds','prepare_seconds','classify_seconds','structure_seconds','build_seconds','price_seconds','reduce_seconds')
DEPTH=2;WIDTH=9
FATAL=(MemoryError,)
WARMUP_BUDGET=3600.


def make_pricers(executor,forward,params,*,worlds,buckets=(1,2,4),block_lanes=2187,blocks_per_call=16,labels=guard_inputs.child_labels):
    """The live pricers: {bucket b: ObservedDepthNPricer(capacity b x W/2, depth 2, k 9, k1 9/9, prior, fixed_capacity)} with
    `forward` (the champion) and one program table shared after binding `params` (block programs compile once, P1 once per
    bucket); `labels` is the effect guard's turn-1 child gather (depthn_guard_inputs)."""
    if forward is None:raise ValueError('an explicit evaluator (the champion) required')
    guard_inputs.check_seam();table={};out={}
    for b in buckets:
        p=guard_inputs.ObservedDepthNPricer(executor,None,capacity=b*(worlds//2),depth=DEPTH,k_own=WIDTH,k_opp=WIDTH,k1_own=WIDTH,k1_opp=WIDTH,block_lanes=block_lanes,
            blocks_per_call=blocks_per_call,selection='prior',fixed_capacity=True,forward=forward,actor_capacity=len(params),labels=labels)
        p.bind(params);p.programs=table;out[b]=p
    return out


def decide(blocks,root_logits,root_legal):
    """One priced root's decision from its K blocks' backed-up worlds (blocks[k] = depthn_backups.backup(frame, row_k,
    learner_max)): search_hedge.averaging over the 2K worlds in block order plays, constructor_worlds' composition only
    reports (module docstring). dict(status COMPLETE or REFUSED, action_index, q_values, refusal, completed_q,
    policy_target, root_value, action_mean_q, action_policy_target, mean_q_agrees, reduction_note)."""
    from .d3_search_latency import compose,reduce_block
    legal=tuple(root_legal);prices=tuple(w for pair in blocks for w in pair)
    out=dict(status='REFUSED',action_index=None,q_values=None,refusal=None,completed_q=None,policy_target=None,root_value=None,action_mean_q=None,
        action_policy_target=None,mean_q_agrees=None,reduction_note=None)
    try:choice=hedge.averaging(prices,legal)
    except ValueError as error:return dict(out,refusal=f'AVERAGING_ERROR: {error}')
    if choice is None:return dict(out,refusal='AVERAGING_UNKNOWN_PRICE')
    action=hedge.top_action(choice.strategy);q=choice.q
    if not legal[action]or any(legal[a]!=(q[a]is not None)or legal[a]and not math.isfinite(q[a])for a in range(9)):
        return dict(out,refusal='AVERAGING_INVALID_RESULT')
    out.update(status='COMPLETE',action_index=action,q_values=list(q))
    try:   # reporting only: whatever happens here never changes the decision (MemoryError and non-Exception errors propagate)
        composed=compose([reduce_block(pair,root_logits,legal,original)for pair in blocks],root_logits,legal,original)
        if composed['status']!='LABELED':return dict(out,reduction_note='REDUCTION_REFUSED: a block reduce_completed_q refused')
        mean_q,target=cw.root_actions(composed['completed_q'],composed['policy_target'],legal)
        report=dict(completed_q=list(composed['completed_q']),policy_target=list(composed['policy_target']),root_value=composed['root_value'],
            action_mean_q=mean_q,action_policy_target=target,mean_q_agrees=mean_q==action)
    except FATAL:raise
    except Exception as error:return dict(out,reduction_note=f'REDUCTION_ERROR: {type(error).__name__}: {error}')
    return dict(out,**report)


def noise_gate(decision,blocks,root_logits,root_legal):
    """NOISE_GATED_K2 on a COMPLETE decide() result (DepthNLiveSearch's root_operator): the played action is
    search_hedge.noise_gated(per_world_q of the 2K worlds in block order, decide's averaging choice, the root logits' argmax over
    the legal actions, NOISE_GATE_K); q_values stay averaging's. Adds noise_gate = dict(candidate, greedy, passed: the candidate is
    played, mean_gap and stderr of Q_w(candidate) - Q_w(greedy), None when they are equal). A gate that cannot be evaluated
    refuses the root (NOISE_GATE_ERROR), never plays the ungated choice."""
    legal=tuple(root_legal);prices=tuple(w for pair in blocks for w in pair)
    candidate=decision['action_index']
    try:
        gs=hedge.grids(prices,legal)
        if gs is None or len(gs)<2:raise ValueError('two or more priced worlds required')
        qw=hedge.per_world_q(gs);greedy=hedge.logits_greedy(root_logits,legal);played=hedge.noise_gated(qw,candidate,greedy,NOISE_GATE_K)
        gaps=None if candidate==greedy else[q[candidate]-q[greedy]for q in qw]
        if not legal[played]or gaps is not None and not all(math.isfinite(g)for g in gaps):raise ValueError('a non-finite gap or illegal action')
    except FATAL:raise
    except Exception as error:
        return dict(decision,status='REFUSED',action_index=None,q_values=None,refusal=f'NOISE_GATE_ERROR: {type(error).__name__}: {error}')
    import statistics
    stats=dict(mean_gap=None,stderr=None)if gaps is None else dict(mean_gap=statistics.fmean(gaps),stderr=statistics.stdev(gaps)/math.sqrt(len(gaps)))
    return dict(decision,action_index=played,noise_gate=dict(candidate=candidate,greedy=greedy,passed=played==candidate,**stats))


def classify(constructor,request):
    """(root_classes, root_class_source) of a protocol request (module docstring): batch 2's protocol facts recomputed with
    the constructor's own helpers. Raises only on a malformed own request."""
    from . import phase_d_protocol_constructor as pc,phase_d_protocol_world as world,phase_d_tail as tail
    lines=tuple(request['lines']);packet=pc._packet(request['own_request_line'],request['owner']);classes=set()
    if any(packet.get('forceSwitch',())):classes.add('forced_switch')
    try:
        if world._sleep_origins(lines):classes.add('sleep')
    except cw.ROOT_ERRORS:pass
    try:
        if callable(getattr(constructor,'public_root',None)):public=constructor.public_root(request)   # prepare's root when held, else recomputed
        else:public=pc.capture_root(config=constructor.owning_config,contract=pc.ActorContract.champion(),lines=lines,battle_id=request['battle_id'],
            address=pc.DecisionAddress(request['stream_id'],request['decision_index'],int(request['owner'][1])-1),seed=constructor.seed&0xffffffff)
        payload=public.materialization_payload;checks,facts=tail.inspect_facts(packet,payload,lines)
    except cw.ROOT_ERRORS:return sorted(classes),'request_and_lines'
    triggered={c['kind']for c in checks if c.get('status')=='TRIGGERED'}
    if facts.get('replacement')is not None or any(k.startswith('PUBLIC_REPLACEMENT')for k in triggered):classes.add('forced_switch')
    if facts.get('fresh_substitutes')or'PUBLIC_SUBSTITUTE_NOT_FRESH_OR_PASSED'in triggered:classes.add('fresh_substitute')
    if facts.get('wish_set_turns')or'PUBLIC_WISH_TIMING_UNAVAILABLE'in triggered:classes.add('wish')
    try:
        if world._defer_sleep(payload)[1]:classes.add('sleep')
    except ValueError:classes.add('sleep')   # PUBLIC_SLEEP_FAINT_CONFLICT: a slp condition is there
    except(KeyError,IndexError,TypeError,AttributeError):pass
    return sorted(classes),'protocol_facts'


CROSSCHECK={'PUBLIC_REPLACEMENT_PHASE':'forced_switch','PUBLIC_FRESH_SUBSTITUTE':'fresh_substitute','PUBLIC_WISH_RECIPIENT_STATE':'wish'}


def class_crosscheck(classes,made):
    """Faithfulness cross-check of the recomputed classes on a fully constructed root, against the CLEAR checks the
    constructor's own world builds wrote (phase_d_tail.restore: PUBLIC_REPLACEMENT_PHASE, PUBLIC_FRESH_SUBSTITUTE,
    PUBLIC_WISH_RECIPIENT_STATE; _restore_sleep: PUBLIC_SLEEP_WORLD_POSTERIOR with CURRENT_SLEEP): 'agrees', or
    'DISAGREES: recomputed only [...]; worlds only [...]'. None when the worlds carry no diagnostics or nothing was classified.
    Reported only; the recomputed classes stay the source."""
    rows=[c for entry in getattr(made,'mapper_diagnostics',())or()for c in(entry.get('checks')or())if isinstance(c,dict)]
    if classes is None or not rows:return None
    seen={CROSSCHECK[c['kind']]for c in rows if c.get('status')=='CLEAR'and c.get('kind')in CROSSCHECK}
    if any(c.get('kind')=='PUBLIC_SLEEP_WORLD_POSTERIOR'and c.get('status')=='CLEAR'and c.get('applicability')=='CURRENT_SLEEP'for c in rows):seen.add('sleep')
    mine=set(classes)
    return'agrees'if mine==seen else f'DISAGREES: recomputed only {sorted(mine-seen)}; worlds only {sorted(seen-mine)}'


def root_class(classes):
    """The first of `classes` by CLASSES' priority, else 'other'."""
    return next((c for c in CLASSES if c in classes),'other')


def carry_structure(world):
    """A world's (state, public) pytree as (treedef, ((path, shape, dtype), ...)): what DepthN's carry signature hashes."""
    flat,tree=jax.tree_util.tree_flatten_with_path((world.state,world.public))
    return str(tree),tuple((jax.tree_util.keystr(path),tuple(np.shape(x)),str(x.dtype if hasattr(x,'dtype')else np.asarray(x).dtype))for path,x in flat)


def structure_diff(got,want,limit=3):
    """A short description of how carry structure `got` differs from `want`, or None when equal."""
    if got==want:return None
    g={p:(s,d)for p,s,d in got[1]};w={p:(s,d)for p,s,d in want[1]}
    out=[f'missing {p}'for p in w if p not in g]+[f'extra {p}'for p in g if p not in w]
    out+=[f'{p} {g[p][0]}/{g[p][1]}!={w[p][0]}/{w[p][1]}'for p in w if p in g and g[p]!=w[p]]
    out=out or['pytree structure differs']
    return'; '.join(out[:limit])+(f' (+{len(out)-limit} more)'if len(out)>limit else'')


def refusal_cause(reason):
    """constructor_refused for a root prepare() refused or with a refused world (Codex's all-W rule), else depthn_refused."""
    return('constructor_refused'if reason.startswith(('constructor-root:','WORLD_CONSTRUCTOR_REFUSED'))or reason=='materialization'
        else'depthn_refused')


def wire_b2_view(view):
    """A root's b2_view (constructor_worlds.b2_view_check) for the wire: dict(equal, keys, numeric_max_abs_diff, error), JSON-safe
    (bools, a finite float or None: a non-finite difference is None), or None when there is none."""
    if view is None:return None
    if not isinstance(view,dict):return dict(equal=None,keys=None,numeric_max_abs_diff=None,error=f'unexpected b2_view {type(view).__name__}')
    keys,diff,error=view.get('keys'),view.get('numeric_max_abs_diff'),view.get('error')
    return dict(equal=None if view.get('equal')is None else bool(view['equal']),keys=None if keys is None else{str(k):bool(v)for k,v in keys.items()},
        numeric_max_abs_diff=float(diff)if isinstance(diff,(int,float))and math.isfinite(diff)else None,error=None if error is None else str(error))


def _battle(key):
    """The battle of a ProtocolConstructor session key (rung, battle, owner, stream), else None."""
    return key[1]if type(key)is tuple and len(key)==4 else None


def _real(x):
    return type(x)in(int,float)and math.isfinite(x)


class DepthNLiveSearch:
    def __init__(self,*,constructor,pricers,params,binding,recipe,tree_commit,min_price_seconds=1.,max_requests=None,session_cap=256,
            archive_sha256=None,device='cpu',clock=time.monotonic,refuse_classes=(),redact_private_reveals=False,peer_view_b=False,reply=None,
            root_operator=ROOT_OPERATOR,ability_guard=False):
        """`pricers`: make_pricers' {bucket: DepthNPricer}; `binding`: the pricer side's WorldBinding (constructor_worlds.binding of
        the executor's source, engine metadata and catalog); `recipe`: the lane-order Recipe; `clock` must be the server's
        arrival clock (time.monotonic); `redact_private_reveals`: constructor_worlds' private-reveal redaction in every
        prepare_roots call (off by default; each response's private_reveals records what it found and redacted);
        `peer_view_b`: constructor_worlds' view b in every prepare_roots call (off by default; it implies the redaction, so
        redact_private_reveals then reads True; each built response carries its peer_view_b record, and a response of a
        search with it off has no peer_view_b key at all). `reply`: a depthn_reply_prior.ReplyPrior or None; with one, every
        priced root's requested worlds use the reply model's logits in place of root_peer_logits, so the averaging played is
        br_fp (the champion's averaging action is reported beside it); every response then carries reply_prior. The reply
        prior reads the worlds prepare_roots completed, so with peer_view_b its peer inputs are view b's. `root_operator` in
        ROOT_OPERATORS: NOISE_GATED_K2 plays search_hedge.noise_gated(per_world_q of the 2K priced worlds, averaging's choice on
        them, the root logits' argmax over the legal actions, 2): with a reply model the worlds are its (br_fp, the MC ladder's
        R5fp_ng); every response then names it and carries noise_gate (candidate, greedy, passed, mean_gap, stderr), and the
        effect guard's (c) reads the played action. The default's decisions and responses are unchanged."""
        if root_operator not in ROOT_OPERATORS:raise ValueError(f'root_operator in {ROOT_OPERATORS} required')
        self.root_operator=root_operator
        self.ability_guard=bool(ability_guard)   # the effect guard's ability arms (depthn_guard_inputs.effect_record's metadata); off by default
        if self.ability_guard and any(getattr(p,'labels',None)is not guard_inputs.ability_labels for p in pricers.values()):
            raise ValueError('ability_guard needs pricers observing with depthn_guard_inputs.ability_labels')
        if self.ability_guard:guard_inputs.ability_facts(constructor.metadata)   # a metadata without the move catalog fails here, at startup
        if type(redact_private_reveals)is not bool:raise ValueError('redact_private_reveals must be a bool')
        if type(peer_view_b)is not bool:raise ValueError('peer_view_b must be a bool')
        self.redact_private_reveals=redact_private_reveals or peer_view_b
        self.peer_view_b=peer_view_b
        W=constructor.count;self.rung=next((r for r,w in RUNGS.items()if w==W),None)
        if self.rung is None:raise ValueError(f'a constructor of W in {tuple(RUNGS.values())} worlds required, got {W!r}')
        self.W,self.K=W,W//2;self.buckets=tuple(sorted(pricers))
        if not self.buckets or any(type(b)is not int or b<=0 for b in self.buckets):raise ValueError('positive integer capacity buckets required')
        first=pricers[self.buckets[0]]
        for b in self.buckets:
            p=pricers[b]
            if(not isinstance(p,dn.DepthNPricer)or p.depth!=DEPTH or p.selection!='prior'or p.fixed_capacity is not True or p.capacity!=b*self.K
                    or p.k_own!=(WIDTH,)or p.k_opp!=(WIDTH,)or(p.k1_own,p.k1_opp)!=(WIDTH,WIDTH)or p.executor is not first.executor
                    or p.forward is not first.forward):
                raise ValueError(f'bucket {b}: a fixed-capacity DepthNPricer of capacity {b}x{self.K}, depth 2, k 9, k1 9/9, prior, sharing the executor and evaluator required')
        self.forward=first.forward;ident=evaluators.identity(self.forward)
        if ident[0]!='champion':raise ValueError('the champion evaluator required (forward=)')
        max_requests=self.buckets[-1]if max_requests is None else max_requests
        if type(max_requests)is not int or not 1<=max_requests<=self.buckets[-1]:raise ValueError(f'max_requests in 1..{self.buckets[-1]} (the largest bucket) required')
        if type(session_cap)is not int or session_cap<max_requests:raise ValueError('a session cap of at least max_requests battles required')
        if not _real(min_price_seconds)or min_price_seconds<0:raise ValueError('finite min_price_seconds >= 0 required')
        if type(tree_commit)is not str or not tree_commit:raise ValueError('the tree commit required')
        if type(params)is not tuple or not params:raise ValueError('the champion bank (params,) required')
        refuse_classes=tuple(refuse_classes)
        if any(c not in CLASSES for c in refuse_classes):raise ValueError(f'refuse_classes from {CLASSES} required, got {refuse_classes!r}')
        self.refuse_classes=tuple(sorted(set(refuse_classes),key=CLASSES.index));self.carry=None;self.warming=False
        if cw.constructor_binding(constructor)!=binding:raise ValueError('the constructor builds under another binding than the pricer\'s')
        if isinstance(self.forward,evaluators.ChampionEvaluator):evaluators.check_binding(self.forward,constructor.catalog)
        self.constructor,self.pricers,self.params,self.binding,self.recipe=constructor,dict(pricers),params,binding,recipe
        self.tree_commit,self.min_price_seconds,self.max_requests,self.session_cap=tree_commit,float(min_price_seconds),max_requests,session_cap
        self.archive_sha256=archive_sha256 or getattr(self.forward,'archive_sha256',None)or ident[1]
        from .depthn_reply_prior import ReplyPrior
        if reply is not None and not isinstance(reply,ReplyPrior):raise ValueError('reply: a depthn_reply_prior.ReplyPrior or None required')
        self.device,self.clock,self.reply=device,clock,reply;self.reply_completed=0   # roots the reply prior completed (warmup's check)
        self.battles=OrderedDict();self.evicted=0;self.warm=None
        self.pricer_identity=dict(engine='DepthNPricer',depth=DEPTH,k=[WIDTH,WIDTH],k1=[WIDTH,WIDTH],selection='prior',W=W,K=self.K,
            buckets=list(self.buckets),capacities=[b*self.K for b in self.buckets],fixed_capacity=True,block_lanes=first.block_lanes,
            blocks_per_call=first.blocks_per_call,precision=getattr(self.forward,'precision',None)or(ident[2]if len(ident)>2 else None),
            evaluator=evaluators.describe(self.forward)['identity'],world_source=cw.SOURCE,backup='learner_max',root_operator=self._operator(),
            root_operator_impl=ROOT_OPERATOR_IMPL,matmul_precision=str(jax.config.jax_default_matmul_precision or'default'))

    def _redaction(self):
        """prepare_roots' keywords for the private-reveal redaction and view b: each given only when it is on (a search built
        before the flag, or without __init__, has none: off)."""
        return{**({'redact_private_reveals':True}if getattr(self,'redact_private_reveals',False)else{}),
            **({'peer_view_b':True}if self._view_b()else{})}

    def _operator(self):
        """The live root operator (ROOT_OPERATOR for a search built before the option, or without __init__)."""
        return getattr(self,'root_operator',ROOT_OPERATOR)

    def _gated(self):return self._operator()==NOISE_GATED_K2

    def _view_b(self):
        """Whether this search builds view b (peer_view_b; False for one built before the flag or without __init__)."""
        return getattr(self,'peer_view_b',False)is True

    @property
    def programs(self):
        """Every compiled program of the live pricers (the union of their tables; one shared table from make_pricers)."""
        return{k for p in self.pricers.values()for k in p.programs}

    # Requests -----------------------------------------------------------------------------------------------------------------------
    def _response(self,request_id,n):
        return dict(schema=SCHEMA,request_id=request_id,rung=self.rung,archive_sha256=self.archive_sha256,status='REFUSED',action_index=None,q_values=None,
            refusal=None,cause=None,requested_worlds=self.W,constructed_worlds=0,decision_seconds=None,root_operator=self._operator(),
            root_operator_impl=ROOT_OPERATOR_IMPL,completed_q=None,policy_target=None,root_value=None,action_mean_q=None,action_policy_target=None,
            mean_q_agrees=None,reduction_note=None,root_view_source=None,b2_view=None,effect_guard=None,private_reveals=None,block_causes=None,world_refusals=None,
            timings=dict.fromkeys(STAGES),
            batch_size=n,bucket=None,
            root_classes=None,root_class=None,root_class_source=None,root_class_crosscheck=None,
            tree_commit=self.tree_commit,pricer=dict(self.pricer_identity),**({'peer_view_b':None}if self._view_b()else{}),
            **({}if getattr(self,'reply',None)is None else dict(reply_prior=None)),**({'noise_gate':None}if self._gated()else{}))

    @staticmethod
    def _problem(request):
        """Why a request is malformed (MALFORMED_REQUEST: ...), or None. A well-formed ARRIVAL stamp is accepted whoever set it:
        process_many cannot tell a client's from the server's, so a client-sent `_arrival_monotonic` is only safe behind
        scripts/depthn_live_server's ArrivalQueue, which overwrites it with the server's own arrival time before any batch."""
        if not isinstance(request,dict):return'MALFORMED_REQUEST: a JSON object required'
        keys=set(request)-{ARRIVAL};want=set(PROTOCOL_KEYS)|set(OURS)
        if keys!=want:return'MALFORMED_REQUEST: missing '+','.join(sorted(want-keys))+'; unexpected '+','.join(sorted(map(str,keys-want)))
        if request['schema']!=SCHEMA:return'MALFORMED_REQUEST: schema'
        if type(request['request_id'])is not str or not request['request_id']:return'MALFORMED_REQUEST: request_id'
        if type(request['rung'])is not str:return'MALFORMED_REQUEST: rung'
        if not _real(request['budget_seconds'])or request['budget_seconds']<=0:return'MALFORMED_REQUEST: budget_seconds (finite > 0)'
        if ARRIVAL in request and not _real(request[ARRIVAL]):return'MALFORMED_REQUEST: arrival stamp'
        return None

    def process_many(self,requests):
        """One response per request, in order (module docstring); batches of at most max_requests."""
        requests=list(requests);out=[]
        for start in range(0,len(requests),self.max_requests):out.extend(self._batch(requests[start:start+self.max_requests]))
        return out

    def _batch(self,requests,*,bucket=None):
        started=self.clock();n=len(requests);out=[None]*n;todo=[]
        for i,request in enumerate(requests):
            rid=request.get('request_id')if isinstance(request,dict)else None
            row=self._response(rid if type(rid)is str else None,n);problem=self._problem(request)
            if isinstance(request,dict)and type(request.get('rung'))is str:row['rung']=request['rung']   # the client matches its own rung
            if problem is None and request['rung']!=self.rung:problem=f'RUNG_NOT_SERVED: {request["rung"]!r} (this server serves {self.rung})'
            if problem is not None:out[i]=dict(row,cause='bad_request',refusal=problem);continue
            wait=max(0.,started-request.get(ARRIVAL,started));row['timings']['queue_seconds']=wait
            if request['budget_seconds']-wait<self.min_price_seconds:
                out[i]=dict(row,cause='expired_in_queue',refusal=f'EXPIRED_IN_QUEUE: waited {wait:.3f}s of a {float(request["budget_seconds"]):.3f}s budget, '
                    f'under the {self.min_price_seconds:.3f}s minimum to price');continue
            todo.append((i,{k:request[k]for k in PROTOCOL_KEYS},row))
        if todo:
            classified={}   # filled per root as soon as it is classified, so a batch exception still reports the classes
            try:done=self._search(todo,started,bucket,classified)
            except FATAL:raise
            except Exception as error:
                done={}
                for i,_,row in todo:
                    labels=dict(classified.get(i,{}));seconds={k:labels.pop(k,None)for k in('classify_seconds','structure_seconds')}
                    done[i]=dict(row,**labels,timings=dict(row['timings'],**seconds),cause='server_error',refusal=f'{type(error).__name__}: {error}')
            finally:self._evict()
            for i,row in done.items():out[i]=row
        for i,row in enumerate(out):
            try:encode(row)
            except(ValueError,TypeError)as error:
                out[i]=dict(self._response(row.get('request_id'),n),cause='server_error',refusal=f'RESPONSE_NOT_ENCODABLE: {type(error).__name__}: {error}')
        return out

    def _search(self,todo,started,bucket,classified):
        """prepare -> classify -> gates (refused classes, carry structure) -> build -> one price -> decide for the batch's valid
        requests; {response index: response}."""
        for _,p,_ in todo:
            if type(p['battle_id'])is str:self.battles[p['battle_id']]=None;self.battles.move_to_end(p['battle_id'])
        prepared=cw.prepare_roots(self.constructor,[p for _,p,_ in todo],device=self.device,
            **self._redaction());t_prepare=time.perf_counter()
        gated=[];gates={}
        for(i,p,_),item in zip(todo,prepared):
            clock=time.perf_counter()
            try:classes,source=classify(self.constructor,p)
            except FATAL:raise
            except Exception as error:classes,source=None,f'UNCLASSIFIED: {type(error).__name__}: {error}'
            full=type(item)is cw.Prepared and item.made.fully_constructed   # a root with a refused world keeps its constructor cause
            classified[i]=dict(root_classes=classes,root_class=None if classes is None else root_class(classes),root_class_source=source,
                root_class_crosscheck=class_crosscheck(classes,item.made)if full else None,classify_seconds=time.perf_counter()-clock,structure_seconds=None)
            refused=[c for c in self.refuse_classes if classes and c in classes]
            if full and refused:gates[i]=('class_refused','ROOT_CLASS_REFUSED:'+','.join(refused))
            elif full and classes is None and self.refuse_classes:gates[i]=('class_refused','ROOT_CLASS_UNKNOWN')
            elif full:
                clock=time.perf_counter();diff=self._structure(item,p,classes);classified[i]['structure_seconds']=time.perf_counter()-clock
                if diff:gates[i]=('depthn_refused','WORLD_STRUCTURE_MISMATCH:'+diff)
            gated.append(cw.RootRefusal(item.request_id,'GATED','',item.seconds,item.binding)if i in gates else item)
        t_classify=time.perf_counter()
        made=cw.build(self.constructor,gated,self.pricers[self.buckets[0]],self.params,binding=self.binding,recipe=self.recipe,only_complete=True)
        t_build=time.perf_counter();chosen=frame=None
        if made.requests:
            chosen=bucket if bucket is not None else next((b for b in self.buckets if len(made.requests)<=b*self.K),None)
            if chosen is None:raise ValueError(f'{len(made.requests)} blocks exceed the largest bucket\'s capacity')
            pricer=self.pricers[chosen];observer=pricer if isinstance(pricer,guard_inputs.ObservedDepthNPricer)else None
            if observer is not None:observer.observe()
            try:frame,_=cw.price(pricer,made,self.binding,self.recipe,self.params)
            finally:children,seam_notes=observer.observed()if observer is not None else([],['the pricer has no observer seam'])
        t_price=time.perf_counter();results=[];causes=backups.refusal_summary(frame)['roots']if made.requests else()
        items={p['request_id']:item for(_,p,_),item in zip(todo,prepared)}
        by_id={p['request_id']:p for _,p,_ in todo};labels=getattr(observer,'labels',guard_inputs.child_labels)if made.requests else None
        for root in made.roots:
            view=dict(root_view_source=root.get('root_view_source'),b2_view=wire_b2_view(root.get('b2_view')))
            if root['status']!='PRICED':results.append(dict(request_id=root['request_id'],status='REFUSED',refusal=root['reason'],block_causes=None,**view));continue
            block=[causes[r]['cause']for r in root['rows']];base=dict(request_id=root['request_id'],block_causes=block,**view)
            if any(block):results.append(dict(base,status='REFUSED',refusal=next(c for c in block if c)));continue   # DepthN's rule: a refused block refuses the root
            try:blocks=[backups.backup(frame,r,backups.learner_max)for r in root['rows']]   # WorldPrices: averaging's value and logit rules
            except ValueError as error:results.append(dict(base,status='REFUSED',refusal=f'AVERAGING_ERROR: {error}'));continue
            reply=None
            if getattr(self,'reply',None)is not None:   # br_fp: the reply model's logits on the same worlds; the champion's averaging is reported
                item=items[root['request_id']]
                try:replied,seconds,check=self.reply.apply(blocks,item.made.worlds,int(item.detached.address.owner))
                except FATAL:raise
                except Exception as error:results.append(dict(base,status='REFUSED',refusal=f'REPLY_PRIOR_ERROR: {type(error).__name__}: {error}'));continue
                champion=decide(blocks,root['root_logits'],root['root_legal'])
                reply=dict(operator='br_fp',views=self.reply.mode,seconds=seconds,champion_action_index=champion['action_index'],champion_q_values=champion['q_values'],
                    **{f'reply_{k}':v for k,v in (getattr(self.reply,'identity',None)or{}).items()})   # the served archive: the proxy's and panel's pin
                if check is not None:   # --reply-views check: the jitted views' logits, averaged and reported beside (eager plays)
                    if check.get('error'):reply.update(jit_error=check['error'])
                    else:
                        alt=decide(check['blocks'],root['root_logits'],root['root_legal'])
                        reply.update(jit_action_index=alt['action_index'],jit_q_values=alt['q_values'],jit_max_abs_logit_diff=check['max_abs_logit_diff'])
                blocks=replied
            decision=decide(blocks,root['root_logits'],root['root_legal'])
            if self._gated()and decision['status']=='COMPLETE':decision=noise_gate(decision,blocks,root['root_logits'],root['root_legal'])
            if reply is not None:
                decision['reply_prior']=reply;self.reply_completed+=decision['status']=='COMPLETE'
            if decision['status']=='COMPLETE':decision['effect_guard']=self._effect_guard(frame,root,made,by_id[root['request_id']],children,seam_notes,
                decision['action_index'],labels,reply_logits=None if reply is None else[None if w is None or not w.opponent_requested else
                    w.opponent_logits for pair in blocks for w in pair])   # br_fp: the union guard on the reply model's logits
            results.append(dict(base,**decision))
        t_reduce=time.perf_counter();decided=self.clock()-started
        if len(prepared)!=len(todo)or len(results)!=len(todo)or any(r['request_id']!=p['request_id']for r,(_,p,_)in zip(results,todo)):
            raise ValueError('the decisions do not answer the batch\'s requests in order')
        out={}
        for(i,_,row),item,result in zip(todo,prepared,results):
            worlds=item.made.worlds if type(item)is cw.Prepared else();labels=dict(classified[i])
            timings=dict(row['timings'],prepare_seconds=item.seconds,classify_seconds=labels.pop('classify_seconds'),structure_seconds=labels.pop('structure_seconds'),
                build_seconds=t_build-t_classify,price_seconds=t_price-t_build,reduce_seconds=t_reduce-t_price)
            base=dict(row,**labels,constructed_worlds=sum(w.state is not None for w in worlds),world_refusals=[dict(world=w.index,kinds=list(w.refusals))for w in worlds if w.refusals],
                timings=timings,bucket=chosen,block_causes=None if result['block_causes']is None else list(result['block_causes']),
                root_view_source=result['root_view_source'],b2_view=result['b2_view'],
                private_reveals=getattr(item,'private_reveals',None)if type(item)is cw.Prepared else None,
                **({'peer_view_b':getattr(item,'peer_view_b',None)if type(item)is cw.Prepared else None}if self._view_b()else{}))
            if i in gates:out[i]=dict(base,cause=gates[i][0],refusal=gates[i][1])
            elif result['status']=='COMPLETE':
                out[i]=dict(base,**{k:result[k]for k in('status','action_index','q_values','completed_q','policy_target','root_value','action_mean_q',
                    'action_policy_target','mean_q_agrees','reduction_note','effect_guard')},decision_seconds=decided,
                    **({}if getattr(self,'reply',None)is None else dict(reply_prior=result.get('reply_prior'))),
                    **({'noise_gate':result.get('noise_gate')}if self._gated()else{}))
            else:out[i]=dict(base,cause=refusal_cause(result['refusal']),refusal=result['refusal'])
        return out

    def _effect_guard(self,frame,root,made,request,children,notes,choice,labels,reply_logits=None):
        """The root's effect guard record (depthn_guard_inputs.effect_record over this frame's observed P1 children); the
        fail-closed error record when its inputs cannot be read (pokezero.depthn_effect_guard.error_record)."""
        try:
            pairs=made.worlds.pairs;worlds=[w for r in root['rows']for w in pairs[r]];owner={'p1':0,'p2':1}[request['owner']]
            return guard_inputs.effect_record(frame,root,worlds,children,request,owner,self.constructor.metadata['vocabulary'],choice,labels=labels,
                notes=notes,reply_logits=reply_logits,metadata=self.constructor.metadata if getattr(self,'ability_guard',False)else None)
        except FATAL:raise
        except Exception as error:return effect_guard.error_record(f'{type(error).__name__}: {error}')

    def _structure(self,item,request,classes):
        """WORLD_STRUCTURE_MISMATCH detail for a fully constructed root whose worlds' carry differs from the reference (recorded
        at warmup from its first fully constructed root, every other warmup root held to it; before any warmup, the first
        such root seen, provisionally), else None. The reference keeps its root's address and classes."""
        for world in item.made.worlds:
            got=carry_structure(world)
            if self.carry is None:
                self.carry=dict(structure=got,signature=dn.carry_signature(((world,),)),provisional=not self.warming,
                    root=dict(battle_id=request['battle_id'],owner=request['owner'],stream_id=request['stream_id'],decision_index=request['decision_index'],
                        prefix_sha256=request['prefix_sha256'],root_classes=classes))
            diff=structure_diff(got,self.carry['structure'])
            if diff:return f'world {world.index}: {diff}'
        return None

    # Sessions -----------------------------------------------------------------------------------------------------------------------
    def _evict(self):
        """Drop the sessions and attempted addresses of the least recently prepared battles beyond session_cap."""
        drop=set()
        while len(self.battles)>self.session_cap:drop.add(self.battles.popitem(last=False)[0])
        if not drop:return
        sessions,seen=self.constructor.sessions,self.constructor.seen
        for key in[k for k in sessions if _battle(k)in drop]:del sessions[key]
        seen.difference_update([a for a in seen if type(a)is tuple and len(a)==3 and _battle(a[0])in drop])
        if callable(getattr(self.constructor,'drop_roots',None)):self.constructor.drop_roots(drop)
        self.evicted+=len(drop)

    def reset_sessions(self):
        """Forget every constructor session and attempted address (warmup does, around each pass; a bench between rosters)."""
        self.constructor.seen.clear();self.constructor.sessions.clear();self.battles.clear()
        if callable(getattr(self.constructor,'drop_roots',None)):self.constructor.drop_roots()

    # Warmup -------------------------------------------------------------------------------------------------------------------------
    def missing_programs(self,b):
        """The programs a live call at bucket b runs that its pricer's table lacks (DepthNPricer program keys): P1 and the prior
        selection at its capacity, the gather from its turn-1 pool, the ply-2 leaf block program (shared by every bucket)."""
        cap=b*self.K;keys=self.pricers[b].programs
        want=dict(P1=lambda k:k[0]=='P1'and k[1]==cap,select=lambda k:k[0]=='select'and k[1][0]==2*cap,
            gather=lambda k:k[0]=='gather'and k[1]==2*cap*dn.LANES,leaf=lambda k:k[0]=='leaf')
        return[name for name,test in want.items()if not any(test(k)for k in keys)]

    def warmup(self,requests):
        """The full path on saved requests (their eight address keys) once cold and once warm per bucket, each forced to that
        bucket, so every program compiles before READY. Windows of the saved requests are tried until one prices at the bucket
        with every program of a live call compiled (missing_programs; RuntimeError if none does, e.g. saved roots whose every
        turn-1 cell is terminal never reach the leaf program). Returns dict(buckets={b: seconds, compiles and outcomes},
        programs, seconds, carry_signature, carry_reference: the reference root's address and classes). It also records the
        carry reference of the structure check: every warmup root must share it (RuntimeError on any WORLD_STRUCTURE_MISMATCH:
        in the priced windows, and over every warmup root prepared once more afterwards), and it must be the only
        constructor-world carry signature in the warmed program keys."""
        trace_release.caches()   # a JAX that moved jit's tracing caches fails here, not after the lowering
        keys=PROTOCOL_KEYS[2:];pool=[]
        for r in requests:
            if not isinstance(r,dict)or any(k not in r for k in keys):raise ValueError('saved requests with '+', '.join(keys)+' required')
            pool.append({k:r[k]for k in keys})
        if not pool:raise ValueError('at least one saved request required')
        started=time.perf_counter();self.carry=None;self.warming=True;self.reply_completed=0;self.reply_warm=None
        # No full (gen-2) collection while the programs trace: each would rescan the trace's own growing live state (tens of
        # millions of objects) and find little garbage. Young collections go on; the thresholds come back only after
        # _release_traces' one collection, so no full pass ever runs over the still-retained traces.
        thresholds=gc.get_threshold();gc.set_threshold(thresholds[0],thresholds[1],1<<30)
        try:
            try:report,batches=self._warm_buckets(pool);labels=self._hold_to_reference(pool)
            finally:self.warming=False;self.reset_sessions()
            missing=[b for b in self.buckets if isinstance(self.pricers[b],guard_inputs.ObservedDepthNPricer)
                and 2*b*self.K*dn.LANES not in self.pricers[b].label_rows]
            if missing or labels is None:
                raise RuntimeError(f'warmup: the effect guard\'s label gather did not compile for buckets {missing} (P1 children) or the root worlds '
                    f'({"missing" if labels is None else "ok"})')
            if getattr(self,'reply',None)is not None and not self.reply_completed:   # a reply archive that fails on every root never reaches READY
                raise RuntimeError('warmup: the reply prior completed no warmup root (every one refused REPLY_PRIOR_ERROR or refused before it)')
            if getattr(self,'reply',None)is not None and set(self.reply_warm or ())!={0,1}:
                raise RuntimeError('warmup: the reply prior did not compile both owner seats (no fully constructed warmup root)')
            signatures={k[-1][2]for k in self.programs if type(k[-1])is tuple and k[-1][:2]==('worlds',cw.SOURCE)}
            if self.carry is None or signatures!={self.carry['signature']}:
                raise RuntimeError(f'warmup: the recorded carry signature {None if self.carry is None else self.carry["signature"]} is not the warmed programs\' {sorted(signatures)}')
            released=self._release_traces(batches)
        finally:gc.set_threshold(*thresholds)
        self.warm=dict(buckets=report,programs=len(self.programs),seconds=time.perf_counter()-started,carry_signature=self.carry['signature'],
            trace_release=released,
            carry_reference=self.carry['root'],labels=dict(p1_rows={b:2*b*self.K*dn.LANES for b in self.buckets},root_rows=labels),
            **({}if getattr(self,'reply',None)is None else dict(reply=dict(views=self.reply.mode,seats_seconds=self.reply_warm))))
        return self.warm

    def _release_traces(self,batches):
        """The end of warmup, once every program has compiled: empty jit's tracing caches (gpu_engine.trace_release: the
        warmed programs' traces, kept alive there by the executor's long-lived jitted functions, are about 66M objects after
        a DN2W16 warmup, and every full collection while serving would rescan them), collect once, then run each bucket's warm
        batch again. That last pass recompiles, before READY, the few small programs whose executable cache is keyed on a
        released trace (eagerly evaluated jaxprs, e.g. vmapped jax.random.fold_in) instead of on their function, so serving
        compiles nothing. Each re-warm must answer as that bucket's warm pass did, (status, cause, bucket, action_index) per
        row (`batches[b]` = (batch, the warm pass's rows)); RuntimeError otherwise, so a broken re-warm never reaches READY.
        Returns dict(entries dropped, gc_seconds, rewarm_seconds, jax: the JAX version released under)."""
        released=trace_release.release();clock=time.perf_counter();gc.collect();collected=time.perf_counter()-clock
        clock=time.perf_counter();outcome=lambda rows:[(r['status'],r['cause'],r['bucket'],r['action_index'])for r in rows]
        for b,(batch,warm) in batches.items():
            self.reset_sessions();again=self._batch(batch,bucket=b)
            if outcome(again)!=outcome(warm):
                self.reset_sessions()
                raise RuntimeError(f'warmup: the re-warm of bucket {b} answered {outcome(again)}, its warm pass {outcome(warm)}')
        self.reset_sessions()
        return dict(entries=released,gc_seconds=collected,rewarm_seconds=time.perf_counter()-clock,jax=jax.__version__)

    def _hold_to_reference(self,pool):
        """Every warmup root, not only those of the windows that priced: prepared once more (fresh sessions and attempts), each
        fully constructed one held to the carry reference; RuntimeError naming every root that differs. The first one's stacked
        worlds also go through the effect guard's label gather (its root-world shape, [W, 1, ...], compiles here whether or not
        a warmup root decided); returns that row count (None without a fully constructed root)."""
        self.reset_sessions()
        requests=[dict(r,schema=SCHEMA,request_id=f'warmup-structure-{j}')for j,r in enumerate(pool)]
        odd=[];labels=None;tried=False;gather=getattr(self.pricers[self.buckets[0]],'labels',guard_inputs.child_labels)
        for request,item in zip(requests,cw.prepare_roots(self.constructor,requests,device=self.device,
                **self._redaction())):
            if type(item)is not cw.Prepared or not item.made.fully_constructed:continue
            if not tried:
                tried=True
                try:jax.block_until_ready(gather(guard_inputs.stack_worlds(item.made.worlds)));labels=len(item.made.worlds)
                except FATAL:raise
                except Exception:labels=None   # warmup then refuses READY: the gather does not run on these worlds
                if getattr(self,'reply',None)is not None:   # both owner seats' reply programs, whichever seat the warmup roots had
                    try:self.reply_warm=self.reply.warm(item.made.worlds)
                    except FATAL:raise
                    except Exception as error:raise RuntimeError(f'warmup: the reply prior did not compile both seats: {type(error).__name__}: {error}')from error
            diff=self._structure(item,request,None)
            if diff:odd.append(f'{request["battle_id"]}/{request["decision_index"]}: {diff}')
        if odd:raise RuntimeError(f'warmup: every warmup root must share one carry structure; against the reference root {self.carry["root"]}: '+'; '.join(odd))
        return labels

    def _warm_buckets(self,pool):
        report={};batches={}
        for b in self.buckets:
            for attempt in range(-(-len(pool)//b)):
                batch=[dict(pool[(attempt*b+j)%len(pool)],schema=SCHEMA,request_id=f'warmup-b{b}-{attempt}-{j}',rung=self.rung,budget_seconds=WARMUP_BUDGET)
                    for j in range(b)]
                self.reset_sessions();clock=time.perf_counter();rows=self._batch(batch,bucket=b);cold=time.perf_counter()-clock
                odd=[r['refusal']for r in rows if str(r['refusal']or'').startswith('WORLD_STRUCTURE_MISMATCH')]
                if odd:raise RuntimeError(f'warmup: every warmup root must share one carry structure; against the reference root {self.carry["root"]}: '+'; '.join(odd))
                if any(r['bucket']==b for r in rows)and not self.missing_programs(b):break
            else:
                self.reset_sessions();missing=self.missing_programs(b)
                raise RuntimeError(f'warmup: no saved request priced at bucket {b} with every program compiled (missing: {", ".join(missing) or "none"}): '
                    +'; '.join(sorted({f'{r["cause"]}: {r["refusal"]}'for r in rows})))
            compiles=self.pricers[b].compiles
            self.reset_sessions();clock=time.perf_counter();again=self._batch(batch,bucket=b);warm=time.perf_counter()-clock
            report[b]=dict(cold_seconds=cold,warm_seconds=warm,attempts=attempt+1,roots=b,complete=sum(r['status']=='COMPLETE'for r in rows),
                causes=sorted({r['cause']for r in rows if r['cause']}),compiled=len(compiles),lower_seconds=sum(c[1]for c in compiles),
                compile_seconds=sum(c[2]for c in compiles),warm_compiled=len(self.pricers[b].compiles)if any(r['bucket']==b for r in again)else None)
            batches[b]=(batch,again)
        return report,batches
