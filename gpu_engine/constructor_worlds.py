"""Codex's Phase D protocol constructor worlds as DepthNPricer input (gap 4, part A): depth-N prices on Showdown roots.

Design. ProtocolConstructor.prepare(request, index) folds a Showdown protocol prefix plus the literal own request (B2)
and returns (detached device_public_worlds.Root, phase_d_live_worlds.WorldSet of W in {2, 8, 16, 32} worlds, B2's own
legal mask). Its worlds are D3's world type (device_public_world_state.World: the engine BattleState plus the public
tracker carry), so DepthN's executor steps them unchanged; only the carry's pytree differs from D3's (from_b2 adds four
dict keys and leaves two members None). This module turns prepared roots into DepthN requests and pre-built worlds:

- prepare_roots: prepare() per protocol request, on the CPU device by default (its eager per-world array updates would
  otherwise dispatch one by one on a GPU), timed and synced per root, each result stamped with the constructor's
  binding. A raised refusal (phase_d_live_worlds.Refusal, or the errors ProtocolConstructor.construct treats as
  refusals) is a RootRefusal: the root gets no request.
- build: W worlds become K = W/2 blocks of two, one Request per block, packed into ONE frame (row = the root's rows[b],
  root-major), since the worlds are pre-built (no per-block recipe or re-materialization). Plans are synthesized:
  selected=True (the quarter selection is bypassed), Start(world index, the constructor's world seed, no fixtures,
  refusals); source_fixture_unknown_mass is NaN (no D3 source-fixture draw exists here; the refused-world share is the
  root's `refused_world_share`). `root_legal` is B2's own mask (pre 2 holds each world's own encoder mask to it, as the
  ladder's legality check). `root_logits` are the pricer's own evaluator (forward=, e.g. the champion) on root_inputs,
  as gate.bind_roots. Actors default to 0/0 (one evaluator bank, the champion's (champion,)).
- root_inputs are world 0's own view under the pricer's executor (the v5 encoder). The turn-1 input check (pre 3+) then
  holds every OTHER world of every block to world 0's view: a consistency check across the sampled worlds only. Unlike
  D3, whose root inputs are the collector's own observation of the real root, nothing gates world 0's view against the
  true observation. As a diagnostic (reported, never a gate), each root's `b2_view` compares the evaluator's view of
  root_inputs (its `view`; the 23-token v4 prefix for an evaluator without one) with B2's own encoding of the literal
  root, detached.view['_b2_root_inputs'], where the root carries it (from_b2 stores it): per key equal, and the
  largest numeric difference.
- Refusals: a world without a state (a constructor world refusal) is pre -1 in its block
  (`physical-materialization:<refusals>` in the world report), so that block refuses and the root refuses: DepthN's
  rule (depthn_backups._root), the constructor's all-W `fully_constructed` rule. `only_complete=True` refuses such
  roots before pricing instead (saves the device work; the frame then never shows them). A root whose world 0 is
  refused takes its root view from the lowest constructed world (it refuses anyway); one with no constructed world, an
  erroring root view or an unusable root forward is refused before pricing with a named reason.
- Bindings: `binding(source, metadata, catalog)` is a DepthN WorldBinding (source hash, ordered rows, the
  source/engine-namespace binding sha256, the engine metadata's canonical-JSON sha256, the encoder catalog's checkpoint,
  category vocabulary and masks). build requires every prepared root's stamp and the constructor's binding to equal the
  pricer side's (the run's D3 Tables source, engine metadata and catalog), and with a champion evaluator
  champion_evaluator.check_binding(evaluator, the constructor's catalog); the pricer checks the binding again on every
  call: fail closed.
- Programs: the worlds go in as DepthN's PrebuiltWorlds under the tag SOURCE, so every compiled-program key gains
  ('worlds', SOURCE, carry signature) after the evaluator's identity: constructor and D3 worlds never share a program,
  nor do two carry structures, nor two evaluators.
- price: `prior` = one packed pricing. `searched` = the full-width depth-2 pass (a DepthNPricer at depth 2, k 9, prior,
  same executor, evaluator and k1) then searched_maps on the host, then depth N with the maps, both passes on the SAME
  PrebuiltWorlds (prepare refuses a second attempt at an address, and none is needed). Nothing in it is D3-only. A batch
  without requests (every root refused) prices nothing and returns (None, None).
- reduce: per root, depthn_backups.backup(rule) and reduce_completed_q per block, then the K blocks by the equal-weight
  mean of their completed-Q (d3_search_latency.compose: every world weighted 1/W); any refused block refuses the root.
  reduce_completed_q's target range check (v5_contract._real, RANGE_EDGE) raises when a block's completed-Q rounds outside
  [-1, 1] (an action winning against every reply in both worlds, its reply prior summing to 1 + ulp): that refuses the
  root only, its cause the first refused block's if any, else 'reduction', the error in `reduction_error`; the batch's
  other roots still label. Any other ValueError there is malformed input (build validated it) and raises, as
  v2_search_head_to_head's _reduce_tolerating_rounding matches the same message.
  Two moves: `action_mean_q`, the argmax of that mean completed-Q over the legal actions (lower action on ties): the
  ladder's root operator (phase_d_ladder_search.reduce_q); and `action_policy_target`, the argmax of the improved
  target softmax(root logits + q) (D's training target, the latency tool's move).

Chance: DepthN samples each node's own keys; constructor states carry kernel_seed-derived keys, so prices are
reproducible but not bit-comparable to the ladder's R1/R7, which fold a domain seed into the keys.

Constructor API this module depends on (part B, the batch 2 and tail sync, must keep it): ProtocolConstructor.prepare's
(Root, WorldSet, legal) return and its `count`, `source`, `metadata`, `catalog` attributes; Refusal's `kind`/`detail`;
WorldSet's `root`, `worlds`, `seeds`, `requested_worlds`; World's fields; Root's `address`, `source_binding_sha256` and
`view['_b2_root_inputs']` (diagnostic only); Source's `source_hash`, `source_rows_sha256`, `binding_sha256`. A changed
carry structure needs no change here (new programs compile under its own signature); a changed engine metadata or
source binding fails closed until both sides agree.
"""
from contextlib import nullcontext
from dataclasses import dataclass,replace
import hashlib
import json
import time

import jax
import numpy as np

from pokezero import v5_gumbel as original
from . import champion_evaluator as evaluators,depthn_backups as backups,device_depth2_pricer as depth2,device_depthn_pricer as dn
from . import device_public_teacher_batch as batch,device_public_worlds as plans

SOURCE='phase-d-protocol-constructor'
# What ProtocolConstructor.construct treats as a root refusal (phase_d_live_worlds.Refusal is a ValueError).
ROOT_ERRORS=(ValueError,KeyError,TypeError,IndexError,AttributeError)
UNNAMED='UNNAMED_CONSTRUCTOR_WORLD_REFUSAL'
RANGE_EDGE='finite in-range numerical value required'   # v5_contract._real's error: on reduce's path only RootImprovement.validate raises it
B2_TOKEN_KEYS=('numeric_features','categorical_ids','token_type_ids','attention_mask')   # B2's encode(): [23,F] / [23]


# Bindings ---------------------------------------------------------------------------------------------------------------------------
def metadata_sha256(metadata):
    """sha256 of the engine metadata's canonical JSON (sorted keys, no spaces)."""
    return hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def encoder_identity(catalog):
    """The encoder catalog's identity: checkpoint sha256, category vocabulary sha256 and size, feature masks, source hash."""
    from pokezero.live_protocol_actor import MASK_NAMES,_vocab_sha
    return (('checkpoint_sha256',catalog.checkpoint_sha256),('category_vocab_sha256',_vocab_sha(catalog.vocabulary)),
        ('category_vocab_size',int(catalog.vocabulary.size)),('feature_masks',tuple((k,getattr(catalog.masks,k))for k in MASK_NAMES)),
        ('source_hash',catalog.source.source_hash))


def binding(source,metadata,catalog):
    """The WorldBinding of a bound public Source, the engine metadata and the encoder catalog. Pricer side: the run's D3
    Tables.source, its engine metadata (Prepared.metadata_json) and catalog; constructor side: constructor_binding."""
    if type(source)is not plans.Source:raise ValueError('original bound device public source required')
    return dn.WorldBinding(source.source_hash,source.source_rows_sha256,source.binding_sha256,metadata_sha256(metadata),encoder_identity(catalog))


def constructor_binding(constructor):
    """The binding a ProtocolConstructor builds its worlds under: its own source, engine metadata and B2 catalog."""
    return binding(constructor.source,constructor.metadata,constructor.catalog)


def _check_binding(built,expected,what):
    if type(expected)is not dn.WorldBinding or type(built)is not dn.WorldBinding or built!=expected:
        differ=[f for f in dn.WorldBinding.__dataclass_fields__ if getattr(built,f,None)!=getattr(expected,f,None)]
        raise ValueError(what+': '+', '.join(differ))


# Prepare ----------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Prepared:
    """One root's ProtocolConstructor.prepare output, its synced wall seconds and the binding of the constructor that made it."""
    request_id:object
    detached:object
    made:object
    legal:tuple
    seconds:float
    binding:object


@dataclass(frozen=True)
class RootRefusal:
    """A root prepare() refused (no worlds): the refusal's kind and detail, and the constructor's binding."""
    request_id:object
    kind:str
    detail:str
    seconds:float
    binding:object


def prepare_roots(constructor,protocol_requests,*,start_index=0,device='cpu'):
    """prepare() per protocol request (index start_index + i) under `device` (default the CPU; None keeps the process
    default), each world's arrays synced before its clock stops. Returns one Prepared or RootRefusal per request, each
    stamped with constructor_binding(constructor) (build refuses a stamp other than its constructor's)."""
    stamp=constructor_binding(constructor)
    if device is None:context=nullcontext()
    else:
        try:context=jax.default_device(jax.devices(device)[0])
        except RuntimeError as error:raise ValueError(f'device {device!r} unavailable ({error}); pass device=None for the process default')from None
    out=[]
    with context:
        for i,request in enumerate(protocol_requests):
            rid=request.get('request_id')if isinstance(request,dict)else None;clock=time.perf_counter()
            try:result=constructor.prepare(request,start_index+i)
            except ROOT_ERRORS as error:
                out.append(RootRefusal(rid,str(getattr(error,'kind',type(error).__name__)),str(getattr(error,'detail',error)),time.perf_counter()-clock,stamp))
                continue
            # Outside the refusal net: a changed prepare() interface raises here instead of refusing every root.
            if type(result)is not tuple or len(result)!=3:
                raise TypeError(f'ProtocolConstructor.prepare must return (Root, WorldSet, legal mask), got {type(result).__name__}')
            detached,made,legal=result
            jax.block_until_ready([(w.state,w.public)for w in made.worlds if w.state is not None])
            out.append(Prepared(rid,detached,made,tuple(map(bool,legal)),time.perf_counter()-clock,stamp))
    return tuple(out)


# Build ------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Batch:
    """DepthN input for prepared roots: `requests` (one per block of two worlds, root-major), `worlds` (PrebuiltWorlds,
    pairs aligned with requests), `roots` (one dict per prepared root: status PRICED with its request `rows`, or REFUSED
    with a reason), `count` W and `blocks` K = W/2."""
    requests:tuple
    worlds:object
    roots:tuple
    count:int
    blocks:int


_ROOT_FORWARDS={}
ROOT_FORWARD_CACHE=8


def _root_forward(forward):
    """One jitted root forward per evaluator object, the ROOT_FORWARD_CACHE most recent kept (each entry holds its
    evaluator, so a live entry's id is never reused)."""
    held=_ROOT_FORWARDS.get(id(forward))
    if held is None or held[0]is not forward:
        while len(_ROOT_FORWARDS)>=ROOT_FORWARD_CACHE:_ROOT_FORWARDS.pop(next(iter(_ROOT_FORWARDS)))
        held=_ROOT_FORWARDS[id(forward)]=(forward,jax.jit(lambda p,x:forward(p,x)))
    return held[1]


def _v4_prefix(inputs):
    """The 23-token v4 prefix of V2 encoder inputs (champion_evaluator's slice), for an evaluator without its own view."""
    t=evaluators.TOKENS
    return dict(numeric_features=inputs['numeric_features'][...,:t,:],categorical_ids=inputs['categorical_ids'][...,:t,:],
        token_type_ids=inputs['token_type_ids'][...,:t],attention_mask=inputs['attention_mask'][...,:t])


def b2_view_check(forward,root_inputs,detached):
    """Diagnostic, never a gate: the evaluator's view of root_inputs against B2's own encoding of the literal root
    (detached.view['_b2_root_inputs']). None when the root carries no B2 encoding; else dict(equal, keys: per B2 token key
    equal, numeric_max_abs_diff), or dict(equal=None, error=...) when the two cannot be compared."""
    view=getattr(detached,'view',None);b2=view.get('_b2_root_inputs')if hasattr(view,'get')else None
    if b2 is None:return None
    try:
        seen=evaluators.digest_view(forward)
        mine=seen(root_inputs)if seen is not None else _v4_prefix(root_inputs)
        keys={}
        for k in B2_TOKEN_KEYS:
            a=np.asarray(mine[k]);b=np.asarray(b2[k])
            keys[k]=bool(a.size==b.size and np.array_equal(a.reshape(b.shape),b))
        a=np.asarray(mine['numeric_features'],np.float64);b=np.asarray(b2['numeric_features'],np.float64)
        diff=float(np.max(np.abs(a.reshape(b.shape)-b)))if a.size==b.size and b.size else None
        return dict(equal=all(keys.values()),keys=keys,numeric_max_abs_diff=diff)
    except(KeyError,ValueError,TypeError,IndexError)as error:
        return dict(equal=None,error=f'{type(error).__name__}: {error}')


def build(constructor,prepared,pricer,params,*,binding,recipe,learner_actor=0,opponent_actor=0,only_complete=False):
    """Requests and pre-built worlds for `prepared` (prepare_roots' output for this constructor), for `pricer` (its executor
    gives the root view, its evaluator the root logits) under the pricer-side `binding`. ValueError on a binding mismatch
    (the pricer's, the constructor's, or a prepared root stamped by another constructor), a champion evaluator not bound
    to the constructor's catalog, a malformed world set or a bad actor index; per-root refusals are recorded, never
    raised (module docstring)."""
    built=constructor_binding(constructor);_check_binding(built,binding,'constructor worlds are built under another binding than the pricer\'s')
    for item in prepared:
        _check_binding(getattr(item,'binding',None),built,'a prepared root comes from a constructor with another binding than this one\'s')
    if isinstance(pricer.forward,evaluators.ChampionEvaluator):evaluators.check_binding(pricer.forward,constructor.catalog)
    if type(recipe)is not original.Recipe:raise ValueError('a random recipe (lane order) required')
    if any(type(x)is not int or not 0<=x<len(params)for x in(learner_actor,opponent_actor)):raise ValueError('verified actor index required')
    W=constructor.count
    if type(W)is not int or W<2 or W%2:raise ValueError('an even declared world budget required')
    K=W//2;views=depth2._make_views(pricer.executor);forward=_root_forward(pricer.forward)
    requests=[];pairs=[];roots=[]
    for index,item in enumerate(prepared):
        base=dict(index=index,request_id=item.request_id,prepare_seconds=item.seconds,worlds=W,rows=())
        if type(item)is RootRefusal:
            roots.append(dict(base,status='REFUSED',reason='constructor-root:'+item.kind+(':'+item.detail if item.detail and item.detail!=item.kind else ''),
                constructed=0,refused_worlds=None,refused_world_share=None));continue
        if type(item)is not Prepared:raise ValueError('prepare_roots output required')
        made,detached=item.made,item.detached
        if (made.requested_worlds!=W or len(made.worlds)!=W or len(made.seeds)!=W or made.root is not detached
                or tuple(w.index for w in made.worlds)!=tuple(range(W))):
            raise ValueError('a world set of the declared budget, worlds 0..W-1, for its own root required')
        if detached.source_binding_sha256!=binding.source_binding_sha256:raise ValueError('root source binding differs from the pricer\'s')
        # A world without a state always names its refusal, so it stays this root's refusal (the pricer refuses the frame otherwise).
        worlds=tuple(w if w.state is not None or w.refusals else replace(w,refusals=(UNNAMED,))for w in made.worlds)
        refused=tuple(w.index for w in worlds if w.state is None);kinds=';'.join(sorted({r for w in worlds for r in w.refusals}))
        base.update(address=(detached.address.stream,detached.address.frame,detached.address.owner),constructed=W-len(refused),refused_worlds=refused,
            refused_world_share=len(refused)/W)
        first=next((w for w in worlds if w.state is not None),None)
        if first is None or only_complete and refused:
            roots.append(dict(base,status='REFUSED',reason='WORLD_CONSTRUCTOR_REFUSED:'+kinds));continue
        state,public=jax.device_get((first.state,first.public))
        own,_,inputs,_=views(state,public,detached.address.owner)
        if int(np.asarray(own.error).reshape(-1)[0])!=0:
            roots.append(dict(base,status='REFUSED',reason='ROOT_VIEW_ENCODER_ERROR'));continue
        root_inputs={k:np.asarray(v)for k,v in inputs.items()};legal=item.legal
        base['b2_view']=b2_view_check(pricer.forward,root_inputs,detached)
        if len(legal)!=9 or not any(legal):
            roots.append(dict(base,status='REFUSED',reason='ROOT_LEGAL_EMPTY'));continue
        output=jax.device_get(forward(params[learner_actor],root_inputs))
        logits=tuple(map(float,np.asarray(output.policy_logits,np.float32).reshape(-1)));value=float(np.asarray(output.value).reshape(-1)[0])
        if not(bool(np.asarray(output.valid).reshape(-1)[0])and all(np.isfinite(logits))and np.isfinite(value)and abs(value)<=1):
            roots.append(dict(base,status='REFUSED',reason='ROOT_FORWARD_REFUSED'));continue
        rows=[]
        for b in range(K):
            pair=worlds[2*b:2*b+2]
            starts=tuple(plans.Start(w.index,made.seeds[w.index],None,(),tuple(w.refusals))for w in pair)
            # No D3 source-fixture draw exists for a constructor world: its unknown source-fixture mass is not defined (NaN).
            p=plans.Plans(detached,True,starts,constructor.source.source_hash,constructor.source.source_rows_sha256,float('nan'))
            rows.append(len(requests));requests.append(batch.Request(p,logits,legal,root_inputs,learner_actor,opponent_actor));pairs.append(pair)
        roots.append(dict(base,status='PRICED',reason=None,rows=tuple(rows),root_logits=logits,root_legal=legal,root_value=value,root_view_world=first.index))
    return Batch(tuple(requests),dn.PrebuiltWorlds(SOURCE,built,tuple(pairs)),tuple(roots),W,K)


# Price and reduce -------------------------------------------------------------------------------------------------------------------
def price(pricer,batch,binding,recipe,params,*,pass_pricer=None):
    """The batch's frame on `pricer` (all K blocks of every root in one call), and the full-width pass frame in `searched`
    mode (else None). `searched` needs `pass_pricer`: a DepthNPricer at depth 2, k 9, `prior`, with the pricer's executor,
    evaluator and k1 widths. Both passes take batch.worlds unchanged. A batch without requests returns (None, None)
    without calling a pricer (reduce takes it)."""
    if pricer.selection=='prior':
        if pass_pricer is not None:raise ValueError('a pass pricer belongs to the searched mode')
    elif (pass_pricer is None or pass_pricer.depth!=2 or pass_pricer.selection!='prior'or(pass_pricer.k_own,pass_pricer.k_opp)!=((9,),(9,))
            or pass_pricer.executor is not pricer.executor or pass_pricer.evaluator_suffix!=pricer.evaluator_suffix
            or(pass_pricer.k1_own,pass_pricer.k1_opp)!=(pricer.k1_own,pricer.k1_opp)):
        raise ValueError('searched needs a full-width depth-2 prior pass pricer with the same executor, evaluator and k1 widths')
    if not batch.requests:return None,None
    if pricer.selection=='prior':return pricer.price(batch.requests,binding,recipe,params,worlds=batch.worlds),None
    raw=pass_pricer.price(batch.requests,binding,recipe,params,worlds=batch.worlds)
    own,opp,_=dn.searched_maps(dn.as_depth2_frame(raw),*pricer.searched_k)
    return pricer.price(batch.requests,binding,recipe,params,selection_maps={2:(own,opp)},worlds=batch.worlds),raw


def root_actions(completed_q,policy_target,legal):
    """(action_mean_q, action_policy_target): the argmax over the legal actions of the mean completed-Q (the ladder's root
    operator, phase_d_ladder_search.reduce_q) and of the improved target, each the lower action on ties."""
    acts=[a for a in range(9)if legal[a]]
    return max(acts,key=lambda a:(completed_q[a],-a)),max(acts,key=lambda a:(policy_target[a],-a))


def reduce(frame,batch,*,rule=backups.learner_max):
    """Per prepared root: status LABELED with the K-block composition (completed_q, policy_target, root_value and the
    moves `action_mean_q` and `action_policy_target`, module docstring), or REFUSED with a cause (the build reason, or the
    first refused block's depthn_backups.refusal_summary cause, 'materialization' for a constructor world refusal, else
    'reduction'; the target range check's ValueError (RANGE_EDGE) refuses that root, its text in `reduction_error`, never
    raised; any other ValueError raises).
    `block_causes` lists each block's cause. `frame` is None for a batch without requests."""
    from .d3_search_latency import compose,reduce_block
    if frame is None and batch.requests:raise ValueError('a frame is required for a batch with requests')
    causes=backups.refusal_summary(frame)['roots']if batch.requests else()
    results=[]
    for root in batch.roots:
        out=dict(index=root['index'],request_id=root['request_id'],status='REFUSED',cause=root['reason'],worlds=batch.count,blocks=batch.blocks,
            block_causes=None,completed_q=None,policy_target=None,root_value=None,action_mean_q=None,action_policy_target=None,reduction_error=None)
        if root['status']!='PRICED':results.append(out);continue
        logits,legal=root['root_logits'],root['root_legal'];out['block_causes']=tuple(causes[row]['cause']for row in root['rows'])
        prices=[backups.backup(frame,row,rule)for row in root['rows']]
        try:composed=compose([reduce_block(pair,logits,legal,original)for pair in prices],logits,legal,original)
        except ValueError as error:   # the rounding edge of the target range check refuses this root only; anything else is a bug
            if str(error)!=RANGE_EDGE:raise
            out.update(cause=next((c for c in out['block_causes']if c),'reduction'),reduction_error=f'{type(error).__name__}: {error}')
            results.append(out);continue
        out.update(status=composed['status'],completed_q=composed['completed_q'],policy_target=composed['policy_target'],root_value=composed['root_value'])
        if composed['status']=='LABELED':
            mean_q,target=root_actions(composed['completed_q'],composed['policy_target'],legal)
            out.update(cause=None,action_mean_q=mean_q,action_policy_target=target)
        else:out['cause']=next((c for c in out['block_causes']if c),'reduction')
        results.append(out)
    return tuple(results)
