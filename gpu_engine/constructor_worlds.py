"""Codex's Phase D protocol constructor worlds as DepthNPricer input (gap 4, part A): depth-N prices on Showdown roots.

Design. ProtocolConstructor.prepare(request, index) folds a Showdown protocol prefix plus the literal own request (B2)
and returns (detached device_public_worlds.Root, phase_d_live_worlds.WorldSet of W in {2, 8, 16, 32} worlds, B2's own
legal mask). Its worlds are D3's world type (device_public_world_state.World: the engine BattleState plus the public
tracker carry), so DepthN's executor steps them unchanged; only the carry's pytree differs from D3's (from_b2 adds four
dict keys and leaves two members None). This module turns prepared roots into DepthN requests and pre-built worlds:

- prepare_roots: prepare() per protocol request, on the CPU device by default (its eager per-world array updates would
  otherwise dispatch one by one on a GPU), timed and synced per root. A raised refusal (phase_d_live_worlds.Refusal, or
  the errors ProtocolConstructor.construct treats as refusals) is a RootRefusal: the root gets no request.
- build: W worlds become K = W/2 blocks of two, one Request per block, packed into ONE frame (row = the root's rows[b],
  root-major), since the worlds are pre-built (no per-block recipe or re-materialization). Plans are synthesized:
  selected=True (the quarter selection is bypassed), Start(world index, the constructor's world seed, no fixtures,
  refusals). `root_inputs` are world 0's own view under the pricer's executor (the v5 encoder): every world of every
  block is then held to it by the turn-1 input check (pre 3+), the information-set check D3 relies on too. `root_legal`
  is B2's own mask (pre 2 holds each world's own encoder mask to it, as the ladder's legality check). `root_logits` are
  the pricer's own evaluator (forward=, e.g. the champion) on root_inputs, as gate.bind_roots. Actors default to 0/0
  (one evaluator bank, the champion's (champion,)).
- Refusals: a world without a state (a constructor world refusal) is pre -1 in its block
  (`physical-materialization:<refusals>` in the world report), so that block refuses and the root refuses: DepthN's
  rule (depthn_backups._root), the constructor's all-W `fully_constructed` rule. `only_complete=True` refuses such
  roots before pricing instead (saves the device work; the frame then never shows them). A root whose world 0 is
  refused takes its root view from the lowest constructed world (it refuses anyway); one with no constructed world, an
  erroring root view or an unusable root forward is refused before pricing with a named reason.
- Bindings: `binding(source, metadata, catalog)` is a DepthN WorldBinding (source hash, ordered rows, the
  source/engine-namespace binding sha256, the engine metadata's canonical-JSON sha256, the encoder catalog's checkpoint,
  category vocabulary and masks). build requires the constructor's binding to equal the pricer side's (the run's D3
  Tables source, engine metadata and catalog) and the pricer checks it again on every call: fail closed.
- Programs: the worlds go in as DepthN's PrebuiltWorlds under the tag SOURCE, so every compiled-program key gains
  ('worlds', SOURCE, carry signature) after the evaluator's identity: constructor and D3 worlds never share a program,
  nor do two carry structures, nor two evaluators.
- price: `prior` = one packed pricing. `searched` = the full-width depth-2 pass (a DepthNPricer at depth 2, k 9, prior,
  same executor, evaluator and k1) then searched_maps on the host, then depth N with the maps, both passes on the SAME
  PrebuiltWorlds (prepare refuses a second attempt at an address, and none is needed). Nothing in it is D3-only.
- reduce: per root, depthn_backups.backup(rule) and reduce_completed_q per block, then the K blocks by the equal-weight
  mean of their completed-Q (d3_search_latency.compose: every world weighted 1/W); any refused block refuses the root.

Chance: DepthN samples each node's own keys; constructor states carry kernel_seed-derived keys, so prices are
reproducible but not bit-comparable to the ladder's R1/R7, which fold a domain seed into the keys.

Constructor API this module depends on (part B, the batch 2 and tail sync, must keep it): ProtocolConstructor.prepare's
(Root, WorldSet, legal) return and its `count`, `source`, `metadata`, `catalog` attributes; Refusal's `kind`/`detail`;
WorldSet's `root`, `worlds`, `seeds`, `requested_worlds`; World's fields; Root's `address` and `source_binding_sha256`;
Source's `source_hash`, `source_rows_sha256`, `binding_sha256`. A changed carry structure needs no change here (new
programs compile under its own signature); a changed engine metadata or source binding fails closed until both sides agree.
"""
from contextlib import nullcontext
from dataclasses import dataclass,replace
import hashlib
import json
import time

import jax
import numpy as np

from pokezero import v5_gumbel as original
from . import depthn_backups as backups,device_depth2_pricer as depth2,device_depthn_pricer as dn
from . import device_public_teacher_batch as batch,device_public_worlds as plans

SOURCE='phase-d-protocol-constructor'
# What ProtocolConstructor.construct treats as a root refusal (phase_d_live_worlds.Refusal is a ValueError).
ROOT_ERRORS=(ValueError,KeyError,TypeError,IndexError,AttributeError)
UNNAMED='UNNAMED_CONSTRUCTOR_WORLD_REFUSAL'


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


def _check_binding(built,expected):
    if type(expected)is not dn.WorldBinding or built!=expected:
        differ=[f for f in dn.WorldBinding.__dataclass_fields__ if getattr(built,f)!=getattr(expected,f,None)]
        raise ValueError('constructor worlds are built under another binding than the pricer\'s: '+', '.join(differ))


# Prepare ----------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Prepared:
    """One root's ProtocolConstructor.prepare output and its synced wall seconds."""
    request_id:object
    detached:object
    made:object
    legal:tuple
    seconds:float


@dataclass(frozen=True)
class RootRefusal:
    """A root prepare() refused (no worlds): the refusal's kind and detail."""
    request_id:object
    kind:str
    detail:str
    seconds:float


def prepare_roots(constructor,protocol_requests,*,start_index=0,device='cpu'):
    """prepare() per protocol request (index start_index + i) under `device` (default the CPU; None keeps the process
    default), each world's arrays synced before its clock stops. Returns one Prepared or RootRefusal per request."""
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
                out.append(RootRefusal(rid,str(getattr(error,'kind',type(error).__name__)),str(getattr(error,'detail',error)),time.perf_counter()-clock))
                continue
            # Outside the refusal net: a changed prepare() interface raises here instead of refusing every root.
            if type(result)is not tuple or len(result)!=3:
                raise TypeError(f'ProtocolConstructor.prepare must return (Root, WorldSet, legal mask), got {type(result).__name__}')
            detached,made,legal=result
            jax.block_until_ready([(w.state,w.public)for w in made.worlds if w.state is not None])
            out.append(Prepared(rid,detached,made,tuple(map(bool,legal)),time.perf_counter()-clock))
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


def _root_forward(forward):
    """One jitted root forward per evaluator object (kept with it, so its id is never reused)."""
    held=_ROOT_FORWARDS.get(id(forward))
    if held is None or held[0]is not forward:
        held=_ROOT_FORWARDS[id(forward)]=(forward,jax.jit(lambda p,x:forward(p,x)))
    return held[1]


def build(constructor,prepared,pricer,params,*,binding,recipe,learner_actor=0,opponent_actor=0,only_complete=False):
    """Requests and pre-built worlds for `prepared` (prepare_roots' output for this constructor), for `pricer` (its executor
    gives the root view, its evaluator the root logits) under the pricer-side `binding`. ValueError on a binding mismatch,
    a malformed world set or a bad actor index; per-root refusals are recorded, never raised (module docstring)."""
    built=constructor_binding(constructor);_check_binding(built,binding)
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
                constructed=0,refused_worlds=None));continue
        if type(item)is not Prepared:raise ValueError('prepare_roots output required')
        made,detached=item.made,item.detached
        if (made.requested_worlds!=W or len(made.worlds)!=W or len(made.seeds)!=W or made.root is not detached
                or tuple(w.index for w in made.worlds)!=tuple(range(W))):
            raise ValueError('a world set of the declared budget, worlds 0..W-1, for its own root required')
        if detached.source_binding_sha256!=binding.source_binding_sha256:raise ValueError('root source binding differs from the pricer\'s')
        # A world without a state always names its refusal, so it stays this root's refusal (the pricer refuses the frame otherwise).
        worlds=tuple(w if w.state is not None or w.refusals else replace(w,refusals=(UNNAMED,))for w in made.worlds)
        refused=tuple(w.index for w in worlds if w.state is None);kinds=';'.join(sorted({r for w in worlds for r in w.refusals}))
        base.update(address=(detached.address.stream,detached.address.frame,detached.address.owner),constructed=W-len(refused),refused_worlds=refused)
        first=next((w for w in worlds if w.state is not None),None)
        if first is None or only_complete and refused:
            roots.append(dict(base,status='REFUSED',reason='WORLD_CONSTRUCTOR_REFUSED:'+kinds));continue
        state,public=jax.device_get((first.state,first.public))
        own,_,inputs,_=views(state,public,detached.address.owner)
        if int(np.asarray(own.error).reshape(-1)[0])!=0:
            roots.append(dict(base,status='REFUSED',reason='ROOT_VIEW_ENCODER_ERROR'));continue
        root_inputs={k:np.asarray(v)for k,v in inputs.items()};legal=item.legal
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
            p=plans.Plans(detached,True,starts,constructor.source.source_hash,constructor.source.source_rows_sha256,sum(w.state is None for w in pair)/2)
            rows.append(len(requests));requests.append(batch.Request(p,logits,legal,root_inputs,learner_actor,opponent_actor));pairs.append(pair)
        roots.append(dict(base,status='PRICED',reason=None,rows=tuple(rows),root_logits=logits,root_legal=legal,root_value=value,root_view_world=first.index))
    return Batch(tuple(requests),dn.PrebuiltWorlds(SOURCE,built,tuple(pairs)),tuple(roots),W,K)


# Price and reduce -------------------------------------------------------------------------------------------------------------------
def price(pricer,batch,binding,recipe,params,*,pass_pricer=None):
    """The batch's frame on `pricer` (all K blocks of every root in one call), and the full-width pass frame in `searched`
    mode (else None). `searched` needs `pass_pricer`: a DepthNPricer at depth 2, k 9, `prior`, with the pricer's executor,
    evaluator and k1 widths. Both passes take batch.worlds unchanged."""
    if pricer.selection=='prior':
        if pass_pricer is not None:raise ValueError('a pass pricer belongs to the searched mode')
        return pricer.price(batch.requests,binding,recipe,params,worlds=batch.worlds),None
    if (pass_pricer is None or pass_pricer.depth!=2 or pass_pricer.selection!='prior'or(pass_pricer.k_own,pass_pricer.k_opp)!=((9,),(9,))
            or pass_pricer.executor is not pricer.executor or pass_pricer.evaluator_suffix!=pricer.evaluator_suffix
            or(pass_pricer.k1_own,pass_pricer.k1_opp)!=(pricer.k1_own,pricer.k1_opp)):
        raise ValueError('searched needs a full-width depth-2 prior pass pricer with the same executor, evaluator and k1 widths')
    raw=pass_pricer.price(batch.requests,binding,recipe,params,worlds=batch.worlds)
    own,opp,_=dn.searched_maps(dn.as_depth2_frame(raw),*pricer.searched_k)
    return pricer.price(batch.requests,binding,recipe,params,selection_maps={2:(own,opp)},worlds=batch.worlds),raw


def reduce(frame,batch,*,rule=backups.learner_max):
    """Per prepared root: status LABELED with the K-block composition (completed_q, policy_target, root_value and the
    move `action`: the target's argmax, lower action on ties), or REFUSED with a cause (the build reason, or the first
    refused block's depthn_backups.refusal_summary cause, 'materialization' for a constructor world refusal, else
    'reduction'). `block_causes` lists each block's cause."""
    from .d3_search_latency import compose,reduce_block
    causes=backups.refusal_summary(frame)['roots']if batch.requests else()
    results=[]
    for root in batch.roots:
        out=dict(index=root['index'],request_id=root['request_id'],status='REFUSED',cause=root['reason'],worlds=batch.count,blocks=batch.blocks,
            block_causes=None,completed_q=None,policy_target=None,root_value=None,action=None)
        if root['status']!='PRICED':results.append(out);continue
        logits,legal=root['root_logits'],root['root_legal']
        composed=compose([reduce_block(backups.backup(frame,row,rule),logits,legal,original)for row in root['rows']],logits,legal,original)
        out.update(block_causes=tuple(causes[row]['cause']for row in root['rows']),status=composed['status'],completed_q=composed['completed_q'],
            policy_target=composed['policy_target'],root_value=composed['root_value'])
        if composed['status']=='LABELED':
            target=composed['policy_target'];out.update(cause=None,action=max(range(9),key=lambda a:(target[a],-a)))
        else:out['cause']=next((c for c in out['block_causes']if c),'reduction')
        results.append(out)
    return tuple(results)

