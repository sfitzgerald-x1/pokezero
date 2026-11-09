"""Codex's Phase D protocol constructor worlds as DepthNPricer input (gap 4, part A): depth-N prices on Showdown roots.

Design. ProtocolConstructor.prepare(request, index) folds a Showdown protocol prefix plus the literal own request (B2)
and returns (detached device_public_worlds.Root, phase_d_live_worlds.WorldSet of W in {2, 8, 16, 32} worlds, B2's own
legal mask). Its worlds are D3's world type (device_public_world_state.World: the engine BattleState plus the public
tracker carry), so DepthN's executor steps them unchanged; only the carry's pytree differs from D3's (from_b2 adds four
dict keys and leaves two members None). This module turns prepared roots into DepthN requests and pre-built worlds:

- prepare_roots: prepare() per protocol request, on the CPU device by default (its eager per-world array updates would
  otherwise dispatch one by one on a GPU), timed and synced per root, each result stamped with the constructor's
  binding. A raised refusal (phase_d_live_worlds.Refusal, or the errors ProtocolConstructor.construct treats as
  refusals) is a RootRefusal: the root gets no request. Every constructed world's public carry is completed (below)
  before its clock stops, so the live search's structure check, build, the pricer and every capture see one world set.
- Carry completion (complete_world; build applies it again, a no-op on a completed world). phase_d_live_worlds.from_b2
  builds the root's tracker carry from B2's public fold, not through the device public hooks, and leaves three things
  the device path (engine_device_public_hooks.initialize, then a hook per engine step) would hold on the same history:
  - shown_types (fill_shown_types): from_b2 writes zeros, so the encoder's TYPE_1/TYPE_2 of every revealed Pokemon of
    the side other than the observer read 0 (<pad>), at the root and at every node and leaf. The hook rule
    (engine_device_public_hooks.install: species_types[clip(species)] where 0 < species < len(species_types), else 0) is
    applied to each (side, public slot) whose row is still (0, 0); a side's active whose engine types differ from its
    base types (a type change the state carries) takes those types instead (the color_change hook's write). A non-zero
    row is never touched (Transform, Color Change). species_types is the executor's own table:
    engine_device_belief.compile_tables(metadata, catalog), which engine_device_observation.compile_tables builds the
    executor's observation (and belief) tables with, here from the constructor's metadata and catalog (the binding
    check makes them the pricer's); the tracker's species are engine ids (from_b2 remaps B2's into the same
    metadata['vocabulary']['species'] the hook's state species index).
  - negative_items (fill_negative_items): from_b2 leaves None, which turns off the Leftovers non-proc rule for the whole
    search; the device path's empty(B) is exact at a root (upkeep clears the evidence) except an active installed after
    the last upkeep (an end-of-turn faint replacement), whose install-time evidence is not recoverable here.
  - the items-removed credits (orient_credits): from_b2 stores them owner-oriented as [B] arrays, which the readout
    passes through unchanged, so the peer's view read the owner's own counts; they become per-side [B, 2] arrays
    (self = items_removed[side], opp = items_removed[1 - side], B2's project_credits), the owner's view unchanged.
  The completion is pure host numpy on the world's own leaves (no other leaf changes) and raises CarryLayoutError (a
  RuntimeError, never a per-root refusal) when a carry is not the engine_device_trackers.Trackers layout it was
  written against.
- build: W worlds become K = W/2 blocks of two, one Request per block, packed into ONE frame (row = the root's rows[b],
  root-major), since the worlds are pre-built (no per-block recipe or re-materialization). Plans are synthesized:
  selected=True (the quarter selection is bypassed), Start(world index, the constructor's world seed, no fixtures,
  refusals); source_fixture_unknown_mass is NaN (no D3 source-fixture draw exists here; the refused-world share is the
  root's `refused_world_share`). `root_legal` is B2's own mask (pre 2 holds each world's own encoder mask to it, as the
  ladder's legality check). Actors default to 0/0 (one evaluator bank, the champion's (champion,)).
- The root prior: `root_logits` and `root_value` are the pricer's own evaluator (forward=, e.g. the champion) on B2's own
  encoding of the literal root, as in Codex's ladder (phase_d_ladder_search.process_many forwards
  detached.view['_b2_root_inputs']): root_inputs with the first TOKENS (23) tokens of each token tensor replaced by B2's
  (splice; history_mask and the later tokens unchanged), through the same jitted root forward; `root_view_source` 'b2'. A
  root without B2 inputs takes world 0's own view (root_inputs; 'world0'). B2 inputs that do not fit (a missing key, a
  shape, a lossy cast) refuse the root, ROOT_B2_INPUTS_UNUSABLE.
- root_inputs (every Request's) stay world 0's own view under the pricer's executor (the v5 encoder), after the carry
  completion. The turn-1 input check (pre 3+) then holds every OTHER world of every block to world 0's view: a
  consistency check across the sampled worlds only. As a diagnostic (reported, never a gate), each root's `b2_view`
  compares the evaluator's view of root_inputs (its `view`; the 23-token v4 prefix for an evaluator without one) with
  B2's encoding, where the root carries it (from_b2 stores it): per key equal, and the largest numeric difference.
- Refusals: a world without a state (a constructor world refusal) is pre -1 in its block
  (`physical-materialization:<refusals>` in the world report), so that block refuses and the root refuses: DepthN's
  rule (depthn_backups._root), the constructor's all-W `fully_constructed` rule. `only_complete=True` refuses such
  roots before pricing instead (saves the device work; the frame then never shows them). A root whose world 0 is
  refused takes its root view from the lowest constructed world (it refuses anyway); one with no constructed world, an
  erroring root view, unusable B2 inputs or an unusable root forward is refused before pricing with a named reason.
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
`view['_b2_root_inputs']` (the root prior's inputs); Source's `source_hash`, `source_rows_sha256`, `binding_sha256`; the
constructor's `metadata` and `catalog` (the completion's tables). A changed carry structure compiles new programs under
its own signature, but the completion fails loudly (CarryLayoutError) until it is reviewed against it; a changed engine
metadata or source binding fails closed until both sides agree.
"""
from contextlib import nullcontext
from dataclasses import dataclass,replace
import hashlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import v5_gumbel as original
from . import champion_evaluator as evaluators,depthn_backups as backups,device_depth2_pricer as depth2,device_depthn_pricer as dn
from . import device_public_teacher_batch as batch,device_public_worlds as plans
from . import engine_device_negative_items as negative_items,engine_device_trackers as public_trackers

SOURCE='phase-d-protocol-constructor'
# What ProtocolConstructor.construct treats as a root refusal (phase_d_live_worlds.Refusal is a ValueError).
ROOT_ERRORS=(ValueError,KeyError,TypeError,IndexError,AttributeError)
UNNAMED='UNNAMED_CONSTRUCTOR_WORLD_REFUSAL'
RANGE_EDGE='finite in-range numerical value required'   # v5_contract._real's error: on reduce's path only RootImprovement.validate raises it
B2_TOKEN_KEYS=('numeric_features','categorical_ids','token_type_ids','attention_mask')   # B2's encode(): [23,F] / [23]
TOKENS=evaluators.TOKENS   # the champion's v4 prefix of the V2 encoding: what splice replaces with B2's tokens
TOKEN_KEYS={'numeric_features':2,'categorical_ids':2,'token_type_ids':1,'attention_mask':1}   # trailing dims from the token axis
CREDIT_KEYS=('self_items_removed_credit','opp_items_removed_credit')   # from_b2's extra field_features keys


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


# Carry completion -------------------------------------------------------------------------------------------------------------------
class CarryLayoutError(RuntimeError):
    """A constructor world's carry is not the layout complete_world was written against (module docstring): never a refusal."""


@dataclass(frozen=True)
class CarryTables:
    """What the completion reads: species_types [S, 2] int32, engine species id -> engine type ids (the public hooks'
    belief_tables.species_types, engine_device_belief.Tables), held as host numpy."""
    species_types:object


def _table(species_types,species_count=None,type_count=None):
    st=np.asarray(species_types)
    if st.ndim!=2 or st.shape[1]!=2 or not st.shape[0] or st.dtype!=np.int32:
        raise CarryLayoutError(f'species_types: an [S, 2] int32 table required, got {st.shape} {st.dtype}')
    if species_count is not None and st.shape[0]!=species_count:
        raise CarryLayoutError(f'species_types: {st.shape[0]} rows for {species_count} engine species')
    if np.any(st<0)or type_count is not None and np.any(st>=type_count):raise CarryLayoutError('species_types: a type id outside the engine namespace')
    st=st.copy();st.setflags(write=False)
    return CarryTables(st)


def carry_tables(metadata,catalog):
    """CarryTables from the engine metadata and encoder catalog through engine_device_belief.compile_tables, the function and
    inputs the executor's observation tables are built with (engine_device_observation.compile_tables(metadata, catalog)
    .belief, whose species_types the public hooks read); compiled on the CPU, only species_types kept."""
    from . import engine_device_belief
    try:context=jax.default_device(jax.devices('cpu')[0])
    except RuntimeError:context=nullcontext()   # no CPU backend in this process: the default device
    with context:belief=engine_device_belief.compile_tables(metadata,catalog)
    v=metadata['vocabulary']
    return _table(jax.device_get(belief.species_types),len(v['species']),len(v['types']))


_CARRY_TABLES={}


def constructor_carry_tables(constructor):
    """carry_tables of the constructor's own metadata and catalog, once per constructor object (the ROOT_FORWARD_CACHE most
    recent kept; each entry holds its constructor, so a live entry's id is never reused)."""
    held=_CARRY_TABLES.get(id(constructor))
    if held is None or held[0]is not constructor:
        while len(_CARRY_TABLES)>=ROOT_FORWARD_CACHE:_CARRY_TABLES.pop(next(iter(_CARRY_TABLES)))
        held=_CARRY_TABLES[id(constructor)]=(constructor,carry_tables(constructor.metadata,constructor.catalog))
    return held[1]


def world_completer(constructor):
    """The completion prepare_roots and build apply to each world of this constructor: (world, owner) -> world."""
    tables=constructor_carry_tables(constructor)
    return lambda world,owner:complete_world(world,tables,owner)


def _host(x):return np.asarray(jax.device_get(x))


def _leaf(x,shape,dtype,name):
    if tuple(np.shape(x))!=shape or np.dtype(getattr(x,'dtype',None))!=np.dtype(dtype):
        raise CarryLayoutError(f'{name}: {shape} {np.dtype(dtype)} required, got {tuple(np.shape(x))} {getattr(x,"dtype",None)}')


def _trackers(public,what='public carry'):
    if type(public)is not public_trackers.Trackers:
        raise CarryLayoutError(f'{what}: an engine_device_trackers.Trackers required, got {type(public).__name__}')
    b=tuple(np.shape(public.error))
    if len(b)!=1 or not b[0]:raise CarryLayoutError(f'{what}: error [B] required, got {b}')
    return b[0]


def fill_shown_types(state,public,tables):
    """The public carry with only shown_types filled (module docstring): per (side, public slot) whose row is (0, 0), the
    hook's species_types[species] for a known species (0 < species < S), or, for a side's active whose engine types differ
    from its base types, those types (the color_change hook's write); any other row unchanged. Host numpy; the same
    carry object when nothing changes. CarryLayoutError on another layout."""
    b=_trackers(public);st=tables.species_types if type(tables)is CarryTables else None
    if st is None:raise CarryLayoutError('CarryTables required')
    for name,value,shape in(('species',public.species,(b,2,6)),('physical_to_public',public.physical_to_public,(b,2,6)),
            ('shown_types',public.shown_types,(b,2,6,2)),('state.active',state.active,(b,2)),('state.types',state.types,(b,2,6,2)),
            ('state.base_types',state.base_types,(b,2,6,2))):_leaf(value,shape,np.int32,name)
    species,shown,mapping,active,types,base=(_host(x)for x in(public.species,public.shown_types,public.physical_to_public,state.active,state.types,state.base_types))
    n=st.shape[0];known=(species>0)&(species<n);values=st[np.clip(species,0,n-1)].copy();retyped=np.zeros(species.shape,bool)
    rows=np.arange(b)
    for side in(0,1):
        physical=active[:,side];p=np.clip(physical,0,5);slot=mapping[rows,side,p];at=np.clip(slot,0,5)
        current,original_types=types[rows,side,p],base[rows,side,p]
        hit=(physical>=0)&(physical<6)&(slot>=0)&(slot<6)&np.any(current!=original_types,axis=-1)
        values[rows[hit],side,at[hit]]=current[hit];retyped[rows[hit],side,at[hit]]=True
    fill=(known|retyped)&np.all(shown==0,axis=-1)
    if not fill.any():return public
    filled=np.where(fill[...,None],values,shown).astype(np.int32)
    return public if np.array_equal(filled,shown)else public._replace(shown_types=jnp.asarray(filled))


def fill_negative_items(public):
    """The public carry with negative_items = engine_device_negative_items.empty(B) where from_b2 left None (the device
    path's initial value; exact at a root, module docstring); an existing one is checked and kept."""
    b=_trackers(public);carry=public.negative_items
    if carry is None:return public._replace(negative_items=negative_items.empty(b))
    if type(carry)is not negative_items.State:raise CarryLayoutError(f'negative_items: an engine_device_negative_items.State required, got {type(carry).__name__}')
    _leaf(carry.percent,(b,2,6),np.int32,'negative_items.percent');_leaf(carry.known,(b,2,6),np.bool_,'negative_items.known')
    return public


def orient_credits(public,owner):
    """The public carry with from_b2's owner-oriented [B] items-removed credits as per-side [B, 2] arrays (module docstring):
    items_removed[:, owner] = self, [:, 1 - owner] = opp; self = items_removed, opp = items_removed[:, ::-1]. A carry
    without them, or with [B, 2] ones already, is unchanged."""
    b=_trackers(public);fields=public.field_features
    if owner not in(0,1):raise CarryLayoutError(f'owner 0 or 1 required, got {owner!r}')
    if fields is None:return public
    if not isinstance(fields,dict):raise CarryLayoutError(f'field_features: a dict required, got {type(fields).__name__}')
    present=[k in fields for k in CREDIT_KEYS]
    if not any(present):return public
    if not all(present):raise CarryLayoutError('field_features: both items-removed credits or neither required')
    values=[fields[k]for k in CREDIT_KEYS];shapes={tuple(np.shape(x))for x in values};kinds={np.dtype(getattr(x,'dtype',None)).kind for x in values}
    if len(shapes)!=1 or kinds-{'i','u'}:raise CarryLayoutError(f'field_features: integer credits of one shape required, got {shapes} {kinds}')
    shape=shapes.pop()
    if shape==(b,2):return public
    if shape!=(b,):raise CarryLayoutError(f'field_features: [B] or [B, 2] credits required, got {shape}')
    own,opp=(_host(x)for x in values);removed=np.zeros((b,2),own.dtype);removed[:,owner]=own;removed[:,1-owner]=opp
    return public._replace(field_features={**fields,CREDIT_KEYS[0]:jnp.asarray(removed),CREDIT_KEYS[1]:jnp.asarray(removed[:,::-1].copy())})


def complete_world(world,tables,owner):
    """A constructor world with its public carry completed (fill_shown_types, fill_negative_items, orient_credits; module
    docstring), `owner` the root's owner; a refused world (no state) and an already completed one are returned as they are.
    Pure: the world's other leaves are the same objects. CarryLayoutError on another carry layout."""
    if world.state is None:return world
    public=world.public
    _trackers(public,f'world {world.index} public carry')
    done=orient_credits(fill_negative_items(fill_shown_types(world.state,public,tables)),owner)
    return world if done is public else replace(world,public=done)


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


def _completed(made,complete,owner):
    """The world set with every constructed world completed; the same object when no world changes."""
    worlds=tuple(w if w.state is None else complete(w,owner)for w in made.worlds)
    return made if all(a is b for a,b in zip(worlds,made.worlds))else replace(made,worlds=worlds)


def prepare_roots(constructor,protocol_requests,*,start_index=0,device='cpu'):
    """prepare() per protocol request (index start_index + i) under `device` (default the CPU; None keeps the process
    default), each world's carry completed (world_completer, module docstring) and its arrays synced before its clock
    stops. Returns one Prepared or RootRefusal per request, each stamped with constructor_binding(constructor) (build
    refuses a stamp other than its constructor's). CarryLayoutError (the completion) is never a refusal."""
    stamp=constructor_binding(constructor)
    if device is None:context=nullcontext()
    else:
        try:context=jax.default_device(jax.devices(device)[0])
        except RuntimeError as error:raise ValueError(f'device {device!r} unavailable ({error}); pass device=None for the process default')from None
    out=[];complete=None
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
            if complete is None:complete=world_completer(constructor)
            made=_completed(made,complete,detached.address.owner)
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


def b2_root_inputs(detached):
    """B2's own encoding of the literal root, detached.view['_b2_root_inputs'], or None when the root carries none."""
    view=getattr(detached,'view',None)
    return view.get('_b2_root_inputs')if hasattr(view,'get')else None


def _bits(a):
    a=np.ascontiguousarray(np.asarray(a))
    return a.view({1:np.uint8,2:np.uint16,4:np.uint32,8:np.uint64}[a.itemsize])if a.dtype.kind=='f'else a


def _prefix_index(key,tokens=TOKENS):
    return(Ellipsis,slice(0,tokens))+((slice(None),)if TOKEN_KEYS[key]==2 else())


def splice(root_inputs,b2,tokens=TOKENS):
    """(X, lossy keys): copies of the V2 root inputs with the first `tokens` tokens of each token tensor (TOKEN_KEYS) replaced
    by B2's (cast to the root's dtype); history_mask and tokens >= `tokens` unchanged. A key whose B2 values do not survive
    the cast is listed. ValueError when the root lacks a token key or B2's tensor does not fit the root's prefix; KeyError
    when B2 lacks a token key."""
    missing=[k for k in TOKEN_KEYS if k not in root_inputs]
    if missing:raise ValueError('root inputs without '+', '.join(missing))
    out,lossy={},[]
    for key,value in root_inputs.items():
        a=np.array(value,copy=True)
        if key in TOKEN_KEYS:
            source=np.asarray(b2[key]);index=_prefix_index(key,tokens);target=a[index]
            if source.shape[0]!=tokens or source.shape!=target.shape[target.ndim-source.ndim:]:
                raise ValueError(f'{key}: B2 {source.shape} does not fit the root prefix {target.shape}')
            cast=source.astype(a.dtype);back=cast.astype(source.dtype)
            if not(back.shape==source.shape and np.array_equal(_bits(back),_bits(source))):lossy.append(key)
            a[index]=cast
        out[key]=a
    return out,lossy


def b2_view_check(forward,root_inputs,detached):
    """Diagnostic, never a gate: the evaluator's view of root_inputs against B2's own encoding of the literal root
    (detached.view['_b2_root_inputs']). None when the root carries no B2 encoding; else dict(equal, keys: per B2 token key
    equal, numeric_max_abs_diff), or dict(equal=None, error=...) when the two cannot be compared."""
    b2=b2_root_inputs(detached)
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
    gives the root view, its evaluator the root prior on B2's encoding, else on world 0's view) under the pricer-side
    `binding`. Every world's carry is completed first (a no-op on prepare_roots' worlds). ValueError on a binding mismatch
    (the pricer's, the constructor's, or a prepared root stamped by another constructor), a champion evaluator not bound
    to the constructor's catalog, a malformed world set or a bad actor index; CarryLayoutError from the completion;
    per-root refusals are recorded, never raised (module docstring). A priced root records root_view_source ('b2' or
    'world0') and b2_view."""
    built=constructor_binding(constructor);_check_binding(built,binding,'constructor worlds are built under another binding than the pricer\'s')
    for item in prepared:
        _check_binding(getattr(item,'binding',None),built,'a prepared root comes from a constructor with another binding than this one\'s')
    if isinstance(pricer.forward,evaluators.ChampionEvaluator):evaluators.check_binding(pricer.forward,constructor.catalog)
    if type(recipe)is not original.Recipe:raise ValueError('a random recipe (lane order) required')
    if any(type(x)is not int or not 0<=x<len(params)for x in(learner_actor,opponent_actor)):raise ValueError('verified actor index required')
    W=constructor.count
    if type(W)is not int or W<2 or W%2:raise ValueError('an even declared world budget required')
    K=W//2;views=depth2._make_views(pricer.executor);forward=_root_forward(pricer.forward)
    requests=[];pairs=[];roots=[];complete=None
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
        if complete is None:complete=world_completer(constructor)
        worlds=tuple(complete(w,detached.address.owner)if w.state is not None else w if w.refusals else replace(w,refusals=(UNNAMED,))
            for w in made.worlds)
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
        # The root prior reads B2's encoding of the literal root (Codex's ladder), else world 0's view (module docstring).
        b2=b2_root_inputs(detached);prior_inputs=root_inputs;base['root_view_source']='world0'
        if b2 is not None:
            base['root_view_source']='b2'
            try:prior_inputs,lossy=splice(root_inputs,b2)
            except(KeyError,ValueError,TypeError,IndexError)as error:
                roots.append(dict(base,status='REFUSED',reason=f'ROOT_B2_INPUTS_UNUSABLE: {type(error).__name__}: {error}'));continue
            if lossy:
                roots.append(dict(base,status='REFUSED',reason='ROOT_B2_INPUTS_UNUSABLE: lossy cast of '+','.join(lossy)));continue
        output=jax.device_get(forward(params[learner_actor],prior_inputs))
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
