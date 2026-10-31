"""Bounded ORIGINAL hypothetical kernel/FULL/public29 grid execution.

No models, ordinary Session/birth claims or caller execution callbacks. Host
preparation is once per hypothesis, routing is batched, and each device chunk
has ONE original kernel call plus the two genuine owning FULL drains. Every
declared cell survives, including failures. Padding never becomes a cell or a
model input. Pure compiled holders retain statics, never game/world carries.
"""
from collections import OrderedDict
from dataclasses import dataclass,asdict
from typing import NamedTuple
import weakref
import hashlib
import json

import jax
import jax.numpy as j
import numpy as np

from . import v5_public_branch as branch,v5_hypothetical_profile as hypothesis
from . import v5_observation,transition,engine_retained_encoder_driver as driver
from . import engine_private_actions as actions,engine_execution_completeness as execution
from . import engine_source_frame_witness as witness,engine_public_events as events

_REGISTERED=weakref.WeakKeyDictionary()
_CACHE=OrderedDict()  # At most four ORIGINAL prepared/adapter pure JIT holders.


def _host():
    if isinstance(j.zeros((),j.int32),jax.core.Tracer):raise ValueError('hypothetical grid host boundary cannot be traced')


@dataclass(frozen=True)
class _Layout:
    tree:object
    rows:tuple[bool,...]
    shapes:tuple
    dtypes:tuple


def _stack(values):
    """Record row-vs-GLOBAL geometry at B1, never infer it from output B=8."""
    if not values:raise ValueError('nonempty original B1 row trees required')
    first,tree=jax.tree.flatten(values[0]);all_leaves=[]
    for value in values:
        leaves,kind=jax.tree.flatten(value)
        if kind!=tree:raise ValueError('same original row tree structure required')
        all_leaves.append(leaves)
    rows=[];out=[];shapes=[];dtypes=[]
    for index,original in enumerate(first):
        shape=getattr(original,'shape',None);dtype=getattr(original,'dtype',None)
        if shape is None or dtype is None:raise ValueError('typed original device leaves required')
        candidates=[leaves[index] for leaves in all_leaves]
        if any(x.shape!=shape or x.dtype!=dtype for x in candidates):raise ValueError('same exact device layout required')
        row=bool(shape and shape[0]==1);rows.append(row);shapes.append(shape);dtypes.append(dtype)
        if row:out.append(j.concatenate(candidates,axis=0))
        else:
            if any(not np.array_equal(np.asarray(original),np.asarray(x)) for x in candidates[1:]):
                raise ValueError('GLOBAL namespace/static leaf differs across hypotheses')
            out.append(original)
    return jax.tree.unflatten(tree,out),_Layout(tree,tuple(rows),tuple(shapes),tuple(dtypes))


def _take(value,layout,index,batch):
    leaves,tree=jax.tree.flatten(value)
    if tree!=layout.tree or len(leaves)!=len(layout.rows):raise ValueError('recorded row layout changed')
    out=[]
    for x,row,shape,dtype in zip(leaves,layout.rows,layout.shapes,layout.dtypes,strict=True):
        expected=(batch,)+shape[1:] if row else shape
        if x.shape!=expected or x.dtype!=dtype:raise ValueError('exact recorded row/GLOBAL output shape required')
        out.append(x[index:index+1] if row else x)
    return jax.tree.unflatten(tree,out)


def _chance(value):
    if type(value) is not transition.ChanceInputs:raise ValueError('original finite ChanceInputs required')
    shapes=((1,2,12),(1,2),(1,2),(1,2),(1,2),(1,))
    limits=(np.array([2,100,100,100,16,16,100,4,8,2,2,2]),60,100,60,2,2)
    for x,shape,limit in zip(value,shapes,limits,strict=True):
        if x.shape!=shape or x.dtype!=j.int32 or np.any(np.asarray(x)<0) or np.any(np.asarray(x)>=limit):
            raise ValueError('original lawful B1 finite chance block required')


class _PreparedRow(NamedTuple):
    core:object
    publics:tuple
    streams:tuple
    record:object


def _planned_transport(root,cell,stream,ordinal,*,padding=None):
    if type(stream) is not int or not 0<stream<2**31:raise ValueError('bounded deterministic transport stream required')
    return dict(schema='pokezero.hypothetical.grid_transport.v1',address=asdict(cell.address),world=cell.world,
        root_action=cell.root_action,reply_action=cell.reply_action,root_index=cell.root_index,
        original_stream=root.history.stream_id,origin_sha256=root.origin_sha256,grid_ordinal=ordinal,
        stream=stream,padding=padding,scope='HYPOTHETICAL_TRANSPORT_NOT_SESSION_OR_BIRTH')


def _unique_transport(contexts):
    ids=[int(np.asarray(value.core.cursor.stream_id)[0]) for value in contexts]
    if len(set(ids))!=len(ids) or any(not 0<x<2**31 for x in ids):
        raise ValueError('unique bounded hypothetical packet namespace required')


def _transport(row,root,cell,stream,ordinal,*,padding=None):
    """Distinct HYPOTHETICAL packet namespace; no ordinary registration claim.

    Original history/address/source and all physical/RNG/word counters remain
    unchanged. The original device projector still enforces unique streams.
    """
    old=root.history.stream_id
    payload=dict(_planned_transport(root,cell,stream,ordinal,padding=padding),
        original_origin_binding=np.asarray(row.record.origin_binding).tolist())
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    digest=hashlib.sha256(raw).digest()
    if not any(digest):raise ValueError('nonzero hypothetical transport binding required')
    def replace(path,value):
        if path and getattr(path[-1],'name',None) in ('stream_id','stream_ids'):
            if value.shape!=(1,) or value.dtype!=j.int32 or int(np.asarray(value)[0])!=old:
                raise ValueError('original stream carrier differs before hypothetical transport')
            return j.array([stream],j.int32)
        return value
    result=jax.tree_util.tree_map_with_path(replace,row)
    record=result.record._replace(origin_binding=j.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32))[None,:])
    result=result._replace(record=record,core=result.core._replace(weather_profile=record))
    return result,dict(payload,binding_sha256=digest.hex())


def _own_view(appended,owners):
    """Select only B-row Observation fields, never namespace/global arrays."""
    if (len(appended)!=2 or any(type(x) is not v5_observation.Observation for x in appended)
            or owners.ndim!=1 or owners.dtype!=j.int32):raise ValueError('two genuine public29 views and exact owners required')
    b=len(owners)
    def choose(a,z):
        if not a.shape or a.shape[0]!=b or a.shape!=z.shape or a.dtype!=z.dtype:
            raise ValueError('public29 Observation must contain declared row-only leaves')
        return j.where((owners==1).reshape((b,)+(1,)*(a.ndim-1)),z,a)
    return jax.tree.map(choose,*appended)


class _DeviceResult(NamedTuple):
    frame:object
    publics:tuple
    appended:tuple
    good:object
    diagnostics:object
    frames:tuple


class _Executor:
    def __init__(self,prepared,adapter):
        self.prepared,self.adapter=prepared,adapter
        r=driver._checked(prepared);driver._checked_witness(r)
        if v5_observation.check_adapter(adapter) is not r['catalog']:
            raise ValueError('same original public29 adapter/catalog required')
        if any(r[name] is None for name in ('execution_compiled','kernel_risk_compiled','speed_profile','frame_witness_compiled')):
            raise ValueError('original hypothetical accounting/risk/profile/FULL statics required')
        self.runtime=r
        names=('request_fast','installations','encode_fast','public','execution_compiled','kernel_risk_compiled',
            'speed_profile','frame_witness_compiled','leftovers_context','status_residual_context','direct_hit_context','direct_hit_mode',
            'substitute_context','toxic_context','recoil_context','calm_mind_context','swords_dance_context','spikes_context','thunder_wave_context','sleep_powder_context')
        self.edges=tuple((name,r[name]) for name in names)
        data=r['speed_profile'].device
        silent=None if r['no_output_compiled'] is None else r['no_output_compiled'].device
        left=r['leftovers_context'];status=r['status_residual_context']
        left=None if left is None else hypothesis.LeftoversContext(left.data,data,silent)
        status=None if status is None else hypothesis.StatusContext(status.data,data)
        self.route=jax.jit(lambda core,choices:(
            actions.route_choice(core.requests[0].actions,core.orders[0],choices[:,0]),
            actions.route_choice(core.requests[1].actions,core.orders[1],choices[:,1])))

        # Separate PURE batched stages avoid cloning the large kernel/FULL
        # compiler graph per owner. Both owners reuse the SAME consume JIT.
        frame_fast=jax.jit(lambda core,record,choices,chance,active:driver._frame(core,
            (None,r['public'].causal.projector.device,r['request']['engine_table']),choices,chance,active,
            leftovers_context=left,status_residual_context=status,execution_tables=r['execution_compiled'].device,
            kernel_risk_tables=r['kernel_risk_compiled'].device,direct_hit_context=r['direct_hit_context'],
            substitute_context=r['substitute_context'],toxic_context=r['toxic_context'],recoil_context=r['recoil_context'],
            calm_mind_context=r['calm_mind_context'],swords_dance_context=r['swords_dance_context'],spikes_context=r['spikes_context'],thunder_wave_context=r['thunder_wave_context'],sleep_powder_context=r['sleep_powder_context'],
            hypothetical_profile=hypothesis.Context(record,data)))
        consume_fast=jax.jit(lambda ledger,frame,stream,public,record,active:execution.consume_frame_device(
            ledger,frame,stream,public,active=active,tables=r['execution_compiled'].device,
            adapter_tables=r['public'].device,catalog=r['catalog'],leftovers=None if left is None else left.data,
            status_residual=None if status is None else status.data,leftovers_no_output=silent,
            hypothetical_hp=hypothesis.Context(record,data)))
        public_commit=jax.jit(driver._public_rows)
        append_fast=jax.jit(lambda encoded,public,owner:v5_observation.append_device(encoded,public,adapter,observer_sides=owner))
        def run(core,publics,streams,record,engine_actions,chance,active):
            b=len(active)
            frame=frame_fast(core,record,engine_actions,chance,active)
            consumed=[];ledgers=[]
            for stream,public in zip(streams,publics,strict=True):
                ledger,result=consume_fast(frame.callbacks.execution.ledger,frame,stream,public,frame.weather_profile,active)
                consumed.append(result);ledgers.append(ledger)
            joined,complete=execution.join_consumed_owners(*ledgers,tables=r['execution_compiled'].device)
            raw=witness._raw_complete(frame.callbacks,r['frame_witness_compiled'].device)
            risks=frame.callbacks.execution.risks.carry
            expected,observed,geometry=witness._geometry(risks,frame.callbacks)
            cutpoints=geometry&j.all(risks.counts==expected,axis=1)&(risks.observed==observed)
            good=active&complete&raw&cutpoints&~frame.invalid_actions&(risks.risk==0)
            good &= (frame.callbacks.refusal&~events.UNINSTRUMENTED)==0
            public_after=tuple(public_commit(good,before,value.public) for before,value in zip(publics,consumed,strict=True))
            orders=tuple(r['installations'](value,frame) for value in core.orders)
            requests=branch._requests_after_frame(r,frame,core,publics,orders,good)
            after=core._replace(engine=frame.state,cursor=frame.cursor,orders=orders,requests=requests,admitted=good)
            appended=[];facts_ok=[];encoded_ok=[]
            for owner,public in enumerate(public_after):
                observer=driver.Observer(j.full(b,owner,j.int32),None,None,public,j.ones(b,bool),j.zeros(b,j.int32),None)
                facts,encoded=r['encode_fast'](after,observer)
                result=append_fast(encoded,public,j.full(b,owner,j.int32))
                facts_ok.append(facts.admitted);encoded_ok.append(encoded.observation.admitted);appended.append(result)
                good &= facts.admitted&encoded.observation.admitted&result.admitted
            diagnostics=j.stack((frame.callbacks.refusal,risks.risk,joined.refusal,raw,cutpoints,
                consumed[0].projection.error,consumed[1].projection.error,*facts_ok,*encoded_ok,
                appended[0].error,appended[1].error),axis=1).astype(j.int32)
            return _DeviceResult(frame,public_after,tuple(appended),good,diagnostics,tuple(x.projection.frames for x in consumed))
        self.run=run
        _REGISTERED[self]=(prepared,adapter,r,self.route,self.run,self.edges)

    def check(self):
        _host();bound=_REGISTERED.get(self) if type(self) is _Executor else None
        if (bound is None or self.prepared is not bound[0] or self.adapter is not bound[1]
                or self.runtime is not bound[2] or self.route is not bound[3] or self.run is not bound[4]
                or self.edges is not bound[5]):
            raise ValueError('original fixed hypothetical executor functions required')
        r=driver._checked(self.prepared);driver._checked_witness(r)
        if r is not self.runtime or any(r[name] is not value for name,value in bound[5]):
            raise ValueError('original static hypothetical executor edges changed')
        if v5_observation.check_adapter(self.adapter) is not r['catalog']:raise ValueError('public29 namespace changed')

    def prepare_row(self,root):
        self.check();r=self.runtime
        current=branch.core(root,r)
        publics=tuple(root.public if side==root.history.owner else branch._opponent_public(root,current,r) for side in (0,1))
        streams=tuple(branch._stream(public,current.cursor,side,r) for side,public in enumerate(publics))
        profile=hypothesis.qualify_origin(root)
        current=current._replace(weather_profile=profile.record)
        return _PreparedRow(current,publics,streams,profile.record)


def _executor(prepared,adapter):
    _host();key=(id(prepared),id(adapter))
    item=_CACHE.get(key)
    if item is None or item.prepared is not prepared or item.adapter is not adapter:
        item=_Executor(prepared,adapter);_CACHE[key]=item
    _CACHE.move_to_end(key)
    while len(_CACHE)>4:_CACHE.popitem(last=False)
    item.check();return item


def iter_execute(batch,adapter,*,max_rows=128):
    """Yield bounded, full-denominator GridChunks with sparse genuine inputs.

    ``device_rows`` names invoked kernel rows, ``active_rows`` the original
    valid routed rows, and padding the remaining frozen rows. They are not
    completed games or scientific throughput. Global failures remain unknown
    for their declared cells; no scalar-kernel fallback or retry is performed.
    """
    _host()
    from .v5_depth1_provider import GridBatch,GridChunk
    if type(batch) is not GridBatch:raise ValueError('exact declared GridBatch required')
    if type(max_rows) is not int or not 1<=max_rows<=128:raise ValueError('bounded literal max_rows1..128 required')
    # Revalidate frozen declaration contents, including object.__setattr__ drift.
    checked=GridBatch(batch.roots,batch.chance_inputs,batch.cells)
    first=checked.roots[0]
    chunks=(len(checked.cells)+max_rows-1)//max_rows
    if len(checked.cells)+chunks*max_rows>=2**31:raise ValueError('deterministic cell/padding transport namespace capacity exceeded')
    for value in checked.chance_inputs:_chance(value)
    executor=_executor(first.prepared,adapter);prepared=OrderedDict()
    for start in range(0,len(checked.cells),max_rows):
        executor.check();cells=checked.cells[start:start+max_rows]
        reasons=[() for _ in cells];contexts={};indices=[];transport=[]
        planned=tuple(_planned_transport(checked.roots[cell.root_index],cell,start+i+1,start+i)
            for i,cell in enumerate(cells))
        for index,cell in enumerate(cells):
            if cell.root_index not in prepared:
                try:value=executor.prepare_row(checked.roots[cell.root_index])
                except (ValueError,KeyError,IndexError) as error:value=(type(error).__name__+': '+str(error),)
                prepared[cell.root_index]=value
            prepared.move_to_end(cell.root_index)
            value=prepared[cell.root_index]
            if type(value) is _PreparedRow:
                contexts[index],mapping=_transport(value,checked.roots[cell.root_index],cell,start+index+1,start+index)
                transport.append((index,mapping));indices.append(index)
            else:reasons[index]=value
        while len(prepared)>128:prepared.popitem(last=False)
        if not indices:
            yield GridChunk(cells,(),None,np.zeros(len(cells),bool),tuple(reasons),
                dict(rows=(),planned_transport=planned,transport=(),padding=()));continue
        core,_=_stack([contexts[i].core for i in indices])
        choices=[]
        for i in indices:
            pair=[0,0];owner=cells[i].address.owner
            pair[owner]=cells[i].root_action;pair[1-owner]=cells[i].reply_action;choices.append(pair)
        routes=jax.block_until_ready(executor.route(core,j.asarray(choices,j.int32)))
        valid=np.asarray(routes[0].admitted&routes[1].admitted,bool)
        selected=[i for i,ok in zip(indices,valid,strict=True) if ok]
        for i,ok in zip(indices,valid,strict=True):
            if not ok:reasons[i]=('hypothetical-action-route-refused',)
        if not selected:
            yield GridChunk(cells,(),None,np.zeros(len(cells),bool),tuple(reasons),
                dict(rows=(),planned_transport=planned,transport=tuple(transport),padding=()));continue
        engine_actions=j.stack([route.engine_action for route in routes],axis=1)[j.asarray(np.flatnonzero(valid),j.int32)]
        count=len(selected);padding=max_rows-count
        repeated=[contexts[i] for i in selected];padding_map=[]
        for index in range(padding):
            original=prepared[cells[selected[0]].root_index]
            stream=len(checked.cells)+(start//max_rows)*max_rows+index+1
            value,mapping=_transport(original,checked.roots[cells[selected[0]].root_index],cells[selected[0]],
                stream,start+selected[0],padding=index)
            repeated.append(value);padding_map.append(mapping)
        _unique_transport(repeated)
        current,_=_stack([value.core for value in repeated])
        publics,_=_stack([value.publics for value in repeated])
        streams,_=_stack([value.streams for value in repeated])
        record,_=_stack([value.record for value in repeated])
        chance,_=_stack([checked.chance_inputs[cells[i].root_index] for i in selected]
            +[checked.chance_inputs[cells[selected[0]].root_index]]*padding)
        if padding:engine_actions=j.concatenate((engine_actions,j.repeat(engine_actions[:1],padding,axis=0)),axis=0)
        active=j.arange(max_rows)<count
        try:result=jax.block_until_ready(executor.run(current,publics,streams,record,engine_actions,chance,active))
        except (ValueError,KeyError,IndexError,RuntimeError) as error:
            for i in selected:reasons[i]=(type(error).__name__+': '+str(error),)
            yield GridChunk(cells,(),None,np.zeros(len(cells),bool),tuple(reasons),
                dict(rows=(),error='batched-device-error-no-retry',planned_transport=planned,
                    transport=tuple(transport),padding=tuple(padding_map)),max_rows,count,padding);continue
        good=np.asarray(result.good[:count],bool);admitted=np.zeros(len(cells),bool)
        diagnostic=[]
        for device_index,index in enumerate(selected):
            admitted[index]=good[device_index]
            if not good[device_index]:reasons[index]=('actual-hypothetical-kernel-FULL-or-public29-refused',)
            diagnostic.append((index,tuple(map(int,np.asarray(result.diagnostics[device_index])))))
        device_indices=np.flatnonzero(good);model_indices=tuple(i for i in selected if admitted[i])
        inputs=None
        if model_indices:
            owners=j.asarray([cells[i].address.owner for i in selected]+[cells[selected[0]].address.owner]*padding,j.int32)
            original=v5_observation.actor_inputs(_own_view(result.appended,owners),adapter)
            inputs={name:(value[j.asarray(device_indices,j.int32)] if hasattr(value,'shape') else value)
                for name,value in original.items()}
        yield GridChunk(cells,model_indices,inputs,admitted,tuple(reasons),
            dict(rows=tuple(diagnostic),planned_transport=planned,transport=tuple(transport),padding=tuple(padding_map)),
            max_rows,count,padding)
