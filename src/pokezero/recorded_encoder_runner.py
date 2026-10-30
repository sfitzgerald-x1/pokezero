"""One shared JAX prefix per battle, then only each actor's private request.

This module does not attest files, select rows, start games or qualify player
visibility. Its caller must stream the complete fixed, hash-attested roster.
Terminal records are validated separately and never become decision inputs.
"""
import hashlib
import json
import numpy as np

from .recorded_encoder_corpus import RecordedEncoderCursor
from .recorded_encoder_parity import SHAPES


def compare_battle(rows,*,battle,stream_id,adapter,tally,with_forward):
    """Consume every boundary, including asymmetric and empty public deltas."""
    if type(with_forward) is not bool:raise ValueError('explicit forward mode required')
    if type(stream_id) is not int or not 0<stream_id<2**31:raise ValueError('registered positive stream ID required')
    cursor=RecordedEncoderCursor(battle_id=battle['battle_id'],seed=battle['seed'])
    adapter.reset(stream_id=stream_id,battle_id=battle['battle_id'])
    boundaries=views=0
    for row in rows:
        if not isinstance(row,dict) or cursor.closed:raise ValueError('invalid or post-terminal record')
        if row.get('event')=='end':
            cursor.close(row);continue
        projection=cursor.advance(row)
        adapter.feed(projection.trusted_nonrequest_delta)
        boundaries+=1
        for view in projection.views:
            actual=adapter.encode(json.loads(view.own_request_json),owner_side=int(view.seat[1])-1)
            kwargs={}
            if with_forward:
                reference=json.loads(view.own_reference_json)
                # Only the owning observation tensors enter either graph.
                kwargs=dict(actual_forward=adapter.forward(actual),
                    reference_forward=adapter.forward({k:reference[k] for k in SHAPES}))
            tally.add(view,actual,**kwargs);views+=1
    if not cursor.closed:raise ValueError('complete terminal record required')
    return dict(battle_id=battle['battle_id'],boundaries=boundaries,views=views,
        source_lines=cursor.source_lines,source_prefix_sha256=cursor.source_prefix_sha256,
        terminal_validated_not_encoded=True)


class JaxRecordedAdapter:
    """Actual device fold/encode with atomic public-boundary host commitment."""
    def __init__(self,catalog,*,model=None,torch_module=None,owner_side=None):
        import jax
        from gpu_engine import v4_encoder
        self.jax=jax;self.encoder=v4_encoder;self.catalog=catalog
        self.model=model;self.torch=torch_module;self.registered={}
        if owner_side is not None and (type(owner_side) is not int or owner_side not in (0,1)):
            raise ValueError('absolute owning side must be zero or one')
        self.owner_side=owner_side
        if (model is None)!=(torch_module is None):raise ValueError('model and Torch supplied together')
        if model is not None:
            if model.training or any(p.dtype!=torch_module.float32 or p.device.type!='cpu' for p in model.parameters()):
                raise ValueError('CPU eval FP32 champion required')
        self.step=jax.jit(lambda state,chunk:v4_encoder.fold_chunk(state,chunk,catalog))
        self.encode_step=jax.jit(lambda state,own,side:v4_encoder.encode(state,own,catalog,observer_sides=side))

    def reset(self,*,stream_id,battle_id):
        from gpu_engine import public_wire
        if (type(stream_id) is not int or not 0<stream_id<2**31 or stream_id in self.registered
                or not isinstance(battle_id,str) or not battle_id.strip() or battle_id in self.registered.values()):
            raise ValueError('fresh unique stream/battle registration required')
        self.stream_ids=np.array([stream_id],np.int32);self.battle_id=battle_id
        owners=None if self.owner_side is None else np.array([self.owner_side],np.int32)
        self.state=self.encoder.initialize(self.catalog,stream_ids=self.jax.numpy.asarray(self.stream_ids),owner_sides=owners)
        self.host=public_wire.WireState();self.bindings=None;self.registered[stream_id]=battle_id

    def feed(self,lines):
        from gpu_engine import public_inputs,public_wire
        if not isinstance(lines,tuple) or any(not isinstance(line,str) or line.startswith('|request|') for line in lines):
            raise ValueError('non-request source delta required')
        host=self.host;projections=[]
        for index,line in enumerate(lines):
            try:host,projection=public_wire.project_line(host,line,**self.encoder.wire_kwargs(self.catalog))
            except ValueError as error:
                kind=line.split('|')[1] if line.startswith('|') else 'malformed'
                raise ValueError('wire refused kind='+kind+' offset='+str(index)+' sha256='+hashlib.sha256(line.encode()).hexdigest()) from error
            projections.append(projection)
        bindings=self.bindings
        if projections:
            bindings={name:np.asarray(getattr(projections[0],name),np.uint32) for name in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
                'active_binding','damage_binding','type_binding','sleep_binding','toxic_binding','tendency_binding')}
        if bindings is None:raise ValueError('complete source prefix must precede first decision')
        events=[event for projection in projections for event in projection.events]
        state=self.state
        for chunk in public_inputs.pack_chunks(self.stream_ids,[events],source_binding=self.catalog.source.device.binding,**bindings):
            state,error=self.step(state,chunk)
            if np.any(np.asarray(error)!=0):raise ValueError('JAX public fold refused boundary')
        # A late host/device failure leaves both previous public cursors intact.
        self.state,self.host,self.bindings=state,host,bindings

    def encode(self,request,*,owner_side):
        if self.owner_side is not None and owner_side != self.owner_side:
            raise ValueError('request owner differs from owning stream')
        own=self.encoder.parse_own_requests(self.catalog,[request],owner_sides=[owner_side])
        out=self.encode_step(self.state,own,self.jax.numpy.array([owner_side],self.jax.numpy.int32))
        observation=out.observation
        valid=bool(np.asarray(observation.admitted).all()) and not np.asarray(out.pending_numeric_columns).size
        for name,coverage in (('numeric_features','numeric_written'),('categorical_ids','categorical_written')):
            values=np.asarray(getattr(observation,name));written=np.asarray(getattr(observation,coverage))
            # Masks name applicable writer cells. Other token/column pairs
            # retain source zero defaults; every column must still be installed
            # somewhere, and an unwritten nonzero cell can never be admitted.
            valid &= bool(written.any(axis=(0,1)).all()) and not np.any(values[~written]!=0)
        if not valid:raise ValueError('incomplete or refused JAX observation')
        return {name:np.asarray(getattr(observation,name))[0] for name in SHAPES}

    def forward(self,observation):
        if self.model is None:raise ValueError('no frozen champion supplied')
        torch=self.torch
        tensors={}
        for name in ('numeric_features','categorical_ids','token_type_ids','attention_mask'):
            dtype=np.float32 if name=='numeric_features' else np.bool_ if name=='attention_mask' else np.int64
            value=np.array(observation[name],dtype=dtype,copy=True)[None,None]
            tensors[name]=torch.from_numpy(value)
        with torch.no_grad():out=self.model(**tensors,history_mask=torch.ones((1,1),dtype=torch.bool))
        return dict(policy_logits=out.policy_logits.detach().cpu().numpy()[0],value=out.value.detach().cpu().numpy()[0])
