"""Conditional observer math at a PUBLIC-SAMPLED hypothetical origin.

These records are deliberately not BirthRecord, ordinary Session provenance,
generator certificates or scientific measurements. They start at the exact
current materialized hypothesis, without resetting HP, turn or public history.
The original fixed-source speed-membership/retention arithmetic is reused.
Unknown public/private histories must already have refused materialization.
"""
from typing import NamedTuple
import hashlib

import jax
import jax.numpy as j
import numpy as np

from . import engine_weather_speed_profile as speed
from . import engine_retained_encoder_driver as driver
from .v5_public_world import HypotheticalRoot


class HypotheticalRecord(NamedTuple):
    qualified_origin:object
    stream_id:object
    engine_binding:object
    source_binding:object
    profile_binding:object
    origin_spe:object
    origin_ability:object
    origin_binding:object


class Context(NamedTuple):
    record:HypotheticalRecord
    data:object


class LeftoversContext(NamedTuple):
    data:object
    weather_profile_data:object
    no_output_tables:object=None


class StatusContext(NamedTuple):
    data:object
    weather_profile_data:object
    hook_context:object=None


def check_context(context,b):
    if type(context) is not Context or type(context.record) is not HypotheticalRecord or type(context.data) is not speed.ProfileData:
        raise ValueError('distinct hypothetical observer Context/record required')
    r=context.record
    for value,shape,dtype in ((r.qualified_origin,(b,),j.bool_),(r.stream_id,(b,),j.int32),
            (r.engine_binding,(b,8),j.uint32),(r.source_binding,(b,8),j.uint32),
            (r.profile_binding,(b,8),j.uint32),(r.origin_binding,(b,8),j.uint32),
            (r.origin_spe,(b,2,6),j.int32),(r.origin_ability,(b,2,6),j.int32)):
        speed._typed(value,shape,dtype)
    return context


def retain_device(record,state,streams,data):
    b=speed._state_shape(state,streams);check_context(Context(record,data),b)
    mask=speed._retention_mask_device(state,streams,data,record.qualified_origin,
        record.stream_id,record.engine_binding,record.source_binding,record.profile_binding,
        record.origin_spe,record.origin_ability)
    mask &= j.any(record.origin_binding!=0,axis=1)
    return record._replace(qualified_origin=mask)


def qualify_origin(branch):
    """Trusted factory seam; a current hypothesis is never an ordinary birth."""
    if type(branch) is not HypotheticalRoot:raise ValueError('reviewed public-root materialization required')
    runtime=driver._checked(branch.prepared);catalog=runtime['catalog']
    if branch.history.catalog is not catalog or branch.history.session_identity[1]!=id(branch.prepared):
        raise ValueError('same original public history/source namespace required')
    profile=runtime['speed_profile']
    if profile is None:raise ValueError('original static speed-math compiler required')
    data=speed.check_profile(profile);state=branch.state
    streams=j.array([branch.history.stream_id],j.int32);speed._state_shape(state,streams)
    try:origin=bytes.fromhex(branch.origin_sha256)
    except (ValueError,TypeError):raise ValueError('hypothetical source/history/sample origin binding required') from None
    if len(origin)!=32:raise ValueError('hypothetical origin SHA256 required')
    bound=hashlib.sha256(b'v5.hypothetical.math.origin\0'+origin).digest()
    binding=j.asarray(np.frombuffer(bound,dtype='>u4').astype(np.uint32))[None,:]
    broadcast=lambda value:j.broadcast_to(value,(1,8))
    good=speed._members(state,data,birth=False)&(streams>0)&~state.done&(state.turn>=0)
    record=HypotheticalRecord(good,streams,broadcast(data.engine_binding),broadcast(data.source_binding),
        broadcast(data.binding),state.stats[...,5],state.base_ability,binding)
    return Context(record,data)
