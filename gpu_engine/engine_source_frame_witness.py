"""Conditional actual-kernel frame input witness; never B4/physics qualification.

This NEW-only seam actually runs BOTH original full owner consumers. It combines
the pinned owning source's literal cutpoint geometry, the entire raw frame,
source-required claims and genuine stage3/cursor/chunk joins. Neither risk0,
all observed bits nor scoped_drained alone is sufficient. Unknown paths refuse.

ORIGIN CONTRACT: root must call check_host_origin outside every owning phase,
initialize only at its actual constructor, retain this carry in the original
Session/Core, and call this seam on the sole just-executed frame BEFORE encoding.
Pure device arrays/copies are trusted same-core transport, NOT origin attestation.
This module does not edit/wire the driver, change full flags or broaden actions.
Conditional covered frames are incremental; every unsupported actual family and
the eventual full self-play/encoder/bf16 throughput/physics canary remain work.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_execution_completeness as execution
from . import engine_kernel_frame_risk as risks
from . import engine_lifecycle_callbacks as life
from . import engine_full_v4_event_device as full
from . import engine_pressure_callbacks as pressure
from . import engine_trace_callbacks as trace
from . import engine_public_events as public, public_toxic, v4_encoder
from .state import BattleState
from . import engine_substitute_callbacks as substitute
from . import engine_toxic_callbacks as toxic
from . import engine_recoil_callbacks as recoil

SCHEMA = 'pokezero.actual_source_frame_witness.v1'
SOURCE, CUTPOINTS, RAW, ACCOUNTING, DRAIN, ORDER, PRIOR = (1<<i for i in range(7))
_COMPILED = weakref.WeakKeyDictionary()
# Exact frozen source audited for this geometry. Root integration/capability
# changes require an explicit reviewed literal refresh, never auto-registration.
SOURCE_PINS = {'absorption.py': '2d684602eb9aa69bc470b215c188533b0e1c4d8495dd5b71ca2508dc38f20623', 'action_history.py': 'a47bdb2712f7eadaed12d91ede70be2ebe9f1c1ae3f70fe56495437997212407', 'baton.py': 'f08fb40dfc1e2d74e7a8fb2e42285c51a2ea4106484951580bf2aa9d23b29706', 'combat.py': '3e802d3fd2f488e9aa2a8ead663f935158c5fec1b7fb745034e54f21d38015af', 'critical.py': 'b66e6cf10c0d4df625b1c4f5e5db238af42052e556d7649a10f911745181401e', 'damage_moves.py': '227c58ab6042cbb0178dcc821e1be7dd2af99aae0f149febef995bdb7662abd1', 'effects.py': '0138f0e90811c4829666960a106d80ed2d59f6cd06a8ff83324d189465f73104', 'encore.py': '63da39bbd181a8c6371b3bca853322291a03674c60f535ad0eacdbe1a440e0dd', 'engine_direct_hit_controls.py': '5ec305fdbfa84841ef93bddfa308095eddb302c7298d75afd46ae1aacc4e1de9', 'engine_execution_completeness.py': '5fb8c5a3ace3da5e5bca423e646ba5d3118cb5ed9e2bb15b8af847fe815c9255', 'engine_full_v4_event_device.py': '69a85b05d4719287b952956094ea4e566abe128b83901e38f8fd0dbdfab6cc9f', 'engine_item_callbacks.py': 'b7984d566834887545526f2ffadd50ec1f7b98360f4d751a761d6bf9eb69eabd', 'engine_kernel_frame_risk.py': '83a44e3416ecd5e8c2a9bea08005dfdf8cbbc7fd22d5c16a9454db12ac3819fb', 'engine_leftovers_callbacks.py': '609858fbbc7ec669fe00da6a2fcd1832187616f3146ddef465f9ad6f9e73dce0', 'engine_leftovers_no_output.py': '29bd8cb04cd935485e877b757fa5056fe81cfaa9a51172e926eae779887540e9', 'engine_lifecycle_callbacks.py': '2eb8a091a86cc1adbbf2d4fabe69967e6f7c4e39eccc70f8a9cfab2fdbbab4c5', 'engine_lifecycle_event_device.py': '083d4568625421c222379af1e87eb014e549a396f5b954f3de6198e23e5cf10c', 'engine_pressure_callbacks.py': '0de20779233ec90be004c0c604809ad5387edfc617cffaa7ce94d51f5ff875e9', 'engine_public_events.py': '091702fcb890ff2eefe0bb7ba7e42b3908e294c386bbb833841a4167a9a84549', 'engine_recoil_callbacks.py': 'ae7183b5eb472d18dff62741c36b421a48db456d158aa45c02216e50cee5de5e', 'engine_request_source.py': 'f58f4999103a4f747088167a7d55ac7168cdfc779d091d4e88aa85aa465a3e80', 'engine_retained_encoder_driver.py': '4bc33d4d0a286ebef12ca1d53b65ec12faba6fcb8077e2d0f50f87c757e296ca', 'engine_status_callbacks.py': '369bde361dbc8cffd357dc92ae576686ee6bbea8e5a96fbd9a8fb08fe86ec23e', 'engine_status_residual_callbacks.py': '8435c5d3eac500fefc829435119ed0eae030c77db5d9541ace50b2bea7a7fe3e', 'engine_substitute_callbacks.py': '04e8d98dd1801a45a6da132d631f53eadb6da1d0f45622e0d1a23210a79be2a4', 'engine_toxic_callbacks.py': '0ff5096482290590dd1647bcb8a7815f0348abaf1c1f8e77081a6aa665be17c1', 'engine_trace_callbacks.py': 'f33f7267e6e03a71c975c723cf59a38882fd499cc682ae666dfaae83bd30d30c', 'engine_weather_callbacks.py': 'b5fee00fa633de1ef03062bffeab107c37ea15124b38a6ffa70c0820e5521d87', 'engine_weather_effect_callbacks.py': '66deb3debe354d8e637524bfcc4b94218863ba501e8da430797d1ae7ac7048f9', 'engine_weather_hp_segment.py': 'ba79f08ef3db40ac557240e7c2589da2683d7441e8e6faa35787bb0ebad43284', 'engine_weather_order_callbacks.py': 'fe21ff43c8e5687b2c1cf6a80596fcad466a93cf06ea57b4415cc8c069c3c6fe', 'engine_weather_progress_callbacks.py': 'a643b28bce10d328ca7359e1685e7b692e316e975c10dc15786cec95b242729e', 'engine_weather_speed_profile.py': 'e1b096cd319bd2f6d030abcbd7bcb12d59ab991dc9204675678d35f1ef51feb1', 'entry.py': '76bca9da8780d31d0efc4e7b74dc7599e935f99226fa8c57174c966920e15744', 'focus_punch.py': '881bd17a84dc22576aa618256ada2b0f67bee29224dbb50508925b55e119585f', 'items.py': 'db32fbb3d0062aa20c965505055b8506c3158d5a408c258095fa947aa0c06c43', 'linked_traps.py': '743c01ee69509162e7829ae555b85f069f1c20ec9bd19cec0779d066894bade4', 'mechanics.py': 'e8476cc3e0029f3ed0bf82276312440e6dd8b340ffeb0c68cdd5611754eef661', 'phazing.py': '8f82d4894f03b39936de0e79b17f0b7715fdfe2f0edb6a7a42d44f1840671bee', 'pressure.py': '73eefb16678fedf1885a4e11cbc8954d15706592742caa6b0b673fb4d63d698c', 'pursuit.py': '2b01102bb3819791fc1edea0ba0fe0a0b913e8dfa5a8f6501f48f3fe260a8309', 'secondary.py': '98fa5cdd3777c4843cb427f141bbd3aa2799cf6c9f84a3cd2630dad3ef52f071', 'self_destruct.py': 'b3c9df0ea093eca7c0960b4c4ffd4277e289a0ded6f32f914466c0752fa9b739', 'sleep_talk.py': '968a78ddde87158b13f68dfcacc2158f07336d82c1be472d2283421e8583c154', 'solar_beam.py': '5463e103392e24df7cc560003740b7e425aba766a0d1bba90ff4428606849155', 'state.py': 'e657242e2d7191c4c7c1acb265d8f5751a6f00acaa08c08998af208d90124fe6', 'sustain.py': '1ff481060f04384eae49da8bfeb663296c8efe582b0a7562fa3e3103ecbfd529', 'transition.py': '6fbd113db922e28f6d1f7b9f8c49de8d8c61a8cde6ff64fda5c302960c82598c', 'variable_moves.py': '7b06b3f0a26abc023036edb4ac0d4050e550522dd231d2d2a74cebf8f569e9b9', 'weather.py': '3c861027e16d27917fa47fa5cf5d6dc0c68d3a74af282108dfb94369d4db57e5'}


_REGISTERED_SOURCE_PINS = tuple(sorted(SOURCE_PINS.items()))


class Tables(NamedTuple):
    binding: object                       # GLOBAL8, including B==8
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    encoder_binding: object
    accounting: object
    risk: object
    adapter: object
    profile: object
    leftovers: object
    status: object
    no_output: object
    secondary_weather: object             # ORIGINAL selected-column proof [M]
    direct_hit_binding: object
    substitute_binding: object
    toxic_binding: object


@dataclass(frozen=True, eq=False)
class Compiled:
    device: Tables


class Carry(NamedTuple):
    binding: object                       # GLOBAL8
    stream_id: object
    frame: object                         # includes constructor frame1
    sequence: object                      # [B,2] genuine owner word cursors
    refusal: object                       # lifetime sticky


class Report(NamedTuple):
    frame_inputs_complete: object         # conditional source/input scope only
    refusal: object
    expected_cutpoints: object
    observed_cutpoints: object
    raw_words: object                     # complete literal frame, not claims only
    raw_count: object
    event_turn: object
    event_phase: object
    event_generation: object
    switch_details: object
    claimed: object
    owner_words: object                   # [B,2] actually drained FULL words
    owner_full_words: object              # [B,2,C,N,6] actual projected chunks
    owner_full_valid: object              # [B,2,C,N], before refusal rollback
    owner_full_more: object               # [B,2,C], before refusal rollback


class Result(NamedTuple):
    carry: Carry
    ledger: object
    owners: tuple                        # actual full.Consumed values, not input flags
    report: Report


def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked frame witness host boundary cannot be traced')


def _source():
    if type(SOURCE_PINS) is not dict or tuple(sorted(SOURCE_PINS.items())) != _REGISTERED_SOURCE_PINS:
        raise ValueError('actual source inventory changed')
    for name, expected in SOURCE_PINS.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError('ACTUAL_KERNEL_SOURCE_DRIFT: '+name)


def compile_witness(prepared):
    _host(); _source()
    from . import engine_retained_encoder_driver as driver
    runtime = driver._checked(prepared)
    if (runtime['public_mode'] != 'retained-full' or runtime['execution_compiled'] is None
            or runtime['kernel_risk_compiled'] is None):
        raise ValueError('original full/accounting/kernel-risk runtime required')
    account, _, adapter = execution._checked(runtime['execution_compiled'])
    risk = risks.check_compiled(runtime['kernel_risk_compiled'])
    catalog = runtime['request']['lifecycle']
    record = risks._COMPILED[runtime['kernel_risk_compiled']]
    if (record[2] is not catalog or record[3] is not runtime['execution_compiled']
            or adapter is not runtime['public'] or adapter.causal.projector.catalog is not catalog):
        raise ValueError('same original kernel/accounting/full compiler edges required')
    table = catalog.engine_table
    # These original modules cause the unconditional per-live-loop guard calls.
    for name in ('effects','sustain','linked_traps','encore','entry','secondary'):
        if getattr(table,name) is None:
            raise ValueError('declared actual guard geometry requires original '+name+' table')
    if table.entry.own_tempo is None:
        raise ValueError('declared entry guard geometry required')
    # Secondary guards are called with acts&hit&~immune, even for zero-power
    # weather. Prove that dispatch from these ORIGINAL immutable columns; do
    # not infer it from HP loss or assume every weather row is eligible.
    m=len(table.power); weather_secondary=np.zeros(m,bool)
    if table.weather is not None:
        typ=np.asarray(table.type);safe=np.clip(typ,0,table.type_num.shape[0]-1)
        weather_secondary=((np.asarray(table.weather.move_weather)>0)&(np.asarray(table.power)==0)
            &np.asarray(table.always_hit)&~np.asarray(table.effects.protectable)
            &(np.asarray(table.effects.status)==0)&(typ>=0)&(typ<table.type_num.shape[0])
            &np.all(np.asarray(table.type_num)[safe]>0,axis=1))
        if table.variable_moves is not None:
            for name in ('hidden_power','returns','frustration','hp_dependent'):
                weather_secondary &= ~np.asarray(getattr(table.variable_moves,name))
    direct = runtime['direct_hit_compiled']
    direct_binding = jnp.zeros(8,jnp.uint32)
    if direct is not None and runtime['direct_hit_mode'] == 'kernel-event':
        from . import engine_direct_hit_controls as hit
        data = hit.check_compiled(direct)
        if hit._COMPILED[direct][2] is not catalog:
            raise ValueError('same original direct-hit catalog required')
        direct_binding = data.binding
    substitute_binding=jnp.zeros(8,jnp.uint32)
    if runtime['substitute_compiled'] is not None and runtime['substitute_mode']=='kernel-event':
        from . import engine_substitute_callbacks as substitute
        data=substitute.check_compiled(runtime['substitute_compiled'])
        if substitute._COMPILED[runtime['substitute_compiled']][2] is not catalog:
            raise ValueError('same original Substitute catalog required')
        substitute_binding=data.binding
    toxic_binding=jnp.zeros(8,jnp.uint32)
    if runtime['toxic_compiled'] is not None and runtime['toxic_mode']=='kernel-event':
        data=toxic.check_compiled(runtime['toxic_compiled'])
        if toxic._COMPILED[runtime['toxic_compiled']][2] is not catalog:raise ValueError('same original Toxic catalog required')
        toxic_binding=data.binding
    own_source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = dict(schema=SCHEMA,source=SOURCE_PINS,witness_source=own_source,prepared=prepared.binding,
        risk=np.asarray(risk.binding).tolist(),account=np.asarray(account.inventory_binding).tolist(),
        full=np.asarray(adapter.device.catalog_binding).tolist() if hasattr(adapter.device,'catalog_binding') else
            np.asarray(runtime['catalog'].binding).tolist(),
        direct=np.asarray(direct_binding).tolist(),substitute=np.asarray(substitute_binding).tolist(),toxic=np.asarray(toxic_binding).tolist(),secondary_weather=weather_secondary.tolist(),
        scope='conditional-actual-frame-inputs-not-origin-or-physics')
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    words = lambda x:jnp.asarray(x,jnp.uint32)
    data = Tables(words(np.frombuffer(digest,dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),
        words(catalog.source_binding),words(catalog.move_binding),account.producer_binding,
        runtime['catalog'].binding,account,risk,adapter.device,
        None if runtime['speed_profile'] is None else runtime['speed_profile'].device,
        None if runtime['leftovers_context'] is None else runtime['leftovers_context'].data,
        None if runtime['status_residual_context'] is None or runtime['status_residual_mode']!='kernel-event'
            else runtime['status_residual_context'].data,
        None if runtime['no_output_compiled'] is None else runtime['no_output_compiled'].device,
        jnp.asarray(weather_secondary),direct_binding,substitute_binding,toxic_binding)
    result = Compiled(data)
    edges = tuple(runtime[k] for k in ('request_compiled','own_compiled','public','execution_compiled','kernel_risk_compiled'))
    _COMPILED[result] = (data,tuple(jax.tree.leaves(data)),weakref.ref(prepared),edges,own_source,
        _REGISTERED_SOURCE_PINS)
    return result


def check_compiled(compiled):
    _host(); _source()
    from . import engine_retained_encoder_driver as driver
    record = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (record is None or compiled.device is not record[0]
            or len(jax.tree.leaves(compiled.device)) != len(record[1])
            or any(a is not b for a,b in zip(jax.tree.leaves(compiled.device),record[1]))):
        raise ValueError('original immutable frame witness compiler required')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=record[4]:
        raise ValueError('actual frame witness source changed')
    if tuple(sorted(SOURCE_PINS.items()))!=record[5]:
        raise ValueError('original compiled source inventory changed')
    prepared = record[2]()
    if prepared is None:
        raise ValueError('original owning prepared runtime no longer exists')
    runtime = driver._checked(prepared)
    if any(runtime[k] is not edge for k,edge in zip(
            ('request_compiled','own_compiled','public','execution_compiled','kernel_risk_compiled'),record[3])):
        raise ValueError('original frame witness runtime edges changed')
    return compiled.device


def check_host_origin(compiled, *, constructed=None, session=None):
    """Required ORIGINAL host objects, never an origin/accepted bool argument.

    Initial integration must pass the exact snapshot returned by the root's
    just-executed constructor. Later integration must pass its original current
    Session; root's _session guard owns exact Core object identity and retention.
    """
    _host(); data = check_compiled(compiled)
    from . import engine_request_source as source, engine_retained_encoder_driver as driver
    if session is not None:
        actual = driver._session(session)
        if session.prepared is not _COMPILED[compiled][2]():
            raise ValueError('same original owning Session required')
        if constructed is not None and constructed is not actual['constructed']:
            raise ValueError('same original constructor snapshot required')
        constructed = actual['constructed']
    if constructed is None:
        raise ValueError('actual original constructor or current Session required')
    source._constructed(constructed)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        if not np.array_equal(np.asarray(getattr(constructed.callbacks,name)),np.asarray(getattr(data,name))):
            raise ValueError('original constructor/source namespace mismatch')
    return data


def initialize_from_constructed(compiled, constructed):
    """Only original constructor host entry; no generic initialize(streams)."""
    data = check_host_origin(compiled,constructed=constructed)
    streams = constructed.cursor.stream_id
    b = streams.shape[0]
    return Carry(data.binding,streams,jnp.zeros(b,jnp.int32),jnp.zeros((b,2),jnp.int32),jnp.zeros(b,jnp.int32))


def _rows_equal(a,b):
    return jnp.all((a==b).reshape((a.shape[0],-1)),axis=1)


def _same_ledger(a,b):
    same = jnp.all(a.inventory_binding==b.inventory_binding)
    for name in a._fields:
        if name!='inventory_binding':same &= _rows_equal(getattr(a,name),getattr(b,name))
    return same


def _geometry(carry,tape):
    initial = carry.frame==0
    raw = tape.words[:,:,0]; present = jnp.arange(raw.shape[1])[None,:] < tape.count[:,None]
    arrivals = jnp.sum(present&((raw==life.INITIAL_REVEAL)|(raw==life.SWITCH_INSTALLED)),axis=1,dtype=jnp.int32)
    moves = jnp.sum(present&(raw==public.MOVE),axis=1,dtype=jnp.int32)
    attempts=present&(raw==toxic.TOXIC_ATTEMPT)
    # Literal actual SECONDARY dispatch mask captured at the primary cutpoint.
    # A miss is not guessed from missing callbacks; unknown attempts still fail
    # raw/site/consumer admission separately.
    secondary_moves=moves-jnp.sum(attempts,axis=1,dtype=jnp.int32)+jnp.sum(
        attempts&((tape.words[:,:,6]&toxic.SECONDARY)!=0),axis=1,dtype=jnp.int32)
    regular = np.full(len(risks.Branch),2,np.int32)
    regular[risks.Branch.CONSTRUCTOR-1]=0; regular[risks.Branch.FRAME_END-1]=1
    regular[risks.Branch.ENCORE-1]=12
    expected = jnp.where(initial[:,None],0,jnp.asarray(regular)[None,:])
    expected = expected.at[:,risks.Branch.CONSTRUCTOR-1].set(initial.astype(jnp.int32))
    expected = expected.at[:,risks.Branch.FRAME_END-1].set(1)
    expected = expected.at[:,risks.Branch.ENTRY_CONFUSION_CURE-1].set(arrivals)
    expected = expected.at[:,risks.Branch.SECONDARY_STATS-1].set(jnp.where(initial,0,3*secondary_moves))
    expected = expected.at[:,risks.Branch.SECONDARY_FLINCH-1].set(jnp.where(initial,0,secondary_moves))
    mask = jnp.sum(jnp.where(expected>0,jnp.left_shift(jnp.int32(1),jnp.arange(len(risks.Branch))),0),axis=1,dtype=jnp.int32)
    initial_rows = jnp.sum(present&(raw==life.INITIAL_REVEAL),axis=1,dtype=jnp.int32)
    geometry = jnp.where(initial,initial_rows==2,initial_rows==0)
    return expected,mask,geometry


def _raw_complete(tape,tables):
    ledger = tape.execution.ledger; kind = tape.words[:,:,0]
    present = jnp.arange(kind.shape[1])[None,:] < tape.count[:,None]
    known = ((kind>=1)&(kind<=17))|((kind>=20)&(kind<=23))|(kind==pressure.PRESSURE_START)|(kind==substitute.SUBSTITUTE_CREATED)|(kind==toxic.TOXIC_ATTEMPT)|(kind==trace.TRACE_COPY)|(kind==recoil.RECOIL_HP)
    direct = (kind==public.MAIN_HP)|(kind==public.DIRECT_FAINT)|((kind>=21)&(kind<=23))
    required = direct|(kind==17)|(kind==20)|(kind==25)|(kind==26)|(kind==trace.TRACE_COPY)|(kind==recoil.RECOIL_HP)
    claim = ledger.claimed; slot = jnp.clip(claim-1,0,ledger.site.shape[1]-1)
    site = jnp.take_along_axis(ledger.site,slot,axis=1)
    expected = jnp.where(direct,int(execution.Site.DIRECT_HIT),jnp.where(kind==17,int(execution.Site.LEFTOVERS),
        jnp.where(kind==25,int(execution.Site.SUBSTITUTE_SELF),
            jnp.where(kind==26,int(execution.Site.TOXIC_ATTEMPT),
                jnp.where(kind==trace.TRACE_COPY,int(execution.Site.TRACE_COPY),
                    jnp.where(kind==recoil.RECOIL_HP,int(execution.Site.RECOIL_HP),int(execution.Site.STATUS_RESIDUAL)))))))
    captured = (claim>0)&(claim<=ledger.count[:,None])&(site==expected)
    registered_direct = getattr(tables.risk,'direct_hit_binding',jnp.zeros(8,jnp.uint32))
    capability = jnp.any(tables.direct_hit_binding!=0)&jnp.all(registered_direct==tables.direct_hit_binding)
    # Every counted MOVE must prove the actual secondary dispatch. Positive
    # direct calls use their fresh stage2 source ticket immediately after MOVE;
    # zero-power calls use the immutable selected-column proof above.
    move=tape.words[:,:,5];valid_move=(move>0)&(move<len(tables.secondary_weather))
    weather=tables.secondary_weather[jnp.clip(move,0,len(tables.secondary_weather)-1)]
    entries=jnp.arange(ledger.stage.shape[1])[None,:]<ledger.count[:,None]
    ticket=(entries&(ledger.site==int(execution.Site.DIRECT_HIT))&(ledger.stage==2)
        &(ledger.no_output_kind==0)&(ledger.end>ledger.start))
    index=jnp.arange(kind.shape[1])[None,:,None]
    follows=jnp.any(ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    candidate=getattr(tables.risk,'direct_candidates',jnp.zeros_like(tables.secondary_weather))
    ordinary=capability&candidate[jnp.clip(move,0,len(candidate)-1)]&follows
    sub_binding=getattr(tables.risk,'substitute_binding',jnp.zeros(8,jnp.uint32))
    sub_capability=jnp.any(tables.substitute_binding!=0)&jnp.all(sub_binding==tables.substitute_binding)
    sub_ticket=(entries&(ledger.site==int(execution.Site.SUBSTITUTE_SELF))&(ledger.stage==2)
        &(ledger.no_output_kind==0)&(ledger.end==ledger.start+1))
    sub_follows=jnp.any(sub_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    sub_candidate=getattr(tables.risk,'substitute_candidates',jnp.zeros_like(tables.secondary_weather))
    captured_sub=sub_capability&sub_candidate[jnp.clip(move,0,len(sub_candidate)-1)]&sub_follows
    # Actual geometry ALSO checks every secondary guard count. Unlike weather,
    # this is no universal no-immunity assertion from a static move name.
    tox_cap=jnp.any(tables.toxic_binding!=0)&jnp.all(tables.risk.toxic_binding==tables.toxic_binding)
    tox_ticket=(entries&(ledger.site==int(execution.Site.TOXIC_ATTEMPT))&(ledger.stage==2)
        &(ledger.no_output_kind==0)&((ledger.end==ledger.start+1)|(ledger.end==ledger.start+2)))
    tox_follows=jnp.any(tox_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    captured_tox=tox_cap&tables.risk.toxic_candidates[jnp.clip(move,0,len(tables.risk.toxic_candidates)-1)]&tox_follows
    dispatch=(kind!=public.MOVE)|(valid_move&(weather|ordinary|captured_sub|captured_tox))
    return jnp.all(~present|(known&(~required|captured)&(~direct|capability)&dispatch),axis=1)


def consume_pair_device(carry, core, frame, streams, publics, tables, catalog, *, active):
    """Actually consume both owners, then derive the conditional frame witness.

    No caller-supplied fold result, accepted flag, empty-span exemption or final
    state-to-event synthesis. Root must supply ORIGINAL host-guarded Core/frame
    and keep this exact carry; device copies cannot establish that origin.
    """
    from . import engine_retained_encoder_driver as driver
    if (type(carry) is not Carry or type(core) is not driver.Core or type(tables) is not Tables
            or type(frame) is not life.LifecycleResult or len(streams)!=2 or len(publics)!=2):
        raise ValueError('typed actual owning frame/carry and two original full owners required')
    tape = frame.callbacks; b = len(carry.stream_id)
    if (type(tape) is not life.LifecycleTape or type(core.engine) is not BattleState
            or type(frame.state) is not BattleState or type(core.cursor) is not life.Cursor
            or type(frame.cursor) is not life.Cursor or type(tape.cursor) is not life.Cursor
            or type(frame.before_registry) is not life.PublicArrivalSnapshot
            or type(frame.after_registry) is not life.PublicArrivalSnapshot
            or type(tables.accounting) is not execution.Tables or type(tables.risk) is not risks.Tables
            or type(tables.adapter) is not full.Tables):
        raise ValueError('exact original source frame/static types required')
    for x in (carry.binding,tables.binding,tables.engine_binding,tables.source_binding,tables.move_binding,
            tables.producer_binding,tables.encoder_binding,tables.direct_hit_binding,tables.substitute_binding,tables.toxic_binding):execution._shape(x,(8,),jnp.uint32)
    for x in (carry.stream_id,carry.frame,carry.refusal,core.frame_number):execution._shape(x,(b,),jnp.int32)
    execution._shape(carry.sequence,(b,2),jnp.int32); execution._shape(active,(b,),jnp.bool_)
    if type(tape.execution) is not execution.Context or type(tape.execution.risks) is not risks.Context:
        raise ValueError('actual accounting and kernel-risk contexts required')
    ledger = tape.execution.ledger; risk = tape.execution.risks.carry
    if (type(ledger) is not execution.Ledger or type(core.execution) is not execution.Ledger
            or type(risk) is not risks.Carry or type(core.risks) is not risks.Carry
            or type(tape.execution.tables) is not execution.Tables or type(tape.execution.risks.tables) is not risks.Tables):
        raise ValueError('exact accounting/risk carry and namespace types required')
    valid = execution._validate(ledger,tables.accounting)&execution._tape(ledger,tape,tables.accounting)
    risk_bound,_ = risks._bound(risk,tape,tables.risk)
    valid &= risk_bound
    valid &= jnp.all(carry.binding==tables.binding)&(carry.stream_id==tape.stream_id)
    valid &= (carry.frame>=0)&(carry.frame<2**31-1)&(ledger.frame==carry.frame+1)
    valid &= (ledger.frame==core.frame_number+1)&ledger.opened&core.admitted
    valid &= _same_ledger(core.execution,ledger)
    for a,z in zip(jax.tree.leaves(core.engine),jax.tree.leaves(frame.state),strict=True):valid &= _rows_equal(a,z)
    for a,z in zip(core.cursor,frame.cursor,strict=True):valid &= _rows_equal(a,z)
    for a,z in zip(tape.cursor,frame.cursor,strict=True):valid &= _rows_equal(a,z)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        valid &= jnp.all(getattr(tape,name)==getattr(tables,name))
    valid &= jnp.all(catalog.binding==tables.encoder_binding)
    for actual,expected in zip(tape.execution.tables,tables.accounting,strict=True):valid &= jnp.all(actual==expected)
    for actual,expected in zip(tape.execution.risks.tables,tables.risk,strict=True):valid &= jnp.all(actual==expected)
    for name in risk._fields:
        if name in ('binding','accounting_binding'):
            valid &= jnp.all(getattr(core.risks,name)==getattr(risk,name))
        else:valid &= _rows_equal(getattr(core.risks,name),getattr(risk,name))
    valid &= jnp.all(risk.binding==tables.risk.binding)&jnp.all(risk.accounting_binding==tables.accounting.inventory_binding)
    valid &= (risk.stream_id==carry.stream_id)&(risk.frame==ledger.frame)&risk.opened
    expected,observed,mode = _geometry(carry,tape)
    cuts = mode & _rows_equal(risk.counts,expected)&(risk.observed==observed)&(risk.risk==0)
    raw = _raw_complete(tape,tables)
    before = (tape.refusal&~public.UNINSTRUMENTED)==0
    before &= (ledger.refusal==0)&((ledger.upstream_refusal&~public.UNINSTRUMENTED)==0)&(carry.refusal==0)
    results=[]; ledgers=[]; drained=jnp.ones(b,bool); amounts=[]
    for owner,(stream,retained) in enumerate(zip(streams,publics)):
        if type(stream) is not full.StreamState or type(retained) is not v4_encoder.State:
            raise ValueError('original typed full stream and retained owner states required')
        execution._shape(stream.sequence,(b,),jnp.int32); execution._shape(stream.causal.owner,(b,),jnp.int32)
        identity = (stream.causal.owner==owner)&(stream.causal.cursor.stream_id==carry.stream_id)
        identity &= (stream.sequence==carry.sequence[:,owner])&(retained.cursor.sequence==stream.sequence)
        visibility = jnp.where(jnp.arange(2)[None,:]==owner,public_toxic.EXACT,public_toxic.PERCENTAGE)
        identity &= _rows_equal(retained.context.toxic.visibility,jnp.broadcast_to(visibility,(b,2)))
        account,result = execution.consume_frame_device(ledger,frame,stream,retained,active=active,
            tables=tables.accounting,adapter_tables=tables.adapter,catalog=catalog,
            weather_hp=None if tables.profile is None else (core.weather_profile,tables.profile),
            leftovers=tables.leftovers,status_residual=tables.status,leftovers_no_output=tables.no_output)
        projection=result.projection; chunks=projection.frames
        n=jnp.sum(chunks.valid,axis=(0,2),dtype=jnp.int32); amounts.append(n)
        used=jnp.any(chunks.valid,axis=2); indices=jnp.arange(used.shape[0])[:,None]
        last=jnp.max(jnp.where(used,indices,-1),axis=0)
        last_more=jnp.take_along_axis(chunks.more,jnp.maximum(last,0)[None,:],axis=0)[0]
        owner_good = identity&(last>=0)&~last_more&(projection.error==0)&projection.component_admitted
        owner_good &= (projection.state.sequence==stream.sequence+n)&(result.public.cursor.sequence==projection.state.sequence)
        owner_good &= (projection.state.causal.cursor.turn==frame.cursor.turn)&(projection.state.causal.cursor.phase==frame.cursor.phase)
        for a,z in zip(life.snapshot(projection.state.causal.cursor),frame.after_registry):owner_good &= _rows_equal(a,z)
        drained &= owner_good
        results.append(result); ledgers.append(account)
    joined,joint = execution.join_consumed_owners(*ledgers,tables=tables.accounting)
    refusal = carry.refusal|jnp.where(active&~valid,SOURCE|ORDER,0)|jnp.where(active&~cuts,CUTPOINTS,0)
    refusal |= jnp.where(active&~raw,RAW,0)|jnp.where(active&~joint,ACCOUNTING,0)
    refusal |= jnp.where(active&~drained,DRAIN,0)|jnp.where(active&~before,PRIOR,0)
    complete = active&(refusal==0)
    sequences=jnp.stack([x.projection.state.sequence for x in results],axis=1)
    updated=carry._replace(frame=jnp.where(complete,ledger.frame,carry.frame),
        sequence=jnp.where(complete[:,None],sequences,carry.sequence),refusal=refusal)
    # Refuse/rollback BOTH views atomically when ANY composed requirement fails.
    safe=[]
    for stream,retained,result in zip(streams,publics,results):
        p=result.projection
        p=p._replace(state=full._commit(complete,stream,p.state),
            error=jnp.where(active&~complete,int(full.inputs.Error.SOURCE),p.error),
            component_admitted=p.component_admitted&complete,
            frames=full.FrameChunks(jnp.where(complete[None,:,None,None],p.frames.words,0),
                p.frames.valid&complete[None,:,None],p.frames.more&complete[None,:]),
            chunk=p.chunk._replace(words=jnp.where(complete[:,None,None],p.chunk.words,0),
                valid=p.chunk.valid&complete[:,None],more=p.chunk.more&complete))
        safe.append(full.Consumed(p,full._commit_public(complete,retained,result.public)))
    report=Report(complete,refusal,expected,risk.counts,tape.words,tape.count,tape.event_turn,
        tape.event_phase,tape.event_generation,tape.switch_details,ledger.claimed,jnp.stack(amounts,axis=1),
        *[jnp.stack([jnp.moveaxis(getattr(result.projection.frames,name),1,0)
            for result in results],axis=1) for name in ('words','valid','more')])
    return Result(updated,execution.commit_rows(active,core.execution,joined),tuple(safe),report)


def commit_rows(active,before,after):
    """Retain original GLOBAL8/stream identity and lifetime refusal in Core."""
    if type(before) is not Carry or type(after) is not Carry:raise ValueError('typed witness carry required')
    if getattr(active,'ndim',None)!=1:raise ValueError('exact bool[B] active rows required')
    b=active.shape[0];execution._shape(active,(b,),jnp.bool_)
    for carry in (before,after):
        execution._shape(carry.binding,(8,),jnp.uint32)
        for name in ('stream_id','frame','refusal'):
            execution._shape(getattr(carry,name),(b,),jnp.int32)
        execution._shape(carry.sequence,(b,2),jnp.int32)
    same=jnp.all(before.binding==after.binding)&(before.stream_id==after.stream_id)
    use=active&same
    return before._replace(frame=jnp.where(use,after.frame,before.frame),
        sequence=jnp.where(use[:,None],after.sequence,before.sequence),
        refusal=before.refusal|jnp.where(active,after.refusal,0)|jnp.where(active&~same,SOURCE,0))
