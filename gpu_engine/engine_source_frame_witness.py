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
from . import engine_flash_fire_end_callbacks as flash_end
from . import engine_protect_callbacks as protect_callbacks
from . import engine_pinch_item_callbacks as pinch_callbacks
from . import engine_bulk_up_callbacks as bulk_up_callbacks
from . import engine_self_heal_callbacks as self_heal_callbacks
from . import engine_belly_drum_callbacks as belly_callbacks
from . import engine_sleep_blocked_callbacks as sleep_blocked_callbacks
from . import engine_encore_callbacks as encore_callbacks
from . import engine_leech_seed_callbacks as leech_callbacks
from . import engine_primary_plain_status_callbacks as plain_primary_callbacks
from . import engine_color_change_callbacks as color_callbacks
from . import engine_kernel_frame_risk as risks
from . import engine_outcome_diagnostics as outcome_diagnostics
from . import engine_secondary_status_callbacks as secondary_callbacks
from . import engine_substitute_damage_callbacks as substitute_damage
from . import engine_lifecycle_callbacks as life
from . import engine_full_v4_event_device as full
from . import engine_pressure_callbacks as pressure
from . import engine_trace_callbacks as trace
from . import engine_public_events as public, public_toxic, v4_encoder
from .state import BattleState
from . import engine_substitute_callbacks as substitute
from . import engine_toxic_callbacks as toxic
from . import engine_recoil_callbacks as recoil
from . import engine_rough_skin_callbacks as rough
from . import engine_no_damage_callbacks as no_damage
from . import engine_swords_dance_callbacks as swords_dance
from . import engine_spikes_callbacks as spikes
from . import engine_thunder_wave_callbacks as thunder_wave
from . import engine_intimidate_callbacks as intimidate
from . import engine_calm_mind_callbacks as calm_mind
from . import engine_sleep_powder_callbacks as powder

SCHEMA = 'pokezero.actual_source_frame_witness.v1'
SOURCE, CUTPOINTS, RAW, ACCOUNTING, DRAIN, ORDER, PRIOR = (1<<i for i in range(7))
_COMPILED = weakref.WeakKeyDictionary()
# Exact frozen source audited for this geometry. Root integration/capability
# changes require an explicit reviewed literal refresh, never auto-registration.
SOURCE_PINS = {'absorption.py': '2d684602eb9aa69bc470b215c188533b0e1c4d8495dd5b71ca2508dc38f20623', 'action_history.py': 'a47bdb2712f7eadaed12d91ede70be2ebe9f1c1ae3f70fe56495437997212407', 'baton.py': 'f08fb40dfc1e2d74e7a8fb2e42285c51a2ea4106484951580bf2aa9d23b29706', 'combat.py': '3e802d3fd2f488e9aa2a8ead663f935158c5fec1b7fb745034e54f21d38015af', 'critical.py': 'b66e6cf10c0d4df625b1c4f5e5db238af42052e556d7649a10f911745181401e', 'damage_moves.py': '227c58ab6042cbb0178dcc821e1be7dd2af99aae0f149febef995bdb7662abd1', 'effects.py': '0138f0e90811c4829666960a106d80ed2d59f6cd06a8ff83324d189465f73104', 'encore.py': '63da39bbd181a8c6371b3bca853322291a03674c60f535ad0eacdbe1a440e0dd', 'engine_bulk_up_callbacks.py': '77324e8acb7b99a9366acb7750432ba606d4f90101a265ce1bc13a0fa16e891e', 'engine_calm_mind_callbacks.py': '12cb052da4c381a6aa414f77383daa01ad7d54f312df88e60c25a87db2c2084d', 'engine_direct_hit_controls.py': 'e9e0a2bcbd8d87bd2bce886243e9e69c8992d021394d1595e0e0a67fbbbf2278', 'engine_direct_ko_callbacks.py': 'df10b12852552e962d9578ba2aeef2b032698a42f3efc8b28bd9676acfd8a161', 'engine_execution_completeness.py': 'ed7a6519610e97ab9d8aeae8b245c83910166283ace75fbc918d9247990a0b05', 'engine_flash_fire_end_callbacks.py': '54019a9edad3010a6d9535b31d206b880e796ab6e0a780fb2e0d445d38ece89f', 'engine_full_v4_event_device.py': 'fc1285f58ed3daf5f5b1264082631e8759e4acdb368752a61679bcf42a42a1cf', 'engine_intimidate_callbacks.py': '8004efa2b7121668e73d9bce898a27e32c6b45cace3eb85fe8403601e761facb', 'engine_item_callbacks.py': 'b7984d566834887545526f2ffadd50ec1f7b98360f4d751a761d6bf9eb69eabd', 'engine_kernel_frame_risk.py': '2c9b12813e4512fafb9c710df23b6960966e7eaeaef9628a10550d2967608120', 'engine_leftovers_callbacks.py': '7076825e29789d44650aca7ee56f2e5804a3c8e2b7c748624b6eb312c3783dca', 'engine_leftovers_no_output.py': 'e56bef0510ec31a69a70e48c3e15372bce7c6ebfd07d83e24f8eeec68c40e9c3', 'engine_lifecycle_callbacks.py': '2150a7d63073f9db362709a79b46532ef0eb32943f04d248625b814149f03f72', 'engine_lifecycle_event_device.py': '7f99e92d6bea9a859c1ca7f7565c7d97c88de3ba9e2af49d182cb6f2b0f582c7', 'engine_no_damage_callbacks.py': '1c13cc4e00d6b2c7a6e37076633963459b521fa908797b2b6c2618f5d197f28c', 'engine_outcome_diagnostics.py': '34acb9c1e32acf943788d478634119e2043d6321c0f51663d8cf4ef9231bfb40', 'engine_pressure_callbacks.py': '0de20779233ec90be004c0c604809ad5387edfc617cffaa7ce94d51f5ff875e9', 'engine_protect_callbacks.py': '71ebafbeb8219a9b1fd7bec374bb3596a3ac7d306bc6a63cd9c2b623395f3ab0', 'engine_public_events.py': '1d151cf0d0a0693862da4cfe0826aaae4515ab3c1b27ac0cbe86e34897e3ebd4', 'engine_recoil_callbacks.py': 'ae7183b5eb472d18dff62741c36b421a48db456d158aa45c02216e50cee5de5e', 'engine_request_source.py': '186cfc0a0ba48e786073003ce30db265c5659435e5b8c32e57ac54ab424ec5ce', 'engine_retained_encoder_driver.py': '9fce2eb6cce874ae3420462076ebbd2a629c40abdef7f03f0f0b40fc15423c19', 'engine_secondary_status_callbacks.py': '1039d607909f776c29496d3112bf3afa77d2a09f99d102dbf4363a657528f77e', 'engine_sleep_powder_callbacks.py': 'b003dacda10137098af9930d469be353fa026fde8399a4f84e221b764e897ce6', 'engine_spikes_callbacks.py': 'e0bc7a7fa2d0c9439de39d0bb94beaef74cbd5fbe35ecdd13266bd78153ee6c0', 'engine_status_callbacks.py': '369bde361dbc8cffd357dc92ae576686ee6bbea8e5a96fbd9a8fb08fe86ec23e', 'engine_status_residual_callbacks.py': 'e3dfad12350d21cc0f53fa6accb4c076a4ea4d5bf25f1cc739ff353b1ee9f153', 'engine_substitute_callbacks.py': '04e8d98dd1801a45a6da132d631f53eadb6da1d0f45622e0d1a23210a79be2a4', 'engine_substitute_damage_callbacks.py': 'd94b89b4849d248a47ea425c90d18f2c1507fa69735ed9f5d7083a2f3e15080a', 'engine_swords_dance_callbacks.py': '3f6f0908f5f7a99a8ce509e9d42f896cbfac99d68abee0ade1ba923fba19a83d', 'engine_terminal_callbacks.py': '934e20dc3b96392a40f303bec5ea0f7eb95b26744b935008526502acbf596781', 'engine_thunder_wave_callbacks.py': 'b514a48d380686c47da51c7bb36e4deb28acb8fc372859ca75325311405be985', 'engine_toxic_callbacks.py': '0ff5096482290590dd1647bcb8a7815f0348abaf1c1f8e77081a6aa665be17c1', 'engine_trace_callbacks.py': '318901fd1a358b83fcac10fa344748e40344cb9d1438a61d21d2d9fdb6fffbe2', 'engine_weather_callbacks.py': 'b5fee00fa633de1ef03062bffeab107c37ea15124b38a6ffa70c0820e5521d87', 'engine_weather_effect_callbacks.py': '66deb3debe354d8e637524bfcc4b94218863ba501e8da430797d1ae7ac7048f9', 'engine_weather_hp_segment.py': '6d812cd6f16accd5d3d82b193bef29d1170e7556c7c758c1ea343c6fc69cbbee', 'engine_weather_order_callbacks.py': 'fe21ff43c8e5687b2c1cf6a80596fcad466a93cf06ea57b4415cc8c069c3c6fe', 'engine_weather_progress_callbacks.py': 'a643b28bce10d328ca7359e1685e7b692e316e975c10dc15786cec95b242729e', 'engine_weather_speed_profile.py': '883bd8df30eb1a53ccd65594a53d8ef7ea66b7a3515c21d81443abd5a3e63832', 'entry.py': 'c17d3f2aaccf4c97ba1a40b52bcdfe31ae2cf5a60e2a66f2bcf4ee823dcdf3ed', 'focus_punch.py': '881bd17a84dc22576aa618256ada2b0f67bee29224dbb50508925b55e119585f', 'items.py': 'db32fbb3d0062aa20c965505055b8506c3158d5a408c258095fa947aa0c06c43', 'linked_traps.py': '743c01ee69509162e7829ae555b85f069f1c20ec9bd19cec0779d066894bade4', 'mechanics.py': 'e8476cc3e0029f3ed0bf82276312440e6dd8b340ffeb0c68cdd5611754eef661', 'phazing.py': '8f82d4894f03b39936de0e79b17f0b7715fdfe2f0edb6a7a42d44f1840671bee', 'pressure.py': '73eefb16678fedf1885a4e11cbc8954d15706592742caa6b0b673fb4d63d698c', 'pursuit.py': '2b01102bb3819791fc1edea0ba0fe0a0b913e8dfa5a8f6501f48f3fe260a8309', 'secondary.py': '7c93fe7d9af04d14493454bb8bedd8b3361afa1c0bcd22c1948760e5ea084c26', 'self_destruct.py': 'b3c9df0ea093eca7c0960b4c4ffd4277e289a0ded6f32f914466c0752fa9b739', 'sleep_talk.py': '968a78ddde87158b13f68dfcacc2158f07336d82c1be472d2283421e8583c154', 'solar_beam.py': '5463e103392e24df7cc560003740b7e425aba766a0d1bba90ff4428606849155', 'state.py': '2147cd0a13b055f3a61b38bec15424f6909b000e617c89213eb7f3a64d668424', 'sustain.py': '1ff481060f04384eae49da8bfeb663296c8efe582b0a7562fa3e3103ecbfd529', 'transition.py': 'b6a5f1757fa46346431e8995f7ee27f9a28fb7f572bbfd35050f9bf4993a065d', 'v5_hypothetical_profile.py': '8f5b621576270dfcc172c28bce4c3dacfb1928666caf9a748cde8d9f3343a53c', 'variable_moves.py': '7b06b3f0a26abc023036edb4ac0d4050e550522dd231d2d2a74cebf8f569e9b9', 'weather.py': '3c861027e16d27917fa47fa5cf5d6dc0c68d3a74af282108dfb94369d4db57e5', 'engine_primary_plain_status_callbacks.py': '21ea803337e60f119bf8282e7c92f07163b929aa280479d9523ca8f067a86c05', 'engine_rough_skin_callbacks.py': 'e67dc05da511c57787dabcdebcd9aee5449f636b9ef3816abe2d3615bd851d60', 'engine_color_change_callbacks.py': '7d64a2c94d3e67ee903a30e97ed476fd608e9b28ca34816adbf9a9ad27ed45d1', 'engine_pinch_item_callbacks.py': 'a7db284293bc675f78fc824d1c5ea41a7b86f1766ce7c8d73a07d805e4fa71b7', 'engine_self_heal_callbacks.py': 'bbc5cb35fa42430029d82232c0eaf754dc423120797e26c2c730ecbc9d1039d6', 'engine_belly_drum_callbacks.py': '191c5302a986cdb4eb985c784e59d249426fe5327fdae5a3b0216f887d8a9eae', 'engine_sleep_blocked_callbacks.py': 'b100f52c46345f90326f8132774807639580d5240a09ecf4829237109a8ba9d2', 'engine_encore_callbacks.py': 'c56e7ceb3a2ab901be2010b6565be1483477e8202431a6516d2e2a3ec2ee59b6', 'engine_leech_seed_callbacks.py': '68d64a376e52b840047e6794fc81e7e6ba7317beacb9530d3508bf2c0b7be065'}


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
    outcome_diagnostics: object = None    # literal private operands, data ONLY


class Result(NamedTuple):
    carry: Carry
    ledger: object
    owners: tuple                        # actual full.Consumed values, not input flags
    report: Report
    terminal_closeout: object = None  # Derived after genuine BOTH owner drains


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
    calm=runtime.get('calm_mind_compiled')
    if calm is not None and (runtime['calm_mind_mode']!='kernel-event' or record[9] is not calm):
        raise ValueError('same original registered Calm Mind/risk compiler edge required')
    swords=runtime.get('swords_dance_compiled')
    if swords is not None and (runtime['swords_dance_mode']!='kernel-event' or record[11] is not swords):
        raise ValueError('same original registered Swords Dance/risk compiler edge required')
    spike=runtime.get('spikes_compiled')
    if spike is not None and (runtime['spikes_mode']!='kernel-event' or record[10] is not spike):
        raise ValueError('same original registered Spikes/risk compiler edge required')
    thunder=runtime.get('thunder_wave_compiled')
    if thunder is not None and (runtime['thunder_wave_mode']!='kernel-event' or record[12] is not thunder):
        raise ValueError('same original registered Thunder Wave/risk compiler edge required')
    powder=runtime.get('sleep_powder_compiled')
    if powder is not None and (runtime['sleep_powder_mode']!='kernel-event' or record[13] is not powder):
        raise ValueError('same original registered Sleep Powder/risk compiler edge required')
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
    # These completed literal callback classes all have the original
    # acts&hit&~immune secondary-dispatch guard FALSE. No absence inference.
    secondary_moves -= jnp.sum(present&(raw==no_damage.TARGET_NO_DAMAGE),axis=1,dtype=jnp.int32)
    protect_attempts=present&(raw==protect_callbacks.PROTECT_ATTEMPT)
    secondary_moves=secondary_moves-jnp.sum(protect_attempts|(present&(raw==protect_callbacks.PROTECT_BLOCK)),axis=1,dtype=jnp.int32)+jnp.sum(protect_attempts&((tape.words[:,:,6]&protect_callbacks.SECONDARY)!=0),axis=1,dtype=jnp.int32)
    bulk_attempts=present&(raw==bulk_up_callbacks.BULK_UP)&(tape.words[:,:,4]==0)
    secondary_moves=secondary_moves-jnp.sum(bulk_attempts,axis=1,dtype=jnp.int32)+jnp.sum(bulk_attempts&(tape.words[:,:,7]==1),axis=1,dtype=jnp.int32)
    heal_attempts=present&(raw==self_heal_callbacks.SELF_HEAL)
    secondary_moves=secondary_moves-jnp.sum(heal_attempts,axis=1,dtype=jnp.int32)+jnp.sum(heal_attempts&(tape.words[:,:,9]==1),axis=1,dtype=jnp.int32)
    seed_attempts=present&(raw==leech_callbacks.LEECH)&(tape.words[:,:,5]==leech_callbacks.START)
    secondary_moves=secondary_moves-jnp.sum(seed_attempts,axis=1,dtype=jnp.int32)+jnp.sum(seed_attempts&((tape.words[:,:,9]&2)!=0),axis=1,dtype=jnp.int32)
    encore_attempts=present&(raw==encore_callbacks.ENCORE)&(tape.words[:,:,3]==encore_callbacks.START)
    secondary_moves=secondary_moves-jnp.sum(encore_attempts,axis=1,dtype=jnp.int32)+jnp.sum(encore_attempts&((tape.words[:,:,9]&4)!=0),axis=1,dtype=jnp.int32)
    belly_attempts=present&(raw==belly_callbacks.BELLY_DRUM)&(tape.words[:,:,4]==0)
    secondary_moves=secondary_moves-jnp.sum(belly_attempts,axis=1,dtype=jnp.int32)+jnp.sum(belly_attempts&(tape.words[:,:,9]==1),axis=1,dtype=jnp.int32)
    calm_attempts=present&(raw==calm_mind.CALM_MIND)&(tape.words[:,:,4]==2)
    secondary_moves=secondary_moves-jnp.sum(calm_attempts,axis=1,dtype=jnp.int32)+jnp.sum(
        calm_attempts&(tape.words[:,:,7]==1),axis=1,dtype=jnp.int32)
    swords_attempts=present&(raw==swords_dance.SWORDS_DANCE)
    secondary_moves=secondary_moves-jnp.sum(swords_attempts,axis=1,dtype=jnp.int32)+jnp.sum(
        swords_attempts&(tape.words[:,:,6]==1),axis=1,dtype=jnp.int32)
    spike_attempts=present&(raw==spikes.SPIKES)
    secondary_moves=secondary_moves-jnp.sum(spike_attempts,axis=1,dtype=jnp.int32)+jnp.sum(
        spike_attempts&(tape.words[:,:,7]==1),axis=1,dtype=jnp.int32)
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
    prefix = present&(raw==life.TURNLOOP_PREFIX)
    geometry &= jnp.where(initial,jnp.sum(prefix,axis=1)==0,
        (jnp.sum(prefix,axis=1)==1)&(raw[:,0]==life.TURNLOOP_PREFIX))
    return expected,mask,geometry


def _raw_complete(tape,tables):
    ledger = tape.execution.ledger; kind = tape.words[:,:,0]
    present = jnp.arange(kind.shape[1])[None,:] < tape.count[:,None]
    known = (kind==no_damage.TARGET_NO_DAMAGE)|(kind==no_damage.ABSORB_FACTS)|((kind>=1)&(kind<=17))|((kind>=20)&(kind<=23))|(kind==pressure.PRESSURE_START)|(kind==substitute.SUBSTITUTE_CREATED)|(kind==toxic.TOXIC_ATTEMPT)|(kind==trace.TRACE_COPY)|(kind==recoil.RECOIL_HP)|(kind==intimidate.INTIMIDATE)|(kind==calm_mind.CALM_MIND)|(kind==swords_dance.SWORDS_DANCE)|(kind==spikes.SPIKES)
    pinch=kind==pinch_callbacks.PINCH_ITEM
    known |= pinch
    prefix=kind==life.TURNLOOP_PREFIX
    known |= prefix|(kind==rough.ROUGH_SKIN_HP)
    secondary=(kind==secondary_callbacks.SECONDARY_STATUS)
    following=jnp.concatenate((kind[:,1:],jnp.zeros_like(kind[:,:1])),axis=1)
    secondary |= (kind==9)&(following==secondary_callbacks.SECONDARY_STATUS)
    color_change=kind==color_callbacks.COLOR_CHANGE
    known |= color_change|(kind==self_heal_callbacks.SELF_HEAL)
    known |= (kind==bulk_up_callbacks.BULK_UP)|secondary|(kind==flash_end.FLASH_FIRE_END)|(kind==protect_callbacks.PROTECT_ATTEMPT)|(kind==protect_callbacks.PROTECT_BLOCK)
    sub_damage=kind==substitute_damage.SUBSTITUTE_DAMAGE
    known |= sub_damage
    direct = (kind==no_damage.TARGET_NO_DAMAGE)|(kind==no_damage.ABSORB_FACTS)|(kind==public.MAIN_HP)|(kind==public.DIRECT_FAINT)|((kind>=21)&(kind<=23))
    required = direct|(kind==17)|(kind==20)|(kind==25)|(kind==26)|(kind==trace.TRACE_COPY)|(kind==recoil.RECOIL_HP)|(kind==intimidate.INTIMIDATE)|(kind==calm_mind.CALM_MIND)|(kind==swords_dance.SWORDS_DANCE)|(kind==spikes.SPIKES)
    required |= pinch
    known |= kind==sleep_blocked_callbacks.SLEEP_BLOCKED
    known |= kind==leech_callbacks.LEECH
    required |= kind==leech_callbacks.LEECH
    known |= kind==encore_callbacks.ENCORE
    required |= kind==encore_callbacks.ENCORE
    known |= kind==belly_callbacks.BELLY_DRUM
    required |= (kind==belly_callbacks.BELLY_DRUM)|(kind==sleep_blocked_callbacks.SLEEP_BLOCKED)
    required |= prefix|(kind==rough.ROUGH_SKIN_HP)
    required |= color_change|(kind==self_heal_callbacks.SELF_HEAL)
    required |= (kind==bulk_up_callbacks.BULK_UP)|secondary|(kind==flash_end.FLASH_FIRE_END)|sub_damage|(kind==protect_callbacks.PROTECT_ATTEMPT)|(kind==protect_callbacks.PROTECT_BLOCK)
    claim = ledger.claimed; slot = jnp.clip(claim-1,0,ledger.site.shape[1]-1)
    site = jnp.take_along_axis(ledger.site,slot,axis=1)
    expected = jnp.where(direct,int(execution.Site.DIRECT_HIT),jnp.where(kind==17,int(execution.Site.LEFTOVERS),
        jnp.where(kind==25,int(execution.Site.SUBSTITUTE_SELF),
            jnp.where(kind==26,jnp.where(tables.risk.thunder_wave_candidates[jnp.clip(tape.words[:,:,5],0,len(tables.risk.thunder_wave_candidates)-1)],int(execution.Site.PRIMARY_PARALYSIS),jnp.where(tables.risk.sleep_powder_candidates[jnp.clip(tape.words[:,:,5],0,len(tables.risk.sleep_powder_candidates)-1)],int(execution.Site.PRIMARY_SLEEP),int(execution.Site.TOXIC_ATTEMPT))),
                jnp.where(kind==trace.TRACE_COPY,int(execution.Site.TRACE_COPY),
                    jnp.where(kind==recoil.RECOIL_HP,int(execution.Site.RECOIL_HP),
                    jnp.where(kind==intimidate.INTIMIDATE,int(execution.Site.INTIMIDATE_INITIAL),jnp.where(kind==calm_mind.CALM_MIND,int(execution.Site.CALM_MIND),jnp.where(kind==swords_dance.SWORDS_DANCE,int(execution.Site.SWORDS_DANCE),jnp.where(kind==spikes.SPIKES,int(execution.Site.SPIKES_WRITE),int(execution.Site.STATUS_RESIDUAL)))))))))))
    expected=jnp.where(kind==rough.ROUGH_SKIN_HP,int(execution.Site.ROUGH_SKIN_HP),expected)
    expected=jnp.where((kind==intimidate.INTIMIDATE)&(tape.event_turn>0),int(execution.Site.INTIMIDATE_ATTEMPT),expected)
    expected=jnp.where(pinch,int(execution.Site.PINCH_ITEM),expected)
    expected=jnp.where(prefix,int(execution.Site.TURNLOOP_PREFIX),expected)
    expected=jnp.where(secondary,int(execution.Site.SECONDARY_STATUS),expected)
    expected=jnp.where(kind==flash_end.FLASH_FIRE_END,int(execution.Site.FLASH_FIRE_END),expected)
    expected=jnp.where(sub_damage,int(execution.Site.SUBSTITUTE_DAMAGE),expected)
    plain_selected=tables.risk.primary_plain_candidates[jnp.clip(tape.words[:,:,5],0,len(tables.risk.primary_plain_candidates)-1)]
    expected=jnp.where((kind==toxic.TOXIC_ATTEMPT)&plain_selected,int(execution.Site.PRIMARY_PLAIN_STATUS),expected)
    expected=jnp.where(color_change,int(execution.Site.COLOR_CHANGE),expected)
    expected=jnp.where(kind==self_heal_callbacks.SELF_HEAL,int(execution.Site.SELF_HEAL),expected)
    expected=jnp.where(kind==bulk_up_callbacks.BULK_UP,int(execution.Site.BULK_UP),expected)
    expected=jnp.where(kind==belly_callbacks.BELLY_DRUM,int(execution.Site.BELLY_DRUM),expected)
    expected=jnp.where(kind==leech_callbacks.LEECH,jnp.where(tape.words[:,:,5]==leech_callbacks.START,int(execution.Site.LEECH_SEED),jnp.where(tape.words[:,:,5]==leech_callbacks.DRAIN,int(execution.Site.LEECH_DRAIN),int(execution.Site.LEECH_HEAL))),expected)
    expected=jnp.where(kind==encore_callbacks.ENCORE,int(execution.Site.ENCORE),expected)
    expected=jnp.where(kind==sleep_blocked_callbacks.SLEEP_BLOCKED,int(execution.Site.SLEEP_BLOCKED),expected)
    expected=jnp.where(kind==protect_callbacks.PROTECT_ATTEMPT,int(execution.Site.PROTECT_ATTEMPT),expected)
    expected=jnp.where(kind==protect_callbacks.PROTECT_BLOCK,int(execution.Site.PROTECT_BLOCK),expected)
    captured = (claim>0)&(claim<=ledger.count[:,None])&(site==expected)
    captured &= (kind!=leech_callbacks.LEECH)|jnp.any(tables.risk.leech_binding!=0)
    captured &= (kind!=encore_callbacks.ENCORE)|jnp.any(tables.risk.encore_binding!=0)
    captured &= (kind!=sleep_blocked_callbacks.SLEEP_BLOCKED)|jnp.any(tables.risk.sleep_blocked_binding!=0)
    secondary_context=tape.execution.secondary_status
    secondary_bound=jnp.asarray(False) if secondary_context is None else jnp.all(secondary_context.data.binding==tables.direct_hit_binding)
    captured &= ~secondary|secondary_bound
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
    sub_damage_ticket=(entries&(ledger.site==int(execution.Site.SUBSTITUTE_DAMAGE))&(ledger.stage==2)
        &(ledger.no_output_kind==0)&(ledger.end==ledger.start+1))
    sub_damage_follows=jnp.any(sub_damage_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    ordinary=capability&candidate[jnp.clip(move,0,len(candidate)-1)]&(follows|sub_damage_follows)
    sub_binding=getattr(tables.risk,'substitute_binding',jnp.zeros(8,jnp.uint32))
    sub_capability=jnp.any(tables.substitute_binding!=0)&jnp.all(sub_binding==tables.substitute_binding)
    sub_ticket=(entries&(ledger.site==int(execution.Site.SUBSTITUTE_SELF))&(ledger.stage==2)
        &(ledger.no_output_kind==0)&(ledger.end==ledger.start+1))
    sub_follows=jnp.any(sub_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    sub_candidate=getattr(tables.risk,'substitute_candidates',jnp.zeros_like(tables.secondary_weather))
    captured_sub=sub_capability&sub_candidate[jnp.clip(move,0,len(sub_candidate)-1)]&sub_follows
    protect_cap=jnp.any(tables.risk.protect_binding!=0)
    protect_ticket=entries&((ledger.site==int(execution.Site.PROTECT_ATTEMPT))|(ledger.site==int(execution.Site.PROTECT_BLOCK)))&(ledger.stage==2)&(ledger.end==ledger.start+1)
    protect_follows=jnp.any(protect_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    captured_protect=protect_cap&protect_follows
    bulk_ticket=entries&(ledger.site==int(execution.Site.BULK_UP))&(ledger.stage==2)&(ledger.no_output_kind==0)&(ledger.end==ledger.start+2)
    captured_bulk=jnp.any(tables.risk.bulk_up_binding!=0)&jnp.any(bulk_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    heal_ticket=entries&(ledger.site==int(execution.Site.SELF_HEAL))&(ledger.stage==2)&(ledger.no_output_kind==0)&(ledger.end==ledger.start+1)
    captured_heal=jnp.any(tables.risk.self_heal_binding!=0)&tables.risk.self_heal_candidates[jnp.clip(move,0,len(tables.risk.self_heal_candidates)-1)]&jnp.any(heal_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    belly_ticket=entries&(ledger.site==int(execution.Site.BELLY_DRUM))&(ledger.stage==2)&(ledger.no_output_kind==0)&(ledger.end==ledger.start+2)
    seed_ticket=entries&(ledger.site==int(execution.Site.LEECH_SEED))&(ledger.stage==2)&(ledger.no_output_kind==0)&(ledger.end==ledger.start+1)
    captured_seed=jnp.any(tables.risk.leech_binding!=0)&(move==tables.risk.leech_move)&jnp.any(seed_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    encore_ticket=entries&(ledger.site==int(execution.Site.ENCORE))&(ledger.stage==2)&(ledger.no_output_kind==0)&(ledger.end==ledger.start+1)
    captured_encore=jnp.any(tables.risk.encore_binding!=0)&(move==tables.risk.encore_move)&jnp.any(encore_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    captured_belly=jnp.any(tables.risk.belly_drum_binding!=0)&jnp.any(belly_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    plain_ticket=entries&(ledger.site==int(execution.Site.PRIMARY_PLAIN_STATUS))&(ledger.stage==2)&(ledger.no_output_kind==0)&(ledger.end==ledger.start+2)
    captured_plain=jnp.any(tables.risk.primary_plain_binding!=0)&jnp.any(plain_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    # Actual geometry ALSO checks every secondary guard count. Unlike weather,
    # this is no universal no-immunity assertion from a static move name.
    tox_cap=jnp.any(tables.toxic_binding!=0)&jnp.all(tables.risk.toxic_binding==tables.toxic_binding)
    tox_ticket=(entries&(ledger.site==int(execution.Site.TOXIC_ATTEMPT))&(ledger.stage==2)
        &(ledger.no_output_kind==0)&((ledger.end==ledger.start+1)|(ledger.end==ledger.start+2)))
    tox_follows=jnp.any(tox_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    captured_tox=tox_cap&tables.risk.toxic_candidates[jnp.clip(move,0,len(tables.risk.toxic_candidates)-1)]&tox_follows
    calm_cap=jnp.any(tables.risk.calm_mind_binding!=0)
    calm_ticket=(entries&(ledger.site==int(execution.Site.CALM_MIND))&(ledger.stage==2)
        &(ledger.no_output_kind==0)&(ledger.end==ledger.start+2))
    calm_follows=jnp.any(calm_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    captured_calm=calm_cap&tables.risk.calm_mind_candidates[jnp.clip(move,0,len(tables.risk.calm_mind_candidates)-1)]&calm_follows
    swords_cap=jnp.any(tables.risk.swords_dance_binding!=0)
    swords_ticket=(entries&(ledger.site==int(execution.Site.SWORDS_DANCE))&(ledger.stage==2)
        &(ledger.no_output_kind==0)&(ledger.end==ledger.start+1))
    swords_follows=jnp.any(swords_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    captured_swords=swords_cap&tables.risk.swords_dance_candidates[jnp.clip(move,0,len(tables.risk.swords_dance_candidates)-1)]&swords_follows
    spike_cap=jnp.any(tables.risk.spikes_binding!=0)
    spike_ticket=(entries&(ledger.site==int(execution.Site.SPIKES_WRITE))&(ledger.stage==2)
        &(ledger.no_output_kind==0)&(ledger.end==ledger.start+1))
    spike_follows=jnp.any(spike_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    captured_spike=spike_cap&tables.risk.spikes_candidates[jnp.clip(move,0,len(tables.risk.spikes_candidates)-1)]&spike_follows
    thunder_cap=jnp.any(tables.risk.thunder_wave_binding!=0)
    thunder_ticket=(entries&(ledger.site==int(execution.Site.PRIMARY_PARALYSIS))&(ledger.stage==2)
        &(ledger.no_output_kind==0)&(ledger.end==ledger.start+2))
    thunder_follows=jnp.any(thunder_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    captured_thunder=thunder_cap&tables.risk.thunder_wave_candidates[jnp.clip(move,0,len(tables.risk.thunder_wave_candidates)-1)]&thunder_follows
    powder_cap=jnp.any(tables.risk.sleep_powder_binding!=0)
    powder_at=jnp.clip(ledger.start,0,tape.words.shape[1]-1)
    powder_raw=tape.words[jnp.arange(tape.count.shape[0])[:,None],powder_at]
    powder_move=jnp.clip(powder_raw[:,:,5],0,len(tables.risk.primary_sleep_miss_candidates)-1)
    powder_miss=powder.supported_miss_raw(powder_raw.reshape((-1,10))).reshape(ledger.site.shape)&tables.risk.primary_sleep_miss_candidates[powder_move]
    powder_ticket=(entries&(ledger.site==int(execution.Site.PRIMARY_SLEEP))&(ledger.stage==2)
        &(ledger.no_output_kind==0)&((ledger.end==ledger.start+2)|((ledger.end==ledger.start+1)&powder_miss)))
    powder_follows=jnp.any(powder_ticket[:,None,:]&(ledger.start[:,None,:]==index+1),axis=2)
    captured_powder=powder_cap&tables.risk.sleep_powder_candidates[jnp.clip(move,0,len(tables.risk.sleep_powder_candidates)-1)]&powder_follows
    dispatch=(kind!=public.MOVE)|(valid_move&(weather|ordinary|captured_seed|captured_encore|captured_plain|captured_bulk|captured_heal|captured_belly|captured_protect|captured_sub|captured_tox|captured_calm|captured_swords|captured_spike|captured_thunder|captured_powder))
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
    from . import engine_terminal_callbacks as terminal
    terminal_source=terminal.matches_final_device(tape,frame.state)
    valid &= ~frame.state.done|terminal_source
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
            for result in results],axis=1) for name in ('words','valid','more')],
        tape.execution.diagnostics if tape.execution.diagnostics is not None else outcome_diagnostics.empty(b))
    return Result(updated,execution.commit_rows(active,core.execution,joined),tuple(safe),report,
        complete & frame.state.done & terminal_source)


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
