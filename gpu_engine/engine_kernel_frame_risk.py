"""Declared actual-kernel cutpoint risks, NOT source/consumer completeness.

First proposed domain is fresh construction + weather-table actions and their
already registered weather/Leftovers transports. Selection reads ORIGINAL
immutable kernel columns, never a move-name roster or original simulator hooks.
Other literal branch observations stick even after HP/effects revert.

The optional actual kernel callers record these declared cutpoints. Compilation
binds identity ONLY. The owning constructor/Core must retain this SAME carry
and compose genuine accounting and BOTH-owner drains. Zero risk,
all observed bits or a copied carry cannot establish exhaustive source coverage.
There is deliberately no complete/admitted/exhaustive/qualification boolean API.
"""
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from typing import NamedTuple
import weakref
import jax
import jax.numpy as jnp
import numpy as np
from . import engine_public_events as public, engine_execution_completeness as accounting
from . import public_mon_state, mechanics
from .state import BattleState


class Branch(IntEnum):
    CONSTRUCTOR=1
    PRE_MOVE=2
    SELECTED_MOVE=3
    HIT_OUTCOME=4
    FRAME_END=5
    SUBSTITUTE=6
    SELF_HEAL=7
    PAIN_SPLIT=8
    BELLY_DRUM=9
    SELF_BOOST=10
    HIT_SPA_DROP=11
    SECONDARY_STATS=12
    SECONDARY_FLINCH=13
    SEED_START=14
    WISH_START=15
    SPIN_CLEAR=16
    LINK_TRAP=17
    ENCORE=18
    SPIKES=19
    HAZE=20
    SELF_DESTRUCT=21
    RECOIL=22
    CONFUSION_HP=23
    ENTRY_CONFUSION_CURE=24
    PRIMARY_STATUS=25
    HEAL_BELL=26


NON_WEATHER, SELF_EFFECT, BOOST, FIELD_VOLATILE, TARGET_OUTCOME, SELF_HP, BLOCKED_ACTION, TERMINAL, CONSTRUCTOR_STATE, NAMESPACE, ORDER, OVERFLOW, UPSTREAM = (1<<n for n in range(13))
SCHEMA='pokezero.actual_kernel.frame_risk.declared.v1'
UNKNOWN_KERNEL_RISK=1024
_COMPILED=weakref.WeakKeyDictionary()


class Tables(NamedTuple):
    binding: object
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    accounting_binding: object
    selected_risk: object
    branch_risk: object
    direct_hit_binding: object       # GLOBAL8; zero means weather-only mode
    direct_candidates: object        # [M] immutable original table capability
    seismic_toss_move: object        # scalar exact original fixed-level source row
    substitute_binding: object      # GLOBAL8, original selected self-write helper
    substitute_candidates: object   # [M], prospective only; actual Site11 REQUIRED
    toxic_binding: object
    toxic_candidates: object
    recoil_binding: object  # GLOBAL8
    recoil_candidates: object
    calm_mind_binding: object
    calm_mind_candidates: object
    swords_dance_binding: object
    swords_dance_candidates: object
    spikes_binding: object
    spikes_candidates: object
    direct_ko_candidates: object
    thunder_wave_binding: object
    sleep_powder_binding: object        # GLOBAL[8]
    sleep_powder_candidates: object     # [M] canonical original source only
    thunder_wave_candidates: object
    no_damage_binding: object          # GLOBAL8 original direct helper capability
    primary_plain_binding: object
    primary_plain_candidates: object
    primary_plain_status: object
    pinch_binding: object
    pinch_stats: object
    bulk_up_binding: object
    bulk_up_candidates: object
    protect_binding: object
    protect_candidates: object
    protect_block_candidates: object
    color_change_binding: object
    color_change_abilities: object
    color_change_candidates: object
    color_change_types: object
    color_change_dynamic: object
    primary_sleep_miss_candidates: object
    self_heal_binding: object
    self_heal_candidates: object
    belly_drum_binding: object
    belly_drum_candidates: object
    sleep_blocked_binding: object
    sleep_blocked_moves: object
    encore_binding: object
    encore_move: object
    leech_binding: object
    leech_move: object
    self_move_binding: object
    self_move_ids: object
    toxic_substitute_failure: object
    substitute_failure: object
    toxic_existing_status: object
    self_heal_family: object
    recoil_ko: object
    recoil_rock_head_source: object
    recoil_rock_head: object
    self_boost_binding: object
    self_boost_stats: object
    self_boost_deltas: object
    self_boost_lengths: object
    heal_bell_binding: object
    heal_bell_move: object
    heal_bell_soundproof: object
    baton_binding: object
    baton_move: object
    baton_abilities: object
    baton_items: object
    self_destruct_binding: object
    self_destruct_candidates: object
    self_destruct_powers: object
    self_destruct_damp: object
    self_destruct_abilities: object
    self_destruct_items: object
    before_move_blocked_binding: object
    before_move_blocked_moves: object
    before_move_blocked_abilities: object
    before_move_blocked_items: object
    thunder_wave_substitute_failure: object
    sleep_talk_toxic_child: object
    levitate_immunity: object
    recoil_immune_source: object
    recoil_immune_ghost_type: object
    recoil_immune_wonder_guard: object
    sleep_talk_rest_failure: object
    self_destruct_own_tempo: object
    baton_outgoing_dormant_abilities: object = None
    thunder_wave_ground_immunity: object = None
    baton_outgoing_insomnia_status: object = None
    primary_plain_wisp_miss: object = None
    whirlwind_binding: object = None
    whirlwind_move: object = None
    whirlwind_abilities: object = None
    whirlwind_items: object = None
    whirlwind_source: object = None
    focus_binding: object = None
    focus_move: object = None
    focus_abilities: object = None
    baton_natural_cure: object = None


@dataclass(frozen=True,eq=False)
class Compiled:
    device: Tables


class Carry(NamedTuple):
    binding: object                    # GLOBAL8, including B==8
    accounting_binding: object         # GLOBAL8
    stream_id: object                  # all remaining fields lead with B
    frame: object
    opened: object
    risk: object
    observed: object                   # bitset of declared cutpoints CALLED
    counts: object                     # [B,24], not public-event counts


class Context(NamedTuple):
    """PRIVATE execution.Context.risks carry; owning Core retains it explicitly."""
    carry: Carry
    tables: Tables


class Outcome(NamedTuple):
    power: object
    critical: object
    hit: object
    immune: object
    numerator: object
    denominator: object
    absorbed: object
    substitute: object
    protected: object
    enduring: object
    recoil: object
    explodes: object


class CapturedDirectHit(NamedTuple):
    """PRIVATE SAME-call source values, not caller attestation or credit.

    The actual kernel passes its original registered context, fresh Site10
    ticket and literal write. The owning driver must establish frame origin;
    hot device arrays/copies cannot attest that origin on their own.
    """
    context: object
    ticket: object
    actor: object
    actor_physical: object
    target_physical: object
    move: object
    write: object


def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked kernel risk boundary cannot be traced')


def compile_risks(engine_catalog,accounting_compiled,*,direct_hit_compiled=None,substitute_compiled=None,toxic_compiled=None,recoil_compiled=None,calm_mind_compiled=None,swords_dance_compiled=None,spikes_compiled=None,thunder_wave_compiled=None,sleep_powder_compiled=None):
    _host();record=accounting._checked(accounting_compiled)
    if record[2].causal.projector.catalog is not engine_catalog:raise ValueError('same original accounting/kernel catalog required')
    public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    table=engine_catalog.engine_table;m=len(table.power);risk=np.zeros(m,np.int32)
    weather=np.asarray(table.weather.move_weather) if table.weather is not None else np.zeros(m,np.int32)
    focus_binding=np.zeros(8,np.uint32);focus_move=0;focus_abilities=np.zeros(0,bool)
    no_damage_binding=np.zeros(8,np.uint32);levitate_immunity=False
    plain_wisp=0;plain_binding=np.zeros(8,np.uint32);plain_candidates=np.zeros(m,bool);plain_status=np.zeros(m,np.int32)
    pinch_binding=np.zeros(8,np.uint32);pinch_stats=np.full(len(table.items.pinch_stat) if table.items is not None else 1,-1,np.int32)
    bulk_binding=np.zeros(8,np.uint32);bulk_candidates=np.zeros(m,bool)
    self_boost_binding=np.zeros(8,np.uint32);self_boost_stats=np.full((m,2),-1,np.int32);self_boost_deltas=np.zeros((m,2),np.int32);self_boost_lengths=np.zeros(m,np.int32)
    belly_binding=np.zeros(8,np.uint32);belly_candidates=np.zeros(m,bool)
    sleep_block_binding=np.zeros(8,np.uint32);sleep_block_moves=np.zeros(m,bool)
    encore_binding=np.zeros(8,np.uint32);encore_move=0
    baton_natural_cure=np.zeros(len(engine_catalog.engine_table.entry.natural_cure),bool)if engine_catalog.engine_table.entry is not None else np.zeros(1,bool);baton_binding=np.zeros(8,np.uint32);baton_move=0;baton_abilities=np.zeros(len(engine_catalog.engine_binding),bool);baton_items=np.zeros(1,bool);baton_outgoing_dormant_abilities=np.zeros_like(baton_abilities);baton_outgoing_insomnia_status=np.zeros(0,np.int32)
    self_move_binding=np.zeros(8,np.uint32);self_move_ids=np.zeros(3,np.int32);self_move_candidates=np.zeros(m,bool);sleep_talk_rest_failure=False
    leech_binding=np.zeros(8,np.uint32);leech_move=0
    protect_binding=np.zeros(8,np.uint32);protect_candidates=np.zeros(m,bool);protect_blocks=np.zeros(m,bool)
    direct_binding=np.zeros(8,np.uint32);candidate=np.zeros(m,bool);seismic=0;ko_candidates=np.zeros(m,bool)
    color_binding=np.zeros(8,np.uint32);color_abilities=np.zeros(len(table.secondary.color_change) if table.secondary is not None else 1,bool);color_candidates=np.zeros(m,bool);color_types=np.zeros(m,np.int32);color_dynamic=np.zeros(m,bool)
    heal_binding=np.zeros(8,np.uint32);heal_candidates=np.zeros(m,bool);heal_family=np.zeros(m,np.int32)
    heal_bell_binding=np.zeros(8,np.uint32);heal_bell_move=0;heal_bell_candidates=np.zeros(m,bool);heal_bell_soundproof=np.zeros(len(table.effects.soundproof)if table.effects is not None else 1,bool)
    self_destruct_binding=np.zeros(8,np.uint32);self_destruct_candidates=np.zeros(m,bool);self_destruct_powers=np.zeros(m,np.int32);self_destruct_damp=np.zeros(1,bool);self_destruct_abilities=np.zeros(1,bool);self_destruct_items=np.zeros(1,bool);self_destruct_own_tempo=np.zeros(1,bool)
    before_move_blocked_binding=np.zeros(8,np.uint32);before_move_blocked_moves=np.zeros((2,m),bool);before_move_blocked_abilities=np.zeros(1,bool);before_move_blocked_items=np.zeros(1,bool)
    if direct_hit_compiled is not None:
        from . import engine_direct_hit_controls as direct
        direct_data=direct.check_compiled(direct_hit_compiled)
        if direct._COMPILED[direct_hit_compiled][2] is not engine_catalog:
            raise ValueError('same original direct-hit/kernel catalog required')
        if int(direct_data.move_count)!=m:raise ValueError('exact direct-hit move namespace required')
        direct_binding=np.asarray(direct_data.binding,np.uint32)
        if direct_data.focus_punch is not None:
            from . import engine_focus_punch_callbacks as focus_callbacks
            fd=direct_data.focus_punch;focus_callbacks.shape(fd);focus_binding=np.asarray(fd.binding);focus_move=int(fd.move);focus_abilities=np.asarray(fd.abilities)
        no_damage_binding=np.asarray(direct_data.no_damage.binding,np.uint32)
        from . import engine_no_damage_callbacks as no_damage
        no_damage._shape(direct_data.no_damage);levitate_immunity=bool(direct_data.no_damage.levitate_source)
        ko_candidates=np.asarray(direct_data.ko_candidates,bool)
        if direct_data.self_destruct is not None:
            from . import engine_self_destruct_callbacks as sd
            data=direct_data.self_destruct;sd.shape(data);self_destruct_binding=np.asarray(data.binding);self_destruct_candidates=np.asarray(data.candidates);self_destruct_powers=np.asarray(data.powers);self_destruct_damp=np.asarray(data.damp);self_destruct_abilities=np.asarray(data.quiet_abilities);self_destruct_items=np.asarray(data.quiet_items);self_destruct_own_tempo=np.asarray(data.own_tempo)
        if direct_data.heal_bell is not None:
            from . import engine_heal_bell_callbacks as heal_bell_callbacks
            data=direct_data.heal_bell;heal_bell_callbacks.shape(data);heal_bell_move=int(data.move)
            heal_bell_binding=np.asarray(data.binding,np.uint32);heal_bell_candidates[heal_bell_move]=True;heal_bell_soundproof=np.asarray(data.installed.soundproof,bool)
        if direct_data.primary_plain_status is not None:
            from . import engine_primary_plain_status_callbacks as plain_primary
            plain_primary.shape(direct_data.primary_plain_status);plain_binding=np.asarray(direct_data.primary_plain_status.binding,np.uint32);plain_candidates=np.asarray(direct_data.primary_plain_status.candidates).copy();plain_status=np.asarray(direct_data.primary_plain_status.status).copy();plain_wisp=0 if direct_data.primary_plain_status.wisp_miss is None else int(direct_data.primary_plain_status.wisp_miss)
        if direct_data.color_change is not None:
            from . import engine_color_change_callbacks as color
            cd=direct_data.color_change;color.shape(cd)
            color_binding=np.asarray(cd.binding);color_abilities=np.asarray(cd.abilities);color_candidates=np.asarray(cd.candidates);color_types=np.asarray(cd.move_types);color_dynamic=np.asarray(cd.dynamic_type)
        if direct_data.pinch_item is not None:
            from . import engine_pinch_item_callbacks as pinch
            pinch.shape(direct_data.pinch_item);pinch_binding=np.asarray(direct_data.pinch_item.binding,np.uint32);pinch_stats=np.asarray(direct_data.pinch_item.stats).copy()
        if direct_data.leech_seed is not None:
            from . import engine_leech_seed_callbacks as leech_callbacks
            ld=direct_data.leech_seed;leech_callbacks.shape(ld);leech_binding=np.asarray(ld.binding);leech_move=int(ld.move_id)
        if direct_data.self_move is not None:
            from . import engine_self_move_callbacks as self_move_callbacks
            sm=direct_data.self_move;self_move_callbacks.shape(sm);self_move_binding=np.asarray(sm.binding);self_move_ids=np.asarray(sm.move_ids);self_move_candidates=np.asarray(sm.candidates)
            sleep_talk_rest_failure=(direct._COMPILED[direct_hit_compiled][3] is not None and sm.talking is not None and int(sm.move_ids[0])>0 and bool(sm.talking_source[int(sm.move_ids[0])]))
        if direct_data.encore is not None:
            from . import engine_encore_callbacks as encore_callbacks
            ed=direct_data.encore;encore_callbacks.shape(ed);encore_binding=np.asarray(ed.binding);encore_move=int(ed.move_id)
        if direct_data.sleep_blocked is not None:
            from . import engine_sleep_blocked_callbacks as sleep_blocked
            sd=direct_data.sleep_blocked;sleep_blocked.shape(sd);sleep_block_binding=np.asarray(sd.binding);sleep_block_moves=np.asarray(sd.moves)
        if direct_data.belly_drum is not None:
            from . import engine_belly_drum_callbacks as belly
            belly.shape(direct_data.belly_drum);belly_binding=np.asarray(direct_data.belly_drum.binding,np.uint32);belly_candidates=np.asarray(direct_data.belly_drum.candidates).copy()
        if direct_data.before_move_blocked is not None:
            from . import engine_before_move_blocked_callbacks as bm
            data=direct_data.before_move_blocked;bm.shape(data);before_move_blocked_binding=np.asarray(data.binding);before_move_blocked_moves=np.asarray(data.moves);before_move_blocked_abilities=np.asarray(data.abilities);before_move_blocked_items=np.asarray(data.items)
        if direct_data.baton_pass is not None:
            from . import engine_baton_pass_callbacks as baton_callbacks
            bd=direct_data.baton_pass;baton_callbacks.shape(bd);baton_binding=np.asarray(bd.binding);baton_move=int(bd.move_id);baton_abilities=np.asarray(bd.quiet_abilities);baton_items=np.asarray(bd.quiet_items);baton_outgoing_dormant_abilities=np.zeros_like(baton_abilities)if bd.outgoing_dormant_abilities is None else np.asarray(bd.outgoing_dormant_abilities);baton_outgoing_insomnia_status=np.zeros(0,np.int32)if bd.outgoing_insomnia_status is None else np.asarray(bd.outgoing_insomnia_status);baton_natural_cure=np.zeros_like(baton_abilities)if bd.natural_cure is None else np.asarray(bd.natural_cure)
        if direct_data.self_boost is not None:
            from . import engine_self_boost_callbacks as self_boost
            sd=direct_data.self_boost;self_boost.shape(sd)
            self_boost_binding=np.asarray(sd.binding);self_boost_stats=np.asarray(sd.stats);self_boost_deltas=np.asarray(sd.deltas);self_boost_lengths=np.asarray(sd.lengths)
        if direct_data.bulk_up is not None:
            from . import engine_bulk_up_callbacks as bulk_up
            bulk_up.shape(direct_data.bulk_up);bulk_binding=np.asarray(direct_data.bulk_up.binding,np.uint32);bulk_candidates=np.asarray(direct_data.bulk_up.candidates).copy()
        if direct_data.self_heal is not None:
            from . import engine_self_heal_callbacks as self_heal
            self_heal.shape(direct_data.self_heal);heal_binding=np.asarray(direct_data.self_heal.binding,np.uint32);heal_candidates=np.asarray(direct_data.self_heal.candidates).copy();heal_family=np.asarray(direct_data.self_heal.family,np.int32).copy()
        if direct_data.protect is not None:
            from . import engine_protect_callbacks as protect_callbacks
            protect_callbacks.shape(direct_data.protect)
            protect_binding=np.asarray(direct_data.protect.binding,np.uint32)
            protect_candidates=np.asarray(direct_data.protect.candidates).copy()
            protect_blocks=np.asarray(direct_data.protect.block_candidates).copy()
        # Candidate only: every actual positive outcome additionally requires
        # its SAME captured Site10 span. Other static/attempt risk bits remain.
        positive=np.asarray(table.power)>0
        if table.variable_moves is not None:
            # These original resolver families have positive power even when
            # the source move's static base power is zero. Actual same-call
            # direct-hit tickets and all other risk guards still apply.
            for name in ('returns','frustration','hidden_power'):
                variable=np.asarray(getattr(table.variable_moves,name))
                if variable.shape!=(m,):raise ValueError('exact selected variable move columns required')
                positive |= variable
        seismic=int(direct_data.seismic_toss_move)
        if seismic>0:positive[seismic]=True
        candidate=positive&(np.asarray(engine_catalog.ids)>0)&(np.asarray(engine_catalog.targets)==1)
    substitute_binding=np.zeros(8,np.uint32);substitute_candidate=np.zeros(m,bool);substitute_failure=False
    if substitute_compiled is not None:
        from . import engine_substitute_callbacks as substitute
        data=substitute.check_compiled(substitute_compiled)
        if substitute._COMPILED[substitute_compiled][2] is not engine_catalog:
            raise ValueError('same original Substitute/kernel catalog required')
        move=int(data.move_id);substitute_failure=bool(data.failure)
        if not 0<move<m:raise ValueError('canonical Substitute move bounds required')
        substitute_binding=np.asarray(data.binding,np.uint32);substitute_candidate[move]=True
    toxic_binding=np.zeros(8,np.uint32);toxic_candidate=np.zeros(m,bool);toxic_substitute_failure=False;toxic_existing_status=False;sleep_talk_toxic_child=False
    if toxic_compiled is not None:
        from . import engine_toxic_callbacks as toxic
        data=toxic.check_compiled(toxic_compiled)
        toxic_substitute_failure=bool(data.substitute_failure);toxic_existing_status=bool(data.existing_status)
        if toxic._COMPILED[toxic_compiled][2] is not engine_catalog:raise ValueError('same original Toxic/kernel catalog required')
        move=int(data.move_id)
        if not 0<move<m:raise ValueError('canonical Toxic move bounds required')
        toxic_binding=np.asarray(data.binding,np.uint32);toxic_candidate[move]=True
        if direct_hit_compiled is not None:
            # Explicit host identity of the genuine original rules, not a
            # borrowed success/other-family mask or a private move waiver.
            from . import engine_direct_hit_controls as direct
            original_rules=direct._COMPILED[direct_hit_compiled][3]
            sleep_talk_toxic_child=(original_rules is not None and toxic._COMPILED[toxic_compiled][3] is original_rules
                and direct_data.self_move is not None and direct_data.self_move.talking is not None)
    whirlwind_binding=np.zeros(8,np.uint32);whirlwind_move=0;whirlwind_abilities=np.zeros((3,1),bool);whirlwind_items=np.zeros(1,bool);whirlwind_source=np.zeros(0,np.int32)
    if direct_hit_compiled is not None and direct_data.whirlwind is not None:
        from . import engine_whirlwind_callbacks as whirlwind
        wd=direct_data.whirlwind;whirlwind.shape(wd);whirlwind_binding=np.asarray(wd.binding);whirlwind_move=int(wd.move);whirlwind_abilities=np.asarray(wd.abilities);whirlwind_items=np.asarray(wd.items);whirlwind_source=np.asarray(whirlwind.vector(wd))
    recoil_binding=np.zeros(8,np.uint32);recoil_candidate=np.zeros(m,bool);recoil_ko=False;recoil_rock_head_source=False;recoil_rock_head=np.zeros(1,bool);recoil_immune_source=False;recoil_immune_ghost_type=0;recoil_immune_wonder_guard=0
    if recoil_compiled is not None:
        from . import engine_recoil_callbacks as recoil
        data=recoil.check_compiled(recoil_compiled)
        if recoil._COMPILED[recoil_compiled][2] is not engine_catalog or direct_hit_compiled is None:
            raise ValueError('same original Double-Edge/direct-hit/kernel catalog required')
        move=int(data.move_id)
        if not 0<move<m or not candidate[move]:raise ValueError('original direct-capable Double-Edge row required')
        recoil_binding=np.asarray(data.binding,np.uint32);recoil_candidate[move]=True;recoil_ko=bool(data.ko_source);recoil_rock_head_source=bool(data.rock_head_source);recoil_rock_head=np.asarray(data.rock_head,bool)
        accounting._shape(data.immune_source,(),jnp.bool_)
        original_rules=direct._COMPILED[direct_hit_compiled][3]
        recoil_immune_source=(bool(data.immune_source) and original_rules is not None
            and recoil._COMPILED[recoil_compiled][3] is original_rules
            and bool(direct_data.no_damage.recoil_immune_source) and int(direct_data.no_damage.recoil_immune_move)==move)
        recoil_immune_ghost_type=int(direct_data.no_damage.ghost_type) if recoil_immune_source else 0
        recoil_immune_wonder_guard=int(direct_data.no_damage.recoil_immune_wonder_guard) if recoil_immune_source else 0
    calm_binding=np.zeros(8,np.uint32);calm_candidate=np.zeros(m,bool)
    if calm_mind_compiled is not None:
        from . import engine_calm_mind_callbacks as calm
        data=calm.check_compiled(calm_mind_compiled)
        if calm._COMPILED[calm_mind_compiled][2] is not engine_catalog:raise ValueError('same original Calm Mind/kernel catalog required')
        move=int(data.move_id)
        if not 0<move<m:raise ValueError('canonical Calm Mind row required')
        calm_binding=np.asarray(data.binding,np.uint32);calm_candidate[move]=True
    swords_binding=np.zeros(8,np.uint32);swords_candidate=np.zeros(m,bool)
    if swords_dance_compiled is not None:
        from . import engine_swords_dance_callbacks as swords
        data=swords.check_compiled(swords_dance_compiled)
        if swords._COMPILED[swords_dance_compiled][2] is not engine_catalog:raise ValueError('same original Swords Dance/kernel catalog required')
        move=int(data.move_id)
        if not 0<move<m:raise ValueError('canonical Swords Dance row required')
        swords_binding=np.asarray(data.binding,np.uint32);swords_candidate[move]=True
    spikes_binding=np.zeros(8,np.uint32);spikes_candidate=np.zeros(m,bool)
    if spikes_compiled is not None:
        from . import engine_spikes_callbacks as spikes
        data=spikes.check_compiled(spikes_compiled)
        if spikes._COMPILED[spikes_compiled][2] is not engine_catalog:raise ValueError('same original Spikes/kernel catalog required')
        move=int(data.move_id)
        if not 0<move<m:raise ValueError('canonical Spikes bounds required')
        spikes_binding=np.asarray(data.binding,np.uint32);spikes_candidate[move]=True
    thunder_binding=np.zeros(8,np.uint32);thunder_candidate=np.zeros(m,bool);thunder_substitute_failure=False;thunder_ground=np.zeros(0,np.int32)
    if thunder_wave_compiled is not None:
        from . import engine_thunder_wave_callbacks as thunder
        data=thunder.check_compiled(thunder_wave_compiled)
        if thunder._COMPILED[thunder_wave_compiled][2] is not engine_catalog:raise ValueError('same original Thunder Wave/kernel catalog required')
        move=int(data.move_id)
        if not 0<move<m:raise ValueError('canonical Thunder Wave row required')
        thunder_binding=np.asarray(data.binding,np.uint32);thunder_candidate[move]=True
        accounting._shape(data.substitute_failure,(),jnp.bool_);thunder_substitute_failure=bool(data.substitute_failure);thunder_ground=np.asarray(thunder.pack_ground(data.ground_immunity))
    powder_binding=np.zeros(8,np.uint32);powder_candidate=np.zeros(m,bool);powder_miss=np.zeros(m,bool)
    if sleep_powder_compiled is not None:
        from . import engine_sleep_powder_callbacks as powder
        data=powder.check_compiled(sleep_powder_compiled)
        if powder._COMPILED[sleep_powder_compiled][2] is not engine_catalog:raise ValueError('same original Sleep Powder/kernel catalog required')
        mask=np.asarray(data.move_mask)
        if mask.shape!=(m,) or mask.dtype!=np.dtype(bool) or not mask.any():raise ValueError('canonical primary sleep mask required')
        powder_miss=np.asarray(data.miss_mask)
        if powder_miss.shape!=(m,) or powder_miss.dtype!=np.dtype(bool) or np.any(powder_miss&~mask) or (not bool(data.primary_sleep) and powder_miss.any()):raise ValueError("original explicit primary sleep miss mask required")
        powder_binding=np.asarray(data.binding,np.uint32);powder_candidate=mask.copy()
    risk[(weather==0)&~((whirlwind_move>0)&(np.arange(m)==whirlwind_move))&~((baton_move>0)&(np.arange(m)==baton_move))&~heal_bell_candidates&~self_move_candidates&~candidate&~substitute_candidate&~toxic_candidate&~calm_candidate&~swords_candidate&~spikes_candidate&~thunder_candidate&~powder_candidate&~protect_candidates&~bulk_candidates&~(self_boost_lengths>0)&~plain_candidates&~heal_candidates&~belly_candidates&~((encore_move>0)&(np.arange(m)==encore_move))&~((leech_move>0)&(np.arange(m)==leech_move))]|=NON_WEATHER
    def flag(module,name,bit):
        if module is None:return
        values=np.asarray(getattr(module,name))
        if values.shape[:1]!=(m,):raise ValueError('exact selected kernel move columns required')
        on=np.any(values!=0,axis=tuple(range(1,values.ndim))) if values.ndim>1 else values!=0
        risk[on]|=bit
    flag(table.effects,'pain_split',SELF_EFFECT)
    if table.effects is not None:risk[np.asarray(table.effects.rest)&~self_move_candidates]|=SELF_EFFECT
    if table.effects is not None:risk[(np.asarray(table.effects.heal_half)|np.asarray(table.effects.weather_heal))&~heal_candidates]|=SELF_EFFECT
    if table.effects is not None:risk[np.asarray(table.effects.belly_drum)&~belly_candidates]|=SELF_EFFECT
    if table.effects is not None:risk[np.asarray(table.effects.substitute)&~substitute_candidate]|=SELF_EFFECT
    if table.effects is not None:risk[np.any(np.asarray(table.effects.self_boosts)!=0,axis=1)&~calm_candidate&~swords_candidate&~bulk_candidates&~(self_boost_lengths>0)]|=BOOST
    for name in ('hit_spa_drop','haze'):flag(table.effects,name,BOOST)
    flag(table.effects,'endure',TARGET_OUTCOME)
    if table.effects is not None:risk[np.asarray(table.effects.protect)&~protect_candidates]|=TARGET_OUTCOME
    if table.effects is not None:risk[np.asarray(table.effects.heal_bell)&~heal_bell_candidates]|=FIELD_VOLATILE
    if table.effects is not None:risk[(np.asarray(table.effects.status)!=0)&~toxic_candidate&~thunder_candidate&~powder_candidate&~plain_candidates]|=FIELD_VOLATILE
    flag(table.sustain,'spin',FIELD_VOLATILE)
    if table.sustain is not None:risk[np.asarray(table.sustain.wish)&~self_move_candidates]|=FIELD_VOLATILE
    if table.sustain is not None:risk[np.asarray(table.sustain.seed)&(np.arange(m)!=leech_move)]|=FIELD_VOLATILE
    if table.entry is not None:risk[np.asarray(table.entry.spikes_move)&~spikes_candidate]|=FIELD_VOLATILE
    flag(table.linked_traps,'move',FIELD_VOLATILE)
    if table.encore is not None:risk[np.asarray(table.encore.move)&(np.arange(m)!=encore_move)]|=FIELD_VOLATILE
    if table.secondary is not None:
        chance=np.asarray(table.secondary.chance)
        if chance.shape!=(m,):raise ValueError('exact selected kernel move columns required')
        # Registered direct hits already visit the original secondary guards.
        # A possible proc is not an actual write: status/reflection callbacks
        # and stats/flinch branch guards retain their existing sticky refusals.
        risk[(chance!=0)&~candidate]|=FIELD_VOLATILE
    if table.self_destruct is not None:risk[np.asarray(table.self_destruct.move)&~self_destruct_candidates]|=SELF_HP
    if table.damage_moves is not None:risk[(np.asarray(table.damage_moves.recoil_den)!=0)&~recoil_candidate]|=SELF_HP
    flag(table.solar_beam,'move',BLOCKED_ACTION);flag(table.focus_punch,'move',BLOCKED_ACTION)
    branch=np.zeros(len(Branch)+1,np.int32)
    for code in Branch:
        if 6<=int(code)<=9:branch[int(code)]=SELF_EFFECT
        elif 10<=int(code)<=12 or code==Branch.HAZE:branch[int(code)]=BOOST
        elif 13<=int(code)<=19 or code==Branch.ENTRY_CONFUSION_CURE:branch[int(code)]=FIELD_VOLATILE
        elif 21<=int(code)<=23:branch[int(code)]=SELF_HP
    branch[Branch.PRIMARY_STATUS]=FIELD_VOLATILE
    branch[Branch.HEAL_BELL]=FIELD_VOLATILE
    files=('transition.py','secondary.py','entry.py','effects.py','absorption.py','sustain.py',
        'linked_traps.py','encore.py','mechanics.py','engine_execution_completeness.py','engine_kernel_frame_risk.py')
    inventory = dict(before_move_blocked_binding=before_move_blocked_binding.tolist(), before_move_blocked_moves=before_move_blocked_moves.tolist(), before_move_blocked_abilities=before_move_blocked_abilities.tolist(), before_move_blocked_items=before_move_blocked_items.tolist(), self_destruct_binding=self_destruct_binding.tolist(), self_destruct_candidates=self_destruct_candidates.tolist(), self_destruct_powers=self_destruct_powers.tolist(), self_destruct_damp=self_destruct_damp.tolist(), self_destruct_abilities=self_destruct_abilities.tolist(), self_destruct_items=self_destruct_items.tolist(), self_destruct_own_tempo=self_destruct_own_tempo.tolist(), heal_bell_binding=heal_bell_binding.tolist(), heal_bell_move=heal_bell_move, heal_bell_soundproof=heal_bell_soundproof.tolist(), schema=SCHEMA, baton_binding=baton_binding.tolist(), baton_move=baton_move, baton_abilities=baton_abilities.tolist(), baton_items=baton_items.tolist(), baton_outgoing_dormant_abilities=baton_outgoing_dormant_abilities.tolist(), baton_outgoing_insomnia_status=baton_outgoing_insomnia_status.tolist(), declared={x.name: int(x) for x in Branch}, source_files={f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in files}, pinch_binding=pinch_binding.tolist(), pinch_stats=pinch_stats.tolist(), scope='DECLARED_CUTPOINT_IDENTITY_ONLY_UNWIRED_NO_COMPLETENESS', kernel=engine_catalog.binding, accounting=np.asarray(record[0].inventory_binding).tolist(), selected_risk=risk.tolist(), branch_risk=branch.tolist(), direct_hit_binding=direct_binding.tolist(), direct_candidates=candidate.tolist(), seismic_toss_move=seismic, substitute_binding=substitute_binding.tolist(), substitute_candidates=substitute_candidate.tolist(), toxic_binding=toxic_binding.tolist(), toxic_candidates=toxic_candidate.tolist(), recoil_binding=recoil_binding.tolist(), recoil_candidates=recoil_candidate.tolist(), recoil_ko=recoil_ko, recoil_rock_head_source=recoil_rock_head_source, recoil_rock_head=recoil_rock_head.tolist(), calm_mind_binding=calm_binding.tolist(), calm_mind_candidates=calm_candidate.tolist(), swords_dance_binding=swords_binding.tolist(), swords_dance_candidates=swords_candidate.tolist(), spikes_binding=spikes_binding.tolist(), spikes_candidates=spikes_candidate.tolist(), direct_ko_candidates=ko_candidates.tolist(), thunder_wave_ground_immunity=thunder_ground.tolist(), thunder_wave_binding=thunder_binding.tolist(), thunder_wave_candidates=thunder_candidate.tolist(), sleep_powder_binding=powder_binding.tolist(), sleep_powder_candidates=powder_candidate.tolist(), no_damage_binding=no_damage_binding.tolist(), primary_plain_binding=plain_binding.tolist(), primary_plain_candidates=plain_candidates.tolist(), primary_plain_status=plain_status.tolist(), primary_plain_wisp_miss=plain_wisp, self_boost_binding=self_boost_binding.tolist(), self_boost_stats=self_boost_stats.tolist(), self_boost_deltas=self_boost_deltas.tolist(), self_boost_lengths=self_boost_lengths.tolist(), bulk_up_binding=bulk_binding.tolist(), bulk_up_candidates=bulk_candidates.tolist(), protect_binding=protect_binding.tolist(), protect_candidates=protect_candidates.tolist(), protect_block_candidates=protect_blocks.tolist(), color_change_binding=color_binding.tolist(), color_change_abilities=color_abilities.tolist(), color_change_candidates=color_candidates.tolist(), color_change_types=color_types.tolist(), color_change_dynamic=color_dynamic.tolist(), primary_sleep_miss_candidates=powder_miss.tolist(), self_heal_binding=heal_binding.tolist(), self_heal_candidates=heal_candidates.tolist(), self_heal_family=heal_family.tolist(), belly_drum_binding=belly_binding.tolist(), belly_drum_candidates=belly_candidates.tolist(), sleep_blocked_binding=sleep_block_binding.tolist(), sleep_blocked_moves=sleep_block_moves.tolist(), encore_binding=encore_binding.tolist(), encore_move=encore_move, leech_binding=leech_binding.tolist(), leech_move=leech_move, self_move_binding=self_move_binding.tolist(), self_move_ids=self_move_ids.tolist(), toxic_substitute_failure=toxic_substitute_failure, substitute_failure=substitute_failure, toxic_existing_status=toxic_existing_status, thunder_wave_substitute_failure=thunder_substitute_failure, sleep_talk_toxic_child=sleep_talk_toxic_child, levitate_immunity=levitate_immunity, recoil_immune_source=recoil_immune_source, recoil_immune_ghost_type=recoil_immune_ghost_type, recoil_immune_wonder_guard=recoil_immune_wonder_guard, sleep_talk_rest_failure=sleep_talk_rest_failure, whirlwind_binding=whirlwind_binding.tolist(), whirlwind_move=whirlwind_move, whirlwind_abilities=whirlwind_abilities.tolist(), whirlwind_items=whirlwind_items.tolist(), whirlwind_source=whirlwind_source.tolist(), focus_binding=focus_binding.tolist(), focus_move=focus_move, focus_abilities=focus_abilities.tolist(), baton_natural_cure=baton_natural_cure.tolist())
    raw=json.dumps(inventory,sort_keys=True,separators=(',',':'))
    words=lambda x:jnp.asarray(x,jnp.uint32)
    device = Tables(words(np.frombuffer(hashlib.sha256(raw.encode()).digest(), dtype='>u4').astype(np.uint32)), words(engine_catalog.engine_binding), words(engine_catalog.source_binding), words(engine_catalog.move_binding), record[0].producer_binding, record[0].inventory_binding, jnp.asarray(risk), jnp.asarray(branch), words(direct_binding), jnp.asarray(candidate), jnp.asarray(seismic, jnp.int32), words(substitute_binding), jnp.asarray(substitute_candidate), words(toxic_binding), jnp.asarray(toxic_candidate), words(recoil_binding), jnp.asarray(recoil_candidate), words(calm_binding), jnp.asarray(calm_candidate), words(swords_binding), jnp.asarray(swords_candidate), words(spikes_binding), jnp.asarray(spikes_candidate), jnp.asarray(ko_candidates), words(thunder_binding), words(powder_binding), jnp.asarray(powder_candidate), jnp.asarray(thunder_candidate), words(no_damage_binding), words(plain_binding), jnp.asarray(plain_candidates), jnp.asarray(plain_status), words(pinch_binding), jnp.asarray(pinch_stats), words(bulk_binding), jnp.asarray(bulk_candidates), words(protect_binding), jnp.asarray(protect_candidates), jnp.asarray(protect_blocks), words(color_binding), jnp.asarray(color_abilities), jnp.asarray(color_candidates), jnp.asarray(color_types), jnp.asarray(color_dynamic), jnp.asarray(powder_miss), words(heal_binding), jnp.asarray(heal_candidates), words(belly_binding), jnp.asarray(belly_candidates), words(sleep_block_binding), jnp.asarray(sleep_block_moves), words(encore_binding), jnp.asarray(encore_move, jnp.int32), words(leech_binding), jnp.asarray(leech_move, jnp.int32), words(self_move_binding), jnp.asarray(self_move_ids, jnp.int32), jnp.asarray(toxic_substitute_failure, jnp.bool_), jnp.asarray(substitute_failure, jnp.bool_), jnp.asarray(toxic_existing_status, jnp.bool_), jnp.asarray(heal_family, jnp.int32), jnp.asarray(recoil_ko, jnp.bool_), jnp.asarray(recoil_rock_head_source, jnp.bool_), jnp.asarray(recoil_rock_head), words(self_boost_binding), jnp.asarray(self_boost_stats), jnp.asarray(self_boost_deltas), jnp.asarray(self_boost_lengths), words(heal_bell_binding), jnp.asarray(heal_bell_move, jnp.int32), jnp.asarray(heal_bell_soundproof), words(baton_binding), jnp.asarray(baton_move, jnp.int32), jnp.asarray(baton_abilities), jnp.asarray(baton_items), words(self_destruct_binding), jnp.asarray(self_destruct_candidates), jnp.asarray(self_destruct_powers), jnp.asarray(self_destruct_damp), jnp.asarray(self_destruct_abilities), jnp.asarray(self_destruct_items), words(before_move_blocked_binding), jnp.asarray(before_move_blocked_moves), jnp.asarray(before_move_blocked_abilities), jnp.asarray(before_move_blocked_items), jnp.asarray(thunder_substitute_failure, jnp.bool_), jnp.asarray(sleep_talk_toxic_child, jnp.bool_), jnp.asarray(levitate_immunity, jnp.bool_), jnp.asarray(recoil_immune_source, jnp.bool_), jnp.asarray(recoil_immune_ghost_type, jnp.int32), jnp.asarray(recoil_immune_wonder_guard, jnp.int32), jnp.asarray(sleep_talk_rest_failure, jnp.bool_), jnp.asarray(self_destruct_own_tempo), jnp.asarray(baton_outgoing_dormant_abilities), jnp.asarray(thunder_ground, jnp.int32), jnp.asarray(baton_outgoing_insomnia_status, jnp.int32), jnp.asarray(plain_wisp, jnp.int32), words(whirlwind_binding), jnp.asarray(whirlwind_move, jnp.int32), jnp.asarray(whirlwind_abilities), jnp.asarray(whirlwind_items), jnp.asarray(whirlwind_source), words(focus_binding), jnp.asarray(focus_move, jnp.int32), jnp.asarray(focus_abilities), jnp.asarray(baton_natural_cure))
    result=Compiled(device);_COMPILED[result]=(device,tuple(device),engine_catalog,accounting_compiled,raw,direct_hit_compiled,substitute_compiled,toxic_compiled,recoil_compiled,calm_mind_compiled,spikes_compiled,swords_dance_compiled,thunder_wave_compiled,sleep_powder_compiled)
    return result


def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (r is None or compiled.device is not r[0] or any(a is not b for a,b in zip(compiled.device,r[1]))
            or any(not isinstance(a,jax.Array) or isinstance(a,jax.core.Tracer) for a in compiled.device)):
        raise ValueError('original immutable kernel risk compiler required')
    packed=compiled.device.thunder_wave_ground_immunity
    if packed.ndim!=1 or packed.dtype!=jnp.dtype(jnp.int32):
        raise ValueError('exact flat int32 Ground source capability required')
    catalog=r[2];public.check_catalog(catalog,catalog.engine_table,expected_binding=catalog.binding,
        expected_engine_binding=np.asarray(catalog.engine_binding,np.uint32))
    accounting._checked(r[3])
    if r[5] is not None:
        from . import engine_direct_hit_controls as direct
        direct.check_compiled(r[5])
    if r[6] is not None:
        from . import engine_substitute_callbacks as substitute
        substitute.check_compiled(r[6])
    if r[7] is not None:
        from . import engine_toxic_callbacks as toxic
        toxic.check_compiled(r[7])
    if r[8] is not None:
        from . import engine_recoil_callbacks as recoil
        recoil.check_compiled(r[8])
    if r[9] is not None:
        from . import engine_calm_mind_callbacks as calm
        calm.check_compiled(r[9])
    if r[11] is not None:
        from . import engine_swords_dance_callbacks as swords
        swords.check_compiled(r[11])
    if r[10] is not None:
        from . import engine_spikes_callbacks as spikes
        spikes.check_compiled(r[10])
    if r[12] is not None:
        from . import engine_thunder_wave_callbacks as thunder
        thunder.check_compiled(r[12])
    if r[13] is not None:
        from . import engine_sleep_powder_callbacks as powder
        powder.check_compiled(r[13])
    return compiled.device


def inventory_json(compiled):
    _host();check_compiled(compiled);return _COMPILED[compiled][4]


def _vector(value,b,dtype=jnp.int32):
    if dtype==jnp.int32:value=public_mon_state._int32(value)
    elif type(value) is bool:value=jnp.asarray(value)
    if getattr(value,'dtype',None)!=jnp.dtype(dtype) or getattr(value,'shape',None) not in ((),(b,)):
        raise ValueError('typed scalar/B actual kernel observation required')
    return jnp.broadcast_to(value,(b,))


def initialize_device(streams,tables):
    accounting._shape(streams,streams.shape,jnp.int32)
    if streams.ndim!=1 or not len(streams):raise ValueError('nonempty physical streams required')
    b=len(streams);ids=jnp.sort(streams);bad=(streams<=0)|jnp.any(ids[1:]==ids[:-1])
    return Carry(tables.binding,tables.accounting_binding,streams,jnp.zeros(b,jnp.int32),jnp.zeros(b,bool),
        jnp.where(bad,NAMESPACE,0),jnp.zeros(b,jnp.int32),jnp.zeros((b,len(Branch)),jnp.int32))


def _bound(carry,tape,tables):
    if type(carry) is not Carry or type(tables) is not Tables:raise ValueError('typed kernel risk carry/tables required')
    b=len(carry.stream_id)
    for x in (carry.binding,carry.accounting_binding,*tables[:6]):accounting._shape(x,(8,),jnp.uint32)
    for name in ('stream_id','frame','risk','observed'):accounting._shape(getattr(carry,name),(b,),jnp.int32)
    accounting._shape(carry.opened,(b,),jnp.bool_);accounting._shape(carry.counts,(b,len(Branch)),jnp.int32)
    accounting._shape(tables.selected_risk,(len(tables.selected_risk),),jnp.int32)
    accounting._shape(tables.branch_risk,(len(Branch)+1,),jnp.int32)
    accounting._shape(tables.no_damage_binding,(8,),jnp.uint32)
    accounting._shape(tables.primary_plain_binding,(8,),jnp.uint32)
    accounting._shape(tables.primary_plain_candidates,tables.direct_candidates.shape,jnp.bool_)
    accounting._shape(tables.primary_plain_status,tables.direct_candidates.shape,jnp.int32)
    accounting._shape(tables.color_change_binding,(8,),jnp.uint32)
    accounting._shape(tables.color_change_abilities,(len(tables.color_change_abilities),),jnp.bool_)
    accounting._shape(tables.color_change_candidates,tables.direct_candidates.shape,jnp.bool_)
    accounting._shape(tables.color_change_types,tables.direct_candidates.shape,jnp.int32)
    accounting._shape(tables.color_change_dynamic,tables.direct_candidates.shape,jnp.bool_)
    accounting._shape(tables.pinch_binding,(8,),jnp.uint32)
    accounting._shape(tables.pinch_stats,tables.pinch_stats.shape,jnp.int32)
    if tables.pinch_stats.ndim!=1 or not len(tables.pinch_stats):raise ValueError("original pinch namespace required")
    accounting._shape(tables.bulk_up_binding,(8,),jnp.uint32)
    accounting._shape(tables.bulk_up_candidates,tables.direct_candidates.shape,jnp.bool_)
    accounting._shape(tables.protect_binding,(8,),jnp.uint32)
    accounting._shape(tables.protect_candidates,tables.direct_candidates.shape,jnp.bool_)
    accounting._shape(tables.protect_block_candidates,tables.direct_candidates.shape,jnp.bool_)
    accounting._shape(tables.self_destruct_binding,(8,),jnp.uint32)
    accounting._shape(tables.self_destruct_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.self_destruct_powers,(len(tables.selected_risk),),jnp.int32)
    for value in (tables.self_destruct_damp,tables.self_destruct_abilities,tables.self_destruct_items,tables.self_destruct_own_tempo):accounting._shape(value,(len(value),),jnp.bool_)
    accounting._shape(tables.recoil_binding,(8,),jnp.uint32)
    accounting._shape(tables.recoil_ko,(),jnp.bool_)
    accounting._shape(tables.recoil_immune_ghost_type,(),jnp.int32)
    accounting._shape(tables.recoil_immune_wonder_guard,(),jnp.int32)
    accounting._shape(tables.recoil_immune_source,(),jnp.bool_)
    accounting._shape(tables.recoil_rock_head_source,(),jnp.bool_)
    accounting._shape(tables.recoil_rock_head,(len(tables.recoil_rock_head),),jnp.bool_)
    accounting._shape(tables.recoil_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.calm_mind_binding,(8,),jnp.uint32)
    accounting._shape(tables.calm_mind_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.swords_dance_binding,(8,),jnp.uint32)
    accounting._shape(tables.swords_dance_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.leech_binding,(8,),jnp.uint32)
    accounting._shape(tables.leech_move,(),jnp.int32)
    accounting._shape(tables.toxic_substitute_failure,(),jnp.bool_)
    accounting._shape(tables.toxic_existing_status,(),jnp.bool_)
    accounting._shape(tables.self_move_binding,(8,),jnp.uint32)
    accounting._shape(tables.self_move_ids,(3,),jnp.int32)
    accounting._shape(tables.encore_binding,(8,),jnp.uint32)
    accounting._shape(tables.encore_move,(),jnp.int32)
    accounting._shape(tables.before_move_blocked_binding,(8,),jnp.uint32)
    accounting._shape(tables.before_move_blocked_moves,(2,len(tables.selected_risk)),jnp.bool_)
    for value in (tables.before_move_blocked_abilities,tables.before_move_blocked_items):accounting._shape(value,(len(value),),jnp.bool_)
    accounting._shape(tables.sleep_blocked_binding,(8,),jnp.uint32)
    accounting._shape(tables.sleep_blocked_moves,tables.direct_candidates.shape,jnp.bool_)
    accounting._shape(tables.belly_drum_binding,(8,),jnp.uint32)
    accounting._shape(tables.belly_drum_candidates,tables.direct_candidates.shape,jnp.bool_)
    accounting._shape(tables.sleep_powder_binding,(8,),jnp.uint32)
    accounting._shape(tables.primary_sleep_miss_candidates,tables.direct_candidates.shape,jnp.bool_)
    accounting._shape(tables.self_heal_binding,(8,),jnp.uint32)
    accounting._shape(tables.self_heal_candidates,tables.direct_candidates.shape,jnp.bool_)
    accounting._shape(tables.self_heal_family,tables.direct_candidates.shape,jnp.int32)
    accounting._shape(tables.sleep_powder_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.sleep_talk_toxic_child,(),jnp.bool_)
    accounting._shape(tables.sleep_talk_rest_failure,(),jnp.bool_)
    accounting._shape(tables.levitate_immunity,(),jnp.bool_)
    accounting._shape(tables.thunder_wave_substitute_failure,(),jnp.bool_)
    accounting._shape(tables.thunder_wave_binding,(8,),jnp.uint32)
    accounting._shape(tables.thunder_wave_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.spikes_binding,(8,),jnp.uint32)
    accounting._shape(tables.spikes_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.direct_ko_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.toxic_binding,(8,),jnp.uint32)
    accounting._shape(tables.toxic_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.substitute_failure,(),jnp.bool_)
    accounting._shape(tables.substitute_binding,(8,),jnp.uint32)
    accounting._shape(tables.substitute_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.direct_hit_binding,(8,),jnp.uint32)
    accounting._shape(tables.direct_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.seismic_toss_move,(),jnp.int32)
    context=accounting._context(tape)
    if context is None:return jnp.zeros(b,bool),None
    if type(context.tables) is not accounting.Tables:raise ValueError('typed actual accounting context required')
    ledger=context.ledger
    same=(jnp.all(carry.binding==tables.binding)&jnp.all(carry.accounting_binding==tables.accounting_binding)
        &(carry.stream_id==tape.stream_id)&(carry.stream_id==ledger.stream_id)
        &jnp.all(ledger.inventory_binding==tables.accounting_binding)&(ledger.frame>0)&ledger.opened
        &(carry.stream_id>0)&(carry.frame>=0)&(carry.risk>=0)&(carry.observed>=0)
        &jnp.all(carry.counts>=0,axis=1))
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        same &= jnp.all(getattr(tape,name)==getattr(tables,name))
    for name in ('engine_binding','source_binding','producer_binding'):
        same &= jnp.all(getattr(context.tables,name)==getattr(tables,name))
    same &= jnp.all(context.tables.inventory_binding==tables.accounting_binding)
    return same,ledger


def begin_frame_device(carry,tape,tables,*,active):
    same,ledger=_bound(carry,tape,tables);b=len(carry.stream_id);active=_vector(active,b,jnp.bool_)
    frame=jnp.zeros(b,jnp.int32) if ledger is None else ledger.frame
    good=same&(carry.frame<2**31-1)&(frame==carry.frame+1)
    use=active&good
    return carry._replace(frame=jnp.where(use,frame,carry.frame),opened=jnp.where(active,good,carry.opened),
        observed=jnp.where(use,0,carry.observed),counts=jnp.where(use[:,None],0,carry.counts),
        risk=carry.risk|jnp.where(active&~good,ORDER|NAMESPACE,0))


def _observe(carry,tape,tables,code,value,*,active):
    same,ledger=_bound(carry,tape,tables);b=len(carry.stream_id);active=_vector(active,b,jnp.bool_)
    frame=jnp.zeros(b,jnp.int32) if ledger is None else ledger.frame
    same &= carry.opened&(carry.frame==frame)
    slot=int(code)-1;overflow=carry.counts[:,slot]>=2**31-1;use=active&same&~overflow
    count=jnp.minimum(carry.counts[:,slot],2**31-2)+1
    counts=carry.counts.at[:,slot].set(jnp.where(use,count,carry.counts[:,slot]))
    return carry._replace(risk=carry.risk|jnp.where(active,
        value|jnp.where(same,0,NAMESPACE|ORDER)|jnp.where(overflow,OVERFLOW,0)
        |jnp.where((tape.refusal&~public.UNINSTRUMENTED)!=0,UPSTREAM,0),0),
        observed=carry.observed|jnp.where(use,1<<slot,0),counts=counts)


def observe_constructor_device(carry,tape,state,tables,*,active):
    if type(state) is not BattleState:raise ValueError('actual constructor BattleState required')
    b=len(carry.stream_id);bad=(state.turn!=0)|state.done
    for name in ('status','confusion_counter','boosts','volatile','substitute_hp','spikes','wish_counter','encore_counter','charge_counter'):
        x=getattr(state,name);bad |= jnp.any(x!=0,axis=tuple(range(1,x.ndim)))
    bad |= jnp.any(state.trap_source!=-1,axis=(1,2))
    return _observe(carry,tape,tables,Branch.CONSTRUCTOR,jnp.where(bad,CONSTRUCTOR_STATE,0),active=active)


def observe_pre_move_device(carry,tape,state,side,before,tables,*,eligible,active,captured_sleep=None,captured_before_move=None):
    if type(before) is not mechanics.BeforeMove:raise ValueError('literal actual BeforeMove result required')
    b=len(carry.stream_id);side=_vector(side,b);eligible=_vector(eligible,b,jnp.bool_)
    accounting._shape(state.active,(b,2),jnp.int32)
    accounting._shape(state.confusion_counter,(b,2,6),jnp.int32)
    rows=jnp.arange(b);safe_side=jnp.clip(side,0,1);physical=state.active[rows,safe_side]
    address=(side>=0)&(side<2)&(physical>=0)&(physical<6)
    confusion=state.confusion_counter[rows,safe_side,jnp.clip(physical,0,5)]
    from . import engine_sleep_blocked_callbacks as sleep_blocked
    supported=sleep_blocked.captured_matches(tape,captured_sleep,state,side,before)
    from . import engine_before_move_blocked_callbacks as bm
    supported |= bm.captured_matches(tape,captured_before_move,state,side,before)
    bad=eligible&address&((~before.can_act&~supported)|before.confusion_self_hit|(confusion>0))
    return _observe(carry,tape,tables,Branch.PRE_MOVE,jnp.where(bad,BLOCKED_ACTION,0)
        |jnp.where(eligible&~address,NAMESPACE,0),active=active)


def observe_selected_device(carry,tape,move,tables,*,eligible,active,captured_focus=None,state=None,side=None,caller_move=None):
    b=len(carry.stream_id);move=_vector(move,b);eligible=_vector(eligible,b,jnp.bool_)
    valid=(move>0)&(move<len(tables.selected_risk));safe=jnp.clip(move,0,len(tables.selected_risk)-1)
    value=jnp.where(eligible,jnp.where(valid,tables.selected_risk[safe],NAMESPACE),0)
    if captured_focus is not None:
        from . import engine_focus_punch_callbacks as focus_callbacks
        supported=focus_callbacks.matches_selected(tape,captured_focus,state,side,move,caller_move)
        value &= ~jnp.where(supported,BLOCKED_ACTION,0)
    return _observe(carry,tape,tables,Branch.SELECTED_MOVE,value,active=active)


def _captured_direct_matches(tape,outcome,tables,captured):
    """Read-only source receipt match; no risk is cleared retroactively."""
    from . import engine_direct_hit_controls as direct
    b=tape.count.shape[0];no=jnp.zeros(b,bool)
    if captured is None:return no
    if (type(captured) is not CapturedDirectHit or type(captured.context) is not direct.RegisteredContext
            or type(captured.context.data) is not direct.Data or type(captured.write) is not direct.HitWrite):
        raise ValueError('typed SAME-call captured direct hit required')
    context=accounting._context(tape)
    if context is None:return no
    ledger=context.ledger;slot,same=accounting._ticket(ledger,captured.ticket);rows=jnp.arange(b)
    same &= accounting._validate(ledger,captured.context.accounting)
    data=captured.context.data
    if not direct._check(tape,data):return no
    move=_vector(captured.move,b)
    same &= (data.seismic_toss_move==tables.seismic_toss_move)
    same &= (data.move_count==len(tables.selected_risk))&(move>0)&(move<data.move_count)
    same &= jnp.any(tables.direct_hit_binding!=0)&jnp.all(data.binding==tables.direct_hit_binding)
    same &= jnp.all(captured.context.accounting.inventory_binding==tables.accounting_binding)
    for name in ('engine_binding','source_binding','producer_binding'):
        same &= jnp.all(getattr(data,name)==getattr(tables,name))
        same &= jnp.all(getattr(captured.context.accounting,name)==getattr(context.tables,name))
    same &= jnp.all(data.move_binding==tables.move_binding)&ledger.opened&(ledger.refusal==0)
    same &= (ledger.stage[rows,slot]==2)&(ledger.end[rows,slot]==tape.count)
    same &= accounting._direct_hit_span_matches(tape,captured.ticket,captured.actor,captured.actor_physical,
        captured.target_physical,captured.move,captured.write,data=data)
    w=captured.write
    for key in ('power','critical','hit','immune','numerator','denominator','substitute','protected','enduring'):
        same &= getattr(outcome,key)==getattr(w,key)
    start=ledger.start[rows,slot]
    span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(
        jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    same &= jnp.all(~span|((ledger.claimed==slot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)
        &(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)
        &jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)
    return same


class CapturedSubstitute(NamedTuple):
    context: object
    ticket: object
    side: object
    physical: object
    move: object
    write: object


def _captured_substitute_matches(tape,tables,captured):
    from . import engine_substitute_callbacks as substitute
    b=len(tape.count);no=jnp.zeros(b,bool)
    if captured is None:return no
    if (type(captured) is not CapturedSubstitute or type(captured.context) is not substitute.RegisteredContext
            or type(captured.context.data) is not substitute.Data):
        raise ValueError('typed SAME-call captured Substitute required')
    context=accounting._context(tape)
    if context is None:return no
    ledger=context.ledger;slot,same=accounting._ticket(ledger,captured.ticket);rows=jnp.arange(b)
    same &= accounting._validate(ledger,captured.context.accounting)
    data=captured.context.data
    same &= substitute._check(tape,data)&jnp.any(tables.substitute_binding!=0)
    same &= jnp.all(data.binding==tables.substitute_binding)
    same &= jnp.all(captured.context.accounting.inventory_binding==tables.accounting_binding)
    move=_vector(captured.move,b);same &= (move==data.move_id)&(move>0)&(move<len(tables.substitute_candidates))
    same &= tables.substitute_candidates[jnp.clip(move,0,len(tables.substitute_candidates)-1)]
    for name in ('engine_binding','source_binding','producer_binding'):
        same &= jnp.all(getattr(data,name)==getattr(tables,name))
        same &= jnp.all(getattr(captured.context.accounting,name)==getattr(context.tables,name))
    same &= jnp.all(data.move_binding==tables.move_binding)&ledger.opened&(ledger.refusal==0)
    same &= (ledger.stage[rows,slot]==2)&(ledger.end[rows,slot]==tape.count)
    same &= accounting._substitute_span_matches(tape,captured.ticket,captured.side,captured.physical,
        captured.move,captured.write)
    start=ledger.start[rows,slot];span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(
        jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    same &= jnp.all(~span|((ledger.claimed==slot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)
        &(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)
        &jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)
    return same


def observe_substitute_device(carry,tape,attempted,tables,*,active,captured=None):
    attempted=_vector(attempted,len(carry.stream_id),jnp.bool_)
    supported=_captured_substitute_matches(tape,tables,captured)
    # Never erase a lifetime bit. This visit adds none only for its OWN fresh
    # exact successful Site11 span; both owner drains remain a separate gate.
    return _observe(carry,tape,tables,Branch.SUBSTITUTE,jnp.where(attempted&~supported,SELF_EFFECT,0),active=active)


def substitute_on_tape(tape,attempted,*,active,captured=None):
    return _on_tape(tape,observe_substitute_device,attempted,active=active,captured=captured)


class CapturedToxic(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    target:object
    move:object
    attempt:object


def _captured_toxic_matches(tape,tables,captured):
    from . import engine_toxic_callbacks as toxic
    b=len(tape.count);no=jnp.zeros(b,bool)
    if captured is None:return no
    if type(captured) is not CapturedToxic or type(captured.context) is not toxic.RegisteredContext:
        raise ValueError('typed SAME-call captured Toxic attempt required')
    context=accounting._context(tape)
    if context is None:return no
    ledger=context.ledger;slot,same=accounting._ticket(ledger,captured.ticket);rows=jnp.arange(b)
    same &= accounting._validate(ledger,captured.context.accounting)
    data=captured.context.data
    same &= toxic._check(tape,data)&jnp.any(tables.toxic_binding!=0)&jnp.all(data.binding==tables.toxic_binding)
    same &= jnp.all(captured.context.accounting.inventory_binding==tables.accounting_binding)
    move=_vector(captured.move,b);same &= (move==data.move_id)&(move>0)&(move<len(tables.toxic_candidates))
    same &= tables.toxic_candidates[jnp.clip(move,0,len(tables.toxic_candidates)-1)]
    for name in ('engine_binding','source_binding','producer_binding'):
        same &= jnp.all(getattr(data,name)==getattr(tables,name))
        same &= jnp.all(getattr(captured.context.accounting,name)==getattr(context.tables,name))
    same &= jnp.all(data.move_binding==tables.move_binding)&ledger.opened&(ledger.refusal==0)
    same &= (ledger.stage[rows,slot]==2)&(ledger.end[rows,slot]==tape.count)
    same &= accounting._toxic_span_matches(tape,captured.ticket,captured.side,captured.physical,
        captured.target,captured.move,captured.attempt)
    start=ledger.start[rows,slot];span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(
        jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    same &= jnp.all(~span|((ledger.claimed==slot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)
        &(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)
        &jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)
    return same


class CapturedThunderWave(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    target:object
    move:object
    attempt:object

class CapturedSleepPowder(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    target:object
    move:object
    attempt:object


def _captured_thunder_wave_matches(tape,tables,captured):
    from . import engine_thunder_wave_callbacks as thunder
    b=len(tape.count);supported=jnp.zeros(b,bool)
    if captured is None:return supported
    if type(captured) is not CapturedThunderWave:raise ValueError('typed SAME-call Thunder Wave capture required')
    context=accounting._context(tape)
    if context is None:return supported
    ledger=context.ledger;slot,same=accounting._ticket(ledger,captured.ticket);rows=jnp.arange(b);move=_vector(captured.move,b)
    supported=(same&jnp.any(tables.thunder_wave_binding!=0)&jnp.all(captured.context.data.binding==tables.thunder_wave_binding)
        &(move>0)&(move<len(tables.thunder_wave_candidates))&tables.thunder_wave_candidates[jnp.clip(move,0,len(tables.thunder_wave_candidates)-1)]
        &(ledger.stage[rows,slot]==2)&(ledger.end[rows,slot]==tape.count)&(ledger.refusal==0))
    supported &= thunder.span_matches(tape,captured.ticket,_vector(captured.side,b),_vector(captured.physical,b),
        _vector(captured.target,b),move,captured.attempt,captured.context)
    span=(jnp.arange(tape.words.shape[1])[None,:]>=ledger.start[rows,slot,None])&(jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    return supported&jnp.all(~span|((ledger.claimed==slot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)
        &(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)
        &jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)

def _captured_sleep_powder_matches(tape,tables,captured):
    from . import engine_sleep_powder_callbacks as powder
    b=len(tape.count);supported=jnp.zeros(b,bool)
    if captured is None:return supported
    if type(captured) is not CapturedSleepPowder:raise ValueError('typed SAME-call Sleep Powder capture required')
    context=accounting._context(tape)
    if context is None:return supported
    ledger=context.ledger;slot,same=accounting._ticket(ledger,captured.ticket);rows=jnp.arange(b);move=_vector(captured.move,b)
    supported=(same&jnp.any(tables.sleep_powder_binding!=0)&jnp.all(captured.context.data.binding==tables.sleep_powder_binding)
        &(move>0)&(move<len(tables.sleep_powder_candidates))&tables.sleep_powder_candidates[jnp.clip(move,0,len(tables.sleep_powder_candidates)-1)]
        &(ledger.stage[rows,slot]==2)&(ledger.refusal==0))
    supported &= powder.span_matches(tape,captured.ticket,_vector(captured.side,b),_vector(captured.physical,b),
        _vector(captured.target,b),move,captured.attempt,captured.context)
    span=(jnp.arange(tape.words.shape[1])[None,:]>=ledger.start[rows,slot,None])&(jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    return supported&jnp.all(~span|((ledger.claimed==slot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)
        &(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)
        &jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)


def observe_primary_status_device(carry,tape,attempted,tables,*,active,captured=None,captured_thunder_wave=None,captured_sleep_powder=None,captured_protect_block=None,captured_primary_plain=None):
    attempted=_vector(attempted,len(carry.stream_id),jnp.bool_)
    supported=_captured_toxic_matches(tape,tables,captured)|_captured_thunder_wave_matches(tape,tables,captured_thunder_wave)|_captured_sleep_powder_matches(tape,tables,captured_sleep_powder)
    from . import engine_protect_callbacks as protect_callbacks
    supported |= protect_callbacks.captured_block_matches(tape,captured_protect_block)
    from . import engine_primary_plain_status_callbacks as plain_primary
    supported |= plain_primary.captured_matches(tape,captured_primary_plain)
    return _observe(carry,tape,tables,Branch.PRIMARY_STATUS,jnp.where(attempted&~supported,FIELD_VOLATILE,0),active=active)


def primary_status_on_tape(tape,attempted,*,active,captured=None,captured_thunder_wave=None,captured_sleep_powder=None,captured_protect_block=None,captured_primary_plain=None):
    return _on_tape(tape,observe_primary_status_device,attempted,active=active,captured=captured,captured_thunder_wave=captured_thunder_wave,captured_sleep_powder=captured_sleep_powder,captured_protect_block=captured_protect_block,captured_primary_plain=captured_primary_plain)


def observe_outcome_device(carry,tape,outcome,tables,*,eligible,active,captured_direct=None,pending_recoil=None,pending_ko=None,captured_no_damage=None,captured_substitute_damage=None,captured_protect_block=None):
    if type(outcome) is not Outcome:raise ValueError('literal typed kernel outcomes required')
    b=len(carry.stream_id);eligible=_vector(eligible,b,jnp.bool_)
    values={name:_vector(x,b,jnp.int32 if name in ('power','numerator','denominator','recoil') else jnp.bool_)
        for name,x in zip(outcome._fields,outcome)}
    o=Outcome(**values)
    unsupported=(o.absorbed|((o.power>0)&(o.critical|~o.hit|o.immune|(o.numerator!=o.denominator)
        |o.substitute|o.protected|o.enduring)))
    direct_mode=jnp.any(tables.direct_hit_binding!=0)
    supported_direct=_captured_direct_matches(tape,o,tables,captured_direct)
    if pending_ko is not None:
        from . import engine_direct_ko_callbacks as ko
        supported_direct |= ko.matches_outcome_device(tape,pending_ko,o,tables)

    candidate=jnp.zeros(b,bool)
    if type(captured_direct) is CapturedDirectHit:
        move=_vector(captured_direct.move,b)
        candidate=(move>0)&(move<len(tables.direct_candidates))&tables.direct_candidates[
            jnp.clip(move,0,len(tables.direct_candidates)-1)]
    # The optional captured-Sub API must request the same canonical zero-power
    # outcome check even when no separate direct-hit Capture was supplied.
    from . import engine_substitute_damage_callbacks as substitute_damage
    supported_sub=jnp.zeros(b,bool)
    if type(captured_substitute_damage) is substitute_damage.Captured:
        move=_vector(captured_substitute_damage.pending.move,b)
        candidate |= (tables.seismic_toss_move>0)&(move==tables.seismic_toss_move)
    from . import engine_no_damage_callbacks as no_damage
    supported_no_damage=no_damage.captured_matches(tape,captured_no_damage,o)
    unsupported=jnp.where(direct_mode & ((o.power>0)|candidate),
        jnp.where(supported_no_damage,False,o.absorbed|~supported_direct),unsupported)
    if type(captured_substitute_damage) is substitute_damage.Captured:
        capture=captured_substitute_damage;w=capture.write;p=capture.pending
        supported_sub=substitute_damage.span_matches(tape,p,w)
        supported_sub &= jnp.any(tables.direct_hit_binding!=0)&jnp.all(p.context.data.binding==tables.direct_hit_binding)
        for key in ('power','hit','immune','critical','numerator','denominator','protected','enduring'):
            supported_sub &= getattr(o,key)==getattr(w,key)
        supported_sub &= o.substitute&~o.absorbed&~o.explodes
        sub_recoil=jnp.zeros(b,bool)
        if pending_recoil is not None:
            from . import engine_recoil_callbacks as recoil
            sub_recoil=(recoil.matches_pending(tape,pending_recoil)
                &recoil.matches_substitute_pending(tape,pending_recoil,capture)
                &jnp.any(tables.recoil_binding!=0)&jnp.all(tables.recoil_binding==pending_recoil.context.data.binding)
                &(o.recoil==pending_recoil.recoil))
        supported_sub &= (o.recoil==0)|sub_recoil
        unsupported &= ~supported_sub
    from . import engine_protect_callbacks as protect_callbacks
    unsupported &= ~protect_callbacks.captured_block_matches(tape,captured_protect_block,o)
    recoil_ok=jnp.zeros(b,bool);recoil_needed=jnp.zeros(b,bool)
    if type(captured_direct) is CapturedDirectHit:
        move=_vector(captured_direct.move,b)
        recoil_needed=(move>0)&(move<len(tables.recoil_candidates))&tables.recoil_candidates[jnp.clip(move,0,len(tables.recoil_candidates)-1)]
    if pending_recoil is not None:
        from . import engine_recoil_callbacks as recoil
        recoil_needed |= pending_recoil.requested
        recoil_ok=(recoil.matches_pending(tape,pending_recoil)&(supported_direct|supported_sub)
            &jnp.any(tables.recoil_binding!=0)&jnp.all(tables.recoil_binding==pending_recoil.context.data.binding)
            &(o.recoil==pending_recoil.recoil))
    from . import engine_self_destruct_callbacks as sd
    sd_supported=sd.matches_outcome(tape,o)
    unsupported &= ~sd_supported
    value=jnp.where(unsupported,TARGET_OUTCOME,0)|jnp.where((o.explodes&~sd_supported)|(((o.recoil>0)|recoil_needed)&~recoil_ok),SELF_HP,0)
    value |= jnp.where((o.power<0)|(o.numerator<0)|(o.denominator<=0)|(o.recoil<0),NAMESPACE,0)
    return _observe(carry,tape,tables,Branch.HIT_OUTCOME,jnp.where(eligible,value,0),active=active)


class CapturedCalmMind(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    move:object
    write:object


def _captured_calm_mind_matches(tape,tables,captured):
    from . import engine_calm_mind_callbacks as calm
    b=tape.count.shape[0];supported=jnp.zeros(b,bool)
    if captured is not None:
        if type(captured) is not CapturedCalmMind:raise ValueError('typed SAME-call Calm Mind capture required')
        context=accounting._context(tape)
        if context is not None:
            slot,same=accounting._ticket(context.ledger,captured.ticket);rows=jnp.arange(b)
            move=_vector(captured.move,b)
            supported=(same&jnp.any(tables.calm_mind_binding!=0)
                &jnp.all(captured.context.data.binding==tables.calm_mind_binding)
                &(move>0)&(move<len(tables.calm_mind_candidates))
                &tables.calm_mind_candidates[jnp.clip(move,0,len(tables.calm_mind_candidates)-1)]
                &(context.ledger.stage[rows,slot]==2)&(context.ledger.refusal==0))
            supported &= calm.span_matches(tape,captured.ticket,_vector(captured.side,b),
                _vector(captured.physical,b),move,captured.write,captured.context)
            span=(jnp.arange(tape.words.shape[1])[None,:]>=context.ledger.start[rows,slot,None])&(jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
            supported &= jnp.all(~span|((context.ledger.claimed==slot[:,None]+1)&jnp.all(context.ledger.words==tape.words,axis=2)
                &(context.ledger.event_turn==tape.event_turn)&(context.ledger.event_phase==tape.event_phase)
                &jnp.all(context.ledger.event_generation==tape.event_generation,axis=2)),axis=1)
    return supported


class CapturedSwordsDance(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    move:object
    write:object


def _captured_swords_dance_matches(tape,tables,captured):
    from . import engine_swords_dance_callbacks as swords
    b=tape.count.shape[0];supported=jnp.zeros(b,bool)
    if captured is None:return supported
    if type(captured) is not CapturedSwordsDance:raise ValueError('typed SAME-call Swords Dance capture required')
    context=accounting._context(tape)
    if context is None:return supported
    slot,same=accounting._ticket(context.ledger,captured.ticket);rows=jnp.arange(b);move=_vector(captured.move,b)
    supported=(same&jnp.any(tables.swords_dance_binding!=0)
        &jnp.all(captured.context.data.binding==tables.swords_dance_binding)
        &(move>0)&(move<len(tables.swords_dance_candidates))
        &tables.swords_dance_candidates[jnp.clip(move,0,len(tables.swords_dance_candidates)-1)]
        &(context.ledger.stage[rows,slot]==2)&(context.ledger.refusal==0))
    supported &= swords.span_matches(tape,captured.ticket,_vector(captured.side,b),_vector(captured.physical,b),move,captured.write,captured.context)
    span=(jnp.arange(tape.words.shape[1])[None,:]>=context.ledger.start[rows,slot,None])&(jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    return supported&jnp.all(~span|((context.ledger.claimed==slot[:,None]+1)&jnp.all(context.ledger.words==tape.words,axis=2)
        &(context.ledger.event_turn==tape.event_turn)&(context.ledger.event_phase==tape.event_phase)
        &jnp.all(context.ledger.event_generation==tape.event_generation,axis=2)),axis=1)


def observe_swords_dance_device(carry,tape,attempted,tables,*,active,captured=None):
    attempted=_vector(attempted,len(carry.stream_id),jnp.bool_)
    supported=_captured_swords_dance_matches(tape,tables,captured)
    # Exactly ONE original SELF_BOOST observation per actual loop. Integration
    # with another registered source unions read-only matches BEFORE _observe.
    return _observe(carry,tape,tables,Branch.SELF_BOOST,jnp.where(attempted&~supported,BOOST,0),active=active)


def swords_dance_on_tape(tape,attempted,*,active,captured=None):
    return _on_tape(tape,observe_swords_dance_device,attempted,active=active,captured=captured)


def observe_self_boost_device(carry,tape,attempted,tables,*,active,captured=None,captured_swords=None,captured_bulk_up=None,captured_self_boost=None):
    attempted=_vector(attempted,len(carry.stream_id),jnp.bool_)
    supported=_captured_calm_mind_matches(tape,tables,captured)|_captured_swords_dance_matches(tape,tables,captured_swords)
    from . import engine_bulk_up_callbacks as bulk_up
    supported |= bulk_up.captured_matches(tape,captured_bulk_up)
    from . import engine_self_boost_callbacks as self_boost
    supported |= self_boost.captured_matches(tape,captured_self_boost)
    return _observe(carry,tape,tables,Branch.SELF_BOOST,jnp.where(attempted&~supported,BOOST,0),active=active)


def self_boost_on_tape(tape,attempted,*,active,captured=None,captured_swords=None,captured_bulk_up=None,captured_self_boost=None):
    return _on_tape(tape,observe_self_boost_device,attempted,active=active,captured=captured,captured_swords=captured_swords,captured_bulk_up=captured_bulk_up,captured_self_boost=captured_self_boost)


class CapturedSpikes(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    move:object
    write:object


def observe_spikes_device(carry,tape,attempted,tables,*,active,captured=None):
    from . import engine_spikes_callbacks as spikes
    b=len(carry.stream_id);supported=jnp.zeros(b,bool)
    if captured is not None:
        if type(captured) is not CapturedSpikes:raise ValueError('typed SAME-call Spikes capture required')
        context=accounting._context(tape)
        if context is not None:
            slot,same=accounting._ticket(context.ledger,captured.ticket);rows=jnp.arange(b);move=_vector(captured.move,b)
            supported=(same&jnp.any(tables.spikes_binding!=0)&jnp.all(captured.context.data.binding==tables.spikes_binding)
                &(move>0)&(move<len(tables.spikes_candidates))&tables.spikes_candidates[jnp.clip(move,0,len(tables.spikes_candidates)-1)]
                &(context.ledger.stage[rows,slot]==2)&(context.ledger.refusal==0))
            supported &= spikes.span_matches(tape,captured.ticket,_vector(captured.side,b),_vector(captured.physical,b),move,captured.write,captured.context)
            at=jnp.clip(context.ledger.start[rows,slot],0,tape.words.shape[1]-1)
            supported &= ((context.ledger.claimed[rows,at]==slot+1)&jnp.all(context.ledger.words[rows,at]==tape.words[rows,at],axis=1)
                &(context.ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(context.ledger.event_phase[rows,at]==tape.event_phase[rows,at])
                &jnp.all(context.ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1))
    attempted=_vector(attempted,b,jnp.bool_)
    return _observe(carry,tape,tables,Branch.SPIKES,jnp.where(attempted&~supported,FIELD_VOLATILE,0),active=active)


def spikes_on_tape(tape,attempted,*,active,captured=None):
    return _on_tape(tape,observe_spikes_device,attempted,active=active,captured=captured)


def observe_branch_device(carry,tape,branch,attempted,tables,*,active,captured_belly_drum=None,captured_encore=None,captured_leech=None,captured_self_move=None,captured_heal_bell=None):
    """Literal named actual branch guard, NOT accepted/coverage/exhaustive.

    Root must place this at that branch's real cutpoint, including unsupported
    no-write/cap attempts. Calling it with False merely records a visited guard;
    it cannot certify the guard is complete or actually wired correctly.
    """
    if type(branch) is not Branch or int(branch)<6:raise ValueError('named actual unsupported write branch required')
    attempted=_vector(attempted,len(carry.stream_id),jnp.bool_)
    supported=jnp.zeros_like(attempted)
    if branch is Branch.SELF_DESTRUCT:
        from . import engine_self_destruct_callbacks as sd
        supported=sd.self_matches(tape,None if tape.execution is None else tape.execution.self_destruct)
    if branch is Branch.HEAL_BELL:
        from . import engine_heal_bell_callbacks as heal_bell_callbacks
        supported=heal_bell_callbacks.captured_matches(tape,captured_heal_bell)
    if branch is Branch.WISH_START:
        from . import engine_self_move_callbacks as self_move_callbacks
        supported=self_move_callbacks.captured_matches(tape,captured_self_move)
        if captured_self_move is not None:supported &= captured_self_move.pending.code==self_move_callbacks.WISH_START
    if branch is Branch.SEED_START:
        from . import engine_leech_seed_callbacks as leech_callbacks
        supported=leech_callbacks.captured_matches(tape,captured_leech)
    if branch is Branch.ENCORE:
        from . import engine_encore_callbacks as encore_callbacks
        supported=encore_callbacks.captured_matches(tape,captured_encore)
    if branch is Branch.BELLY_DRUM:
        from . import engine_belly_drum_callbacks as belly
        supported=belly.captured_matches(tape,captured_belly_drum)
    return _observe(carry,tape,tables,branch,jnp.where(attempted&~supported,tables.branch_risk[int(branch)],0),active=active)


def observe_end_device(carry,tape,state,tables,*,active):
    from . import engine_terminal_callbacks as terminal
    supported=terminal.matches_final_device(tape,state)
    return _observe(carry,tape,tables,Branch.FRAME_END,jnp.where(state.done&~supported,TERMINAL,0),active=active)


def commit_rows(active,before,after):
    """Explicit GLOBAL8 versus B-row carry; owning Core must actually call it."""
    if type(before) is not Carry or type(after) is not Carry:raise ValueError('typed kernel risk carries required')
    active=_vector(active,len(before.stream_id),jnp.bool_)
    return before._replace(**{name:jnp.where(active.reshape((len(active),)+(1,)*(old.ndim-1)),new,old)
        for name,old,new in zip(before._fields,before,after) if name not in ('binding','accounting_binding')})


def context_on_tape(tape):
    """Absent opt-in is inert; original execution ledger/tables are preserved."""
    execution=getattr(tape,'execution',None)
    context=getattr(execution,'risks',None)
    if context is not None and type(context) is not Context:
        raise ValueError('typed actual kernel risk context required')
    return context


def _on_tape(tape,observe,*args,**kwargs):
    context=context_on_tape(tape)
    if context is None:return tape
    carry=observe(context.carry,tape,*args,context.tables,**kwargs)
    active=_vector(kwargs['active'],len(carry.stream_id),jnp.bool_)
    return tape._replace(execution=tape.execution._replace(risks=context._replace(carry=carry)),
        refusal=tape.refusal|jnp.where(active&(carry.risk!=0),UNKNOWN_KERNEL_RISK,0))


def pre_move_on_tape(tape,state,side,before,*,eligible,active,captured_sleep=None,captured_before_move=None):
    return _on_tape(tape,observe_pre_move_device,state,side,before,eligible=eligible,active=active,captured_sleep=captured_sleep,captured_before_move=captured_before_move)


def selected_on_tape(tape,move,*,eligible,active,captured_focus=None,state=None,side=None,caller_move=None):
    return _on_tape(tape,observe_selected_device,move,eligible=eligible,active=active,captured_focus=captured_focus,state=state,side=side,caller_move=caller_move)


def outcome_on_tape(tape,outcome,*,eligible,active,captured_direct=None,pending_recoil=None,pending_ko=None,captured_no_damage=None,captured_substitute_damage=None,captured_protect_block=None):
    return _on_tape(tape,observe_outcome_device,outcome,eligible=eligible,active=active,captured_direct=captured_direct,pending_recoil=pending_recoil,pending_ko=pending_ko,captured_no_damage=captured_no_damage,captured_substitute_damage=captured_substitute_damage,captured_protect_block=captured_protect_block)


def observe_recoil_device(carry,tape,amount,tables,*,active,pending,before):
    from . import engine_recoil_callbacks as recoil
    amount=_vector(amount,len(carry.stream_id));needed=pending.requested|(amount>0)
    same=(recoil.matches_before(tape,pending,before)&(amount==pending.recoil)
        &jnp.all(tables.direct_hit_binding==pending.context.direct.data.binding)
        &jnp.any(tables.recoil_binding!=0)&jnp.all(tables.recoil_binding==pending.context.data.binding))
    return _observe(carry,tape,tables,Branch.RECOIL,jnp.where(needed&~same,SELF_HP,0),active=active)


def recoil_on_tape(tape,amount,*,active,pending,before):
    return _on_tape(tape,observe_recoil_device,amount,active=active,pending=pending,before=before)


def observe_self_heal_device(carry,tape,attempted,tables,*,active,captured=None,captured_self_move=None):
    from . import engine_self_heal_callbacks as self_heal
    from . import engine_self_move_callbacks as self_move_callbacks
    self_move_supported=self_move_callbacks.captured_matches(tape,captured_self_move)
    if captured_self_move is not None:self_move_supported &= (captured_self_move.pending.code==self_move_callbacks.REST_FAIL)|(captured_self_move.pending.code==self_move_callbacks.REST_STATUS)|(captured_self_move.pending.code==self_move_callbacks.REST_ASLEEP_FAIL)
    supported=self_heal.captured_matches(tape,captured)|self_move_supported
    return _observe(carry,tape,tables,Branch.SELF_HEAL,jnp.where(_vector(attempted,len(carry.stream_id),jnp.bool_)&~supported,SELF_EFFECT,0),active=active)


def self_heal_on_tape(tape,attempted,*,active,captured=None,captured_self_move=None):
    return _on_tape(tape,observe_self_heal_device,attempted,active=active,captured=captured,captured_self_move=captured_self_move)


def branch_on_tape(tape,branch,attempted,*,active,captured_belly_drum=None,captured_encore=None,captured_leech=None,captured_self_move=None,captured_heal_bell=None):
    return _on_tape(tape,observe_branch_device,branch,attempted,active=active,captured_belly_drum=captured_belly_drum,captured_encore=captured_encore,captured_leech=captured_leech,captured_self_move=captured_self_move,captured_heal_bell=captured_heal_bell)


def end_on_tape(tape,state,*,active):
    return _on_tape(tape,observe_end_device,state,active=active)
