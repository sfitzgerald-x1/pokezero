"""Pinned Gen3 request/choice algebra from literal Native evidence only.

The owning Python/B2 request mask and Native numeric acceptance are different
surfaces. Neither is substituted for the other. This builds a selection plan;
actual dispatch-slot history, full physical agreement and owner-stream binding
remain separate gates. No Native getter, GPU state, RNG or encoder is used.
"""
from dataclasses import dataclass
import math
import re

from .fresh200_native_identity import PartyMap


class _Unknown:
    pass


UNKNOWN = _Unknown()


@dataclass(frozen=True)
class LockResults:
    """Actual passive results at the frozen boundary, never guessed defaults.

    A restricted SemiLockMove query skipped by maybeLocked is UNKNOWN, not
    None. A semi-lock's physical slot must come from the Native causal ledger;
    collapsed request move IDs cannot establish it, even with duplicate IDs.
    """
    hard: object = UNKNOWN
    semi: object = UNKNOWN
    semi_slot: object = UNKNOWN


def _owner(owner):
    if type(owner) is not int or owner not in (0, 1):
        raise ValueError('literal absolute Native owner required')


def _request(request, owner):
    _owner(owner)
    if request is None:
        return {}, [], [], False
    if type(request) is not dict:
        raise ValueError('literal Native request required')
    side = request.get('side') or {}
    if type(side) is not dict or side.get('id') not in (None, f'p{owner+1}'):
        raise ValueError('owning Native request belongs to another seat')
    team = side.get('pokemon') or []
    active = request.get('active') or []
    if (type(team) is not list or len(team) > 6 or any(type(x) is not dict for x in team)
            or type(active) is not list or len(active) > 1 or any(type(x) is not dict for x in active)):
        raise ValueError('literal singles Native team/active required')
    force = request.get('forceSwitch')
    if force is not None and (type(force) is not list or len(force) > 1):
        raise ValueError('literal singles forced replacement required')
    moves = active[0].get('moves') or [] if active else []
    if type(moves) is not list or len(moves) > 4:
        raise ValueError('literal positional Native move rows required')
    return request, team, moves, bool(force and any(bool(x) for x in force))


def owner_request_mask(request, *, owner):
    """Literal original Python/B2 nine bits, in REQUEST roster coordinates.

    PP alone never disables a row. A positional invalid move row stays absent;
    maybeTrapped is an advertised trap. This is a contract helper, not a new
    reference encoder or a replacement for running the literal Python/B2 path.
    """
    raw, team, moves, forced = _request(request, owner)
    mask = [False]*9
    if request is None or raw.get('wait', False):
        return mask
    active_rows = raw.get('active') or []
    active = active_rows[0] if active_rows else {}
    if not forced:
        for slot, move in enumerate(moves):
            mask[slot] = type(move) is dict and not bool(move.get('disabled', False))
    indices = [i for i, mon in enumerate(team) if mon.get('active')]
    if not indices:
        return mask
    active_index = indices[0]  # original argmax behavior, not a physical identity join
    trapped = active.get('trapped') is True or active.get('maybeTrapped') is True
    switching = forced or bool(active_rows) and not trapped
    for lane in range(5):
        target = lane + (lane >= active_index)
        if switching and target < len(team):
            mon = team[target]
            mask[4+lane] = not bool(mon.get('active')) and not str(mon.get('condition') or '').startswith('0 ')
    return mask


def _id(value):
    if type(value) is not str or re.fullmatch('[a-z0-9]+', value) is None:
        raise ValueError('literal Native move ID required')
    return value


def _lock(value):
    if value is UNKNOWN:
        raise ValueError('missing actual Native lock result; no proactive query permitted')
    return None if value is None else _id(value)


def native_action_relation(request, party: PartyMap, *, owner, locks: LockResults):
    """Enumerate all nine physical selection lanes and numeric aliases.

    Ordinary Gen3 duplicate IDs use ANY unrestricted enabled row for numeric
    acceptance, as Native chooseMove does. The selected slot is not attestation
    of Native getMoveData/PP/dispatch history. Hard/semi locks take precedence
    over ordinary PP/Disable/Choice filters. No selected-action-only admission.
    """
    raw, team, moves, forced = _request(request, owner)
    if not isinstance(locks, LockResults):
        raise ValueError('explicit passive Native lock results required')
    result = dict(schema='pokezero.fresh200.native_action_relation.v1', owner=owner,
        owner_request_mask=owner_request_mask(request, owner=owner), physical_mask=[False]*9,
        physical_commands=[None]*9, numeric_relations=[], qualification=False,
        dispatch_history_verified=False, owner_stream_boundary_verified=False)
    if request is None or raw.get('wait', False) or party.battle.get('ended') is True:
        result['kind'] = 'inactive'
        return result
    if len(team) != 6:
        raise ValueError('complete six-member owning Native request required')
    active = party.active(owner)
    current_active = party.original_to_current[owner][active]
    flags = [i for i, mon in enumerate(team) if mon.get('active')]
    if flags != [current_active]:
        raise ValueError('request active/current Native roster join differs')
    mon = party.mon(owner, active)
    if not (type(mon.get('trapped')) is bool or mon.get('trapped') == 'hidden'):
        raise ValueError('actual Native trapped cache required')
    for lane, target in enumerate(x for x in range(6) if x != active):
        candidate = party.mon(owner, target)
        if type(candidate.get('fainted')) is not bool:
            raise ValueError('actual Native faint cache required')
        if (forced or not mon['trapped']) and not candidate['fainted']:
            command = 'switch '+str(party.original_to_current[owner][target]+1)
            result['physical_mask'][4+lane] = True
            result['physical_commands'][4+lane] = command
            result['numeric_relations'].append(dict(command=command, kind='switch', original_target=target, physical_lanes=[4+lane]))
    if forced:
        result['kind'] = 'forced-replacement'
        return result
    active_rows = raw.get('active') or []
    if len(active_rows) != 1 or not moves or any(type(x) is not dict for x in moves):
        raise ValueError('actual actionable Native move request required')
    slots = mon.get('moveSlots')
    if type(slots) is not list or not 1 <= len(slots) <= 4 or any(type(x) is not dict for x in slots):
        raise ValueError('actual current Native move slots required, including Transform')
    ids = [_id(x.get('id')) for x in slots]
    request_ids = [_id(x.get('id')) for x in moves]
    hard = _lock(locks.hard)
    if hard:
        if (hard not in ('recharge', 'solarbeam') or request_ids != [hard] or mon['trapped'] is not True
                or active_rows[0].get('trapped') is not True):
            raise ValueError('unsupported or inconsistent Native hard-lock request')
        lanes = [0] if hard == 'recharge' else [i for i, move in enumerate(ids) if move == hard]
        if not lanes:
            lanes = [0]
            result['hard_slot_fallback'] = 'existing SolarBeam physical-lane0 fallback; not executed-slot evidence'
        result['physical_mask'][4:] = [False]*5
        result['physical_commands'][4:] = [None]*5
        result['numeric_relations'] = []
        result['kind'] = 'hard-lock'
        for lane in lanes:
            result['physical_mask'][lane] = True
            result['physical_commands'][lane] = 'move 1'
        result['numeric_relations'].append(dict(command='move 1', kind=hard, physical_lanes=lanes))
        return result
    semi = _lock(locks.semi)
    if semi:
        slot = locks.semi_slot
        if type(slot) is not int or not 0 <= slot < len(slots) or ids[slot] != semi:
            raise ValueError('semi-lock requires exact causal Native physical slot')
        collapsed = request_ids == [semi]
        if not collapsed and request_ids != ids:
            raise ValueError('semi-lock request slots differ from actual Native slots')
        command = 'move '+str(1 if collapsed else slot+1)
        result['kind'] = 'semi-lock'
        result['physical_mask'][slot] = True
        result['physical_commands'][slot] = command
        # Native accepts any parsed numeric request row, then overrides the ID.
        for index in range(len(moves)):
            result['numeric_relations'].append(dict(command='move '+str(index+1), kind='semi-lock',
                request_slot=index, move_id=semi, physical_lanes=[slot]))
        return result
    enabled = set()
    for move, move_id in zip(slots, ids):
        pp, disabled = move.get('pp'), move.get('disabled')
        if (type(pp) not in (int, float) or not math.isfinite(pp) or pp < 0
                or not (type(disabled) is bool or type(disabled) is str and disabled == 'hidden')):
            raise ValueError('literal Native PP/Disable cache required')
        if pp > 0 and disabled is False:
            enabled.add(move_id)
    if not enabled:
        if request_ids != ['struggle'] and request_ids != ids:
            raise ValueError('unrestricted Struggle numeric aliases lack matching Native request')
        result['kind'] = 'struggle'
        result['physical_mask'][0] = True
        result['physical_commands'][0] = 'move 1'
        for index in range(len(moves)):
            result['numeric_relations'].append(dict(command='move '+str(index+1), kind='struggle', request_slot=index, physical_lanes=[0]))
        return result
    if request_ids != ids:
        raise ValueError('ordinary Native request/current physical slots differ')
    result['kind'] = 'ordinary'
    for slot, move_id in enumerate(ids):
        if move_id in enabled:
            command = 'move '+str(slot+1)
            result['physical_mask'][slot] = True
            result['physical_commands'][slot] = command
            result['numeric_relations'].append(dict(command=command, kind='move', request_slot=slot,
                move_id=move_id, physical_lanes=[slot], execution_slot_requires_causal_witness=True))
    return result
