"""Public absent-Leftovers evidence, with the original pre-item HP snapshot.

Only actual switch/HP-line cutpoints supply these shown percentages. Endpoints
are not evidence for the HP at an earlier item slot. Unordered HP lines discard
the snapshot; upkeep consumes it once. No held item or private party is read.
"""
from typing import NamedTuple
import jax.numpy as j
from . import engine_device_trackers as tr
from . import engine_device_visible_belief as visible_belief
from . import lane_index as _li

HP_ACTION = 0
HP_PRE_ITEM = 1
HP_POST_ITEM = 2
HP_UNKNOWN = 3


class State(NamedTuple):
    percent: object                 # [B,2,6], public encounter coordinates
    known: object                   # [B,2,6], current-turn pre-slot evidence


def empty(batch):
    return State(j.zeros((batch, 2, 6), j.int32), j.zeros((batch, 2, 6), bool))


def hp_line(trackers, state, side, use, *, source_class):
    """One actual HP line, or the condition on an actual switch line."""
    carry = trackers.negative_items
    if carry is None:
        return trackers
    r = j.arange(len(trackers.error)); physical = _li.take(state.active,r,side)
    slot = _li.take(trackers.physical_to_public,r,side,j.clip(physical, 0, 5))
    at = j.clip(slot, 0, 5)
    use = j.broadcast_to(j.asarray(use, bool), r.shape) & (slot >= 0) & (slot < 6)
    use &= (physical >= 0) & (physical < 6)
    hp = _li.take(state.hp,r,side,physical); maximum = _li.take(state.max_hp,r,side,physical)
    # _percent clamps for display. A malformed condition is not a known HP
    # observation and must not preserve prior evidence, even after the slot.
    valid = (maximum > 0) & (hp >= 0) & (hp <= maximum)
    percent = tr._percent(hp, maximum)
    update = use & valid & ((source_class == HP_ACTION) | (source_class == HP_PRE_ITEM))
    discard = use & (~valid | ((source_class != HP_ACTION) & (source_class != HP_PRE_ITEM) & (source_class != HP_POST_ITEM)))
    values = _li.put(carry.percent,r,(side,at),j.where(update, percent, _li.take(carry.percent,r,side,at)))
    known = _li.put(carry.known,r,(side,at),j.where(discard, False, update | _li.take(carry.known,r,side,at)))
    return trackers._replace(negative_items=State(values, known))


def upkeep(trackers, belief_tables, leftovers, use, *, current_state=None):
    """The original upkeep non-proc rules, Leftovers, Lum and the pinch berries, then expire evidence.

    B2's sweep (gpu_engine/public_item_nonproc.py sweep, its three triggers at :171-172) and the champion's Python
    belief (src/pokezero/belief.py _sweep_end_of_turn_non_procs: 'Status persisted without an instant Lum cure' and
    'Action-phase HP at or below 25% with no pinch-berry activation') run, at |upkeep, for each side's living active whose item is unrevealed and
    unmutated, in this order: Leftovers (a known pre-slot HP snapshot below full and no Leftovers heal), Lum Berry (any
    status: Showdown data/items.ts lumberry eats at onAfterSetStatus and onUpdate while statused, so a holder cannot
    keep a status to the upkeep) and Salac/Petaya/Liechi as one rule (the snapshot at or below a quarter and no [eat]:
    data/mods/gen3/items.ts eats them at the residual 10/4 when hp <= maxhp/4; an [eat] reveals the item, which the
    eligibility already excludes). Each rule refreshes once when it rules out a new name (visible_belief.exclude).

    ``leftovers`` is the existing startup item-namespace vector, not a current holder lookup; the Lum and pinch IDs
    come from belief_tables.nonproc_items (engine_device_belief.compile_tables; tables without it skip those rules).
    Positive item reveals and mutations suppress these conclusions. Hard-mask inconsistency/fallback remains the
    original projector's concern.
    """
    carry = trackers.negative_items
    if carry is None:
        return trackers
    use = j.broadcast_to(j.asarray(use, bool), trackers.error.shape)
    nonproc = None if belief_tables is None else getattr(belief_tables, 'nonproc_items', None)
    if belief_tables is not None and (leftovers is not None or nonproc is not None):
        if leftovers is not None and (leftovers.shape != (belief_tables.item_mask.shape[1],) or leftovers.dtype != j.bool_):
            raise ValueError('Leftovers and belief item namespaces must match')
        r = j.arange(len(trackers.error))
        for side in range(2):
            slot = _li.take(trackers.active_slot,r,side); at = j.clip(slot, 0, 5)
            species = _li.take(trackers.species,r,side,at)
            changed = _li.take(trackers.item_removed,r,side,at)
            # Fidelity M1: B2's non-proc sweep skips a mutated item (public_item_nonproc: ~items.mutated), the
            # per-mon ledger flag item_mutated, which outlives the seat's per-stint item_swapped (an arrival
            # resets that one; it implies a mutation, so a carry without the ledger still uses it).
            for mutation in ('item_mutated', 'item_swapped'):
                if trackers.mon_features is not None and mutation in trackers.mon_features:
                    changed |= _li.take(trackers.mon_features[mutation],r,side,at) != 0
            valid = use & (slot >= 0) & (slot < 6) & _li.take(trackers.present,r,side,at)
            # Validate the current condition itself, not its clamped display
            # or unrelated sticky identity/status errors. No current operands
            # means no absence conclusion. Exact values are not retained.
            if current_state is None:
                valid &= False
            else:
                physical = _li.take(current_state.active,r,side)
                hp = _li.take(current_state.hp,r,side,physical)
                maximum = _li.take(current_state.max_hp,r,side,physical)
                valid &= (maximum > 0) & (hp >= 0) & (hp <= maximum)
            valid &= (species > 0) & (species < belief_tables.item_mask.shape[0])
            valid &= (_li.take(trackers.hp_percent,r,side,at) > 0) & ~_li.take(trackers.fainted,r,side,at)
            valid &= (_li.take(trackers.shown_item,r,side,at) == 0) & ~changed
            known = _li.take(carry.known,r,side,at); percent = _li.take(carry.percent,r,side,at)
            physical = _li.take(trackers.active_physical,r,side); b = len(use)
            if leftovers is not None:
                # This vector names the one canonical item, not a private holder.
                item = j.argmax(leftovers).astype(j.int32)
                trackers = visible_belief.exclude(trackers,belief_tables,side,physical,j.full(b,item,j.int32),
                    valid & known & (percent < 100) & (j.sum(leftovers.astype(j.int32)) == 1),kind='item')
            if nonproc is not None:
                ids = j.asarray(nonproc, j.int32)
                statused = _li.take(trackers.status,r,side,at) != 0
                trackers = visible_belief.exclude(trackers,belief_tables,side,physical,j.full(b,ids[0],j.int32),
                    valid & statused & (ids[0] > 0),kind='item')
                trackers = visible_belief.exclude(trackers,belief_tables,side,physical,j.broadcast_to(ids[1:],(b,3)),
                    valid & known & (percent <= 25) & (ids[1] > 0),kind='item')
    # No new HP publication next turn means no new evidence, even if HP stayed low.
    carry = carry._replace(known=j.where(use[:, None, None], False, carry.known))
    return trackers._replace(negative_items=carry)
