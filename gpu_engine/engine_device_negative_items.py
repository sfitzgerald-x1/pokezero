"""Public absent-Leftovers evidence, with the original pre-item HP snapshot.

Only actual switch/HP-line cutpoints supply these shown percentages. Endpoints
are not evidence for the HP at an earlier item slot. Unordered HP lines discard
the snapshot; upkeep consumes it once. No held item or private party is read.
"""
from typing import NamedTuple
import jax.numpy as j
from . import engine_device_trackers as tr
from . import engine_device_visible_belief as visible_belief

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
    r = j.arange(len(trackers.error)); physical = state.active[r, side]
    slot = trackers.physical_to_public[r, side, j.clip(physical, 0, 5)]
    at = j.clip(slot, 0, 5)
    use = j.broadcast_to(j.asarray(use, bool), r.shape) & (slot >= 0) & (slot < 6)
    use &= (physical >= 0) & (physical < 6)
    hp = state.hp[r, side, physical]; maximum = state.max_hp[r, side, physical]
    # _percent clamps for display. A malformed condition is not a known HP
    # observation and must not preserve prior evidence, even after the slot.
    valid = (maximum > 0) & (hp >= 0) & (hp <= maximum)
    percent = tr._percent(hp, maximum)
    update = use & valid & ((source_class == HP_ACTION) | (source_class == HP_PRE_ITEM))
    discard = use & (~valid | ((source_class != HP_ACTION) & (source_class != HP_PRE_ITEM) & (source_class != HP_POST_ITEM)))
    values = carry.percent.at[r, side, at].set(j.where(update, percent, carry.percent[r, side, at]))
    known = carry.known.at[r, side, at].set(j.where(discard, False, update | carry.known[r, side, at]))
    return trackers._replace(negative_items=State(values, known))


def upkeep(trackers, belief_tables, leftovers, use, *, current_state=None):
    """Apply only the original Leftovers non-proc rule, then expire evidence.

    ``leftovers`` is the existing startup item-namespace vector, not a current
    holder lookup. Positive item reveals and mutations suppress this conclusion.
    Hard-mask inconsistency/fallback remains the original projector's concern.
    """
    carry = trackers.negative_items
    if carry is None:
        return trackers
    use = j.broadcast_to(j.asarray(use, bool), trackers.error.shape)
    if belief_tables is not None and leftovers is not None:
        if leftovers.shape != (belief_tables.item_mask.shape[1],) or leftovers.dtype != j.bool_:
            raise ValueError('Leftovers and belief item namespaces must match')
        # This vector names the one canonical item, not a private holder.
        item=j.argmax(leftovers).astype(j.int32)
        r = j.arange(len(trackers.error))
        for side in range(2):
            slot = trackers.active_slot[r, side]; at = j.clip(slot, 0, 5)
            species = trackers.species[r, side, at]
            changed = trackers.item_removed[r, side, at]
            if trackers.mon_features is not None and 'item_swapped' in trackers.mon_features:
                changed |= trackers.mon_features['item_swapped'][r, side, at] != 0
            valid = use & (slot >= 0) & (slot < 6) & trackers.present[r, side, at]
            # Validate the current condition itself, not its clamped display
            # or unrelated sticky identity/status errors. No current operands
            # means no absence conclusion. Exact values are not retained.
            if current_state is None:
                valid &= False
            else:
                physical = current_state.active[r, side]
                hp = current_state.hp[r, side, physical]
                maximum = current_state.max_hp[r, side, physical]
                valid &= (maximum > 0) & (hp >= 0) & (hp <= maximum)
            valid &= (species > 0) & (species < belief_tables.item_mask.shape[0])
            valid &= (trackers.hp_percent[r, side, at] > 0) & ~trackers.fainted[r, side, at]
            valid &= carry.known[r, side, at] & (carry.percent[r, side, at] < 100)
            valid &= (trackers.shown_item[r, side, at] == 0) & ~changed
            valid &= j.sum(leftovers.astype(j.int32))==1
            trackers=visible_belief.exclude(trackers,belief_tables,side,trackers.active_physical[r,side],
                j.full(len(use),item,j.int32),valid,kind='item')
    # No new HP publication next turn means no new evidence, even if HP stayed low.
    carry = carry._replace(known=j.where(use[:, None, None], False, carry.known))
    return trackers._replace(negative_items=carry)
