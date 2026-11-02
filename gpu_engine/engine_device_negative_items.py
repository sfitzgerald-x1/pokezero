"""Public absent-Leftovers evidence, with the original pre-item HP snapshot.

Only actual switch/HP-line cutpoints supply these shown percentages. Endpoints
are not evidence for the HP at an earlier item slot. Unordered HP lines discard
the snapshot; upkeep consumes it once. No held item or private party is read.
"""
from typing import NamedTuple
import jax.numpy as j
from . import engine_device_trackers as tr

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
    percent = tr._percent(state.hp[r, side, physical], state.max_hp[r, side, physical])
    update = use & ((source_class == HP_ACTION) | (source_class == HP_PRE_ITEM))
    discard = use & (source_class != HP_ACTION) & (source_class != HP_PRE_ITEM) & (source_class != HP_POST_ITEM)
    values = carry.percent.at[r, side, at].set(j.where(update, percent, carry.percent[r, side, at]))
    known = carry.known.at[r, side, at].set(j.where(discard, False, update | carry.known[r, side, at]))
    return trackers._replace(negative_items=State(values, known))


def upkeep(trackers, belief_tables, leftovers, use):
    """Apply only the original Leftovers non-proc rule, then expire evidence.

    ``leftovers`` is the existing startup item-namespace vector, not a current
    holder lookup. Positive item reveals and mutations suppress this conclusion.
    Hard-mask inconsistency/fallback remains the original projector's concern.
    """
    carry = trackers.negative_items
    if carry is None:
        return trackers
    use = j.broadcast_to(j.asarray(use, bool), trackers.error.shape)
    hard = trackers.belief_mask
    if belief_tables is not None and leftovers is not None:
        if leftovers.shape != (belief_tables.item_mask.shape[1],) or leftovers.dtype != j.bool_:
            raise ValueError('Leftovers and belief item namespaces must match')
        excluded = j.bitwise_or.reduce(j.where(leftovers[None, :, None], belief_tables.item_mask, j.uint32(0)), axis=1)
        r = j.arange(len(trackers.error))
        for side in range(2):
            slot = trackers.active_slot[r, side]; at = j.clip(slot, 0, 5)
            species = trackers.species[r, side, at]
            changed = trackers.item_removed[r, side, at]
            if trackers.mon_features is not None and 'item_swapped' in trackers.mon_features:
                changed |= trackers.mon_features['item_swapped'][r, side, at] != 0
            valid = use & (slot >= 0) & (slot < 6) & trackers.present[r, side, at]
            valid &= (species > 0) & (species < excluded.shape[0])
            valid &= (trackers.hp_percent[r, side, at] > 0) & ~trackers.fainted[r, side, at]
            valid &= carry.known[r, side, at] & (carry.percent[r, side, at] < 100)
            valid &= (trackers.shown_item[r, side, at] == 0) & ~changed
            bits = excluded[j.clip(species, 0, excluded.shape[0] - 1)]
            old = hard[r, side, at]
            hard = hard.at[r, side, at].set(j.where(valid[:, None], old & ~bits, old))
    # No new HP publication next turn means no new evidence, even if HP stayed low.
    carry = carry._replace(known=j.where(use[:, None, None], False, carry.known))
    return trackers._replace(belief_mask=hard, negative_items=carry)
