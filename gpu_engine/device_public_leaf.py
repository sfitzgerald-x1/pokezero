"""Opt-in Arm D leaf observation through the same direct public29 encoder.

Only root PUBLIC knowledge is cloned for sampled worlds. The module never
initializes knowledge from a sampled world and never reads opponent private
state. A caller may advance the returned public carry with step_with_public,
then use from_public on that actual resulting carry. Provenance of an advanced
carry belongs to that caller; a copied array does not prove its origin.

This module does not construct worlds, run transitions/models, or qualify D.
"""
from typing import NamedTuple

from . import engine_device_observation as observation
from . import engine_device_trackers as public


class Leaf(NamedTuple):
    public: object
    observation: object
    inputs: object
    observer_sides: object


def _enabled(enable_device_leaf, legacy_champion):
    if enable_device_leaf is not True:
        raise ValueError('explicit direct-device leaf enablement required')
    if type(legacy_champion) is not bool:
        raise ValueError('legacy champion layout requires an explicit bool')


def from_roots(world_state, root_public, root_rows, tables, *, observer_sides,
               legal_action_mask, enable_device_leaf=False, legacy_champion=False):
    """Gather/repeat genuine root knowledge BEFORE observing sampled worlds.

    root_rows is the original int32 root index of each flattened world. Legal
    masks are supplied explicitly by the caller, without a hidden-world lookup
    in this adapter. All source encoder rows/error flags remain present.
    """
    _enabled(enable_device_leaf, legacy_champion)
    cloned = public.clone_for_worlds(root_public, root_rows)
    return from_public(world_state, cloned, tables, observer_sides=observer_sides,
        legal_action_mask=legal_action_mask, enable_device_leaf=True,
        legacy_champion=legacy_champion)


def from_public(world_state, world_public, tables, *, observer_sides,
                legal_action_mask, enable_device_leaf=False, legacy_champion=False):
    """Observe root-cloned knowledge, possibly advanced by actual public steps.

    Start with from_roots. Subsequent world_public must be its returned carry or
    the actual carry returned by step_with_public; never initialize a tracker
    from sampled private opponent state. Own sampled-world party data is legal
    input to the SAME encoder, exactly as for the real owning player.
    """
    _enabled(enable_device_leaf, legacy_champion)
    view = public.readout(world_public, observer_sides)
    encoded = observation.encode(world_state, view, tables,
        observer_sides=observer_sides, legal_action_mask=legal_action_mask)
    inputs = (observation.champion_inputs(encoded) if legacy_champion else
        observation.actor_inputs(encoded))
    return Leaf(world_public, encoded, inputs, observer_sides)


def map_actions(leaf, world_state, selected):
    """Use the exact real-encoder request -> engine dispatch mapping."""
    return observation.map_actions(leaf.observation, world_state, leaf.observer_sides, selected)
