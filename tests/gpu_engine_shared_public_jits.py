"""One compiled causal projection and one compiled V4 fold per tables/catalog object. Test helper, not a test module.

Public-consumer component tests compare the full projector against Showdown reference lines folded through
v4_encoder.fold_chunk, and lower frames with the causal projector, both scan-shaped. Called eagerly, each
call compiles its scan body again (a fresh closure per call), so a module paid one compile per call. These
shared jitted programs compile once per (program, tables or catalog object) and process; inputs, outputs and
assertions are unchanged. The cache keys on object identity: do not use it for tests that patch engine
internals or mutate tables/catalog arrays in place after the first call.
"""
from unittest.mock import patch

import jax

from gpu_engine import engine_full_v4_event_device as full, v4_encoder as v4

_HELD={}


def _one(kind,obj,build):
    hit=_HELD.get((kind,id(obj)))
    if hit is None or hit[0] is not obj:hit=_HELD[(kind,id(obj))]=(obj,build())
    return hit[1]


def fold(catalog):
    """v4_encoder.fold_chunk(state, chunk, catalog), compiled once per catalog."""
    return _one('fold',catalog,lambda:jax.jit(lambda state,chunk:v4.fold_chunk(state,chunk,catalog)))


def lower(tables):
    """The full adapter's causal project_frame(state, frame, tables), compiled once per tables object."""
    return _one('lower',tables,lambda:jax.jit(lambda state,frame:full.causal.project_frame(state,frame,tables)))


def project(tables):
    """The full adapter's project_frame(state, frame, tables), compiled once per tables object."""
    return _one('project',tables,lambda:jax.jit(lambda state,frame:full.project_frame(state,frame,tables)))


class _Routed:
    """A module stand-in: the given attributes overridden, everything else delegated."""
    def __init__(self,module,**overrides):
        self._module=module;vars(self).update(overrides)

    def __getattr__(self,name):
        return getattr(self._module,name)


def compare(fixture_module):
    """StatusCallbacksTests.compare with its causal projection and reference folds on the shared programs.

    Only the fixture module's own `full`/`v4` names are swapped, and only during the call; a routing failure
    (renamed aliases, or compare no longer calling them) raises.
    """
    original=fixture_module.StatusCallbacksTests.compare
    def routed(self,state,frame,current,wire):
        calls=[]
        def project_frame(state_,frame_,tables):
            calls.append('lower');return lower(tables)(state_,frame_)
        def fold_chunk(state_,chunk,catalog):
            calls.append('fold');return fold(catalog)(state_,chunk)
        causal=_Routed(fixture_module.full.causal,project_frame=project_frame)
        with patch.object(fixture_module,'full',_Routed(fixture_module.full,causal=causal)),\
                patch.object(fixture_module,'v4',_Routed(fixture_module.v4,fold_chunk=fold_chunk)):
            out=original(self,state,frame,current,wire)
        if calls.count('lower')!=1 or 'fold' not in calls:
            raise AssertionError('StatusCallbacksTests.compare did not run through the shared programs')
        return out
    return routed
