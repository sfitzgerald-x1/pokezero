"""Strict narrowing-only regression attribution, independent of the general effect/cascade allowances.

Admission mathematics remain certified by host corpus parity. This audit uses the producer's returned reduction to
check its write scope at each invocation, then checks every changed output against the exact baseline and witnessed
public slots. A later refused call cannot borrow an earlier call's admission, even if its write is subsequently undone.
"""
from __future__ import annotations

import numpy as np

PIN_LEAVES = tuple('trackers.visible_belief.' + n for n in
                   ('pin', 'pin_active', 'visible_pin', 'visible_pin_active', 'conflicts'))


def validate_options(plugins, *, waive=(), faint_clear=False):
    if 'NAR' in plugins and (plugins != ['NAR'] or waive or faint_clear):
        raise ValueError('NAR requires its own pass, without other plug-ins, waivers or faint-clear rules')


def unexpected(entries, post, base, admitted, bad=False):
    """One lane: only five pin leaves, with every differing public slot witnessed at its damage event."""
    admitted = np.asarray(admitted, bool)
    if admitted.shape != (2, 6):raise ValueError('narrowing evidence must have shape (2, 6)')
    failures = []
    for leaf, side in sorted(entries, key=str):
        if leaf not in PIN_LEAVES or side not in (0, 1) or leaf not in post or leaf not in base:
            failures.append((leaf, side));continue
        a, b = np.asarray(post[leaf]), np.asarray(base[leaf])
        if a.shape != b.shape or a.shape[:2] != (2, 6) or a.dtype != b.dtype:
            failures.append((leaf, side));continue
        changed = (a[side] != b[side]).reshape(6, -1).any(-1)
        # A digest difference with no corresponding witness difference is also an invalid attribution.
        if not changed.any() or (changed & ~admitted[side]).any():failures.append((leaf, side))
    if bad:failures.append(('narrowing_evidence', None))
    return failures


class Recorder:
    """Optional JAX trace-time wrappers of the strike hook and in_tree (ordinary damage and Pursuit).

    No extra likelihood evaluation: consume the returned reduction, not changed pin bits. State and pre-call tracker
    arrays are immutable snapshots; the hook's mutable Context is never used as the before image. All tracker leaves
    are audited per invocation, including forbidden non-pin leaves. The final strict comparison also covers state.
    """
    def __init__(self, lanes):
        import jax.numpy as j
        self.lanes = lanes
        self.admitted = j.zeros((lanes, 2, 6), bool)
        self.bad = j.zeros(lanes, bool)
        self.calls = 0
        self.subject = None

    def install(self):
        from gpu_engine import device_set_posterior as dsp, engine_device_public_hooks as hooks
        self.original = dsp.in_tree
        self.hook = hooks.set_posterior_strike
        dsp.in_tree = self.record
        hooks.set_posterior_strike = self.record_hook

    def restore(self):
        from gpu_engine import device_set_posterior as dsp, engine_device_public_hooks as hooks
        dsp.in_tree = self.original
        hooks.set_posterior_strike = self.hook

    @staticmethod
    def named(trackers):
        import jax
        return {'trackers' + jax.tree_util.keystr(p): x
                for p, x in jax.tree_util.tree_flatten_with_path(trackers)[0]}

    def record_hook(self, c, state, side, move, caller_move, critical, hits_sub, endured, dealt,
                    hp_before, hp_after, num, den, use, **kwargs):
        import jax.numpy as j
        before = self.named(c.trackers)  # Context._replace mutates c; retain immutable pre-call arrays.
        self.subject = (c.trackers, c.belief_tables, state, side, use)
        self.call_admitted = j.zeros_like(self.admitted)
        out = self.hook(c, state, side, move, caller_move, critical, hits_sub, endured, dealt,
                        hp_before, hp_after, num, den, use, **kwargs)
        self.audit(before, self.named(out.trackers), self.call_admitted)
        self.subject = None
        return out

    def record(self, trackers, tables, state, side, move, caller_move, critical, hits_sub, endured, dealt,
               hp_before, hp_after, num, den, use, **kwargs):
        import jax.numpy as j
        before = self.named(trackers)
        if self.subject is None:raise ValueError('narrowing producer called outside its damage hook')
        out, reduced = self.original(trackers, tables, state, side, move, caller_move, critical, hits_sub, endured,
                                     dealt, hp_before, hp_after, num, den, use, **kwargs)
        # Admission is tied to the original outer hook subject/use, so a widened inner call cannot move that subject.
        trackers, tables, state, side, use = self.subject
        r = j.arange(self.lanes);side = j.broadcast_to(j.asarray(side, j.int32), (self.lanes,))
        seat = j.clip(side, 0, 1);own = state.active[r, seat];physical = j.clip(own, 0, 5)
        slot = trackers.physical_to_public[r, seat, physical];safe_slot = j.clip(slot, 0, 5)
        species = trackers.species[r, seat, safe_slot]
        source = tables.source_species[j.clip(species, 0, len(tables.source_species) - 1)]
        valid = ((side >= 0) & (side < 2) & (own >= 0) & (own < 6) & (slot >= 0) & (slot < 6)
                 & trackers.present[r, seat, safe_slot]
                 & (species > 0) & (species < len(tables.source_species)) & (source > 0))
        admitted = (j.broadcast_to(j.asarray(use, bool), (self.lanes,)) & valid
                    & (reduced.reason == 0) & j.any(reduced.fit != 0, axis=-1))
        at = ((j.arange(2)[None, :, None] == side[:, None, None])
              & (j.arange(6)[None, None, :] == slot[:, None, None]) & admitted[:, None, None])
        self.admitted = self.admitted | at
        self.call_admitted = self.call_admitted | at
        self.audit(before, self.named(out), at)
        self.calls += 1
        return out, reduced

    def audit(self, before, after, at):
        import jax.numpy as j
        if before.keys() != after.keys():raise ValueError('narrowing changed the tracker leaf structure')
        for leaf, old in before.items():
            new = after[leaf]
            if old is new:continue
            if old.shape != new.shape or old.dtype != new.dtype:raise ValueError('narrowing changed leaf type: ' + leaf)
            diff = old != new
            if leaf in PIN_LEAVES:
                if diff.shape[:3] != (self.lanes, 2, 6):raise ValueError('invalid pin shape: ' + leaf)
                diff = diff.reshape(self.lanes, 2, 6, -1).any(-1) & ~at
            if diff.ndim == 0:self.bad = self.bad | diff
            elif diff.shape[0] == self.lanes:self.bad = self.bad | diff.reshape(self.lanes, -1).any(-1)
            else:self.bad = self.bad | j.any(diff)
    def result(self):
        return dict(admitted=self.admitted, bad=self.bad, calls=np.int32(self.calls))
