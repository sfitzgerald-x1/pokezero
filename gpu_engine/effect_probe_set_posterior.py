"""Device mode's narrowing check (effect_probe.py --device --set-posterior): the device's standing pin against set_posterior.

With the run's device belief tables carrying the producer (device_set_posterior.enable), the device narrows every
opponent's standing pin in-tree at each damage reveal (engine_device_public_hooks.set_posterior_strike). This tree's B2
carries no set_posterior pin, so the B2 comparison's pin fields (belief_pin, visible_pin, belief_conflicts) are replaced
by this check. After every compared step, for each side s and present public slot k of the fold owned by side 1-s:

- posterior_support: the variants the encoder reads, the hard mask intersected with the visible standing pin when it is
  active (engine_device_belief.project), equals set_posterior's support, computed from the owner's channel lines with
  set_posterior's own strike parse and likelihood (parse_strikes, the single-variant skip, _counts, _encounter's
  off-model drop) over B2's hard mask (the hard support set_posterior's two-way check holds equal to B2's);
- posterior_active: the device's visible pin is active exactly when an admitted strike narrowed set_posterior's weights;
- posterior_conflicts: the device counted no pin conflict (set_posterior never empties a support).

The literal probe source is registered with device_set_posterior so its tables compile from the same variants. The
reference needs pokezero.set_posterior (the host producer); without it a --set-posterior run refuses at startup.
"""
from __future__ import annotations

import numpy as np

PIN_FIELDS = ('belief_pin', 'visible_pin', 'belief_conflicts')


def _words_to_bits(words):
    words = [int(w) for w in words]
    return [bool((words[i // 32] >> (i % 32)) & 1) for i in range(32 * len(words))]


class Narrowing:
    def __init__(self, catalog, source, dex, metadata):
        try:
            from pokezero import set_posterior
        except ImportError as error:
            raise RuntimeError('the narrowing probe needs pokezero.set_posterior, the host producer') from error
        from gpu_engine import device_set_posterior as dsp
        self.sp, self.dsp, self.catalog, self.source, self.dex = set_posterior, dsp, catalog, dsp.register_source(source), dex
        self.posterior = dsp.compile_tables(source, dex, metadata, source_tables=catalog.source)[0]
        self.lines = {}

    def tables(self, belief_tables):
        return self.dsp.enable(belief_tables, self.posterior)

    def feed(self, lane, step, channels):
        """The lane's owner channels (p1's, p2's) from battle start: reset at 'initial', extended at every step."""
        if step == 'initial' or lane not in self.lines:self.lines[lane] = ([], [])
        for owner in range(2):self.lines[lane][owner].extend(channels[owner] or ())

    def reference(self, lane, owner, species, hard_words):
        """set_posterior's (support bits, narrowed) for the opponent of `owner` holding `species` with B2's hard mask."""
        sp = self.sp
        lines = self.lines[lane][owner]
        seat = 'p1' if owner == 0 else 'p2'
        key = sp.species_key(species)
        universe = self.source.universe_for(species)
        variants = universe.variants if universe is not None else ()
        hard = tuple(_words_to_bits(hard_words)[:len(variants)])
        request = sp._first_request(lines) or {}
        from pokezero import tier2
        team = tier2.own_team_from_request(request)
        rows = []
        single = sum(hard) == 1
        for strike in sp.parse_strikes(lines, seat):
            if strike['attacker_key'] != key:continue
            reason, counts = strike['reason'], None
            if reason is None and single:reason = sp.SINGLE_VARIANT
            elif reason is None:reason, counts = sp._counts(strike, species, variants, hard, team, source=self.source, dex=self.dex)
            rows.append((strike, reason, counts))
        post = sp._encounter('p2' if owner == 0 else 'p1', seat, key, hard, rows, '')
        narrowed = any(s.admitted for s in post.strikes)
        return list(post.support) + [False] * (64 - len(post.support)), narrowed

    def diffs(self, trackers, b2_view, lane):
        """The narrowing diffs of one compared step (keys side:slot:field, as compare_views keys them)."""
        out = []
        vb = trackers.visible_belief
        for s in range(2):
            owner = 1 - s
            for k in range(6):
                mon = b2_view[s]['mons'][k]
                if mon is None:continue
                hard_words = mon['belief_hard']
                hard = np.asarray(_words_to_bits(hard_words))
                active = bool(vb.visible_pin_active[lane, s, k])
                pin = np.asarray(_words_to_bits(np.asarray(vb.visible_pin[lane, s, k])))
                device = hard & pin if active and (hard & pin).any() else hard
                support, narrowed = self.reference(lane, owner, mon['species'], hard_words)
                want = np.asarray(support)
                if not np.array_equal(device, want):
                    out.append(dict(field=f'{s}:{k}:posterior_support', b2=np.flatnonzero(want).tolist(),
                                    device=np.flatnonzero(device).tolist()))
                if active != narrowed:
                    out.append(dict(field=f'{s}:{k}:posterior_active', b2=narrowed, device=active))
                if int(vb.conflicts[lane, s, k]):
                    out.append(dict(field=f'{s}:{k}:posterior_conflicts', b2=0, device=int(vb.conflicts[lane, s, k])))
        return out
