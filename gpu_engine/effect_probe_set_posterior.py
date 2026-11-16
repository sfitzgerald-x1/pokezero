"""Device mode's narrowing check (effect_probe.py --device --set-posterior): the device's standing pin against set_posterior.

With the run's device belief tables carrying the producer (device_set_posterior.enable), the device narrows every
opponent's standing pin in-tree at each damage reveal (engine_device_public_hooks.set_posterior_strike). This tree's B2
carries no set_posterior pin, so the B2 comparison's pin fields (belief_pin, visible_pin, belief_conflicts) are replaced
by this check. After every compared step, for each side s and present public slot k of the fold owned by side 1-s:

- posterior_support: the variants the encoder reads, the hard mask intersected with the visible standing pin when it is
  active (engine_device_belief.project), equals set_posterior's support, computed from the owner's channel lines with
  set_posterior's own strike parse and likelihood (parse_strikes, the single-variant skip, _counts, _encounter's
  off-model drop) over B2's hard mask (the hard support set_posterior's two-way check holds equal to B2's);
- posterior_active: the device's visible pin stays active after the reference admits nonempty survivors over the hard
  support captured from B2 at the damage event, before later reveals in the same step;
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
        self.vocab = metadata['vocabulary']
        self.reset()

    def reset(self):
        """A new scenario's battles, including raw-line offsets and event-time evidence."""
        self.lines, self.teams, self.last_rows = {}, {}, []
        self.active, self.damage, self.evaluated = {}, {}, {}

    @staticmethod
    def _own_damage(line, owner):
        parts = line.split('|')
        return (len(parts) > 3 and parts[1] == '-damage' and parts[2].startswith(f'p{owner + 1}')
                and not any(p.startswith('[from]') for p in parts[4:]))

    def feed_b2(self, b2, deltas):
        """Fold complete raw-line prefixes through canonical B2, cutting immediately after each owner's damage line.

        The snapshots are full hard support, never the visible basis or device pins. Offsets count original channel
        lines, including those B2 filters. Crit/effectiveness/refusal evaluation waits for the completed host window.
        The ordinary flag-off feed is unchanged. Independent streams can reach their next cut in the same batch."""
        if len(deltas) != b2.streams:raise ValueError('one channel delta per B2 stream')
        offsets, cursors = list(b2.lines), [0] * b2.streams
        fresh = []
        while any(cursors[q] < len(lines or ()) for q, lines in enumerate(deltas)):
            parts, cuts = [], {}
            for q, lines in enumerate(deltas):
                lines = lines or ()
                start, end = cursors[q], len(lines)
                for index in range(start, end):
                    if self._own_damage(lines[index], q % 2):
                        end = index + 1;cuts[q] = offsets[q] + index;break
                parts.append(lines[start:end] or None);cursors[q] = end
            fresh.extend(b2.feed(parts))
            if not cuts:continue
            host = b2.host()
            for q, offset in cuts.items():
                if b2.refused[q] is not None:continue
                owner, attacker = q % 2, 1 - q % 2
                slot = int(host.mons.active[q, attacker])
                if not 0 <= slot < 6:raise ValueError(f'no attacker slot at damage line {offset} in stream {q}')
                species = b2.catalog.species.names[int(host.mons.species[q, attacker, slot])]
                snapshot = dict(species=self.sp.species_key(species), slot=slot,
                    encounter=int(host.mons.encounter[q, attacker, slot]),
                    hard=np.asarray(host.beliefs.hard[q, owner, slot], np.uint32).copy())
                self.damage.setdefault(q, {})[offset] = snapshot
        for q, refusal in enumerate(b2.refused):
            if refusal is not None:self.damage.pop(q, None)
        return fresh

    def tables(self, belief_tables):
        return self.dsp.enable(belief_tables, self.posterior)

    def feed(self, lane, step, channels, state=None):
        """The lane's owner channels (p1's, p2's) from battle start: reset at 'initial', extended at every step. At
        'initial' the engine state gives each owner's own team, the exact facts set_posterior reads from the owner's
        first request (the probe channels carry no |request| line): species, level, stats with max HP, the original
        ability and item, the moves."""
        if step == 'initial' or lane not in self.lines:
            if lane in self.lines:
                for owner in range(2):self.damage.pop(2 * lane + owner, None)
            self.lines[lane] = ([], [])
            self.active[lane] = set()
            self.evaluated[lane] = set()
            if state is not None:self.teams[lane] = tuple(self._team(state, lane, owner) for owner in range(2))
        for owner in range(2):self.lines[lane][owner].extend(channels[owner] or ())

    def _team(self, state, lane, owner):
        from pokezero import tier2
        v = self.vocab
        names = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
        team = []
        for k in range(6):
            species = int(state.species[lane, owner, k])
            if species <= 0:continue
            stats = dict(zip(names, (int(x) for x in state.stats[lane, owner, k])))
            stats['hp'] = int(state.max_hp[lane, owner, k])
            team.append(tier2.OwnMon(species=v['species'][species], level=int(state.level[lane, owner, k]), stats=stats,
                ability=v['abilities'][int(state.base_ability[lane, owner, k])] if int(state.base_ability[lane, owner, k]) else None,
                item=v['items'][int(state.item[lane, owner, k])] if int(state.item[lane, owner, k]) else None,
                moves=tuple(tier2.canonical_move_id(v['moves'][int(m)]) for m in state.moves[lane, owner, k] if int(m))))
        return tuple(team)

    def reference(self, lane, owner, species, hard_words):
        """set_posterior's (support bits, narrowed) for the opponent of `owner` holding `species` with B2's hard mask."""
        sp = self.sp
        lines = self.lines[lane][owner]
        seat = 'p1' if owner == 0 else 'p2'
        key = sp.species_key(species)
        universe = self.source.universe_for(species)
        variants = universe.variants if universe is not None else ()
        hard = tuple(_words_to_bits(hard_words)[:len(variants)])
        request = sp._first_request(lines)
        if request is not None:
            from pokezero import tier2
            team = tier2.own_team_from_request(request)
        else:
            team = self.teams.get(lane, ((), ()))[owner]
        rows = []
        single = sum(hard) == 1
        # compare() runs only after a complete Showdown step. Its channels omit requests, including at a forced
        # replacement after a KO, so mark that known boundary for the parser without inventing an own-team request.
        for strike in sp.parse_strikes([*lines, '|request|{}'], seat):
            if strike['attacker_key'] != key:continue
            self._activity(lane, owner, key, species, variants, team, strike)
            reason, counts = strike['reason'], None
            if reason is None and single:reason = sp.SINGLE_VARIANT
            elif reason is None:reason, counts = sp._counts(strike, species, variants, hard, team, source=self.source, dex=self.dex)
            rows.append((strike, reason, counts))
        post = sp._encounter('p2' if owner == 0 else 'p1', seat, key, hard, rows, '')
        # Activity comes only from event-time host evidence. Support above is independently recomputed at root time,
        # so a later reveal cannot erase activity or hide a strike-time/root-time support discrepancy.
        narrowed = (owner, key) in self.active[lane]
        self.last_reasons = [(s.line, s.move, s.reason or 'admitted') for s in post.strikes]
        return list(post.support) + [False] * (64 - len(post.support)), narrowed

    def _activity(self, lane, owner, key, species, variants, team, strike):
        identity = (owner, strike['line'])
        if identity in self.evaluated[lane]:return
        if strike['reason'] is None:
            captures = [s for i, s in self.damage.get(2 * lane + owner, {}).items()
                        if strike['line'] < i <= strike['last']]
            if len(captures) != 1 or captures[0]['species'] != key:
                raise ValueError(f'expected one B2 damage capture for {key} at line {strike["line"]}, got {captures}')
            hard = tuple(_words_to_bits(captures[0]['hard'])[:len(variants)])
            if sum(hard) > 1:
                reason, counts = self.sp._counts(strike, species, variants, hard, team, source=self.source, dex=self.dex)
                if reason is None and any(keep and count > 0 for keep, count in zip(hard, counts)):
                    self.active[lane].add((owner, key))
        self.evaluated[lane].add(identity)

    def diffs(self, trackers, b2_view, lane):
        """The narrowing diffs of one compared step (keys side:slot:field, as compare_views keys them)."""
        out, self.last_rows = [], []
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
                                    device=np.flatnonzero(device).tolist(), strikes=self.last_reasons))
                if active != narrowed:
                    out.append(dict(field=f'{s}:{k}:posterior_active', b2=narrowed, device=active))
                if self.last_reasons or active:
                    self.last_rows.append(dict(side=s, slot=k, species=mon['species'], hard=np.flatnonzero(hard).tolist(),
                        device=np.flatnonzero(device).tolist(), reference=np.flatnonzero(want).tolist(), strikes=self.last_reasons))
                if int(vb.conflicts[lane, s, k]):
                    out.append(dict(field=f'{s}:{k}:posterior_conflicts', b2=0, device=int(vb.conflicts[lane, s, k])))
        return out
