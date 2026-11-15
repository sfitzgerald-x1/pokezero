"""R3pa's live lane labels: each priced root's worlds' opponent reply lanes labelled as mc_ladder_backline.labels labels them
offline, for depthn_r3pa's label-mapped root prior (the live search's R3pa hook; built, warmed and called only when that hook is
on, so a search without it compiles and runs nothing of this module).

Offline (scripts/depthn_rr_reprice.py, then mc_ladder_backline.labels): per world, from that world's own engine state and completed
public carry, opponent_facts (the opponent's active, its four move ids, its charge move, its party's species through the
constructor's vocabulary) and PeerLanes (the opponent's encoder lanes: device_depth2_pricer._make_views' peer observation, legal
[9], move_targets [4], switch_targets [5], and gpu_engine.state.usable_moves on its side) give depthn_rr_reprice.lane_labels:
lane b < 4 is move:struggle (b 0, no usable move), move:<charge> (b 0, a charge move held), else move:<the active's move in slot
move_targets[b]>; lane 4 + i is switch:<species of party slot switch_targets[i]>; None on an illegal lane or a target outside the
party. mc_ladder_backline.labels then keeps move labels and maps switch:<species> to switch:<species_key> when that species is in
every world's party (seen_species) and to switch:? otherwise; a world whose party has a member without a species makes the root
unlabelled (world_sets_missing).

Live (this module): the same facts from one jitted program per owner seat over the root's stacked worlds (LaneLabels.views: the
same _make_views on the pricer's executor and the same usable_moves, vmapped over the worlds; only integer outputs, so XLA keeps
just what they need), then lane_labels_live, a pure host mapper over the constructor vocabulary, with the offline helpers' exact
string rules (phase_d_live_worlds._source_move for moves, pokezero.dex.normalize_id then fp_reply_evidence.species_key for
species). Integer outputs per world (LaneViews): legal [9] (the peer view's legal mask), kind [9] (NONE, MOVE, STRUGGLE, SWITCH),
lane_id [9] (the engine move id of a MOVE lane, the engine species id of a SWITCH lane, else 0) and party [6] (the opponent's
party species ids, slot order). The pricer's own frame keeps only root_peer_requested / legal / logits (no move or switch
targets) and the effect guard's root gather runs after the decision and has no targets, so the labels cost this one extra small
device pass per priced root.

peer_legal_mismatch: the requested worlds whose kept reply lanes (WorldPrices.opponent_legal) differ from the peer view's legal
lanes; mc_ladder_backline.check_labels refuses such a root (kept_matches_peer_legal false, lane_unlabelled), which a WorldPrices
alone cannot show.
"""
from dataclasses import dataclass
import re
import time

import numpy as np

UNSEEN = 'switch:?'   # mc_ladder_backline.UNSEEN
NONE, MOVE, STRUGGLE, SWITCH = 0, 1, 2, 3   # LaneViews.kind
LANES = 9


class WorldSetsMissing(ValueError):
    """A world's opponent party has a member without a species: mc_ladder_backline.seen_species is None (world_sets_missing)."""
    reason = 'world_sets_missing'


@dataclass(frozen=True)
class LaneViews:
    """One root's views program outputs, host arrays in world (block) order, and the device pass's wall seconds."""
    legal: np.ndarray     # [W, 9] bool: the peer view's legal mask
    kind: np.ndarray      # [W, 9] int32: NONE, MOVE, STRUGGLE or SWITCH
    lane_id: np.ndarray   # [W, 9] int32: engine move id (MOVE), engine species id (SWITCH), else 0
    party: np.ndarray     # [W, 6] int32: the opponent's party species ids, slot order
    seconds: float = 0.


def species_key(species):
    """fp_reply_evidence.species_key (fp_reply_reweight.toid, then Unown's letter forms as one species), without its imports."""
    mon = re.sub(r'[^a-z0-9]', '', str(species or '').lower())
    return 'unown' if mon.startswith('unown') else mon


class Names:
    """The constructor vocabulary's engine ids as depthn_rr_reprice.opponent_facts names them, and the label strings."""
    def __init__(self, vocabulary):
        from pokezero.dex import normalize_id
        from .phase_d_live_worlds import _source_move
        self._move = [None] + [_source_move(x) for x in vocabulary['moves'][1:]]
        self._species = [None] + [normalize_id(x) for x in vocabulary['species'][1:]]
        self._key = [species_key(str(x)) for x in self._species]   # backline's species_key of the lane's 'switch:<species>' text

    def move_label(self, i):
        """'move:' + opponent_facts' move(i) (None for i <= 0, as its f-string writes it); IndexError past the vocabulary."""
        return f'move:{None if i <= 0 else self._move[i]}'

    def species(self, i):
        """opponent_facts' species(i): None for i <= 0, else normalize_id of the vocabulary entry; IndexError past it."""
        return None if i <= 0 else self._species[i]

    def key(self, i):
        """species_key of the lane text depthn_rr_reprice.lane_labels writes for species id i."""
        return self._key[0] if i <= 0 else self._key[i]


def lane_function(executor, owner):
    """One world's (legal [9], kind [9], lane_id [9], party [6]) for the opponent of `owner` (traced; module docstring), from its
    state and public carry with their one-battle leading axis [1, ...]: depth2._make_views(executor)'s peer observation and
    gpu_engine.state.usable_moves, with depthn_rr_reprice.lane_labels' rules."""
    import jax.numpy as j
    from . import device_depth2_pricer as depth2
    from .state import usable_moves
    views = depth2._make_views(executor);foe = 1 - owner

    def lanes(state, public):
        _, peer, _, _ = views(state, public, owner)
        legal = peer.legal_action_mask.reshape(LANES).astype(bool)
        move_targets = peer.move_targets.reshape(4).astype(j.int32);switch_targets = peer.switch_targets.reshape(5).astype(j.int32)
        usable = usable_moves(state)[0, foe]
        party = state.species[0, foe].astype(j.int32);slots = party.shape[0]
        active = state.active[0, foe];inside = (active >= 0) & (active < slots);at = j.clip(active, 0, slots - 1)
        moves = state.moves[0, foe, at].astype(j.int32);charge = state.charge_move[0, foe, at].astype(j.int32)
        struggle = ~j.any(usable);charged = inside & (charge > 0);first = j.arange(4) == 0
        in_moves = inside & (move_targets >= 0) & (move_targets < moves.shape[0])
        slot_move = moves[j.clip(move_targets, 0, moves.shape[0] - 1)]
        move_kind = j.where(first & struggle, STRUGGLE, j.where(first & charged, MOVE, j.where(in_moves, MOVE, NONE)))
        move_id = j.where(first & struggle, 0, j.where(first & charged, charge, j.where(in_moves, slot_move, 0)))
        in_party = (switch_targets >= 0) & (switch_targets < slots)
        switch_kind = j.where(in_party, SWITCH, NONE);switch_id = j.where(in_party, party[j.clip(switch_targets, 0, slots - 1)], 0)
        kind = j.where(legal, j.concatenate([move_kind, switch_kind]), NONE).astype(j.int32)
        lane_id = j.where(kind != NONE, j.concatenate([move_id, switch_id]), 0).astype(j.int32)
        return legal, kind, lane_id, party
    return lanes


def stack_worlds(worlds):
    """The worlds' (states, publics) fetched to the host and stacked [W, 1, ...] (as the re-pricer reads a world: jax.device_get)."""
    import jax
    if not worlds:raise ValueError('one or more worlds required')
    if any(w.state is None or w.public is None for w in worlds):raise ValueError('a priced root with an unconstructed world')
    host = jax.device_get([(w.state, w.public) for w in worlds])
    return jax.tree.map(lambda *x: np.stack([np.asarray(v) for v in x]), *host)


class LaneLabels:
    """The live labels of one search (module docstring): one jitted views program per owner seat on `executor` (the pricers'),
    built on first use; `vocabulary` is the constructor's metadata['vocabulary']. Nothing is traced or compiled until views()
    or warm() runs. `traces[owner]` counts the program's traces (one per seat and world count)."""
    def __init__(self, executor, vocabulary):
        self.executor, self.names = executor, Names(vocabulary);self.programs, self.traces = {}, {0: 0, 1: 0}

    def _program(self, owner):
        if owner not in (0, 1):raise ValueError(f'owner 0 or 1 required, got {owner!r}')
        if owner not in self.programs:
            import jax
            lanes = lane_function(self.executor, owner)

            def program(states, publics):
                self.traces[owner] += 1
                return jax.vmap(lanes)(states, publics)
            self.programs[owner] = jax.jit(program)
        return self.programs[owner]

    def views(self, worlds, owner):
        """LaneViews of a priced root's worlds (block order: the constructor's world order) for `owner`: one device pass."""
        import jax
        started = time.perf_counter();program = self._program(owner);states, publics = stack_worlds(worlds)
        legal, kind, lane_id, party = (np.asarray(x) for x in jax.device_get(program(states, publics)))
        return LaneViews(legal.astype(bool), kind.astype(np.int32), lane_id.astype(np.int32), party.astype(np.int32), time.perf_counter() - started)

    def warm(self, worlds):
        """Compile both seats' programs on these worlds' shapes (a live root's) before READY: {owner: seconds}."""
        seconds = {}
        for owner in (0, 1):
            started = time.perf_counter();self.views(worlds, owner);seconds[owner] = time.perf_counter() - started
        return seconds

    def __call__(self, worlds, owner):
        """lane_labels_live of this root's views."""
        return lane_labels_live(self.views(worlds, owner), self.names)


def seen_keys(views, names):
    """mc_ladder_backline.seen_species on the views: the species_keys in every world's party; WorldSetsMissing when a world's
    party has a member without a species (fp_reply_evidence.world_party None) or there is no world."""
    seen = None
    for w, row in enumerate(np.asarray(views.party)):
        species = [names.species(int(i)) for i in row]
        if not species or any(not s for s in species):raise WorldSetsMissing(f'world_sets_missing: world {w} has a party member without a species')
        keys = {species_key(s) for s in species};seen = keys if seen is None else seen & keys
    if seen is None:raise WorldSetsMissing('world_sets_missing: no world')
    return seen


def lane_labels_live(views, names):
    """Per world (block order) its nine reply lanes' labels, exactly mc_ladder_backline.labels(world, seen_species(worlds)) of
    depthn_rr_reprice.lane_labels: 'move:<id>', 'switch:<species_key>' (seen in every world's party), 'switch:?' (unseen), None
    (an illegal lane, or one without a label). Raises WorldSetsMissing (module docstring)."""
    kind, lane_id = np.asarray(views.kind), np.asarray(views.lane_id)
    if kind.ndim != 2 or kind.shape[1] != LANES or lane_id.shape != kind.shape or len(views.party) != len(kind):
        raise ValueError('views of [W, 9] lanes and [W, 6] parties required')
    seen = seen_keys(views, names);out = []
    for w in range(len(kind)):
        row = []
        for b in range(LANES):
            k, i = int(kind[w, b]), int(lane_id[w, b])
            if k == NONE:row.append(None)
            elif k == STRUGGLE:row.append('move:struggle')
            elif k == MOVE:row.append(names.move_label(i))
            elif k == SWITCH:
                key = names.key(i);row.append('switch:' + key if key in seen else UNSEEN)
            else:raise ValueError(f'world {w} lane {b}: unknown lane kind {k}')
        out.append(row)
    return out


def peer_legal_mismatch(world_prices, views):
    """The requested worlds (block order indices) whose kept reply lanes differ from their peer view's legal lanes
    (mc_ladder_backline.check_labels' kept_matches_peer_legal false: the hook refuses the root, lane_unlabelled). A world without a
    price surface (None) or whose opponent waits is not compared."""
    prices = list(world_prices);legal = np.asarray(views.legal, bool)
    if len(prices) != len(legal):raise ValueError(f'{len(prices)} priced worlds for {len(legal)} viewed')
    return [w for w, p in enumerate(prices) if p is not None and p.opponent_requested
            and [bool(x) for x in p.opponent_legal] != [bool(x) for x in legal[w]]]
