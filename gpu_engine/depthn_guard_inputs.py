"""The live effect guard's inputs (pokezero.depthn_effect_guard), read from one DepthN price call: the seam and the record.

Seam (observer only, our layer; no pricer file changes). ObservedDepthNPricer is a DepthNPricer whose `_program` returns the
base program unchanged unless `observing` is set; while set, the P1 program (key 'P1') is wrapped so that each call's
output, before DepthNPricer.price pops its turn-1 children ('states') to expand ply 2, is read through one jitted gather
(`labels`, child_labels by default) into `children` together with the output's lane_cell. The wrapper calls the compiled
program with the same arguments and returns its output object itself, so the frame, the programs and the control flow are
the base pricer's in either mode (tests pin that bitwise). check_seam verifies, before anything prices, that
DepthNPricer.price still reaches P1 through self._program under that key and that make_p1's output names 'states' and
'lane_cell'; a P1 output without them leaves `children` empty and the guard fails closed (turn2_available False).

Record (effect_record): our root prior (build's root_logits / root_legal: B2's prior) on our request's lanes (own_moves);
per world the peer's turn-1 prior (frame.root_peer_logits / root_peer_legal, requested) on the lanes of that world's peer
active (labels of the root world states); per expanded turn-1 cell (t1_status 2) of every world, each requested side's
turn-2 prior (t1_own_logits / t1_own_legal, t1_peer_logits / t1_peer_legal) on that child's lanes, the child's conditions
from its own state (statuses, faints, Rest origin) over the root's (items, abilities); the search's own action for (c).
"""
import dis

import jax
import jax.numpy as j
import numpy as np

from pokezero import depthn_effect_guard as guard
from pokezero.dex import normalize_id
from . import device_depthn_pricer as dn

LANES = dn.LANES


def _labels(states, ability=False):
    """Per battle row of a stacked BattleState [N, 1, ...]: active [N, 2], the active's species, four moves, charge move and
    usable mask (gpu_engine.state.usable_moves), and each party slot's species, status, alive (hp > 0) and Rest origin; for the
    ability guard (`ability`: ability_labels) also each party slot's ability, hp, max_hp, types and Hidden Power type."""
    from .state import usable_moves
    s = jax.tree.map(lambda x: x[:, 0], states)
    act = j.clip(s.active, 0, 5)
    at = lambda x: j.take_along_axis(x, act[:, :, None], axis=2)[:, :, 0]
    moves = j.take_along_axis(s.moves, act[:, :, None, None], axis=2)[:, :, 0]
    out = dict(active=act, species=at(s.species), moves=moves, charge=at(s.charge_move), usable=usable_moves(s), party=s.species, status=s.status,
        alive=s.hp > 0, rest=s.rest_origin)
    if ability:out.update(ability=s.ability, hp=s.hp, max_hp=s.max_hp, types=s.types, hp_type=s.hidden_power_type)
    return out


_PROGRAM = {}


def child_labels(states):
    """_labels through one jitted program per process (device arrays; the caller fetches them)."""
    if 'program' not in _PROGRAM:_PROGRAM['program'] = jax.jit(_labels)
    return _PROGRAM['program'](states)


def ability_labels(states):
    """child_labels plus each party slot's ability, hp, max_hp, types and Hidden Power type: the ability guard's gather (a server
    with --ability-guard observes with it; without the flag the gather, its transfer and the record are child_labels' alone)."""
    if 'ability' not in _PROGRAM:_PROGRAM['ability'] = jax.jit(lambda states: _labels(states, ability=True))
    return _PROGRAM['ability'](states)


class ObservedDepthNPricer(dn.DepthNPricer):
    """A DepthNPricer with the observer seam (module docstring). `labels(states)`: the per-row label gather (child_labels)."""
    observing = False

    def __init__(self, *arguments, labels=child_labels, **keywords):
        super().__init__(*arguments, **keywords);self.labels = labels;self.children = [];self.notes = [];self.label_rows = set()

    def _program(self, key, build, *args):
        compiled = super()._program(key, build, *args)
        if not self.observing or not (type(key) is tuple and key and key[0] == 'P1'):return compiled
        pricer = self

        def run(*call):
            out = compiled(*call)
            try:
                if not isinstance(out, dict) or 'states' not in out or 'lane_cell' not in out:raise KeyError('P1 output without states / lane_cell')
                pricer.children.append(dict(lane_cell=out['lane_cell'], labels=pricer.labels(out['states'])))
                pricer.label_rows.add(int(np.shape(out['lane_cell'])[0]) * LANES)   # the gathered shape (warmup checks it compiled)
            except Exception as error:pricer.notes.append(f'P1 children: {type(error).__name__}: {error}'[:300])
            return out
        return run

    def observe(self):
        """Start one observed price call: children and notes cleared, observing on."""
        self.children, self.notes, self.observing = [], [], True

    def observed(self):
        """End it: observing off; (the children fetched to the host, notes)."""
        self.observing = False
        children = [dict(lane_cell=np.asarray(c['lane_cell']), labels={k: np.asarray(v) for k, v in jax.device_get(c['labels']).items()})
            for c in self.children]
        self.children = []
        return children, list(self.notes)


def check_seam(price=None):
    """True when DepthNPricer.price reaches P1 through self._program with a 'P1' key and make_p1's output names 'states' and
    'lane_cell'; RuntimeError otherwise (the seam would observe nothing)."""
    price = dn.DepthNPricer.price if price is None else price
    names = {i.argval for i in dis.get_instructions(price)}
    if not {'_program', 'P1'} <= names or not {'states', 'lane_cell'} <= _strings(dn.make_p1.__code__):
        raise RuntimeError('the DepthN pricer no longer reaches P1 through self._program, or P1 no longer outputs states and lane_cell')
    return True


def _strings(code):
    """Every string constant of a code object and its nested code objects (keyword names included)."""
    out = set()
    for c in code.co_consts:
        if isinstance(c, str):out.add(c)
        elif isinstance(c, tuple):out.update(x for x in c if isinstance(x, str))
        elif hasattr(c, 'co_consts'):out |= _strings(c)
    out.update(code.co_names);return out


# The record -----------------------------------------------------------------------------------------------------------------------
class Names:
    """Engine ids -> normalized move and species names (metadata['vocabulary'])."""
    def __init__(self, vocabulary):
        self.moves = [normalize_id(x) for x in vocabulary['moves']];self.species = [normalize_id(x) for x in vocabulary['species']]

    def move(self, i):return self.moves[i] if 0 < i < len(self.moves) else None

    def kind(self, i):return self.species[i] if 0 < i < len(self.species) else None


def side_lanes(labels, row, seat, names):
    """(lanes [4], user species) of one seat at one labelled row."""
    moves = [names.move(int(m)) for m in labels['moves'][row, seat]];charge = names.move(int(labels['charge'][row, seat]))
    return guard.lanes_of(moves, charge, [bool(x) for x in labels['usable'][row, seat]]), names.kind(int(labels['species'][row, seat]))


def side_members(labels, row, seat, names):
    return [(names.kind(int(labels['party'][row, seat, k])), int(labels['status'][row, seat, k]), bool(labels['alive'][row, seat, k]),
        bool(labels['rest'][row, seat, k])) for k in range(labels['party'].shape[2])]


def stack_worlds(worlds):
    """The worlds' states stacked [W, 1, ...] (host), a world without a state as the first constructed one's (as pack_turn1)."""
    first = next((w for w in worlds if w.state is not None), None)
    if first is None:return None
    host = [jax.device_get(w.state if w.state is not None else first.state) for w in worlds]
    return jax.tree.map(lambda *x: np.stack([np.asarray(v) for v in x]), *host)   # [W, 1, ...]: one-battle rows, as P1's children


UNION_FIELDS = ('tripped', 'b', 'b_turn1', 'b_turn2', 'c', 'arms', 'm_own', 'm_opp')
_FACTS = {}


def ability_facts(metadata):
    """AbilityFacts of one metadata object, built once per process (the live search's constructor metadata never changes)."""
    key = id(metadata)
    if key not in _FACTS or _FACTS[key][0] is not metadata:_FACTS[key] = (metadata, AbilityFacts(metadata))
    return _FACTS[key][1]


class AbilityFacts:
    """The ability guard's view of engine metadata: ability and type names (vocabulary), the move catalog (category, type name,
    contact; metadata['moves'], vocabulary ordered) and the Gen 3 type chart by name (metadata['type_chart'][attack][defend] over
    vocabulary types 1..; type 0 is typeless)."""
    def __init__(self, metadata):
        v = metadata['vocabulary'];self.abilities = [normalize_id(x) for x in v['abilities']];self.types = [normalize_id(x) for x in v['types']]
        self.catalog = {}
        for row in metadata.get('moves') or ():
            if not row or 'id' not in row:continue
            t = int(row.get('type') or 0)
            self.catalog[normalize_id(row['id'])] = dict(category=row.get('category'), type=self.types[t] if 0 < t < len(self.types) else None,
                contact=bool((row.get('flags') or {}).get('contact')))
        if not self.catalog:raise ValueError('the ability guard needs the move catalog (metadata moves)')
        names = self.types[1:];chart = metadata['type_chart']
        self.chart = {a: {d: float(chart[i][j]) for j, d in enumerate(names)} for i, a in enumerate(names)}
        self.relevant = frozenset(i for i, name in enumerate(self.abilities) if name in guard.ABILITY_FAMILIES)

    def mon(self, labels, row, seat, slot):
        """dict(ability, hp, max_hp, types, hp_type, species index) of one party slot at one labelled row."""
        a = int(labels['ability'][row, seat, slot]);t = [int(x) for x in labels['types'][row, seat, slot]];h = int(labels['hp_type'][row, seat, slot])
        return dict(ability=self.abilities[a] if 0 < a < len(self.abilities) else None, hp=int(labels['hp'][row, seat, slot]),
            max_hp=int(labels['max_hp'][row, seat, slot]), types=[self.types[x] for x in t if 0 < x < len(self.types)],
            hp_type=self.types[h] if 0 < h < len(self.types) else None)

    def active(self, labels, row, seat):
        return self.mon(labels, row, seat, int(labels['active'][row, seat]))

    def touches(self, labels, row):
        """Whether either active at this row holds an ability the guard watches (else no lane there can be guarded)."""
        return any(int(labels['ability'][row, seat, int(labels['active'][row, seat])]) in self.relevant for seat in (0, 1))

    def lanes(self, labels, row, seat, lanes, pinch='third', actives=None):
        """ability_lanes of `seat` at a row; `actives` {seat: active facts} when the caller already read them."""
        actives = actives or {s: self.active(labels, row, s) for s in (0, 1)}
        return guard.ability_lanes(actives[seat], actives[1 - seat], lanes, self.catalog, self.chart, pinch=pinch)

    def unknown(self, lanes):
        """How many of a side's lanes name a move the catalog does not know (recorded: such a lane is never guarded)."""
        return sum(1 for m in list(lanes)[:4] if m and m not in self.catalog)


def ability_root(facts, labels, world, owner, own_lanes, peer_lanes):
    """The ability guard's turn-1 inputs of one world's root row: (turn1 entry, switch-in entries)."""
    lanes = {owner: list(own_lanes)[:4], 1 - owner: list(peer_lanes)[:4]};side = {owner: 'own', 1 - owner: 'opp'}
    entry = dict(world=world);actives = {s: facts.active(labels, world, s) for s in (0, 1)}
    for seat in (owner, 1 - owner):
        entry[side[seat]] = dict(third=facts.lanes(labels, world, seat, lanes[seat], actives=actives),
            two_thirds=facts.lanes(labels, world, seat, lanes[seat], 'two_thirds', actives=actives), damaging=guard.damaging_lanes(lanes[seat], facts.catalog))
    switchin = []
    for seat in (owner, 1 - owner):
        active = int(labels['active'][world, seat]);other = 1 - seat
        for i in range(5):
            slot = i + (i >= active)
            if slot >= labels['ability'].shape[2] or not bool(labels['alive'][world, seat, slot]):continue
            mon = facts.mon(labels, world, seat, slot);family = guard.holder_family(mon)
            if family is None:continue
            against = guard.ability_lanes(actives[other], mon, lanes[other], facts.catalog, facts.chart)[family]
            if any(against):switchin.append(dict(side=side[seat], lane=4 + i, family=family, holder=normalize_id(str(mon['ability'])), against=list(against)))
    return entry, switchin


def effect_record(frame, root, worlds, children, request, owner, vocabulary, choice, *, labels=child_labels, notes=(), reply_logits=None,
                  metadata=None):
    """The guard record of one priced root (module docstring). `frame`: the DepthN frame; `root`: build's root dict (rows,
    root_logits, root_legal); `worlds`: the root's worlds in block order (block k's world j is world 2k + j); `children`:
    ObservedDepthNPricer.observed()'s children of this frame's price call (one P1 call); `request`: the protocol request
    (lines, own_request_line); `owner` 0 or 1; `choice`: the search's action.
    `reply_logits` (a served reply model, br_fp): per world in block order its reply-model logits [9], None for a waiting world.
    The union guard: the guard is evaluated a second time with each requested world's turn-1 peer prior replaced by the reply
    model's (turn-1 peer mass and the reach's per-world reply weights; our prior and the turn-2 priors unchanged). The record
    stays the champion's evaluation; `union` adds dict(tripped=champion OR reply, champion=..., reply=...) over UNION_FIELDS (with
    the ability guard also ability_b and ability_own_lanes, its (c) lanes: those can differ between the arms),
    `union_only` whether only the reply arm tripped (then the record's b, c, arms and m_* describe the champion, which did not), and
    the top-level `tripped` becomes union['tripped'] (the live actor plays greedy on it). Without reply_logits: unchanged. Only (b)
    can differ between the arms: (c) reads our own prior and lanes, the same in both.
    `metadata` (the engine metadata: vocabulary, moves, type_chart): with it the ability guard runs too (AbilityFacts,
    ability_root at each world's root row, ability_lanes at each expanded child's two sides, exact; the record's `ability`);
    without it the record is the move guard's alone."""
    names = Names(vocabulary);cond = guard.conditions(request['lines'], owner, request.get('own_request_line'))
    own_lanes = guard.own_moves(request.get('own_request_line')) or (None,) * 4
    own_turn1 = dict(lanes=list(own_lanes), logits=[float(x) for x in root['root_logits']], legal=[bool(x) for x in root['root_legal']])
    stacked = stack_worlds(worlds);root_labels = None if stacked is None else {k: np.asarray(v) for k, v in jax.device_get(labels(stacked)).items()}
    peer, turn2, expanded, missing = [], [], 0, 0
    facts = None if metadata is None else ability_facts(metadata);ab = dict(turn1=[], switchin={}, turn2=[], unknown_lanes=0)
    if facts is not None and root_labels is not None and 'ability' not in root_labels:raise ValueError('the ability guard needs ability_labels')
    if reply_logits is not None and len(reply_logits) != len(worlds):raise ValueError(f'{len(reply_logits)} reply logits for {len(worlds)} worlds')
    child = children[0] if len(children) == 1 else None
    rows = {}
    if child is not None:
        lane_cell = child['lane_cell']
        for w, lane in zip(*np.nonzero(lane_cell < LANES)):rows[(int(w), int(lane_cell[w, lane]))] = int(w) * LANES + int(lane)
    for k, r in enumerate(root['rows']):
        for jw in (0, 1):
            world = 2 * k + jw
            if root_labels is not None and world < len(worlds) and worlds[world].state is not None:
                lanes, user = side_lanes(root_labels, world, 1 - owner, names)
                peer.append(dict(world=world, lanes=lanes, user=user, logits=[float(x) for x in frame.root_peer_logits[r, jw]],
                    legal=[bool(x) for x in frame.root_peer_legal[r, jw]], requested=bool(frame.root_peer_requested[r, jw])))
                if facts is not None:
                    entry, switchin = ability_root(facts, root_labels, world, owner, own_lanes, lanes)
                    ab['turn1'].append(entry);ab['unknown_lanes'] += facts.unknown(lanes) + (facts.unknown(own_lanes) if not ab['turn1'][1:] else 0)
                    if switchin:ab['switchin'][world] = switchin
            status = frame.t1_status[r, jw]
            for a, b in zip(*np.nonzero(status == 2)):
                expanded += 1;cell = int(a) * 9 + int(b);row = rows.get((2 * r + jw, cell), -1)
                if row < 0:missing += 1;continue
                sides = None;touched = facts is not None and facts.touches(child['labels'], row);actives = None
                for side, seat, requested, logits, legal in (('own', owner, frame.t1_own_requested, frame.t1_own_logits, frame.t1_own_legal),
                        ('opp', 1 - owner, frame.t1_peer_requested, frame.t1_peer_logits, frame.t1_peer_legal)):
                    if not bool(requested[r, jw, a, b]):continue
                    lanes, user = side_lanes(child['labels'], row, seat, names)
                    if touched:
                        actives = actives or {s: facts.active(child['labels'], row, s) for s in (0, 1)}
                        masks = facts.lanes(child['labels'], row, seat, lanes, actives=actives)
                        if any(any(v) for v in masks.values()):
                            ab['turn2'].append(dict(world=world, cell=cell, side=side, lanes=masks, logits=[float(x) for x in logits[r, jw, a, b]],
                                legal=[bool(x) for x in legal[r, jw, a, b]]))
                    if not any(m in guard.WATCHED for m in lanes):continue   # nothing to weigh at this child
                    if sides is None:
                        sides = dict(own_side=guard.child_side(cond['own'], side_members(child['labels'], row, owner, names)),
                            opp_side=guard.child_side(cond['opp'], side_members(child['labels'], row, 1 - owner, names), opponent=True))
                    turn2.append(dict(world=world, cell=cell, side=side, lanes=lanes, user=user, logits=[float(x) for x in logits[r, jw, a, b]],
                        legal=[bool(x) for x in legal[r, jw, a, b]], **sides))
    available = not (expanded and (child is None or missing))
    ability = None if facts is None else ab
    record = guard.evaluate(cond, own_turn1=own_turn1, peer_turn1=peer, turn2=[dict(t) for t in turn2], turn2_available=available, choice=choice,
        ability=ability)
    record.update(turn2_cells=dict(expanded=expanded, unlabelled=missing, p1_captures=len(children)), notes=list(notes))
    if reply_logits is not None:
        replied = []
        for w in peer:
            logits = reply_logits[w['world']]
            if w['requested'] and (logits is None or len(logits) != len(w['logits'])):
                raise ValueError(f'reply logits for requested world {w["world"]}: {None if logits is None else len(logits)} values, not {len(w["logits"])}')
            replied.append(dict(w, logits=[float(x) for x in logits]) if w['requested'] else dict(w))
        reply = guard.evaluate(cond, own_turn1=own_turn1, peer_turn1=replied, turn2=[dict(t) for t in turn2], turn2_available=available, choice=choice,
            ability=ability)
        def summary(r):
            out = {k: r.get(k) for k in UNION_FIELDS}
            if 'ability' in r:   # the ability guard's (b) and its (c) lanes (the root PINCH rule reads the reply prior: (c) can differ)
                ab = r['ability'];choice = lambda lanes: sorted(f for f in lanes or () if 'choice' in ab['arms'].get(f, ())) or None
                out.update(ability_b=bool(ab['b']), ability_own_lanes=[choice(x) for x in ab['own_guarded_lanes']])
            return out
        record['union'] = dict(tripped=bool(record['tripped'] or reply['tripped']), champion=summary(record), reply=summary(reply))
        record['union_only'] = bool(record['union']['tripped'] and not record['tripped'])   # only the reply arm tripped: b, c, arms, m_* are the champion's
        record['tripped'] = record['union']['tripped']
    return record
