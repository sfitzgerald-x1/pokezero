"""UNREPRESENTED_EFFECT: the DepthN live guard against pricing a move the GPU engine does not model (pure Python, no JAX).

The GPU device engine keys every move effect by name and reads no effect fields (generate_teams.cjs:75-81); the static
table of its handling of the Gen 3 random-battle pool is reports/gpu-engine-move-effect-table-20261109.md. GUARDED holds
the moves whose missing effect carries over into the price (the lead's decision):
- tier A, unconditional: transform, yawn, wrap, perishsong, counter, mirrorcoat, curse, superpower, knockoff, charm,
  lovelykiss, bonemerang, gigadrain;
- tier C, guarded for a side only while its condition holds (`conditions`): spore, sleeppowder, hypnosis while Sleep Clause
  would bind (the target's side already has a non-fainted Pokemon asleep from a non-allied source: Showdown f76228a1
  data/rulesets.ts sleepclausemod, whose onSetStatus skips a sleeper whose statusState.source is its ally, i.e. Rest; a
  Yawn or move-induced sleep counts; a wake (-curestatus slp), a faint or a party cure ends it), refresh while the user is
  burned, paralyzed or poisoned, sacredfire while the user is frozen, psychoboost while the user's White Herb is unconsumed
  (ours: the request's item; the opponent's: Deoxys and Deoxys-Attack always hold it in the pool, until an -enditem or an
  -item for that Pokemon is seen), facade while the user has Guts and a burn, paralysis or poison (every pool Facade user
  has Guts; ours: the request's ability), aromatherapy while the user's party has a statused, non-fainted Pokemon (ours:
  the request; the opponent's: its public statuses), trick while either side has a non-fainted Pokemon whose item was
  publicly knocked off (-enditem ... [from] move: Knock Off: the engine models Trick since M1, but live roots do not carry
  item_knocked yet, so an in-tree Trick on that Pokemon would succeed where Showdown fails; roots after a public item swap
  stay refused by the world constructor, PUBLIC_ITEM_SWAP_NOT_SUPPORTED).
TIER_B (irontail, signalbeam, steelwing, meteormash: chance secondaries) is never guarded, only reported (tier_b_legal).
RETIRED (tier 'R') holds the moves the guard no longer trips on because the engine now models their effect, each with the
integration merge from which it does (a screen declares an engine at or after it): destinybond (A2, 286a5ffe: the DESTINY_BOND
volatile, Showdown-probed), hyperbeam (M1, 8b212312: the recharge turn; the live roots' recharge refusals and the live actor's
own-recharge guard, RECHARGE_MOVES, are separate and unchanged). They stay WATCHED, so their masses are still recorded (turn-2
entries: guarded False, tier 'R'), and never count toward (a), (b) (its turn-1 arm and the reach) or (c). Roots where a side is
already bonded stay refused by the world constructor, not here.

The guard trips on (b) or (c) (RULE 'reach', the lead's live rule from screen-r2):
- (b) turn 1: in any constructed world, either side's turn-1 prior puts >= THRESHOLD of its mass on a lane whose move is
  guarded for that side (our root prior over our legal lanes; the peer's prior per world);
- (b) turn 2, the reach: M_own >= THRESHOLD or M_opp >= THRESHOLD, M_s = sum_w p_w sum_(a,b) pi_own(a) pi_peer,w(b)
  q_(w,ab),s, q the side's turn-2 prior mass at child cell a*9 + b of world w on moves guarded for it at that child (tier A,
  tier C under that child's own conditions: each recorded entry's `guarded`), pi_own our root prior, pi_peer,w world w's
  turn-1 reply prior (a waiting world: reply 0 at mass 1), p_w uniform over the priced worlds (`reach`, exact here);
- (c) our chosen action is a move lane whose move is guarded for us (for the RR, per operator on its own choice: trips).
The earlier any-cell turn-2 arm (either side's turn-2 prior >= THRESHOLD on a guarded lane at ANY expanded cell) is kept in
every record for analysis only: any_cell_turn2, any_cell_b and any_cell_tripped.
`evaluate` returns the record: tripped (for the search's own choice), b (b_turn1, b_turn2: the reach arm), c, m_own and m_opp
(the reach M; reach: their arm's own summary), arms (which tripped: turn1, reach, unavailable, choice; error for an error
record), the any-cell fields above (its turn-2 hits under any_cell_reasons), reasons (only what tripped: per move x side for turn1
and choice, with the max mass, the tier and the effect class; one per tripping side for the reach, with M as max_mass, its
contributing tiers' masses, its moves' weighted masses and its worlds; turn2_unavailable), tiers (the tripped reasons'), the (a) flag (a guarded move legal for
either side at turn 1 in any world: reporting only), tier_b_legal, our guarded lanes, the conditions per side and the masses
of every WATCHED move (GUARDED, TIER_B, RETIRED; turn 1 per side and world, turn 2 per side, world and cell a * 9 + b with our lane a and the
reply b, each turn-2 entry also carrying `guarded` (this guard's own decision at that child), the move's `tier` and its `user`
species). A turn-2 capture that is missing while some turn-1 cell expanded fails closed: (b) trips (the reach is
unavailable; the any-cell arm's source turn2_unavailable). `reach` weighs those turn-2 masses by the root and reply priors:
exact from `guarded` (the live arm), bounds for an older record without it (the RR's report on old rows).

Lane labels follow the encoder (engine_device_observation.encode): move lanes 0-3 are the active's four moves in slot order,
except lane 0 alone (the charge move) while one charges and lane 0 alone (Struggle) when no move is usable; switch lanes
4-8 are never moves. Our root lanes come from our own request (own_moves).
"""
import json
import math

SCHEMA = 'pokezero.depthn-effect-guard.v1'
THRESHOLD = .05
CAUSE = 'UNREPRESENTED_EFFECT'
RULE = 'reach'   # the live turn-2 arm: the reach M_s (screen-r2); records before it used the any-cell arm (any_cell_* now)

# move id -> (tier, effect class, condition) ; tier A conditions are None.
GUARDED = {
    'transform': ('A', 'other', None),
    'yawn': ('A', 'volatile', None), 'wrap': ('A', 'volatile', None), 'perishsong': ('A', 'volatile', None),
    'counter': ('A', 'retaliation', None), 'mirrorcoat': ('A', 'retaliation', None),
    'curse': ('A', 'self_boost', None), 'superpower': ('A', 'self_stat_drop', None), 'knockoff': ('A', 'item_removal', None),
    'charm': ('A', 'target_stat_change', None), 'lovelykiss': ('A', 'status', None), 'bonemerang': ('A', 'other', None),
    'gigadrain': ('A', 'other', None),
    'spore': ('C', 'status', 'sleep_clause'), 'sleeppowder': ('C', 'status', 'sleep_clause'), 'hypnosis': ('C', 'status', 'sleep_clause'),
    'refresh': ('C', 'status', 'user_brn_par_psn'), 'sacredfire': ('C', 'status', 'user_frozen'),
    'psychoboost': ('C', 'self_stat_drop', 'white_herb'), 'facade': ('C', 'other', 'guts_status'),
    'aromatherapy': ('C', 'status', 'party_statused'), 'trick': ('C', 'item_removal', 'item_knocked'),
}
TIER_B = {'irontail': 'target_stat_change', 'signalbeam': 'volatile', 'steelwing': 'self_boost', 'meteormash': 'self_boost'}
# move id -> (effect class, the integration merge from which the engine models it): watched and recorded, never guarded.
RETIRED = {'destinybond': ('volatile', '286a5ffe'), 'hyperbeam': ('multi_turn', '8b212312')}
WATCHED = frozenset(GUARDED) | frozenset(TIER_B) | frozenset(RETIRED)
# The guarded tiers in force while records carried no per-entry `guarded` (before 49530d2a; GUARDED unchanged from 2c5c7b80 until
# RETIRED): reach's bounds for those records use these, so a later guard set does not move them.
PRE_FLAG_TIERS = {**dict.fromkeys(('hyperbeam', 'destinybond', 'transform', 'yawn', 'wrap', 'perishsong', 'trick', 'counter', 'mirrorcoat',
    'curse', 'superpower', 'knockoff', 'charm', 'lovelykiss', 'bonemerang', 'gigadrain'), 'A'),
    **dict.fromkeys(('spore', 'sleeppowder', 'hypnosis', 'refresh', 'sacredfire', 'psychoboost', 'facade', 'aromatherapy'), 'C')}
# Showdown's Gen 3 move table (Dex.forGen(3)): every standard move with flags.recharge (the live actor's own-recharge guard).
RECHARGE_MOVES = frozenset(('hyperbeam', 'blastburn', 'hydrocannon', 'frenzyplant'))
STATUSES = ('par', 'brn', 'psn', 'tox', 'slp', 'frz')
STATUS_CODES = {1: 'par', 2: 'brn', 3: 'psn', 4: 'tox', 5: 'slp', 6: 'frz'}   # the engine's state.status (phase_d_live_worlds)
FACADE_STATUSES = frozenset(('brn', 'par', 'psn', 'tox'))
WHITE_HERB_SPECIES = frozenset(('deoxys', 'deoxysattack'))


def move_id(value):
    return ''.join(c for c in str(value).lower() if c.isascii() and c.isalnum())


# Our own request ------------------------------------------------------------------------------------------------------------------
def own_moves(request_line):
    """The owner's move id on each action lane 0-3, from its |request| line: active[0].moves in request order (the lane
    order: showdown._legal_action_mask), each normalized like a Showdown id (its 'id', else its 'move'), None on a lane
    without a move (a forceSwitch or wait request has no active row: four Nones). None for a line that is not a JSON
    |request| object."""
    packet = _packet(request_line)
    if packet is None:return None
    active = packet.get('active');active = active[0] if type(active) is list and active and type(active[0]) is dict else {}
    moves = active.get('moves');moves = moves if type(moves) is list else []
    out = []
    for lane in range(4):
        move = moves[lane] if lane < len(moves) and type(moves[lane]) is dict else {}
        name = next((move[k] for k in ('id', 'move') if type(move.get(k)) is str and move_id(move[k])), None)
        out.append(None if name is None else move_id(name))
    return tuple(out)


def recharge_lane(moves, action):
    """True when `action` is a move lane whose own move is a recharge move (RECHARGE_MOVES); False for a switch lane, an
    unknown move or unknown moves (None)."""
    return moves is not None and type(action) is int and 0 <= action < 4 and moves[action] in RECHARGE_MOVES


def recharge_guarded(moves, searched, greedy):
    """The own-recharge guard: True when the searched action goes INTO a recharge move greedy did not choose (play greedy).
    An override away from greedy's recharge move, or the same action, is never guarded."""
    return searched != greedy and recharge_lane(moves, searched) and not recharge_lane(moves, greedy)


def _packet(request_line):
    if type(request_line) is not str or not request_line.startswith('|request|'):return None
    try:packet = json.loads(request_line[len('|request|'):])
    except ValueError:return None
    return packet if type(packet) is dict else None


def _condition(text):
    """(fainted, status) of a protocol condition ('250/300 par', '0 fnt', '100/100')."""
    parts = str(text or '').split()
    if not parts:return False, ''
    if parts[0] == '0' or 'fnt' in parts[1:]:return True, ''
    return False, next((x for x in parts[1:] if x in STATUSES), '')


def _species(details):
    return move_id(str(details or '').split(',')[0])


# Conditions ---------------------------------------------------------------------------------------------------------------------
def _member():
    return dict(status='', fainted=False, sleep_source=None, white_herb=None, guts=None, knocked=False)


def conditions(lines, owner, own_request_line=None):
    """Both sides' guard conditions at the root (module docstring): {'own': side, 'opp': side}, each dict(active, members:
    {species: dict(status, fainted, sleep_source 'rest'/'other'/None, white_herb, guts)}, sleep_clause: this side's sleep
    moves would hit Sleep Clause (the OTHER side has a non-fainted Pokemon asleep from a non-allied source), party_statused,
    item_knocked: this side has a non-fainted Pokemon whose item was publicly knocked off; each member's `knocked`).
    Ours: the public lines, then our own request (statuses, faints, item and ability exact); the opponent's: public lines only.
    `owner` 0 or 1 (p1 / p2)."""
    if owner not in (0, 1):raise ValueError(f'owner 0 or 1 required, got {owner!r}')
    seats = ('p1', 'p2');own, opp = seats[owner], seats[1 - owner]
    sides = {s: dict(active=None, members={}) for s in seats};names = {}
    def member(seat, name):
        species = names.get((seat, name)) or move_id(name)
        return sides[seat]['members'].setdefault(species, _member()), species
    for line in lines or ():
        parts = str(line).split('|')
        if len(parts) < 3 or parts[0] != '':continue
        kind = parts[1];seat, _, name = parts[2].partition(':');seat = seat.strip()[:2];name = name.strip()
        if seat not in sides:continue
        if kind in ('switch', 'drag', 'replace') and len(parts) > 3:
            species = _species(parts[3]);names[(seat, name)] = species;sides[seat]['active'] = species
            m = sides[seat]['members'].setdefault(species, _member())
            if len(parts) > 4:
                m['fainted'], status = _condition(parts[4])
                if status != m['status']:_set_status(m, status, 'other' if status == 'slp' else None)   # an unchanged status keeps its source
            continue
        m, species = member(seat, name)
        tags = [x.strip().lower() for x in parts[3:]]
        if kind == 'faint':m['fainted'] = True;_set_status(m, '', None)
        elif kind == '-status' and len(parts) > 3:
            status = move_id(parts[3]);source = 'rest' if any(t == '[from] move: rest' for t in tags) else 'other'
            _set_status(m, status if status in STATUSES else m['status'], source if status == 'slp' else None)
        elif kind == '-curestatus' and len(parts) > 3:
            if move_id(parts[3]) in STATUSES:_set_status(m, '', None)
        elif kind == '-cureteam':
            for x in sides[seat]['members'].values():_set_status(x, '', None)
        elif kind in ('-damage', '-heal', '-sethp') and len(parts) > 3:
            fainted, status = _condition(parts[3])
            if fainted:m['fainted'] = True;_set_status(m, '', None)
            elif status != m['status']:_set_status(m, status, 'other' if status == 'slp' else None)
        elif kind == '-enditem':
            m['white_herb'] = False
            if '[from] move: knock off' in tags:m['knocked'] = True
        elif kind == '-item' and len(parts) > 3:m['white_herb'] = move_id(parts[3]) == 'whiteherb'
    packet = _packet(own_request_line)
    if packet is not None:
        for row in ((packet.get('side') or {}).get('pokemon') or ()):
            if type(row) is not dict:continue
            species = _species(row.get('details'));m = sides[own]['members'].setdefault(species, _member())
            m['fainted'], status = _condition(row.get('condition'))
            if status != m['status']:_set_status(m, status, 'other' if status == 'slp' else None)
            m['white_herb'] = move_id(row.get('item') or '') == 'whiteherb'
            m['guts'] = move_id(row.get('ability') or row.get('baseAbility') or '') == 'guts'
            if row.get('active') is True:sides[own]['active'] = species
    for species, m in sides[opp]['members'].items():_opponent_defaults(species, m)
    out = {}
    for key, seat in (('own', own), ('opp', opp)):
        other = sides[opp if seat == own else own]['members'].values()
        members = sides[seat]['members']
        out[key] = dict(active=sides[seat]['active'], members={s: dict(m) for s, m in sorted(members.items())},
            sleep_clause=any(not m['fainted'] and m['status'] == 'slp' and m['sleep_source'] != 'rest' for m in other),
            party_statused=any(not m['fainted'] and m['status'] for m in members.values()), item_knocked=_knocked(members))
    return out


def _knocked(members):
    """Whether a side has a non-fainted Pokemon whose item was publicly knocked off (Trick's condition; `knocked` is absent
    from a record written before it: False)."""
    return any(not m.get('fainted') and m.get('knocked') for m in members.values())


def _opponent_defaults(species, m):
    """The opponent's public pool facts on a member whose item or ability is not known otherwise: Deoxys and Deoxys-Attack hold
    White Herb, every pool Facade user has Guts."""
    if m['white_herb'] is None:m['white_herb'] = species in WHITE_HERB_SPECIES
    if m['guts'] is None:m['guts'] = True
    return m


def _set_status(m, status, source):
    m['status'] = status;m['sleep_source'] = source if status == 'slp' else None


def child_side(side, members, *, opponent=False):
    """A side's conditions at a turn-2 child: the root's `side` (item and ability facts) with the child's own statuses,
    `members` [(species, status code, alive, rest_origin)] read from the child state (the sleep source: rest_origin). A member
    the root does not know (an unrevealed opponent in this world) takes the opponent's pool facts when `opponent`."""
    out = dict(side, members={s: dict(m) for s, m in side['members'].items()})
    for species, code, alive, rest in members:
        if not species:continue
        m = out['members'].setdefault(species, _member())
        if opponent:_opponent_defaults(species, m)
        m['status'] = STATUS_CODES.get(int(code), '');m['fainted'] = not alive
        m['sleep_source'] = ('rest' if rest else 'other') if m['status'] == 'slp' else None
    out['party_statused'] = any(not m['fainted'] and m['status'] for m in out['members'].values())
    out['item_knocked'] = _knocked(out['members'])
    return out


def sleep_clause(other):
    """Whether a sleep move against the side `other` (its conditions) would hit Sleep Clause."""
    return any(not m['fainted'] and m['status'] == 'slp' and m['sleep_source'] != 'rest' for m in other['members'].values())


def guarded(move, side, user, *, other=None):
    """(tier, effect class) when `move` is guarded for a side (its conditions `side`, its active `user`), else None. `other`:
    the other side's conditions (Sleep Clause; Trick: either side's knocked-off item), default the side's own flags."""
    entry = GUARDED.get(move)
    if entry is None:return None
    tier, effect, condition = entry
    if condition is None:return tier, effect
    m = side['members'].get(user) or _member()
    holds = dict(sleep_clause=lambda: sleep_clause(other) if other is not None else side['sleep_clause'],
        user_brn_par_psn=lambda: m['status'] in FACADE_STATUSES, user_frozen=lambda: m['status'] == 'frz',
        white_herb=lambda: bool(m['white_herb']), guts_status=lambda: bool(m['guts']) and m['status'] in FACADE_STATUSES,
        party_statused=lambda: side['party_statused'],
        item_knocked=lambda: bool(side.get('item_knocked')) or bool((other or {}).get('item_knocked')))[condition]()
    return (tier, effect) if holds else None


# Masses and lanes --------------------------------------------------------------------------------------------------------------
def tier_of(move):
    """A watched move's tier: 'A' or 'C' (GUARDED), 'B' (TIER_B), 'R' (RETIRED), else None."""
    return GUARDED[move][0] if move in GUARDED else 'B' if move in TIER_B else 'R' if move in RETIRED else None


def masses(logits, legal):
    """The prior over the legal lanes: softmax of `logits` restricted to `legal` (9 each), in float64 with math.fsum; 0 on
    an illegal lane; all zeros when no lane is legal or a legal logit is not finite."""
    lanes = [a for a in range(9) if legal[a]]
    if not lanes or any(not math.isfinite(float(logits[a])) for a in lanes):return [0.] * 9
    top = max(float(logits[a]) for a in lanes);w = {a: math.exp(float(logits[a]) - top) for a in lanes};total = math.fsum(w.values())
    return [w[a] / total if a in w else 0. for a in range(9)]


def lanes_of(moves, charge=None, usable=None):
    """The four move-lane labels of a side at a node (module docstring): moves [4] (ids, None for an empty slot), the charge
    move (an id or None) and usable [4] (None: all)."""
    if charge:return [charge, None, None, None]
    if usable is not None and not any(usable):return ['struggle', None, None, None]
    return [m if m else None for m in list(moves)[:4]] + [None] * (4 - len(list(moves)[:4]))


# The guard ----------------------------------------------------------------------------------------------------------------------
def evaluate(cond, *, own_turn1, peer_turn1, turn2, turn2_available=True, choice=None, threshold=THRESHOLD, ability=None):
    """The guard record (module docstring). `cond`: conditions(); `own_turn1`: dict(lanes [4], logits [9], legal [9]) (our
    root prior, the same in every world); `peer_turn1`: per world dict(world, lanes [4], logits, legal, requested, user);
    `turn2`: per expanded cell and side dict(world, cell, side 'own'/'opp', lanes, logits, legal, user, own_side, opp_side)
    (the child's conditions: child_side), `turn2_available` False when cells expanded but no labels were captured;
    `choice`: the search's own action (None: none). `ability`: the ability guard's inputs (ability_evaluate); with them the
    record gains `ability` (its own record) and `tripped` / `arms` include it ('ability'); without them the record is unchanged."""
    found = {};recorded = dict(turn1=[], turn2=[]);a_moves = set();b_moves = set()
    def note(move, side, source, mass, world=None, cell=None, tier=None, effect=None):
        key = (move, side, source)
        r = found.setdefault(key, dict(move=move, side=side, source=source, tier=tier, effect_class=effect, max_mass=0., worlds=set()))
        r['max_mass'] = max(r['max_mass'], mass)
        if world is not None:r['worlds'].add(world)
    def lanes_masses(side_name, side, user, lanes, logits, legal, source, world=None, cell=None, other=None):
        p = masses(logits, legal)
        for lane in range(4):
            move = lanes[lane] if lane < len(lanes) else None
            if move not in WATCHED or not legal[lane]:continue
            if source == 'turn1':
                if move in TIER_B:b_moves.add(move)
                elif guarded(move, side, user, other=other):a_moves.add(move)
            hit = guarded(move, side, user, other=other)
            if p[lane] > 0:
                item = dict(side=side_name, world=world, move=move, mass=p[lane])
                if cell is not None:   # turn 2: the guard's own decision at this child (reach), the move's tier and its user
                    item.update(cell=cell, guarded=hit is not None, tier=tier_of(move), user=user)
                recorded[source].append(item)
            if hit is not None and p[lane] >= threshold:note(move, side_name, source, p[lane], world, cell, *hit)
    own, opp = cond['own'], cond['opp']
    lanes_masses('own', own, own['active'], own_turn1['lanes'], own_turn1['logits'], own_turn1['legal'], 'turn1', other=opp)
    for w in peer_turn1:
        if not w.get('requested', True):continue
        lanes_masses('opp', opp, w.get('user') or opp['active'], w['lanes'], w['logits'], w['legal'], 'turn1', w['world'], other=own)
    for c in turn2:
        side = c.get('own_side') if c['side'] == 'own' else c.get('opp_side')
        side = side or (own if c['side'] == 'own' else opp)
        other = (c.get('opp_side') or opp) if c['side'] == 'own' else (c.get('own_side') or own)
        lanes_masses(c['side'], side, c.get('user'), c['lanes'], c['logits'], c['legal'], 'turn2', c['world'], c['cell'], other)
    if not turn2_available:found[(None, None, 'turn2_unavailable')] = dict(move=None, side=None, source='turn2_unavailable', tier=None,
        effect_class=None, max_mass=None, worlds=set())
    own_guarded = [guarded(m, own, own['active'], other=opp) if m else None for m in own_turn1['lanes']]
    c_hit = type(choice) is int and 0 <= choice < 4 and own_guarded[choice] is not None
    if c_hit:
        tier, effect = own_guarded[choice];note(own_turn1['lanes'][choice], 'own', 'choice', 1., tier=tier, effect=effect)
    found_all = sorted(({**r, 'worlds': sorted(r['worlds'])} for r in found.values()), key=lambda r: (str(r['source']), str(r['side']), str(r['move'])))
    b_turn1 = any(r['source'] == 'turn1' for r in found_all)
    any_cell_reasons = [r for r in found_all if r['source'] in ('turn2', 'turn2_unavailable')]   # the pre-reach arm, kept for analysis
    any_cell_turn2 = bool(any_cell_reasons)
    # The live turn-2 arm: the reach M_s over the recorded turn-2 masses (their per-child `guarded`), our root prior and each
    # world's reply prior (a waiting world: reply 0), p_w uniform over the worlds peer_turn1 lists (the priced worlds).
    by_world = {w['world']: dict(requested=w.get('requested', True), logits=w['logits'], kept=w['legal']) for w in peer_turn1}
    worlds = [by_world.get(i) for i in range(max(by_world) + 1)] if by_world else []
    r = reach(dict(masses=recorded, turn2_available=turn2_available, conditions=cond), own_turn1['logits'], own_turn1['legal'], worlds, threshold=threshold)
    b_turn2 = bool(r['tripped']);b = b_turn1 or b_turn2
    arms = [name for name, hit in (('turn1', b_turn1), ('reach', b_turn2 and not r['unavailable']), ('unavailable', r['unavailable']),
        ('choice', c_hit)) if hit]
    # reasons list only what tripped: turn 1, the reach per tripping side (its contributing tiers, moves and worlds), the choice,
    # an unavailable capture
    reasons = [x for x in found_all if x['source'] in ('turn1', 'choice')]
    if r['unavailable']:reasons.append(dict(move=None, side=None, source='turn2_unavailable', tier=None, effect_class=None, max_mass=None, worlds=[]))
    elif b_turn2:
        for side in ('own', 'opp'):
            m_side = r[f'm_{side}_lower']   # the exact M here (every live turn-2 entry carries `guarded`)
            if m_side < threshold:continue
            parts = {t: v for t, v in r['tiers']['lower'][side].items() if v > 0};detail = r['contributions']['lower'][side]
            reasons.append(dict(move=None, side=side, source='reach', tier=None, effect_class=None, max_mass=m_side, worlds=detail['worlds'],
                tiers=parts, moves=detail['moves']))
    reasons.sort(key=lambda x: (str(x['source']), str(x['side']), str(x['move'])))
    tiers = sorted({x['tier'] for x in reasons if x['tier']} | {t for x in reasons for t in x.get('tiers') or ()})
    out = dict(schema=SCHEMA, rule=RULE, threshold=threshold, tripped=b or c_hit, b=b, b_turn1=b_turn1, b_turn2=b_turn2, c=c_hit, choice=choice,
        m_own=r['m_own'], m_opp=r['m_opp'], arms=arms,
        reach=dict(m_own=r['m_own'], m_opp=r['m_opp'], tripped=r['tripped'], unavailable=r['unavailable'], tiers=None if r['tiers'] is None else r['tiers']['lower']),
        any_cell_turn2=any_cell_turn2, any_cell_b=b_turn1 or any_cell_turn2, any_cell_tripped=b_turn1 or any_cell_turn2 or c_hit,
        any_cell_reasons=any_cell_reasons, reasons=reasons, tiers=tiers, a=bool(a_moves), a_moves=sorted(a_moves),
        tier_b_legal=bool(b_moves), tier_b_moves=sorted(b_moves), own_lanes=list(own_turn1['lanes']),
        own_guarded_lanes=[None if g is None else list(g) for g in own_guarded], conditions=cond, masses=recorded,
        turn2_available=turn2_available)
    if ability is not None:
        ab = ability_evaluate(own_turn1, peer_turn1, ability, choice=choice, threshold=threshold)
        out['ability'] = ab;out['tripped'] = bool(out['tripped'] or ab['tripped'])
        if ab['tripped']:out['arms'] = out['arms'] + ['ability']
    return out


# Reach (the live turn-2 arm): the expected turn-2 guarded mass under the search's own root and reply priors ---------------------
def _psychoboost(record, side):
    """Whether Psycho Boost is certainly guarded for `side` (the lower bound): a revealed Deoxys or Deoxys-Attack of that side
    holds its White Herb in the root's item facts (a turn-2 child keeps them; one Deoxys form per side, species clause). With
    none revealed the user may be Deoxys-Speed, which learns Psycho Boost and holds no White Herb: then it counts only in the
    upper bound."""
    members = ((record.get('conditions') or {}).get(side) or {}).get('members') or {}
    return any(bool(m.get('white_herb')) for species, m in members.items() if species in WHITE_HERB_SPECIES)


def reach(record, root_logits, root_legal, worlds, *, threshold=THRESHOLD):
    """The guard's turn-2 reach, the live turn-2 arm (RULE 'reach'; the RR's bounds for older records), per side s:
    M_s = sum_w p_w sum_(a,b) pi_own(a) pi_peer,w(b) q_(w,ab),s, with q the side's turn-2 prior mass on moves guarded for it at
    child cell a*9 + b of world w (record['masses']['turn2']), pi_own our root prior (root_logits over root_legal), pi_peer,w
    world w's turn-1 peer prior (its root_peer_logits over kept; a world whose opponent waits has the single reply 0 at mass 1:
    the pricer expands its cells (a, 0)), p_w uniform over the priced worlds. `worlds`: the row's worlds in block order, each
    dict(requested, logits, kept, priced=True). Exact when every turn-2 entry carries `guarded` (the guard's own decision at
    that child, its recorded tier the bucket: a record written before a move was RETIRED keeps its M); otherwise bounds, under
    the tiers in force when such records were written (PRE_FLAG_TIERS): lower counts tier A and Psycho Boost when a revealed
    Deoxys or Deoxys-Attack of that side holds
    its White Herb (_psychoboost), upper every
    tier-C mass as guarded. tripped: True / False when the bounds (or the exact M) agree about THRESHOLD on either side, None
    when they straddle it; a record without its turn-2 masses (an error record, an unavailable capture) trips, fail closed.
    contributions: per bound and side, each counted move's weighted mass and the worlds it came from (the reasons' detail)."""
    out = dict(exact=None, m_own=None, m_opp=None, m_own_lower=None, m_own_upper=None, m_opp_lower=None, m_opp_upper=None, tiers=None,
        tripped=True, tripped_lower=True, tripped_upper=True, unavailable=True, contributions=None)
    entries = ((record or {}).get('masses') or {}).get('turn2')
    if entries is None or not record.get('turn2_available', True):return out
    if not entries:   # no turn-2 mass on any watched move: M is 0 whatever the worlds
        zero = {side: {'A': 0., 'C': 0.} for side in ('own', 'opp')}
        out.update(exact=True, m_own=0., m_opp=0., m_own_lower=0., m_own_upper=0., m_opp_lower=0., m_opp_upper=0.,
            tiers=dict(lower=zero, upper={k: dict(v) for k, v in zero.items()}), tripped=False, tripped_lower=False, tripped_upper=False, unavailable=False,
            contributions={scope: {side: dict(moves={}, worlds=[]) for side in ('own', 'opp')} for scope in ('lower', 'upper')})
        return out
    priced = [w for w, x in enumerate(worlds) if x is not None and x.get('priced', True)]
    if not priced:return out
    own = masses(root_logits, root_legal);p = 1 / len(priced)
    peer = {w: (masses(worlds[w]['logits'], worlds[w]['kept']) if worlds[w].get('requested') else [1.] + [0.] * 8) for w in priced}
    exact = all('guarded' in e for e in entries)
    sums = {scope: {side: {'A': [], 'C': []} for side in ('own', 'opp')} for scope in ('lower', 'upper')}
    moves = {scope: {side: {} for side in ('own', 'opp')} for scope in ('lower', 'upper')}
    where = {scope: {side: set() for side in ('own', 'opp')} for scope in ('lower', 'upper')}
    for e in entries:
        # A record keeps its M under a later guard set (RETIRED): an exact entry counts by its own guard's decision (the bucket: its
        # recorded tier, else the move's current or pre-flag tier, else A); an older record's bounds use the tiers in force when
        # it was written.
        w = e['world']
        if exact:
            if not e['guarded']:continue
            tier = next((t for t in (e.get('tier'), tier_of(e['move']), PRE_FLAG_TIERS.get(e['move'])) if t in ('A', 'C')), 'A')
        else:
            tier = PRE_FLAG_TIERS.get(e['move'])
            if tier not in ('A', 'C'):continue
        if w not in peer:continue
        a, b = divmod(int(e['cell']), 9);weight = p * own[a] * peer[w][b] * float(e['mass'])
        if not weight:continue
        if exact:counted = dict(lower=True, upper=True)
        else:counted = dict(lower=tier == 'A' or e['move'] == 'psychoboost' and _psychoboost(record, e['side']), upper=True)
        for scope, yes in counted.items():
            if yes:
                sums[scope][e['side']][tier].append(weight);moves[scope][e['side']].setdefault(e['move'], []).append(weight)
                where[scope][e['side']].add(w)
    tiers = {scope: {side: {t: math.fsum(v) for t, v in per.items()} for side, per in sides.items()} for scope, sides in sums.items()}
    m = {(scope, side): math.fsum(tiers[scope][side].values()) for scope in tiers for side in ('own', 'opp')}
    lower, upper = (max(m[(scope, 'own')], m[(scope, 'opp')]) >= threshold for scope in ('lower', 'upper'))
    out.update(exact=exact, m_own_lower=m[('lower', 'own')], m_own_upper=m[('upper', 'own')], m_opp_lower=m[('lower', 'opp')],
        m_opp_upper=m[('upper', 'opp')], tiers=tiers, tripped_lower=lower, tripped_upper=upper, tripped=True if lower else False if not upper else None,
        unavailable=False, contributions={scope: {side: dict(moves={k: math.fsum(v) for k, v in sorted(moves[scope][side].items())},
            worlds=sorted(where[scope][side])) for side in ('own', 'opp')} for scope in ('lower', 'upper')})
    if exact:out.update(m_own=m[('lower', 'own')], m_opp=m[('lower', 'opp')])
    return out


def error_record(error):
    """The fail-closed record of a root whose guard inputs could not be read: tripped, with the error."""
    return dict(schema=SCHEMA, rule=RULE, threshold=THRESHOLD, tripped=True, b=True, b_turn1=False, b_turn2=False, c=False, choice=None,
        m_own=None, m_opp=None, arms=['error'], reach=dict(m_own=None, m_opp=None, tripped=True, unavailable=True, tiers=None),
        any_cell_turn2=False, any_cell_b=True, any_cell_tripped=True, any_cell_reasons=[],
        reasons=[dict(move=None, side=None, source='guard_error', tier=None, effect_class=None, max_mass=None, worlds=[])], tiers=[], a=False,
        a_moves=[], tier_b_legal=False, tier_b_moves=[], own_lanes=None, own_guarded_lanes=None, conditions=None, masses=None,
        turn2_available=False, error=str(error)[:500])


def trips(record, choice):
    """The guard for an action `choice` of ours at a root whose record is `record` (the RR's per-operator rule): dict(tripped,
    b, c). (b) is the root's; (c) is whether `choice` is a move lane guarded for us."""
    c = type(choice) is int and 0 <= choice < 4 and (record.get('own_guarded_lanes') or [None] * 4)[choice] is not None
    ab = record.get('ability') or {};b = bool(record['b']) or bool(ab.get('b'))
    if type(choice) is int and 0 <= choice < 4 and (ab.get('own_guarded_lanes') or [None] * 4)[choice]:   # the ability guard's (c)
        c = c or any(a in ab.get('arms', ABILITY_ARMS).get(f, ()) for a in ('choice',) for f in ab['own_guarded_lanes'][choice])
    return dict(tripped=b or c, b=b, c=c)


# Abilities the engine does not model (the ability guard; the lead's decisions of 2026-11-12) -------------------------------------
# Showdown f76228a (gen3 inherits data/mods/gen4/abilities.ts): Wonder Guard (gen4 :530-543) blocks every damaging move that is not
# super-effective, except a Status move or a typeless one (Struggle, gen4/moves.ts:1273-1281; Beat Up, gen4/moves.ts:81; the Future
# Sight / Doom Desire hit, gen3/moves.ts:228, 244), immune moves included; Blaze / Torrent / Overgrow / Swarm (gen4 :25-35 and
# alike) give their type 1.5x base power while hp <= maxhp / 3; Poison Point, Flame Body, Effect Spore (gen3 abilities.ts:12-30,
# 135-144) may status a contact attacker that dealt damage. The engine models none of them (Static and Rough Skin it does). A lane
# is guarded at a node for a family when its move's outcome there depends on that family's ability (ability_lanes); the families'
# arms are those of the move guard: turn 1 (b), the reach (b), a turn-1 switch-in into a WG or CONTACT holder (b, new), the choice
# (c). ABILITY_ARMS is the declared configuration (a family shipped choice-only keeps 'choice' and, for WG, 'switchin').
ABILITY_SCHEMA = 'pokezero.depthn-ability-guard.v1'
ABILITY_FAMILIES = {'wonderguard': 'WG', 'blaze': 'PINCH', 'torrent': 'PINCH', 'overgrow': 'PINCH', 'swarm': 'PINCH',
                    'poisonpoint': 'CONTACT', 'flamebody': 'CONTACT', 'effectspore': 'CONTACT'}
FAMILIES = ('WG', 'PINCH', 'CONTACT')
PINCH_TYPES = {'blaze': 'fire', 'torrent': 'water', 'overgrow': 'grass', 'swarm': 'bug'}
WG_TYPELESS = frozenset(('struggle', 'beatup', 'futuresight', 'doomdesire'))
ABILITY_ARMS = {'WG': ('turn1', 'reach', 'switchin', 'choice'), 'PINCH': ('turn1', 'reach', 'choice'),
                'CONTACT': ('turn1', 'reach', 'switchin', 'choice')}
# The root PINCH rule: 'two_thirds' guards a pinch-type lane at the root while hp <= 2 maxhp / 3 and the other side puts >= THRESHOLD
# on damaging lanes in that world (the holder may fall to a third during turn 1, then move); 'third' only while hp <= maxhp / 3.
# Turn-2 children are always exact ('third' on the child's own hp).
PINCH_ROOT = 'two_thirds'


def effectiveness(move_type, types, chart):
    """The type multiplier of a `move_type` hit on a Pokemon of `types` (Gen 3 chart[attack][defend]; a typeless move: 1)."""
    out = 1.
    if not move_type:return out
    for t in types or ():
        if t:out *= chart[move_type][t]
    return out


def ability_lanes(user, target, lanes, catalog, chart, *, pinch='third'):
    """{family: [bool] * 4}: the side's move lanes 0-3 whose outcome at this node depends on an ability the engine does not model.
    `user` / `target`: this side's and the other side's active, dict(ability, hp, max_hp, types, hp_type) (None: no such
    Pokemon); `lanes`: the side's move ids per lane (lanes_of); `catalog`: move id -> dict(category, type, contact); `chart`: the
    Gen 3 type chart by name; `pinch`: 'third' or 'two_thirds' (the root rule). Hidden Power's type is the user's own."""
    out = {f: [False] * 4 for f in FAMILIES}
    user = user or {};target = target or {}
    pinch_type = PINCH_TYPES.get(user.get('ability'));hp, top = int(user.get('hp') or 0), int(user.get('max_hp') or 0)
    low = hp > 0 and top > 0 and 3 * hp <= (2 * top if pinch == 'two_thirds' else top)
    wg = target.get('ability') == 'wonderguard';contact = ABILITY_FAMILIES.get(target.get('ability')) == 'CONTACT'
    for lane in range(4):
        move = lanes[lane] if lane < len(lanes) else None;info = catalog.get(move) if move else None
        if not info or info.get('category') == 'Status':continue
        kind = user.get('hp_type') if move.startswith('hiddenpower') and user.get('hp_type') else info.get('type')
        if wg and move not in WG_TYPELESS and effectiveness(kind, target.get('types'), chart) < 2:out['WG'][lane] = True
        if pinch_type and low and kind == pinch_type:out['PINCH'][lane] = True
        if contact and info.get('contact'):out['CONTACT'][lane] = True
    return out


def damaging_lanes(lanes, catalog):
    """[bool] * 4: the lanes whose move is not a Status move (the root PINCH rule's other side)."""
    return [bool(m) and bool(catalog.get(m)) and catalog[m].get('category') != 'Status' for m in (list(lanes) + [None] * 4)[:4]]


def holder_family(mon):
    """WG or CONTACT for a Pokemon holding such an ability (a turn-1 switch-in that takes the other side's move), else None."""
    family = ABILITY_FAMILIES.get((mon or {}).get('ability'))
    return family if family in ('WG', 'CONTACT') else None


def _nine(mask):
    return [bool(x) for x in mask[:4]] + [False] * 5


def ability_evaluate(own_turn1, peer_turn1, ability, *, choice=None, threshold=THRESHOLD, arms=None, pinch_root=None):
    """The ability guard over one root (module section above). `own_turn1`: dict(logits, legal) (our root prior); `peer_turn1`: the
    move guard's per-world list (world, logits, legal, requested); `ability`: dict(
      turn1: per world dict(world, own, opp), each side dict(third, two_thirds: {family: [bool] * 4} (ability_lanes under each rule),
        damaging [bool] * 4);
      switchin: per world [dict(side 'own'/'opp' (who switches), lane 4-8, family, holder, against [bool] * 4 (the other side's lanes
        guarded against the incoming holder))];
      turn2: [dict(world, cell, side, lanes {family: [bool] * 4} (exact at the child), logits, legal)]).
    Returns the ability record: tripped, b (b_turn1 / b_reach / b_switchin), c, per family and arm the masses that decide them, the
    reasons (source ability_turn1 / ability_reach / ability_switchin / ability_choice, each with its family), own_guarded_lanes,
    and for any other configuration: reach_subsets (the reach M of every family subset) and PINCH's turn1_third / choice_third
    (the exact root rule), so a record decides every ABILITY_ARMS / PINCH_ROOT variant exactly (ability_trips)."""
    arms = ABILITY_ARMS if arms is None else arms;pinch_root = PINCH_ROOT if pinch_root is None else pinch_root
    on = lambda family, arm: arm in arms.get(family, ())
    own_p = masses(own_turn1['logits'], own_turn1['legal'])
    peers = {w['world']: w for w in peer_turn1}
    priced = sorted(peers);pw = 1 / len(priced) if priced else 0.
    peer_p = {w: (masses(peers[w]['logits'], peers[w]['legal']) if peers[w].get('requested', True) else None) for w in priced}
    by = {f: dict(turn1=dict(own=0., opp=0.), reach=dict(own=0., opp=0.), switchin=dict(own=0., opp=0.), choice=False) for f in FAMILIES}
    by['PINCH'].update(turn1_third=dict(own=0., opp=0.), choice_third=False)   # PINCH under the exact root rule too (the fallback)
    own_third = [False] * 4
    reasons = [];own_lanes = [set() for _ in range(4)]
    def root_mask(side, family, other_damaging):
        if family != 'PINCH' or pinch_root == 'third' or other_damaging < threshold:return side['third'][family]
        return side['two_thirds'][family]
    # turn 1 (b) and the choice (c): any world
    turn1_worlds = {}
    for t in ability.get('turn1') or ():
        w = t['world'];pp = peer_p.get(w)
        dmg = dict(own=math.fsum(own_p[k] for k in range(4) if t['own']['damaging'][k]),
                   opp=0. if pp is None else math.fsum(pp[k] for k in range(4) if t['opp']['damaging'][k]))
        for side, prior, other in (('own', own_p, 'opp'), ('opp', pp, 'own')):
            third = t[side]['third']['PINCH']
            if side == 'own':own_third = [a or bool(b) for a, b in zip(own_third, third)]
            if prior is not None:
                by['PINCH']['turn1_third'][side] = max(by['PINCH']['turn1_third'][side], math.fsum(prior[k] for k in range(4) if third[k]))
            for f in FAMILIES:
                mask = root_mask(t[side], f, dmg[other])
                if side == 'own':
                    for k in range(4):
                        if mask[k]:own_lanes[k].add(f)
                if prior is None:continue
                m = math.fsum(prior[k] for k in range(4) if mask[k])
                by[f]['turn1'][side] = max(by[f]['turn1'][side], m)
                if m >= threshold and on(f, 'turn1'):
                    x = turn1_worlds.setdefault((side, f), [0., set()]);x[0] = max(x[0], m);x[1].add(w)
    for (side, f), (m, ws) in sorted(turn1_worlds.items()):
        reasons.append(dict(source='ability_turn1', family=f, side=side, max_mass=m, worlds=sorted(ws)))
    c_hit = type(choice) is int and 0 <= choice < 4 and any(on(f, 'choice') for f in own_lanes[choice])
    for f in FAMILIES:by[f]['choice'] = type(choice) is int and 0 <= choice < 4 and f in own_lanes[choice]
    by['PINCH']['choice_third'] = type(choice) is int and 0 <= choice < 4 and own_third[choice]
    if c_hit:reasons.append(dict(source='ability_choice', family=sorted(f for f in own_lanes[choice] if on(f, 'choice')), side='own',
        max_mass=1., worlds=[]))
    # the turn-1 switch-in: S = sum_w p_w pi_switcher(lane) sum_x pi_other(x) against(x), credited to the attacking side
    sums = {(f, s): [] for f in FAMILIES for s in ('own', 'opp')};where = {(f, s): set() for f in FAMILIES for s in ('own', 'opp')}
    for w, entries in (ability.get('switchin') or {}).items():
        w = int(w);pp = peer_p.get(w)
        if w not in peers or pp is None:continue   # a waiting opponent neither switches nor attacks
        for e in entries:
            mover, attacker = (own_p, pp) if e['side'] == 'own' else (pp, own_p)
            hit = math.fsum(attacker[k] for k in range(4) if e['against'][k])
            weight = pw * mover[int(e['lane'])] * hit
            if weight > 0:
                side = 'opp' if e['side'] == 'own' else 'own';sums[(e['family'], side)].append(weight);where[(e['family'], side)].add(w)
    for (f, side), v in sums.items():by[f]['switchin'][side] = math.fsum(v)
    b_switchin = False
    for f in FAMILIES:
        for side in ('own', 'opp'):
            m = by[f]['switchin'][side]
            if m >= threshold and on(f, 'switchin'):
                b_switchin = True;reasons.append(dict(source='ability_switchin', family=f, side=side, max_mass=m, worlds=sorted(where[(f, side)])))
    # the reach: M_s over the enabled families' lanes (their union per lane), and each family's own M
    msum = {s: [] for s in ('own', 'opp')};fsum = {(f, s): [] for f in FAMILIES for s in ('own', 'opp')}
    subsets = [tuple(f for i, f in enumerate(FAMILIES) if bits >> i & 1) for bits in range(1, 2 ** len(FAMILIES))]
    ssum = {(sub, s): [] for sub in subsets for s in ('own', 'opp')}   # the reach M of every family subset (any configuration, exact)
    rworlds = {s: set() for s in ('own', 'opp')}
    for e in ability.get('turn2') or ():
        w = e['world']
        if w not in peers:continue
        reply = peer_p[w] if peer_p[w] is not None else [1.] + [0.] * 8   # a waiting world: reply 0 at mass 1 (as reach)
        a, b = divmod(int(e['cell']), 9);weight = pw * own_p[a] * reply[b]
        if not weight:continue
        q = masses(e['logits'], e['legal'])
        union = [any(e['lanes'][f][k] for f in FAMILIES if on(f, 'reach')) for k in range(4)]
        u = math.fsum(q[k] for k in range(4) if union[k])
        if u:msum[e['side']].append(weight * u);rworlds[e['side']].add(w)
        for f in FAMILIES:
            v = math.fsum(q[k] for k in range(4) if e['lanes'][f][k])
            if v:fsum[(f, e['side'])].append(weight * v)
        for sub in subsets:
            v = math.fsum(q[k] for k in range(4) if any(e['lanes'][f][k] for f in sub))
            if v:ssum[(sub, e['side'])].append(weight * v)
    m_reach = {s: math.fsum(v) for s, v in msum.items()}
    for (f, s), v in fsum.items():by[f]['reach'][s] = math.fsum(v)
    b_reach = max(m_reach.values()) >= threshold
    for s in ('own', 'opp'):
        if m_reach[s] >= threshold:
            reasons.append(dict(source='ability_reach', family=sorted(f for f in FAMILIES if on(f, 'reach') and by[f]['reach'][s] > 0), side=s,
                max_mass=m_reach[s], worlds=sorted(rworlds[s]), families={f: by[f]['reach'][s] for f in FAMILIES if by[f]['reach'][s] > 0}))
    b_turn1 = bool(turn1_worlds);b = b_turn1 or b_reach or b_switchin
    reasons.sort(key=lambda x: (x['source'], x['side'], str(x['family'])))
    return dict(schema=ABILITY_SCHEMA, threshold=threshold, arms={f: list(arms.get(f, ())) for f in FAMILIES}, pinch_root=pinch_root,
        tripped=b or c_hit, b=b, b_turn1=b_turn1, b_reach=b_reach, b_switchin=b_switchin, c=c_hit, m_own=m_reach['own'], m_opp=m_reach['opp'],
        families=by, reasons=reasons, own_guarded_lanes=[sorted(x) or None for x in own_lanes],
        reach_subsets={'+'.join(sub): {s: math.fsum(ssum[(sub, s)]) for s in ('own', 'opp')} for sub in subsets},
        cells=dict(turn1=len(ability.get('turn1') or ()), switchin=sum(len(v) for v in (ability.get('switchin') or {}).values()),
                   turn2=len(ability.get('turn2') or ())))


def ability_trips(ab, *, arms=None, pinch_root=None, threshold=None):
    """Whether an ability record trips under another configuration (`arms` per family, `pinch_root`), from the record alone:
    dict(tripped, b_turn1, b_reach, b_switchin, c). The record's own configuration reproduces its own tripped."""
    arms = ab['arms'] if arms is None else arms;pinch_root = ab['pinch_root'] if pinch_root is None else pinch_root
    threshold = ab['threshold'] if threshold is None else threshold;fam = ab['families']
    on = lambda f, arm: arm in arms.get(f, ())
    def turn1(f):
        m = fam[f]['turn1_third'] if f == 'PINCH' and pinch_root == 'third' else fam[f]['turn1']
        return max(m.values()) >= threshold
    def choice(f):
        return bool(fam[f]['choice_third'] if f == 'PINCH' and pinch_root == 'third' else fam[f]['choice'])
    b_turn1 = any(on(f, 'turn1') and turn1(f) for f in FAMILIES)
    sub = '+'.join(f for f in FAMILIES if on(f, 'reach'))
    b_reach = bool(sub) and max(ab['reach_subsets'][sub].values()) >= threshold
    b_switchin = any(on(f, 'switchin') and max(fam[f]['switchin'].values()) >= threshold for f in FAMILIES)
    c = any(on(f, 'choice') and choice(f) for f in FAMILIES)
    return dict(tripped=b_turn1 or b_reach or b_switchin or c, b_turn1=b_turn1, b_reach=b_reach, b_switchin=b_switchin, c=c)
