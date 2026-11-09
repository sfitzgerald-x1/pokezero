"""UNREPRESENTED_EFFECT: the DepthN live guard against pricing a move the GPU engine does not model (pure Python, no JAX).

The GPU device engine keys every move effect by name and reads no effect fields (generate_teams.cjs:75-81); the static
table of its handling of the Gen 3 random-battle pool is reports/gpu-engine-move-effect-table-20261109.md. GUARDED holds
the moves whose missing effect carries over into the price (the lead's decision):
- tier A, unconditional: hyperbeam, destinybond, transform, yawn, wrap, perishsong, trick, counter, mirrorcoat, curse,
  superpower, knockoff, charm, lovelykiss, bonemerang, gigadrain;
- tier C, guarded for a side only while its condition holds (`conditions`): spore, sleeppowder, hypnosis while Sleep Clause
  would bind (the target's side already has a non-fainted Pokemon asleep from a non-allied source: Showdown f76228a1
  data/rulesets.ts sleepclausemod, whose onSetStatus skips a sleeper whose statusState.source is its ally, i.e. Rest; a
  Yawn or move-induced sleep counts; a wake (-curestatus slp), a faint or a party cure ends it), refresh while the user is
  burned, paralyzed or poisoned, sacredfire while the user is frozen, psychoboost while the user's White Herb is unconsumed
  (ours: the request's item; the opponent's: Deoxys and Deoxys-Attack always hold it in the pool, until an -enditem or an
  -item for that Pokemon is seen), facade while the user has Guts and a burn, paralysis or poison (every pool Facade user
  has Guts; ours: the request's ability), aromatherapy while the user's party has a statused, non-fainted Pokemon (ours:
  the request; the opponent's: its public statuses).
TIER_B (irontail, signalbeam, steelwing, meteormash: chance secondaries) is never guarded, only reported (tier_b_legal).

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
record), the any-cell fields above, reasons (per move x side x
source: turn1 / turn2 / choice, with the max mass, the tier and the effect class), the (a) flag (a guarded move legal for
either side at turn 1 in any world: reporting only), tier_b_legal, our guarded lanes, the conditions per side and the masses
of every GUARDED or TIER_B move (turn 1 per side and world, turn 2 per side, world and cell a * 9 + b with our lane a and the
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
    'hyperbeam': ('A', 'multi_turn', None), 'destinybond': ('A', 'volatile', None), 'transform': ('A', 'other', None),
    'yawn': ('A', 'volatile', None), 'wrap': ('A', 'volatile', None), 'perishsong': ('A', 'volatile', None),
    'trick': ('A', 'item_removal', None), 'counter': ('A', 'retaliation', None), 'mirrorcoat': ('A', 'retaliation', None),
    'curse': ('A', 'self_boost', None), 'superpower': ('A', 'self_stat_drop', None), 'knockoff': ('A', 'item_removal', None),
    'charm': ('A', 'target_stat_change', None), 'lovelykiss': ('A', 'status', None), 'bonemerang': ('A', 'other', None),
    'gigadrain': ('A', 'other', None),
    'spore': ('C', 'status', 'sleep_clause'), 'sleeppowder': ('C', 'status', 'sleep_clause'), 'hypnosis': ('C', 'status', 'sleep_clause'),
    'refresh': ('C', 'status', 'user_brn_par_psn'), 'sacredfire': ('C', 'status', 'user_frozen'),
    'psychoboost': ('C', 'self_stat_drop', 'white_herb'), 'facade': ('C', 'other', 'guts_status'),
    'aromatherapy': ('C', 'status', 'party_statused'),
}
TIER_B = {'irontail': 'target_stat_change', 'signalbeam': 'volatile', 'steelwing': 'self_boost', 'meteormash': 'self_boost'}
WATCHED = frozenset(GUARDED) | frozenset(TIER_B)
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
    return dict(status='', fainted=False, sleep_source=None, white_herb=None, guts=None)


def conditions(lines, owner, own_request_line=None):
    """Both sides' guard conditions at the root (module docstring): {'own': side, 'opp': side}, each dict(active, members:
    {species: dict(status, fainted, sleep_source 'rest'/'other'/None, white_herb, guts)}, sleep_clause: this side's sleep
    moves would hit Sleep Clause (the OTHER side has a non-fainted Pokemon asleep from a non-allied source), party_statused).
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
        elif kind == '-enditem':m['white_herb'] = False
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
            party_statused=any(not m['fainted'] and m['status'] for m in members.values()))
    return out


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
    return out


def sleep_clause(other):
    """Whether a sleep move against the side `other` (its conditions) would hit Sleep Clause."""
    return any(not m['fainted'] and m['status'] == 'slp' and m['sleep_source'] != 'rest' for m in other['members'].values())


def guarded(move, side, user, *, other=None):
    """(tier, effect class) when `move` is guarded for a side (its conditions `side`, its active `user`), else None. `other`:
    the other side's conditions (Sleep Clause), default the side's own sleep_clause flag."""
    entry = GUARDED.get(move)
    if entry is None:return None
    tier, effect, condition = entry
    if condition is None:return tier, effect
    m = side['members'].get(user) or _member()
    holds = dict(sleep_clause=lambda: sleep_clause(other) if other is not None else side['sleep_clause'],
        user_brn_par_psn=lambda: m['status'] in FACADE_STATUSES, user_frozen=lambda: m['status'] == 'frz',
        white_herb=lambda: bool(m['white_herb']), guts_status=lambda: bool(m['guts']) and m['status'] in FACADE_STATUSES,
        party_statused=lambda: side['party_statused'])[condition]()
    return (tier, effect) if holds else None


# Masses and lanes --------------------------------------------------------------------------------------------------------------
def tier_of(move):
    """A watched move's tier: 'A' or 'C' (GUARDED), 'B' (TIER_B), else None."""
    return GUARDED[move][0] if move in GUARDED else 'B' if move in TIER_B else None


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
def evaluate(cond, *, own_turn1, peer_turn1, turn2, turn2_available=True, choice=None, threshold=THRESHOLD):
    """The guard record (module docstring). `cond`: conditions(); `own_turn1`: dict(lanes [4], logits [9], legal [9]) (our
    root prior, the same in every world); `peer_turn1`: per world dict(world, lanes [4], logits, legal, requested, user);
    `turn2`: per expanded cell and side dict(world, cell, side 'own'/'opp', lanes, logits, legal, user, own_side, opp_side)
    (the child's conditions: child_side), `turn2_available` False when cells expanded but no labels were captured;
    `choice`: the search's own action (None: none)."""
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
    reasons = sorted(({**r, 'worlds': sorted(r['worlds'])} for r in found.values()), key=lambda r: (str(r['source']), str(r['side']), str(r['move'])))
    b_turn1 = any(r['source'] == 'turn1' for r in reasons)
    any_cell_turn2 = any(r['source'] in ('turn2', 'turn2_unavailable') for r in reasons)   # the pre-reach rule, kept for analysis
    # The live turn-2 arm: the reach M_s over the recorded turn-2 masses (their per-child `guarded`), our root prior and each
    # world's reply prior (a waiting world: reply 0), p_w uniform over the worlds peer_turn1 lists (the priced worlds).
    by_world = {w['world']: dict(requested=w.get('requested', True), logits=w['logits'], kept=w['legal']) for w in peer_turn1}
    worlds = [by_world.get(i) for i in range(max(by_world) + 1)] if by_world else []
    r = reach(dict(masses=recorded, turn2_available=turn2_available, conditions=cond), own_turn1['logits'], own_turn1['legal'], worlds, threshold=threshold)
    b_turn2 = bool(r['tripped']);b = b_turn1 or b_turn2
    arms = [name for name, hit in (('turn1', b_turn1), ('reach', b_turn2 and not r['unavailable']), ('unavailable', r['unavailable']),
        ('choice', c_hit)) if hit]
    return dict(schema=SCHEMA, rule=RULE, threshold=threshold, tripped=b or c_hit, b=b, b_turn1=b_turn1, b_turn2=b_turn2, c=c_hit, choice=choice,
        m_own=r['m_own'], m_opp=r['m_opp'], arms=arms,
        reach=dict(m_own=r['m_own'], m_opp=r['m_opp'], tripped=r['tripped'], unavailable=r['unavailable'], tiers=None if r['tiers'] is None else r['tiers']['lower']),
        any_cell_turn2=any_cell_turn2, any_cell_b=b_turn1 or any_cell_turn2, any_cell_tripped=b_turn1 or any_cell_turn2 or c_hit,
        reasons=reasons, tiers=sorted({r['tier'] for r in reasons if r['tier']}), a=bool(a_moves), a_moves=sorted(a_moves),
        tier_b_legal=bool(b_moves), tier_b_moves=sorted(b_moves), own_lanes=list(own_turn1['lanes']),
        own_guarded_lanes=[None if g is None else list(g) for g in own_guarded], conditions=cond, masses=recorded,
        turn2_available=turn2_available)


# Reach (report only): the expected turn-2 guarded mass under the search's own root and reply priors ------------------------------
def _psychoboost(record, side):
    """Whether Psycho Boost is certainly guarded for `side` (the lower bound): a revealed Deoxys or Deoxys-Attack of that side
    holds its White Herb in the root's item facts (a turn-2 child keeps them; one Deoxys form per side, species clause). With
    none revealed the user may be Deoxys-Speed, which learns Psycho Boost and holds no White Herb: then it counts only in the
    upper bound."""
    members = ((record.get('conditions') or {}).get(side) or {}).get('members') or {}
    return any(bool(m.get('white_herb')) for species, m in members.items() if species in WHITE_HERB_SPECIES)


def reach(record, root_logits, root_legal, worlds, *, threshold=THRESHOLD):
    """The guard's turn-2 reach (the lead's narrower, report-only scope), per side s:
    M_s = sum_w p_w sum_(a,b) pi_own(a) pi_peer,w(b) q_(w,ab),s, with q the side's turn-2 prior mass on moves guarded for it at
    child cell a*9 + b of world w (record['masses']['turn2']), pi_own our root prior (root_logits over root_legal), pi_peer,w
    world w's turn-1 peer prior (its root_peer_logits over kept; a world whose opponent waits has the single reply 0 at mass 1:
    the pricer expands its cells (a, 0)), p_w uniform over the priced worlds. `worlds`: the row's worlds in block order, each
    dict(requested, logits, kept, priced=True). Exact when every turn-2 entry carries `guarded` (the guard's own decision at
    that child); otherwise bounds: lower counts tier A and Psycho Boost when a revealed Deoxys or Deoxys-Attack of that side holds
    its White Herb (_psychoboost), upper every
    tier-C mass as guarded. tripped: True / False when the bounds (or the exact M) agree about THRESHOLD on either side, None
    when they straddle it; a record without its turn-2 masses (an error record, an unavailable capture) trips, fail closed."""
    out = dict(exact=None, m_own=None, m_opp=None, m_own_lower=None, m_own_upper=None, m_opp_lower=None, m_opp_upper=None, tiers=None,
        tripped=True, tripped_lower=True, tripped_upper=True, unavailable=True)
    entries = ((record or {}).get('masses') or {}).get('turn2')
    if entries is None or not record.get('turn2_available', True):return out
    if not entries:   # no turn-2 mass on any watched move: M is 0 whatever the worlds
        zero = {side: {'A': 0., 'C': 0.} for side in ('own', 'opp')}
        out.update(exact=True, m_own=0., m_opp=0., m_own_lower=0., m_own_upper=0., m_opp_lower=0., m_opp_upper=0.,
            tiers=dict(lower=zero, upper={k: dict(v) for k, v in zero.items()}), tripped=False, tripped_lower=False, tripped_upper=False, unavailable=False)
        return out
    priced = [w for w, x in enumerate(worlds) if x is not None and x.get('priced', True)]
    if not priced:return out
    own = masses(root_logits, root_legal);p = 1 / len(priced)
    peer = {w: (masses(worlds[w]['logits'], worlds[w]['kept']) if worlds[w].get('requested') else [1.] + [0.] * 8) for w in priced}
    exact = all('guarded' in e for e in entries)
    sums = {scope: {side: {'A': [], 'C': []} for side in ('own', 'opp')} for scope in ('lower', 'upper')}
    for e in entries:
        tier = tier_of(e['move']);w = e['world']
        if tier not in ('A', 'C') or w not in peer:continue
        a, b = divmod(int(e['cell']), 9);weight = p * own[a] * peer[w][b] * float(e['mass'])
        if not weight:continue
        if exact:counted = dict(lower=e['guarded'], upper=e['guarded'])
        else:counted = dict(lower=tier == 'A' or e['move'] == 'psychoboost' and _psychoboost(record, e['side']), upper=True)
        for scope, yes in counted.items():
            if yes:sums[scope][e['side']][tier].append(weight)
    tiers = {scope: {side: {t: math.fsum(v) for t, v in per.items()} for side, per in sides.items()} for scope, sides in sums.items()}
    m = {(scope, side): math.fsum(tiers[scope][side].values()) for scope in tiers for side in ('own', 'opp')}
    lower, upper = (max(m[(scope, 'own')], m[(scope, 'opp')]) >= threshold for scope in ('lower', 'upper'))
    out.update(exact=exact, m_own_lower=m[('lower', 'own')], m_own_upper=m[('upper', 'own')], m_opp_lower=m[('lower', 'opp')],
        m_opp_upper=m[('upper', 'opp')], tiers=tiers, tripped_lower=lower, tripped_upper=upper, tripped=True if lower else False if not upper else None,
        unavailable=False)
    if exact:out.update(m_own=m[('lower', 'own')], m_opp=m[('lower', 'opp')])
    return out


def error_record(error):
    """The fail-closed record of a root whose guard inputs could not be read: tripped, with the error."""
    return dict(schema=SCHEMA, rule=RULE, threshold=THRESHOLD, tripped=True, b=True, b_turn1=False, b_turn2=False, c=False, choice=None,
        m_own=None, m_opp=None, arms=['error'], reach=dict(m_own=None, m_opp=None, tripped=True, unavailable=True, tiers=None),
        any_cell_turn2=False, any_cell_b=True, any_cell_tripped=True,
        reasons=[dict(move=None, side=None, source='guard_error', tier=None, effect_class=None, max_mass=None, worlds=[])], tiers=[], a=False,
        a_moves=[], tier_b_legal=False, tier_b_moves=[], own_lanes=None, own_guarded_lanes=None, conditions=None, masses=None,
        turn2_available=False, error=str(error)[:500])


def trips(record, choice):
    """The guard for an action `choice` of ours at a root whose record is `record` (the RR's per-operator rule): dict(tripped,
    b, c). (b) is the root's; (c) is whether `choice` is a move lane guarded for us."""
    c = type(choice) is int and 0 <= choice < 4 and (record.get('own_guarded_lanes') or [None] * 4)[choice] is not None
    return dict(tripped=bool(record['b']) or c, b=bool(record['b']), c=c)
