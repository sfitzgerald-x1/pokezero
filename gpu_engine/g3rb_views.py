"""Human players' views from public Gen 3 random-battle replays (the opponent-history design, section 4, steps 3-6).

usage: python -m gpu_engine.g3rb_views INPUT.jsonl OUT --showdown-root R --node-binary N --model-checkpoint PT
           [--shard I --shards K] [--limit N] [--imputations M]

INPUT.jsonl: one replay per line, {index, id, vintage, rating, log} (index: its row in the sanitized table; vintage: strict,
extended or outside; out-of-pin replays are never in it).

Each replay (the hashed, sanitized log of a game) tagged strict, extended or outside (revealed facts consistent with the pinned
Gen 3 random-battle support) is rebuilt, per seat, at every decision of that seat:

- Decision points. A move decision at every |turn| line (prefix: the log through it), for each seat with an unfainted active
  mon. A switch decision before every replacement |switch| that follows that seat's |faint| (forced) or its own Baton Pass
  (mid-turn); a |drag| is never a decision, and the leads are not decisions.
- Label. Move decision: the seat's first action of the turn: a |switch| of its own before any action of its own (voluntary
  switch, labelled 'switch <species>'), or its |move| line ('move <id>'; a [from]lockedmove line or Struggle labels the move
  and sets forced). Censored, never imputed: |cant| (sleep, freeze, paralysis, flinch, recharge, Truant, ...; the chosen
  move is unrevealed), the active fainting before acting, no action at all. Switch decision: the replacement's species.
- Own team (imputation index m). Every species the seat revealed, with everything it revealed about it over the whole game
  (the player knows its own team): moves used, an announced ability, an item seen (Leftovers / a berry / an ended item), its
  level; one variant of the pinned source matching those facts, drawn with a seeded RNG. The slots it never revealed draw a
  species and a variant from the source. Party order: the lead first, then the rest in order of appearance, the unrevealed
  last; every switch-in swaps the incoming mon with slot 0 (Showdown's side.switchIn). Stats are the generator's (85 EVs,
  31 IVs, Hidden Power IVs, attack dump); HP = the log's own exact HP; PP = max PP minus the uses seen (Pressure's extra PP is
  not modelled); an item ended in public is gone. Choice lock, Disable, Taunt, Encore and 0 PP disable moves as Showdown's
  request does; trapping is not modelled (switches stay legal; labels are unaffected).
- View. pokezero.policy_opponent_view.build_policy_opponent_view on the public prefix and that request: the same parser,
  belief source and observation writer as the live player; the label is decoded on the view's own state
  (teacher_capture.action_index_from_choice_string).

Writes shard-<I>.npz in fp_reply_data's layout (categorical_ids, numeric_features, token_type_ids, attention_mask, legal,
label (-1 censored), forced (a forced switch, as in fp_reply_data), locked (a [from]lockedmove or Struggle label), turn,
kind (0 move, 1 switch), seat (0/1), replay (row index in CLEAN), vintage, rating, imputation, censor). Censored rows keep
their reason (cant and fainted_before_acting only happen to a chosen move: a trainer can treat them as 'some move' partial
labels; recharge had no choice), and shard-<I>.json with the counts by vintage, kind and censor reason."""
import argparse
import collections
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np

STATS = ('atk', 'def', 'spa', 'spd', 'spe')
VINTAGES = ('strict', 'extended', 'outside')
CENSORS = ('none', 'cant', 'fainted_before_acting', 'no_action', 'label_undecodable', 'view_error', 'recharge')
KINDS = ('move', 'switch')
FORMAT = 'gen3randombattle'


def toid(x):return re.sub(r'[^a-z0-9]', '', str(x or '').lower())


def species_id(details):
    s = toid(str(details).split(',')[0])
    return 'unown' if s.startswith('unown') else s


def ident(text):
    """('p1', 'Nick') from 'p1a: Nick' / 'p1: Nick'; None otherwise."""
    m = re.match(r'(p[12])[a-z]?: (.*)$', str(text or ''))
    return (m.group(1), m.group(2)) if m else None


def exact_hp(text):
    """(cur, max) from a server-side condition ('238/273 par'); None for '0 fnt' or a malformed one."""
    m = re.match(r'(\d+)/(\d+)', str(text or ''))
    return (int(m.group(1)), int(m.group(2))) if m else None


PRIVATE = 'opponent [silent] ability, opponent Struggle activation, -hint'


def seat_lines(lines, seat, counts=None):
    """The log as `seat` received it. These logs are the server's: exact HP for both sides (the view builder projects the
    opponent's to the public percent) and lines Showdown sends to one side only, dropped here for the other side: an
    opponent's [silent] ability announcement (Pressure; our live wires carry it for our own mons only), an opponent's
    -activate ... move: Struggle, and every -hint line. Every other [silent] kind is on both sides of our wires (public)."""
    out = []
    for line in lines:
        p = line.split('|');kind = p[1] if len(p) > 1 else '';who = ident(p[2]) if len(p) > 2 else None
        opponent = who is not None and who[0] != seat
        if kind == '-hint' or opponent and (kind == '-ability' and '[silent]' in line or kind == '-activate' and 'move: Struggle' in line):
            if counts is not None:counts['private_line_dropped:' + kind] += 1
            continue
        out.append(line)
    return out


def condition(text):
    """(percent 0..100, status or '') from a public condition ('45/100 par', '0 fnt', '100/100')."""
    head, *rest = str(text or '').split()
    if head.startswith('0') and (not rest or rest[0] == 'fnt') and '/' not in head:return 0, 'fnt'
    cur, _, top = head.partition('/')
    try:pct = float(cur) * 100. / float(top or 100)
    except ValueError:return None, ''
    return pct, (rest[0] if rest and rest[0] != 'fnt' else ('fnt' if pct == 0 else ''))


# ---- the log ---------------------------------------------------------------------------------------------------------------------

def facts(lines):
    """Per seat: {nick: dict(species, details, level, moves (ordered), ability, item)} over the whole game, and the lead's nick."""
    mons = {'p1': {}, 'p2': {}};lead = {}
    for line in lines:
        p = line.split('|')
        if len(p) < 3:continue
        kind, who = p[1], ident(p[2])
        if kind in ('switch', 'drag', 'replace') and who and len(p) > 3:
            seat, nick = who;level = re.search(r', L(\d+)', p[3])
            mons[seat].setdefault(nick, dict(species=species_id(p[3]), details=p[3], level=int(level.group(1)) if level else 100, moves=[],
                                             ability=None, item=None))
            lead.setdefault(seat, nick)
            continue
        if not who or who[1] not in mons[who[0]]:continue
        mon = mons[who[0]][who[1]]
        if kind == 'move' and len(p) > 3 and '[from]' not in line:
            m = toid(p[3])
            if m and m != 'struggle' and m not in mon['moves']:mon['moves'].append(m)
        elif kind == '-ability' and len(p) > 3:
            mon['ability'] = 'trace' if '[from] ability: Trace' in line else toid(p[3])
        elif kind == '-enditem' and len(p) > 3 and '[from] move:' not in line and mon['item'] is None:
            mon['item'] = toid(p[3])
        if '[of]' not in line and kind not in ('-ability', 'move'):
            a = re.search(r'\[from\] ability: ([^|]+)', line);i = re.search(r'\[from\] item: ([^|]+)', line)
            if a and mon['ability'] is None:mon['ability'] = toid(a.group(1))
            if i and mon['item'] is None:mon['item'] = toid(i.group(1))
    return mons, lead


def decisions(lines):
    """[(prefix_end, seat, kind, label, locked, censor, turn)] in log order: prefix_end is the number of lines of the prefix.

    Replacements after faints: both seats choose at once when both actives fainted (Showdown sends forceSwitch to both, then runs
    the switches), so both prefixes end before the first replacement switch. A mon KOed by Pursuit while switching out is no
    new decision: Showdown carries out the switch chosen at turn start, which labels that turn's decision."""
    out, turn, active, fainted = [], 0, {'p1': None, 'p2': None}, {'p1': False, 'p2': False}
    passing = {'p1': False, 'p2': False};pursued = {'p1': None, 'p2': None};requeued = {'p1': False, 'p2': False}
    waiting = {'p1': None, 'p2': None}   # a seat that chose its replacement while the other seat's switch-in was already under way
    for i, line in enumerate(lines):
        p = line.split('|')
        if len(p) < 2:continue
        kind = p[1];who = ident(p[2]) if len(p) > 2 else None
        if kind == 'turn':
            turn = int(p[2]) if len(p) > 2 and p[2].isdigit() else turn + 1;pursued = {'p1': None, 'p2': None};waiting = {'p1': None, 'p2': None}
            for seat in ('p1', 'p2'):
                if active[seat] and not fainted[seat]:out.append([i + 1, seat, 'move', None, False, None, turn])
        elif kind == '-activate' and who and 'move: Pursuit' in line:pursued[who[0]] = who[1]   # Pursuit hits this seat's switching mon
        elif kind in ('switch', 'replace') and who:
            seat = who[0];other = 'p2' if seat == 'p1' else 'p1'
            if requeued[seat]:requeued[seat] = False   # the switch chosen at turn start, carried out after a Pursuit KO
            elif active[seat] is not None and (fainted[seat] or passing[seat]):   # a replacement after a faint, or Baton Pass's switch
                end = i
                if fainted[seat]:   # both replaced at once: the first switch ends both prefixes, once (a later re-replacement is its own)
                    if waiting[seat] is not None:end, waiting[seat] = waiting[seat], None
                    elif fainted[other] and active[other] is not None:waiting[other] = i
                out.append([end, seat, 'switch', 'switch ' + species_id(p[3]), False, 'none', turn])
            active[seat], fainted[seat], passing[seat], pursued[seat] = who[1], False, False, None
        elif kind == 'drag' and who:active[who[0]], fainted[who[0]] = who[1], False
        elif kind == 'faint' and who and who[1] == active[who[0]]:
            fainted[who[0]] = True
            if pursued[who[0]] == who[1] and not passing[who[0]]:requeued[who[0]] = True   # a Baton Passer KOed still gets a replacement request
        elif kind == 'move' and who and len(p) > 3 and toid(p[3]) == 'batonpass' and '[from]' not in line:passing[who[0]] = True
    # labels of the move decisions: the seat's first own action after the turn line
    for d in out:
        if d[2] != 'move':continue
        start, seat = d[0], d[1];d[5] = 'no_action';pursued_now = False
        for j, line in enumerate(lines[start:], start):
            p = line.split('|')
            if len(p) < 3:continue
            kind, who = p[1], ident(p[2])
            if kind == 'turn':break
            if not who or who[0] != seat:continue
            if kind == '-activate' and 'move: Pursuit' in line:pursued_now = True;continue
            if kind == 'switch':d[3], d[5] = 'switch ' + species_id(p[3]), 'none';break
            if kind == 'move' and len(p) > 3 and ('[from]' not in line or '[from]lockedmove' in line):
                m = toid(p[3]);d[3], d[5] = 'move ' + m, 'none';d[4] = '[from]lockedmove' in line or m == 'struggle';break
            if kind == 'cant':d[5] = 'recharge' if len(p) > 3 and toid(p[3]) == 'recharge' else 'cant';break   # recharge: no choice existed
            if kind == 'faint':
                if pursued_now:   # KOed by Pursuit while switching out: the chosen switch is the seat's next switch this turn
                    nxt = next((l for l in lines[j + 1:] if l.startswith(f'|switch|{seat}') or l.startswith('|turn|')), '')
                    if nxt.startswith('|switch|'):d[3], d[5] = 'switch ' + species_id(nxt.split('|')[3]), 'none';break
                d[5] = 'fainted_before_acting';break
    return [tuple(d) for d in out]


# ---- the seat's own team -----------------------------------------------------------------------------------------------------------

class Tables:
    """The pinned source, the dex and the generator tables, loaded once."""
    def __init__(self, showdown_root, node):
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat import load_gen3_randbat_source_cached
        from gpu_engine import fp_reply_reweight as rw
        self.rw = rw;self.source = load_gen3_randbat_source_cached(showdown_root);self.dex = load_showdown_dex_cached(showdown_root)
        self.gen = rw.generator_tables(showdown_root, node)
        self.universes = {species_id(u.species): u for u in self.source.universes.values()}


def draw_team(tables, seat_facts, lead, rng):
    """[member] of 6 in the initial party order, or None when a revealed mon matches no pinned variant. member: dict(nick, species,
    level, details, moves, ability, item, revealed (bool))."""
    team = []
    order = [lead] + [n for n in seat_facts if n != lead] if lead in seat_facts else list(seat_facts)
    for nick in order:
        f = seat_facts[nick];u = tables.universes.get(f['species'])
        if u is None:return None
        variants = u.filter_variants(revealed_moves=tuple(f['moves']), revealed_ability=f['ability'], revealed_item=f['item'])
        if not variants:variants = u.filter_variants(revealed_moves=tuple(f['moves']), revealed_ability=f['ability'])   # item evidence is loose
        if not variants:return None
        v = rng.choice(list(variants))
        team.append(dict(nick=nick, species=f['species'], level=f['level'], details=f['details'], moves=list(v.moves), ability=toid(v.ability),
                         item=toid(v.item), revealed=True))
    taken = {m['species'] for m in team};pool = sorted(s for s in tables.universes if s not in taken)
    while len(team) < 6 and pool:
        s = pool.pop(rng.randrange(len(pool)));u = tables.universes[s];v = rng.choice(list(u.variants))
        name = tables.dex.species_info(s).name if tables.dex.species_info(s) else s
        team.append(dict(nick=name, species=s, level=u.level, details=f'{name}, L{u.level}', moves=list(v.moves), ability=toid(v.ability),
                         item=toid(v.item), revealed=False))
    return team


def max_hp(tables, member):
    info = tables.dex.species_info(member['species'])
    if member['species'] == 'shedinja':return 1
    hidden = [m for m in member['moves'] if m.startswith('hiddenpower') and m != 'hiddenpower']
    iv = tables.gen['hp_ivs'].get(hidden[0][len('hiddenpower'):], {}).get('hp', 31) if hidden else 31
    return (2 * int(info.base_stats['hp']) + iv + 85 // 4) * member['level'] // 100 + member['level'] + 10


def family(m):
    """A move as the log names it: every typed Hidden Power is 'hiddenpower'."""
    return 'hiddenpower' if m.startswith('hiddenpower') else m


STRUGGLE = {'move': 'Struggle', 'id': 'struggle', 'pp': 1, 'maxpp': 1, 'target': 'randomNormal', 'disabled': False}


def choice(req, label):
    """The /choose body for `label` on `req`, by position (decoding names fails on a typed Hidden Power id and Unown's forms):
    'move N' (N the 1-based slot in the active request; any Hidden Power matches the typed one) or 'switch N' (the party
    position of that species); None when it is not there."""
    kind, _, name = label.partition(' ')
    if kind == 'move':
        entries = (req.get('active') or [{}])[0].get('moves') or []
        for j, e in enumerate(entries):
            ident = gv_toid(e.get('id'))
            if ident == name or name == 'hiddenpower' and ident.startswith('hiddenpower'):return f'move {j + 1}'
        return None
    for j, mon in enumerate(req['side']['pokemon']):
        if species_id(mon['details']) == name and not mon.get('active'):return f'switch {j + 1}'
    return None


def gv_toid(x):return re.sub(r'\d+$', '', toid(x))


def request(tables, team, lines, seat, kind, struggle=False):
    """Showdown's request for `seat` at the end of the public prefix `lines` (module docstring), and the imputed flags."""
    rw = tables.rw;order = list(range(len(team)));by_nick = {m['nick']: i for i, m in enumerate(team)}
    hp = {i: (100., '') for i in order};exact = {};used = collections.Counter();active = None;since = None
    items = {i: m['item'] for i, m in enumerate(team)}   # the current item: an ended one (eaten, knocked off) is gone
    volatile = dict(disable=None, taunt=False, encore=None);last_move = None
    for line in lines:
        p = line.split('|')
        if len(p) < 3:continue
        k, who = p[1], ident(p[2])
        if not who or who[0] != seat or who[1] not in by_nick:continue
        i = by_nick[who[1]]
        if k in ('switch', 'drag', 'replace'):
            if active is None:active = i
            else:
                a, b = order.index(i), 0;order[a], order[b] = order[b], order[a];active = i
            since, last_move = [], None;volatile = dict(disable=None, taunt=False, encore=None)
            hp[i] = condition(p[4]) if len(p) > 4 else hp[i];e = exact_hp(p[4]) if len(p) > 4 else None
            if e:exact[i] = e
        elif k in ('-damage', '-heal', '-sethp') and len(p) > 3:
            c = condition(p[3]);hp[i] = c if c[0] is not None else hp[i];e = exact_hp(p[3])
            if e:exact[i] = e
        elif k == '-status' and len(p) > 3:hp[i] = (hp[i][0], p[3])
        elif k == '-curestatus':hp[i] = (hp[i][0], '')
        elif k == '-cureteam':hp.update({j: (v[0], '') for j, v in hp.items() if v[1] != 'fnt'})
        elif k == '-enditem':items[i] = ''
        elif k == '-item' and len(p) > 3:items[i] = toid(p[3])
        elif k == 'faint':hp[i] = (0., 'fnt')
        elif k == 'move' and len(p) > 3 and '[from]' not in line:
            m = family(toid(p[3]));used[(i, m)] += 1;last_move = m
            if since is not None:since.append(m)
        elif k == '-start' and len(p) > 3:
            what = toid(p[3])
            if what == 'disable' and len(p) > 4:volatile['disable'] = family(toid(p[4]))
            elif what in ('taunt', 'movetaunt'):volatile['taunt'] = True
            elif what == 'encore':volatile['encore'] = last_move
        elif k == '-end' and len(p) > 3:
            what = toid(p[3]).replace('move', '') if toid(p[3]).startswith('move') else toid(p[3])
            if what == 'disable':volatile['disable'] = None
            elif what == 'taunt':volatile['taunt'] = False
            elif what == 'encore':volatile['encore'] = None
    pokemon = []
    for slot in order:
        m = team[slot];top = max_hp(tables, m);pct, status = hp[slot]
        cur = 0 if status == 'fnt' or not pct else max(1, min(top, int(round(pct * top / 100.))))
        if slot in exact and status != 'fnt':cur, top = exact[slot]   # the server log's own exact HP (the seat knows it)
        stats = rw.generator_stats(tables.dex.species_info(m['species']).base_stats, m['level'], m['moves'], tables.dex, tables.gen)
        pokemon.append({'ident': f'{seat}: {m["nick"]}', 'details': m['details'], 'condition': f'{cur}/{top}' + (f' {status}' if status and status != 'fnt' else '')
                        if cur else '0 fnt', 'active': slot == active, 'stats': stats, 'moves': [rw.request_move_id(x) for x in m['moves']],
                        'baseAbility': m['ability'], 'ability': m['ability'], 'item': items[slot], 'pokeball': 'pokeball'})
    out = {'side': {'id': seat, 'name': seat, 'pokemon': pokemon}}
    if kind == 'switch':out['forceSwitch'] = [True];return out
    m = team[active];entries = []
    locked = items[active] == 'choiceband' and bool(since)
    for x in m['moves']:
        f = family(x)
        disabled = (locked and f != since[0]) or f == volatile['disable'] or (volatile['encore'] is not None and f != volatile['encore']) \
            or (volatile['taunt'] and (tables.dex.move_info(x) is not None and tables.dex.move_info(x).gen3_category == 'Status'))
        e = rw.fresh_move_entry(tables.dex, x, disabled);e['pp'] = max(0, e['maxpp'] - used[(active, f)])
        e['disabled'] = bool(e['disabled'] or e['pp'] <= 0);entries.append(e)   # Showdown disables a move with no PP left
    out['active'] = [{'moves': [dict(STRUGGLE)] if struggle else entries}]   # Showdown offers Struggle alone when no move is usable
    return out


# ---- one replay ------------------------------------------------------------------------------------------------------------------

def replay_rows(tables, model, row_index, row, vintage, imputations, counts):
    """The rows of one replay (both seats, every decision, `imputations` own-team draws each)."""
    from pokezero.policy_opponent_view import PolicyOpponentViewError, build_policy_opponent_view
    from pokezero.teacher_capture import action_index_from_choice_string
    lines = [l for l in str(row['log']).split('\n') if l.startswith('|')];all_facts, lead = facts(lines);points = decisions(lines)
    for seat in ('p1', 'p2'):seat_lines(lines, seat, counts)   # count the private lines each seat's view drops
    rows = []
    for imp in range(imputations):
        for seat in ('p1', 'p2'):
            seed = int(hashlib.sha256(f'{row["id"]}:{seat}:{imp}'.encode()).hexdigest()[:16], 16)
            team = draw_team(tables, all_facts[seat], lead.get(seat), random.Random(seed))
            if team is None:counts['seat_team_unmatched'] += 1;continue
            for end, s, kind, label, locked, censor, turn in points:
                if s != seat:continue
                obs = None
                try:
                    public = seat_lines(lines[:end], seat)   # what the seat received
                    req = request(tables, team, public, seat, kind, struggle=label == 'move struggle')
                    view = build_policy_opponent_view(public_lines=public, hp_visibility={'p1': 'exact', 'p2': 'exact'},
                        sampled_self_request=req, opponent_slot=seat, battle_id=str(row['id']), battle_seed=0, format_id=FORMAT,
                        set_source=tables.source, spec=model['spec'], feature_masks=model['masks'])
                    obs = view.observation(category_vocab=model['vocab'], dex=tables.dex)
                    body = choice(req, label) if censor == 'none' and label else None
                    action = action_index_from_choice_string(view.state, body) if body else None
                    if censor == 'none' and action is None:censor = 'label_undecodable'
                except (PolicyOpponentViewError, ValueError, KeyError, IndexError, TypeError) as error:
                    counts['view_error:' + type(error).__name__] += 1;counts.setdefault('view_error_example', f'{type(error).__name__}: {str(error)[:200]}')
                    censor, action = 'view_error', None
                if obs is None:continue
                rows.append(dict(obs=obs, label=-1 if censor != 'none' else int(action), forced=kind == 'switch', locked=bool(locked), turn=turn, kind=KINDS.index(kind),
                                 seat=int(seat[1]) - 1, replay=row_index, vintage=VINTAGES.index(vintage), rating=float(row['rating'] or 0),
                                 imputation=imp, censor=CENSORS.index(censor)))
                counts[f'{vintage}:{kind}:{censor}'] += 1
    return rows


def load_model(checkpoint, showdown_root):
    from pokezero.neural_policy import (category_vocab_from_model_config, feature_masks_from_model_config, load_transformer_checkpoint,
                                        observation_spec_from_model_config)
    _, result = load_transformer_checkpoint(checkpoint, map_location='cpu')
    return dict(spec=observation_spec_from_model_config(result.model_config), masks=feature_masks_from_model_config(result.model_config),
                vocab=category_vocab_from_model_config(result.model_config, showdown_root))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('input');p.add_argument('out')
    p.add_argument('--showdown-root', required=True);p.add_argument('--node-binary', default='node');p.add_argument('--model-checkpoint', required=True)
    p.add_argument('--shard', type=int, default=0);p.add_argument('--shards', type=int, default=1);p.add_argument('--limit', type=int, default=0)
    p.add_argument('--imputations', type=int, default=1)
    args = p.parse_args(argv)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True);started = time.time()
    with open(args.input) as f:table = [json.loads(line) for line in f if line.strip()]
    tables = Tables(args.showdown_root, args.node_binary);model = load_model(args.model_checkpoint, args.showdown_root)
    counts = collections.Counter();columns = collections.defaultdict(list);done = 0
    for position, row in enumerate(table):
        if position % args.shards != args.shard or row['vintage'] not in VINTAGES:continue
        if args.limit and done >= args.limit:break
        try:rows = replay_rows(tables, model, int(row['index']), row, row['vintage'], args.imputations, counts)
        except Exception as error:counts['replay_error:' + type(error).__name__] += 1;continue
        done += 1;counts['replays'] += 1
        for r in rows:
            o = r.pop('obs')
            columns['categorical_ids'].append(np.asarray(o.categorical_ids, np.int32));columns['numeric_features'].append(np.asarray(o.numeric_features, np.float32))
            columns['token_type_ids'].append(np.asarray(o.token_type_ids, np.int16));columns['attention_mask'].append(np.asarray(o.attention_mask, bool))
            columns['legal'].append(np.asarray(o.legal_action_mask, bool))
            for k, v in r.items():columns[k].append(v)
        if done % 50 == 0:print(json.dumps(dict(event='PROGRESS', shard=args.shard, replays=done, rows=len(columns['label']), seconds=round(time.time() - started, 1))), flush=True)
    stacked = ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask', 'legal')
    arrays = {k: np.stack(v) if k in stacked else np.asarray(v) for k, v in columns.items()}
    if arrays:np.savez(out / f'shard-{args.shard}.npz', **arrays)
    summary = dict(schema='v2s-g3rb-views.v1', shard=args.shard, shards=args.shards, rows=len(columns['label']), counts=dict(counts),
                   vintages=list(VINTAGES), kinds=list(KINDS), censors=list(CENSORS), seconds=round(time.time() - started, 1))
    (out / f'shard-{args.shard}.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(dict(event='SHARD_DONE', shard=args.shard, replays=done, rows=summary['rows'], seconds=summary['seconds'])), flush=True)
    return 0


if __name__ == '__main__':raise SystemExit(main())
