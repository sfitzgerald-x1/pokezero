"""Rebuild Foul Play's choice journal for a live screen that recorded none (screen-r1 ran opponent_journal 'addressed' and
emitted no entries), so its games can be replayed exactly through the training env (fp_reply_evidence's export).

usage: python -m gpu_engine.fp_journal_rebuild RUN_DIRECTORY OUT [--shard I --shards N] [--ordinals ORD ...]
       python -m gpu_engine.fp_journal_rebuild --report OUT

Per game (results/<ordinal>.json, telemetry/, owning-wire/ and requests/ of RUN_DIRECTORY, Foul Play's seat from the
benchmark): LocalShowdownEnv.reset(battle seed), then round by round:
- our seat plays its recorded action (telemetry's played_action_index), and its request must equal the capture's
  own_request_line for that decision (rqid and the side's username ignored, fp_reply_data._request_body);
- when Foul Play is requested, its choice is the one whose step reproduces the turn: the env's new protocol lines, filtered
  (SKIP_KINDS) and with Foul Play's conditions turned into the public percent our seat saw, must equal the capture's next
  lines filtered the same way. The choice the capture implies (Foul Play's next |move| without [from], a |cant| naming its
  move, or its |switch|) is tried first, then every other legal action, each from an env snapshot;
- without Foul Play requested, the step must reproduce the turn as it is.
A game is dropped (counted by reason, never patched) when no choice reproduces a turn, a request or a turn differs, the
capture has lines left at the end, our decisions are not all used, or the winner differs from the result's.

Each journal entry is {round, seat, choice, action, tag, reason}: tag 'chosen' when Foul Play's choice is public in that turn
(its |move| without [from], a |cant| naming its move, its |switch|, a faint replacement), 'fill' otherwise. A fill only drives
the replay and is never evidence (fp_reply_reweight skips it): reason is the |cant| cause (slp, par, flinch, frz, recharge,
truant, ...), 'fainted_before_moving', 'dragged' (Roar / Whirlwind moved it before it acted) or 'no_public_action'.
Cross-check against RR's reply proxy (scripts/depthn_rr_score.reply_for on the owner's capture) at each of our decisions in a
round where Foul Play chose: a 'chosen' entry's move id or species must equal RR's b_FP; any other pairing is counted.

Output: OUT/run/ mirrors RUN_DIRECTORY for fp_reply_data.load_game (results/<ordinal>.json with source_game.opponent_moves the
rebuilt journal and journal_rebuild metadata; telemetry, owning-wire and requests linked), OUT/games-<shard>.jsonl (per game:
status, reason, entries by tag and reason, cross-check), OUT/shard-<shard>.json; --report writes OUT/report.json.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

from gpu_engine import fp_reply_data as data_mod

SCHEMA = 'v2s-fp-journal-rebuild.v1'
SKIP_KINDS = frozenset(('t:', 'request', 'player', 'j', 'J', 'l', 'L', 'n', 'c', 'chat', 'raw', 'html', 'uhtml', 'uhtmlchange',
                        'inactive', 'inactiveoff', 'gametype', 'gen', 'tier', 'rule', 'teamsize', 'title', 'timer', 'seed', 'debug',
                        'win', 'tie', 'message', 'bigerror', 'error', 'badge', 'name', 'split', '-hint'))
SIDE = re.compile(r'^(p[1-4]): .*$')   # a side named by its player ('p1: PokeZeroBot' live, 'p1: PokeZero p1' in the env)
HP_FIELD = {'switch': 4, 'drag': 4, 'replace': 4, '-damage': 3, '-heal': 3, '-sethp': 3}


def _showdown_id(name):
    value = re.sub('[^a-z0-9]', '', str(name).lower())
    return 'hiddenpower' if value.startswith('hiddenpower') else value


def shown_percent(hp, maximum):
    """Showdown's public HP percent: ceil(100 hp / max), 99 for a damaged Pokémon that would show 100, 0 when fainted."""
    top = max(int(maximum), 1);hp = min(max(int(hp), 0), top);shown = (100 * hp + top - 1) // top
    return 99 if hp < top and shown == 100 else shown


def public_condition(text):
    """'219/250 tox' -> '88/100 tox'; '0 fnt' and unreadable conditions unchanged."""
    head, sep, rest = str(text).partition(' ')
    hp, slash, top = head.partition('/')
    if not slash or not hp.isdigit() or not top.isdigit():return text
    return f'{shown_percent(int(hp), int(top))}/100' + (sep + rest if sep else '')


def comparable(lines, fp):
    """Battle lines as our seat saw them: SKIP_KINDS (with -hint), [silent] and blank lines dropped, a side named by its player
    ('p1: PokeZeroBot') as 'p1', Foul Play's conditions at the public percent."""
    out = []
    for line in lines:
        if not line.startswith('|') or line.strip() == '|':continue
        parts = line.split('|')
        if parts[1] in SKIP_KINDS or '[silent]' in parts[3:]:continue   # silent lines (Gen 3 Pressure) reach the omniscient stream only
        if parts[1] == '-activate' and 'move: Struggle' in parts[3:]:continue   # 'has no moves left': sent to its own side only
        parts = [SIDE.sub(r'\1', x) for x in parts]   # usernames differ between the live players and the env
        index = HP_FIELD.get(parts[1])
        if index is not None and len(parts) > index and parts[2].strip()[:2] == fp:
            parts[index] = public_condition(parts[index])
        out.append('|'.join(parts))
    return out


def _who(ident):return str(ident).split(':', 1)[0].strip()[:2]


def implied_choice(lines, fp):
    """The choice string our capture's next lines show Foul Play making, or None: its first |move| without [from], a |cant|
    naming its own move, or its |switch|, before the next |turn|."""
    from scripts.depthn_rr_score import cant_named_move
    for line in lines:
        parts = line.split('|')
        if len(parts) < 3:continue
        if parts[1] == 'turn':return None
        if _who(parts[2]) != fp or len(parts) < 4:continue
        if parts[1] == 'move' and not any(p.strip().startswith('[from]') for p in parts[4:]):return 'move ' + _showdown_id(parts[3])
        if parts[1] == 'cant' and not any(p.startswith('[of]') for p in parts[3:]):
            named = cant_named_move(parts)
            if named is not None:return 'move ' + named
        if parts[1] == 'switch':return 'switch ' + _showdown_id(parts[3].split(',')[0])
    return None


def classify(lines, fp, forced):
    """(tag, reason, (kind, id) or None) of Foul Play's choice from one step's (omniscient) lines (module docstring)."""
    from scripts.depthn_rr_score import cant_named_move
    cant = None
    for line in lines:
        parts = line.split('|')
        if len(parts) < 3 or _who(parts[2]) != fp:continue
        kind = parts[1]
        if kind == 'faint' and not forced:return 'fill', cant or 'fainted_before_moving', None   # |faint|IDENT has three fields
        if len(parts) < 4:continue
        if kind == 'switch':return 'chosen', 'replacement' if forced else 'switch', ('switch', _showdown_id(parts[3].split(',')[0]))
        if forced:continue
        if kind == 'move' and not any(p.strip().startswith('[from]') for p in parts[4:]):return 'chosen', 'move', ('move', _showdown_id(parts[3]))
        if kind == 'cant' and not any(p.startswith('[of]') for p in parts[3:]):
            named = cant_named_move(parts)
            if named is not None:return 'chosen', 'cant_named', ('move', named)
            if cant is None:cant = _showdown_id(parts[3].replace('move: ', ''))
        if kind == 'drag':return 'fill', cant or 'dragged', None
    return 'fill', cant or 'no_public_action', None


def _request_choice(state, request, choice):
    """The action index of a choice string on Foul Play's own state ('move <id>' by the request's move ids, Hidden Power
    collapsed; 'switch <species>' by the side's order), or None."""
    from pokezero.teacher_capture import action_index_from_choice_string
    kind, _, arg = choice.partition(' ')
    if kind == 'move':
        moves = [_showdown_id(m.get('id') or m.get('move')) for m in ((request.get('active') or [{}])[0].get('moves') or ())]
        if arg in moves:return action_index_from_choice_string(state, f'move {moves.index(arg) + 1}')
        return None
    side = [_showdown_id(str(p.get('details') or '').split(',')[0]) for p in (request.get('side') or {}).get('pokemon') or ()]
    if arg in side:return action_index_from_choice_string(state, f'switch {side.index(arg) + 1}')
    return None


class Dropped(Exception):
    def __init__(self, reason, detail=None):super().__init__(reason);self.reason, self.detail = reason, detail


def _first_difference(expected, got):
    for i, (x, y) in enumerate(zip(expected, got)):
        if x != y:return dict(at=i, expected=expected[max(0, i - 2):i + 3], got=got[max(0, i - 2):i + 3])
    return dict(at=min(len(expected), len(got)), expected=expected[len(got):len(got) + 3], got=got[len(expected):len(expected) + 3])


def captured(run, ordinal):
    """(the capture's lines, {decision_index: own_request_line}) from owning-wire/<ordinal>.jsonl's WIRE and ROOT rows."""
    lines, roots = [], {}
    for raw in (run / 'owning-wire' / f'{ordinal}.jsonl').read_bytes().splitlines():
        if not raw.strip():continue
        row = json.loads(raw)
        if row.get('event') == 'WIRE':lines.extend(row['lines'])
        elif row.get('event') == 'ROOT':roots[row['decision_index']] = row['own_request_line']
    return lines, roots


def load(run, ordinal):
    result = json.loads((run / 'results' / f'{ordinal}.json').read_bytes())
    if result.get('status') != 'COMPLETE':return None, 'not_complete'
    game = result.get('source_game') or {};bench = result.get('benchmark') or {}
    fp = bench.get('foulplay_player');ours = {'p1': 'p2', 'p2': 'p1'}.get(fp)
    if ours is None:return None, 'unknown_seats'
    telemetry = sorted((json.loads(x) for x in (run / 'telemetry' / f'{ordinal}.jsonl').read_bytes().splitlines() if x.strip()),
                       key=lambda r: r['decision_index'])
    telemetry = [r for r in telemetry if r.get('owner') in (None, ours)]
    if [r['decision_index'] for r in telemetry] != list(range(len(telemetry))):return None, 'telemetry_gap'
    lines, roots = captured(run, ordinal)
    request = json.loads((run / 'requests' / f'{ordinal}.json').read_bytes())
    return dict(ordinal=ordinal, result=result, seed=int(game['seed']), fp=fp, ours=ours, our_actions=[r['played_action_index'] for r in telemetry],
                our_requests=[roots.get(i) for i in range(len(telemetry))], lines=lines, winner=(game.get('terminal_witness') or {}).get('winner_player'),
                base_config=request['base_config']), None


def rebuild(env, game):
    """(journal entries, counters) for one game; raises Dropped(reason)."""
    from pokezero.showdown import showdown_choice_for_action
    fp, ours = game['fp'], game['ours'];env.reset(seed=game['seed'])
    cap = comparable(game['lines'], fp);pos = 0;counts = collections.Counter()
    start = comparable(env.protocol_lines, fp)
    if cap[:len(start)] != start:raise Dropped('start_differs')
    pos = len(start);journal = [];decision = 0;round_ = 0
    while env.terminal() is None:
        requested = env.requested_players()
        if not requested:raise Dropped('no_requested_players')
        actions = {}
        if ours in requested:
            if decision >= len(game['our_actions']):raise Dropped('our_decisions_exhausted')
            line = game['our_requests'][decision]
            if line is None or data_mod._request_body(line) != data_mod._request_body(json.dumps(env._latest_requests.get(ours))):
                raise Dropped('request_differs')
            actions[ours] = int(game['our_actions'][decision]);decision += 1
        before = len(env._lines)
        if fp in requested:
            state = env._state_for_player(fp);request = env._latest_requests.get(fp) or {};forced = bool(request.get('forceSwitch'))
            legal = [a for a in range(9) if state.legal_action_mask[a]]
            implied = implied_choice(cap[pos:], fp);first = None if implied is None else _request_choice(state, request, implied)
            order = ([first] if first in legal else []) + [a for a in legal if a != first]
            if not order:raise Dropped('no_legal_action')
            snapshot = env.snapshot() if len(order) > 1 else None;matched = None;tries = []
            for tried, action in enumerate(order):
                if tried:env.restore(snapshot);before = len(env._lines)
                env.step({**actions, fp: action})
                new = comparable(env._lines[before:], fp)
                if cap[pos:pos + len(new)] == new:matched = action;break
                if action == first or not tries:tries.append(dict(action=action, **_first_difference(cap[pos:pos + len(new)], new)))
            if matched is None:raise Dropped('no_choice_reproduces_the_turn', dict(round=round_, implied=implied, first=first, legal=legal, tries=tries))
            counts['implied_first'] += matched == first;counts['retries'] += tried
            journal.append(dict(round=round_, seat=fp, choice=showdown_choice_for_action(state, matched), action=matched, forced=forced,
                                our_decision=decision - 1 if ours in requested else None, start=before))
        else:
            env.step(actions);new = comparable(env._lines[before:], fp)
            if cap[pos:pos + len(new)] != new:raise Dropped('turn_differs', dict(round=round_, **_first_difference(cap[pos:pos + len(new)], new)))
        pos += len(new);round_ += 1
        if round_ > data_mod.MAX_ROUNDS:raise Dropped('runaway_replay')
    if decision != len(game['our_actions']):raise Dropped('our_decisions_unused')
    if pos != len(cap):raise Dropped('capture_lines_left')
    if (env.terminal().winner or None) != (game['winner'] or None):raise Dropped('winner_differs')
    # A choice resolves in its own step or a later one before Foul Play's next request (our Baton Pass replacement pauses the
    # turn before Foul Play moves): classify each entry on the lines from its step to the next entry's, cut at the turn's end.
    for entry, after in zip(journal, journal[1:] + [None]):
        lines = env._lines[entry.pop('start'):None if after is None else after['start']]
        end = next((i for i, line in enumerate(lines) if line.startswith('|turn|')), len(lines))
        tag, reason, public = classify(lines[:end], fp, entry['forced'])
        entry.update(tag=tag, reason=reason, public=None if public is None else list(public))
    return journal, counts


def cross_check(run, game, journal):
    """Counter of (tag, RR's reply) pairings at our decisions, plus disagreements (a chosen entry whose id differs from b_FP)."""
    from scripts import depthn_rr_score as rr
    rr.load_capture.cache_clear();capture = rr.load_capture(str(run / 'owning-wire' / f"{game['ordinal']}.jsonl"))
    by_decision = {e['our_decision']: e for e in journal if e['our_decision'] is not None}
    pairs, disagreements = collections.Counter(), []
    for index, request_count in sorted(capture.roots.items()):
        line = game['our_requests'][index] if index < len(game['our_requests']) else None
        kind = 'switch' if line and (json.loads(line[len('|request|'):]).get('forceSwitch')) else 'move'
        reply = rr.reply_for(capture, dict(owner=game['ours'], decision_index=index, request_count=request_count, request_kind=kind))
        entry = by_decision.get(index);tag = entry['tag'] if entry else 'not_requested'
        proxy = reply.kind if reply.cause is None else 'censored:' + reply.cause
        pairs[f'{tag}|{proxy}'] += 1
        if entry and entry['tag'] == 'chosen' and reply.cause is None and tuple(entry['public']) != (reply.kind, reply.id):
            disagreements.append(dict(kind='chosen_id_differs', ordinal=game['ordinal'], decision_index=index, round=entry['round'],
                                      journal=entry['public'], rr=[reply.kind, reply.id], rr_detail=reply.detail))
        elif entry and entry['tag'] == 'fill' and reply.cause is None and reply.kind != 'wait':
            disagreements.append(dict(kind='fill_but_rr_reply', ordinal=game['ordinal'], decision_index=index, round=entry['round'],
                                      journal_reason=entry['reason'], rr=[reply.kind, reply.id], rr_detail=reply.detail))
        elif entry and entry['tag'] == 'chosen' and reply.cause is not None:
            disagreements.append(dict(kind='chosen_but_rr_censored', ordinal=game['ordinal'], decision_index=index, round=entry['round'],
                                      journal=entry['public'], rr_cause=reply.cause, rr_detail=reply.detail))
    return pairs, disagreements


def mirror(run, out, game, journal):
    """OUT/run/results/<ordinal>.json with the rebuilt journal; the other directories linked once."""
    target = out / 'run';(target / 'results').mkdir(parents=True, exist_ok=True)
    for name in ('telemetry', 'owning-wire', 'requests'):
        link = target / name
        if not link.exists():
            try:os.symlink(run.resolve() / name, link)
            except FileExistsError:pass
    result = json.loads(json.dumps(game['result']))
    result['source_game']['opponent_moves'] = [dict(round=e['round'], seat=e['seat'], choice=e['choice'], action=e['action'], tag=e['tag'],
                                                    reason=e['reason']) for e in journal]
    result['journal_rebuild'] = dict(schema=SCHEMA, source_run=str(run.resolve()), entries=len(journal),
                                     chosen=sum(e['tag'] == 'chosen' for e in journal), fill=sum(e['tag'] == 'fill' for e in journal))
    (target / 'results' / f"{game['ordinal']}.json").write_text(json.dumps(result) + '\n')


def run_shard(args):
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError
    run, out = Path(args.run_directory), Path(args.out);out.mkdir(parents=True, exist_ok=True);started = time.time()
    ordinals = sorted(int(Path(p).stem) for p in glob.glob(str(run / 'results' / '*.json')))
    if args.ordinals:ordinals = [o for o in ordinals if o in set(args.ordinals)]
    ordinals = [o for o in ordinals if o % args.shards == args.shard]
    env = None;key = None;totals = collections.Counter();rows = []
    for ordinal in ordinals:
        game, skip = load(run, ordinal)
        if game is None:rows.append(dict(ordinal=ordinal, status='skipped', reason=skip));totals[f'skipped_{skip}'] += 1;continue
        base = game['base_config'];this = (base['showdown_root'], base['node_binary'], bool(base['belief_set_source']))
        if key is None:key = this
        elif this != key:rows.append(dict(ordinal=ordinal, status='skipped', reason='other_runtime'));continue
        if env is None:env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=base['showdown_root'], node_binary=base['node_binary'],
                                                                  set_belief_source=bool(base['belief_set_source'])))
        try:journal, counts = rebuild(env, game)
        except Dropped as error:
            rows.append(dict(ordinal=ordinal, seed=game['seed'], status='dropped', reason=error.reason, detail=error.detail));totals['dropped'] += 1;continue
        except (LocalShowdownError, ValueError, KeyError) as error:
            rows.append(dict(ordinal=ordinal, seed=game['seed'], status='dropped', reason=f'env_error:{type(error).__name__}', detail=str(error)[:300]))
            totals['dropped'] += 1;env.close();env = None;continue
        pairs, disagreements = cross_check(run, game, journal);mirror(run, out, game, journal)
        tags = collections.Counter(f"{e['tag']}:{e['reason']}" for e in journal)
        rows.append(dict(ordinal=ordinal, seed=game['seed'], battle_id=(game['result'].get('source_game') or {}).get('battle_id'), status='rebuilt',
                         entries=len(journal), tags=dict(tags), counts=dict(counts), cross_check=dict(pairs), disagreements=disagreements))
        totals['rebuilt'] += 1
    if env is not None:env.close()
    (out / f'games-{args.shard}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = dict(schema=SCHEMA, run_directory=str(run), shard=args.shard, shards=args.shards, games=len(rows), **totals,
                   seconds=round(time.time() - started, 1))
    (out / f'shard-{args.shard}.json').write_text(json.dumps(summary, indent=1) + '\n');print(json.dumps(summary), flush=True)


def report(out, rr_ordinals=None):
    """Totals over every games-*.jsonl; with rr_ordinals (the RR battles), the same restricted to them and the dropped list."""
    rows = [json.loads(x) for p in sorted(glob.glob(str(Path(out) / 'games-*.jsonl'))) for x in Path(p).read_text().splitlines() if x.strip()]

    def totals(selected):
        tags, pairs, statuses, reasons = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
        disagreements = []
        for r in selected:
            statuses[r['status']] += 1
            if r['status'] != 'rebuilt':reasons[r['reason']] += 1;continue
            tags.update(r['tags']);pairs.update(r['cross_check']);disagreements += r['disagreements']
        chosen = sum(v for k, v in tags.items() if k.startswith('chosen:'));fill = sum(v for k, v in tags.items() if k.startswith('fill:'))
        return dict(games=len(selected), statuses=dict(statuses), dropped_reasons=dict(reasons), entries=dict(chosen=chosen, fill=fill),
                    by_tag_reason=dict(sorted(tags.items())), cross_check=dict(sorted(pairs.items())),
                    disagreements=dict(collections.Counter(d['kind'] for d in disagreements)), disagreement_list=disagreements)
    result = dict(schema=SCHEMA + ':report', all=totals(rows))
    if rr_ordinals is not None:
        chosen = [r for r in rows if r['ordinal'] in rr_ordinals]
        result['rr'] = dict(totals(chosen), missing=sorted(set(rr_ordinals) - {r['ordinal'] for r in chosen}),
                            dropped=[dict(ordinal=r['ordinal'], reason=r['reason']) for r in chosen if r['status'] != 'rebuilt'])
    (Path(out) / 'report.json').write_text(json.dumps(result, indent=1) + '\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('run_directory', nargs='?');p.add_argument('out')
    p.add_argument('--shard', type=int, default=0);p.add_argument('--shards', type=int, default=1)
    p.add_argument('--ordinals', type=int, nargs='+');p.add_argument('--report', action='store_true')
    p.add_argument('--rr-ordinals', help='JSON list of the RR battles\' ordinals (with --report)')
    args = p.parse_args(argv)
    if args.report:
        result = report(args.out, None if not args.rr_ordinals else set(json.loads(Path(args.rr_ordinals).read_text())))
        print(json.dumps({k: (v if k == 'schema' else {kk: vv for kk, vv in v.items() if kk != 'disagreement_list'}) for k, v in result.items()}))
        return 0
    if not args.run_directory:p.error('RUN_DIRECTORY required')
    run_shard(args);return 0


if __name__ == '__main__':sys.exit(main())
