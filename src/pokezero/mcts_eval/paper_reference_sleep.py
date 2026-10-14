"""Gen 3 induced sleep: condition the uniform 2..5 timer on public survival.

Never accepts a simulator snapshot or a private sleep timer. Nickname-free
identities and settled ordinary public boundaries are required.
"""
import re

from ..local_showdown import LocalShowdownError


def induced_sleep_certificates(state):
    ledgers = {}
    last_move = None
    for event in state.replay.public_events:
        parts = event.raw_line.split('|')
        if len(parts) < 2:
            continue
        kind = parts[1]
        if kind in ('upkeep', 'turn'):
            for row in ledgers.values():
                if row['pending']:
                    row['pending'] = False
                    row['skipped'] = 0
        if kind == 'move':
            last_move = parts[2:5]
        if len(parts) < 3:
            continue
        ident = parts[2]
        match = re.fullmatch(r'(p[12])(?:[a-f])?: (.+)', ident)
        if match is None:
            continue
        key = match[1] + ':' + re.sub('[^a-z0-9]', '', match[2].lower())
        if kind == '-status' and len(parts) > 3 and parts[3] == 'slp':
            if any('move: Rest' == p.replace('[from] ', '') for p in parts[4:]):
                ledgers.pop(key, None)
                continue
            if (not last_move or not last_move[0].startswith(('p2' if match[1] == 'p1' else 'p1') + 'a: ')
                    or last_move[2] != ident or not any(p == '[from] move: ' + last_move[1] for p in parts[4:])):
                raise LocalShowdownError('Induced sleep needs an explicit public opposing move source.')
            ledgers[key] = dict(attempts=0, refunded=0, skipped=0, pending=False,
                survival=[], source_player=last_move[0][:2], source_name=last_move[0].split(': ', 1)[1])
        elif kind in ('-curestatus', 'faint'):
            if kind == 'faint' or len(parts) > 3 and parts[3] == 'slp':
                ledgers.pop(key, None)
        elif key in ledgers:
            row = ledgers[key]
            if kind in ('switch', 'drag'):
                row['refunded'] += row['skipped']
                row['skipped'] = 0
            elif kind == 'cant' and len(parts) > 3:
                if row['pending'] or parts[3] != 'slp':
                    raise LocalShowdownError('Induced sleep attempt has an ambiguous public outcome.')
                row['attempts'] += 1
                row['survival'].append([row['attempts'], row['refunded']])
                row['pending'] = True
            elif kind == 'move' and row['pending']:
                if parts[3] not in ('Sleep Talk', 'Snore'):
                    raise LocalShowdownError('Induced sleep public move is not sleep-usable.')
                row['skipped'] += 1
                row['pending'] = False
    if any(r['pending'] for r in ledgers.values()):
        raise LocalShowdownError('Induced sleep requires a settled public action boundary.')
    return {k: {n: v for n, v in r.items() if n != 'pending'} for k, r in ledgers.items()}


def induced_sleep_support(certificate, ability):
    cost = 2 if re.sub('[^a-z0-9]', '', ability.lower()) == 'earlybird' else 1
    return [dict(time=start - certificate['attempts'] * cost + certificate['refunded'],
                 startTime=start, skippedTime=certificate['skipped'])
        for start in range(2, 6)
        if all(start - a * cost + refund > 0 for a, refund in certificate['survival'])
        and start - certificate['attempts'] * cost + certificate['refunded'] > 0]
