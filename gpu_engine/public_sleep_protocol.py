"""Raw sleep facts for a canonical producer's already-resolved encounter.

These are lexical facts, not host-computed counters or ability deductions.
Arrival/faint, team cure, TURN and exact UPKEEP use shared stream events.
The producer emits these facts after the mon field update in the same chunk
transaction; a public line is not committed until all its consumers accept.
"""
from pokezero.showdown import _ident_has_position

from . import public_sleep as sleep
from .public_mon_protocol import condition_words, STATUS


def facts(line):
    """Return (operation, first, second) facts; identity stays with the registry."""
    if not isinstance(line,str) or not line.startswith('|'):
        raise ValueError('sleep protocol requires a raw public line')
    parts=line.split('|');kind=parts[1] if len(parts)>1 else ''
    if kind=='-status' and len(parts)>=4:
        if parts[3] not in STATUS:raise ValueError('sleep status requires a recognized public status')
        status=STATUS[parts[3]]
        return ((sleep.STATUS,status,int(status==sleep.SLEEP and 'move: Rest' in line)),)
    if kind=='-curestatus' and len(parts)>=4:
        if parts[3] not in STATUS:raise ValueError('sleep cure requires a recognized public status')
        return ((sleep.CURE,int(not _ident_has_position(parts[2])),0),)
    if kind in ('-damage','-heal','-sethp') and len(parts)>=4:
        _,_,status,_=condition_words(parts[3])
        condition=parts[3].strip() if kind=='-sethp' else parts[3]
        flags=int(condition=='0 fnt')|int(condition.split()==['0','fnt'])*2
        return ((sleep.CONDITION,status,flags),)
    return sleep.lexical_facts(line)
