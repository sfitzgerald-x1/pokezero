"""Canonical annotation/setup main payloads for the current V4 consumer.

These never erase raw events: switch boundaries, Trick closure, raw procs and
all registered bookkeeping must still run. Crit/effectiveness/hit count and accuracy misses remain
meaningful transition-history facts outside this V4 mask contract. Per-mon
Tier-2 pins are retired at V4 and investment belief narrowing is not admitted.
This module does not qualify transition tokens or a complete player wire.
"""
import re


def classify(line):
    """Return a supported main-payload category, or no admission authority."""
    if not isinstance(line,str) or '\n' in line or '\r' in line or line.strip()!=line:
        return None
    parts=line.split('|')
    if len(parts)<2 or parts[0]:return None
    kind=parts[1]
    # Incidental proc tags retain their existing consumers; text cannot
    # manufacture a source ability/item reveal under this annotation contract.
    if any(tag in line for tag in ('[from]','[of]')):return None
    ident=len(parts)>2 and re.fullmatch(r'p[12]a: \S(?:[^|]*\S)?',parts[2])
    if kind=='-miss':
        # The pinned Gen3 emitter names both the move user and its target.
        # Both identities must remain explicit; raw bookkeeping still runs.
        target=len(parts)==4 and re.fullmatch(r'p[12]a: \S(?:[^|]*\S)?',parts[3])
        return 'miss' if len(parts)==4 and ident and target else None
    if kind in ('-crit','-supereffective','-resisted'):
        return 'target' if len(parts)==3 and ident else None
    if kind=='-hitcount':
        # Deliberate Gen3 admission restriction; the source parser's integer
        # decoder is broader. Unsupported counts refuse rather than coerce.
        return 'target' if len(parts)==4 and ident and parts[3] in ('1','2','3','4','5') else None
    if kind in ('-hint','-message'):
        return 'text' if len(parts)==3 and parts[2].strip() else None
    if line=='|-fieldactivate|move: Perish Song':return 'perish_announcement'
    if kind=='start':return 'setup' if line=='|start' else None
    if kind=='gametype':return 'setup' if line=='|gametype|singles' else None
    if kind=='gen':return 'setup' if line=='|gen|3' else None
    if kind=='tier':
        return 'setup' if len(parts)==3 and re.sub('[^a-z0-9]','',parts[2].lower())=='gen3randombattle' else None
    if kind=='rule':return 'setup' if len(parts)==3 and parts[2].strip() else None
    if kind=='teamsize':
        return 'setup' if len(parts)==4 and parts[2] in ('p1','p2') and parts[3] in tuple(str(i) for i in range(7)) else None
    if kind=='player':
        return 'player' if 4<=len(parts)<=6 and parts[2] in ('p1','p2') and parts[3].strip() else None
    return None
