"""Canonical annotation/setup main payloads for the current V4 consumer.

These never erase raw events: switch boundaries, Trick closure, raw procs and
all registered bookkeeping must still run. Crit/effectiveness/hit count and accuracy misses remain
meaningful transition-history facts outside this V4 mask contract. Per-mon
Tier-2 pins are retired at V4 and investment belief narrowing is not admitted.
This module does not qualify transition tokens or a complete player wire.
"""
import re


# Pinned Showdown f76228a sim/battle.ts:1842-1848 emits this at turn500.
# It is a public warning, not a terminal, state assignment or generic error.
TURN_500_WARNING = ("|bigerror|You will auto-tie if the battle doesn't end in "
                    "500 turns (on turn 1000).")

# Pinned Gen3 Custom Game with the single Sleep Clause Mod override emits
# these literal setup announcements before installing the initial active slots.
GEN3_CUSTOM_SETUP = (
    '|tier|[Gen 3] Custom Game',
    '|raw|<div class="infobox"><details class="readmore" open><summary><strong>'
    '1 custom rule:</strong></summary> sleepclausemod</details></div>',
)


def attract_idents(line):
    """Closed singles before-move announcement, including its explicit source."""
    if not isinstance(line,str) or '\n' in line or '\r' in line or line.strip()!=line:
        return None
    parts=line.split('|')
    if (len(parts)!=5 or parts[:2]!=['','-activate'] or parts[3]!='move: Attract'
            or not parts[4].startswith('[of] ') or '[from]' in line):return None
    target,source=parts[2],parts[4][5:]
    if (not re.fullmatch(r'p[12]a: \S(?:[^|]*\S)?',target)
            or not re.fullmatch(r'p[12]a: \S(?:[^|]*\S)?',source)
            or target[1]==source[1]):return None
    return target,source


def classify(line):
    """Return a supported main-payload category, or no admission authority."""
    if not isinstance(line,str) or '\n' in line or '\r' in line or line.strip()!=line:
        return None
    parts=line.split('|')
    if len(parts)<2 or parts[0]:return None
    kind=parts[1]
    if line==TURN_500_WARNING:return 'text'
    # Showdown f76228a1 emits these exact diagnostics in battle-actions.ts:1331
    # and abilities.ts:4185. Keep the raw line and its shared consumers, with
    # no failure, ability reveal or secondary-effect event inferred from it.
    if line in ('|debug|move failed because it did nothing', '|debug|Shield Dust prevent secondary'):
        return 'text'
    # Pinned f76228a1 pokemon.ts:1714,2257, battle.ts:2114, and the Gen 4
    # Serene Grace handler inherited by Gen 3 (abilities.ts:377). These carry
    # no literal recipient, status, HP delta, or ability reveal.
    if line in ('|debug|natural status immunity', '|debug|weather immunity',
                '|debug|immune to status', '|debug|doubling secondary chance'):
        return 'text'
    # Frozen Gen 3 owner streams report the Protect roll and sunlight scaling.
    if line in ('|debug|Success chance: 50%', '|debug|Sunny Day water suppress'):
        return 'text'
    if attract_idents(line) is not None:return 'attract'
    # Incidental proc tags retain their existing consumers; text cannot
    # manufacture a source ability/item reveal under this annotation contract.
    if any(tag in line for tag in ('[from]','[of]')):return None
    ident=len(parts)>2 and re.fullmatch(r'p[12]a: \S(?:[^|]*\S)?',parts[2])
    if kind=='-activate' and len(parts)==4 and parts[3]=='move: Destiny Bond' and ident:
        # The pinned emitter announces this AFTER the user's |faint| and
        # BEFORE the attacker's separate |faint|. It adds no V4 fact itself;
        # the wire must prove the exact current, already-fainted encounter.
        return 'fainted_target'
    if (kind, parts[3] if len(parts)>3 else '') in (
            ('-activate','move: Heal Bell'), ('-activate','move: Struggle'), ('-singleturn','move: Focus Punch')):
        # These observed Gen3 announcements have no additional V4 feature
        # writer. Keep the exact active target and every shared raw consumer;
        # healing statuses and actual move use still require their own events.
        return 'target' if len(parts)==4 and ident else None
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
    # These are exact diagnostics from the pinned Native damage calculation.
    # The Gen 3 Reversal callback emits BP: 20 on the frozen root 29 prefix.
    if line in ('|debug|BP: 20', '|debug|BP: 70', '|debug|Negating (sp)def boost/penalty.'):
        return 'text'
    # These pinned diagnostics describe a modifier or refused status attempt.
    if line in ('|debug|rain water boost', '|debug|set status [tox] interrupted'):
        return 'text'
    if kind in ('-hint','-message'):
        return 'text' if len(parts)==3 and parts[2].strip() else None
    # The frozen owner streams include the Focus Punch callback and Reversal power.
    if line in ('|debug|before turn callback: focuspunch', '|debug|BP: 40'):
        return 'text'
    if line=='|-fieldactivate|move: Perish Song':return 'perish_announcement'
    # The pinned Native damage and sleep handlers emit these literal diagnostics.
    if line in ('|debug|damage event failed', '|debug|set status [slp] interrupted'):
        return 'text'
    if line in GEN3_CUSTOM_SETUP:return 'setup'
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
