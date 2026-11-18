"""Public identities for a partial trap suspended at a replacement boundary."""
import numpy as np

from pokezero.dex import normalize_id
from .phase_d_constructor_rules import _identity


def inactive_sources(physical, wrapped, facts, *, lines, request, detached, metadata, assignment, owner, replacement):
    """Join a departed, living source to its literal party slot before the pending residual.

    Showdown f76228a1, data/mods/gen5/conditions.ts:8-23 (inherited by Gen3), retains the source until its residual:
    an inactive, fainted or newly returned source (activeTurns == 0) then releases the trap without damage. The engine
    already has that lifecycle. An arbitrary current active or a negative sentinel would lose the re-entry identity.
    Ordinary boundaries and an intervening residual retain the existing refusal; no hidden clock is reconstructed.
    """
    from .phase_d_protocol_world import PARTIAL_TRAP_MOVES,wrap_support
    missing=[side for side in wrapped if facts[side] is not None and not facts[side]['trapper_active']
        and wrap_support(facts[side]['k']) and facts[side]['k']==facts[side]['upkeeps']]
    if not missing:return {}
    if replacement is None or not replacement['pending_residual']:raise ValueError('PUBLIC_WRAP_TRAPPER_NOT_ACTIVE')
    rows=[str(line).split('|') for line in lines if not str(line).startswith('|request|')]
    vocab=metadata['vocabulary']['species'];result={}
    def species(index):return normalize_id(vocab[int(index)])
    def refuse():raise ValueError('PUBLIC_WRAP_SOURCE_IDENTITY_UNAVAILABLE')
    for side in missing:
        source=1-side;seat=f'p{source+1}'
        starts=[i for i,p in enumerate(rows) if len(p)>3 and p[1]=='-activate' and p[2].startswith(f'p{side+1}a:')
            and p[3].startswith('move: ') and normalize_id(p[3][6:]) in PARTIAL_TRAP_MOVES]
        if not starts:refuse()
        start=starts[-1];of=[p[5:] for p in rows[start][4:] if p.startswith('[of] ')]
        if len(of)!=1 or not of[0].startswith(seat+'a: '):refuse()
        arrivals=[(i,p) for i,p in enumerate(rows) if len(p)>4 and p[1] in ('switch','drag','replace') and p[2].startswith(seat+'a:')]
        before=[(i,p) for i,p in arrivals if i<start]
        if not before or before[-1][1][2]!=of[0]:refuse()
        ident=_identity(of[0]);name=normalize_id(before[-1][1][3].split(',')[0])
        if not ident or not name:refuse()
        # Nicknames are identities, never species. Conflicting or duplicate public joins are unavailable.
        if any((_identity(p[2])==ident and normalize_id(p[3].split(',')[0])!=name) or
               (normalize_id(p[3].split(',')[0])==name and _identity(p[2])!=ident) for _,p in arrivals):refuse()
        departures=[i for i,p in arrivals if i>start]
        if not departures or any(p[1:2]==['upkeep'] for p in rows[departures[0]+1:]):
            raise ValueError('PUBLIC_WRAP_TRAPPER_NOT_ACTIVE')
        if source==owner:
            if request['side']['id']!=seat:refuse()
            party=request['side']['pokemon']
            matches=[k for k,p in enumerate(party) if p.get('ident','').startswith(seat+': ') and _identity(p['ident'])==ident]
            if len(party)!=6 or len(matches)!=1:refuse()
            slot=matches[0]
            if normalize_id(party[slot]['details'].split(',')[0])!=name:refuse()
        else:
            view=detached.view;present=np.asarray(view['present'])[0];shown=np.asarray(view['species'])[0]
            matches=[k for k in range(6) if present[k] and species(shown[k])==name]
            if len(matches)!=1:refuse()
            public=matches[0];slot=assignment.get(public)
            if slot is None or list(assignment.values()).count(slot)!=1:refuse()
        physical_matches=[k for k in range(6) if species(physical.species[0,source,k])==name]
        if physical_matches!=[slot] or int(physical.hp[0,source,slot])<=0:refuse()
        result[side]=slot
    return result
