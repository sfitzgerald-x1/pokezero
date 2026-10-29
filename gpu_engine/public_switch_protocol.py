"""Lexical pending-switch facts from one public line and its prior registry.

No request, posterior, queue, callback count or observation is consulted. The
registry must be the pre-line registry: an incoming arrival cannot replace the
identity against which an earlier pending switch is resolved.
"""
from dataclasses import dataclass
import numpy as np

from pokezero.belief import _confirmed_ability_from_event
from pokezero.showdown import _public_event_from_line,_slot_from_ident,_species_from_ident,_normalize_identifier
from .public_switch_evidence import lexical_control


@dataclass(frozen=True)
class Facts:
    arrival: bool
    boundary: bool
    confirmed_ability: int
    confirmed_side: int
    confirmed_slot: int
    confirmed_encounter: int
    exact_ident: bool
    species_ident: int


def lexical_facts(line,registry,*,species_catalog,abilities):
    if not isinstance(line,str) or not line.startswith('|'):raise ValueError('public raw line required')
    if line.split('|')[1]=='request':raise ValueError('private request cannot enter switch prelude')
    if not np.array_equal(np.asarray(species_catalog.device.ability_binding),np.asarray(abilities.device.binding)):
        raise ValueError('switch lexical facts require the same species/ability namespace')
    arrival,boundary=lexical_control(line)
    ident,name=_confirmed_ability_from_event(_public_event_from_line(line))
    ability=abilities.ability_id(name) if name else 0
    side={'p1':0,'p2':1}.get(_slot_from_ident(ident or ''),-1)
    slot=-1;encounter=0;exact=False
    if ident:
        key=_normalize_identifier(ident)
        matches=[entry.address for rows in registry.encounters for entry in rows if _normalize_identifier(entry.ident)==key]
        if len(matches)>1:raise ValueError('normalized full public ident is ambiguous')
        if matches:
            address=matches[0];side,slot,encounter=address.side,address.slot,address.encounter;exact=True
    # Direct full-name lookup deliberately avoids canonical randbat species
    # and cosmetic/base-forme aliases. The source compares the raw spelling.
    raw=_normalize_identifier(_species_from_ident(ident or '') or '')
    names=[i for i,name in enumerate(species_catalog.names) if i and _normalize_identifier(name)==raw]
    if len(names)>1:raise ValueError('normalized full public species is ambiguous')
    species=names[0] if names else 0
    return Facts(arrival,boundary,ability,side,slot,encounter,exact,species)
