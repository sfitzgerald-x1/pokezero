"""Raw public wire decoding for the named move/ability/Transform consumers.

This emits typed lexical facts, not observation vectors. Identity resolution
uses the already registered public ledger. Every projection retains its raw
line and explicitly lacks complete raw-line consumer coverage; it cannot admit
a production observation or silently filter an entire replay.
"""
from dataclasses import dataclass, field
from typing import Mapping

from pokezero.belief import _confirmed_ability_from_event, _trace_copy_source, _species_from_ident
from pokezero.showdown import _public_event_from_line, _slot_from_ident

from .public_inputs import PublicEvent, Kind, Phase
from .public_moves import lexical_move_tags


@dataclass(frozen=True)
class PublicAddress:
    side: int
    slot: int
    encounter: int

    def __post_init__(self):
        if (any(type(x) is not int for x in (self.side,self.slot,self.encounter))
                or self.side not in (0,1) or not 0<=self.slot<6 or not 0<self.encounter<2**31):
            raise ValueError("resolved public address requires exact side/slot/encounter integers")


@dataclass(frozen=True)
class PublicAddresses:
    active: Mapping[int, PublicAddress]
    known_idents: Mapping[str, PublicAddress]
    public_species: Mapping[PublicAddress, str] = field(default_factory=dict)

    def resolve(self, ident):
        """Match belief._target_belief: public active first, then known ident.

        A missing registry entry is a refusal; no private order or species-only
        join is invented to manufacture a party slot.
        """
        seat=_slot_from_ident(ident or "")
        if seat not in ("p1","p2"):raise ValueError("unresolved public protocol side")
        side=int(seat[1])-1
        address=self.active.get(side) or self.known_idents.get(ident)
        if not isinstance(address,PublicAddress) or address.side!=side:
            raise ValueError("protocol identity requires a registered public encounter")
        return address

    def transform_target(self,ident):
        seat=_slot_from_ident(ident or "")
        if seat not in ("p1","p2") or not isinstance(self.active.get(int(seat[1])-1),PublicAddress):
            raise ValueError("Transform requires a publicly active donor")
        address=self.resolve(ident)
        tracked=self.public_species.get(address)
        named=_species_from_ident(ident)
        # Transform's source resolver is deliberately narrower than ordinary
        # active-first attribution. This component can copy only the tracked
        # donor's public species ID, so differing named species must refuse.
        if not tracked or not named or not (tracked.strip().lower()==named.strip().lower()
                or tracked.strip().lower().startswith(named.strip().lower()+"-")):
            raise ValueError("Transform requires the source's exact species or hyphen-forme binding")
        return address


@dataclass(frozen=True)
class ContextProjection:
    raw_line: str
    events: tuple[PublicEvent,...]
    full_line_consumers_complete: bool = False


def project_line(line, *, sequence, turn, phase=Phase.ACTION, addresses, moves, abilities):
    """Decode one supported line without discarding tags or raw provenance.

    A single raw line can yield both original-ability and parser-Trace facts;
    their source lexical discriminators intentionally differ. All unhandled
    lines fail explicitly. Other consumers of a handled line (HP, status,
    choice/item/sleep/clocks/tendencies, etc.) remain the outer adapter's duty.
    """
    if not isinstance(line,str) or not line.startswith("|"):raise ValueError("public protocol line required")
    if type(sequence) is not int or type(turn) is not int:
        raise ValueError("protocol sequence and turn require exact integers")
    event=_public_event_from_line(line)
    kind=event.event_type;parts=line.split("|");events=[]
    def emit(kind,payload,actor=None,target=None):
        value=PublicEvent(kind,sequence+len(events),turn,phase,
            actor_side=actor.side if actor else -1,actor_slot=actor.slot if actor else -1,
            target_side=target.side if target else -1,target_slot=target.slot if target else -1,
            payload=tuple(payload)+(0,)*(8-len(payload)))
        value.words();events.append(value)
    if kind=="move" and event.primary and event.actor_slot:
        actor=addresses.resolve(event.actor_ident)
        if len(parts)>4 and parts[4].strip() and not event.target_slot:
            raise ValueError("nonempty move target has an unresolved public side")
        target=addresses.resolve(event.target_ident) if event.target_slot else None
        caller,has_from=lexical_move_tags(line)
        emit(Kind.MOVE,[actor.encounter,moves.move_id(event.primary),caller,int(has_from)],actor,target)
    elif kind=="-transform" and event.actor_slot and event.primary:
        actor=addresses.resolve(event.actor_ident);target=addresses.transform_target(event.primary)
        emit(Kind.TRANSFORM,[actor.encounter,target.encounter],actor,target)
    elif kind in ("-ability","ability") and event.primary and event.target_slot:
        actor=addresses.resolve(event.target_ident);ability=abilities.ability_id(event.primary)
        is_trace,ident=_trace_copy_source(line)
        parser_feature=kind=="-ability" and "ability: Trace" in line and bool(parts[3].strip())
        if is_trace:
            # Invalid/missing [of] side is no belief copy, but can still update
            # the independently keyed parser Trace feature on a -ability line.
            donor=addresses.resolve(ident) if _slot_from_ident(ident or "") in ("p1","p2") else None
            emit(Kind.TRACE,[actor.encounter,ability,donor.encounter if donor else 0,int(parser_feature)],actor,donor)
        else:
            emit(Kind.ABILITY,[actor.encounter,ability,1],target=actor)
            if parser_feature:emit(Kind.TRACE,[actor.encounter,ability,0,1],actor)
    else:
        ident,name=_confirmed_ability_from_event(event)
        if not ident or not name:raise ValueError("line has no implemented public context consumer")
        holder=addresses.resolve(ident)
        emit(Kind.ABILITY,[holder.encounter,abilities.ability_id(name),0],target=holder)
    return ContextProjection(line,tuple(events))
