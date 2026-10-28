"""Raw public arrival/condition/boost facts for the named mon field consumers.

The public registry must already resolve encounter identities and frozen dex IDs.
This does not assign opponent private party order or encode observations. Other
consumers of each raw line remain required; every projection retains that caveat.
"""
import re

from pokezero.dex import normalize_id
from pokezero.determinization import _gender_from_details
from pokezero.showdown import (_species_from_details, _level_from_details, _slot_from_ident,
                              _ident_has_position, _line_mentions_baton_pass, _canonical_replacement_marker,
                              _canonical_turn_number, _canonical_faint_marker, _normalize_name, _species_from_ident)

from . import public_mon_state as mon
from .public_inputs import PublicEvent, Kind, Phase, protocol_boundary
from .public_context_protocol import PublicAddress, ContextProjection


STATUS={"slp":mon.SLEEP,"par":mon.PARALYSIS,"brn":mon.BURN,"frz":mon.FREEZE,
        "psn":mon.POISON,"tox":mon.TOXIC}


def condition_words(condition):
    """Decode integer wire HP/status facts; device computes any HP fraction."""
    if not isinstance(condition,str):raise ValueError("public condition must be a wire string")
    fields=condition.split()
    if fields==["0","fnt"]:return 0,0,mon.NONE,1
    if not fields or not re.fullmatch(r"[0-9]+/[0-9]+",fields[0]) or len(fields)>2:
        raise ValueError("integer public HP fraction and optional status required")
    hp,maximum=map(int,fields[0].split("/"))
    status=fields[1] if len(fields)>1 else ""
    if status and status not in STATUS and status!="fnt":raise ValueError("unrecognized public condition status")
    if maximum<=0 or not 0<=hp<=maximum or status=="fnt" and hp!=0:
        raise ValueError("invalid public condition HP bounds")
    return hp,maximum,STATUS.get(status,mon.NONE),int(status=="fnt")


def _exact_active(addresses,active_idents,ident):
    side=_slot_from_ident(ident)
    value=addresses.known_idents.get(ident)
    if (side not in ("p1","p2") or not isinstance(value,PublicAddress)
            or value.side!=int(side[1])-1 or addresses.active.get(value.side)!=value
            or active_idents.get(value.side)!=ident):
        raise ValueError("condition requires the exact registered public active ident")
    return value


def project_line(line,*,sequence,turn,addresses,species_ids,source,active_idents,phase=Phase.ACTION,baton_pass=False,damage_metadata=False):
    """Decode a named mon field event or refuse it; never filter a full replay.

    Species IDs come from a frozen public dex namespace, separate from source
    predicate IDs. Baton Pass's pending successful-switch fact is supplied by
    its public sequence consumer or an explicit wire tag, never inferred from
    a sampled battle state. Benched cure aliases need a prior public registry
    entry; unresolved aliases refuse instead of erasing the active mon's status.
    """
    if not isinstance(line,str) or not line.startswith("|"):raise ValueError("public protocol line required")
    if type(sequence) is not int or type(turn) is not int or type(baton_pass) is not bool or type(damage_metadata) is not bool:
        raise ValueError("exact sequence/turn integers and public Baton Pass flag required")
    parts=line.split("|");kind=parts[1] if len(parts)>1 else ""
    if line=="|-clearallboost":
        event=PublicEvent(Kind.CLEAR_ALL_BOOSTS,sequence,turn,phase)
        event.words();return ContextProjection(line,(event,))
    if kind=="-cureteam" and len(parts)==3:
        side=_slot_from_ident(parts[2])
        if side not in ("p1","p2"):raise ValueError("team cure requires a public side")
        event=PublicEvent(Kind.CURE_TEAM,sequence,turn,phase,target_side=int(side[1])-1)
        event.words();return ContextProjection(line,(event,))
    if line in ("|","|upkeep") or kind=="turn":
        if kind=="turn" and _canonical_turn_number(line) is None:
            raise ValueError("canonical positive public turn marker required")
        event=protocol_boundary(line,sequence=sequence,turn=turn,phase=phase)
        event.words();return ContextProjection(line,(event,))
    def emit(kind,data,target,actor=None):
        event=PublicEvent(kind,sequence,turn,phase,actor_side=actor.side if actor else -1,
            actor_slot=actor.slot if actor else -1,target_side=target.side,target_slot=target.slot,
            payload=tuple(data)+(0,)*(8-len(data)))
        event.words();return ContextProjection(line,(event,))
    if kind in ("switch","drag","replace") and len(parts)>=5:
        if not _canonical_replacement_marker(line,kind,line.strip().split('|')):
            raise ValueError("noncanonical public arrival suffix or active-slot payload")
        target=_exact_active(addresses,active_idents,parts[2]);species=_species_from_details(parts[3]);key=normalize_id(species)
        species_id=species_ids.get(key)
        if type(species_id) is not int or species_id<=0:raise ValueError("arrival species outside frozen public dex namespace")
        hp,maximum,status,fainted=condition_words(parts[4])
        if fainted:raise ValueError("fainted public arrival is outside the admitted arrival contract")
        level=_level_from_details(parts[3]);gender={"M":1,"F":2}.get(_gender_from_details(parts[3]),0)
        if level is None:raise ValueError("arrival details must include public species/level information")
        event_kind=Kind.BATON_PASS if kind=="switch" and (baton_pass or _line_mentions_baton_pass(parts)) else {
            "switch":Kind.SWITCH,"drag":Kind.DRAG,"replace":Kind.REPLACE}[kind]
        return emit(event_kind,[target.encounter,species_id,source.species_id(species),level,gender,hp,maximum,status],target)
    if kind in ("-damage","-heal","-sethp") and len(parts)>=4:
        if kind=="-sethp" and len(parts)>4 and _slot_from_ident(parts[4]):
            raise ValueError("paired sethp requires its complete source consumer")
        target=_exact_active(addresses,active_idents,parts[2]);hp,maximum,status,fainted=condition_words(parts[3])
        metadata=(0,0,0)
        if damage_metadata:
            from .public_damage import condition_tags
            metadata=condition_tags(line)
        return emit(Kind.CONDITION,[target.encounter,hp,maximum,status,fainted,*metadata],target)
    if kind in ("-status","-curestatus") and len(parts)>=4:
        if kind=="-curestatus" and not _ident_has_position(parts[2]):
            target=addresses.known_idents.get(parts[2]);side=_slot_from_ident(parts[2])
            if (not isinstance(target,PublicAddress) or side!=f"p{target.side+1}"
                    or addresses.active.get(target.side)==target):
                raise ValueError("benched cure requires its registered inactive public encounter")
            named=_normalize_name(_species_from_ident(parts[2]))
            matches={a for a,s in addresses.public_species.items() if a.side==target.side and
                (_normalize_name(s)==named or _normalize_name(str(s).split('-',1)[0])==named)}
            if not named or matches!={target}:
                raise ValueError("benched cure requires the source's unique public species/base-forme match")
        else:target=_exact_active(addresses,active_idents,parts[2])
        if parts[3] not in STATUS:raise ValueError("unrecognized public status event")
        return emit(Kind.STATUS,[target.encounter,STATUS[parts[3]] if kind=="-status" else mon.NONE],target)
    if kind=="faint" and len(parts)>=3:
        if not _canonical_faint_marker(line,line.strip().split('|')):raise ValueError("canonical public faint marker required")
        target=_exact_active(addresses,active_idents,parts[2]);return emit(Kind.FAINT,[target.encounter],target)
    if kind in ("-copyboost","-transform") and len(parts)>=4:
        actor=addresses.resolve(parts[2]);target=addresses.transform_target(parts[3]) if kind=="-transform" else addresses.resolve(parts[3])
        return emit(Kind.COPY_BOOST,[actor.encounter,target.encounter],target,actor)
    clears={"-clearboost":mon.CLEAR,"-restoreboost":mon.CLEAR,
            "-clearpositiveboost":mon.CLEAR_POSITIVE,"-clearnegativeboost":mon.CLEAR_NEGATIVE}
    if kind in clears and len(parts)>=3:
        target=addresses.resolve(parts[2]);return emit(Kind.BOOST,[target.encounter,0,0,clears[kind]],target)
    if kind in ("-boost","-unboost","-setboost") and len(parts)>=5:
        target=addresses.resolve(parts[2]);stat=parts[3].strip()
        if stat not in mon.STATS or not re.fullmatch(r"[+-]?[0-9]+",parts[4]):raise ValueError("known public boost stat and integer stage required")
        amount=int(parts[4])*(-1 if kind=="-unboost" else 1)
        return emit(Kind.BOOST,[target.encounter,mon.STATS.index(stat),amount,mon.SET if kind=="-setboost" else mon.DELTA],target)
    raise ValueError("line has no implemented public mon field consumer")
