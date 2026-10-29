"""Public encounter registration and composition of the implemented wire consumers.

Slots follow first public encounter order, never either player's private party
order. This is an incremental input producer, not full replay admission: every
projection keeps the original line and the incomplete-consumer flag. Unknown
lines and ambiguous identity changes refuse without advancing host state.
"""
from dataclasses import dataclass, replace
from types import MappingProxyType
import numpy as np

from pokezero.showdown import (_canonical_replacement_marker, _pokemon_from_public_line,
    _flag_baton_pass, _slot_from_ident, _species_from_ident, _normalize_name,
    _public_event_from_line)
from pokezero.belief import _confirmed_ability_from_event

from . import public_mon_protocol, public_context_protocol, public_items, public_field, public_active, public_damage, public_types, public_sleep, public_sleep_protocol, public_toxic_protocol
from . import public_tendency, public_tendency_protocol
from .public_context_protocol import PublicAddress, PublicAddresses, ContextProjection
from .public_inputs import Kind, Phase, PublicEvent


@dataclass(frozen=True)
class Encounter:
    ident: str
    species: str
    address: PublicAddress


@dataclass(frozen=True)
class WireState:
    encounters: tuple[tuple[Encounter, ...], tuple[Encounter, ...]] = ((), ())
    active: tuple[int, int] = (-1, -1)
    live: tuple[bool, bool] = (False, False)
    pending_baton: tuple[bool, bool] = (False, False)
    next_sequence: int = 1
    turn: int = 0
    phase: Phase = Phase.ACTION
    species_binding: tuple[int, ...] | None = None
    move_binding: tuple[int, ...] | None = None
    item_binding: tuple[int, ...] | None = None
    context_binding: tuple[int, ...] | None = None
    field_binding: tuple[int, ...] | None = None
    active_binding: tuple[int, ...] | None = None
    damage_binding: tuple[int, ...] | None = None
    type_binding: tuple[int, ...] | None = None
    sleep_binding: tuple[int, ...] | None = None
    toxic_binding: tuple[int, ...] | None = None
    tendency_binding: tuple[int, ...] | None = None

    def addresses(self):
        active = {side: rows[self.active[side]].address for side, rows in enumerate(self.encounters)
                  if self.active[side] >= 0 and self.live[side]}
        known = {entry.ident: entry.address for rows in self.encounters for entry in rows}
        species = {entry.address: entry.species for rows in self.encounters for entry in rows}
        return PublicAddresses(MappingProxyType(active), MappingProxyType(known), MappingProxyType(species))

    def active_idents(self):
        return MappingProxyType({side: rows[self.active[side]].ident for side, rows in enumerate(self.encounters)
                                 if self.active[side] >= 0})


def _arrive(state, line, parts):
    if not _canonical_replacement_marker(line, parts[1], line.strip().split('|')):
        raise ValueError("canonical singles public arrival required")
    pokemon = _pokemon_from_public_line(parts)
    if pokemon is None or not pokemon.species:
        raise ValueError("arrival requires a public species and side")
    side = int(pokemon.showdown_slot[1]) - 1
    entries = state.encounters[side]
    matches = [entry for entry in entries if entry.ident == pokemon.ident]
    if matches:
        entry = matches[0]
        if entry.species != pokemon.species:
            # A forme/replacement identity needs its own proven source consumer.
            # It must not silently inherit a different species' set evidence.
            raise ValueError("changed species on an existing public ident requires explicit replacement reconciliation")
    else:
        if len(entries) == 6:
            raise ValueError("public encounter capacity exceeded; no truncation or private-order refill")
        entry = Encounter(pokemon.ident, pokemon.species, PublicAddress(side, len(entries), len(entries) + 1))
        entries = entries + (entry,)
    encounters = tuple(entries if s == side else state.encounters[s] for s in (0, 1))
    active = tuple(entry.address.slot if s == side else state.active[s] for s in (0, 1))
    live = tuple(True if s == side else state.live[s] for s in (0, 1))
    return replace(state, encounters=encounters, active=active, live=live), side


def _bench_alias(addresses, parts):
    """Resolve only the source's unique public species/base-forme bench cure."""
    if len(parts) < 3 or parts[1] != '-curestatus':
        return addresses
    ident = parts[2]
    side = _slot_from_ident(ident)
    if side not in ('p1', 'p2') or ident.startswith(side + 'a:'):
        return addresses
    named = _normalize_name(_species_from_ident(ident))
    matches = {a for a, species in addresses.public_species.items() if a.side == int(side[1]) - 1
               and (_normalize_name(species) == named or _normalize_name(species.split('-', 1)[0]) == named)}
    if len(matches) != 1:
        raise ValueError("bench cure requires one publicly encountered species, not a private roster")
    target = next(iter(matches))
    return PublicAddresses(addresses.active, MappingProxyType(dict(addresses.known_idents, **{ident: target})),
                           addresses.public_species)


MON_KINDS = frozenset(('switch', 'drag', 'replace', '-damage', '-heal', '-sethp', '-status',
    '-curestatus', 'faint', '-copyboost', '-transform', '-clearboost', '-restoreboost',
    '-clearpositiveboost', '-clearnegativeboost', '-boost', '-unboost', '-setboost',
    '-clearallboost', '-cureteam', 'turn', 'upkeep'))
CONTEXT_KINDS = frozenset(('move', '-transform', '-ability', 'ability'))
# These named protocol families may carry a proc as their only implemented
# consumer. An arbitrary raw kind cannot become supported by adding a tag.
PROC_ONLY_KINDS = frozenset(('-activate','-immune','-fail','-block','-miss','-crit','-supereffective','-resisted'))
ACTIVE_MAIN_KINDS = frozenset(('move','cant','-damage','turn','-anim','-mustrecharge','-singleturn',
    '-fail','-item','-enditem','-ability','-activate','-end','-start','-singlemove','-prepare'))


@dataclass(frozen=True)
class WireProjection(ContextProjection):
    species_binding: tuple[int, ...] = ()
    ability_binding: tuple[int, ...] = ()
    move_binding: tuple[int, ...] = ()
    item_binding: tuple[int, ...] | None = None
    context_binding: tuple[int, ...] | None = None
    field_binding: tuple[int, ...] | None = None
    active_binding: tuple[int, ...] | None = None
    damage_binding: tuple[int, ...] | None = None
    type_binding: tuple[int, ...] | None = None
    sleep_binding: tuple[int, ...] | None = None
    toxic_binding: tuple[int, ...] | None = None
    tendency_binding: tuple[int, ...] | None = None


def project_line(state, line, *, species_catalog, source, moves, abilities, items=None, context_tables=None, field_catalog=None, active_catalog=None, damage_tables=None, type_catalog=None, sleep_tables=None, toxic_tables=None, tendency_catalog=None):
    """Return a new registry and ordered named-field events, or refuse atomically.

    Callers commit this candidate state only with successful device fold of all
    emitted events. Reusing a public ident preserves its encounter; two distinct
    idents with the same species remain distinct. Nickname changes and species
    changes are not joined by species. Unknown kinds cannot be skipped through
    this API. Other raw-line consumers remain unimplemented and visibly flagged.
    """
    if not isinstance(state, WireState) or not isinstance(line, str) or not line.startswith('|'):
        raise ValueError("public WireState and raw protocol line required")
    binding = tuple(int(x) for x in np.asarray(species_catalog.device.binding))
    move_binding = tuple(int(x) for x in np.asarray(moves.device.binding))
    if (len(binding)!=8 or state.species_binding not in (None,binding)
            or state.move_binding not in (None,move_binding)
            or not np.array_equal(np.asarray(abilities.device.binding),np.asarray(species_catalog.device.ability_binding))
            or not np.array_equal(np.asarray(moves.device.source_binding),np.asarray(source.device.binding))
            or not np.array_equal(np.asarray(species_catalog.device.source_binding),np.asarray(source.device.binding))):
        raise ValueError('public wire requires the frozen species/ability/move namespaces')
    item_binding=context_binding=field_binding=active_binding=damage_binding=type_binding=sleep_binding=toxic_binding=tendency_binding=None
    if items is not None:
        if context_tables is None or context_tables.items is None or context_tables.binding is None:
            raise ValueError("item wire requires a frozen composed context")
        for actual,expected in ((context_tables.items.binding,items.device.binding),
                (context_tables.abilities.binding,abilities.device.binding),(context_tables.moves.binding,moves.device.binding),
                (items.device.source_binding,source.device.binding)):
            if not np.array_equal(np.asarray(actual),np.asarray(expected)):
                raise ValueError("item wire context namespace mismatch")
        item_binding=tuple(int(x) for x in np.asarray(items.device.binding))
        context_binding=tuple(int(x) for x in np.asarray(context_tables.binding))
    elif context_tables is not None and context_tables.items is not None:
        raise ValueError("item-enabled context cannot drop its public item producer")
    if field_catalog is not None:
        if (context_tables is None or context_tables.field is None or items is None
                or not np.array_equal(np.asarray(context_tables.field.binding),np.asarray(field_catalog.device.binding))
                or not np.array_equal(np.asarray(field_catalog.device.vocabulary_binding),np.asarray(species_catalog.device.vocabulary_binding))):
            raise ValueError("field wire requires the same frozen field/context/vocabulary namespaces")
        field_binding=tuple(int(x) for x in np.asarray(field_catalog.device.binding))
    elif context_tables is not None and context_tables.field is not None:
        raise ValueError("field-enabled context cannot drop its public field producer")
    if active_catalog is not None:
        if (context_tables is None or context_tables.active is None or field_catalog is None
                or not np.array_equal(np.asarray(context_tables.active.binding),np.asarray(active_catalog.device.binding))
                or not np.array_equal(np.asarray(active_catalog.device.vocabulary_binding),np.asarray(species_catalog.device.vocabulary_binding))):
            raise ValueError("active wire requires the same frozen active/context/vocabulary namespaces")
        active_binding=tuple(int(x) for x in np.asarray(active_catalog.device.binding))
        if active_catalog.device.native_truant is not None:
            if (active_catalog.device.native_truant.shape!=(len(species_catalog.names),)
                    or not np.array_equal(np.asarray(active_catalog.device.public_species_binding),np.asarray(species_catalog.device.binding))):
                raise ValueError('Truant wire requires its exact compiled public species namespace')
    elif context_tables is not None and context_tables.active is not None:
        raise ValueError("active-enabled context cannot drop its public active producer")
    if damage_tables is not None:
        if (context_tables is None or context_tables.damage is None or field_catalog is None
                or not np.array_equal(np.asarray(context_tables.damage.binding),np.asarray(damage_tables.binding))
                or not np.array_equal(np.asarray(damage_tables.vocabulary_binding),np.asarray(species_catalog.device.vocabulary_binding))):
            raise ValueError("damage wire requires the same frozen damage/context/vocabulary namespaces")
        damage_binding=tuple(int(x) for x in np.asarray(damage_tables.binding))
    elif context_tables is not None and context_tables.damage is not None:
        raise ValueError("damage-enabled context cannot drop its public damage producer")
    if type_catalog is not None:
        if (context_tables is None or context_tables.types is None or field_catalog is None
                or not np.array_equal(np.asarray(context_tables.types.binding),np.asarray(type_catalog.device.binding))
                or not np.array_equal(np.asarray(type_catalog.device.vocabulary_binding),np.asarray(species_catalog.device.vocabulary_binding))
                or not np.array_equal(np.asarray(type_catalog.device.species_binding),np.asarray(species_catalog.device.binding))):
            raise ValueError('live-type wire requires matching type/context/species/vocabulary namespaces')
        type_binding=tuple(int(x) for x in np.asarray(type_catalog.device.binding))
    elif context_tables is not None and context_tables.types is not None:
        raise ValueError('live-type context cannot drop its producer')
    if sleep_tables is not None:
        if (context_tables is None or context_tables.sleep is None or field_catalog is None
                or not np.array_equal(np.asarray(context_tables.sleep.binding),np.asarray(sleep_tables.binding))
                or not np.array_equal(np.asarray(sleep_tables.vocabulary_binding),np.asarray(species_catalog.device.vocabulary_binding))
                or not np.array_equal(np.asarray(sleep_tables.species_binding),np.asarray(species_catalog.device.binding))):
            raise ValueError('sleep wire requires matching sleep/context/species/vocabulary namespaces')
        sleep_binding=tuple(int(x) for x in np.asarray(sleep_tables.binding))
    elif context_tables is not None and context_tables.sleep is not None:
        raise ValueError('sleep context cannot drop its raw producer')
    if toxic_tables is not None:
        if (context_tables is None or context_tables.toxic is None or field_catalog is None
                or not np.array_equal(np.asarray(context_tables.toxic.binding),np.asarray(toxic_tables.binding))
                or not np.array_equal(np.asarray(toxic_tables.species_binding),np.asarray(species_catalog.device.binding))
                or not np.array_equal(np.asarray(toxic_tables.vocabulary_binding),np.asarray(species_catalog.device.vocabulary_binding))):
            raise ValueError('Toxic wire requires matching Toxic/context/species/vocabulary namespaces')
        toxic_binding=tuple(int(x) for x in np.asarray(toxic_tables.binding))
    elif context_tables is not None and context_tables.toxic is not None:
        raise ValueError('Toxic context cannot drop its raw producer')
    tendency_projection=None
    if tendency_catalog is not None:
        if (context_tables is None or context_tables.tendency is None or field_catalog is None
                or not np.array_equal(np.asarray(context_tables.tendency.binding),np.asarray(tendency_catalog.device.binding))
                or not np.array_equal(np.asarray(tendency_catalog.device.species_binding),np.asarray(species_catalog.device.binding))
                or not np.array_equal(np.asarray(tendency_catalog.device.move_binding),np.asarray(moves.device.binding))
                or not np.array_equal(np.asarray(tendency_catalog.device.vocabulary_binding),np.asarray(species_catalog.device.vocabulary_binding))):
            raise ValueError('tendency wire requires exact context/species/move/vocabulary namespaces')
        tendency_binding=tuple(int(x) for x in np.asarray(tendency_catalog.device.binding))
        tendency_projection=public_tendency_protocol.project_line(line,catalog=tendency_catalog)
    elif context_tables is not None and context_tables.tendency is not None:
        raise ValueError('tendency context cannot drop its raw producer')
    if state.next_sequence>1 and (state.item_binding!=item_binding or state.context_binding!=context_binding or state.field_binding!=field_binding or state.active_binding!=active_binding or state.damage_binding!=damage_binding or state.type_binding!=type_binding or state.sleep_binding!=sleep_binding or state.toxic_binding!=toxic_binding or state.tendency_binding!=tendency_binding):
        raise ValueError("item context cannot change or be dropped across continuation")
    parts = line.split('|')
    kind = parts[1] if len(parts) > 1 else ''
    main_handled=kind in MON_KINDS or kind in CONTEXT_KINDS or line=='|'
    if kind == 'request':
        raise ValueError('private requests cannot enter the public event producer')
    candidate = replace(state,species_binding=binding,move_binding=move_binding,
        item_binding=item_binding,context_binding=context_binding,field_binding=field_binding,active_binding=active_binding,damage_binding=damage_binding,type_binding=type_binding,sleep_binding=sleep_binding,toxic_binding=toxic_binding,tendency_binding=tendency_binding)
    side = None
    if kind in ('switch', 'drag', 'replace'):
        candidate, side = _arrive(candidate, line, parts)
        # Enforce the bound catalog's lookup rather than accepting arbitrary
        # ID maps at the producer boundary.
        species_catalog.species_id(candidate.encounters[side][candidate.active[side]].species)
    addresses = _bench_alias(candidate.addresses(), parts)
    events = []
    item_projection=public_items.project_line(line,items=items) if items is not None else None
    if item_projection is not None:
        main_handled |= item_projection.item_event or bool(item_projection.trick_idents)
        kwargs={}
        family=1 if item_projection.item_event else 0
        payload=(family,0,0,0,0,0,0,0)
        if item_projection.trick_idents:
            actor,target=(addresses.resolve(x) for x in item_projection.trick_idents)
            if addresses.active.get(actor.side)!=actor or addresses.active.get(target.side)!=target:
                raise ValueError("Trick requires two live public encounters")
            kwargs=dict(actor_side=actor.side,actor_slot=actor.slot,target_side=target.side,target_slot=target.slot)
            payload=(2,actor.encounter,target.encounter,0,0,0,0,0)
        event=PublicEvent(Kind.RAW_BEGIN,state.next_sequence,state.turn,state.phase,payload=payload,**kwargs)
        event.words();events.append(event)
    # The belief engine records a raw proc BEFORE dispatching the line's main
    # consumer. A proc on a move can reveal Pressure before that move's PP
    # charge; a proc on Transform can establish the donor's current ability.
    # Direct/Trace declarations have their own authoritative attribution path.
    public_event = _public_event_from_line(line)
    confirmed = _confirmed_ability_from_event(public_event)
    # Explicit ability declarations are a supported main consumer of -start.
    # A volatile with an incidental [from] ability tag is not that declaration.
    main_handled |= (kind == '-start' and bool(public_event.primary)
        and public_event.primary.strip().lower().startswith('ability:') and all(confirmed))
    if kind not in ('-ability', 'ability') and all(confirmed):
        ident, name = confirmed
        holder = addresses.resolve(ident)
        proc = PublicEvent(Kind.ABILITY, state.next_sequence + len(events), state.turn, state.phase,
            target_side=holder.side, target_slot=holder.slot,
            payload=(holder.encounter, abilities.ability_id(name), 0, 0, 0, 0, 0, 0))
        proc.words()
        events.append(proc)
    if item_projection is not None:
        for ident,operation,item_id in item_projection.facts:
            holder=addresses.resolve(ident)
            event=PublicEvent(Kind.ITEM,state.next_sequence+len(events),state.turn,state.phase,
                target_side=holder.side,target_slot=holder.slot,
                payload=(holder.encounter,item_id,operation,0,0,0,0,0))
            event.words();events.append(event)
    if field_catalog is not None:
        field_projection=public_field.project_line(line,catalog=field_catalog)
        main_handled |= bool(field_projection.facts)
        for operation,field_side,value,upkeep,ability_source in field_projection.facts:
            if operation==public_field.TURN:
                continue  # The same TURN event advances both cursor and field clock.
            event=PublicEvent(Kind.FIELD,state.next_sequence+len(events),state.turn,state.phase,
                target_side=field_side,payload=(operation,value,int(upkeep),int(ability_source),0,0,0,0))
            event.words();events.append(event)
    if active_catalog is not None:
        active_projection=public_active.project_line(line,catalog=active_catalog)
        main_handled |= bool(active_projection.facts) and kind in ACTIVE_MAIN_KINDS
        for operation,active_side,value in active_projection.facts:
            if operation==public_active.TURN:
                continue  # The one TURN event advances cursor, field and active clocks together.
            holder=addresses.resolve(parts[2])
            if holder.side!=active_side or candidate.active[holder.side]!=holder.slot or not candidate.live[holder.side]:
                raise ValueError("public active fact requires the current live encounter")
            event=PublicEvent(Kind.ACTIVE,state.next_sequence+len(events),state.turn,state.phase,
                target_side=holder.side,target_slot=holder.slot,payload=(holder.encounter,operation,value,0,0,0,0,0))
            event.words();events.append(event)
    if type_catalog is not None:
        type_projection=public_types.project_line(line,catalog=type_catalog)
        main_handled |= bool(type_projection.facts)
        for operation,type_side,first,second in type_projection.facts:
            # The champion parser attributes these facts by public side, even
            # for a positionless/stale nickname. Preserve that source rule.
            ident=candidate.active_idents().get(type_side)
            if not ident:raise ValueError('live-type fact requires an introduced active public encounter')
            holder=addresses.resolve(ident)
            event=PublicEvent(Kind.TYPES,state.next_sequence+len(events),state.turn,state.phase,
                target_side=holder.side,target_slot=holder.slot,payload=(holder.encounter,operation,first,second,0,0,0,0))
            event.words();events.append(event)
    if damage_tables is not None:
        damage_facts=public_damage.control_facts(line)
        main_handled |= bool(damage_facts) and kind in ('cant','-damage','-enditem','-activate')
        for operation,damage_side in damage_facts:
            event=PublicEvent(Kind.DAMAGE_CONTROL,state.next_sequence+len(events),state.turn,state.phase,
                target_side=damage_side,payload=(operation,0,0,0,0,0,0,0))
            event.words();events.append(event)
    if kind in MON_KINDS or line == '|':
        projected = public_mon_protocol.project_line(line, sequence=state.next_sequence + len(events), turn=state.turn,
            addresses=addresses, active_idents=candidate.active_idents(), species_ids=species_catalog.ids,
            source=source, phase=state.phase, baton_pass=side is not None and state.pending_baton[side],
            damage_metadata=damage_tables is not None)
        events.extend(projected.events)
    if kind in CONTEXT_KINDS:
        projected = public_context_protocol.project_line(line, sequence=state.next_sequence + len(events),
            turn=state.turn, phase=state.phase, addresses=addresses, moves=moves, abilities=abilities)
        events.extend(projected.events)
    if sleep_tables is not None:
        for operation,first,second in public_sleep_protocol.facts(line):
            main_handled=True
            holder=(addresses.known_idents.get(parts[2]) if operation==public_sleep.CURE and first==1
                else addresses.resolve(parts[2]))
            if not isinstance(holder,PublicAddress):raise ValueError('sleep fact requires its registered public encounter')
            event=PublicEvent(Kind.SLEEP,state.next_sequence+len(events),state.turn,state.phase,
                target_side=holder.side,target_slot=holder.slot,payload=(holder.encounter,operation,first,second,0,0,0,0))
            event.words();events.append(event)
    tendency_main=tendency_projection is not None and tendency_projection.payload[0]!=public_tendency.OTHER
    main_handled |= tendency_main
    proc_only=kind in PROC_ONLY_KINDS and (all(confirmed) or item_projection is not None and bool(item_projection.facts))
    main_handled |= proc_only
    if toxic_tables is not None:
        # Poison's raw state must see each accepted line exactly once. Common
        # lifecycle/clock events carry it directly; all other implemented lines
        # receive one lexical control. A generic Toxic no-op cannot turn an
        # otherwise unimplemented wire kind into an accepted full prefix.
        identity_ids={ident:address.encounter for ident,address in addresses.known_idents.items()}
        projection=public_toxic_protocol.project_line(line,identity_ids=identity_ids)
        if not projection.shared_event:
            if not main_handled and not projection.timestamp:
                raise ValueError('Toxic no-op cannot drop another raw line consumer')
            event=PublicEvent(Kind.TOXIC,state.next_sequence+len(events),state.turn,state.phase,
                target_side=projection.side,payload=projection.payload)
            event.words();events.append(event)
    if not main_handled and not (toxic_tables is not None and projection.timestamp):
        raise ValueError('raw main kind has no implemented consumer; incidental proc tags are insufficient')
    if (not events or item_projection is not None and len(events)==1 and not item_projection.trick_idents) and not tendency_main:
        raise ValueError("raw line has no implemented consumer; it must not be silently dropped")
    if tendency_projection is not None:
        last=events[-1]
        event=PublicEvent(Kind.TENDENCY,state.next_sequence+len(events),last.turn,last.phase,
            target_side=tendency_projection.side,payload=tendency_projection.payload)
        event.words();events.append(event)
    pending = {f'p{s+1}' for s in (0, 1) if state.pending_baton[s]}
    _flag_baton_pass(parts, pending)
    if side is not None:
        pending.discard(f'p{side+1}')
    if kind == 'turn':
        pending.clear()
    if kind == 'faint':
        faint_side = _slot_from_ident(parts[2])
        candidate = replace(candidate, live=tuple(False if faint_side == f'p{s+1}' else candidate.live[s]
                                                for s in (0, 1)))
    last = events[-1]
    candidate = replace(candidate, next_sequence=state.next_sequence + len(events),
        turn=last.turn, phase=last.phase, pending_baton=tuple(f'p{s+1}' in pending for s in (0, 1)))
    return candidate, WireProjection(line, tuple(events), species_binding=binding,
        ability_binding=tuple(int(x) for x in np.asarray(abilities.device.binding)),move_binding=move_binding,
        item_binding=item_binding,context_binding=context_binding,field_binding=field_binding,active_binding=active_binding,damage_binding=damage_binding,type_binding=type_binding,sleep_binding=sleep_binding,toxic_binding=toxic_binding,tendency_binding=tendency_binding)


def project_lines(state, lines, **tables):
    """Prepare a whole public chunk transaction without mutating its input."""
    candidate, projections = state, []
    for line in lines:
        candidate, projection = project_line(candidate, line, **tables)
        projections.append(projection)
    return candidate, tuple(projections)
