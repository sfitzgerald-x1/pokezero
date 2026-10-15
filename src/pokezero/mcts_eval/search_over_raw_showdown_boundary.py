"""Private, read-only Showdown snapshot/request projection and identity joins.

No simulator is opened. This reads the pinned serializer's original-team
permutation, NOT species/nickname first matches. Inputs remain private. A frozen
map is a structurally checked construction claim, not owning-run provenance;
these exports never certify replay, snapshot freshness or scientific runtime.
"""
from copy import deepcopy
import re

from .search_over_raw import digest, require

SEATS = ("p1", "p2")
STATUSES = {"":"none", "brn":"burn", "par":"paralyze", "psn":"poison",
    "tox":"toxic", "slp":"sleep", "frz":"freeze"}
MAP_SCHEMA = "pokezero.search-over-raw.boundary-party-map.v1"


class NativeReplayUnavailable(ValueError):
    """Valid observed boundary cannot execute the next frozen joint action.

    Malformed requests, mappings and ambiguous commands are NOT this outcome.
    Legacy strict replay still fails because this remains a ValueError.
    """
    def __init__(self, reason, actors=()):
        require(reason in {"REFERENCE_TERMINAL", "REQUEST_LEGALITY_UNAVAILABLE", "NATIVE_ACTION_UNAVAILABLE"}
            and type(actors) is tuple and all(s in SEATS for s in actors),
            "closed native replay unavailability category required")
        self.reason, self.actors = reason, actors
        super().__init__("native replay unavailable: " + reason)


def _species(mon):
    value = mon.get("species")
    require(type(value) is str and re.fullmatch(r"\[Species:[a-z0-9]+\]",value) is not None,
        "pinned serialized current species reference required")
    species = value[len("[Species:"):-1]
    # Same documented cosmetic Unown collapse as the native world constructor.
    return "unown" if species.startswith("unown") else species


def _snapshot(snapshot):
    require(type(snapshot) is dict and set(snapshot) == {"battle", "boundaryRequests", "terminalScheduled"}
        and type(snapshot["terminalScheduled"]) is bool and type(snapshot["battle"]) is dict
        and type(snapshot["boundaryRequests"]) is dict and set(snapshot["boundaryRequests"]) <= set(SEATS),
        "paired private snapshot/request payload required")
    battle = snapshot["battle"]
    require(battle.get("formatid") in {"gen3randombattle", "gen3customgame"}
        and battle.get("gameType") == "singles" and type(battle.get("ended")) is bool
        and type(battle.get("sides")) is list and len(battle["sides"]) == 2,
        "known Gen 3 singles snapshot required")
    sides = {}
    for seat, side in zip(SEATS,battle["sides"]):
        require(type(side) is dict and side.get("id") == seat
            and type(side.get("pokemon")) is list and 1 <= len(side["pokemon"]) <= 6
            and type(side.get("team")) is str, "pinned serialized singles party required")
        count = len(side["pokemon"])
        require(len(side["team"]) == count and set(side["team"]) == set(str(v) for v in range(1,count+1)),
            "original-team/current-party permutation required")
        original_to_current = [int(v)-1 for v in side["team"]]
        current_to_original = [original_to_current.index(v) for v in range(count)]
        require(side.get("active") == [f"[Pokemon:{seat}a]"], "one actual serialized active reference required")
        for current, mon in enumerate(side["pokemon"]):
            require(type(mon) is dict and type(mon.get("position")) is int and mon["position"] == current
                and type(mon.get("isActive")) is bool
                and mon["isActive"] is (current == 0 and mon.get("fainted") is not True)
                and type(mon.get("set")) is dict and type(mon.get("hp")) is int
                and type(mon.get("maxhp")) is int and mon["maxhp"] > 0
                and 0 <= mon["hp"] <= mon["maxhp"] and type(mon.get("fainted")) is bool
                and mon.get("status") in STATUSES, "serialized party health/position/status drift")
            _species(mon)
        sides[seat] = dict(side=side,original_to_current=original_to_current,
            current_to_original=current_to_original)
    digest(snapshot)  # Full input remains bound, including fields outside this scoped projection.
    return sides


def freeze_party_mapping(root_snapshot, native_root, *, native_to_original):
    """Bind an EXPLICIT construction-slot join; never discover it by species.

    The caller must independently bind this construction manifest to the actual
    consumed world and reference root before accepting a boundary comparison.
    Duplicate species are safe only with this explicit, bijective slot claim.
    """
    sides = _snapshot(root_snapshot)
    require(type(native_root) is dict and native_root.get("root") is True
        and type(native_to_original) is dict and set(native_to_original) == set(SEATS),
        "bound native root and explicit construction slot manifest required")
    maps, sets = {}, {}
    for seat in SEATS:
        count = len(sides[seat]["side"]["pokemon"])
        order = native_to_original[seat]
        require(type(order) is list and len(order) == count and all(type(v) is int for v in order)
            and sorted(order) == list(range(count)), "complete bijective construction-slot map required")
        party = native_root.get(seat,{}).get("party_identities")
        require(type(party) is list and len(party) == count
            and [v.get("slot") for v in party] == list(range(count)), "native construction party inventory drift")
        reference = sides[seat]
        for native_slot, original in enumerate(order):
            mon = reference["side"]["pokemon"][reference["original_to_current"][original]]
            require(party[native_slot].get("engine_id") == _species(mon), "construction species/slot join drift")
        sets[seat] = [digest(reference["side"]["pokemon"][current]["set"])
            for current in reference["original_to_current"]]
        maps[seat] = deepcopy(order)
    return dict(schema=MAP_SCHEMA,root_snapshot_sha256=digest(root_snapshot),
        native_root_export_sha256=digest(native_root),native_to_original=maps,
        original_set_sha256s=sets,construction_provenance_validated=False,
        snapshot_provenance_validated=False,runtime_qualified=False)


def _mapped_sides(snapshot, mapping):
    require(type(mapping) is dict and set(mapping) == {"schema", "root_snapshot_sha256",
        "native_root_export_sha256", "native_to_original", "original_set_sha256s",
        "construction_provenance_validated", "snapshot_provenance_validated", "runtime_qualified"}
        and mapping["schema"] == MAP_SCHEMA and all(mapping[k] is False for k in
            ("construction_provenance_validated", "snapshot_provenance_validated", "runtime_qualified"))
        and all(type(mapping[k]) is dict and set(mapping[k]) == set(SEATS)
            for k in ("native_to_original", "original_set_sha256s")), "unpromoted frozen party map required")
    sides = _snapshot(snapshot)
    for seat in SEATS:
        ref = sides[seat]; count = len(ref["side"]["pokemon"])
        order, sets = mapping["native_to_original"][seat], mapping["original_set_sha256s"][seat]
        require(type(order) is list and len(order) == count and all(type(v) is int for v in order)
            and sorted(order) == list(range(count)) and type(sets) is list and len(sets) == count,
            "frozen party map inventory drift")
        require(sets == [digest(ref["side"]["pokemon"][current]["set"])
            for current in ref["original_to_current"]], "immutable original-team identity changed")
        ref["current_to_native"] = [order.index(original) for original in ref["current_to_original"]]
    return sides


def _move_id(move_id, mon):
    require(type(move_id) is str and re.fullmatch(r"[a-z0-9]+",move_id) is not None,
        "actual request move id required")
    if move_id == "hiddenpower":
        kind, power = mon.get("hpType"), mon.get("hpPower")
        require(type(kind) is str and re.fullmatch(r"[A-Za-z]+",kind) is not None
            and type(power) is int and 30 <= power <= 70, "actual Hidden Power type/power required")
        return f"hiddenpower{kind.lower()}{power}"
    return move_id


def _move_slots(mon):
    slots = mon.get("moveSlots")
    require(type(slots) is list and 1 <= len(slots) <= 4, "serialized active move slots required")
    for value in slots:
        require(type(value) is dict and type(value.get("pp")) is int and value["pp"] >= 0,
            "serialized move PP required")
        _move_id(value.get("id"),mon)
    return slots


def _request_choices(request, reference, *, terminal):
    """All request choices in native party/move slots; no inferred commitments."""
    if request is None: return ([], [], []) if terminal else (None, None, ["missing_paired_request"])
    require(type(request) is dict, "actual paired request object required")
    require(all(type(request[k]) is bool for k in ("wait","teamPreview") if k in request),
        "exact request wait/preview flags required")
    if terminal: return [], [], []
    party = reference["side"]["pokemon"]
    request_side = request.get("side")
    require(type(request_side) is dict and request_side.get("id") == reference["side"]["id"]
        and type(request_side.get("pokemon")) is list and len(request_side["pokemon"]) == len(party),
        "request/current-party inventory drift")
    # Verify request order against the exact private serialized state, not names.
    for current, row in enumerate(request_side["pokemon"]):
        mon = party[current]
        require(type(row) is dict and type(row.get("active")) is bool and row["active"] is (current == 0)
            and type(row.get("condition")) is str, "request/current active drift")
        condition = row["condition"].split()
        expected = [f"{mon['hp']}/{mon['maxhp']}"] + ([mon["status"]] if mon["status"] else [])
        require(condition == (["0","fnt"] if mon["hp"] == 0 else expected),
            "request/serialized exact health-status drift")
    if request.get("wait") is True:
        return [dict(kind="none",slot=None,engine_id="No Move")], [], []
    if request.get("teamPreview") is True:
        return None, None, ["team_preview_outside_boundary_request_law"]
    force = request.get("forceSwitch")
    require(force is None or type(force) is list and len(force) == 1 and type(force[0]) is bool,
        "singles replacement request required")
    replacing = force == [True]
    choices, commands, limitations = [], [], []
    withheld = False
    allow_switch = replacing
    if not replacing:
        active = request.get("active")
        require(type(active) is list and len(active) == 1 and type(active[0]) is dict
            and type(active[0].get("moves")) is list and 1 <= len(active[0]["moves"]) <= 4,
            "full singles move request required")
        active = active[0]
        flags = [active.get(k,False) for k in ("maybeTrapped","maybeDisabled","maybeLocked")]
        require(all(type(flag) is bool for flag in flags), "exact request withholding flags required")
        withheld = any(flags)
        require(type(active.get("trapped",False)) is bool, "explicit request trap flag required")
        allow_switch = not active.get("trapped",False)
        slots = _move_slots(party[0])
        for request_slot, move in enumerate(active["moves"]):
            require(type(move) is dict and type(move.get("disabled",False)) is bool,
                "explicit request disabled flag required")
            move_id = move.get("id")
            _move_id(move_id,party[0])  # A withheld/disabled row cannot hide malformed identity.
            if move.get("disabled",False): continue
            if move_id in {"struggle","recharge"}:
                choice = dict(kind="struggle" if move_id == "struggle" else "none",slot=None,
                    engine_id="struggle" if move_id == "struggle" else "No Move")
            else:
                matches = [slot for slot,v in enumerate(slots) if v["id"] == move_id]
                if len(matches) != 1:
                    return None, None, ["request_move_slot_ambiguous_or_absent"]
                choice = dict(kind="move",slot=matches[0],engine_id=_move_id(move_id,party[0]))
            choices.append(choice); commands.append(dict(command=f"move {request_slot+1}",choice=choice))
    if allow_switch:
        for current, mon in enumerate(party):
            if current == 0 or mon["fainted"] or mon["hp"] <= 0: continue
            choice = dict(kind="switch",slot=reference["current_to_native"][current],engine_id=_species(mon))
            choices.append(choice); commands.append(dict(command=f"switch {current+1}",choice=choice))
    require(len(set(digest(v) for v in choices)) == len(choices), "duplicate full effective choice identity")
    if withheld: return None, None, ["request_withholds_effective_legality"]
    return sorted(choices,key=lambda v:(v["kind"],-1 if v["slot"] is None else v["slot"],v["engine_id"])), commands, limitations


def commands_for_native_actions(snapshot, mapping, actions):
    """Resolve one frozen full native joint action from the actual PRE-state.

    No species-first lookup, policy choice, saved commitment, fallback or RNG
    coupling is inferred. Omission requires an explicit wait request. A target
    unavailable after divergence refuses the full path instead of reselecting.
    """
    from .search_over_raw_native_boundary import _choice
    before = digest(dict(snapshot=snapshot, mapping=mapping, actions=actions))
    require(type(actions) is dict and set(actions) == set(SEATS), "resolved full native joint action required")
    for action in actions.values(): _choice(action)
    projected = project_showdown_boundary(snapshot, mapping)
    from .search_over_raw_boundary_transport import _requests
    requested = _requests(snapshot)  # Missing/malformed paired state is not ordinary unavailability.
    commands, withheld, absent = {}, [], []
    for seat in SEATS:
        catalog = projected["command_catalogs"][seat]
        if catalog is None:
            require(projected["limitations"][seat] == ["request_withholds_effective_legality"],
                "ambiguous or unsupported request catalog is not ordinary native action unavailability")
            withheld.append(seat)
            continue
        matches = [row["command"] for row in catalog if digest(row["choice"]) == digest(actions[seat])]
        if matches:
            require(len(matches) == 1, "native action has ambiguous executable request commands")
            commands[seat] = matches[0]
        else:
            waiting = (not catalog and snapshot["boundaryRequests"].get(seat, {}).get("wait") is True
                and digest(actions[seat]) == digest(dict(kind="none", slot=None, engine_id="No Move")))
            if not waiting: absent.append(seat)
    require(before == digest(dict(snapshot=snapshot, mapping=mapping, actions=actions)),
        "native command resolution inputs changed")
    # Validate both catalogs before reporting any normal stopping category.
    if projected["reference_terminal"]: raise NativeReplayUnavailable("REFERENCE_TERMINAL")
    if withheld: raise NativeReplayUnavailable("REQUEST_LEGALITY_UNAVAILABLE", tuple(withheld))
    if absent: raise NativeReplayUnavailable("NATIVE_ACTION_UNAVAILABLE", tuple(absent))
    require(commands and set(commands) == set(requested), "native commands must cover actual requested actors")
    return commands


def project_showdown_boundary(snapshot, mapping, *, before_snapshot=None, incoming_choices=None):
    """Derive partial fields from actual paired raw inputs, with no certification.

    Incoming actions require exact choice strings and the previous snapshot;
    omitted/nonactionable requests never become native saved commitments. The
    future owning transport validator must bind those strings to actual sends.
    """
    before = digest(dict(snapshot=snapshot,mapping=mapping,before_snapshot=before_snapshot,incoming=incoming_choices))
    sides = _mapped_sides(snapshot,mapping)
    projection = {k:{} for k in ("active_identities","hp","fainted","status","effective_legal_choices","native_pp")}
    catalogs, limitations = {}, {}
    terminal = snapshot["battle"]["ended"]
    for seat in SEATS:
        ref = sides[seat]; mon = ref["side"]["pokemon"][0]
        projection["active_identities"][seat] = dict(slot=ref["current_to_native"][0],engine_id=_species(mon))
        projection["hp"][seat] = [mon["hp"],mon["maxhp"]]
        projection["fainted"][seat] = mon["fainted"]
        projection["status"][seat] = STATUSES[mon["status"]]
        choices, commands, limits = _request_choices(snapshot["boundaryRequests"].get(seat),ref,terminal=terminal)
        projection["effective_legal_choices"][seat], catalogs[seat], limitations[seat] = choices, commands, limits
        slots = _move_slots(mon)
        projection["native_pp"][seat] = [dict(slot=slot,engine_id=_move_id(v["id"],mon),pp=v["pp"])
            for slot,v in enumerate(slots)] + [dict(slot=slot,engine_id="none",pp=0) for slot in range(len(slots),4)]
    if any(v is None for v in projection["effective_legal_choices"].values()):
        projection["effective_legal_choices"] = None
    projection["action_identity"] = None
    if before_snapshot is not None and incoming_choices is not None:
        require(type(incoming_choices) is dict and set(incoming_choices) <= set(SEATS), "actual joint choice commands required")
        previous = _mapped_sides(before_snapshot,mapping); actions = {}
        for seat in SEATS:
            available, commands, _ = _request_choices(before_snapshot["boundaryRequests"].get(seat),previous[seat],
                terminal=before_snapshot["battle"]["ended"])
            if commands is None: break
            if not commands and available == [dict(kind="none",slot=None,engine_id="No Move")]:
                if seat in incoming_choices: break
                actions[seat] = available[0]
            else:
                matches = [v["choice"] for v in commands if v["command"] == incoming_choices.get(seat)]
                if len(matches) != 1: break
                actions[seat] = matches[0]
        if set(actions) == set(SEATS): projection["action_identity"] = actions
    # These are reference facts only; no claim of matched native transitions.
    projection["branch_transition"] = None
    require(before == digest(dict(snapshot=snapshot,mapping=mapping,before_snapshot=before_snapshot,incoming=incoming_choices)),
        "private boundary inputs changed during projection")
    return dict(showdown=projection,snapshot_sha256=digest(snapshot),mapping_sha256=digest(mapping),
        command_catalogs=catalogs,limitations=limitations,
        reference_terminal=terminal,terminal_scheduled=snapshot["terminalScheduled"],
        independently_validated=False,snapshot_provenance_validated=False,
        construction_provenance_validated=False,branch_transition_validated=False,
        runtime_qualified=False,scientific_collection_authorized=False)
