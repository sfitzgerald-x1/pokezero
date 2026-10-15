"""Private observed world/native assembly and conservative original-slot joins.

These witnesses bind actual constructor outputs, not a caller's party map.
Hashes are drift guards, not operator authentication. Owning-run and complete
ancestry joins remain separate; no construction witness certifies dynamics.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from .search_over_raw import digest, require

SCHEMA = "pokezero.search-over-raw.native-construction.v1"
SEATS = ("p1","p2")
STATS = ("hp","atk","def","spa","spd","spe")
IDENTITY_FIELDS = ("species","moves","ability","item","level","nature","gender","evs","ivs")


def construction_source_hashes():
    package = Path(__file__).parents[1]
    paths = [Path(__file__),*(package/name for name in
        ("engine_world.py","engine_search.py","poke_engine_adapter.py","showdown_fixture.py"))]
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _text_sha(text):
    require(type(text) is str and text,"actual constructed native state text required")
    return hashlib.sha256(text.encode()).hexdigest()


class NativeConstructionCapture:
    """One actual draw/world/native assembly; failed or repeated hooks poison it."""
    def __init__(self, *, binding, ordinal, draw_sha256):
        require(type(binding) is dict and binding and type(ordinal) is int and ordinal >= 0,
            "externally bound native construction ordinal required")
        self.binding,self.ordinal,self.draw_sha = deepcopy(binding),ordinal,draw_sha256
        self.sources,self.events,self.failed = construction_source_hashes(),[],False
        self.inputs = digest(dict(binding=self.binding,ordinal=ordinal,draw_sha256=draw_sha256,sources=self.sources))

    def _check(self):
        require(not self.failed and self.sources == construction_source_hashes()
            and self.inputs == digest(dict(binding=self.binding,ordinal=self.ordinal,
                draw_sha256=self.draw_sha,sources=self.sources)),"construction witness failed or inputs drifted")

    def world_built(self, value):
        try:
            self._check()
            require(not self.events and type(value) is dict and set(value) == {"payload","override","world"}
                and digest(value["override"]) == self.draw_sha,"one actual drawn world assembly required")
            self.events.append(dict(kind="world_built",value=deepcopy(value)))
        except BaseException:
            self.failed = True
            raise

    def native_built(self, value):
        try:
            self._check()
            require(len(self.events) == 1 and type(value) is dict and set(value) == {"spec","native_state"}
                and digest(value["spec"]) == digest(self.events[0]["value"]["world"]["spec"]),
                "native assembly must consume the actual constructed spec")
            _text_sha(value["native_state"])
            self.events.append(dict(kind="native_built",value=deepcopy(value)))
        except BaseException:
            self.failed = True
            raise

    def export(self, *, expected_world, expected_state):
        self._check()
        value = dict(schema=SCHEMA,binding=deepcopy(self.binding),ordinal=self.ordinal,
            draw_sha256=self.draw_sha,source_hashes=deepcopy(self.sources),events=deepcopy(self.events),
            private_controller_only=True)
        read_native_construction(value,binding=self.binding,ordinal=self.ordinal,draw_sha256=self.draw_sha,
            constructed_sha256=digest(expected_world),state_utf8_sha256=_text_sha(expected_state),source_hashes=self.sources)
        return value


def _identity(mon):
    """Identity-bearing sampled-set fields, not an effective-state equivalence test."""
    from ..dex import normalize_id
    return dict(species=normalize_id(mon.species),moves=[normalize_id(v) for v in mon.moves],
        ability=normalize_id(mon.ability or ""),item=normalize_id(mon.item or ""),level=mon.level,
        nature=normalize_id(mon.nature or "serious"),gender=mon.gender or "",
        evs={s:(mon.evs or {}).get(s,0) for s in STATS},ivs={s:(mon.ivs or {}).get(s,31) for s in STATS})


def _snapshot_identity(value):
    """Malformed or incomplete immutable sets remain unknown, never normalized away."""
    from ..showdown_fixture import FixturePokemon
    required = {"species","moves","ability","item","level","nature","gender","evs","ivs"}
    if type(value) is not dict or not required <= set(value): return None
    if "happiness" in value and (type(value["happiness"]) is not int or value["happiness"] != 255): return None
    if "shiny" in value and value["shiny"] is not False: return None
    if any(type(value[k]) is not str for k in ("species","ability","item","nature","gender")) \
            or not value["species"] or not value["nature"] or value["gender"] not in {"","M","F"} \
            or type(value["level"]) is not int or not 1 <= value["level"] <= 100 \
            or type(value["moves"]) is not list or not 1 <= len(value["moves"]) <= 4 \
            or any(type(v) is not str or not v for v in value["moves"]): return None
    for key,ceiling in (("evs",255),("ivs",31)):
        if type(value[key]) is not dict or set(value[key]) != set(STATS) \
                or any(type(v) is not int or not 0 <= v <= ceiling for v in value[key].values()): return None
    return _identity(FixturePokemon(**{k:deepcopy(value[k]) for k in required}))


def read_native_construction(capture, *, binding, ordinal, draw_sha256, constructed_sha256,
                             state_utf8_sha256, source_hashes):
    """Independent schema/input/order reader; no callback hash certifies ownership."""
    from ..dex import normalize_id
    from ..engine_world import unpack_team, _engine_species_id
    before = digest(dict(capture=capture,binding=binding,ordinal=ordinal,draw=draw_sha256,
        world=constructed_sha256,state=state_utf8_sha256,sources=source_hashes))
    saved = deepcopy(capture)
    require(type(saved) is dict and set(saved) == {"schema","binding","ordinal","draw_sha256",
        "source_hashes","events","private_controller_only"} and saved["schema"] == SCHEMA
        and saved["binding"] == binding and type(saved["ordinal"]) is int and saved["ordinal"] == ordinal
        and saved["draw_sha256"] == draw_sha256 and saved["source_hashes"] == source_hashes
        and source_hashes == construction_source_hashes() and saved["private_controller_only"] is True,
        "exact actual-construction bindings required")
    events = saved["events"]
    require(type(events) is list and len(events) == 2 and all(type(v) is dict and set(v) == {"kind","value"}
        for v in events) and [v["kind"] for v in events] == ["world_built","native_built"],
        "complete ordered world/native construction hooks required")
    world_event,native_event = (v["value"] for v in events)
    require(type(world_event) is dict and set(world_event) == {"payload","override","world"}
        and type(native_event) is dict and set(native_event) == {"spec","native_state"},"construction event schema drift")
    draw,world = world_event["override"],world_event["world"]
    require(type(draw) is dict and digest(draw) == draw_sha256
        and type(draw.get("player_teams")) is dict and set(draw["player_teams"]) == set(SEATS)
        and type(world) is dict and set(world) == {"spec","party_species","slot_sides"}
        and world["slot_sides"] == dict(p1="side_one",p2="side_two") and digest(world) == constructed_sha256
        and digest(world["spec"]) == digest(native_event["spec"])
        and _text_sha(native_event["native_state"]) == state_utf8_sha256,"actual draw/spec/native-output join drift")
    slots = {}
    for seat in SEATS:
        packed = draw["player_teams"][seat]
        require(type(packed) is str and packed,"actual packed team required")
        # The inverse parser intentionally omits these fields. Do not let a
        # nondefault happiness/extended tail silently alias its default set.
        parts = [mon.split("|") for mon in packed.split("]")]
        require(all(len(mon) == 12 and mon[9] == "" and mon[11] in {"","255"} for mon in parts),
            "unsupported packed identity fields remain unqualified")
        team = unpack_team(packed)
        side = world["spec"][world["slot_sides"][seat]]
        party = side["pokemon"]
        require(1 <= len(team) <= 6 and len(party) == len(team)
            and list(world["party_species"][seat]) == [normalize_id(mon.species) for mon in team]
            and type(side["active_index"]) is int and 0 <= side["active_index"] < len(team),
            "actual sampled-team/native-construction ordinal drift")
        rows = []
        for index,(mon,built) in enumerate(zip(team,party)):
            # Transform retains the constructor's original slot, but changes
            # current species. Use its captured pre-copy identity for this join.
            origin = built.get("pre_transform") or built
            require(origin["id"] == _engine_species_id(normalize_id(mon.species)),
                "actual construction species/slot drift")
            rows.append(dict(native_slot=index,sampled_identity=_identity(mon),
                current_engine_id=built["id"],built_pokemon_sha256=digest(built)))
        slots[seat] = rows
    require(before == digest(dict(capture=capture,binding=binding,ordinal=ordinal,draw=draw_sha256,
        world=constructed_sha256,state=state_utf8_sha256,sources=source_hashes)),"construction inputs mutated during read")
    return dict(schema=SCHEMA,capture_sha256=digest(saved),state_utf8_sha256=state_utf8_sha256,
        slots=slots,identity_fields=list(IDENTITY_FIELDS),nickname_excluded=True,
        default_happiness=255,nonshiny_only=True,construction_order_consistent=True,process_ownership_validated=False,
        construction_provenance_validated=False,branch_transition_validated=False,
        runtime_qualified=False,scientific_collection_authorized=False)


def derive_party_mapping(root_snapshot, native_root, construction):
    """Unique immutable identity-field match, never first-match or caller permutation.

    Missing fields or colliding identities leave the mapping unknown. This
    structural result still requires construction/owning-run evidence joins.
    """
    from .search_over_raw_showdown_boundary import _snapshot,freeze_party_mapping,_species
    before = digest(dict(snapshot=root_snapshot,native=native_root,construction=construction))
    require(type(construction) is dict and construction.get("schema") == SCHEMA
        and construction.get("construction_order_consistent") is True
        and native_root.get("state_utf8_sha256") == construction.get("state_utf8_sha256")
        and native_root.get("root") is True,"native root must join observed constructor output")
    sides = _snapshot(root_snapshot)
    orders,limitations = {},[]
    for seat in SEATS:
        original_sets = [sides[seat]["side"]["pokemon"][current]["set"]
            for current in sides[seat]["original_to_current"]]
        identities = [_snapshot_identity(value) for value in original_sets]
        rows = construction["slots"][seat]
        if len(rows) != len(original_sets):
            limitations.append(f"{seat}:party_inventory_mismatch")
            continue
        order = []
        for row in rows:
            matches = [i for i,value in enumerate(identities)
                if value is not None and digest(value) == digest(row["sampled_identity"])]
            if len(matches) != 1:
                limitations.append(f"{seat}:missing_or_ambiguous_immutable_set_identity")
                break
            order.append(matches[0])
        if len(order) == len(rows) and len(set(order)) == len(order): orders[seat] = order
        elif len(order) == len(rows): limitations.append(f"{seat}:nonbijective_set_identity")
    mapping = None
    if not limitations:
        for seat in SEATS:
            party = native_root.get(seat, {}).get("party_identities")
            order = orders[seat]
            if type(party) is not list or len(party) != len(order) or any(type(v) is not dict for v in party) \
                    or [v.get("slot") for v in party] != list(range(len(order))) \
                    or any(type(v.get("slot")) is not int for v in party):
                limitations.append(f"{seat}:native_root_party_inventory_mismatch")
                continue
            ref = sides[seat]
            if any(party[slot].get("engine_id") != _species(ref["side"]["pokemon"][
                    ref["original_to_current"][original]]) for slot, original in enumerate(order)):
                limitations.append(f"{seat}:current_species_slot_mismatch")
    if not limitations:
        # The existing projector checks CURRENT engine identities against the
        # actual native root. Transform cannot be converted to an origin species.
        mapping = freeze_party_mapping(root_snapshot,native_root,native_to_original=orders)
    require(before == digest(dict(snapshot=root_snapshot,native=native_root,construction=construction)),
        "party identity inputs mutated during read")
    return dict(mapping=mapping,limitations=limitations,construction_capture_sha256=construction["capture_sha256"],
        snapshot_sha256=digest(root_snapshot),native_root_sha256=digest(native_root),
        unique_set_mapping_derived=mapping is not None,construction_provenance_validated=False,
        snapshot_provenance_validated=False,runtime_qualified=False,scientific_collection_authorized=False)
