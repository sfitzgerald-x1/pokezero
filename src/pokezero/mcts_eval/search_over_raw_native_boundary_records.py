"""Pure native-boundary record laws, not native extraction or provenance.

Validate already supplied records and project their required fields. This shared
kernel starts no native engine, ancestry capture, callback, game or model. A valid
record does not establish physics, party mapping, branch correspondence, source
ownership, runtime qualification or scientific collection authority.
"""
from copy import deepcopy
import hashlib
import json

def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _state_sha(text):
    require(type(text) is str and bool(text), "captured serialized state required")
    return hashlib.sha256(text.encode()).hexdigest()


SCHEMA = "pokezero.search-over-raw.native-scoped-boundary.v1"
SEATS = ("p1", "p2")
SIDE_KEYS = {"active_identity", "hp", "fainted", "status", "active_moves",
    "party_identities", "engine_options", "search_arms", "interior_empty_fallback",
    "effective_legal_choices", "force_switch", "force_trapped", "slow_uturn_move"}
FLAGS = {"party_mapping_validated", "showdown_requests_inferred", "branch_transition_validated",
    "runtime_qualified", "scientific_collection_authorized"}


def _identity(value, maximum):
    require(type(value) is dict and set(value) == {"slot", "engine_id"}
        and type(value["slot"]) is int and 0 <= value["slot"] < maximum
        and type(value["engine_id"]) is str and bool(value["engine_id"]),
        "actual native slot identity required")


def _choice(value):
    require(type(value) is dict and set(value) == {"kind", "slot", "engine_id"},
        "native option identity field drift")
    kind = value["kind"]
    require(kind in {"move", "switch", "struggle", "none"}, "native option kind drift")
    if kind in {"move", "switch"}:
        _identity({k:value[k] for k in ("slot", "engine_id")}, 4 if kind == "move" else 6)
    else:
        require(value["slot"] is None and value["engine_id"] ==
            ("struggle" if kind == "struggle" else "No Move"), "synthetic option identity drift")


def read_native_boundary(report, *, state_text, root):
    """Exact report bindings and internal laws; not an independent physics audit."""
    saved = deepcopy(report)
    require(type(root) is bool and type(saved) is dict and set(saved) ==
        {"schema", "state_utf8_sha256", "root", "team_preview", "native_outcome", "terminal",
         "terminal_choice_law", "root_search_admissible", "p1", "p2", "native_pp_class"} | FLAGS,
        "native boundary schema field drift")
    require(saved["schema"] == SCHEMA and saved["state_utf8_sha256"] == _state_sha(state_text)
        and saved["root"] is root and saved["native_pp_class"] == "engine_pp_model"
        and saved["terminal_choice_law"] == "no_new_choices_at_terminal"
        and all(saved[k] is False for k in FLAGS), "native boundary binding/scope drift")
    require(type(saved["native_outcome"]) in (int, float) and saved["native_outcome"] in (-1,0,1)
        and saved["terminal"] is (saved["native_outcome"] != 0)
        and type(saved["team_preview"]) is bool, "native terminal fact drift")
    for seat in SEATS:
        side = saved[seat]
        require(type(side) is dict and set(side) == SIDE_KEYS, "native seat field drift")
        _identity(side["active_identity"],6)
        hp = side["hp"]
        require(type(hp) is list and len(hp) == 2 and all(type(v) is int for v in hp)
            and hp[1] > 0 and 0 <= hp[0] <= hp[1] and side["fainted"] is (hp[0] <= 0)
            and side["status"] in {"none","burn","paralyze","poison","toxic","sleep","freeze"},
            "native active health/status drift")
        require(all(type(side[k]) is bool for k in ("force_switch", "force_trapped",
            "slow_uturn_move", "interior_empty_fallback")), "native side flag drift")
        moves, party = side["active_moves"], side["party_identities"]
        require(type(moves) is list and len(moves) == 4 and type(party) is list
            and 1 <= len(party) <= 6, "native move/party inventory drift")
        for slot, move in enumerate(moves):
            require(type(move) is dict and set(move) == {"slot","engine_id","pp","disabled"}
                and move["slot"] == slot and type(move["pp"]) is int
                and type(move["disabled"]) is bool, "native move slot/PP drift")
            _identity({k:move[k] for k in ("slot","engine_id")},4)
        for identity in party: _identity(identity,6)
        require([v["slot"] for v in party] == sorted(set(v["slot"] for v in party))
            and side["active_identity"] in party, "native active/party identity drift")
        for field in ("engine_options", "search_arms", "effective_legal_choices"):
            require(type(side[field]) is list and len(side[field]) <= 10, "native option vector drift")
            for value in side[field]:
                _choice(value)
                if value["kind"] == "move":
                    require(moves[value["slot"]]["engine_id"] == value["engine_id"],
                        "native move option/catalog drift")
                if value["kind"] == "switch":
                    require({k:value[k] for k in ("slot","engine_id")} in party
                        and (value["slot"] != side["active_identity"]["slot"]
                            or (root and saved["team_preview"])),
                        "native switch option/catalog drift")
            require(len(set(digest(v) for v in side[field])) == len(side[field]),
                "duplicate native option slot")
        raw = side["engine_options"]
        fallback = not root and not raw
        require(side["interior_empty_fallback"] is fallback
            and side["search_arms"] == (raw if not fallback else
                [{"kind":"none","slot":None,"engine_id":"No Move"}])
            and side["effective_legal_choices"] == ([] if saved["terminal"] else raw),
            "native option/fallback/terminal law drift")
    admissible = not root or all(saved[s]["engine_options"] for s in SEATS)
    require(saved["root_search_admissible"] is bool(admissible), "native root rejection drift")
    require(digest(report) == digest(saved), "native export mutated during read")
    return saved


def project_native_boundary(report):
    """Required-field view of an already structurally validated native export."""
    native = {field:{seat:deepcopy(report[seat][source]) for seat in SEATS}
        for field,source in (("active_identities","active_identity"),("hp","hp"),
            ("fainted","fainted"),("status","status"))}
    native["effective_legal_choices"] = {seat:sorted(deepcopy(report[seat]["effective_legal_choices"]),
        key=lambda v:(v["kind"],-1 if v["slot"] is None else v["slot"],v["engine_id"])) for seat in SEATS}
    native["native_pp"] = {seat:[{k:v[k] for k in ("slot","engine_id","pp")}
        for v in report[seat]["active_moves"]] for seat in SEATS}
    native["action_identity"] = None
    native["branch_transition"] = None
    return native
