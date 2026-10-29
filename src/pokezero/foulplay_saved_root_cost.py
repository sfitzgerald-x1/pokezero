"""Parser-only Foul Play inputs for a separately admitted saved-TRAIN-root cost gate.

This module never selects a move, executes search, submits an action, starts a
simulator, or imports the opponent at module import time. The caller owns the
frozen roster, production parser reconstruction, source/runtime attestation and
any later explicit cost-only think. Historical recorded actions are DATA used
only to maintain Foul Play's own last-selected-move bookkeeping before the root.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Callable

SHOWDOWN_SOURCE = "f76228a1354b5d0f307ca2d16101294ad3a2308b"
_HEALTH = re.compile(r"^(\d+)/(\d+)(?: ([a-z]+))?$")
_IDENT = re.compile(r"^(p[12])(?:[a-z])?:")


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def actor_protocol_line(line: str, actor: str) -> str | None:
    """Project the trusted omniscient+request archive onto one Gen3 player wire.

    Gen3 Random Battle inherits Standard AG's HP Percentage Mod. This is the
    pinned Pokemon.getHealth shared branch, including its near-full 99 guard;
    own exact HP and own private requests remain exact. Other requests never
    cross the seam. A new unrecognized HP-bearing command refuses, rather than
    passing an omniscient numerator/denominator to the opponent.
    """
    _require(actor in ("p1", "p2") and isinstance(line, str) and line.startswith("|"),
             "one owning seat and a protocol line required")
    parts = line.split("|")
    _require(len(parts) >= 2, "malformed protocol")
    command = parts[1]
    if command == "request":
        _require(len(parts) == 3, "one request JSON payload required")
        request = json.loads(parts[2])
        owner = request.get("side", {}).get("id") if isinstance(request, dict) else None
        _require(owner in ("p1", "p2"), "request owner missing")
        return line if owner == actor else None
    fields: list[tuple[int, int]] = []
    if command in ("switch", "drag", "replace"):
        fields = [(2, 4)]
    elif command in ("-damage", "-heal"):
        fields = [(2, 3)]
    elif command == "-sethp":
        fields = [(2, 3)]
        if len(parts) > 5 and _IDENT.match(parts[4]):
            fields.append((4, 5))
    covered = {hp for _, hp in fields}
    for index, value in enumerate(parts):
        _require(not _HEALTH.fullmatch(value) or index in covered,
                 "unrecognized HP-bearing protocol command")
    for ident_index, hp_index in fields:
        _require(len(parts) > hp_index, "incomplete HP-bearing protocol")
        identity = _IDENT.match(parts[ident_index])
        _require(identity is not None, "health has no owning player identity")
        value = parts[hp_index]
        if value == "0 fnt":
            continue
        health = _HEALTH.fullmatch(value)
        _require(health is not None, "unknown health representation")
        hp, maximum = int(health[1]), int(health[2])
        _require(0 < hp <= maximum, "invalid exact health fraction")
        if identity[1] != actor:
            percentage = (100 * hp + maximum - 1) // maximum
            if percentage == 100 and hp < maximum:
                percentage = 99
            parts[hp_index] = f"{percentage}/100" + (f" {health[3]}" if health[3] else "")
    return "|".join(parts)


@dataclass(frozen=True)
class FoulPlayParserAPI:
    battle: Callable[[str], Any]
    random_battle_type: Any
    initialize_dataset: Callable[[str], Any]
    update_battle: Callable[[Any, str], Any]
    process_updates: Callable[[Any], Any]
    last_used_move: Callable[[str, str, int], Any]
    randbats_cache_sha256: str | None = None

    @classmethod
    def from_bound_runtime(cls, *, randbats_cache_sha256: str) -> "FoulPlayParserAPI":
        # An owner verifies the complete patched source/dependency/native tree
        # BEFORE this import. No configuration, think, or process is started here.
        from constants import BattleType
        from data.pkmn_sets import PKMN_SETS_CACHE_DIR, RandomBattleTeamDatasets
        from fp.battle import Battle, LastUsedMove
        from fp.battle_modifier import process_battle_updates, update_battle
        from config import FoulPlayConfig
        _require(FoulPlayConfig.pokemon_format == "gen3randombattle"
                 and FoulPlayConfig.search_time_ms == 1000
                 and FoulPlayConfig.parallelism == FoulPlayConfig.search_threads == 1
                 and getattr(FoulPlayConfig, "search_iterations", 0) == 0,
                 "bound production FP1000ms one-worker configuration required")
        _require(re.fullmatch(r"[0-9a-f]{64}", randbats_cache_sha256) is not None,
                 "frozen opponent dataset hash required")
        snapshot = (Path(PKMN_SETS_CACHE_DIR) / "gen3randombattle.json").read_bytes()
        _require(sha256(snapshot).hexdigest() == randbats_cache_sha256,
                 "opponent dataset snapshot drift")
        _require(isinstance(json.loads(snapshot), dict) and bool(json.loads(snapshot)),
                 "nonempty frozen opponent dataset required")

        def initialize_dataset(generation):
            _require(generation == "gen3", "only the frozen Gen3 dataset is admitted")
            # This is RandomBattleTeamDatasets.initialize's reset and transform
            # with its cache read replaced by the immutable bytes just hashed.
            # Never reopen a mutable cache or trigger get_sets_file's network.
            RandomBattleTeamDatasets.raw_pkmn_sets = json.loads(snapshot)
            RandomBattleTeamDatasets.pkmn_sets = {}
            RandomBattleTeamDatasets.pkmn_mode = generation
            RandomBattleTeamDatasets._initialize_pkmn_sets()

        return cls(Battle, BattleType.RANDOM_BATTLE, initialize_dataset,
                   update_battle, process_battle_updates, LastUsedMove, randbats_cache_sha256)


class SavedRootFoulPlayParser:
    """Feed the production opponent parser without running historical decisions."""
    def __init__(self, *, actor: str, battle_id: str, api: FoulPlayParserAPI):
        _require(actor in ("p1", "p2") and bool(battle_id), "actor and battle identity required")
        self.actor, self.api, self.battle_id = actor, api, battle_id
        self.battle = api.battle(battle_id)
        self.battle.user.name = actor
        self.battle.opponent.name = "p2" if actor == "p1" else "p1"
        self.battle.pokemon_format = "gen3randombattle"
        self.battle.generation = "gen3"
        self.battle.battle_type = api.random_battle_type
        api.initialize_dataset("gen3")
        self._started = self._initialized = self._selected = False
        self._pending: list[str] = []
        self._wire: list[str] = []
        self._next_rqid = 1
        self._round = 0
        self._requests = self._other_requests = self._past_choices = 0
        self._selected_requested = False

    def feed_boundary(self, row: dict[str, Any], *, selected: bool,
                      recorded_own_choice: str | None = None) -> None:
        _require(not self._selected and type(selected) is bool, "one terminal selected boundary only")
        _require(row["battle_id"] == self.battle_id and type(row["decision_round_index"]) is int
                 and row["decision_round_index"] == self._round, "source battle identity/order changed")
        roles = row["requested_players"]
        _require(isinstance(roles, list) and bool(roles) and len(set(roles)) == len(roles)
                 and set(roles) <= {"p1", "p2"}, "unique requested source roles required")
        _require((recorded_own_choice is None) if selected else
                 (recorded_own_choice is not None) == (self.actor in row["requested_players"]),
                 "historical own choice required only before the selected root")
        for raw in row["protocol_delta"]:
            projected = actor_protocol_line(raw, self.actor)
            if projected is None:
                self._other_requests += 1
                continue
            self._wire.append(projected)
            if projected == "|start":
                _require(not self._started, "duplicate battle start")
                self._started = self.battle.started = True
                continue
            if not self._started:
                continue
            if projected.startswith("|request|"):
                request = json.loads(projected[len("|request|"):])
                if "rqid" not in request:
                    request = dict(request, rqid=self._next_rqid)
                    self._next_rqid += 1
                _require(type(request["rqid"]) is int and request["rqid"] > 0, "request rqid invalid")
                self._wire[-1] = '|request|' + json.dumps(request, separators=(',', ':'))
                if not self._initialized:
                    _require(bool(request.get('active')) and not request.get('wait') and not request.get('teamPreview'),
                             'complete actionable opening request required')
                    self.battle.request_json = request
                    self.battle.user.initialize_first_turn_user_from_json(request)
                    self.battle.rqid = request["rqid"]
                    self.battle.msg_list = self._pending
                    self.api.process_updates(self.battle)
                    self._initialized = True
                else:
                    line = "|request|" + json.dumps(request, separators=(",", ":"))
                    self.api.update_battle(self.battle, "\n".join([*self._pending, line]))
                self._pending = []
                self._requests += 1
            elif not (not self._initialized and projected.startswith(f"|switch|{self.actor}")):
                self._pending.append(projected)
        _require(self._initialized, "first own request absent; no parser repair")
        latest = dict(self.battle.request_json)
        latest.pop("rqid", None)
        source_request = dict(row["requests"][self.actor])
        source_request.pop("rqid", None)
        _require(latest == source_request, "Foul Play own private request differs from saved boundary")
        if selected:
            self._selected = True
            self._selected_requested = self.actor in row["requested_players"]
        elif recorded_own_choice is not None:
            # Mirror async_pick_move's own bookkeeping on a COPY. This never
            # consults the current selected root's recorded action or a model.
            candidate = self._decision_copy()
            words = recorded_own_choice.split()
            _require(len(words) == 2 and words[0] in ("move", "switch") and words[1].isdigit(),
                     "canonical recorded own choice required")
            index = int(words[1])
            if words[0] == "move":
                _require(1 <= index <= len(candidate.user.active.moves), "recorded move slot missing")
                choice = candidate.user.active.moves[index - 1].name
            else:
                targets = [pokemon for pokemon in candidate.user.reserve if pokemon.index == index]
                _require(len(targets) == 1, "recorded switch slot missing or ambiguous")
                choice = "switch " + targets[0].name
            self.battle.user.last_selected_move = self.api.last_used_move(
                self.battle.user.active.name, choice, self.battle.turn)
            self._past_choices += 1
        self._round += 1

    def _decision_copy(self):
        candidate = deepcopy(self.battle)
        _require(not candidate.team_preview, "Gen3 Random Battle team preview forbidden")
        candidate.user.update_from_request_json(candidate.request_json)
        return candidate

    def selected_decision_copy(self):
        _require(self._selected and self._selected_requested, "opponent has no requested selected-root decision")
        _require(not self._pending, "unprocessed selected-root public lines")
        return self._decision_copy()

    def proof(self) -> dict[str, Any]:
        _require(self._selected, "selected root not reached")
        return {"schema": "pokezero.A4-saved-root-FP-parser.v1", "actor": self.actor,
                "format": "gen3randombattle", "showdown_source": SHOWDOWN_SOURCE,
                "actor_wire_sha256": sha256(json.dumps(self._wire, separators=(",", ":")).encode()).hexdigest(),
                "own_requests": self._requests, "foreign_requests_excluded": self._other_requests,
                "immutable_randbats_cache_sha256": self.api.randbats_cache_sha256,
                "historical_own_choices_mapped_without_execution": self._past_choices,
                "selected_root_action_consumed": False, "selected_opponent_requested": self._selected_requested,
                "native_search_queries": 0, "model_forwards": 0, "simulator_steps": 0, "new_games": 0,
                "opponent_parser_auxiliary_native_damage_inference": "production parser may call this; no MCTS/choice here"}
