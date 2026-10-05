"""Construct the sampled opponent's information state without seat swapping.

Only a public protocol prefix and that seat's sampled private request enter
this boundary. No PolicyContext, subject request, simulator snapshot, or full
two-seat engine state is accepted. Use the same canonical parser, belief
source, and observation writer as the incumbent, not EngineEnv's smoke-grade
reveals-only belief approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Any, Mapping, Sequence

from .actions import ACTION_COUNT
from .belief import PokemonSetSource, PublicBattleBeliefEngine
from .dex import ShowdownDex
from .local_showdown import PublicBattleMaterializationState, _public_materialization_payload
from .observation import (
    ObservationFeatureMasks, ObservationSpec, PokeZeroObservationV0,
    FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
    TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
)
from .showdown import (
    PlayerRelativeBattleState, _observation_metadata, normalize_for_player,
    observation_from_player_state, parse_showdown_replay,
)


class PolicyOpponentViewError(ValueError):
    """An observation cannot be certified; never replace it with uniform play."""


_HP_EVENTS = {"switch", "drag", "replace", "-damage", "-heal", "-sethp"}
_HP = re.compile(r"^(\d+)/(\d+)(.*)$")


def public_policy_lines(
    lines: Sequence[str], *, hp_visibility: Mapping[str, str],
) -> tuple[str, ...]:
    """Remove secret split branches and project exact HP into shared /100 HP.

    The current random-battle contract uses HP Percentage Mod. Its rule is
    ceil(100*hp/maxhp), except a wounded Pokemon must never display 100.
    Existing percentage lines are preserved; exact 100-HP Pokemon require
    explicit provenance and cannot be guessed from their denominator.
    """
    result: list[str] = []
    index = 0
    while index < len(lines):
        # Preserve protocol spelling: the canonical observer intentionally does
        # not grant chronology evidence to a whitespace-decorated |upkeep.
        line = str(lines[index])
        index += 1
        parts = line.split("|")
        event = parts[1] if len(parts) > 1 else ""
        split_shared = False
        if event == "split":
            if len(parts) != 3 or parts[2] not in {"p1", "p2"} or index + 1 >= len(lines):
                raise PolicyOpponentViewError("incomplete or malformed public split")
            # Both live private alternatives are irrelevant: our own request
            # must come from the REGISTERED SAMPLED world, not live truth.
            line = str(lines[index + 1])
            index += 2
            parts = line.split("|")
            event = parts[1] if len(parts) > 1 else ""
            split_shared = True
        if event in {"request", "split"} or (line and not line.startswith("|")):
            raise PolicyOpponentViewError("private/nonpublic input at public-policy boundary")
        if event in _HP_EVENTS:
            # switch/drag/replace have an intervening details field; -sethp
            # may update two seats in one line.
            hp_fields = [4] if event in {"switch", "drag", "replace"} else [3]
            if event == "-sethp" and len(parts) > 4:
                # Pain Split is normally TWO single-target lines, with public
                # [from]/[silent] qualifiers. Those are not a second HP owner.
                qualifiers_start = 4
                if not parts[4].startswith("["):
                    if len(parts) <= 5:
                        raise PolicyOpponentViewError("missing second public HP condition")
                    hp_fields.append(5)
                    qualifiers_start = 6
                if any(not field.startswith("[") for field in parts[qualifiers_start:]):
                    raise PolicyOpponentViewError("malformed public HP qualifiers")
            for hp_index in hp_fields:
                if hp_index >= len(parts):
                    raise PolicyOpponentViewError("missing public HP condition")
                ident_index = 2 if hp_index in {3, 4} else hp_index - 1
                slot = parts[ident_index][:2]
                if slot not in {"p1", "p2"}:
                    raise PolicyOpponentViewError("unidentified public HP owner")
                match = _HP.fullmatch(parts[hp_index])
                if match is None:
                    if parts[hp_index] != "0 fnt":
                        raise PolicyOpponentViewError("unsupported public HP condition")
                    continue
                hp, maxhp = int(match[1]), int(match[2])
                if maxhp <= 0 or hp > maxhp:
                    raise PolicyOpponentViewError("invalid public HP range")
                visibility = "percentage" if split_shared else hp_visibility.get(slot)
                if visibility == "percentage":
                    if maxhp != 100:
                        raise PolicyOpponentViewError("public policy requires /100 percentage HP")
                elif visibility == "exact":
                    percent = (100 * hp + maxhp - 1) // maxhp
                    if percent == 100 and hp < maxhp:
                        percent = 99
                    parts[hp_index] = f"{percent}/100{match[3]}"
                else:
                    raise PolicyOpponentViewError("missing HP visibility provenance")
            line = "|".join(parts)
        if line:
            result.append(line)
    return tuple(result)


@dataclass(frozen=True)
class PolicyOpponentView:
    state: PlayerRelativeBattleState
    materialization: PublicBattleMaterializationState
    public_lines: tuple[str, ...]
    spec: ObservationSpec
    feature_masks: ObservationFeatureMasks
    battle_seed: int
    native_action_indices: tuple[int | None, ...] | None = None

    def row_inputs(self, *, dex: ShowdownDex) -> dict[str, Any]:
        metadata = _observation_metadata(self.state, dex=dex, schema_version=self.spec.schema_version)
        metadata["belief_view"] = self.state.belief_view.to_overlay_payload()
        return {
            "battle_id": self.state.battle_id,
            "battle_seed": self.battle_seed,
            "format_id": self.materialization.observation_format_id,
            "player_id": self.state.player_id,
            "observation_schema_version": self.spec.schema_version,
            "observation_metadata": metadata,
            "public_materialization": _public_materialization_payload(self.materialization),
        }

    def observation(self, *, category_vocab: Any, dex: ShowdownDex) -> PokeZeroObservationV0:
        observation = observation_from_player_state(
            self.state, category_vocab=category_vocab, spec=self.spec,
            dex=dex, feature_masks=self.feature_masks,
        )
        return replace(observation, metadata={
            **dict(observation.metadata), "belief_view": self.state.belief_view.to_overlay_payload(),
        })


def build_policy_opponent_view(
    *, public_lines: Sequence[str], hp_visibility: Mapping[str, str],
    sampled_self_request: Mapping[str, Any], opponent_slot: str,
    battle_id: str, battle_seed: int, format_id: str,
    set_source: PokemonSetSource, spec: ObservationSpec,
    feature_masks: ObservationFeatureMasks,
    sampled_self_move_states: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> PolicyOpponentView:
    """Rebuild at root OR child from a complete prefix and sampled own request.

    Each caller must construct the private request from the opponent's sampled
    side only. Appending realized public branch lines and supplying its updated
    request evolves this view, including newly revealed subject identities.
    This is a correctness reference; native caching/batching is a later step.
    """
    if opponent_slot not in {"p1", "p2"} or set_source is None:
        raise PolicyOpponentViewError("opponent slot and canonical set source are required")
    # Damage conclusions mutate belief when narrowing is enabled and occupy
    # encoded columns in legacy schemas. V4 explicitly retired BOTH the
    # history region and pinned residual/investment columns: with narrowing
    # disabled those provenance flags have no input consumer. Keep the flags
    # unchanged, and refuse whenever a real observer-dependent surface exists.
    feature_pack = spec.schema_version in FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS
    if feature_masks.investment_belief_narrowing or (feature_masks.tier2_investment and not feature_pack):
        raise PolicyOpponentViewError("investment observer is not implemented for policy opponent")
    if feature_masks.tier2_residuals and not feature_pack:
        raise PolicyOpponentViewError("residual observer is not implemented for policy opponent history")
    clean_lines = public_policy_lines(public_lines, hp_visibility=hp_visibility)
    if "|start" not in clean_lines:
        raise PolicyOpponentViewError("public policy requires a complete battle-start prefix")
    # JSON cloning prevents mutation by a provider/client after certification.
    try:
        request = json.loads(json.dumps(sampled_self_request, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise PolicyOpponentViewError("sampled own request is not finite JSON") from error
    if not isinstance(request, dict):
        raise PolicyOpponentViewError("sampled own request must be an object")
    side = request.get("side")
    if not isinstance(side, dict) or side.get("id") != opponent_slot:
        raise PolicyOpponentViewError("sampled own request belongs to a different seat")
    team = side.get("pokemon")
    if not isinstance(team, list) or not team:
        raise PolicyOpponentViewError("sampled own request has no party")
    if any(not isinstance(mon, dict) or not str(mon.get("ident", "")).startswith(f"{opponent_slot}:") for mon in team):
        raise PolicyOpponentViewError("sampled party identity belongs to a different seat")
    move_states = _sampled_move_states(request, sampled_self_move_states)
    replay = parse_showdown_replay(
        clean_lines, battle_id=battle_id, complete_prefix=True,
        hp_visibility={"p1": "percentage", "p2": "percentage"},
    )
    if replay.winner is not None or any(line.split("|")[1] == "tie" for line in clean_lines if "|" in line):
        raise PolicyOpponentViewError("public policy cannot infer at a terminal root")
    belief = PublicBattleBeliefEngine.from_events(
        replay.public_events, format_id=format_id, set_source=set_source,
        item_belief_narrowing=feature_masks.item_belief_narrowing,
    )
    state = normalize_for_player(
        replace(replay, requests={opponent_slot: request}), player_id=opponent_slot,
        configured_showdown_slot=opponent_slot, format_id=format_id, belief_engine=belief,
        include_turn_merged=spec.schema_version in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
    )
    materialization = PublicBattleMaterializationState(
        player_id=opponent_slot, format_id=format_id, observation_format_id=format_id,
        replay=replace(replay, requests={}), belief_engine=belief,
        self_request=request, self_initial_request=request,
        self_move_states=move_states,
    )
    return PolicyOpponentView(state, materialization, clean_lines, spec, feature_masks, battle_seed)


def _sampled_move_states(
    request: Mapping[str, Any], supplied: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Clone complete native own-party PP, including Pokemon on the bench.

    The older request-only reference remains available for observer tests, but
    native integration must use the bundle boundary below. An active request
    alone cannot certify PP after a switch or a forced-replacement boundary.
    """
    if supplied is None:
        return {}
    try:
        states = json.loads(json.dumps(supplied, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise PolicyOpponentViewError("sampled move states are not finite JSON") from error
    team = request["side"]["pokemon"]
    identities = [mon["ident"].split(":", 1)[-1].strip().casefold() for mon in team]
    if len(set(identities)) != len(identities) or not isinstance(states, dict) or set(states) != set(identities):
        raise PolicyOpponentViewError("sampled move states do not cover the exact own party")

    def move_id(name: str) -> str:
        normalized = "".join(c for c in name.casefold() if c.isalnum())
        return "hiddenpower" if normalized.startswith("hiddenpower") else normalized

    result = {}
    for identity, mon in zip(identities, team):
        rows = states[identity]
        known = mon.get("moves")
        if not isinstance(rows, list) or not isinstance(known, list) or not all(isinstance(name, str) for name in known):
            raise PolicyOpponentViewError("sampled own moves must be complete lists")
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
                or not isinstance(row.get("move"), str) or not row["move"]
                or type(row.get("pp")) is not int or type(row.get("maxpp")) is not int
                or not 0 <= row["pp"] <= row["maxpp"] or row["maxpp"] <= 0
                or type(row.get("disabled")) is not bool):
                raise PolicyOpponentViewError("invalid sampled own move PP state")
        ids = [move_id(row["id"]) for row in rows]
        if len(set(ids)) != len(ids) or ids != [move_id(name) for name in known]:
            raise PolicyOpponentViewError("sampled move-state identity differs from own request")
        result[identity] = tuple(rows)
    return result


def build_policy_opponent_view_from_native_bundle(
    *, native_request_bundle: Mapping[str, Any], **view_arguments: Any,
) -> PolicyOpponentView:
    """Certify the native side-only request, full PP and exact action surface.

    This is the production integration boundary, not a constructor accepting a
    full engine state. Native callers supply only the sampled acting side's
    request bundle and a public transcript.
    """
    if not isinstance(native_request_bundle, Mapping) or set(native_request_bundle) != {
        "request", "self_move_states", "native_action_indices",
    }:
        raise PolicyOpponentViewError("incomplete native private-request bundle")
    if not isinstance(native_request_bundle["self_move_states"], Mapping):
        raise PolicyOpponentViewError("native private-request bundle lacks complete own PP")
    raw_indices = native_request_bundle["native_action_indices"]
    if not isinstance(raw_indices, (list, tuple)) or not raw_indices:
        raise PolicyOpponentViewError("empty native private-request action map")
    indices = tuple(raw_indices)
    if any(index is not None and (type(index) is not int or not 0 <= index < ACTION_COUNT) for index in indices):
        raise PolicyOpponentViewError("invalid native private-request action map")
    mapped = [index for index in indices if index is not None]
    if len(set(mapped)) != len(mapped) or (None in indices and indices != (None,)):
        raise PolicyOpponentViewError("ambiguous native private-request action map")
    view = build_policy_opponent_view(
        sampled_self_request=native_request_bundle["request"],
        sampled_self_move_states=native_request_bundle["self_move_states"],
        **view_arguments,
    )
    legal = {index for index, enabled in enumerate(view.state.legal_action_mask) if enabled}
    if legal != set(mapped):
        raise PolicyOpponentViewError("native and canonical private-request legal surfaces differ")
    return replace(view, native_action_indices=indices)


def advance_policy_opponent_view(
    parent: PolicyOpponentView, *, public_branch_lines: Sequence[str],
    branch_hp_visibility: Mapping[str, str], sampled_self_request: Mapping[str, Any],
    set_source: PokemonSetSource,
    sampled_self_move_states: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> PolicyOpponentView:
    """Rebuild a child without confusing root percentage HP with exact branch HP.

    Native engine branch events can carry exact HP even when the root transcript
    was already percentage-only. Project the new suffix independently before
    replaying the combined, entirely public prefix. No parent state is mutated;
    siblings may have different realized public events and sampled requests.
    """
    suffix = public_policy_lines(public_branch_lines, hp_visibility=branch_hp_visibility)
    return build_policy_opponent_view(
        public_lines=(*parent.public_lines, *suffix),
        hp_visibility={"p1": "percentage", "p2": "percentage"},
        sampled_self_request=sampled_self_request, opponent_slot=parent.state.player_id,
        battle_id=parent.state.battle_id, battle_seed=parent.battle_seed,
        format_id=parent.materialization.observation_format_id, set_source=set_source,
        spec=parent.spec, feature_masks=parent.feature_masks,
        sampled_self_move_states=sampled_self_move_states,
    )
