"""Direction 2 of the oracle: every SEARCHED world's public projection must
match the observed log.

Direction 1 (:mod:`pokezero.truth_differential`) asks *does the true world
construct*. It kills guards that refuse too much, which is the loud failure
direction. It says nothing about the other one. Report 4 section 4.5 states the
asymmetry plainly: relaxing a guard flips the failure from *refusing too much*,
which is loud, to *answering wrongly*, which is silent, and **a silently wrong
searched world is worse than the refusal that was removed to get it**.

This module is the instrument for the silent direction, and it is one sentence:

    project every world the search actually used back into the public protocol
    facts the opponent can already see, and assert the projection equals what
    was observed.

A mismatch means search spent budget on a world the OPPONENT could prove is
impossible. Per PLAN section 3 a guard change is correct iff BOTH directions
hold on the census block.

Two comparators, and they are NOT the same claim
------------------------------------------------
``state_projection_mismatches`` (the headline). Runs on EVERY world the sampler
produced, at EVERY decision, and compares the CONSTRUCTED ENGINE STATE against
the observed public record. Cheap: no search, no extra sampling -- the worlds
arrive through :class:`WorldObserver`, which ``EngineMctsPolicy`` calls with
exactly the worlds it is about to hand to the search.

``render_projection_mismatch`` (the renderer's own projection). The state
comparator reads the world at the decision BOUNDARY and is therefore blind to
everything the renderer does downstream of it -- and the renderer is where
#1211 lives. This comparator renders the branch set for the joint action that
was ACTUALLY played, through the shipped attribution-safe mapper
(``pokezero_search.branch_events``), and requires the transition Showdown
actually took to lie in the rendered support. It costs an engine
re-enumeration per decision, so it is opt-in and its coverage is published as a
floor.

Why this is not a tautology
---------------------------
The world is constructed FROM ``_public_materialization_payload``, so comparing
a world against that payload would largely restate the constructor to itself.
Every observed value here is therefore read from a source UPSTREAM of the
payload:

* the raw protocol lines (``replay.public_events[*].raw_line``), folded by
  ``engine_fidelity.showdown_turn_features`` -- the same fold the engine
  fidelity differential has used against the engine since PR #727;
* the acting seat's RAW Showdown request (``self_request``) -- ``active[0].moves[]``
  carries the id, pp and disabled flag of every move the seat may pick, and
  ``side.pokemon[]`` carries its own party verbatim;
* the parser's public opponent record (``replay.public_revealed``), which is the
  definition of "what the protocol revealed" and has no upstream.

That matters for the four guards relaxed on 2026-08-09, three of which are
visible to the state comparator precisely because the request states the fact
the guard now infers:

* #1210 (Transform PP overlay) -> ``self_move_pp``: the request publishes the
  copy's live PP every round, so a wrong overlay is a numeric disagreement.
* #1212 (a third Encore resolution source) -> ``self_move_disabled``: under
  Encore the request disables every move but the encored one, so resolving the
  WRONG move produces a different disabled set. This is the axis that makes the
  relaxation falsifiable at all -- ``encore_move_unknown`` used to refuse.
* #1209 (toxic stage: demand a weaker proof) -> ``toxic_count``: the engine's
  ``side_conditions.toxic_count`` must equal the parser's public
  ``replay.toxic_stage`` whenever that stage is known.
* #1211 (absorb guard narrowed to HP headroom) is NOT visible here. It is a
  renderer branch, and it is why ``render_projection_mismatch`` exists.

Units, kept apart (plan 4 reporting rules, report 4 section 9.2)
----------------------------------------------------------------
``projection_mismatched_worlds`` counts WORLDS. ``projection_mismatched_decisions``
counts DECISIONS. ``axis_worlds`` counts WORLDS per axis. ``render_mismatched_boundaries``
counts BOUNDARIES. They are never co-ranked in one table.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from .engine_fidelity import (
    _STATUS_TO_ENGINE,
    _engine_side_conditions,
    showdown_turn_features,
)

__all__ = [
    "AXES",
    "ProjectionMismatch",
    "WorldProjectionRecord",
    "DecisionProjectionRecord",
    "WorldObserver",
    "PublicProjectionProbe",
    "aggregate_projection_records",
    "observed_public_view",
    "render_projection_mismatch",
    "render_self_consistency_mismatches",
    "state_projection_mismatches",
]


# --- taxonomy -----------------------------------------------------------------

#: Every axis this oracle can fire on. A closed set, so a new axis cannot appear
#: in a census without appearing here first -- and so `test_axes_are_closed`
#: fails when one is added without a pin.
AXES = (
    # -- from the raw protocol fold (independent of the constructor's payload) --
    "active_hp",
    "active_status",
    "weather",
    "side_conditions",
    # -- from the acting seat's RAW request --
    "self_move_set",
    "self_move_pp",
    "self_move_disabled",
    "self_party_species",
    "self_party_hp",
    "self_item",
    "self_ability",
    # -- from the parser's public opponent record --
    "opponent_revealed_species",
    "opponent_revealed_moves",
    "opponent_revealed_item",
    "opponent_revealed_ability",
    # -- public counters --
    "boosts",
    "toxic_count",
    # -- the renderer's projection (a different comparator; see the docstring) --
    "render_unmatched_transition",
    "render_no_usable_branch",
    "render_post_state_disagreement",
)

#: Axes whose observed side comes from the acting seat's own request. Only
#: meaningful for `context.player_id`; the opponent's request is not ours to see.
_SELF_AXES = frozenset(
    {
        "self_move_set",
        "self_move_pp",
        "self_move_disabled",
        "self_party_species",
        "self_party_hp",
        "self_item",
        "self_ability",
    }
)

_DETAIL_LIMIT = 160


def _bounded(text: str) -> str:
    text = str(text)
    return text if len(text) <= _DETAIL_LIMIT else text[: _DETAIL_LIMIT - 3] + "..."


def _norm(value: Any) -> str:
    """The identifier normalisation the parser and the engine already share."""

    from .showdown import _normalize_identifier  # noqa: PLC0415 - import-light

    return _normalize_identifier(value)


@dataclass(frozen=True)
class ProjectionMismatch:
    """One public fact a searched world contradicts.

    ``axis`` is the closed-set bucket; ``predicate`` is the queue key (axis plus
    the discriminator that makes two occurrences the same defect); ``detail``
    carries the two values, bounded.
    """

    axis: str
    slot: str
    predicate: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "slot": self.slot,
            "predicate": self.predicate,
            "detail": self.detail,
        }


# --- the observed side --------------------------------------------------------


@dataclass(frozen=True)
class ObservedPublicView:
    """What the public protocol has actually shown, at one decision boundary.

    Every field is read from a source UPSTREAM of the world constructor's input
    payload. ``None`` and ``-1`` mean *not publicly determined*; an axis never
    fires on an undetermined value, because "the protocol did not say" is not a
    disagreement.
    """

    #: Acting seat.
    slot: str
    opponent_slot: str
    #: `engine_fidelity.TurnFeatures` over every raw public line so far.
    turn_features: Any
    #: The acting seat's raw request, verbatim.
    self_request: Mapping[str, Any]
    #: `{slot: {stat: stage}}` from the parser's public boost tracker.
    boosts: Mapping[str, Mapping[str, int]]
    #: `{slot: stage}` public Toxic chronology; missing/None = not determined.
    toxic_stage: Mapping[str, int]
    #: The parser's public opponent record.
    opponent_revealed: Sequence[Any]


def observed_public_view(context: Any) -> ObservedPublicView | None:
    """Read the observed public record for this decision, or None if unavailable."""

    state = getattr(context, "public_materialization_state", None)
    replay = getattr(state, "replay", None)
    if state is None or replay is None:
        return None
    slot = str(getattr(context, "player_id", "p1"))
    opponent = "p2" if slot == "p1" else "p1"
    lines = [event.raw_line for event in getattr(replay, "public_events", ())]
    return ObservedPublicView(
        slot=slot,
        opponent_slot=opponent,
        turn_features=_fold_public_lines(lines),
        self_request=dict(getattr(state, "self_request", {}) or {}),
        boosts={
            key: dict(value) for key, value in (getattr(replay, "boosts", {}) or {}).items()
        },
        toxic_stage=dict(getattr(replay, "toxic_stage", {}) or {}),
        opponent_revealed=tuple(
            (getattr(replay, "public_revealed", {}) or {}).get(opponent, ())
        ),
    )


def _fold_public_lines(lines: Sequence[str]) -> Any:
    """Fold raw protocol lines into comparable features.

    `showdown_turn_features` takes an object with a `protocol_lines` attribute
    (its production caller is a one-turn fixture result). Reused verbatim rather
    than reimplemented: it is the fold the engine fidelity differential has
    matched engine states against since PR #727, so this oracle and that
    differential cannot drift on what "the protocol said" means.
    """

    import types  # noqa: PLC0415

    return showdown_turn_features(types.SimpleNamespace(protocol_lines=tuple(lines)))


#: Protocol tags that state an absolute HP for a slot. `|switch|` and `|drag|`
#: carry it in field 4; the rest in field 3.
_HP_TAGS_FIELD3 = ("-damage", "-heal", "-sethp")
_HP_TAGS_FIELD4 = ("switch", "drag", "replace")


@dataclass(frozen=True)
class StepProjection:
    """One STEP's public effect, per side, with unstated values resolved.

    Deliberately NOT ``showdown_turn_features``. That fold only records an HP
    from ``-damage``/``-heal`` for a slot it has already seen a ``|switch|`` for,
    which is correct over a whole log (a log opens with switches) and silently
    blinding over a single step (a step usually has none). Worse, it returns -1
    for "not stated", and a comparator that skips -1 cannot see the one thing the
    render arm exists to see: **a heal that the log shows and the render omits.**
    Measured on the unit fixture -- the #1211 shape passed the comparator until
    an unstated side was resolved to "unchanged" instead of "unknown".
    """

    hp: Mapping[str, int]
    status: Mapping[str, str]
    fainted: frozenset[str]
    side_conditions: Mapping[str, tuple[str, ...]]


def fold_step_lines(lines: Sequence[str], pre: Any) -> StepProjection:
    """Fold ONE step's lines. An unstated HP resolves to the pre-state HP."""

    from .engine_fidelity import _SIDESTART_IDS  # noqa: PLC0415
    from .engine_fidelity import _parse_condition  # noqa: PLC0415
    from .showdown import _normalize_identifier  # noqa: PLC0415

    hp: dict[str, int] = {"p1": int(pre.p1_hp), "p2": int(pre.p2_hp)}
    status: dict[str, str] = {"p1": pre.p1_status, "p2": pre.p2_status}
    fainted: set[str] = set()
    conditions: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
    for line in lines:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1]
        side = parts[2].split(":", 1)[0].strip()[:2]
        if side not in ("p1", "p2"):
            side = ""
        if tag in _HP_TAGS_FIELD3 and side and len(parts) >= 4:
            value, token = _parse_condition(parts[3])
            hp[side] = value
            status[side] = _STATUS_TO_ENGINE.get(token, status[side])
        elif tag in _HP_TAGS_FIELD4 and side and len(parts) >= 5:
            value, token = _parse_condition(parts[4])
            hp[side] = value
            status[side] = _STATUS_TO_ENGINE.get(token, "NONE")
            fainted.discard(side)
        elif tag == "-status" and side and len(parts) >= 4:
            status[side] = _STATUS_TO_ENGINE.get(parts[3].strip(), status[side])
        elif tag == "-curestatus" and side:
            status[side] = "NONE"
        elif tag == "faint" and side:
            fainted.add(side)
            hp[side] = 0
        elif tag == "-sidestart" and side and len(parts) >= 4:
            key = _SIDESTART_IDS.get(_normalize_identifier(parts[3].split(":")[-1]))
            if key:
                conditions[side][key] = conditions[side].get(key, 0) + 1
        elif tag == "-sideend" and side and len(parts) >= 4:
            key = _SIDESTART_IDS.get(_normalize_identifier(parts[3].split(":")[-1]))
            if key:
                conditions[side][key] = 0
    return StepProjection(
        hp=hp,
        status=status,
        fainted=frozenset(fainted),
        side_conditions={
            slot: tuple(sorted(k for k, v in values.items() if v))
            for slot, values in conditions.items()
        },
    )


# --- the projected side -------------------------------------------------------


def _sides_by_slot(state: Any, slot_sides: Mapping[str, str]) -> dict[str, Any]:
    sides = {"side_one": state.side_one, "side_two": state.side_two}
    return {slot: sides[slot_sides[slot]] for slot in ("p1", "p2")}


def _engine_turn_features(state: Any, slot_sides: Mapping[str, str]) -> Any:
    """The engine state's own projection into `TurnFeatures`.

    Field-for-field `engine_transition_differential.engine_features_by_slot`.
    Duplicated here rather than imported because that lives in a script, and a
    shipped module importing a script would make the script's import-time side
    effects (it mutates `sys.path`) part of the library's contract.
    """

    from .engine_fidelity import TurnFeatures  # noqa: PLC0415

    sides = _sides_by_slot(state, slot_sides)
    actives = {
        slot: side.pokemon[int(str(side.active_index))] for slot, side in sides.items()
    }
    return TurnFeatures(
        p1_hp=int(actives["p1"].hp),
        p2_hp=int(actives["p2"].hp),
        p1_status=str(actives["p1"].status).upper(),
        p2_status=str(actives["p2"].status).upper(),
        fainted=frozenset(slot for slot, mon in actives.items() if mon.hp <= 0),
        weather=str(state.weather).upper(),
        side_conditions={slot: _engine_side_conditions(side) for slot, side in sides.items()},
    )


def _live_party(side: Any) -> list[Any]:
    """The side's real party. Engine sides are padded to six with ``NONE`` slots."""

    return [mon for mon in side.pokemon if str(mon.id).lower() != "none"]


def _identity_species(mon: Any) -> str:
    """The species the PROTOCOL calls this mon, not the one the engine is running.

    An already-Transformed active carries the DONOR's species, stats and moves in
    `PokemonSpec.id`, with its own form in `pre_transform`. The protocol never
    renames it: a Ditto that copied Dugtrio is still `p2a: Ditto` in every line
    and still `Ditto` in the request's own `details`. Comparing `mon.id` therefore
    reported the Transform mechanic as a projection defect -- measured on the
    first six-game shard, `self_party_species` and `opponent_revealed_species`
    firing on 8 worlds each with `ditto` missing and `dugtrio` extra.
    """

    base = _pre_transform_species(mon)
    return base if base else _norm(mon.id)


def _pre_transform_species(mon: Any) -> str:
    """The transformer's own species, or "".

    ``Pokemon.pre_transform`` reads back as the engine's SERIALIZED form
    (``"ditto;1;1;1;1;1;transform:32;none:0;..."``), not as a nested object --
    checked against the binding rather than assumed, because the obvious
    ``pre.id`` returns None on it and would have made this whole exclusion a
    silent no-op.
    """

    raw = getattr(mon, "pre_transform", None)
    if not raw:
        return ""
    head = str(raw).split(";", 1)[0]
    token = _norm(head)
    return "" if token in ("", "none") else token


def _is_transformed(mon: Any) -> bool:
    return bool(_pre_transform_species(mon))


# --- the comparator, one function per axis ------------------------------------
#
# ONE FUNCTION PER AXIS is a mutation-testing requirement, not a style choice.
# Report 4 section 4.4: a battery can lie by never applying a mutant, and an
# inline `if` inside a 200-line comparator is exactly the shape whose deletion
# survives. Independent review's M20 on direction 1 deleted an inline gate and
# survived the whole suite; the remedy there was to extract it into a named
# function, and the same remedy is applied here from the start.


def _axis_active_hp(observed: ObservedPublicView, engine: Any) -> list[ProjectionMismatch]:
    out: list[ProjectionMismatch] = []
    for slot, obs, eng in (
        ("p1", observed.turn_features.p1_hp, engine.p1_hp),
        ("p2", observed.turn_features.p2_hp, engine.p2_hp),
    ):
        # -1 is the fold's "the protocol never stated this side's HP". Silence,
        # not a mismatch: an axis that fires on an undetermined value measures
        # the instrument.
        if obs < 0 or obs == eng:
            continue
        out.append(
            ProjectionMismatch(
                axis="active_hp",
                slot=slot,
                predicate="active_hp",
                detail=_bounded(f"observed {obs} != world {eng}"),
            )
        )
    return out


def _currently_fainted(features: Any) -> frozenset[str]:
    """Which side's ACTIVE is down right now, read from HP.

    NOT ``TurnFeatures.fainted``. That set is populated by every ``|faint|`` line
    and never cleared, which is right for the one-turn fold it was written for and
    catastrophic over a whole-log fold: the first faint of the game marks the side
    permanently, and every later decision then compares a stale flag. Measured on
    the first smoke game before this was fixed -- 272 of 312 worlds "mismatched",
    all of them this. A comparator that fires on 87% of worlds is reporting itself.
    """

    return frozenset(
        slot
        for slot, hp in (("p1", features.p1_hp), ("p2", features.p2_hp))
        if hp == 0
    )


def _axis_active_status(observed: ObservedPublicView, engine: Any) -> list[ProjectionMismatch]:
    out: list[ProjectionMismatch] = []
    down = _currently_fainted(observed.turn_features)
    for slot, obs, eng in (
        ("p1", observed.turn_features.p1_status, engine.p1_status),
        ("p2", observed.turn_features.p2_status, engine.p2_status),
    ):
        # A fainted mon's `0 fnt` condition string carries no status while the
        # engine keeps it -- the known faint conflation the fidelity harness
        # documents. Not a projection defect.
        if slot in down or slot in engine.fainted:
            continue
        if obs.startswith("?") or obs == eng:
            continue
        out.append(
            ProjectionMismatch(
                axis="active_status",
                slot=slot,
                predicate=f"active_status:{obs.lower()}_vs_{eng.lower()}",
                detail=_bounded(f"observed {obs} != world {eng}"),
            )
        )
    return out


# NO `fainted` AXIS. It would restate `active_hp`: a side whose active HP the
# protocol states as 0 and whose world HP is not 0 already fires there, and a
# separate faint axis would double-count the same disagreement into two
# predicates and two rows of one queue.


def _axis_weather(observed: ObservedPublicView, engine: Any) -> list[ProjectionMismatch]:
    obs = observed.turn_features.weather
    if obs.startswith("?") or obs == engine.weather:
        return []
    return [
        ProjectionMismatch(
            axis="weather",
            slot="both",
            predicate=f"weather:{obs.lower()}_vs_{engine.weather.lower()}",
            detail=_bounded(f"observed {obs} != world {engine.weather}"),
        )
    ]


def _axis_side_conditions(
    observed: ObservedPublicView, engine: Any
) -> list[ProjectionMismatch]:
    obs = observed.turn_features.presence()
    eng = engine.presence()
    if obs == eng:
        return []
    return [
        ProjectionMismatch(
            axis="side_conditions",
            slot="both",
            predicate="side_conditions",
            detail=_bounded(f"observed {obs} != world {eng}"),
        )
    ]


def _request_active_moves(request: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    active = request.get("active")
    if not isinstance(active, Sequence) or not active:
        return None
    first = active[0]
    if not isinstance(first, Mapping):
        return None
    moves = first.get("moves")
    if not isinstance(moves, Sequence):
        return None
    return [row for row in moves if isinstance(row, Mapping)]


def _axis_self_moves(
    observed: ObservedPublicView, side: Any
) -> list[ProjectionMismatch]:
    """The request's own move rows vs the world's active moveset.

    THE REQUEST IS NOT A BELIEF. ``active[0].moves[]`` is what Showdown told this
    seat it may pick, with the id, the remaining PP and the disabled flag of each.
    A searched world whose active carries a different move id, a different PP or a
    different disabled set is a world this seat can already see is wrong.

    Three axes and not one, because they fail for different reasons and a merged
    key would rank them together: a wrong move id is a construction/sampling
    defect, a wrong PP is an overlay defect (#1210's shape), a wrong disabled set
    is a lock-resolution defect (#1212's shape).

    A ``Struggle``-only request (no usable move) and a force-switch request carry
    no comparable move rows and are skipped -- the world's moveset is then not
    what the request is describing.
    """

    rows = _request_active_moves(observed.self_request)
    if not rows:
        return []
    if observed.self_request.get("forceSwitch"):
        return []
    active = side.pokemon[int(str(side.active_index))]
    world_moves = [move for move in active.moves if str(move.id).lower() != "none"]
    observed_ids = [_norm(row.get("id") or row.get("move")) for row in rows]
    world_ids = [_norm(move.id) for move in world_moves]
    if observed_ids == ["struggle"]:
        return []

    out: list[ProjectionMismatch] = []
    # Hidden Power is the one id the engine deliberately spells differently: the
    # request says `hiddenpower` and the engine carries the typed+BP id
    # (`hiddenpowerice60`). Compare on the request's own prefix rule rather than
    # excluding the move, which would blind the PP axis on it too.
    def _same_move(observed_id: str, world_id: str) -> bool:
        if observed_id == world_id:
            return True
        return observed_id == "hiddenpower" and world_id.startswith("hiddenpower")

    if len(observed_ids) != len(world_ids) or not all(
        _same_move(a, b) for a, b in zip(observed_ids, world_ids)
    ):
        out.append(
            ProjectionMismatch(
                axis="self_move_set",
                slot=observed.slot,
                predicate="self_move_set",
                detail=_bounded(f"request {observed_ids} != world {world_ids}"),
            )
        )
        # PP and disabled are per-INDEX comparisons; without an aligned moveset
        # they would report noise attributed to the wrong axis.
        return out

    for index, row in enumerate(rows):
        pp = row.get("pp")
        if isinstance(pp, int) and int(world_moves[index].pp) != pp:
            out.append(
                ProjectionMismatch(
                    axis="self_move_pp",
                    slot=observed.slot,
                    predicate=f"self_move_pp:{world_ids[index]}",
                    detail=_bounded(
                        f"{world_ids[index]}: request pp {pp} != world "
                        f"{int(world_moves[index].pp)}"
                    ),
                )
            )
        disabled = bool(row.get("disabled"))
        if bool(world_moves[index].disabled) != disabled:
            out.append(
                ProjectionMismatch(
                    axis="self_move_disabled",
                    slot=observed.slot,
                    predicate=f"self_move_disabled:{world_ids[index]}",
                    detail=_bounded(
                        f"{world_ids[index]}: request disabled={disabled} != world "
                        f"{bool(world_moves[index].disabled)}"
                    ),
                )
            )
    return out


def _request_side_rows(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    side = request.get("side")
    if not isinstance(side, Mapping):
        return []
    rows = side.get("pokemon")
    if not isinstance(rows, Sequence):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _condition_hp_status(condition: Any) -> tuple[int | None, str]:
    """Parse a Showdown ``condition`` string into ``(hp, engine status)``."""

    text = str(condition or "").strip()
    if not text:
        return None, "NONE"
    if text.startswith("0 fnt") or text == "0":
        return 0, "NONE"
    head, _, tail = text.partition(" ")
    hp_text = head.split("/", 1)[0]
    try:
        hp = int(hp_text)
    except ValueError:
        return None, "NONE"
    token = tail.strip().split(" ")[0] if tail else ""
    return hp, _STATUS_TO_ENGINE.get(token, f"?{token}")


def _axis_self_party(
    observed: ObservedPublicView, side: Any
) -> list[ProjectionMismatch]:
    """The seat's OWN party, verbatim from its request, vs the world's party.

    This is the strongest axis available, because the seat's own team is not
    inferred at all: the request states its species, its HP, its status, its item
    and its ability. A sampled world that disagrees with it is wrong about
    information the SEARCHING PLAYER holds, never mind the opponent.

    Compared as a SET keyed by species, not positionally: `EngineWorld`'s
    docstring is explicit that party order is the sampled override's order and
    not the request's active-first permutation, so a positional comparison would
    report an ordering convention as a defect.
    """

    rows = _request_side_rows(observed.self_request)
    if not rows:
        return []
    out: list[ProjectionMismatch] = []
    party = _live_party(side)
    world_by_species: dict[str, Any] = {}
    for mon in party:
        world_by_species.setdefault(_identity_species(mon), mon)

    request_species = sorted(
        _norm(str(row.get("details", "")).split(",", 1)[0]) for row in rows
    )
    world_species = sorted(world_by_species)
    if request_species != world_species:
        out.append(
            ProjectionMismatch(
                axis="self_party_species",
                slot=observed.slot,
                predicate="self_party_species",
                detail=_bounded(f"request {request_species} != world {world_species}"),
            )
        )
        return out

    for row in rows:
        species = _norm(str(row.get("details", "")).split(",", 1)[0])
        mon = world_by_species[species]
        hp, status = _condition_hp_status(row.get("condition"))
        if hp is not None and int(mon.hp) != hp:
            out.append(
                ProjectionMismatch(
                    axis="self_party_hp",
                    slot=observed.slot,
                    predicate="self_party_hp",
                    detail=_bounded(f"{species}: request hp {hp} != world {int(mon.hp)}"),
                )
            )
        if not status.startswith("?") and str(mon.status).upper() != status:
            out.append(
                ProjectionMismatch(
                    axis="self_party_hp",
                    slot=observed.slot,
                    predicate="self_party_status",
                    detail=_bounded(
                        f"{species}: request status {status} != world "
                        f"{str(mon.status).upper()}"
                    ),
                )
            )
        # `item` is "" once the item is gone and absent on an unrevealed one; the
        # engine spells the empty item `none`. Only a STATED item is compared.
        item = row.get("item")
        if isinstance(item, str) and item:
            if _norm(item) != _norm(mon.item):
                out.append(
                    ProjectionMismatch(
                        axis="self_item",
                        slot=observed.slot,
                        predicate="self_item",
                        detail=_bounded(
                            f"{species}: request item {_norm(item)} != world {_norm(mon.item)}"
                        ),
                    )
                )
        ability = row.get("ability") or row.get("baseAbility")
        if (
            isinstance(ability, str)
            and ability
            and _norm(ability) != "trace"
            # A Transformed copy runs the DONOR's ability while the request keeps
            # reporting the transformer's own. Measured: `ditto: request ability
            # limber != world arenatrap`, 8 worlds on a six-game shard, where the
            # world is right and the request is describing the base form.
            and not _is_transformed(mon)
        ):
            # TRACE IS EXCLUDED, and the exclusion is the narrowest one that
            # works. A Tracer's request keeps reporting `trace` while the
            # constructor bakes in the ability that was actually copied
            # (`_apply_traced_ability_materialization_state`), so the world is
            # RIGHT and the request is stating the base. Measured: `gardevoir:
            # request ability trace != world rockhead`, 24 worlds on the first
            # six-game shard. Excluding the whole axis would have been the easy
            # move and would have blinded it everywhere else; excluding the one
            # producer keeps every other ability disagreement live.
            if _norm(ability) != _norm(mon.ability):
                out.append(
                    ProjectionMismatch(
                        axis="self_ability",
                        slot=observed.slot,
                        predicate="self_ability",
                        detail=_bounded(
                            f"{species}: request ability {_norm(ability)} != world "
                            f"{_norm(mon.ability)}"
                        ),
                    )
                )
    return out


def _axis_opponent_revealed(
    observed: ObservedPublicView, side: Any
) -> list[ProjectionMismatch]:
    """Everything the opponent has SHOWN must be in the world it is sampled into.

    Subset, not equality: the hidden remainder is exactly what sampling is for.
    A revealed species absent from the world, a revealed move the world's copy of
    that mon does not know, a revealed item or ability the world contradicts --
    each is a world the opponent can see is impossible.
    """

    out: list[ProjectionMismatch] = []
    world_by_species: dict[str, Any] = {}
    for mon in _live_party(side):
        world_by_species.setdefault(_identity_species(mon), mon)

    for record in observed.opponent_revealed:
        species = _norm(getattr(record, "species", "") or "")
        if not species:
            continue
        mon = world_by_species.get(species)
        if mon is None:
            # Cosmetic formes are collapsed engine-side (Unown letters), so a
            # revealed `unownc` legitimately lands on `unown`. Retry through the
            # same collapse the constructor applies rather than reporting the
            # convention as a defect.
            from .engine_world import _engine_species_id  # noqa: PLC0415

            mon = world_by_species.get(_engine_species_id(species))
        if mon is None:
            out.append(
                ProjectionMismatch(
                    axis="opponent_revealed_species",
                    slot=observed.opponent_slot,
                    predicate="opponent_revealed_species",
                    detail=_bounded(
                        f"revealed {species} absent from world party "
                        f"{sorted(world_by_species)}"
                    ),
                )
            )
            continue
        if _is_transformed(mon):
            # A Transformed copy carries the DONOR's moveset, so the revealed
            # moves of the transformer (typically just `transform`) are not in it
            # and never should be. The species axis above still holds it to
            # account; the moveset axis has nothing to say here.
            continue
        world_moves = {_norm(move.id) for move in mon.moves if str(move.id).lower() != "none"}
        for move_id in getattr(record, "moves", ()) or ():
            token = _norm(move_id)
            if not token or token == "struggle":
                continue
            if token in world_moves:
                continue
            if token == "hiddenpower" and any(
                candidate.startswith("hiddenpower") for candidate in world_moves
            ):
                continue
            out.append(
                ProjectionMismatch(
                    axis="opponent_revealed_moves",
                    slot=observed.opponent_slot,
                    predicate="opponent_revealed_moves",
                    detail=_bounded(
                        f"{species}: revealed move {token} absent from world "
                        f"{sorted(world_moves)}"
                    ),
                )
            )
        item = getattr(record, "item", None)
        if isinstance(item, str) and item and _norm(item) != _norm(mon.item):
            out.append(
                ProjectionMismatch(
                    axis="opponent_revealed_item",
                    slot=observed.opponent_slot,
                    predicate="opponent_revealed_item",
                    detail=_bounded(
                        f"{species}: revealed item {_norm(item)} != world {_norm(mon.item)}"
                    ),
                )
            )
        ability = getattr(record, "ability", None)
        if isinstance(ability, str) and ability and _norm(ability) != _norm(mon.ability):
            out.append(
                ProjectionMismatch(
                    axis="opponent_revealed_ability",
                    slot=observed.opponent_slot,
                    predicate="opponent_revealed_ability",
                    detail=_bounded(
                        f"{species}: revealed ability {_norm(ability)} != world "
                        f"{_norm(mon.ability)}"
                    ),
                )
            )
    return out


_ENGINE_BOOST_FIELD = {
    "atk": "attack_boost",
    "def": "defense_boost",
    "spa": "special_attack_boost",
    "spd": "special_defense_boost",
    "spe": "speed_boost",
    "accuracy": "accuracy_boost",
    "evasion": "evasion_boost",
}


def _axis_boosts(
    observed: ObservedPublicView, sides: Mapping[str, Any]
) -> list[ProjectionMismatch]:
    """Stat stages are announced on the protocol; both sides' are public."""

    out: list[ProjectionMismatch] = []
    for slot, side in sides.items():
        stated = observed.boosts.get(slot) or {}
        for key, field_name in _ENGINE_BOOST_FIELD.items():
            want = int(stated.get(key, 0) or 0)
            got = int(getattr(side, field_name, 0) or 0)
            if want != got:
                out.append(
                    ProjectionMismatch(
                        axis="boosts",
                        slot=slot,
                        predicate=f"boosts:{key}",
                        detail=_bounded(f"{key}: observed {want} != world {got}"),
                    )
                )
    return out


#: Showdown's own cap. `replay.toxic_stage` uses 16 as an internal saturation
#: sentinel meaning "already capped at 15 at an ordinary request", which is a
#: statement about the PARSER's certainty and not a stage the engine can hold.
_TOXIC_SATURATION_SENTINEL = 16


def _axis_toxic_count(
    observed: ObservedPublicView, sides: Mapping[str, Any], engine: Any
) -> list[ProjectionMismatch]:
    """The public Toxic ramp vs the engine's counter -- #1209's axis.

    Fires only when the stage is publicly KNOWN and the active is actually
    Toxic-statused. #1209 relaxed the proof this counter is allowed to be built
    from; if that proof is now too weak, the world carries a counter the log does
    not support and this is where it shows.
    """

    out: list[ProjectionMismatch] = []
    for slot, side in sides.items():
        stage = observed.toxic_stage.get(slot)
        if stage is None or int(stage) >= _TOXIC_SATURATION_SENTINEL:
            continue
        status = engine.p1_status if slot == "p1" else engine.p2_status
        if status != "TOXIC":
            continue
        got = int(getattr(side.side_conditions, "toxic_count", 0) or 0)
        if got != int(stage):
            out.append(
                ProjectionMismatch(
                    axis="toxic_count",
                    slot=slot,
                    predicate="toxic_count",
                    detail=_bounded(f"observed stage {int(stage)} != world {got}"),
                )
            )
    return out


def state_projection_mismatches(
    context: Any,
    world: Any,
    state: Any,
    *,
    observed: ObservedPublicView | None = None,
) -> list[ProjectionMismatch]:
    """Every public fact this world contradicts. Empty list = projection matches.

    ``world`` is an ``EngineWorld`` (its ``slot_sides`` maps p1/p2 onto the
    engine's side_one/side_two) and ``state`` is the built ``poke_engine.State``
    that search is about to receive -- not a re-derivation of it.
    """

    observed = observed_public_view(context) if observed is None else observed
    if observed is None:
        return []
    slot_sides = world.slot_sides
    engine = _engine_turn_features(state, slot_sides)
    sides = _sides_by_slot(state, slot_sides)
    mismatches: list[ProjectionMismatch] = []
    mismatches += _axis_active_hp(observed, engine)
    mismatches += _axis_active_status(observed, engine)
    mismatches += _axis_weather(observed, engine)
    mismatches += _axis_side_conditions(observed, engine)
    mismatches += _axis_self_moves(observed, sides[observed.slot])
    mismatches += _axis_self_party(observed, sides[observed.slot])
    mismatches += _axis_opponent_revealed(observed, sides[observed.opponent_slot])
    mismatches += _axis_boosts(observed, sides)
    mismatches += _axis_toxic_count(observed, sides, engine)
    return mismatches


# --- the renderer's projection -------------------------------------------------


#: Markers whose telemetry is incomplete while the public action window is still
#: exact, so the branch stays usable for matching. Mirrors
#: `engine_transition_differential._TELEMETRY_ONLY_LOSSY_MARKERS` -- an ALLOWLIST,
#: because an exclusion ("skip attribution_unsafe") fails open as the renderer
#: grows, and that file records exactly that mistake being made and reverted.
RENDER_TELEMETRY_ONLY_LOSSY = frozenset(
    {"sleeptalk_called_unidentified", "attract_immobilization_source_unknown"}
)


def render_branch_is_usable(lossy: Sequence[str]) -> bool:
    return not lossy or set(lossy) <= RENDER_TELEMETRY_ONLY_LOSSY


def render_projection_mismatch(
    *,
    state_string: str,
    slot_sides: Mapping[str, str],
    party_display: Mapping[str, Sequence[str]],
    turn: int,
    choices: Mapping[str, str],
    observed_lines: Sequence[str],
    pre_features: Any,
    module: Any | None = None,
) -> tuple[list[ProjectionMismatch], dict[str, Any]]:
    """Does the transition Showdown took lie in the RENDERED branch support?

    The state comparator reads the world at the decision boundary. The renderer
    runs downstream of it, on branches nothing else compares against a log -- and
    that is where #1211 lives: a guard that decides whether a zero-amount Heal is
    Protect's no-op or an absorb ability's. Widen it and the mapper renders
    ``|-activate|...|Protect`` over an event the engine emitted as a real heal.
    Nothing in direction 1 or in the state comparator can see that, because the
    refusal it replaced is exactly what was removed.

    So: take the joint action that was ACTUALLY played, render every branch the
    engine enumerates from this world through the shipped
    ``pokezero_search.branch_events`` mapper, fold each render into the same
    ``TurnFeatures`` vocabulary the observed lines fold into, and require at
    least one usable branch to match.

    HONEST LIMIT, stated where it is used and not only in a report. The match is
    the NET-HP band comparator (`engine_fidelity._mismatch_reason`'s shape: exact
    status/faint/weather/side-conditions, banded HP), NOT
    `engine_transition_differential`'s per-source strict decomposition. A
    sub-band systematic error passes. It is the coarser of the two instruments
    and it is chosen because it needs only the rendered lines, while the strict
    matcher needs a per-source attribution pass that is not reusable outside that
    script. Anything this reports is real; what it does NOT report is bounded by
    the band.

    Returns ``(mismatches, diagnostics)``.
    """

    if module is None:
        import pokezero_search as module  # noqa: PLC0415

    side_one_choice = choices["p1"] if slot_sides["p1"] == "side_one" else choices["p2"]
    side_two_choice = choices["p2"] if slot_sides["p2"] == "side_two" else choices["p1"]
    ctx = json.dumps(
        {
            "p1": list(party_display["p1" if slot_sides["p1"] == "side_one" else "p2"]),
            "p2": list(party_display["p2" if slot_sides["p2"] == "side_two" else "p1"]),
            "turn": int(turn),
        }
    )
    try:
        report = json.loads(
            module.branch_events(
                state_string, side_one_choice, side_two_choice, ctx, True, False
            )
        )
    except Exception as error:  # noqa: BLE001 - a probe must never break the run
        return [], {"render_error": f"{type(error).__name__}: {error}"}

    observed = fold_step_lines(observed_lines, pre_features)
    branches = report.get("branches") or []
    # SELF-CONSISTENCY FIRST, on every branch, not only the one the game took.
    # It needs no log and rides the same `branch_events` call, so leaving it out
    # would discard the counterfactual branches entirely -- and those are most of
    # the tree.
    self_consistency = render_self_consistency_mismatches(
        branches, slot_sides=slot_sides, pre_features=pre_features
    )
    usable = 0
    unusable_markers: Counter[str] = Counter()
    reasons: list[str] = []
    for branch in branches:
        lossy = list(branch.get("lossy") or [])
        if branch.get("attribution_unsafe") or not render_branch_is_usable(lossy):
            for marker in sorted(set(lossy)) or ["attribution_unsafe"]:
                unusable_markers[str(marker).split(":")[0]] += 1
            continue
        usable += 1
        rendered = fold_step_lines(
            [line for line in (branch.get("events") or []) if line and line != "|"],
            pre_features,
        )
        reason = _render_mismatch_reason(observed, rendered, pre_features)
        if reason is None:
            return self_consistency, {
                "branches": len(branches),
                "usable_branches": usable,
                "matched": True,
                "self_consistency": len(self_consistency),
            }
        reasons.append(reason)

    diagnostics = {
        "branches": len(branches),
        "usable_branches": usable,
        "matched": False,
        "unusable_markers": dict(unusable_markers),
        "reasons": [_bounded(reason) for reason in reasons[:4]],
        "self_consistency": len(self_consistency),
    }
    if usable == 0:
        # NOT a mismatch verdict on the renderer's content: the renderer told us
        # it could not describe any branch. Counted on its own axis so a census
        # can never read "no unmatched transitions" off a block where nothing was
        # comparable in the first place.
        return self_consistency + [
            ProjectionMismatch(
                axis="render_no_usable_branch",
                slot="both",
                predicate="render_no_usable_branch",
                detail=_bounded(f"{len(branches)} branches, markers {dict(unusable_markers)}"),
            )
        ], diagnostics
    return self_consistency + [
        ProjectionMismatch(
            axis="render_unmatched_transition",
            slot="both",
            predicate="render_unmatched_transition",
            detail=_bounded(f"{usable} usable branches, first reason: {reasons[0]}"),
        )
    ], diagnostics


def render_self_consistency_mismatches(
    branches: Sequence[Mapping[str, Any]],
    *,
    slot_sides: Mapping[str, str],
    pre_features: Any,
) -> list[ProjectionMismatch]:
    """Does each branch's RENDER describe that branch's own outcome?

    The transition comparator above needs the log, so it only ever sees the ONE
    branch the game actually took. Every other branch search spent budget on --
    the counterfactuals the tree is made of -- is never compared to anything.
    This closes that: `branch_events` already returns each branch's post-state
    summary alongside its rendered lines, so the render can be held against the
    engine's own outcome for the SAME branch, at zero extra cost and with no log
    required.

    This is the arm that sees #1211's over-broad direction. Render
    ``|-activate|...|Protect`` over an instruction the engine emitted as a real
    heal and the rendered lines say the defender's HP did not move while the
    branch's own post-state says it did.

    WHAT IT CANNOT SEE, stated here rather than in a report: an over-broad
    marker over a ZERO-amount heal. A full-HP absorber's no-op and Protect's
    no-op have the same post-state by construction -- that is what makes them
    ambiguous in the first place -- so no state-based comparator can separate
    them. Separating those two needs the per-source LINE decomposition in
    ``scripts/engine_transition_differential.py``, and this arm does not claim
    to replace it.
    """

    out: list[ProjectionMismatch] = []
    engine_label = {
        slot: ("p1" if slot_sides[slot] == "side_one" else "p2") for slot in ("p1", "p2")
    }
    for index, branch in enumerate(branches):
        post = branch.get("post")
        if not isinstance(post, Mapping):
            continue
        if branch.get("attribution_unsafe") or not render_branch_is_usable(
            list(branch.get("lossy") or [])
        ):
            continue
        rendered = fold_step_lines(
            [line for line in (branch.get("events") or []) if line and line != "|"],
            pre_features,
        )
        for slot in ("p1", "p2"):
            side_post = post.get(engine_label[slot]) or {}
            want_hp = side_post.get("active_hp")
            if want_hp is None:
                continue
            got_hp = rendered.hp[slot]
            if got_hp < 0 or int(want_hp) == int(got_hp):
                continue
            out.append(
                ProjectionMismatch(
                    axis="render_post_state_disagreement",
                    slot=slot,
                    predicate="render_post_state_disagreement:active_hp",
                    detail=_bounded(
                        f"branch {index}: render says hp {got_hp}, the branch's own "
                        f"post-state says {int(want_hp)}"
                    ),
                )
            )
    return out


def _render_mismatch_reason(
    observed: StepProjection, rendered: StepProjection, pre: Any
) -> str | None:
    """Exact on everything deterministic; banded on HP only.

    The band is anchored on THIS step's HP movement, as `engine_fidelity` anchors
    it: an engine branch carries the representative damage roll while Showdown
    sampled one of sixteen, so a same-mechanic hit differs by up to ~16%.
    Everything else -- status, faint set, side conditions -- is exact, because
    none of it is roll-scaled.

    WEATHER IS NOT COMPARED, deliberately. Both sides fold ONE step, an absent
    `|-weather|` is indistinguishable from "unchanged", and the axis would then
    only ever fire on an upkeep-line rendering convention. The STATE comparator
    does compare weather, against the whole-log fold, where it is well defined.
    """

    from .engine_fidelity import _DAMAGE_TOLERANCE, _MIN_TOLERANCE_HP  # noqa: PLC0415

    if observed.fainted != rendered.fainted:
        return f"fainted {sorted(observed.fainted)} != {sorted(rendered.fainted)}"
    for slot in ("p1", "p2"):
        if slot in observed.fainted:
            continue
        obs, ren = observed.status[slot], rendered.status[slot]
        if obs.startswith("?") or obs == ren:
            continue
        return f"{slot} status {obs} != {ren}"
    if observed.side_conditions != rendered.side_conditions:
        return (
            f"side conditions {observed.side_conditions} != {rendered.side_conditions}"
        )
    for slot, start_hp in (("p1", pre.p1_hp), ("p2", pre.p2_hp)):
        obs, ren = observed.hp[slot], rendered.hp[slot]
        if obs < 0 or ren < 0 or obs == ren:
            continue
        moved = max(abs(int(start_hp) - obs), abs(int(start_hp) - ren))
        tolerance = max(_MIN_TOLERANCE_HP, int(moved * _DAMAGE_TOLERANCE) + 1)
        if abs(obs - ren) > tolerance:
            return f"{slot} hp {obs} != {ren} (moved {moved}, tolerance {tolerance})"
    return None


# --- records and the probe ----------------------------------------------------


@dataclass
class WorldProjectionRecord:
    """One searched world's verdict."""

    world_index: int
    mismatches: list[ProjectionMismatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_index": self.world_index,
            "mismatches": [m.to_dict() for m in self.mismatches],
        }


@dataclass
class DecisionProjectionRecord:
    battle_id: str
    seed: int
    seat: str
    round: int
    turn: int
    arm: str
    worlds: list[WorldProjectionRecord] = field(default_factory=list)
    render: dict[str, Any] | None = None
    exemplar: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "battle_id": self.battle_id,
            "seed": self.seed,
            "seat": self.seat,
            "round": self.round,
            "turn": self.turn,
            "arm": self.arm,
            "worlds": [world.to_dict() for world in self.worlds],
            "render": self.render,
            "exemplar": self.exemplar,
        }


#: A world-level corruption applied for instrument testing. Returns a label.
Forcing = Callable[[Any, Any], str]


class WorldObserver:
    """Collects the worlds a policy constructed and projects each one.

    Installed on ``EngineMctsPolicy`` through its ``world_observer`` hook, which
    fires once per successfully constructed world with exactly the ``(EngineWorld,
    State)`` pair search is about to receive. Never the probe's own re-sample:
    report 4 section 4.2 is four different ways a harness measured one thing while
    believing it measured two, and a re-sampled world would be a fifth.
    """

    def __init__(
        self,
        *,
        arm: str,
        records: list[DecisionProjectionRecord],
        exemplar_store: MutableMapping[str, dict[str, Any]] | None = None,
        forcing: Forcing | None = None,
    ) -> None:
        self.arm = arm
        self.records = records
        self.exemplar_store: MutableMapping[str, dict[str, Any]] = (
            {} if exemplar_store is None else exemplar_store
        )
        self.forcing = forcing
        self.errors: list[str] = []
        self.observed_worlds = 0
        self.seed = 0
        self._current: DecisionProjectionRecord | None = None
        self._current_key: tuple[Any, ...] | None = None
        self._observed: ObservedPublicView | None = None
        self.forced_labels: Counter[str] = Counter()

    # -- the hook ------------------------------------------------------------

    def __call__(self, context: Any, world: Any, state: Any) -> None:
        try:
            self._observe(context, world, state)
        except Exception as error:  # noqa: BLE001 - telemetry never breaks a run
            self.errors.append(
                f"{getattr(context, 'battle_id', '?')}"
                f"/{getattr(context, 'decision_round_index', '?')}"
                f"/{getattr(context, 'player_id', '?')}: {type(error).__name__}: {error}"
            )

    def _observe(self, context: Any, world: Any, state: Any) -> None:
        key = (
            str(getattr(context, "battle_id", "?")),
            str(getattr(context, "player_id", "?")),
            getattr(context, "decision_round_index", None),
        )
        if key != self._current_key:
            self._observed = observed_public_view(context)
            replay = getattr(
                getattr(context, "public_materialization_state", None), "replay", None
            )
            self._current = DecisionProjectionRecord(
                battle_id=key[0],
                seed=self.seed,
                seat=key[1],
                round=-1 if key[2] is None else int(key[2]),
                turn=int(getattr(replay, "turn_number", 0) or 0),
                arm=self.arm,
            )
            self._current_key = key
            self.records.append(self._current)
        assert self._current is not None
        if self.forcing is not None:
            # The forcing REPLACES the world under test. It returns a rebuilt
            # `(world, state)` rather than mutating in place, because the engine's
            # pyo3 objects are read-only from Python and an in-place version
            # swallowed 312 AttributeErrors per game while reporting zero
            # mismatches. A forcing that cannot force is the failure this whole
            # apparatus exists to prevent, so it must be structurally impossible
            # here: if the rebuild raises, `__call__` records an instrument error
            # and no world is projected at all.
            label, world, state = self.forcing(world, state)
            self.forced_labels[label] += 1
        self.observed_worlds += 1
        mismatches = state_projection_mismatches(
            context, world, state, observed=self._observed
        )
        self._current.worlds.append(
            WorldProjectionRecord(
                world_index=len(self._current.worlds), mismatches=mismatches
            )
        )
        if mismatches and self._current.exemplar is None:
            self._current.exemplar = self._exemplar(context, mismatches)

    def _exemplar(self, context: Any, mismatches: Sequence[ProjectionMismatch]) -> Any:
        novel = [m.predicate for m in mismatches if m.predicate not in self.exemplar_store]
        payload = {
            "battle_id": str(getattr(context, "battle_id", "?")),
            "seed": self.seed,
            "seat": str(getattr(context, "player_id", "?")),
            "round": getattr(context, "decision_round_index", None),
            "arm": self.arm,
            "mismatches": [m.to_dict() for m in mismatches[:8]],
        }
        for predicate in novel:
            self.exemplar_store[predicate] = payload
        return payload if novel else None


class PublicProjectionProbe:
    """Policy wrapper that keeps the driver's decision and records the projection.

    The wrapped policy plays; nothing here can change an action. Its only job
    beyond the observer is to remember the action index each seat chose, because
    the render comparator needs the joint action that was ACTUALLY played and a
    world observer never sees it.
    """

    def __init__(self, *, primary: Any, observer: WorldObserver) -> None:
        self.primary = primary
        self.observer = observer
        self.last_decision: dict[str, Any] = {}

    @property
    def policy_id(self) -> str:
        return getattr(self.primary, "policy_id", "public-projection")

    @property
    def stats(self) -> Any:
        return self.primary.stats

    def select_action(self, observation: Any, *, rng: Any) -> Any:
        return self.primary.select_action(observation, rng=rng)

    def select_action_with_context(self, context: Any, *, rng: Any) -> Any:
        decision = self.primary.select_action_with_context(context, rng=rng)
        self.last_decision[str(getattr(context, "player_id", "?"))] = decision
        return decision


# --- aggregation ---------------------------------------------------------------


def aggregate_projection_records(
    records: Sequence[DecisionProjectionRecord | Mapping[str, Any]]
) -> dict[str, Any]:
    """Roll the per-decision records into the direction-2 inventory.

    THREE UNITS, three tables, never one ranking: ``*_worlds`` counts WORLDS,
    ``*_decisions`` counts DECISIONS, ``render_*_boundaries`` counts BOUNDARIES.
    """

    rows = [r if isinstance(r, Mapping) else r.to_dict() for r in records]
    decisions = 0
    worlds = 0
    mismatched_worlds = 0
    mismatched_decisions = 0
    axis_worlds: Counter[str] = Counter()
    predicate_worlds: Counter[str] = Counter()
    predicate_decisions: Counter[str] = Counter()
    predicate_axis: dict[str, str] = {}
    predicate_exemplar: dict[str, Any] = {}
    per_arm: Counter[str] = Counter()
    per_arm_worlds: Counter[str] = Counter()
    battles: set[str] = set()
    render_boundaries = 0
    render_mismatched = 0
    render_axis: Counter[str] = Counter()
    render_errors: Counter[str] = Counter()

    for row in rows:
        decisions += 1
        battles.add(str(row.get("battle_id")))
        arm = str(row.get("arm", "?"))
        per_arm[arm] += 1
        decision_predicates: set[str] = set()
        for world in row.get("worlds") or []:
            worlds += 1
            per_arm_worlds[arm] += 1
            mismatches = world.get("mismatches") or []
            if mismatches:
                mismatched_worlds += 1
            for mismatch in mismatches:
                axis = str(mismatch["axis"])
                predicate = str(mismatch["predicate"])
                axis_worlds[axis] += 1
                predicate_worlds[predicate] += 1
                predicate_axis[predicate] = axis
                decision_predicates.add(predicate)
        if decision_predicates:
            mismatched_decisions += 1
            for predicate in decision_predicates:
                predicate_decisions[predicate] += 1
        exemplar = row.get("exemplar")
        if exemplar:
            for mismatch in exemplar.get("mismatches", []):
                predicate_exemplar.setdefault(str(mismatch["predicate"]), exemplar)
        render = row.get("render")
        if render:
            if render.get("error"):
                render_errors[str(render["error"])[:80]] += 1
                continue
            render_boundaries += 1
            axes = render.get("axes") or []
            if axes:
                render_mismatched += 1
                for axis in axes:
                    render_axis[str(axis)] += 1

    return {
        "decisions_seen": decisions,
        "battles": len(battles),
        "worlds_projected": worlds,
        "projection_mismatched_worlds": mismatched_worlds,
        "projection_mismatched_decisions": mismatched_decisions,
        "projection_world_mismatch_rate": (mismatched_worlds / worlds) if worlds else None,
        "projection_decision_mismatch_rate": (
            (mismatched_decisions / decisions) if decisions else None
        ),
        "distinct_open_predicates": len(predicate_worlds),
        "decisions_per_arm": dict(per_arm),
        "worlds_per_arm": dict(per_arm_worlds),
        # WORLDS per axis. Never compared with the DECISIONS column below.
        "axis_worlds": dict(axis_worlds.most_common()),
        "predicates": [
            {
                "predicate": predicate,
                "axis": predicate_axis[predicate],
                "worlds": count,
                "decisions": predicate_decisions[predicate],
                "exemplar": predicate_exemplar.get(predicate),
            }
            for predicate, count in predicate_worlds.most_common()
        ],
        # BOUNDARIES.
        "render_boundaries_compared": render_boundaries,
        "render_mismatched_boundaries": render_mismatched,
        "render_mismatch_rate": (
            (render_mismatched / render_boundaries) if render_boundaries else None
        ),
        "render_axis_boundaries": dict(render_axis.most_common()),
        "render_errors": dict(render_errors.most_common(10)),
    }


# --- identity witness ----------------------------------------------------------

#: A symbol that exists ONLY in the tree carrying this module. `__file__` alone
#: does not prove which SOURCE is loaded: a stale `.pyc` has the right path and
#: the wrong bytes, and report 4 section 4.2 case 2 is a size-preserving edit
#: whose stale bytecode scored a clean pass.
CONTENT_FINGERPRINT_SYMBOL = "state_projection_mismatches"


def identity_witness() -> dict[str, Any]:
    """Which tree is loaded, read from the LOADED modules. Never from argv."""

    import sys  # noqa: PLC0415

    import pokezero
    from pokezero import engine_search, public_projection

    witness: dict[str, Any] = {
        "sys_executable": sys.executable,
        "sys_path_head": list(sys.path[:4]),
        "pokezero_file": pokezero.__file__,
        "engine_search_file": engine_search.__file__,
        "public_projection_file": public_projection.__file__,
        "public_projection_present": hasattr(
            public_projection, CONTENT_FINGERPRINT_SYMBOL
        ),
        "engine_search_world_observer_hook": "world_observer"
        in getattr(engine_search.EngineMctsPolicy.__init__, "__code__").co_varnames,
        "public_projection_axis_count": len(AXES),
        "source_sha256": {},
    }
    for name, module in (
        ("public_projection", public_projection),
        ("engine_search", engine_search),
    ):
        path = getattr(module, "__file__", None)
        if path:
            try:
                with open(path, "rb") as handle:
                    witness["source_sha256"][name] = hashlib.sha256(
                        handle.read()
                    ).hexdigest()[:16]
            except OSError as error:  # pragma: no cover - diagnostics only
                witness["source_sha256"][name] = f"unreadable: {error}"
    try:
        import pokezero_search

        witness["pokezero_search_file"] = pokezero_search.__file__
        witness["pokezero_search_model_feature"] = bool(
            getattr(pokezero_search, "MODEL_FEATURE_ENABLED", False)
        )
        witness["pokezero_search_so_sha256"] = _extension_hash(pokezero_search)
    except Exception as error:  # noqa: BLE001
        witness["pokezero_search_file"] = f"unavailable: {type(error).__name__}: {error}"
        witness["pokezero_search_model_feature"] = None
    try:
        import torch

        witness["torch_version"] = torch.__version__
    except Exception as error:  # noqa: BLE001
        witness["torch_version"] = f"unavailable: {type(error).__name__}: {error}"
    return witness


def _extension_hash(module: Any) -> str:
    """SHA-256 of the compiled crate, not of its Python shim."""

    import pathlib  # noqa: PLC0415

    root = pathlib.Path(getattr(module, "__file__", "")).parent
    candidates = sorted(root.glob("*.so")) + sorted(root.glob("*.pyd"))
    if not candidates:
        return "no extension module found"
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def witness_json() -> str:
    return json.dumps(identity_witness(), indent=2, sort_keys=True)


if __name__ == "__main__":  # pragma: no cover - the neutral-cwd child entrypoint
    print(witness_json())
