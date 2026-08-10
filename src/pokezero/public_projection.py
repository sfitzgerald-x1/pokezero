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

That matters for the four guards relaxed on 2026-08-09, **two** of which are
visible to the state comparator. The count is MEASURED, not counted off the
list below: the first revision of this header said *three* and reached three by
naming an axis for #1212 that does not exist.

* #1210 (Transform PP overlay) -> ``self_move_pp``: the request publishes the
  copy's live PP every round, so a wrong overlay is a numeric disagreement.
  **COVERED-MEASURED** -- reverting ``_copied_move_spec``'s overlay takes this
  axis from 0 to 2,219 WORLDS on the census block.
* #1209 (toxic stage: demand a weaker proof) -> ``toxic_count``, whose observed
  side is ``observed_toxic_multiplier``: the multiplier the log shows was
  actually PAID, recovered from raw ``|-damage|...|[from] psn`` damage.
  **COVERED-MEASURED, by a PRODUCER mutant and not by the census reading zero**
  -- see ``_axis_toxic_count`` and the two paragraphs below, which are the whole
  reason a zero on this axis must not be read as coverage.
* #1212 (a third Encore resolution source) -> **NO AXIS. NOT COVERED, BY
  CONSTRUCTION**, and this is where the first revision of this header was
  simply wrong. ``_apply_encore_locks`` writes only
  ``last_used_move=f"move:{index}"`` and never touches ``disabled``; the
  per-move ``disabled`` flag comes from ``_move_specs`` via ``known_pp``, built
  from the request's own rows -- so ``self_move_disabled`` compares the
  request's flag against the request's flag. Separately, #1212's class is the
  OPPONENT seat, and every ``_SELF_AXES`` member is evaluated only for
  ``context.player_id``. RETRACTED: the claim that ``self_move_disabled`` is
  "the axis that makes the relaxation falsifiable at all". It is not an axis on
  this relaxation at all, and no axis here is.
* #1211 (absorb guard narrowed to HP headroom) is NOT visible here. It is a
  renderer branch, and it is why ``render_projection_mismatch`` exists.
  COVERED-MEASURED there, on a separately built crate.

Two things about #1209's axis that a zero must NOT be read as
-------------------------------------------------------------
**The two sides are not independent DERIVATIONS.** They are two
implementations of the same arithmetic over the same ``[from] psn`` line:
``showdown._reseed_toxic_stage_from_residual`` and
``observed_toxic_multiplier`` both compute ``damage // (maxhp // 16)`` and both
refuse on a non-integral quotient. So zero variance across a whole census block
is that IDENTITY restating itself, not agreement between two witnesses. Nothing
about a zero here is evidence of power.

**What establishes the power is a producer mutant, and it fires.** Widening
``local_showdown._materialization_toxic_stage`` from
``min(14, max(0, tracked_stage - 1))`` to ``min(14, max(0, tracked_stage))`` --
an over-broad #1209 crediting every world one extra tick -- takes this axis
from 0 to **11,568 WORLDS / 1,320 DECISIONS** on the 731-battle block
(mismatched worlds 406 -> 10,962). So #1209's over-credit shape is
COVERED-MEASURED. Command and per-arm source sha256 in the direction-2 report.

**And one arm of #1209 is unreachable here, which is the honest reason its
literal site is not covered.** A producer mutant at #1209's own site --
``if tracked_stage == 0:`` returning 0 immediately, dropping every proof
requirement -- is a NULL MUTANT: the census records are **bit-identical** on
both arms. That is not "the mutant was not run". It was run; this block does
not reach that arm, and reachability is the limit, not the instrument.

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
    "observed_toxic_multiplier",
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

#: Tokens a Showdown request can offer that are not moves any engine moveset
#: carries. Kept as a named constant so deleting it is a killable mutation.
_REQUEST_PSEUDO_MOVES = frozenset({"struggle", "recharge"})


def _bounded(text: str, limit: int = _DETAIL_LIMIT) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


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
    #: `{slot: multiplier or None}` DERIVED FROM RAW DAMAGE, not from the
    #: parser's toxic tracker -- see `observed_toxic_multiplier` for why the
    #: tracker cannot be used here.
    toxic_multiplier: Mapping[str, int | None]
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
        toxic_multiplier=observed_toxic_multiplier(lines),
        opponent_revealed=tuple(
            (getattr(replay, "public_revealed", {}) or {}).get(opponent, ())
        ),
    )


def _fold_public_lines(lines: Sequence[str]) -> Any:
    """Fold raw protocol lines into comparable features.

    `showdown_turn_features` supplies weather and side conditions: it is the fold
    the engine fidelity differential has matched engine states against since PR
    #727, so this oracle and that differential cannot drift on what "the protocol
    said" means for those.

    HP AND STATUS ARE RE-DERIVED HERE, and the reason is measured rather than
    stylistic. That function handles `switch`, `-damage`, `-heal`, `-status`,
    `-curestatus` and `faint` -- and **not `-sethp`**, which is correct for the
    one-turn fixtures it was written for and wrong over a whole log. Pain Split
    sets both sides with `|-sethp|...|[from] move: Pain Split|[silent]`, so the
    observed side kept the pre-Pain-Split HP while the world -- which reads the
    line -- held the right one. That produced the whole `active_hp` class on the
    first census: 360 worlds over 45 decisions, every one of them the OBSERVED
    side being wrong. Exemplar dumped at `ppc-s0-9800144` p1 round 5, world
    `armaldo 155/260`, protocol `|-sethp|p2a: Armaldo|155/260|[from] move: Pain
    Split|[silent]`.

    It also drops that function's requirement that a `|switch|` be seen for a
    slot before its `-damage` lines count, which is right for a one-turn fixture
    and needlessly lossy over a log that opens with switches anyway.
    """

    import types  # noqa: PLC0415

    from .engine_fidelity import TurnFeatures, _parse_condition  # noqa: PLC0415

    base = showdown_turn_features(types.SimpleNamespace(protocol_lines=tuple(lines)))
    hp: dict[str, int] = {"p1": -1, "p2": -1}
    status: dict[str, str] = {"p1": "NONE", "p2": "NONE"}
    for line in lines:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1]
        slot = parts[2].split(":", 1)[0].strip()[:2]
        if slot not in ("p1", "p2"):
            continue
        if tag in _HP_TAGS_FIELD4 and len(parts) >= 5:
            value, token = _parse_condition(parts[4])
            hp[slot] = value
            status[slot] = _STATUS_TO_ENGINE.get(token, "NONE") if token else "NONE"
        elif tag in _HP_TAGS_FIELD3 and len(parts) >= 4:
            value, token = _parse_condition(parts[3])
            hp[slot] = value
            if token:
                status[slot] = _STATUS_TO_ENGINE.get(token, status[slot])
        elif tag == "-status" and len(parts) >= 4:
            status[slot] = _STATUS_TO_ENGINE.get(parts[3].strip(), status[slot])
        elif tag in _CURE_TAGS:
            status[slot] = "NONE"
        elif tag == "faint":
            hp[slot] = 0
    return TurnFeatures(
        p1_hp=hp["p1"],
        p2_hp=hp["p2"],
        p1_status=status["p1"],
        p2_status=status["p2"],
        fainted=frozenset(slot for slot in ("p1", "p2") if hp[slot] == 0),
        weather=base.weather,
        side_conditions=base.side_conditions,
    )


#: Protocol tags that state an absolute HP for a slot. `|switch|` and `|drag|`
#: carry it in field 4; the rest in field 3.
#: Tags that clear a status. `|-cureteam|` is the one that is easy to miss and
#: it cost 1,168 worlds over 146 decisions on a census before it was handled:
#: Aromatherapy and Heal Bell cure the WHOLE TEAM and Showdown announces them
#: with `|-cureteam|`, not `|-curestatus|`, so the observed side stayed on the
#: status last printed by a `-damage` condition string while the world correctly
#: held none. Exemplar `ppc-s1-10100064` p1 round 75:
#: `|-cureteam|p2a: Blissey|[from] move: Aromatherapy`, observed TOXIC, world
#: NONE, and the WORLD was right.
_CURE_TAGS = ("-curestatus", "-cureteam")

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
            # `if token` IS LOAD-BEARING. `_STATUS_TO_ENGINE` carries "" as a KEY
            # mapping to "NONE", so `.get(token, status[side])` never reaches its
            # default and an ordinary `|-damage|p2a: X|253/335` WIPED the folded
            # status to NONE. Independent review measured the cost: 969 of 3,078
            # render boundaries "mismatched", 830 of them on `status X != NONE`,
            # and only 31 of the 969 contained any switch line -- so the
            # mechanism this PR first published for that number was wrong.
            if token:
                status[side] = _STATUS_TO_ENGINE.get(token, status[side])
        elif tag in _HP_TAGS_FIELD4 and side and len(parts) >= 5:
            value, token = _parse_condition(parts[4])
            hp[side] = value
            # A switch DOES restate the incoming mon's status absolutely, so an
            # empty token here really does mean "no status".
            status[side] = _STATUS_TO_ENGINE.get(token, "NONE") if token else "NONE"
            fainted.discard(side)
        elif tag == "-status" and side and len(parts) >= 4:
            status[side] = _STATUS_TO_ENGINE.get(parts[3].strip(), status[side])
        elif tag in _CURE_TAGS and side:
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


# NO `_currently_fainted` HELPER. It used to recompute "which active is down"
# from HP beside a fold that already computes exactly that, and the duplication
# made the fold's own `fainted` field DEAD -- so the battery's mutant for the
# sticky-faint hazard was unkillable, because the value it corrupted was never
# read. One producer, one reader: `_fold_public_lines` derives `fainted` from HP
# and this module trusts it, which puts the hazard back inside the blast radius
# of `S07-sticky-faint-set-restored-at-the-fold`.
def _axis_active_status(observed: ObservedPublicView, engine: Any) -> list[ProjectionMismatch]:
    out: list[ProjectionMismatch] = []
    down = observed.turn_features.fainted
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

    THE RULE, restated after review: **every move the request says this seat may
    pick must exist in the searched world, with the PP and the disabled flag the
    request states.** Matched BY ID, not positionally, and the world is allowed
    to carry moves the request does not name.

    The first revision compared positionally and required equal length, and it
    was wrong in two documented ways at once.

    * A LOCK -- a two-turn charge, a recharge, a Choice lock -- restricts the
      request to one move while the world legitimately keeps all four. 304 of
      710 firings on the first census were this.
    * A TRANSFORMED active carries the belief-sampled DONOR's moveset, and
      ``_copied_move_spec`` (#1210's own fix) says in terms that a slot the
      request does not name is *left usable so that the fiction stays
      COUNTABLE* -- it is deliberately routed to ``unmapped_choices``, which is
      how the class is measured. The remaining 406 firings were this: the
      published exemplar was a Ditto transformed into Cradily, i.e. the FOURTH
      instance of a Transform artifact already excluded on three other axes.
      Independent review classified all 8 shards: 66 transformed, 32 locks,
      **zero residue.**

    WHAT SURVIVES THIS NARROWING, and it is the part with real power. ``pp`` and
    ``disabled`` are still compared for every move the request DOES name,
    including on a transformed active -- and that is exactly where #1210 lives.
    ``_copied_move_spec`` is the only producer of those values that is not a
    copy of the request rows (``_move_specs`` reads ``known_pp`` straight from
    the request for every other case), so **``self_move_pp`` is tautological
    everywhere except the Transform overlay, where it is the one axis in this
    module with measured power over one of the four relaxations.** See the
    ``M1210`` mutant in the battery.
    """

    rows = _request_active_moves(observed.self_request)
    if not rows:
        return []
    if observed.self_request.get("forceSwitch"):
        return []
    active = side.pokemon[int(str(side.active_index))]
    world_moves = [move for move in active.moves if str(move.id).lower() != "none"]
    observed_ids = [_norm(row.get("id") or row.get("move")) for row in rows]
    if observed_ids == ["struggle"]:
        # Struggle is the ENGINE's substitution, not a moveset claim.
        return []
    # REQUEST PSEUDO-MOVES. `recharge` is not a move the engine's moveset can
    # carry -- a recharging seat holds MUSTRECHARGE and the engine's own option
    # surface offers only "No Move", which is why
    # `engine_transition_differential.engine_choice_for_action` translates it
    # rather than looking it up. Measured firing 32 times on a 40-game slice
    # before it was excluded; it says nothing about any world.
    observed_ids = [token for token in observed_ids if token not in _REQUEST_PSEUDO_MOVES]
    rows = [
        row
        for row in rows
        if _norm(row.get("id") or row.get("move")) not in _REQUEST_PSEUDO_MOVES
    ]
    if not rows:
        return []

    by_id: dict[str, Any] = {}
    for move in world_moves:
        by_id.setdefault(_norm(move.id), move)

    def _lookup(observed_id: str) -> Any | None:
        if observed_id in by_id:
            return by_id[observed_id]
        if observed_id == "hiddenpower":
            # The request says `hiddenpower`; the engine carries the typed+BP id.
            typed = [key for key in by_id if key.startswith("hiddenpower")]
            if len(typed) == 1:
                return by_id[typed[0]]
        return None

    out: list[ProjectionMismatch] = []
    # SPLIT BY PRODUCER, so the queue classifies itself instead of needing a
    # human pass. A non-transformed self active takes its moveset from the
    # request rows themselves (`_move_specs` + `known_pp`), so a request-offered
    # move can only be missing when the active is a Transform copy whose
    # belief-SAMPLED donor lacks it. The two are different defects with different
    # owners -- one is the sampler drawing the wrong donor variant, the other
    # would be a self-team construction bug -- and one predicate would rank them
    # together.
    transformed = _is_transformed(active)
    for observed_id, row in zip(observed_ids, rows):
        move = _lookup(observed_id)
        if move is None:
            out.append(
                ProjectionMismatch(
                    axis="self_move_set",
                    slot=observed.slot,
                    predicate=(
                        "self_move_set:request_move_absent_from_transformed_copy"
                        if transformed
                        else "self_move_set:request_move_absent_from_world"
                    ),
                    detail=_bounded(
                        f"request offers {observed_id}, world moveset "
                        f"{sorted(by_id)}"
                        + (" (active is a Transform copy)" if transformed else "")
                    ),
                )
            )
            continue
        pp = row.get("pp")
        if isinstance(pp, int) and int(move.pp) != pp:
            out.append(
                ProjectionMismatch(
                    axis="self_move_pp",
                    slot=observed.slot,
                    predicate=f"self_move_pp:{_norm(move.id)}",
                    detail=_bounded(
                        f"{_norm(move.id)}: request pp {pp} != world {int(move.pp)}"
                    ),
                )
            )
        disabled = bool(row.get("disabled"))
        if bool(move.disabled) != disabled:
            out.append(
                ProjectionMismatch(
                    axis="self_move_disabled",
                    slot=observed.slot,
                    predicate=f"self_move_disabled:{_norm(move.id)}",
                    detail=_bounded(
                        f"{_norm(move.id)}: request disabled={disabled} != world "
                        f"{bool(move.disabled)}"
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


#: Gen 3 Toxic charges ``floor(maxhp / 16) * stage`` at each residual, so the
#: multiplier is recoverable from the DAMAGE the protocol printed. That is the
#: only route to an observed side for this axis that is independent of the
#: constructor's input, and independence is the whole point -- see below.
_TOXIC_DENOMINATOR = 16


def observed_toxic_multiplier(lines: Sequence[str]) -> dict[str, int | None]:
    """The last Toxic multiplier each side actually PAID, read from raw damage.

    WHY THIS EXISTS, and what the first version of this axis got wrong.
    ------------------------------------------------------------------
    The first revision compared ``replay.toxic_stage[slot]`` against the engine's
    ``side_conditions.toxic_count``. **That was a tautology**, and independent
    review is what established it: ``local_showdown._materialization_toxic_stage``
    RETURNS ``min(14, max(0, tracked_stage - 1))`` where ``tracked_stage`` IS
    ``replay.toxic_stage[player]``, and ``engine_world`` writes that straight into
    ``side_conditions["toxic_count"]``. The comparison was ``x`` against ``f(x)``,
    the uniform +1 delta over 12,416 of 12,416 worlds was the forced output of
    that identity, and the axis had **zero power over #1209**.

    It was worse than powerless. Corrected to the documented convention it is
    silent by construction, so an over-broad #1209 -- one that credits every world
    an extra tick -- makes the oracle read *cleaner*, not dirtier. An axis whose
    response to a defect is to stop firing is an anti-instrument.

    So the observed side is rebuilt from a source the constructor never touches:
    the raw ``|-damage|SLOT|hp/max|[from] psn`` line. Gen 3 charges
    ``floor(maxhp / 16) * stage``, so ``stage = damage / floor(maxhp / 16)``, and
    that arithmetic passes through none of the parser's toxic trackers.

    Returns ``{slot: multiplier or None}``. ``None`` means *not determined* --
    no tick observed since the current active came in, a non-integral quotient,
    a percentage-mod HP grid too coarse to divide, or **a tick whose own
    condition token is plain ``psn`` rather than ``tox``**. An axis never fires
    on an undetermined value.
    """

    from .engine_fidelity import _parse_condition  # noqa: PLC0415

    last_hp: dict[str, int] = {}
    maxhp: dict[str, int] = {}
    multiplier: dict[str, int | None] = {"p1": None, "p2": None}
    for line in lines:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1]
        slot = parts[2].split(":", 1)[0].strip()[:2]
        if slot not in ("p1", "p2"):
            continue
        if tag in ("switch", "drag", "replace") and len(parts) >= 5:
            value, _token = _parse_condition(parts[4])
            last_hp[slot] = value
            maxhp[slot] = _condition_maxhp(parts[4]) or maxhp.get(slot, 0)
            # A gen 3 switch-out RESETS the counter, so nothing observed before
            # this line says anything about the mon now standing there.
            multiplier[slot] = None
            continue
        if tag not in ("-damage", "-heal", "-sethp") or len(parts) < 4:
            continue
        value, _token = _parse_condition(parts[3])
        maxhp[slot] = _condition_maxhp(parts[3]) or maxhp.get(slot, 0)
        previous = last_hp.get(slot)
        last_hp[slot] = value
        if tag != "-damage" or previous is None:
            continue
        if "[from] psn" not in line:
            continue
        # PLAIN POISON IS NOT TOXIC, and this side used to price it as though it
        # were. The PARSER applies this gate explicitly and this one did not --
        # `showdown._reseed_toxic_stage_from_residual` opens with
        # `if "tox" not in new_condition.split(): return`. Gen 3 plain poison
        # charges `maxhp / 8`, which is exactly `2 * (maxhp // 16)`, so a plain
        # `psn` tick divided cleanly and came back as TOXIC STAGE 2: a
        # fabricated value on the side of the comparison whose entire job is to
        # be observed. Only `_axis_toxic_count`'s engine-status gate stood
        # between that and a firing, and where the engine DOES say TOXIC the
        # fabricated 2 can MATCH a pre-tick counter of 2 and silently absorb a
        # real status disagreement -- the anti-instrument shape (a defect making
        # the oracle read cleaner) that this axis was rebuilt once to escape.
        if "tox" not in parts[3].split():
            multiplier[slot] = None
            continue
        unit = maxhp.get(slot, 0) // _TOXIC_DENOMINATOR
        damage = previous - value
        if unit <= 0 or damage <= 0 or damage % unit:
            # Non-integral: a percentage grid, or a tick that hit the HP floor.
            # Silence beats a fabricated stage.
            multiplier[slot] = None
            continue
        multiplier[slot] = damage // unit
    return multiplier


def _condition_maxhp(condition: str) -> int:
    head = str(condition or "").split(" ", 1)[0]
    if "/" not in head:
        return 0
    try:
        return int(head.split("/", 1)[1])
    except ValueError:
        return 0


def _axis_toxic_count(
    observed: ObservedPublicView, sides: Mapping[str, Any], engine: Any
) -> list[ProjectionMismatch]:
    """#1209's axis, with an observed side the constructor cannot have written.

    The engine's counter is PRE-tick and the engine charges ``toxic_count + 1``
    (``engine_world.py`` says so at the write site: *"a bridge-only pre-tick
    counter, not the public multiplier"*), so a mon that has just paid multiplier
    ``m`` must be holding counter ``m``. Compared only when a multiplier was
    actually recovered from a tick since this active came in, and only when the
    engine agrees the active is Toxic-statused.
    """

    out: list[ProjectionMismatch] = []
    for slot, side in sides.items():
        paid = observed.toxic_multiplier.get(slot)
        if paid is None:
            continue
        status = engine.p1_status if slot == "p1" else engine.p2_status
        if status != "TOXIC":
            continue
        got = int(getattr(side.side_conditions, "toxic_count", 0) or 0)
        if got != int(paid):
            out.append(
                ProjectionMismatch(
                    axis="toxic_count",
                    slot=slot,
                    predicate="toxic_count",
                    detail=_bounded(
                        f"last tick paid multiplier {int(paid)}, world pre-tick "
                        f"counter {got}"
                    ),
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
    pre_summary: Mapping[str, Any] | None = None,
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
    self_consistency = (
        render_self_consistency_mismatches(
            branches, slot_sides=slot_sides, pre_summary=pre_summary
        )
        if pre_summary is not None
        else []
    )
    usable = 0
    unusable_markers: Counter[str] = Counter()
    reasons: list[str] = []
    #: The reason set of the CLOSEST branch -- fewest disagreements -- which is
    #: the informative one when nothing matched.
    best: list[str] = []
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
        branch_reasons = _render_mismatch_reasons(observed, rendered, pre_features)
        if not branch_reasons:
            return self_consistency, {
                "branches": len(branches),
                "usable_branches": usable,
                "self_consistency_branches": len(
                    [b for b in branches
                     if isinstance(b.get("post"), Mapping)
                     and not b.get("attribution_unsafe")
                     and render_branch_is_usable(list(b.get("lossy") or []))]
                ),
                "matched": True,
                "self_consistency": len(self_consistency),
            }
        # ALL of them, not the first. The first revision returned on the first
        # difference and checked status BEFORE hp and side conditions, so on the
        # ~28% of boundaries where a status difference fired, the hp and
        # side-condition checks never ran at all -- masking, measured by review.
        best = min(best, branch_reasons, key=len) if best else branch_reasons
        reasons.extend(branch_reasons)

    diagnostics = {
        "branches": len(branches),
        "usable_branches": usable,
        "matched": False,
        "unusable_markers": dict(unusable_markers),
        "reasons": [_bounded(reason) for reason in best[:6]],
        "all_reasons": len(reasons),
        "self_consistency": len(self_consistency),
        "self_consistency_branches": len(
            [b for b in branches
             if isinstance(b.get("post"), Mapping)
             and not b.get("attribution_unsafe")
             and render_branch_is_usable(list(b.get("lossy") or []))]
        ),
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
            detail=_bounded(
                f"{usable} usable branches, closest branch disagreed on: {best}"
            ),
        )
    ], diagnostics


#: Every field `events.rs::post_state_summary` publishes per side. The
#: self-consistency arm compares ALL of them. The first revision compared
#: `active_hp` and nothing else, and independent review measured what that cost:
#: over a nine-branch probe, an `active_hp` disagreement and a faint were caught
#: while a status in `post` with no status line, a `-status brn` with no status
#: in `post`, a `-boost atk 2` against `post` 0, a benched mon at 0 hp and a
#: `|switch|` against `active_index 0` were ALL SILENT -- using facts already in
#: the payload being read, at zero extra cost.
#:
#: `force_switch` is the one published field still not compared, and the reason
#: is that the protocol has no line for it: it is a request-level fact the
#: renderer never emits, so there is nothing to fold against it. Side conditions
#: are not compared either because `post_state_summary` does not publish them --
#: the TRANSITION arm covers those against the log.
_BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")

_ENGINE_BOOST_ATTR = {
    "atk": "attack_boost",
    "def": "defense_boost",
    "spa": "special_attack_boost",
    "spd": "special_defense_boost",
    "spe": "speed_boost",
    "accuracy": "accuracy_boost",
    "evasion": "evasion_boost",
}

_BOOST_ALIAS = {"atk": "atk", "def": "def", "spa": "spa", "spd": "spd",
                "spe": "spe", "accuracy": "accuracy", "evasion": "evasion"}


def pre_state_summary(state: Any, slot_sides: Mapping[str, str]) -> dict[str, Any]:
    """The world's pre-branch state in exactly `post_state_summary`'s shape.

    Keyed by PLAYER SLOT. Built so the rendered lines can be folded ONTO it and
    the result compared field-for-field with the branch's own post-state; that
    is a much stronger and much simpler contract than comparing two independently
    derived summaries.
    """

    sides = _sides_by_slot(state, slot_sides)
    summary: dict[str, Any] = {}
    for slot, side in sides.items():
        party = _live_party(side)
        active_index = int(str(side.active_index))
        summary[slot] = {
            "active_index": active_index,
            "active_hp": int(side.pokemon[active_index].hp),
            "active_status": str(side.pokemon[active_index].status).lower(),
            "boosts": {
                key: int(getattr(side, _ENGINE_BOOST_ATTR[key], 0) or 0)
                for key in _BOOST_KEYS
            },
            "pokemon": [
                {"hp": int(mon.hp), "status": str(mon.status).lower()} for mon in party
            ],
            "species": [_norm(mon.id) for mon in party],
        }
    return summary


def fold_lines_onto_summary(
    summary: Mapping[str, Any], lines: Sequence[str]
) -> dict[str, Any]:
    """Apply one branch's rendered protocol lines to a pre-state summary."""

    import copy  # noqa: PLC0415

    from .engine_fidelity import _parse_condition  # noqa: PLC0415

    out = copy.deepcopy(dict(summary))

    def _write_active(slot: str, hp: int | None, status: str | None) -> None:
        side = out[slot]
        index = side["active_index"]
        if hp is not None:
            side["active_hp"] = hp
            if 0 <= index < len(side["pokemon"]):
                side["pokemon"][index]["hp"] = hp
        if status is not None:
            side["active_status"] = status
            if 0 <= index < len(side["pokemon"]):
                side["pokemon"][index]["status"] = status

    for line in lines:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1]
        slot = parts[2].split(":", 1)[0].strip()[:2]
        if slot not in out:
            continue
        if tag in ("switch", "drag", "replace") and len(parts) >= 5:
            species = _norm(str(parts[3]).split(",", 1)[0])
            candidates = [
                index for index, name in enumerate(out[slot]["species"]) if name == species
            ]
            if candidates:
                out[slot]["active_index"] = candidates[0]
            value, token = _parse_condition(parts[4])
            # A regular switch clears stat stages with NO protocol echo; the
            # engine emits reset_boosts instructions for it.
            out[slot]["boosts"] = {key: 0 for key in _BOOST_KEYS}
            _write_active(slot, value, _ENGINE_STATUS.get(token, "none"))
        elif tag in ("-damage", "-heal", "-sethp") and len(parts) >= 4:
            value, token = _parse_condition(parts[3])
            _write_active(slot, value, _ENGINE_STATUS.get(token) if token else None)
        elif tag == "-status" and len(parts) >= 4:
            _write_active(slot, None, _ENGINE_STATUS.get(parts[3].strip(), "none"))
        elif tag in _CURE_TAGS:
            _write_active(slot, None, "none")
            if tag == "-cureteam":
                for mon in out[slot]["pokemon"]:
                    mon["status"] = "none"
        elif tag == "faint":
            _write_active(slot, 0, None)
        elif tag in ("-boost", "-unboost", "-setboost") and len(parts) >= 5:
            key = _BOOST_ALIAS.get(parts[3].strip())
            if key is None:
                continue
            try:
                amount = int(parts[4])
            except ValueError:
                continue
            current = out[slot]["boosts"][key]
            if tag == "-setboost":
                out[slot]["boosts"][key] = amount
            else:
                delta = amount if tag == "-boost" else -amount
                out[slot]["boosts"][key] = max(-6, min(6, current + delta))
        elif tag in ("-clearboost", "-clearnegativeboost", "-invertboost"):
            out[slot]["boosts"] = {key: 0 for key in _BOOST_KEYS}
        elif tag == "-clearallboost":
            for other in out:
                out[other]["boosts"] = {key: 0 for key in _BOOST_KEYS}
    return out


#: Showdown status token -> the engine's lowercase spelling, which is what
#: `post_state_summary` prints.
_ENGINE_STATUS = {
    "": "none",
    "par": "paralyze",
    "brn": "burn",
    "psn": "poison",
    "tox": "toxic",
    "slp": "sleep",
    "frz": "freeze",
}


def render_self_consistency_mismatches(
    branches: Sequence[Mapping[str, Any]],
    *,
    slot_sides: Mapping[str, str],
    pre_summary: Mapping[str, Any],
) -> list[ProjectionMismatch]:
    """Does each branch's RENDER describe that branch's own outcome?

    The transition comparator needs the log, so it only ever sees the ONE branch
    the game took. Every other branch search spent budget on -- the
    counterfactuals the tree is made of -- is never compared to anything. This
    closes that: `branch_events` already returns each branch's post-state summary
    beside its rendered lines, so the render can be held against the engine's own
    outcome for the SAME branch, at zero extra cost and with no log required.

    This is the arm that sees a renderer-side relaxation. Render
    ``|-activate|...|Protect`` over an instruction the engine emitted as a real
    heal and the rendered lines leave HP where it started while the branch's own
    post-state has moved.

    NOW COMPARES EVERY FIELD ``post`` PUBLISHES -- active index, active HP,
    active status, all seven boosts, and per-mon HP and status for the whole
    party -- because the first revision compared one scalar per side and review
    demonstrated five distinct silent regions using facts already in the payload.

    WHAT IT STILL CANNOT SEE, disclosed rather than discovered later:

    * ``force_switch``: a request-level fact with no protocol line to fold.
    * side conditions: ``post_state_summary`` does not publish them. The
      TRANSITION arm compares those against the log.
    * an over-broad marker over a ZERO-amount heal. A full-HP absorber's no-op
      and Protect's no-op have the SAME post-state on every published field --
      that identity is what makes them ambiguous in the first place. No
      state-based comparator can separate them; only the per-source LINE
      decomposition in ``scripts/engine_transition_differential.py`` can.
    """

    out: list[ProjectionMismatch] = []
    #: The rendered lines of the first disagreeing branch, attached to the
    #: mismatch detail. An inventory row that cannot be adjudicated without
    #: re-running the census is a row nobody adjudicates -- report 4 section 2.1
    #: is four mechanisms that were all wrong until someone dumped the inputs.
    witness_lines: list[str] = []
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
        branch_lines = [
            line for line in (branch.get("events") or []) if line and line != "|"
        ]
        rendered = fold_lines_onto_summary(pre_summary, branch_lines)
        before = len(out)
        for slot in ("p1", "p2"):
            side_post = post.get(engine_label[slot])
            if not isinstance(side_post, Mapping):
                continue
            got = rendered[slot]
            for field in ("active_index", "active_hp", "active_status"):
                want = side_post.get(field)
                if want is None:
                    continue
                if _same_field(field, want, got[field]):
                    continue
                if field == "active_status" and _is_unrendered_cure(
                    want, got[field], pre_summary[slot]["active_status"]
                ):
                    continue
                out.append(
                    _render_disagreement(index, slot, field, want, got[field])
                )
            want_boosts = side_post.get("boosts") or {}
            for key in _BOOST_KEYS:
                if key not in want_boosts:
                    continue
                if int(want_boosts[key]) == int(got["boosts"][key]):
                    continue
                out.append(
                    _render_disagreement(
                        index, slot, f"boost:{key}", want_boosts[key], got["boosts"][key]
                    )
                )
            want_party = side_post.get("pokemon") or []
            if len(want_party) != len(got["pokemon"]):
                continue
            for party_index, (want_mon, got_mon) in enumerate(
                zip(want_party, got["pokemon"])
            ):
                pre_party = pre_summary[slot]["pokemon"]
                for field in ("hp", "status"):
                    if want_mon.get(field) is None:
                        continue
                    if _same_field(field, want_mon[field], got_mon[field]):
                        continue
                    if field == "status" and party_index < len(pre_party) and (
                        _is_unrendered_cure(
                            want_mon[field],
                            got_mon[field],
                            pre_party[party_index]["status"],
                        )
                    ):
                        continue
                    out.append(
                        _render_disagreement(
                            index,
                            slot,
                            f"party_{field}",
                            want_mon[field],
                            got_mon[field],
                            extra=f"party slot {party_index}",
                        )
                    )
        if len(out) > before and not witness_lines:
            witness_lines = branch_lines[:16]
    if witness_lines and out:
        # Carried in the FIRST row's detail rather than as a row of its own: a
        # witness that is itself counted would inflate every figure it explains.
        first = out[0]
        out[0] = ProjectionMismatch(
            axis=first.axis,
            slot=first.slot,
            predicate=first.predicate,
            detail=first.detail + " || rendered: " + " ".join(witness_lines)[:400],
        )
    return out


def _is_unrendered_cure(want: Any, got: Any, pre_status: Any) -> bool:
    """A status the branch CLEARED, which the renderer deliberately does not say.

    ``events.rs``'s own header lists the lines the fold provably ignores and that
    are therefore never emitted: ``|-singleturn|``, **``|-curestatus|``**,
    ``|-fail|``, ``|-ability|``, ``|-enditem|``, ``|-mustrecharge|``,
    ``|-start|`` (except absorb signatures), ``|-anim|``, ``|debug|``. So a
    branch in which the sleeper WOKE has ``post.status == "none"`` while the
    render still shows the pre-state's sleep, and that is a documented omission,
    not a fabricated fact.

    Measured the moment the status axis was added: the very first run of the
    #1211 fixture fired twice on exactly this, on the BASE crate, in the wake-up
    branch.

    NARROW BY CONSTRUCTION, and the narrowness is the point. It excludes only
    ``post`` says NONE **and** the render is still showing the status the world
    came in with. A render that ASSERTS a status ``post`` denies, or omits one
    ``post`` gained, is untouched -- and those are the two directions a
    fabricating renderer would move in.
    """

    return (
        str(want).lower() == "none"
        and str(got).lower() != "none"
        and str(got).lower() == str(pre_status).lower()
    )


def _same_field(field: str, want: Any, got: Any) -> bool:
    if field.endswith("status"):
        return str(want).lower() == str(got).lower()
    return int(want) == int(got)


def _render_disagreement(
    branch_index: int, slot: str, field: str, want: Any, got: Any, extra: str = ""
) -> ProjectionMismatch:
    tail = f" ({extra})" if extra else ""
    return ProjectionMismatch(
        axis="render_post_state_disagreement",
        slot=slot,
        predicate=f"render_post_state_disagreement:{field}",
        detail=_bounded(
            f"branch {branch_index}{tail}: render says {field}={got}, the branch's "
            f"own post-state says {want}"
        ),
    )


def _render_mismatch_reasons(
    observed: StepProjection, rendered: StepProjection, pre: Any
) -> list[str]:
    """EVERY way this branch's render disagrees with the observed step.

    Returns a list, and an empty list means match. The first revision returned
    the FIRST reason and tested status before HP and side conditions, which
    masked both on every boundary where a status difference fired -- and, given
    the `fold_step_lines` status bug that has now been fixed, that was ~28% of
    them.

    Exact on everything deterministic; banded on HP only. The band is anchored on
    this step's HP movement, as `engine_fidelity` anchors it: an engine branch
    carries the representative damage roll while Showdown sampled one of sixteen.

    THE BAND'S WIDTH IS THE DOMINANT DETERMINANT OF THE LARGEST RENDER FIGURE --
    HP is 159 of the 206 `render_unmatched_transition` reasons -- and it was
    pinned by nothing. Measured: `_DAMAGE_TOLERANCE` could be widened 0.16 ->
    0.75 (4.7x) and `_MIN_TOLERANCE_HP` 5 -> 40 (8x) with the whole module green,
    and TIGHTENED to 0 with the whole module green too, so the boundary was open
    in both directions. `RenderBandWidthTests` now pins both constants from both
    sides at the two anchors where the floor and the proportional term each
    dominate; widen or tighten either and a named test fails.

    WEATHER IS NOT COMPARED. Both sides fold ONE step, an absent `|-weather|` is
    indistinguishable from "unchanged", and the axis would only ever fire on an
    upkeep-line rendering convention. The STATE comparator does compare weather,
    against the whole-log fold, where it is well defined.
    """

    from .engine_fidelity import _DAMAGE_TOLERANCE, _MIN_TOLERANCE_HP  # noqa: PLC0415

    out: list[str] = []
    if observed.fainted != rendered.fainted:
        out.append(f"fainted {sorted(observed.fainted)} != {sorted(rendered.fainted)}")
    for slot, start_hp in (("p1", pre.p1_hp), ("p2", pre.p2_hp)):
        obs, ren = observed.hp[slot], rendered.hp[slot]
        if obs < 0 or ren < 0 or obs == ren:
            continue
        moved = max(abs(int(start_hp) - obs), abs(int(start_hp) - ren))
        tolerance = max(_MIN_TOLERANCE_HP, int(moved * _DAMAGE_TOLERANCE) + 1)
        if abs(obs - ren) > tolerance:
            out.append(f"{slot} hp {obs} != {ren} (moved {moved}, tolerance {tolerance})")
    if observed.side_conditions != rendered.side_conditions:
        out.append(
            f"side conditions {observed.side_conditions} != {rendered.side_conditions}"
        )
    for slot in ("p1", "p2"):
        if slot in observed.fainted:
            continue
        obs, ren = observed.status[slot], rendered.status[slot]
        if obs.startswith("?") or obs == ren:
            continue
        out.append(f"{slot} status {obs} != {ren}")
    return out


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
    #: BOUNDARIES per axis -- a boundary is counted ONCE however many rows it
    #: carried. See `render_axis_rows` for the other unit and why they differ.
    render_axis: Counter[str] = Counter()
    #: ROWS per axis: one per `ProjectionMismatch`. `render_post_state_disagreement`
    #: emits one row per (branch, slot, field), so this is several times the
    #: boundary count and is NOT a boundary figure.
    render_axis_rows: Counter[str] = Counter()
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
                # Two different units off one list, kept apart at the source.
                # Incrementing ONE counter per row and publishing it under a
                # `_boundaries` name is the bug this replaces: it labelled 80
                # rows as 80 boundaries, over 23 boundaries -- a 3.5x
                # overstatement in the artifact the plan calls the deliverable.
                for axis in axes:
                    render_axis_rows[str(axis)] += 1
                for axis in set(str(axis) for axis in axes):
                    render_axis[axis] += 1

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
        # BOUNDARIES per axis: distinct boundaries, so the key's own name is true.
        "render_axis_boundaries": dict(render_axis.most_common()),
        # ROWS per axis: one per mismatch row. A DIFFERENT UNIT, published beside
        # the boundary count rather than instead of it, because the row count is
        # the honest size of the render arm's output and the boundary count is
        # the honest denominator-comparable figure.
        "render_axis_rows": dict(render_axis_rows.most_common()),
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
