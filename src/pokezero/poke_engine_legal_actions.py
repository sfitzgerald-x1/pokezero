"""Fixture-only legal-action equivalence for curated Gen 3 singles.

This is a narrow, optional seam for the poke-engine evaluation spike (step 3 of
``docs/poke_engine_assessment.md``). It derives the legal action set two ways and
compares them:

- From a curated Showdown-style **request** payload (the input side a real client
  receives each turn): active moves in request order minus disabled ones, plus the
  legal bench switches.
- From an adapter-built ``poke_engine.State`` via the **engine's own** root-option
  enumeration, *if the binding exposes one*.

The two derivations are intentionally independent: the request side reads the
Showdown payload, and the engine side resolves the engine's option output against
the engine state. We never derive both sides from the same fixture data, so a
match is evidence of agreement rather than a tautology.

Scope is **singles only**. The module is optional and lazy: importing it never
requires the native wheel, and the engine probe degrades to a clearly-explained
``supported=False`` result when the Python binding does not export legal options
(which, as of poke-engine 0.0.47, it does not -- see ``engine_legal_actions``).

It is deliberately disconnected from rollout, training, search, and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .poke_engine_backend import PokeEngineUnavailableError, require_poke_engine

# The two seats of a singles battle, matching ``poke_engine.State`` attributes.
SIDE_NAMES = ("side_one", "side_two")

# Candidate names a future PyO3 export of ``State::root_get_all_options`` might use.
# Probed (in order) on the supplied option provider, the state, then the module.
# poke-engine 0.0.47's Python binding exposes none of these, so the real engine
# path reports an actionable unsupported result rather than a fake comparison.
ENGINE_OPTION_PROVIDER_CANDIDATES = (
    "get_all_options",
    "root_get_all_options",
    "get_root_options",
)

ENGINE_UNSUPPORTED_REASON = (
    "poke-engine's Python binding does not expose root legal-option enumeration "
    "(probed: {candidates}). Lower-level legal-option export is the next step: a small "
    "PyO3 wrapper over State::root_get_all_options would let this compare true engine "
    "options instead of reporting unsupported."
).format(candidates=", ".join(ENGINE_OPTION_PROVIDER_CANDIDATES))


def move_action_label(move_id: str) -> str:
    """Stable label for a move action, normalized to poke-engine id style."""

    return f"move:{_normalize_id(move_id)}"


def switch_action_label(team_index: int) -> str:
    """Stable label for a switch action, keyed by party (team) index.

    The request side uses Showdown's party index. Future engine option exports
    must confirm whether their switch indices are also party indices or reserve
    indices before treating these labels as comparable.
    """

    return f"switch:{int(team_index)}"


@dataclass(frozen=True)
class EngineLegalActions:
    """Legal actions derived from the engine state, or why they were unavailable."""

    supported: bool
    actions: tuple[str, ...]
    reason: str | None = None


@dataclass(frozen=True)
class LegalActionEquivalence:
    """Result of comparing request-derived vs. engine-derived legal actions.

    ``missing_from_engine`` are actions Showdown offers that the engine did not;
    ``extra_from_engine`` are actions the engine offered that Showdown did not.
    When the engine cannot enumerate options, ``supported`` is ``False`` and
    ``reason`` explains what is missing; the comparison fields are left empty so
    callers never mistake an unsupported probe for an agreement.
    """

    supported: bool
    request_actions: tuple[str, ...]
    engine_actions: tuple[str, ...]
    missing_from_engine: tuple[str, ...]
    extra_from_engine: tuple[str, ...]
    reason: str | None = None

    @property
    def equivalent(self) -> bool:
        return self.supported and not self.missing_from_engine and not self.extra_from_engine

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "equivalent": self.equivalent,
            "request_actions": list(self.request_actions),
            "engine_actions": list(self.engine_actions),
            "missing_from_engine": list(self.missing_from_engine),
            "extra_from_engine": list(self.extra_from_engine),
            "reason": self.reason,
        }


def request_legal_actions(request: Mapping[str, Any]) -> tuple[str, ...]:
    """Derive expected legal action labels from a singles Showdown request payload.

    Active moves are emitted in request order, skipping ``disabled`` ones. Legal
    switches are bench slots (party index) that are not active, not fainted, and
    not blocked by confirmed ``trapped``. ``maybeTrapped`` is not conclusive and
    the Showdown client still offers switches in that state. ``forceSwitch``
    overrides move selection, in which case only switches are legal. A ``wait``
    request has no actions. Raises ``ValueError`` for non-singles payloads.
    """

    if not isinstance(request, Mapping):
        raise TypeError(f"request must be a mapping, got {type(request).__name__}")
    if request.get("wait"):
        return ()

    force_switch = _force_switch_requested(request)
    actions: list[str] = []
    if not force_switch:
        actions.extend(_request_move_labels(request))
    if force_switch or _switching_allowed(request):
        actions.extend(_request_switch_labels(request))
    return tuple(actions)


def _force_switch_requested(request: Mapping[str, Any]) -> bool:
    force_switch = request.get("forceSwitch")
    if not isinstance(force_switch, list):
        return False
    if len(force_switch) > 1:
        raise ValueError("singles only: forceSwitch carries more than one slot")
    return any(bool(slot) for slot in force_switch)


def _active_row(request: Mapping[str, Any]) -> Mapping[str, Any] | None:
    active_rows = request.get("active")
    if active_rows is None:
        return None
    if not isinstance(active_rows, list):
        raise TypeError("request 'active' must be a list")
    if len(active_rows) > 1:
        raise ValueError("singles only: request carries more than one active slot")
    if not active_rows:
        return None
    row = active_rows[0]
    return row if isinstance(row, Mapping) else None


def _request_move_labels(request: Mapping[str, Any]) -> list[str]:
    active = _active_row(request)
    moves = active.get("moves") if isinstance(active, Mapping) else None
    if not isinstance(moves, list):
        return []
    labels: list[str] = []
    for move in moves:
        if not isinstance(move, Mapping) or move.get("disabled"):
            continue
        labels.append(move_action_label(_request_move_id(move)))
    return labels


def _switching_allowed(request: Mapping[str, Any]) -> bool:
    active = _active_row(request)
    if isinstance(active, Mapping) and active.get("trapped") is True:
        return False
    return True


def _request_switch_labels(request: Mapping[str, Any]) -> list[str]:
    side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
    pokemon = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(pokemon, list):
        return []
    labels: list[str] = []
    for team_index, member in enumerate(pokemon):
        if isinstance(member, Mapping) and _can_switch_to(member):
            labels.append(switch_action_label(team_index))
    return labels


def _can_switch_to(member: Mapping[str, Any]) -> bool:
    if member.get("active"):
        return False
    return not _is_fainted(member)


def _is_fainted(member: Mapping[str, Any]) -> bool:
    condition = str(member.get("condition") or "")
    return condition.startswith("0 ") or condition == "0" or "fnt" in condition


def _request_move_id(move: Mapping[str, Any]) -> str:
    for key in ("id", "move"):
        value = move.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"request move is missing an id/move name: {move!r}")


def engine_legal_actions(
    state: Any,
    side: str,
    *,
    module: Any | None = None,
    option_provider: Callable[[Any], Any] | None = None,
) -> EngineLegalActions:
    """Derive legal action labels from a built engine ``state`` for one seat.

    Looks for a root-option enumerator in this order: the explicit
    ``option_provider`` argument, then a method on ``state``, then a function on
    ``module`` (or the lazily-imported real engine when ``module`` is ``None``),
    using :data:`ENGINE_OPTION_PROVIDER_CANDIDATES`. The provider must return a
    pair of per-seat option lists ``(side_one_options, side_two_options)``; each
    option is resolved against the engine state (not the request fixture), so a
    later match is real agreement.

    When no enumerator is exposed -- the case for poke-engine 0.0.47 -- this returns
    ``supported=False`` with an actionable ``reason`` instead of raising.
    """

    if side not in SIDE_NAMES:
        raise ValueError(f"side must be one of {SIDE_NAMES}, got {side!r}")

    provider = option_provider or _resolve_option_provider(state, module)
    if provider is None:
        return EngineLegalActions(supported=False, actions=(), reason=ENGINE_UNSUPPORTED_REASON)

    options_by_side = provider(state)
    side_options = _select_side_options(options_by_side, side)
    actions = tuple(_resolve_engine_option(state, side, option) for option in side_options)
    return EngineLegalActions(supported=True, actions=actions, reason=None)


def _resolve_option_provider(state: Any, module: Any | None) -> Callable[[Any], Any] | None:
    for name in ENGINE_OPTION_PROVIDER_CANDIDATES:
        bound = getattr(state, name, None)
        if callable(bound):
            return lambda _state, _bound=bound: _bound()

    engine = module
    if engine is None:
        try:
            engine = require_poke_engine()
        except PokeEngineUnavailableError:
            return None
    for name in ENGINE_OPTION_PROVIDER_CANDIDATES:
        fn = getattr(engine, name, None)
        if callable(fn):
            return fn
    return None


def _select_side_options(options_by_side: Any, side: str) -> Sequence[Any]:
    if not isinstance(options_by_side, Sequence) or len(options_by_side) != len(SIDE_NAMES):
        raise ValueError(
            "engine option provider must return a (side_one, side_two) pair, "
            f"got {type(options_by_side).__name__}"
        )
    return options_by_side[SIDE_NAMES.index(side)]


def _resolve_engine_option(state: Any, side: str, option: Any) -> str:
    """Translate one engine root option into a label using the engine state.

    Supports the contract a future ``root_get_all_options`` export would plausibly
    expose: each option carries ``kind`` ("move"/"switch") with either a move slot
    index (``move_index``) or a party index (``switch_index``). If a future
    binding exposes reserve-relative switch indices instead, this resolver should
    be updated before equivalence results are considered binding.
    """

    kind = getattr(option, "kind", None)
    if kind == "switch":
        return switch_action_label(getattr(option, "switch_index"))
    if kind == "move":
        seat = getattr(state, side)
        active_index = int(seat.active_index)
        move_id = seat.pokemon[active_index].moves[getattr(option, "move_index")].id
        return move_action_label(move_id)
    raise ValueError(f"unrecognized engine option {option!r} (expected kind 'move' or 'switch')")


def compare_legal_actions(
    request: Mapping[str, Any],
    state: Any,
    side: str,
    *,
    module: Any | None = None,
    option_provider: Callable[[Any], Any] | None = None,
) -> LegalActionEquivalence:
    """Compare request-derived vs. engine-derived legal actions for one seat.

    Returns a :class:`LegalActionEquivalence`. When the engine cannot enumerate
    options, ``supported`` is ``False`` and the comparison fields are empty.
    """

    request_actions = request_legal_actions(request)
    engine = engine_legal_actions(state, side, module=module, option_provider=option_provider)
    if not engine.supported:
        return LegalActionEquivalence(
            supported=False,
            request_actions=request_actions,
            engine_actions=(),
            missing_from_engine=(),
            extra_from_engine=(),
            reason=engine.reason,
        )

    engine_set = set(engine.actions)
    request_set = set(request_actions)
    missing = tuple(action for action in request_actions if action not in engine_set)
    extra = tuple(action for action in engine.actions if action not in request_set)
    return LegalActionEquivalence(
        supported=True,
        request_actions=request_actions,
        engine_actions=engine.actions,
        missing_from_engine=missing,
        extra_from_engine=extra,
        reason=None,
    )


def _normalize_id(value: str) -> str:
    """Lowercase and strip to alphanumerics, matching poke-engine id style."""

    return "".join(ch for ch in str(value).lower() if ch.isalnum())
