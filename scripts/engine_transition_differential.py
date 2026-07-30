#!/usr/bin/env python
"""Game-level Showdown-vs-poke-engine TRANSITION differential over fresh random games.

This is the tier-2 real-game sweep that ``docs/engine_fidelity_findings.md``
lists as "Next": instead of curated fixtures, it plays whole
``gen3randombattle`` games in the real Node sim with uniform-random legal
actions and, at EVERY full decision boundary (both seats act), asks one
question:

    does the transition Showdown actually took lie in the branch support
    ``poke_engine.generate_instructions`` enumerates for the same joint action,
    from the same state?

Design points that make this different from
:mod:`pokezero.engine_fidelity_multiturn`:

* **Fresh world per boundary, not a followed engine trajectory.** The engine
  state is rebuilt at every boundary through the PRODUCTION world constructor
  (``engine_world.world_battle_spec`` with the game's TRUE packed teams as a
  fixed ``BattleStartOverride`` — the omniscient world, no belief sampling).
  Roll drift therefore never accumulates, and the constructor itself is under
  test on live states rather than on hand-built specs.
* **Pre-state gate.** A boundary only receives a transition verdict when the
  constructed engine pre-state matches Showdown's observed pre-state exactly
  (active HP, status, weather, side-condition presence). Boundaries that fail
  the gate are reported separately as world-construction divergences, so
  constructor error is never charged to the engine's transition model (and
  never silently passes either).
* **Fail-closed outcomes are counted, not hidden.** ``EngineWorldUnsupported``
  reasons, single-seat (force-switch) boundaries and unmappable choices each
  get their own counted bucket. Public-information limits use their existing
  ``limit:*`` family rather than masquerading as an engine transition result.

Matching reuses the shipped matchers: exact boost-delta filter
(:func:`pokezero.engine_fidelity_multiturn.observed_boost_deltas`), then exact
status/faint/weather/side-condition equality with the +/-16%-of-this-turn's-
damage HP band (:func:`pokezero.engine_fidelity._mismatch_reason`). Because the
pre-state is exact, the band scales to a single turn's damage — but it is still
a band, and a sub-band systematic damage error would pass (documented limit,
inherited from the one-turn harness).

Read-only: no training, no search, no production behavior. Measurement only.

Usage::

    PYTHONPATH=src python scripts/engine_transition_differential.py \\
        --showdown-root <showdown> --games 200 --seed-start 900000 \\
        --json report.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import random
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import poke_engine  # noqa: E402
import pokezero_search  # noqa: E402

from pokezero.dex import load_showdown_dex, normalize_id  # noqa: E402
from pokezero.engine_fidelity import (  # noqa: E402
    _DAMAGE_TOLERANCE,
    _MIN_TOLERANCE_HP,
    TurnFeatures,
    _engine_side_conditions,
    _features_payload,
    showdown_turn_features,
)
from pokezero.engine_fidelity_multiturn import observed_boost_deltas  # noqa: E402
from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    unpack_team,
    world_battle_spec,
)
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.golden_corpus import _true_teams_from_bridge_snapshot  # noqa: E402
from pokezero.local_showdown import (  # noqa: E402
    DEFAULT_SHOWDOWN_ROOT,
    LocalShowdownConfig,
    LocalShowdownEnv,
    _public_materialization_payload,
)
from pokezero.poke_engine_adapter import build_poke_engine_state  # noqa: E402
from pokezero.randbat import Gen3RandbatSource, canonical_gen3_randbat_species_id  # noqa: E402

# Engine "no action" choice string (waiting seat / MUSTRECHARGE).
_ENGINE_NO_MOVE = "none"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from engine_build_fingerprint import assert_fresh  # noqa: E402
from fidelity_gate_events import truant_loaf_slots  # noqa: E402

_ENGINE_BOOST_FIELDS = (
    "attack_boost",
    "defense_boost",
    "special_attack_boost",
    "special_defense_boost",
    "speed_boost",
    "accuracy_boost",
    "evasion_boost",
)


# ---------------------------------------------------------------------------------------------
# Feature extraction keyed by PLAYER SLOT (the engine's side_one/side_two assignment
# is an EngineWorld detail; the protocol is always p1/p2).
# ---------------------------------------------------------------------------------------------


def _sides_by_slot(state: Any, slot_sides: Mapping[str, str]) -> dict[str, Any]:
    sides = {"side_one": state.side_one, "side_two": state.side_two}
    return {slot: sides[slot_sides[slot]] for slot in ("p1", "p2")}


def engine_features_by_slot(state: Any, slot_sides: Mapping[str, str]) -> TurnFeatures:
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


def engine_boost_deltas_by_slot(
    before: Any, after: Any, slot_sides: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    pre = _sides_by_slot(before, slot_sides)
    post = _sides_by_slot(after, slot_sides)
    deltas: dict[str, dict[str, int]] = {}
    for slot in ("p1", "p2"):
        stats: dict[str, int] = {}
        for name in _ENGINE_BOOST_FIELDS:
            delta = int(getattr(post[slot], name, 0) or 0) - int(getattr(pre[slot], name, 0) or 0)
            if delta:
                stats[name[: -len("_boost")]] = delta
        deltas[slot] = stats
    return deltas


def _fold(lines: Sequence[str]) -> TurnFeatures:
    return showdown_turn_features(types.SimpleNamespace(protocol_lines=tuple(lines)))


# ---------------------------------------------------------------------------------------------
# Choice resolution: chosen action index -> engine move/switch string.
# ---------------------------------------------------------------------------------------------


class UnmappableChoice(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def engine_choice_for_action(
    *,
    action_index: int,
    candidates: Sequence[Mapping[str, Any]],
    engine_side: Any,
) -> str:
    """Translate one seat's chosen action index into an engine choice string.

    Move ids are resolved against the BUILT engine active's own move list (which
    is how Hidden Power's typed+BP engine id is recovered from the request's
    plain ``hiddenpower``); switch targets are resolved against the engine party
    order, with the cosmetic-forme collapse the world constructor applies.
    """

    candidate = next(
        (
            row
            for row in candidates
            if isinstance(row, Mapping) and row.get("action_index") == action_index
        ),
        None,
    )
    if candidate is None:
        raise UnmappableChoice("no_candidate_row")
    kind = candidate.get("kind")
    if kind == "move":
        move_id = normalize_id(str(candidate.get("move_id") or ""))
        if not move_id:
            raise UnmappableChoice("blank_move_id")
        active = engine_side.pokemon[int(str(engine_side.active_index))]
        engine_moves = [normalize_id(str(move.id)) for move in active.moves]
        if move_id in engine_moves:
            return move_id
        if move_id.startswith("hiddenpower"):
            typed = [m for m in engine_moves if m.startswith("hiddenpower")]
            if len(typed) == 1:
                return typed[0]
            raise UnmappableChoice("hidden_power_ambiguous")
        if move_id == "recharge":
            # A recharging seat carries MUSTRECHARGE, under which the engine's own
            # option surface offers only "No Move" — the recharge is not a
            # submittable move id (passing "recharge" raises "Invalid move for sN").
            return _ENGINE_NO_MOVE
        if move_id == "struggle":
            # Struggle is engine-INTERNAL: gen3 MoveChoice::from_string resolves
            # only ids present on the active's move list, and the engine
            # substitutes Struggle itself when nothing else is usable. There is no
            # choice string for it, so the boundary is not drivable by this harness.
            raise UnmappableChoice("struggle_not_submittable")
        raise UnmappableChoice(f"move_not_in_engine_set:{move_id}")
    if kind == "switch":
        pokemon = candidate.get("pokemon")
        species = (
            normalize_id(str(pokemon.get("species") or ""))
            if isinstance(pokemon, Mapping)
            else ""
        )
        if not species:
            raise UnmappableChoice("blank_switch_species")
        # gen3 ``MoveChoice::from_string`` (third_party/poke-engine-src/src/gen3/
        # state.rs:51) resolves a switch by matching ``pkmn.id`` on the BUILT
        # state, and it takes the BARE species id — the ``"switch <species>"``
        # form raises ValueError("Invalid move for sN").
        #
        # Resolve against the built party's real ids, NOT ``EngineWorld.
        # party_species``: the two disagree for cosmetic Unown formes, where the
        # world reports ``unownb`` while the engine stores the collapsed
        # ``unown``. Species clause keeps the collapsed key unique per team.
        party = [normalize_id(str(mon.id)) for mon in engine_side.pokemon]
        if species in party:
            return species
        canonical = canonical_gen3_randbat_species_id(species)
        matches = [s for s in party if canonical_gen3_randbat_species_id(s) == canonical]
        if len(matches) == 1:
            return matches[0]
        raise UnmappableChoice("switch_species_not_in_party")
    raise UnmappableChoice(f"unknown_kind:{kind}")


# ---------------------------------------------------------------------------------------------
# STRICT matcher: per-damage-source comparison instead of a net-HP band.
#
# The banded matcher compared NET active HP within +/-16 % of the turn's damage.
# That is unsound in both directions, and Appendix A.3's D11 scenario proved it
# empirically: a Leftovers tick riding on an attack manufactured a 12-39 %
# apparent "bias" that vanished once the item was removed. A band can hide a real
# error and invent a fake one.
#
# The strict matcher decomposes BOTH sides of the comparison into per-source
# components and requires each to agree:
#
#   Showdown side: every |-damage|/|-heal| line carries its own attribution in a
#     `[from]` tag ("[from] psn", "[from] Sandstorm", "[from] item: Leftovers",
#     "[from] Leech Seed", "[from] Spikes", "[from] Recoil", ...). A bare
#     |-damage| is direct move damage; a bare |-heal| is a move heal.
#   Engine side: the SAME vocabulary, produced by the shipped instruction->event
#     mapper (`pokezero_search.branch_events`, PR #727), which renders a branch's
#     instruction list as protocol lines with attribution.
#
# Comparing rendered-vs-real protocol keeps the two sides in one vocabulary
# instead of guessing at structural positions in the instruction list.
#
# EXACT vs ROLL-SCALED. Everything that is a deterministic fraction — status
# residuals, weather chip, Leftovers, Leech Seed, hazards, move heals — must
# match to the HP point.
#
# Roll-scaled components (direct move damage, recoil, drain, confusion self-hit)
# are accepted when they are, IN ORDER:
#   1. equal to the engine's value, OR
#   2. a member of the enumerated legal roll set — gen3 computes
#      `floor(base * random(85,100) / 100)` and `poke_engine.calculate_damage`
#      returns the base for non-crit and crit, so the set is enumerable, OR
#   3. within +/-9 % of the engine's representative roll.
#
# Rung 3 is a band, and saying otherwise would be an over-claim. The baseline
# legal set is computed from the PRE-state, but a mapper branch that switches or
# changes a stat before direct damage is repriced from its branch-local post-state
# below. The honest description of the predicate is "equal, or in the enumerated
# legal set, or within +/-9 % of the engine's representative roll". What is
# genuinely gone is the NET-HP band: every tolerance here is scoped to a single
# roll-scaled component, and no deterministic component gets any tolerance at all.
# ---------------------------------------------------------------------------------------------

# Components that scale with the damage roll; everything else must be exact.
#
# ``heal_to_full`` is the subtle one. A move heal that CAPS at max HP (Rest, and
# Recover/Soft-Boiled/Morning Sun when the mon is above half) restores
# ``maxhp - hp``, so its magnitude is set by whatever damage landed earlier in
# the SAME turn — it inherits that hit's roll. Demanding an exact match on it was
# a matcher defect, not an engine bug: it produced the whole
# "move-heal invisible to the engine branch" class (ledger B.4, seed 1310001
# step 72 — Showdown healed 251 from 2 HP, the engine healed 247 from 6 HP, same
# mechanic, different Surf roll). A bare heal that does NOT reach full is a pure
# fraction (Recover = maxhp/2) and stays EXACT.
_ROLL_SCALED_SOURCES = frozenset(
    {"", "recoil", "drain", "confusion", "capped_lethal", "move_unknown_callee"}
)

# The mapper cannot recover WHICH move Sleep Talk called from the instruction
# delta (rust/pokezero-search/src/events.rs:1230, "documented insufficiency"), so
# it flags the branch and renders the called move's damage with the generic
# `[from] residual` tag. That routed real move damage into the EXACT bucket,
# where it can never match Showdown's bare `-damage` line.
#
# It does not have to. The engine still BRANCHES over the candidate call set, and
# each branch carries a concrete called move's damage — replaying two residue
# rows showed the matching branch plainly present (seed 1350014 step 55: −78
# exact; seed 1350019 step 99: −97 against Showdown's −103, inside the roll
# window). The information needed to validate is there; only the LABEL is
# missing. So within such a branch the unattributed damage is reclassified as
# roll-scaled move damage of an unknown callee, and the realized outcome is
# validated against the UNION of the candidate branches' supports — the same
# support-based principle already used for hidden counters.
#
# IMPLEMENTED PREDICATE, stated exactly: this reclassifies ANY `-damage` line
# inside a Sleep-Talk-flagged branch whose source fell through to the mapper's
# generic `residual` tag. It is NOT "the called move's damage" — nothing in the
# rendered stream identifies which line the callee produced, which is the whole
# reason the branch is flagged. In practice the callee's damage is the only
# unattributed damage there, because every other gen3 residual the mapper can
# emit is named (psn/brn/Sandstorm/Hail/Leech Seed/partialtrap); the fall-through
# is reachable only for a residual with no cause branch, and the two candidates
# (Nightmare, Ghost-Curse) are both absent from the gen3 randbats pool. So the
# broader predicate is latent, not exercised — but it is the predicate, and
# `test_named_residual_is_NOT_reclassified` pins the boundary.
_SLEEPTALK_LOSSY_MARKER = "sleeptalk_called_unidentified"
_UNATTRIBUTED_DAMAGE_SOURCE = "residual"
_UNKNOWN_CALLEE_SOURCE = "move_unknown_callee"
# Sources whose rendering the mapper is known not to reproduce line-for-line;
# counted and excluded rather than silently mismatched.
_IGNORED_SOURCES = frozenset({"lockedmove"})

# Showdown tags a partial-trap tick with the MOVE that caused it
# ("[from] move: Wrap"); the engine carries only a generic PARTIALLYTRAPPED
# volatile and its mapper tags it "partiallytrapped". The move identity is not
# recoverable engine-side and does not affect state — every gen3 Wrap-class move
# ticks maxhp/16 — so both sides normalize to one canonical source. Without this
# the pair read as a component MISMATCH on 11 % of divergences (18 rows of the
# 1350000-1350059 census) purely on naming.
_PARTIAL_TRAP_SOURCES = frozenset({
    "partiallytrapped", "movewrap", "movebind", "movefirespin",
    "moveclamp", "movewhirlpool", "movesandtomb",
})
_CANONICAL_PARTIAL_TRAP = "partialtrap"


def damage_components(
    lines: Sequence[str],
    initial_hp: Mapping[str, int] | None = None,
    *,
    unattributed_damage_as_roll: bool = False,
) -> dict[str, list[tuple[str, int]]]:
    """Per-source HP deltas from a protocol slice, keyed by slot.

    ``initial_hp`` seeds the running HP per slot with the PRE-STEP value. Without
    it the first HP line for a slot only established the baseline and its delta
    was dropped — which silently hid the step's PRIMARY move damage whenever the
    slot had no earlier line (no switch, no second hit), i.e. most steps. Found
    by replaying a `roll_scaled_component` row (seed 1340001 step 47): the label
    said "roll disagreement", the replay showed the Hidden Power damage was not
    in the component list at all.

    ``-sethp`` is consumed alongside ``-damage``/``-heal``. Omitting it silently
    dropped Pain Split's HP change from the observation and folded its effect
    into the NEXT attributed delta on that slot, so the instrument reported
    impossible components such as a Leftovers heal of -73 and blamed the engine
    for them (seed 1500008 step 101: true Leftovers +13/+25, harness said
    +9/+28, engine said +13/+25 — the engine was right). Pain Split is the only
    move in the gen3 pool that emits the tag, and it emits TWO lines, one per
    slot, the target's first and `[silent]`; both must be read.

    Returns ``{"p1": [(source, delta), ...], "p2": [...]}`` where ``delta`` is
    signed (negative = damage) and ``source`` is the normalized ``[from]`` tag
    ("" for a bare damage line = direct move damage, "heal" for a bare heal).
    Deltas are computed against the running HP for that slot, so a slot's
    components sum to its net change and each is independently comparable.
    """

    running: dict[str, int] = dict(initial_hp or {})
    out: dict[str, list[tuple[str, int]]] = {"p1": [], "p2": []}
    for line in lines:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1]
        if tag in ("switch", "drag", "replace") and len(parts) > 4:
            slot = parts[2].split(":", 1)[0].strip()[:2]
            running[slot] = _hp_of(parts[4])
            continue
        if tag not in ("-damage", "-heal", "-sethp") or len(parts) < 4:
            continue
        slot = parts[2].split(":", 1)[0].strip()[:2]
        if slot not in out:
            continue
        new_hp = _hp_of(parts[3])
        max_hp = _maxhp_of(parts[3])
        fainted_here = new_hp == 0
        source = ""
        for extra in parts[4:]:
            extra = extra.strip()
            if extra.startswith("[from]"):
                source = normalize_id(extra[len("[from]"):].strip())
                break
        if source in _PARTIAL_TRAP_SOURCES:
            source = _CANONICAL_PARTIAL_TRAP
        if (
            unattributed_damage_as_roll
            and tag == "-damage"
            and source == _UNATTRIBUTED_DAMAGE_SOURCE
        ):
            # Scoped to branches the mapper flagged: this is a called move's
            # damage wearing a generic tag, not a genuine residual.
            source = _UNKNOWN_CALLEE_SOURCE
        if not source and tag == "-heal":
            source = "heal"
        if not source and tag == "-sethp":
            # Defensive: every `-sethp` Showdown emits carries `[from] move:
            # Pain Split`, but an untagged one must NOT fall through as "" —
            # that is the roll-scaled bucket, and Pain Split is deterministic
            # (floor((targetHP + userHP) / 2)), so its magnitude must be
            # compared exactly.
            source = "sethp"
        if tag == "-heal" and max_hp and new_hp >= max_hp:
            # ANY heal that tops the mon out is roll-scaled, whatever its tag:
            # it restores `maxhp - hp`, so its magnitude is set by the damage
            # that landed earlier in the same turn. This covers Rest and Recover
            # (bare) and equally a Leftovers or Wish tick that happens to cap.
            # The tag is preserved so attribution is still compared; only the
            # magnitude is relaxed, and only in the capped direction.
            source = f"{source}_to_full"
        if fainted_here and tag == "-damage":
            # A residual that KILLS is capped by the HP that happened to be
            # left, so its magnitude inherits the roll of whatever damaged the
            # mon earlier in the turn — Showdown reports 20 where the uncapped
            # tick would be 26. Comparing that exactly against a branch with a
            # different roll is a false divergence (seed 1310000 step 193,
            # exonerated by the engine lane in #893). Bucket it as roll-scaled.
            source = "capped_lethal"
        previous = running.get(slot)
        if previous is not None and new_hp != previous:
            # ZERO deltas are dropped. The engine emits no-op instructions
            # (`Heal SideTwo: 0` for a Rest that cannot heal a full-HP mon)
            # where Showdown emits `|-fail|` and no HP line at all. A component
            # that changed nothing carries no information, and recording it made
            # the component LISTS differ in length — surfacing as a spurious
            # roll-scaled mismatch (seed 1340000 step 110).
            out[slot].append((source, new_hp - previous))
        running[slot] = new_hp
    return out


def _maxhp_of(condition: str) -> int:
    """Max HP from a ``cur/max`` condition string, or 0 when unavailable."""

    head = condition.strip().split()[0] if condition.strip().split() else ""
    _, _, tail = head.partition("/")
    try:
        return int(tail)
    except ValueError:
        return 0


def _hp_of(condition: str) -> int:
    condition = condition.strip()
    head = condition.split()[0] if condition.split() else condition
    if head in ("0", "0.0") or "fnt" in condition:
        return 0
    current, _, _ = head.partition("/")
    try:
        return int(current)
    except ValueError:
        return 0


def _split_components(
    components: Sequence[tuple[str, int]]
) -> tuple[Counter, list[tuple[str, int]]]:
    """Partition one slot's components into (exact multiset, roll-scaled pairs).

    The roll-scaled half keeps its SOURCE, because ``capped_lethal`` is compared
    as an inequality rather than a window (see :func:`roll_components_agree`).
    """

    exact: Counter = Counter()
    rolled: list[tuple[str, int]] = []
    for source, delta in components:
        if source in _IGNORED_SOURCES:
            continue
        if source in _ROLL_SCALED_SOURCES or source.endswith("_to_full"):
            rolled.append((source, delta))
        else:
            exact[(source, delta)] += 1
    return exact, rolled


def legal_roll_damages(base_rolls: Sequence[int]) -> set[int]:
    """Every gen3-legal damage value for a move whose 100 % rolls are ``base_rolls``.

    gen3 damage is ``floor(base * random(85, 100) / 100)``;
    ``poke_engine.calculate_damage`` returns the base for (non-crit, crit), so
    the achievable set is exactly enumerable — membership, not a band.
    """

    values: set[int] = set()
    for base in base_rolls:
        if base <= 0:
            continue
        for roll in range(85, 101):
            values.add(base * roll // 100)
    return values


class BranchLegalRollError(ValueError):
    """A state-changing branch could not supply its required local support."""


def _event_changes_roll_state(event: object) -> bool:
    """Whether a rendered pre-hit event changes the damage-calculation state."""

    if not isinstance(event, str):
        return False
    if event.startswith("|switch|"):
        return True
    if not event.startswith(("|-boost|", "|-unboost|")):
        return False
    fields = event.split("|")
    try:
        return int(fields[4]) != 0
    except (IndexError, ValueError):
        # A malformed stage event must not regain the stale-support fallback.
        return True


def branch_event_legal_rolls(
    branch: Mapping[str, Any],
    *,
    side_one_choice: str,
    side_two_choice: str,
) -> set[int] | None:
    """Return support repriced after a pre-damage switch or stat event.

    ``calculate_damage`` on the boundary pre-state prices the wrong target when
    a switch resolves first, and the wrong stages when Boost/Unboost resolves
    first. The mapper supplies ``legal_roll_state`` only when it has serialized
    the exact branch prefix immediately before the first defender damage. The
    completed ``post_state`` is deliberately never accepted here: it may include
    the hit itself and later effects (for example Knock Off's item removal),
    which cannot affect the hit being compared.
    """

    events = branch.get("events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise BranchLegalRollError("branch events are missing or malformed")

    acting_side: str | None = None
    direct_damage_index: int | None = None
    for index, event in enumerate(events):
        if not isinstance(event, str):
            continue
        fields = event.split("|")
        if event.startswith("|move|"):
            actor = fields[2] if len(fields) > 2 else ""
            candidate = actor.split(":", maxsplit=1)[0][:2]
            acting_side = candidate if candidate in {"p1", "p2"} else None
            continue
        if not event.startswith("|-damage|") or "[from]" in event:
            continue
        target = fields[2] if len(fields) > 2 else ""
        target_side = target.split(":", maxsplit=1)[0][:2]
        # A bare Substitute HP cost is not an opponent hit. It needs no damage
        # support and the mapper correctly has no pre-hit snapshot for it.
        # Malformed idents remain fail-closed as potential direct damage.
        if acting_side is None or target_side not in {"p1", "p2"} or target_side != acting_side:
            if any(_event_changes_roll_state(prior) for prior in events[:index]):
                direct_damage_index = index
                break
    if direct_damage_index is None:
        return None

    legal_roll_state = branch.get("legal_roll_state")
    if not isinstance(legal_roll_state, str) or not legal_roll_state:
        raise BranchLegalRollError(
            "state-changing branch omitted pre-damage legal_roll_state"
        )
    try:
        state = poke_engine.State.from_string(legal_roll_state)
        side_one_rolls, side_two_rolls = poke_engine.calculate_damage(
            state, side_one_choice, side_two_choice, True
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:  # noqa: BLE001
        raise BranchLegalRollError(
            f"could not calculate branch-local legal rolls: {type(error).__name__}"
        ) from error
    return legal_roll_damages(list(side_one_rolls) + list(side_two_rolls))


def _roll_damage_scale(components: Sequence[tuple[str, int]]) -> int:
    """Total roll-scaled DAMAGE on a slot this step (the spread that can move a
    capped heal). Heals are positive and excluded; only damage carries a roll."""

    return sum(abs(delta) for _source, delta in components if delta < 0)


def roll_components_agree(
    observed: Sequence[tuple[str, int]],
    engine: Sequence[tuple[str, int]],
    legal: set[int] | None,
) -> bool:
    """Compare roll-scaled components: same count, each observed value legal.

    With a ``legal`` set from ``calculate_damage`` this is exact membership. When
    that is unavailable (the pre-state calculation does not survive a same-turn
    stat change) it degrades to a proportional window around the engine's
    representative roll — recorded by the caller as a separate, counted bucket.
    """

    if len(observed) != len(engine):
        return False
    for (obs_source, obs), (_eng_source, eng) in zip(
        sorted(observed, key=lambda pair: pair[1]),
        sorted(engine, key=lambda pair: pair[1]),
    ):
        if obs == eng:
            continue
        if obs_source.endswith("_to_full"):
            # A heal that tops the mon out restores `maxhp - hp_before`, so the
            # two sims differ by exactly their difference in `hp_before` — which
            # is bounded by the 85-100 % spread of the damage that preceded it in
            # THIS step. Crucially this caps in BOTH directions: a larger
            # preceding roll leaves less HP and makes the heal LARGER, so
            # `obs > eng` is legitimate and the one-sided `obs <= eng + 1` test
            # is inverted for this class (it rejected the motivating Rest case,
            # 251 vs 247, while accepting a 24x-too-small 10 vs 247).
            #
            # Observed damage d satisfies d >= 0.85 * base, so base <= d / 0.85
            # and the spread 0.15 * base <= 0.176 * d. Round to 0.18 with 1 HP
            # of flooring slack.
            scale = max(_roll_damage_scale(observed), _roll_damage_scale(engine))
            if abs(abs(obs) - abs(eng)) <= 0.18 * scale + 1:
                continue
            return False
        if obs_source == "capped_lethal":
            # A residual that KILLED was clipped by the HP that happened to
            # remain, so it can only ever be SMALLER than the uncapped tick the
            # engine carries — here the one-sided inequality IS the sound test.
            if abs(obs) <= abs(eng) + 1:
                continue
            return False
        if (obs < 0) != (eng < 0):
            return False
        magnitude = abs(obs)
        # ``legal`` is an ADDITIONAL accept path, never a veto. It is computed
        # from the PRE-state with an assumed move order, so it is unreliable
        # whenever the turn reorders or a same-turn stat change moves the base —
        # letting it reject would fail boundaries where the engine and Showdown
        # agree to the HP point (observed in the first revision of this matcher).
        if legal is not None and magnitude in legal:
            continue
        # Window: the engine carries a ~92 % representative roll, so the 85-100 %
        # spread is [0.924, 1.087] of it. One HP of slack for flooring. This is
        # scoped to the ROLL-SCALED component alone, not to net HP.
        low = abs(eng) * 0.92 - 1
        high = abs(eng) * 1.09 + 1
        if not (low <= magnitude <= high):
            return False
    return True


# ---------------------------------------------------------------------------------------------
# Hidden-counter mechanics: support-based validation instead of exact-state gating.
#
# WHY. Some gen3 counters are genuinely not public. The clearest is sleep:
# Showdown rolls `this.effectState.time = this.random(2, 6)` once, privately
# (pokemon-showdown/data/mods/gen3/conditions.ts `slp.onStart`), so the mon is
# unable to act for 1-4 attempted turns. The engine does NOT store "turns
# remaining"; it models wake-up as a HAZARD conditioned on turns already slept —
# `chance_to_wake_up(turns_asleep) = 1/(1 + MAX_SLEEP_TURNS - turns_asleep)` with
# `MAX_SLEEP_TURNS = 4` (third_party/poke-engine-src/src/gen3/generate_instructions.rs:44-71),
# which reproduces Showdown's uniform 1-4 duration exactly given the elapsed count.
#
# Demanding an exact counter match for such a mechanic is wrong IN PRINCIPLE: the
# harness is asking the world constructor to reproduce a number no observer can
# see. Strict mode therefore used to fail the whole boundary closed
# (`status_unsupported`), discarding 27.5% of full-round boundaries — the D8
# coverage ceiling in docs/engine_divergence_ledger_20260728.md.
#
# WHAT REPLACES IT. For hidden-counter mechanics only, the bar becomes
# SUPPORT-BASED: build one world per legal hidden-counter assignment and accept
# the boundary if the realized Showdown transition lies in the UNION of those
# worlds' branch supports with nonzero probability. That still fails a genuine
# mechanic error (a wake outside the legal window, a wrong post-wake effect, a
# wrong damage roll) because no legal counter value can produce it, while no
# longer punishing the engine for not knowing a private number.
#
# WHICH MECHANICS. Exactly two, both counter-hidden and both bounded:
#   * SLEEP  — `sleep_turns` swept 0..MAX_SLEEP_TURNS, plus `rest_turns` 1..2 for
#     a Rest-induced sleep (a separate engine code path with a fixed duration).
#   * CONFUSION — bounded duration since PR #875; the engine rolls the snap-out
#     the same way and the remaining count is private.
# Everything else keeps EXACT gating. In particular damage, status application,
# hazards, screens, weather, boosts and faints are all publicly observable and
# are NOT relaxed — the divergence bar for observable mechanics is unchanged.
# ---------------------------------------------------------------------------------------------

# third_party/poke-engine-src/src/gen3/generate_instructions.rs:44
MAX_SLEEP_TURNS = 4
# gen3 Rest: asleep for two attempted turns; the engine panics outside 0..2
# (generate_instructions.rs:1699), so the sweep stays inside the legal domain.
MAX_REST_TURNS = 2
# generate_instructions.rs:45 — the confusion ladder's top rung.
MAX_CONFUSION_TURNS = 4
# Guard rail: a pathological cross-product must never explode the boundary cost.
MAX_HIDDEN_COUNTER_WORLDS = 64


def _sleep_counter_variants(spec: Any) -> list[Any]:
    """Every legal (sleep_turns, rest_turns) assignment for the asleep actives."""

    def side_variants(side: Any) -> list[Any]:
        active = side.pokemon[side.active_index]
        if str(getattr(active, "status", "none")).lower() != "sleep":
            return [side]
        variants = []
        for sleep_turns in range(0, MAX_SLEEP_TURNS + 1):
            party = list(side.pokemon)
            party[side.active_index] = dataclasses.replace(
                active, sleep_turns=sleep_turns, rest_turns=0
            )
            variants.append(dataclasses.replace(side, pokemon=tuple(party)))
        for rest_turns in range(1, MAX_REST_TURNS + 1):
            party = list(side.pokemon)
            party[side.active_index] = dataclasses.replace(
                active, sleep_turns=0, rest_turns=rest_turns
            )
            variants.append(dataclasses.replace(side, pokemon=tuple(party)))
        return variants

    one = side_variants(spec.side_one)
    two = side_variants(spec.side_two)
    return [
        dataclasses.replace(spec, side_one=a, side_two=b) for a in one for b in two
    ][:MAX_HIDDEN_COUNTER_WORLDS]


def _confusion_counter_variants(spec: Any) -> list[Any]:
    """Every legal confusion-ladder rung for actives carrying CONFUSION.

    Post-PR #875 the engine prices the snap-out as a hazard on
    ``volatile_status_durations.confusion`` with
    ``chance_confusion_ends(n) = 1/(1 + MAX_CONFUSION_TURNS - n)``
    (generate_instructions.rs:106-114), reproducing Showdown's uniform 2-5 roll
    given the elapsed count. The remaining count is private, so the rungs
    0..MAX_CONFUSION_TURNS are swept exactly like sleep's.
    """

    def side_variants(side: Any) -> list[Any]:
        volatiles = {str(v).lower() for v in (side.volatile_statuses or ())}
        if "confusion" not in volatiles:
            return [side]
        variants = []
        for rung in range(0, MAX_CONFUSION_TURNS + 1):
            durations = dict(side.volatile_status_durations or {})
            durations["confusion"] = rung
            variants.append(dataclasses.replace(side, volatile_status_durations=durations))
        return variants

    one = side_variants(spec.side_one)
    two = side_variants(spec.side_two)
    return [
        dataclasses.replace(spec, side_one=a, side_two=b) for a in one for b in two
    ][:MAX_HIDDEN_COUNTER_WORLDS]


# Which fail-closed reason is recoverable by which SINGLE widening. The retry
# widens ONLY the mechanic that caused the failure — flipping the whole
# approximate bundle would admit yawn / partial-trap / substitute-health
# guesses whose effects (a sleep landing, a chip tick) are OBSERVABLE, which
# would break the "observable mechanics keep exact gating" rule this mode rests
# on.
_UNSUPPORTED_VOLATILE_RE = re.compile(r"\[([^\]]*)\]")


def hidden_counter_recovery(error: EngineWorldUnsupported) -> str | None:
    """Which hidden counter (if any) this fail-closed reason is about.

    Returns "sleep", "confusion", or None. ``volatile_unsupported`` recovers
    ONLY when confusion is the sole unsupported volatile: yawn also rides the
    same engine_world flag, and yawn's sleep landing is publicly observable, so
    admitting it would relax an observable mechanic.
    """

    if error.reason == "status_unsupported":
        return "sleep"
    if error.reason != "volatile_unsupported":
        return None
    match = _UNSUPPORTED_VOLATILE_RE.search(error.detail or "")
    if not match:
        return None
    names = {
        part.strip().strip("'\"").lower()
        for part in match.group(1).split(",")
        if part.strip()
    }
    return "confusion" if names == {"confusion"} else None


def world_construction_limit(error: EngineWorldUnsupported) -> str | None:
    """Return a named comparison limit for public state that cannot be known."""

    if error.reason == "substitute_health_unknown":
        return "limit:world_substitute_health_unknown"
    return None


def count_world_construction_limit(
    counts: Counter[str], error: EngineWorldUnsupported
) -> bool:
    """Count a public-information limit and report whether it was handled."""

    limit = world_construction_limit(error)
    if limit is None:
        return False
    counts[limit] += 1
    return True


# ---------------------------------------------------------------------------------------------
# Per-boundary evaluation.
# ---------------------------------------------------------------------------------------------


def _prestate_mismatch(observed: TurnFeatures, engine: TurnFeatures) -> str | None:
    """Exact pre-state comparison (no damage band: both sides claim the SAME state)."""

    if observed.p1_status != engine.p1_status:
        return f"p1 status {observed.p1_status} != {engine.p1_status}"
    if observed.p2_status != engine.p2_status:
        return f"p2 status {observed.p2_status} != {engine.p2_status}"
    if observed.weather != engine.weather:
        return f"weather {observed.weather} != {engine.weather}"
    if observed.presence() != engine.presence():
        return f"side conditions {observed.presence()} != {engine.presence()}"
    for side, obs_hp, eng_hp in (
        ("p1", observed.p1_hp, engine.p1_hp),
        ("p2", observed.p2_hp, engine.p2_hp),
    ):
        if obs_hp < 0:
            return f"{side} hp unknown in protocol fold"
        if obs_hp != eng_hp:
            return f"{side} hp {obs_hp} != {eng_hp}"
    return None


def _transition_mismatch(
    observed: TurnFeatures,
    branch: TurnFeatures,
    *,
    start_hp: Mapping[str, int],
    branch_maxhp: Mapping[str, int],
    active_changed: Mapping[str, bool],
) -> str | None:
    """This step's observed delta vs one engine branch, with the real-game carve-outs.

    Differences from :func:`pokezero.engine_fidelity._mismatch_reason`, each
    forced by a live-game shape the curated fixtures never reach:

    * a side that FAINTED this step is not status-compared — Showdown's
      ``0 fnt`` condition string carries no status, while the engine keeps the
      status on the fainted mon (the known faint-pattern conflation);
    * a side whose ACTIVE CHANGED anchors its damage band on the incoming mon's
      max HP rather than on the outgoing mon's HP, which is not a damage scale
      at all.
    """

    for slot, obs_status, br_status in (
        ("p1", observed.p1_status, branch.p1_status),
        ("p2", observed.p2_status, branch.p2_status),
    ):
        if slot in observed.fainted or slot in branch.fainted:
            continue
        if obs_status != br_status:
            return f"{slot} status {obs_status} != {br_status}"
    if observed.fainted != branch.fainted:
        return f"fainted {sorted(observed.fainted)} != {sorted(branch.fainted)}"
    if observed.weather != branch.weather:
        return f"weather {observed.weather} != {branch.weather}"
    if observed.presence() != branch.presence():
        return f"side conditions {observed.presence()} != {branch.presence()}"
    for slot, obs_hp, br_hp in (
        ("p1", observed.p1_hp, branch.p1_hp),
        ("p2", observed.p2_hp, branch.p2_hp),
    ):
        if obs_hp < 0:
            continue
        if obs_hp == 0 or br_hp == 0:
            if obs_hp != br_hp:
                return f"{slot} hp {obs_hp} != {br_hp} (faint boundary)"
            continue
        anchor = branch_maxhp[slot] if active_changed[slot] else start_hp[slot]
        damage_scale = max(abs(anchor - br_hp), _MIN_TOLERANCE_HP)
        if abs(obs_hp - br_hp) > max(_MIN_TOLERANCE_HP, _DAMAGE_TOLERANCE * damage_scale):
            return f"{slot} hp {obs_hp} outside tolerance of {br_hp} (anchor {anchor})"
    return None



_MISS_COMPONENTS_RE = re.compile(
    r"observed_only=\[(?P<obs>.*?)\]\s+engine_only=\[(?P<eng>.*?)\]"
)
_MISS_SOURCE_RE = re.compile(r"\('([a-z0-9_]*)',")


_MISS_PCT_RE = re.compile(r"pct=(?P<pct>[\d.]+)")

# Residuals whose presence in a miss makes that miss a candidate for the
# majority override. Deliberately an allow-list of NAMED end-of-turn effects:
# an unattributed or roll-scaled component is never "the residual".
_ADJUDICABLE_RESIDUALS = frozenset({"itemleftovers", "psn", "brn", "sandstorm", "tox"})


def _majority_miss(misses: Sequence[str]) -> str | None:
    """The miss carrying the largest share of the branch probability mass."""

    best: tuple[float, str] | None = None
    for miss in misses:
        match = _MISS_PCT_RE.search(miss)
        if not match:
            continue
        pct = float(match.group("pct"))
        if best is None or pct > best[0]:
            best = (pct, miss)
    return best[1] if best else None


def _residual_only_sources(miss: str) -> set[str]:
    """The named residuals a miss is about, or empty if it is about anything else.

    Empty is the safe answer: it means "do not override". A miss mentioning a
    roll-scaled component, or any source outside the allow-list, is not a
    residual-only miss and must keep its own classification.
    """

    if not miss or "roll-scaled" in miss:
        return set()
    body = miss.split(": ", 1)[1] if ": " in miss else miss
    match = _MISS_COMPONENTS_RE.search(body)
    if not match:
        return set()
    sources = set(_MISS_SOURCE_RE.findall(match.group("obs"))) | set(
        _MISS_SOURCE_RE.findall(match.group("eng"))
    )
    if not sources or not sources <= _ADJUDICABLE_RESIDUALS:
        return set()
    return sources


def classify_divergence(step_lines: Sequence[str], misses: Sequence[str]) -> str:
    """Name every divergence. No divergence may land in an unnamed bucket.

    Under the strict matcher the FAILING COMPONENT is always known — it is in the
    miss reason — so classification is driven by that first, and only falls back
    to step-protocol evidence when there is no parsable miss. An earlier version
    classified from the protocol alone, which left ~28 % of strict divergences
    ``unclassified``; "zero divergence" cannot gate on a bucket nobody can name.
    """

    reason = misses[0] if misses else ""
    # MAJORITY OVERRIDE (#946 adjudication, made mechanical).
    #
    # `misses` is in branch order, so `misses[0]` can be a minority branch. When
    # a row's FIRST miss is a named residual (Leftovers, poison...) but the
    # branches carrying most of the probability mass complain only about a
    # damage component, the residual is not the disagreement — it is present and
    # numerically identical in the majority branch, and only the low-probability
    # branch lacks it. Classifying from `misses[0]` then files a damage
    # disagreement under the residual's name.
    #
    # s1500014 st69: three branches. The 6.25% branch reports a missing
    # `itemleftovers`; the 75.00% and 18.75% branches (93.75% together) report
    # `observed=[('', -214)] engine=[('', -116)]` — a damage disagreement of
    # nearly 2x. The row is damage_calc, and was labelled
    # `component_missing_in_engine:itemleftovers` for four cycles.
    #
    # Deliberately narrow: this only fires when the first miss is residual-named
    # AND the majority miss is roll-scaled. Reordering every row by probability
    # would re-classify rows nobody has adjudicated.
    secondary = _residual_only_sources(reason)
    if secondary:
        majority = _majority_miss(misses)
        if (
            majority is not None
            and "roll-scaled" in majority
            and "capped_lethal" not in majority
        ):
            # The `capped_lethal` exclusion is load-bearing and deliberate. A
            # majority miss carrying it classifies as
            # `limit:roll_divergent_lethality` — an ADJUDICATED NON-DIVERGENCE —
            # so allowing the override there would move 15 rows out of the
            # outside-limit count and hand the acceptance gate a 15-row credit
            # that no one adjudicated. #946 adjudicated these rows as
            # damage_calc, not as a comparison limit. A relabel must never
            # reduce the residue; if those rows belong in a limit class, that is
            # its own decision, taken on its own evidence.
            reason = majority
    body = reason.split(": ", 1)[1] if ": " in reason else reason

    # PHAZE FIRST — it explains the components rather than the other way round.
    # Whirlwind/Roar drag a RANDOM target: the engine fans out uniformly over its
    # world's alive reserve, while Showdown drew from the real hidden team. When
    # the realized target is not the one a branch dragged, the entry-hazard
    # arithmetic lands on a different mon with a different max HP, and the
    # component diff reads as a hazard/residual disagreement that it is not.
    # The engine lane verified these are determinization limits, not engine bugs
    # (feeding the exact repro state back produced correct fan-out including the
    # observed tick). Named so the residue table stops charging them to hazards.
    if any(line.startswith("|drag|") for line in step_lines):
        return "limit:world_sample_drag_target"

    if "boost deltas" in body:
        return "boost_delta_support"
    if "roll-scaled" in body:
        if "capped_lethal" in body:
            # Showdown's roll left the mon at N HP and the residual killed it;
            # the engine's roll left it at M and the residual did not (or vice
            # versa). The component sets then differ in LENGTH — one sim has a
            # lethal residual, the other an ordinary one — and no per-component
            # comparison can align two different stochastic outcomes. This is a
            # limit of the comparison, not an engine fault; named so it can be
            # excluded explicitly rather than sitting in an anonymous bucket.
            return "limit:roll_divergent_lethality"
        return "roll_scaled_component"
    match = _MISS_COMPONENTS_RE.search(body)
    if match:
        observed = set(_MISS_SOURCE_RE.findall(match.group("obs")))
        engine = set(_MISS_SOURCE_RE.findall(match.group("eng")))
        shared = observed & engine
        if shared and observed == engine:
            return "component_magnitude:" + ",".join(sorted(shared))
        if observed and not engine:
            return "component_missing_in_engine:" + ",".join(sorted(observed))
        if engine and not observed:
            return "component_extra_in_engine:" + ",".join(sorted(engine))
        if observed or engine:
            return "component_mismatch:%s|%s" % (
                ",".join(sorted(observed)),
                ",".join(sorted(engine)),
            )
        return "component_set_equal_but_unmatched"
    if "every branch rendered lossy" in body:
        return "mapper_lossy"
    if "mapper produced no usable branch" in body:
        return "no_usable_branch"

    # Banded matcher (or an unparsable miss): fall back to protocol evidence.
    if " status " in body:
        return "status_support"
    if "fainted " in body:
        return "faint_boundary"
    if " hp " in body:
        return "damage_band"
    # PROTOCOL-EVIDENCE fallbacks only. These name what the step CONTAINED, not
    # what went wrong, and they are reached only when the miss reason could not
    # be parsed. The old `faint_ply_residual_deferral` label was actively
    # misleading: it pointed the residue table at D2 for boundaries where
    # nothing faints in the move phase (#893). Prefixed so no reader mistakes
    # evidence for attribution.
    fainted = any(line.startswith("|faint|") for line in step_lines)
    upkeep = any(line.strip() == "|upkeep" for line in step_lines)
    if fainted and not upkeep:
        return "evidence:faint_ply_no_upkeep"
    if any("[from] Spikes" in line for line in step_lines):
        return "evidence:spikes_in_step"
    if any("|-crit|" in line for line in step_lines):
        return "evidence:crit_in_step"
    if not reason:
        return "no_miss_recorded"
    return "unclassified"


def _active_maxhp_by_slot(state: Any, slot_sides: Mapping[str, str]) -> dict[str, int]:
    sides = _sides_by_slot(state, slot_sides)
    return {
        slot: int(side.pokemon[int(str(side.active_index))].maxhp) for slot, side in sides.items()
    }


def evaluate_boundary_strict(
    *,
    states: Sequence[Any],
    slot_sides: Mapping[str, str],
    choices: Mapping[str, str],
    party_display: Mapping[str, Sequence[str]],
    turn: int,
    pre_features: TurnFeatures,
    observed: TurnFeatures,
    step_lines: Sequence[str],
    observed_boosts: Mapping[str, Mapping[str, int]],
    active_changed: Mapping[str, bool],
    counts: Counter,
) -> tuple[str, list[str], int]:
    """Per-damage-source comparison against the mapper-rendered engine branches."""

    side_one_choice = choices["p1"] if slot_sides["p1"] == "side_one" else choices["p2"]
    side_two_choice = choices["p2"] if slot_sides["p2"] == "side_two" else choices["p1"]
    # The mapper renders side_one/side_two; map its p1/p2 labels back to slots.
    engine_label_for_slot = {
        slot: ("p1" if slot_sides[slot] == "side_one" else "p2") for slot in ("p1", "p2")
    }
    ctx = json.dumps({
        "p1": list(party_display["p1" if slot_sides["p1"] == "side_one" else "p2"]),
        "p2": list(party_display["p2" if slot_sides["p2"] == "side_two" else "p1"]),
        "turn": int(turn),
    })

    # Both sides start from the same pre-state (the pre-state gate proved the
    # HP equal), so both extractions are seeded with it.
    pre_hp = {"p1": pre_features.p1_hp, "p2": pre_features.p2_hp}
    observed_components = damage_components(step_lines, pre_hp)
    obs_exact = {slot: _split_components(observed_components[slot])[0] for slot in ("p1", "p2")}
    obs_rolled = {slot: _split_components(observed_components[slot])[1] for slot in ("p1", "p2")}

    misses: list[tuple[int, str]] = []
    branch_total = 0
    usable_branches = 0
    for state in states:
        try:
            rendered = json.loads(
                pokezero_search.branch_events(
                    state.to_string(), side_one_choice, side_two_choice, ctx, True, True
                )
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # noqa: BLE001
            counts[f"strict:branch_events_error:{type(error).__name__}"] += 1
            continue
        branches = rendered.get("branches") or []
        branch_total += len(branches)
        # Baseline legal damage set for this pre-state; unavailable when the
        # engine refuses the pair (recorded, then the proportional window is
        # used). State-changing branches below replace this with their local
        # support rather than silently reusing a stale target or stat stage.
        pre_legal: set[int] | None = None
        try:
            s1_rolls, s2_rolls = poke_engine.calculate_damage(
                state, side_one_choice, side_two_choice, True
            )
            pre_legal = legal_roll_damages(list(s1_rolls) + list(s2_rolls))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # noqa: BLE001
            counts["strict:no_damage_rolls"] += 1

        for branch in branches:
            if float(branch.get("percentage") or 0.0) <= 0.0:
                continue
            lossy = list(branch.get("lossy") or [])
            # A branch whose ONLY defect is the known Sleep Talk callee-identity
            # gap is still usable: its damage is real, only its attribution is
            # generic. Any other lossy marker is a different insufficiency and
            # still disqualifies the branch.
            sleeptalk_union = bool(lossy) and set(lossy) == {_SLEEPTALK_LOSSY_MARKER}
            if lossy and not sleeptalk_union:
                counts["strict:lossy_render"] += 1
                continue
            if sleeptalk_union:
                counts["strict:sleeptalk_union_branch"] += 1
            try:
                legal = branch_event_legal_rolls(
                    branch,
                    side_one_choice=side_one_choice,
                    side_two_choice=side_two_choice,
                )
            except BranchLegalRollError as error:
                counts[f"strict:branch_event_legal_error:{type(error).__name__}"] += 1
                continue
            if legal is None:
                legal = pre_legal
            usable_branches += 1
            engine_components = damage_components(
                branch.get("events") or [],
                {engine_label_for_slot[slot]: pre_hp[slot] for slot in ("p1", "p2")},
                unattributed_damage_as_roll=sleeptalk_union,
            )
            ok = True
            reason = None
            for slot in ("p1", "p2"):
                label = engine_label_for_slot[slot]
                eng_exact, eng_rolled = _split_components(engine_components[label])
                # ROLL FIRST. A branch whose roll does not match is the wrong
                # branch, and its deterministic components are then compared
                # against a different damage history — reporting THAT as the
                # miss points at the wrong mechanic. Reject on the roll and let
                # a later branch be judged on its residuals.
                if not roll_components_agree(obs_rolled[slot], eng_rolled, legal):
                    reason = (
                        f"{slot} roll-scaled components differ: "
                        f"observed={sorted(obs_rolled[slot], key=lambda p: p[1])} "
                        f"engine={sorted(eng_rolled, key=lambda p: p[1])}"
                    )
                    ok = False
                    break
                if eng_exact != obs_exact[slot]:
                    only_obs = obs_exact[slot] - eng_exact
                    only_eng = eng_exact - obs_exact[slot]
                    reason = (
                        f"{slot} attributed components differ: "
                        f"observed_only={sorted(only_obs.elements())} "
                        f"engine_only={sorted(only_eng.elements())}"
                    )
                    ok = False
                    break
            if ok:
                return "matched", [], branch_total
            if reason:
                # RANK the miss. Roll-first rejection means a branch that failed
                # on its roll reports a roll reason even when ANOTHER branch
                # passes the roll and fails on a deterministic component — and
                # the first-listed miss drives classification, so 35 % of the
                # `roll_scaled_component` class were really exact-component
                # divergences wearing the wrong label (triage of seeds
                # 1350000-1350059). Report the branch that got FURTHEST: one
                # that cleared the rolls outranks one that did not.
                rank = 0 if "roll-scaled" in reason else 1
                misses.append((rank, f"pct={float(branch.get('percentage') or 0):.2f}: {reason}"))
    if usable_branches == 0:
        # Every branch was a lossy render: the mapper itself is telling us it
        # cannot reproduce this turn, so the boundary is unmeasurable, not
        # divergent.
        return "skip_lossy", ["every branch rendered lossy"], branch_total
    ordered = [text for _rank, text in sorted(misses, key=lambda m: -m[0])][:12]
    return "diverged", ordered, branch_total


def evaluate_boundary(
    *,
    states: Sequence[Any],
    slot_sides: Mapping[str, str],
    choices: Mapping[str, str],
    pre_features: TurnFeatures,
    observed: TurnFeatures,
    observed_boosts: Mapping[str, Mapping[str, int]],
    active_changed: Mapping[str, bool],
) -> tuple[str, list[str], int]:
    """Return ``(verdict, misses, branch_count)`` for one full decision boundary."""

    side_one_choice = choices["p1"] if slot_sides["p1"] == "side_one" else choices["p2"]
    side_two_choice = choices["p2"] if slot_sides["p2"] == "side_two" else choices["p1"]

    # Union of branch support across the hidden-counter candidates. With a single
    # candidate (the normal case) this is exactly the old exact-world check;
    # with several it is the support-based bar documented above. Zero-probability
    # branches are never admitted: only branches the engine actually enumerates.
    rows: list[dict[str, Any]] = []
    for state in states:
        for branch in poke_engine.generate_instructions(
            state, side_one_choice, side_two_choice
        ):
            if float(branch.percentage) <= 0.0:
                continue
            applied = state.apply_instructions(branch)
            rows.append(
                {
                    "percentage": float(branch.percentage),
                    "features": engine_features_by_slot(applied, slot_sides),
                    "boost_deltas": engine_boost_deltas_by_slot(state, applied, slot_sides),
                    "maxhp": _active_maxhp_by_slot(applied, slot_sides),
                }
            )

    # A REGULAR switch clears stat stages with no protocol echo (Showdown emits
    # nothing; the engine emits reset_boosts instructions), so a side whose
    # active changed is exempt from the exact boost-delta filter.
    def _comparable(deltas: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
        return {
            slot: ({} if active_changed[slot] else dict(deltas.get(slot) or {}))
            for slot in ("p1", "p2")
        }

    normalized_observed = _comparable(observed_boosts)
    candidates = [row for row in rows if _comparable(row["boost_deltas"]) == normalized_observed]
    if not candidates:
        misses = [
            f"observed boost deltas {normalized_observed} not in branch support: "
            + "; ".join(
                f"pct={row['percentage']:.2f}: {_comparable(row['boost_deltas'])}" for row in rows
            )
        ]
        return "diverged", misses, len(rows)

    start_hp = {"p1": pre_features.p1_hp, "p2": pre_features.p2_hp}
    misses = []
    for row in candidates:
        reason = _transition_mismatch(
            observed,
            row["features"],
            start_hp=start_hp,
            branch_maxhp=row["maxhp"],
            active_changed=active_changed,
        )
        if reason is None:
            return "matched", [], len(rows)
        misses.append(f"pct={row['percentage']:.2f}: {reason}")
    return "diverged", misses, len(rows)


# ---------------------------------------------------------------------------------------------
# Game driver.
# ---------------------------------------------------------------------------------------------


def run_game(
    *,
    env: LocalShowdownEnv,
    flags_policy: EngineMctsPolicy,
    seed: int,
    dex: Any,
    max_steps: int,
    keep_repro: int,
    repros: list[dict[str, Any]],
    approximate_sleep: bool,
    hidden_counter_support: bool,
    matcher: str,
) -> Counter:
    """Run one game. Divergence repros are appended to ``repros`` (capped by
    ``keep_repro``); the caller owns whether that list is per-game (checkpoint
    records) or run-global (the final report)."""

    counts: Counter = Counter()
    env.reset(seed=seed, format_id="gen3randombattle")
    true_teams = _true_teams_from_bridge_snapshot(env.snapshot().bridge_snapshot)
    packed = {slot: true_teams[slot]["packed"] for slot in ("p1", "p2")}
    override = BattleStartOverride(player_teams=packed)
    teams = {slot: unpack_team(packed[slot]) for slot in ("p1", "p2")}
    rng = random.Random(seed ^ 0x5EED)

    cumulative: list[str] = list(env.protocol_lines)
    cursor = len(cumulative)
    steps = 0

    while env.terminal() is None and steps < max_steps:
        steps += 1
        requested = tuple(env.requested_players())
        actions: dict[str, int] = {}
        for player in requested:
            mask = env.legal_actions(player)
            legal = [i for i, allowed in enumerate(mask) if allowed]
            if not legal:
                counts["abort:no_legal_action"] += 1
                return counts
            actions[player] = rng.choice(legal)

        prepared: dict[str, Any] | None = None
        if set(requested) == {"p1", "p2"}:
            counts["boundaries_full_round"] += 1
            prepared = _prepare_boundary(
                env=env,
                flags_policy=flags_policy,
                override=override,
                teams=teams,
                dex=dex,
                actions=actions,
                cumulative=cumulative,
                counts=counts,
                approximate_sleep=approximate_sleep,
                hidden_counter_support=hidden_counter_support,
            )
        else:
            counts["skip:single_seat_boundary"] += 1

        env.step(actions)
        step_lines = tuple(str(line) for line in env.protocol_lines[cursor:])
        cursor = len(env.protocol_lines)
        cumulative.extend(step_lines)

        if prepared is None:
            continue

        observed = _fold(cumulative)
        observed = TurnFeatures(
            p1_hp=observed.p1_hp,
            p2_hp=observed.p2_hp,
            p1_status=observed.p1_status,
            p2_status=observed.p2_status,
            fainted=_fold(step_lines).fainted,
            weather=observed.weather,
            side_conditions=observed.side_conditions,
        )
        active_changed = {
            slot: any(
                line.startswith((f"|switch|{slot}a", f"|drag|{slot}a", f"|replace|{slot}a"))
                for line in step_lines
            )
            for slot in ("p1", "p2")
        }
        try:
            if matcher == "strict":
                verdict, misses, branch_count = evaluate_boundary_strict(
                    states=prepared["states"],
                    slot_sides=prepared["slot_sides"],
                    choices=prepared["choices"],
                    party_display=prepared["party_display"],
                    turn=prepared["turn"],
                    pre_features=prepared["pre_features"],
                    observed=observed,
                    step_lines=step_lines,
                    observed_boosts=observed_boost_deltas(step_lines),
                    active_changed=active_changed,
                    counts=counts,
                )
            else:
                verdict, misses, branch_count = evaluate_boundary(
                    states=prepared["states"],
                    slot_sides=prepared["slot_sides"],
                    choices=prepared["choices"],
                    pre_features=prepared["pre_features"],
                    observed=observed,
                    observed_boosts=observed_boost_deltas(step_lines),
                    active_changed=active_changed,
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # pyo3 panics do not derive from Exception
            counts["engine_error"] += 1
            # Strip operands so the reason histogram stays low-cardinality
            # ("Invalid move for s1: firepunch" -> "invalid_move").
            detail = "invalid_move" if "Invalid move for" in str(error) else "other"
            counts[f"engine_error:{type(error).__name__}:{detail}"] += 1
            if detail == "invalid_move":
                bad = str(error).split(": ", 1)[-1].strip()
                counts[f"engine_error_choice:{bad}"] += 1
            if len(repros) < keep_repro:
                repros.append(
                    {
                        "kind": "engine_error",
                        "seed": seed,
                        "step": steps,
                        "error": f"{type(error).__name__}: {error}",
                        "choices": prepared["choices"],
                        "engine_state": prepared["states"][0].to_string(),
                    # EVERY hidden-counter candidate, so scripts/replay_residue.py
                    # reproduces the exact branch union the matcher judged rather
                    # than only the first sweep rung.
                    "engine_states": [st.to_string() for st in prepared["states"]],
                    "gating": prepared["gating"],
                        "engine_states": [st.to_string() for st in prepared["states"]],
                    }
                )
            continue

        if verdict == "skip_lossy":
            counts["skip:strict_all_branches_lossy"] += 1
            continue
        counts[f"transition:{verdict}"] += 1
        if verdict == "diverged":
            counts[f"divergence_class:{classify_divergence(step_lines, misses)}"] += 1
        if verdict == "diverged" and len(repros) < keep_repro:
            repros.append(
                {
                    "kind": "transition_diverged",
                    "seed": seed,
                    "step": steps,
                    "choices": prepared["choices"],
                    "engine_state": prepared["states"][0].to_string(),
                    # EVERY hidden-counter candidate, so scripts/replay_residue.py
                    # reproduces the exact branch union the matcher judged rather
                    # than only the first sweep rung.
                    "engine_states": [st.to_string() for st in prepared["states"]],
                    "gating": prepared["gating"],
                    "pre_features": _features_payload(prepared["pre_features"]),
                    "observed": _features_payload(observed),
                    "observed_boost_deltas": observed_boost_deltas(step_lines),
                    "active_changed": active_changed,
                    "divergence_class": classify_divergence(step_lines, misses),
                    "branch_count": branch_count,
                    "branch_misses": misses[:12],
                    "protocol": list(step_lines),
                }
            )
    if steps >= max_steps:
        counts["abort:max_steps"] += 1
    return counts


def _prepare_boundary(
    *,
    env: LocalShowdownEnv,
    flags_policy: EngineMctsPolicy,
    override: BattleStartOverride,
    teams: Mapping[str, tuple],
    dex: Any,
    actions: Mapping[str, int],
    cumulative: Sequence[str],
    counts: Counter,
    approximate_sleep: bool,
    hidden_counter_support: bool,
) -> dict[str, Any] | None:
    """Build the engine world + resolve both choices, or return None with a counted skip."""

    try:
        mstate = env.public_materialization_state("p1")
    except Exception as error:  # noqa: BLE001 — a materialization refusal is a skip
        counts[f"skip:no_materialization:{type(error).__name__}"] += 1
        return None

    observation = env.observe("p1")
    context = types.SimpleNamespace(observation=observation, player_id="p1")
    # Production derivation of the public item/Transform/Encore signals the
    # payload cannot carry (engine_search.EngineMctsPolicy._public_effect_signals):
    # reused verbatim so the differential builds the same world the live searcher
    # would. It needs only ``observation.metadata`` + ``player_id``.
    blocked, encored, removed, overridden, transformed = flags_policy._public_effect_signals(
        context
    )

    candidates_by_slot: dict[str, Sequence[Mapping[str, Any]]] = {}
    recharging: list[str] = []
    for slot in ("p1", "p2"):
        metadata = env.observe(slot).metadata
        rows = metadata.get("action_candidates")
        if not isinstance(rows, Sequence):
            counts["skip:no_action_candidates"] += 1
            return None
        candidates_by_slot[slot] = rows
        chosen = next(
            (r for r in rows if isinstance(r, Mapping) and r.get("action_index") == actions[slot]),
            None,
        )
        if (
            isinstance(chosen, Mapping)
            and chosen.get("kind") == "move"
            and normalize_id(str(chosen.get("move_id") or "")) == "recharge"
        ):
            recharging.append(slot)

    try:
        payload = _public_materialization_payload(mstate)
        truant = truant_loaf_slots(list(cumulative), payload, teams)
    except Exception:  # noqa: BLE001
        truant = []

    def _build(*, approximate_sleep_turns: bool, hidden_volatiles: bool) -> Any:
        return world_battle_spec(
            mstate,
            override,
            dex=dex,
            approximate_sleep_turns=approximate_sleep_turns,
            approximate_substitute_health=True,
            approximate_hidden_duration_volatiles=hidden_volatiles,
            blocked_slots=blocked,
            encored_moves=encored,
            removed_item_species=removed,
            current_item_overrides=overridden,
            recharging_slots=tuple(recharging),
            truant_slots=tuple(truant),
            transformed_slots=transformed,
        )

    world = None
    specs: list[Any] = []
    gating = "exact"
    try:
        world = _build(approximate_sleep_turns=approximate_sleep, hidden_volatiles=False)
        specs = [world.spec]
    except EngineWorldUnsupported as error:
        if count_world_construction_limit(counts, error):
            return None
        mechanic = hidden_counter_recovery(error) if hidden_counter_support else None
        if mechanic is None:
            counts[f"skip:world_unsupported:{error.reason}"] += 1
            return None
        # Widen ONLY the mechanic that failed, then sweep ONLY its counter.
        try:
            if mechanic == "sleep":
                world = _build(approximate_sleep_turns=True, hidden_volatiles=False)
                specs = _sleep_counter_variants(world.spec)
            else:
                world = _build(
                    approximate_sleep_turns=approximate_sleep, hidden_volatiles=True
                )
                specs = _confusion_counter_variants(world.spec)
        except EngineWorldUnsupported as inner:
            if count_world_construction_limit(counts, inner):
                return None
            counts[f"skip:world_unsupported:{inner.reason}"] += 1
            return None
        gating = "support"
        counts[f"hidden_counter_support:{mechanic}"] += 1
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:  # noqa: BLE001
        counts[f"skip:world_error:{type(error).__name__}"] += 1
        return None

    states: list[Any] = []
    for spec in specs:
        try:
            states.append(build_poke_engine_state(spec, module=poke_engine))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # noqa: BLE001 — an illegal counter combination
            continue
    if not states:
        counts["skip:world_error:no_constructible_candidate"] += 1
        return None
    state = states[0]

    sides = _sides_by_slot(state, world.slot_sides)
    choices: dict[str, str] = {}
    for slot in ("p1", "p2"):
        try:
            choices[slot] = engine_choice_for_action(
                action_index=actions[slot],
                candidates=candidates_by_slot[slot],
                engine_side=sides[slot],
            )
        except UnmappableChoice as error:
            counts[f"skip:unmappable_choice:{error.reason}"] += 1
            return None

    pre_features = engine_features_by_slot(state, world.slot_sides)
    observed_pre = _fold(cumulative)
    mismatch = _prestate_mismatch(observed_pre, pre_features)
    if mismatch is not None:
        counts["world_prestate_mismatch"] += 1
        counts[f"world_prestate_mismatch:{mismatch.split(' ')[0]}_{mismatch.split(' ')[1]}"] += 1
        return None

    counts["boundaries_measured"] += 1
    counts[f"gating:{gating}"] += 1
    turn = 0
    try:
        turn = int(_public_materialization_payload(mstate).get("turn") or 0)
    except Exception:  # noqa: BLE001
        turn = 0
    return {
        "party_display": {
            slot: [str(m.species) for m in teams[slot]] for slot in ("p1", "p2")
        },
        "turn": turn,
        "states": states,
        "slot_sides": world.slot_sides,
        "choices": choices,
        "pre_features": pre_features,
        "gating": gating,
    }


# ---------------------------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------
# Checkpointing: one JSONL record per completed game, appended as it finishes.
#
# A 2000-game run was lost to a supervisor kill because the report was only
# serialised at exit (docs/engine_divergence_ledger_20260728.md §3.5). With
# --checkpoint the run is restartable and shardable: --resume skips seeds already
# present, and --merge-from aggregates any number of shard files into one report
# with the SAME schema the single-process path emits.
# ---------------------------------------------------------------------------------------------

CHECKPOINT_SCHEMA = "engine-transition-differential/1"


def checkpoint_record(
    *,
    seed: int,
    counts: Mapping[str, int],
    repros: Sequence[Mapping[str, Any]],
    seconds: float,
    build_check: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        # Carried per RECORD, not just per report, so a merge of many shards can
        # tell that ANY of them ran ungated.
        "build_check": build_check,
        "seed": int(seed),
        "seconds": round(float(seconds), 3),
        "counters": {str(k): int(v) for k, v in counts.items()},
        "repros": list(repros),
    }


def append_checkpoint(handle: Any, record: Mapping[str, Any]) -> None:
    """Append one record and flush, so a kill loses at most the in-flight game."""

    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    handle.flush()


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    """Read a checkpoint file, tolerating a truncated final line from a hard kill."""

    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    raw = [line for line in path.read_text().splitlines()]
    last_index = len(raw) - 1
    for line_number, line in enumerate(raw):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line_number != last_index:
                # Mid-file corruption would silently shrink the denominator of
                # every rate computed from this shard, so it is fatal. Only the
                # FINAL line may be a torn write from an interrupted run.
                raise ValueError(
                    f"{path}: unparseable line {line_number + 1} of {len(raw)} — only the "
                    "final line may be a torn write; refusing to load a corrupt shard"
                ) from None
            print(
                f"warning: {path}: discarding torn final line {line_number + 1} "
                "(interrupted run)",
                file=sys.stderr,
            )
            continue
        if isinstance(record, Mapping) and record.get("schema") == CHECKPOINT_SCHEMA:
            records.append(dict(record))
    return records


def build_report(
    records: Sequence[Mapping[str, Any]],
    *,
    elapsed: float | None,
    approximate_sleep: bool | None,
    matcher: str | None,
    keep_repro: int,
    repros_per_game: int | None = None,
    sources: Sequence[str] = (),
) -> dict[str, Any]:
    """Aggregate per-game records into the report schema (live and merge paths).

    ``keep_repro`` and ``repros_per_game`` are DIFFERENT knobs and both are recorded in
    the payload, because conflating them silently invalidates identity diffs.
    ``--repros-per-game`` bounds what each GAME retains (and therefore what lands in the
    checkpoint); ``--keep-repro`` bounds what the aggregated REPORT carries. A run invoked
    with ``--repros-per-game 40`` and no ``--keep-repro`` still writes a report truncated to
    the ``--keep-repro`` default, so a diff computed from ``report["repros"]`` compares two
    truncated samples and can report "0 cleared, 0 new" from a real change. That happened on
    2026-07-29 (Z5.3). The retention block below makes the truncation legible from the
    artifact instead of resting on the runner's memory of the flags.
    """

    totals: Counter = Counter()
    repros: list[dict[str, Any]] = []
    seeds: list[int] = []
    total_seconds = 0.0
    build_checks: set[str] = set()
    for record in records:
        build_checks.add(str(record.get("build_check", "unknown")))
        totals.update({str(k): int(v) for k, v in (record.get("counters") or {}).items()})
        seeds.append(int(record.get("seed", -1)))
        total_seconds += float(record.get("seconds") or 0.0)
        for repro in record.get("repros") or ():
            if len(repros) < keep_repro:
                repros.append(dict(repro))

    games = len(records)
    measured = totals["boundaries_measured"]
    diverged = totals["transition:diverged"] + totals["engine_error"]
    wall = elapsed if elapsed is not None else total_seconds
    # A report from an ungated run must never read as a gated one — same
    # label-the-output rule as --allow-partial. "unknown" covers pre-field
    # checkpoints, which are equally not-proven-gated.
    gated = build_checks <= {"gated"} and build_checks
    report: dict[str, Any] = {
        "build_check": (
            "gated" if gated
            else "NOT-GATED: " + ",".join(sorted(build_checks or {"unknown"}))
        ),
        "acceptance_eligible": bool(gated),
        "games": games,
        "seeds": {"min": min(seeds), "max": max(seeds), "distinct": len(set(seeds))} if seeds else None,
        "approximate_sleep_turns": approximate_sleep,
        "matcher": matcher,
        # Provenance for identity diffs: is report["repros"] the full divergent set or a
        # truncated sample? Never assert it in prose -- read `repros_complete` here.
        "repro_retention": {
            "repros_per_game": repros_per_game,
            "keep_repro": keep_repro,
            "repros_retained": len(repros),
            "transitions_diverged": totals["transition:diverged"],
            "repros_complete": len(repros) >= totals["transition:diverged"],
        },
        "gating_exact": totals["gating:exact"],
        "gating_support_based": totals["gating:support"],
        "elapsed_seconds": round(wall, 2) if wall else None,
        "games_per_hour": round(games / wall * 3600, 1) if wall else None,
        "boundaries_full_round": totals["boundaries_full_round"],
        "boundaries_measured": measured,
        "transitions_matched": totals["transition:matched"],
        "transitions_diverged": totals["transition:diverged"],
        "engine_errors": totals["engine_error"],
        "divergent_transitions_per_game": round(diverged / games, 4) if games else None,
        "measured_fraction_of_full_rounds": (
            round(measured / totals["boundaries_full_round"], 4)
            if totals["boundaries_full_round"]
            else None
        ),
        "divergence_classes": {
            key.split(":", 1)[1]: value
            for key, value in sorted(totals.items())
            if key.startswith("divergence_class:")
        },
        "counters": dict(sorted(totals.items())),
        "repros": repros,
    }
    if sources:
        report["merged_from"] = list(sources)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showdown-root", default=DEFAULT_SHOWDOWN_ROOT)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=900000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--keep-repro", type=int, default=25)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--approximate-sleep",
        action="store_true",
        help="approximate hidden sleep counters instead of failing the world closed "
             "(default: strict — a publicly-asleep mon with an unknown counter is a "
             "counted SKIP, never a guessed world)",
    )
    parser.add_argument(
        "--skip-build-check",
        action="store_true",
        help="skip the engine build-freshness assertion. Only for offline analysis "
             "(--merge-from), where no engine call is made — a stale build does not "
             "error, it produces a plausible number.",
    )
    parser.add_argument(
        "--matcher",
        choices=("strict", "banded"),
        default="strict",
        help="strict (default): compare per-damage-source components. Deterministic "
             "components (residuals, weather, items, hazards, move heals) must match "
             "exactly; roll-scaled ones (move damage, recoil, drain, confusion) are "
             "accepted if equal, or in the enumerated legal roll set, or within +/-9%% of "
             "the engine's representative roll — component-scoped, never net-HP. "
             "banded: the legacy +/-16%%-of-net-HP band, kept for continuity with the "
             "pre-hardening numbers.",
    )
    parser.add_argument(
        "--no-hidden-counter-support",
        action="store_true",
        help="legacy strict behaviour: fail a boundary closed when a hidden counter "
             "(sleep / confusion duration) is unknown, instead of validating the "
             "realized transition against the union of the counter's legal branch "
             "support. Use only to reproduce the pre-hardening coverage number.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="append one JSONL record per completed game to this path (crash-safe; "
             "pairs with --resume and --merge-from)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="with --checkpoint: skip seeds already recorded and fold their counters "
             "into the final report",
    )
    parser.add_argument(
        "--merge-from",
        type=Path,
        nargs="+",
        default=None,
        help="aggregate these checkpoint files into one report and exit (no games run). "
             "Use to combine acceptance-run shards.",
    )
    parser.add_argument(
        "--repros-per-game",
        type=int,
        default=8,
        help="per-game cap on repros written to the checkpoint (keeps shard files small)",
    )
    args = parser.parse_args(argv)

    # The engine must have been built from the CHECKED-OUT patch set. A stale
    # wheel measured 4.43 % divergence where a HEAD build measured 1.11 % on
    # identical seeds — it fails as a believable number, not as an error.
    skipped_check = bool(args.skip_build_check) and not args.merge_from
    assert_fresh(skip=args.skip_build_check or bool(args.merge_from))
    build_check = "skipped" if skipped_check else "gated"

    # --- merge mode: pure aggregation, no simulator ---
    if args.merge_from:
        records: list[dict[str, Any]] = []
        for path in args.merge_from:
            loaded = load_checkpoint(path)
            print(f"{path}: {len(loaded)} games", flush=True)
            records.extend(loaded)
        seen: set[int] = set()
        deduped: list[dict[str, Any]] = []
        for record in records:
            seed = int(record.get("seed", -1))
            if seed in seen:
                continue
            seen.add(seed)
            deduped.append(record)
        if len(deduped) != len(records):
            print(
                f"warning: dropped {len(records) - len(deduped)} duplicate seeds across shards "
                "(shards should use disjoint seed ranges)",
                file=sys.stderr,
            )
        report = build_report(
            deduped,
            elapsed=None,
            approximate_sleep=None,
            matcher=None,
            keep_repro=args.keep_repro,
            sources=[str(p) for p in args.merge_from],
        )
        if not report.get("acceptance_eligible"):
            print(
                f"WARNING: merged report is {report['build_check']} — at least one shard "
                "ran without the engine build-freshness gate, so this report is NOT "
                "acceptance-eligible.",
                file=sys.stderr,
            )
        print(json.dumps({k: v for k, v in report.items() if k != "repros"}, indent=2))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2))
            print(f"-> {args.json}")
        return 1 if (report["transitions_diverged"] or report["engine_errors"]) else 0

    if args.resume and not args.checkpoint:
        parser.error("--resume requires --checkpoint")

    done_records: list[dict[str, Any]] = []
    done_seeds: set[int] = set()
    if args.checkpoint and args.resume:
        done_records = load_checkpoint(args.checkpoint)
        done_seeds = {int(r["seed"]) for r in done_records}
        print(f"resume: {len(done_seeds)} games already in {args.checkpoint}", flush=True)

    todo = [
        args.seed_start + offset
        for offset in range(args.games)
        if (args.seed_start + offset) not in done_seeds
    ]
    if not todo:
        print("resume: nothing left to run", flush=True)

    dex = load_showdown_dex(args.showdown_root)
    env = LocalShowdownEnv(
        LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True)
    )
    flags_policy = EngineMctsPolicy(
        dex=dex,
        set_source=Gen3RandbatSource.from_showdown_root(args.showdown_root),
        config=EngineMctsConfig(worlds=1, search_time_ms=1),
    )

    records = list(done_records)
    started = time.perf_counter()
    handle = None
    try:
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            handle = args.checkpoint.open("a")
        for index, seed in enumerate(todo, start=1):
            game_started = time.perf_counter()
            game_repros: list[dict[str, Any]] = []
            counts = run_game(
                env=env,
                flags_policy=flags_policy,
                seed=seed,
                dex=dex,
                max_steps=args.max_steps,
                keep_repro=args.repros_per_game,
                repros=game_repros,
                approximate_sleep=args.approximate_sleep,
                hidden_counter_support=not args.no_hidden_counter_support,
                matcher=args.matcher,
            )
            record = checkpoint_record(
                seed=seed,
                counts=counts,
                repros=game_repros,
                seconds=time.perf_counter() - game_started,
                build_check=build_check,
            )
            records.append(record)
            if handle is not None:
                append_checkpoint(handle, record)
            if args.progress_every and index % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                running = build_report(
                    records, elapsed=None, approximate_sleep=None, matcher=None, keep_repro=0
                )
                print(
                    f"[{index}/{len(todo)}] {elapsed:.0f}s "
                    f"({index / elapsed * 3600:.0f} games/h) "
                    f"measured={running['boundaries_measured']} "
                    f"matched={running['transitions_matched']} "
                    f"diverged={running['transitions_diverged']}",
                    flush=True,
                )
    finally:
        if handle is not None:
            handle.close()
        env.close()

    elapsed = time.perf_counter() - started
    report = build_report(
        records,
        # Only the games run in THIS process contribute to the wall clock; a
        # resumed run's throughput would otherwise be nonsense.
        elapsed=elapsed if len(records) == len(todo) else None,
        approximate_sleep=bool(args.approximate_sleep),
        matcher=args.matcher,
        keep_repro=args.keep_repro,
        repros_per_game=args.repros_per_game,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "repros"}, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"-> {args.json}")
    return 1 if (report["transitions_diverged"] or report["engine_errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
