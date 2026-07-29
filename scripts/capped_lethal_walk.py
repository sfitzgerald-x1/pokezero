#!/usr/bin/env python
"""Per-row branch walk for `capped_lethal`-majority rows, under Appendix X as
amended (#951): mass-gated, roll-consistent, four-exit.

The question, per row: is Showdown's observed outcome reachable under some
legal (branch, roll assignment) the engine already prices — with the
reproducing assignments' TOTAL probability under the engine's own distribution
clearing a 1% floor — or is it reachable under none, making the row a
`damage_calc` disagreement with a quantifiable gap?

Implementation of the standard:

* COMPARISON TARGET (X.2): observed post-state HP vector of both actives, the
  faint set, AND the multiset of non-residual, non-capped damage components.
  Residual and faint-capped components are exempt (they are what
  `capped_lethal` makes incomparable); ordinary move damage is not.
* ROLL CONSISTENCY (X.3.2): one roll index per MOVE INSTANCE, applied to that
  move's damage wherever it appears in the branch. Distinct moves roll
  independently; each roll index carries probability 1/16.
* MASS (X.3.1): sum over reproducing (branch, assignment) pairs of
  branch_pct x (1/16)^n_moves. `limit` requires >= 1%. Reported per candidate
  engine state (the hidden-counter sweep); the row's mass is the max across
  candidates, and the candidate index is recorded.
* FOUR EXITS (X.3.3): limit / damage_calc(+gap) / limit_not_established /
  cannot_enumerate. `cannot_enumerate` only on the named preconditions
  (X.3.4); each claim quotes the failing precondition.

Branch arithmetic: the engine's branches APPLY trunc(0.925 * max) (symbol
`avg_damage_dealt`, gen3/generate_instructions.rs), while the legal roll set
for a move with 100%-roll base m is {floor(m * p / 100) for p in 85..100}
(Z.0). For each bare `-damage` event the walk recovers m from
`poke_engine.calculate_damage` on the branch's own candidate state (both move
orders, crit and non-crit arms) by requiring trunc(0.925 * m) == the branch's
applied value; when the applied value was itself faint-capped or no candidate
matches (same-turn stat change), m is inferred from the applied value and the
inference is recorded on the row. Substituting a roll re-walks the branch's
event list with faint-pattern consistency enforced at every damage event: an
assignment that faints where the branch did not (or vice versa) belongs to a
DIFFERENT branch of the engine's distribution and is discarded here.

Usage::

    PYTHONPATH=src python scripts/capped_lethal_walk.py \\
        --report c9_run.json --rows 1500012:24,1500050:33 [--json out.json]

Exit code 0 always — adjudication evidence, not a gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import poke_engine  # noqa: E402
import pokezero_search  # noqa: E402

from engine_transition_differential import (  # noqa: E402
    _ROLL_SCALED_SOURCES,
    damage_components,
)

MASS_FLOOR_PCT = 1.0  # X.3.1: limit requires reproducing mass >= 1%
_ROLLS = tuple(range(85, 101))  # gen3: floor(m * p / 100), 16 outcomes

_HP_RE = re.compile(r"^(\d+)(?:/(\d+))?")


def _parse_hp(token: str) -> tuple[int, int | None]:
    """Return (hp, maxhp|None) from a Showdown condition token ('157/209',
    '0 fnt')."""
    token = token.strip()
    if token.startswith("0 fnt") or token == "0":
        return 0, None
    m = _HP_RE.match(token)
    if not m:
        return 0, None
    return int(m.group(1)), int(m.group(2)) if m.group(2) else None


def _slot_of(ident: str) -> str | None:
    ident = ident.strip()
    if ident.startswith("p1"):
        return "p1"
    if ident.startswith("p2"):
        return "p2"
    return None


class BranchOp:
    """One HP-affecting event of a rendered branch, in order."""

    __slots__ = ("kind", "slot", "delta", "move_key", "to_full", "maxhp",
                 "source")

    def __init__(self, kind: str, slot: str, delta: int, move_key: int | None,
                 to_full: bool, maxhp: int | None, source: str = "") -> None:
        self.kind = kind          # "roll" (bare move -damage) | "fixed"
        self.slot = slot
        self.delta = delta        # branch's own signed delta
        self.move_key = move_key  # roll-group id for kind == "roll"
        self.to_full = to_full    # heal that topped the mon out in-branch
        self.maxhp = maxhp
        self.source = source      # normalized [from] tag for kind == "fixed"


def _normalize_source(from_tag: str | None) -> str:
    if not from_tag:
        return ""
    body = from_tag.partition("]")[2].strip()
    body = body.partition(":")[2].strip() if ":" in body else body
    return re.sub(r"[^a-z0-9]", "", body.lower())


def _uncapped_residual_tick(source: str, maxhp: int,
                            ctx: Mapping[str, Any]) -> int | None:
    """The engine's own UNCAPPED residual magnitude, transcribed from the
    vendored gen3/generate_instructions.rs residual blocks (10.4-10.6):

      burn:      max(trunc(maxhp * 0.125), 1)
      poison:    max(1, trunc(maxhp * 0.125))
      toxic:     max(maxhp // 16, 1) * (toxic_count + 1)   [per-stage floored]
      sandstorm: max(maxhp // 16, 1)
      leechseed: trunc(maxhp * 0.125)                       [no min-1 clamp]

    A branch caps these at remaining HP; substituting a different move roll
    changes the cap, so the walk needs the uncapped value. Returns None for a
    source this table does not carry (the caller records the row as
    unresolved rather than guessing)."""

    if source == "psn":
        if str(ctx.get("status", "")).upper() == "TOXIC":
            return max(maxhp // 16, 1) * (int(ctx.get("toxic_count", 0)) + 1)
        return max(1, int(maxhp * 0.125))
    if source == "brn":
        return max(int(maxhp * 0.125), 1)
    if source == "sandstorm":
        return max(maxhp // 16, 1)
    if source == "leechseed":
        return int(maxhp * 0.125)
    return None


def _parse_branch(events: Sequence[str], pre_hp: Mapping[str, int]) -> tuple[
        list[BranchOp], dict[int, dict[str, Any]], list[str]]:
    """Parse rendered branch events into ordered ops + per-move roll groups.

    Returns (ops, move_groups, warnings). move_groups[key] = {"crit": bool,
    "applied": int (uncapped-if-known branch value), "capped": bool}.
    """

    ops: list[BranchOp] = []
    groups: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    running = dict(pre_hp)
    maxhp: dict[str, int | None] = {"p1": None, "p2": None}
    current_move: int | None = None
    crit_pending = False
    next_key = 0

    for line in events:
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag == "move":
            current_move = None  # allocate lazily on first bare -damage
            crit_pending = False
            continue
        if tag == "-crit":
            crit_pending = True
            continue
        if tag in {"switch", "drag"}:
            slot = _slot_of(parts[2]) if len(parts) > 2 else None
            if slot and len(parts) > 4:
                hp, mx = _parse_hp(parts[4])
                running[slot] = hp
                if mx:
                    maxhp[slot] = mx
            warnings.append(f"active changed mid-branch ({tag}); HP tracking rebased")
            continue
        if tag not in {"-damage", "-heal", "-sethp"}:
            continue
        slot = _slot_of(parts[2]) if len(parts) > 2 else None
        if slot is None:
            continue
        hp, mx = _parse_hp(parts[3]) if len(parts) > 3 else (0, None)
        if mx:
            maxhp[slot] = mx
        prev = running.get(slot)
        if prev is None:
            running[slot] = hp
            continue
        delta = hp - prev
        running[slot] = hp
        if delta == 0:
            continue
        from_tag = next((p for p in parts[4:] if p.startswith("[from]")), None)
        bare_damage = tag == "-damage" and from_tag is None
        if bare_damage:
            # One roll group per bare -damage event. A single-hit move has
            # exactly one; a multi-hit move rolls per hit in the gen3 sim, so
            # per-event groups are the faithful reading of X.3.2's "a move
            # rolls once" for the states these rows actually contain.
            current_move = next_key
            next_key += 1
            groups[current_move] = {"crit": crit_pending, "applied": -delta,
                                    "capped": hp == 0}
            ops.append(BranchOp("roll", slot, delta, current_move, False, maxhp[slot]))
        else:
            source = _normalize_source(from_tag)
            to_full = tag == "-heal" and maxhp[slot] is not None and hp == maxhp[slot]
            # Recoil/drain DERIVE from the preceding move's dealt damage
            # (engine: trunc(damage_dealt * fraction), gen3/
            # generate_instructions.rs recoil arm; drain analogous), so they
            # re-scale with a substituted roll. Link them to the move's roll
            # group; the fraction is inferred from the branch's own pair.
            link = current_move if source in {"recoil", "drain"} else None
            ops.append(BranchOp("fixed", slot, delta, link, to_full, maxhp[slot],
                                source))
        crit_pending = False
    return ops, groups, warnings


def _slot_context(state_str: str,
                  slot_sides: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Per-slot active maxhp / status / toxic_count from the candidate state,
    for reconstructing faint-capped residual magnitudes."""

    state = poke_engine.State.from_string(state_str)
    sides = {"side_one": state.side_one, "side_two": state.side_two}
    out: dict[str, dict[str, Any]] = {}
    for slot in ("p1", "p2"):
        side = sides[slot_sides.get(slot, "side_one" if slot == "p1" else "side_two")]
        try:
            active = side.pokemon[int(str(side.active_index))]
        except (ValueError, IndexError):
            out[slot] = {}
            continue
        volatiles = {str(v).upper() for v in side.volatile_statuses}
        future_sight = getattr(side, "future_sight", (0, "0"))
        out[slot] = {
            "maxhp": int(active.maxhp),
            "status": str(active.status),
            "toxic_count": int(getattr(side.side_conditions, "toxic_count", 0)),
            # PERISH1 at the pre-state means this side's generic `[from]
            # residual` damage this turn is the Perish Song kill, which the
            # engine deals as the mon's ENTIRE remaining HP (gen3/
            # generate_instructions.rs, Perish Song residual: damage_amount =
            # active_pkmn.hp) — always lethal under any substituted roll.
            "perish1": "PERISH1" in volatiles,
            # A branch whose status tick (order 10.6) KILLED shows nothing
            # after the faint. Un-fainting that mon under a substituted roll
            # is sound only when the engine's own residual order table
            # (gen3/generate_instructions.rs, "The gen3 order table") has
            # nothing left to fire on it: partial trap is 10.9, Future Sight
            # 11, Perish Song 12. Checked against the candidate state, not
            # assumed.
            "safe_unfaint_after_status": (
                "PARTIALLYTRAPPED" not in volatiles
                and not any(v.startswith("PERISH") for v in volatiles)
                and int(future_sight[0]) == 0
            ),
        }
    return out


def _candidate_maxes(state_str: str, s1: str, s2: str) -> dict[str, set[tuple[int, bool]]]:
    """All (100%-roll base, crit) candidates per side from calculate_damage,
    both move orders."""

    out: dict[str, set[tuple[int, bool]]] = {"p1": set(), "p2": set()}
    state = poke_engine.State.from_string(state_str)
    for first in (True, False):
        try:
            s1_rolls, s2_rolls = poke_engine.calculate_damage(state, s1, s2, first)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # noqa: BLE001
            continue
        for side, rolls in (("p1", s1_rolls), ("p2", s2_rolls)):
            if len(rolls) >= 1 and rolls[0] > 0:
                out[side].add((int(rolls[0]), False))
            if len(rolls) >= 2 and rolls[1] > 0:
                out[side].add((int(rolls[1]), True))
    return out


def _roll_bases_for_group(group: Mapping[str, Any], attacker_slot: str,
                          maxes: Mapping[str, set[tuple[int, bool]]]) -> tuple[list[int], str]:
    """Candidate 100%-roll bases m for a move whose branch applied `applied`.

    Prefers exact trunc(0.925 * m) == applied against calculate_damage's
    candidates for the ATTACKING slot; falls back to inverting the 0.925
    factor (recorded). A faint-capped applied value cannot pin m, so the
    fallback then spans every m whose minimum roll still reaches the cap."""

    applied = int(group["applied"])
    capped = bool(group["capped"])
    exact = [m for m, _crit in maxes.get(attacker_slot, ())
             if int(m * 0.925) == applied]
    if exact and not capped:
        return sorted(set(exact)), "calculate_damage"
    if capped:
        # Any m from calculate_damage whose branch application would reach the
        # cap; if none, infer the minimal family from the cap value itself.
        reach = [m for m, _crit in maxes.get(attacker_slot, ())
                 if int(m * 0.925) >= applied]
        if reach:
            return sorted(set(reach)), "calculate_damage(capped)"
        return [applied], "inferred-from-cap(min-family)"
    # No exact candidate (same-turn stat change moved the base): invert.
    inferred = [m for m in range(int(applied / 0.925) - 1, int((applied + 1) / 0.925) + 2)
                if m > 0 and int(m * 0.925) == applied]
    return inferred or [applied], "inverted-0.925"


def _simulate(ops: Sequence[BranchOp], pre_hp: Mapping[str, int],
              assignment: Mapping[int, int],
              bases: Mapping[int, int],
              slot_ctx: Mapping[str, Mapping[str, Any]]) -> tuple[
                  dict[str, int], set[str], list[int], bool, bool]:
    """Re-walk a branch under a roll assignment.

    Returns (post_hp, fainted, NON-CAPPED substituted move damages,
    consistent, unresolved). A faint-capped substituted damage is exempt from
    the multiset target (X.2), so it is not returned. A substitution whose
    faint pattern diverges from the branch's own is inconsistent (that
    outcome lives in a different branch). `unresolved` is True when a
    faint-capped residual's uncapped magnitude could not be transcribed from
    the engine source table — the caller must refuse rather than guess."""

    hp = dict(pre_hp)
    hp_branch = dict(pre_hp)
    fainted: set[str] = set()
    sub_damages: list[int] = []
    sub_dealt_by_group: dict[int, int] = {}
    branch_dealt_by_group: dict[int, int] = {}
    n_ops = len(ops)

    def _later_op_for_slot(idx: int, slot: str) -> bool:
        return any(ops[j].slot == slot for j in range(idx + 1, n_ops))

    for idx, op in enumerate(ops):
        if op.kind == "roll":
            m = bases[op.move_key]
            pct = _ROLLS[assignment[op.move_key]]
            value = m * pct // 100
            dealt = min(value, hp[op.slot])
            kills_sub = value >= hp[op.slot]
            kills_branch = -op.delta >= hp_branch[op.slot]
            if not kills_sub:
                sub_damages.append(dealt)
            sub_dealt_by_group[op.move_key] = dealt
            branch_dealt_by_group[op.move_key] = -op.delta
            hp[op.slot] -= dealt
            hp_branch[op.slot] = max(0, hp_branch[op.slot] + op.delta)
            if kills_sub != kills_branch:
                return hp, fainted, sub_damages, False, False
            if kills_sub:
                fainted.add(op.slot)
        elif op.source in {"recoil", "drain"} and op.move_key is not None:
            # Derived from the linked move's DEALT damage: engine computes
            # trunc(dealt * fraction). The fraction is recovered from the
            # branch's own (dealt, recoil) pair over the engine's move data
            # constants; if the candidates disagree on the substituted value
            # the walk refuses rather than guesses.
            b_dealt = branch_dealt_by_group.get(op.move_key)
            s_dealt = sub_dealt_by_group.get(op.move_key)
            if b_dealt is None or s_dealt is None or b_dealt <= 0:
                return hp, fainted, sub_damages, False, True
            magnitude = abs(op.delta)
            fracs = [f for f in (0.25, 0.33, 1.0 / 3.0, 0.5, 1.0)
                     if int(b_dealt * f) == magnitude]
            sub_values = {int(s_dealt * f) for f in fracs}
            if not fracs or len(sub_values) != 1:
                return hp, fainted, sub_damages, False, True
            sub_mag = sub_values.pop()
            if op.delta < 0:
                value = sub_mag
                dealt = min(value, hp[op.slot])
                kills_sub = value >= hp[op.slot]
                kills_branch = magnitude >= hp_branch[op.slot]
                if not kills_sub:
                    sub_damages.append(dealt)
                hp[op.slot] -= dealt
                hp_branch[op.slot] = max(0, hp_branch[op.slot] - magnitude)
                if kills_sub != kills_branch:
                    if kills_sub and not _later_op_for_slot(idx, op.slot):
                        fainted.add(op.slot)
                        continue
                    return hp, fainted, sub_damages, False, False
                if kills_sub:
                    fainted.add(op.slot)
            else:
                new = hp[op.slot] + sub_mag
                if op.maxhp is not None:
                    new = min(new, op.maxhp)
                hp[op.slot] = new
                hp_branch[op.slot] = min(
                    hp_branch[op.slot] + op.delta,
                    op.maxhp if op.maxhp is not None else 10 ** 9)
        else:
            kills_branch = op.delta < 0 and hp_branch[op.slot] + op.delta <= 0
            if op.delta < 0 and kills_branch:
                # The branch capped this residual at ITS remaining HP; under a
                # substituted roll the cap moves, so the uncapped tick is
                # needed. Transcribed from the engine's own residual blocks.
                ctx = slot_ctx.get(op.slot, {})
                maxhp = op.maxhp or int(ctx.get("maxhp", 0))
                if op.source == "residual" and ctx.get("perish1"):
                    tick = None
                    dealt = hp[op.slot]      # Perish Song: kills outright
                    kills_sub = True
                else:
                    tick = _uncapped_residual_tick(op.source, maxhp, ctx)
                    if tick is None:
                        return hp, fainted, sub_damages, False, True
                    dealt = min(tick, hp[op.slot])
                    kills_sub = tick >= hp[op.slot]
                hp[op.slot] -= dealt
                hp_branch[op.slot] = max(0, hp_branch[op.slot] + op.delta)
                if kills_sub != kills_branch:
                    if kills_branch and not kills_sub:
                        # The branch's capped tick killed; this roll leaves the
                        # mon alive. Sound only when nothing later in the
                        # engine's residual order could still fire on it.
                        if op.source in {"psn", "brn"} and ctx.get(
                                "safe_unfaint_after_status"):
                            continue
                        return hp, fainted, sub_damages, False, True
                    return hp, fainted, sub_damages, False, False
                if kills_sub:
                    fainted.add(op.slot)
                continue
            if op.to_full and op.maxhp is not None:
                new = op.maxhp
            elif op.delta < 0 and op.source in {"psn", "brn", "sandstorm",
                                                "leechseed"}:
                # An uncapped-in-branch residual can still CAP under a
                # substituted roll that left less HP: the tick magnitude is
                # state-fixed, the cap is min(tick, hp).
                new = hp[op.slot] - min(-op.delta, hp[op.slot])
            else:
                new = hp[op.slot] + op.delta
                if op.maxhp is not None:
                    new = min(new, op.maxhp)
            if op.delta < 0:
                kills_sub = new <= 0
                if kills_sub != kills_branch:
                    if kills_sub and not _later_op_for_slot(idx, op.slot):
                        # The substituted roll left little enough HP that this
                        # residual now kills. Nothing later in the branch
                        # touches the slot, so the reconstruction is complete.
                        hp[op.slot] = 0
                        hp_branch[op.slot] = max(0, hp_branch[op.slot] + op.delta)
                        fainted.add(op.slot)
                        continue
                    return hp, fainted, sub_damages, False, False
                if kills_sub:
                    fainted.add(op.slot)
            hp[op.slot] = max(0, new)
            hp_branch[op.slot] = max(0, min(
                hp_branch[op.slot] + op.delta,
                op.maxhp if op.maxhp is not None else 10 ** 9))
    return hp, fainted, sub_damages, True, False


def _observed_target(row: Mapping[str, Any]) -> tuple[dict[str, int], set[str], list[int]]:
    pre = row.get("pre_features") or {}
    pre_hp = {"p1": int(pre.get("p1_hp", 0)), "p2": int(pre.get("p2_hp", 0))}
    obs = row.get("observed") or {}
    post_hp = {"p1": int(obs.get("p1_hp", 0)), "p2": int(obs.get("p2_hp", 0))}
    fainted = {str(s) for s in (obs.get("fainted") or ())}
    protocol = [l for l in (row.get("protocol") or []) if not l.startswith("|request|")]
    components = damage_components(protocol, pre_hp)
    move_damages: list[int] = []
    for slot in ("p1", "p2"):
        for source, delta in components[slot]:
            if delta < 0 and source in _ROLL_SCALED_SOURCES and source != "capped_lethal":
                move_damages.append(-delta)
    return post_hp, fainted, sorted(move_damages)


def walk_row(row: Mapping[str, Any]) -> dict[str, Any]:
    seed, step = row.get("seed"), row.get("step")
    result: dict[str, Any] = {
        "seed": seed, "step": step,
        "divergence_class": row.get("divergence_class"),
        "choices": row.get("choices"),
    }
    states = row.get("engine_states") or (
        [row["engine_state"]] if row.get("engine_state") else [])
    if not states:
        result["verdict"] = "cannot_enumerate"
        result["precondition"] = "the recorded row lacks a field the walk requires (no engine_states)"
        return result
    pre = row.get("pre_features")
    if not pre:
        result["verdict"] = "cannot_enumerate"
        result["precondition"] = "the recorded row lacks a field the walk requires (no pre_features)"
        return result

    post_hp_obs, fainted_obs, move_dmg_obs = _observed_target(row)
    result["observed_target"] = {
        "post_hp": post_hp_obs, "fainted": sorted(fainted_obs),
        "non_residual_move_damage": move_dmg_obs,
    }
    pre_hp = {"p1": int(pre.get("p1_hp", 0)), "p2": int(pre.get("p2_hp", 0))}
    choices = row.get("choices") or {}
    slot_sides = row.get("slot_sides") or {"p1": "side_one", "p2": "side_two"}
    s1 = choices["p1"] if slot_sides.get("p1") == "side_one" else choices["p2"]
    s2 = choices["p2"] if slot_sides.get("p2") == "side_two" else choices["p1"]
    ctx = json.dumps({"p1": [], "p2": [], "turn": 0})

    candidates_out: list[dict[str, Any]] = []
    best_mass = 0.0
    best_candidate = None
    closest: tuple[int, dict[str, Any]] | None = None
    any_branches = False
    total_unresolved = 0

    for index, state_str in enumerate(states):
        try:
            rendered = json.loads(
                pokezero_search.branch_events(state_str, s1, s2, ctx, True, True))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # noqa: BLE001
            candidates_out.append({
                "candidate": index,
                "error": f"branch rendering failed: {type(error).__name__}: {error}"})
            continue
        branches = rendered.get("branches") or []
        if not branches:
            candidates_out.append({"candidate": index, "error": "no branches"})
            continue
        any_branches = True
        maxes = _candidate_maxes(state_str, s1, s2)
        slot_ctx = _slot_context(state_str, slot_sides)
        cand_mass = 0.0
        cand_unresolved = 0
        branch_rows: list[dict[str, Any]] = []
        for branch in branches:
            pct = float(branch.get("percentage") or 0.0)
            ops, groups, warnings = _parse_branch(branch.get("events") or [], pre_hp)
            brow: dict[str, Any] = {"pct": pct, "n_roll_moves": len(groups),
                                    "warnings": warnings}
            if not groups:
                # No roll-scaled move damage: the branch is deterministic.
                post_hp, fainted, subs, ok, unresolved = _simulate(
                    ops, pre_hp, {}, {}, slot_ctx)
                cand_unresolved += int(unresolved)
                match = (ok and post_hp == post_hp_obs and fainted == fainted_obs
                         and sorted(subs) == move_dmg_obs)
                brow["deterministic_post_hp"] = post_hp
                if match:
                    cand_mass += pct
                    brow["reproduces"] = True
                branch_rows.append(brow)
                dist = abs(post_hp["p1"] - post_hp_obs["p1"]) + abs(post_hp["p2"] - post_hp_obs["p2"])
                if ok and (closest is None or dist < closest[0]):
                    closest = (dist, {"candidate": index, "pct": pct, "post_hp": post_hp,
                                      "assignment": None})
                continue
            keys = sorted(groups)
            base_options: list[list[int]] = []
            base_srcs: list[str] = []
            damaged_slot: dict[int, str] = {}
            for key in keys:
                slot = next(op.slot for op in ops if op.kind == "roll" and op.move_key == key)
                damaged_slot[key] = slot
                attacker = "p2" if slot == "p1" else "p1"
                bases, src = _roll_bases_for_group(groups[key], attacker, maxes)
                base_options.append(bases)
                base_srcs.append(src)
            brow["roll_bases"] = {str(k): {"options": base_options[i], "source": base_srcs[i]}
                                  for i, k in enumerate(keys)}
            reproducing = 0
            total = 0
            ladder: list[dict[str, Any]] = []
            # Base ambiguity: iterate every combination of candidate bases, but
            # count assignment probability ONCE per roll assignment (a real
            # engine has ONE base per move; ambiguity may only widen the
            # reachable set, so mass uses the best single base combination).
            best_combo_mass = 0.0
            for combo in product(*base_options):
                bases = dict(zip(keys, combo))
                combo_hits = 0
                combo_total = 0
                for assignment_tuple in product(range(16), repeat=len(keys)):
                    assignment = dict(zip(keys, assignment_tuple))
                    post_hp, fainted, subs, ok, unresolved = _simulate(
                        ops, pre_hp, assignment, bases, slot_ctx)
                    combo_total += 1
                    cand_unresolved += int(unresolved)
                    if not ok:
                        continue
                    # X.2 target: post-state HP vector + faint set + multiset
                    # of non-residual, NON-CAPPED move damage (a substituted
                    # damage that fainted its target is faint-capped and
                    # exempt, mirroring the observed side's capped_lethal
                    # exemption; _simulate already excludes it).
                    match = (post_hp == post_hp_obs and fainted == fainted_obs
                             and sorted(subs) == move_dmg_obs)
                    if match:
                        combo_hits += 1
                    dist = (abs(post_hp["p1"] - post_hp_obs["p1"])
                            + abs(post_hp["p2"] - post_hp_obs["p2"]))
                    if ok and (closest is None or dist < closest[0]):
                        closest = (dist, {"candidate": index, "pct": pct,
                                          "post_hp": post_hp,
                                          "assignment": dict(assignment),
                                          "bases": dict(bases)})
                combo_mass = pct * combo_hits / (16 ** len(keys))
                if combo_mass > best_combo_mass:
                    best_combo_mass = combo_mass
                    reproducing, total = combo_hits, combo_total
                    ladder = [{"move": k, "base": bases[k],
                               "rolls": [bases[k] * p // 100 for p in _ROLLS]}
                              for k in keys]
            brow["reproducing_assignments"] = reproducing
            brow["assignments_total"] = total
            brow["mass_pct"] = best_combo_mass
            brow["roll_ladder"] = ladder
            cand_mass += best_combo_mass
            branch_rows.append(brow)
        candidates_out.append({"candidate": index, "branches": branch_rows,
                               "mass_pct": cand_mass,
                               "unresolved_assignments": cand_unresolved})
        total_unresolved += cand_unresolved
        if cand_mass > best_mass:
            best_mass = cand_mass
            best_candidate = index

    result["candidates"] = candidates_out
    result["mass_pct"] = best_mass
    result["mass_candidate"] = best_candidate
    result["unresolved_assignments"] = total_unresolved
    if not any_branches:
        result["verdict"] = "cannot_enumerate"
        result["precondition"] = (
            "generate_instructions raises, or returns no branch, on every recorded state")
        return result
    if best_mass >= MASS_FLOOR_PCT:
        result["verdict"] = "limit:roll_divergent_lethality"
    elif best_mass > 0.0:
        result["verdict"] = "limit_not_established"
    elif total_unresolved:
        # Some assignments could not be resolved (a capped residual whose
        # uncapped magnitude is not in the transcription table). "Reachable
        # under NO assignment" cannot be claimed over a walk with holes, and
        # X.3.4 does not license cannot_enumerate for an inconclusive walk:
        # the row keeps its label.
        result["verdict"] = "limit_not_established"
        result["note"] = ("walk incomplete: unresolved capped-residual "
                          "assignments; damage_calc NOT claimed")
    else:
        result["verdict"] = "damage_calc"
        if closest is not None:
            result["quantified_gap"] = {
                "closest_achievable_post_hp": closest[1]["post_hp"],
                "observed_post_hp": post_hp_obs,
                "l1_distance": closest[0],
                "at": {k: v for k, v in closest[1].items() if k != "post_hp"},
            }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--rows", required=True,
                    help="comma-separated seed:step list, e.g. 1500012:24,1500050:33")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    wanted = set()
    for chunk in args.rows.split(","):
        seed_s, _, step_s = chunk.strip().partition(":")
        wanted.add((int(seed_s), int(step_s)))

    report = json.loads(args.report.read_text())
    repros = report.get("repros") or []
    by_key = {(int(r.get("seed", -1)), int(r.get("step", -1))): r for r in repros}

    results = []
    for key in sorted(wanted):
        row = by_key.get(key)
        if row is None:
            results.append({"seed": key[0], "step": key[1],
                            "verdict": "cannot_enumerate",
                            "precondition": "row not present in the report "
                                            "(no recorded repro to walk)"})
            continue
        results.append(walk_row(row))

    for res in results:
        print(f"s{res['seed']} st{res['step']}: {res['verdict']}"
              f"  mass={res.get('mass_pct', 0):.3f}%"
              + (f"  [{res.get('precondition')}]" if res.get("precondition") else ""))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
