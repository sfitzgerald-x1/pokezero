#!/usr/bin/env python
"""Per-row replay triage of the ``roll_scaled_component`` residue class.

Applies the ledger's method rule (replay before you label) to every row of a
class at once, instead of eyeballing samples. For each recorded divergent
boundary it re-runs the exact state + joint action through the engine and asks,
in order:

  (a) LEGITIMATE ROLL — is the observed magnitude inside the engine's own legal
      roll set for this state (``calculate_damage`` gives the 100 % base for
      non-crit and crit, and gen3 rolls ``floor(base * random(85,100) / 100)``)?
      If yes the engine CAN produce what Showdown produced, so a divergence here
      is a MATCHER accounting bug, not an engine one.

  (b) DAMAGE-CALC DISAGREEMENT — otherwise, how far off is it? The ratio
      observed/engine is bucketed against the multipliers a gen3 damage
      calculation is built from (1/2 screen, 2x crit, 2x/4x/1/2/1/4 type,
      1.5x STAB / Sand SpD), and the pre-state's screens, weather, boosts,
      items, abilities and types are dumped alongside so the implicated
      mechanism is visible rather than guessed.

  (c) STRUCTURAL — component counts differ (one sim hit, the other missed; an
      extra residual), which is not a magnitude question at all.

Usage::

    PYTHONPATH=src python scripts/triage_roll_components.py \\
        --report triage_full.json [--class roll_scaled_component] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import poke_engine  # noqa: E402
import pokezero_search  # noqa: E402

from engine_transition_differential import (  # noqa: E402
    _split_components,
    damage_components,
    legal_roll_damages,
)

# Ratio observed/engine -> the gen3 multiplier it implicates. Tolerance is the
# roll spread (+/-9 %), so these do not overlap.
_MULTIPLIER_BUCKETS = (
    (0.25, "quarter (type 1/4x, or double resist)"),
    (0.50, "half (screen: Reflect/Light Screen, or type 1/2x)"),
    (0.67, "two-thirds"),
    (1.50, "one-and-a-half (STAB, or Sand SpD)"),
    (2.00, "double (crit, or type 2x)"),
    (4.00, "quadruple (type 4x)"),
)
# The engine reports `trunc(0.925 * max_damage)`, so its value is NOT the mean
# roll and NOT the maximum. Against Showdown's 85%-100% ladder the honest low
# edge is 0.85/0.925 = 0.919, not 0.92. The old 0.92 was a guess, and a guessed
# threshold is what made an eleven-row band look like a filter artifact in the
# first place (Appendix Y) -- the reasoning was sound, the constant was invented.
#
# Prefer `observed_in_legal_set` (roll-set membership) wherever it is available:
# it needs no constant and cannot grow a floor artifact. These bounds remain only
# for the descriptive ratio labels.
_ROLL_LOW, _ROLL_HIGH = 0.919, 1.09


def _state_context(state: Any) -> dict[str, Any]:
    """Pre-state facts a damage disagreement could plausibly hinge on."""

    out: dict[str, Any] = {}
    for label, side in (("p1", state.side_one), ("p2", state.side_two)):
        active = side.pokemon[int(str(side.active_index))]
        out[label] = {
            "species": str(active.id),
            "types": [str(t) for t in active.types],
            "item": str(active.item),
            "ability": str(active.ability),
            "status": str(active.status),
            "hp": f"{int(active.hp)}/{int(active.maxhp)}",
            "atk": int(active.attack), "def": int(active.defense),
            "spa": int(active.special_attack), "spd": int(active.special_defense),
            "boosts": {
                n: int(getattr(side, f"{n}_boost", 0) or 0)
                for n in ("attack", "defense", "special_attack", "special_defense", "speed")
                if int(getattr(side, f"{n}_boost", 0) or 0)
            },
            "reflect": int(getattr(side.side_conditions, "reflect", 0) or 0),
            "light_screen": int(getattr(side.side_conditions, "light_screen", 0) or 0),
        }
    out["weather"] = str(state.weather)
    return out


def _pair_by_source(
    observed: Sequence[tuple[str, int]], engine: Sequence[tuple[str, int]]
) -> list[tuple[str, int, int]]:
    """Pair observed against engine components BY SOURCE, magnitude only within it.

    The previous version sorted both sides by bare magnitude and zipped. In a
    slot with more than one component that pairs unlike things: a 2-point
    residual against a 136-point move hit, reported as a ratio of 0.015 that
    describes nothing. 15 of 54 multi-component findings in the cycle-seven
    brief were unusable for this reason, and were excluded rather than trusted —
    dark data feeding nothing.

    Source is the meaningful key: `recoil` belongs with `recoil` and a move hit
    with a move hit, whatever their magnitudes. Within a source, magnitude order
    is the only available tiebreak and is used. Components with no counterpart
    on the other side are dropped here — an unmatched COUNT is a structural
    difference, which `structural_component_count` already reports; it is not a
    ratio and must not be manufactured into one.
    """

    by_source: dict[str, list[list[int]]] = {}
    for index, side in ((0, observed), (1, engine)):
        for source, delta in side:
            by_source.setdefault(source, [[], []])[index].append(abs(delta))
    out: list[tuple[str, int, int]] = []
    for source in sorted(by_source):
        obs, eng = by_source[source]
        for a, b in zip(sorted(obs), sorted(eng)):
            out.append((source, a, b))
    return out


def _classify_ratio(ratio: float) -> str:
    if _ROLL_LOW <= ratio <= _ROLL_HIGH:
        return "within_roll_window"
    for value, label in _MULTIPLIER_BUCKETS:
        if value * _ROLL_LOW <= ratio <= value * _ROLL_HIGH:
            return label
    return f"unexplained_ratio_{ratio:.2f}"


def triage_row(row: Mapping[str, Any], *, verbose: bool) -> dict[str, Any]:
    choices = row.get("choices") or {}
    slot_sides = row.get("slot_sides") or {"p1": "side_one", "p2": "side_two"}
    s1 = choices["p1"] if slot_sides.get("p1") == "side_one" else choices["p2"]
    s2 = choices["p2"] if slot_sides.get("p2") == "side_two" else choices["p1"]
    engine_label = {slot: ("p1" if slot_sides.get(slot) == "side_one" else "p2")
                    for slot in ("p1", "p2")}

    pre = row.get("pre_features") or {}
    pre_hp = {"p1": int(pre.get("p1_hp", 0)), "p2": int(pre.get("p2_hp", 0))}
    protocol = [l for l in (row.get("protocol") or []) if not l.startswith("|request|")]
    observed = damage_components(protocol, pre_hp)
    obs_rolled = {slot: _split_components(observed[slot])[1] for slot in ("p1", "p2")}

    states = row.get("engine_states") or ([row["engine_state"]] if row.get("engine_state") else [])
    party = row.get("party_display") or {"p1": [], "p2": []}
    ctx = json.dumps({"p1": party.get("p1", []), "p2": party.get("p2", []), "turn": 0})

    # Union of every legal damage the engine could roll from any candidate.
    legal: set[int] = set()
    context = None
    for state_str in states:
        try:
            state = poke_engine.State.from_string(state_str)
        except BaseException:  # noqa: BLE001
            continue
        if context is None:
            context = _state_context(state)
        for first in (True, False):
            try:
                r1, r2 = poke_engine.calculate_damage(state, s1, s2, first)
                legal |= legal_roll_damages(list(r1) + list(r2))
            except BaseException:  # noqa: BLE001
                pass

    # Best engine branch = the one whose roll-scaled multiset is closest.
    best: dict[str, Any] | None = None
    for state_str in states:
        try:
            rendered = json.loads(
                pokezero_search.branch_events(state_str, s1, s2, ctx, True, True)
            )
        except BaseException:  # noqa: BLE001
            continue
        for branch in rendered.get("branches") or []:
            if branch.get("lossy") or float(branch.get("percentage") or 0) <= 0:
                continue
            eng = damage_components(
                branch.get("events") or [],
                {engine_label[slot]: pre_hp[slot] for slot in ("p1", "p2")},
            )
            eng_rolled = {slot: _split_components(eng[engine_label[slot]])[1]
                          for slot in ("p1", "p2")}
            score = 0.0
            structural = False
            for slot in ("p1", "p2"):
                o = sorted(abs(d) for _s, d in obs_rolled[slot])
                e = sorted(abs(d) for _s, d in eng_rolled[slot])
                if len(o) != len(e):
                    structural = True
                    score += 100.0
                    continue
                for a, b in zip(o, e):
                    score += abs(a - b) / max(b, 1)
            if best is None or score < best["score"]:
                best = {"score": score, "structural": structural,
                        "eng_rolled": eng_rolled, "pct": float(branch.get("percentage") or 0)}

    # ---- verdict ----
    all_obs = [abs(d) for slot in ("p1", "p2") for _s, d in obs_rolled[slot]]
    findings: list[str] = []
    if best is None:
        bucket = "no_usable_branch"
    elif best["structural"]:
        bucket = "structural_component_count"
    else:
        for slot in ("p1", "p2"):
            for source, a, b in _pair_by_source(
                obs_rolled[slot], best["eng_rolled"][slot]
            ):
                if a == b:
                    continue
                label = _classify_ratio(a / max(b, 1))
                findings.append(f"{slot}:{a}vs{b}:{label}:{source or 'move'}")
        if not findings:
            bucket = "matches_best_branch"
        elif all(v in legal for v in all_obs) and legal:
            bucket = "legitimate_roll_in_legal_set"
        else:
            labels = {f.split(":", 2)[2] for f in findings}
            bucket = "damage_calc:" + ";".join(sorted(labels))

    result = {
        "seed": row.get("seed"), "step": row.get("step"), "choices": dict(choices),
        "bucket": bucket, "findings": findings,
        "observed_rolled": {k: [[s, d] for s, d in v] for k, v in obs_rolled.items()},
        "engine_rolled": (
            {k: [[s, d] for s, d in v] for k, v in best["eng_rolled"].items()} if best else None
        ),
        "observed_in_legal_set": [v in legal for v in all_obs],
        "context": context,
    }
    if verbose:
        print(f"  seed {result['seed']} step {result['step']} -> {bucket}  {findings}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--class", dest="klass", default="roll_scaled_component")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="triage a deliberate sample; shares are then over the sample, "
                         "not the class")
    args = ap.parse_args(argv)

    report = json.loads(args.report.read_text())
    rows = [
        r for r in (report.get("repros") or [])
        if r.get("kind") == "transition_diverged"
        and args.klass in str(r.get("divergence_class", ""))
    ]

    # POPULATION GUARD. Percentages here are quoted as shares of a class, so they
    # are only meaningful over the WHOLE class. `--repros-per-game` (default 8)
    # silently caps games with many divergences: the first run of this triage
    # covered 64 of 86 rows and produced confident-looking percentages over an
    # incomplete population. The census records the true class total; refuse
    # rather than narrate the shortfall.
    expected = int((report.get("divergence_classes") or {}).get(args.klass, -1))
    if expected < 0:
        # Substring match (e.g. a class family) has no single census total.
        matching = [
            v for k, v in (report.get("divergence_classes") or {}).items()
            if args.klass in k
        ]
        expected = sum(matching) if matching else -1
    if expected >= 0 and len(rows) < expected and not args.allow_partial:
        raise SystemExit(
            f"\nINCOMPLETE POPULATION — refusing to compute shares.\n"
            f"  census says class {args.klass!r} has {expected} rows; this report "
            f"carries only {len(rows)} repros for it.\n"
            f"  The repro set is capped per game. Regenerate with a higher cap:\n"
            f"    --keep-repro 5000 --repros-per-game 300\n"
            f"  or pass --allow-partial to triage a deliberate sample (shares will "
            f"then be over the sample, not the class).\n"
        )
    scope = "class" if expected >= 0 and len(rows) >= expected else "sample"
    print(f"triaging {len(rows)} rows of class {args.klass!r} "
          f"(census total {expected if expected >= 0 else 'unknown'}; shares over the {scope})\n")

    results = [triage_row(r, verbose=args.verbose) for r in rows]
    buckets: Counter = Counter(r["bucket"] for r in results)
    examples: dict[str, Any] = {}
    for r in results:
        examples.setdefault(r["bucket"], r)

    print(f"{'count':>6}  {'share':>7}  bucket")
    total = len(results) or 1
    for bucket, count in buckets.most_common():
        ex = examples[bucket]
        print(f"{count:>6}  {100*count/total:>6.1f}%  {bucket}")
        print(f"{'':>15}repro: seed {ex['seed']} step {ex['step']} {ex['choices']}")

    if args.json:
        args.json.write_text(json.dumps(
            {"class": args.klass, "rows": len(results),
             "buckets": dict(buckets), "results": results}, indent=2))
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
